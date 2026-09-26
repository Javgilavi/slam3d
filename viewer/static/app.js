// slam3d synchronized viewer: 360 video <-> floor plan <-> 3D reconstruction.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
const fmt = (s) => { const m = Math.floor(s / 60); return `${m}:${(s - 60 * m).toFixed(1).padStart(4, '0')}`; };
const CLASS_COLORS = ['#ff6b6b', '#4dabf7', '#ffd43b', '#69db7c', '#da77f2', '#ffa94d', '#38d9a9', '#f783ac', '#a9e34b', '#74c0fc', '#e599f7', '#ffc078'];

const S = {
  run: null, frames: [], t0: 0, traj: null, objects: [], walls: null, gt: null, pf: null, particles: null,
  idx: 0, t: 0, trajIdx: -1, lon: 0, lat: 0, fov: 85, layers: {}, picking: false, pairs: [],
  plan: { s: 10, ox: 0, oy: 0 }, labelColor: {}, perf: { frames: 0, last: performance.now(), fps: 0, planMs: 0, syncMs: 0 },
};

// ------------------------------------------------------------------ utils
function lowerBound(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }
function nearestIdx(arr, x) { const i = lowerBound(arr, x); if (i <= 0) return 0; if (i >= arr.length) return arr.length - 1; return (x - arr[i - 1] <= arr[i] - x) ? i - 1 : i; }
function colorFor(label) { if (!(label in S.labelColor)) S.labelColor[label] = CLASS_COLORS[Object.keys(S.labelColor).length % CLASS_COLORS.length]; return S.labelColor[label]; }

// alignment: plan_xy = S @ grav_xy ; z_plan = scale * (z - floor_z)
function objectPlanXYZ(o) {
  const a = S.run.alignment; const c = o.centroid_grav_units;
  if (!a) return [c[0], c[1], c[2]];
  const M = a.S_plan_from_grav;
  return [M[0][0] * c[0] + M[0][1] * c[1] + M[0][2], M[1][0] * c[0] + M[1][1] * c[1] + M[1][2], a.scale * (c[2] - a.floor_z_units)];
}

// ------------------------------------------------------------------ panorama (explicit UV sphere in slam3d pano convention)
const video = $('#video');
let panoRenderer, panoScene, panoCam, panoTex;
function buildPanoSphere(lonSeg = 128, latSeg = 64, R = 100) {
  const pos = [], uv = [], idx = [];
  for (let j = 0; j <= latSeg; j++) {
    const lat = -Math.PI / 2 + Math.PI * j / latSeg;           // + = down in pano frame
    for (let i = 0; i <= lonSeg; i++) {
      const lon = -Math.PI + 2 * Math.PI * i / lonSeg;        // + = right
      const px = Math.cos(lat) * Math.sin(lon), py = Math.sin(lat), pz = Math.cos(lat) * Math.cos(lon);
      pos.push(R * px, -R * py, -R * pz);                      // three.js: x right, y up, -z forward
      uv.push(i / lonSeg, 1 - j / latSeg);
    }
  }
  for (let j = 0; j < latSeg; j++) for (let i = 0; i < lonSeg; i++) {
    const a = j * (lonSeg + 1) + i, b = a + lonSeg + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}
function initPano() {
  const el = $('#pano');
  panoRenderer = new THREE.WebGLRenderer({ antialias: false });
  panoRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  el.appendChild(panoRenderer.domElement);
  panoScene = new THREE.Scene();
  panoCam = new THREE.PerspectiveCamera(S.fov, 1, 0.1, 1000);
  panoCam.rotation.order = 'YXZ';
  panoTex = new THREE.VideoTexture(video);
  panoTex.colorSpace = THREE.SRGBColorSpace;
  panoTex.minFilter = THREE.LinearFilter; panoTex.generateMipmaps = false;
  panoScene.add(new THREE.Mesh(buildPanoSphere(), new THREE.MeshBasicMaterial({ map: panoTex, side: THREE.DoubleSide })));
  let drag = null;
  el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, lon: S.lon, lat: S.lat }; el.setPointerCapture(e.pointerId); el.style.cursor = 'grabbing'; });
  el.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const k = (S.fov / el.clientHeight) * Math.PI / 180;
    // "grab the image": dragging right turns the view left, dragging down tilts the view up (S.lat + = down)
    S.lon = drag.lon - (e.clientX - drag.x) * k; S.lat = Math.max(-1.5, Math.min(1.5, drag.lat - (e.clientY - drag.y) * k));
    drawPlanDynamic();
  });
  el.addEventListener('pointerup', () => { drag = null; el.style.cursor = 'grab'; });
  el.addEventListener('wheel', (e) => { e.preventDefault(); S.fov = Math.max(25, Math.min(110, S.fov + e.deltaY * 0.05)); }, { passive: false });
}
function renderPano() {
  const el = $('#pano');
  const w = el.clientWidth, h = el.clientHeight;
  if (panoRenderer.domElement.width !== Math.floor(w * panoRenderer.getPixelRatio())) { panoRenderer.setSize(w, h); }
  panoCam.aspect = w / h; panoCam.fov = S.fov; panoCam.updateProjectionMatrix();
  panoCam.rotation.y = -S.lon; panoCam.rotation.x = -S.lat;
  panoRenderer.render(panoScene, panoCam);
}

// ------------------------------------------------------------------ time sync
function fpsEnc() { return S.run.fps_encoded || 29.97; }
function onFrame(mediaTime) {
  const idx = Math.max(0, Math.min(S.frames.length - 1, Math.round(mediaTime * fpsEnc())));
  const tPrev = S.t;
  S.idx = idx; S.t = S.frames[idx];
  if (S.traj) {
    const k = nearestIdx(S.traj.t, S.t);
    S.trajIdx = Math.abs(S.traj.t[k] - S.t) < 0.1 ? k : -1;
    S.perf.syncMs = S.trajIdx >= 0 ? Math.abs(S.traj.t[k] - S.t) * 1000 : NaN;
  }
  if (S.t !== tPrev) onTimeChanged();
}
function onTimeChanged() {
  const rel = S.t - S.t0;
  $('#time-label').textContent = `${fmt(rel)} · #${S.idx}`;
  $('#hud-time').textContent = `${fmt(rel)}  (t=${S.t.toFixed(3)})`;
  if (document.activeElement !== $('#seek')) $('#seek').value = rel / Math.max(1e-6, S.frames[S.frames.length - 1] - S.t0);
  if (S.trajIdx >= 0) {
    const p = S.traj.xyz[S.trajIdx];
    $('#hud-pose').textContent = `${S.traj.frame} x ${p[0].toFixed(2)} y ${p[1].toFixed(2)} z ${p[2].toFixed(2)}  heading ${(viewHeading() * 180 / Math.PI).toFixed(0)}°`;
  } else $('#hud-pose').textContent = 'no pose for this frame (SLAM not tracking / not initialised)';
  drawPlanDynamic(); updateCloudCamera(); maybeFetchParticles();
}
function seekTime(t) { const idx = nearestIdx(S.frames, t); video.currentTime = (idx + 0.5) / fpsEnc(); if (video.paused) onFrame(idx / fpsEnc()); }
function viewHeading(lon = S.lon) {
  const X = S.traj.xaxis[S.trajIdx], Z = S.traj.zaxis[S.trajIdx];
  return Math.atan2(Math.sin(lon) * X[1] + Math.cos(lon) * Z[1], Math.sin(lon) * X[0] + Math.cos(lon) * Z[0]);
}
function initControls() {
  const play = () => { if (video.paused) { video.play(); $('#btn-play').textContent = '⏸'; } else { video.pause(); $('#btn-play').textContent = '▶'; } };
  $('#btn-play').onclick = play;
  $('#btn-prev').onclick = () => { video.pause(); video.currentTime = Math.max(0, (S.idx - 1 + 0.5) / fpsEnc()); };
  $('#btn-next').onclick = () => { video.pause(); video.currentTime = (S.idx + 1 + 0.5) / fpsEnc(); };
  $('#seek').oninput = (e) => seekTime(S.t0 + e.target.value * (S.frames[S.frames.length - 1] - S.t0));
  $('#speed').onchange = (e) => { video.playbackRate = parseFloat(e.target.value); };
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.code === 'Space') { e.preventDefault(); play(); }
    if (e.code === 'ArrowLeft') $('#btn-prev').click();
    if (e.code === 'ArrowRight') $('#btn-next').click();
  });
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = (_now, meta) => { onFrame(meta.mediaTime); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
  } else video.addEventListener('timeupdate', () => onFrame(video.currentTime));
  video.addEventListener('seeked', () => onFrame(video.currentTime));
}

// ------------------------------------------------------------------ plan canvas
const cStatic = $('#plan-static'), cDyn = $('#plan-dyn');
const imgs = { structure: new Image(), drawing: new Image(), asbuilt: new Image() };
const w2c = (x, y) => [x * S.plan.s + S.plan.ox, -y * S.plan.s + S.plan.oy];
const c2w = (cx, cy) => [(cx - S.plan.ox) / S.plan.s, -(cy - S.plan.oy) / S.plan.s];
function resizePlan() {
  for (const c of [cStatic, cDyn]) { c.width = c.clientWidth * devicePixelRatio; c.height = c.clientHeight * devicePixelRatio; }
  drawPlanStatic(); drawPlanDynamic();
}
function fitPlan() {
  let xs, ys;
  if (S.traj && S.traj.frame === 'plan') { xs = S.traj.xyz.map((p) => p[0]); ys = S.traj.xyz.map((p) => p[1]); }
  else { const P = S.run.plan; xs = [0, P.width_px * P.res_m_per_px]; ys = [0, P.height_px * P.res_m_per_px]; }
  const x0 = Math.min(...xs) - 4, x1 = Math.max(...xs) + 4, y0 = Math.min(...ys) - 4, y1 = Math.max(...ys) + 4;
  const W = cStatic.width, H = cStatic.height;
  S.plan.s = Math.min(W / (x1 - x0), H / (y1 - y0));
  S.plan.ox = W / 2 - S.plan.s * (x0 + x1) / 2; S.plan.oy = H / 2 + S.plan.s * (y0 + y1) / 2;
}
function drawPlanStatic() {
  const t0 = performance.now();
  const ctx = cStatic.getContext('2d'); const L = S.layers;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#f4f4f2'; ctx.fillRect(0, 0, cStatic.width, cStatic.height);
  const P = S.run.plan;
  if (L.drawing && S.run.drawing && imgs.drawing.complete && imgs.drawing.naturalWidth) {
    const d = S.run.drawing, px = d.src_res_m_per_px;
    const [x, y] = w2c(-0.5 * px, (d.height - 0.5) * px);
    ctx.globalAlpha = 0.45; ctx.drawImage(imgs.drawing, x, y, d.width * px * S.plan.s, d.height * px * S.plan.s); ctx.globalAlpha = 1;
  }
  if (L.structure && imgs.structure.complete && imgs.structure.naturalWidth) {
    const r = P.res_m_per_px; const [x, y] = w2c(-0.5 * r, (P.height_px - 0.5) * r);
    ctx.globalCompositeOperation = 'multiply'; ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgs.structure, x, y, P.width_px * r * S.plan.s, P.height_px * r * S.plan.s);
    ctx.globalCompositeOperation = 'source-over'; ctx.imageSmoothingEnabled = true;
  }
  if (L.asbuilt && imgs.asbuilt.complete && imgs.asbuilt.naturalWidth) {
    const r = P.res_m_per_px; const [x, y] = w2c(-0.5 * r, (P.height_px - 0.5) * r);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(imgs.asbuilt, x, y, P.width_px * r * S.plan.s, P.height_px * r * S.plan.s);
    ctx.imageSmoothingEnabled = true;
  }
  if (L.walls && S.walls) {
    for (let i = 0; i < S.walls.xy.length; i++) {
      const [x, y] = w2c(...S.walls.xy[i]); ctx.fillStyle = S.walls.d[i] < 0.2 ? '#1f9d55' : '#ff8c1a'; ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    }
  }
  if (L.gt && S.gt && S.gt.xyz.length) polyline(ctx, S.gt.xyz, '#d63333', 1.5);
  if (L.path && S.traj && S.traj.frame === 'plan') polyline(ctx, S.traj.xyz, '#1c6dd0', 2.5);
  if (L.pf && S.pf) polyline(ctx, S.pf.x.map((x, i) => [x, S.pf.y[i]]), '#b43ad6', 1.5);
  if (L.objects) for (const o of S.objects) {
    if (o.status === 'dynamic' && !L.dynamic) continue;
    const p = objectPlanXYZ(o); const [x, y] = w2c(p[0], p[1]);
    ctx.beginPath(); ctx.arc(x, y, o.status === 'confirmed' ? 6 : 4, 0, 2 * Math.PI);
    ctx.strokeStyle = colorFor(o.label); ctx.lineWidth = 2;
    if (o.status === 'dynamic') { ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]); }
    else { ctx.fillStyle = colorFor(o.label) + 'aa'; ctx.fill(); ctx.stroke(); }
  }
  S.perf.planMs = performance.now() - t0;
}
function polyline(ctx, pts, color, w) {
  ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = w2c(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.stroke();
}
function drawPlanDynamic() {
  if (!S.run) return;
  const ctx = cDyn.getContext('2d'); ctx.clearRect(0, 0, cDyn.width, cDyn.height);
  if (S.layers.pf && S.particles) {
    ctx.fillStyle = '#8e2de2aa';
    for (let i = 0; i < S.particles.x.length; i++) { const [x, y] = w2c(S.particles.x[i], S.particles.y[i]); ctx.fillRect(x - 1, y - 1, 2, 2); }
  }
  if (S.layers.pf && S.pf) {
    const k = lowerBound(S.pf.t, S.t) - 1;
    if (k >= 0) { const [x, y] = w2c(S.pf.x[k], S.pf.y[k]); ctx.beginPath(); ctx.arc(x, y, Math.max(4, S.pf.std[k] * S.plan.s), 0, 2 * Math.PI); ctx.strokeStyle = '#b43ad6'; ctx.lineWidth = 2; ctx.stroke(); }
  }
  for (const pr of S.pairs) { const [x, y] = w2c(...pr.plan_xy); ctx.fillStyle = '#ff00aa'; ctx.fillRect(x - 4, y - 4, 8, 8); }
  if (S.trajIdx >= 0 && S.traj.frame === 'plan') {
    const p = S.traj.xyz[S.trajIdx]; const [x, y] = w2c(p[0], p[1]);
    const hd = viewHeading(); const half = (S.fov * Math.PI / 180) / 2 * (cDyn.clientWidth ? 1 : 1);
    const R = 38 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, R, -(hd + half), -(hd - half)); ctx.closePath();
    ctx.fillStyle = '#ffae0055'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fillStyle = '#ff7b00'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
  }
}
function initPlan() {
  imgs.structure.onload = drawPlanStatic; imgs.drawing.onload = drawPlanStatic; imgs.asbuilt.onload = drawPlanStatic;
  imgs.asbuilt.onerror = () => {};  // layer not computed (e.g. alignment not confident)
  imgs.structure.src = '/media/plan_structure.png';
  imgs.asbuilt.src = '/media/discrepancy.png';
  if (S.run.drawing) imgs.drawing.src = '/media/plan_drawing.jpg';
  let drag = null, moved = false;
  cDyn.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, ox: S.plan.ox, oy: S.plan.oy }; moved = false; cDyn.setPointerCapture(e.pointerId); });
  cDyn.addEventListener('pointermove', (e) => {
    if (!drag) return; const dx = (e.clientX - drag.x) * devicePixelRatio, dy = (e.clientY - drag.y) * devicePixelRatio;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    S.plan.ox = drag.ox + dx; S.plan.oy = drag.oy + dy; drawPlanStatic(); drawPlanDynamic();
  });
  cDyn.addEventListener('pointerup', (e) => { const wasDrag = moved; drag = null; if (!wasDrag) planClick(e); });
  cDyn.addEventListener('wheel', (e) => {
    e.preventDefault(); const r = cDyn.getBoundingClientRect(); const cx = (e.clientX - r.left) * devicePixelRatio, cy = (e.clientY - r.top) * devicePixelRatio;
    const f = Math.exp(-e.deltaY * 0.0015); S.plan.ox = cx - (cx - S.plan.ox) * f; S.plan.oy = cy - (cy - S.plan.oy) * f; S.plan.s *= f;
    drawPlanStatic(); drawPlanDynamic();
  }, { passive: false });
  document.querySelectorAll('#layers input[type=checkbox]').forEach((cb) => {
    S.layers[cb.dataset.layer] = cb.checked;
    cb.onchange = async () => { S.layers[cb.dataset.layer] = cb.checked; if (cb.dataset.layer === 'pf') await loadPF(); drawPlanStatic(); drawPlanDynamic(); updateCloudObjects(); };
  });
}
async function planClick(e) {
  const r = cDyn.getBoundingClientRect(); const cx = (e.clientX - r.left) * devicePixelRatio, cy = (e.clientY - r.top) * devicePixelRatio;
  const [wx, wy] = c2w(cx, cy);
  if (S.picking) {
    S.pairs.push({ t: S.t, frame_idx: S.idx, plan_xy: [+wx.toFixed(3), +wy.toFixed(3)] }); S.picking = false;
    $('#btn-pick').classList.remove('active'); renderPairs(); drawPlanDynamic(); return;
  }
  if (S.layers.objects) {
    let best = null, bd = 10 * devicePixelRatio;
    for (const o of S.objects) {
      if (o.status === 'dynamic' && !S.layers.dynamic) continue;
      const p = objectPlanXYZ(o); const [x, y] = w2c(p[0], p[1]); const d = Math.hypot(x - cx, y - cy);
      if (d < bd) { bd = d; best = o; }
    }
    if (best) { inspect(best); return; }
  }
  const vis = await api(`/api/visits?x=${wx}&y=${wy}&radius=${Math.max(0.6, 12 / S.plan.s * devicePixelRatio)}`);
  if (!vis.length) return;
  if (vis.length === 1) { seekTime(vis[0].t); return; }
  const box = $('#visits'); box.innerHTML = '<b>This spot was visited several times:</b>';
  vis.sort((a, b) => a.t - b.t).forEach((v) => {
    const d = document.createElement('div'); d.className = 'v'; d.textContent = `${fmt(v.t - S.t0)}  (${v.dist.toFixed(2)} m)`;
    d.onclick = () => { seekTime(v.t); box.classList.add('hidden'); }; box.appendChild(d);
  });
  box.style.left = `${e.clientX + 8}px`; box.style.top = `${e.clientY + 8}px`; box.classList.remove('hidden');
  setTimeout(() => window.addEventListener('pointerdown', function h(ev) { if (!box.contains(ev.target)) { box.classList.add('hidden'); window.removeEventListener('pointerdown', h); } }), 0);
}

// ------------------------------------------------------------------ alignment tool
function renderPairs() {
  const el = $('#pairs'); el.innerHTML = '';
  S.pairs.forEach((p, i) => {
    const d = document.createElement('div'); d.innerHTML = `<span>${fmt(p.t - S.t0)} → (${p.plan_xy[0].toFixed(1)}, ${p.plan_xy[1].toFixed(1)})</span><span class="del">✕</span>`;
    d.querySelector('.del').onclick = () => { S.pairs.splice(i, 1); renderPairs(); drawPlanDynamic(); };
    el.appendChild(d);
  });
  $('#btn-realign').disabled = S.pairs.length < 2;
}
function initAlignTool() {
  $('#btn-pick').onclick = () => { S.picking = !S.picking; $('#btn-pick').classList.toggle('active', S.picking); };
  $('#btn-realign').onclick = async () => {
    await fetch('/api/correspondences', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pairs: S.pairs }) });
    await fetch('/api/realign', { method: 'POST' });
    $('#align-diag').textContent = 're-aligning…';
    const poll = setInterval(async () => {
      const st = await api('/api/status');
      if (st.align && st.align.state !== 'running') { clearInterval(poll); await reloadAlignment(); }
    }, 1500);
  };
}
async function reloadAlignment() {
  S.run = await api('/api/run');
  [S.traj, S.walls] = await Promise.all([api('/api/trajectory'), api('/api/wall_evidence')]);
  showAlignDiag(); drawPlanStatic(); drawPlanDynamic(); await loadCloud();
}
function showAlignDiag() {
  const a = S.run.alignment; if (!a) { $('#align-diag').textContent = 'not aligned yet'; return; }
  const c = a.confidence || {}; const ev = S.run.alignment_eval;
  let s = `${a.method}: ${c.status || ''}\nwall inliers ${(c.wall_inlier_frac ?? NaN).toFixed(2)}  margin ${c.margin != null ? c.margin.toFixed(2) : 'n/a'}\nscale ${a.scale.toFixed(4)} m/unit`;
  if (ev) s += `\nvs GT: median ${ev.plan_frame_2d_error_no_extra_alignment.median.toFixed(2)} m`;
  $('#align-diag').textContent = s;
}

// ------------------------------------------------------------------ particle filter
async function loadPF() {
  const exp = $('#pf-exp').value; if (!exp || !S.layers.pf) return;
  S.pf = await api(`/api/pf/${exp}`); S.particles = null; maybeFetchParticles(true);
}
let lastParticleFetch = 0;
async function maybeFetchParticles(force = false) {
  if (!S.layers.pf || !$('#pf-exp').value) return;
  const now = performance.now(); if (!force && now - lastParticleFetch < 250) return; lastParticleFetch = now;
  try { S.particles = await api(`/api/particles/${$('#pf-exp').value}?t=${S.t}`); drawPlanDynamic(); } catch (_) { /* no snapshot */ }
}

// ------------------------------------------------------------------ 3D cloud
let cloudRenderer, cloudScene, cloudCam, controls, cloudPoints, trajLine, camMarker, objGroup;
function initCloud() {
  const el = $('#cloud');
  cloudRenderer = new THREE.WebGLRenderer({ antialias: true }); cloudRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  el.appendChild(cloudRenderer.domElement);
  cloudScene = new THREE.Scene(); cloudScene.background = new THREE.Color('#0b0d10');
  cloudCam = new THREE.PerspectiveCamera(55, 1, 0.05, 2000); cloudCam.up.set(0, 0, 1);
  controls = new OrbitControls(cloudCam, cloudRenderer.domElement); controls.enableDamping = true;
  camMarker = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.6, 12), new THREE.MeshBasicMaterial({ color: 0xff7b00 }));
  cloudScene.add(camMarker); objGroup = new THREE.Group(); cloudScene.add(objGroup);
  cloudScene.add(new THREE.AxesHelper(1));
  const ray = new THREE.Raycaster();
  cloudRenderer.domElement.addEventListener('click', (e) => {
    const r = cloudRenderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), cloudCam);
    const hit = ray.intersectObjects(objGroup.children.filter((c) => c.userData.pick), false)[0];
    if (hit) inspect(hit.object.userData.obj);
  });
}
async function loadCloud() {
  const buf = await fetch('/api/pointcloud.bin?voxel=0.04').then((r) => r.arrayBuffer());
  const n = new Uint32Array(buf, 0, 1)[0];
  const xyz = new Float32Array(buf, 4, n * 3); const rgb = new Uint8Array(buf, 4 + n * 12, n * 3);
  const col = new Float32Array(n * 3); for (let i = 0; i < n * 3; i++) col[i] = rgb[i] / 255;
  if (cloudPoints) cloudScene.remove(cloudPoints);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(xyz.slice(), 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  cloudPoints = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true }));
  cloudScene.add(cloudPoints);
  if (trajLine) cloudScene.remove(trajLine);
  if (S.traj) {
    const tg = new THREE.BufferGeometry().setFromPoints(S.traj.xyz.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
    trajLine = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: 0x4da3ff })); cloudScene.add(trajLine);
    g.computeBoundingBox(); const c = new THREE.Vector3(); new THREE.Box3().setFromPoints(S.traj.xyz.map((p) => new THREE.Vector3(...p))).getCenter(c);
    controls.target.copy(c); cloudCam.position.set(c.x - 15, c.y - 25, c.z + 25); controls.update();
  }
  updateCloudObjects();
}
function updateCloudObjects() {
  if (!objGroup) return;
  objGroup.clear();
  if (!S.layers.objects || !S.run.alignment) return;
  for (const o of S.objects) {
    if (o.status === 'dynamic' && !S.layers.dynamic) continue;
    const p = objectPlanXYZ(o); const e = o.observed_extent_m.map((v) => Math.max(0.25, Math.min(v, 4)));
    const box = new THREE.Mesh(new THREE.BoxGeometry(e[0], e[1], e[2]), new THREE.MeshBasicMaterial({ color: colorFor(o.label), wireframe: true }));
    box.position.set(p[0], p[1], p[2]); box.userData = { pick: true, obj: o }; objGroup.add(box);
  }
}
function updateCloudCamera() {
  if (!camMarker || S.trajIdx < 0) return;
  const p = S.traj.xyz[S.trajIdx]; camMarker.position.set(p[0], p[1], p[2]);
  const hd = viewHeading(); camMarker.rotation.set(0, 0, hd - Math.PI / 2);
}
function renderCloud() {
  const el = $('#cloud'); const w = el.clientWidth, h = el.clientHeight;
  if (cloudRenderer.domElement.width !== Math.floor(w * cloudRenderer.getPixelRatio())) cloudRenderer.setSize(w, h);
  cloudCam.aspect = w / h; cloudCam.updateProjectionMatrix(); controls.update(); cloudRenderer.render(cloudScene, cloudCam);
}

// ------------------------------------------------------------------ inspector
function inspect(o) {
  const p = objectPlanXYZ(o);
  const rows = o.observations.slice().sort((a, b) => a.t - b.t).map((ob, i) =>
    `<tr class="obs" data-i="${i}"><td>${fmt(ob.t - S.t0)}</td><td>${ob.label}</td><td>${ob.conf.toFixed(2)}</td><td>${ob.src}</td><td>${ob.n_pts}</td></tr>`).join('');
  $('#inspector-body').innerHTML = `
    <h3 style="margin:2px 0">#${o.id} ${o.label} <span class="tag ${o.status}">${o.status}</span></h3>
    ${o.thumb ? `<img src="/api/thumb/${o.id}.jpg">` : ''}
    <table>
      <tr><td>confidence</td><td>${o.confidence.toFixed(2)} (${o.n_observations} observations)</td></tr>
      <tr><td>location (${S.run.alignment ? 'plan m' : 'SLAM units'})</td><td>x ${p[0].toFixed(2)}, y ${p[1].toFixed(2)}, z ${p[2].toFixed(2)}</td></tr>
      <tr><td>observed extent m</td><td>${o.observed_extent_m.map((v) => v.toFixed(2)).join(' × ')}</td></tr>
      <tr><td>position spread</td><td>${o.position_spread_m.toFixed(2)} m</td></tr>
      <tr><td>labels</td><td>${Object.entries(o.labels).map(([k, v]) => `${k}:${v.toFixed(1)}`).join(', ')}</td></tr>
      <tr><td>geometry</td><td>${o.geometry_sources.join(', ')}</td></tr>
    </table>
    <div class="small">${o.note}</div>
    <b>Supporting observations</b> <span class="small">(click → jump & look at it)</span>
    <table><tr><td>time</td><td>label</td><td>conf</td><td>3D from</td><td>pts</td></tr>${rows}</table>`;
  const sorted = o.observations.slice().sort((a, b) => a.t - b.t);
  $('#inspector-body').querySelectorAll('tr.obs').forEach((tr) => tr.onclick = () => {
    const ob = sorted[+tr.dataset.i]; video.pause(); $('#btn-play').textContent = '▶';
    seekTime(ob.t); const b = ob.bearing_pano; const n = Math.hypot(b[0], b[1], b[2]);
    S.lon = Math.atan2(b[0], b[2]); S.lat = Math.asin(Math.max(-1, Math.min(1, b[1] / n)));  // pano y down == S.lat positive
  });
  $('#inspector').classList.remove('hidden');
}

// ------------------------------------------------------------------ status + perf
async function pollStatus() {
  try {
    const st = await api('/api/status'); const el = $('#stage-chips'); el.innerHTML = '';
    for (const [k, v] of Object.entries(st)) {
      const c = document.createElement('span'); c.className = `chip ${v.state}`; c.textContent = k;
      c.title = `${v.state}${v.runtime_s ? ` · ${v.runtime_s.toFixed(1)} s` : ''}${v.error ? `\n${v.error}` : ''}`; el.appendChild(c);
    }
  } catch (_) { /* server busy */ }
}
function loop() {
  renderPano(); renderCloud();
  const pf = S.perf; pf.frames++; const now = performance.now();
  if (now - pf.last > 1000) {
    pf.fps = pf.frames * 1000 / (now - pf.last); pf.frames = 0; pf.last = now;
    $('#perf').textContent = `render ${pf.fps.toFixed(0)} fps · plan ${pf.planMs.toFixed(1)} ms · sync Δ ${isNaN(pf.syncMs) ? 'n/a' : pf.syncMs.toFixed(1) + ' ms'}`;
    window.__slam3dPerf = { fps: pf.fps, planMs: pf.planMs, syncMs: pf.syncMs, idx: S.idx, t: S.t, trajIdx: S.trajIdx };
  }
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ boot
async function main() {
  S.run = await api('/api/run');
  $('#run-name').textContent = S.run.run;
  S.frames = (await api('/api/frames')).t; S.t0 = S.frames[0]; S.t = S.t0;
  const safe = (p) => p.catch(() => null);
  [S.traj, S.objects, S.walls, S.gt] = await Promise.all([safe(api('/api/trajectory')), safe(api('/api/objects')), safe(api('/api/wall_evidence')), safe(api('/api/gt'))]);
  S.objects = S.objects || [];
  const sel = $('#pf-exp'); (S.run.pf_experiments || []).forEach((e) => { const o = document.createElement('option'); o.value = e; o.textContent = e; sel.appendChild(o); });
  sel.onchange = loadPF;
  initPano(); initControls(); initPlan(); initAlignTool(); initCloud(); showAlignDiag();
  $('#inspector-close').onclick = () => $('#inspector').classList.add('hidden');
  video.src = '/media/pano.mp4';
  window.addEventListener('resize', resizePlan);
  cStatic.width = cStatic.clientWidth * devicePixelRatio; cStatic.height = cStatic.clientHeight * devicePixelRatio;
  cDyn.width = cStatic.width; cDyn.height = cStatic.height;
  fitPlan(); resizePlan(); await loadCloud(); pollStatus(); setInterval(pollStatus, 4000);
  video.addEventListener('loadeddata', () => onFrame(0));
  loop();
  window.__slam3d = { S, seekTime };
}
main().catch((e) => { document.body.insertAdjacentHTML('beforeend', `<div class="panel" style="position:fixed;bottom:10px;left:10px;color:#ff5a5a">viewer error: ${e.message}</div>`); console.error(e); });
