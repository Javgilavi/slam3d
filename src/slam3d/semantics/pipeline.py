"""Persistent 3D object mapping from 360 keyframes.

Per keyframe (subsampled SLAM keyframes, final optimised poses):
  1. render overlapping perspective views sharing the panorama centre (covers the seam, low distortion);
  2. open-vocabulary instance segmentation (YOLOE text prompts; fallback: any Ultralytics -seg model);
  3. map each instance polygon back to exact panorama pixels, de-duplicate across overlapping views;
  4. lift to 3D:
       a) SLAM landmarks observed in this keyframe whose pixels fall inside the (eroded) mask, with
          depth-consistency filtering against the median to reject background leakage;
       b) fallback for floor-standing objects: intersect the bearing of the lowest mask pixel with the
          estimated floor plane (reported as geometry_source='floor_ray');
  5. store the observation in the KEYFRAME CAMERA frame (so later pose-graph corrections re-project it)
     together with world coordinates computed from the final poses.

Association: per label group, observations are merged into objects with a 3D distance gate (metres)
and a colour-histogram appearance gate; an observation never merges with an object already observed
in the same keyframe. Objects whose observations move (> dynamic_spread_m) or whose label is in
`dynamic_classes` are marked dynamic. Extents are observed-point extents only; they are NOT complete
object shapes or precise poses.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from slam3d.geometry import sphere


def _views(n_yaw: int, pitch_deg: float, fov: float, size: int):
    out = []
    for k in range(n_yaw):
        yaw = 2 * np.pi * k / n_yaw
        R = sphere.perspective_view_rotation(yaw, np.deg2rad(pitch_deg))
        out.append((yaw, R, sphere.perspective_intrinsics(fov, size)))
    return out


def _hist(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [16, 8], [0, 180, 0, 256]).ravel()
    return h / max(h.sum(), 1e-9)


def _bhatt(a, b):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


class Detector:
    def __init__(self, model_name: str, prompts: list[str], weights_dir: Path, conf: float = 0.3, imgsz: int = 640):
        import os

        os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never modify the environment at run time
        from ultralytics import YOLO, YOLOE

        cwd = os.getcwd()
        weights_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(weights_dir)  # keep downloaded weights inside third_party
        try:
            if "yoloe" in model_name:
                self.model = YOLOE(model_name)
                self.model.set_classes(prompts, self.model.get_text_pe(prompts))
                self.open_vocab = True
            else:
                self.model = YOLO(model_name)
                self.open_vocab = False
        finally:
            os.chdir(cwd)
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, images):
        res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, verbose=False)
        out = []
        for r in res:
            dets = []
            if r.masks is not None:
                for i in range(len(r.boxes)):
                    poly = r.masks.xy[i]
                    if len(poly) < 3:
                        continue
                    dets.append({"label": r.names[int(r.boxes.cls[i])], "conf": float(r.boxes.conf[i]),
                                 "bbox": r.boxes.xyxy[i].tolist(), "poly": np.asarray(poly, np.float32)})
            out.append(dets)
        return out


def run_objects(cfg, rdir: Path, log=print) -> dict:
    from slam3d.config import REPO
    from slam3d.io.tum import read_tum
    from slam3d.slam import stella

    oc = cfg.get("objects", {})
    if not oc.get("enabled", True):
        return {"enabled": False}
    out = rdir / "objects"
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    geo = json.loads((rdir / "geometry/geometry.json").read_text())
    align = json.loads((rdir / "align/alignment.json").read_text()) if (rdir / "align/alignment.json").exists() else None
    mpu = align["scale"] if align else geo["scale"]["metres_per_unit"] or 1.0
    floor_z = align["floor_z_units"] if align else None
    meta = json.loads((rdir / "ingest/stitch.json").read_text())
    W, H = int(meta["width"]), int(meta["height"])
    frames = np.genfromtxt(rdir / "ingest/frames.csv", delimiter=",", skip_header=1, dtype=str)
    ft = frames[:, 1].astype(float)
    static_mask = cv2.imread(str(rdir / "ingest/static_mask.png"), cv2.IMREAD_GRAYSCALE)

    # keyframes in grav frame (final poses) + landmarks in grav frame
    kcsv = np.genfromtxt(rdir / "geometry/keyframes_grav.csv", delimiter=",", skip_header=1)
    kid, kt = kcsv[:, 0].astype(int), kcsv[:, 1]
    KT = np.tile(np.eye(4), (len(kid), 1, 1))
    KT[:, :3, :] = kcsv[:, 2:14].reshape(-1, 3, 4)
    lms = stella.load_landmarks(rdir / "slam/landmarks.bin")
    Rg = np.array(geo["R_grav_slam"])
    L = lms["xyz"] @ Rg.T - np.array(geo["origin_offset_in_rotated_slam"])
    lm2i = {int(k): i for i, k in enumerate(lms["id"])}
    kobs = stella.load_kf_obs(rdir / "slam/kf_obs.bin")
    order = np.argsort(kobs["kf"], kind="stable")
    kobs = kobs[order]
    ukf, starts = np.unique(kobs["kf"], return_index=True)
    ends = np.append(starts[1:], len(kobs))
    kf_slice = {int(k): (s, e) for k, s, e in zip(ukf, starts, ends)}
    if floor_z is None:
        floor_z = float(np.median(KT[:, 2, 3]) - geo["scale"]["camera_height_units"])

    prompts = oc.get("prompts", ["person"])
    det = Detector(oc.get("model", "yoloe-11s-seg.pt"), prompts, REPO / "third_party/weights/yoloe",
                   conf=oc.get("conf", 0.3), imgsz=oc.get("imgsz", 640))
    views = _views(oc.get("n_views", 6), oc.get("pitch_deg", -20.0), oc.get("fov_deg", 100.0), oc.get("view_size", 640))
    step = int(oc.get("keyframe_step", 3))
    observations = []
    import time

    t_infer = 0.0
    sel = np.arange(0, len(kid), step)
    for n_done, ki in enumerate(sel):
        fi = int(np.argmin(np.abs(ft - kt[ki])))
        pano = cv2.imread(str(rdir / "ingest" / frames[fi][2]))
        if pano is None:
            continue
        imgs, maps = [], []
        for yaw, R, K in views:
            v, uv = sphere.equirect_to_perspective(pano, R, K, K.shape[0] and oc.get("view_size", 640))
            imgs.append(v)
            maps.append(uv)
        t0 = time.time()
        dets = det(imgs)
        t_infer += time.time() - t0
        # landmarks observed in this keyframe
        s, e = kf_slice.get(int(kid[ki]), (0, 0))
        ob = kobs[s:e]
        li = np.array([lm2i.get(int(l), -1) for l in ob["lm"]])
        ok = li >= 0
        lm_uv = ob["uv"][ok].astype(np.float64) - 0.5  # stella px -> our pixel-centre convention
        lm_w = L[li[ok]]
        Tw = KT[ki]
        lm_c = (lm_w - Tw[:3, 3]) @ Tw[:3, :3]  # into keyframe camera frame
        kf_obs_list = []
        for vi, dl in enumerate(dets):
            size = imgs[vi].shape[0]
            for d in dl:
                m = np.zeros((size, size), np.uint8)
                cv2.fillPoly(m, [d["poly"].round().astype(np.int32)], 1)
                if m.sum() < 50:
                    continue
                # exact panorama pixels of the instance
                ys, xs = np.nonzero(m)
                pu = sphere.wrap_u(maps[vi][ys, xs, 0], W)
                pv = maps[vi][ys, xs, 1]
                pano_mask = np.zeros((H, W), np.uint8)
                pano_mask[np.clip(pv.round().astype(int), 0, H - 1), np.clip(pu.round().astype(int), 0, W - 1)] = 1
                pano_mask = cv2.morphologyEx(pano_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                if static_mask is not None and (static_mask[pano_mask > 0] == 0).mean() > 0.5:
                    continue  # detection on the rig / operator's own equipment
                ero = cv2.erode(pano_mask, np.ones((5, 5), np.uint8))
                bc = sphere.pixel_to_bearing(pu.mean(), pv.mean(), W, H)  # rough centre bearing
                inside = ero[np.clip(lm_uv[:, 1].round().astype(int), 0, H - 1), np.clip(lm_uv[:, 0].round().astype(int), 0, W - 1)] > 0
                pts_c = lm_c[inside]
                src = None
                if len(pts_c) >= 3:
                    rng = np.linalg.norm(pts_c, axis=1)
                    med = np.median(rng)
                    keep = np.abs(rng - med) < 0.3 * med
                    pts_c = pts_c[keep]
                    if len(pts_c) >= 3:
                        src = "landmarks"
                dpath = rdir / "dense/depth" / f"kf_{int(kid[ki])}.npy"
                if src is None and dpath.exists():
                    # dense PanoVGGT radial depth (already scaled to SLAM units per keyframe)
                    dep = np.load(dpath).astype(np.float32)
                    dh, dw = dep.shape
                    ys_e, xs_e = np.nonzero(ero)
                    if len(xs_e) >= 20:
                        sub = np.linspace(0, len(xs_e) - 1, min(400, len(xs_e))).astype(int)
                        du = np.clip(((xs_e[sub] + 0.5) / W * dw - 0.5).round().astype(int), 0, dw - 1)
                        dv = np.clip(((ys_e[sub] + 0.5) / H * dh - 0.5).round().astype(int), 0, dh - 1)
                        r = dep[dv, du]
                        med = np.median(r)
                        keep = np.abs(r - med) < 0.2 * med  # reject background leakage at mask borders
                        if keep.sum() >= 10:
                            b = sphere.pixel_to_bearing(xs_e[sub][keep], ys_e[sub][keep], W, H)
                            pts_c = b * r[keep, None]
                            src = "dense_depth"
                if src is None:
                    # floor ray from the lowest mask pixel (object standing on the floor)
                    j = int(np.argmax(pv))
                    b = sphere.pixel_to_bearing(pu[j], pv[j], W, H)
                    bw = Tw[:3, :3] @ b
                    if bw[2] < -0.05:
                        lam = (floor_z - Tw[2, 3]) / bw[2]
                        if 0 < lam * mpu < 15.0:
                            pts_c = (lam * b)[None]
                            src = "floor_ray"
                if src is None:
                    continue
                pts_w = pts_c @ Tw[:3, :3].T + Tw[:3, 3]
                x0, y0, x1, y1 = [int(round(v)) for v in d["bbox"]]
                crop = imgs[vi][max(0, y0):y1, max(0, x0):x1]
                hist = _hist(imgs[vi], m)
                kf_obs_list.append({"kf": int(kid[ki]), "t": float(kt[ki]), "frame_idx": fi, "view": vi, "label": d["label"],
                                    "conf": d["conf"], "src": src, "n_pts": int(len(pts_c)),
                                    "p_cam": pts_c.mean(0).tolist(), "p_w": pts_w.mean(0).tolist(),
                                    "ext_w": (np.percentile(pts_w, 95, 0) - np.percentile(pts_w, 5, 0)).tolist() if len(pts_w) > 2 else [0, 0, 0],
                                    "bearing_pano": bc.tolist(), "bbox_pano_uv": [float(pu.min()), float(pv.min()), float(pu.max()), float(pv.max())],
                                    "pix": int(len(xs)), "_hist": hist, "_crop": crop})
        # de-duplicate across overlapping views of the same panorama (same label, overlapping bearing)
        kf_obs_list.sort(key=lambda o: -o["conf"] * np.sqrt(o["pix"]))
        kept = []
        for o in kf_obs_list:
            dup = False
            for k in kept:
                if k["label"] == o["label"] and np.degrees(sphere.angular_distance(np.array(k["bearing_pano"]), np.array(o["bearing_pano"]))) < 8.0:
                    dup = True
                    break
            if not dup:
                kept.append(o)
        observations.extend(kept)
        if n_done % 50 == 0:
            log(f"[objects] keyframe {n_done}/{len(sel)}: {len(observations)} observations")

    # ------------------------------------------------------------------ camera-rigid detections
    # The operator carrying the camera, their helmet and rig parts stay at the same position in the
    # KEYFRAME CAMERA frame across many keyframes; they are not scene objects.
    n_rigid = 0
    if observations:
        pc = np.array([o["p_cam"] for o in observations]) * mpu
        labels = np.array([o["label"] for o in observations])
        kfo = np.array([o["kf"] for o in observations])
        rigid = np.zeros(len(observations), bool)
        min_kf = max(5, oc.get("camera_rigid_min_kf_frac", 0.15) * max(len(sel), 1))
        for lab in np.unique(labels):
            idx = np.nonzero(labels == lab)[0]
            for i in idx:
                near = idx[np.linalg.norm(pc[idx] - pc[i], axis=1) < oc.get("camera_rigid_radius_m", 0.35)]
                if len(np.unique(kfo[near])) >= min_kf:
                    rigid[near] = True
        n_rigid = int(rigid.sum())
        observations = [o for o, r in zip(observations, rigid) if not r]
        log(f"[objects] removed {n_rigid} camera-rigid observations (operator / rig)")

    # ------------------------------------------------------------------ association
    gate_m = oc.get("gate_m", 0.8)
    app_gate = oc.get("appearance_gate", 0.6)
    dyn_classes = set(oc.get("dynamic_classes", ["person"]))
    objects = []
    for oi, o in enumerate(sorted(range(len(observations)), key=lambda i: observations[i]["t"])):
        ob = observations[o]
        pw = np.array(ob["p_w"])
        best, best_d = None, np.inf
        for obj in objects:
            if obj["label_votes"].get(ob["label"], 0) == 0 or ob["kf"] in obj["kfs"]:
                continue
            dist = np.linalg.norm(obj["centroid"] - pw) * mpu
            tol = max(gate_m, 0.5 * np.linalg.norm(obj["ext"]) * mpu)
            if dist < tol and _bhatt(obj["hist"], ob["_hist"]) < app_gate and dist < best_d:
                best, best_d = obj, dist
        if best is None:
            best = {"obs": [], "kfs": set(), "label_votes": {}, "centroid": pw, "ext": np.array(ob["ext_w"]), "hist": ob["_hist"],
                    "thumb": ob["_crop"], "best_conf": 0.0}
            objects.append(best)
        best["obs"].append(o)
        best["kfs"].add(ob["kf"])
        best["label_votes"][ob["label"]] = best["label_votes"].get(ob["label"], 0) + ob["conf"]
        ws = np.array([observations[i]["conf"] * (2.0 if observations[i]["src"] == "landmarks" else 1.0) for i in best["obs"]])
        P = np.array([observations[i]["p_w"] for i in best["obs"]])
        best["centroid"] = (ws[:, None] * P).sum(0) / ws.sum()
        best["ext"] = np.maximum(best["ext"], np.array(ob["ext_w"]))
        best["hist"] = 0.8 * best["hist"] + 0.2 * ob["_hist"]
        if ob["conf"] > best["best_conf"] and ob["_crop"].size:
            best["best_conf"], best["thumb"] = ob["conf"], ob["_crop"]

    S = np.array(align["S_plan_from_grav"]) if align else None
    out_objs = []
    for i, obj in enumerate(objects):
        label = max(obj["label_votes"], key=obj["label_votes"].get)
        P = np.array([observations[j]["p_w"] for j in obj["obs"]])
        spread = float(np.max(np.linalg.norm(P - np.median(P, 0), axis=1)) * mpu) if len(P) > 1 else 0.0
        confs = np.array([observations[j]["conf"] for j in obj["obs"]])
        n = len(obj["obs"])
        dynamic = label in dyn_classes or (n >= 3 and spread > oc.get("dynamic_spread_m", 1.5))
        conf = float(confs.mean() * (1 - np.exp(-n / 3.0)))
        rec = {"id": i, "label": label, "labels": obj["label_votes"], "confidence": conf, "n_observations": n,
               "status": "dynamic" if dynamic else ("confirmed" if n >= 3 else "tentative"),
               "centroid_grav_units": obj["centroid"].tolist(), "observed_extent_m": (obj["ext"] * mpu).tolist(),
               "position_spread_m": spread, "t_first": float(min(observations[j]["t"] for j in obj["obs"])),
               "t_last": float(max(observations[j]["t"] for j in obj["obs"])),
               "geometry_sources": sorted({observations[j]["src"] for j in obj["obs"]}),
               "observations": [{k: v for k, v in observations[j].items() if not k.startswith("_")} for j in obj["obs"]],
               "note": "extent = spread of observed points only; not a full shape or precise pose"}
        if S is not None:
            c = obj["centroid"]
            rec["plan_xy"] = (S[:2, :2] @ c[:2] + S[:2, 2]).tolist()
            rec["height_above_floor_m"] = float((c[2] - floor_z) * mpu)
        if obj["thumb"] is not None and obj["thumb"].size:
            cv2.imwrite(str(out / "thumbs" / f"{i}.jpg"), obj["thumb"])
        out_objs.append(rec)
    (out / "objects.json").write_text(json.dumps(out_objs, indent=1, default=float))

    # ------------------------------------------------------------------ consistency diagnostics
    static = [o for o in out_objs if o["status"] != "dynamic"]
    # re-projection of object centroids into their supporting panoramas: inside the detection box?
    hits, tot = 0, 0
    kpos = {int(k): i for i, k in enumerate(kid)}
    for o in static:
        c = np.array(o["centroid_grav_units"])
        for ob in o["observations"]:
            Tw = KT[kpos[ob["kf"]]]
            b = (c - Tw[:3, 3]) @ Tw[:3, :3]
            u, v = sphere.bearing_to_pixel(b, W, H)
            u0, v0, u1, v1 = ob["bbox_pano_uv"]
            pad = 0.1 * max(u1 - u0, v1 - v0) + 5
            wrap_ok = (u0 - pad <= u <= u1 + pad) or (u0 - pad <= u + W <= u1 + pad) or (u0 - pad <= u - W <= u1 + pad)
            hits += int(wrap_ok and v0 - pad <= v <= v1 + pad)
            tot += 1
    # duplicate candidates: same label, close, never co-observed
    dup = 0
    for a in range(len(static)):
        for b in range(a + 1, len(static)):
            if static[a]["label"] == static[b]["label"]:
                d = np.linalg.norm(np.array(static[a]["centroid_grav_units"]) - np.array(static[b]["centroid_grav_units"])) * mpu
                ka = {x["kf"] for x in static[a]["observations"]}
                kb = {x["kf"] for x in static[b]["observations"]}
                if d < gate_m and not (ka & kb):
                    dup += 1
    per_class = {}
    for o in out_objs:
        pc = per_class.setdefault(o["label"], {"objects": 0, "confirmed": 0, "dynamic": 0, "observations": 0})
        pc["objects"] += 1
        pc["confirmed"] += o["status"] == "confirmed"
        pc["dynamic"] += o["status"] == "dynamic"
        pc["observations"] += o["n_observations"]
    rep = {"keyframes_processed": int(len(sel)), "views_per_keyframe": len(views), "inference_s": t_infer,
           "n_observations": len(observations), "n_camera_rigid_observations_removed": n_rigid, "n_objects": len(out_objs),
           "n_confirmed_static": sum(o["status"] == "confirmed" for o in out_objs),
           "n_tentative": sum(o["status"] == "tentative" for o in out_objs), "n_dynamic": sum(o["status"] == "dynamic" for o in out_objs),
           "obs_geometry_source": {s: sum(o["src"] == s for o in observations) for s in ("landmarks", "dense_depth", "floor_ray")},
           "centroid_reprojection_inside_detection_frac": hits / max(tot, 1), "n_reprojection_checks": tot,
           "label_conflict_objects": sum(len(o["labels"]) > 1 for o in out_objs),
           "duplicate_candidates_same_label_within_gate_never_coobserved": dup,
           "metres_per_unit": mpu, "open_vocabulary": det.open_vocab, "prompts": prompts, "per_class": per_class}
    (out / "objects_report.json").write_text(json.dumps(rep, indent=2, default=float))
    log(f"[objects] {rep['n_objects']} objects ({rep['n_confirmed_static']} confirmed static, {rep['n_dynamic']} dynamic) "
        f"from {rep['n_observations']} observations; reprojection consistency {rep['centroid_reprojection_inside_detection_frac']:.2f}")
    return {k: v for k, v in rep.items() if k != "per_class"}
