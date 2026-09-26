import * as THREE from 'three';
import {OrbitControls} from 'three/addons/OrbitControls.js';
const $=s=>document.querySelector(s), V=$('#view');
const state={ready:false,walk:false,follow:false,playing:false,time:0,fps:0,frames:0,errors:[],sortMs:0};
window.__world=state;
const renderer=new THREE.WebGLRenderer({antialias:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));V.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#0b1118');
const camera=new THREE.PerspectiveCamera(100,1,.08,250);camera.up.set(0,0,1);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
const objectsGroup=new THREE.Group();scene.add(objectsGroup);
let shTexture=null,shDimensions=new THREE.Vector2(1,1);
let report,data,mesh,attributes,recordIndex=0,frameIndex=0,lastImage=-1,bounds,home,keys={},drag=null,sortAt=0,previousPose='',mapPoints=[];
function nearest(a,t){let l=0,r=a.length;while(l<r){const m=(l+r)>>1;if(a[m]<t)l=m+1;else r=m;}return l===0?0:l===a.length?l-1:(t-a[l-1]<a[l]-t?l-1:l);}
function semantic(id){return id<0?new THREE.Color(.20,.24,.28):new THREE.Color().setHSL((id*.61803398875)%1,.75,.55);}
const vertex=`attribute vec3 center, tint, ca, cb; attribute float opacity; uniform vec2 viewport, shDimensions; uniform sampler2D shTexture; uniform float useSH; attribute float shIndex; varying vec2 vQ; varying vec3 vColor; varying float vAlpha;
vec3 coeff(float k){float t=shIndex*16.+k;return texture2D(shTexture,(vec2(mod(t,shDimensions.x),floor(t/shDimensions.x))+.5)/shDimensions).rgb;}
vec3 appearance(){vec3 d=normalize(center-cameraPosition);float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
vec3 c=.5+.28209479177387814*coeff(0.);
c+=.4886025119029199*(-y*coeff(1.)+z*coeff(2.)-x*coeff(3.));
c+=1.0925484305920792*x*y*coeff(4.)-1.0925484305920792*y*z*coeff(5.)+.31539156525252005*(2.*zz-xx-yy)*coeff(6.)-1.0925484305920792*x*z*coeff(7.)+.5462742152960396*(xx-yy)*coeff(8.);
c+=-.5900435899266435*y*(3.*xx-yy)*coeff(9.)+2.890611442640554*x*y*z*coeff(10.)-.4570457994644658*y*(4.*zz-xx-yy)*coeff(11.)+.3731763325901154*z*(2.*zz-3.*xx-3.*yy)*coeff(12.)-.4570457994644658*x*(4.*zz-xx-yy)*coeff(13.)+1.445305721320277*z*(xx-yy)*coeff(14.)-.5900435899266435*x*(xx-3.*yy)*coeff(15.);
return max(c,vec3(0.));}
void main(){vec4 p=modelViewMatrix*vec4(center,1.);if(p.z>-.08){gl_Position=vec4(2.,2.,2.,1.);vAlpha=0.;return;}
mat3 C=mat3(ca.x,ca.y,ca.z,ca.y,cb.x,cb.y,ca.z,cb.y,cb.z); mat3 R=mat3(modelViewMatrix);C=R*C*transpose(R);
float fx=projectionMatrix[0][0]*viewport.x*.5, fy=projectionMatrix[1][1]*viewport.y*.5;
vec2 limit=1.3/vec2(projectionMatrix[0][0],projectionMatrix[1][1]);
vec2 slope=clamp(p.xy/-p.z,-limit,limit);
vec3 jx=vec3(fx/-p.z,0.,fx*slope.x/-p.z), jy=vec3(0.,fy/-p.z,fy*slope.y/-p.z);
float a=dot(jx,C*jx)+.3,b=dot(jx,C*jy),d=dot(jy,C*jy)+.3;float mid=(a+d)*.5,delta=sqrt(max(0.,(a-d)*(a-d)*.25+b*b));
float l1=max(.3,mid+delta),l2=max(.3,mid-delta);vec2 e=abs(b)>1e-6?normalize(vec2(b,l1-a)):(a>=d?vec2(1.,0.):vec2(0.,1.));
vec2 offset=3.*(position.x*sqrt(l1)*e+position.y*sqrt(l2)*vec2(-e.y,e.x));
gl_Position=projectionMatrix*p;
vec2 ndc=gl_Position.xy/gl_Position.w,extent=6.*sqrt(vec2(a,d))/viewport;
if(any(greaterThan(abs(ndc),vec2(1.)+extent))){gl_Position=vec4(2.,2.,2.,1.);vAlpha=0.;return;}
gl_Position.xy+=offset*2./viewport*gl_Position.w;vQ=position.xy*3.;vColor=useSH>.5?appearance():tint;vAlpha=opacity;}`;
const fragment=`precision highp float;varying vec2 vQ;varying vec3 vColor;varying float vAlpha;void main(){float q=dot(vQ,vQ);if(q>9.)discard;float a=min(.995,vAlpha*exp(-.5*q));if(a<.003)discard;gl_FragColor=vec4(vColor,a);}`;
function build(){
 const g=new THREE.InstancedBufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));g.setIndex([0,1,2,0,2,3]);g.instanceCount=data.length/14;
 attributes={};for(const [name,size] of [['center',3],['tint',3],['ca',3],['cb',3],['opacity',1],['shIndex',1]]){attributes[name]=new THREE.InstancedBufferAttribute(new Float32Array(g.instanceCount*size),size);g.setAttribute(name,attributes[name]);}
 mesh=new THREE.Mesh(g,new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,uniforms:{viewport:{value:new THREE.Vector2()},shTexture:{value:shTexture||new THREE.DataTexture(new Float32Array(4),1,1,THREE.RGBAFormat,THREE.FloatType)},shDimensions:{value:shDimensions},useSH:{value:shTexture?1:0}},transparent:true,depthWrite:false,depthTest:true,side:THREE.DoubleSide}));mesh.frustumCulled=false;mesh.renderOrder=1;scene.add(mesh);
 state.gaussians=g.instanceCount;renderer.setPixelRatio(g.instanceCount>150000?.65:Math.min(devicePixelRatio,1.5));sort(true);
}
function sort(force=false){
 const sig=camera.position.toArray().concat(camera.quaternion.toArray()).map(x=>x.toFixed(3)).join(',')+$('#appearance').value;
 if(!force&&(sig===previousPose||performance.now()-sortAt<130))return;
 previousPose=sig;sortAt=performance.now();camera.updateMatrixWorld();const m=camera.matrixWorldInverse.elements;
 const n=data.length/14;
 if(!state.sortBuffers||state.sortBuffers.depth.length!==n)state.sortBuffers={depth:new Float32Array(n),order:new Uint32Array(n),counts:new Uint32Array(65536)};
 const {depth,order,counts}=state.sortBuffers;counts.fill(0);let lo=Infinity,hi=-Infinity;
 for(let i=0;i<n;i++){const z=m[2]*data[i*14]+m[6]*data[i*14+1]+m[10]*data[i*14+2]+m[14];depth[i]=z;lo=Math.min(lo,depth[i]);hi=Math.max(hi,depth[i]);}
 const scale=65535/Math.max(hi-lo,1e-9),bin=z=>Math.min(65535,Math.max(0,Math.floor((z-lo)*scale)));
 for(let i=0;i<n;i++)counts[bin(depth[i])]++;
 let start=0;for(let b=0;b<65536;b++){const count=counts[b];counts[b]=start;start+=count;}
 for(let i=0;i<n;i++)order[counts[bin(depth[i])]++]=i;
 const center=attributes.center.array,ca=attributes.ca.array,cb=attributes.cb.array,tint=attributes.tint.array,alpha=attributes.opacity.array,sid=attributes.shIndex.array,sem=$('#appearance').value==='semantic',palette=new Map();
 for(let k=0;k<n;k++){const i=order[k]*14,j=k*3;sid[k]=order[k];alpha[k]=data[i+6];for(let d=0;d<3;d++){center[j+d]=data[i+d];ca[j+d]=data[i+7+d];cb[j+d]=data[i+10+d];tint[j+d]=data[i+3+d];}if(sem){const id=data[i+13];if(!palette.has(id))palette.set(id,semantic(id).toArray());const c=palette.get(id);tint[j]=c[0];tint[j+1]=c[1];tint[j+2]=c[2];}}
 mesh.material.uniforms.useSH.value=shTexture&&$('#appearance').value==='rgb'?1:0;for(const a of Object.values(attributes))a.needsUpdate=true;state.sortMs=performance.now()-sortAt;
}
function recordedPose(){const tr=report.trajectory,p=tr.xyz[recordIndex],R=tr.R[recordIndex];camera.position.fromArray(p);const lon=state.lon||0,lat=state.lat||0;const v=[Math.sin(lon)*Math.cos(lat),Math.sin(lat),Math.cos(lon)*Math.cos(lat)];const dir=new THREE.Vector3(...R.map(row=>row.reduce((s,x,i)=>s+x*v[i],0)));camera.up.set(-R[0][1],-R[1][1],-R[2][1]);camera.lookAt(camera.position.clone().add(dir));controls.target.copy(camera.position).add(dir);}
function seek(t){state.time=Math.max(report.frame_times[0],Math.min(report.frame_times.at(-1),t));recordIndex=nearest(report.trajectory.t,state.time);frameIndex=nearest(report.frame_times,state.time);$('#seek').value=(state.time-report.frame_times[0])/(report.frame_times.at(-1)-report.frame_times[0]);$('#time').textContent=` ${(state.time-report.frame_times[0]).toFixed(1)} s`;if(frameIndex!==lastImage){$('#frame').src=window.__frameURL?window.__frameURL(frameIndex):`/api/frame/${frameIndex}.jpg`;lastImage=frameIndex;}if(state.follow)recordedPose();}
function inspect(o){$('#objects').value=o.id;const el=$('#inspect');el.replaceChildren();const text=document.createElement('p');text.textContent=`${o.label} #${o.id} · ${o.status} · ${(o.confidence*100).toFixed(0)}%\nPosition: ${o.position.map(x=>x.toFixed(2)).join(', ')}\n${o.n_observations} observations. Extent is visible geometry only.`;el.append(text);for(const ob of o.observations.slice(0,10)){const b=document.createElement('button');b.textContent=`${(ob.t-report.frame_times[0]).toFixed(1)}s`;b.onclick=()=>seek(ob.t);el.append(b);}}
function drawMap(){const c=$('#map'),ctx=c.getContext('2d');ctx.fillStyle='#08111a';ctx.fillRect(0,0,c.width,c.height);const sx=(c.width-28)/(bounds[1][0]-bounds[0][0]),sy=(c.height-28)/(bounds[1][1]-bounds[0][1]),s=Math.min(sx,sy);const map=(x,y)=>[14+(x-bounds[0][0])*s,c.height-14-(y-bounds[0][1])*s];state.mapTransform={s,x:bounds[0][0],y:bounds[0][1]};ctx.fillStyle='#304955';for(const p of mapPoints){const q=map(...p);ctx.fillRect(q[0],q[1],1.5,1.5);}ctx.strokeStyle='#e3a65b';ctx.beginPath();report.trajectory.xyz.forEach((p,i)=>{const q=map(...p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();for(const [p,col]of [[report.trajectory.xyz[recordIndex],'#ffb65e'],[camera.position.toArray(),'#71ffdc']]){const q=map(...p);ctx.beginPath();ctx.arc(...q,6,0,7);ctx.fillStyle=col;ctx.fill();}const q=map(camera.position.x,camera.position.y),dir=new THREE.Vector3();camera.getWorldDirection(dir);ctx.strokeStyle='#71ffdc';ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(q[0]+dir.x*25,q[1]-dir.y*25);ctx.stroke();}
function setup(){
 const tr=report.trajectory;bounds=[new Array(3).fill(Infinity),new Array(3).fill(-Infinity)];for(const p of tr.xyz)for(let j=0;j<3;j++){bounds[0][j]=Math.min(bounds[0][j],p[j]-3);bounds[1][j]=Math.max(bounds[1][j],p[j]+3);}for(let i=0;i<data.length;i+=14*8)mapPoints.push([data[i],data[i+1]]);
 home=new THREE.Vector3((bounds[0][0]+bounds[1][0])/2,(bounds[0][1]+bounds[1][1])/2,bounds[1][2]+10);camera.position.copy(home);controls.target.set(home.x,home.y,0);controls.update();
 const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(tr.xyz.map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color:0xf0b967}));scene.add(line);
 for(const o of report.objects.filter(o=>o.status!=='dynamic')){const box=new THREE.Mesh(new THREE.BoxGeometry(...o.observed_extent_m.map(v=>Math.max(.15,v))),new THREE.MeshBasicMaterial({color:semantic(o.id),wireframe:true}));box.position.fromArray(o.position);box.userData.object=o;objectsGroup.add(box);const option=document.createElement('option');option.value=o.id;option.textContent=`${o.label} #${o.id} (${o.status})`;$('#objects').append(option);}
 $('#objects').onchange=e=>{const o=report.objects.find(o=>o.id===Number(e.target.value));if(o)inspect(o);};$('#boxes').onchange=e=>objectsGroup.visible=e.target.checked;
 $('#walk').onclick=()=>{state.walk=!state.walk;state.follow=false;controls.enabled=!state.walk;$('#follow').textContent='Follow video: off';$('#walk').textContent=state.walk?'Exit walk mode':'Enter walk mode';};
 $('#home').onclick=()=>{state.follow=false;state.walk=false;controls.enabled=true;camera.up.set(0,0,1);camera.position.copy(home);controls.target.set(home.x,home.y,0);$('#walk').textContent='Enter walk mode';$('#follow').textContent='Follow video: off';};
 $('#jump').onclick=()=>{recordedPose();};$('#follow').onclick=()=>{state.follow=!state.follow;state.walk=false;controls.enabled=!state.follow;$('#follow').textContent=`Follow video: ${state.follow?'on':'off'}`;$('#walk').textContent='Enter walk mode';if(state.follow)recordedPose();};
 $('#play').onclick=()=>{state.playing=!state.playing;$('#play').textContent=state.playing?'Pause':'Play';};$('#seek').oninput=e=>seek(report.frame_times[0]+Number(e.target.value)*(report.frame_times.at(-1)-report.frame_times[0]));
 $('#map').onclick=e=>{const c=$('#map'),r=c.getBoundingClientRect(),m=state.mapTransform,x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height,wx=(x-14)/m.s+m.x,wy=(c.height-14-y)/m.s+m.y;const hits=[];let last=-Infinity;for(let i=0;i<tr.t.length;i++){const p=tr.xyz[i];if(Math.hypot(p[0]-wx,p[1]-wy)<.8&&tr.t[i]-last>3){hits.push(i);last=tr.t[i];}}$('#visits').replaceChildren();for(const i of hits){const b=document.createElement('button');b.textContent=`Visit ${(tr.t[i]-report.frame_times[0]).toFixed(1)}s`;b.onclick=()=>seek(tr.t[i]);$('#visits').append(b);}if(hits.length===1)seek(tr.t[hits[0]]);};
 renderer.domElement.addEventListener('pointerdown',e=>{if(state.walk||state.follow){drag=[e.clientX,e.clientY];renderer.domElement.setPointerCapture(e.pointerId);}});
 renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;const dx=(e.clientX-drag[0])*.004,dy=(e.clientY-drag[1])*.004;drag=[e.clientX,e.clientY];if(state.follow){state.lon=(state.lon||0)-dx;state.lat=Math.max(-1.4,Math.min(1.4,(state.lat||0)-dy));recordedPose();}else{camera.rotateOnWorldAxis(new THREE.Vector3(0,0,1),-dx);camera.rotateX(-dy);}});
 renderer.domElement.addEventListener('pointerup',()=>drag=null);
 renderer.domElement.addEventListener('dblclick',e=>{const r=renderer.domElement.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),camera);const hit=ray.intersectObjects(objectsGroup.children)[0];if(hit)inspect(hit.object.userData.object);});
 window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;keys[e.code]=true;if(e.code==='Escape'&&state.walk)$('#walk').click();});window.addEventListener('keyup',e=>keys[e.code]=false);window.addEventListener('blur',()=>keys={});
 if(report.method&&Number.isFinite(report.after?.held_out?.ssim)){const h=report.after.held_out;$('#metrics').textContent=`${state.gaussians.toLocaleString()} Gaussians shown${report.gaussians_full?` (of ${report.gaussians_full.toLocaleString()})`:''} · held-out views: PSNR ${h.psnr_db.toFixed(2)} dB, SSIM ${h.ssim.toFixed(3)} · offline processing ${(report.total_s/60).toFixed(1)} min. Units: ${report.units}.`;const sc=$('#scope');if(sc)sc.textContent=`${report.method}. Poses: ${report.poses}. `+(report.limitations||[]).join(' ');}
 else $('#metrics').textContent=`${state.gaussians.toLocaleString()} Gaussians · ${report.total_s.toFixed(1)}s additional CPU processing · held-out appearance PSNR ${report.after.held_out.psnr_db.toFixed(2)}dB. Units: ${report.units}.`;
 seek(report.trajectory.t[0]);state.seek=seek;state.camera=camera;state.ready=true;state.report=report;
}
let previous=performance.now(),fpsAt=previous,count=0;
function loop(now){requestAnimationFrame(loop);const elapsed=(now-previous)/1000,dt=Math.min(elapsed,.1);previous=now;if(!state.ready)return;const w=V.clientWidth,h=V.clientHeight;if(renderer.domElement.clientWidth!==w||renderer.domElement.clientHeight!==h||camera.aspect!==w/h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}mesh.material.uniforms.viewport.value.set(w*renderer.getPixelRatio(),h*renderer.getPixelRatio());if(state.playing){seek(state.time+elapsed*Number($('#speed').value));if(state.time>=report.frame_times.at(-1)){state.playing=false;$('#play').textContent='Play';}}if(state.walk){const speed=dt*(keys.ShiftLeft?6:2),f=new THREE.Vector3();camera.getWorldDirection(f);f.z=0;f.normalize();const right=f.clone().cross(new THREE.Vector3(0,0,1));camera.position.addScaledVector(f,speed*((!!keys.KeyW)-(!!keys.KeyS)));camera.position.addScaledVector(right,speed*((!!keys.KeyD)-(!!keys.KeyA)));camera.position.z+=speed*((!!keys.KeyE)-(!!keys.KeyQ));controls.target.copy(camera.position).add(camera.getWorldDirection(new THREE.Vector3()));}else if(!state.follow)controls.update();sort();renderer.render(scene,camera);drawMap();count++;state.frames++;if(now-fpsAt>1000){state.fps=count*1000/(now-fpsAt);count=0;fpsAt=now;}$('#hud').textContent=`${state.walk?'WALK':state.follow?'RECORDED CAMERA':'ORBIT'} · ${state.fps.toFixed(0)} rendering fps\nVirtual camera ${camera.position.toArray().map(x=>x.toFixed(2)).join(', ')} · ${report.units}`;}
async function main(){const r=await fetch('/api/gaussians');if(!r.ok)throw Error('Gaussian scene unavailable. Run: slam3d gaussians RUN_DIR');report=await r.json();const binary=await fetch('/api/gaussians.bin');if(!binary.ok)throw Error('Scene binary unavailable');data=new Float32Array(await binary.arrayBuffer());if(data.length!==report.gaussians*14)throw Error('Invalid scene buffer length');if(report.browser_sh_degree===3){const response=await fetch('/api/gaussians.sh.bin');if(!response.ok)throw Error('SH appearance unavailable');const sh=new Float32Array(await response.arrayBuffer());if(sh.length!==report.gaussians*64)throw Error('Invalid SH buffer');const width=Math.min(4096,renderer.capabilities.maxTextureSize),height=Math.ceil(sh.length/4/width);if(height>renderer.capabilities.maxTextureSize)throw Error('Scene exceeds GPU texture capacity');const padded=new Float32Array(width*height*4);padded.set(sh);shTexture=new THREE.DataTexture(padded,width,height,THREE.RGBAFormat,THREE.FloatType);shTexture.needsUpdate=true;shDimensions.set(width,height);}build();setup();requestAnimationFrame(loop);}
main().catch(e=>{state.errors.push(String(e));$('#error').textContent=String(e);$('#hud').textContent='Could not load Gaussian world';console.error(e);});
