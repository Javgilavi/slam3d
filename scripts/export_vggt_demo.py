"""Standalone navigable viewer for a trained VGGT Gaussian scene."""
import argparse
import base64
import json
import re
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('scene',type=Path);a=p.parse_args()
    report=json.loads((a.scene/'report.json').read_text())
    rec=json.loads((Path(report['source'])/'reconstruction.json').read_text())
    static=Path('viewer/static')
    # Reuse the existing projected covariance shaders, not a point renderer.
    js=(static/'world.js').read_text()
    vertex=re.search(r'const vertex=`(.*?)`;',js,re.S).group(1)
    fragment=re.search(r'const fragment=`(.*?)`;',js,re.S).group(1)
    encode=lambda data:base64.b64encode(data).decode()
    modules={'three':'data:text/javascript;base64,'+encode((static/'vendor/three.module.js').read_bytes()),'three/addons/OrbitControls.js':'data:text/javascript;base64,'+encode((static/'vendor/OrbitControls.js').read_bytes())}
    data=dict(scene=encode((a.scene/'scene.bin').read_bytes()),frames=rec['frames'],images=['data:image/jpeg;base64,'+encode(Path(f['image']).read_bytes()) for f in rec['frames']],report={k:report[k] for k in ['gaussians','after','units','limitations']})
    html='''<!doctype html><meta charset="utf-8"><title>Construction · VGGT Gaussian reconstruction</title>
<style>body{margin:0;background:#10171e;color:#eee;font:14px system-ui}header{padding:12px}#layout{display:flex;height:82vh}#view{width:75%;position:relative}#reference{width:25%;padding:8px}img{width:100%}button,input{margin:8px}#hud{position:absolute;bottom:12px;left:12px;white-space:pre}small{color:#b9c5cd}</style>
<header><b>Construction · Gaussian reconstruction</b><button id="reset">Recorded view</button><button id="play">Play views</button><input id="seek" type="range" min="0" value="0"><span id="label"></span><br><small>Drag to orbit · right-drag to pan · scroll to move closer · WASD to move · Q/E down/up. One camera direction; unseen areas are incomplete. Scale is arbitrary.</small></header>
<div id="layout"><div id="view"><div id="hud"></div></div><div id="reference">Recorded image<img id="image"><p id="metrics"></p><small>Playback uses processed camera views. It is not live reconstruction. Semantic labels have not been transferred to this map.</small></div></div>
<script type="importmap">IMPORTS</script><script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/OrbitControls.js';
const D=DATA,vertex=VERTEX,fragment=FRAGMENT,$=s=>document.querySelector(s),view=$('#view');
const renderer=new THREE.WebGLRenderer({antialias:false,preserveDrawingBuffer:true});renderer.setPixelRatio(1);view.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x000000);const camera=new THREE.PerspectiveCamera(90,1,.001,1000);camera.up.set(0,-1,0);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
const a=new Float32Array(Uint8Array.from(atob(D.scene),c=>c.charCodeAt(0)).buffer),n=a.length/14;
const g=new THREE.InstancedBufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));g.setIndex([0,1,2,0,2,3]);g.instanceCount=n;
const attrs={};for(const [name,size] of [['center',3],['tint',3],['ca',3],['cb',3],['opacity',1]]){attrs[name]=new THREE.InstancedBufferAttribute(new Float32Array(n*size),size);g.setAttribute(name,attrs[name]);}
const mat=new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,uniforms:{viewport:{value:new THREE.Vector2()}},transparent:true,depthWrite:false,side:THREE.DoubleSide});const mesh=new THREE.Mesh(g,mat);mesh.frustumCulled=false;scene.add(mesh);
let previous='',index=0,playing=false,lastStep=0,keys={},fps=0,count=0,epoch=performance.now();
function sort(){camera.updateMatrixWorld();const sig=camera.matrixWorld.elements.join(',');if(sig===previous)return;previous=sig;const m=camera.matrixWorldInverse.elements,order=Array.from({length:n},(_,i)=>i);order.sort((i,j)=>(m[2]*a[i*14]+m[6]*a[i*14+1]+m[10]*a[i*14+2])-(m[2]*a[j*14]+m[6]*a[j*14+1]+m[10]*a[j*14+2]));for(let k=0;k<n;k++){const i=order[k]*14;attrs.center.array.set(a.subarray(i,i+3),k*3);attrs.tint.array.set(a.subarray(i+3,i+6),k*3);attrs.opacity.array[k]=a[i+6];attrs.ca.array.set(a.subarray(i+7,i+10),k*3);attrs.cb.array.set(a.subarray(i+10,i+13),k*3);}Object.values(attrs).forEach(v=>v.needsUpdate=true);}
function recorded(i){index=i;const f=D.frames[i],m=new THREE.Matrix4().set(...f.c2w.flat());m.multiply(new THREE.Matrix4().makeScale(1,-1,-1));camera.position.setFromMatrixPosition(m);camera.quaternion.setFromRotationMatrix(m);camera.up.set(-f.c2w[0][1],-f.c2w[1][1],-f.c2w[2][1]);const dir=camera.getWorldDirection(new THREE.Vector3());controls.target.copy(camera.position).add(dir);camera.fov=2*Math.atan(518/(2*f.K[1][1]))*180/Math.PI;camera.updateProjectionMatrix();controls.update();$('#seek').value=i;$('#image').src=D.images[i];$('#label').textContent=`View ${i+1}/${D.frames.length}`;}
$('#seek').max=D.frames.length-1;$('#seek').oninput=e=>recorded(+e.target.value);$('#reset').onclick=()=>recorded(index);$('#play').onclick=()=>{playing=!playing;$('#play').textContent=playing?'Pause':'Play views';};
window.addEventListener('keydown',e=>keys[e.code]=true);window.addEventListener('keyup',e=>keys[e.code]=false);window.addEventListener('blur',()=>keys={});
$('#metrics').textContent=`${n.toLocaleString()} Gaussians. Held-out appearance: ${D.report.after.held_out.psnr_db.toFixed(2)} dB, SSIM ${D.report.after.held_out.ssim.toFixed(3)}.`;
let then=performance.now();function loop(now){requestAnimationFrame(loop);const dt=Math.min((now-then)/1000,.1);then=now;const w=view.clientWidth,h=view.clientHeight;if(renderer.domElement.width!==w||renderer.domElement.height!==h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}mat.uniforms.viewport.value.set(w,h);if(playing&&now-lastStep>500){recorded((index+1)%D.frames.length);lastStep=now;}const f=camera.getWorldDirection(new THREE.Vector3()),r=new THREE.Vector3().crossVectors(f,camera.up).normalize(),move=f.multiplyScalar((!!keys.KeyW)-(!!keys.KeyS)).addScaledVector(r,(!!keys.KeyD)-(!!keys.KeyA)).addScaledVector(camera.up,(!!keys.KeyE)-(!!keys.KeyQ)).multiplyScalar(dt*.3);camera.position.add(move);controls.target.add(move);controls.update();sort();renderer.render(scene,camera);count++;if(now-epoch>1000){fps=count*1000/(now-epoch);epoch=now;count=0;}$('#hud').textContent=`${fps.toFixed(0)} rendering FPS\nCamera ${camera.position.toArray().map(v=>v.toFixed(3)).join(', ')} · arbitrary units`;window.__vggt.fps=fps;}
recorded(0);window.__vggt={ready:true,camera,recorded,renderer,scene,frames:D.frames.length};requestAnimationFrame(loop);
</script>'''
    html=html.replace('IMPORTS',json.dumps({'imports':modules})).replace('DATA',json.dumps(data)).replace('VERTEX',json.dumps(vertex)).replace('FRAGMENT',json.dumps(fragment))
    (a.scene/'demo.html').write_text(html)
    print(a.scene/'demo.html')


if __name__=='__main__':main()
