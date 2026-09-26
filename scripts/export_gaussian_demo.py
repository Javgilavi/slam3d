"""Bundle the Gaussian viewer into one local HTML file, with no local server.

Uses keyframe-only thumbnails (explicitly labelled), embedded three.js, scene and
metadata. No external uploads, network imports, or floor-plan dependencies.
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from slam3d.config import REPO
from slam3d.io.tum import read_tum


def export(rdir):
    rdir = Path(rdir).resolve()
    out = rdir / 'gaussians'
    report = json.loads((out/'report.json').read_text())
    t, T = read_tum(rdir/'geometry/trajectory_grav.tum')
    report['trajectory']={'t':t.tolist(),'xyz':(T[:,:3,3]*report['metres_per_slam_unit']).tolist(),'R':T[:,:3,:3].tolist()}
    ft = np.loadtxt(rdir/'ingest/frames.csv',delimiter=',',skiprows=1,usecols=(1,))
    report['frame_times']=ft.tolist()
    kf = np.atleast_2d(np.loadtxt(rdir/'geometry/keyframes_grav.csv',delimiter=',',skiprows=1))
    thumbs=[]
    for row in kf:
        i=int(np.argmin(abs(ft-row[1])))
        img=cv2.imread(str(rdir/f'ingest/frames/{i:06d}.jpg'))
        img=cv2.resize(img,(480,240),interpolation=cv2.INTER_AREA)
        ok, buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,65])
        if not ok: raise RuntimeError('Thumbnail encoding failed')
        thumbs.append([i,float(ft[i]),'data:image/jpeg;base64,'+base64.b64encode(buf).decode()])
    static=REPO/'viewer/static'
    html=(static/'world.html').read_text()
    modules={'three':'data:text/javascript;base64,'+base64.b64encode((static/'vendor/three.module.js').read_bytes()).decode(),
             'three/addons/OrbitControls.js':'data:text/javascript;base64,'+base64.b64encode((static/'vendor/OrbitControls.js').read_bytes()).decode()}
    old='<script type="importmap">{"imports":{"three":"/static/vendor/three.module.js","three/addons/":"/static/vendor/"}}</script>'
    html=html.replace(old,'<script type="importmap">'+json.dumps({'imports':modules})+'</script>')
    html=html.replace('<a href="/">Synchronized viewer</a>','<small>Standalone · keyframe image previews</small>')
    binary=base64.b64encode((out/'scene.bin').read_bytes()).decode()
    sh_binary=base64.b64encode((out/'scene.sh.bin').read_bytes()).decode() if (out/'scene.sh.bin').exists() else ''
    bootstrap='''<script>
const embeddedReport=REPORT, embeddedScene="BINARY", embeddedSH="SHDATA", thumbs=THUMBS;
window.fetch=async function(url){
 if(url==='/api/gaussians')return new Response(JSON.stringify(embeddedReport),{status:200});
 if(url==='/api/gaussians.bin'){const a=Uint8Array.from(atob(embeddedScene),c=>c.charCodeAt(0));return new Response(a,{status:200});}
 if(url==='/api/gaussians.sh.bin'){const a=Uint8Array.from(atob(embeddedSH),c=>c.charCodeAt(0));return new Response(a,{status:200});}
 throw new Error('Standalone demo does not request external resources: '+url);
};
window.__frameURL=function(index){const t=thumbs.reduce((best,t)=>Math.abs(t[0]-index)<Math.abs(best[0]-index)?t:best,thumbs[0]);document.querySelector('#frame').title='Keyframe preview at '+(t[1]-embeddedReport.frame_times[0]).toFixed(2)+'s (not frame-exact video)';return t[2];};
</script>'''.replace('REPORT',json.dumps(report).replace('</','<\\/')).replace('BINARY',binary).replace('SHDATA',sh_binary).replace('THUMBS',json.dumps(thumbs))
    app=(static/'world.js').read_text()
    app_url='data:text/javascript;base64,'+base64.b64encode(app.encode()).decode()
    html=html.replace('<script type="module" src="/static/world.js"></script>',bootstrap+'<script type="module" src="'+app_url+'"></script>')
    path=out/'demo.html';path.write_text(html)
    print(f'{path} ({path.stat().st_size/2**20:.2f} MiB); {len(thumbs)} keyframe previews')
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');export(parser.parse_args().run_dir)
