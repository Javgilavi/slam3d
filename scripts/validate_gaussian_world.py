"""Validate artifact/API contracts and attempt real-browser interaction checks.

Route handlers are checked directly, so they work without socket permission.
Browser-launch failures are reported as blocked, never as passed checks.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from slam3d.viewer.server import create_app


def check_api(rdir):
    app=create_app(rdir)
    routes={r.path:r.endpoint for r in app.routes if hasattr(r,'endpoint')}
    r=routes['/api/gaussians']()
    a=np.fromfile(routes['/api/gaussians.bin']().path,'<f4').reshape(-1,14)
    assert len(a)==r['gaussians'] and np.isfinite(a).all()
    C=np.empty((len(a),3,3));C[:,0,0]=a[:,7];C[:,0,1]=C[:,1,0]=a[:,8];C[:,0,2]=C[:,2,0]=a[:,9];C[:,1,1]=a[:,10];C[:,1,2]=C[:,2,1]=a[:,11];C[:,2,2]=a[:,12]
    assert np.linalg.eigvalsh(C).min()>0 and ((a[:,6]>=0)&(a[:,6]<=1)).all()
    world=Path(routes['/world']().path);assert world.is_file()
    for name in ['world.js','vendor/three.module.js','vendor/OrbitControls.js']:
        assert (world.parent/name).is_file()
    assert Path(routes['/api/frame/{idx}.jpg'](0).path).is_file()
    assert len(r['trajectory']['t'])==len(r['trajectory']['xyz'])==len(r['trajectory']['R'])
    assert np.diff(r['trajectory']['t']).min()>0
    return {'artifact_finite':True,'covariance_positive_definite':True,'opacity_valid':True,'route_handlers_and_files':True,'trajectory_contract':True,'gaussians':len(a),'http_transport_tested':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--no-browser',action='store_true');a=p.parse_args();rdir=Path(a.run_dir).resolve();out=rdir/'gaussians'
    result={'api':check_api(rdir),'browser':{'state':'pending'}}
    if a.no_browser:
        result['browser']={'state':'not_run','reason':'--no-browser selected; this session separately observed Chromium sandbox_host_linux.cc shutdown Operation not permitted, and socket() PermissionError'}
        (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b=pw.chromium.launch(headless=True,args=['--enable-gpu','--ignore-gpu-blocklist','--use-gl=angle','--use-angle=gl-egl'])
            page=b.new_page(viewport={'width':1440,'height':900});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
            page.goto((out/'demo.html').as_uri());page.wait_for_function('window.__world && window.__world.ready', timeout=60000)
            page.wait_for_timeout(1500)
            page.click('#walk');before=page.evaluate('window.__world.camera.position.toArray()')
            page.keyboard.down('w');page.wait_for_timeout(700);page.keyboard.up('w')
            after=page.evaluate('window.__world.camera.position.toArray()');assert np.linalg.norm(np.array(after)-before)>.2
            page.click('#follow');page.click('#play');t0=page.evaluate('window.__world.time');page.wait_for_timeout(1500);assert page.evaluate('window.__world.time')>t0+1
            page.click('#play');page.select_option('#appearance','semantic');page.select_option('#objects',index=1);assert page.locator('#inspect').inner_text()
            page.click('#home');page.wait_for_timeout(800);page.screenshot(path=str(out/'world_screenshot.png'))
            result['browser']={'state':'passed' if not errors else 'failed','errors':errors,'fps':page.evaluate('window.__world.fps'),'walk_displacement':float(np.linalg.norm(np.array(after)-before))}
            b.close()
    except Exception as e:
        result['browser']={'state':'blocked_or_failed','error':str(e)}
    (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
