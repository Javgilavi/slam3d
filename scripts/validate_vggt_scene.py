"""Validate exported Gaussians and actual standalone browser rendering/navigation."""
import argparse
import json
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser();p.add_argument('scene',type=Path);a=p.parse_args()
    report=json.loads((a.scene/'report.json').read_text())
    packed=np.fromfile(a.scene/'scene.bin',dtype='<f4').reshape(-1,14)
    z=np.load(a.scene/'scene.npz')
    assert len(packed)==report['gaussians'] and np.isfinite(packed).all()
    assert np.linalg.eigvalsh(z['covariance']).min()>0
    assert ((z['opacity']>=0)&(z['opacity']<=1)).all()
    assert np.allclose(packed[:,:3],z['xyz']) and np.allclose(packed[:,3:6],z['rgb'])
    errors=[]
    with sync_playwright() as pw:
        b=pw.chromium.launch(headless=True,args=['--enable-webgl','--ignore-gpu-blocklist','--use-gl=angle','--use-angle=gl-egl','--enable-gpu'])
        page=b.new_page(viewport={'width':1280,'height':800})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
        page.goto((a.scene/'demo.html').resolve().as_uri())
        page.wait_for_function('window.__vggt?.ready',timeout=60000)
        page.wait_for_timeout(1000)
        before=page.evaluate('window.__vggt.camera.position.toArray()')
        page.keyboard.down('w');page.wait_for_timeout(700);page.keyboard.up('w')
        after=page.evaluate('window.__vggt.camera.position.toArray()')
        assert np.linalg.norm(np.array(after)-before)>.01
        page.click('#reset')
        page.evaluate('window.__vggt.recorded(Math.floor(window.__vggt.frames/2))')
        page.wait_for_timeout(1000)
        page.screenshot(path=str(a.scene/'browser_world.png'))
        # Square canvas, same recorded camera: inspect only actual rendered pixels.
        page.evaluate("document.querySelector('#view').style.cssText='width:384px;height:384px;flex:none'")
        page.evaluate('window.__vggt.recorded(0)')
        page.wait_for_timeout(600)
        page.locator('#view canvas').screenshot(path=str(a.scene/'browser_render.png'))
        renderer=page.evaluate("(()=>{let g=window.__vggt.renderer.getContext(),e=g.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)})()")
        result=dict(artifacts='passed',browser='passed' if not errors else 'failed',errors=errors,renderer=renderer,rendering_fps=page.evaluate('window.__vggt.fps'),walk_displacement=float(np.linalg.norm(np.array(after)-before)),gaussians=len(packed),note='Browser FPS belongs to the renderer named above, not reconstruction throughput.')
        b.close()
    (a.scene/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
