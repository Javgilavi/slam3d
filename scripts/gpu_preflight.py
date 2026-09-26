"""Check actual GPU execution, without changing drivers or system settings."""
import argparse
import glob
import json
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/gpu_preflight.json');a=p.parse_args()
    report={'device_nodes':glob.glob('/dev/nvidia*'),'render_nodes':glob.glob('/dev/dri/*')}
    path=Path('/proc/driver/nvidia/version')
    report['kernel_driver']=path.read_text() if path.exists() else None
    for name,cmd in [('nvidia_smi',['nvidia-smi']),('nvcc',['nvcc','--version'])]:
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            report[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except (OSError,subprocess.TimeoutExpired) as e:report[name]={'error':str(e)}
    import torch
    report.update(torch_version=torch.__version__,torch_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
    try:
        x=torch.ones((512,512),device='cuda');y=x@x;torch.cuda.synchronize()
        assert torch.allclose(y,torch.full_like(y,512))
        report.update(execution_passed=True,gpu=torch.cuda.get_device_name(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,capability=torch.cuda.get_device_capability())
    except Exception as e:report.update(execution_passed=False,execution_error=str(e))
    if not report['device_nodes'] and report['kernel_driver']:
        report['diagnosis']='NVIDIA driver loaded, but no NVIDIA device nodes visible in this execution environment. GPU access must be provided by the environment; package reinstall cannot expose hardware.'
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 0 if report['execution_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
