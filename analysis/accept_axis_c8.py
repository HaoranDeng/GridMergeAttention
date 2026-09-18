"""Run full-call speed and multi-head checks inside one exclusive GPU guard."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from analysis.axis_c8_speed_gate import environment,source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--full-heads',type=Path,required=True)
    a=p.parse_args();v=json.loads(a.validation.read_text())
    assert v['passed'] and v['environment']==environment() and v['source_sha256']==source_hashes()
    subprocess.run([sys.executable,'-m','analysis.benchmark_axis_c8','--output',str(a.benchmark),'--profile'],check=True)
    result=json.loads(a.benchmark.read_text())
    if not result['passed']:
        print(json.dumps({'speed_passed':False,'next':'tune'}),flush=True)
        return
    subprocess.run([sys.executable,'-m','analysis.check_axis_c8_full_heads','--output',str(a.full_heads)],check=True)
    print(json.dumps({'speed_passed':True,'full_heads_passed':True}),flush=True)


if __name__=='__main__':main()
