"""CPU-only compiler preparation; never allocates a CUDA tensor."""
import argparse
import json
from attention.axis_current_c8_cuda import _compiled


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args()
    configurations=[(2,80,45,1,1),(21,80,45,1,1)]
    if a.full:configurations.append((21,80,45,40,4))
    for shape in configurations:
        for implementation in ('r2','slow_reference'):
            print(json.dumps({'compiling':shape,'implementation':implementation}),flush=True)
            _compiled(*shape,implementation)
    print(json.dumps({'compiled':True}),flush=True)


if __name__=='__main__':main()
