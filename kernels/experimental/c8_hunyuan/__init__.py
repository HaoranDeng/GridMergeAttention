"""GMA-C8 ported to the Hunyuan Video 13B shape [1,33,30,45,24,128].

Round-0 (get-it-running) port from kernels/best/c8_r2 (the Wan
[1,21,45,80,40,128] shape). See gma_c8h_r0.py's module docstring and the
four .cu files here for the full constant-by-constant mapping and the
engineering choices made where the new shape doesn't divide evenly.

No integrity gate (unlike kernels/best/*): this has not been screened for
numerical fidelity, only smoke-tested for shape/finiteness.
"""
from .gma_c8h_r0 import create, NativeC8Hunyuan

__all__ = ["create", "NativeC8Hunyuan"]
