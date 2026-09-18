"""Experimental graph replay of the complete native Fisher operator.

Each instance owns its workspace and is restricted to one caller stream.
Replay includes three input copies and an independent output copy. Captured
selection is executed on every call; no seeds or attention results are cached.
"""
import torch


class GraphFisherAttention:
    def __init__(self, operator, q, k, v, scale=128**-.5):
        self.operator = operator
        self.scale = scale
        self.static = [torch.empty_like(x) for x in (q, k, v)]
        for target, source in zip(self.static, (q, k, v)):
            target.copy_(source)
        self.stream = torch.cuda.current_stream(q.device)
        warmup = torch.cuda.Stream(device=q.device)
        warmup.wait_stream(self.stream)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                operator(*self.static, scale=scale)
        self.stream.wait_stream(warmup)
        torch.cuda.synchronize(q.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = operator(*self.static, scale=scale)

    @torch.no_grad()
    def __call__(self, q, k, v):
        if torch.cuda.current_stream(q.device) != self.stream:
            raise ValueError('Graph instance must remain on its original caller stream')
        for target, source in zip(self.static, (q, k, v)):
            if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device or not source.is_contiguous():
                raise ValueError('Graph replay requires matching contiguous native Q/K/V')
        for target, source in zip(self.static, (q, k, v)):
            target.copy_(source)
        self.graph.replay()
        return self.output.clone()
