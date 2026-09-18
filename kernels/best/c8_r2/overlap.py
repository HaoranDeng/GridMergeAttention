"""Two-stream merge/route pipeline for the exact-capacity Fisher C8 back half.

Chunk ``i + 1`` can merge while chunk ``i`` routes because the two operations
use different compressed workspaces.  The schedule therefore alternates two
``mk``/``mv``/``bias`` buffer sets and orders their reuse with CUDA events.

This changes only scheduling and workspace count.  Each route still consumes
exactly eight merged K/V representatives; there is no singleton, residual, raw
K/V, or ninth route token.
"""

from __future__ import annotations

import torch


class MergeRouteOverlap:
    """Run an exact-C8 merge/route pair with double-buffered CUDA streams."""

    def __init__(self, backend, merger=None, router=None):
        self.backend = backend
        self.merger = backend.merge if merger is None else merger
        self.router = backend.route if router is None else router
        self.workspaces = [
            (backend.mk, backend.mv, backend.bias),
            (
                torch.empty_like(backend.mk),
                torch.empty_like(backend.mv),
                torch.empty_like(backend.bias),
            ),
        ]
        self.merge_stream = torch.cuda.Stream(device=backend.device)
        self.route_stream = torch.cuda.Stream(device=backend.device)
        self.inputs_ready = torch.cuda.Event()
        self.merge_done = [torch.cuda.Event() for _ in self.workspaces]
        self.route_done = [torch.cuda.Event() for _ in self.workspaces]
        self.finished = torch.cuda.Event()

    @property
    def extra_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.workspaces[1]
        )

    def run(self, q, k, v, seeds, output, scale=128 ** -0.5):
        """Merge and route all 40 heads after selector and pack have run."""
        backend = self.backend
        current = torch.cuda.current_stream(device=backend.device)
        self.inputs_ready.record(current)
        self.merge_stream.wait_event(self.inputs_ready)
        self.route_stream.wait_event(self.inputs_ready)

        # At the start of a call both workspaces are reusable once the caller's
        # stream reaches this point.  Within the call route_done protects reuse
        # of a workspace two chunks later.
        for done in self.route_done:
            done.record(current)

        for index, head in enumerate(range(0, 40, 5)):
            slot = index & 1
            mk, mv, bias = self.workspaces[slot]
            self.merge_stream.wait_event(self.route_done[slot])
            with torch.cuda.stream(self.merge_stream):
                self.merger.run(
                    backend.packed_q,
                    k,
                    v,
                    seeds,
                    mk,
                    mv,
                    bias,
                    head,
                    5,
                    105,
                    scale,
                    backend.empty,
                    0,
                    1,
                )
                self.merge_done[slot].record(self.merge_stream)

            self.route_stream.wait_event(self.merge_done[slot])
            with torch.cuda.stream(self.route_stream):
                self.router.run(q, mk, mv, bias, output, head, 5, scale)
                self.route_done[slot].record(self.route_stream)

        self.finished.record(self.route_stream)
        current.wait_event(self.finished)
        return output
