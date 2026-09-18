"""Optional CUDA-event timing for Wan self-attention benchmarks."""

from __future__ import annotations

import torch


class AttentionRuntimeRecorder:
    def __init__(self):
        self.active = False
        self.core_events = []
        self.full_events = []

    def begin_video(self):
        self.core_events.clear()
        self.full_events.clear()
        self.active = True

    def start(self):
        if not self.active:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _stop(start, events):
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        events.append((start, end))

    def stop_core(self, start):
        self._stop(start, self.core_events)

    def stop_full(self, start):
        self._stop(start, self.full_events)

    def finish_video(self):
        torch.cuda.synchronize()
        self.active = False
        return {
            "attention_total_ms": sum(
                start.elapsed_time(end) for start, end in self.core_events
            ),
            "attention_calls": len(self.core_events),
            "full_attention_total_ms": sum(
                start.elapsed_time(end) for start, end in self.full_events
            ),
            "full_attention_calls": len(self.full_events),
        }


ATTENTION_RUNTIME_RECORDER = AttentionRuntimeRecorder()

