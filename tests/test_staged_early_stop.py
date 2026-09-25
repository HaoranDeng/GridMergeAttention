import json

from analysis.evaluate_staged_early_stop import Gate, gate_decision, runtime_latency


def metrics(psnr, ssim, lpips):
    return {
        "rgb": {"summary": {"psnr": psnr, "ssim": ssim}},
        "lpips": {"summary": {"lpips": lpips}},
    }


def test_gate_promotes_candidate_within_stage_margins():
    gate = Gate(-0.35, -0.025, 0.035, 0.98)
    deltas, checks, passed = gate_decision(
        metrics(12.1, 0.38, 0.40),
        metrics(12.3, 0.39, 0.38),
        runtime_ratio=0.90,
        theoretical_ratio=0.68,
        gate=gate,
    )
    assert passed
    assert all(checks.values())
    assert abs(deltas["psnr"] + 0.2) < 1e-12


def test_gate_rejects_quality_or_flop_failure():
    gate = Gate(-0.15, -0.015, 0.020, 0.95)
    _, checks, passed = gate_decision(
        metrics(11.5, 0.34, 0.48),
        metrics(12.3, 0.39, 0.38),
        runtime_ratio=0.80,
        theoretical_ratio=0.72,
        gate=gate,
    )
    assert not passed
    assert not checks["psnr"]
    assert not checks["ssim"]
    assert not checks["lpips"]
    assert not checks["theoretical_flops"]


def test_runtime_latency_supports_records_output_schema(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "output": "000.mp4",
                        "full_attention_calls": 160,
                        "full_attention_total_ms": 320.0,
                    },
                    {
                        "output": "001.mp4",
                        "full_attention_calls": 160,
                        "full_attention_total_ms": 160.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = runtime_latency(runtime, ["000.mp4", "001.mp4"])
    assert summary["mean_full_attention_ms"] == 1.0
    assert summary["excludes_first_video"]
