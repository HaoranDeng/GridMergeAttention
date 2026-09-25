"""CPU-only checks for the streamed diagnostic's aggregation and scope."""
import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "screen_native_seed_trajectory.py"
spec = importlib.util.spec_from_file_location("streamed_seed_screen", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_uses_element_weighted_error():
    records = [{"methods": {"a": {"numel": n, "squared_error_sum": e,
                 "reference_squared_sum": r, "attention_ms": t}}}
               for n, e, r, t in [(10, 20, 40, 3), (30, 90, 180, 5)]]
    result = module.summarize(records, ["a"])["a"]
    assert result["mse"] == 110 / 40
    assert result["relative_l2"] == pytest.approx((110 / 220) ** .5)
    assert result["diagnostic_sampled_head_mean_attention_ms"] == 4
    assert result["calls"] == 2


def test_empty_summary_is_not_fake_zero_error():
    assert module.summarize([], ["a"]) == {}


def test_reference_pipeline_and_call_contract_are_explicit():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "inference" for n in ast.walk(tree))
    assert "denoising_step_list[:-1]" not in source
    assert 'state["call"] != 160' in source
    assert 'state["layer"] != call % 40' in source
    assert 'return dense' in source
    assert 'torch.randn(list(config.image_or_video_shape)' in source
    assert 'set_seed(index)' in source


def test_frozen_controls_do_not_enable_rescue():
    assert set(module.MODES) == {"rf0", "mass_c32", "mass_c16", "partition_c16"}
    for mode in module.MODES.values():
        assert "_sq1_qm_fp8score_skv_nl_ml2" in mode
        assert "_u16_" in mode
        assert "_t1_rf0_u16_" in mode
        assert "_lpf" not in mode and "_lyrc" not in mode


@pytest.mark.parametrize("key", ["TOKEN_MERGING_SAVE_FIRST_QKV_PATH", "TOKEN_MERGING_EXIT_AFTER_QKV_SAVE"])
def test_capture_environment_is_rejected(key):
    with pytest.raises(ValueError, match=key):
        module.check_capture_environment({key: "1"})
    module.check_capture_environment({key: ""})
