from types import SimpleNamespace
import unittest
from analysis.run_wan22_ear_c4 import (ExpertProxy, density_summary, validate_dispatch,
                                       configure_vae_math_decode, DENSE_CALLS, SPARSE_CALLS, CALLS)


def test_two_experts_share_live_global_timestep():
    pipeline = SimpleNamespace(_current_timestep=990)
    first, second = ExpertProxy(pipeline, object()), ExpertProxy(pipeline, object())
    pipeline._current_timestep = 400
    assert first._current_timestep == second._current_timestep == 400


def test_density_includes_overlap_only_once():
    assert (CALLS, DENSE_CALLS, SPARSE_CALLS) == (4000, 880, 3120)
    result = density_summary([0.2595] * SPARSE_CALLS, 1)
    assert abs(result["all_call_density"] - 0.42241) < 1e-10
    with unittest.TestCase().assertRaisesRegex(ValueError, "incomplete"):
        density_summary([0.2595] * (SPARSE_CALLS - 1), 1)


def make_dispatch():
    high, low = [], []
    for layer in range(40):
        high.append(dict(dense=40 if layer == 0 else 20, sparse=0 if layer == 0 else 20, gca=0))
        low.append(dict(dense=60 if layer == 0 else 0, sparse=0 if layer == 0 else 60, gca=0))
    forwards = [dict(step=step, expert="high_noise" if step < 20 else "low_noise")
                for step in range(50) for _ in range(2)]
    return [SimpleNamespace(layers=high), SimpleNamespace(layers=low)], forwards


def test_warmup_does_not_restart_at_expert_switch():
    audits, forwards = make_dispatch()
    result = validate_dispatch(audits, forwards, "ear")
    assert result["dense"] == 880
    assert result["sparse"] == 3120
    audits[1].layers[1]["dense"] += 20
    audits[1].layers[1]["sparse"] -= 20
    with unittest.TestCase().assertRaisesRegex(ValueError, "warm-up"):
        validate_dispatch(audits, forwards, "ear")


def test_missing_cfg_expert_call_is_rejected():
    audits, forwards = make_dispatch()
    with unittest.TestCase().assertRaisesRegex(ValueError, "CFG"):
        validate_dispatch(audits, forwards[:-1], "ear")


def test_vae_backend_is_scoped_even_after_decode_failure():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    def decoder(fail=False):
        assert torch.backends.cuda.math_sdp_enabled()
        assert not torch.backends.cuda.flash_sdp_enabled()
        if fail:
            raise RuntimeError("decode failure")
        return "decoded"
    vae = SimpleNamespace(decode=decoder)
    configure_vae_math_decode(vae)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        assert vae.decode() == "decoded"
        assert not torch.backends.cuda.math_sdp_enabled()
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "decode failure"):
            vae.decode(fail=True)
        assert not torch.backends.cuda.math_sdp_enabled()
        assert torch.backends.cuda.flash_sdp_enabled()


if __name__ == "__main__":
    for test in (test_two_experts_share_live_global_timestep,
                 test_density_includes_overlap_only_once,
                 test_warmup_does_not_restart_at_expert_switch,
                 test_missing_cfg_expert_call_is_rejected,
                 test_vae_backend_is_scoped_even_after_decode_failure):
        test()
        print(test.__name__ + ": PASS")
