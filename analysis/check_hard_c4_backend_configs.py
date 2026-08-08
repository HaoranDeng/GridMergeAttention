from attention.hard_route_backend import parse_direct_route_backend


BASE_DIRECT_C4 = "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"


def backend(*suffixes):
    suffix = "_".join([*suffixes, "tokenmajor", "inline_local_seed"])
    return f"{BASE_DIRECT_C4}_{suffix}"


def expect_flags(name, config, **flags):
    if not config.supported:
        raise AssertionError(f"{name}: backend should be supported.")
    for field, expected in flags.items():
        actual = getattr(config, field)
        if actual != expected:
            raise AssertionError(f"{name}: {field}={actual!r}, expected {expected!r}.")


def main():
    cases = {
        "bestlogit": (
            backend("bestlogit"),
            {"use_assigned_logit": True},
        ),
        "topkey": (
            backend("bestlogit", "topkey"),
            {"use_assigned_logit": True, "use_top_key": True},
        ),
        "topkv": (
            backend("bestlogit", "topkv"),
            {"use_assigned_logit": True, "use_top_kv": True},
        ),
        "topvalue": (
            backend("bestlogit", "topvalue"),
            {"use_assigned_logit": True, "use_top_value": True},
        ),
        "seedvalue": (
            backend("bestlogit", "seedvalue"),
            {"use_assigned_logit": True, "use_seed_value": True},
        ),
        "seedkey_topvalue": (
            backend("seedkey", "bestlogit", "topvalue"),
            {"use_seed_key": True, "use_assigned_logit": True, "use_top_value": True},
        ),
        "seedkeyproj_topvalue": (
            backend("seedkeyproj", "bestlogit", "topvalue"),
            {
                "use_seed_key": True,
                "use_seed_key_scale": True,
                "use_assigned_logit": True,
                "use_top_value": True,
            },
        ),
    }

    for name, (text, flags) in cases.items():
        expect_flags(name, parse_direct_route_backend(text), **flags)

    forbidden = {
        "seedkey_topkey": backend("seedkey", "bestlogit", "topkey"),
        "seedkey_topkv": backend("seedkey", "bestlogit", "topkv"),
    }
    for name, text in forbidden.items():
        config = parse_direct_route_backend(text)
        if not (config.use_seed_key and (config.use_top_key or config.use_top_kv)):
            raise AssertionError(f"{name}: parser did not expose the expected conflict flags.")

    print("hard_c4_backend_config_check_ok")


if __name__ == "__main__":
    main()
