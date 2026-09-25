from types import SimpleNamespace

from analysis.estimate_attention_io import hard_c4_traffic


def make_args(merge_q_group):
    return SimpleNamespace(
        batch=1,
        outer=21,
        rows=45,
        cols=80,
        heads=40,
        dim=128,
        c=4,
        dtype_bytes=2,
        bias_bytes=4,
        merge_q_group=merge_q_group,
    )


def traffic_by_name(args):
    return {item.name: item.bytes for item in hard_c4_traffic(args)}


def test_full_query_group_loads_each_original_kv_token_once():
    args = make_args(merge_q_group=21)
    traffic = traffic_by_name(args)
    one_tensor = (
        args.batch
        * args.heads
        * args.outer
        * args.rows
        * args.cols
        * args.dim
        * args.dtype_bytes
    )
    assert traffic["hard.merge.k_load"] == one_tensor
    assert traffic["hard.merge.v_load"] == one_tensor


def test_smaller_query_groups_reload_kv_once_per_group():
    full_group = traffic_by_name(make_args(merge_q_group=21))
    seven_frames = traffic_by_name(make_args(merge_q_group=7))
    assert seven_frames["hard.merge.k_load"] == 3 * full_group["hard.merge.k_load"]
    assert seven_frames["hard.merge.v_load"] == 3 * full_group["hard.merge.v_load"]
