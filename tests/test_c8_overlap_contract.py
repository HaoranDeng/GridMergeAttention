import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OVERLAP = ROOT / "kernels/best/c8_r2/overlap.py"
RESIDENT = ROOT / "analysis/hold_fisher_c8_clean_h200.py"
N128_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_n128_v0.cu"
N96_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_n96_v0.cu"
OFFSET0_RCP_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_rcp_v0.cu"
DEPHASE_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_dephase1250_v0.cu"
WAVE_DEPHASE_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_wave132_dephase1250_v0.cu"
SMSLOT_DEPHASE_MERGE = ROOT / "native_fisher/fisher_c8_merge_offset0_smslot_dephase1250_v0.cu"


class ExactC8OverlapContractTests(unittest.TestCase):
    def test_overlap_double_buffers_only_merged_workspaces(self):
        source = OVERLAP.read_text(encoding="utf-8")
        self.assertIn("torch.empty_like(backend.mk)", source)
        self.assertIn("torch.empty_like(backend.mv)", source)
        self.assertIn("torch.empty_like(backend.bias)", source)
        self.assertIn("range(0, 40, 5)", source)
        self.assertNotIn("singleton_count", source)
        self.assertNotIn("8 + 1", source)

    def test_route_consumes_only_exact_c8_workspaces(self):
        tree = ast.parse(OVERLAP.read_text(encoding="utf-8"))
        route_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "router"
        ]
        self.assertEqual(len(route_calls), 1)
        rendered = ast.unparse(route_calls[0])
        self.assertIn("q, mk, mv, bias, output", rendered)
        self.assertNotIn("q, k, v", rendered)

    def test_resident_benchmark_requires_overlap_headroom(self):
        source = RESIDENT.read_text(encoding="utf-8")
        self.assertIn('"--combined-overlap"', source)
        self.assertIn("args.backend_headroom_gib < 75", source)
        self.assertIn('"two_stream_double_buffer"', source)

    def test_n128_score_probe_keeps_exact_c8_storage(self):
        source = N128_MERGE.read_text(encoding="utf-8")
        self.assertIn("float s[64]", source)
        self.assertIn("MMA_64x128x16_F32BF16BF16_RS<Major::K,Major::K>", source)
        self.assertIn("seeds.size(1)", source)
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertNotIn("kb*9", source)
        self.assertNotIn("singleton", source.lower())

    def test_n96_score_probe_keeps_exact_c8_storage(self):
        source = N96_MERGE.read_text(encoding="utf-8")
        self.assertIn("float s[48]", source)
        self.assertIn("MMA_64x96x16_F32BF16BF16_RS<Major::K,Major::K>", source)
        self.assertIn("E40(s),E8(s,40)", source)
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertNotIn("kb*9", source)

    def test_offset0_rcp_keeps_exact_c8_storage(self):
        source = OFFSET0_RCP_MERGE.read_text(encoding="utf-8")
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertIn("rcp.approx.ftz.f32", source)
        self.assertNotIn("kb*9", source)
        self.assertNotIn("singleton", source.lower())

    def test_dephase_candidate_changes_only_schedule_not_capacity(self):
        source = DEPHASE_MERGE.read_text(encoding="utf-8")
        self.assertIn("__nanosleep(phase_ns)", source)
        self.assertIn("(blockIdx.x&3u)*1250u", source)
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertNotIn("kb*9", source)
        self.assertNotIn("singleton", source.lower())

    def test_wave_dephase_uses_h200_residency_wave_not_block_modulo(self):
        source = WAVE_DEPHASE_MERGE.read_text(encoding="utf-8")
        self.assertIn("/132u)&3u", source)
        self.assertIn("gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z)", source)
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertNotIn("kb*9", source)
        self.assertNotIn("singleton", source.lower())

    def test_sm_slot_dephase_does_not_assume_grid_traversal(self):
        source = SMSLOT_DEPHASE_MERGE.read_text(encoding="utf-8")
        self.assertIn("fisher_c8_dephase_slots[256]", source)
        self.assertIn("atomicAdd(&fisher_c8_dephase_slots[smid],1u)&3u", source)
        self.assertIn("mov.u32 %0, %%smid", source)
        self.assertIn("kb*8+seed_for_row(row)", source)
        self.assertNotIn("kb*9", source)
        self.assertNotIn("singleton", source.lower())


if __name__ == "__main__":
    unittest.main()
