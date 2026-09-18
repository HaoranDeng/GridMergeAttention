import unittest

from wan.modules.model import (
    resolve_temporal_router_cycle,
    resolve_weighted_span_layer_schedule,
)


class TemporalRouterCycleTest(unittest.TestCase):
    def test_weighted_span_per_frame_schedule_uses_ranked_layers(self):
        mode = "weighted_span_alpha0p3_pool45_lpf3"
        self.assertTrue(
            resolve_weighted_span_layer_schedule(mode, 9).endswith(
                "_hg1o0_j4_p16_v8"
            )
        )
        self.assertTrue(
            resolve_weighted_span_layer_schedule(mode, 8).endswith("_t1_j4")
        )

    def test_weighted_span_per_frame_schedule_rejects_other_methods(self):
        with self.assertRaisesRegex(ValueError, "weighted-span"):
            resolve_weighted_span_layer_schedule("hellinger_span_lpf3", 9)

    def test_two_frame_router_cycle_varies_by_layer_and_call(self):
        mode = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_lyrc2_cg45"
        self.assertIn("_rf5_", resolve_temporal_router_cycle(mode, 0, 0))
        self.assertIn("_rf15_", resolve_temporal_router_cycle(mode, 1, 0))
        self.assertIn("_rf15_", resolve_temporal_router_cycle(mode, 0, 1))


    def test_four_frame_router_cycle_covers_all_router_frames(self):
        mode = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_lyrc4_cg45"
        resolved = [
            resolve_temporal_router_cycle(mode, layer, 0) for layer in range(4)
        ]
        self.assertEqual(
            [
                next(part for part in item.split("_") if part.startswith("rf"))
                for item in resolved
            ],
            ["rf0", "rf7", "rf13", "rf20"],
        )


    def test_router_cycle_rejects_multiple_markers(self):
        with self.assertRaisesRegex(ValueError, "at most one"):
            resolve_temporal_router_cycle("weighted_span_lyrc2_lyrc4", 0, 0)


if __name__ == "__main__":
    unittest.main()
