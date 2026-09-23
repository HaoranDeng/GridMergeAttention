# Q80K90 C4/C6/C8 native 20-video experiment

Generate the canonical MovieGen first20 prompts at seed0, 720x1280, 81 frames,
50 UniPC steps (flow shift5), CFG5. GPU0 handles indices000-009; GPU4 handles
010-019 for each capacity. Use the hash-frozen v4 source and selected per-capacity
H200 configs from the preceding 384-case numerical validation.

Before output generation, each card runs a first-step native dense-trajectory
integration and paired speed gate at layers0,5,10,15,20,25,30,35 and both CFG
branches. This engineering probe does not replace the frozen numerical scan or
estimate full-video speedup. Generation requires4000 GMA calls and100 per layer.

A separate CPU process evaluates all81 frames at original resolution against
the prompt/seed-matched dense baseline, using the hash-frozen evaluator.
Timings remain separated by GPU. Results and videos are kept under the canonical
local `C:/GMA/QKC scan/Q80K90C4_C6_C8_video20` directory.

Server paths are explicit for reproducibility. Borrowing and restoring the
user-authorized original queue jobs is managed separately; private environments
are never included in public artifacts. Runs are pending until their completion
and quality markers are verified.
