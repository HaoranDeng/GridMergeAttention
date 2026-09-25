# Q80K90 C4–C12 优化实测

| 容量 | 同卡旧版 ms | 新版 ms | 耗时下降 | 对同卡 Dense 加速 | 50步 PFLOPs | 推荐 |
|---|---:|---:|---:|---:|---:|---|
| C4 | 115.703 | 61.392 | 46.94% | 5.400x | 56.716886 | v4 |
| C6 | 109.317 | 108.874 | 0.41% | 3.061x | 84.678967 | v4 |
| C8 | 121.785 | 121.261 | 0.43% | 2.757x | 112.641049 | v4 |
| C10 | 254.962 | 224.672 | 11.88% | 1.444x | 140.603130 | v4 |
| C12 | 267.697 | 242.189 | 9.53% | 1.335x | 168.565211 | v4 |

全部使用 scai7 独占 NVIDIA H200：GPU0 为 C4/C6/C8，GPU4 为 C10/C12。旧版、新版及 Dense 在各自同一 GPU、同一 QKV 测点内交替测量；不按不同 GPU 的记录计算速度增益。
协议：BF16 [1,75600,40,128]，3 prompts × 4 冻结 StepDistill 步 × 层0/5/10/15/20/25/30/35，共96测点，每点3次，40个真实独立head。表中是完整 attention 调用耗时，不是视频端到端时间。
C6/C8 旧版是已验证v3，C10/C12旧版是原native实现。C4旧版是保守的8行内部tile实现，用作本轮工程对照；此前没有正式C4延迟数据。推荐依据为逐测点旧版减新版的均值及近似95%区间；未取得明确增益时保留此前实现。

## 数值验证与两个质量基线

| 方法 | Mean relative L2 to dense | 对 SVG-EAR 变化 | 对原始 C8 变化 | PSNR/SSIM/LPIPS |
|---|---:|---:|---:|---|
| SVG-EAR top_p=0.85 | 0.06526199 | — | — | 本轮待测 |
| 原始 Q45K80C8 | 0.08862509 | — | — | 本轮待测 |
| Q80K90C4 | 0.15449854 | +136.74% | +74.33% | 待测，未生成视频 |
| Q80K90C6 | 0.12410684 | +90.17% | +40.04% | 待测，未生成视频 |
| Q80K90C8 | 0.10388787 | +59.19% | +17.22% | 待测，未生成视频 |
| Q80K90C10 | 0.08979765 | +37.60% | +1.32% | 待测，未生成视频 |
| Q80K90C12 | 0.07847927 | +20.25% | -11.45% | 待测，未生成视频 |

每种容量独立完成固定八层384个数值样本。另检查全部输出位置、尾部head、RF0复用、exact C、两交点、空簇零K/V和负无穷bias、常量V保持。该实现验证不是方法质量胜过比较基线的结论。

## FLOPs口径

完整50步×40层×CFG两路=4000次self-attention；逻辑major-matmul，FMA=2。不含投影、softmax、Fisher/GS标量运算、归约、布局和硬件tile padding。
`F_call = 6*40*128*C*945*840*90 + 4*40*128*75600*C*840 + 2*40*128*16*840*45*32`；50步总量为每次调用×4000。
SVG-EAR严格50步FLOPs待实际50步密度与聚类迭代trace；原始Q45K80C8的对应实现计数待核对。历史四步单次等效值48.75106610096/43.47592704 TFLOPs保留原口径，不直接外推SVG-EAR。两基线匹配本轮的H200完整调用延迟待测。

## 原任务恢复

- GPU0 `p50_svg_s035`：已核实恢复生成PID [854199]；原有17个视频保留，队列失败次数未增加。
- GPU4 `n50_p50i_fisher_c8f`：已核实恢复生成PID [589809]；原有40个视频保留，队列失败次数未增加。

逐配置、逐数值样本、逐计时测点、源码哈希、GPU独占监控和恢复验证见同目录结构化结果；私有环境不进入公共证据包。

## 最终默认配置

| 容量 | 默认实现 | 实测完整调用 ms | 50步 PFLOPs |
|---|---|---:|---:|
| C4 | v4 | 61.392 | 56.716886 |
| C6 | v4 | 108.874 | 84.678967 |
| C8 | v4 | 121.261 | 112.641049 |
| C10 | v4 | 224.672 | 140.603130 |
| C12 | v4 | 242.189 | 168.565211 |

`recommended.make_q80k90(C)` 按容量返回上表配置；保留更快的旧版时不会切换到新版。
每个并发调用/stream需独立实例；输出拥有独立存储，debug seeds属于可复用工作区。H200/sm_90a，BF16，D=128，N为3600的正整数倍；完整性能证据对应N=75600。

```python
from recommended import make_q80k90
attention = make_q80k90(10)
output = attention(q, k, v)
```

历史v3基线配置保留于 `validated_h200_config.json`；当前五种容量的默认选择在 `validated_h200_configs.json`。完整原始调优证据及数值/测速脚本随交付保留。

Memory tradeoff: selected C4 and C8 use head_chunk=8. Their merged K/V workspace is twice the corresponding head_chunk=4 workspace. C8 improves latency only about 0.4%; the previous v3 HC4 implementation and configuration remain available when lower workspace use matters.

Published text uses LF line endings. gpu_evidence.json retains both measured source hashes and published source hashes; normalization changes no operators or configuration values. Raw measured sources remain with the experiment artifacts.

## Strided input packing (2026-09-25)

The fused-layout path now reads the actual strides of Q, K and V. Diffusers
passes token-major views backed by head-major storage; the previous path
materialized three contiguous tensors before packing them again. Removing
those copies leaves selection, merge weights, entropy bias and routing
unchanged. The `fused_layout=False` reference path is unchanged.

`check_input_layouts.py` validates all packed tensors against independent
PyTorch layout operations, including mixed strides, nonzero storage offsets,
tail heads and the full `[1,75600,40,128]` shape. All 16 cases passed exactly
on an exclusively idle L40S. With a frozen original v4 packer as the baseline,
head-major input packing decreased from 15.527 to 8.445 ms and eliminated
2,322,432,000 bytes of temporary allocations. Contiguous token-major input
packing was 8.609 versus 8.626 ms. These are packing-stage L40S results only;
they do not establish a new H200 attention or full-video latency.

Evidence: `input_layout_validation_20260925.json`. The historical H200 timing
tables above remain unchanged. See
[`docs/q80k90_c4_c12_kernels.md`](../../../docs/q80k90_c4_c12_kernels.md)
for the historical 61.4/67.3/72.1 ms timing audit and completed H200 validation.

H200 follow-up completed later on2026-09-25: same GPU7, same native50
prompt/seed,4000calls each. Full attention fell from71.193183 to68.540536ms
(3.73%), and pipeline time from608.271085 to598.595445s (1.59%). All16 paired
first-step checks and the final decoded output hash matched exactly;603
monitor samples confirmed exclusivity. The repeated first-step attention
gate was65.086850/62.988589ms and isolated packing4.170104/1.312200ms.
One full50 run per implementation, old then new; this does not eliminate or
causally explain the remaining gate-versus-generation gap. Full-pipeline peak
allocated memory was unchanged. See `h200_input_layout_validation_20260925.json`
and the linked documentation for full scope, telemetry and limitations.
