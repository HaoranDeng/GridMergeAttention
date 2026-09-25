# Q80K90C8 / C6 第三轮优化实测

两种容量均完成独占 H200 数值验证与真实 40 头测速。原任务已交回 GPU0/GPU4，原有输出保留，队列失败次数未增加。

| 配置 | H200 GPU | v2 | v3 完整调用 | 耗时下降 | 同卡 dense | 对 dense 加速 | 目标 | 达标 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Q80K90C8 | scai7 GPU0 | 139.518 ms | 122.212 ms | 12.40% | 333.278 ms | 2.727× | 100 ms | 否 |
| Q80K90C6 | scai7 GPU4 | 127.274 ms | 108.708 ms | 14.59% | 326.866 ms | 3.007× | 120 ms | 是 |

协议：3 prompts × 4 个冻结 StepDistill timestep × 固定层 0、5、10、15、20、25、30、35 × 40 个独立 head；96 个测点，每点与同卡 dense 交替测量 3 次，BF16 `[1,75600,40,128]`。这是完整 attention 调用延迟，不是 50 步视频耗时。v2/v3 使用同一物理 GPU 和匹配协议，属于不同轮次测量。

本轮采用 ballot/ffs 归属选择、连续向量直接写出、两路累加和新的 attention 流水配置。exact C、RF0 索引复用、候选池、两个交点、熵偏置、空簇语义和全量压缩 attention 保持原算子定义。没有新增代表点或 dense 特例。

所选配置：`head_chunk=4, native_tiles=42, selector_groups=2, sum_groups=2, owner_mode=1, direct_store=true, vector_store=true, min_blocks=4, route_m=128, route_n=128, route_warps=8, route_stages=3`；FP32 selector、局部 fast math、融合输入布局。构造器仍保留保守默认值，复现实测应显式传入 `selected_config.json`。

C8 的 HC8 形状均值仅比选中 HC4 快 0.080 ms，合并工作区翻倍，因此保留 HC4。形状调优使用四个已捕获 head 重复至 40 头，最终表格使用真实独立 40 头。所有未入选试验保留原始记录，不作为已验证推荐配置。

尝试过的未入选方向包括硬件 masked reduction、更宽 K/V 合并、更多 warp groups、shared Q、跨流重叠、指数复用、exact80 路由和 FlashInfer FA3 熵偏置路由。FA3 初步数值检查通过，但 C8 约 155 ms，慢于选中 Triton 路由。

## 数值检查及两个基线

| 方法 | 对 dense 平均相对 L2 | 比 SVG-EAR 增幅 | 比原始 C8 增幅 | 匹配协议的 H200 延迟 | PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|---|
| SVG-EAR top_p=0.85 | 0.06526199 | — | — | 待测 | 本协议待测 |
| 原始 Q45K80C8 | 0.08862509 | — | — | 待测 | 本协议待测 |
| Q80K90C8 v3 | 0.10388787 | +59.19% | +17.22% | 122.212 ms | 待测，未生成视频 |
| Q80K90C6 v3 | 0.12410684 | +90.17% | +40.04% | 108.708 ms | 待测，未生成视频 |

两个基线取自相同冻结八层 384 样本扫描。SVG-EAR 使用 H200，原始 C8 使用 L40S；不跨 GPU 计算速度比。SVG-EAR p=0.85 数值基线与工作簿 p=0.8749 视频基线不同。当前方法误差仍高于两个基线，未达到 QKC shortlist 的质量要求；实现验证通过不表示质量胜过基线。

| 配置 | 数值样本 | 对自身算子平均相对 L2 | 最大相对 L2 | seed 变化比例 |
|---|---:|---:|---:|---:|
| C8 | 384 | 0.00170018 | 0.00457633 | 0.000006330 |
| C6 | 384 | 0.00169054 | 0.00200066 | 0.000003445 |

另对所选配置检查了全部输出位置与独立 Triton 实现的一致性、尾部 head、RF0、exact C、强制交点、空簇零 K/V 与负无穷偏置、常量 V 保持。工程门槛为最大算子 L2 < 0.02、平均 seed 变化比例 < 0.005、对 dense 平均 L2 相比自身冻结算子绝对回退不超过 0.0005；这是实现审查标准，不是用户规定的通用误差容限。

## 完整 50 步 FLOPs

| 方法 | 50 步、4000 次 attention 调用的逻辑 major-matmul 总量 |
|---|---:|
| Q80K90C8 | 112.641048576 PFLOPs |
| Q80K90C6 | 84.678967296 PFLOPs |
| SVG-EAR | 待实际 50 步 trace |
| 原始 Q45K80C8 | 待完整调度核对 |

以上为固定形状 exact C 的调度计数，v3 未改变逻辑运算量，不是硬件计数或实测视频耗时。每调用 core=`6·40·128·C·945·840·90 + 4·40·128·75600·C·840`，selector=`2·40·128·16·840·45·32`，共乘 4000。排除 softmax、reduction、布局和 tensor-core padding。

历史四步平均口径保留为 SVG-EAR 48.75106610096 TFLOPs、原始 Q45K80C8 43.47592704 TFLOPs；不与 50 步总量混算，不将 SVG-EAR 四步均值直接外推。

## GPU 交回与证据

| GPU | 原任务 | 中断前已有视频 | 已核对恢复生成 PID |
|---|---|---:|---|
| 0 | p50_svg_s035 | 16 | 3229081 |
| 4 | n50_p50i_fisher_c8f | 39 | 3238466 |

原任务按原配置、seed、输出目录和 skip_existing 恢复。已有输出文件名及大小均未变化，失败次数未增加。完整原环境仅私存在 scai7，不进入代码仓库或公共证据包。

- C8：102 项形状配置测试，其中 98 项通过初步检查；2018 个独占监控样本，外来计算进程 0 次。
- C6：102 项形状配置测试，其中 98 项通过初步检查；2018 个独占监控样本，外来计算进程 0 次。
- `summary.json` 保存数值、时间、基线差值、FLOPs 和源码哈希；`C8/`、`C6/` 保存逐配置和逐测点原始结果。
- `source_snapshots/` 保留后期每次切换的源码；`restoration_verification.json` 保存原任务实际 GPU/PID/输出目录核验。

## Reproduce the validated configuration

The constructors retain conservative defaults. Pass `validated_h200_config.json`
explicitly. Only this selected configuration has the full v3 validation reported
above; other backend, occupancy and pipeline controls are experimental ablations.
Use one instance per concurrent call/stream. Returned outputs own their storage;
debug seeds use reusable workspace and must be cloned before another call.

Run the frozen numerical benchmark first, then the real40 benchmark after its
gate passes. Use the existing exclusive-GPU guard around each command, and supply
the frozen eight-layer data, reference, model repository and checkpoint paths:

```bash
python benchmark_q80k90c8.py --root RUN/numerical --data FROZEN_DATA --reference FROZEN_REFERENCE.py --operator compare_block_extensions_rf0.py --phase validate --config validated_h200_config.json
python benchmark_q80_real40_c8.py --root RUN/real40 --repo MODEL_REPO --checkpoint CHECKPOINT.pt --config validated_h200_config.json --prompts 3
```

Use the corresponding C6 benchmark files for C6. This kernel targets Hopper
sm_90a, BF16 inputs with D=128 and N a multiple of 3600, Q80/K90 and exact C6/C8.
The measured environment is the lab's `monarch_rt` environment with CUDA,
PyTorch, Triton and FlashInfer's bundled CUTLASS headers. The optional custom FA3
route was tested with FlashInfer 0.6.6 and was not selected.

Historical v2 documentation, measured config and CPU/GPU evidence are retained
under `history/v2/`; v2 source is commit `cc6bec9`. The existing `run_gpu.py`
tuning grid and offline compiler scripts remain historical v2 utilities; they
do not reproduce the selected v3 tuning configuration automatically.

Raw evidence paths in this report are relative to
`C:/GMA/QKC scan/Q80K90C8_C6_kernel/borrow_gpu0_gpu4_v3`.
They are also retained on scai7 under
`/local2/denghaoran/GridMergeAttention/experiments/q80c6_c8_borrow_gpu0_gpu4_20260923_v3`.
Private original-job environments are excluded from public evidence.
