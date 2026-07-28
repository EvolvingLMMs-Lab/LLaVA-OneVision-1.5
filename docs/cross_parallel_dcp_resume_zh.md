# 跨并行方式加载 DCP 权重操作手册

> 场景：先以 `{TP1, PP1}` 加载 torch 格式 ckpt 训练若干步，把权重保存为 `torch_dist` (DCP) 格式；随后改并行方式为 `{TP4, PP1}` 并修改其它训练参数，**仅复用 DCP 中的权重，优化器与训练参数全部使用当前脚本**。
>
> 适用仓库：`LLaVA-OneVision-2.0`（Megatron-LM + `aiak_megatron`，DCP reshard 已验证）。

---

## 0. TL;DR

```text
[torch ckpt @ TP1/PP1]
        │  torchrun 训练 (stage A), --save / --save-interval N
        ▼
[DCP ckpt @ TP1/PP1]  (fully_sharded_model_space, 由 --ckpt-fully-parallel-save 生成)
        │  torchrun 训练 (stage B), 改 TP/PP/训练参数, --finetune --no-load-rng --load <dcp>
        ▼
[从 iter 0 开始用新训练参数继续训练, 优化器/RNG/调度器全部重新初始化]
```

核心三条命令参数：


| Stage                 | 关键 flag                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| A (TP1/PP1, 保存 DCP) | `--save <OUT_A> --save-interval N --ckpt-format torch_dist --ckpt-fully-parallel-save`         |
| B (TP4/PP1, 加载 DCP) | `--load <OUT_A> --finetune --no-load-rng --auto-detect-ckpt-format --ckpt-fully-parallel-load` |

---

## 1. 实际运行（参考脚本）

仓库内已配套两个可直接跑的示例脚本，覆盖完整两阶段流程：

### Stage A：torch `{TP1, PP1}` → torch_dist `{TP1, PP1}`

- 参考脚本：`./examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_base.sh`
- **关键：先禁用 `export TORCH_COMPILE_LAYERS=1`**（注释掉或不设），否则 Stage A 落盘 DCP 时会带上 compile 相关 hook，与 Stage B 的 reshard 不兼容。
- 权重保存到：`./examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_base/`
- 产物含 `iter_0000040/` + `latest_checkpointed_iteration.txt`（内容为 40）+ `metadata.json` (`sharded_backend=torch_dist`)。

```bash
cd /workspace/train_ov2/LLaVA-OneVision-2.0
# 禁用 torch.compile
# export TORCH_COMPILE_LAYERS=1   # 注释掉
bash examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_base.sh 1 1 20480 1 32 40
```

校验：

```bash
cat examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_base/latest_checkpointed_iteration.txt
# 期望: 40
```

### Stage B：torch_dist `{TP1, PP1}` → torch_dist `{TP4, PP1}`

- 参考脚本：`./examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp.sh`
- **可以开启 `export TORCH_COMPILE_LAYERS=1`**（Stage B 是消费端，不影响落盘 DCP 兼容性）。
- 权重保存到：`./examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp/`
- `--load` 指向 Stage A 的 DCP 目录，并加 `--finetune --no-load-rng`。

```bash
cd /workspace/train_ov2/LLaVA-OneVision-2.0
export TORCH_COMPILE_LAYERS=1   # Stage B 可开
bash examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp.sh 4 1 20480 1 32 100
```

> 两个脚本里的 TP/PP/GBS/SEQ_LEN/NSTEP 等参数均可通过位置参数或环境变量覆盖，便于不同实验配置复用。

---

## 2. 前置约束（必须满足，否则 reshard 失败）

1. **Stage A 保存 DCP 时必须开 `--ckpt-fully-parallel-save`**
   产生 `fully_sharded_model_space` 布局，这是跨 TP reshard 的硬要求。
   `checkpointing.py:1290` 检测到 `ckpt_tp_pp != run_tp_pp` 且 sharding_type 不是 `fully_sharded_model_space` 时会 `raise RuntimeError`。
2. **Stage A 首次加载的 torch ckpt 必须是 TP1/PP1**（或与当时 run 的 TP/PP 一致）。
   否则首次加载本身就需要走 `tools/checkpoint/loader_mcore.py` 结构对齐。
3. **Stage A 与 Stage B 的参数 key 名一致**。

   - 算子实现可换（TE ↔ Mcore ↔ Apex 的 `ColumnParallelLinear` 等，state_dict key 都是 `weight/bias`）
   - adapter 的 `ADAPTER_TYPE` 可在 `LINEAR ↔ TP_LINEAR` 间切换（key 相同）
   - 但 `Conv2d ↔ Linear`、fused QKV 与非 fused 等会改变 key/shape 的操作**不能直接 reshard**
4. **vocab padding 不一致可接受**（如 151936 → 152064），DCP 会自动 pad/truncate，仅 warning。
5. **FP8 `_extra_state`** 由 commit `27e12f92` 的 `fix_te.py` + `te_env.py` 处理：
   若源端 TE 算子有 `_extra_state` 而目标端换为非 TE 实现，需在目标端设
   `TE_EXTRA_STATE_MISSING_CHECK=1` 与 `TE_FILTER_MODULE_OPT="<modules>;<ops>"`
   （modules 与 ops 1:1 配对）以主动过滤。

---

## 3. Stage A：以 `{TP1, PP1}` 加载 torch ckpt 并训练保存 DCP

### 3.1 关键脚本参数

```bash
TP=1
PP=1
NSTEP=40                       # 训练步数，到达后保存
SAVE_INTERVAL=${NSTEP}         # 在最后一步落盘

TRAINING_ARGS=(
    ...
    --save        "$OUT_A"
    --save-interval   "$SAVE_INTERVAL"
    --ckpt-format     torch_dist
    --ckpt-fully-parallel-save          # ★ 必须
    --load        "$TORCH_CKPT_DIR"     # 原始 torch 格式 ckpt
    --auto-detect-ckpt-format
    --finetune                          # torch ckpt 也用 finetune, 不带优化器
    --no-load-rng
    ...
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size "$PP"   # 1
    --tensor-model-parallel-size   "$TP"   # 1
    --use-distributed-optimizer
    --sequence-parallel                 # SP 可与 TP1 共存
    ...
)
```

### 3.2 运行与产物

```bash
bash examples/llava_onevision2/quick_start_video_2b/your_stageA.sh
```

成功后 `$OUT_A` 下应有：

```
$OUT_A/
├── iter_0000040/
│   ├── __0_0.distcp  ...  __7_1.distcp
│   ├── common.pt
│   └── metadata.json        # {"sharded_backend": "torch_dist", ...}
├── latest_checkpointed_iteration.txt   # 内容: 40
└── dataloader/ ...
```

校验：

- `cat $OUT_A/latest_checkpointed_iteration.txt` 应为 `40`
- `cat $OUT_A/iter_0000040/metadata.json | jq .sharded_backend` 应为 `"torch_dist"`

---

## 4. Stage B：改并行方式为 `{TP4, PP1}`、修改训练参数、仅复用权重

### 4.1 关键脚本参数

```bash
TP=4
PP=1
GBS=8                        # 训练参数可改
SEQ_LEN=20480
NSTEP=100
LR=1.0e-5

# 用 DCP 作权重源
CHECKPOINT_PATH="$OUT_A"     # 指向 Stage A 的 save 目录

TRAINING_ARGS=(
    ...
    --train-iters    "$NSTEP"
    --lr             "$LR"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --global-batch-size  "$GBS"
    --seq-length         "$SEQ_LEN"

    --save        "$OUT_B"
    --save-interval   50
    --ckpt-format     torch_dist
    --ckpt-fully-parallel-load          # 加载端可选, 加速并行读

    --load        "$CHECKPOINT_PATH"    # Stage A 的 DCP 目录
    --auto-detect-ckpt-format
    --finetune                          # ★ 只取权重
    --no-load-rng                       # ★ 不恢复 RNG
    --override-opt_param-scheduler      # ★ 冗余但无害, 强化用脚本 scheduler

    --dataloader-save "$OUT_B/dataloader"
    ...
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size "$PP"   # 1
    --tensor-model-parallel-size   "$TP"   # 4
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-backend nccl
    ...
)
```

环境变量（按需）：

```bash
export PATCH_EMBED_TYPE="${PATCH_EMBED_TYPE:-LINEAR}"   # 或 TP_LINEAR / CONV2D
export SCATTER_BEFORE_PATCH_EMBED="${SCATTER_BEFORE_PATCH_EMBED:-0}"
export TE_EXTRA_STATE_MISSING_CHECK=1                  # 如需过滤 _extra_state
export TE_FILTER_MODULE_OPT="vision_model,adapter;patch_embed,fc1"  # modules;ops 1:1 配对
```

### 4.2 运行

```bash
bash examples/llava_onevision2/quick_start_video_2b/your_stageB.sh
```

### 4.3 加载阶段日志校验（关键行）

成功路径应看到：

```
successfully loaded checkpoint from ... [ t 1/4, p 1/1 ] at iteration 40
> loaded checkpoint with TP=1 PP=1, reshard to run TP=4 PP=1 ...
... overriding lr_max ... to <script value>          # override-opt_param-scheduler 生效
... checkpoint version 0.0
... [before the start of training step] datetime: <stage B 启动时刻>
```

之后应正常进入第一步训练并打出 loss/tflops。若日志在 `[before the start of training step]` 3 秒内静默结尾、tensorboard 无 scalar、GPU 空闲、`dmesg` 无 OOM——是运行被外部 kill，不是加载问题（见排错章节）。

---

## 5. `--finetune` 行为对照（来自 `aiak_megatron/megatron/training/checkpointing.py`）


| 加载项                     | `--finetune` (本手册方案)                           | 不带`--finetune` 的 resume         |
| -------------------------- | --------------------------------------------------- | ---------------------------------- |
| 模型权重                   | ✅ 加载                                             | ✅ 加载                            |
| 优化器 state (`optimizer`) | ❌ 不加载（`checkpointing.py:1425`）                | ✅ 加载                            |
| RNG state                  | ❌ 不加载                                           | ✅ 加载（`checkpointing.py:1472`） |
| `opt_param_scheduler`      | ❌ 不加载，用当前脚本（`checkpointing.py:1446`）    | ✅ 加载                            |
| starting`iteration`        | 强制`0`（`checkpointing.py:1377`）                  | 读 ckpt 里的`iteration`            |
| `consumed_train_samples`   | 不读 ckpt，沿用脚本（`checkpointing.py:1399` 跳过） | 从 ckpt 恢复                       |
| `rerun_state_machine`      | 不加载                                              | 仅当 TP/PP 一致才加载              |

结论：`--finetune` 正是"只拿权重，丢弃所有训练状态"的官方开关，与本手册需求完全对应。`--no-load-rng` 在 `--finetune` 下冗余，但显式声明可读性更好。

---

## 6. 为什么能省掉 mcore 并行互转脚本

DCP 在 `fully_sharded_model_space` 布局下，`torch.distributed.checkpoint.load` 会：

1. 读取源端 metadata，把所有分片合并为完整张量
2. 按目标端 `generate_state_dict` (`checkpointing.py:1326`) 声明的 shard 描述重新切片
3. 把切片分发给目标 TP/PP 各 rank

这条路径覆盖了 `tools/checkpoint/loader_mcore.py` 在 TP/PP 互转上所做的事。仍需要结构对齐（HF → mcore 层名映射、qkv 合并、ln 位置等）的场景，DCP 不管——所以**只有第一次 HF→mcore 转换不可省**，之后的 mcore↔mcore 的 TP/PP 互转都可以走 DCP。

---

## 7. 算子更换（TE→Mcore/Apex）的兼容判定


| 切换类型                                                                   | 能否直接从 DCP 加载 | 备注                                                       |
| -------------------------------------------------------------------------- | ------------------- | ---------------------------------------------------------- |
| `ColumnParallelLinear` 实现库更换（TE / Mcore / `apex.transformer`），bf16 | ✅                  | key 都是`weight/bias`，shard 轴一致                        |
| `ADAPTER_TYPE = LINEAR ↔ TP_LINEAR`                                       | ✅                  | 同上，仅 TP 切分差异由 reshard 处理                        |
| `FusedRMSNorm` ↔ `FusedLayerNorm`（bias 存在性差异）                      | ⚠️                | 若源端有`bias` 而目标端只有 `weight`，bias 被 DCP 静默丢弃 |
| TE FP8 算子 → 非 TE 算子                                                  | ⚠️                | 需用`TE_FILTER_MODULE_OPT` 主动过滤源端 `_extra_state`     |
| `Conv2d` ↔ `Linear`（权重 flatten 维度不同）                              | ❌                  | DCP 不做 reshape，需手写转换                               |
| fused QKV ↔ 三个独立 Linear                                               | ❌                  | 拼接顺序差异，DCP 报 size mismatch                         |

---

## 8. 排错清单


| 现象                                                                                                         | 原因                                                                | 处理                                                           |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| `RuntimeError: ... not supported for DistributedOptimizer with sharding type dp_zero_gather_scatter`         | Stage A 没开`--ckpt-fully-parallel-save`                            | 重跑 Stage A，加上 flag                                        |
| `size mismatch ... saved torch.Size([151936, 2048]) and current: torch.Size([152064, 2048])`                 | vocab padding 警告，**可忽略**                                      | DCP 自动 pad/truncate                                          |
| `Unable to load optimizer from checkpoint ... Specify --no-load-optim or --finetune`                         | 没加`--finetune`，尝试加载优化器失败                                | 加`--finetune`                                                 |
| 日志在`[before the start of training step]` 后 3 秒内静默结束，tensorboard 无 scalar，GPU 空闲，dmesg 无 OOM | 进程被外部 kill（调度/租约/OOM-killer 未记录）                      | 检查 sched/CI/lease；在脚本里`trap` + 单独 `tee` stderr 到文件 |
| `Filtering out modules ['vision_model'] ...`                                                                 | `fix_te.py` 在过滤 `_extra_state`，**正常**                         | 无需处理                                                       |
| `TE_FILTER_MODULE_OPT: modules count (X) != ops count (Y)`                                                   | `te_env.py:get_te_filter_modules_and_ops` 校验 modules/ops 1:1 配对 | 检查 env，两边数量对齐                                         |
| resumed`iteration = 40` 而不是 `0`                                                                           | 没加`--finetune`，按 resume 加载                                    | `--finetune` 强制 iter=0                                       |
| `consumed_train_samples` 不为 0                                                                              | 没加`--finetune`，从 ckpt 恢复                                      | `--finetune` 跳过 `args` 恢复                                  |

---

## 9. 速查命令

```bash
# Stage A 产物校验
cat $OUT_A/latest_checkpointed_iteration.txt
python -c "import json;print(json.load(open('$OUT_A/iter_0000040/metadata.json'))['sharded_backend'])"

# Stage B 启动
bash your_stageB.sh 2>&1 | tee run_stageB.log

# 日志中关键行
grep -E "successfully loaded|overriding|checkpoint version|before the start|Filtering out" run_stageB.log

# 确认未误取优化器状态
grep -E "Could not load optimizer|loaded optimizer" run_stageB.log
```


---

## 10. 参考

- 加载主路径：`aiak_megatron/megatron/training/checkpointing.py:1200-1480`
- reshard 硬校验：`checkpointing.py:1290-1295`
- `--finetune` 行为分支：`checkpointing.py:1264, 1377, 1399, 1425, 1446, 1472`
- `_extra_state` 过滤：`aiak_megatron/megatron/training/fix_te.py` + `aiak_training_llm/utils/te_env.py`（commit `27e12f92`）
- `override_opt_param_scheduler`：`aiak_megatron/megatron/core/optimizer_param_scheduler.py:211`
- adapter TP 类型切换：`aiak_training_llm/models/llava_onevision2/adapter.py:70-128`
- 参考脚本：`examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_PETP_SP_adapterTP/sft_ckpt-dcp-to-dcp_with-parallel-change_PETP_SP_adapterTP.sh`
