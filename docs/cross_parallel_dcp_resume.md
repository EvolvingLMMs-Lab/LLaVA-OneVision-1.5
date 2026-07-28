# Cross-Parallel DCP Resumability Playbook

> Scenario: load a torch-format checkpoint under `{TP1, PP1}`, train for a number of steps, then save a `torch_dist` (DCP) checkpoint — `ckpt_40`. After that, change the parallel layout to `{TP4, PP1}` and other training parameters; **reuse only the weights from `ckpt_40`; the optimizer state and all training parameters come from the current script**.
>
> Target repo: `LLaVA-OneVision-2.0` (Megatron-LM + `aiak_megatron`, DCP reshard verified).

---

## 0. TL;DR

```text
[torch ckpt @ TP1/PP1]
        │  torchrun training (Stage A), --save / --save-interval N
        ▼
[DCP ckpt @ TP1/PP1]  (fully_sharded_model_space, produced by --ckpt-fully-parallel-save)
        │  torchrun training (Stage B), change TP/PP/training params,
        │  --finetune --no-load-rng --load <dcp>
        ▼
[Continue training from iter 0 with the new training parameters;
 optimizer / RNG / scheduler are all re-initialized from the current script]
```

The three key command-line flags per stage:

| Stage                   | Key flags                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| A (TP1/PP1, save DCP)   | `--save <OUT_A> --save-interval N --ckpt-format torch_dist --ckpt-fully-parallel-save`          |
| B (TP4/PP1, load DCP)   | `--load <OUT_A> --finetune --no-load-rng --auto-detect-ckpt-format --ckpt-fully-parallel-load`  |

---

## 1. Reference Runs (Ready-to-Use Scripts)

The repo ships two runnable example scripts that cover the full two-stage flow.

### Stage A: torch `{TP1, PP1}` → torch_dist `{TP1, PP1}`

- Script: `./examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_base.sh`
- **Key: disable `export TORCH_COMPILE_LAYERS=1` first** (comment it out or leave it unset). Otherwise Stage A will persist torch.compile-related hooks in the DCP and they are not compatible with Stage B's reshard.
- Output dir: `./examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_base/`
- Artifacts include `iter_0000040/`, `latest_checkpointed_iteration.txt` (content `40`), and `metadata.json` (`sharded_backend=torch_dist`).

```bash
cd /workspace/train_ov2/LLaVA-OneVision-2.0
# Disable torch.compile
# export TORCH_COMPILE_LAYERS=1   # commented out
bash examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_base.sh 1 1 20480 1 32 40
```

Verify:

```bash
cat examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_base/latest_checkpointed_iteration.txt
# expected: 40
```

### Stage B: torch_dist `{TP1, PP1}` → torch_dist `{TP4, PP1}`

- Script: `./examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp.sh`
- **You may enable `export TORCH_COMPILE_LAYERS=1`** here. Stage B is a consumer, so it does not affect the on-disk DCP compatibility.
- Output dir: `./examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp/`
- `--load` points at Stage A's DCP directory, plus `--finetune --no-load-rng`.

```bash
cd /workspace/train_ov2/LLaVA-OneVision-2.0
export TORCH_COMPILE_LAYERS=1   # safe to enable on Stage B
bash examples/llava_onevision2/quick_start_video_2b/sft_ckpt-dcp-to-dcp_with-parallel-change_4tp.sh 4 1 20480 1 32 100
```

> Both scripts accept TP/PP/GBS/SEQ_LEN/NSTEP via positional parameters or environment variables, so you can re-use them for different experiment configs.

---

## 2. Prerequisites (mandatory; otherwise reshard fails)

1. **Stage A must enable `--ckpt-fully-parallel-save`** when writing the DCP.
   This produces the `fully_sharded_model_space` layout — the hard requirement for cross-TP reshard.
   `checkpointing.py:1290` raises a `RuntimeError` if `ckpt_tp_pp != run_tp_pp` and `sharding_type` is not `fully_sharded_model_space`.
2. **The first torch-format checkpoint loaded in Stage A must be TP1/PP1** (or match the run's TP/PP).
   Otherwise the first load itself needs `tools/checkpoint/loader_mcore.py` to handle structural alignment.
3. **Parameter key names must match between Stage A and Stage B**.
   - You can swap implementations (TE ↔ Mcore ↔ Apex `ColumnParallelLinear`, etc.; state-dict keys are `weight`/`bias`).
   - You can toggle adapter `ADAPTER_TYPE` between `LINEAR` and `TP_LINEAR` (same keys).
   - But operations that change keys/shapes — e.g. `Conv2d` ↔ `Linear`, fused QKV vs. separate Q/K/V — **cannot be directly resharded**.
4. **Vocab padding mismatch is acceptable** (e.g. 151936 → 152064). DCP will auto pad/truncate; you only get a warning.
5. **FP8 `_extra_state`** is handled by commit `27e12f92` (`fix_te.py` + `te_env.py`).
   If the source side has TE operator `_extra_state` and the target side uses a non-TE implementation, set on the target:
   `TE_EXTRA_STATE_MISSING_CHECK=1` and `TE_FILTER_MODULE_OPT="<modules>;<ops>"`
   (modules and ops must be 1:1 paired).

---

## 3. Stage A: load torch ckpt under `{TP1, PP1}`, train and save DCP

### 3.1 Key script arguments

```bash
TP=1
PP=1
NSTEP=40                       # total training steps; save at the end
SAVE_INTERVAL=${NSTEP}         # checkpoint at the final step

TRAINING_ARGS=(
    ...
    --save              "$OUT_A"
    --save-interval     "$SAVE_INTERVAL"
    --ckpt-format       torch_dist
    --ckpt-fully-parallel-save          # ★ required
    --load              "$TORCH_CKPT_DIR"     # original torch-format checkpoint
    --auto-detect-ckpt-format
    --finetune                          # torch ckpt also loaded with --finetune; no optimizer
    --no-load-rng
    ...
)

MODEL_PARALLEL_ARGS=(
    --pipeline-model-parallel-size "$PP"   # 1
    --tensor-model-parallel-size   "$TP"   # 1
    --use-distributed-optimizer
    --sequence-parallel                 # SP is allowed even with TP1
    ...
)
```

### 3.2 Run and artifacts

```bash
bash examples/llava_onevision2/quick_start_video_2b/your_stageA.sh
```

After success, `$OUT_A` should contain:

```
$OUT_A/
├── iter_0000040/
│   ├── __0_0.distcp  ...  __7_1.distcp
│   ├── common.pt
│   └── metadata.json        # {"sharded_backend": "torch_dist", ...}
├── latest_checkpointed_iteration.txt   # content: 40
└── dataloader/ ...
```

Verify:

- `cat $OUT_A/latest_checkpointed_iteration.txt` should print `40`
- `cat $OUT_A/iter_0000040/metadata.json | jq .sharded_backend` should print `"torch_dist"`

---

## 4. Stage B: change parallel layout to `{TP4, PP1}`, adjust training parameters, reuse weights only

### 4.1 Key script arguments

```bash
TP=4
PP=1
GBS=8                        # training params can change
SEQ_LEN=20480
NSTEP=100
LR=1.0e-5

# Use the DCP as the weight source
CHECKPOINT_PATH="$OUT_A"     # point to Stage A's save dir

TRAINING_ARGS=(
    ...
    --train-iters        "$NSTEP"
    --lr                 "$LR"
    --lr-decay-iters     "$NSTEP"
    --lr-decay-style     cosine
    --lr-warmup-fraction 0.002
    --global-batch-size  "$GBS"
    --seq-length         "$SEQ_LEN"

    --save               "$OUT_B"
    --save-interval      50
    --ckpt-format        torch_dist
    --ckpt-fully-parallel-load          # optional on the load side, speeds up parallel read

    --load               "$CHECKPOINT_PATH"    # Stage A's DCP directory
    --auto-detect-ckpt-format
    --finetune                              # ★ load weights only
    --no-load-rng                           # ★ do not restore RNG
    --override-opt_param_scheduler          # ★ redundant but harmless, forces script's scheduler

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

Optional environment variables:

```bash
export PATCH_EMBED_TYPE="${PATCH_EMBED_TYPE:-LINEAR}"   # or TP_LINEAR / CONV2D
export SCATTER_BEFORE_PATCH_EMBED="${SCATTER_BEFORE_PATCH_EMBED:-0}"
export TE_EXTRA_STATE_MISSING_CHECK=1                  # enable _extra_state filtering
export TE_FILTER_MODULE_OPT="vision_model,adapter;patch_embed,fc1"  # modules;ops 1:1 paired
```

### 4.2 Run

```bash
bash examples/llava_onevision2/quick_start_video_2b/your_stageB.sh
```

### 4.3 Log checkpoints (key lines) for the load phase

A successful run should show:

```
successfully loaded checkpoint from ... [ t 1/4, p 1/1 ] at iteration 40
> loaded checkpoint with TP=1 PP=1, reshard to run TP=4 PP=1 ...
... overriding lr_max ... to <script value>          # override-opt_param-scheduler takes effect
... checkpoint version 0.0
... [before the start of training step] datetime: <Stage B launch time>
```

After that, the run should enter the first training step and print loss/tflops as usual. If the log silently ends within ~3 seconds after `[before the start of training step]` — TensorBoard has no scalars, GPUs are idle, and `dmesg` shows no OOM — then the process was killed externally, not a load issue (see the troubleshooting section).

---

## 5. `--finetune` Behavior Matrix (from `aiak_megatron/megatron/training/checkpointing.py`)

| Loaded item                | `--finetune` (this playbook)                              | Resume without `--finetune`          |
| -------------------------- | --------------------------------------------------------- | ------------------------------------ |
| Model weights              | ✅ Loaded                                                 | ✅ Loaded                            |
| Optimizer state            | ❌ Not loaded (`checkpointing.py:1425`)                   | ✅ Loaded                            |
| RNG state                  | ❌ Not loaded                                             | ✅ Loaded (`checkpointing.py:1472`)  |
| `opt_param_scheduler`      | ❌ Not loaded; uses current script (`checkpointing.py:1446`) | ✅ Loaded                          |
| Starting `iteration`       | Forced to `0` (`checkpointing.py:1377`)                   | Reads `iteration` from checkpoint    |
| `consumed_train_samples`   | Not read from ckpt; uses script (`checkpointing.py:1399` skip) | Restored from checkpoint       |
| `rerun_state_machine`      | Not loaded                                                | Loaded only when TP/PP match         |

Conclusion: `--finetune` is precisely the official switch for "take the weights only, drop all training state", which fully matches the playbook's needs. `--no-load-rng` is redundant under `--finetune` but explicit is better for readability.

---

## 6. Why You Can Skip the Megatron-Core Parallel-Conversion Scripts

With the `fully_sharded_model_space` layout, `torch.distributed.checkpoint.load` will:

1. Read the source metadata and gather all shards back into the full tensor.
2. Re-slice the full tensor according to the target's shard description from `generate_state_dict` (`checkpointing.py:1326`).
3. Distribute the slices to the target TP/PP ranks.

This path covers everything `tools/checkpoint/loader_mcore.py` does for TP/PP inter-conversion. Where structural alignment is still required (HF → mcore layer-name mapping, qkv fusion, ln placement, etc.), DCP does not help — so **only the very first HF → mcore conversion is unavoidable**; after that, mcore↔mcore TP/PP inter-conversion can all go through DCP.

---

## 7. Operator Swap (TE → Mcore / Apex) Compatibility

| Switch type                                                       | Can be loaded directly from DCP? | Notes                                                                 |
| ----------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------- |
| `ColumnParallelLinear` library swap (TE / Mcore / `apex.transformer`), bf16 | ✅                            | Keys are `weight`/`bias`; shard axis matches                         |
| `ADAPTER_TYPE = LINEAR ↔ TP_LINEAR`                               | ✅                                | Same keys; only the TP partition differs and is handled by reshard    |
| `FusedRMSNorm` ↔ `FusedLayerNorm` (bias presence differs)         | ⚠️                                | If source has `bias` but target only has `weight`, bias is silently dropped |
| TE FP8 op → non-TE op                                             | ⚠️                                | Need `TE_FILTER_MODULE_OPT` to actively filter the source `_extra_state` |
| `Conv2d` ↔ `Linear` (weight flatten dim differs)                 | ❌                                | DCP does not reshape; a manual conversion is required                 |
| Fused QKV ↔ three separate Linears                                | ❌                                | Concatenation order differs; DCP reports `size mismatch`               |

---

## 8. Troubleshooting

| Symptom                                                                                                    | Cause                                                              | Fix                                                                                 |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `RuntimeError: ... not supported for DistributedOptimizer with sharding type dp_zero_gather_scatter`        | Stage A did not enable `--ckpt-fully-parallel-save`                | Re-run Stage A with the flag                                                         |
| `size mismatch ... saved torch.Size([151936, 2048]) and current: torch.Size([152064, 2048])`                | Vocab padding warning, **can be ignored**                          | DCP auto pad/truncate                                                                |
| `Unable to load optimizer from checkpoint ... Specify --no-load-optim or --finetune`                        | Missing `--finetune`; optimizer load attempt failed                | Add `--finetune`                                                                     |
| Log silently ends within ~3 s after `[before the start of training step]`; no TB scalars; GPUs idle; no OOM in `dmesg` | Process killed externally (scheduler/lease/OOM-killer not recorded) | Check scheduler/CI/lease; add `trap` in the script and `tee` stderr to a separate file |
| `Filtering out modules ['vision_model'] ...`                                                                | `fix_te.py` is filtering `_extra_state`, **normal**                | No action needed                                                                    |
| `TE_FILTER_MODULE_OPT: modules count (X) != ops count (Y)`                                                  | `te_env.py:get_te_filter_modules_and_ops` enforces 1:1 pairing     | Align the counts in the env value                                                   |
| Resumed `iteration = 40` instead of `0`                                                                     | Missing `--finetune`; resumed instead of fresh-start              | `--finetune` forces `iter=0`                                                        |
| `consumed_train_samples` is non-zero                                                                        | Missing `--finetune`; restored from ckpt                           | `--finetune` skips the `args` restore                                               |

---

## 9. Quick Reference

```bash
# Stage A artifact checks
cat $OUT_A/latest_checkpointed_iteration.txt
python -c "import json;print(json.load(open('$OUT_A/iter_0000040/metadata.json'))['sharded_backend'])"

# Stage B launch
bash your_stageB.sh 2>&1 | tee run_stageB.log

# Key log lines
grep -E "successfully loaded|overriding|checkpoint version|before the start|Filtering out" run_stageB.log

# Confirm optimizer state was NOT taken
grep -E "Could not load optimizer|loaded optimizer" run_stageB.log
```

---

## 10. References

- Main load path: `aiak_megatron/megatron/training/checkpointing.py:1200-1480`
- Reshard hard check: `checkpointing.py:1290-1295`
- `--finetune` behavior branches: `checkpointing.py:1264, 1377, 1399, 1425, 1446, 1472`
- `_extra_state` filter: `aiak_megatron/megatron/training/fix_te.py` + `aiak_training_llm/utils/te_env.py` (commit `27e12f92`)
- `override_opt_param_scheduler`: `aiak_megatron/megatron/core/optimizer_param_scheduler.py:211`
- Adapter TP-type switch: `aiak_training_llm/models/llava_onevision2/adapter.py:70-128`
- Reference script: `examples/llava_onevision2/quick_start_video_2b/output/sft_ckpt-dcp-to-dcp_with-parallel-change_PETP_SP_adapterTP/sft_ckpt-dcp-to-dcp_with-parallel-change_PETP_SP_adapterTP.sh`
