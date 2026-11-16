# DSpark + QuaRot on Atlas A2: host-side weight alignment

This branch is `develop` (upstream `unified-cache-management`, base `6e3dbaa`) plus a runtime
patch that lets **DSpark speculative decoding start on a QuaRot-quantized target on Atlas A2**.

Without it every worker dies while loading the draft model:

```
vllm_ascend/models/qwen3_dspark.py:101  align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:48   weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
RuntimeError: operator(): OpParamMaker.cpp:400 OPS function error: Identity, error code is 500002
EZ3002 Unsupported_Operator: Optype [TransData] of Ops kernel [DNN_VM_HOST_CPU_OP_STORE] is unsupported.
        Reason: Transdata op, groups should be greater than 1, but now is 1.
No supported Ops kernel and engine are found for [trans_TransData_0]
```

## Why it fails

vllm-ascend aligns the draft (projection, `embed_tokens`, `lm_head`) with the target's rotation
matrix `Q` **on the device**. On A2 the host→device move of those 2-D matrices is lowered to a
single-op graph node `Identity` plus a `TransData` whose `groups` is 1, and the AICPU op store
(host CPU and `aicpu_ascend_kernel`) only ships the grouped variant, so the graph cannot be built.

A plain `param.copy_(cpu_tensor)` is *not* automatically safe either: after
`process_weights_after_loading` a bf16 weight may already be `FRACTAL_NZ`
(`additional_config.weight_nz_mode`), and writing ND data into it needs the same `TransData`.
Weight loading works only because it happens *before* that conversion.

## What this branch changes

New module `ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py`, registered at the end
of `apply_all_patches()`, so every process that imports `vllm` (engine core and all workers)
installs it when `ENABLE_UCM_PATCH=1`.

Four layers, all host-side:

1. `llama_eagle3.load_quarot_target_layer` — chunked host alignment
   (`target.float() @ Q.T`, written in ≤8 MiB blocks, unused tail zeroed).
2. `qwen3_dspark.align_draft_weights` — same ownership/PP rules as upstream, but the projection is
   rotated on the host and written into a **fresh ND** device buffer; the layout the layer expects
   (`maybe_trans_nz`) is re-applied afterwards. Rebound on `qwen3_dspark`, `qwen3_dflash2`,
   `kimi_k3_dspark`, `glm5_dspark` and `quarot_utils`.
3. `qwen3_dspark.process_weight` — rotates on the host for a device input too, instead of moving
   `Q` to the device.
4. `safe_tensor_to` — size-gated net on `Tensor.to` / `Tensor.npu`: large non-parameter CPU→NPU
   moves go through `empty(...) + copy_` in blocks. Kill switch `UCM_QUAROT_SAFE_TO=0`, threshold
   override `UCM_QUAROT_SAFE_TO_MIN_NUMEL`.

Nothing outside UCM is modified; no image rebuild, no site-packages change.

## Use

```bash
# this branch is the whole tree; install it like any other UCM checkout
ENABLE_UCM_PATCH=1 vllm serve <model> --speculative-config '{"method":"dspark", ...}'
```

Expected log lines (one per worker / per rank):

```
UCM QuaRot safe Tensor.to installed (>= 1048576 elements, ...)
UCM QuaRot host-alignment patch applied to vllm_ascend.models.llama_eagle3 (load_quarot_target_layer)
UCM QuaRot host-alignment patch applied to vllm_ascend.models.qwen3_dspark (process_weight, align_draft_weights)
UCM QuaRot host alignment: rotated draft projection (4096, 4096) (acl format 29 -> 29, nz_mode=2)
UCM QuaRot host alignment: loaded draft embed_tokens.weight ... (rotated=True)
UCM QuaRot host alignment: loaded draft lm_head.weight ... (rotated=True)
```

`acl format 29 -> 29` means the destination was already `FRACTAL_NZ` and is written back as
`FRACTAL_NZ` through a fresh ND buffer; `? -> ?` means the ACL format could not be read.

## Diagnostics

`tools/probe_a2_h2d.py` (single card) reports which host→device transfer forms this box accepts:
ND target, fp32 source, `.to('npu')`, `FRACTAL_NZ` target whole vs. chunked, and the NZ→ND cast.
Run it if a copy still fails, and keep its output with the bug report.

## Known limits

- Depends on vllm-ascend 0.32.0 internals (`_find_safetensors_weight`, `process_weight`,
  `TARGET_*_WEIGHT_NAMES`, the ownership flags). If upstream renames them the patch logs
  `UCM QuaRot host alignment skipped: ...` and falls back to the stock (failing) code.
- `align_draft_weights` is a copy of upstream logic with a different write-back; it must be
  re-synced if upstream changes ownership handling (vllm-ascend#21687, #21783).
- Rotating on the host costs one one-time host matmul per rank (seconds), and lowers both host and
  device peak memory compared with staging the fp32 product on the device.
