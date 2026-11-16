# DSpark 在 Atlas A2 上启动失败（QuaRot 权重对齐 → TransData/EZ3002）修复包 v2

本目录是**已打好补丁的 UCM develop 源码**，用于在「不能改 vllm_ascend 镜像、只能从 UCM 打 patch」
的机器上修复下面这个启动失败。

- 上游基线：`ModelEngine-Group/unified-cache-management` @ `develop` / `6e3dbaa`（2026-11-16）
- 打包提交：`c05a40b`（v1，修 checkpoint 加载路径）+ `5646fc7`（v2，修 draft projection 路径）
- 适用环境：vLLM 0.32.0 + vllm-ascend 0.32.0rc1、Atlas A2（910B）、MiniMax-M3.1-Flash-w8a8 + `method=dspark`

## 1. 现象：两个不同的崩溃点，同一个根因

**v1 之前（第一次启动）**——draft 从 target checkpoint 重建 embed/LM head 时炸：

```
vllm_ascend/models/qwen3_dspark.py:101   align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:70    load_quarot_target_layer(..., rotation, ...)
vllm_ascend/models/llama_eagle3.py:86    rotation = rotation.to(device=layer.weight.device, dtype=torch.float32)
EZ3002 Unsupported_Operator: Optype [TransData] of Ops kernel [DNN_VM_HOST_CPU_OP_STORE]
```

**v1 之后（打了只修 `load_quarot_target_layer` 的补丁）**——栈前进了，但仍炸在同一条路上：

```
vllm_ascend/models/qwen3_dspark.py:101   align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:48    weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
vllm/model_executor/parameter.py:124     __torch_function__
RuntimeError: operator(): OpParamMaker.cpp:400 OPS function error: Identity, error code is 500002
EZ3002 Unsupported_Operator: Optype [TransData] ... groups should be greater than 1, but now is 1
```

## 2. 根因

`align_draft_weights()` 里有**两条**旋转路径，v1 只覆盖了第二条：

1. **draft 自己的 projection（`self.model.fc` / `self.model.context_proj`）**——第 44-48 行，
   先把 device 上的权重 `.cpu()` 下来，用 `process_weight()` 在 host 上算好，再
   **`.to(weight.device)` 整体搬回 device**。这个 `Tensor.to(device)` 就是第一次崩溃的那个 op。
2. **从 checkpoint 重建 embed_tokens / lm_head**——第 58-73 行，走 `load_quarot_target_layer`，
   v1 已经换成 host 侧分批对齐。

为什么 `.to(device)` 会炸而 vLLM 加载权重不炸：`Tensor.to(device)` 在 torch_npu 上被 lower 成
单算子图里的一个 `Identity` 节点，GE 为它补了一个 `TransData`，而这个 TransData 的 `groups` 属性
在这条路径上被推成 1；A2 + 当前 CANN 的 AICPU op store（`DNN_VM_HOST_CPU_OP_STORE`）没有
`groups=1` 的 TransData 内核，于是 `EZ3002`。而 `param.copy_(cpu_tensor)`（vLLM 加载每个模型都用）
走的是同 dtype 的直接拷贝，不建这个图，所以一直没问题。

注意错误是**异步**抛的：日志里那串 `MoeDistributeDispatchV2` / hifloat4_scale 初始化失败是 GE
初始化算子库时的噪声，真正失败的节点是 `trans_TransData_0`；栈里显示 `Identity` 也是同一原因
（stacktrace may be inaccurate）。

## 3. 改动（只动 UCM，不动 vllm_ascend）

`ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py`：

- 替换 `llama_eagle3.load_quarot_target_layer`：host 侧按 4096 行分批做
  `target_shard.float() @ rotation.float().T`，cast 到 `layer.weight.dtype` 后用
  `layer.weight.data[start:stop].copy_(...)` 写回，尾部行清零。
- **（v2 新增）** 替换 `qwen3_dspark.align_draft_weights`：保留上游的所有权判断与 PP 回退逻辑，
  只把那行改成
  `weight.data.copy_(process_weight(weight.cpu(), rotation))`——host 上算完，同 dtype 写回，
  全程不出现 `Tensor.to(device)`。
- 因为 `qwen3_dflash2` / `kimi_k3_dspark` / `glm5_dspark` 都是
  `from vllm_ascend.models.qwen3_dspark import align_draft_weights`（各自持有一份绑定），
  这三个模块的属性也一并替换；另外 hook 了 `vllm_ascend.models.quarot_utils`
  （vllm-ascend#21783 正在把 helper 迁到该模块），并用 `sys.modules` 扫描兜底"模块先导入、补丁后到"。
- 语义不变：`process_weight` 仍是"按 hidden 分块、fp32 matmul、cast 回原 dtype"，只是不复用
  device 中转。附带收益：fp32 中间结果不再占显存。

在 `apply_patch.py` 的 `apply_all_patches()` 末尾加了一行 import 注册。

## 4. 为什么从 UCM 打能打到 worker 里

site-packages 的 `ucm_patch.pth` 在解释器启动时装一个 meta-path finder，**每个进程第一次
`import vllm` 结束后调用 `apply_all_patches()`**（日志里每个 pid 一条
`UCM patch initialization completed!`），所以 engine core 和 8 个 worker 都会执行到。
补丁模块本身只注册惰性钩子，等 vllm_ascend 的 DSpark 模块被导入时才落地——正好在
`load_draft_model` 内部、`post_process()` 之前。

## 5. 安装

前提：`ENABLE_UCM_PATCH=1`（启动脚本里已有）。

### 方式 A：整包替换

```bash
tar xzf unified-cache-management-develop-6e3dbaa-dspark.tar.gz -C /
python -c "import ucm, os; print(os.path.dirname(ucm.__file__))"   # 确认目标目录
rsync -a --delete unified-cache-management-develop/ <上面打印的父目录>/
find <上面打印的父目录> -name '__pycache__' -prune -exec rm -rf {} +
```

### 方式 B：只打两个文件的增量

```bash
cd <UCM 根目录>
cp dspark_quarot_host_align_patch.py ucm/integration/vllm/patch/
git apply -p1 0001-add-dspark-quarot-host-align.patch     # 已校验可对 develop@6e3dbaa 干净应用
python -c "import py_compile;py_compile.compile('ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py',doraise=True);py_compile.compile('ucm/integration/vllm/patch/apply_patch.py',doraise=True);print('compile ok')"
find ucm -name '__pycache__' -prune -exec rm -rf {} +
```

若 `apply_patch.py` 上下文对不上，手工在 `apply_all_patches()` 的 try 块里加一行：

```python
        import ucm.integration.vllm.patch.v0300.mamba_layer_hooks_patch
+       import ucm.integration.vllm.patch.dspark_quarot_host_align_patch
        logger.info("UCM patch initialization completed!")
```

然后按原脚本重启 `vllm serve`。若 UCM 是**非 editable 安装**（site-packages 里一份拷贝），
新增 `.py` 需要 `pip install --force-reinstall --no-deps .`，或直接改那份拷贝。

## 6. 验证

1. 进程级（每个 worker 各一条）：

   ```
   UCM QuaRot host-alignment patch applied to vllm_ascend.models.llama_eagle3 (load_quarot_target_layer)
   UCM QuaRot host-alignment patch applied to vllm_ascend.models.qwen3_dspark (align_draft_weights)
   ```

2. 权重级（每个 rank，先 projection 再 embed/lm_head）：

   ```
   UCM QuaRot host alignment: rotated draft projection (4096, 4096)
   UCM QuaRot host alignment: loaded draft embed_tokens.weight from model-00001-....safetensors (...) (rotated=True)
   UCM QuaRot host alignment: loaded draft lm_head.weight from ... (rotated=True)
   ```

3. 原来那两条栈（`llama_eagle3.py:86` 和 `qwen3_dspark.py:48`）都不再出现。

4. `python verify_ucm_patch.py`（可选，需 torch）：校验两个函数都被替换、输出仍等于
   `w.float() @ R.T`、尾部行清零。

5. 起来后对比 DSpark 接受长度，确认 draft 权重没算错。

离线自测（不需要 torch/NPU，任何有 python+numpy 的机器都能跑，已全部 PASS）：

- `_selftest/run.py`：先打补丁再导入 dspark 模块；校验 loader 数值（fp32/bf16、`rotation=None`）。
- `_selftest/run_preloaded.py`：vllm_ascend 先导入再打补丁（`_sweep_aliases` 路径）。
- `_selftest/run_align.py`：**复现第二条崩溃**——stub torch 把任何 `to(device="npu")` 都抛成
  `EZ3002`，先证明未打补丁的 `align_draft_weights` 必炸，再证明打补丁后 projection 被正确旋转
  且全程没有 host→device 的 `.to()`。

## 7. 回滚

```bash
cd <UCM 根目录>
git checkout -- ucm/integration/vllm/patch/apply_patch.py
rm -f ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py
find ucm -name '__pycache__' -prune -exec rm -rf {} +
```

## 8. 边界

- 纯运行时 monkeypatch，不改镜像文件；只对导入过 vllm 的进程生效。
- 依赖 vllm-ascend 0.32.0 的内部实现：`llama_eagle3._find_safetensors_weight` /
  `qwen3_dspark.process_weight` / `TARGET_*_WEIGHT_NAMES`。升版若改名，补丁会打
  `UCM QuaRot host alignment skipped: ...` 并退回原实现（也就是又会炸），升级时盯这条日志。
- `align_draft_weights` 是"抄一份上游实现 + 改一行"，上游若改所有权逻辑（如 vllm-ascend#21687、
  #21783），需要同步；补丁里对缺失的 helper 做了兜底（缺什么就跳过并记 warning）。
- UCM 自带 `test/suites/Unit/test_apply_patch_version.py` 可能断言 patch 列表，若跑 UCM 单测需同步加一条。

## 9. 版本记录

| 版本 | 提交 | 覆盖的崩溃点 |
| --- | --- | --- |
| v1 | `c05a40b` | `llama_eagle3.py:86`（checkpoint 重建 embed/lm head 的 `rotation.to(device)`） |
| v2 | `5646fc7` | v1 + `qwen3_dspark.py:48`（draft projection 的 `process_weight(...).to(weight.device)`），并 hook `quarot_utils` |

## 10. 本目录内容

| 文件 | 说明 |
| --- | --- |
| `unified-cache-management-develop-6e3dbaa-dspark.tar.gz` | 打好补丁（v2）的完整 UCM develop 源码（git archive，不含 .git） |
| `dspark_quarot_host_align_patch.py` | 新增的 UCM patch 模块（v2） |
| `0001-add-dspark-quarot-host-align.patch` | 对 develop@6e3dbaa 的增量 patch（新文件 + 注册行） |
| `verify_ucm_patch.py` | 机器上的运行时校验脚本 |
| `SHA256SUMS.txt` | 校验和 |
