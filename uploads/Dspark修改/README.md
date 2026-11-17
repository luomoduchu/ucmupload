# DSpark 在 Atlas A2 上启动失败（QuaRot 权重对齐 → TransData/EZ3002）修复包 v3

本目录是**已打好补丁的 UCM develop 源码**，用于在「不能改 vllm_ascend 镜像、只能从 UCM 打 patch」
的机器上修复下面这个启动失败。

- 上游基线：`ModelEngine-Group/unified-cache-management` @ `develop` / `1771cfc`（2026-11-17）
- 打包提交：`c05a40b`(v1) → `5646fc7`(v2) → `b75e72b`(v3)
- 适用环境：vLLM 0.32.0 + vllm-ascend 0.32.0rc1、Atlas A2（910B）、MiniMax-M3.1-Flash-w8a8 + `method=dspark`

## 1. 现象：栈一路往前走，每修一处换一行

```
vllm_ascend/models/qwen3_dspark.py:101   align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:70    load_quarot_target_layer(..., rotation, ...)
vllm_ascend/models/llama_eagle3.py:86    rotation = rotation.to(device=layer.weight.device, dtype=torch.float32)   # v1 修掉
```
```
vllm_ascend/models/qwen3_dspark.py:101   align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:48    weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))    # v2 修掉
vllm/model_executor/parameter.py:124     __torch_function__
RuntimeError: operator(): OpParamMaker.cpp:400 OPS function error: Identity, error code is 500002
EZ3002 Unsupported_Operator: Optype [TransData] ... groups should be greater than 1, but now is 1
```

两者都是 `EZ3002` / `TransData`，都不是 UCM 的问题。

## 2. 根因

**A2 上「host 上新建的大张量整体搬去 device」这条路走不通。**

`Tensor.to(device=...)`（以及 `Tensor.npu()`）在 torch_npu 上被 lower 成单算子图里的一个
`Identity` 节点，GE 为它补一个 `TransData`，这条路径上 `groups` 被推成 1；A2 + 当前 CANN 的
AICPU op store（`DNN_VM_HOST_CPU_OP_STORE`）没有 `groups=1` 的 TransData 内核 → `EZ3002`。

对照证据（同一台机器、同一次启动里都在跑）：

| 写法 | 结果 | 说明 |
| --- | --- | --- |
| `param.copy_(cpu_tensor)`（vLLM 加载每个模型权重） | 正常 | 同 dtype 直拷，不建这个图 |
| `w8a8` 的 `torch.from_numpy(...).to(scale.device)`（KB 级 1-D scale） | 正常 | 小张量不触发 |
| `rotation.to(device, dtype=fp32)`（4096×4096） | **EZ3002** | 16 MiB 2-D 矩阵 |
| `process_weight(...).to(weight.device)`（hidden×3hidden） | **EZ3002** | 更大的 2-D 矩阵 |

错误是**异步**抛的，所以栈会漂：日志里 `MoeDistributeDispatchV2` / `hifloat4_scale` 初始化失败
是 GE 初始化算子库时的噪声，真正失败的节点是 `trans_TransData_0`；栈里显示 `Identity`、
`parameter.py:__torch_function__` 也只是"最近一个同步点"。

## 3. 审计：vllm-ascend 里所有命中这个模式的 QuaRot 路径

`vllm_ascend` 全量 grep（`.to(device=…)`、`.to(X.device)`、`.npu()`、`torch.tensor(...,device=)`、
`quarot|rotation`），并逐个读上下文后的结论：

| # | 位置 | 表达式 | 会被本次修复覆盖吗 |
| --- | --- | --- | --- |
| 1 | `qwen3_dspark.py:48` `align_draft_weights` | `process_weight(...).to(weight.device)` | 层 2（直接替换函数） |
| 2 | `llama_eagle3.py:86` `load_quarot_target_layer` | `rotation.to(device=…, dtype=fp32)` | 层 1（直接替换函数） |
| 3 | `llama_eagle3.py:90` 同函数 | `target_weight.to(device=…, dtype=fp32)`（整块 embedding 分片） | 层 1（整函数替换，已按 4096 行分块） |
| 4 | `qwen3_dspark.py:29` `process_weight` | `rotation_weight.to(device=linear_weight.device)` | 层 3-A（替换为 host 计算） |
| 5 | `deepseek_v41/model.py:1108` Engram | `engram_rotation.to(device=self.device)`（QuaRot 门控，target 加载期） | 层 3-B（只有 `(num_blocks,32,32)` 这种尺寸，见 §8 阈值） |
| 6 | `w8a8_dynamic.py:54` / `w4a8.py:295` / `w4a4_flatquant.py:34` | `.to(scale.device)` / `.npu()` | 层 3-B（这些是 int 位重解释，`copy_` 为同 dtype 直拷，语义不变） |
| 7 | `turboquant/latent.py:86`、`deepseek_v4{,_41}/vision.py`、`xlite.py` | `torch.tensor(...,device=)` / `.to(device=x.device)` | 3-B 覆盖大张量；与本次场景无关 |

明确**不需要**改的：`llama_eagle3.py:113-150`（Eagle3 的 `load_weights`）本来就是 host 侧
`@ Q3` / `@ Q.T`，是这套代码里"正确写法"的参考实现；`kimi_k3.py:1256/1273` 是
`nn.Module.to()`（参数迁移，走的是另一条路，且已被层 3-B 的 `requires_grad` 条件排除）。

## 4. 改动：三层，全部只在 UCM 内

`ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py`：

**层 1 — 替换 `llama_eagle3.load_quarot_target_layer`**
host 侧按 4096 行分批 `target_shard.float() @ rotation.float().T`，cast 到 `layer.weight.dtype`
后用 `layer.weight.data[start:stop].copy_(...)` 写回，尾部行 `zero_()`。
（顺带把原来那条 3.1 GB 的 fp32 H2D 传输消掉，host 峰值内存也降了。）

**层 2 — 替换 `align_draft_weights`**
保留上游的全部所有权判断与 PP 回退逻辑，只把写回改成
`aligned = module.process_weight(weight.cpu(), rotation)` + `weight.data.copy_(aligned)`。
因为 `qwen3_dflash2` / `kimi_k3_dspark` / `glm5_dspark` 都是
`from vllm_ascend.models.qwen3_dspark import align_draft_weights`（各自持一份绑定），
这三个模块的属性也一并替换；另外 hook 了 `vllm_ascend.models.quarot_utils`
（vllm-ascend#21783 正在把 helper 迁到该模块），并用 `sys.modules` 扫描兜底"模块先导入、补丁后到"。

**层 3-A — 替换 `process_weight`**
上游是 `rotation.to(device=linear_weight.device)` 开头，只要调用方传进来的是 device 张量就必然
搬运。新实现：device 输入 → `.cpu()` 后在 host 上算，再 `empty(...) + copy_` 回 device；
CPU 输入原样交给上游实现。返回值仍在调用方 device 上，语义不变。

**层 3-B — `safe_tensor_to` 兜底网**
包装 `torch.Tensor.to` / `torch.Tensor.npu`：**只**拦截
`源在 CPU` + `目标非 CPU` + `numel ≥ 阈值`（默认 1 Mi）+ `requires_grad=False` 的调用，
改成 `torch.empty(shape, dtype, device=目标) + copy_`。
- 语义完全等价（都是"分配 + 拷贝"，dtype 变化仍在 host 上先 cast）；
- `requires_grad=True` 的参数不拦，所以 `nn.Module.to()` 的模型迁移不受影响；
- 小张量（ids、mask、量化 scale）不拦，热路径开销为零；
- 开关：`UCM_QUAROT_SAFE_TO=0` 关闭；阈值：`UCM_QUAROT_SAFE_TO_MIN_NUMEL=1024` 可把
  DSv4.1 Engram 那种 Ki 级矩阵也纳进来。

在 `apply_patch.py` 的 `apply_all_patches()` 末尾加了一行 import 注册；层 3-B 在补丁模块
被 import 时就装好（`apply_all_patches()` 在 `import vllm` 之后立刻执行，早于任何模型构造）。

## 5. 为什么从 UCM 打能打到 worker 里

site-packages 的 `ucm_patch.pth` 在解释器启动时装一个 meta-path finder，**每个进程第一次
`import vllm` 结束后调用 `apply_all_patches()`**（日志里每个 pid 一条
`UCM patch initialization completed!`），所以 engine core 和 8 个 worker 都会执行到。
补丁模块本身只注册惰性钩子，等 vllm_ascend 的模块被导入时才落地——正好在
`load_draft_model` 内部、`post_process()` 之前。

## 6. 安装

前提：`ENABLE_UCM_PATCH=1`（启动脚本里已有）。

### 方式 A：整包替换

```bash
tar xzf unified-cache-management-develop-1771cfc-dspark.tar.gz -C /
python -c "import ucm, os; print(os.path.dirname(ucm.__file__))"   # 确认目标目录
rsync -a --delete unified-cache-management-develop/ <上面打印的父目录>/
find <上面打印的父目录> -name '__pycache__' -prune -exec rm -rf {} +
```

### 方式 B：只打两个文件的增量

```bash
cd <UCM 根目录>
cp dspark_quarot_host_align_patch.py ucm/integration/vllm/patch/
git apply -p1 0001-add-dspark-quarot-host-align.patch     # 已校验可对 develop@1771cfc 干净应用
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

## 7. 验证

1. 进程级（每个 worker 各一条）：

   ```
   UCM QuaRot safe Tensor.to installed (>= 1048576 elements, ...)
   UCM QuaRot host-alignment patch applied to vllm_ascend.models.llama_eagle3 (load_quarot_target_layer)
   UCM QuaRot host-alignment patch applied to vllm_ascend.models.qwen3_dspark (process_weight, align_draft_weights)
   ```

2. 权重级（每个 rank，先 projection 再 embed/lm_head）：

   ```
   UCM QuaRot host alignment: rotated draft projection (4096, 4096)
   UCM QuaRot host alignment: loaded draft embed_tokens.weight from model-00001-....safetensors (...) (rotated=True)
   UCM QuaRot host alignment: loaded draft lm_head.weight from ... (rotated=True)
   ```

3. 原来那些栈（`llama_eagle3.py:86`、`qwen3_dspark.py:48`）都不再出现。

4. `python verify_ucm_patch.py`（可选，需 torch）：校验两个函数都被替换、输出仍等于
   `w.float() @ R.T`、尾部行清零。

5. 起来后对比 DSpark 接受长度，确认 draft 权重没算错。

离线自测（不需要 torch/NPU，任何有 python+numpy 的机器都能跑，四套已全 PASS）：

- `_selftest/run.py`：先打补丁再导入 dspark 模块；loader 数值（fp32/bf16、`rotation=None`）。
- `_selftest/run_preloaded.py`：vllm_ascend 先导入再打补丁（`_sweep_aliases` 路径）。
- `_selftest/run_align.py`：复现第二条崩溃——stub torch 把任何 CPU→NPU `to()` 抛成 `EZ3002`，
  先证明未打补丁的 `align_draft_weights` 必炸，再证明打补丁后 projection 被正确旋转且全程无 `.to()`。
- `_selftest/run_safe_to.py`：层 3 —— 大张量 `.to("npu")` 被改道、低于阈值仍走原实现、
  device 输入 `process_weight` 在 host 上算完写回。

## 8. 边界

- 纯运行时 monkeypatch，不改镜像文件；只对导入过 vllm 的进程生效。
- 层 1/2/3-A 依赖 vllm-ascend 0.32.0 的内部实现（`_find_safetensors_weight`、`process_weight`、
  `TARGET_*_WEIGHT_NAMES`、所有权标志）。升版若改名，补丁打
  `UCM QuaRot host alignment skipped: …` 并退回原实现（也就是又会炸），升级时盯这条日志。
- `align_draft_weights` 是"抄一份上游实现 + 改一行"，上游若改所有权逻辑（如 vllm-ascend#21687、
  #21783），需要同步。
- 层 3-B 是 `torch.Tensor.to` 的进程级包装：只影响"大 + host + 不要梯度"的 CPU→NPU 搬运，
  语义等价但多一层 Python 调用；默认阈值 1 Mi 下热路径不经过。若担心，`UCM_QUAROT_SAFE_TO=0` 关掉，
  层 1/2/3-A 仍有效（等于 v2 行为）。
- DSv4.1 的 Engram rotation 只有 `(32,32)`（1024）或 `(num_blocks,32,32)`（≤~16 万）个元素，
  低于默认阈值；如果将来那条路也炸，把 `UCM_QUAROT_SAFE_TO_MIN_NUMEL` 设小即可。
- UCM 自带 `test/suites/Unit/test_apply_patch_version.py` 可能断言 patch 列表，若跑 UCM 单测需同步加一条。

## 9. 版本记录

| 版本 | 提交 | 内容 |
| --- | --- | --- |
| v1 | `c05a40b` | 只修 `llama_eagle3.load_quarot_target_layer`（`llama_eagle3.py:86/90`） |
| v2 | `5646fc7` | + 替换 `align_draft_weights`（`qwen3_dspark.py:48`），并 hook `quarot_utils` |
| v3 | `b75e72b` | + 替换 `process_weight`，+ `safe_tensor_to` 兜底网；按审计结果覆盖 §3 表中全部路径 |
| v4 | `4a1a58c` | 现场复现后修正写回方式：不再直接写活的参数，改为"新鲜 ND 缓冲区 + 分块拷贝 + 重新施加 NZ 布局" |
| v5 | `a37d56f` | 探测证明写入方向正常，补上读回方向：读 weight.cpu() 前显式做 NZ→ND cast，避免 D2H 的格式转换 |

## 10. 本目录内容

| 文件 | 说明 |
| --- | --- |
| `unified-cache-management-develop-1771cfc-dspark.tar.gz` | 打好补丁（v5）的完整 UCM develop 源码（git archive，不含 .git） |
| `dspark_quarot_host_align_patch.py` | 新增的 UCM patch 模块（v3） |
| `0001-add-dspark-quarot-host-align.patch` | 对 develop@1771cfc 的增量 patch（新文件 + 注册行） |
| `verify_ucm_patch.py` | 机器上的运行时校验脚本 |
| `probe_a2_h2d.py` | 单卡探测脚本：判断 EZ3002 到底由"目标布局（FRACTAL_NZ）"还是"传输尺寸"触发 |
| `SHA256SUMS.txt` | 校验和 |

## 11. v4：为什么 v3 还炸，改了什么

现场日志（`qwen3_dspark.py:101 → align_draft_weights → _rotate_projection → weight.data.copy_(aligned)`）

```
RuntimeError: operator(): OpParamMaker.cpp:400 OPS function error: Identity, error code is 500002
Optype [TransData] of Ops kernel [DNN_VM_HOST_CPU_OP_STORE] ... groups should be greater than 1, but now is 1
Optype [TransData] of Ops kernel [aicpu_ascend_kernel]   ... groups should be greater than 1, but now is 1
No supported Ops kernel and engine are found for [trans_TransData_0]
```

说明我 v1–v3 的一个前提错了：**"`copy_` 一定安全"不成立**。同一次启动里 draft 自己的 `fc.weight`
是能被正常加载进去的，差别在**目标张量的状态**——`process_weights_after_loading` 之后，bf16 权重
可能已经被 `maybe_trans_nz` 转成 FRACTAL_NZ（`additional_config.weight_nz_mode`），此后任何
"host ND → device NZ" 的拷贝都会被 lower 成那个 groups=1 的 TransData。

v4 的写回流程改成和"正常加载 + 后处理"完全一致的顺序：

1. host 上算好旋转结果（dtype 已对齐）；
2. `torch.empty(...)` 造一个**全新的 ND** device 张量；
3. 按 8 MiB 分块 `copy_` 进去（避免单个大传输）；
4. 用 `maybe_trans_nz` 重新施加该层需要的布局（ND 时是 no-op，所以不会二次量化）；
5. `weight.data = 新的张量`。

日志会打印 `acl format before -> after` 与 `nz_mode`，例如：

```
UCM QuaRot host alignment: rotated draft projection (4096, 4096) (acl format 29 -> 29, nz_mode=2)
```

- `format ? -> ?`：拿不到 ACL 格式（正常，非 A2 或 torch_npu 无该接口）；
- `29 -> 29`：目标原本就是 NZ，现在按 NZ 写回，正是修复点；
- `0 -> 29` / `0 -> 0`：原本是 ND，按层策略决定是否转 NZ。

若 v4 仍失败，请跑 `probe_a2_h2d.py`（单卡即可）并把输出发回来，它会直接指出是"大张量传输"
还是"NZ 布局"的问题。

