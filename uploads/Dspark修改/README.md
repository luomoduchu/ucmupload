# DSpark 在 Atlas A2 上启动失败（QuaRot 权重对齐 → TransData/EZ3002）修复包

本目录是**已打好补丁的 UCM develop 源码**，用于在「不能改 vllm_ascend 镜像、只能从 UCM 打 patch」
的机器上修复下面这个启动失败。

- 上游基线：`ModelEngine-Group/unified-cache-management` @ `develop` / `6e3dbaa`（2026-11-16）
- 打包提交：`c05a40b feat(dspark): align QuaRot draft weights on the host`
- 适用环境：vLLM 0.32.0 + vllm-ascend 0.32.0rc1、Atlas A2（910B）、MiniMax-M3.1-Flash-w8a8 + `method=dspark`

## 1. 现象

`vllm serve ... --speculative-config '{"method":"dspark",...}'` 时，8 个 worker 全部在 draft 加载阶段退出；
关掉 DSpark 就正常。

```
worker/v2/model_runner.py:350                     super().load_model()
vllm/gpu/model_runner.py:417                      self.speculator.load_model(...)
vllm/gpu/spec_decode/speculator.py:228            self.load_draft_model(...)
vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:98   model.post_process(vllm_config)
vllm_ascend/models/qwen3_dspark.py:101            align_draft_weights(self, self.model.fc, vllm_config)
vllm_ascend/models/qwen3_dspark.py:70             load_quarot_target_layer(..., rotation, ...)
vllm_ascend/models/llama_eagle3.py:86             rotation = rotation.to(device=layer.weight.device, dtype=torch.float32)

EZ3002 Unsupported_Operator: Optype [TransData] of Ops kernel [DNN_VM_HOST_CPU_OP_STORE] is unsupported.
        Reason: Transdata op, groups should be greater than 1, but now is 1.
```

## 2. 根因

DSpark 的 draft 要从 target checkpoint 重建 `embed_tokens` / `lm_head`，并左乘 QuaRot 的全局旋转矩阵
（`quant_model_description.json` 里的 `optional.quarot.rotation_map.global_rotation`）。上游实现把
**rotation 和 shard 都 `.to(device=npu, dtype=fp32)` 后在 device 上做 fp32 MatMul**；在 A2 + 当前 CANN 上，
这个 2D fp32 矩阵的 host→device 拷贝被拆出一个 `TransData(groups=1)`，AICPU op store 不接受，于是抛
`EZ3002`。异常是异步抛的（栈里显示 `Identity`），真实失败点就是这次拷贝/MatMul 链路上的 TransData。

关掉 DSpark 时 `align_draft_weights()` 根本不会被调用，QuaRot 矩阵永远不上 device，所以能正常起来。

## 3. 改动（只动 UCM，不动 vllm_ascend）

新增 `ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py`：

- 用 `@when_imported("vllm_ascend.models.llama_eagle3")` 和
  `@when_imported("vllm_ascend.models.qwen3_dspark")` 在 DSpark 模块被导入时替换
  `load_quarot_target_layer`（`qwen3_dspark` 是 `from ... import` 的独立绑定，必须一起换；
  另有一遍 `sys.modules` 扫描兜底“模块先导入、补丁后到”的顺序）。
- 替换后的实现把**旋转、dtype cast、fp32 中间结果全部留在 host 内存**，按 4096 行分批写入 device 参数，
  使最后的 H2D 只是同 dtype 的普通 memcpy。
- 语义不变：仍是 `target_shard.float() @ rotation.float().T`，未用到的尾部行清零；
  `rotation=None`（非 QuaRot / PP 场景）走同样的分批 plain copy。
- 顺带收益：全词表 fp32 中间结果不再占显存（8 个 rank 原本合计几十 GB）。

在 `ucm/integration/vllm/patch/apply_patch.py` 的 `apply_all_patches()` 末尾加了一行 import 注册
（`logger.info("UCM patch initialization completed!")` 之前）。

## 4. 为什么从 UCM 打能打到 worker 里

site-packages 的 `ucm_patch.pth` 在解释器启动时装一个 meta-path finder，**每个进程第一次 `import vllm`
结束后调用 `apply_all_patches()`**（日志里每个 pid 一条 `UCM patch initialization completed!`），
所以 engine core 和 8 个 worker 都会执行到。补丁模块本身只注册惰性钩子，等 vllm_ascend 的 DSpark 模块
被导入时才落地——正好在 `load_draft_model` 内部、`post_process()` 之前。

## 5. 安装

前提：`ENABLE_UCM_PATCH=1`（启动脚本里已有）。

### 方式 A：整包替换（推荐，等价于本目录的 tar 包）

```bash
cd / && tar xzf unified-cache-management-develop-6e3dbaa-dspark.tar.gz -C /
# 得到 unified-cache-management-develop/；把它同步到你实际 import 的 UCM 目录
python -c "import ucm, os; print(os.path.dirname(ucm.__file__))"   # 确认目标目录
rsync -a --delete unified-cache-management-develop/ <上面打印的父目录>/
find <上面打印的父目录> -name '__pycache__' -prune -exec rm -rf {} +
```

### 方式 B：只打两个文件的增量（适合 UCM 目录被 bind-mount、想最小改动）

```bash
cd <UCM 根目录>
cp dspark_quarot_host_align_patch.py ucm/integration/vllm/patch/
git apply -p1 0001-add-dspark-quarot-host-align.patch     # 已校验可对 develop@6e3dbaa 干净应用
python -c "import py_compile;py_compile.compile('ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py',doraise=True);py_compile.compile('ucm/integration/vllm/patch/apply_patch.py',doraise=True);print('compile ok')"
find ucm -name '__pycache__' -prune -exec rm -rf {} +
```

若 `apply_patch.py` 上下文对不上（版本不同），手工在 `apply_all_patches()` 的 try 块里加一行：

```python
        import ucm.integration.vllm.patch.v0300.mamba_layer_hooks_patch
+       import ucm.integration.vllm.patch.dspark_quarot_host_align_patch
        logger.info("UCM patch initialization completed!")
```

然后按原脚本重启 `vllm serve`（只重启服务进程，无需重建 vllm_ascend 镜像）。
若 UCM 是**非 editable 安装**（site-packages 里一份拷贝），新增 `.py` 需要
`pip install --force-reinstall --no-deps .`，或直接改那份拷贝。

## 6. 验证

1. 进程级（每个 worker 各一条）：`UCM QuaRot host-alignment patch applied to vllm_ascend.models.llama_eagle3`
2. 权重级（每个 rank 两条，`rotated=True` 表示确实做了旋转对齐）：
   ```
   UCM QuaRot host alignment: loaded draft embed_tokens.weight from model-00001-....safetensors (...) (rotated=True)
   UCM QuaRot host alignment: loaded draft lm_head.weight from ... (rotated=True)
   ```
3. 原来那条 `llama_eagle3.py:86` + `EZ3002 TransData groups=1` 不再出现。
4. `python verify_ucm_patch.py`：校验钩子确实替换了 `qwen3_dspark` 里的绑定，且输出仍等于
   `w.float() @ R.T`、尾部行清零（会把 chunk 缩到 2 行以覆盖分批循环；可用 `cpu` 参数强制 CPU）。
5. 起来后对比 DSpark 接受长度，确认 draft 权重没算错。

离线自测（不需要 torch/NPU）已在提交前跑过：两种导入顺序（先打补丁再导入 / vllm_ascend 先导入再打补丁）
都通过，fp32 与 bf16 数值一致、尾部清零、`rotation=None` 分支正常。

## 7. 回滚

```bash
cd <UCM 根目录>
git checkout -- ucm/integration/vllm/patch/apply_patch.py
rm -f ucm/integration/vllm/patch/dspark_quarot_host_align_patch.py
find ucm -name '__pycache__' -prune -exec rm -rf {} +
```

## 8. 边界

- 纯运行时 monkeypatch，不改镜像文件；只对导入过 vllm 的进程生效。
- 依赖 vllm-ascend 0.32.0 的内部函数 `llama_eagle3._find_safetensors_weight`。将来升版若改名/改签名，
  补丁会打一条 `UCM QuaRot host alignment skipped: ...` 并退回原实现（也就是又会炸），升级时盯这条日志。
- 替换的是 `load_quarot_target_layer`，所有走 `align_draft_weights()` 的模型一并生效
  （eagle3、dflash2、kimi_k3 dspark、glm5 dspark、qwen3 dspark），语义等价。
- UCM 自带 `test/suites/Unit/test_apply_patch_version.py` 可能断言 patch 列表，若跑 UCM 单测需同步加一条。

## 9. 本目录内容

| 文件 | 说明 |
| --- | --- |
| `unified-cache-management-develop-6e3dbaa-dspark.tar.gz` | 打好补丁的完整 UCM develop 源码（git archive，不含 .git） |
| `dspark_quarot_host_align_patch.py` | 新增的 UCM patch 模块（可直接放到 `ucm/integration/vllm/patch/`） |
| `0001-add-dspark-quarot-host-align.patch` | 对 develop@6e3dbaa 的增量 patch（新文件 + 注册行），可 `git apply -p1` |
| `verify_ucm_patch.py` | 机器上的运行时校验脚本 |
| `SHA256SUMS.txt` | 校验和 |
