# 双 H100：vLLM 0.27.1 / Qwen3-32B / UcmTransferGraph

## v2 修正（2026-10-23）

修复 `check_metrics.py` 在 graph on 时误检查 legacy 后端计数的问题。
on 检查 `ucm:cache_transfer_graph_backend_shards_total`，off 检查
`ucm:cache_load_backend_shards_total`；on 仍要求 graph 批次增长。
报错区分指标未导出和指标无增量。无需修改 YAML、重启服务或重新编译 UCM。
可只替换 `check_metrics.py`，用旧快照重新校验：

```bash
python transfer_graph_h100/check_metrics.py on /path/to/1.before.prom /path/to/1.after.prom /path/to/1.json
```

如果测的是 off，将上述 on 改为 off。off 无增量仍需排查真实命中或指标导出问题。
本修正通过 10 项无 GPU 回归检查；运行 `python transfer_graph_h100/test_check_metrics.py` 可复验。

使用用户仓库 `https://github.com/luomoduchu/unified-cache-management.git` 的 graph 分支，默认固定到 `89909fd0c5c6fb140a5dd9e0ec0ded213e4fa6e5`。脚本与被测源码分开放置，安装脚本不会覆盖已有 checkout。

默认按 **2 × H100 80GB、BF16、TP=2** 设计。主机需要 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；模型已下载到本地。主存至少要为两个 rank 的 UCM buffer 留出 16 GiB，另留模型加载及系统开销。A800 以 Linux 本地挂载/NFS 路径提供。建议缓存目录至少空闲 128 GiB；测试目录不要与生产缓存混用。

## 1. 准备容器与安装

把本目录复制到 Linux 主机 `/data/ucm-test/transfer_graph_h100`，或把提供的 tar.gz 解压到 `/data/ucm-test`。修改以下三个主机路径：

```bash
mkdir -p /data/ucm-test
docker run --name ucmgraph-h100 -it \
  --gpus '"device=0,1"' --ipc=host --network=host \
  --ulimit memlock=-1:-1 \
  -v /data/models:/models:ro \
  -v /mnt/a800:/cache \
  -v /data/ucm-test:/workspace \
  --entrypoint /bin/bash vllm/vllm-openai:v0.27.1

# 以下在容器内执行；会安装构建依赖、clone、固定源码并编译 CUDA 后端。
bash /workspace/transfer_graph_h100/setup.sh
```

`setup.sh` 中实际 clone/checkout 命令为：

```bash
git clone --branch graph https://github.com/luomoduchu/unified-cache-management.git /workspace/ucm-source
git -C /workspace/ucm-source checkout --detach 89909fd0c5c6fb140a5dd9e0ec0ded213e4fa6e5
```

已有 vLLM 0.27.1 环境也可运行安装脚本（脚本的 apt 安装部分要求 root）。CUDA 构建需要 `/usr/local/cuda/bin/nvcc` 及对应开发头文件；缺少时脚本会停止，需补齐与环境匹配的 CUDA toolkit。不要另行升级 torch/vLLM。安装完成保存 `/workspace/ucm-pip-freeze.txt`；记录容器镜像 digest、驱动和模型 revision，以便复现。

## 2. 关闭 graph 的基线

容器内终端 A：

```bash
export MODEL=/models/Qwen3-32B
export STORAGE_ROOT=/cache/ucm-graph-bench
bash /workspace/transfer_graph_h100/serve.sh off
```

等待服务器 ready。在主机另开终端 B：

```bash
docker exec -it ucmgraph-h100 bash
export MODEL=/models/Qwen3-32B
bash /workspace/transfer_graph_h100/bench.sh off
```

默认 16 条提示词，输入 8192 tokens、输出 128 tokens、并发 4、seed 42。先跑 seed 和 warmup 两轮，再记录三轮。使用 `/v1/completions` 的随机 token 负载，适合传输路径对比，不用于评估 Qwen 的聊天/思考质量。输出通过 ignore_eos 固定长度；每轮使用同一 tokenizer 和随机种子。

## 3. 开启 graph

在终端 A 用 Ctrl+C 正常停止基线，确认 GPU worker 退出、显存释放，再启动：

```bash
bash /workspace/transfer_graph_h100/serve.sh on
```

服务 ready 后在终端 B：

```bash
bash /workspace/transfer_graph_h100/bench.sh on
```

两组使用相同存储目录。脚本不会删除缓存。每个模式的有效 YAML、完整启动命令、服务日志和 GPU 信息存入 `/workspace/results/off` 或 `on`；每次压测创建独立 `bench-*` 子目录，保存 JSON 和前后 `/metrics` 快照。若要保留多次重启的服务日志，给两个终端设置相同的独立 `RUN_DIR`，例如 `/workspace/results/on-repeat2`。

建议再按 on → off 顺序复测，以减小运行顺序偏差。先用默认 16 请求验证接入，正式测试可两组统一设置 `NUM_PROMPTS=64`（更大数据集需要更多磁盘容量），或逐组固定测试 `CONCURRENCY=1/4/8`、`INPUT_LEN=4096/8192/12288`。修改负载后重新进行 seed/warmup，勿混用不同参数的结果。

## 实际启动命令和配置

`serve.sh` 会从同一份 `ucm.yaml` 生成 off/on 配置，仅修改 graph 开关和指定的存储路径。核心启动命令等价于：

```bash
export ENABLE_UCM_PATCH=1
export CUDA_VISIBLE_DEVICES=0,1
vllm serve /models/Qwen3-32B --served-model-name qwen3-32b \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --block-size 128 \
  --no-enable-prefix-caching --seed 42 \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/results/on/ucm.yaml"}}'
```

没有设置 `--enforce-eager`，保留 vLLM 默认 CUDA Graph 行为，且两组一致。UcmTransferGraph 与 CUDA Graph 是不同层次的机制。`ENABLE_UCM_PATCH=1` 在启动 Python 之前设置；本仓库提供 0.27.1 patch，无需手工应用 diff。

YAML 使用 `Cache|Posix`、`psync`、`io_direct: true`、32 个 POSIX I/O worker、每 rank 8 GiB buffer。graph 上限为 32 shards / 8 MiB，实际批次还受当前请求 shard 数约束。tensor/shard 大小由 connector 根据模型布局计算，不在 YAML 中猜测填写。如果挂载文件系统不支持 O_DIRECT，应两组一起改为 `io_direct: false`，并记录此变更。

`cache_load_backend_only: true` 绕过 UCM DRAM 读命中；`--no-enable-prefix-caching` 避免 GPU 前缀命中遮蔽传输。这里衡量的是后端读路径。在评估完整生产多级缓存时，两组一起把 backend-only 设为 false，并按实际策略启用 GPU 前缀缓存，单独记录结果。

## 如何判定结果

先看每轮 `*.check.txt`：请求必须全部成功，后端读取计数必须增长；on 组还必须出现 graph 批次与 shard 计数。脚本检测到相关 UCM 错误计数增加会停止。检查通过仅证明路径被执行，不代表所有请求均命中或性能已改善。

对比三轮 JSON 的 TTFT 中位数和 P95、请求吞吐、输出 token/s、TPOT。TTFT 改善率为 `(off - on) / off × 100%`；吞吐提升率为 `(on / off - 1) × 100%`。16 请求的 P99 不适合统计结论，正式评估应增加样本并重复。还要核对两组的缓存命中覆盖、后端读取 shard 数、请求输入/输出长度一致；若只加载少量缓存，增加预热并检查写入/lookup 日志，不能直接宣称加速。

graph 的 `graph_shards / graph_batches` 表示实际每批 shard 数；接近 1 时，批量优化空间很小。默认等待 10 秒仅为异步写入/指标导出留时间，不是持久化完成保证。warmup 有失败或命中不稳定时，先排查，不将该轮用于结论。

本测试不执行全机 drop_caches。默认 O_DIRECT 旨在绕过客户端文件数据页缓存，但不能消除存储控制器/NFS 服务端缓存；若改为 buffered I/O，还会受到 Linux 页缓存影响。POSIX 调用不等于每次实际访问 SSD。报告中注明本地/NFS、挂载参数、A800 型号、冷热状态；需要实盘吞吐时结合客户端/服务端 I/O 监控单独实验。psync 严格完成语义可能等待挂起的 NFS I/O，不能用超时数值推断服务必然及时退出。

脚本已做本地语法与配置检查；尚未在 H100 上完成端到端运行，也没有预设性能提升比例。

参考：[vLLM 0.27.1 bench serve 参数](https://docs.vllm.ai/en/v0.27.1/cli/bench/serve/)、[Qwen3-32B 官方 recipe](https://recipes.vllm.ai/Qwen/Qwen3-32B)、[vLLM 0.27.1 benchmark 实现](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/benchmarks/serve.py)。
