# Qwen3 Embedding 本地基线：运行协议与当前状态

**当前状态：模型与运行时资产已校验，14 项相关测试通过，CPU / Iris Xe Vulkan 实机性能预检及官方示例数值检查已完成。完整检索质量评测尚未完成。** 首轮预检曾因可用物理内存低于 256 MiB 退出；缩小符合输入上限的运行缓冲后，第二次顺序测试完成，失败记录保留。

2026-10-13，i5-1130G7 / Intel Iris Xe / 16GB 共享电脑上的一次测量如下。三个配置使用同一组按 ID 哈希选择的 16 篇摘要，全部达到 128-token 上限，关闭读取向量缓存；每次仅启动一个本项目模型。

| 后端 | CPU threads | 摘要/秒 | 5,183 篇文档时间投影 |
|---|---:|---:|---:|
| CPU | 1 | 0.3085 | 4.67 小时 |
| CPU | 2 | 0.2946 | 4.89 小时 |
| Iris Xe Vulkan | 2 | 1.0702 | 80.7 分钟 |

Vulkan 在本次小样本测试中约为 CPU 单线程的 3.47 倍；主机上其他任务及内存压力会影响速度，不能推广成硬件的普遍加速比。时间投影不是全量实测耗时，不包括查询编码、加载和后续排名。Vulkan 与 CPU 的样本向量最小余弦相似度为 0.99963；官方四文本示例相似度相对公开全精度参考的最大绝对误差分别为 CPU 0.00355、Vulkan 0.00266。Vulkan 进程峰值 working set 约 1.19GB，不等于包含所有共享 GPU 分配的系统总开销。

[完整性能记录](../results/modern/cpu_vulkan_benchmark.json) · [首次资源保护记录](../results/modern/benchmark_resource_wait.json)。这些数字只支持输入正确性和本机可行性判断，不能代替检索质量实验。

本模块接入 Qwen 在 2025 年发布的 Qwen3 Embedding 系列，用于补充 MiniLM 之外的预训练编码器对照。“现代基线”是项目中的比较角色，不表示当前全球最新模型、已达到某个榜单成绩或已经优于学生模型。[Qwen 2025 年发布说明](https://qwenlm.github.io/blog/qwen3-embedding/)。

## 1. 固定的模型和本地运行时

当前实现面向 **Windows x64**，默认 CPU；可选使用 llama.cpp 已枚举的 Vulkan 设备。没有云端推理、API key 或按调用付费服务。本项目未训练 Qwen 权重。

| 项目 | 固定资产 |
| --- | --- |
| 模型 | 官方 `Qwen/Qwen3-Embedding-0.6B-GGUF`，revision `370f27d7550e0def9b39c1f16d3fbaa13aa67728` |
| 量化文件 | `Qwen3-Embedding-0.6B-Q8_0.gguf`，639,150,592 bytes |
| CPU runtime | llama.cpp `b11580`，`llama-b11580-bin-win-cpu-x64.zip` |
| Vulkan runtime | llama.cpp `b11580`，`llama-b11580-bin-win-vulkan-x64.zip` |
| 输出 | 1,024 维，last-token pooling，L2 归一化 |
| 许可 | Qwen 权重 Apache-2.0；llama.cpp MIT |

来源：[固定模型版本](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/tree/370f27d7550e0def9b39c1f16d3fbaa13aa67728)、[固定运行时发布页](https://github.com/ggml-org/llama.cpp/releases/tag/b11580)。下载器校验固定大小与 SHA256，运行前验证已有资产，支持继续未完成的下载。

```text
Q8_0 model: 06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439
CPU runtime: 430efe5b4f10db617ae0f5732f7e9360fa7be4344d1d97688e09bbd3b5b084da
Vulkan runtime: 6e520f0b8bbc6608b53d70065ed80ee90d33acf1636f886d965e0c87f708dff2
```

## 2. 128-token 输入究竟包括什么

查询采用固定任务指令；文档直接输入 `title + newline + abstract`，不加文档指令。查询格式是：

```text
Instruct: Given a scientific claim, retrieve scientific abstracts that provide evidence about the claim.
Query:<claim>
```

先通过真实 GGUF runtime 的 tokenizer 分词，再显式右截断。默认 `--max-tokens 128` **包括最终特殊 token**：正文最多保留 127 tokens，末尾追加官方 EOS `151643`；不额外加 BOS。查询中的指令、格式前缀和 claim 共享这 127-token 预算，不能把它描述成“保留 128 个 claim tokens”。原始分词请求关闭自动 special-token 插入，避免重复追加；提交 embedding 请求时发送最终整数 token 列表。

这对应固定 GGUF/官方 tokenizer 的 `add_eos_token=true`、`add_bos_token=false` 约定。Python 内部接口的 `append_eos=False` 是非默认消融，不是正式基线配方。微型预检使用[官方模型示例](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage)检查输入、pooling 与归一化是否出现明显偏差；其容差检查不能证明与全精度实现逐值一致。

MiniLM 学生也使用 128-token 上限，但 tokenizer 不同、Qwen 额外消耗查询指令 tokens，**相同 token 数不等于相同文本覆盖率**。未来得出的分数只属于本地 `Q8_0 + 固定指令 + 128 tokens` 配置，不能冒充官方全精度分数或默认长上下文能力。

## 3. 下载与速度预检

在已安装项目的虚拟环境中，从仓库根目录运行。以下是实际模块入口，不是 `evidenceforge adapt` 的子阶段：

```bash
python -m evidenceforge.modern fetch --cache-dir cache --backend cpu
python -m evidenceforge.modern fetch --cache-dir cache --backend vulkan
```

第二条只用于需要 Vulkan runtime 的场景。两者使用同一个已校验的 Q8 模型文件；`fetch` 不启动推理服务。准备 SciFact 数据后，CPU 预检命令为：

```bash
python -m evidenceforge download
python -m evidenceforge.modern benchmark --data-dir data/scifact --cache-dir cache --output artifacts/modern/cpu_benchmark.json --max-tokens 128
```

该命令依次比较 CPU 1 thread、CPU 2 threads。它先执行官方示例语义检查和一次预热，再编码按文档 ID 哈希确定的 16 篇摘要；选择不依赖相关性标签。**16 篇摘要的吞吐预检不是 300-query 检索质量评测。** 模型加载与完整性检查单独计时，吞吐样本显式设置 `use_cache=False`。

若本机 runtime 确实枚举出 `Vulkan0`，可在两组 CPU 之后增加该设备：

```bash
python -m evidenceforge.modern benchmark --data-dir data/scifact --cache-dir cache --output artifacts/modern/cpu_vulkan_benchmark.json --max-tokens 128 --include-vulkan --vulkan-device Vulkan0
```

Vulkan 是可选硬件后端，不是 CPU 成绩；需要记录设备、runtime、内存和与 CPU 输出的一致性。当前尚无数据可以断言它更快、占用更少或在这台机器上能完成完整评测。不要与正在运行的学生训练并发执行新模型预检。

llama.cpp 的 context/batch 缓冲是运行时容量设置，不会扩大 `--max-tokens` 指定的实际文本预算；输入仍逐条编码，`--parallel 1`，没有并发提交一批文档。缓冲和日志参数以运行产物保存的 metadata/server command 为准。

## 4. 完整 SciFact 质量评测

本仓库当前的运行协议已固定为 Vulkan、2 threads、128 tokens，附加编码独立的 809 条训练查询文本，保存于 [protocol.json](../results/modern/scifact-full/protocol.json)。可先只登记/验证协议，再用完全相同参数启动模型：

```bash
python -m evidenceforge.modern_experiment --output-dir results/modern/scifact-full --backend vulkan --device Vulkan0 --threads 2 --max-tokens 128 --encode-extra-train --prepare-only
python -m evidenceforge.modern_experiment --output-dir results/modern/scifact-full --backend vulkan --device Vulkan0 --threads 2 --max-tokens 128 --encode-extra-train
```

`--prepare-only` 不加载模型，不生成质量指标。下方 CPU/其他目录示例是独立配置，不代表本仓库已完成这些额外质量对照。

满足内存条件并通过预检后，运行完整 CPU 入口：

```bash
python -m evidenceforge.modern_experiment --data-dir data/scifact --cache-dir cache --output-dir artifacts/modern/scifact-full --threads 2 --max-tokens 128 --backend cpu
```

可选 Vulkan 入口使用独立输出目录：

```bash
python -m evidenceforge.modern_experiment --data-dir data/scifact --cache-dir cache --output-dir artifacts/modern/scifact-full-vulkan --threads 2 --max-tokens 128 --backend vulkan --device Vulkan0
```

入口强制要求原始完整 **5,183 篇文档、300 条官方 test 查询**；每个 test query 与全部文档做归一化向量点积，保存 top-50 TREC 排名并计算 nDCG@10、MRR@10、Recall@10/50。它不缩小为 MiniLM 候选集，不向测试候选补入标注正例，不按测试结果调指令或选择模型。当前入口只覆盖 SciFact，不能写成已经完成 NFCorpus 上的 Qwen 对照。

第一次启动前先登记 `protocol.json`，固定模型、后端、输入身份、token 配置和查询集合。改变 threads、backend、token 上限或其他协议参数时使用新输出目录，避免混合两次实验。查看以下文件区分“已开始”与“已完成”：

```text
artifacts/modern/scifact-full/
  protocol.json
  documents_progress.json
  test_queries_progress.json
  report.json
  runs/
    qwen3_06b_q8.trec
    qwen3_06b_q8.per_query.json
```

只有在全部文档和 test queries 编码、排序与指标计算成功之后才写入本轮最终 `report.json`。进度文件、协议文件或已下载模型都不是质量结果。

## 5. 中断恢复与可选训练查询向量

每条文本完成后独立、原子保存 embedding checkpoint；每 16 条汇总一次阶段进度。内存保护、中断或下一次调用后，使用相同参数重跑即可复用兼容缓存，不必从第 1 篇文档重新推理。

缓存身份包含模型/量化、runtime、后端、输入配方与 token 配置；查询和文档分别标记。线程数有意不作为向量缓存身份的一部分，但每条记录保留原始推理线程数与耗时。改变线程数需新实验输出目录，若复用向量，报告会暴露历史线程构成。CPU 与 Vulkan 后端的缓存不视为同一配置。缓存命中时仍验证向量维度、有限值和归一化。

如需为**另一项预先登记的、仅使用训练查询的教师实验**准备额外向量，可选：

```bash
python -m evidenceforge.modern_experiment --data-dir data/scifact --cache-dir cache --output-dir artifacts/modern/scifact-with-train --threads 2 --max-tokens 128 --backend cpu --encode-extra-train
```

额外编码全部 809 条官方 train 查询文本。它们使用 `train_queries` 命名空间，与 `test_queries` 分开；导出的 `cache/modern/teacher_vectors/<spec_sha256>/vectors.npz` 包含文档向量和 train-query 向量，**不包含 test-query 向量**。这项导出不会改变 300 条测试查询的质量统计，也不会自动替换当前 LoRA 三组实验中已固定的 MiniLM 教师。

809 条官方 train 包含内部训练和内部验证查询。后续若沿用 647/162 划分，教师过滤仍只能使用允许参与训练的那 647 条；“导出了训练向量”不代表已经训练了教师、更不代表完成了知识蒸馏。

## 6. 如何读耗时，如何处理内存不足

| 字段 | 含义 |
| --- | --- |
| 预检 `documents_per_second` | 本次无缓存的 16 篇样本文档编码吞吐；不含模型启动，包含分词、本地请求与 checkpoint 写入等编码路径开销 |
| `estimated_5183_document_seconds` | 从上述样本外推的估计，不能当作已测全量时间 |
| `cache_hits` | 本次复用历史向量的条数 |
| `uncached_inference_seconds` / `uncached_item_latency_seconds` | 本次新编码项目的累计耗时及分位数；全缓存时无新编码延迟样本 |
| `recorded_original_inference_seconds` | 各向量保存时的原始推理耗时之和，可能跨多次运行 |
| `recorded_threads_item_counts` | 历史缓存中不同线程配置对应的条数 |
| `wall_seconds_this_invocation` | 本次调用的墙钟时间，不是整个跨断点实验的累计耗时 |
| `dot_product_only_seconds` | 已编码向量的排序耗时，排除加载、文档编码和查询编码 |

不要把全缓存重跑的速度写成模型推理吞吐，也不要把点积排序时间写成端到端查询时延。进度中的截断率和 token 数需要与最终质量指标一起查看。

Windows 内存保护分两层：启动前要求至少 1.5 GiB 可用物理内存；模型运行后，在新的未缓存请求前检查，低于 256 MiB 时退出。后者正是首轮预检退出的原因。它是主动停止条件，不保证模型在任何内存压力下都不会 OOM，也不表示固定的峰值占用。

编码器只终止自己通过 `subprocess.Popen` 创建并持有句柄的 llama.cpp 进程；不会按进程名批量结束程序，不会关闭用户其他应用或学生训练。已完成 checkpoint 保留。等待已有任务结束、内存恢复后再重跑，不降低保护阈值来强行并发。

目前能陈述的是：固定资产和可恢复完整评测入口已就绪，首轮预检被内存保护中止。吞吐、CPU/Vulkan 差异、完整检索质量及其相对学生模型的优劣，都必须等待实际运行产物。
