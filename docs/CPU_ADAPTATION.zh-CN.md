# 在 CPU 上训练、评测和使用 LoRA 编码器

本指南对应 EvidenceForge v2 的 CPU 编码器适配流程：在 SciFact 上训练三种负例采样方案，在完整 SciFact 和 NFCorpus 测试集上比较，然后通过 CLI 或本地 API 加载适配器。训练和推理都在本机运行，不调用付费模型 API。

本文是使用说明，**不预设 `filtered` 优于 `retrieved`**。下文选择 `filtered` 只是路径示例，训练完成后也可加载 `random` 或 `retrieved`。质量指标将在实际评测完成后写入独立结果说明；Qwen 基线尚未完成吞吐与质量评测。

## 1. 安装 CPU 环境

推荐 Python 3.12 虚拟环境。在仓库根目录、已启用的虚拟环境中执行：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-adaptation.lock -e ".[dev,adaptation]"
```

先安装官方 CPU PyTorch wheel，再按 v2 约束文件安装项目。`requirements.lock` 是 v1 环境，**不要将它与 `requirements-adaptation.lock` 混用**。现有环境不必删除；需要隔离时可新建虚拟环境。

Windows PowerShell 也可直接使用虚拟环境的解释器，无需先激活环境：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -c requirements-adaptation.lock -e ".[dev,adaptation]"
```

首次运行会下载公开模型权重和数据，之后使用校验通过的本地文件与缓存。仅安装基础依赖时，v1 检索仍可使用；v2 神经训练和适配器推理需要 `adaptation` extra。

## 2. 从数据准备开始

先准备源数据 SciFact，再固定本轮适配协议：

```bash
python -m evidenceforge download
python -m evidenceforge --threads 2 adapt --stage prepare --output-dir results/adaptation --transfer-dir data/nfcorpus
```

`prepare` 下载并校验浮点 MiniLM 基础权重、完整 NFCorpus 数据，读取本地 SciFact，并写入 `results/adaptation/protocol.json`；这个阶段不进行梯度更新。SciFact 应先通过 `download` 命令准备好。

所有全局参数都放在子命令前，例如 `--data-dir`、`--cache-dir`、`--threads`：

```bash
python -m evidenceforge --data-dir data/scifact --cache-dir cache --threads 2 adapt --stage prepare --output-dir results/adaptation --transfer-dir data/nfcorpus
```

默认实验使用全部 **647 条内部训练查询、162 条内部验证查询、2 epochs、seed 42、3 个采样组**。CLI 的 `--max-train-queries` 默认值是 10000，表示上限；实际内部训练分区只有 647 条，不会产生 10000 条训练样本。正式比较不要临时改变这个上限。

模型只训练最后一个 Transformer block 的 Q/V LoRA，rank 8，共 12,288 个可训练参数。三组是随机负例 `random`、检索负例 `retrieved`、联合过滤负例 `filtered`。采样规则、分组依据和归因限制详见[实验协议](ADAPTATION_PROTOCOL.zh-CN.md)。

## 3. 分阶段训练与评测

单独运行训练：

```bash
python -m evidenceforge --threads 2 adapt --stage train --output-dir results/adaptation --transfer-dir data/nfcorpus
```

三个组依次训练，默认每组计划 648 次更新。日志显示当前组、已完成步数、loss 和梯度范数；loss 不同可能来自负例难度不同，不能当作检索质量排名。验证集只监测固定负例上的 InfoNCE，最终 epoch 在训练前固定，不按测试分数挑 checkpoint。

三个适配器都完成后运行：

```bash
python -m evidenceforge --threads 2 adapt --stage evaluate --output-dir results/adaptation --transfer-dir data/nfcorpus
```

评测包含浮点未适配 `base` 与三个适配器，只比较四种密集检索方法，使用相同的 128-token 截断和完整语料的精确余弦排序：

| 数据集 | 语料文档数 | 官方测试查询数 | 用途 |
| --- | ---: | ---: | --- |
| BEIR SciFact | 5,183 | 300 | 源任务测试 |
| BEIR NFCorpus | 3,633 | 323 | 查询/任务分布迁移测试 |

NFCorpus 的 1/2 两级相关度保留到 nDCG 计算；本轮采用线性 graded gain。两数据集仍有生物医学主题重叠，不能将其描述为完全无关领域。评测不会用 NFCorpus 标签更新模型或选择采样规则。

也可一次完成全部阶段；省略 `--stage` 等同于 `--stage all`：

```bash
python -m evidenceforge --threads 2 adapt --output-dir results/adaptation --transfer-dir data/nfcorpus
```

CPU 运行时间取决于机器、文档长度、首次缓存与其他进程负载。默认 2 threads 用于控制本机资源占用，不保证实时训练或特定吞吐量。

## 4. 中断后继续

再次执行**相同参数、相同输出目录**的 `train` 或 `all` 命令，会校验计划、配置、数据、基础权重和已有适配器：

- 已完成的组通过身份与校验和检查后复用。
- 未完成的组从最近保存的 checkpoint 恢复适配器和 Adam 状态；最后一个 checkpoint 之后的少量步骤可能重算。
- 评测嵌入按块缓存，重跑可复用校验通过的块。

默认每 50 步及组结束时保存 checkpoint。不要手动混搭不同运行的 `metadata.json`、adapter 或 checkpoint 文件。若改变 epochs、训练查询上限等已登记配置，使用新的 `--output-dir`；身份检查会拒绝覆盖不同协议的结果。重新运行评测时也要沿用该实验的训练参数。

训练 metadata/report 中的 `elapsed_seconds` 计量当前调用，不会把已有 checkpoint 所耗的历史时间补加回来；中断恢复后不能将它当作从零训练的总耗时。比较总成本时应另行累计各次运行记录。

## 5. 检查产物

完成后主要文件如下：

```text
results/adaptation/
  protocol.json
  frozen_adapters.json
  report.json
  training/
    training_plan.json
    training_report.json
    random/
      adapter.safetensors
      metadata.json
      checkpoints/
    retrieved/
      adapter.safetensors
      metadata.json
      checkpoints/
    filtered/
      adapter.safetensors
      metadata.json
      checkpoints/
  scifact/runs/
    base.trec
    base.per_query.json
    ...
  nfcorpus/runs/
    base.trec
    base.per_query.json
    ...
```

`training_plan.json` 保存每个组的查询、正负例和过滤决策；各组 `metadata.json` 记录真实参数变化、梯度、完成步数与训练曲线。`report.json` 包含完整测试指标及配对比较，预先指定的主比较是 `filtered − retrieved` 的 nDCG@10，无论正负都应报告。

评测在每个数据集完成后更新 `report.json`，因此**文件存在不代表两个数据集都完成**；确认 `datasets` 与 `comparisons` 中均有 `scifact`、`nfcorpus`，且每个数据集都有 `base/random/retrieved/filtered` 四种方法。报告中的 `dot_product_only_seconds` 只测已编码向量的点积排序，不是完整在线请求耗时。

无需再次训练即可审计负例构造：

```bash
python scripts/audit_negative_mining.py --plan results/adaptation/training/training_plan.json --data-dir data/scifact --output results/adaptation/negative_mining_audit.json
```

脚本分别统计家族保护、教师相似度过滤及二者交集，记录实际选中负例的分数/排名、替换率和随机补足。缺少独立标签的“假负例准确率”不会被编造成结果。

## 6. 用训练好的适配器搜索

对应组的 `adapter.safetensors` 与 `metadata.json` 都完成后，就可以使用该组。示例路径可以替换为 `training/random` 或 `training/retrieved`：

```bash
python -m evidenceforge --threads 2 search "Vitamin D supplementation and fracture risk" --method dense --adapter-dir results/adaptation/training/filtered
python -m evidenceforge --threads 2 search "Vitamin D supplementation and fracture risk" --method rrf --adapter-dir results/adaptation/training/filtered --top-k 5
```

`dense` 使用该适配器的向量检索，`rrf` 将它与 BM25 融合。RRF 是可用的推理方式，本轮四编码器实验没有评测其融合质量，不能将 `report.json` 中的密集检索指标视为适配器 RRF 的指标。`--adapter-dir` 在 CLI 中只与 `dense` 或 `rrf` 搭配，不能与 v1 的 `ltr` 模型混用。首次使用若没有相应语料缓存，会先编码文档；已完成本轮评测时可复用导出的兼容向量缓存。

## 7. 启动本地 API

```bash
python -m evidenceforge --threads 2 serve --adapter-dir results/adaptation/training/filtered --host 127.0.0.1 --port 8000
```

打开 [API 文档](http://127.0.0.1:8000/docs)，或在 PowerShell 发起请求：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/search -Method Post -ContentType 'application/json' -Body '{"query":"Vitamin D supplementation and fracture risk","method":"dense","top_k":5}'
```

`/health` 显示就绪状态，`/metadata` 提供当前索引和编码器信息。适配器模式不会自动加载不兼容的 v1 LambdaMART；在该实例请求 `ltr` 会返回 409，应使用 `dense`、`rrf` 或 BM25。服务限制同时活动的检索请求，繁忙时返回 429；这是本地演示接口，不是已经验证吞吐或 SLA 的生产部署。

搜索结果只表示文献检索相关性，不自动判定支持、反驳或医疗有效性。跨域质量、过滤是否带来改善以及它为何改善，应依据完整实验与审计结果解释。
