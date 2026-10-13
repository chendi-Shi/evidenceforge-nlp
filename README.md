# EvidenceForge · 科学文献检索与学习排序

[![Offline tests](https://github.com/chendi-Shi/evidenceforge-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/chendi-Shi/evidenceforge-nlp/actions/workflows/ci.yml)

面向 NLP 算法实践的可复现实验项目：给定英文研究论断，在完整 SciFact 论文摘要库中检索相关文献，比较稀疏检索、神经语义检索、融合召回与监督学习排序。

项目包含真实公开数据、难负样本挖掘、训练内分组验证、排序消融、逐查询评测、误差分析、命令行与 FastAPI 服务。神经编码器保持冻结；训练的是 LambdaMART 排序模型。检索相关性不能判断论断真假，也不代表文献支持该论断。

```mermaid
flowchart LR
    Q[英文论断] --> B[BM25]
    Q --> D[MiniLM ONNX INT8]
    C[5,183 篇论文摘要] --> B
    C --> D
    B --> R[RRF 候选融合]
    D --> R
    R --> L[LambdaMART 重排]
    L --> E[文献 ID / 原文 / 排序分数]
    T[训练查询与相关性标注] --> H[难负样本 + 分组验证]
    H --> L
```

## 运行

Python 3.11–3.13，建议 3.12。支持 CPU，不需要 API Key 或付费服务。首次运行需要联网下载公开语料与固定版本的模型；后续复用本地缓存。

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 使用 .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
evidenceforge download
evidenceforge index
evidenceforge experiment --output-dir results/scifact
evidenceforge search "Vitamin D deficiency is associated with cardiovascular disease." --method ltr
evidenceforge serve
```

在浏览器访问 `http://127.0.0.1:8000/docs` 可交互调用 API。首次模型下载/索引需要时间；索引完成后再启动服务。

Windows 上也可以不激活环境，直接运行 `.\.venv\Scripts\python.exe -m evidenceforge serve`。

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Vitamin D deficiency is associated with cardiovascular disease.","method":"rrf","top_k":5}'
```

只体验词法检索时，运行 `evidenceforge index --lexical-only` 与 `evidenceforge search "your query" --method bm25`。此模式仍需要数据，但不下载神经模型。

全局路径参数放在子命令之前，例如：

```bash
evidenceforge --data-dir data/scifact --cache-dir cache --threads 2 experiment --output-dir results/scifact
```

复现本仓库记录的版本时，用 Python 3.12 执行 `pip install -c requirements.lock -e ".[dev]"`。锁文件不包含本机路径或模型权重。

## 实验协议

| 项目 | 设计 |
|---|---|
| 数据 | BEIR SciFact，完整 5,183 篇文献；809 个训练查询，300 个测试查询 |
| 稀疏基线 | BM25，确定性分词和分数并列排序 |
| 神经基线 | 固定版本 all-MiniLM-L6-v2；ONNX INT8；均值池化、L2 归一化、余弦检索 |
| 融合 | Reciprocal Rank Fusion |
| 可训练模型 | LightGBM LambdaMART，检索分数、排名和文本匹配特征 |
| 负样本 | RRF 前 50 候选中的非标注相关文献；仅训练时允许补入缺失的正样本 |
| 验证 | 仅在训练查询内部划分，按共享相关文献/规范化论断构建连通组，减少变体泄漏 |
| 测试 | 固定测试集；不注入正确文献、不用测试标签选参 |
| 消融 | 相同 RRF 候选池，仅移除重排器中的 dense 与 RRF 特征；不是端到端移除 dense |
| 指标 | nDCG@10、MRR@10、Recall@10/50、候选召回上限、配对 bootstrap 置信区间 |
| 工程 | 固定模型版本、下载校验、缓存身份检查、模型文本序列化、离线测试、请求边界限制 |

BEIR 的 `test` 对应原始 SciFact 的公开 `dev`，并非原始隐藏测试集。其相关性标注源于 cited document IDs；“相关文献”与“有标注支持/反驳证据的文献”并不等价。详见 [数据说明](docs/DATA_CARD.md)。

## 实测结果

2026-10-13，在完整 5,183 篇文献、300 条官方 BEIR test 查询上实测。超参数仅由训练集内的分组验证选择，最终用全部 809 条训练查询、40,512 个候选文献对重训。

| 方法 | nDCG@10 | MRR@10 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|
| BM25 | 0.6634 | 0.6309 | 0.7876 | 0.8704 |
| MiniLM INT8 dense | 0.6426 | 0.6009 | 0.7806 | 0.8887 |
| RRF | 0.6870 | 0.6504 | 0.8182 | 0.9447 |
| **RRF + LambdaMART** | **0.7149** | **0.6741** | **0.8549** | 0.9447 |
| 重排特征消融：移除 dense/RRF 特征 | 0.6728 | 0.6378 | 0.8074 | 0.9447 |

学习排序相对 RRF 的 nDCG@10 **绝对增加 0.0279，相对增加 4.06%**；2,000 次查询级配对 bootstrap 的差值 95% 区间为 `[0.0117, 0.0445]`。这是单一公开基准的实测差异；区间未按论断家族聚类或进行多重比较校正。官方 train/test 有 6 个规范化论断签名重叠，已在报告中披露。

RRF 候选集的 oracle nDCG@10 为 0.9461，说明仍存在排序改进空间。重排保留相同 50 个候选，所以三个使用相同候选池的方法 Recall@50 一致。

- [完整机器可读报告](results/scifact/report.json)、[逐查询结果](results/scifact/per_query.jsonl)、[错误案例](results/scifact/errors.json)、[TREC 排名](results/scifact/runs)
- [结果解读与具体案例](docs/RESULTS.zh-CN.md)、[可用于简历的表述](docs/RESUME.zh-CN.md)
- [独立指标复算记录](results/scifact/independent_audit.json)：20 个汇总指标与候选上限均核验一致

本地 Windows Python 3.12 的 **37 项测试通过**，GitHub Actions 的 Linux Python 3.11/3.12 检查通过。训练好的两个 LightGBM 文本模型已随结果保存；原始数据、预训练神经权重与嵌入缓存仍按首次运行命令下载/构建。

## 代码入口

```text
src/evidenceforge/
  data.py          # 校验与加载 BEIR 数据
  neural.py        # 固定版本 ONNX 编码器
  retrieval.py     # BM25 / dense / RRF
  ranker.py        # 特征、难负样本、分组验证、LambdaMART
  evaluation.py    # 指标、bootstrap
  experiment.py    # 完整实验与报告
  cli.py           # 可复现命令行
  api.py           # FastAPI，使用相同检索/排序代码
tests/             # 小型合成数据仅用于离线正确性测试
docs/              # 数据、实验方法与面试材料
```

```bash
pytest -q
ruff check src tests
python scripts/verify_results.py --results-dir results/scifact
```

CI 在 Linux Python 3.11/3.12 上运行离线测试；完整基准需要单独运行。Dockerfile 提供服务打包入口，运行时将本地 `data/`、`cache/`、`results/` 所在目录挂载到 `/artifacts`。本地 API 默认只绑定回环地址；开放公网需要另行配置认证和部署设施。

`scripts/verify_results.py` 使用独立的标准库实现，从保存的 TREC 排名与相关性标注重新计算指标，检查报告是否一致，不需要重新训练模型。

## 简历与面试

- [实验方法与边界](docs/METHODOLOGY.zh-CN.md)
- [技术取舍、面试问题与简历模板](docs/INTERVIEW.zh-CN.md)
- [含实测数字的简历表述](docs/RESUME.zh-CN.md)
- [数据来源、标注语义与许可证](docs/DATA_CARD.md)

这是有公开数据实验依据的算法工程项目；不要表述为已经上线服务真实用户、训练大模型，或取得未经测量的线上收益。理解并能复现代码、实验和失败原因后，再在简历中描述自己的实际贡献。

## 来源与许可证

项目原创代码采用 MIT。数据和模型保持各自许可证，不随本仓库重新授权；原始语料、模型权重与缓存不提交 Git。

- [BEIR](https://github.com/beir-cellar/beir) / [BEIR paper](https://arxiv.org/abs/2104.08663)
- [SciFact 原始项目](https://github.com/allenai/scifact) / [数据许可证](https://github.com/allenai/scifact/blob/master/LICENSE.md)
- [MiniLM 模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [LightGBM LambdaRank 参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
