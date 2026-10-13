# EvidenceForge 数据卡

本项目使用 **BEIR 格式的 SciFact 科学文献检索数据**。输入是英文科学论断，输出是相关论文摘要的排序。相关性标注不能直接用作论断真伪标签，检索命中也不代表文献支持该论断。

## 来源与版本

| 项目 | 内容 |
| --- | --- |
| 原始数据 | [AllenAI SciFact](https://github.com/allenai/scifact) |
| 原始论文 | [Wadden et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.609/) |
| 检索基准 | [BEIR](https://github.com/beir-cellar/beir) |
| 下载包 | [BEIR scifact.zip](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip) |
| 上游公布的包 MD5 | `5f7d1de60b170fc8027bb7898e2efca1` |
| 本项目核验日期 | 2026-10-13 |
| 语言与文档单位 | 英文；每篇论文的标题与完整摘要为一个检索文档 |

下载源、发布包 MD5 与公开测试集规模见 [BEIR 数据集表](https://github.com/beir-cellar/beir#available-datasets)。实际复现同时保留下载包与文件的哈希；MD5 用于与上游发布物核对，不作为安全保证。

## 实际规模与划分

以下计数来自对上述 BEIR 发布包的实际解析，不是根据网页的“5K”近似值填写。两个上游包的哈希、查询映射和标签集合核验记录见 [data_audit.json](data_audit.json)。

| 文件/划分 | 数量 | 含义 |
| --- | ---: | --- |
| `corpus.jsonl` | 5,183 | 全部候选论文摘要 |
| `queries.jsonl` | 1,109 | train 与 test 查询的并集 |
| `qrels/train.tsv` | 809 个查询、919 个相关查询文档对 | 可用于监督训练；项目从这里划内部验证集 |
| `qrels/test.tsv` | 300 个查询、339 个相关查询文档对 | BEIR 公开检索测试集 |

训练和测试使用同一篇数为 5,183 的固定语料库；划分对象是查询。对全量候选语料建 BM25 索引、计算冻结模型的文档向量符合这一固定语料检索设置，不等于使用测试相关性标签训练。

**BEIR 的 `test` 不是原始 SciFact 的隐藏测试集。** 本项目把 BEIR 查询 ID 与文本同[原始发布包](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz)逐条核对，得到：

| BEIR 划分 | 原始 SciFact 文件 | ID 与文本一致的查询数 |
| --- | --- | ---: |
| train | `claims_train.jsonl` | 809/809 |
| test | `claims_dev.jsonl` | 300/300 |
| 不参与本项目 | `claims_test.jsonl` | 原始隐藏测试 300 条 |

原始 SciFact 的测试标签不公开，其评估包含论断立场与证据句任务。本项目的检索指标不能与原论文完整事实核验系统的指标直接比较。[原始数据说明](https://github.com/allenai/scifact#dataset)

## 标签到底表示什么

原始 SciFact 区分 `cited_doc_ids` 和 `evidence`：前者是产生该论断的引用句所引用的文献；后者标出真正提供支持或反驳证据的摘要、立场与句子位置。被引用的文献不一定包含足够证据。[原始 schema](https://github.com/allenai/scifact/blob/master/doc/data.md)

2026-10-13 对两个发布包逐对核验：**BEIR qrels 与原始 `cited_doc_ids` 完全相等，并非只取 `evidence` 的文档。** 这是本项目通过数据比对得出的结果。

| BEIR 划分 | qrels 文档对 | 同时在原始 evidence 的文档对 | 未在原始 evidence 的文档对 | 原始 evidence 为空的查询 |
| --- | ---: | ---: | ---: | ---: |
| train | 919 | 564 | 355 | 304 |
| test | 339 | 209 | 130 | 112 |

核验方法：按 `(str(claim.id), str(doc_id))` 构建三组集合：BEIR qrels、原始 cited 文档对、原始 evidence 文档对；比较集合相等性与差集，同时逐条核对 query 文本。不同文档对与查询数不可混用。

因此，项目训练的是**论文相关性排序**。返回的摘要是供后续阅读或核验的候选材料，不能将 `qrel=1` 解读为“支持该论断”，也不能据此声称完成自动事实核验。

## 项目中的数据处理

- 保留原始查询和文档 ID，标题与摘要拼接后参与检索；不把 qrels、支持/反驳标签或证据位置写入文本特征。
- BM25 使用词级分词；冻结 MiniLM 使用自己的 tokenizer。稠密编码存在最大输入长度截断，当前没有滑窗或句子级多向量索引。
- 内部训练/验证划分只读取 `qrels/train.tsv`。共享相关文档，或小写、去标点及数字归一化后文本相同的训练查询，按连通分组整体划分，减少相近论断跨内部划分的情况。
- 验证与测试的候选文档由检索器自然产生，不注入标签中的相关文档。训练阶段可以补入未召回的正例，以构造有效的排序学习样本；补入行为必须在训练报告中单独记录。

分组使用相关性标签是一种划分策略，应如实披露；它不是与标签无关的随机切分，也不能证明消除了全部语义重叠。原有 BEIR train/test 划分保持不变。

## 许可与署名

以[原始 SciFact LICENSE.md](https://github.com/allenai/scifact/blob/master/LICENSE.md)对不同组成部分的声明为准：

| 组成部分 | 上游声明 |
| --- | --- |
| Claims 与 evidence annotations | CC BY 4.0 |
| 来自 S2ORC 的 abstracts | ODC-By 1.0 |
| 原始 SciFact 仓库代码 | Apache 2.0 |

本仓库代码许可不替代数据或模型的许可。发布项目时保留以上出处；数据与模型通过脚本从上游下载，不把第三方语料误标为本项目原创。部分托管平台卡片的单一 license 字段与原始声明不一致，不能据此将所有组成部分归入一种许可。[BEIR](https://github.com/beir-cellar/beir#disclaimer)也说明其统一数据格式不等于为使用者授予所有源数据的使用权。

## 适用范围与局限

1. **规模与领域有限。** 5,183 篇摘要足以进行完整对照实验，但不能代表百万级语料的内存、延迟或召回行为，也没有证明中文或跨领域能力。
2. **未标注不等于不相关。** 训练把未出现在 qrels 的候选暂作负例；其中可能存在漏标相关文献。难负例挖掘会增加接触此类样本的机会。
3. **文本相关不等于可信。** 该检索任务没有估计研究质量、时效性、因果强度或论断真伪。
4. **预训练暴露无法完全排除。** MiniLM 上游训练材料包含 S2ORC、SPECTER 等科学文本资源；当前未做逐样本预训练污染审计。只能声称本项目监督训练未使用 BEIR test 标签，不能声称模型从未接触过相关文献。[MiniLM 模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
5. **公开测试集可被反复观察。** 后续若依据 test 错例修改参数，应将结果视为迭代开发结果，并另设未观察的评估集。

引用数据时请同时署名 SciFact 原作者与 BEIR 作者；BibTeX 可从[原论文](https://aclanthology.org/2020.emnlp-main.609/)和 [BEIR 仓库](https://github.com/beir-cellar/beir#citation)获取。
