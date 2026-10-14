# v3：资源受限的检索蒸馏与迁移退化

**状态：2026-10-14 已启动全量教师向量生成，完整训练和迁移评测待产出。** 数据准备与实机预检已完成；当日上午第一次启动因低内存退出，内存恢复后于 11:18（Asia/Shanghai）启动，并确认查询向量进度增长。v1/v2 的原有成绩保留，不能将它们写成这一轮蒸馏的收益。固定协议与预检产物公开在 [`results/distillation`](../results/distillation/)；预检权重全部丢弃。

## 问题

在固定本机计算预算下，小编码器学习较大教师的检索分布，能否改善检索质量，同时保留原有的跨任务能力？v2 中难负例训练明显退化，过滤并未建立稳定超过原模型的优势。本轮针对训练目标和迁移能力建立对照，不以增加接口或模型名称作为算法贡献。

## 数据和实验

- 已完整扫描固定版本的 `sentence-transformers/msmarco-bm25/triplet` 的 502,931 行，按规范化查询分组和全局哈希抽样，固定 12,000 条训练查询、1,000 条验证查询和 25,677 篇去重文档。排除 43,907 条正负文档同文的三元组；训练与验证共享 49 篇文档，查询互斥。详见 [数据审计](../results/distillation/data_manifest.json)。
- 保留该来源中可观测的所有正例；排除同文负例和已知正例。未知相关文档仍可能被当作负例。划分保证查询互斥，允许部分文档交叠并披露数量。
- 教师为固定版本 Qwen3-Embedding-0.6B Q8_0，本机 Iris Xe/Vulkan 推理，512-token 上限；学生为 MiniLM，256-token 上限。教师和学生输入覆盖不同，不能单独归因为参数量差异。
- 学生冻结词嵌入和前四层，更新最后两层全部 3,548,928 个参数。FP32 前缀缓存和原始模型向量保存在本地，关闭 dropout 并检查缓存与完整前向的一致性。计算图仍保留学生后两层的梯度。按长度排序、微批前向后恢复顺序，再对完整候选池计算损失，不将微批当成更小的负例池。
- 四组：监督 CE、CE+教师 KL、CE+原始表示保持、CE+KL+表示保持；三个优化随机种子（17/42/73），共享数据、初始化、批次顺序和最大训练步数。保持项是与原始模型向量的余弦距离。方法是已有目标的受控比较，不宣称算法首创。
- 固定一轮、batch 16、学习率 `2e-5`，检索温度 `0.05`，KL 权重 1、表示保持权重 10；KL 不乘温度平方。精确配置和实现哈希在运行前写入 `results/distillation/protocol.json`。

## 验证、封存和解释

全部验证查询搜索同一个去重后的验证文档池，以验证 nDCG@10 选择固定周期保存的 checkpoint。它是抽样语料上的验证，**不是 MS MARCO 880 万段落的官方检索成绩**。不按验证损失代替排名选模，也不按新测试集选择训练组。

所有选定模型保存并校验后，才加载 BEIR SCIDOCS 完整 25,657 篇文档、1,000 条测试查询；按 BEIR 规则排除 query ID 与 document ID 相同的条目。它是经典的论文引用检索子任务，不是原 SciDocs 全套任务，也不是新发布的基准。已看过的 SciFact/NFCorpus 只能作为回归集，不能再次充当全新验证证据。

主比较是 SCIDOCS 上 `kd_anchor − kd`，分别报告三个 seed、seed 标准差及每个 seed 的配对查询区间。同时必须看相对冻结 base 的表现、源域验证收益和向量漂移。如果保持项只是让模型几乎不更新，或只超过一个严重退化的组，就不能写成泛化能力提升。四组结果均保留；不把 seed×query 当成独立样本扩大显著性，不承诺正结果。

预设的次要交互比较为 `(kd_anchor − kd) − (anchor − ce)`：它检查表示保持对蒸馏组的影响是否不同于普通监督组，不能用单独的 `kd_anchor − kd` 代替。当前记录的漂移来自**源域验证**的查询与文档，没有测量 SCIDOCS 表示漂移；它只辅助解释训练行为，不能证明目标域几何保持。基础权重固定且 dropout 关闭，各训练 seed 的主要变化来自批次顺序。

结果解释还须保留每组在 250/500/750 步的源域验证记录，作相同步数比较。不同组可能选择不同的 best step，较小的漂移也可能来自较早选模；不能完全归因于保持项。目标测试集只评测预先选定的权重，不能反过来重新挑 checkpoint。即使主差值为正，若仍低于 frozen base，也只能说缓解了该配置下的负迁移；与 base 的差异区间包含 0 不等于证明无损保持。

## 本机运行

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,distillation]"
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment prepare
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment pilot --max-hours 1
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment register
.\.venv\Scripts\python.exe scripts/launch_distillation.py --max-hours 11.5
Get-Content results/distillation/live_status.json
```

长实验只使用本机，教师与学生顺序运行；本次启动预算 11.5 小时，程序允许的单次上限为 12 小时，在批次边界检查。教师向量逐批落盘；学生每 50 步及验证前保存完整断点，正常预算退出时额外保存当前步，非正常中断可能重算最后一小段。再次执行可恢复，必须匹配数据、协议、基础模型、配置和优化状态。缓存默认位于 `%LOCALAPPDATA%/EvidenceForge/distillation-v3`，避免大型前缀缓存频繁同步到 OneDrive。电脑睡眠或断电会影响完成时间，程序不更改系统电源设置。

实机 pilot 只估计吞吐和检查梯度，不是训练结果。其短时速度不能保证整夜速度；完整评测和验证成本必须另算。进程结束也不自动意味着实验成功，应核对 `live_status.json`、全部权重、排名、逐查询指标和最终报告。

2026-10-14 的最后一次 [64 查询学生预检](../results/distillation/pilot_student.json) 完成 4 次非零梯度更新，每步完整批次耗时中位数 1.411 秒；8 条文本的缓存/完整前向抽查逐位相同。[教师预检](../results/distillation/pilot_teacher.json) 使用 64 查询和 128 文档，无向量缓存命中。按这些短测外推，源域教师推理约 5.83 小时、12 次学生训练的纯更新约 3.53 小时；尚未计入全量缓存构建、验证、最终 SCIDOCS 编码和持续负载影响。这是预算估计，不是实测整轮耗时，也不是质量成绩。

本地 Windows Python 3.12 的 167 项测试通过，Ruff 检查通过。测试包含完整候选池与微批梯度等价、继续训练与断点恢复一致、冻结前禁止读取目标数据及篡改权重拒绝复用。小模型和合成数据只用于代码正确性检查；[预检记录与预算计算](../results/distillation/preflight.json) 均不作为模型效果证据。

## 完成后的独立复算

[`audit_distillation.py`](../scripts/audit_distillation.py) 不调用模型或生产评测函数，从保存的 TREC 和本地 qrels 复算 13 份排名的四项指标及每个 seed 的主比较区间，核对固定协议、数据身份、12 组训练轨迹和权重文件。所有必需结果尚未齐全时返回 `pending`（退出码 2），不会提前读取 SCIDOCS；一致性通过为 0，失败为 1。文件校验只能证明保存产物之间一致，不能独立证明历史计算过程。

```powershell
.\.venv\Scripts\python.exe scripts/audit_distillation.py --check
# 完整复算通过后，显式保存两个单独产物；默认只读输出，不覆盖已有文件。
.\.venv\Scripts\python.exe scripts/audit_distillation.py --output results/distillation/independent_audit.json
.\.venv\Scripts\python.exe scripts/analyze_distillation_results.py --output results/distillation/analysis.json
```

[`analyze_distillation_results.py`](../scripts/analyze_distillation_results.py) 只读取已完成的 JSON 结果，保留每个 seed 的目标基线差值、四组交互量、选中步数及源域验证轨迹。次要比较的区间使用独立的 Python 随机抽样，有限次抽样的端点可能与主比较程序不同；不会替代独立 TREC 复算，也不会自动把正差值解释为普遍泛化提升。两个工具的小型合成测试均只验证程序逻辑。

## 来源与使用范围

- [训练数据与固定格式](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25)：项目只公开抽样规则、身份和统计，不重新分发原始训练文本。
- [MS MARCO 条款](https://microsoft.github.io/msmarco/#terms-and-conditions)：非商业研究用途；本仓库代码许可证不改变数据许可。
- [Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
- [BEIR 数据说明](https://github.com/beir-cellar/beir#available-datasets)、[SCIDOCS 原始许可证](https://github.com/allenai/scidocs/blob/master/LICENSE)、[BEIR SCIDOCS 数据卡](https://huggingface.co/datasets/BeIR/scidocs)：分别保存来源许可信息，不将原始项目与 BEIR 分发卡的声明混同。
- [Sentence Transformers 蒸馏说明](https://sbert.net/examples/sentence_transformer/training/distillation/README.html)：教师到学生的知识迁移是已有训练范式；本项目研究本地预算下的实验取舍与失败边界。
