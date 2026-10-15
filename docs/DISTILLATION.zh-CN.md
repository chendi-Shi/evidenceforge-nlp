# v3：资源受限的检索蒸馏与迁移退化

**状态：2026-10-15 已完成全部 12 组训练、SCIDOCS 评测和独立产物复算。** 主比较显示表示保持缓解了 KD 的负迁移，但没有超过冻结原模型。完整数表见[结果解读](DISTILLATION_RESULTS.zh-CN.md)，机器产物见 [`results/distillation`](../results/distillation/)。本文说明实验设计、复现和证据边界；v1/v2 成绩仍属于各自历史实验。

## 问题

在固定本机计算预算下，小编码器学习较大教师的检索分布，能否改善检索质量，同时保留原有的跨任务能力？v2 中难负例训练明显退化，过滤并未建立稳定超过原模型的优势。本轮针对训练目标和迁移能力建立对照，不以增加接口或模型名称作为算法贡献。

## 数据和实验

- 已完整扫描固定版本的 `sentence-transformers/msmarco-bm25/triplet` 的 502,931 行，按规范化查询分组和全局哈希抽样，固定 12,000 条训练查询、1,000 条验证查询和 25,677 篇去重文档。排除 43,907 条正负文档同文的三元组；训练与验证共享 49 篇文档，查询互斥。详见 [数据审计](../results/distillation/data_manifest.json)。
- 保留该来源中可观测的所有正例；排除同文负例和已知正例。未知相关文档仍可能被当作负例。划分保证查询互斥，允许部分文档交叠并披露数量。
- 教师为固定版本 Qwen3-Embedding-0.6B Q8_0，本机 Iris Xe/Vulkan 推理，512-token 上限；学生为 MiniLM，256-token 上限。教师和学生输入覆盖不同，不能单独归因为参数量差异。
- 学生冻结词嵌入和前四层，更新最后两层全部 3,548,928 个参数。FP32 前缀缓存和原始模型向量保存在本地，关闭 dropout 并检查缓存与完整前向的一致性。计算图仍保留学生后两层的梯度。按长度排序、微批前向后恢复顺序，再对完整候选池计算损失，不将微批当成更小的负例池。
- 四组：监督 CE、CE+教师 KL、CE+原始表示保持、CE+KL+表示保持；三个批次顺序种子（17/42/73）。同一个 seed 内四组共享数据、初始化、批次顺序和最大训练步数；不同 seed 改变批次顺序。保持项是与原始模型向量的余弦距离。方法是已有目标的受控比较，不宣称算法首创。
- 固定一轮、batch 16、学习率 `2e-5`，检索温度 `0.05`，KL 权重 1、表示保持权重 10；KL 不乘温度平方。精确配置和实现哈希在运行前写入 `results/distillation/protocol.json`。

每个 batch 的 16 条查询共享各自正例和 BM25 负例组成的最多 32 篇候选；去重并屏蔽非主正例的其他已知正例，避免将它们作为负例。CE 指向所选标注正例，KL 比较同一有效候选集合上的教师与学生相似度分布，保持项取查询和唯一文档对原始表示的余弦距离均值，各占一半。总目标为 `CE + λKD × KL(teacher || student) + λanchor × anchor`。这与 v2 的每查询两负例、无 in-batch negatives 不同。

## 验证、封存和解释

全部验证查询搜索同一个去重后的验证文档池，以验证 nDCG@10 选择固定周期保存的 checkpoint。它是抽样语料上的验证，**不是 MS MARCO 880 万段落的官方检索成绩**。不按验证损失代替排名选模，也不按新测试集选择训练组。

所有选定模型保存并校验后，才加载 BEIR SCIDOCS 完整 25,657 篇文档、1,000 条测试查询；按 BEIR 规则排除 query ID 与 document ID 相同的条目。它是经典的论文引用检索子任务，不是原 SciDocs 全套任务，也不是新发布的基准。已看过的 SciFact/NFCorpus 只能作为回归集，不能再次充当全新验证证据。

主比较是 SCIDOCS 上 `kd_anchor − kd`，分别报告三个 seed、seed 标准差及每个 seed 的配对查询区间。同时必须看相对冻结 base 的表现、源域验证收益和向量漂移。如果保持项只是让模型几乎不更新，或只超过一个退化的组，就不能写成泛化能力提升。四组结果均保留；不把 seed×query 当成独立样本扩大显著性。

预设的次要交互比较为 `(kd_anchor − kd) − (anchor − ce)`：它检查表示保持对蒸馏组的影响是否不同于普通监督组，不能用单独的 `kd_anchor − kd` 代替。当前记录的漂移来自**源域验证**的查询与文档，没有测量 SCIDOCS 表示漂移；它只辅助解释训练行为，不能证明目标域几何保持。基础权重固定且 dropout 关闭，各训练 seed 的主要变化来自批次顺序。

结果解释还须保留每组在 250/500/750 步的源域验证记录，作相同步数比较。不同组可能选择不同的 best step，较小的漂移也可能来自较早选模；不能完全归因于保持项。目标测试集只评测预先选定的权重，不能反过来重新挑 checkpoint。即使主差值为正，若仍低于 frozen base，也只能说缓解了该配置下的负迁移；与 base 的差异区间包含 0 不等于证明无损保持。

## 完成后的结论

SCIDOCS 的冻结 base nDCG@10 为 **0.216406**；CE、KD、anchor、KD+anchor 的三种子均值分别为 **0.205354、0.209372、0.212952、0.215630**。主差值均值 **+0.006258**，相对 KD 均值 **+2.99%**；seed 17/42/73 的配对查询 95% 区间分别为 `[0.003196, 0.011209]`、`[0.001048, 0.009643]`、`[0.002233, 0.010232]`。每个 KD+anchor 种子仍低于 base，不构成超过原模型、SOTA 或普遍泛化改进的证据。

四组源域验证 nDCG@10 较原模型的平均收益分别为 +0.035870、+0.023352、+0.038574、+0.011437，说明源域提升和目标域下降可以同时发生。次要交互量均值 −0.001340，三个区间均跨 0；没有建立表示保持与蒸馏的特殊协同优势。KD+anchor 的源域漂移较小，但这里同时改变了目标约束与选中步数，且没有测目标域漂移，不能把这种关联解释为已证明的机制。[次要分析](../results/distillation/analysis.json)保留每组源域轨迹、漂移与交互量。

三个 seed 使用相同预训练初始化、固定数据、关闭 dropout，只覆盖训练批次顺序的变化；逐查询 bootstrap 条件于已训练模型。它们没有覆盖数据抽样、预训练暴露、硬件非确定性等全部不确定性，也没有进行查询家族聚类或多重比较校正。教师在 12,000 个训练三元组中有 4,458 次更偏好 BM25 负例；[教师标注一致性记录](../results/distillation/teacher_label_agreement.json)披露这种冲突，但不将其判定为标注错误，也未据此删除样本。

## 选中权重与最终权重

每组训练到 750 步；测试使用源域验证选中的 checkpoint，不能根据 SCIDOCS 分数重选。选择步骤如下：

| 组 | seed 17 | seed 42 | seed 73 |
|---|---:|---:|---:|
| CE | 500 | 250 | 250 |
| KD | 750 | 750 | 750 |
| anchor | 750 | 750 | 500 |
| KD+anchor | 500 | 750 | 750 |

权重目录为 `results/distillation/training/<arm>/<seed>/`，`arm` 为 `ce/kd/anchor/kd_anchor`。**当前有效文件由 `selected/metadata.json` 与 `final/metadata.json` 的 `filename` 字段指定**，相应 `suffix_sha256` 是文件 SHA256，`parameter_sha256` 是参数内容身份，不能把两种哈希混用。本地 `selected/` 可能保留较早候选文件；公开产物仅包含当前 metadata 引用的 selected/final 权重，不包含旧候选或优化器断点。[产物清单](../results/distillation/artifact_manifest.json)列出 218 份文件（不含清单自身），其中 24 个权重文件，共 340,780,416 字节。

例如 [KD+anchor / seed 17 的 selected metadata](../results/distillation/training/kd_anchor/17/selected/metadata.json) 指向第 500 步 `suffix-56088c97fcbd2452fe9f.safetensors`；[final metadata](../results/distillation/training/kd_anchor/17/final/metadata.json) 指向第 750 步 `suffix-35cb146f63b62521fc86.safetensors`。SCIDOCS 分数属于前者。所有组的[冻结记录](../results/distillation/frozen_students.json)保存 selected metadata 的 SHA256 和最终权重身份，独立审计核对实际文件。复用时还需 metadata 中固定的基础模型、tokenizer、配置与 256-token 长度；后两层权重不是可单独运行的完整编码器。

## 本机运行

以下命令针对 **Windows、Python 3.12 和可用 Vulkan 后端**。正式 runner 固定教师后端为 Vulkan，学生使用 CPU；不能直接当作 Linux 或纯 CPU 教师运行指南。先在仓库根目录创建环境、按记录依赖安装，并显式下载固定浮点 MiniLM：`prepare` 本身只准备源数据，不会下载学生基础权重。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -c requirements-distillation.lock -e ".[dev,distillation]"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from evidenceforge.model_assets import download_float_minilm; print(download_float_minilm(Path('cache')))"

# 独立目录：不要把新运行写入仓库已发布的 results/distillation。
$distillReproArgs = @('--output-dir', 'results/distillation-reproduction', '--work-dir', (Join-Path $env:LOCALAPPDATA 'EvidenceForge/distillation-v3-reproduction'))
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment prepare @distillReproArgs
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment pilot @distillReproArgs --max-hours 1
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment register @distillReproArgs
.\.venv\Scripts\python.exe -m evidenceforge.distill_experiment run @distillReproArgs --max-hours 11.5
Get-Content results/distillation-reproduction/live_status.json
```

浮点模型下载入口校验固定版本资产，默认目录与 `--base-model-dir` 一致。Qwen GGUF 和固定 Windows llama.cpp 资产由教师初始化入口下载校验。数据默认使用 `data/distillation`，SCIDOCS 使用其相邻 `data/scidocs`；共享已验证的下载资产不会重用新目录中的训练 checkpoint，但已有逐文本教师缓存可能命中，因此重跑耗时不等于首次运行耗时。新实验若再次开始而不是恢复，需另选尚未使用的 output/work 目录。

预算到期后用相同 `run` 命令恢复；不是每次都重新 `prepare/pilot/register`。程序允许的单次上限为 12 小时，在批次边界检查，这不是整个实验的总耗时或总预算。教师向量逐批落盘；学生每 50 步及验证前保存完整断点，正常预算退出时额外保存当前步，非正常中断可能重算最后一小段。恢复必须匹配数据、协议、基础模型、配置和优化状态。上述 work 目录位于本机 LocalAppData，避免大型前缀缓存频繁同步到 OneDrive。电脑睡眠或断电会影响完成时间，程序不更改系统电源设置。

历史隐藏启动器 `scripts/launch_distillation.py` 固定写入 `results/distillation`，不支持上述独立 output/work 参数，因此新复现使用模块的 `run` 命令。已完成环境优先使用下方只读审计，不重新注册或覆盖已发布协议。命令与参数按当前实现核对；没有为这次文档更新重新执行一轮训练。

实机 pilot 只估计吞吐和检查梯度，不是训练结果。其短时速度不能保证整夜速度；完整评测和验证成本必须另算。进程结束也不自动意味着实验成功，应核对 `live_status.json`、全部权重、排名、逐查询指标和最终报告。

实际 [执行账本](../results/distillation/execution_summary.json) 记录 12 × 750 = **9,000 次有效更新**，本地调用墙钟累计计费 **15.5736273522 小时**，在 23 小时长实验预算内。它不是 CPU/GPU 利用时长之和，也不包含预检、数据准备和调用间等待。一次 Windows 更新关机使用系统关闭时间作为结束时间保守上界，比最后确认进度多 **115.741178 秒**，不能精确声称为工作进程退出时间。`ce/42` 从第 600 步断点恢复，重算此前观察到的 601–644 共 44 步；实际尝试计算多于最终轨迹，不能把它们累加成额外有效训练步。各训练 report 的 `current_invocation_training_seconds` 只计当次调用，不能直接相加冒充端到端总时间。

历史 [64 查询学生预检](../results/distillation/pilot_student.json) 和[教师预检](../results/distillation/pilot_teacher.json) 只用于启动前吞吐估算与正确性抽查，权重未进入正式评测。完整实验耗时以账本口径报告，不能用 pilot 每步耗时乘步数替代。

最终发布前，本地 Windows Python 3.12 的 **213 项测试通过（62.27 秒）**，`ruff check src tests scripts` 通过。测试包含完整候选池与微批梯度等价、继续训练与断点恢复一致、冻结前禁止读取目标数据及篡改权重拒绝复用。小模型和合成数据只用于代码正确性检查；[预检记录与预算计算](../results/distillation/preflight.json)均不作为模型效果证据。

## 完成后的独立复算

下面默认检查仓库中已发布的 `results/distillation`。需要事先按固定来源取得 `data/distillation/prepared` 和 `data/scidocs` 的原始数据；Git 仓库不附带这些文本与 qrels。只有结果文件的干净 clone 不能直接完成审计。前述完整重跑会准备数据；若只补齐审计输入，可先对独立 output 目录执行 `prepare`，再调用 `evidenceforge.distill_eval.load_or_download_scidocs(Path('data/scidocs'))`，均无需启动模型。审计新重跑的结果时，为以下两个脚本添加 `--results-dir results/distillation-reproduction`。

[`audit_distillation.py`](../scripts/audit_distillation.py) 不调用模型或生产评测函数，从保存的 TREC 和本地 qrels 复算 13 份排名的四项指标及每个 seed 的主比较区间，核对固定协议、数据身份、12 组训练轨迹、验证选模和权重文件。[已保存审计](../results/distillation/independent_audit.json) 为 `status=passed, ok=true`；[次要分析](../results/distillation/analysis.json) 为 `status=complete, ok=true`。所有必需结果尚未齐全时审计返回 `pending`（退出码 2），不会提前读取 SCIDOCS；一致性通过为 0，失败为 1。

审计证明的是本地保存产物的一致性，**不重新训练或推理，不验证计时真实性，不认证历史注册/封存时间，不检查张量数值，也不能证明保存的 top-50 之外实际发生了何种计算**。哈希不等于外部时间戳或真实性认证；合成测试通过也不等于模型效果成立。

```powershell
.\.venv\Scripts\python.exe scripts/audit_distillation.py --check
.\.venv\Scripts\python.exe scripts/analyze_distillation_results.py
# 可选：另存本次复查，不覆盖仓库中已发布的审计与分析。
.\.venv\Scripts\python.exe scripts/audit_distillation.py --output results/distillation/audit.recheck.json
.\.venv\Scripts\python.exe scripts/analyze_distillation_results.py --output results/distillation/analysis.recheck.json
```

[`analyze_distillation_results.py`](../scripts/analyze_distillation_results.py) 只读取已完成的 JSON 结果，保留每个 seed 的目标基线差值、四组交互量、选中步数及源域验证轨迹。次要比较的区间使用独立的 Python 随机抽样，有限次抽样的端点可能与主比较程序不同；不会替代独立 TREC 复算，也不会自动把正差值解释为普遍泛化提升。两个工具的小型合成测试均只验证程序逻辑。

## 来源与使用范围

- [训练数据与固定格式](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25)：项目只公开抽样规则、身份和统计，不重新分发原始训练文本。
- [MS MARCO 条款](https://microsoft.github.io/msmarco/#terms-and-conditions)：非商业研究用途；本仓库代码许可证不改变数据许可。
- [Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
- [BEIR 数据说明](https://github.com/beir-cellar/beir#available-datasets)、[SCIDOCS 原始许可证](https://github.com/allenai/scidocs/blob/master/LICENSE)、[BEIR SCIDOCS 数据卡](https://huggingface.co/datasets/BeIR/scidocs)：分别保存来源许可信息，不将原始项目与 BEIR 分发卡的声明混同。
- [Sentence Transformers 蒸馏说明](https://sbert.net/examples/sentence_transformer/training/distillation/README.html)：教师到学生的知识迁移是已有训练范式；本项目研究本地预算下的实验取舍与失败边界。
