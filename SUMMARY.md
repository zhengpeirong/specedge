# SpecEdge 代码库总结

## 项目概览

SpecEdge 是一个**边缘辅助的 LLM 推理框架**，发表于 NeurIPS 2025（[arXiv:2505.17052](https://arxiv.org/abs/2505.17052)）。其核心思想是将大语言模型推理任务分配到边缘设备（消费级 GPU）和服务器（高性能 GPU）之间，利用**投机解码（Speculative Decoding）**技术实现高效推理。

**主要性能优势：**
- 服务器吞吐量提升 **2.22×**
- 端到端延迟降低 **11.24%**（相比纯服务器方案）

## 目录结构

```
specedge/
├── config/              # 各部署模式的 YAML 配置模板
│   ├── specedge.example.yaml        # SpecEdge 主配置（多节点 + gRPC）
│   ├── auto_batch.example.yaml      # 自回归批处理基线配置
│   └── server_only.example.yaml     # 纯服务器基线配置
├── data/                # 基准测试数据集（JSON/JSONL 格式）
│   ├── c4_prompts.json              # C4 数据集
│   ├── mtbench_prompts.json         # 多轮对话基准
│   ├── oasst_prompts.json           # Open Assistant 数据集
│   ├── wikitext_prompts.json        # WikiText 数据集
│   └── specbench_prompts.jsonl      # SpecBench（多轮推理）
├── img/                 # 文档图片（架构图、性能图、参数图）
├── script/              # Bash 入口脚本
│   ├── batch_server.sh              # 启动 SpecEdge 批处理服务器
│   ├── client_host.sh               # 启动多节点客户端协调器
│   ├── client.sh                    # 单客户端执行
│   ├── auto_batch.sh                # 自回归批处理基线
│   ├── server_only.sh               # 纯服务器基线
│   └── compile_proto.sh             # 编译 Protobuf 定义
├── src/                 # Python 源代码（约 3,919 行）
│   ├── config.py        # 配置元类与参数加载器
│   ├── log.py           # 日志系统（含 JSONL 结果记录）
│   ├── util.py          # 工具函数（模型加载、数据集、计时、性能分析）
│   ├── metric/          # 性能指标分析脚本
│   ├── model/           # 自定义 LLM 模型实现
│   ├── script/          # 各模式的可执行入口
│   ├── specedge/        # 核心 SpecEdge 框架
│   ├── specedge_grpc/   # 自动生成的 Protobuf 文件
│   └── strategy/        # 推理策略（起草、验证、请求管理）
├── specedge.proto        # gRPC 服务接口定义
└── pyproject.toml        # 项目元数据与依赖
```

## 核心模块详解

### 1. gRPC 通信接口（`specedge.proto`）

定义边缘客户端与服务器之间的双向通信协议：

| 方法 | 描述 |
|------|------|
| `Validate` | 边缘发送候选 token 树 → 服务器返回验证结果 |
| `Sync` | 多客户端同步屏障 |

**消息字段（`ValidateRequest`）：** `client_idx`、`req_idx`、`input_ids`、`position_ids`、`cache_seq_indices`、`parent_indices`、`attention_mask`、`prefill`、`prefix`

### 2. 核心框架（`src/specedge/`）

| 模块 | 说明 |
|------|------|
| **`tree.py`** | 投机树数据结构，管理 token ID、位置、父节点索引、对数概率及状态标志 |
| **`engine/graph.py`** | CUDA 图优化推理引擎，支持静态计算图捕获以加速推理 |
| **`network/grpc.py`** | 异步 gRPC 客户端，负责将张量编码为字节流并发送至服务器 |
| **`client/specexec.py`** | 主客户端逻辑：维护投机树、调用边缘起草、提交服务器验证、接受/拒绝 token |
| **`client/proactive.py`** | 主动起草策略：在等待服务器响应期间，利用最优候选节点扩展树，减少空闲时间 |

#### 投机树节点状态转换

```
PROMPT → GENERATED → PROCESSED → CANDIDATE → POST_CANDIDATE → POST_PROCESSED
```

### 3. 推理策略（`src/strategy/`）

| 模块 | 职责 |
|------|------|
| **`request_manager.py`** | 批处理槽分配与请求生命周期管理（`Prefill → Generating → Finished`） |
| **`edge_draft/specexec.py`** | 边缘起草阶段：加载数据集提示词，填充批处理槽，生成投机树 |
| **`edge_verify/specexec.py`** | 边缘预验证：为服务器验证准备注意力掩码和位置 ID |
| **`server_verify/specexec/grpc.py`** | 服务端异步 gRPC 服务器：接收客户端请求，路由至验证引擎 |
| **`server_verify/specexec/server_only.py`** | 服务端验证引擎：在目标模型上执行 token 采样 |

### 4. 自定义模型实现（`src/model/`）

替换标准 Transformers 注意力机制，支持投机解码所需的 KV 缓存管理：

| 模块 | 说明 |
|------|------|
| **`llama.py`** | 自定义 `LlamaAttention`，支持 KV 缓存索引管理 |
| **`qwen3.py`** | 自定义 `Qwen3Attention`，含头部归一化 |
| **`cache.py`** | `KVCache` 类：每层存储 K/V 张量，支持批处理/序列索引及树节点的 gather 操作 |
| **`layer_split.py`** | 模型分层部署：`PartialFirstHalf`（边缘前半段）、`PartialSecondHalf`（服务器后半段） |

### 5. 性能指标（`src/metric/`）

| 脚本 | 分析目标 |
|------|----------|
| **`specedge.py`** | 主分析脚本：加载客户端/服务器 JSONL 日志，按任务类型计算延迟/吞吐量 |
| **`server_only.py`** | 纯服务器基线：分析起草阶段与目标阶段的耗时 |
| **`auto_batch.py`** | 批处理基线：跟踪前向传播时间、预填充/非预填充成本 |
| **`layer_split.py`** | 分层部署变体的性能分析 |

GPU 成本常量（`__init__.py`）：
- A100 40GB: $0.001125/秒
- A100 80GB: $0.001403/秒
- RTX 4090: $0.000097/秒

### 6. 配置系统（`src/config.py`）

基于**元类**的配置管理，使用环境变量进行懒加载：

| 配置类 | 用途 |
|--------|------|
| `SpecEdgeClientConfig` | 单客户端（基于提示词） |
| `SpecEdgeBatchClientConfig` | 批处理客户端（多请求） |
| `SpecEdgeServerConfig` | 目标模型服务器（单请求） |
| `SpecEdgeBatchServerConfig` | 批处理服务器（多请求） |
| `AutoregressiveBatchConfig` | 自回归批处理基线 |

### 7. 日志系统（`src/log.py`）

- **双通道输出**：控制台（INFO 级别以上）+ 文件（DEBUG 级别以上）
- **结果记录**：结构化 JSONL 格式保存实验数据
- **线程安全**：基于 `multiprocessing.Queue` + `QueueListener` 的异步日志

## 部署模式

### 模式一：SpecEdge（主要模式）

```
边缘（客户端）                    服务器
├─ 加载起草模型                   ├─ 加载目标模型
├─ 投机树起草                     └─ gRPC 服务（端口 8000）
└─ 循环：生成候选 → gRPC 验证 → 更新树
```

```bash
# 服务器端
./script/batch_server.sh -f config/specedge.example.yaml

# 边缘端（多节点）
./script/client_host.sh -f config/specedge.example.yaml
```

### 模式二：Auto Batch（自回归批处理基线）

单 GPU 标准自回归批量推理，无投机解码。

```bash
./script/auto_batch.sh -f config/auto_batch.example.yaml
```

### 模式三：Server-Only（纯服务器基线）

服务器与客户端在同一机器，无边缘辅助。

```bash
./script/server_only.sh -f config/server_only.example.yaml
```

### 模式四：Layer Split（分层部署变体）

将模型分为前后两半，分别在边缘和服务器上运行。

## 投机解码参数

![](img/param.svg)

| 参数 | 作用 |
|------|------|
| `max_n_beams` | 每次迭代最多转发的节点数（控制计算开销） |
| `max_beam_len` | 投机树深度（连续起草迭代次数） |
| `max_branch_width` | 每个父节点的最大子节点数（top-k 续词数） |
| `max_budget` | 树中最大节点总数（按累积对数概率剪枝，衰减因子 0.9） |

**主动起草（Proactive）模式：**
- `included`：包含已选 token 的子树进行主动扩展
- `excluded`：从未被选中的候选节点开始主动扩展
- `disabled`：禁用主动起草

## 技术依赖

| 类别 | 主要依赖 |
|------|----------|
| 深度学习 | PyTorch ~2.9、Transformers ~4.57、Accelerate ~1.10 |
| 分布式通信 | gRPC ~1.75（grpcio） |
| 数据分析 | Polars ~1.34、Matplotlib ~3.10、Plotly ~6.3 |
| 文本处理 | SentencePiece ~0.2、TikToken ~0.12 |
| 界面输出 | Rich ~14.2 |
| 运行环境 | Python ~3.14 |

## 实验结果分析

运行指标脚本前，需将服务器和边缘的 JSONL 日志文件汇集到同一目录：

```bash
. .venv/bin/activate

# SpecEdge 指标
python src/metric/specedge.py -d result/demo/specedge --gpu "A100-40"

# Auto Batch 指标
python src/metric/auto_batch.py -d result/demo/auto_batch --gpu "A100-80"

# Server-Only 指标
python src/metric/server_only.py -d result/demo/server_only --gpu "A100-40"
```

## 引用

```bibtex
@inproceedings{park2025specedge,
  author = {Jinwoo Park and Seunggeun Cho and Dongsu Han},
  title = {SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs},
  booktitle = {Annual Conference on Neural Information Processing Systems},
  year = {2025},
  eprint = {2505.17052},
  archivePrefix = {arXiv},
  primaryClass= {cs.CL}
}
```
