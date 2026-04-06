# SpecEdge 项目总结 / Project Summary

## 概述 / Overview

**SpecEdge** 是一个发表于 NeurIPS 2025 的边缘辅助大语言模型推理服务框架。其核心思想是利用消费级边缘 GPU（如 RTX 4090）进行草稿 token 生成，配合服务器端高端 GPU（如 A100）进行验证，通过投机解码（Speculative Decoding）实现高吞吐、低延迟的 LLM 在线服务。

**SpecEdge** is an edge-assisted LLM inference serving framework published at NeurIPS 2025. It combines consumer-grade edge GPUs (e.g., RTX 4090) for draft token generation with server-side high-end GPUs (e.g., A100) for verification via speculative decoding, achieving high-throughput, low-latency LLM serving.

### 关键指标 / Key Results

- **2.22×** 服务器吞吐量提升（对比纯服务器基线）
- **11.24%** 端到端延迟降低
- 通过将草稿生成分散至边缘节点，显著降低服务成本

---

## 系统架构 / System Architecture

```
┌───────────────────────────────────────┐
│         边缘节点 / Edge Nodes          │
│  ┌───────────────────────────────┐    │
│  │  草稿模型 Draft Model (小模型)  │    │
│  │  SpecExecClient               │    │
│  │  树结构候选词 Tree Candidates   │    │
│  └───────────────────────────────┘    │
└──────────────┬────────────────────────┘
               │  gRPC Validate RPC
               │  (树结构数据 tree data)
               ▼
┌───────────────────────────────────────┐
│         服务器 / Server (A100)         │
│  ┌───────────────────────────────┐    │
│  │  目标模型 Target Model (大模型)  │    │
│  │  SpecExecBatchServer          │    │
│  │  验证逻辑 Verification Logic    │    │
│  └───────────────────────────────┘    │
└───────────────────────────────────────┘
```

### 主要组件 / Main Components

| 组件 | 文件 | 职责 |
|------|------|------|
| 树结构 Tree | `src/specedge/tree.py` | 投机解码候选词树的数据结构 |
| 边缘客户端 Edge Client | `src/specedge/client/specexec.py` | 草稿生成与验证循环编排 |
| 主动草稿 Proactive Draft | `src/specedge/client/proactive.py` | 利用奖励 token 进行异步草稿生成 |
| 推理引擎 Graph Engine | `src/specedge/engine/graph.py` | CUDA Graph 捕获与回放加速 |
| 网络层 Network | `src/specedge/network/grpc.py` | 边缘-服务器 gRPC 通信 |
| 请求管理 Request Manager | `src/strategy/request_manager.py` | 请求状态跟踪（预填充/生成/完成）|
| 边缘草稿策略 Edge Draft | `src/strategy/edge_draft/specexec.py` | 批量草稿生成 |
| 服务器验证策略 Server Verify | `src/strategy/server_verify/specexec/grpc.py` | 服务器端验证与结果广播 |
| 模型实现 Models | `src/model/llama.py`, `src/model/qwen3.py` | Llama/Qwen 自定义模型实现 |
| KV 缓存 KV Cache | `src/model/cache.py` | 高效注意力计算缓存管理 |
| 配置系统 Config | `src/config.py` | 基于元类的环境变量驱动配置 |
| 评估指标 Metrics | `src/metric/specedge.py` | 性能分析（吞吐、延迟、成本）|

---

## 投机解码实现 / Speculative Decoding Implementation

SpecEdge 基于 **SpecExec 算法**，使用树形投机解码：边缘节点并行预测多个 token 序列（树形结构），服务器一次性批量验证。

### 树生长阶段（边缘）/ Tree Growth Phase (Edge)

```
初始树 T = {前缀 prefix}
重复 max_beam_len 次:
  candidates = 按对数概率排名的 top-K 节点（max_n_beams 个）
  logits = 通过草稿模型前向推理(candidates)
  new_tokens = 每个候选节点的 top-branch_width 个 token
  按预算修剪 new_tokens 到 max_budget 个节点（累积对数概率）
  T = T ∪ new_tokens
将 T 发送至服务器进行验证
```

### 关键参数 / Key Parameters

| 参数 | 含义 |
|------|------|
| `max_n_beams` | 每次前向推理最多处理的节点数 |
| `max_beam_len` | 投机树最大深度（生成步数）|
| `max_branch_width` | 每个父节点最多生成的子 token 数 |
| `max_budget` | 树中允许的最大节点总数（触发修剪）|

### Token 状态机 / Token State Machine

```
PROMPT → GENERATED → PROCESSED → CANDIDATE → POST_CANDIDATE → POST_PROCESSED
```

---

## 关键优化技术 / Key Optimization Techniques

### 1. CUDA Graph 加速 / CUDA Graph Acceleration

针对不同 beam 大小（1 到 max_n_beams）预先捕获 CUDA 核函数序列，推理时直接回放，大幅减少核函数启动开销。

Pre-captures CUDA kernel sequences for each beam size and replays them during inference, significantly reducing kernel launch overhead.

### 2. 主动草稿 / Proactive Edge Drafting

在等待服务器验证结果期间，边缘节点从"奖励 token"（被拒绝但概率较高的备选 token）继续草稿生成，充分利用空闲 GPU 时间。

While waiting for server validation, the edge continues drafting from "bonus tokens" (high-probability rejected alternatives), fully utilizing otherwise-idle GPU time.

### 3. 异步 gRPC / Asynchronous gRPC

服务器采用多进程异步架构（接收队列 → 推理进程 → 响应队列），将网络通信与模型推理解耦。

The server uses a multiprocess async architecture (recv_queue → inference process → resp_queue) to decouple network I/O from model inference.

### 4. 预分配 KV 缓存 / Pre-allocated KV Cache

KV 缓存按固定形状 `(num_layers, batch_size, num_kv_heads, max_len, head_dim)` 预分配，兼容 CUDA Graph 的静态形状要求。

KV caches are pre-allocated with fixed shape to satisfy CUDA Graph's static shape requirements.

### 5. 预算修剪 / Budget Pruning

使用带 0.9 衰减因子的累积对数概率对树节点进行优先级排序，保留最有可能的 `max_budget` 个节点。

Uses cumulative log-probability with a 0.9 decay factor to rank tree nodes, retaining only the top `max_budget` most likely nodes.

---

## 配置系统 / Configuration System

配置基于元类 `_ConfigMeta` 实现，通过环境变量（`SPECEDGE_*`）驱动，支持懒初始化。

Configuration is driven by environment variables (`SPECEDGE_*`) via a metaclass-based system with lazy initialization.

### 主要配置类 / Main Config Classes

- **`SpecEdgeClientConfig`**: 边缘客户端（草稿模型、投机参数、数据集等）
- **`SpecEdgeServerConfig`**: 服务器端（目标模型、批大小、温度等）
- **`SpecEdgeBatchServerConfig`**: 批处理服务器（批类型、缓存预填充等）
- **`AutoregressiveBatchConfig`**: 自回归批处理基线配置

---

## 评估指标 / Evaluation Metrics

### 追踪指标 / Tracked Metrics

- **草稿阶段** / Draft Phase: 端到端时间、每次迭代前向时间
- **验证阶段** / Validation Phase: 端到端时间、服务器处理时间、预填充 token 数
- **结果** / Results: 接受的 token 数、吞吐量（tokens/s）

### 成本分析 / Cost Analysis

以下单价来自论文实验时所用云平台报价（仅供参考，实际费率因时间和供应商而异）。

The following unit costs are based on cloud platform pricing used during paper experiments (for reference only; actual rates vary by time and provider).

| 硬件 | 单价（美元/秒）|
|------|--------------|
| A100 40GB | $0.001125 |
| A100 80GB | $0.001403 |
| RTX 4090 | $0.0000972 |

---

## 运行模式 / Running Modes

| 模式 | 脚本 | 说明 |
|------|------|------|
| SpecEdge（完整）| `script/batch_server.sh` + `script/client_host.sh` | 边缘+服务器完整框架 |
| Auto Batch | `script/auto_batch.sh` | 单机批处理基线 |
| Server Only | `script/server_only.sh` | 纯服务器基线 |

---

## 支持的模型与数据集 / Supported Models & Datasets

- **模型**: Llama 系列, Qwen3 系列（可扩展至其他模型）
- **数据集**: C4, MTBench, OASST, Wikitext, SpecBench

---

## 引用 / Citation

```
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
