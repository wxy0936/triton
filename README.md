# Triton 入门教程：Chapter 00-17

这是一个面向 Triton 初学者的渐进式教程项目。每章同时提供：

- `.ipynb`：包含 Markdown 讲解和可逐格运行的完整代码。
- `.py`：与 notebook 最终效果等价的独立可运行程序。
- PyTorch reference、Triton implementation、correctness check 和准确的 CUDA benchmark（Chapter 00 只做环境检查，不写 kernel）。

## 环境要求

- Python 3.10+
- 支持 CUDA 的 NVIDIA GPU
- CUDA 版本的 PyTorch
- Triton
- Jupyter Notebook

建议先根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择与你的 CUDA 环境匹配的 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

先检查环境：

```bash
python chapter_00_setup/setup_check.py
```

成功时会输出 `Triton tutorial environment is ready`。

## 运行 Notebook

在项目根目录执行：

```bash
jupyter notebook
```

浏览器打开后进入对应章节目录，从上到下依次运行所有 cell。Notebook 是展开的教学版本；同章 `.py` 是整理后的完整版本。

## 运行章节脚本

所有命令都在项目根目录执行，例如：

```bash
python chapter_01_vector_add/vector_add.py
python chapter_06_fused_softmax/fused_softmax.py
python chapter_07_matmul_baseline/matmul_baseline.py
python chapter_10_layernorm/layernorm.py
python chapter_13_flashattention_v1/flashattention_v1.py
python chapter_14_mini_project/mini_project.py
python chapter_15_performance_mental_model/performance_mental_model.py
python chapter_16_profiling_and_benchmarking/profiling_and_benchmarking.py
python chapter_17_attention_memory_analysis/attention_memory_analysis.py
```

每个脚本会创建温和规模的输入，运行 correctness check，并输出 PyTorch 与 Triton 的 benchmark。

## 章节目录

| 章节 | 主题 | 新概念 |
| --- | --- | --- |
| 00 | 环境检查 | PyTorch CUDA、Triton 和 GPU 信息 |
| 01 | Vector Add | program id、offset、mask、grid |
| 02 | Triton 编程模型 | program/block/grid、向量化 offset |
| 03 | Mask 与内存访问 | 边界 mask、stride、non-contiguous slice |
| 04 | Unary / Binary Ops | 逐元素 kernel、kernel fusion |
| 05 | Reduction Sum / Max | 一行一个 program、`tl.sum`、`tl.max` |
| 06 | Fused Softmax | 数值稳定、row-wise fusion |
| 07 | Matmul Baseline | 二维 tile、K 维分块、`tl.dot` |
| 08 | Matmul Optimized | grouped ordering、`num_warps`、`num_stages` |
| 09 | Autotune 与 Benchmark | CUDA 正确计时、warmup/repeat、`triton.autotune` |
| 10 | LayerNorm Forward | fp32 reduction、affine weight/bias |
| 11 | Attention 基础（进阶） | Q/K/V shape、causal mask、scaled attention |
| 12 | Online Softmax（进阶） | running max、denominator、output accumulator |
| 13 | FlashAttention v1 Forward（进阶） | Triton block attention、online softmax |
| 14 | Tiny Self-Attention（进阶） | projection、multi-head reshape、完整 attention 子层 |
| 15 | Performance Mental Model（CS336/LLM systems） | FLOPs、bytes、arithmetic intensity、roofline 思维 |
| 16 | Profiling 与 Benchmark（CS336/LLM systems） | 同步计时、forward/backward/train step、profiler |
| 17 | Attention Memory Analysis（CS336/LLM systems） | S² activation、tile working set、peak memory |

## 建议学习路线

1. 先完成 Chapter 00，确认 CUDA 和 Triton 能正常工作。
2. Chapter 01 写出第一个 kernel，理解最小的 Triton 工作流。
3. Chapter 02 暂时忽略性能，专门建立 program 和 grid 的心智模型。
4. Chapter 03 理解真实 tensor 地址为何需要 mask 和 stride。
5. Chapter 04 观察多个逐元素操作如何融合成一个 kernel。
6. Chapter 05 学习 reduction，为 softmax 和后续 layer normalization 打基础。
7. Chapter 06 把 load、reduction、数值稳定和 fusion 串起来。
8. Chapter 07 从二维 tile 开始实现矩阵乘法。
9. Chapter 08 只加入少量手写优化，观察 program ordering 和 launch 参数。
10. Chapter 09 学会可靠 benchmark，再使用少量 Config 做 autotune。
11. Chapter 10 将 reduction 组合成完整 LayerNorm forward。
12. Chapter 11–12 进入 attention/FlashAttention 进阶部分，先用 PyTorch 理解数学和 online softmax。
13. Chapter 13 把 online softmax 写成教学版 Triton FlashAttention forward。
14. Chapter 14 将该 kernel 接入 tiny self-attention，理解它与 Transformer attention 子层的关系。
15. Chapter 15–17 转向 CS336 / LLM systems 的系统分析视角：先估算性能上限，再做可信计时和 profiling，最后分析 attention 的显存瓶颈。
