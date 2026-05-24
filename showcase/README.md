# QuantumGPT Phase 1-2 可展示内容

## 文件清单

| 文件 | 说明 |
|------|------|
| `QuantumGPT_Phase1-2.pptx` | 10 页 PPT (中文) |
| `Phase1-2_Report_CN.md` | 中文进展报告 |
| `physics_demo_output/` | 标准化 physics-facing Rabi demo 输出目录 |
| `physics_demo_output/rabi_report.md` | Rabi 实验可读报告，包含 trace/state/memory/safety/dry-run 诊断 |
| `physics_demo_output/rabi_trace.json` | Rabi demo 原始 trace JSON，含完整 observations 和 artifacts |
| `physics_demo_output/rabi_oscillation.png` | Rabi 振荡图 (Nature 风格) |
| `physics_demo_output/rabi_oscillation.pdf` | 矢量版图 |
| `physics_demo_output/rabi_talk_track.txt` | 展示讲稿 |
| `eval_results.json` | 7×4 评测原始数据 |
| `run_all_demos.sh` | 一键运行所有 demo |

## 快速运行

```bash
cd /home/wangshuchang/quantumgpt
source .venv/bin/activate

# 运行所有 demo (约 3 分钟)
bash showcase/run_all_demos.sh

# 或单独运行:
PYTHONPATH=. python3 - <<'PY'               # Rabi demo
from demos.rabi_demo import run_rabi_demo
run_rabi_demo(save_dir='showcase/physics_demo_output')
PY
PYTHONPATH=. python3 eval/run_eval.py       # 评测
```

## PPT 内容 (10 slides)

1. 封面
2. 项目概览
3. 系统架构图
4. Phase 1 成果
5. Phase 2 成果
6. 评测结果表
7. Rabi Demo (含图)
8. 代码结构
9. 可运行命令
10. 下一步计划
