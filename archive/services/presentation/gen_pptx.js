const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Wang Shuchang";
pres.title = "QuantumGPT: LLM-Driven Autonomous Quantum Computing Agent";

// Color palette — Ocean/Quantum theme
const C = {
  navy: "1B2838",
  deepBlue: "0D47A1",
  teal: "00838F",
  accent: "00BCD4",
  lightBg: "F5F9FC",
  white: "FFFFFF",
  darkText: "1E293B",
  mutedText: "64748B",
  green: "2E7D32",
  orange: "E65100",
  purple: "6A1B9A",
};

// ═══════════════════════════════════════════════════════
// Slide 1: Title
// ═══════════════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.navy };
s1.addText("QuantumGPT", {
  x: 0.8, y: 1.2, w: 8.5, h: 1.2,
  fontSize: 48, fontFace: "Arial Black", color: C.accent, bold: true,
});
s1.addText("LLM-Driven Autonomous Quantum Computing Agent", {
  x: 0.8, y: 2.4, w: 8.5, h: 0.8,
  fontSize: 22, fontFace: "Arial", color: C.white,
});
s1.addText("Phase 1-2 Progress Report", {
  x: 0.8, y: 3.4, w: 8.5, h: 0.6,
  fontSize: 16, fontFace: "Arial", color: C.mutedText, italic: true,
});
s1.addText("Wang Shuchang | 2026.05", {
  x: 0.8, y: 4.6, w: 8.5, h: 0.5,
  fontSize: 14, fontFace: "Arial", color: C.mutedText,
});

// ═══════════════════════════════════════════════════════
// Slide 2: Project Overview
// ═══════════════════════════════════════════════════════
let s2 = pres.addSlide();
s2.background = { color: C.lightBg };
s2.addText("项目概览", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});
s2.addText([
  { text: "目标: ", options: { bold: true } },
  { text: "��建一个 LLM 驱动的自主量子计算 Agent，能够:", options: {} },
], { x: 0.5, y: 1.3, w: 9, h: 0.5, fontSize: 15, color: C.darkText });

s2.addText([
  { text: "监控量子设备健康状态 (T1/T2, 门错误率, 漂移)", options: { bullet: true, breakLine: true } },
  { text: "自主运行和优化量子电路", options: { bullet: true, breakLine: true } },
  { text: "应用误差缓解 (ZNE) 提升保真度", options: { bullet: true, breakLine: true } },
  { text: "执行脉冲级实验 (Rabi 振荡) 进行量子比特表征", options: { bullet: true, breakLine: true } },
  { text: "基于 ReAct 推理框架做出自适应决策", options: { bullet: true } },
], { x: 0.7, y: 1.9, w: 8.5, h: 2.5, fontSize: 14, color: C.darkText, paraSpaceAfter: 6 });

s2.addText("技术栈: Qiskit + AerSimulator + DeepSeek/Claude API + ReAct Agent", {
  x: 0.5, y: 4.6, w: 9, h: 0.5, fontSize: 12, color: C.mutedText, italic: true,
});

// ═══════════════════════════════════════════════════════
// Slide 3: Architecture
// ═══════════════════════════════════════════════════════
let s3 = pres.addSlide();
s3.background = { color: C.white };
s3.addText("系统架构", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});

// Architecture boxes
const boxes = [
  { label: "User Prompt", x: 0.5, y: 1.5, color: C.purple },
  { label: "ReAct Agent\n(Thought→Action→Obs)", x: 2.8, y: 1.5, color: C.deepBlue },
  { label: "14 Quantum Tools", x: 5.5, y: 1.5, color: C.teal },
  { label: "Shadow Backend\n(FakeBrisbane 127q)", x: 8.0, y: 1.5, color: C.green },
];
boxes.forEach(b => {
  s3.addShape(pres.shapes.RECTANGLE, {
    x: b.x, y: b.y, w: 2.0, h: 1.0,
    fill: { color: b.color }, shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
  });
  s3.addText(b.label, {
    x: b.x, y: b.y, w: 2.0, h: 1.0,
    fontSize: 10, color: C.white, align: "center", valign: "middle", bold: true,
  });
});

// Arrows
[3.5, 5.8, 8.2].forEach((_, i) => {
  s3.addText("→", {
    x: 2.4 + i * 2.5, y: 1.75, w: 0.5, h: 0.5,
    fontSize: 20, color: C.mutedText, align: "center", valign: "middle",
  });
});

// Lower section: Fidelity Budget
s3.addShape(pres.shapes.RECTANGLE, {
  x: 2.8, y: 3.0, w: 4.5, h: 0.8,
  fill: { color: C.orange },
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
});
s3.addText("Fidelity Budget Controller\n(目标保真度 → 自动缓解/停止)", {
  x: 2.8, y: 3.0, w: 4.5, h: 0.8,
  fontSize: 10, color: C.white, align: "center", valign: "middle", bold: true,
});
s3.addText("↑ 反馈", {
  x: 4.7, y: 2.55, w: 1.0, h: 0.5,
  fontSize: 10, color: C.mutedText, align: "center",
});

// Tool list
s3.addText("14 Tools: get_backend_health, run_circuit, transpile_circuit,\napply_mitigation (ZNE), predict_fidelity, rabi_experiment, fit_rabi,\nget_qubit_properties, get_coupling_map, detect_drift, ...", {
  x: 0.5, y: 4.0, w: 9, h: 1.2, fontSize: 11, color: C.mutedText, fontFace: "Consolas",
});

// ═══════════════════════════════════════════════════════
// Slide 4: Phase 1 Results
// ═══════════════════════════════════════════════════════
let s4 = pres.addSlide();
s4.background = { color: C.lightBg };
s4.addText("Phase 1: 基础设施", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s4.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});

const p1Items = [
  ["Shadow Backend", "FakeBrisbane 127-qubit 模拟器 + 噪声模型 + 漂移注入"],
  ["9 基础工具", "health, qubit_props, coupling_map, drift, calibration_age, compare, run_circuit, list, diagnose"],
  ["Rabi 脉冲模拟", "Schrödinger 方程数值求解, square/gaussian 脉冲, 30点扫描"],
  ["Agent Loop v1", "While-loop + RulePlanner + OpenAI/Anthropic/DeepSeek API"],
  ["数据层", "DuckDB 存储 + InstrumentedExecutor + W&B 集成"],
];

p1Items.forEach((item, i) => {
  const y = 1.3 + i * 0.75;
  s4.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 0.06, h: 0.55, fill: { color: C.teal },
  });
  s4.addText(item[0], {
    x: 0.75, y: y, w: 2.5, h: 0.55,
    fontSize: 13, color: C.darkText, bold: true, valign: "middle",
  });
  s4.addText(item[1], {
    x: 3.2, y: y, w: 6.5, h: 0.55,
    fontSize: 12, color: C.mutedText, valign: "middle",
  });
});

// ════════��══════════════════════════════════════════════
// Slide 5: Phase 2 Results
// ═══════════════════════════════════════════════════════
let s5 = pres.addSlide();
s5.background = { color: C.lightBg };
s5.addText("Phase 2: Agent 升级 + 评测", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s5.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});

const p2Items = [
  ["5 新工具 → 14 总计", "transpile, apply_mitigation(ZNE), predict_fidelity, rabi_experiment, fit_rabi"],
  ["ReAct Agent", "Thought→Action→Observation 结构化推理, 支持多 provider"],
  ["Fidelity Budget", "目标保真度跟踪, 自动触发缓解/停止, 预算注入 prompt"],
  ["4 系统对照", "Static Pipeline / LLM Single-Shot / ReAct(no budget) / ReAct+Budget"],
  ["7×4 评测", "100% 成功率(智能系统), 86% static baseline — 符合预期"],
  ["Rabi Demo", "端到端: prompt → pulse sim → fit → π=52.52 MHz, R²=1.0"],
];

p2Items.forEach((item, i) => {
  const y = 1.2 + i * 0.65;
  s5.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 0.06, h: 0.5, fill: { color: C.accent },
  });
  s5.addText(item[0], {
    x: 0.75, y: y, w: 2.8, h: 0.5,
    fontSize: 12, color: C.darkText, bold: true, valign: "middle",
  });
  s5.addText(item[1], {
    x: 3.5, y: y, w: 6.2, h: 0.5,
    fontSize: 11, color: C.mutedText, valign: "middle",
  });
});

// ═══════════════════════════════════════════════════════
// Slide 6: Evaluation Table
// ═══════════════════════════════════════════════════════
let s6 = pres.addSlide();
s6.background = { color: C.white };
s6.addText("评测结果: 7 Tasks × 4 Systems", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s6.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.9, w: 2.0, h: 0.04, fill: { color: C.accent },
});

const tableHeader = [
  { text: "Task", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } },
  { text: "Static", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } },
  { text: "Single-Shot", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } },
  { text: "ReAct", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } },
  { text: "ReAct+Budget", options: { bold: true, fill: { color: C.navy }, color: C.white, fontSize: 10 } },
];

const tableRows = [
  ["Health+GHZ-5", "100%", "100%", "100%", "100%"],
  ["Multi-Circuit", "100%", "100%", "100%", "100%"],
  ["Diagnose+Mitigate", "0%", "100%", "100%", "100%"],
  ["Fidelity Predict", "100%", "100%", "100%", "100%"],
  ["Transpile+Run", "100%", "100%", "100%", "100%"],
  ["Rabi Experiment", "100%", "100%", "100%", "100%"],
  ["Full Pipeline", "100%", "100%", "100%", "100%"],
  ["Overall", "86%", "100%", "100%", "100%"],
];

const tData = [tableHeader];
tableRows.forEach((row, i) => {
  const isLast = i === tableRows.length - 1;
  const rowFill = isLast ? { color: "E8F5E9" } : (i % 2 === 0 ? { color: "F8FAFC" } : { color: C.white });
  tData.push(row.map((cell, j) => ({
    text: cell,
    options: {
      fontSize: 10,
      fill: rowFill,
      bold: isLast || j === 0,
      color: cell === "0%" ? "D32F2F" : C.darkText,
    },
  })));
});

s6.addTable(tData, {
  x: 0.5, y: 1.2, w: 9.0,
  colW: [2.2, 1.5, 1.8, 1.5, 2.0],
  border: { type: "solid", pt: 0.5, color: "E0E0E0" },
  rowH: [0.35, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.38],
});

s6.addText("Static Pipeline 缺乏推理能力，无法执行诊断+缓解任务 → 验证了 Agent 推理的必要性", {
  x: 0.5, y: 4.8, w: 9, h: 0.5, fontSize: 12, color: C.orange, italic: true,
});

// ═══════════════════════════════════════════════════════
// Slide 7: Rabi Demo
// ═══════════════════════════════════════════════════════
let s7 = pres.addSlide();
s7.background = { color: C.white };
s7.addText("Rabi 振荡端到端 Demo", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 28, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s7.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.9, w: 2.0, h: 0.04, fill: { color: C.accent },
});

// Left: workflow
s7.addText([
  { text: "Agent 工作流:\n", options: { bold: true, breakLine: true } },
  { text: "1. 检查后端健康状态\n", options: { breakLine: true } },
  { text: "2. 运行 Rabi 振荡实验\n", options: { breakLine: true } },
  { text: "   (5 GHz, gaussian, 100ns)\n", options: { breakLine: true } },
  { text: "3. 拟合 sin² 模型\n", options: { breakLine: true } },
  { text: "4. 报告 π-pulse 幅度\n", options: { breakLine: true } },
  { text: "\n", options: { breakLine: true } },
  { text: "结果:\n", options: { bold: true, breakLine: true } },
  { text: "  π-pulse = 52.52 MHz\n", options: { breakLine: true } },
  { text: "  π/2-pulse = 26.26 MHz\n", options: { breakLine: true } },
  { text: "  R² = 1.0000\n", options: { breakLine: true } },
  { text: "  4 tool calls, 11.3s", options: {} },
], { x: 0.5, y: 1.2, w: 4.0, h: 4.0, fontSize: 12, color: C.darkText, valign: "top" });

// Right: figure placeholder
s7.addImage({
  path: path.resolve(__dirname, "../demos/output/rabi_oscillation.png"),
  x: 4.8, y: 1.2, w: 4.8, h: 3.6,
});

// ═══════════════════════════════════════════════════════
// Slide 8: Code Structure
// ═══════════════════════════════════════════════════════
let s8 = pres.addSlide();
s8.background = { color: C.lightBg };
s8.addText("代码结构", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s8.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});

s8.addText(
`quantumgpt/
├── agent/
│   ├── loop.py          # 基础 while-loop agent
│   ├── react.py         # ReAct agent + 结构化 trace
│   ├── budget.py        # Fidelity budget controller
│   └── baselines.py     # 4 系统对照实现
├── backends/
│   ├── base.py          # ShadowBackend 抽象基类
│   └── fake_adapter.py  # FakeBrisbane 127q 适配器
├── tools/
│   └── quantum_tools.py # 14 个量子工具定义+执行
├── dynamics/
│   └── rabi.py          # Rabi 脉冲模拟 (Schrödinger)
├── mitigation/
│   └── __init__.py      # ZNE 误差缓解
├── eval/
│   └── run_eval.py      # 7×4 评测框架
└── demos/
    └── rabi_demo.py     # 端到端 Rabi demo`,
  { x: 0.5, y: 1.2, w: 9, h: 4.0, fontSize: 11, fontFace: "Consolas", color: C.darkText, valign: "top" }
);

// ═══════════════════════════════════════════════════════
// Slide 9: Demo Commands
// ═══════════════════════════════════════════════════════
let s9 = pres.addSlide();
s9.background = { color: C.navy };
s9.addText("可运行 Demo", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.accent, bold: true, margin: 0,
});

s9.addText(
`# 1. Rabi 端到端 demo
PYTHONPATH=. python3 demos/rabi_demo.py

# 2. 7×4 系统评测
PYTHONPATH=. python3 eval/run_eval.py

# 3. ReAct Agent 交互
PYTHONPATH=. python3 -c "
from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter
be = FakeBackendAdapter('FakeBrisbane')
agent = ReActAgent(be, provider='mock')
trace = agent.run('Run GHZ-5 and apply mitigation')
print(trace.final_answer)
"

# 4. 单工具测试
PYTHONPATH=. python3 -c "
from tools.quantum_tools import ToolExecutor
from backends.fake_adapter import FakeBackendAdapter
ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
print(ex.execute('get_backend_health', {}))
"`,
  { x: 0.5, y: 1.1, w: 9, h: 4.2, fontSize: 11, fontFace: "Consolas", color: "B2DFDB", valign: "top" }
);

// ═══════════════════════════════════════════════════════
// Slide 10: Next Steps
// ═══════════════════════════════════════════════════════
let s10 = pres.addSlide();
s10.background = { color: C.lightBg };
s10.addText("下一步计划", {
  x: 0.5, y: 0.3, w: 9, h: 0.7,
  fontSize: 32, fontFace: "Arial", color: C.navy, bold: true, margin: 0,
});
s10.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.95, w: 2.0, h: 0.04, fill: { color: C.accent },
});

s10.addText([
  { text: "Phase 3: 真实 LLM 集成 + 论文实验\n\n", options: { bold: true, breakLine: true } },
  { text: "接入 DeepSeek API 运行完整 ReAct loop (非 mock)", options: { bullet: true, breakLine: true } },
  { text: "对比 LLM 推理 vs Rule-based 在复杂任务上的差异", options: { bullet: true, breakLine: true } },
  { text: "增加更多量子算法 benchmark (VQE, QAOA 变分)", options: { bullet: true, breakLine: true } },
  { text: "实现 T1/T2 衰减实验 + Ramsey 干涉", options: { bullet: true, breakLine: true } },
  { text: "论文写作: 系统设计 + 实验结果 + 消融分析", options: { bullet: true, breakLine: true } },
  { text: "目标: ICLR 2027 投稿 (~2026-10-03)", options: { bullet: true } },
], { x: 0.7, y: 1.3, w: 8.5, h: 3.5, fontSize: 14, color: C.darkText, paraSpaceAfter: 8 });

// Write file
const outPath = path.resolve(__dirname, "QuantumGPT_Phase1-2.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPT saved to: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
