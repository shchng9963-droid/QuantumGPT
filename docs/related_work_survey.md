# Related Work Survey: LLM + Quantum Computing (2024–2026)

> Compiled for QuantumGPT project. Focus: identifying gaps for a **device-aware closed-loop quantum agent**.

---

## 1. Core Competing Papers

### 1.1 QUASAR (2025)

- **Title**: QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL
- **Authors**: Cong Yu, Valter Uotila, Shilong Deng, Qingyuan Wu, Tuo Shi, Songlin Jiang, Lei You, Bo Zhao
- **arXiv**: 2510.00967 (Oct 2025)
- **Venue**: Preprint

**Method**: Agentic RL framework for quantum circuit generation. Uses tool-augmented LLMs with:
  - External quantum simulator verification
  - Hierarchical reward mechanism in RL training
  - Fine-tunes a 4B LLM

**Key Results**: 99.31% validity (Pass@1), 100% (Pass@10), outperforms GPT-4o, GPT-5, DeepSeek-V3

**Abstraction Level**: Gate-level (OpenQASM generation)
**Closed-loop**: Partial — RL feedback loop during training, but no runtime closed-loop with hardware
**Real Hardware**: No — simulator only
**Limitations**:
  - Only generates circuits, does NOT operate/run them on hardware
  - No drift awareness, no hardware perception
  - No pulse-level or device-level capabilities
  - Training-time RL, not inference-time adaptation

**Gap for QuantumGPT**: QUASAR generates code but doesn't *operate* quantum hardware. It has zero awareness of hardware state, drift, or the need to adapt at runtime. QuantumGPT fills this by being a runtime agent that perceives and reacts to hardware conditions.

---

### 1.2 El Agente Quntur (2026)

- **Title**: El Agente Quntur: A research collaborator agent for quantum chemistry
- **Authors**: Juan B. Pérez-Sánchez, Yunheng Zou, Jorge A. Campos-Gonzalez-Angulo, Marcel Müller, Ignacio Gustin, et al.
- **arXiv**: 2602.04850 (Feb 2026)
- **Venue**: Preprint

**Method**: Hierarchical multi-agent AI system for computational quantum chemistry. Three design strategies:
  1. Reasoning-driven decisions (no hard-coded procedures)
  2. General/composable actions for generalization
  3. Guided deep research (integrates documentation + scientific literature)

Currently instantiated in ORCA 6.0 quantum chemistry package.

**Key Results**: Supports full range of ORCA 6.0 calculations; plans/executes/adapts/analyzes chemistry experiments

**Abstraction Level**: Quantum chemistry layer (above gate-level; works with molecular Hamiltonians)
**Closed-loop**: Partial — adapts based on computation results, but not hardware feedback
**Real Hardware**: No — classical simulation of quantum chemistry (not quantum computing hardware)
**Limitations**:
  - Operates on classical computers running quantum chemistry software
  - No interaction with actual quantum hardware (QPU)
  - No noise/drift/calibration awareness
  - Designed for chemistry, not hardware operations

**Gap for QuantumGPT**: Quntur is chemistry-focused and runs on classical computers. QuantumGPT targets the hardware operations layer — transpilation, execution, error mitigation, calibration under real noise/drift.

---

### 1.3 Agent-Q (2025)

- **Title**: Agent-Q: Fine-Tuning Large Language Models for Quantum Circuit Generation and Optimization
- **arXiv**: 2504.11109 (Apr 2025)
- **Venue**: Preprint

**Method**: LLM fine-tuning system for generating parameterized quantum circuits:
  - Generates training data (14K circuits: QAOA, VQE, adaptive VQE)
  - End-to-end pipeline to fine-tune LLMs
  - Produces OpenQASM 3.0 circuits with initial parameters

**Key Results**: Syntactically correct parameterized circuits; better parameters than random initialization

**Abstraction Level**: Gate-level (circuit generation)
**Closed-loop**: No — generates circuits offline, no runtime adaptation
**Real Hardware**: No
**Limitations**:
  - Only generates circuits, not executes them
  - No hardware awareness, no runtime adaptation
  - Parameters are "better than random" but not optimized on hardware

**Gap for QuantumGPT**: Agent-Q is a circuit generator; QuantumGPT is a circuit operator. Agent-Q's outputs could be consumed by QuantumGPT for execution.

---

### 1.4 QAgent (2025)

- **Title**: QAgent: An LLM-based Multi-Agent System for Autonomous OpenQASM Programming
- **arXiv**: 2508.20134 (Aug 2025)
- **Venue**: Preprint

**Method**: Multi-agent system for automating OpenQASM programming:
  - Task planning + in-context few-shot learning
  - RAG for long-term context
  - Predefined generation tools + CoT reasoning

**Key Results**: 71.6% improvement in QASM code generation accuracy vs static LLM approaches

**Abstraction Level**: Gate-level (code generation)
**Closed-loop**: No — generates code, does not execute or adapt
**Real Hardware**: No
**Limitations**:
  - Pure code generation system
  - No hardware interaction or runtime feedback
  - No noise/drift awareness

**Gap for QuantumGPT**: Same as QUASAR/Agent-Q — generates code but doesn't operate hardware.

---

### 1.5 AlphaQubit (2024)

- **Title**: (Google DeepMind) — ML-based quantum error correction decoder
- **Published**: Nature, 2024
- **Note**: The exact arXiv ID was not located in search; this is the Google DeepMind paper on neural network decoders for surface codes.

**Method**: Recurrent neural network for decoding quantum error correction syndromes. Trained on real hardware data from Google's superconducting processors.

**Abstraction Level**: QEC decoding (specialized subsystem)
**Closed-loop**: No — offline training, inference-time decoding only
**Real Hardware**: Yes — trained on real hardware syndrome data
**Limitations**:
  - Narrow scope: only QEC decoding
  - Not an agent system
  - No tool use, planning, or multi-step reasoning

**Gap for QuantumGPT**: AlphaQubit solves one specific problem (QEC decoding) very well. QuantumGPT is a general-purpose agent that could potentially use AlphaQubit as one of its tools.

---

## 2. Benchmarks

### 2.1 QCoder Benchmark (2025)

- **Title**: QCoder Benchmark: Bridging Language Generation and Quantum Hardware through Simulator-Based Feedback
- **arXiv**: 2510.26101 (Oct 2025)

**What it does**: Evaluates LLMs on quantum programming with feedback from simulated hardware. Supports domain-specific metrics (circuit depth, execution time, error classification).

**Key Results**: GPT-4o ~18.97% accuracy; o3 ~78%; human average 39.98%

**Relevance**: Validates that current LLMs struggle with quantum code without RL/fine-tuning. Motivates tool-augmented agent approaches.

---

### 2.2 QHackBench (2025)

- **Title**: QHackBench: Benchmarking Large Language Models for Quantum Code Generation Using PennyLane Hackathon Challenges
- **arXiv**: 2506.20008 (Jun 2025)

**What it does**: Benchmarks LLMs on PennyLane code generation using real QHack challenges. Evaluates vanilla prompting and RAG.

**Relevance**: Another data point showing LLMs need structured tooling to perform well on quantum tasks.

---

### 2.3 GenQC (2024)

- **Title**: Quantum circuit synthesis with diffusion models
- **arXiv**: 2311.02041
- **Venue**: Nature Machine Intelligence, 2024

**What it does**: Uses a diffusion model to synthesize quantum circuits from specifications.

**Relevance**: Generative model approach (not agentic). Different paradigm from tool-augmented LLM agents.

---

## 3. Landscape Summary

| Paper | Year | Level | Agent? | Closed-Loop? | Real HW? | What it does |
|---|---|---|---|---|---|---|
| **QUASAR** | 2025 | Gate | RL-agent | Train-time only | No | Generate circuits |
| **El Agente Quntur** | 2026 | Chemistry | Multi-agent | Partial (results) | No (classical) | Plan/run chemistry |
| **Agent-Q** | 2025 | Gate | SFT pipeline | No | No | Generate circuits |
| **QAgent** | 2025 | Gate | Multi-agent | No | No | Generate QASM code |
| **AlphaQubit** | 2024 | QEC | No (ML model) | No | Yes (training) | Decode syndromes |
| **QCoder** | 2025 | Gate | Benchmark | Sim feedback | No | Evaluate LLMs |
| **GenQC** | 2024 | Gate | No (diffusion) | No | No | Synthesize circuits |
| **QuantumGPT (ours)** | 2026 | **Gate + Pulse** | **Yes** | **Yes (runtime)** | **Shadow + replay** | **Operate hardware** |

---

## 4. Identified Gaps (QuantumGPT's Unique Position)

### Gap 1: No existing work operates quantum hardware at runtime
All current LLM-quantum works either:
- Generate circuits offline (QUASAR, Agent-Q, QAgent, GenQC)
- Run on classical computers (El Agente Quntur)
- Solve a narrow ML problem (AlphaQubit)

**QuantumGPT is the first to actually operate a QPU (even simulated) with runtime closed-loop feedback.**

### Gap 2: No drift-awareness
No existing work monitors hardware drift and adapts execution plans. This is because none of them interact with hardware at runtime. QuantumGPT's drift detector + replanning is completely novel in this space.

### Gap 3: No cross-level (gate + pulse) agent
All existing LLM-quantum work stays at the gate/QASM level. Nobody has built an agent that can drop down to pulse-level when gate-level doesn't suffice. QuantumGPT's Rabi tool demonstrates this capability.

### Gap 4: No quantum-agent benchmark with hardware realism
QCoder and QHackBench evaluate code generation quality. Nobody evaluates whether an agent can successfully *operate* hardware under realistic conditions (drift, failure, noise). QC-Agent-Bench fills this gap.

### Gap 5: No ExperimentRecord / episodic memory for quantum ops
No existing system maintains structured memory of past quantum experiments to inform future decisions.

---

## 5. Paper Positioning Statement

> While recent works (QUASAR, Agent-Q, QAgent) have demonstrated that LLMs can *generate* quantum circuits with increasing quality, **no existing system closes the loop between circuit generation and hardware execution under realistic operating conditions**. QuantumGPT is the first device-aware, closed-loop agent that perceives hardware state (including drift), dispatches tools across gate and pulse levels, maintains structured experiment memory, and adapts execution plans in real time. We complement this with QC-Agent-Bench — the first benchmark evaluating quantum agents under drift and failure conditions.

---

*Last updated: 2026-05-15*
*Sources: arXiv search, paper abstracts*
