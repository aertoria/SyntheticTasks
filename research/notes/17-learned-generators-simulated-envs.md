# Learned task generators, LLM-simulated environments and agent-environment co-evolution, including the DeepSeek-V4.1-Flash update to the frontier-lab picture

*Scope: this note covers five things. (1) Task or environment generators that are trained or adapted on the learner's own success statistics. (2) LLM "world models" that stand in for executable environments. (3) Loops where the agent and its environment co-evolve. (4) Self-play for open-ended tasks without a verifier. (5) The 2026 frontier reports DeepSeek-V4.1-Flash, DSec, Kimi K3, Nemotron-IMO and SU-01. Compiled 2026-09-30. Verification: the researcher supplied 30 entries covering 43 papers. I checked each against the arXiv full text (HTML), corrected 6, dropped 0 and added 3, for 46 papers in total.*

## TL;DR

- **The frontier picture changed in September 2026. DeepSeek-V4.1-Flash (arXiv 2609.19969, §5.1.1) describes training the model itself as a task constructor.** Each task is a triplet (problem, environment, verification system), scored for *difficulty* (the task is non-trivial) and *correctness* (none of the three parts has a critical flaw). "Using difficulty and correctness as reward signals, we iteratively train the model to construct better tasks." Every new RL run re-audits the tasks it used. Coding environments are built by a multi-agent pipeline: build → self-test → leak-scrub → several distinct solver agents → an independent inspector → a repair agent that "adjusts evaluation points that are too easy or too difficult". The report says data and environment pipelines explain "essentially all of the observed gains" under an unremarkable SFT → RL → OPD recipe. It gives no reward formula and no ablation. Note 12's statement that frontier reports contain no synthesis detail is true of DeepSeek-V4 (2606.19348) and false for V4.1-Flash.
- **The standard academic recipe is a generator steered by the current policy's pass rate, targeting about 50%.** GenEnv uses α = 0.5 with reward-weighted regression. STRETCH peaks its GRPO reward at 50%; a 0.5 target beat 0.2 and 0.8. SCOPE targets τ = 0.5 and discards tasks outside [0.2, 0.8]. SEAD samples profiles with weight ∝ 1 − |CR − 0.5|. DreamGym selects seeds by group reward variance. UI-Simulator-Grow uses the 25th–75th percentile of teacher-forcing loss. **The cheapest version to adopt** is RLAnything's rule: when accuracy exceeds 0.8, ask an LLM for a harder rewrite and accept it only if 0.2 < acc(q′) < acc(q).
- **A learned component may choose tasks or initial states, but it must never be the grader.** SEAD's user simulator, once trained, "force[d] success rates around 50% by directly accepting or rejecting regardless of agent performance". SWE-World's reward model without chain-of-thought (CoT) was hacked: trajectory length collapsed after about step 20 as short invalid patches were scored correct. LSP's solver hacked an off-the-shelf reward model by answering in Python. Qwen-AgentWorld's world model learned to add "self-praise" phrases to fool its judge.
- **LLM simulators are too agreeable, and they fail exactly where complexification pushes.** EnvSimBench finds a universal *state-change cliff*: near-perfect accuracy when state does not change, collapse when several variables must update at once. WebWorld reports sycophantic, "overly optimistic outcomes". LLM user simulators cut non-buyers' expressed resistance from 25.1% to 13.5%. The rule: keep state in code or a database, and let the LLM render only surface text or user behaviour.
- **A simulator helps harden tasks only when you steer it.** In Qwen-AgentWorld, uncontrolled simulated RL gave no gain (Tool Decathlon went from 32.4 to 31.5). Instructed perturbations (intermittent errors, pagination, partial results, withheld answers) gave +3.7 on Tool Decathlon and +12.3 on MCPMark. Fictional SQL-backed worlds gave +16.29 WideSearch item-F1. A dedicated, trained simulator beat a prompted general model, which produced only marginal gains.
- **Many methods here make tasks *easier*. Teams with saturated tasks should run them in reverse.** Examples are SEAL's interface cues, CuES's hint ladder, Environment Tuning's corrective hints and Simia's explanatory errors. Reversed operators: strip the hints, return terse errors (Environment Tuning's final stage does exactly this), delete the answer's source document (LiteResearcher), move details into a hidden user profile (SynthAgent), or invert labels in the observations (AutoEnv). None of these touches the verifier.
- **Explore-then-question pipelines guarantee solvability because the solution was witnessed during exploration.** This covers AgentEvolver, CuES, CoEvolve, WebWorld's Abstract-and-Instantiate and UI-Simulator. The catch is that they are limited to what the explorer can already do. Seed hardening with CoEvolve's *forgetting* and *boundary* signals, then compose or strip hints, and re-verify by replaying the witnessed trajectories.
- **Self-play for open-ended tasks gives real but modest gains and needs stabilisers.** G-Zero raises the average from 33.95 to 35.43. MAE gains 4.54% on a 3B model. R-Few gains +3.0 over R-Zero. SCOPE gains up to +10.4. The stabilisers are golden replay (STRETCH collapses without it), 1–5% human anchors (R-Few), quality gates (MAE's floor of 0.7; SCOPE's source-relevance gate), KL anchors, and keeping only lower-half-δ pairs (G-Zero). The strongest signals come from *hidden grounding* rather than judge scores: SCOPE's source document that only the judge sees, and SpyRL's known spy identity.
- **Other 2026 frontier reports still select hard human problems rather than synthesise new ones.** Nemotron-IMO keeps problems solved in 1–3 of 4 attempts (9,597). SU-01 rejects prompts that are too easy or too hard. Kimi K3 makes tasks hard through environment design: randomised agent harnesses, multi-day mock-application event streams, and Autonomous Execution Tasks that pair public verifiers with hidden verifiers under a submission budget. The DSec paper documents concrete exploits and the defences against them.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash task constructor + DSec | 2026 | [2609.19969](https://arxiv.org/abs/2609.19969), [2609.22978](https://arxiv.org/abs/2609.22978) | General agents (mocked SaaS/enterprise), repo-level coding | SFT+RL | Constructor trained on difficulty+correctness rewards; failure replay; complex implementation directions; eval-point re-tuning; multi-scaffold RL | Correctness reward; F2P/P2P tests; multiple solvers; inspector reads trajectories; leak scrub; builder/runtime account split; AppArmor + eBPF |
| Kimi K3 §4.2 | 2026 | [2607.24653](https://arxiv.org/abs/2607.24653) | Knowledge work, search, kernels, PA workflows, AET, webdev | SFT+RL | KG-guided concept composition; harness randomization; multi-day event streams; AET | Final-state verifiers; public+hidden verifiers with submission budgets; kernel hacking detector; webdev fake-artifact zeroing |
| Nemotron-IMO recipe + SU-01 | 2026 | [2609.10712](https://arxiv.org/abs/2609.10712), [2605.13301](https://arxiv.org/abs/2605.13301) | Olympiad math/physics | SFT+RL | Selection only: 1–3/4 pass band; too-easy/too-hard rejection; reverse-perplexity SFT order | Proof verifier + meta-verifier traces; human-sourced problems |
| EnterpriseBench CoreCraft | 2026 | [2602.16179](https://arxiv.org/abs/2602.16179) | Enterprise customer-support tools (MCP) | RL | Expert-authored tasks from lookups to policy-constrained multi-step workflows | Expert rubrics of verifiable assertions, LLM-judged; pass = all criteria |
| DreamGym | 2025 | [2511.03773](https://arxiv.org/abs/2511.03773) | WebShop, ALFWorld, WebArena-Lite | RL | Reward-variance seed selection → harder variations; bounded synthetic fraction | Learned experience model gives rewards; sim-to-real check |
| GenEnv | 2025 | [2512.19682](https://arxiv.org/abs/2512.19682) | Tool use, embodied, QA, planning | RL | Trained environment policy; α=0.5 bell reward via RWR | Exact execution vs simulator target (structured) / soft similarity; validity filter |
| RLAnything | 2026 | [2602.02488](https://arxiv.org/abs/2602.02488) | OSWorld, ALFWorld, coding | RL | Critic-guided harder/easier rewrite; hints; task-template switch; co-generated unit tests | Directional acceptance α_low < acc(q′) < acc(q); pre-built verifier files; UT reward vs gt code |
| STRETCH | 2026 | [2609.18642](https://arxiv.org/abs/2609.18642) | Negotiation, operations research | RL | Scaffolder adds constraints to a base context; 50%-peaked reward | OR: code execution + optimality, infeasible = 0; negotiation: LLM judge; golden replay |
| SEAD | 2026 | [2602.03548](https://arxiv.org/abs/2602.03548) | Service dialogue with simulated users | RL | Statistics-driven sampling of user initial states at CR≈0.5 | Outcome decided by an untrained role-play simulator |
| COvolve | 2026 | [2603.28386](https://arxiv.org/abs/2603.28386) | Code-as-env/policy: MiniGrid, 2D nav, CARLA | analysis (program synthesis) | Best-response env mutation vs MSNE policy mixture | Reachability/feasibility checks; execution-measured payoff |
| From Trainee to Trainer (+EnvGen, SimWorld Studio) | 2026 (2024) | [2606.17682](https://arxiv.org/abs/2606.17682), [2403.12014](https://arxiv.org/abs/2403.12014), [2605.09423](https://arxiv.org/abs/2605.09423) | MAPF grids; Crafter/Heist; UE5 navigation | RL | Policy writes next-stage generator config from failure breakdown | Procedural generators with exact solvers (CBS), game engines, NavMesh + physics/VLM checks |
| SEAL | 2026 | [2605.24426](https://arxiv.org/abs/2605.24426) | Multi-turn tool use (BFCL) | RL | Diagnosis-driven interface cues (typically makes easier; reversible) | Tool backend, labels, rewards unchanged |
| **Added:** Environment Tuning | 2025 | [2510.10197](https://arxiv.org/abs/2510.10197) | BFCL multi-turn tool use | RL | 4-stage curriculum; corrective hints on failure, removed in final stage | Original BFCL state/response checks; progress reward |
| CoEvolve | 2026 | [2604.15840](https://arxiv.org/abs/2604.15840) | AppWorld, BFCL | RL | Forgetting/boundary/rare signals → LLM re-exploration → new tasks | Execution in real environment (with a positive-reward loophole) |
| AgentEvolver (self-questioning) | 2025 | [2511.10395](https://arxiv.org/abs/2511.10395) | AppWorld, BFCL | RL | Explore-then-question; scale #entities/attributes/operations | Reference solution from exploration trace; feasibility re-execution; reference-based judge |
| CuES | 2025 | [2512.01311](https://arxiv.org/abs/2512.01311) | AppWorld, BFCL, WebShop | RL | Memory-tree curiosity exploration; hint-depth rewrite ladder | Judge agent re-executes; accept only reward = 1.0 + path faithfulness |
| **Added:** UI-Simulator / -Grow | 2025 | [2510.14969](https://arxiv.org/abs/2510.14969) | WebArena, AndroidWorld | SFT | LLM UI simulator; target tasks at 25–75th pct teacher-forcing loss; content-preserving variants | Teacher-produced trajectories; retrieval-grounded simulation |
| Simia | 2025 | [2511.01824](https://arxiv.org/abs/2511.01824) | τ²-Bench, OfficeBench, AgentBench | SFT+RL | Seed amplification by full-trajectory simulation | Tool-spec anchoring, JSON/tool-name repair, LLM checks (no state verifier) |
| WebWorld | 2026 | [2602.14721](https://arxiv.org/abs/2602.14721) | Open-web agents | SFT | Abstract-and-Instantiate task synthesis in a trained world model | Rejection sampling on success inside the world model |
| Qwen-AgentWorld | 2026 | [2606.24597](https://arxiv.org/abs/2606.24597) | 7 domains; OpenClaw, MCP, search | SFT+RL | Instructed perturbations; answer-withholding snippets; fictional SQL worlds; OpenClaw seed expansion | SQL-derived answers; unsearchability checks; rubric verifiers; rule anchor in WM reward |
| **Added:** LiteResearcher | 2026 | [2604.17931](https://arxiv.org/abs/2604.17931) | Deep-research search agents | RL | Source masking (delete answer's page); corpus co-expansion; 1–7/8 pass band | 7-criterion rubric filter; local search engine over real pages |
| SWE-World | 2026 | [2602.03419](https://arxiv.org/abs/2602.03419) | SWE-bench-style repos | SFT+RL | Docker-free: learned transition + reward models | Real file edits; SWR trained on Docker test outcomes; git log/show disabled |
| MDLM text world models | 2026 | [2607.16204](https://arxiv.org/abs/2607.16204) | Text agent envs (AppWorld, ScienceWorld, ALFWorld) | RL | Steering directives; anchor-aware denoising for diverse transitions | Deterministic state checks; 100-state human study |
| EnvACE (+WebEvolver, CoMAP, PaW) | 2026 (2025) | [2608.06197](https://arxiv.org/abs/2608.06197) | Tool use, web | RL | Policy rehearses its own environment responses / co-trained world models | Outcome evaluator or checklist judge only |
| SynthAgent (Mock Worlds, Real Skills) | 2026 | [2601.22511](https://arxiv.org/abs/2601.22511) | Tool use, math/search | RL | Underspecified instruction + private user context; per-task virtual tools; forbidden behaviours | Subgoal + required-interaction rubric, zeroed by forbidden actions; per-task call map |
| EnvFactory | 2026 | [2605.18703](https://arxiv.org/abs/2605.18703) | MCP tool use | SFT+RL | Topology-aware tool chains; implicit reference, action compression, ambiguity, goal expansion | Unit-tested executable DB environments; trajectory + state reward |
| AutoEnv | 2025 | [2511.19304](https://arxiv.org/abs/2511.19304) | Text/visual game-like envs | eval | Observation/skin layers; inverse semantics; level generator | Execution tests, level validators, weak-vs-strong differential testing |
| EnvSimBench + Simulated Customers | 2026 | [2605.07247](https://arxiv.org/abs/2605.07247), [2606.20708](https://arxiv.org/abs/2606.20708) | LLM env/user simulators | analysis | Difficulty = number of simultaneous state changes | Programmatic transition labels; real purchase outcomes |
| G-Zero | 2026 | [2605.09959](https://arxiv.org/abs/2605.09959) | Open-ended generation | RL (proposer) + DPO | Proposer maximizes Hint-δ (hint-induced log-prob shift) | Verifier-free; lower-half-δ + length/compression filters |
| Multi-Agent Evolve + Language Self-Play | 2025 | [2510.23595](https://arxiv.org/abs/2510.23595), [2509.07414](https://arxiv.org/abs/2509.07414) | General reasoning; instruction following | RL | Difficulty reward 1 − judged solve score; challenger advantage V − V(q) | Judge ≥ 0.7 quality gate; KL; reference-model quality self-reward |
| R-Few | 2025 | [2512.02472](https://arxiv.org/abs/2512.02472) | Math, general reasoning | RL | Anchor-conditioned challenger; mid-uncertainty quantile curriculum | Majority-vote pseudo-labels + 1–5% human anchors |
| SCOPE | 2026 | [2605.31433](https://arxiv.org/abs/2605.31433) | Open-ended research QA | RL | Document-grounded tasks needing multi-turn retrieval; τ=0.5, window [0.2, 0.8] | Frozen self-judge writes rubrics from hidden source doc; quality gates; cosine length penalty |
| SpyRL / RLSVR | 2026 | [2607.23802](https://arxiv.org/abs/2607.23802) | Summarization, creative writing, math | RL | Latent-variable injection (masked span to one "spy") | Detection reward exact vs environment-assigned spy index |

## Method notes

### DeepSeek-V4.1-Flash task constructor + DSec — DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (§5.1) and DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale (DeepSeek-AI / Xu et al., 2026; Huang et al., 2026)
Link: https://arxiv.org/abs/2609.19969 (17 Sep 2026); https://arxiv.org/abs/2609.22978 (19 Sep 2026)
- **Mechanism**:
  - *Premise.* "We observe that the model is already beginning to exhibit the ability to construct its own training tasks, though this capability remains far from perfect."
  - *Task formalism and rewards.* Each task is a triplet (problem, environment, verification system). Quality is judged on *difficulty*, "ensuring the task is non-trivial", and *correctness*, "guaranteeing that no critical flaws exist among the three components". Both are used as rewards to "iteratively train the model to construct better tasks".
  - *Lifecycle monitoring.* "Whenever a task is used in a new RL run, the resulting trajectories provide fresh evidence for quality re-auditing."
  - *General agents.* Employees and partners voluntarily return interaction data. From the interfaces seen in that data, DeepSeek builds mocked tools that reproduce "input formats, output structures, API schemas, and behavioral constraints" of SaaS, enterprise and back-end systems. Negative feedback and failure cases submitted by employees are rebuilt (tool context, user interaction patterns, failure conditions) into single- and multi-turn environments, enabling "systematic replay of failures and targeted reinforcement learning against observed model weaknesses".
  - *Coding agents, sources.* Internal and partner coding-agent sessions, filtered to "highly complex tasks or tasks on which model performance is poor" and deduplicated by trajectory, plus GitHub repositories above a star threshold.
  - *Coding agents, pipeline.* (1) An agent checks that the project builds, runs fully in a container and can be verified automatically. It picks a turn or commit as the start, designs "several sufficiently complex implementation directions", and writes fail-to-pass (F2P) and pass-to-pass (P2P) evaluation points plus a construction report, fetching web resources as needed. (2) A second agent sets up dependencies, working directory, tests and task description in an isolated container, self-tests, "removes any traces that could leak the task solution", and packages a new image layer. (3) Several distinct agents attempt the task. (4) An independent quality-inspection agent reviews the environment *together with the solvers' trajectories*, checking environment issues, factual errors, mismatches between evaluation points and the description, and hackability. (5) If inspection fails, a repair agent fixes the errors, "adjusts evaluation points that are too easy or too difficult", and the task is verified again.
  - *RL (§5.1.2).* Asynchronous RL is scaled along two axes: compute and number of scaffolds (several Claude Code versions; OpenCode, Pi, DeepSeek Harness in Standard and PTC modes). A worker container normalises heterogeneous scaffold interactions into one trajectory schema. Checkpoints from runs on different scaffolds are *merged* to initialise the next RL run.
  - *Environment building in DSec §6.1.* The paper calls it environments "of Agents, by Agents, for Agents". `pack_diff` snapshots an interactive sandbox into an incremental disk image that becomes a reusable environment. Builder and runtime agents use separate accounts. Build-time residue is removed from the writable layer so "reference answers are not carried into the resulting image".
- **How it makes tasks harder**: A learned difficulty reward on the constructor. Sources pre-filtered to complex or poorly solved sessions. Several complex implementation directions per repository. The repair agent re-tunes F2P points that are too easy. Replay of observed failures. Scaffold diversity shifts the interface, so a task solved in one harness can still be hard in another.
- **Correctness / verification**: Correctness is itself a constructor reward. F2P and P2P tests. Several independent solvers. An inspector that reads trajectories, which catches hackable tasks that output-only checks miss. Leak scrubbing and separate builder/runtime accounts. DSec adds per-sandbox AppArmor profiles (file read/write, sockets, the command daemon's logs) and per-task eBPF allowlists by IP, port and protocol. If an agent crashes its environment, the trajectory counts as failed and a "repercussion" signal goes to the RL framework.
- **Difficulty control**: The difficulty reward, the source filter, eval-point re-tuning and the re-audit on each run. The report claims output that is "correct, discriminative, and controllable in length and difficulty". It gives no pass-rate thresholds.
- **Reported results**:
  - *Attribution.* Under a fixed SFT → RL → OPD recipe, better scale, diversity and verifiability of synthesised data and environments "account for essentially all of the observed gains".
  - *Infrastructure scale.* Sub-NUMA-partitioned worker VMs raised density from about 1,000 to more than 2,500 concurrent live containers per physical node. One DSec scale unit spans about 160 nodes, serves about 3M sandboxes a day, sustains more than 380K concurrent sandboxes and more than 5,000 creations per second.
  - *Documented answer-seeking exploits:* forged RPC messages to the `chronus` command sockets; reading `chronus` logs for leaked answers; overwriting `/bin/bash`; `XFS_IOC_SWAPEXT` to read protected files (this corrupted XFS metadata and forced a filesystem shutdown); scanning ports for reachable mirrors; fetching GitHub code via Go module proxies; installing newer package releases that already contain the implementation; answers leaked from package-mirror services.
  - *Accidental damage:* `grep` from `/` read `/proc/kpagecgroup` and crashed the kernel; `yes` produced tens of GB of logged output.
  - No ablation of the constructor.
- **Limitations / failure modes**: No reward formula, model size, retraining schedule or ablation, so the claim cannot be checked from outside. The replayed failures come from employees and deployment, not from the current RL checkpoint's failures. Mocked-tool fidelity is not measured. The approach assumes very large sandbox infrastructure.
- **How to reuse with easy seed tasks**: Store every seed as a (problem, environment, verifier) triplet. Train or fine-tune a constructor with reward = [passes multi-solver and inspector checks] × [pass rate of the current policy inside a target band]. Log per-task pass rates for every RL run and send tasks to re-audit when behaviour jumps (sudden 0→100%, reward-length anomalies). For repo tasks, copy the loop: leak scrub, an inspector that *reads solver trajectories*, and a repair step that tightens F2P tests every solver passes. Build hardened environments from sandbox snapshots with builder and runtime accounts kept apart.

### Kimi K3 task synthesis and agentic environments — Kimi K3: Open Frontier Intelligence (§4.2) (Kimi Team / Bai et al., 2026)
Link: https://arxiv.org/abs/2607.24653
- **Mechanism**:
  - *Unified white-box environment (§4.2.1).* A harness is represented as composable modules (tool interfaces, system prompts, context management, skills, memories, subagents). It can reproduce Kimi Code, Claude Code, Codex, OpenClaw, Hermes "as well as entirely new ones". During RL, "we dynamically construct different harness configurations for different task groups".
  - *Knowledge-graph-guided synthesis (§4.2.2).* The knowledge graph is a directed acyclic graph grown recursively by agents: each node gets an agent that runs web searches, reuses existing nodes to avoid duplicates, and stops when a concept is "sufficiently atomic". Nodes are sampled at different granularities, "individually or in related combinations". Their keywords plus ancestor context become web queries, and a synthesis agent turns the retrieved material into tasks of a chosen type.
  - *Personal-assistant tasks (§4.2.5).* Mock Gmail, Notion, Slack and Canvas. Tasks span several simulated days with "dozens of interdependent events", up to thousands of tool calls and millions of context tokens. Each event has its own rule-based or LLM evaluator. The initial workspace is built by agents from web reference material.
  - *Autonomous Execution Tasks (AET, §4.2.6).* Each task specifies an initial state, a constrained goal, a tool action space, an execution budget and an independent verifier. Examples: black-box system replication (rebuild a hidden system through oracle queries), quantitative factor discovery, tax auditing.
- **How it makes tasks harder**: Composing several concepts in one task. Distribution shift across harnesses. Long-horizon event streams whose events depend on each other. AET agents see "only the objective, context, constraints, and verification interfaces, without reference trajectories or predefined procedures".
- **Correctness / verification**: Rewards come from the verifier's view of the *final environment state*, "rather than the agent's self-reported completion".
  - *AET:* agents are isolated from verifiers; public verifiers give diagnostic feedback and hidden verifiers score held-out scenarios, with penalty-based rewards under limited submission budgets.
  - *Kernels:* any error above a numeric threshold against a PyTorch reference scores 0; matching the expert implementation scores 0.5; approaching the hardware roofline pushes the score toward 1. A hacking detector penalises CUDA graph replay, input caching and precision reduction, and is extended as new hacks appear.
  - *Web development:* reward is zeroed when a project fails to build, runs with errors, or "fakes rather than implements the artifact".
- **Difficulty control**: No learned generator and no pass-rate band in §4.2. Difficulty comes from environment design. Reasoning effort is controlled per domain by a curriculum over a budget multiplier τ, annealed from large (max effort) to smaller (high and low effort).
- **Reported results**: Scaling RL compute (FLOPs) raises capability across domains, and the number of tool-call steps grows with it (Fig. 8). Three domain experts × three effort levels = nine experts, merged by multi-teacher on-policy distillation. No task-synthesis ablations.
- **Limitations / failure modes**: The fidelity of the mock applications, the yield of the knowledge graph, and all filtering thresholds are unreported.
- **How to reuse with easy seed tasks**: Put seed topics into a concept DAG and sample *related combinations* to build multi-concept tasks. Run saturated tasks under randomised harness modules. For long tasks, attach a verifier to each event, and split verification into a public diagnostic part and a hidden held-out part with a submission budget, so the agent cannot fit the checker.

### Nemotron-IMO recipe and SU-01 (selection, not synthesis) — An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics (Moshkov et al., 2026); Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling (Li et al., 2026)
Link: https://arxiv.org/abs/2609.10712 ; https://arxiv.org/abs/2605.13301
- **Mechanism**:
  - *Nemotron SFT data.* 15,879 hard AoPS proof problems from Nemotron-Math-Proofs-v1, "selected using prior pass-rate evaluations". DeepSeek-V4-Pro (Max mode) writes proofs of up to 400K tokens. Problems not judged fully solved get up to three more refinement rounds on verifier feedback (350K-token cap). Verifier traces (scores in {0, 0.5, 1}) and meta-verifier traces are kept; score-0 verifier traces are subsampled. Result: 414,890 examples over 15,818 problems (58,543 proof, 67,971 refinement, 236,360 verification, 52,016 meta-verification).
  - *Nemotron RL.* Keeps problems that Nemotron-3-Ultra solves in 1–3 of 4 attempts, judged by DeepSeek-V3.2-Speciale: 9,597 problems. Reward follows DeepSeekMath-V2 without the self-analysis term. Dynamic sampling removes zero-advantage groups.
  - *SU-01.* Starts from P1-30B-A3B. SFT on about 340K sub-8K-token trajectories, ordered by *reverse perplexity* (highest perplexity first). Then 200 RL steps. The prompt pool (AoPS, competition books, Evan Chen's materials, the Shuzhimi forum, OPC proofs) is deduplicated and decontaminated. Rejection sampling removes prompts "already too easy or too hard for the current policy", leaving 8,967 verifiable and 16,287 non-verifiable prompts.
- **How it makes tasks harder**: Only by selecting the hardest human problems and refreshing the band. Neither pipeline synthesises new problems.
- **Correctness / verification**: Proof verifiers and meta-verifiers. Problem correctness is inherited from human sources.
- **Difficulty control**: Nemotron uses a 1–3/4 band. SU-01 rejects against the current policy. Nemotron's test-time development set has 3 easy, 7 medium, 10 hard and 10 unsolved problems, defined by the refinement round of first acceptance.
- **Reported results**: Nemotron scored 30/42 at IMO 2026, above the gold cutoff of 29, with full credit on P1, P2, P4 and P5. *Correction to the researcher's summary:* SU-01 reaches gold only with test-time scaling (35 points on both IMO 2025 and USAMO 2026). Without it, SU-01 clears the IPhO 2024 and 2025 gold lines but only bronze on IMO 2025 and USAMO 2026. IMO-ProofBench: 57.6% direct, 70.2% with test-time scaling. Low-perplexity-first was the weakest SFT order (24.3 on AnswerBench, 15.0 on AMO-Bench).
- **Limitations / failure modes**: No generator. These reports support note 12's "select a pass band" pattern and are the clearest evidence that frontier olympiad recipes do not synthesise harder problems.
- **How to reuse with easy seed tasks**: Recompute a 1–3/4 (or 1–7/8) band with the current checkpoint before every stage. Before RL, distil long generate → verify → refine traces from a stronger model so that the policy's pass rate on hard items is non-zero.

### EnterpriseBench CoreCraft — EnterpriseBench Corecraft: Training Generalizable Agents on High-Fidelity RL Environments (Mehta et al., 2026)
Link: https://arxiv.org/abs/2602.16179
- **Mechanism**: A simulated enterprise with more than 2,500 entities of 14 types and 23 MCP tools, served from stateful Docker containers. Design is "task-centric": every entity and tool exists to support tasks, not to maximise world size. Domain experts write tasks and rubrics. GLM 4.6 (357B total, 32B active) is trained with GRPO: 16 rollouts per prompt, each in its own container. Reward = the proportion of rubric criteria satisfied, judged by an LLM.
- **How it makes tasks harder**: Expert-designed tiers, from lookups to multi-step workflows under policy constraints (e.g., checking build compatibility, processing returns).
- **Correctness / verification**: Each rubric criterion is a verifiable factual assertion. A task passes only if every criterion is satisfied.
- **Difficulty control**: Human-set. Claude Opus 4.6 solves fewer than 31% of tasks.
- **Reported results**: 1,000 training and 150 held-out tasks. One epoch raised held-out pass rate from 25.37% to 36.76%. Transfer: BFCL Parallel +4.5%, τ²-Bench Retail +7.4%, Toolathlon 18.8% → 25.6% (+6.8%).
- **Limitations / failure modes**: Hand-authored and expensive. It shows what fidelity buys, not how to generate hardness.
- **How to reuse with easy seed tasks**: Use it as a *fidelity control*. At equal compute, compare RL on LLM-simulated hardened variants of your seeds against RL on a small, state-grounded, expert-built environment.

### DreamGym — Scaling Agent Learning via Experience Synthesis (Chen et al., 2025; ICLR 2026)
Link: https://arxiv.org/abs/2511.03773
- **Mechanism**: An "experience model" is fine-tuned from Llama-3.1-8B-Instruct on offline trajectories annotated with reasoning. It predicts the next *abstract textual* state and the reward using CoT, conditioned on the interaction history, the task, and the top-k semantically similar transitions from a replay buffer. The buffer is seeded with offline data and grows with on-policy transitions. The same parameters also act as a curriculum task generator, τ_t = M_task({τ_{t−1}^i}): it takes seed tasks with high group reward *variance* V_τ = (1/n)Σ(r_i − r̄)². The paper calls this "reward entropy"; it is maximal at 50/50 success. It then writes "progressively more challenging variations" of them. A hyperparameter λ caps the share of synthetic tasks per iteration. The policy is trained with PPO or GRPO entirely inside the model. The optional S2R variant (sim-to-real) adds a short real-environment RL phase (5K real rollouts).
- **How it makes tasks harder**: Variance-selected seeds are rewritten into harder variants. For PPO, tasks are clustered semantically so that groups exist.
- **Correctness / verification**: There is no executable verifier in simulation; rewards come from the learned model. Ablations show that removing reasoning or retrieval increases hallucination (judged by GPT-4o). Final correctness is measured on real benchmarks.
- **Difficulty control**: Variance-based seed selection plus the λ cap.
- **Reported results**:
  - *WebArena-Lite, zero real interactions:* 13.3 / 9.1 / 12.7 (Llama-3.2-3B / Llama-3.1-8B / Qwen2.5-7B) vs 7.3 / 6.1 / 6.1 for GRPO on 80K real transitions.
  - *WebShop, Qwen2.5-7B:* 68.3 vs 66.1.
  - *Ablation:* removing the task generator costs 6.6 points (WebShop) and 6.0 (WebArena).
  - *S2R:* more than 40% improvement over training from scratch with under 10% of the external data.
  - *Cost:* roughly one-third to one-fifth of real-environment RL.
  - *Data efficiency:* with the experience model trained on only 10K offline samples, the Llama-3.1-8B setup exceeds 50% success on WebShop.
- **Limitations / failure modes**: The reward is learned and can be exploited. Transfer drops sharply across large domain gaps (web → ALFWorld). *(Corrected: I removed the researcher's unverifiable claim that there is no official code.)*
- **How to reuse with easy seed tasks**: Compute per-task reward variance from existing GRPO groups. Send only mixed-outcome tasks to a rewriter prompted for a strictly harder variant. Cap synthetic tasks at a fixed fraction per batch. Keep an executable checker wherever one exists.

### GenEnv — GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators (Guo et al., 2025)
Link: https://arxiv.org/abs/2512.19682
- **Mechanism**: Two policies are initialised from Qwen2.5-7B-Instruct: an agent trained with GRPO, and an environment simulator that produces batches of task variations, each with an evaluation spec and a target action or answer. For each batch the agent's empirical success p̂ is computed. The simulator is rewarded with R_env = exp(−β(p̂ − α)²), α = 0.5, and updated by reward-weighted regression (RWR): weighted SFT on its own higher-reward generations, with a KL penalty to the initial simulator and a per-step KL cap. The agent's training pool admits only valid traces (parse, execute, checker runs without error).
- **How it makes tasks harder**: As the agent masters a task family, p̂ moves away from α and the simulator is pushed toward harder families. Required response length rises from 137 to 204 tokens (+49%) by epoch 6.
- **Correctness / verification**: The agent reward is exact match for structured actions and soft similarity (token-F1 or embedding) for free-form answers, both against the *simulator's* target. Correctness therefore depends on the simulator's reference.
- **Difficulty control**: α and β. Batches with |p̂ − α| > 0.1 are excluded from simulator updates. A proposition proves the α reward ranks task families consistently.
- **Reported results**:
  - *Base → GenEnv (Qwen2.5-7B):* ALFWorld 14.2 → 54.5; BFCL 7.0 → 41.8; API-Bank 61.6 → 79.1; Bamboogle 68.0 → 76.0; TravelPlanner 14.3 → 16.6; average 33.0 → 53.6.
  - *Data efficiency (BFCL validation):* 0.458 vs 0.438 for offline augmentation by Gemini-2.5-Pro with about 3.3× more data.
  - *Ablation:* 12.3% better than GenEnv-Random (a simulator that is never trained).
  - *Caveat:* ALFWorld multi-turn tasks are "decomposed into single steps for evaluation".
- **Limitations / failure modes**: The simulator writes both task and answer, so errors propagate. Soft-similarity rewards are weak. Only tested at 7B.
- **How to reuse with easy seed tasks**: Fine-tune a small generator on your seeds. Each epoch, weight its generations by exp(−β(p̂ − 0.5)²) using your policy's pass rate, and apply RWR with a KL penalty. Keep your own executable checker as the only source of agent reward.

### RLAnything — RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System (Wang et al., 2026; ICML 2026)
Link: https://arxiv.org/abs/2602.02488
- **Mechanism**: Policy, step-wise reward model and environment are all optimised together. When a task's rollout accuracy leaves [α_low = 0.2, α_high = 0.8], a language model rewrites it to be harder or easier "while preserving the essence". The rewrite is guided by a summary of steps the reward model flagged as possibly wrong (steps with at least one −1 score); in coding, the summary is unit-test results.
  - *ALFWorld:* the rewriter (Qwen3-4B) also receives a structured summary of object locations and properties.
  - *Coding:* the rewriter emits a new task together with unit tests.
  - *OSWorld:* the rewriter adds or removes hints, or switches to one of the pre-made perturbed templates for 47 of the 230 training tasks, each with its own evaluator and verifier files.
- **How it makes tasks harder**: Critic-guided rewrites, hint removal, and harder templates (e.g., "copy the Revenue column to Sheet2" becomes "…then rename it to Profit").
- **Correctness / verification**: A harder q′ is accepted only if α_low < acc(q′) < acc(q), and an easier one only if acc(q) < acc(q′) < α_high, so solvability is witnessed by the policy's own successes. *(Corrected mechanism for generated unit tests:)* a test is "gt" if it passes all ground-truth code. A gt test is rewarded by the number of non-gt solutions it makes fail; a non-gt test is penalised by the number of non-gt solutions it wrongly lets pass.
- **Difficulty control**: The two thresholds plus the directional acceptance rule.
- **Reported results**: Environment adaptation on top of policy + reward co-training: OSWorld in-domain 49.6 → 52.1, out-of-distribution (OOD) 20.0 → 21.3; ALFWorld in-domain 55.6 → 60.2, OOD 61.8 → 63.6. Totals versus before optimisation: OSWorld in-domain +11.7, ALFWorld +21.2 / +18.7, LiveBench +11.9. The abstract reports +9.1% on OSWorld after scaling for Qwen3-VL-8B-Thinking.
- **Limitations / failure modes**: Harder GUI objectives need hand-made verifier files. The acceptance test proves solvability only by the current policy and says nothing about whether the rewrite drifted from the intended skill.
- **How to reuse with easy seed tasks**: This is the simplest loop to adopt. Whenever acc > 0.8, prompt for a harder rewrite with a summary of near-misses, roll out, and keep the rewrite only if 0.2 < acc′ < acc. For code, have the rewriter emit tests and validate them against reference solutions.

### STRETCH — STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution (Yu et al., 2026)
Link: https://arxiv.org/abs/2609.18642
- **Mechanism**: One set of parameters plays two roles through system prompts. In the *difficulty-alignment* loop, the Scaffolder samples a group of adaptive questions (constraints) for a base context. A frozen Learner snapshot solves each several times, an LLM judge or verifier gives accuracy p̂, and the Scaffolder is trained with GRPO on a reward that peaks at p̂ = 0.5 (sharpness α). In the *exploration* loop, the Learner is trained with GRPO on questions from the frozen Scaffolder. Each step does exactly one Scaffolder update and one Learner update. Every epoch, "golden" successful interactions (adaptive questions plus optimal traces) are replayed with SFT.
- **How it makes tasks harder**: The Scaffolder adds constraints to OR instances or bargaining scenarios so they land in the Learner's "stretch zone".
- **Correctness / verification**: OR tasks: binary reward "strictly based on code execution and optimality", and infeasible constraints get zero verifier reward. Negotiation: an LLM judge scores deal success and strategy.
- **Difficulty control**: The target p. Ablation: p = 0.5 beats 0.8 (too often already solved) and 0.2 (too few successful trajectories). It reaches 67.9 execution rate and 63.6 solving accuracy on Mano Complex.
- **Reported results**: Without golden replay, performance "drops sharply" and formatting collapses by epoch 4.
- **Limitations / failure modes**: Narrow domains (Craigslist bargaining, Mano Complex, Complex OR). Negotiation depends on a judge.
- **How to reuse with easy seed tasks**: For optimisation or OR seeds, let the model propose extra constraints. Check feasibility and optimality with a solver, and reward proposals that land near 50% learner success. Always replay verified good trajectories.

### SEAD — SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue (Dai et al., 2026)
Link: https://arxiv.org/abs/2602.03548
- **Mechanism**: The user side is split into a Profile Controller that sets initial user states and a User Role-Play Model that plays them. States combine cooperation (5 levels) × emotion (4) × trust (6) = 120 combinations, plus behaviour patterns taken from more than 100K anonymised real dialogues. State evolves turn by turn with agent quality. The service agent (Qwen2.5-14B) is trained with GRPO on the success state the role-play model reaches at the end. *(Corrected:)* the controller is **not a trained adversary**. It samples randomly in the first iteration, then by statistics: p(p₀) ∝ 1 − |CR − 0.5| over the completion rate (CR) recorded for each state combination. The role-play model is not trained.
- **How it makes tasks harder**: As the agent improves, sampling mass moves to less cooperative, lower-trust profiles.
- **Correctness / verification**: The outcome comes from an untrained role-play simulator whose realism was rated by GPT-5.1 with few-shot human annotations (humanness above 4.5/5; violation score 1.15, "matching real behavior").
- **Difficulty control**: Completion-rate-targeted sampling.
- **Reported results**: 52.0% completion vs 44.2% for GPT-4o. The abstract's "+17.6%" is a *relative* gain (7.8 points); it is 34.4% relative over the base 14B model. Lowest average turns to target (9.6). *Key negative finding:* training the role-play model "degrades role-play quality, as URM will force success rates around 50% by directly accepting or rejecting regardless of agent performance". *(Venue claim "Findings of ACL 2026" could not be verified and was removed.)*
- **Limitations / failure modes**: One enterprise scenario (restaurant promotion calls). Realism is judged by an LLM.
- **How to reuse with easy seed tasks**: For user-simulated tasks, let a sampler or generator choose only the persona and initial state (resistance, missing information, mood) at CR ≈ 0.5. Never train the component that decides the outcome.

### COvolve — COvolve: Adversarial Co-Evolution of Large-Language-Model-Generated Policies and Environments via Two-Player Zero-Sum Game (Sygkounas et al., 2026; GECCO 2026)
Link: https://arxiv.org/abs/2603.28386
- **Mechanism**: GPT-5.2 acts as both Environment Designer and Policy Designer, each writing Python. Every policy is run on every level (100 episodes) to build a payoff matrix. Solving the mixed-strategy Nash equilibrium (MSNE) gives a policy mixture p*. The Environment Designer proposes K mutations of the current level and keeps the one that minimises the MSNE mixture's expected return (a best response). The Policy Designer mutates policies for the new level. This is PSRO (policy-space response oracles) adapted to unsupervised environment design (UED).
- **How it makes tasks harder**: Levels evolve from ad hoc code to parameterised designs with explicit solvability checks, chains of key–door dependencies, narrow chokepoints, denser obstacles, and more aggressive CARLA actors.
- **Correctness / verification**: Feasibility checks. For 2D navigation, a reachability check on an occupancy grid with obstacles inflated by the agent's radius; invalid placements are rejected. Payoffs are measured by running the code.
- **Difficulty control**: Adversarial minimisation against the equilibrium mixture rather than the latest policy.
- **Reported results**: Versus UED-Greedy (latest policy only) and UED-Uniform (uniform mixture), the MSNE mixture keeps performance on the whole archive while the latest policy forgets earlier levels. Standard PPO, SAC and QR-DQN fail on the evolved fully observable tasks (mostly qualitative and figure-based).
- **Limitations / failure modes**: The policies are programs, not LLM weights. Evaluation costs O(policies × levels). The feasibility heuristics are hand-written.
- **How to reuse with easy seed tasks**: Keep a checkpoints × tasks payoff matrix. When mutating tasks to be harder, choose the mutation that minimises success of the *equilibrium mixture of checkpoints*. This stops the generator from chasing the latest model while older skills are forgotten.

### From Trainee to Trainer (+ EnvGen, SimWorld Studio) — From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning (Chen et al., 2026); EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents (Zala et al., 2024); SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning (Kang et al., 2026)
Link: https://arxiv.org/abs/2606.17682 ; https://arxiv.org/abs/2403.12014 ; https://arxiv.org/abs/2605.09423
- **Mechanism**:
  - *From Trainee to Trainer.* After each RL stage, the *current policy* (Qwen3-4B) receives structured summaries of its training behaviour, validation failures (parse errors, illegal moves, conflicts, hole collisions and more, per map size) and environment statistics. It then writes the next configuration of MAPF-FrozenLake, a generator built on Conflict-Based Search. For each map size the configuration sets a data ratio, a hole ratio, and a wait ratio (the share of instances that need at least one wait action to resolve conflicts). The model never writes instances itself.
  - *EnvGen.* An LLM writes Crafter/Heist configurations (terrain, starting items, pre-achieved subgoals) and revises them from per-skill success in the original environment, using about 4 LLM calls in total.
  - *SimWorld Studio.* SimCoder writes Unreal Engine 5 code. A rule-based verifier checks collisions and support, a VLM critiques scenes, NavMesh produces solvable navigation tasks, and agent feedback adapts difficulty.
- **How it makes tasks harder**: Parameter edits targeted at failures (larger maps, more holes, more conflicts that need waits), or new scenes aimed at weak skills.
- **Correctness / verification**: Instances come from generators with exact solvers or checkers. The LLM chooses only parameters.
- **Difficulty control**: Explicit knobs. "Naively maximizing difficulty is also insufficient: overly difficult environments may collapse the learning signal."
- **Reported results**: The 4B environment engineer beats fixed curricula and larger proprietary designers (GPT-5.4, Gemini-3.1-Pro). Successful updates "rely on failure evidence and preserve configurations that already work", and the RL checkpoint is a better engineer than the base model. EnvGen: an RL agent under 5M parameters beats a GPT-4 agent on Crafter using 4 LLM calls (SPRING averages 2.7K calls). SimWorld Studio: +18 success-rate points over fixed-environment training and +40 over an untrained agent.
- **Limitations / failure modes**: Needs a parameterised, verifiable generator. Toy and embodied domains only.
- **How to reuse with easy seed tasks**: If your seeds come from a procedural generator, give the current checkpoint the knobs plus a failure histogram per knob, let it propose the next mixture, and keep the previous mixture as a fallback when validation drops.

### SEAL — SEAL: Synergistic Co-Evolution of Agents and Learning Environments (Hu et al., 2026)
Link: https://arxiv.org/abs/2605.24426
- **Mechanism**: Rollouts are diagnosed into turn-level failure labels grounded in verifier evidence (invalid or missing tool call, argument or state mismatch, failure to recover), and the labels are aggregated into a failure profile. Only the training-time *learning interface* changes, chosen by failure type: constraint cues, schema-implied tool-affordance cues (for missing_tool_call), and structured recovery feedback "without revealing the correct answer" (for recovery_failure). The same diagnoses reweight GRPO advantages by a clipped positive weight w_j, which leaves each advantage's sign unchanged. The evolved interface is removed at evaluation.
- **How it makes tasks harder**: It doesn't: it scaffolds. Run in reverse, it is a clean hardening operator.
- **Correctness / verification**: Tool backend, labels, rewards and evaluation are unchanged.
- **Difficulty control**: Driven by the failure profile.
- **Reported results**: With only 400 training samples, +8.25 to +26.25 average points across three backbones, with positive OOD transfer.
- **Limitations / failure modes**: Low-resource setting only. Makes tasks easier.
- **How to reuse with easy seed tasks**: Reverse it. Strip schema-affordance and constraint cues and return generic errors on saturated tool tasks while the verifier stays fixed. Use SEAL-style diagnoses to decide which cue to remove first.

### ADDED — Environment Tuning — Don't Just Fine-tune the Agent, Tune the Environment (Lu et al., 2025)
Link: https://arxiv.org/abs/2510.10197
- **Mechanism**: RL from only 400 BFCL V3 problem instances, with no expert trajectories. The four-stage curriculum:
  - *Stage 1:* syntactic and schema correctness, with a format-only reward.
  - *Stage 2:* the Base split, with "actionable environment augmentation" and a fine-grained progress reward.
  - *Stage 3:* adds the Missing-Parameters, Missing-Functions and Long-Context splits.
  - *Stage 4:* the augmentation is *disabled*, "forc[ing] the agent to … handle uninformative or standard error messages, just as it would during final evaluation".
  - *What the augmentation does:* failure messages reveal inter-tool dependencies (e.g., look up the airport code before booking) and tool-internal constraints (e.g., `rm` takes relative paths), rather than bare errors like `FileNotFoundError`.
  - *Stage transitions:* decided by validation and stability.
- **How it makes tasks harder**: A curriculum by split, then removal of the scaffolds (stage 4).
- **Correctness / verification**: The original BFCL state and response checks. The progress reward decomposes completion into dense components (appendix).
- **Difficulty control**: Stage schedule. Hints on or off.
- **Reported results**: Lifts Qwen2.5-7B from near zero to strong in-distribution performance with 400 instances. Better OOD generalisation (BFCL V4, ACEBench) than SFT baselines, which collapse. Ablations show both augmentation and progress reward matter.
- **Limitations / failure modes**: One benchmark family. The hints are hand-designed or LLM-generated per tool.
- **How to reuse with easy seed tasks**: Use the stage-4 idea directly: once a task is saturated *with* informative errors, retrain on the same tasks with uninformative errors. The verifier does not change.

### CoEvolve — CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution (Yang et al., 2026; ACL 2026)
Link: https://arxiv.org/abs/2604.15840
- **Mechanism**: During GRPO, rollouts are mined for three weakness signals. *Forgetting*: a task that previously succeeded now fails. *Boundary*: a group has both successes and failures. *Rare*: an action pattern below a frequency threshold (e.g., θ = 5) recurs. Flagged trajectories prompt an LLM to re-explore the environment around those patterns. The resulting (action, observation) triplets are grouped by exploration run and abstracted into task–solution pairs, which are validated by execution and appended to the training set.
- **How it makes tasks harder**: New tasks concentrate on unstable, forgotten and under-explored behaviour.
- **Correctness / verification**: An LLM agent executes each task and solution in the real environment. Accepted if it completes the objective, *or* "if execution fails but the environment returns a positive reward". The second clause is a loophole.
- **Difficulty control**: Implicit, through signal targeting. Boundary signals dominate: 51.4% on AppWorld, 45.5% on BFCL.
- **Reported results**: Average gains of +19.43 (Qwen2.5-7B), +15.58 (Qwen3-4B) and +18.14 (Qwen3-30B-A3B). AppWorld challenge split +23.21 TGC / +21.43 SGC for 30B-A3B. Training performance rose 0.21 → 0.35 while a static baseline went 0.17 → 0.29 → 0.23.
- **Limitations / failure modes**: The positive-reward acceptance clause. Results depend on how similar synthesised tasks are to validation tasks (the paper studies this).
- **How to reuse with easy seed tasks**: Log per-task success history across GRPO steps. Forgetting and boundary tasks are the best seeds for LLM-guided hardening. Accept a new task only when its abstracted solution replays successfully.

### AgentEvolver (self-questioning) — AgentEvolver: Towards Efficient Self-Evolving Agent System (Zhai et al., 2025)
Link: https://arxiv.org/abs/2511.10395
- **Mechanism**: A high-temperature LLM (Qwen-Plus, T = 1) explores a sandbox guided by an environment profile of entities, attributes and operations: breadth-first for N_b steps, then depth-first for N_d steps (3 + 17 on AppWorld; 3 + 27 on BFCL). Tasks are synthesised from the exploration traces, and difficulty is set by "the number of entities, attributes, and operations involved". The experiments request 2 entities, 3 attributes and 3 operations at a "hard" level. Because tasks are written after exploration, a prompted LLM extracts each task's reference solution from the trace. Filters: lexical deduplication in real time and after generation, plus a feasibility check that *executes the reference solution*. Rewards come from a judge that compares the agent's trajectory with the reference solution. Proxy and real tasks can be mixed: p_hybrid = (1 − λ)p_target + λp_task.
- **How it makes tasks harder**: Scaling the entity, attribute and operation counts.
- **Correctness / verification**: Solvable by construction, confirmed by re-execution.
- **Difficulty control**: Preference-specified counts and λ.
- **Reported results**: Qwen2.5-7B avg@8. AppWorld: zero-shot 1.8, original tasks 16.1, synthetic only 23.2, hybrid 21.8. BFCL: 29.8 / 58.8 / 49.0 / 65.3. Qwen2.5-14B AppWorld: 18.0 / 46.1 / 44.3 / 48.4.
- **Limitations / failure modes**: Limited to what the explorer reached. The judge reward is weaker than state checks.
- **How to reuse with easy seed tasks**: Harden by composition: chain two witnessed sub-solutions into one task (more entities and operations), then verify by replaying the concatenated reference.

### CuES — CuES: A Curiosity-driven and Environment-grounded Synthesis Framework for Agentic RL (Mai et al., 2025)
Link: https://arxiv.org/abs/2512.01311
- **Mechanism**: A concept pool and principles are extracted from the environment description (seed goals are optional). An explorer uses a per-environment *memory tree* to prefer actions not yet tried in similar states. Sliding windows over the trajectory are abstracted into (goal, guideline) pairs with a confidence score. A judge agent *re-executes* each goal and accepts only reward = 1.0 plus path faithfulness. Each accepted goal is then rewritten for exactly L steps, each adding hints distilled from the witnessed trajectory (action verbs, landmarks, parameter examples, preconditions).
- **How it makes tasks harder**: Mainly the opposite: the ladder adds hints. The terse L = 0 goal is the hardest rung.
- **Correctness / verification**: Execution-based acceptance. The witnessed trajectory is stored as the solution.
- **Difficulty control**: Rewrite depth L.
- **Reported results**: Abstract only: generated tasks "match or surpass manually curated datasets in both diversity and executability" with "substantial downstream" RL gains.
- **Limitations / failure modes**: Hardness capped by exploration. Few numbers in the abstract.
- **How to reuse with easy seed tasks**: Invert the ladder. Remove parameter exemplars, landmarks and precondition hints from goals the policy solves, and keep the witnessed trajectory as the reference.

### ADDED — UI-Simulator / UI-Simulator-Grow — LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training (Wang et al., 2025)
Link: https://arxiv.org/abs/2510.14969
- **Mechanism**: An LLM (GPT-4o-mini) simulates structured UI states, either without retrieval or grounded in a small offline corpus of test-environment transitions via a BM25 + semantic retrieval pipeline. A teacher agent explores under step-wise "task controls": it proposes a high-level task from the state and updates it when the current one is done. A trajectory wrapper infers the user instruction afterwards and rebuilds the reasoning. *UI-Simulator-Grow* is the targeted part:
  - Each iteration measures the *teacher-forcing loss* of the student against teacher actions on a validation set.
  - Tasks between the 25th and 75th loss percentiles are selected as targets.
  - Variants are made by "lightweight task rewriting" that keeps action types and flow (e.g., "search running shoes" → "search slippers"), and the simulated states are edited to match.
  - Representative earlier tasks are replayed to prevent forgetting.
- **How it makes tasks harder**: Resources go to mid-difficulty tasks, measured without a verifier.
- **Correctness / verification**: No state verifier. Trajectories are teacher-generated; retrieval grounds the simulation.
- **Difficulty control**: The loss-percentile band, and the validation set is refreshed each iteration.
- **Reported results**: WebArena success: UI-Simulator-F 6.28, -R 6.40, Grow-R 7.14 (Llama-3-8B base, 2.34 before training), matching Llama-3-70B-Instruct (7.02) with 66% of the trajectories. AndroidWorld 0 → 8.6 (-F), 12.9 (-R), 13.4 (Grow-R). GPT-4o scores 13.10 / 11.7.
- **Limitations / failure modes**: Low absolute success rates. Variants are content swaps, not structural hardening.
- **How to reuse with easy seed tasks**: If you lack a cheap verifier, use teacher-forcing loss (student vs strong teacher) as the difficulty proxy and keep the middle band. Pair it with structural operators rather than content swaps.

### Simia (Simia-SFT / Simia-RL) — Simulating Environments with Reasoning Models for Agent Training (Li et al., 2025)
Link: https://arxiv.org/abs/2511.01824
- **Mechanism**: A reasoning LLM is prompted with tool specifications, policy rules, output formats and one seed trajectory. It simulates whole trajectories (user turns, reasoning, tool calls, tool outputs) with no backend. Seeds are pre-filtered by an LLM for completeness and logic. Outputs are repaired by rules (JSON fixes) and filtered on tool names. Simia-RL uses o4-mini (T = 1) as both environment and reward.
- **How it makes tasks harder**: It doesn't: it amplifies breadth.
- **Correctness / verification**: Only structural validity and LLM checks.
- **Difficulty control**: None.
- **Reported results**: About 5K APIGen-MT seeds grow to 90K trajectories; 668 AgentTuning seeds grow to 15K. The 32B model beats GPT-4o and xLAM-2-70B on τ²-Bench (58.9 average). On OfficeBench, RL in the *simulated* environment beat RL in the real one (2-apps 64.7 vs 60.8; 3-apps 34.5 vs 28.6), which the authors attribute to more explanatory simulated error messages (e.g., "overlaps with a lunch break").
- **Limitations / failure modes**: Simulated feedback richer than reality makes tasks *easier* than reality. No fidelity metric.
- **How to reuse with easy seed tasks**: Use it for cheap SFT breadth. For hardening, instruct the simulator to return the real environment's terse errors and to withhold explanations.

### WebWorld — WebWorld: A Large-Scale World Model for Web Agent Training (Xiao et al., 2026)
Link: https://arxiv.org/abs/2602.14721
- **Mechanism**: World models of 8B, 14B and 32B trained on 1.06M open-web trajectories. Data comes from rule-based crawls of URLs in pre-training corpora (43.3%), autonomous exploration with self-generated tasks, and task-directed synthesis (seed intent → parameter-perturbed variants → paraphrases). Training includes CoT for next-state prediction, and simulation runs 30+ turns. *Abstract-and-Instantiate* turns a concrete seed ("Book a flight to London on March 15th") into an underspecified goal ("Book a flight to somewhere on sometime"). The agent acts under the abstract goal, each trajectory is instantiated back into a concrete task, the agent retries the concrete task, and only successes are kept.
- **How it makes tasks harder**: It diversifies rather than hardens.
- **Correctness / verification**: Rejection sampling on success inside the world model. Intrinsic evaluation with WebWorld-Bench: Factuality and Web Turing scores over nine dimensions, judged by GPT-4o.
- **Difficulty control**: None explicit.
- **Reported results**: 8,000 trajectories give Qwen3-8B +9.9 on MiniWob++ and +10.9 on WebArena (Reddit +18.3, GitLab +12.0). Qwen3-14B gains +9.2 on WebArena. Simulation quality is comparable to Gemini-3-Pro, and it beats GPT-5 as a world model for lookahead search. Limitation stated by the authors: "sycophancy by generating overly optimistic outcomes that cater to the agent's action".
- **Limitations / failure modes**: Optimism makes simulated tasks easier than real ones. Weak at generating detailed content.
- **How to reuse with easy seed tasks**: Use Abstract-and-Instantiate as a diversity operator. Add explicit pessimism instructions and real-environment spot checks before counting simulated difficulty as real difficulty.

### Qwen-AgentWorld — Qwen-AgentWorld: Language World Models for General Agents (Zuo et al., 2026)
Link: https://arxiv.org/abs/2606.24597
- **Mechanism**: Language world models (35B-A3B and 397B-A17B) for seven domains: MCP, Search, Terminal, SWE, Android, Web and OS. Training is continued pre-training (CPT) → SFT → RL on more than 10M real interaction trajectories. The RL reward is a five-dimension LLM rubric (format, factuality, consistency, realism, quality) combined 9:1 with an executable rule verifier where one exists. Every prompt can carry a *simulation instruction* that controls the simulator. Three applications:
  - *OpenClaw (§6.1.1).* Real traces are distilled into seed scenarios (initial state plus query). These are expanded into 4K environments: "vary concrete states while preserving the underlying workflow structure; … rephrase intents, adjust difficulty, and compose multi-step goals". Each task gets a rubric verifier.
  - *MCP (§6.1.2).* Each simulator prompt holds the initial state, a summary of the *hidden* environment state (database contents, permissions, service availability) behind a verified frontier-agent trajectory, and control instructions: "injected errors, withheld intermediate results, response formats, and the final tool and environment state required for task success".
  - *Search.* 1K fictional worlds, each built on a 300–500-row relational database of consistent invented facts. Four strategies: time-shifted (e.g., a 2029 phone ranking with real brands and invented models), granular long-tail, private simulation, and realistic grounded. Answers come from running SQL; queries are reverse-generated. Snippets must "tease information without fully answering".
- **How it makes tasks harder**: Perturbations that real deployments rarely produce: intermittent API errors, paginated responses, incomplete intermediate results, partial batch failures. Withholding answers forces page extraction. Fictional facts block answering from memory.
- **Correctness / verification**: SQL-derived answers. "Automated retrieval checks and dual rule-based & LLM validation" confirm facts fit real-world patterns yet are "unsearchable". World-model reward hacking by *self-praise* (inserting "operation completed successfully…") was countered with three measures: the rule-verifier anchor, content-type scoping (deterministic content judged by exact match) and strict tag extraction so the judge never sees reasoning.
- **Difficulty control**: Natural-language control instructions. Task-side difficulty adjustment and goal composition for OpenClaw.
- **Reported results**:
  - *OpenClaw Sim RL (397B simulator):* Claw-Eval 65.4 → 69.7, QwenClawBench 47.9 → 55.0. With Qwen3.6-Plus prompted as the simulator: 66.7 / 47.8.
  - *MCP:* uncontrolled Sim RL moved Tool Decathlon 32.4 → 31.5; controlled Sim RL gave 36.1 (+3.7), and MCPMark 21.5 → 33.8 (+12.3).
  - *WideSearch (fictional worlds):* item-F1 34.02 → 50.31 and row-F1 13.72 → 24.21 at 35B; +3.87 / +6.05 at 397B.
  - *Sim RL vs live-search RL (first 60 steps):* 50.3% vs 45.6%. Page-extraction calls rose from 2.5 to 4.0 per trajectory under Sim RL but fell to 1.5 under real RL.
  - *Authors' takeaway:* "State is the bottleneck": without a detailed initial state, fidelity and gains degrade.
- **Limitations / failure modes**: Needs a heavily trained dedicated simulator. Simulated-task verifiers are mostly rubrics. No per-state-change fidelity breakdown.
- **How to reuse with easy seed tasks**: Replay saturated tool-use seeds in a simulator told to paginate, fail intermittently, drop fields or return partial batches, with the ground-truth end state and hidden-state summary fixed. For search seeds, build a fictional database, derive answers by SQL, and serve only partial snippets.

### ADDED — LiteResearcher — LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent (Qu et al., 2026)
Link: https://arxiv.org/abs/2604.17931
- **Mechanism**: A local "virtual world" built from real pages rather than an LLM simulator. Question–answer (QA) pairs are extracted from a seed corpus (Wikipedia, BBC News). Each surviving pair triggers live-web fetches of related pages, which are appended to an ever-growing local corpus. A local search engine and browse tool serve training. **Source masking**: "after a QA pair is generated, we deliberately delete its original information source from the local corpus", forcing alternative multi-hop search paths.
- **How it makes tasks harder**: Source deletion removes the one-hop shortcut. Stage-wise difficulty filtering.
- **Correctness / verification**: A 7-criterion LLM rubric (independence, ambiguity, answer verifiability and others); all seven must pass.
- **Difficulty control**: Before each RL stage, 8 rollouts per query; keep only 1 ≤ correct ≤ 7.
- **Reported results**: LiteResearcher-4B reaches 71.3% on GAIA and 78.0% on Xbench (Tongyi DeepResearch 30B: 70.9 / 75.0). The run issued 73.2M tool calls; commercial APIs would have cost an estimated $59K–$243K, with 10–46× higher latency.
- **Limitations / failure modes**: Answers are LLM-extracted, so the rubric is the only guard. Web-only.
- **How to reuse with easy seed tasks**: For retrieval seeds solved in one hop, delete (or block in the index) the document the answer was extracted from, and keep only questions still answerable through other pages.

### SWE-World — SWE-World: Building Software Engineering Agents in Docker-Free Environments (Sun et al., 2026)
Link: https://arxiv.org/abs/2602.03419
- **Mechanism**: A lightweight sandbox applies real file edits, so the workspace state is exact. A transition model (SWT) trained on real Docker rollouts predicts ⟨stdout, stderr, exit_code⟩ for execution commands. A reward model (SWR) reads repository context, the final patch and the test content, writes a structured test report, and then emits a binary reward. Both are Qwen2.5-32B/72B fine-tuned on CoT → ground truth. SWR drives SFT filtering, RL and test-time selection (multi-sample voting).
- **How it makes tasks harder**: Indirectly. Task supply is no longer limited to repositories that containerise successfully.
- **Correctness / verification**: SWR is benchmarked against Docker outcomes. "To prevent git hacking … we disallow solution-revealing git commands (e.g., git log, git show)."
- **Difficulty control**: None.
- **Reported results**:
  - *SWE-bench Verified, Qwen2.5-Coder-32B:* 6.2% → 52.0% (SFT) → 55.0% in the table (54.8% in the text; RL) → 68.2% (TTS@8).
  - *SWR-32B:* accuracy 0.754, precision 0.779.
  - *CoT ablation (4K training examples):* SWR accuracy 0.578 → 0.712, precision 0.609 → 0.754. CoT adds only 0.8% for SWT.
  - *Hacking:* with the non-CoT SWR, "trajectory length collapses drastically after step 20" as short invalid patches are scored correct.
- **Limitations / failure modes**: Reward precision caps how hard tasks can get. Subtler tests mean more reward-model error.
- **How to reuse with easy seed tasks**: If you simulate execution to scale hard SWE tasks, keep file state real, use a *reasoning* reward model with high precision, and treat early reward spikes combined with length collapse as a hacking alarm.

### Masked-diffusion text world models — Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL (Deshpande, 2026)
Link: https://arxiv.org/abs/2607.16204
- **Mechanism**: World modelling is framed as steerable transition dynamics with five parts: initial state, task context, tool schemas, domain rules and steering directives. Dataset: 239,403 grounded trajectories from nine environments and twelve model families. About 80% carry Claude-written "hindsight instructions" describing ground-truth context (extra database state, reward structure, schemas). Masked-diffusion LMs (SDAR, LLaDA-2.1, WeDLM) are compared with autoregressive LMs. Bidirectional denoising lets each output condition on global anchors such as schemas and expected outcomes. A plug-and-play GRPO framework uses the world model as the environment, with deterministic state checks.
- **How it makes tasks harder**: Steering directives. The *behaviour probes* (not a curriculum) covered adjacent-but-not-answering database states, acting with insufficient information, verbose outputs, length limits, infeasible tasks, and pagination.
- **Correctness / verification**: Deterministic state checks. Four experts rated 100 SDAR next states: realism 4.75, outcome correctness 4.25, training utility 4.50 (Krippendorff's α ≥ 0.89).
- **Difficulty control**: Directive-based, not evaluated as a curriculum.
- **Reported results**: Up to 47% absolute zero-shot gains on OOD environments (ScienceWorld, ALFWorld, AppWorld) for 1.2B–7B agents. Masked-diffusion LMs beat autoregressive LMs more than 4× their size on coherence and diversity. SDAR-8B: Self-BLEU 0.601, MAUVE 0.900.
- **Limitations / failure modes**: Single author. Small human study.
- **How to reuse with easy seed tasks**: If you run a text simulator, add hindsight state context and hardening directives (forced tool failure, near-miss database rows) to prompts, and compare a masked-diffusion simulator for diversity.

### EnvACE (+ WebEvolver, CoMAP, PaW) — EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning (Xu et al., 2026); WebEvolver (Fang et al., 2025); CoMAP (Liu et al., 2026); Policy and World Modeling Co-Training for Language Agents (Lu et al., 2026)
Link: https://arxiv.org/abs/2608.06197 ; https://arxiv.org/abs/2504.21024 ; https://arxiv.org/abs/2606.02372 ; https://arxiv.org/abs/2606.02388
- **Mechanism**:
  - *EnvACE.* One policy (Qwen3-8B) alternates between acting (a tool call) and *rehearsal* (writing the environment's response itself). Both roles are optimised with role-wise GRPO on the task-success reward, from an outcome evaluator or checklist judge on the CM2 dataset, and no external environment is queried in training.
  - *WebEvolver (EMNLP 2025).* A co-trained web world model serves as a virtual server for self-instructed data and as a look-ahead engine.
  - *CoMAP (paper name "CoMap", EMNLP 2026).* The world model predicts feedback for candidate actions and is self-distilled on on-policy data.
  - *PaW (EMNLP 2026).* Adds auxiliary next-observation prediction to the policy during RL, with data selection by action entropy, a noise-tolerant (clipped MAE) loss and reward-adaptive loss balancing.
- **How it makes tasks harder**: None of them do; they change where transitions come from.
- **Correctness / verification**: Only the outcome reward. Rehearsed observations are not grounded.
- **Difficulty control**: None.
- **Reported results**: EnvACE overall 32.91% (+0.99 over EnvScaler-8B, +0.37 over AWM-14B). τ²-Bench 31.2% → 36.7% versus plain GRPO. Sharing parameters between roles gives 35.5 → 36.7. Parallel rehearsal at test time (N = 2) reaches 40.9%. WebEvolver: about a 10% gain over prior self-evolving web agents. CoMap: 16.75% relative gain with Qwen3-4B.
- **Limitations / failure modes**: A policy that writes its own observations can learn convenient worlds. No hacking analysis.
- **How to reuse with easy seed tasks**: Use as an auxiliary loss or warm-up. Do not use self-rehearsed environments to measure or raise difficulty.

### SynthAgent — Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards (Lyu et al., 2026)
Link: https://arxiv.org/abs/2601.22511
- **Mechanism**: A teacher (Qwen3-235B-A22B-Instruct, deployed locally) takes a Persona Hub persona, infers a workflow, and builds a task-specific virtual tool suite with per-task forbidden behaviours (e.g., no system reboot). It then *rewrites the detailed workflow into an underspecified instruction* and moves the critical details into a *private user context* that an LLM user simulator reveals only when asked. An LLM mock tool system answers calls, consulting a per-task map of earlier calls and responses so answers stay consistent.
- **How it makes tasks harder**: The information gap forces clarification and extra tool calls.
- **Correctness / verification**: Reward = mean of the subgoal fraction and the required-interaction fraction, zeroed if any forbidden behaviour occurs; judged by an LLM against the rubric. Subgoals come from four teacher executions, and tasks whose workflow is largely unexecuted are dropped. Persona-rewritten math and search problems are kept only if Qwen3-235B gives 3 consistent answers.
- **Difficulty control**: The size of the information gap. No adaptive loop.
- **Reported results**: 15,096 tool tasks generated with local open models. Qwen3-8B and 14B trained on them beat larger baselines across 14 datasets. A small tool simulator suffices, and extra teacher demonstrations add little.
- **Limitations / failure modes**: No state grounding. Inconsistency across rollouts corrupts advantages, which is why the call map exists.
- **How to reuse with easy seed tasks**: "Move the details into a hidden user profile" is a cheap hardening operator for fully specified tool tasks. Grade by subgoals, required clarifications and forbidden actions.

### EnvFactory — EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL (Xu et al., 2026)
Link: https://arxiv.org/abs/2605.18703
- **Mechanism**: A Search agent grounds environment proposals in real sources. A Code agent implements a stateful database schema and Python tools. A Test agent writes unit tests covering interface consistency, execution, expected behaviour and database state transitions, iterating until they pass. Tool chains are sampled from a dependency graph over the 842 tools so that every required input is user-provided or produced by an earlier tool. QueryGen splits a chain into subgoals and writes turns, then applies four calibrated refinements: (1) implicit reference (replace explicit IDs, omit deducible parameters), (2) action compression (skip inferable intermediate steps), (3) ambiguity introduction, (4) goal expansion (add a related secondary objective).
- **How it makes tasks harder**: The four refinements, plus chain topology.
- **Correctness / verification**: Executable, unit-tested environments. For each query, k candidate trajectories are generated and the best is chosen by database state changes. RL reward R = α·R_traj + (1 − α)·R_state − γ·P_length.
- **Difficulty control**: Topology and the refinements. No adaptive loop.
- **Reported results**: 85 environments across 7 domains; 1,622 SFT and 953 RL conversations. Up to +15% on BFCLv3, +8.6% on MCP-Atlas, +6% on τ² and VitaBench. BFCL-v3 improves as the environment pool grows from 50 to 75 to 85.
- **Limitations / failure modes**: Static generation. Belongs with the code- and database-backed family in note 06 (AWM, AgentScaler, COVERT).
- **How to reuse with easy seed tasks**: Apply the four refinements to over-specified seed queries while the executable environment keeps the ground-truth end state fixed.

### AutoEnv — AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning (Zhang et al., 2025)
Link: https://arxiv.org/abs/2511.19304
- **Mechanism**: A theme is turned into a YAML DSL for three layers: BaseEnv (dynamics, reward, full state), ObsEnv (what the agent is shown) and SkinEnv (rendering as text or image). The DSL also specifies a level generator. Coding agents implement the layers, a level generator, a validator with a max-reward function, and documentation, then self-repair. Semantics can be *aligned* or *inverse* (e.g., "poison restores health").
- **How it makes tasks harder**: Partial observability, inverse or misleading semantics, and level-generator parameters, all without touching the dynamics.
- **Correctness / verification**: Three stages: execution tests; level generation checked by reachability and reward-structure validators; reliability by *differential model testing*. The environment is discarded if a weaker model (GPT-4o-mini ReAct) consistently beats a stronger one (DeepSeek-V3.1), which signals near-random reward.
- **Difficulty control**: Observation policy, skin and generator knobs.
- **Reported results**: About $4.12 per environment. Of 100 themes, 65 passed: 90.0% execution, 96.7% level generation, 74.7% reliability. Human-reviewed themes pass 80% vs 60% for LLM-only themes. AutoEnv-36 (358 levels): seven LLMs score 12–49% normalised reward. *(Corrected example:)* in the FieldDetection "skin-inverse" ablation, the original already uses a 3×3 local patch. Only the displayed values are inverted (3 → 0 …), with a legend claiming "0 = Critical". Gemini-2.5-Flash averaged 26.67% even when warned about inversions.
- **Limitations / failure modes**: Built for evaluation and cross-environment learning, not large-scale RL.
- **How to reuse with easy seed tasks**: Re-render solved procedural seeds under partial observability or inverted labels, with dynamics and checker unchanged. Use weak-vs-strong differential testing to reject generated environments whose reward is essentially random.

### Simulator-fidelity diagnostics — EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation (Liu et al., 2026); Simulated Customers Never Walk Away: Decision Fidelity of LLM User Simulators Measured Against Real Purchase Outcomes (Chen, 2026)
Link: https://arxiv.org/abs/2605.07247 ; https://arxiv.org/abs/2606.20708
- **Mechanism**:
  - *EnvSimBench.* 400 self-contained samples over 167 environments, derived from EnvScaler environments. Each gives the before-state, the tool call and the tool's *implementation code*, and asks for the observation and after-state. Samples are stratified by action outcome, argument cardinality and the number of state variables changed: 0 (80 samples), 1–2 (50), 3–6 (200), 7–12. Its "constraint-driven" paradigm makes every step independently checkable.
  - *Simulated Customers.* Teacher-forced probes over 2,790 production sales conversations, 793 of them with verified payment outcomes.
- **How it makes tasks harder**: Not applicable. Both are diagnostics.
- **Correctness / verification**: Programmatic transition labels; real purchase outcomes.
- **Difficulty control**: The number of simultaneous state changes is a difficulty axis for *simulators*.
- **Reported results**: The state-change cliff appears in all evaluated frontier models and is "orthogonal to model scale and general reasoning ability". Worse, simulators can produce "superficially correct feedback" together with incorrect state transitions, "silently corrupting training signals". A 4B constraint-driven simulator beats frontier LLMs on configuration match, raises EnvScaler synthesis yield by 6.8% and cuts cost by more than 90%. For user simulators: eventual buyers are reproduced almost exactly (depth bias +0.09), but non-buyers are inflated toward purchase (+0.40). Their resistance falls from 25.1% to 13.5% and deliberation rises from 21.9% to 40.1%. The effect replicates with a DeepSeek simulator. Explicitly allowing disengagement cuts marginal bias five-fold but barely moves the outcome-conditioned contrast.
- **Limitations / failure modes**: Diagnostic only.
- **How to reuse with easy seed tasks**: Before trusting a simulator on hardened tasks, measure its accuracy on transitions that change many variables and its disengagement rate. Keep multi-variable updates in code.

### G-Zero — G-Zero: Self-Play for Open-Ended Generation from Zero Data (Huang et al., 2026)
Link: https://arxiv.org/abs/2605.09959
- **Mechanism**: A Proposer, trained with GRPO, writes ⟨question, hint⟩ pairs. Its reward is Hint-δ = the per-token mean of [log π_G(a_t | q, a_<t) − log π_G(a_t | q, h, a_<t)] over the frozen Generator's *unassisted* answer, minus length and BLEU-duplication penalties. δ is large only when the query is hard *and* the hint is informative. The Generator is then trained with length-normalised DPO: the hint-assisted response is "chosen" and the unassisted one "rejected".
- **How it makes tasks harder**: The Proposer searches for the Generator's blind spots.
- **Correctness / verification**: No verifier. Filters keep only lower-half-δ pairs, because high-δ pairs break DPO's implicit KL constraint. They also drop length inflation above 2.5×, chosen responses outside 100–10,000 characters, and responses with zlib compression ratio below 0.15 (repetitive). A best-iterate guarantee is proved for an idealised variant.
- **Difficulty control**: δ maximisation plus a δ band.
- **Reported results**: Two rounds. Qwen3-8B-Base average 33.95 → 35.43, AIME25 7.19 → 12.40. Llama-3.1-8B-Instruct 42.77 → 43.90. Over 70% of the DPO pool is non-verifiable (advice, writing and similar). R-Zero gains on math but loses on conversation and instruction-following evaluations.
- **Limitations / failure modes**: Small gains. Verbosity is an exploit that needs penalties.
- **How to reuse with easy seed tasks**: For non-verifiable seeds, measure difficulty as the shift a self-generated hint causes in the model's answer distribution, and train on moderate-δ pairs.

### Multi-Agent Evolve (+ Language Self-Play) — Multi-Agent Evolve: LLM Self-Improve through Co-evolution (Chen et al., 2025); Language Self-Play For Data-Free Training (Kuba et al., 2025)
Link: https://arxiv.org/abs/2510.23595 ; https://arxiv.org/abs/2509.07414
- **Mechanism**:
  - *MAE.* Proposer, Solver and Judge are instantiated from one LLM (Qwen2.5-3B-Instruct) and trained with Task-Relative REINFORCE++. Proposer reward = ⅓·quality (judge) + ⅓·difficulty + ⅓·format, where R_difficulty = 1 − the mean judge score of N Solver answers. Only questions with judge quality ≥ 0.7 enter the pool.
  - *LSP.* One model (Llama-3.2-3B-Instruct) plays Challenger and Solver via prompts. The Solver's advantage is R − V(q). The Challenger's advantage is V̄ − V(q), so it is rewarded when the Solver's value on its query is below average. KL regularisation "prevents Challenger from mindlessly generating adversarial sequences". A quality self-reward from the reference model is added to both players, which makes the game non-zero-sum.
- **How it makes tasks harder**: A difficulty reward against the current Solver.
- **Correctness / verification**: Judge-based only.
- **Difficulty control**: The difficulty term, quality floors, KL.
- **Reported results**: MAE improves the average by 4.54%. SFT on the seed data with ground truth *degraded* results (53.87 vs base 55.33). LSP improves instruction-following, math and coding with self-play alone. With an off-the-shelf reward model (OpenAssistant DeBERTa), the Solver hacked by answering most queries in Python.
- **Limitations / failure modes**: Judge capability caps difficulty. Collapse of one role spreads: e.g., a Proposer producing only open-ended writing questions.
- **How to reuse with easy seed tasks**: Without a verifier, pair a difficulty reward with a strict quality gate and a KL anchor on the proposer, and monitor the topic mix of its proposals.

### R-Few — Guided Self-Evolving LLMs with Minimal Human Supervision (Yu et al., 2025)
Link: https://arxiv.org/abs/2512.02472
- **Mechanism**: At each rollout the Challenger is given k ∈ [0, 5] randomly sampled human anchor examples (1% or 5% of WebInstruct's 232K). It is rewarded for moderate uncertainty and for embedding similarity to the anchors. The Solver trains on synthetic and human items whose uncertainty falls between two quantiles; human anchors are up-weighted (λ = 2.0) against forgetting. With k = 0 the method reduces to R-Zero.
- **How it makes tasks harder**: An uncertainty-targeted challenger, kept on-distribution.
- **Correctness / verification**: Majority-vote pseudo-labels (as in R-Zero) plus human-labelled anchors.
- **Difficulty control**: The quantile window [τ_low, τ_high].
- **Reported results**: Qwen3-8B-Base gains +3.0 over R-Zero on math. It reaches 55.1 (1% anchors) and 56.7 (5%) versus General-Reasoner's 56.0, which used 232K human examples. Qwen3-4B-Base: base 41.9, R-Zero 48.2.
- **Limitations / failure modes**: Pseudo-labels are unreliable on the hardest items.
- **How to reuse with easy seed tasks**: Use your easy seeds as in-context anchors so the challenger produces harder, on-distribution items, and keep mixing seeds into solver batches.

### SCOPE — SCOPE: Self-Play via Co-Evolving Policies for Open-Ended Tasks (Kwan et al., 2026)
Link: https://arxiv.org/abs/2605.31433
- **Mechanism**: Starting from a corpus document and a task type, the Challenger retrieves more evidence and writes a task that needs multi-turn retrieval. A *frozen* copy of the initial model acts as Judge. It applies binary quality gates (including source relevance), writes weighted rubrics *from the source document*, and grades N rollouts from the previous Solver. Challenger reward = format + 𝟙[gate]·f_diff(ḡ; τ), with f_diff = max(0, 1 − |ḡ − τ| / min(τ, 1 − τ)) and τ = ½. Tasks outside [0.2, 0.8] are discarded. Solver reward = a rubric score with a cosine length penalty + format + search terms.
- **How it makes tasks harder**: Tasks require evidence the Solver must retrieve, and the difficulty target moves with the Solver.
- **Correctness / verification**: Rubrics come from a source document the Solver never sees. Without the gates, "the Challenger degenerates to producing generic tasks unrelated to the source document". The length penalty blocks rewards for verbosity.
- **Difficulty control**: τ plus the window.
- **Reported results**: Up to +10.4 on eight open-ended benchmarks (Qwen2.5-7B: 24.4 → 34.8). Matches or beats GRPO trained on about 9K curated prompts. Up to +13.8 on seven held-out short-form QA benchmarks. A frozen Challenger stops improving after the first iteration. "Rubric generation quality, not grading capacity, is the bottleneck."
- **Limitations / failure modes**: Needs a corpus. Rubric quality caps progress.
- **How to reuse with easy seed tasks**: Harden open-ended seeds by requiring evidence from hidden documents, and have a frozen judge write rubrics from those documents.

### SpyRL / RLSVR — From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement (Wang et al., 2026; COLM 2026)
Link: https://arxiv.org/abs/2607.23802
- **Mechanism**: The environment samples an instance and a spy index u. With n = 5 players, civilians get the full input and the spy gets a copy with one continuous span masked: 20% for GovReport summaries and WritingPrompts stories. For math (Nemotron-CC-Math), 40% of the source text is masked and players must *formulate and solve* a question from what remains. Everyone performs the task, then everyone votes on who the spy is. Performer rewards are zero-sum: r_u = −β(m_u − m̄_c), and civilians share the complement minus a within-group penalty λ(m_cj − m̄_c). Role-advantage estimation corrects the imbalance between roles. The performing and detection stages are trained alternately.
- **How it makes tasks harder**: Detection gets harder as the spy imitates civilians and civilians must stand out through quality.
- **Correctness / verification**: The detection reward is exact because the environment set the latent spy index.
- **Difficulty control**: Masking ratio (20% vs 40% were nearly equivalent for summarisation), number of players, co-evolution.
- **Reported results**: Qwen3-8B wins 75.4% (summarisation) and 77.3% (creative writing) of GPT-4o pairwise comparisons against the base model. Math: +8.97% (Qwen3-4B) and +6.16% (Qwen3-8B) across seven benchmarks.
- **Limitations / failure modes**: Detectability is a proxy for quality, and stylistic tells could be rewarded.
- **How to reuse with easy seed tasks**: Make a verifiable variant of a non-verifiable seed by injecting a known latent (a withheld span, a perturbed fact) and rewarding detection. SpyRL's math variant also works as a *task generator*: formulate a problem from a partially masked document.

## Complexification operators from this area

1. **Variance-targeted generator reward (ZPD band)**
   - *What it does:* Selects, rewrites or trains the generator on the current policy's success rate p̂, with the objective peaking at p̂ ≈ 0.5 (maximum Bernoulli variance and GRPO signal). The same idea appears as variance-based seed selection, a bell-shaped α reward, a 50%-peaked scaffolder reward, a [0.2, 0.8] window, CR-targeted sampling, and a percentile band of teacher-forcing loss when no verifier exists.
   - *Easy → hard:* A WebShop "buy X under $30" task solved 16/16 becomes "cheapest X in colour Y, ≥4 stars, delivered by date D, apply the coupon if eligible", kept if solved about 8/16.
   - *Keeping it verifiable:* The generator never grades; the agent's reward comes from an executable checker or frozen verifier. Exclude noisy batches (GenEnv drops |p̂ − α| > 0.1). Regularise the generator with KL or RWR.
   - *Sources:* DreamGym, GenEnv, STRETCH, SCOPE, SEAD, UI-Simulator-Grow, LiteResearcher, DeepSeek-V4.1 (difficulty reward).
2. **Directional rewrite with accuracy-band acceptance**
   - *What it does:* When accuracy leaves [0.2, 0.8], an LLM rewrites the task toward harder or easier using critic or near-miss summaries. The rewrite is accepted only if accuracy moves in the intended direction and stays inside the band.
   - *Easy → hard:* "Put a clean mug in the cabinet" (acc 1.0) becomes "put two clean mugs and a heated potato in different cabinets, then turn off the lamp".
   - *Keeping it verifiable:* Accept a harder rewrite iff α_low < acc(q′) < acc(q), so solvability is witnessed by the policy. For GUI or code, ship new verifier files or tests validated against reference solutions.
   - *Sources:* RLAnything.
3. **Failure-signal-conditioned re-exploration and failure replay**
   - *What it does:* Mine rollouts and deployment data for forgetting, boundary, rare-pattern or turn-level failures. Re-explore around those states, or rebuild the failure context, and abstract new tasks from what was witnessed.
   - *Easy → hard:* A reliably solved AppWorld task becomes a task built around a rarely used paginated contact search followed by a batch update on which the agent recently regressed.
   - *Keeping it verifiable:* Re-execute the abstracted solution in the real environment. Do *not* accept "execution failed but reward was positive" (the CoEvolve loophole).
   - *Sources:* CoEvolve, SEAL (diagnosis), DeepSeek-V4.1 (failure replay through mocked tools).
4. **Simulator perturbation injection (oracle-preserving)**
   - *What it does:* Replay a solved task in a simulator instructed to add realistic adversity (intermittent API errors, pagination, incomplete intermediate results, partial batch failures, snippets that withhold the answer) while the goal and ground-truth end state stay fixed.
   - *Easy → hard:* "List open tickets for customer C and close duplicates" with one clean response becomes: `list_tickets` returns 10 per page, one call times out, and the batch close partially fails and must be retried.
   - *Keeping it verifiable:* Give the simulator a summary of the hidden state from a *verified* trajectory and state the required final state. Check the end state with code or an anchored rubric. See note 06's COVERT for the code-backed version.
   - *Sources:* Qwen-AgentWorld, MDLM world models, COVERT (cross-reference).
5. **Fictional-world grounding / answer-source removal**
   - *What it does:* Answers exist only inside a self-consistent invented database (SQL-derived), or the page an answer came from is deleted from the corpus. Either way the model cannot answer from memory or in one hop.
   - *Easy → hard:* "Best-selling phone of 2024?" (memorised) becomes "Among 2029 brand-B phones with model numbers starting X-, which gained the most quarter-over-quarter share in Q3?", answerable only through partial snippets from the fictional index. Alternatively, delete the source page of a Wikipedia QA so the answer must be assembled from other pages.
   - *Keeping it verifiable:* SQL-executed answers. Retrieval checks that facts are unsearchable on the real web. For source masking, re-verify that the answer is still reachable, e.g., with a solver pass rate between 1/8 and 7/8.
   - *Sources:* Qwen-AgentWorld, LiteResearcher.
6. **Information withholding / underspecification / implicit intent**
   - *What it does:* Move details from the instruction into hidden state (a private user profile, an abstract goal), or rewrite queries with implicit references, compressed steps, ambiguity and secondary goals.
   - *Easy → hard:* "Book AA123 on 3 May for Alice, seat 12A, card ending 4411" becomes "Can you sort out my trip for the conference?", with details released by a simulated user only when asked.
   - *Keeping it verifiable:* Grade on subgoals, required clarifications and forbidden actions against the original detailed workflow. Keep the hidden profile fixed per task. With EnvFactory's refinements, the executable ground-truth end state does not change.
   - *Sources:* SynthAgent, EnvFactory, WebWorld (Abstract-and-Instantiate).
7. **Hint and scaffold stripping ladder**
   - *What it does:* Build rungs of the same task with decreasing hints or interface cues. Train on the rung near 50% success, and for saturated tasks remove the cues entirely (including informative error messages).
   - *Easy → hard:* "Use Insert → PivotTable; put Promotion in Column Fields" becomes the bare objective, with generic errors and no schema affordance cues.
   - *Keeping it verifiable:* Hints change only prompt or feedback text, never the checked end state.
   - *Sources:* Environment Tuning (stage 4), CuES (ladder), RLAnything (hints), SEAL (reversed), G-Zero (Hint-δ as a difficulty measure).
8. **Adversarial initial-state sampling with an outcome-neutral simulator**
   - *What it does:* A sampler or learned controller chooses the simulated user's or environment's initial state. The component that decides the outcome stays frozen and realistic.
   - *Easy → hard:* A cooperative, trusting customer becomes a low-trust, irritated, low-cooperation one, sampled because CR ≈ 0.5 for that profile.
   - *Keeping it verifiable:* Never train the grader or the episode-ending component. Validate simulator realism against real outcomes (disengagement rates).
   - *Sources:* SEAD; Simulated Customers (fidelity caveat).
9. **Explore-then-question with entity/operation scaling and composition**
   - *What it does:* Explore first, write tasks whose solutions were witnessed, and scale the number of entities, attributes and operations, or chain witnessed segments.
   - *Easy → hard:* "Show my last order" becomes "For each March order over $50, return items rated below 3 stars and email the summary to my manager".
   - *Keeping it verifiable:* Extract the reference from the trace and re-execute it; replay the concatenated trace for chained tasks.
   - *Sources:* AgentEvolver, CuES, CoEvolve, UI-Simulator.
10. **Policy-written generator configuration**
    - *What it does:* The current checkpoint reads its failure breakdown and the generator's knobs, and writes the next stage's parameter mixture.
    - *Easy → hard:* MAPF with 2 agents on 3×3–5×5 grids becomes 8×8–10×10 maps with a higher wait ratio where failures show collisions, while configurations that already work are kept.
    - *Keeping it verifiable:* Instances come from generators with exact solvers or checkers (Conflict-Based Search, game engines, NavMesh).
    - *Sources:* From Trainee to Trainer, EnvGen, SimWorld Studio.
11. **Observation and semantic transformation**
    - *What it does:* Keep dynamics and checker fixed; change observability, rendering, or label semantics, including inverted labels.
    - *Easy → hard:* A FieldDetection grid with honest labels becomes the same grid with displayed intensities inverted and a legend claiming "0 = Critical".
    - *Keeping it verifiable:* Underlying state and max-reward validator untouched. Use weak-vs-strong differential tests to drop environments with essentially random reward.
    - *Sources:* AutoEnv.
12. **Equilibrium best-response task mutation**
    - *What it does:* Keep a payoff matrix of checkpoints × tasks and mutate tasks to minimise the success of the *equilibrium mixture*, not only of the latest checkpoint.
    - *Easy → hard:* A maze all policies solve becomes a mutated maze with a key–door chain and a chokepoint that the MSNE mixture solves least often.
    - *Keeping it verifiable:* Reachability checks (e.g., an occupancy grid inflated by the agent's radius); execution-measured success.
    - *Sources:* COvolve; LSP (V̄ − V(q) challenger advantage).
13. **Latent-variable injection (verifiable games for open-ended tasks)**
    - *What it does:* Turn a non-verifiable task into a game whose ground truth is a latent chosen by the environment (who received the degraded input). Reward detection or indistinguishability.
    - *Easy → hard:* "Summarise this report" (judge-scored) becomes a five-player game in which one player's input has 20% masked and all players vote on the spy.
    - *Keeping it verifiable:* Detection is exact. Watch for stylistic tells that let players detect the spy without regard to quality.
    - *Sources:* SpyRL/RLSVR.
14. **Build → self-test → leak-scrub → multi-solver → inspector → repair**
    - *What it does:* A multi-agent pipeline for repository tasks. Choose a start commit, design complex implementation directions with F2P and P2P points, build and self-test in a container, scrub solution traces, have several different agents attempt it, have an inspector audit environment plus trajectories, and repair or re-tune evaluation points.
    - *Easy → hard:* "Fix a one-line bug with one failing test" becomes "implement a multi-module feature from a chosen commit, with several F2P tests and P2P regression tests re-tuned until only some solvers pass".
    - *Keeping it verifiable:* Separate builder and runtime accounts; remove residue from image layers; eBPF network allowlists and AppArmor file/socket controls; re-audit on every new RL run.
    - *Sources:* DeepSeek-V4.1-Flash §5.1.1; DSec.
15. **Harness and scaffold randomisation**
    - *What it does:* Run the same tasks under differently composed harnesses (tool schemas, prompts, context management, subagents, memory).
    - *Easy → hard:* A SWE task always run in one scaffold becomes the same task under Claude-Code-, Codex- or OpenClaw-style module combinations and different context policies.
    - *Keeping it verifiable:* The verifier is independent of the harness. Normalise trajectories to one schema.
    - *Sources:* Kimi K3 (white-box environment), DeepSeek-V4.1 (multi-scaffold RL plus checkpoint merging).
16. **Public/hidden verifier split under a submission budget**
    - *What it does:* The agent iterates against a diagnostic public verifier. The reward comes from hidden held-out scenarios, with penalties for extra submissions.
    - *Easy → hard:* "Make these 5 visible tests pass" becomes "replicate this black-box system": the visible oracle queries and public checks guide the work, and hidden scenarios decide the reward.
    - *Keeping it verifiable:* Isolate agents from verifier code and grade the final environment state.
    - *Sources:* Kimi K3 (AET).

## Insights & pitfalls

- **Update to note 12 (frontier practices).**
  - *"No visible task-synthesis detail."* True for DeepSeek-V4 (2606.19348), which has only DSec infrastructure and evaluation text, but superseded by DeepSeek-V4.1-Flash §5.1.1. The claim that detail is "shrinking" should become "uneven": V4.1 and K3 are detailed, V4 is not.
  - *"No report closes the loop from failures to the generator."* Now partly closed. V4.1 trains the constructor on difficulty and correctness, re-audits tasks with fresh RL trajectories, replays failures, and re-tunes evaluation points that are too easy. Caveats: the replayed failures come from employees and deployment rather than the current checkpoint, and there is no formula or ablation.
  - *"Bootstrapping hardness requires a frontier-scale synthesiser."* V4.1 uses the model being trained as its constructor.
  - *"No frontier report uses a learned pass-rate predictor."* Still true for V4.1 and K3. PROPEL (notes 02 and 11) is an academic counterexample.
  - *Simulator fidelity.* "No standard test that simulated hardness matches real hardness" still holds, but intrinsic simulator benchmarks now exist: EnvSimBench, WebWorld-Bench and AgentWorldBench.
  - *Exploit taxonomy.* DSec §6.4 now gives a concrete catalogue.
- **Everyone targets about 50%. The details that matter are elsewhere:** (a) exclude noisy batches from generator updates; (b) accept a rewrite only if difficulty moves in the intended direction *and* the task stays solvable; (c) prefer RWR or KL-anchored updates over unconstrained RL on the generator; (d) refresh the band every stage. STRETCH's ablation (0.5 > 0.8, 0.2) is the cleanest evidence for the 50% target.
- **A generator trained against the policy beats more static teacher data.** GenEnv at 1× data beat offline Gemini-2.5-Pro augmentation at 3.3× (0.458 vs 0.438), and beat an untrained simulator by 12.3%. CoEvolve kept improving (0.21 → 0.35) where a static baseline peaked and fell. DreamGym lost about 6 points without its generator. UI-Simulator-Grow matched a 70B model with 66% of the data.
- **The trained policy is a better task designer than the base model.** From Trainee to Trainer: the RL checkpoint beat the base model and larger proprietary designers. Good redesigns cite failure evidence and keep configurations that work. DeepSeek likewise trains the model under RL as its own constructor. Implication: re-run the proposer from the latest checkpoint rather than a frozen teacher.
- **Freeze or rule-anchor every grader.** Trained user simulators collapse to accept/reject (SEAD). Learned reward models get hacked within about 20 steps if their precision is low (SWE-World, where CoT raised precision from 0.609 to 0.754). Judges get self-praised (Qwen-AgentWorld, fixed with a 9:1 rubric:rule mix, exact-match scoping and tag extraction). Reward models get Python-dumped (LSP). Challengers drift to generic tasks without source gates (SCOPE).
- **Simulated difficulty is not real difficulty.** Simulators are sycophantic (WebWorld), explain more than real environments do (Simia's OfficeBench sim-RL beat real RL partly for this reason), and user simulators do not walk away. A task that is "50% hard" in simulation may be easy, or impossible, in reality.
- **Complexification pushes simulators off the cliff.** More entities, chained operations and batch updates mean more simultaneous state changes, which is exactly where EnvSimBench shows every model collapsing. Keep state transitions in code or a database (AWM, EnvScaler, EnvFactory, COVERT), give the simulator the before-state and tool code (constraint-driven), or give it a hidden-state summary from a verified trajectory (Qwen-AgentWorld MCP). Qwen-AgentWorld's own conclusion: "State is the bottleneck."
- **Steering is a prerequisite, not a bonus.** Without grounded control instructions, simulated MCP RL was "too noisy to yield any gain". A prompted general model as simulator gave marginal gains; the dedicated trained simulator gave +4.3 / +7.1.
- **Controlled simulation can shape behaviour in ways real environments cannot.** Search snippets that withhold answers raised page-extraction calls (2.5 → 4.0), while real-search RL lowered them (→ 1.5). Real environments often allow shortcuts, and perturbation operators can remove them.
- **Several popular operators make tasks easier; reverse them for saturated sets.** SEAL cues, CuES hints, Environment Tuning augmentation, RLAnything "easier" rewrites, Simia's explanatory feedback. Environment Tuning's stage 4 (same tasks, uninformative errors) is a ready-made recipe for the reversal.
- **Explore-then-question is safe but conservative.** Witnessed solutions guarantee solvability and make references cheap, but the tasks are biased toward the explorer's competence. Layer composition, hint stripping, perturbation or source masking on top, and re-verify by replay.
- **Stabilisers are mandatory for self-generated curricula.** Golden replay (STRETCH format collapse by epoch 4 without it). Human anchors (R-Few). A capped synthetic fraction (DreamGym). KL (GenEnv, LSP). A quality floor of 0.7 (MAE). Lower-half-δ filtering (G-Zero). Replay of representative earlier tasks (UI-Simulator-Grow).
- **In verifier-free self-play, hidden grounding beats judge scores.** SCOPE (rubrics from a source document the solver never sees) and SpyRL (a known latent) give the most robust signals. SCOPE finds that rubric *generation*, not grading, is the bottleneck. Gains remain modest at 3–8B over 2–5 rounds.
- **The attack surface grows with task difficulty.** DSec shows agents on hard tasks attacking the sandbox itself: sockets, logs, `/bin/bash`, filesystem ioctls, package mirrors, Go proxies. Treat "the agent found the answer elsewhere" as a verifier failure. Kimi's public/hidden verifier split and DeepSeek's inspector that reads trajectories are the two frontier answers.

## Open problems & research opportunities

- **An open replication of DeepSeek-V4.1's learned constructor.** How should correctness of a (problem, environment, verifier) triplet be scored as a reward? How often should the constructor be retrained? How much of the gain comes from the constructor versus scale and scaffold diversity? A matched-compute ablation (constructor trained on difficulty × correctness vs a prompted constructor vs a static pool) is missing.
- **Calibrating simulated hardness against real hardness.** Build paired benchmarks (the same tasks in a database-backed environment and in an LLM simulator, with and without perturbations) and report pass-rate correlation per band, stratified by the number of state changes.
- **Hybrid simulators.** Code or database state machines for dynamics, LLMs for rendering and user behaviour, with fidelity reported at 0 / 1–2 / 3–6 / 7–12 state changes, which is where complexification operators operate.
- **Cheap difficulty predictors for long-horizon agentic tasks.** Every co-evolution method here pays for full rollouts per candidate. Activation probes (PROPEL) or teacher-forcing-loss proxies (UI-Simulator-Grow) are unproven for multi-turn tool and GUI tasks.
- **Lifecycle re-audit as a method.** Public detectors for tasks that have become broken or hackable (sudden 0 → 100%, reward/length anomalies, cross-solver disagreement), and statistics on how quickly pools decay.
- **Outcome-neutral adversarial simulators.** SEAD's recipe (adversarial choice of initial state, frozen role-player) works, but a general method for training adversarial users and environments that stay realistic and outcome-neutral is missing. User-simulator decision fidelity (disengagement) resists prompting fixes.
- **Population and equilibrium curricula at LLM-weight scale.** COvolve's MSNE prevents forgetting in small code domains. No study compares best-response-to-latest with best-response-to-equilibrium-mixture for synthetic task pools used to train LLM weights.
- **Scaling verifier-free self-play.** Evidence stops at 3–8B and a few rounds. Does Hint-δ, document-grounded rubric generation, or latent-variable games keep improving, and how do we stop rubric-generation quality from capping difficulty?
- **Diversity telemetry under co-evolution.** Most papers report success-rate or length curves only. Coverage and entropy of the generated task distribution over rounds would catch early collapse (e.g., writing-only proposers) before downstream damage.
- **Red-teaming environments that models build.** As agents build environments for other agents (DSec's `pack_diff`, V4.1's constructor), leakage channels multiply. A shared exploit benchmark and a standard pre-RL red-team protocol for generated environments do not exist.

## References

1. DeepSeek-AI (Xu, A. et al.) (2026). *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*. arXiv:2609.19969. https://arxiv.org/abs/2609.19969
2. Huang, J., Tang, H., Chen, J., Liu, Y., Chen, Y. et al. (2026). *DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale*. arXiv:2609.22978. https://arxiv.org/abs/2609.22978
3. DeepSeek-AI (Xu, A. et al.) (2026). *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*. arXiv:2606.19348. https://arxiv.org/abs/2606.19348
4. Kimi Team (Bai, T., Bai, Y. et al.) (2026). *Kimi K3: Open Frontier Intelligence*. arXiv:2607.24653. https://arxiv.org/abs/2607.24653
5. Moshkov, I., Ge, S., Armstrong, G., Du, W., Mahdavi, S., Gitman, I. (2026). *An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics*. arXiv:2609.10712. https://arxiv.org/abs/2609.10712
6. Li, Y., Zhan, R., Zhang, H. et al. (2026). *Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling*. arXiv:2605.13301. https://arxiv.org/abs/2605.13301
7. Mehta, S., Ritchie, L., Garre, S., Niebres, I., Heiner, N., Chen, E. (2026). *EnterpriseBench Corecraft: Training Generalizable Agents on High-Fidelity RL Environments*. arXiv:2602.16179. https://arxiv.org/abs/2602.16179
8. Chen, Z., Zhao, Z., Zhang, K., Liu, B., Qi, Q., Wu, Y. et al. (2025). *Scaling Agent Learning via Experience Synthesis* (DreamGym). ICLR 2026; arXiv:2511.03773. https://arxiv.org/abs/2511.03773
9. Guo, J., Yang, L., Chen, P., Xiao, Q., Wang, Y., Juan, X. et al. (2025). *GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators*. arXiv:2512.19682. https://arxiv.org/abs/2512.19682
10. Wang, Y., Xie, T., Shen, K., Wang, M., Yang, L. (2026). *RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System*. ICML 2026; arXiv:2602.02488. https://arxiv.org/abs/2602.02488
11. Yu, Y., Lee, M., Feng, Y. (2026). *STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution*. arXiv:2609.18642. https://arxiv.org/abs/2609.18642
12. Dai, Y., Gao, N., Zhang, W., Wang, J., Luo, Z. et al. (2026). *SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue*. arXiv:2602.03548. https://arxiv.org/abs/2602.03548
13. Sygkounas, A., Hazra, R., Persson, A., Zuidberg Dos Martires, P., Loutfi, A. (2026). *COvolve: Adversarial Co-Evolution of Large-Language-Model-Generated Policies and Environments via Two-Player Zero-Sum Game*. GECCO 2026; arXiv:2603.28386. https://arxiv.org/abs/2603.28386
14. Chen, C., Li, C., Li, Z., Liu, Y., Guo, Z. (2026). *From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning*. arXiv:2606.17682. https://arxiv.org/abs/2606.17682
15. Zala, A., Cho, J., Lin, H., Yoon, J., Bansal, M. (2024). *EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents*. COLM 2024; arXiv:2403.12014. https://arxiv.org/abs/2403.12014
16. Kang, H., Ye, X., Liu, Y. et al. (2026). *SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning*. arXiv:2605.09423. https://arxiv.org/abs/2605.09423
17. Hu, Y., Wen, Z., Liu, X., Wang, P., Zhang, X., Wu, W. (2026). *SEAL: Synergistic Co-Evolution of Agents and Learning Environments*. arXiv:2605.24426. https://arxiv.org/abs/2605.24426
18. Lu, S., Wang, Z., Zhang, H., Wu, Q., Gan, L. et al. (2025). *Don't Just Fine-tune the Agent, Tune the Environment* (Environment Tuning). arXiv:2510.10197. https://arxiv.org/abs/2510.10197
19. Yang, S., Ma, Z., Huang, T., Hu, Y., Wang, Y., Chu, X. (2026). *CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution*. ACL 2026; arXiv:2604.15840. https://arxiv.org/abs/2604.15840
20. Zhai, Y., Tao, S., Chen, C., Zou, A., Chen, Z., Fu, Q. et al. (2025). *AgentEvolver: Towards Efficient Self-Evolving Agent System*. arXiv:2511.10395. https://arxiv.org/abs/2511.10395
21. Mai, S., Zhai, Y., Chen, Z., Chen, C., Zou, A., Tao, S. et al. (2025). *CuES: A Curiosity-driven and Environment-grounded Synthesis Framework for Agentic RL*. arXiv:2512.01311. https://arxiv.org/abs/2512.01311
22. Wang, Y., Yin, D., Cui, Y., Zheng, R., Li, Z. et al. (2025). *LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training* (UI-Simulator). arXiv:2510.14969. https://arxiv.org/abs/2510.14969
23. Li, Y., Inan, H. A., Yue, X., Chen, W.-N., Wutschitz, L., Kulkarni, J. et al. (2025). *Simulating Environments with Reasoning Models for Agent Training* (Simia). arXiv:2511.01824. https://arxiv.org/abs/2511.01824
24. Xiao, Z., Tu, J., Zou, C., Zuo, Y., Li, Z., Wang, P. et al. (2026). *WebWorld: A Large-Scale World Model for Web Agent Training*. arXiv:2602.14721. https://arxiv.org/abs/2602.14721
25. Zuo, Y., Xiao, Z., Sheng, L., Huang, F., Tu, J., Liu, Y. et al. (2026). *Qwen-AgentWorld: Language World Models for General Agents*. arXiv:2606.24597. https://arxiv.org/abs/2606.24597
26. Qu, B., Li, W., Pan, B., Zhang, J., Liu, Z. et al. (2026). *LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent*. arXiv:2604.17931. https://arxiv.org/abs/2604.17931
27. Sun, S., Song, H., Huang, L., Jiang, J., Le, R., Lv, Z. et al. (2026). *SWE-World: Building Software Engineering Agents in Docker-Free Environments*. arXiv:2602.03419. https://arxiv.org/abs/2602.03419
28. Deshpande, D. (2026). *Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL*. arXiv:2607.16204. https://arxiv.org/abs/2607.16204
29. Xu, Z., Yao, Z., Chen, Y., Guo, Y., Lu, Z., Lu, Y. et al. (2026). *EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning*. arXiv:2608.06197. https://arxiv.org/abs/2608.06197
30. Fang, T., Zhang, H., Zhang, Z., Ma, K., Yu, W., Mi, H. et al. (2025). *WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model*. EMNLP 2025; arXiv:2504.21024. https://arxiv.org/abs/2504.21024
31. Liu, Y., Wang, J., Wang, H., Li, W. (2026). *CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents*. EMNLP 2026; arXiv:2606.02372. https://arxiv.org/abs/2606.02372
32. Lu, N., Lin, B., Liu, S., Wu, J., Lv, H., Wei, Y. et al. (2026). *Policy and World Modeling Co-Training for Language Agents* (PaW). EMNLP 2026; arXiv:2606.02388. https://arxiv.org/abs/2606.02388
33. Lyu, Y., Wang, C., Shen, L., Huang, J., Xu, T. (2026). *Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards* (SynthAgent). arXiv:2601.22511. https://arxiv.org/abs/2601.22511
34. Xu, M., Wang, Z., Deng, M., Li, Z., Yang, Z., Zhu, X. et al. (2026). *EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL*. arXiv:2605.18703. https://arxiv.org/abs/2605.18703
35. Zhang, J., Peng, Y., Kong, F., Yang, C., Wu, Y., Yu, Z. et al. (2025). *AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning*. arXiv:2511.19304. https://arxiv.org/abs/2511.19304
36. Liu, Y., Hui, T., Zhang, W., Sun, L., Su, N., Wang, J. et al. (2026). *EnvSimBench: A Benchmark for Evaluating and Improving LLM-Based Environment Simulation*. arXiv:2605.07247. https://arxiv.org/abs/2605.07247
37. Chen, L. (2026). *Simulated Customers Never Walk Away: Decision Fidelity of LLM User Simulators Measured Against Real Purchase Outcomes*. arXiv:2606.20708. https://arxiv.org/abs/2606.20708
38. Huang, C., Liu, H., Zheng, T., Dai, R., Huang, L., Li, J. et al. (2026). *G-Zero: Self-Play for Open-Ended Generation from Zero Data*. arXiv:2605.09959. https://arxiv.org/abs/2605.09959
39. Chen, Y., Wang, Y., Zhu, S., Yu, H., Feng, T., Zhang, M. et al. (2025). *Multi-Agent Evolve: LLM Self-Improve through Co-evolution*. arXiv:2510.23595. https://arxiv.org/abs/2510.23595
40. Kuba, J. G., Gu, M., Ma, Q., Tian, Y., Mohan, V., Chen, J. (2025). *Language Self-Play For Data-Free Training*. arXiv:2509.07414. https://arxiv.org/abs/2509.07414
41. Yu, W., Liang, Z., Huang, C., Panaganti, K., Fang, T., Mi, H. et al. (2025). *Guided Self-Evolving LLMs with Minimal Human Supervision* (R-Few). arXiv:2512.02472. https://arxiv.org/abs/2512.02472
42. Kwan, W.-C., Gema, A. P., Leang, J. O. J., Minervini, P. (2026). *SCOPE: Self-Play via Co-Evolving Policies for Open-Ended Tasks*. arXiv:2605.31433. https://arxiv.org/abs/2605.31433
43. Wang, Q., Shi, J., Wang, H., Wan, K., Wu, Y., Liu, B. et al. (2026). *From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement* (SpyRL). COLM 2026; arXiv:2607.23802. https://arxiv.org/abs/2607.23802
44. Xu, S., Li, S., Liu, X., Liu, T., Li, Y. et al. (2026). *Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning* (COVERT; cross-reference, covered in note 06). arXiv:2604.09813. https://arxiv.org/abs/2604.09813
