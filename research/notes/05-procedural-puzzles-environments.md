# Procedural reasoning environments & logic puzzles with difficulty knobs (RLVR gyms)

*Scope: procedural generator + verifier suites and single-family puzzle generators (logic grids, K&K, SAT, NP-hard optimization, games), adaptive-difficulty controllers, LLM-written environments, and the controlled studies that say which complexification axes matter for SFT/RL. Compiled 2026-09-30. Verification: 28 entries checked against primary sources (arXiv full text, plus READMEs where noted); 17 corrected, 0 dropped, 6 added. Many unverifiable venue claims and several numbers were removed.*

## TL;DR

- **Near-zero GRPO signal on easy tasks comes from static difficulty. Fix it with generators that have unbounded knobs plus an online controller, not with a bigger static "hard" set.** RLVE showed that a static range with a low cap drives the effective-prompt ratio (share of prompts whose rollouts disagree) to 0. Its sliding window (promote at 90% accuracy, keep the last 4 levels) beat even an oracle static range. Other controllers:
  - SCALER: proportional controller plus environment retirement.
  - Frontier Learning: a PLR regret buffer plus mutation, for when knobs interact non-monotonically.
  - INTELLECT-3 (cheapest to adopt): easy/normal/hard pools, and discard groups where every rollout passes or every one fails.
- **Get correctness from solve–verify asymmetry and planted solutions, not from model-written answer keys.** Build the solution first and derive the problem from it: a solved Sudoku grid that is then masked, a Hamiltonian path planted in a graph, satisfiable CNF, operations-research (OR) instances solved before the environment code exists. Or check the answer cheaply: differentiate a proposed antiderivative, check a SAT assignment with PySAT, check feasibility and then compare to a heuristic value.
  - ReSyn's reward ablation (BBH): verifier-based rewards 75.24, code-computed reference 74.94, LLM-generated answers 68.83.
- **Many environments beat many instances, but select them by ability coverage.**
  - ReSyn at about 16K instances (BBH): 400 envs × 40 instances = 75.19; 25 × 640 = 71.20.
  - RLVE: held-out accuracy rose over nested sets of 1, 4, 16 and 256 environments.
  - SCALER: gains grew from 8 to 2,739 environments.
  - Forge: 10 tasks beat 3 tasks on out-of-domain (OOD) average (53.6 vs 53.0).
  - Caveat (AES): 30 environments chosen for ability coverage out of 200 gave +95.6% relative gain; all 200 gave +43.4%.
- **"Harder" should mean orthogonal axes, not just bigger instances.**
  - Depth × distractor complexity: joint coverage beats either axis alone.
  - Harness/scaffold removal.
  - Information hiding: VHD-Play's same solved instances score 0.962 written-out vs 0.204 when the parameters sit behind tools.
  - Problem-type upgrades: decision → search → MaxSAT → minimal unsatisfiable subset (MUS), as in SATQuest.
  - Feasibility → optimality with ratio rewards (NP-Engine/Forge).
- **LLM-written environments now scale, but only behind staged gates.**
  - Scale so far: ReSyn 418; InternBootcamp 704 from about 100 seeds; SCALER 2,739; EvoEnv 840 in 100 steps; DeepSeek-V3.2 1,827; VHD-Play 3,300 at about $0.01–0.03 each.
  - Gates:
    - execution across seeds and levels;
    - determinism;
    - non-triviality;
    - scorer perturbation tests;
    - agreement of multiple reference solutions;
    - a solve-rate-vs-level monotonicity test;
    - a pass band (InternBootcamp keeps 3–85%);
    - any-reject semantic review.
  - Even hand-built gyms contain bugs: an audit found material defects in 13 of 105 Reasoning Gym tasks and 9 SynLogic generators.
- **RL on compositions of known atomic skills teaches genuinely new skills; rejection fine-tuning (RFT) does not.** RL on depth-2 string-function compositions reached about 30% on unseen depth-3 (RFT ≤2.6%). This requires the atoms to be learned first.
- **Transfer to math, code and general reasoning is real but uneven, and it shrinks as the model gets stronger.**
  - Positive: Reasoning Gym MATH +9.7 at 3B; SynLogic AIME24 4.5→19.6 at 32B.
  - None or small: Enigmata saw no general-reasoning transfer on Qwen2.5-32B, and only +0.8 to +1.9 on Seed1.5-Thinking.
  - Negative: K&K-only RL dropped the code average from 67.46 to 56.09.
  - So keep puzzles a minority of the mix, alongside the target domains.
- **Aim for an intermediate pass rate and keep a band of difficulties.**
  - EvoEnv targets 30% pass and chose below 50% because 50%-difficulty environments saturate quickly.
  - LURE's teacher is rewarded min(p, 1−p).
  - Evidence on mixing: uniform mixing beat staged curricula at a fixed budget; Enigmata found easy:medium:hard 1:1:1 beat 2:6:2; NP-Engine (TSP ablation) preferred 5:4:1.
- **Reward design has to change as tasks get harder.**
  - Partial credit for multi-cell outputs: RLVE (x/N)^10; on Logic Puzzle Baron, binary reward collapsed and partial reward was needed.
  - Quality ratios for optimization: in Forge, a fixed reward for mere feasibility invited reward hacking.
  - Solver-derived process rewards (CAST, SPRING) when successes become too sparse.
- **Evaluate on perturbed and held-out generators, not just fresh seeds.**
  - K&K fine-tuning memorizes: near-perfect on training puzzles, brittle under one-statement perturbations.
  - Some "reasoning collapse" results are generator artifacts: River Crossing with N>5 actors and boat capacity 3 is unsolvable, and token limits also played a role.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Reasoning Gym | 2025 | [2505.24760](https://arxiv.org/abs/2505.24760) | 100+ algebra/algorithm/logic/graph/game generators | RL, eval | size/range knobs, structural knobs, curriculum | answer from generator or task-specific score function |
| Reasoning Core (v1–v3) | 2025–26 | [2603.02208](https://arxiv.org/abs/2603.02208) | PDDL, first-order logic (FOL), context-free grammars (CFG), Bayesian nets, equations; v3 has 50 generators | RL, pretraining, SFT | rule/world randomization, continuous difficulty knob | external solvers (planners, provers, parsers, exact inference) |
| SynLogic | 2025 | [2505.19641](https://arxiv.org/abs/2505.19641) | 35 logic tasks | RL | per-task knobs calibrated to reference models | rule-based verifier per task |
| Enigmata | 2025 | [2505.19914](https://arxiv.org/abs/2505.19914) | 36 puzzles, 7 categories | SFT (RFT) + RL | grid size / blanks / mask rate; E/M/H tiers | rule-based verifier per task |
| InternBootcamp | 2025 | [2508.08636](https://arxiv.org/abs/2508.08636) | 1000+ tasks, 8 domains | SFT + RL | LLM-written "bootcamps", task-count scaling | execution-feedback refinement + 0.03–0.85 accuracy filter |
| NPPC | 2025 | [2504.11239](https://arxiv.org/abs/2504.11239) | 25 NP-complete problems | eval | unbounded size ("ever-scaling") | polynomial-time certificate check |
| NP-Engine / Forge | 2025–26 | [2510.16476](https://arxiv.org/abs/2510.16476), [2605.08905](https://arxiv.org/abs/2605.08905) | 10 NP-hard optimization tasks | RL | feasibility→optimality, size/density tiers | exact feasibility check + heuristic reference ratio |
| RLVE / RLVE-Gym | 2025 | [2511.07317](https://arxiv.org/abs/2511.07317) | 400 algorithmic/math/logic/NP envs | RL | integer level d → size; adaptive sliding window | algorithmic verifiers; solve–verify asymmetry |
| SCALER | 2026 | [2601.04809](https://arxiv.org/abs/2601.04809) | CodeContests → 2,739 envs | RL | scale-parameter lifting, proportional controller, env retirement | multiple reference solutions must agree; unique-output problems only |
| Frontier Learning | 2026 | [2609.35426](https://arxiv.org/abs/2609.35426) | RG-style generators | RL | regret-prioritized level buffer, mutation, exploration | inherits generator verifiers |
| INTELLECT-3 difficulty pools | 2025 | [2512.16144](https://arxiv.org/abs/2512.16144) | math/code/logic (SynLogic) envs | RL | offline solve-rate pools + online filter | env verifiers |
| LURE | 2026 | [2608.21871](https://arxiv.org/abs/2608.21871) | PhantomWiki, IFEval-style, ZebraLogic generators | RL | learned "evader" places difficulty at p≈0.5 | programmatic generators and verifiers only |
| SPIRAL (+ Stratagem) | 2025 | [2506.24119](https://arxiv.org/abs/2506.24119) | zero-sum TextArena games | RL | self-play opponent escalation, multi-game | game engine decides win/loss |
| Logic-RL | 2025 | [2502.14768](https://arxiv.org/abs/2502.14768) | Knights & Knaves | RL | people count 2–8, operator nesting 1–4 | unique solution by construction; strict format reward |
| K&K perturbation benchmark | 2024 | [2410.23123](https://arxiv.org/abs/2410.23123) | Knights & Knaves | SFT, eval | statement/leaf perturbation, role flip, renaming | generator re-solves; perturbation must change the solution |
| ZebraLogic | 2025 | [2502.01100](https://arxiv.org/abs/2502.01100) | logic grid CSPs | eval | N×M scaling, clue minimization, Z3-conflict hardness | SAT solver confirms uniqueness after each clue removal |
| ZebraArena | 2026 | [2603.18614](https://arxiv.org/abs/2603.18614) | logic grids with withheld clues + tools | eval (RL-ready) | information hiding, missing-clue count | unique ground truth; constructive query budget K_ref |
| SATURN | 2025 | [2505.16368](https://arxiv.org/abs/2505.16368) | (n,k,l)-SAT | RL | analytic LLM-difficulty D(n,k,l); pass@1-gated curriculum | satisfiable-only instances; assignment checked |
| SATBench | 2025 | [2505.14615](https://arxiv.org/abs/2505.14615) | SAT/UNSAT as story puzzles | eval | clause count, narrative rendering | LLM checks + solver equivalence of back-translated formula |
| SATQuest | 2025 | [2509.00930](https://arxiv.org/abs/2509.00930) | CNF tasks in 5 problem types × 4 formats | eval, RFT | problem-type upgrade, format shift | PySAT checks binary-string answers |
| AutoLogi | 2025 | [2502.16906](https://arxiv.org/abs/2502.16906) | LSAT-style open-ended puzzles | eval, DPO/RFT | constraint expansion/reduction | LLM verifier cross-checked by exhaustive traversal |
| Illusion of Thinking (+ comment) | 2025 | [2506.06941](https://arxiv.org/abs/2506.06941) | Tower of Hanoi, Checker Jumping, River Crossing, Blocks World | eval | compositional size N | simulators check move sequences (with known pitfalls) |
| LogicPro | 2024 | [2409.12929](https://arxiv.org/abs/2409.12929) | LeetCode → text reasoning | SFT | program lifting, many test inputs | run reference program; intermediate variables guide the reasoning trace |
| Game-RL / Code2Logic | 2025 | [2505.13886](https://arxiv.org/abs/2505.13886) | 30 games, 158 multimodal tasks | RL | game-code QA generators, difficulty grades | answers from executing game logic |
| Loong | 2025 | [2509.03059](https://arxiv.org/abs/2509.03059) | 12 domains, code-backed seeds | SFT, RL | few-shot / Self-Instruct / Evol-Instruct from seeds | code-executed answer vs agent chain-of-thought (CoT) answer |
| ReSyn | 2026 | [2602.20117](https://arxiv.org/abs/2602.20117) | 418 LLM-written envs | RL | keyword→env synthesis, 5 levels | reference-free verifiers; Wald test on level vs solve rate |
| EvoEnv | 2026 | [2605.14392](https://arxiv.org/abs/2605.14392) | self-written Python envs from 10 seeds | RL | policy writes envs; target 30% pass; novelty pressure | 5 mechanical gates + 3× any-reject self-review |
| DeepSeek-V3.2 agentic synthesis | 2025 | [2512.02556](https://arxiv.org/abs/2512.02556) | 1,827 tool envs (trip planning etc.) | RL | agent iteratively hardens task, adds tools | solution function must pass co-updated verifier; pass@100>0 |
| VHD-Play | 2026 | [2609.27321](https://arxiv.org/abs/2609.27321) | OR mechanisms as stateful tool envs | RL | solve-first, hide parameters behind tools, size/horizon | reward vs pre-solved u*, u0; replay admission |
| f(g(x)) composition | 2025 | [2509.25123](https://arxiv.org/abs/2509.25123) | string-function compositions | analysis (RL vs RFT) | composition depth, hidden definitions | execute composed functions |
| Depth × Environment Complexity | 2026 | [2605.26934](https://arxiv.org/abs/2605.26934) | synthetic KG worlds | analysis | depth D × complexity T × task family | ground truth is the generated graph |
| AES + HDC | 2026 | [2608.03571](https://arxiv.org/abs/2608.03571) | 200 multimodal agent envs | RL | harness removal H0→H4, state scale, ability-aware selection | unchanged env verifiers |
| Can One Domain Help Others? | 2025 | [2507.17512](https://arxiv.org/abs/2507.17512) | math/code/K&K/Logic Puzzle Baron | analysis | K&K level curriculum + policy refresh; reward shapes | rule verifiers |
| RAGEN / RAGEN-2 | 2025–26 | [2504.20073](https://arxiv.org/abs/2504.20073), [2604.06268](https://arxiv.org/abs/2604.06268) | Bandit, Sokoban, FrozenLake, WebShop | RL (multi-turn) | static → interactive; instance filtering | env rules |
| CAST | 2026 | [2607.25308](https://arxiv.org/abs/2607.25308) | Sokoban, Minesweeper, Rush Hour | RL | unseen-difficulty generalization via solver credit | game solver value changes as turn-level advantage |

## Method notes

### Reasoning Gym — REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards (Stojanovski et al., 2025)
Link: https://arxiv.org/abs/2505.24760 · code https://github.com/open-thought/reasoning-gym · NeurIPS 2025 Spotlight (per arXiv comment)
- **Mechanism**: 100+ data generators, each paired with a verifier. They span algebra, arithmetic, computation, cognition, geometry, graph theory, logic and common games. Each task is a seeded generative algorithm whose parameters "continuously modulate problem characteristics", in three classes:
  - *Difficulty* parameters, e.g., graph node counts, polynomial degree, word length.
  - *Structural* parameters, e.g., dimensionality, constraint types, proof depth.
  - *Stylistic* parameters, e.g., variable names, number formats, framing. These change presentation, not difficulty.
  
  The appendix ships paired easy/hard configs (e.g., coordinate or complex-number ranges widened from ±10 to ±100).
- **How it makes tasks harder**: raise difficulty parameters; change structural parameters; curriculum mode.
- **Correctness / verification**: answers are computed by the generator. A task-specific scoring function checks responses, which lets puzzles with many valid answers be checked by constraints.
- **Difficulty control**: config ranges. In the curriculum experiment, training starts at the easiest level and moves up when performance exceeds 70% over 20 steps (Qwen2.5-3B-Instruct, GRPO). The baseline samples all levels uniformly.
- **Reported results**:
  - Zero-shot on hard configs: o3-mini 63.5%, DeepSeek-R1 59.5%, Grok 3 Mini 55.1%, Llama 4 Maverick 41.5%, Claude 3.5 Sonnet 40.3%, Gemma 3 27B 20.3%.
  - Easy→hard drop for o3-mini: code −71.9%, graphs −33.8%, geometry −33.1%, algorithms −25.6%.
  - Curriculum vs uniform levels: Spell Backward +40.67 (word length 4), Mini Sudoku +13.33 (8–10 empty cells), Count Primes +26.67 (range 100–500, table value). On Count Primes the curriculum run never left level 1 and still won.
  - Cross-domain (Acc@3): algorithmic training raised algebra 23.83→52.89 (+29.1) and geometry 0.83→23.17 (+22.3), but ARC fell by 2.3.
  - RG-Math (800 GRPO steps on algebra/arithmetic/geometry): MATH 48.5→58.2, BBH 8.68→16.34, GSM8K +0.5, MMLU-Pro Physics 38.49→44.19.
- **Limitations / failure modes**:
  - Knowledge-heavy or creative domains are hard to generate. Verifiers may miss solution quality. Tasks are single-turn text only, and training sampled tasks uniformly and i.i.d. (authors).
  - The Reasoning Core v3 audit confirmed material default-path defects in 13 of 105 RG tasks, e.g., a scorer that gives full credit to any nonempty answer.
- **How to reuse with easy seed tasks**: Wrap each seed in the `(generator(config, seed), score_answer)` interface and expose at least one difficulty parameter. Tag stylistic parameters separately so they can be used as anti-memorization noise. Per the README, RG plugs directly into `verifiers` (Prime Intellect) and is used by NVIDIA ProRL/BroRL, Nous Atropos and MILA's Self-Evolving Curriculum. Audit the scorers of every RG task you enable. Sibling suites with the same generator + verifier pattern, useful as extra diversity or multi-turn upgrades:
  - TextArena: 57+ text games.
  - KORGym: 50+ textual/visual games with multi-turn and RL support.
  - GEM: a Gym-style agentic API that wraps RG, games, code, math and QA.
  - GlyphBench: 360+ game tasks rendered as Unicode grids.

### Reasoning Core — Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training (Lacombe et al., 2026)
Link: https://arxiv.org/abs/2603.02208 (v2) · v1 "A Scalable RL Environment for LLM Symbolic Reasoning" https://arxiv.org/abs/2509.18083 (2025) · v3 "Designing Broad Procedural Data for Completion-Supervised Reasoning Training" (Sileo, Lacombe, Kachler) https://arxiv.org/abs/2608.05148
- **Mechanism**:
  - The generators randomize the formal system itself, not just instance size. Domains: PDDL planning over randomized domains, first-order logic with equality, CFG parsing and generation, causal reasoning over random Bayesian networks, and systems of equations.
  - Each task has an external solver and a continuous difficulty knob, and examples can include solver-derived reasoning traces.
  - v3 grows to 50 generators (math, logic, planning, state tracking, formal languages, structured data, games, causality, code), with semantic scorers and evaluators.
- **How it makes tasks harder**: turn the continuous knob; draw new worlds (new domains, grammars, networks) so the model must reason from stated rules rather than recall domain patterns.
- **Correctness / verification**: planners, provers, parsers and exact inference compute or verify answers. v3 adds repository-scale audits: model-assisted review, human adjudication and regression tests.
- **Difficulty control**: a continuous parameter per generator.
- **Reported results**:
  - v2: mixing Reasoning Core data into pre-training improves downstream reasoning while preserving or slightly improving language-modeling quality. The tasks challenge GPT-5 zero-shot.
  - v3, matched completion-supervised protocol at 3B: highest mean on DROP, LogiQA and ARC-Challenge, above no-procedural data, Procedural Warmup, Reasoning Gym and SynLogic.
  - In isolated RG-task runs, tasks with *intermediate* final rewards transferred best to held-out benchmarks; fully saturated tasks did not transfer better.
  - Grid/board tasks formed a negative cluster, largely explained by long prompts and answers.
  - A 50-task RG subset selected by BBH-dev NLL degraded most other metrics.
- **Limitations / failure modes**: formal domains may transfer less to knowledge-heavy tasks. The authors stress that "semantic validity alone does not ensure training utility": compact targets and calibrated difficulty matter.
- **How to reuse with easy seed tasks**: if seeds are templated, randomize the *rules* (new operators, predicates, grammars), not only sizes. Use solver traces as SFT or midtraining data before RL. Prefer compact answer formats over full-state reconstruction.

### SynLogic — SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond (Liu et al., 2025)
Link: https://arxiv.org/abs/2505.19641 · MiniMax
- **Mechanism**: 35 tasks (Sudoku, Game of 24, ciphers, Zebra, ARC-AGI, BBH/BBEH-style tasks), chosen from logic-puzzle communities and from BBH/BBEH. Each has generation code, a rule-based verifier and difficulty hyperparameters. 33 tasks are generated in-house; Zebra and ARC-AGI data are adopted from existing sources.
- **How it makes tasks harder**: task-specific knobs such as Sudoku grid size.
- **Correctness / verification**: a dedicated rule-based verifier per task.
- **Difficulty control**: calibrated offline against reference models.
  - Upper bound: the hardest setting where DeepSeek-R1 or o3-mini still has pass@10 > 0.
  - Lower bound: the lowest setting where chat models get a pass rate in (0, 0.5).
  - Two releases: SynLogic-Hard (all 35 tasks, 33k samples, for Qwen2.5-32B) and SynLogic-Easy (27 tasks, 16k). The 8 tasks dropped from Easy stayed at zero training accuracy for 7B even after lowering difficulty.
- **Reported results**:
  - 7B: KOR-Bench 48.1, about 10 points above Qwen2.5-7B-Instruct.
  - 32B: BBEH 25.5 vs R1-Distill-Qwen-32B 19.2; AIME24 4.5→19.6; MATH500 82.0; AMC23 57.5.
  - Zero-Mix-3 (35k math + 9k code + 17k SynLogic) vs Zero-Mix-2 (math+code): BBEH 28.6 vs 18.5, KOR 65.0 vs 58.6, LCB 40.7 vs 39.5, AIME24 35.8 vs 34.5, GPQA-D 57.5 vs 55.2.
  - 7B logic+math matched math-only on math benchmarks with fewer math samples, and was about 10 KOR points higher.
- **Limitations / failure modes**: calibration is one-shot and offline, so it goes stale as the policy improves. Templated puzzles. The Reasoning Core v3 audit found 9 SynLogic generators with material defects, e.g., displayed constraints that contradict the stored solution.
- **How to reuse with easy seed tasks**: set each knob's upper bound where a strong reference model still sometimes succeeds, and the lower bound where your policy is below 50%. Re-calibrate periodically or add an online controller. INTELLECT-3 reused 29 SynLogic tasks (11.6K problems) as its logic environment.

### Enigmata — Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles (Chen et al., 2025)
Link: https://arxiv.org/abs/2505.19914 · ByteDance Seed / Fudan
- **Mechanism**:
  - 36 puzzle tasks in 7 categories. 30 tasks have auto-generators at any difficulty; the other 6 draw from fixed pools. All 36 have auto-verifiers (manually validated).
  - Training starts with RFT: DeepSeek-R1 samples 8 solutions per puzzle and the correct ones are kept. Puzzles and math are mixed 1:1, plus ARC-AGI train data, because ARC is "too difficult to learn without RFT".
  - Then VC-PPO RL, either:
    - Mix-training: Enigmata + ARC-AGI 1/2 + AIME 1983–2023; or
    - Multi-stage: ARC first, then Enigmata while retaining the earlier data.
- **How it makes tasks harder**: key variables per task (grid size, blank-cell count, mask rate r).
- **Correctness / verification**: a rule-based verifier per task.
- **Difficulty control**: Easy/Medium/Hard tiers per task are set from pass@k trends (n=200; k=1, 10, 100) across parameter settings. The per-task, per-level sample counts N_{i,d} are explicit mixing knobs.
- **Reported results**:
  - Qwen2.5-32B-Enigmata: Enigmata-Eval 62.6 (o3-mini-high 59.9, o1 54.9, o4-mini-high 65.1), ARC-AGI-1 32.8, ARC-AGI-2 0.6, KOR-Bench 65.0.
  - The authors "do not observe" general-reasoning transfer at Qwen2.5-32B.
  - Seed1.5-Thinking (20B active / 200B MoE) plus 20K Enigmata samples: AIME24 87.5 (+0.8), AIME25 75.9 (+1.9), BeyondAIME 48.4 (+0.4), GPQA-D 78.1 (+0.8).
  - Stage-2 mix (N_i=400 per task): balanced 1:1:1 beat medium-heavy 2:6:2.
  - Excess Enigmata data in stage 2 caused catastrophic forgetting.
- **Limitations / failure modes**: tiers are static; transfer is small at frontier scale; forgetting as the puzzle share grows.
- **How to reuse with easy seed tasks**: RFT cold-start is needed for seeds whose hard tiers have near-zero pass (ARC-like). Keep all three tiers in the mix, and track OOD metrics as you raise the puzzle share.

### InternBootcamp — InternBootcamp: Boosting LLM Reasoning with Verifiable Task Scaling (Li et al., 2025)
Link: https://arxiv.org/abs/2508.08636 (v1 titled "InternBootcamp Technical Report: …")
- **Mechanism**:
  - A "bootcamp" is a class with a case generator, a prompt function and a verify function.
  - About 100 hand-built bootcamps plus task descriptions seeded an agent workflow. DeepSeek-R1 fills in the interfaces and refines them over iterations using execution feedback: does it run, and does it generate more than a few special instances?
- **How it makes tasks harder**: configurable difficulty per bootcamp, plus training on many tasks at once.
- **Correctness / verification**: execution feedback, then a "self-consistent unit test": the bootcamp generates instances, R1-Distill-Qwen-32B solves them, and the bootcamp's own verifier scores the answers. Bootcamps with accuracy above 0.85 (oversimplified) or below 0.03 (semantically broken) are dropped. Manual inspection confirmed that these thresholds separate the two failure types.
- **Difficulty control**: per-bootcamp configuration; task-count scaling (8, 32, 128, 512 of the 704 retained tasks).
- **Reported results**:
  - Simplification keywords ("to simplify", "let's … temporarily") appeared in the generator's reasoning in 97.93% of 228 runs at iteration 1, 54.39% at iteration 2 and 32.46% at iteration 3. Problematic bootcamps fell from 33/228 to 14/228.
  - Performance scaled roughly linearly with task count (512 > 128 > 32 > 8), and 8-task RL saw costly batch-generation growth.
  - "Emergent moment": Hyperbaton, PropositionalLogicFormalization and Wordscapes did not learn in isolation (7B) but became solvable after about 300 steps in the 512-task mix.
  - Qwen2.5-32B-Instruct: Bootcamp-Eval 24.4 → 46.9 (RL), 61.1 (SFT on 55K R1 CoTs + 11K math), 59.5 (SFT→RL). OOD average 42.3 → 43.0 (RL only), 53.2 (SFT), 61.8 (SFT→RL).
  - R1-Distill-Qwen-32B with RL: OOD 52.5→56.9.
- **Limitations / failure modes**: LLM generators drift toward oversimplifying tasks. RL alone on a non-reasoning base barely moved OOD (+0.7); an SFT cold start was needed.
- **How to reuse with easy seed tasks**: have a strong model write generator and verifier code for many seed types. Feed back execution and "narrow-instance" signals over at least 3 iterations. Filter by a solver pass band, treating near-zero pass as "probably broken". Train on hundreds of tasks jointly.

### NPPC — Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs (Yang et al., 2025)
Link: https://arxiv.org/abs/2504.11239 · TMLR (per arXiv comment)
- **Mechanism**: npgym provides a unified interface to 25 NP-complete problems and generates any number of instances at any complexity. npsolver evaluates API and local models; npeval analyzes answers, tokens and errors.
- **How it makes tasks harder**: raise the size parameters without bound. There are four "ever-scaling" desiderata:
  - complexity (against being crushed);
  - instance (against hacking and memorization);
  - oversight (efficient verification);
  - coverage (real-world relevance).
- **Correctness / verification**: NP-complete certificates are verified in polynomial time. Rationale: problems in P can be solved by tool-using agents that write code, while NP-hard problems without polynomial verification cannot be graded at scale.
- **Difficulty control**: per-problem size parameters.
- **Reported results**: drives advanced LLMs below 10% accuracy. DeepSeek-R1 beats Claude-3.7-Sonnet and o1/o3-mini on most problems. Reasoning-token counts first rise, then fall, as instances get harder.
- **Limitations / failure modes**: evaluation-focused; very large instances run into context limits.
- **How to reuse with easy seed tasks**: if a seed reduces to an NP-complete core (scheduling, covering, coloring), generate instances with a planted certificate and grade the certificate, so difficulty can keep rising without a stronger grader.

### NP-Engine / Forge — NP-Engine: Empowering Optimization Reasoning in Large Language Models with Verifiable Synthetic NP Problems (Li et al., 2025); Forge: Quality-Aware Reinforcement Learning for NP-Hard Optimization in LLMs (Li et al., 2026)
Link: https://arxiv.org/abs/2510.16476 · https://arxiv.org/abs/2605.08905 (same first authors; Forge is an expanded and renamed version, with a benchmark called Forge-Bench)
- **Mechanism**: 10 NP-hard optimization tasks in 5 categories (graph clustering, resource scheduling, graph partitioning, subset selection, path planning). Each has:
  - a controllable generator;
  - a rule-based feasibility verifier;
  - a heuristic solver for a near-optimal reference. For TSP: multi-start nearest neighbor, then local search until no improvement or timeout.
- **How it makes tasks harder**: turn "find a feasible X" into "find the best X"; scale size and density. TSP tiers: Easy 10–20 cities, Medium 20–30, Hard 35–45, Benchmark 45–55.
- **Correctness / verification**: exact feasibility checks. The reward is R_format (±1) + R_feasibility, where R_feasibility is −1.5 if infeasible and R_optimal = M_s/M_h (maximization) or M_h/M_s (minimization) if feasible, designed to lie in (0, 1].
- **Difficulty control**: Easy/Medium/Hard tiers chosen from success-rate and quality trends across parameters; curriculum; multi-stage RL.
- **Reported results**:
  - NP-Engine, Qwen2.5-7B-Instruct-1M, zero-RL on 5K examples, NP-Bench: success rate (SR) 29.6→93.1, average ratio (AR) 14.6→46.6. For comparison, GPT-4o 62.1/36.2 and Qwen3-32B 70.7/57.6.
  - OOD average 50.4→53.6: KORBench +1.2, MATH500 +2.2, OlympiadBench +2.1, GPQA-D +4.1, IFEval +6.1.
  - TSP ablation:
    - Even without the heuristic ground truth ("w/o GT"), SR rose 4.0→90.0 and AR 1.9→25.6.
    - E:M:H = 5:4:1 with curriculum was best (AR 29.0, OOD 52.9), vs 1:4:5 (28.2 / 52.8) and 1:1:1 (27.1 / 52.2). The differences are small.
  - Multi-stage vs one-stage RL: overall SR 93.0 vs 72.6.
  - Forge (15K examples): reports the same 93.1% SR / 46.6% quality ratio (QR). The abstract claims quality-aware rewards beat binary rewards by 28.8%, and that 10 tasks beat 3 tasks on OOD (53.6 vs 53.0) despite 3.3× less data per task. It also notes that a fixed 1.0 reward for mere feasibility induced early reward hacking, because feasibility is easy.
  - *Correction*: Forge's abstract lists "GPT-4o (29.6% SR, 14.6% QR)", but those are the Qwen2.5-7B base numbers. Forge's own intro and table give GPT-4o 62.1% SR / 36.2% QR.
- **Limitations / failure modes**: the heuristic reference caps and biases the reward, and the ratio is designed to stay ≤1, which under-rewards beating the heuristic. Output length grows with instance size.
- **How to reuse with easy seed tasks**: convert any "valid solution" seed into an optimization variant rewarded by a ratio to a heuristic (or exact solver for small n). Keep a feasibility gate and penalty, and never pay a flat reward for feasibility alone.

### RLVE — RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments (Zeng et al., 2025)
Link: https://arxiv.org/abs/2511.07317 · ICML 2026 (per arXiv comment)
- **Mechanism**: 400 hand-engineered environments (RLVE-Gym). Each has a generator P_d conditioned on an unbounded integer level d, an input template and a verifier. Example mappings:
  - Sorting: array length ≈ 3·1.1^d. *Corrected*: N = d+3 is the bubble-sort permutation-counting environment, not sorting.
  - Integral: expression tree with d+2 nodes.
  - PolynomialMinimum: degree 2(d+1).
  - Sudoku: max(N, M) ≤ d+2.
  - HamiltonianPathExistence: N = d+3.
  
  The algorithm is DAPO with dynamic sampling.
- **How it makes tasks harder**: grow the dominant size/depth variable with d. Exploit solve–verify asymmetry:
  - Integral: sample F, give F′, and check the model's answer by differentiating it with SymPy, which avoids non-elementary integrals.
  - Sudoku: randomly transform a canonical solved grid, then mask cells.
  - Hamiltonian path: plant a random permutation path, then add random edges.
- **Correctness / verification**: exact algorithmic verifiers with shaped rewards:
  - −1 for bad format;
  - sorting: −0.5 for wrong length, else (x/N)^10, where x = correct positions;
  - numeric programming environments: (min/max)^10;
  - Hamiltonian path: −0.5 if not a permutation, else (valid edges/(N−1))^5.
- **Difficulty control**: a per-environment window [ℓ, h] starting at [0, 0].
  - Sample d uniformly in the window.
  - When h-level accuracy is at least τ_acc = 0.9 over at least τ_num = 8 × rollouts samples, set h+1 and ℓ = h − d_Δ + 1 with d_Δ = 4.
  - No upper cap. The health metric is the effective-prompt ratio.
- **Reported results**:
  - Qwen2.5-7B-Base on Sorting/Multiplication: static d∼[0,1] drove the effective-prompt ratio to 0 (saturation). Static [0,100] stayed nonzero but far below adaptive.
  - Joint training on 256 environments: adaptive upper bounds reached levels 0–12 by step 400. An oracle static [0,20] range covering them was consistently worse.
  - Nested C1⊂C4⊂C16⊂C256 improved accuracy on 50 held-out environments.
  - ProRL-1.5B-v2: +3.37 absolute average over 6 benchmarks (AIME24/25, OMEGA-500, OlympiadBench, LiveCodeBench, BBEH) in about 1,100 H100 hours. Continuing ProRL's own RL gave +0.49 in 3,600 hours.
  - From OpenThinker3-1.5B: about 2 points better than DeepMath-103K (which cost about $138K and 127K GPU hours to build).
- **Limitations / failure modes**:
  - Environments and difficulty maps are hand-designed.
  - Experiments are at 1.5B–7B.
  - The authors' attempt at frontier-LM automatic environment engineering struggled with template ambiguity, generator reliability and diversity, and verifier robustness.
- **How to reuse with easy seed tasks**: write each seed's generator as f(d, seed) along its dominant variable (length, depth, node count, digits). Plant solutions wherever solving is hard. Use (x/N)^k partial credit for long outputs. Deploy the [h−3, h] window with 0.9 promotion, and log the effective-prompt ratio.

### SCALER — SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning (Xu et al., 2026)
Link: https://arxiv.org/abs/2601.04809
- **Mechanism**:
  - GLM-4.6 extracts metadata from CodeContests problems: scale parameters from the constraints, output type in {number, array, string}, and whether the output is unique. Non-unique-output problems are discarded.
  - A test-case generator agent emits randomized inputs for a target scale configuration. Reference solutions produce the answers, executed in SandboxFusion.
  - Result: 4,973 problems → 2,739 environments, about 70M synthesis output tokens.
- **How it makes tasks harder**: turn the scale parameters up. A binary search over one global scale factor finds s_max that fits the 4,096-token prompt budget and the original execution time limit. The range is discretized into levels (arithmetic progression for small spans, geometric for large).
- **Correctness / verification**:
  - *Breadth check*: diverse scale configurations are answered by multiple independent ground-truth solutions that must agree. This also re-validates uniqueness.
  - *Depth check*: repeated generation at a fixed configuration must produce diverse outputs, which prevents answer-pattern hacking.
- **Difficulty control**:
  - Controller: d_{t+1} = clip(d_t + β(acc_t − τ), 0, D).
  - Active set of 64 environments, one problem each per step.
  - Retire an environment when the least-squares slope of difficulty over K_slope = 10 steps is ≤ 0, when accuracy is 0 for K_zero = 5 steps, or when it sits at max level for K_sat = 5 steps. Retired environments return to the pool.
- **Reported results**:
  - Qwen3-4B-Base, average of 5 benchmarks: SCALER 54.25 vs RLVE 53.52, MATH-7.5k 52.04, DeepMath-103K 51.08, base 35.91. RLVE was higher on BBEH (16.52 vs 14.56) and MATH-500.
  - Qwen3-1.7B-Base: 40.18 vs DeepMath 39.07 vs RLVE 37.80.
  - Performance rose monotonically across 8, 64, 512 and 2,739 environments.
  - Removing either the controller or the curation hurt.
  - Environment interaction was 16.85% of step time.
- **Limitations / failure modes**: only unique-output problems are usable. The effects of environment properties and scaling laws are unexplored (authors).
- **How to reuse with easy seed tasks**: if the seeds are coding or algorithmic problems with a reference solution, lift them. Use constraint variables as knobs, require at least two agreeing reference solutions, cap by token and time budgets, and run the proportional controller with slope-based retirement.

### Frontier Learning — Frontier Learning: Training LLM Reasoners at the Edge of Capability (Faro et al., 2026)
Link: https://arxiv.org/abs/2609.35426
- **Mechanism**:
  - The generator's attributes form a multidimensional space of "levels". The method does *not* assume a known or monotonic difficulty order.
  - Problem regret: ρ(x) = 1[s ≥ 1] − s/n (solvable but not reliably solved). Level regret is averaged over a moving window of that level's problems, using only rollouts GRPO already collects.
  - Priority P(ℓ, t) = R(ℓ) + λ_s·(t − t_ℓ), a staleness bonus.
  - Batch assembly: draw levels from the buffer by a Zipfian distribution over priority. Replace each slot with a uniformly random new level with probability φ (exploration). Otherwise mutate one attribute with a state-dependent probability p_mut(ℓ) (separate p_inf, p_easy, p_hard). When the buffer is full, evict the lowest priority.
- **How it makes tasks harder**: mutation expands the buffer into neighboring, harder parameter settings as regret moves.
- **Correctness / verification**: inherits the procedural generators' verifiers.
- **Difficulty control**: learned online from regret; no scalar difficulty needed.
- **Reported results**:
  - At 200 GRPO steps vs the best baseline (domain randomization, uniform, SEC, PLR, ACE-GRPO): Countdown 50.9 vs 44.7 (+13.9% relative), Sokoban 45.2 vs 41.1, Decimal Arithmetic 34.7 vs 31.0.
  - Dice at 500 steps: 71.8 vs SEC 33.3 (+115.6% relative); baselines plateau.
  - Ablations: PLR+Explore 33.5, PLR+Mutation 62.5, Uniform+FlatMutation 47.1, full 71.8.
  - Llama-3.2-3B Countdown: 39.2 vs 33.8. Olmo3-7B Largest Island: 73.7 vs PLR 52.0 (+41.7%).
- **Limitations / failure modes**: needs an explicit parametric generator; evaluated per task, not across large environment mixes.
- **How to reuse with easy seed tasks**: when a seed has several interacting knobs (operand count × magnitude × target range; grid × boxes × solution depth), keep a regret-prioritized buffer of settings and mutate the high-regret ones instead of hand-ordering levels.

### INTELLECT-3 difficulty pools — INTELLECT-3: Technical Report (Prime Intellect Team, 2025) [added]
Link: https://arxiv.org/abs/2512.16144
- **Mechanism**: production RL, built on the GLM-4.5-Air base, with prime-rl and verifiers environments.
  - Offline: problems are annotated with a small model's solve rate — Qwen3-4B-Thinking-2507 over 8 generations for math, Qwen3-4B-Instruct-2507 over 16 for code and logic — and too-easy samples are filtered.
  - Online: problems are sorted into easy/normal/hard pools by observed solve rate, with per-step control of how many samples come from each pool. An online filter "discards trivial rollouts — such as those that the model always fails or always solves". A prompt with pass rate 1 goes to the easy pool and is not sampled again.
  - Logic environment: 29 SynLogic tasks, 11.6K problems.
- **How it makes tasks harder**: it doesn't generate harder tasks; it re-weights the existing ones by current difficulty.
- **Correctness / verification**: environment verifiers.
- **Difficulty control**: pool proportions per step; online filtering.
- **Reported results**: none isolating this component. The RL setup was 256 prompts × 16 rollouts, with contexts up to 65,536 tokens.
- **Limitations / failure modes**: bounded by the ceiling of the fixed pool, so it saturates once the hard pool is solved.
- **How to reuse with easy seed tasks**: day-1 baseline. Annotate solve rates with a small model, bucket the data, drop all-pass and all-fail groups, then add generators with knobs so the "hard" pool can be refilled.

### LURE — The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning (Yu et al., 2026)
Link: https://arxiv.org/abs/2608.21871
- **Mechanism**: three separately trained models — an evader (challenger) plus a planner and an executor (together the pursuer, or solver). In each round the evader emits candidate tasks per environment, each placed at a target difficulty of that environment's programmatic generator. The pursuer attacks each task with G = 8 multi-turn rollouts. All three take one GRPO update with a KL penalty toward the round-start snapshot. Setup: 8 rounds, 48 candidates per environment per round.
- **How it makes tasks harder**:
  - Evader reward: min(p, 1−p) − ρ(x) for well-formed tasks and −1 − ρ for malformed ones, where p is the capture (solve) rate. It peaks at p = 0.5, where the group reward variance p(1−p) is largest.
  - ρ(x) = (n_sig − 1)/N penalizes repeated coarse signatures: hop count and population for multi-hop QA, constraint count for instruction-following, grid dimensions for logic grids.
- **Correctness / verification**: the evader only chooses where along the generator's difficulty axis to place tasks; generators and verifiers are programmatic. The pursuer also gets dense credit from verifier progress (λ = 0.25, cost for zero-progress steps).
- **Difficulty control**: a learned curriculum policy rather than post-hoc rejection bands (e.g., R-Zero).
- **Reported results** (Qwen2.5-7B-Instruct):
  - Unified model: PhantomWiki 65.8 vs R-Zero 50.8, GRPO 43.3, GRPO-Zero 37.5, base 25.0. ZebraLogic puzzle accuracy 16.9 vs 15.4 (GRPO).
  - Specialist models: PhantomWiki 80.0 vs 56.7; ZebraLogic 18.3 vs R-Zero 15.0.
  - Removing the learned challenger was the largest ablation (−10.6).
  - IFEval: the unified model (61.2) is below base (63.0) at 7B.
  - Integrity finding: R-Zero's role-conditioned prompt leaked the gold grid on ZebraLogic; removing it cut performance by 62 points.
- **Limitations / failure modes**: only three environments; modest, benchmark-dependent OOD gains.
- **How to reuse with easy seed tasks**: train a small "teacher" that picks generator parameters, rewarded by min(p, 1−p) of the student, with signature-repetition penalties. Audit every role prompt for answer leakage.

### SPIRAL (+ Stratagem) — SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning (Liu et al., 2025)
Link: https://arxiv.org/abs/2506.24119 · ICLR 2026 (per arXiv comment) · follow-up Stratagem https://arxiv.org/abs/2604.17696 (Feng et al., 2026; ACL 2026 Main per arXiv comment)
- **Mechanism**: a single policy plays both roles in TextArena zero-sum games (TicTacToe, Kuhn Poker, Simple Negotiation) against continuously updated copies of itself. The system is fully online, multi-agent and multi-turn (Oat + vLLM). Role-conditioned Advantage Estimation (RAE) centers returns per game and role.
- **How it makes tasks harder**: the opponent improves, giving an automatic curriculum. Fixed opponents (Mistral, Gemini) get exploited.
- **Correctness / verification**: the game engine decides the outcome.
- **Difficulty control**: implicit, through opponent strength.
- **Reported results**:
  - Final version: up to +10% across 8 reasoning benchmarks on 4 models. Qwen3-4B-Base multi-game went 34.0→44.5 average, beating SFT on 25,000 expert trajectories.
  - Without RAE, models abandon reasoning traces after about 200 steps ("thinking collapse").
  - R1-Distill-Qwen-7B still benefits.
  - Stratagem modulates the game advantage: A_mod = A_game·φ + β·ψ, with LLM-judged transferability φ ∈ {0, 0.5, 1} and reasoning-evolution reward ψ ∈ {−1, 0, +1}; β = 0.2 was best. Results: AIME24 10→20%, AIME25 3.3→13.3%, AMC23 60% (SPIRAL 45%), MATH500 76% (+5 over SPIRAL).
- **Limitations / failure modes**: few, simple games; multi-agent RL instability; the Stratagem judge adds an LLM-in-the-loop reward.
- **How to reuse with easy seed tasks**: recast a seed as a two-player game (proposer vs solver, hidden-information bargaining, adversarial puzzle setting). Use per-role baselines and mix several games.

### Logic-RL — Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning (Xie et al., 2025)
Link: https://arxiv.org/abs/2502.14768
- **Mechanism**: procedurally generated Knights & Knaves puzzles, trained with a modified REINFORCE++ (γ = 1) on Qwen2.5-7B-Instruct-1M. Run: 3,600 steps, learning rate 4e-7, temperature 0.7, fewer than 5K puzzles mixing 3–7 people.
- **How it makes tasks harder**: 2–8 characters; 1–4 combined Boolean operators per statement.
- **Correctness / verification**:
  - A unique ground truth is guaranteed by the generator.
  - Format reward +1/−1: each tag appears exactly once and in order, and the reasoning must be genuine.
  - Answer reward +2 for a full match, −1.5 for a partially wrong answer, −2 for an unparseable one.
  - Seven observed hacks: skipping <think>; reasoning inside <answer>; repeated guessing; filler text; unextractable ordering; re-thinking after answering; echoing the question.
- **Difficulty control**: people count and operator depth; a mixed 3–7 range.
- **Reported results**:
  - Relative gains of +125% on AIME and +38% on AMC.
  - OOD to 8-person puzzles.
  - Response length grew about 500→2,000 tokens over 1K steps.
  - PPO was most accurate but 138% slower than REINFORCE++; GRPO was weakest in their setup.
  - Curriculum vs mixed difficulty: slightly higher mid-training, practically negligible overall.
  - Longer responses did not by themselves cause better reasoning.
- **Limitations / failure modes**: a single puzzle family; transfer measured at small absolute scale.
- **How to reuse with easy seed tasks**: K&K is the cheapest first gym. Scale people and nesting, enforce strict formats, and penalize partial answers. Warm Up Before You Train (https://arxiv.org/abs/2505.13718) found that distilling K&K long CoT before small-data RLVR improved MATH, HumanEval+ and MMLU-Pro, and kept cross-domain generality.

### K&K perturbation benchmark — On Memorization of Large Language Models in Logical Reasoning (Xie et al., 2024) [added]
Link: https://arxiv.org/abs/2410.23123
- **Mechanism**: a dynamic K&K generator with:
  - an abstract-puzzle module: N people, each statement a logic tree of max width W and depth D over and/or/not/implication/equivalence;
  - a solver;
  - a CoT generator;
  - a *Perturber*;
  - a natural-language module with random names and templates.
  
  Core data: 1,000/100/50 train/test/val puzzles per N (2 ≤ N ≤ 8; default W = D = 2).
- **How it makes tasks harder**: more people and wider/deeper statements. The 8-person space (W = D = 2) has about 10^24 puzzles, of which about 30% have a unique solution. Six perturbations per puzzle:
  - 2 problem-level: replace a statement; replace a leaf. The perturbed puzzle must have a *different* solution.
  - 4 language-level: random role-pair names, uncommon person names, reordered statements, flipped roles (knights lie).
- **Correctness / verification**: the generator's solver checks uniqueness; perturbations are re-solved.
- **Difficulty control**: N, W, D.
- **Reported results**: fine-tuned models reach near-perfect training accuracy but fail on slight perturbations. The local-inconsistency memorization score (LiMem) is higher under math-level than language-level perturbations. Role-flipped accuracy is near zero, and Llama3-8B's memorization score under role-flipping is about 80%. Yet fine-tuning still improves generalization, even when trained on wrong answers.
- **Limitations / failure modes**: an SFT-centric study; a single family.
- **How to reuse with easy seed tasks**: build a perturber for each seed family, i.e., minimal edits that flip the answer, plus counterfactual rule flips. Use it both to make hard variants and as an anti-memorization evaluation.

### ZebraLogic — ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning (Lin et al., 2025)
Link: https://arxiv.org/abs/2502.01100 · ICML 2025 (per arXiv comment)
- **Mechanism**: sample a solution grid; enumerate all true clues (FoundAt, SameHouse, NotAt, DirectLeft/Right, SideBySide, Left/RightOf, One/TwoBetween). Then repeatedly pick a clue and delete it if a SAT solver confirms the solution is still unique. Weighted sampling makes simpler clue types (FoundAt) likelier to be removed, which keeps the puzzle hard. The problem is NP-complete (reduction from Quasigroup Completion; FoundAt + NotAt suffice).
- **How it makes tasks harder**: larger N × M grids; minimal clue sets.
- **Correctness / verification**: a SAT solver checks uniqueness after every removal.
- **Difficulty control**: two measures. Search space (N!)^M, binned Small < 10^3, Medium < 10^6, Large < 10^10, X-Large ≥ 10^10. And the mean number of Z3 conflicts over 32 runs: 0 conflicts means forward-chaining suffices.
- **Reported results**:
  - 1,000 puzzles, N, M ∈ {2..6}, 40 per size, mean 10.4 clues.
  - Overall accuracy (X-Large in brackets): o1 81.0 (42.5), R1 78.7 (28.5), o1-preview 71.4, o1-mini 59.7, Claude 3.5 Sonnet 36.2 (1.0), Llama-3.1-405B 32.6 (0.0).
  - Most models collapse beyond about 10^7 search space (e.g., 4×5) or about 20 Z3 conflicts.
  - GPT-4o best-of-N with an oracle at N = 128: 69.1. Majority vote: 31.7→38.0 at N = 32, with no further gain.
- **Limitations / failure modes**: a benchmark, not a training recipe; a narrow family.
- **How to reuse with easy seed tasks**: the "sample ground truth → enumerate true facts → solver-prune to a minimal unique set" operator works for any seed with a hidden assignment (schedules, seatings, allocations). Bucket by solver conflicts, not surface size.

### ZebraArena — ZEBRAARENA: A Diagnostic Simulation Environment for Studying Reasoning-Action Coupling in Tool-Augmented LLMs (Zhao et al., 2026) [added]
Link: https://arxiv.org/abs/2603.18614
- **Mechanism**: a ZebraLogic-style puzzle with some clues withheld, so the given clues admit more than one solution. The agent recovers the missing constraints through tool calls to a rule-based server:
  - fact queries (is house i attribute A equal to v?);
  - relation queries (left_of, adjacent, distance_k, same_house, …).
  
  Queries are validated against a JSON schema and canonicalized.
- **How it makes tasks harder**: turns a static, fully specified puzzle into partially observed, multi-step information gathering. Difficulty grows with grid size and the number of missing clues.
- **Correctness / verification**: a unique ground-truth solution. K_ref = number of masked clues gives a constructive recovery policy of exactly K_ref queries. Per-step information gain IG_t = log|X(C_{t−1})| − log|X(C_t)|, where X(C) is the set of solutions consistent with the clues known so far.
- **Difficulty control**: grid (N, M); missing-clue count; query budget and pricing.
- **Reported results**: Gemini 2.5 Pro reaches only 60% on Large instances. GPT-5 uses 70–270% more tool calls than K_ref on Medium.
- **Limitations / failure modes**: a diagnostic, not a training study.
- **How to reuse with easy seed tasks**: take any solved seed puzzle, withhold k constraints behind a query tool, and reward correctness plus query efficiency against K_ref. This gives a cheap, exactly verifiable multi-turn upgrade.

### SATURN — SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning (Liu et al., 2025)
Link: https://arxiv.org/abs/2505.16368 · NeurIPS 2025 Spotlight (per arXiv comment)
- **Mechanism**: SAT_Construction(n, k, l, m) samples m satisfiable (n, k, l)-SAT instances: n literals per clause, k variables, l clauses. The run is a curriculum-estimation loop plus a GRPO training loop, starting from DeepSeek-R1-Distill-Qwen-1.5B/7B. Initial settings: (3, 5, 5) for 1.5B and (3, 5, 13) for 7B; 2 curriculum iterations.
- **How it makes tasks harder**: increase the difficulty estimate D(n, k, l) = log₂k + 2·log₂l − n + k/n, fitted to LLM pass@3 (R² 0.707 and 0.724). The authors argue that phase-transition hardness, which is built for heuristic solvers, is misaligned with LLM reasoning.
- **Correctness / verification**: all instances are satisfiable; the model's full assignment is checked against every clause.
- **Difficulty control**: advance when validation pass@1 > ε (0.5 for 1.5B, 0.75 for 7B), evaluated every 250 training samples.
- **Reported results**:
  - SATURN-2.6k = 1,500 train, 160 same-level test, 1,000 test across 10 harder unseen levels.
  - Unseen harder levels, pass@3: 1.5B 10.1→24.2; 7B 36.1→64.2 (+14.0 / +28.1).
  - Math/code average: +4.9 (1.5B) and +1.8 (7B). 7B MATH500 93.2→95.0, LiveCodeBench 35.4→37.7.
  - On DeepScaleR-1.5B: 49.9→52.3 (AIME +5.0, GPQA-D +6.0).
  - On Qwen2.5-7B-Instruct-1M, 1K SAT examples gave 32.5→36.2, 8.8% relative above Logic-RL trained on 5K.
- **Limitations / failure modes**: no domain knowledge is learned; context-window bottlenecks as the needed reasoning grows exponentially; plasticity loss and forgetting across stages.
- **How to reuse with easy seed tasks**: compile constraint-flavored seeds to CNF, fit your own difficulty regression on your policy's pass rates, and demand certificates rather than yes/no answers.

### SATBench — SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas (Wei et al., 2025)
Link: https://arxiv.org/abs/2505.14615 · authors Wei, Wu, Wan, Suresh, Tan, Zhou, Koyejo, Wang, Aiken
- **Mechanism**: sample CNF formulas. An LLM (e.g., GPT-4o) writes a story background and a variable mapping (e.g., "musician i performs in genre j") and turns each clause into a condition. The SAT/UNSAT label comes from a solver. 2,100 puzzles.
- **How it makes tasks harder**: more clauses — Easy 4–19, Medium 20–30, Hard 31–50; UNSAT questions; narrative noise.
- **Correctness / verification**: LLM checks for omissions and additions. Then an LLM converts the narrative back into a formula, and a solver checks bidirectional equivalence with the original (non-equivalent puzzles are discarded). Human audit of 100 puzzles: mapping 100%, clause faithfulness 97%, LLM trace judgment 93%.
- **Difficulty control**: clause-count tiers; SAT vs UNSAT.
- **Reported results**:
  - Average accuracy fell from 78.5 (easy) to 53.0 (hard).
  - o4-mini on hard: 78.0 overall, 91.1 SAT, 65.0 UNSAT.
  - "Satisfiability bias" (answering SAT with an invalid assignment) was 68.7% of o4-mini's errors and 40.5% of R1's.
  - LoRA SFT of Qwen2.5-14B on 1,100 o4-mini traces: 51.9→53.6.
- **Limitations / failure modes**: evaluation set; LLM rendering can introduce errors (mitigated by the checks).
- **How to reuse with easy seed tasks**: render formal seeds into narratives, but always parse back and solver-check equivalence. Reward assignment certificates, not SAT/UNSAT labels.

### SATQuest — SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs (Zhao et al., 2025)
Link: https://arxiv.org/abs/2509.00930 · ACL 2026 Main (per arXiv comment). *Correction*: a separate paper by different authors (Zhao, Li, Bo, Takezoe, …), not part of SATBench.
- **Mechanism**: generates tasks directly from DIMACS CNF along three orthogonal axes.
  - *Instance*: n, m, and solver statistics (decisions, conflicts, propagations).
  - *Problem type*:
    - SATDP (decision);
    - SATSP (find an assignment);
    - MaxSAT;
    - MCS (minimal correction subset of an UNSAT formula);
    - MUS (minimal unsatisfiable core).
  - *Question format*: Math, DIMACS, Story, DualStory (the negated, AND-semantics phrasing).
  - Answers are binary strings (1, n or m bits).
- **How it makes tasks harder**: move up the problem-type ladder on the same instance; switch to machine or narrative formats; scale n.
- **Correctness / verification**: PySAT checks every answer and accepts any valid one. SAT/UNSAT evaluation pairs share nearly identical structure.
- **Difficulty control**: evaluation set n ∈ [3, 16], m = 4n (140 pairs). RFT set n ∈ [3, 8] with clause/variable ratio 2.1–4.0 (3,000 pairs).
- **Reported results**:
  - o3-mini leads at 0.56.
  - R1 and QwQ drop 9 and 10 points on SATSP in DIMACS vs Math format.
  - GRPO RFT on Qwen2.5-7B (and 14B) boosts the targeted tasks and generalizes to larger instances. Cross-task transfer is asymmetric, and cross-format robustness remains limited.
- **Limitations / failure modes**: long binary outputs strain small models' format adherence.
- **How to reuse with easy seed tasks**: MUS and MCS give *checkable certificates for UNSAT*: a core is verified by one solver call plus minimality checks. That is the practical way to reward "no solution" answers without 50% guessing.

### AutoLogi — AutoLogi: Automated Generation of Logic Puzzles for Evaluating Reasoning Abilities of Large Language Models (Zhu et al., 2025)
Link: https://arxiv.org/abs/2502.16906
- **Mechanism**:
  - Stage 1 extracts background and constraints from AR-LSAT/LogiQA-style sources.
  - Stage 2 has GPT-4/GPT-4o write a JSON format spec plus format and constraint verifiers, then a traversal function that enumerates the domain.
  - Stage 3 augments difficulty in both directions.
- **How it makes tasks harder**:
  - *Expansion*: an LLM adds constraints and verifiers until the solution space shrinks to 1 or attempts run out.
  - *Reduction*: remove constraints one at a time, down to 1, for easier variants.
- **Correctness / verification**: traversal and verifiers cross-check each other, and an empty solution space or code errors trigger regeneration.
  - *Corrected numbers*: 23% of initial Chinese problems (11% English) were unsolvable.
  - About 30% of LLM-added constraints made puzzles unsolvable, which the traversal caught.
  - About 3% of validation results remain wrong.
- **Difficulty control**: constraint count, driving the solution-space size to 1.
- **Reported results**:
  - 1,575 English and 883 Chinese puzzles. Model scores spread 35.25–72.61 vs 21.04–37.39 on the multiple-choice source.
  - Qwen2.5-72B self-alignment DPO: +6.61 (EN), +5.62 (ZH), AR-LSAT +7.05, LiveBench +6.13.
  - Strong-to-weak RFT (72B sampler): +4.69 (EN), +5.46 (ZH).
- **Limitations / failure modes**: needs domains small enough to enumerate exhaustively; LLM-added constraints can be unnatural.
- **How to reuse with easy seed tasks**: for finite-domain seeds, write the brute-force traversal once. Then generate graded variants by adding or removing constraints, and admit each only after traversal confirms the solution count.

### Illusion of Thinking (+ comment) — The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity (Shojaee et al., 2025) [added]
Link: https://arxiv.org/abs/2506.06941 (NeurIPS 2025 per arXiv comment) · critique: Lawsen, *Comment on The Illusion of Thinking*, https://arxiv.org/abs/2506.09250
- **Mechanism**: controllable puzzle simulators (Tower of Hanoi, Checker Jumping, River Crossing, Blocks World) scale compositional complexity while keeping the logical structure fixed. Traces and final answers are checked move by move.
- **How it makes tasks harder**: increase N (disks, checkers, actors, blocks).
- **Correctness / verification**: the simulator validates move sequences.
- **Difficulty control**: N.
- **Reported results**: three regimes, comparing large reasoning models (LRMs) with standard LLMs:
  - low complexity: non-thinking models are better;
  - medium: thinking helps;
  - high: both collapse.
  
  Thinking tokens rise with complexity, then *fall* near collapse despite remaining budget.
- **Limitations / failure modes** (the comment):
  - Tower of Hanoi outputs can exceed output-token limits, and models said so.
  - River Crossing instances with N ≥ 6 and boat capacity 3 are *unsolvable*, yet were scored as failures.
- **How to reuse with easy seed tasks**: a direct warning for complexification. Check solvability of every generated instance with a solver. Separate "can't reason" from "can't emit the answer within the budget" by allowing compact answers (e.g., a program or algorithm).

### LogicPro — LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning (Jiang et al., 2024)
Link: https://arxiv.org/abs/2409.12929 · ACL 2025 (per arXiv comment)
- **Mechanism**: pair LeetCode-style problems with test cases and turn each (problem, input) pair into a text reasoning problem. Running the reference Python solution gives the answer plus key intermediate-variable values, which guide synthesis of the reasoning trace.
- **How it makes tasks harder**: difficulty is inherited from algorithmic complexity and input size; more inputs give more problems.
- **Correctness / verification**: answers and intermediate states come from executing the program.
- **Difficulty control**: none adaptive.
- **Reported results**: 540K examples from 2,360 algorithm problems improved several models on BBH (27 tasks), LogicBench, DROP, AR-LSAT and GSM8K.
- **Limitations / failure modes**: rationales can be mechanical; no policy-aware difficulty.
- **How to reuse with easy seed tasks**: for any seed with a reference program, scale the inputs and use execution traces as verified SFT rationales. Then switch to RL with the program as verifier.

### Game-RL / Code2Logic — Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning (Tong et al., 2025)
Link: https://arxiv.org/abs/2505.13886 (earlier title: Code2Logic)
- **Mechanism**: Code2Logic has three steps:
  1. LLMs construct game code.
  2. LLM-assisted design of question and analysis templates, each capturing one reasoning pattern.
  3. A data engine that reuses the core game code to emit (image, question, step-by-step analysis, answer).
- **How it makes tasks harder**: controllable difficulty grades per task, via game parameters such as board size or number of steps.
- **Correctness / verification**: answers come from executing game logic, and the image and text share one latent state.
- **Difficulty control**: per-task difficulty grades.
- **Reported results**: GameQA has 30 games, 158 tasks and 140K questions. GRPO on GameQA alone gave Qwen2.5-VL-7B +2.33% across 7 vision-language benchmarks.
- **Limitations / failure modes**: modest transfer; game domains are narrow.
- **How to reuse with easy seed tasks**: any simulator is a free verifier. Emit "state after k moves" or "best next move" questions and raise k or the board size as pass rates rise.

### Loong — Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers (Huang et al., 2025)
Link: https://arxiv.org/abs/2509.03059 · CAMEL-AI
- **Mechanism**: LoongBench has 8,729 human-vetted seeds across 12 domains, each with executable code. LoongEnv generates new questions plus solving code from the seeds using few-shot prompting, Self-Instruct or Evol-Instruct, executes the code for the answer, and compares it with the agent's CoT answer.
- **How it makes tasks harder**: Evol-Instruct keeps the core semantics but compounds the reasoning.
- **Correctness / verification**: sandboxed execution, a judge agent (DeepSeek-R1), and agreement between CoT and code.
- **Difficulty control**: the generation strategy; no parametric knob.
- **Reported results** (GPT-4.1-mini as generator):
  - Logic domain: few-shot passes 92.6%; Self-Instruct is rejected 44.8% of the time; Evol-Instruct code is 55.0% non-executable.
  - Physics: few-shot 93.9% and Self-Instruct 82.0% pass; Evol-Instruct 29.8% judged incorrect and 14.0% non-executable.
  - Advanced Physics accuracy by strategy (few-shot / Self-Instruct / Evol-Instruct): GPT-4.1-mini 92.0 / 83.0 / 62.0; R1 93.2 / 87.4 / 70.3.
- **Limitations / failure modes**: difficulty and validity trade off sharply. Generated code is only as correct as its author.
- **How to reuse with easy seed tasks**: LLM "evolve" operators do make seeds harder, but budget for about 30–55% rejection. Gate every variant on executable code, determinism and agreement between independent solves.

### ReSyn — ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models (He et al., 2026)
Link: https://arxiv.org/abs/2602.20117 · CMU & AWS
- **Mechanism**:
  - About 100 1–2-word keywords come from an LLM shown one problem per BBH and KOR-Bench subtask, plus manual algorithm and data-structure keywords.
  - For each keyword, the LLM proposes 8 tasks, each implemented as ρ₀(d, n) (instance sampler, d ∈ 1..5), O(s) (question renderer) and R(s, a) → {0, 1} (verifier).
- **How it makes tasks harder**: a 5-level difficulty argument inside each environment; tasks chosen for "computational advantage" (easy with tools, hard by hand).
- **Correctness / verification**: a two-stage LLM judge.
  - Code stage: the environment must satisfy *reference-free verification* (checking by logic rather than against a possibly hallucinated reference) or *computational advantage*, plus completeness and difficulty scaling.
  - Question stage: well-specified and loophole-free. Failed environments get one revision.
  - Then a *difficulty calibration*: keep an environment only if solve rate falls with level (one-sided Wald test, α = 0.05). This removes trivial and impossible environments.
- **Difficulty control**: levels 1–5, validated statistically.
- **Reported results**:
  - 418 environments, 16K instances.
  - Qwen2.5-7B-Instruct → ReSyn: BBH 65.9→75.2, BBEH 11.2→14.3 (+27% relative), GSM8K 82.3→91.4, AIME24 mean@128 9.8→14.0.
  - Reward ablation, BBH / BBEH: Verifier-RL 75.24 / 14.61, Code-RL 74.94 / 14.24, Answer-RL 68.83 / 14.33.
  - Fixed about 16K instances (BBH): 400×40 = 75.19, 100×160 = 69.85, 25×640 = 71.20.
- **Limitations / failure modes**: only 418 environments trained (compute); coarse levels; no online adaptation.
- **How to reuse with easy seed tasks**: prompt with seed-derived keywords. Require verifier-first environments with a level argument, and run the Wald monotonicity test on your own policy before admission.

### EvoEnv — Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis (Shi et al., 2026)
Link: https://arxiv.org/abs/2605.14392 (tech report)
- **Mechanism**: a single policy alternates two roles in GRPO (generator advantage scale 0.3; 100 steps; 10 seed environments covering sorting, DP, graph and number theory).
  - Generator: writes VerifiableEnvironment classes (seeded instance sampler with difficulty, prompt renderer, hidden reference, scorer).
  - Solver: answers prompts sampled from the accepted pool.
- **How it makes tasks harder**:
  - Difficulty reward: m = 8 calibration instances, one solver response each, reward Q = exp(−(â − 0.3)²/2σ²). The target is below 0.5 because near-50% environments saturate quickly. Admission requires 0 < â < 1.
  - Novelty: frozen all-MiniLM-L6-v2 embeddings of the prompt template and the generator code (λ = 0.5). The adaptive novelty weight γ ∈ [2, 5] turns on between similarity thresholds 0.45 and 0.65; the pool-admission gate is similarity 0.80.
  - The design principle is stable solve–verify asymmetry: equality oracles (DP, graph traversal, recurrences) and feasibility oracles (planted subset-sum, constraint satisfaction).
- **Correctness / verification**:
  - Mechanical gates L1–L5: parse; execute across seeds and difficulties; determinism; non-triviality; scorer contract (the reference scores positive; perturbed, malformed and type-mismatched answers do not; the parser does not leak the reference).
  - Then K_rev = 3 same-policy reviews with any-reject. Review verdicts gate the pool but are not rewarded, so the generator cannot optimize for the reviewer.
  - Audit of 79 environments labeled by GPT-5.4: F1 87.0% (P 85.7, R 88.2). The reviewer is weakest on cross-method data flow.
- **Difficulty control**: solver-relative calibration, plus pool rotation.
- **Reported results**:
  - 840 environments and 45 tag prototypes from 10 seeds. Pass rates rose over training: L2/L3 53.4→~73.6%, L4 42.4→63.0%.
  - Average over 8 benchmarks: Qwen3-4B-Instruct 49.2→53.1; Qwen3-4B-Thinking 72.4→74.8 (fixed public-data RLVR and fixed hand-crafted-environment RLVR both *lowered* it); Nemotron-Cascade-8B 71.0→73.2.
  - Solver training score fell 0.88→0.61 while held-out accuracy on 50 unseen RLVE environments rose 72.4→80.4% (fixed baseline about 72%).
  - Ablation: removing the quality reward cut +2.4 to +0.5; removing novelty cut it to +0.6.
- **Limitations / failure modes**: deterministic Python tasks only; requires a sandbox; residual data-flow bugs and hallucinated edge cases.
- **How to reuse with easy seed tasks**: give seeds as few-shot environment classes. Admit new environments only through L1–L5, perturbation probes, any-reject review, the 0 < pass < 1 band and novelty. Target a pass rate of about 0.3.

### DeepSeek-V3.2 agentic task synthesis — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek-AI, 2025)
Link: https://arxiv.org/abs/2512.02556
- **Mechanism**: given a category (e.g., travel itinerary) and a sandbox with bash and search tools, an agent collects data into a database. It then proposes a simple task with Python *solution* and *verification* functions. The solution function may only call tools or do logic; it cannot touch the database directly.
- **How it makes tasks harder**: the agent "iteratively increases the difficulty of the task and updates the corresponding solution and verification functions", adding tools when the toolset is insufficient. Example: a multi-day trip with no repeats and constraints that interact with budget, hotel tier and ratings.
- **Correctness / verification**: the solution's output must pass the verifier, or the agent revises one of them. Only instances with non-zero pass@100 under DeepSeek-V3.2 RL are kept.
- **Difficulty control**: iterative escalation plus the pass@100 > 0 filter.
- **Reported results**:
  - 1,827 environments / 4,417 tasks, plus 50,275 search and 24,667 code-agent tasks.
  - On 50 sampled tasks, pass@1: DeepSeek-V3.2-Exp 12%, Sonnet-4.5 34%, Gemini-3.0 Pro 51%, GPT-5-Thinking 62%.
  - RL on synthetic general-agent data alone (from the SFT checkpoint, non-thinking) improved Tau2Bench, MCP-Mark and MCP-Universe. RL only on code and search did not.
- **Limitations / failure modes**: solution and verifier share an author, so they can share blind spots.
- **How to reuse with easy seed tasks**: "make it harder while your reference still passes your verifier", looped. Admit only bundles with 0 < pass@k < 1 on your policy, and audit verifiers independently.

### VHD-Play — Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms (Shen et al., 2026)
Link: https://arxiv.org/abs/2609.27321 · Qwen Technical Report
- **Mechanism**: sample parameters θ for an OR mechanism family (inventory DP, routing, knapsack allocation, LP/QP, scheduling, search, …). Solve for u*(θ) (optimum) and u₀(θ) (default) *before* any environment code exists. A frozen setter (the same Qwen3.6-35B-A3B suffices) receives θ and a passage from a real-world document, and writes a scenario, a database, at least 10 stateful tools and instructions.
- **How it makes tasks harder**:
  - Presentation: written-out (all parameters in the prompt) → informed-agentic (parameters shown, actions through tools) → agentic (parameters discoverable only through tools).
  - Mechanism size and horizon.
- **Correctness / verification**:
  - Reward r = clip[0,1]((u − u₀)/(u* − u₀)).
  - Admission replays reference policies through the generated code: optimal π*, greedy π₀ and idle π∅.
    - C1: executability.
    - C2: the default outcome lies in the admissible region.
    - C3: π* reproduces u* within tolerance.
  - Post-hoc audit: 480 executions, no exceptions, all repeats agreed, and no raw utility exceeded u*.
- **Difficulty control**: presentation mode; size and horizon (knapsack 6×5 up to 11×11).
- **Reported results** (*corrected*: these are means over five families, not inventory alone):
  - Base: written-out 0.962, informed-agentic 0.231, agentic 0.204.
  - Training: (+0.030, +0.644, +0.611), closing 84% of the parameter-revealed gap and 77% of the full gap. Agentic 0.204→0.815.
  - Held-out training families +0.56; near-OOD families +0.63; far-OOD +0.24 / +0.16.
  - Knapsack scale sweep, gains +0.412 → +0.861; at 11×11 base 0.085 → 0.946.
  - External: BFCL V4 (10 cells) +2.84; TravelBench 0.700→0.794; storefront ending balance 3.4× base.
  - About $0.01–0.03 per admitted environment; 3,300 environments.
- **Limitations / failure modes**: only mechanisms with a solver and a parameterization.
- **How to reuse with easy seed tasks**: if the model aces a seed written out, keep the solved instance but move its parameters behind query tools and add a horizon. The same u* keeps grading exact.

### f(g(x)) — From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (Yuan et al., 2025)
Link: https://arxiv.org/abs/2509.25123
- **Mechanism**: 25 atomic string transformations with meaningless names (func_16). Stage 1 teaches the atoms (definitions shown during data collection, hidden in training). Stage 2 trains on compositions from one subset of functions and evaluates on held-out functions, comparing RL with RFT, using Llama-3.1-8B-Instruct.
- **How it makes tasks harder**: composition depth (Level k = k nested functions); hidden definitions.
- **Correctness / verification**: execute the composed Python functions.
- **Difficulty control**: nesting depth.
- **Reported results**:
  - RL on Level-2: unseen Level-3 went from near zero to about 30% (27% in one figure) and Level-4 to about 15%. RFT on the same data stayed ≤2.6% at Level-3.
  - RL on Level-1 only: <1% gain at Level-3.
  - Transfer to Countdown Level-3 reached 35% when Countdown atoms were known. After RL, failures shift to atomic errors (55%).
- **Limitations / failure modes**: toy domain; atoms are a prerequisite.
- **How to reuse with easy seed tasks**: once the atomic seeds are mastered, generate chained variants (output feeds input) at depth ≥ 2 and train them with RL, not SFT.

### Depth × Environment Complexity — Reasoning Depth and Environment Complexity: A Controlled Study of RLVR Data Allocation across Logical Reasoning Tasks (Zhu et al., 2026)
Link: https://arxiv.org/abs/2605.26934 · EMNLP 2026 Main (per arXiv comment)
- **Mechanism**: a synthetic knowledge-graph world with people, objects, time-ordered ownership-changing events and static relations. Instances vary along:
  - depth D (events on the target's causal chain);
  - environment complexity T (parallel chains, person scale, cross-chain entity overlap, exchange frequency);
  - task family (deductive state tracking, abductive recovery, inductive rule induction, analogy).
  
  Models are trained from scratch: 107M main, sweep 51M–1.02B. Pretraining covers D ∈ 1..4 and T ∈ 1..2 (3.2M graphs, 3.14B tokens).
- **How it makes tasks harder**: depth and distractors/interactions as separate axes; inverting forward tasks into abductive ones.
- **Correctness / verification**: the generated graph is the ground truth.
- **Difficulty control**: an explicit D × T grid.
- **Reported results**:
  - Joint depth–complexity coverage beats single-axis recipes.
  - Abductive training barely generalizes outside RL coverage; the other families keep about 50–70% of in-recipe gains.
  - Tasks correlate in deductive–abductive and inductive–analogy pairs, and analogy shows mode collapse.
  - Uniform mixing beats a staged 3-block curriculum at fixed budget.
  - Off-the-shelf models show the same deductive > abductive gap.
- **Limitations / failure modes**: small from-scratch models.
- **How to reuse with easy seed tasks**: add a distractor/interaction axis next to depth, cover the grid jointly, and generate inverted (abductive) versions explicitly.

### AES + HDC — Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning (Zhu et al., 2026)
Link: https://arxiv.org/abs/2608.03571
- **Mechanism**: 200 multimodal environments (from VisGym and Gym-V); Qwen3-VL-4B/8B.
  - AES: build meta-ability profiles from trajectory-level atomic abilities (merged by GPT-5), then select environments for coverage while penalizing redundancy and optimization conflict.
  - HDC: an outer harness curriculum H0 → H4. H0 gives text observations, text state, hints and rules; H4 gives only visual observations and the task description. Earlier levels stay sampled. An inner state-scale curriculum samples from a per-environment sliding window.
- **How it makes tasks harder**: remove scaffolds; grow the state scale.
- **Correctness / verification**: unchanged environment rewards.
- **Difficulty control**: harness level × state-scale frontier.
- **Reported results**:
  - Adding environment types under a fixed budget can fluctuate or even degrade performance.
  - Average relative gain over base: AES-selected 30 environments +95.6% vs all 200 +43.4%; AES+HDC +143.2%.
  - Visual state extraction is a major failure source.
  - (The "72 meta-abilities" figure in the source JSON could not be confirmed and was removed.)
- **Limitations / failure modes**: multimodal games only; profiling needs an LLM annotator.
- **How to reuse with easy seed tasks**: easy tasks are often easy because of scaffolding (rules restated, state summarized, hints). Strip it progressively as a difficulty axis orthogonal to size, and select environments for skill coverage.

### Can One Domain Help Others? — Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning (Li et al., 2025)
Link: https://arxiv.org/abs/2507.17512
- **Mechanism**: GRPO on Qwen2.5-7B base and instruct, over math (DeepScaleR, Countdown), code, and puzzles (K&K, Logic Puzzle Baron (LPB)), singly and combined.
- **How it makes tasks harder**: K&K levels 3PPL→8PPL (PPL = people per puzzle) with "policy refresh": every 175 steps, reset the reference model to the actor and reset the optimizer state.
- **Correctness / verification**: rule verifiers.
  - K&K rewards compared: binary, partial (N_c/N), format and rescaled.
  - LPB uses a proportional cell reward.
- **Difficulty control**: sequential level curriculum.
- **Reported results**:
  - K&K-only (base): MATH500 56.4→68.4, AIME24 10.0→20.0, K&K 94.29, but code average 67.46→56.09. KK+LPB softened the code drop (71.35).
  - Math+Puzzle cost code −22.56; three domains together were more balanced.
  - K&K curriculum with refresh 99.71 vs plain curriculum 97.29 vs mixed 94.29.
  - Binary reward was best on K&K. On LPB, binary collapsed, partial peaked at 38.63 at step 200 then declined, and format and rescaled rewards eventually won.
- **Limitations / failure modes**: one model family at 7B.
- **How to reuse with easy seed tasks**: when adding complexified puzzles, keep every target domain in the mix and monitor code. Move from binary to partial or format-shaped rewards once success becomes sparse.

### RAGEN / RAGEN-2 — RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (Wang et al., 2025) [added]
Link: https://arxiv.org/abs/2504.20073 · RAGEN-2: Reasoning Collapse in Agentic RL (Wang et al., 2026) https://arxiv.org/abs/2604.06268
- **Mechanism**: StarPO trajectory-level RL for agents in Bandit, Sokoban, FrozenLake and WebShop. RAGEN-2 decomposes reasoning quality into within-input entropy and cross-input mutual information.
- **How it makes tasks harder**: turns puzzles into multi-turn, stochastic, irreversible interaction.
- **Correctness / verification**: environment rules.
- **Difficulty control**:
  - StarPO-S filters instances by uncertainty (reward variance), adds a critic and stabilizes gradients.
  - RAGEN-2's SNR-aware filtering selects high-reward-variance prompts each iteration.
- **Reported results**:
  - "Echo Trap": reward-variance cliffs, entropy drop and gradient spikes. PPO collapses later than GRPO on symbolic tasks.
  - Rollouts benefit from diverse initial states, medium interaction granularity and frequent sampling.
  - RAGEN-2's "template collapse": outputs look diverse but are input-agnostic, invisible to entropy. Mutual information tracks final performance much better.
- **Limitations / failure modes**: stylized environments.
- **How to reuse with easy seed tasks**: when converting static puzzles to multi-turn ones, filter prompts by reward variance and monitor cross-input mutual information, not just entropy.

### CAST — CAST: Game Solvers as Turn-Level Teachers for LLM Agents (Wang et al., 2026) [added]
Link: https://arxiv.org/abs/2607.25308
- **Mechanism**: change in a game solver's state value → per-turn "solver advantage", injected into RLVR, stabilized with asinh compression and batch RMS normalization. Under a soft-optimal-solver assumption this equals on-policy distillation from the solver without teacher logits.
- **How it makes tasks harder**: makes long-horizon, harder puzzle instances learnable despite sparse terminal rewards. It is evaluated at unseen difficulties.
- **Correctness / verification**: an exact solver, or a learned value network when none exists (reported comparable).
- **Difficulty control**: the game instances' difficulty levels; in-domain vs unseen-difficulty splits.
- **Reported results**: best of all trained methods on Sokoban, Minesweeper and Rush Hour, both in-domain and at unseen difficulty. Beats outcome-only RLVR and the GiGPO process baseline, with the best zero-shot ALFWorld and WebShop averages. The exact solver adds negligible overhead.
- **Limitations / failure modes**: needs a solver or value model for the domain.
- **How to reuse with easy seed tasks**: when complexified puzzles drop the pass rate toward zero, add solver-value-difference credit per step rather than lowering difficulty. SPRING (https://arxiv.org/abs/2609.34660) does the analogue for single-turn logic: SMT-checked "novel deduction" step rewards, up to +49.71 ZebraLogic puzzle accuracy over base and +15.43 over the best outcome-only baseline.

## Complexification operators from this area

1. **Size / scale escalation**
   - *What it does*: grow the dominant variable (items, grid, nodes, digits, clauses) as a function of a level d.
   - *Example*: sort 5 numbers → about 3·1.1^d numbers; 4×4 Sudoku → max(N,M) ≤ d+2; TSP 10–20 cities → 45–55.
   - *Keeping it verifiable*: programmatic checker, not a stored answer. Partial credit (x/N)^k for long outputs. Cap by prompt and output tokens and by solver time (SCALER binary-searches a global scale factor).
   - *Sources*: RLVE, Reasoning Gym, SCALER, NP-Engine.
   - *Caveat*: size alone mostly adds length (see Illusion of Thinking critique; Reasoning Core grid tasks).
2. **Planted solution / solve-first generation**
   - *What it does*: sample the answer first, then derive a problem it solves; hardness sits on the solving side.
   - *Example*: random 9×9 solved grid → masked puzzle; planted Hamiltonian path + random edges; random F → ask for ∫F′; satisfiable CNF; planted subset-sum with distractors; OR mechanism with u* precomputed.
   - *Keeping it verifiable*: verify the model's answer against constraints, not against the planted answer when several answers are valid.
   - *Sources*: RLVE, SATURN, EvoEnv, VHD-Play.
3. **Constraint minimization / expansion to uniqueness**
   - *What it does*: enumerate facts true of a hidden solution, then prune to a minimal unique set; or add constraints until one solution remains.
   - *Example*: 3-house Zebra with direct clues → 6×6 with minimal relational clues; LSAT puzzle with 4 constraints → expanded until unique.
   - *Keeping it verifiable*: SAT/SMT or exhaustive traversal after every edit. About 30% of LLM-added constraints make puzzles unsolvable (AutoLogi).
   - *Sources*: ZebraLogic, AutoLogi.
4. **Solver-effort hardness targeting**
   - *What it does*: bucket by the search effort a solver needs rather than by surface size.
   - *Example*: Zebra puzzles with 0 Z3 conflicts → more than 20; CNF chosen by SATURN's D(n,k,l) or by SATQuest's decision/conflict/propagation counts.
   - *Keeping it verifiable*: certificates are checked in polynomial time. Regress the metric on your *policy's* pass rate before trusting it (SATURN R² about 0.71).
   - *Sources*: ZebraLogic, SATURN, SATQuest, NPPC.
5. **Composition / chaining**
   - *What it does*: feed one seed's output into another; depth = number of composed skills.
   - *Example*: func_15(x) → func_16(func_15(func_3(x))) with definitions hidden.
   - *Keeping it verifiable*: execute the pipeline; compose only atoms with exact verifiers. Train compositions with RL, not RFT.
   - *Sources*: f(g(x)), OMEGA (compositional generalization "remains limited" under SFT), InternBootcamp (skills emerge only in mixtures).
6. **Reasoning-depth increase**
   - *What it does*: longer inference chains or deeper nesting.
   - *Example*: K&K with 2 people and 1 operator → 8 people and 4 nested operators (W/D trees); 1 ownership transfer → 8+ events; expression tree with d+2 nodes.
   - *Keeping it verifiable*: the generator holds the world state. Prefer compact answers.
   - *Sources*: Logic-RL, K&K benchmark, Depth × Complexity, RLVE.
7. **Distractor / interacting-structure injection**
   - *What it does*: add plausible irrelevant structure the model must filter.
   - *Example*: one ownership chain → 6 parallel chains with shared people and frequent exchanges; knowledge-graph QA with reranked, semantically similar distractor triples; subset-sum with near-miss numbers.
   - *Keeping it verifiable*: build distractors knowing the true answer; re-check uniqueness with a solver.
   - *Sources*: Depth × Complexity, SIE (https://arxiv.org/abs/2509.23330), EvoEnv.
8. **Problem-type upgrade and inversion**
   - *What it does*: move the same instance up a ladder, or ask for causes rather than effects.
   - *Example*: SAT decision → satisfying assignment → MaxSAT → MCS → MUS on the *same* CNF; forward state tracking → abductive "which event is missing?"; derivative → antiderivative.
   - *Keeping it verifiable*: reward certificates (assignments, cores, antiderivatives checked by differentiation). Never reward bare SAT/UNSAT labels (SATBench satisfiability bias: 68.7% of o4-mini errors). Abduction needs explicit coverage.
   - *Sources*: SATQuest, SATBench, RLVE, Depth × Complexity.
9. **Feasibility → optimality**
   - *What it does*: turn "any valid X" into "best X" with continuous quality rewards.
   - *Example*: any tour → shortest tour; any allocation → horizon-optimal allocation.
   - *Keeping it verifiable*: exact feasibility gate with a penalty; ratio to heuristic, exact or pre-solved u*. No flat reward for feasibility, which was hacked in Forge.
   - *Sources*: NP-Engine, Forge, VHD-Play.
10. **Program / game-code lifting**
    - *What it does*: an existing program becomes the oracle, and its input constraints become the knobs.
    - *Example*: one LeetCode sample → random inputs at controller-chosen scale; game engine → "state after k moves" for large k.
    - *Keeping it verifiable*: at least two independent reference solutions must agree; unique outputs only; diversity check at a fixed scale.
    - *Sources*: SCALER, LogicPro, Game-RL, Loong.
11. **Learned teacher / self-play difficulty**
    - *What it does*: an opponent or teacher policy sets the difficulty.
    - *Example*: TicTacToe vs fixed opponent → Kuhn Poker vs latest self; an evader choosing ZebraLogic sizes near p = 0.5.
    - *Keeping it verifiable*: the game engine or generator verifies; the teacher only chooses parameters. Add signature-repetition penalties or vocabulary dropout (+4.4 at 8B in R-Zero; https://arxiv.org/abs/2604.03472) to prevent proposer collapse.
    - *Sources*: SPIRAL, LURE, Vocabulary Dropout.
12. **Information hiding / static → interactive**
    - *What it does*: keep the solved instance but make the model acquire the parameters through tools or partial observation.
    - *Example*: written-out knapsack (0.962 mean over 5 families) → agentic 11-period version (0.204 mean); a Zebra puzzle with k clues withheld behind fact/relation queries; knowledge-graph context with 100%→0% of the supporting triples retained (SIE).
    - *Keeping it verifiable*: pre-solve and grade arithmetically. Replay reference policies through the generated environment (VHD-Play C1–C3). K_ref gives an efficiency baseline.
    - *Sources*: VHD-Play, ZebraArena, SIE.
13. **Scaffold / harness removal**
    - *What it does*: remove the aids that make tasks easy.
    - *Example*: H0 (text observations, state, hints, rules) → H4 (raw visual + task description); visible function definitions → opaque names.
    - *Keeping it verifiable*: the verifier is unchanged. Keep sampling earlier levels to avoid abrupt shifts.
    - *Sources*: AES+HDC, f(g(x)).
14. **Representation / format shift**
    - *What it does*: render the same latent instance differently.
    - *Example*: CNF in math notation → DIMACS → story → negated "DualStory"; English → 14 languages (Multilingual Reasoning Gym: 94 tasks, native-speaker validation in 10 languages); text → image (Game-RL; TRON online visual generator–verifier programs).
    - *Keeping it verifiable*: the answer key is unchanged. For LLM renderings, back-translate and solver-check equivalence (SATBench). Expect cross-format robustness to lag (SATQuest).
    - *Sources*: SATQuest, SATBench, Multilingual RG (https://arxiv.org/abs/2603.10793), TRON (https://arxiv.org/abs/2606.01599), Game-RL.
15. **Local perturbation and counterfactual rules**
    - *What it does*: minimal edits that flip the answer; flip a familiar rule.
    - *Example*: replace one K&K statement or leaf (the new solution must differ); knights lie instead of tell the truth.
    - *Keeping it verifiable*: re-solve every perturbation.
    - *Sources*: K&K benchmark; Sudoku-Bench variants with unusual interacting rules, where SOTA solves <15% unaided (https://arxiv.org/abs/2505.16135).
16. **Rule / world randomization**
    - *What it does*: randomize the formal system itself.
    - *Example*: standard Blocksworld → a fresh PDDL domain; a Sudoku variant with a new interacting constraint.
    - *Keeping it verifiable*: an external planner, parser or prover; discard worlds where it times out or finds no solution.
    - *Sources*: Reasoning Core, Sudoku-Bench.
17. **Agentic iterative escalation with a co-updated verifier**
    - *What it does*: an LLM agent repeatedly hardens a seed and updates solution, tests and environment in sync.
    - *Example*: 1-day trip with one budget → multi-day trip with interacting tier, budget and rating constraints plus new tools; terminal tasks rewritten by increase, reduce or diversify actions chosen by a MILP (mixed-integer linear program) around a target frontier (Envs-FORGE: tb-core 40.0→49.2 on Qwen 3.5 35B); off-policy "environment evolution" along three difficulty directions (Terminal-Bench 2.1 +14.4 / +18.0).
    - *Keeping it verifiable*: admit only gold-verified bundles (the reference passes, perturbations fail, and 0 < policy pass rate < 1). Audit independently, since solution and verifier share an author.
    - *Sources*: DeepSeek-V3.2, Envs-FORGE (https://arxiv.org/abs/2608.14312), Environment Evolution (https://arxiv.org/abs/2609.04128), EvoEnv.

## Insights & pitfalls

- **The effective-prompt ratio is the primary health metric.** Static low-cap difficulty goes to 0 (RLVE). A high static cap wastes most rollouts, and even an oracle static range covering all adaptive levels lost to per-environment adaptation. Every environment has its own notion of "level", so tune per environment, not globally.
- **Controller menu, from cheapest to most flexible:**
  - INTELLECT-3 pools plus all-pass/all-fail filtering (no generation);
  - SATURN's pass@1 gate;
  - RLVE's window (τ = 0.9, d_Δ = 4);
  - SCALER's proportional controller with slope/zero/saturation retirement (K = 10/5/5);
  - Frontier Learning's regret buffer with mutation, for non-monotone multi-knob generators;
  - a learned teacher (LURE).
  
  Mutation was the key ingredient in Frontier Learning (PLR+Explore 33.5 vs PLR+Mutation 62.5).
- **Where to aim.** Regret peaks for problems that are solvable but not reliably solved. EvoEnv deliberately targets 0.3, not 0.5, because near-half environments saturate quickly as the solver improves. Admit only 0 < pass < 1.
- **Keep bands and mixes rather than hard stage switches.** A controlled study found uniform mixing beat staged curricula at fixed budget; Logic-RL saw negligible curriculum benefit. But sequential K&K curricula with a policy refresh did help (99.71 vs 94.29 mixed). If you stage, reset the reference model and optimizer at each stage.
- **Diversity vs selection.** More distinct environments generally beat more instances (ReSyn, RLVE, SCALER, InternBootcamp, Forge). But selecting 30 of 200 environments by ability coverage beat all 200 (AES). A 50-task Reasoning Gym subset chosen by BBH-dev NLL hurt most other metrics (Reasoning Core v3). Select by skill coverage, not by a single dev metric.
- **Generator bugs are the dominant correctness risk, even in popular gyms.**
  - 13 of 105 Reasoning Gym tasks and 9 SynLogic generators had material defects: a scorer paying full credit for any nonempty answer; displayed constraints contradicting the stored solution.
  - AutoLogi found 11–23% of initial LLM-built puzzles unsolvable.
  - Apple's River Crossing included unsolvable N ≥ 6, b = 3 instances.
  - InternBootcamp's generator tried to "simplify" in 97.93% of first-iteration runs.
  - Mandatory gates:
    - execution across seeds and levels;
    - determinism;
    - uniqueness or solvability by solver;
    - at least two agreeing reference solutions;
    - scorer perturbation probes;
    - a monotonicity test between level and solve rate;
    - any-reject review;
    - periodic human audit.
- **Very low pass rate on a new environment usually means it is broken, not hard.** InternBootcamp's manual inspection confirmed that <3% accuracy mostly flagged semantic errors and >85% flagged oversimplification. Inspect the extremes before training.
- **LLM "evolve" operators trade validity for difficulty.** Loong's Evol-Instruct cut GPT-4.1-mini accuracy from 92.0 to 62.0 on physics, but 55% of its logic-domain code failed to run. Budget for heavy rejection.
- **Reward hacking follows the answer format.**
  - Logic-RL catalogued 7 shortcuts.
  - SATBench's satisfiability bias shows that bare labels are gameable.
  - Forge's flat feasibility reward was hacked early.
  - SCALER keeps unique-output problems only and checks output diversity at a fixed scale.
  - Use certificates such as assignments, cores or paths.
- **Reward shape must track difficulty.**
  - Binary rewards were best for K&K. On LPB, binary collapsed and partial, format or rescaled rewards were needed.
  - For large instances, use steep partial credit, (x/N)^10 in RLVE.
  - For optimization, use quality ratios.
  - For long-horizon puzzles, use solver-value (CAST) or SMT-checked step rewards (SPRING).
- **Transfer is real but uneven.**
  - Positive: RG MATH +9.7 (3B); ReSyn BBEH +27% relative; SynLogic AIME24 4.5→19.6 (32B); SATURN +4.9/+1.8; SPIRAL up to +10%; GlyphBench RL on 100 game tasks reached 63.48% on held-out Reasoning Gym, beating math- and code-trained baselines (https://arxiv.org/abs/2609.34214).
  - Negative or null: Enigmata saw no general transfer at 32B and +0.8–1.9 at Seed1.5-Thinking scale; K&K-only dropped code by 11.4 points; Guru found logic, simulation and tabular need in-domain RL data (https://arxiv.org/abs/2506.14965).
  - Keep puzzles a minority and track OOD metrics every evaluation (Enigmata saw forgetting as puzzle share grew).
- **Static public gyms stop helping strong models.** On Qwen3-4B-Thinking, fixed public RLVR data and fixed hand-crafted environments both reduced the average, while self-evolving environment synthesis raised it (EvoEnv). Plan for rising ceilings: unbounded knobs, policy-written environments, learned teachers.
- **SFT cold start matters for hard gyms.** InternBootcamp RL-only on Qwen2.5-32B-Instruct moved OOD +0.7, while SFT→RL gave +19.5. Enigmata needed RFT for ARC-like tasks. K&K long-CoT warmup helps small-data RLVR.
- **Composition beats imitation for new skills.** RL on depth-2 compositions generalizes to depth 3–4, while RFT and atom-only RL do not. OMEGA shows SFT gives exploratory but little compositional and "little to no" transformative generalization (https://arxiv.org/abs/2506.18880).
- **Hidden difficulty is often interaction, not reasoning.** VHD-Play's base model solved the mechanisms written out (0.962) but failed when it had to carry decisions through state (0.231) or acquire parameters (0.204). Training closed most of that gap, with gains growing at larger scales.
- **Memorization confound.** K&K fine-tuning memorizes training puzzles, and role-flip accuracy is near zero. Chess-trained LMs' benchmark scores are largely pattern-matching (https://arxiv.org/abs/2605.17565). Always evaluate on held-out generator families, held-out parameter regions and perturbed variants.
- **Audit prompts, not just answer keys.** In multi-role self-play, a role prompt leaked ZebraLogic gold grids, and removing it cut the baseline by 62 points (LURE).
- **Multi-turn collapse modes.** RAGEN's Echo Trap and RAGEN-2's template collapse. Filter by reward variance and watch cross-input mutual information.
- **Multi-domain scheduling can be learned cheaply.** The Transfer-Aware Curriculum (TAC) prioritizes domains by gradient-alignment transferability at <1% overhead, up to +2.8 over a learnability-only bandit (https://arxiv.org/abs/2606.25178).

## Open problems & research opportunities

- **Difficulty that is not length.** Most knobs scale size or depth. Generators that control "insight" (Sudoku-variant break-ins, transformative strategies in OMEGA) are mostly missing, and the Illusion of Thinking critique shows how easily size-scaling conflates reasoning with output length.
- **Automatic knob discovery.** Frontier Learning and RLVE need explicit parameters. ReSyn, SCALER and EvoEnv rely on LLM-declared levels checked by solve-rate correlation. Learning a monotone difficulty embedding for an arbitrary seed set is open.
- **Universal claims.** UNSAT, optimality and "no solution" answers need certificates. MUS/MCS (SATQuest) and exact solvers help for small n. At scale, systems either avoid UNSAT (SATURN) or use heuristic ceilings (NP-Engine caps its ratio at ≤1).
- **Verifier correctness for model-written environments.** Self-review reaches about 87% F1 against a stronger auditor (EvoEnv), and it is weakest on cross-method data flow. Solution/verifier co-authorship (DeepSeek-V3.2) risks shared blind spots. Adversarial red-teaming of scorers is not yet standard.
- **Predicting transfer before training.** TAC, AES ability coverage and Guru's pretraining-exposure hypothesis are early. A transfer-aware *generator* (not just a scheduler) is open.
- **Environment scaling laws.** Environment count is measured only at a few points: RLVE 256, ReSyn 418, InternBootcamp 512, SCALER 2,739. The joint laws in environments × instances × model size × compute are unmeasured (SCALER lists this as future work).
- **Abductive and inductive generators.** Abduction does not generalize beyond coverage, and analogy mode-collapses. Exactly verifiable inverse-problem and rule-induction generators lag forward-deduction puzzles.
- **Keeping self-evolving curricula diverse over long runs.** Proposer collapse, template collapse and drift toward verifier-gaming are handled only by heuristics (vocabulary dropout, signature penalties, embedding novelty).
- **Dense yet unhackable process rewards without solvers.** CAST can fall back to a learned value network. SPRING needs SMT encodings.
- **Continual addition of environments.** Adding complexified environments without re-running multitask RL or forgetting earlier ones is just starting to be studied (Beyond Forgetting, https://arxiv.org/abs/2608.18574).
- **Evaluation hygiene.** Generator families (ZebraLogic, K&K, Reasoning Gym) now appear in both training and evaluation. Held-out *mechanism families* (VHD-Play's eight unseen families) and perturbation suites should become standard.

## References

1. Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye, A., Kaddour, J., Köpf, A. (2025). *REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards*. NeurIPS 2025 Spotlight; arXiv:2505.24760. https://arxiv.org/abs/2505.24760
2. Lacombe, V., Quesnel, V., Sileo, D. (2025). *Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning*. arXiv:2509.18083. https://arxiv.org/abs/2509.18083
3. Lacombe, V., Quesnel, V., Sileo, D. (2026). *Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training*. arXiv:2603.02208. https://arxiv.org/abs/2603.02208
4. Sileo, D., Lacombe, V., Kachler, D. (2026). *Reasoning Core: Designing Broad Procedural Data for Completion-Supervised Reasoning Training*. arXiv:2608.05148. https://arxiv.org/abs/2608.05148
5. Liu, J., Fan, Y., Jiang, Z., et al. (2025). *SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond*. arXiv:2505.19641. https://arxiv.org/abs/2505.19641
6. Chen, J., He, Q., Yuan, S., et al. (2025). *Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles*. arXiv:2505.19914. https://arxiv.org/abs/2505.19914
7. Li, P., Ye, J., Chen, Y., et al. (2025). *InternBootcamp: Boosting LLM Reasoning with Verifiable Task Scaling* (v1: InternBootcamp Technical Report). arXiv:2508.08636. https://arxiv.org/abs/2508.08636
8. Yang, C., Wang, R., Jiang, J., et al. (2025). *Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs*. TMLR; arXiv:2504.11239. https://arxiv.org/abs/2504.11239
9. Li, X., Fang, X., Ding, S., Li, L., Duan, H., Liu, Q., Chen, K. (2025). *NP-Engine: Empowering Optimization Reasoning in Large Language Models with Verifiable Synthetic NP Problems*. arXiv:2510.16476. https://arxiv.org/abs/2510.16476
10. Li, X., Fang, X., Ding, S., et al. (2026). *Forge: Quality-Aware Reinforcement Learning for NP-Hard Optimization in LLMs*. arXiv:2605.08905. https://arxiv.org/abs/2605.08905
11. Zeng, Z., Ivison, H., Wang, Y., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*. ICML 2026; arXiv:2511.07317. https://arxiv.org/abs/2511.07317
12. Xu, C., Xiao, C., Peng, Z., Wang, X., Cao, Y. (2026). *SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning*. arXiv:2601.04809. https://arxiv.org/abs/2601.04809
13. Faro, R., Ramesh, S. S., Bogunovic, I., Lucchi, A. (2026). *Frontier Learning: Training LLM Reasoners at the Edge of Capability*. arXiv:2609.35426. https://arxiv.org/abs/2609.35426
14. Prime Intellect Team, Senghaas, M., Obeid, F., et al. (2025). *INTELLECT-3: Technical Report*. arXiv:2512.16144. https://arxiv.org/abs/2512.16144
15. Yu, J., Chen, S., Tan, Y. (2026). *The Chase Is the Curriculum, the Capture Anchors the Credit: Pursuit-Evasion Self-Play for Zero-Data LLM Reasoning*. arXiv:2608.21871. https://arxiv.org/abs/2608.21871
16. Liu, B., Guertler, L., Yu, S., et al. (2025). *SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning*. ICLR 2026; arXiv:2506.24119. https://arxiv.org/abs/2506.24119
17. Feng, X., Yin, D., Feng, X., et al. (2026). *Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play*. ACL 2026; arXiv:2604.17696. https://arxiv.org/abs/2604.17696
18. Xie, T., Gao, Z., Ren, Q., et al. (2025). *Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning*. arXiv:2502.14768. https://arxiv.org/abs/2502.14768
19. Xie, C., Huang, Y., Zhang, C., Yu, D., Chen, X., et al. (2024). *On Memorization of Large Language Models in Logical Reasoning*. arXiv:2410.23123. https://arxiv.org/abs/2410.23123
20. Lin, B. Y., Le Bras, R., Richardson, K., Sabharwal, A., Poovendran, R., Clark, P., Choi, Y. (2025). *ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning*. ICML 2025; arXiv:2502.01100. https://arxiv.org/abs/2502.01100
21. Zhao, W., Schmidt, L., Choi, Y., Zou, J., Balachandran, V. (2026). *ZEBRAARENA: A Diagnostic Simulation Environment for Studying Reasoning-Action Coupling in Tool-Augmented LLMs*. arXiv:2603.18614. https://arxiv.org/abs/2603.18614
22. Liu, H., Li, G., Li, J., Zhu, H., Zhang, K., Dong, Y. (2025). *SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning*. NeurIPS 2025 Spotlight; arXiv:2505.16368. https://arxiv.org/abs/2505.16368
23. Wei, A., Wu, Y., Wan, Y., Suresh, T., Tan, H., Zhou, Z., Koyejo, S., Wang, K., Aiken, A. (2025). *SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas*. arXiv:2505.14615. https://arxiv.org/abs/2505.14615
24. Zhao, Y., Li, Y., Bo, Z., Takezoe, R., et al. (2025). *SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs*. ACL 2026; arXiv:2509.00930. https://arxiv.org/abs/2509.00930
25. Zhu, Q., Huang, F., Peng, R., Lu, K., Yu, B., Cheng, Q., Qiu, X., Huang, X., et al. (2025). *AutoLogi: Automated Generation of Logic Puzzles for Evaluating Reasoning Abilities of Large Language Models*. arXiv:2502.16906. https://arxiv.org/abs/2502.16906
26. Shojaee, P., Mirzadeh, I., Alizadeh, K., Horton, M., Bengio, S., et al. (2025). *The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity*. NeurIPS 2025; arXiv:2506.06941. https://arxiv.org/abs/2506.06941
27. Lawsen, A. (2025). *Comment on The Illusion of Thinking*. arXiv:2506.09250. https://arxiv.org/abs/2506.09250
28. Jiang, J., Yan, Y., Liu, Y., et al. (2024). *LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning*. ACL 2025; arXiv:2409.12929. https://arxiv.org/abs/2409.12929
29. Tong, J., Tang, J., Li, H., Mou, Y., Zhang, M., Zhao, J., et al. (2025). *Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning*. arXiv:2505.13886. https://arxiv.org/abs/2505.13886
30. Huang, X., Rishabh, Franke, G., Yang, Z., et al. (2025). *Loong: Synthesize Long Chain-of-Thoughts at Scale through Verifiers*. arXiv:2509.03059. https://arxiv.org/abs/2509.03059
31. He, A., Weir, N., Bostrom, K., Nie, A., Cassel, D., Bayless, S., Rangwala, H. (2026). *ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models*. arXiv:2602.20117. https://arxiv.org/abs/2602.20117
32. Shi, Y., Liang, Z., Panaganti, K., Yu, D., Yu, W., Mi, H. (2026). *Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis*. arXiv:2605.14392. https://arxiv.org/abs/2605.14392
33. DeepSeek-AI (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556. https://arxiv.org/abs/2512.02556
34. Shen, X., Fan, W., Guo, X., Tu, J., Su, Y., Kuang, C., Zhang, Y., Liu, D. (2026). *Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms*. Qwen Technical Report; arXiv:2609.27321. https://arxiv.org/abs/2609.27321
35. Yuan, L., Chen, W., Zhang, Y., Cui, G., Wang, H., You, Z., et al. (2025). *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*. arXiv:2509.25123. https://arxiv.org/abs/2509.25123
36. Zhu, Y., Liu, Q., Cheng, F., Wang, J., Aizawa, A., Kurohashi, S., Shimodaira, H. (2026). *Reasoning Depth and Environment Complexity: A Controlled Study of RLVR Data Allocation across Logical Reasoning Tasks*. EMNLP 2026; arXiv:2605.26934. https://arxiv.org/abs/2605.26934
37. Zhu, K., Jin, Z., Huang, D., Yuan, H., Hao, Y., Liu, K., Zhao, J. (2026). *Beyond Simply Environment Scaling: Designing Effective Environment Distributions for Multimodal Agent Learning*. arXiv:2608.03571. https://arxiv.org/abs/2608.03571
38. Li, Y., Pan, Z., Lin, H., Sun, M., He, C., Wu, L. (2025). *Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning*. arXiv:2507.17512. https://arxiv.org/abs/2507.17512
39. Wang, Z., Wang, K., Wang, Q., Zhang, P., Li, L., et al. (2025). *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*. arXiv:2504.20073. https://arxiv.org/abs/2504.20073
40. Wang, Z., Gui, C., Jin, X., et al. (2026). *RAGEN-2: Reasoning Collapse in Agentic RL*. arXiv:2604.06268. https://arxiv.org/abs/2604.06268
41. Wang, Y., Zhang, Y.-K., Shi, W., et al. (2026). *CAST: Game Solvers as Turn-Level Teachers for LLM Agents*. arXiv:2607.25308. https://arxiv.org/abs/2607.25308
42. Ali, M. A., Wang, W., Wang, H., et al. (2026). *Rewarding Novel Deductions: Solver-guided Process Rewards for Logical Reasoning* (SPRING). arXiv:2609.34660. https://arxiv.org/abs/2609.34660
43. Creus Castanyer, R., Côté, M.-A., Sargent, M. J., et al. (2026). *GlyphBench: A Playground for Language-Model Reinforcement Learning*. arXiv:2609.34214. https://arxiv.org/abs/2609.34214
44. Shrestha, S., Kim, M., Nepal, A., Shrestha, A., Ross, K. (2025). *Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings*. EMNLP 2025; arXiv:2505.13718. https://arxiv.org/abs/2505.13718
45. Sun, Y., Hu, S., Zhou, G., Zheng, K., Hajishirzi, H., et al. (2025). *OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization*. arXiv:2506.18880. https://arxiv.org/abs/2506.18880
46. Dineen, J., RRV, A., Xu, Z., Zhou, B. (2026). *Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution*. COLM 2026; arXiv:2604.03472. https://arxiv.org/abs/2604.03472
47. Yu, P., Zhao, Z., Zhang, S., Fu, L., Wang, X., et al. (2025). *Structured In-context Environment Scaling for Large Language Model Reasoning* (SIE). arXiv:2509.23330. https://arxiv.org/abs/2509.23330
48. Dobler, K., Lehnerer, S., Scozzafava, F., Janke, J., Ali, M., et al. (2026). *Multilingual Reasoning Gym: Multilingual Scaling of Procedural Reasoning Environments*. arXiv:2603.10793. https://arxiv.org/abs/2603.10793
49. Yang, T., Shi, Y., Sun, R., Huang, J., Liu, N. (2026). *TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL*. arXiv:2606.01599. https://arxiv.org/abs/2606.01599
50. Wu, X., Yang, C., Liu, H., Lin, X., Shi, Z., et al. (2026). *Envs-FORGE: Frontier-Optimized Reward-Grounded Environment Synthesis for Agent RL*. arXiv:2608.14312. https://arxiv.org/abs/2608.14312
51. Fan, Z., Yu, T., Cai, Y., Zhou, J., Guan, J., et al. (2026). *Environment Evolution for Terminal Agents*. arXiv:2609.04128. https://arxiv.org/abs/2609.04128
52. Seely, J., Imajuku, Y., Zhao, T., Cetin, E., Jones, L. (2025). *Sudoku-Bench: Evaluating creative reasoning with Sudoku variants*. arXiv:2505.16135. https://arxiv.org/abs/2505.16135
53. Tang, E. (2026). *Generalization or Memorization? Brittleness Testing for Chess-Trained Language Models*. arXiv:2605.17565. https://arxiv.org/abs/2605.17565
54. Yang, Y., Liu, J., He, Y., Zhang, L., Schölkopf, B., et al. (2026). *Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR* (TAC). arXiv:2606.25178. https://arxiv.org/abs/2606.25178
55. Luo, L., Zhang, G., Xu, H., Li, R., Fang, C., et al. (2026). *Beyond Forgetting: Diagnosing and Harnessing Shared Reasoning in Continual RLVR*. arXiv:2608.18574. https://arxiv.org/abs/2608.18574
56. Cheng, Z., Hao, S., Liu, T., Zhou, F., Xie, Y., et al. (2025). *Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective* (Guru). arXiv:2506.14965. https://arxiv.org/abs/2506.14965
57. Shi, J., Yang, J., Liu, J., et al. (2025). *KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation*. arXiv:2505.14552. https://arxiv.org/abs/2505.14552
58. Liu, Z., Sims, A., Duan, K., et al. (2025). *GEM: A Gym for Agentic LLMs*. arXiv:2510.01051. https://arxiv.org/abs/2510.01051
59. Guertler, L., Cheng, B., Yu, S., et al. (2025). *TextArena*. arXiv:2504.11442. https://arxiv.org/abs/2504.11442
