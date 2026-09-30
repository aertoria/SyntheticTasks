# Dynamic benchmark generation, adversarial filtering, and open-ended/quality-diversity task generation

*Scope: dynamic and contamination-resistant evaluation, adversarial data filtering, and open-ended / quality-diversity (QD) task generation, 2019–Sep 2026, read as generators of harder SFT/RL tasks from easy seeds. Compiled 2026-09-30. Verification: 29 researcher entries checked against primary sources (arXiv full text or HTML); 10 corrected, 0 dropped, 5 added (KUMO, DynaMath, Putnam-AXIOM, MR-GSM8K, MMLU-Pro). A further 30+ secondary citations were resolved by arXiv ID. Every number below appears in the cited source.*

## TL;DR

- **Lift saturated seeds into parameterized generators and adapt difficulty online. This is the best-supported fix for zero-advantage GRPO groups.** RLVE (400 hand-built environments, each with an integer difficulty `d` and a sliding window promoted at 90% accuracy) gave +3.37 points average on six benchmarks when continuing from the already-saturated ProRL-1.5B-v2. Continuing the original RL gave +0.49 with more than 3× the compute. SCALER automatically lifts 2,739 CodeContests problems into scale-parameterized environments and reached a 54.25 average for Qwen3-4B-Base, versus 52.04 (MATH-7.5k), 51.08 (DeepMath-103K) and 53.52 (RLVE).
- **Structural operators on an executable intermediate representation give monotone, verifiable difficulty.** Intermediate representations here are reasoning graphs, computational DAGs and programs. Examples: DARG depth +4 took Claude-3-Opus from about 95% to 41.0% on GSM8K. GSM-Infinite accuracy declines along a sigmoid as operation count grows, and its reverse mode is consistently harder than forward mode. The label comes from executing the graph, so correctness is close to free.
- **Free-form "make it harder" prompting is unreliable. Composing from verified parts is not.** In CHASE's direct-generation baseline, 34 of 100 Evol-Instruct-style hard math problems had errors, and frontier models still scored 82.5–88.9% on them. CHASE's bottom-up pipeline, which verifies every step, had 6, 3 and 7 errors per 100 audited items (QA, Code, Math) while holding state-of-the-art models to about 40–60%.
- **"Keep what strong models fail" enriches label noise, so every adversarially selected item needs independent verification.** HLE estimates 15.4% expert disagreement on its public set. HLE-Verified kept only 668 of 2,500 items unchanged: 1,143 were revised and 689 were marked uncertain. Models gain 30–40 points on the items whose original statement or answer was wrong. AFLite-style filtering oversamples low-annotator-agreement items and makes model rankings depend on which adversary was used.
- **Select items by a policy-relative signal, and measure which signal works.** SwS keeps only synthesized problems with policy accuracy in [25%, 75%]; about 35% of generated problems land there. In DataEnvGym, learning gains peak at mid-range difficulty. In eva, regret-style advantage selection reached 60.0–60.1 on Arena-Hard versus 57.5 for uniform evolution, but reward variance (54.8) and inverse advantage (52.3) did worse than uniform. CompassPlay, which rewards gradient alignment with target tasks, beat Absolute Zero's difficulty reward (+1.5 coding, +2.7 math) and needed 40% fewer GPU-hours in Lean.
- **Method-breaking and functional perturbations of solved seeds are a cheap source of signal.** On MATH-P-Hard, o1-mini dropped 16.49%. On Putnam-AXIOM variations, o1-preview dropped 19.6 points (46.8% relative). EvoEval's "Subtle" set costs 24.0% on average at roughly unchanged difficulty. When a seed is solved, try a hard variant of it before discarding it.
- **Every synthetic task should ship with a checker, a known-good solution and known-bad candidates.** This is Self-Challenging's Code-as-Task: the example solution must pass and the failure cases must fail. Keep checkers hidden from proposers. In the Darwin Gödel Machine, objective hacking was more frequent when the checking functions were visible.
- **Diversity is a first-order variable.** Keep a descriptor-indexed QD archive (ACES uses 21,700 skill-combination niches) plus a novelty filter; ACD still accepted about 20% of proposals as novel after 5,000 generations. In Self-Challenging, training on 200 synthetic tasks slightly degraded test performance, 400 tasks gave marginal gains, and 800 tasks gave steady gains. RLVE and SCALER both improve as the number of environments grows.
- **Reframing a solved task yields hard tasks with derivable labels.** Options include locating the error in a solution, scoring a solution, expanding answer options, or probing a sub-ability. In MR-GSM8K, GPT-4-injected errors produced diagnostic training data that was more than 90% accurate, even though GPT-4 correctly identified incorrect solutions only about 40% of the time. MMLU-Pro expanded 4 options to 10 but had to remove 1,953 false-negative options (correct answers generated as distractors) in its MMLU-derived part alone.
- **Guard against cheap hardness and lucky rewards.** Check whether added difficulty survives tool access (BBEH), since length, distractors and trivia inflate difficulty without adding reasoning. Score a seed by its worst case across variants rather than its average (DynaMath: worst-case accuracy is at most about 50% of average-case for all 14 VLMs tested). TRACE found about 28% spurious guessing in mid-sized models, meaning correct answers with invalid reasoning.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| DyVal / DyVal 2 (MPA) | 2023 / 2024 | [2309.17167](https://arxiv.org/abs/2309.17167), [2402.14865](https://arxiv.org/abs/2402.14865) | Arithmetic, logic, graph algorithms; rewrites of MMLU/GSM8K/ARC-C/BBH | Eval; SFT augmentation | DAG nodes/depth/width, extra random links, irrelevant descriptions, order shuffle; paraphrase, extra context, extra plausible option | Label by executing the DAG; MPA judge agent checks each rewrite |
| DARG | 2024 | [2406.17271](https://arxiv.org/abs/2406.17271) | GSM8K, BBQ, BBH Navigate, Dyck | Eval; SFT | Reasoning-graph extraction, then numeric/depth/width increments, then re-render in original style | Rule-based label from the graph; code-executing solver must reproduce it |
| GSM-Infinite | 2025 | [2502.05252](https://arxiv.org/abs/2502.05252) | GSM-style math, long context | Eval | Op-count scaling, reverse mode, 2/3-entity abstraction, spider-topology noise | Graph executed at generation; noise edges point only outward from the core graph |
| RLVE | 2025 | [2511.07317](https://arxiv.org/abs/2511.07317) | 400 procedural environments (math, algorithms, puzzles, NP-complete problems) | RL | Integer difficulty `d` per environment; sliding difficulty window promoted by accuracy | Hand-written generator and verifier, often using solve/verify asymmetry |
| SCALER | 2026 | [2601.04809](https://arxiv.org/abs/2601.04809) | 2,739 environments lifted from CodeContests | RL | Extract scale parameters; proportional difficulty controller; environment retirement | Several reference solutions must agree (breadth check); instance diversity required (depth check) |
| KUMO *(added)* | 2025 | [2504.02810](https://arxiv.org/abs/2504.02810) | LLM-proposed domains; partially observable deduction games | Eval | More candidate truths and actions; new domains | SAT engine generates outcomes consistent with the hidden truth |
| DynaMath *(added)* | 2024 | [2411.00836](https://arxiv.org/abs/2411.00836) | Visual math (VLMs) | Eval | Seed-as-program variants (numbers, geometry, function type, color) | Seed program computes each variant's answer; worst-case-over-variants scoring |
| Putnam-AXIOM Variation *(added)* | 2025 | [2508.08292](https://arxiv.org/abs/2508.08292) | Putnam competition math | Eval | Variable renaming, constant change, rephrasing via scripts | Script recomputes the answer; SymPy equivalence check |
| Benchmark Self-Evolving | 2024 | [2402.11443](https://arxiv.org/abs/2402.11443) | GSM8K, CLUTRR, StrategyQA, BoolQ | Eval | Question alternating/complicating, paraphrase, context noise, polarity reversal, sub-ability probes | GPT-4 pre-filter plus double verification (answer passes, wrong option fails) |
| EvoEval | 2024 | [2403.19114](https://arxiv.org/abs/2403.19114) | HumanEval-derived code | Eval | Difficult, Creative, Subtle, Combine, Tool_Use; Verbose/Concise | GPT-4 solution and tests, self-consistency, manual review of every problem |
| MATH-Perturb | 2025 | [2502.06453](https://arxiv.org/abs/2502.06453) | MATH level-5 | Eval | Simple vs hard (method-breaking) minimal edits | PhD annotators, independent cross-validation, re-check where o1-mini disagrees |
| CHASE | 2025 | [2502.14678](https://arxiv.org/abs/2502.14678) | Long-document QA, repository code, math | Eval (SFT pilot) | Bottom-up composition, answer-first documents with distractors, continuation chaining (depth 2–8), rejection sampling | Every step verified by LLM verifier(s); test code must pass; verifier ensemble |
| TaskCraft | 2025 | [2506.10055](https://arxiv.org/abs/2506.10055) | Multi-tool agent tasks | SFT (RL initialization) | Atomic tool task, then depth extension (identifier-obfuscation hops) or width extension (merge) | Answer inherited from a verified atomic task; superset and leakage checks |
| AutoBencher | 2024 | [2407.08351](https://arxiv.org/abs/2407.08351) | Knowledge QA, math, multilingual, safety | Eval | Search over dataset descriptions for difficulty, novelty and salience | Privileged information: retrieval, Python/sympy, translation |
| MR-GSM8K *(added)* | 2023 | [2312.17080](https://arxiv.org/abs/2312.17080) | GSM8K meta-reasoning | Eval (training-data recipe) | Reframe "solve" as "score this solution / find first error step / explain error"; POT and reversed variants | Expert double annotation; error injection into correct solutions for training data |
| MMLU-Pro *(added)* | 2024 | [2406.01574](https://arxiv.org/abs/2406.01574) | Multi-domain multiple choice | Eval | Filter trivial items with 8 small models, expand 4 to 10 options, add reasoning-heavy sources | Two-round expert review; LLM flags false-negative options |
| BBEH | 2025 | [2502.19187](https://arxiv.org/abs/2502.19187) | 23 reasoning tasks | Eval | Harder task per BBH skill; iterate until reference models score below 70%; close shortcuts | Programmatic or known-correct answers |
| AF / AFLite | 2019 / 2020 | [1905.07830](https://arxiv.org/abs/1905.07830), [2002.04108](https://arxiv.org/abs/2002.04108) | Commonsense, NLI | Eval (and training) | Discriminator-driven distractor swapping; predictability-based removal | Human gold labels; human accuracy re-checked |
| ANLI / Dynabench | 2019 / 2021 | [1910.14599](https://arxiv.org/abs/1910.14599), [2104.14337](https://arxiv.org/abs/2104.14337) | NLI, QA, sentiment, hate speech | Training and eval | Human writers attack a live model over multiple rounds | Independent human validation of model-fooling examples |
| HLE / HLE-Verified | 2025 / 2026 | [2501.14249](https://arxiv.org/abs/2501.14249), [2602.13964](https://arxiv.org/abs/2602.13964) | Expert closed-ended QA | Eval | Pre-submission filter against frontier LLMs | Two-round expert review; post-hoc audit and repair |
| UQ | 2025 | [2508.17580](https://arxiv.org/abs/2508.17580) | Unanswered Stack Exchange questions | Eval | Mine naturally unsolved questions | Validators exploiting the generator-validator gap, then community verification |
| MathDuels (+SKATE) | 2026 / 2025 | [2604.21916](https://arxiv.org/abs/2604.21916), [2508.06111](https://arxiv.org/abs/2508.06111) | Scalar-answer math; code-output prediction | Eval | Meta-prompting, hardening rounds, cross-model duels, Rasch calibration | Independent verifier picks the reference answer; execution (SKATE) |
| DataEnvGym | 2024 | [2410.06215](https://arxiv.org/abs/2410.06215) | MATH, LiveCodeBench, GQA, NaturalBench, MnMs | SFT | Teacher agent conditioned on student errors and skills | Teacher LLM; no formal guarantee |
| EvalTree (+AutoDetect) | 2025 / 2024 | [2503.08893](https://arxiv.org/abs/2503.08893), [2406.16714](https://arxiv.org/abs/2406.16714) | MATH, WildChat, general | SFT | Capability tree, then weakness profile, then targeted synthesis | Generator LLM |
| SwS | 2025 | [2506.08989](https://arxiv.org/abs/2506.08989) | Math RLVR | RL | Failure mining, concept recombination, synthesis, [25, 75]% accuracy band | Teacher majority ≥50%, student agreement ≥25%, quality filters |
| eva | 2024 | [2411.00062](https://arxiv.org/abs/2411.00062) | Chat alignment | RL (DPO/RLOO/…) | Regret-weighted prompt selection plus EvolInstruct-style rewrites | Reward model only (no ground truth) |
| ACCEL | 2022 | [2203.01302](https://arxiv.org/abs/2203.01302) | MiniGrid mazes, BipedalWalker | RL | Small edits to high-regret levels | Simulator; regret gate before a level enters the buffer |
| OMNI / OMNI-EPIC | 2023 / 2024 | [2306.01711](https://arxiv.org/abs/2306.01711), [2405.15568](https://arxiv.org/abs/2405.15568) | Simulated embodied tasks | RL | Foundation model proposes next task; interestingness filter; environment, reward and success code | Code success function (72.7% human agreement) |
| ACD | 2025 | [2502.07577](https://arxiv.org/abs/2502.07577) | Open-ended task families | Eval | Scientist model writes task-family code; novelty filter; difficulty adaptation | Task `score()` code or GPT-4o judge |
| ACES | 2023 | [2310.10692](https://arxiv.org/abs/2310.10692) | Python programming puzzles | Eval | MAP-Elites over 20-skill combinations; difficulty-seeking generation | Execution (`f(g())`); never-solved puzzles discarded |
| Rainbow Teaming (+QDAIF) | 2024 / 2023 | [2402.16822](https://arxiv.org/abs/2402.16822), [2310.13032](https://arxiv.org/abs/2310.13032) | Safety, QA, cybersecurity | SFT (safety) | MAP-Elites grid, targeted mutation, BLEU similarity filter, pairwise judge | Judge LLM / Llama Guard; oracle model for QA |
| AC/DC | 2026 | [2604.14969](https://arxiv.org/abs/2604.14969) | Natural-language tasks co-evolved with merged LLMs | Model discovery / analysis | Scientist makes tasks harder, easier or novel; impossible-task filter; Dominated Novelty Search | Self-solve reflection; 97.8% human-rated correct |
| Self-Challenging | 2025 | [2506.01716](https://arxiv.org/abs/2506.01716) | Multi-turn tool use | RL | Challenger explores tools, then writes a Code-as-Task | Verifier function; example solution must pass; failure cases must fail |
| CompassPlay | 2026 | [2609.32228](https://arxiv.org/abs/2609.32228) | Coding self-play; Lean 4 | RL | Proposer reward = cosine of gradients with reference tasks | Execution; Lean kernel |

## Method notes

*Family A: structure-based and parametric generation. The label comes from execution.*

### DyVal / DyVal 2 (Meta Probing Agents) — DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks (Zhu et al., 2023)
Link: https://arxiv.org/abs/2309.17167 (ICLR 2024 spotlight; code at aka.ms/dyval). DyVal 2 / MPA: *Dynamic Evaluation of Large Language Models by Meta Probing Agents* (Zhu et al., 2024, ICML 2024), https://arxiv.org/abs/2402.14865 (code in microsoft/promptbench).
- **Mechanism**
  - DyVal samples a DAG, executes it for the label, and renders it with a description function.
  - Tree-based T-DAGs cover arithmetic, boolean, deductive and abductive logic; general G-DAGs cover reachability and max-sum path. Leaves hold random values and internal nodes hold operations or relations.
  - MPA instead rewrites existing benchmark items. A GPT-4-Turbo probing agent (temperature 0.7) applies five psychometric principles: paraphrase the question, paraphrase the choices, permute the choices (done in code, no agent), add non-essential context, and add a plausible but wrong option E.
  - A GPT-4-Turbo judge agent (temperature 0) checks that the rewrite still tests the same concept, or that option E is relevant but not an alternative correct answer.
- **How it makes tasks harder**
  - DyVal: more nodes and depth, extra random links between nodes, embedded irrelevant descriptions, shuffled description order.
  - MPA: surface-form shift, distractor context and extra options, which break memorized forms of seed items.
- **Correctness / verification**: DyVal labels are correct by construction. MPA depends on the judge; the authors note that at the time only GPT-4-class models could reliably play both the probing and judging roles.
- **Difficulty control**: four complexity levels, D1–D4. For G-DAG tasks, node counts are {7, 10, 15, 20} with at most {3, 4, 6, 8} links per node. MPA toggles principles per ability (language understanding, problem solving, domain knowledge).
- **Reported results**
  - Models from Flan-T5-large to GPT-4 degrade as complexity rises. Xwin-13B, phi-1.5 and WizardMath-13B scored 0 on every DyVal task.
  - Fine-tuning Llama2-13B-chat on DyVal data improved few-shot results on GSM8K, SVAMP, FOLIO, RACO, DP and LCS, and on harder-than-training DyVal samples.
  - MPA with GPT-4-Turbo (original → probed): MMLU 84.40 → 68.86, GSM8K 95.22 → 88.50, ARC-C 96.16 → 84.67.
  - Fine-tuning GPT-3.5-Turbo on MPA rewrites of the MMLU and ARC-C training splits, plus the originals, gave about 2% average gains.
- **Limitations / failure modes**: templated, unnatural text; narrow algorithmic tasks. MPA mostly tests robustness rather than deeper reasoning, and judge errors let meaning drift through.
- **How to reuse with easy seed tasks**
  - For seeds with an executable core (arithmetic, logic, graph algorithms, SQL or table operations), write a graph sampler plus a renderer. Expose node count, depth, extra links and distractor text as curriculum knobs.
  - Use MPA-style rewrites (extra option, extra context, paraphrase) as cheap anti-memorization augmentation for saturated seeds, followed by a judge check and an answer re-check.

### DARG — DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph (Zhang et al., 2024)
Link: https://arxiv.org/abs/2406.17271
- **Mechanism**
  - An LLM extracts a reasoning graph from each benchmark item using in-context examples. For GSM8K, nodes are numbers and edges are operations.
  - A rule-based function computes the label from the graph. If it does not match the gold label, the LLM is re-prompted at high temperature until it does.
  - The graph is perturbed and decoded back into text in the benchmark's style.
  - A code-augmented LLM agent solves the new text by writing and executing code, and generation repeats until its answer matches the graph label.
- **How it makes tasks harder**
  - GSM8K has three independent axes:
    - Numerical complexity, counted as unit additions: +2, +4, +6, +8, by resampling node values.
    - Depth, the longest leaf-to-answer path: +1 to +4, by splitting the start node of the longest path into two nodes with the same numerical complexity.
    - Width, extra node pairs outside the longest path: +1 to +4, by decomposing start nodes of the other paths.
  - BBH Navigate: linear graphs grow by +2, +4, +8 or +16 nodes, splitting an action node into several with the same net effect.
  - Dyck: the input and output parts grow by +2 to +16.
- **Correctness / verification**: labels are computed from the graph, a code-executing solver must reproduce them on the rendered text, and a human evaluation of generated samples is reported in the paper's appendix.
- **Difficulty control**: integer increments on each axis, with each axis reported separately.
- **Reported results**
  - On 15 LLMs, almost all models decline as complexity grows.
  - Claude-3-Opus went from about 95% on original GSM8K to 41.0% at depth +4 (reported drop 54.2).
  - Models show more social bias on BBQ at higher complexity.
  - Mistral-7B and Llama2-7B fine-tuned on DARG data outperformed the same models fine-tuned on an equal amount of GSM8K training data, measured on DARG-evolved test sets. The training CoTs were generated by GPT-4 Turbo from question plus graph.
- **Limitations / failure modes**: needs faithful graph extraction and fluent re-rendering; only works where explicit graph structure exists; perturbed stories can become unnatural.
- **How to reuse with easy seed tasks**: convert solved seeds (pass rate near 100%) into graphs, then emit a ladder of depth and width +1…+4 siblings with verified labels. Train RL where policy pass@k sits in the middle band and step up as it saturates. Contamination caveat: hold out some graph operators and renderers for evaluation.

### GSM-Infinite — GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? (Zhou et al., 2025)
Link: https://arxiv.org/abs/2502.05252 (open-source generator)
- **Mechanism**
  - Problems are sampled as computational graphs (nodes are variables, edges are operations) with a chosen operation count.
  - They are rendered through three real-world templates: children-animal-zoo, teachers-school-district and awards-movies-festival.
  - There are three subsets: Symbolic, Medium (2-entity variables, implicit +/−) and Hard (3-entity variables, implicit ×/÷).
  - Reverse mode defines abstract variables before specific ones, which forces implicit − and ÷ and backward reasoning.
- **How it makes tasks harder**
  - More operations; reverse mode; 3-entity abstraction.
  - Context scaling by noise. Edges point outward from core-graph nodes to noise nodes, so noise cannot enter the solution. Most noise edges attach to core nodes, making them semantically close, which defeats RAG retrievers.
- **Correctness / verification**: the graph is executed at generation; outward-only noise edges guarantee that distractors are irrelevant to the answer.
- **Difficulty control**: operation count (unbounded), subset, forward vs reverse mode, and context length. Metrics are AUC over operation count and the first operation count at which accuracy falls below 50% or 10%.
- **Reported results**
  - Accuracy follows a consistent sigmoid or exponential decay as operation count grows.
  - Exponentially more inference compute (repeated sampling) buys only linear gains.
  - Reverse problems are harder than forward ones.
  - On the zero-noise Hard subset, DeepSeek-R1's accuracy first falls below 50% at 100 operations; o3-mini's does at 70.
  - With noise, RAG retrievers could not distinguish essential chunks from noise chunks.
- **Limitations / failure modes**: templated language; narrow arithmetic domain; much of the difficulty is bookkeeping and length.
- **How to reuse with easy seed tasks**: an unbounded, verifiable warm-up environment. Sample RL prompts at the operation count on the steep part of the current policy's sigmoid and step up as it saturates. Mix in reverse-mode and noisy variants to block forward-only heuristics.

### RLVE — RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments (Zeng et al., 2025)
Link: https://arxiv.org/abs/2511.07317 (ICML 2026; code released)
- **Mechanism**
  - Each environment has an input template, a problem generator `P_d` conditioned on an integer difficulty `d`, and a verifier.
  - Each environment keeps a difficulty range `[ℓ, h]` and samples `d` from it.
  - After at least `τ_num` rollouts at level `h`, if accuracy is ≥ `τ_acc`, `h` increments and the statistics reset. `ℓ` is capped so the window width never exceeds `d_Δ`.
  - All 400 RLVE-Gym environments train jointly with DAPO-style dynamic sampling.
- **How it makes tasks harder** (example environments):
  - Sorting: array length N ∝ 3×1.1^d.
  - Integration: an elementary function F whose expression tree has d+2 nodes; the model is asked to integrate F′.
  - Hamiltonian path: N = d+3 vertices.
  - Sudoku: max(N, M) ≤ d+2.
  - A bubble-sort permutation problem: N = d+3.
- **Correctness / verification**
  - Generators are hand-engineered and exploit the solve/verify asymmetry.
  - Integration is checked by differentiating the answer and comparing with F′, never by integrating.
  - Hamiltonian-path graphs have a planted random-permutation path and are checked edge by edge.
  - Sudoku grids are made by random equivalence transformations of a canonical solution and then masked.
  - Unparsable output gets a reward of −1.
- **Difficulty control**: per-environment `d`, promoted online. Defaults: `τ_acc` = 0.9, `τ_num` = 8 × rollouts per problem, `d_Δ` = 4. The key metric is the effective prompt ratio, the share of prompts whose rollouts do not all get the same reward.
- **Reported results**
  - Data-saturation scenario: ProRL-1.5B-v2 had already had more than 20,000 H100-hours of RLVR. Continuing with RLVE gave +3.37 points absolute average over six benchmarks; continuing the original RL gave +0.49 using more than 3× the compute.
  - Compute-constrained scenario: about 2 points better than training on DeepMath-103K.
  - More environments consistently improve held-out performance, and adaptive difficulty keeps the effective prompt ratio higher than static distributions.
- **Limitations / failure modes**: manual environment engineering (400 environments); algorithmic and puzzle skew; transfer to open-ended domains is untested.
- **How to reuse with easy seed tasks**: wrap each saturated seed family in a generator with an integer knob and an asymmetric checker, and apply the window-promotion rule per family. Track the effective prompt ratio as the health metric.

### SCALER — SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning (Xu et al., 2026)
Link: https://arxiv.org/abs/2601.04809
- **Mechanism**
  - An LLM agent (GLM-4.6) extracts meta-information from each programming problem: scale parameters and output requirements. Only problems with unique numeric, array or string outputs are kept.
  - The agent writes a randomized test-case generator for each problem.
  - Two checks validate the generator. The breadth check samples diverse scale settings, runs several independent ground-truth solutions, and requires them to agree; this catches malformed generators and non-unique outputs. The depth check calls the generator repeatedly at a fixed scale and requires diverse outputs; this blocks narrow patterns and reward hacking.
  - A binary search over a global scale factor finds the largest configuration that fits the prompt-length and runtime budget, which fixes the maximum difficulty D.
  - Code runs in SandboxFusion.
- **How it makes tasks harder**: larger input scales and constraints (array length, edge count) within each lifted problem.
- **Correctness / verification**: labels come from executing several agreeing reference solutions.
- **Difficulty control**
  - Continuous difficulty per environment: `d_{t+1} = clip(d_t + β·(acc_t − τ), 0, D)`, realized as a multiset of integer levels whose mean matches `d_{t+1}`.
  - Environment curation: an environment is retired when its learning slope is ≤ 0 over K_slope = 10 steps, or when it sits at zero accuracy or saturates at maximum difficulty (K_zero = K_sat = 5). Replacements are sampled from the pool.
  - 64 active environments, one prompt from each per step; GRPO with 8 responses.
- **Reported results**
  - 4,973 CodeContests problems yielded 2,739 environments.
  - Qwen3-4B-Base, average over MATH-500, AMC23, AIME24, MMLU-Pro and BBEH: SCALER 54.25; RLVE 53.52; MATH-7.5k 52.04; DeepMath-103K 51.08. AIME24: 27.29 vs 24.16 (MATH-7.5k) and 25.42 (RLVE).
  - Qwen3-1.7B-Base: SCALER 40.18; DeepMath 39.07; MATH-7.5k 38.89; RLVE 37.80.
  - RLVE beat SCALER on BBEH at 4B (16.52 vs 14.56).
- **Limitations / failure modes**: needs scalable, unique-output problems with correct and efficient reference solutions; prompt-length caps bound the maximum scale.
- **How to reuse with easy seed tasks**: this is the automated route from a fixed seed to an unbounded generator. Extract size and constraint parameters, validate with at least two reference solvers, drive scale with a proportional controller toward target accuracy τ, and retire environments with no learning slope.

### KUMO *(added)* — Generative Evaluation of Complex Reasoning in Large Language Models (Lin et al., 2025)
Link: https://arxiv.org/abs/2504.02810
- **Mechanism**
  - A game instance is a set of candidate truths, a set of actions whose outcomes rule truths out, and a natural-language knowledge book explaining how outcomes eliminate truths.
  - One valid truth is hidden. The player repeatedly picks an action, sees its outcome, and eventually predicts the truth, aiming to use as few actions as possible.
  - Pipeline: o1 proposes domains (for example medical diagnosis, chemical detection, education) and generates seed configurations (truths, actions, outcomes); an SAT-based engine samples the action subset and generates outcomes consistent with the valid truth; GPT-4o writes the knowledge book.
- **How it makes tasks harder**: more candidate truths and actions; new domains with different truth-action graph topologies; partial observability that requires multi-turn evidence gathering.
- **Correctness / verification**: the symbolic engine guarantees that outcomes are consistent with exactly the valid truth; success rate and relative action count are computed automatically.
- **Difficulty control**: Easy has 4 truths and 6 actions; Hard has 12 truths and 16 actions. Domain count is unbounded.
- **Reported results**
  - 23 LLMs on 5,200 tasks across 100 domains.
  - Many LLMs beat university students in the Easy setting; reasoning models reach student level in Hard.
  - KUMO scores correlate with newer benchmarks (MMLU-Pro, LongBench-V2, LiveBench-Reason).
  - Fine-tuning Qwen2.5-0.5B and 3B on optimal-search "golden trajectories" helped in-distribution but degraded sharply out of domain and across difficulty transitions (Easy to Hard and Hard to Easy).
  - Domains with similar entity-relation graph topologies produce similar performance.
- **Limitations / failure modes**: a single game family (elimination deduction); LLM-written knowledge books can be ambiguous; efficiency is scored against an optimal searcher.
- **How to reuse with easy seed tasks**: convert static classification or diagnosis seeds into multi-turn, partially observable games with an SAT or CSP backend. The reward is exact, and difficulty is set by the size of the truth and action sets. Diversify the graph topologies, not just the surface domains, since topology predicts difficulty.

### DynaMath *(added)* — DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models (Zou et al., 2024)
Link: https://arxiv.org/abs/2411.00836 (ICLR 2025)
- **Mechanism**: 501 human-curated seed questions, each written as a Python program that generates the image, the question and the answer. Running the program yields concrete variants: numerical values, geometric transformations, function types, colors and other variant types.
- **How it makes tasks harder**: less by raising per-item difficulty than by demanding consistency. A seed counts as solved only if every variant is solved.
- **Correctness / verification**: the seed program computes the ground truth for every variant.
- **Difficulty control**: variant type per seed (one type per seed; combining types is listed as future work); number of variants.
- **Reported results**
  - 10 variants per seed gave 5,010 problems, and 14 VLMs were evaluated.
  - For every model, worst-case accuracy (all 10 variants correct) was close to or below 50% of average-case accuracy.
  - Best zero-shot average was Claude-3.5 at 64.8%, against a human score of 77.3%.
  - Failures are consistent rather than random: GPT-4o fails one variant with 90% repetition consistency, insisting the corner of a shifted |x| is at x = 0.
- **Limitations / failure modes**: seeds are hand-written; one variant type per seed; visual domain.
- **How to reuse with easy seed tasks**: write seeds as programs and use a worst-case-over-k-variants reward, or group variants into one GRPO group. A seed the policy "solves" on average often fails on some variants, and that failure is the missing signal.

### Putnam-AXIOM *(added)* — Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs (Gulati et al., 2025)
Link: https://arxiv.org/abs/2508.08292 (code: github.com/brando90/putnam-axiom)
- **Mechanism**
  - 522 Putnam problems with boxed final answers form the Original set.
  - For 100 of them, Python scripts generate functional variations: 63 variable-only and 37 constant-plus-variable.
  - Variable changes rename symbols and leave the answer unchanged. Constant changes alter numeric constants, which changes intermediate steps and the final answer (for example the year 2011 becomes 4680 and the answer changes from 10053 to 23398). Problems are also rephrased.
  - Scoring canonicalizes TeX into SymPy and tests equivalence.
- **How it makes tasks harder**: it removes the memorization path; the paper frames variants as equally difficult.
- **Correctness / verification**: scripted recomputation of answers; problems with problem-specific constants, non-generalizable solutions or non-boxable answers were excluded from variation.
- **Difficulty control**: none beyond the choice of original versus variant; an unlimited stream of variants.
- **Reported results**: o1-preview scored 41.9% on the Original set and dropped 19.6 points (46.8% relative) on the paired variations. All 18 other models trended down, 10 with non-overlapping 95% confidence intervals. The paper also proposes Teacher-Forced Accuracy (TFA), a lightweight proxy for scoring reasoning traces.
- **Limitations / failure modes**: only about 19% of problems were variable; variants are hand-scripted.
- **How to reuse with easy seed tasks**: variabilize solved competition seeds. Pairing the original with its variant in the same batch detects and penalizes memorized-answer shortcuts.

*Family B: LLM rewriting, composition and reframing, with verification.*

### Benchmark Self-Evolving — Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation (Wang et al., 2024)
Link: https://arxiv.org/abs/2402.11443 (code: github.com/NanshineLoong/Self-Evolving-Benchmark)
- **Mechanism**
  - Four GPT-4 agents: instance pre-filter, instance creator, instance verifier and candidate-option formulator.
  - Six reframing operations:
    - Scalable: question alternating; question complicating (more reasoning steps, context unchanged).
    - Robust: context paraphrasing; context noising; polarity reversing (edit key details so the answer flips).
    - Fine-grained: sub-ability questions on task planning, implicit knowledge and relevant-context retrieval.
- **How it makes tasks harder**: extra reasoning steps, adversarial edits that flip the answer, and probing of sub-skills behind the seed.
- **Correctness / verification**
  - The pre-filter keeps only instances the system can handle.
  - Double verification: the new (context, question, answer) must verify as correct, and a generated wrong option must verify as inconsistent.
  - Correction to the researcher entry: the human audit sampled 5 instances per reframing operation per dataset that ChatGPT answered incorrectly (115 total, 110 correct, 95.7%). It is an audit of "hard" items, not a random sample.
- **Difficulty control**: categorical, by choice of operation. Question complicating disrupted models most, followed by polarity reversing and question alternating; paraphrasing and noising had limited impact.
- **Reported results**: most models decline versus their original results, and gaps between models and across tasks widen. GPT-4, which also generated the items, fell from 100.0 to 85.00 on GSM8K scalable-evaluation instances.
- **Limitations / failure modes**: the same model family generates and verifies, so self-preference is a risk; about 5% residual error among model-failed items is harmful as an RL reward; limited to QA and reading-comprehension formats.
- **How to reuse with easy seed tasks**: apply question complicating and polarity reversing to saturated seeds. Add sub-ability questions as auxiliary SFT or RL tasks. Adopt the double-verification gate (the answer passes and a distractor fails) before trusting any reward.

### EvoEval — Top Leaderboard Ranking = Top Coding Proficiency, Always? EvoEval: Evolving Coding Benchmarks via LLM (Xia et al., 2024)
Link: https://arxiv.org/abs/2403.19114
- **Mechanism**: GPT-4 evolves HumanEval seeds with targeted instructions.
  - Semantic-altering transformations: Difficult (add constraints, replace common requirements with less common ones, add reasoning steps), Creative, Subtle (small change to one requirement), Combine (merge two problems) and Tool_Use (main problem plus helper functions).
  - Semantic-preserving transformations: Verbose and Concise.
  - For the altering sets, GPT-4 writes a new solution and test inputs, and self-consistency between solutions on those inputs is checked.
- **How it makes tasks harder**: stacked constraints, rarer requirements, composition, and small semantic flips that punish memorized solutions.
- **Correctness / verification**: self-consistency plus manual examination and fixing of every problem, ground truth and test set. The preserving variants reuse HumanEval ground truth.
- **Difficulty control**: categorical by transformation; 100 problems per altering set; transformations can be re-applied iteratively.
- **Reported results**: across 51 LLMs, an average drop of 39.4% versus HumanEval, ranging from 19.6% to 47.7%, with drastic ranking changes. On Subtle, which is roughly the same difficulty, the average drop is 24.0%. Instruction-tuned models are brittle to rewording, and composing and decomposing problems is a weakness.
- **Limitations / failure modes**: manual validation required; small; toy-function seeds.
- **How to reuse with easy seed tasks**: use Difficult, Subtle and Combine as mutation operators. Replace manual review with differential testing (at least two independent reference solutions agree on generated inputs) plus Code-as-Task failure cases. Subtle variants are especially valuable for RL on saturated coding seeds.

### MATH-Perturb — MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations (Huang et al., 2025)
Link: https://arxiv.org/abs/2502.06453
- **Mechanism**
  - 12 PhD-student annotators with strong math backgrounds minimally edit level-5 MATH problems.
  - MATH-P-Simple: the original method still applies but the answer changes.
  - MATH-P-Hard: the edit invalidates the original solution method while keeping the text close to the original.
  - Answers are always required to differ from the originals.
- **How it makes tasks harder**: it breaks the learned solution template, not the surface form.
- **Correctness / verification**: annotators double-check their work; an independent annotator cross-validates each problem; every case where o1-mini's answer disagreed with the annotation was manually re-checked. Result: 279 pairs (164 from the MATH train split, 115 from test).
- **Difficulty control**: binary (simple vs hard) under a minimal-edit constraint; edit distance and embedding similarity to the original are reported.
- **Reported results**
  - On MATH-P-Hard, o1-mini dropped 16.49% and gemini-2.0-flash-thinking dropped 12.9%.
  - A new memorization failure: models blindly apply learned skills without checking whether they apply. Outputting the original answer verbatim is only a minority of errors (for example 9 of o1-mini's 60 Hard errors); most failures ignore the modified assumption in subtler ways.
  - Using the original problem and solution as a one-shot example helps on Simple, but on Hard its helpful effect is offset by a misleading effect.
- **Limitations / failure modes**: expert labor; small; no automatic generator.
- **How to reuse with easy seed tasks**: automate method-breaking edits with an LLM, then require (a) the new answer to differ from the original and (b) confirmation by a CAS, a program or independent solvers. Put the seed and its hard variant together in the same RL batch as contrastive pairs.

### CHASE — How to Get Your LLM to Generate Challenging Problems for Evaluation (Patel et al., 2025)
Link: https://arxiv.org/abs/2502.14678
- **Mechanism**: build hard problems bottom-up from simpler components, each verified separately.
  - CHASE-QA: generate scenarios, then QA pairs whose answer points are spread over several documents, then similar-but-irrelevant QA pairs, then documents containing both.
  - CHASE-Code: generate helper functions in a domain, then a problem statement and answer code that must call 4–6 of them (objectives with 6–8 sub-goals), then test code. The result is hidden in a repository with m = 100 irrelevant helpers spread across 10 files.
  - CHASE-Math: split a seed word problem into context and question, then repeatedly generate a continuation whose context silently uses the previous answer, up to reasoning depth 8.
- **How it makes tasks harder**: composition depth, long distracting context, hidden dependencies between steps, and rejection sampling against a reference model.
- **Correctness / verification**
  - QA: a verifier checks that each document contains its answer points and no extra relevant information.
  - Code: up to 10 generated test programs; an example is kept only if some test passes on the answer code.
  - Math: GPT-4o-mini generates and an ensemble of Gemini-1.5-Flash and Llama-3.1-70B verifies; failed continuations are discarded and retried.
  - Post-hoc recipe (appendix): keep an item if at least k > 1 held-out verifiers match the ground truth and the rest do not agree on a different answer. Discard items where a majority agree on another answer, and send split cases to humans.
- **Difficulty control**: chain depth 2–8; 15 continuation iterations × 3 passes; number of helpers; distractor volume; rejection rate.
- **Reported results**
  - The best of 15 models scored about 40–60%.
  - CHASE-Math: 2.3k GSM8K/SVAMP seeds gave about 1,500 problems; discarding about 75% of those GPT-4o-mini could solve left 500.
  - Manual audit of 100 items per benchmark: 6 errors (QA), 3 (Code), 7 (Math).
  - Correction to the researcher entry: the direct-prompting baseline scored 82.5–88.9% on math and 73–81% on QA, and 34 of 100 direct-generated math problems had errors.
  - Fine-tuning used about 10k problems that Llama-3.1-8B generated and verified itself. Gains were marginal: Llama-3.1-8B 30 → 34.7, Mistral-7B 3.3 → 4.7, Qwen2-7B 12.7 → 15.3.
  - GPT-4o as answer judge agreed with 3 annotators 91% of the time (κ = 0.82).
- **Limitations / failure modes**: LLM-only soft verification; some unnatural text; cost limits scale.
- **How to reuse with easy seed tasks**: one of the most transferable recipes.
  - Chain solved seeds into multi-hop continuations with per-link verification by a model family different from the generator.
  - Build answer-first long-context tasks with planted near-miss distractors.
  - Rejection-sample against the current policy rather than a fixed small model.
  - Apply the k-of-n held-out verifier filter before items enter the RLVR pool.

### TaskCraft — TaskCraft: Automated Generation of Agentic Tasks (Shi et al., 2025)
Link: https://arxiv.org/abs/2506.10055
- **Mechanism**
  - Atomic task: a tool input i_T retrieves content C; an LLM picks an answer a and a relation R, and writes a question.
  - Depth extension: recursively replaces the identifier with a new one that must first be resolved through a reversible superset relation, adding one hop per step.
  - Width extension: merges sub-tasks into one compound question.
- **How it makes tasks harder**: more hops and more merged sub-goals, across search, browsing, PDF and image tools.
- **Correctness / verification**
  - Atomic check: the tool agent's output (within about 3 tool steps) is compared against an LLM without tools; a judge LLM keeps the task only if the agent's output alone contains the gold answer.
  - Extensions are checked linguistically for a strict superset relation (avoiding "pseudo-supersets") and for information leakage, meaning an infer-LLM must not easily derive the answer from the merged task.
- **Difficulty control**: number of depth hops and width merges.
- **Reported results**
  - About 36,000 tasks.
  - Prompt optimization with generated tasks raised the atomic-generation pass rate from 54.9% to 68.1% and cut generation time from 29.1 s to 23.5 s; depth-wise extension at 6 hops went from 41.0% to 51.2%.
  - SFT on TaskCraft trajectories improved average scores on HotpotQA, Musique and Bamboogle by +14.0% (Qwen2.5-3B-Base) and +6.0% (3B-Instruct) over the base workflow, and was positioned as an initialization for RL.
- **Limitations / failure modes**: artificial chains of lookups ("obfuscated trivia"); leakage checks are LLM-based.
- **How to reuse with easy seed tasks**: treat easy single-tool tasks as atoms. Deepen and widen them while the reward stays exact, because the answer is inherited. Reject any extension a no-tool LLM can answer.

### AutoBencher — AutoBencher: Towards Declarative Benchmark Construction (Li et al., 2024)
Link: https://arxiv.org/abs/2407.08351 (ICLR 2025)
- **Mechanism**
  - A proposer LM iteratively proposes dataset descriptions (topics), conditioned on the history of (description, candidate-model accuracy).
  - A generator LM with privileged information writes QA pairs for each description: retrieved Wikipedia for knowledge, Python libraries such as sympy for math, a translation LM for multilingual items.
  - Small pilot sets are scored and descriptions re-ranked by the declared objectives; the best are expanded into datasets.
- **How it makes tasks harder**: it searches for topics where models fail, often tail knowledge, and uses tool-computed answers the test-taker cannot reproduce by recall.
- **Correctness / verification**: information asymmetry between generator and test-taker. Mechanical Turk verification found a 5% error rate overall and 3% for math and economics.
- **Difficulty control**: explicit objective. Difficulty is the lowest achieved error rate (1 − best model accuracy); novelty is 1 − rank correlation with predictions from existing benchmarks; salience is also an objective.
- **Reported results**: 22% more model errors than existing benchmarks and a 27% decrease in ranking correlation. Specific gaps found: Gemini-Pro on the Permian extinction and Fordism, and GPT-4o not declining cryptocurrency-scam requests.
- **Limitations / failure modes**: drifts toward obscure tail knowledge rather than reasoning depth; can overfit to quirks of the candidate models; 5% label error is non-trivial as RL reward.
- **How to reuse with easy seed tasks**: run the topic search against your own policy to find low-pass-rate sub-domains, then synthesize tool-grounded problems there. Add a reasoning-depth term, or restrict to executable answers, so the search does not drift into trivia.

### MR-GSM8K *(added)* — MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation (Zeng et al., 2023)
Link: https://arxiv.org/abs/2312.17080 (code: github.com/dvlab-research/MR-GSM8K)
- **Mechanism**
  - Reframes the model from "solver" to "teacher". Given a question and a solution, it must judge whether the solution is correct, find the first error step, and explain the error. The scoring metric is MR-Score.
  - Questions are original GSM8K, Program-of-Thought (code) variants, and reversed variants (hide an input and give the answer).
  - Solutions were sampled from MetaMath-7B at temperature 1, targeting about 50% correct.
- **How it makes tasks harder**: judging must be process-level. A solution with the right final answer but flawed reasoning is labeled incorrect. Reversed questions add backward reasoning.
- **Correctness / verification**: each item is labeled twice by different annotators; disagreements go to a supervisor; 50% get a second QC round; about 10% are author-inspected.
- **Difficulty control**: question type (original, POT, reversed) and error position.
- **Reported results**
  - Deepseek-v2 and Claude3-Sonnet were close to GPT-4 on GSM8K, but the gap widened to more than 20 points on MR-GSM8K.
  - Training-data recipe: GPT-4 injected an error at a random step of a correct solution and completed it. The resulting diagnostic data was more than 90% accurate by expert spot-check, although GPT-4 identified incorrect solutions only about 40% of the time on the test set.
- **Limitations / failure modes**: human-labeled test set; GSM8K-level math.
- **How to reuse with easy seed tasks**: every solved seed with a gold solution can yield "find the first wrong step" tasks. Inject an error programmatically or with an LLM at a known step; the label (step index) is known by construction, which avoids the verifier bottleneck.

### MMLU-Pro *(added)* — MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (Wang et al., 2024)
Link: https://arxiv.org/abs/2406.01574 (NeurIPS 2024 Datasets & Benchmarks)
- **Mechanism**
  - Removes MMLU questions answered correctly by more than 4 of 8 small models (Llama-2-7B/13B and their chat versions, Mistral-7B, Gemma-7B, Yi-6B, Yi-6B-Chat). This filtered 5,886 questions.
  - Adds reasoning-heavy sources: a STEM website, TheoremQA and SciBench. GPT-4-Turbo converted the STEM-website and TheoremQA problems to multiple choice.
  - GPT-4-Turbo expands each question from 4 to 10 options.
- **How it makes tasks harder**: removes items that small models already solve (adversarial filtering against weak models), adds 3× more distractors, and shifts toward multi-step reasoning.
- **Correctness / verification**: two review rounds. Experts verify answers; then Gemini-1.5-Pro re-evaluates all options to flag false negatives, meaning distractors that are actually correct, and experts review the flags.
  - In the MMLU-derived part, experts found 350 incorrect answers, 1,953 false-negative options and 385 bad questions.
  - Final set: 12,032 questions; 83% have 10 options (mean 9.47).
- **Difficulty control**: small-model agreement threshold; option count.
- **Reported results**: 16–33% accuracy drop versus MMLU; prompt sensitivity fell from 4–5% to 2% across 24 prompt styles; chain-of-thought helps on MMLU-Pro, unlike on MMLU.
- **Limitations / failure modes**: still multiple choice; LLM-generated distractors produce many false negatives, which require expert review.
- **How to reuse with easy seed tasks**: for multiple-choice RL seeds, expanding options is a cheap hardener. Always run a false-negative pass, because a strong model marks every option it believes correct. For RLVR, prefer converting items to free-form answers where possible.

*Family C: adversarial selection and model-in-the-loop authoring.*

### BIG-Bench Extra Hard (BBEH) — BIG-Bench Extra Hard (Kazemi et al., 2025)
Link: https://arxiv.org/abs/2502.19187 (data: github.com/google-deepmind/bbeh)
- **Mechanism**
  - Each of the 23 BBH tasks is replaced by a novel task probing a similar skill at higher difficulty.
  - Construction was semi-adversarial: iterate on each task until both reference models score below 70% accuracy. The reference models were Gemini 1.5 Flash and Gemini-2.0-Flash-Thinking-Exp-01-21.
  - Authors inspected model solutions to remove shortcuts. Example: the reference model solved long Boolean expressions by running Python, so some True/False operands were replaced with statements of known truth value ("The capital of Canada is Ottawa"; "24 − 2 is greater than 48 / 2").
- **How it makes tasks harder**: many-hop reasoning, long context, going against strong priors, finding errors in reasoning traces, distractors, long-range dependencies, and fewer guessable answer options.
- **Correctness / verification**: automatic scoring against programmatically known answers.
- **Difficulty control**: stopping rule (both reference models below 70%). Macro-average context is about 6× BBH and Gemini 2.0 Flash outputs are about 7× longer. 200 questions per task (120 for Disambiguation QA); BBEH Mini has 460.
- **Reported results**
  - At release, harmonic-mean accuracy was 9.8% for the best general-purpose model and 44.8% for the best reasoning model; the reference Thinking model scored 20.2%.
  - Zebra puzzles: top models did better on 8×8 puzzles without distracting clues than on smaller 6×6 or 7×7 puzzles with distractors.
- **Limitations / failure modes**: hardened against specific reference models; will saturate; part of the difficulty is length.
- **How to reuse with easy seed tasks**: institutionalize "solve, read solutions, find the shortcut, redesign" with a target accuracy. The reframings "embed operands in world-knowledge statements" and "find the error in this trace" transfer to many seed families.

### Adversarial Filtering / AFLite — HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019); Adversarial Filters of Dataset Biases (Le Bras et al., 2020)
Links: https://arxiv.org/abs/1905.07830 (ACL 2019) · https://arxiv.org/abs/2002.04108 (ICML 2020) · critique: Phang et al. 2021, https://arxiv.org/abs/2111.08181
- **Mechanism**
  - AF: an LM generates many candidate wrong endings. Discriminators are trained iteratively, and wrong endings they find easy are replaced by ones they misclassify.
  - AFLite: train many linear classifiers on precomputed embeddings over random partitions, remove instances predicted correctly with high confidence (solvable via spurious features), and iterate.
- **How it makes tasks harder**: removes artifact-solvable items and makes distractors that fool models while staying easy for humans.
- **Correctness / verification**: gold labels come from humans or real continuations; only distractors are machine-made; human accuracy is re-checked.
- **Difficulty control**: number of rounds, predictability threshold, adversary strength.
- **Reported results**: HellaSwag: humans above 95%, state-of-the-art models below 48% at release. AFLite: SNLI accuracy fell from 92% to 62% while human performance stayed high, and models trained on filtered data generalize better out of distribution.
- **Limitations / failure modes**: Phang et al. adapted AFLite to evaluation sets with 18 adversary models.
  - Harder sets result, but model rankings become unstable and depend on the adversary.
  - AFLite oversamples low-annotator-agreement items, so comparisons hinge on contentiously labeled examples.
  - Correction to the researcher entry: the finding that the adversary model is disproportionately hurt comes from model-in-the-loop sets (ANLI, AdversarialQA).
- **How to reuse with easy seed tasks**: predictability filtering against an ensemble that includes the current policy is a good pre-filter for shortcut-solvable items. Always pair it with label verification, because the filter enriches contentious or mislabeled items.

### ANLI / Dynabench — Adversarial NLI: A New Benchmark for Natural Language Understanding (Nie et al., 2019); Dynabench: Rethinking Benchmarking in NLP (Kiela et al., 2021)
Links: https://arxiv.org/abs/1910.14599 (ACL 2020) · https://arxiv.org/abs/2104.14337 (NAACL 2021)
- **Mechanism**: annotators write examples against a live target model; other humans verify the ones that fool it; the model is retrained on everything collected and the next round targets the stronger model (ANLI ran 3 rounds). Dynabench turns this into a platform where datasets and models co-evolve.
- **How it makes tasks harder**: each round attacks a stronger model, often with longer or more complex contexts.
- **Correctness / verification**: independent human validators must agree with the writer's label.
- **Difficulty control**: round index; strength of the adversary.
- **Reported results**
  - ANLI training gave state-of-the-art results on several NLI benchmarks while its test set stayed hard, and non-expert annotators found real weaknesses.
  - Kaushik et al. 2021 (https://arxiv.org/abs/2106.00872), a randomized study: models trained on adversarial QA data did better on other adversarial sets but worse on diverse out-of-domain sets.
  - Bartolo et al. 2021 (https://arxiv.org/abs/2104.08678) amplified a small human-adversarial set synthetically: +3.7 F1 on AdversarialQA, better generalization on 9 of 12 MRQA sets, and the human fooling rate fell from 17.6% to 8.8%.
- **Limitations / failure modes**: expensive; overfits to adversary blind spots; non-experts can no longer fool frontier LLMs.
- **How to reuse with easy seed tasks**: replace human writers with LLM attackers that see the policy's live failures, run several rounds against updated checkpoints, reserve experts or strong verifiers for label validation, and always mix adversarial with standard data.

### Humanity's Last Exam + HLE-Verified — Humanity's Last Exam (Phan et al., 2025); HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam (Zhai et al., 2026)
Links: https://arxiv.org/abs/2501.14249 · https://arxiv.org/abs/2602.13964 (data on Hugging Face: skylenage/HLE-Verified)
- **Mechanism**
  - Experts submit closed-ended questions (exact-match or multiple choice) with unambiguous, non-searchable answers.
  - Each submission is first tested against frontier LLMs: GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet and o1, plus o1-mini and o1-preview for text-only questions. Exact-match questions must stump all models; multiple-choice questions must stump all but one (the main text says, equivalently, that models must do worse than random on average).
  - Two expert review rounds follow; reviewers hold graduate degrees.
  - HLE-Verified, Stage I: binary validation of each problem and answer by domain experts plus model cross-checks. Stage II: dual independent expert repairs of flawed-but-fixable items, model-assisted audit, and adjudication.
- **How it makes tasks harder**: binary adversarial filter against the frontier at submission time.
- **Correctness / verification**
  - Correction to the researcher entry: HLE's own pre-release audits (two rounds of 200 questions) estimate 15.4% expert disagreement on the public set and about 18% on a biology/chemistry/health subset. The "about 30% of chemistry/biology answers likely wrong" figure is an external website that HLE cites, not HLE's estimate.
  - HLE-Verified: 668 items verified unchanged, 1,143 revised and certified, 689 released as an uncertain set.
- **Difficulty control**: the frontier-model filter.
- **Reported results**
  - More than 70,000 attempts were logged; about 13,000 LLM-stumping questions went to expert review; 2,500 were released publicly.
  - Release accuracy: GPT-4o 2.7%, o1 8.0%, DeepSeek-R1 8.5% (text-only), o3-mini (high) 13.4%; RMS calibration errors of 73–89%.
  - HLE-Verified: models score 7–10 points higher overall and 30–40 points higher on items whose original statement or answer was erroneous; model confidence is strongly associated with item errors.
- **Limitations / failure modes**: "models fail" filtering preferentially selects mislabeled, ambiguous or obscure items; static; costly.
- **How to reuse with easy seed tasks**: if you adversarially filter synthetic RL problems, add an independent verification stage. Treat items where several strong models confidently agree on a non-reference answer as probable label errors, not hard problems.

### UQ — UQ: Assessing Language Models on Unsolved Questions (Nie et al., 2025)
Link: https://arxiv.org/abs/2508.17580 (uq.stanford.edu)
- **Mechanism**
  - Unanswered questions were crawled from 80 Stack Exchange communities, about 3 million candidates.
  - Rule filters (age ≥ 2 years, site-dependent view and vote thresholds, top 10% by votes) cut this to 33,916. LLM filters for well-posedness and difficulty cut it to 7,685. Human review left 500.
  - Candidate answers are screened by UQ-Validators, which exploit the generator-validator gap (models validate better than they generate). The default pipeline runs cycle consistency, then fact/logic check, then correctness check, with repeated unanimous votes and a precision-first design.
  - Answers that pass go to a platform for expert verification.
- **How it makes tasks harder**: the questions are hard because the askers could not solve them, which yields difficulty and realism at once.
- **Correctness / verification**: no ground truth at creation time; validators pre-screen and humans verify later, asynchronously.
- **Difficulty control**: the filtering pipeline.
- **Reported results**: the top model passes UQ validation on only 15% of questions, and preliminary human verification has already confirmed correct answers among those that passed.
- **Limitations / failure modes**: not usable as an RL reward until answers are verified; validator false positives; slow.
- **How to reuse with easy seed tasks**: mine unsolved items in your domain as a frontier pool; use precision-oriented validator ensembles for best-of-n distillation; promote items to RLVR once tools or humans confirm them. Related never-saturating targets:
  - FrontierMath (https://arxiv.org/abs/2411.04872): unpublished expert problems with automated verification; state-of-the-art models solved under 2% at release.
  - FrontierMath Erdős (https://arxiv.org/abs/2609.25050): 68 Erdős problems open as of August 2026, to be resolved in Lean. At $300 per problem, one of five AI systems (GPT-6 Astra) scored 3% and the others 0%.

### MathDuels (+SKATE) — MathDuels: A Self-Play Benchmark That Grows (Xu et al., 2026)
Links: https://arxiv.org/abs/2604.21916 · SKATE: *SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges* (Gould et al., 2025), https://arxiv.org/abs/2508.06111
- **Mechanism**
  - Each of 19 frontier models authors problems in three stages:
    - (1) Meta-prompting: write the prompt for authoring a hard problem in a domain; sampled meta-prompts diversify strategies.
    - (2) Generation: a problem with a scalar gold answer.
    - (3) Difficulty amplification: one or more hardening rounds producing variants that need deeper reasoning.
  - Every non-author model solves every problem; answers are checked by symbolic equivalence.
  - An independent verifier (GPT-5.4-high) excludes ill-posed items and selects the reference answer, using solver traces when answers disagree.
  - A Rasch model jointly fits solver ability and problem difficulty; author quality comes from the difficulty of each model's authored problems.
  - SKATE: models set code-output-prediction challenges for each other, scored by execution and ranked with TrueSkill.
- **How it makes tasks harder**: author-side adversarial amplification, and new authors keep entering.
- **Correctness / verification**: the verifier can override the author. Two verifier backbones agreed on the keep/exclude decision for 97.5% of problems, and on answers for 99.4% of problems both kept.
- **Difficulty control**: number of hardening rounds; post-hoc Rasch difficulty.
- **Reported results**
  - 570 problems, 11 excluded, 559 valid, 10,062 solve observations.
  - Correction to the researcher entry: the error rates come from a pipeline ablation with the four strongest authors (432 model-pair observations). Error rose from 5.1% (direct generation) to 9.5% (meta-prompting) to 25.0% (meta-prompting + amplification). With code execution at solve time: 3.0%, 6.9%, 21.1%.
  - Authoring and solving ability are partially decoupled: Gemini-3.1-Pro-high ranked first overall because its problems had the lowest average solve rate (62.9%), although GPT-5.4-high led as a solver.
  - SKATE: weaker models can reliably differentiate stronger ones, and models show self-preferencing, posing questions that suit their own strengths.
- **Limitations / failure modes**: the verifier's ability is a ceiling; scalar answers only; authors may exploit verifier blind spots.
- **How to reuse with easy seed tasks**: several strong authors with explicit "harden this" rounds; an independent verifier ensemble that picks the answer from candidates; Rasch/IRT fit on policy rollouts to pick items near P(correct) = 0.5. Watch for self-preferencing when the author is also the policy.

*Family D: teacher agents and weakness-targeted synthesis.*

### DataEnvGym — DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback (Khan et al., 2024)
Link: https://arxiv.org/abs/2410.06215 (ICLR 2025 Spotlight)
- **Mechanism**
  - Training-data creation is framed as sequential decision-making.
  - The teacher has a data-generation policy (plans the data) and a data-generation engine (produces it).
  - Each iteration trains the student, evaluates it, and returns errors or skill scores as the next state.
  - Three environments add structure: Open-Ended (raw per-example results), Skill-List (per-skill accuracy over induced skills) and Skill-Tree (actions are explore, which grows subskills, and exploit, which reallocates the data budget).
- **How it makes tasks harder**: indirectly, by targeting weak skills.
- **Correctness / verification**: teacher LLM (GPT-4o) plus domain checks; no formal guarantee.
- **Difficulty control**: implicit. Per-skill gains were largest at mid-range question difficulty and on more frequent skills.
- **Reported results**
  - GPT-4o teacher, with-state policies: Gemma2-2B on MATH 15.78 → 23.44 (+7.66, Open-Ended); Llama3-8B on LiveCodeBench +2.41; PaliGemma-3B on GQA +5.58 (Skill-Tree).
  - Without student state, Open-Ended gave 0.00 on LiveCodeBench. With-state policies beat no-state ones by 3.5, 2.05 and 1.08 points (Open-Ended, Skill-List, Skill-Tree).
  - A GPT-4o-mini teacher gave much smaller gains (+0.77 to +1.81 on average).
- **Limitations / failure modes**: modest gains; depends on teacher strength; SFT loop; no correctness guarantee.
- **How to reuse with easy seed tasks**: feed per-skill failure statistics of the current checkpoint into the generator prompt every round, and keep a skill tree so coverage persists while the budget shifts toward skills below the target pass rate.

### EvalTree (+AutoDetect) — EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees (Zeng et al., 2025)
Links: https://arxiv.org/abs/2503.08893 (COLM 2025) · AutoDetect: *AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models* (Cheng et al., 2024, EMNLP 2024 Findings), https://arxiv.org/abs/2406.16714
- **Mechanism**
  - EvalTree annotates each benchmark instance with a capability description, embeds the descriptions, clusters them recursively with K-means into a tree, and has an LM describe each node.
  - It extracts weakness nodes by traversing from the root: a binomial test checks whether node accuracy is significantly below a user threshold τ, and a node is extracted only if its sufficiently large children also pass.
  - The weakness descriptions condition data synthesis.
  - AutoDetect uses Examiner, Questioner and Assessor agents to probe and synthesize targeted questions iteratively.
- **How it makes tasks harder**: it concentrates generation where the model is statistically weak.
- **Correctness / verification**: profiling is statistical; the generated data relies on the generator LLM.
- **Difficulty control**: the threshold τ and minimum node sizes.
- **Reported results**
  - On MATH, GPT-4o mini scores 75.1% on "calculating combinations and arrangements of elements" but 49.1% on "analyzing geometric relationships using trigonometric principles".
  - Weakness-guided synthesis gave an accuracy gain 2.5× larger than guidance by a generic capability.
  - AutoDetect: weakness-identification success above 30% on ChatGPT and Claude; gains of over 10% for Llama-series models and Mistral-7B after targeted training.
- **Limitations / failure modes**: the generator must produce correct data exactly where models are weak; profiles shift after training.
- **How to reuse with easy seed tasks**: after each RL stage, build the tree over failures and zero-advantage prompts, then synthesize verified harder or more varied items for the lowest nodes.

### SwS — SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning (Liang et al., 2025)
Link: https://arxiv.org/abs/2506.08989
- **Mechanism**
  - A preliminary RL run logs per-problem accuracy each epoch. A weakness is a problem that never reaches 50% accuracy and whose accuracy slope is negative.
  - Core concepts are extracted from weaknesses, grouped by category and recombined using co-occurrence and embedding similarity.
  - Llama-3.3-70B-Instruct synthesizes problems, excluding multiple-choice, multi-part and proof problems.
  - Quality is judged by Llama-3.3-70B-Instruct and Qwen-2.5-72B-Instruct.
  - Correction to the researcher entry: reference answers come from Skywork-OR1-Math-7B for models up to 7B and from QwQ-32B for 32B, by self-consistency with Math-Verify equivalence.
  - Only problems with initial-policy accuracy in [25%, 75%] enter augmented RL, and the budget per category is re-allocated.
- **How it makes tasks harder**: recombines the concepts behind persistent failures into new problems.
- **Correctness / verification**: the most frequent answer must cover at least 50% of teacher samples, plus quality filters. The quality filter is the strictest stage, removing 78.35%.
  - In a weak-teacher study, also requiring the student to match the teacher's answer in at least 25% of its responses raised labeling accuracy on MATH-500 (used as a stand-in synthetic set) from 80.6% to 97.5%.
- **Difficulty control**: the policy-relative [25, 75]% band avoids all-correct or all-wrong GRPO groups.
- **Reported results**: average gains of 10.0% (7B) and 7.7% (32B) across eight benchmarks. About 35% of synthesized problems fall in the band across the four base models.
- **Limitations / failure modes**: self-consistency labels rather than formal checks; heavy over-generation; recombines known concepts rather than discovering new skills. The self-evolving variant helped less on AIME-level problems.
- **How to reuse with easy seed tasks**: directly applicable when GRPO saturates. Log per-prompt accuracy, mine persistent and worsening failures, recombine concepts, label with teacher majority plus student agreement, keep the 25–75% band, and budget about 3× over-generation.

*Family E: open-ended, quality-diversity and regret-driven task generation.*

### eva — Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play (Ye et al., 2024)
Link: https://arxiv.org/abs/2411.00062
- **Mechanism**
  - Post-training is framed as a creator-solver game.
  - The creator scores each prompt's informativeness with reward-advantage regret proxies over sampled responses, scored by a reward model (ArmoRM-8B by default). Examples: A*_avg = |mean reward − max reward| and A*_dts = |second-best − best|.
  - It samples prompts in proportion to informativeness, evolves them with in-depth and in-breadth rewrites, and fills a prioritized generative buffer mixed with the original prompts.
  - The solver trains with DPO, SPPO, SimPO, ORPO or RLOO.
  - Correction to the researcher entry: the claim that training eventually bootstraps from evolved prompts alone was not found and has been removed.
- **How it makes tasks harder**: evolves prompts where the policy's responses vary most in reward, i.e. high regret.
- **Correctness / verification**: reward model only; there are no ground-truth answers.
- **Difficulty control**: informativeness-weighted sampling.
- **Reported results**
  - gemma-2-9b-it on Arena-Hard: 51.6 → 60.1 (DPO) and 52.6 → 62.4 (RLOO).
  - DPO+eva (60.1) edged out adding new human prompts (59.8).
  - Informativeness ablation (DPO): uniform evolution 57.5; var(r) 54.8; avg(r) 58.5; 1/avg(r) 56.7; 1/A*_min 52.3; A*_avg 60.0; A*_dts 60.0.
- **Limitations / failure modes**: reward-model hacking; prompt quality drift.
- **How to reuse with easy seed tasks**: for RLVR, use group outcome statistics as the informativeness signal (for example `0 < pass < 1`, or advantage magnitude) and mutate near-frontier prompts with verified complexification operators. Test the signal choice: in eva's own ablation, variance-based selection underperformed uniform selection.

### ACCEL — Evolving Curricula with Regret-Based Environment Design (Parker-Holder et al., 2022)
Link: https://arxiv.org/abs/2203.01302 (accelagent.github.io)
- **Mechanism**
  - Maintains a level-replay buffer scored by estimated regret, using positive value loss as in PLR.
  - Samples high-regret levels, trains on them, and applies small edits (add or remove obstacles, change terrain).
  - Edited children enter the buffer only if their regret score is high; they are not trained on immediately.
  - Starting from empty or simple levels, complexity compounds.
- **How it makes tasks harder**: accumulated small edits at the agent's frontier.
- **Correctness / verification**: the simulator defines levels; the regret gate filters uninformative ones.
- **Difficulty control**: regret-based admission; number of edits.
- **Reported results**: correction to the researcher entry: the domains are MiniGrid mazes and 2D BipedalWalker (no car racing). ACCEL reached POET-comparable level complexity while training on less than 0.05% of POET's environment samples, on a single GPU, and showed strong zero-shot transfer to held-out levels.
- **Limitations / failure modes**: noisy regret proxies; assumes small edits change difficulty smoothly; pre-LLM.
- **How to reuse with easy seed tasks**: keep a buffer of problems scored by learning potential, apply one small verified edit at a time (one constraint, one hop, one parameter bump), and admit children only if they remain in the frontier band.

### OMNI / OMNI-EPIC — OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code (Faldor et al., 2024)
Links: https://arxiv.org/abs/2405.15568 · OMNI: *OMNI: Open-endedness via Models of human Notions of Interestingness* (Zhang et al., 2023), https://arxiv.org/abs/2306.01711
- **Mechanism**
  - OMNI uses a foundation model as a "model of interestingness" (MoI) to prefer learnable and interesting tasks over trivial variants. It outperformed uniform sampling and learning progress alone.
  - OMNI-EPIC keeps an archive of learned and failed tasks and runs this loop:
    - A task generator retrieves similar archive tasks as stepping stones and proposes a new natural-language task.
    - An environment generator writes executable simulation code: world, reward, and a separate `get_success()`.
    - A post-generation MoI rejects tasks not "interestingly new".
    - RL trains the agent; a success detector (code or VLM) judges the outcome.
    - Failed tasks are retried up to a limit and then archived as failed. The generator then tends to propose easier variants, for example fewer obstacles.
- **How it makes tasks harder**: it builds on learned stepping stones and elaborates tasks after success.
- **Correctness / verification**: code-defined environments and a separate success function. In a 50-participant study, humans agreed with the success detector 72.7% of the time.
- **Difficulty control**: learnability inferred from archive successes and failures; simplify after failure, elaborate after success.
- **Reported results**: a run produced 16 learned tasks, 6 failed and 1 rejected as uninteresting. Cell coverage and ANNECS-OMNI were significantly higher than controls (p < 0.05).
- **Limitations / failure modes**: expensive RL inner loop; LLM judgments of interestingness; buggy generated code; not shown on LLM reasoning.
- **How to reuse with easy seed tasks**: keep an archive of tasks-as-code with checkers; propose new tasks conditioned on nearby archive entries and the policy's solve/fail status; route failures to a "simplify" operator and successes to a "harden" operator; reject near-duplicates with an MoI or embedding filter.

### Automated Capability Discovery (ACD) — Automated Capability Discovery via Foundation Model Self-Exploration (Lu et al., 2025)
Link: https://arxiv.org/abs/2502.07577
- **Mechanism**
  - A scientist model writes new task families as Python classes with instructions, task instances and a score function, conditioned on sampled archive tasks.
  - A proposal is kept only if it is "interestingly new" relative to its nearest archive neighbors (text-embedding-3-small).
  - The subject attempts tasks; scoring is programmatic or by a GPT-4o judge.
  - Results are clustered into capability and failure areas.
- **How it makes tasks harder**: the scientist adapts difficulty to subject success and keeps exploring novel regions.
- **Correctness / verification**: 92.2% of tasks were rated clear and valid by humans; automated-versus-human scoring F1 was 0.86 with a slight positive bias. Judge F1 drops on tasks humans rated "Very Difficult".
- **Difficulty control**: scientist-driven adaptation plus the novelty filter.
- **Reported results**: with GPT-4o as both scientist and subject, 5,000 generations produced 1,330 "interestingly new" tasks in 25 clusters; about 20% of new proposals were still novel after 5,000 generations.
- **Limitations / failure modes**: judge scoring is least reliable exactly on the hardest tasks; some discovered "failures" are ambiguous tasks.
- **How to reuse with easy seed tasks**: run ACD against your policy to discover failure clusters. Promote families with programmatic `score()` to RL environments; keep judge-scored families for SFT or evaluation.

### ACES — ACES: Generating Diverse Programming Puzzles with Autotelic Generative Models (Pourcel et al., 2023)
Link: https://arxiv.org/abs/2310.10692 (the arXiv title contains the typo "with with")
- **Mechanism**
  - Puzzles use the P3 format: a test program `f`, where a valid solution `g` satisfies `f(g()) == True`.
  - An LLM labels each puzzle with a subset of 20 programming skills. Each skill combination is a niche, giving 21,700 possible niches.
  - Each iteration samples a target niche and picks 3 in-context examples from the nearest filled niches, sampled by softmax (temperature 0.2) over normalized fitness.
  - The generator is asked for a puzzle with the target skills; each example's difficulty score is included in the prompt, with an instruction to reach 90–100.
  - Llama-3-70B attempts each new puzzle 50 times.
- **How it makes tasks harder**: difficulty-seeking generation within each niche.
- **Correctness / verification**: execution. Puzzles never solved in 50 attempts are discarded as possibly unsolvable.
- **Difficulty control**: fitness is the negative solver success rate rescaled to 0–100.
- **Reported results**: more diverse and more challenging than baseline generators; on average across 11 code LLMs, three times more challenging than existing Python benchmarks. The best solver (Mixtral-8x22B-Instruct) reached 47.3% pass@1, versus 86.6% for GPT-4-turbo on HumanEval+.
- **Limitations / failure modes**: dropping never-solved puzzles caps difficulty at the solver's reach; LLM-assigned skill labels; evaluation-oriented.
- **How to reuse with easy seed tasks**: keep a MAP-Elites archive of RL tasks keyed by skill-descriptor tuples, with fitness = 1 − policy pass rate. Exclude pass-0 tasks unless independently certified. Sampling target niches prevents collapse onto a few hard templates.

### Rainbow Teaming (+QDAIF) — Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts (Samvelyan et al., 2024)
Links: https://arxiv.org/abs/2402.16822 · QDAIF: *Quality-Diversity through AI Feedback* (Bradley et al., 2023), https://arxiv.org/abs/2310.13032
- **Mechanism**
  - Quality-diversity search over a feature archive. For safety the archive is 10 risk categories × 10 attack styles, initialized from scratch without seed data.
  - A parent is sampled uniformly; a mutator LLM applies descriptor-targeted mutations toward a target cell; candidates too similar to the parent (high BLEU) are dropped.
  - A judge compares the target's responses to the candidate and to the cell incumbent, with multiple evaluations and position swapping; the winner keeps the cell.
  - QA variant: a 3-D archive (topic × interrogative word × question length). Fitness is a question the target (Llama 2-chat 7B) answers incorrectly while a stronger oracle (Llama 2-chat 70B) answers correctly.
  - QDAIF generalizes the idea to quality and diversity axes defined by AI feedback.
- **How it makes tasks harder**: each cell keeps its most effective prompt.
- **Correctness / verification**: GPT-4 and Llama Guard as independent attack-success classifiers (neither directly optimized); oracle answers for QA, which are not guaranteed correct.
- **Difficulty control**: judge-based pairwise fitness per cell.
- **Reported results**
  - Correction to the researcher entry: attack success rate (ASR) is the fraction of the 100 archive prompts that elicit unsafe responses. It was about 92% against Llama 2-chat 7B and Llama 3-Instruct 8B and 98% against Mistral 7B and Vicuna 7B after 2,000 iterations, and 90% or higher on Llama 2-chat 7B, 13B and 70B.
  - Prompts transfer across models; fine-tuning on Rainbow data improved safety without hurting general capability or helpfulness.
- **Limitations / failure modes**: judge fitness can be gamed; the oracle can be wrong; descriptor grids are hand-designed.
- **How to reuse with easy seed tasks**: use a descriptor grid (skill × format × length, or topic × reasoning type) as the backbone of a hard-task generator. Mutate elites toward empty or weak cells and replace an incumbent only on more verified policy failures.

### AC/DC — Discovering Novel LLM Experts via Task-Capability Coevolution (Dai et al., 2026)
Link: https://arxiv.org/abs/2604.14969 (ICLR 2026)
- **Mechanism**
  - Co-evolves a population of LLMs with an archive of natural-language tasks.
  - Models are made by merging, with mutation by noise on the leading singular values of weight matrices.
  - A scientist LLM mutates parent tasks into harder, easier ("conceptually related but significantly easier") or novel variants, using their difficulty profile across the population; a novelty judge filters.
  - Reflection and validation repair tasks.
  - Minimal-criterion filters remove gibberish models and impossible tasks that no model solves.
  - Dominated Novelty Search selects models by skill vectors.
- **How it makes tasks harder**: population-relative difficulty mutation.
- **Correctness / verification**: self-solve reflection, code checks, the impossible-task filter. Humans rated 97.8% ± 2.2% of synthetic tasks correct and 68.9% ± 6.9% out-of-distribution relative to existing benchmarks.
- **Difficulty control**: per-task difficulty profile across models; explicit harder/easier mutations.
- **Reported results**: evolved populations achieve broader coverage of downstream benchmarks than larger same-family models and curated baselines, without benchmark optimization; collectives reach or exceed GPT-4o-level knowledge coverage with far fewer parameters; coverage improves over generations and helps multi-agent best-of-N.
- **Limitations / failure modes**: model merging limits how far capabilities move; coverage is a best-of-population metric; tasks are LLM-verified.
- **How to reuse with easy seed tasks**: use difficulty profiles across checkpoints or models to decide whether to harden or soften each task. Keep tasks that discriminate between models; drop never-solved tasks unless a verifier certifies them.

### Self-Challenging Agents — Self-Challenging Language Model Agents (Zhou et al., 2025)
Link: https://arxiv.org/abs/2506.01716
- **Mechanism**
  - The same LLM first acts as challenger: it interacts with the tools to learn what is feasible, then writes a Code-as-Task (CaT).
  - A CaT has four parts: an instruction, an executable verification function over the final state and answer, an example solution, and failure cases. Everything except the instruction is code.
  - A task is kept only if the example solution passes and every failure case fails.
  - The model then trains as executor with RL (rejection fine-tuning or PPO-style online RL), using the verifier as a 0/1 reward.
- **How it makes tasks harder**: failure cases force non-trivial verifiers; exploration grounds tasks in feasible but non-trivial goals.
- **Correctness / verification**: executable checks. In a manual audit of 50 Llama-3.1-8B rollouts per variant in tau-bench Retail, CaT sharply reduced both false negatives (impossible tasks, e.g. returning a non-existent order) and false positives (lenient verifiers); some ambiguity remained.
- **Difficulty control**: implicit.
- **Reported results**
  - More than a two-fold improvement for Llama-3.1-8B-Instruct on M3ToolEval and tau-bench from self-generated data (800 tasks, 12k rollouts).
  - In the distillation setting, more than 20% absolute gain in average success rate (PAE: 18%).
  - Diversity mattered: 200 training tasks slightly degraded test performance, 400 gave marginal gains, 800 gave steady improvement.
  - A gap remains versus training on oracle tasks.
- **Limitations / failure modes**: residual ambiguous tasks; no explicit difficulty control; quality depends on challenger exploration.
- **How to reuse with easy seed tasks**: require every synthetic or complexified RL task to carry a known-good solution plus known-bad candidates that the checker must reject. This is the cheapest guard against lenient or unsolvable verifiers. Scale the number of distinct tasks, not only the number of rollouts.

### CompassPlay — CompassPlay: Rewarding the Proposer for Where It Moves the Solver (Pu et al., 2026)
Link: https://arxiv.org/abs/2609.32228
- **Mechanism**
  - In proposer-solver self-play, a candidate task is eligible only if solver success s satisfies 0 < s < 1.
  - Eligible tasks are rewarded by the cosine between the solver's cross-entropy-loss gradient on the verified candidate solution and the gradient of a reference loss for the target capabilities, both at current solver parameters.
  - This is a first-order approximation of the learning progress the task would induce, computed without extra solver training.
  - Coding: built on AZR, with a small fixed external reference set with gold solutions, not shown to the proposer.
  - Lean 4: built on SGS, with the solver's verified proofs of solved seed theorems as local references.
- **How it makes tasks harder**: it steers toward tasks that move the solver toward the targets, not merely toward hard tasks.
- **Correctness / verification**: execution for code; the Lean kernel for proofs.
- **Difficulty control**: difficulty is only the eligibility gate; value is gradient alignment.
- **Reported results**
  - Qwen2.5-Coder-7B: +1.5 points in-domain coding and +2.7 out-of-domain math over AZR's difficulty reward.
  - Lean 4: matches the difficulty baseline's 150-iteration cumulative coverage with 40% fewer GPU-hours.
  - A 200-step run cost 95 GPU-hours versus 109 for a single 20-step SOAR reimplementation, which needs 16 inner RL runs per outer step.
  - Failure mode found: when the proposer sees the seed, optimizing alignment encourages renamed copies of it; weighting by a novelty factor suppresses this.
- **Limitations / failure modes**: needs per-example gradients and a representative reference; risk of narrowing diversity toward the reference.
- **How to reuse with easy seed tasks**: when many verified hard variants pass the difficulty gate, rank them by gradient alignment with a small held-out set of target tasks. Add a novelty term relative to the seed.

## Complexification operators from this area

1. **Structural scaling of an executable reasoning graph**
   - What it does: lift the item into a DAG or computational graph; raise nodes, depth, width, operation count or cross-links; re-render.
   - Example (illustrative): "Tom has 9 apples and buys 10 more" (1 operation) becomes the same story from a 12-operation DAG where intermediate quantities feed several later nodes. DARG depth +4 took Claude-3-Opus from about 95% to 41.0%.
   - Keep verifiable: label by executing the graph; require a code-writing solver to reproduce it from the rendered text; regenerate on mismatch.
   - Sources: DyVal, DARG, GSM-Infinite, NPPC (2504.11239, 25 NP-complete problem generators).
2. **Lift the seed into a parameterized generator with an adaptive difficulty knob**
   - What it does: extract the scale and constraint parameters of a fixed seed to get an unbounded instance generator, then control difficulty online from policy accuracy.
   - Example: a CodeContests problem fixed at n = 10 becomes the same problem at whatever scale holds accuracy near τ (SCALER controller); RLVE sorting length ∝ 3×1.1^d, promoted when top-level accuracy reaches 90%.
   - Keep verifiable: several reference solvers must agree across scales; prefer asymmetric checkers (differentiate a proposed antiderivative; check a path edge by edge); plant a solution at generation time (planted Hamiltonian path, transformed canonical Sudoku).
   - Sources: RLVE, SCALER, NPPC, S3Eval (2310.15147), Reasoning Core (2603.02208), TRACE (2607.04784; Pearson r ≈ −0.96 between its difficulty metric and accuracy).
3. **Compositional chaining and merging**
   - What it does: make one task's answer an unstated input to the next (sequential), or merge two specifications (parallel).
   - Example: a single GSM8K problem becomes a 5-link chain where each context silently uses the previous answer (CHASE-Math goes to depth 8); EvoEval Combine merges two HumanEval problems.
   - Keep verifiable: verify every link independently (a verifier ensemble from a different family) before appending; for merged code, use differential tests across independent reference implementations.
   - Sources: CHASE, EvoEval, TaskCraft (width), BBEH.
4. **Answer-first / privileged-information construction**
   - What it does: fix the answer first from a source the solver lacks (tool output, database row, retrieved document, executed code, hidden truth in an SAT model), then build question and context around it.
   - Example: CHASE-QA writes QA pairs and then long documents that scatter the answer points among near-miss distractors; TaskCraft starts from a tool-returned answer and adds identifier-obfuscation hops; EDGEGEN (2609.24115) generates database-grounded tasks designed to violate an agent's compliance rules.
   - Keep verifiable: the answer is known by construction. Run a leakage test (a model without tools or context must fail) and a completeness test (the context contains every answer point and nothing else relevant).
   - Sources: CHASE, AutoBencher, TaskCraft, Self-Challenging, KUMO, EDGEGEN.
5. **Distractor / noise injection and context scaling**
   - What it does: add related but irrelevant facts, numbers, documents or functions, or scale context without changing the answer path.
   - Example: a 3-sentence problem is padded with semantically close noise nodes to 8K+ tokens (GSM-Infinite); a target function is hidden among 100 irrelevant helpers in 10 files (CHASE-Code); a single seemingly relevant clause cut accuracy by up to 65% (GSM-Symbolic's GSM-NoOp, 2410.05229).
   - Keep verifiable: noise must be provably disconnected (edges only outward from the core graph) or verified absent from the answer path; re-solve with a strong solver.
   - Sources: GSM-Infinite, GSM-Plus (2402.19255), GSM-Symbolic, CHASE, Benchmark Self-Evolving, MPA, BBEH (Zebra puzzles with distracting clues).
6. **Option / distractor expansion for multiple choice**
   - What it does: add more plausible wrong options, or a new plausible option E.
   - Example: MMLU 4 options become 10 (MMLU-Pro); the MPA principle "add a new plausible choice".
   - Keep verifiable: run a false-negative sweep, where a strong model re-evaluates every option and humans or a stronger check adjudicate. MMLU-Pro removed 1,953 false-negative options in its MMLU part.
   - Sources: MMLU-Pro, MPA.
7. **Method-breaking minimal perturbation (hard perturbation / Subtle / polarity reversal)**
   - What it does: make the smallest textual edit that changes the answer and invalidates the memorized method.
   - Example: a one-condition change to a MATH level-5 problem so the standard trick no longer applies (MATH-P-Hard: o1-mini −16.49%); EvoEval Subtle (−24.0% at roughly the same difficulty); Benchmark Self-Evolving polarity reversal.
   - Keep verifiable: require the new answer to differ from the original; confirm with a CAS, program or independent solvers; flag outputs equal to the original answer as memorization.
   - Sources: MATH-Perturb, EvoEval, Benchmark Self-Evolving.
8. **Functional variation / variabilization with worst-case scoring**
   - What it does: turn the seed into a script with free variables and constants; sample variants; score the seed by its worst case over variants.
   - Example: Putnam 2011 A1 with 2011 → 4680, which changes the answer from 10053 to 23398 (o1-preview −19.6 points on variations); DynaMath's 10 variants per seed, where worst-case accuracy is at most about 50% of average.
   - Keep verifiable: the script recomputes the answer; check equivalence with SymPy; exclude seeds whose solution depends on the specific constant.
   - Sources: Putnam-AXIOM, DynaMath, GSM-Symbolic.
9. **Counterfactual rule or world change**
   - What it does: alter the semantics of the task world (number base, operator meaning, a contradicting new premise) so priors mislead.
   - Example: base-9 arithmetic (Reasoning or Reciting, 2307.02477); MatheMagic (2510.05962) alters the interpretation of numbers and operators, with answers still automatically verifiable; RE-IMAGINE Level 3 ("Imagine") adds a predicate that revises a stated fact; BBEH tasks that go against strong priors.
   - Keep verifiable: represent the task as a symbolic program, apply the mutation in code, and execute it for the new answer.
   - Sources: Reasoning or Reciting, MatheMagic, RE-IMAGINE (2506.15455), BBEH.
10. **Inversion (make a given the unknown)**
    - What it does: swap given and asked quantities to force backward reasoning.
    - Example: "3 boxes of 4 pens: how many pens?" becomes "after buying some boxes of 4 pens and receiving 5 more she has 17; how many boxes?" (illustrative); GSM-Infinite reverse mode; MR-GSM8K reversed questions; GSM-Plus reversing operation.
    - Keep verifiable: generate from a ground-truth forward assignment, then solve symbolically to confirm a unique, well-typed solution.
    - Sources: GSM-Plus, GSM-Infinite, MR-GSM8K, TaskCraft (reversible relations).
11. **Constraint stacking / requirement rarefaction**
    - What it does: add constraints, edge cases or reasoning steps, or replace common requirements with rarer ones.
    - Example (illustrative): "sort the list" becomes "sort by digit sum, break ties by original index descending, exclude primes, handle negatives by absolute digit sum".
    - Keep verifiable: regenerate reference solutions and tests; differential testing; Code-as-Task failure cases that each violate one constraint and must be rejected.
    - Sources: EvoEval Difficult, WizardLM Evol-Instruct (2304.12244), Auto Evol-Instruct (2406.00770), Benchmark Self-Evolving (question complicating), Self-Challenging.
12. **Task reframing (verify, locate error, score, predict, probe a sub-ability)**
    - What it does: change the format while keeping ground truth derivable.
    - Example: "solve this GSM8K problem" becomes "here is a solution; is it correct, where is the first wrong step, why?" (MR-GSM8K); BBEH "find the error in the reasoning trace"; Benchmark Self-Evolving sub-ability questions; SKATE code-output prediction on model-authored programs.
    - Keep verifiable: inject the error programmatically or with an LLM at a known step (GPT-4 injection data was more than 90% accurate even though GPT-4 detects errors at about 40%); execute code for output prediction.
    - Sources: MR-GSM8K, BBEH, Benchmark Self-Evolving, MPA, SKATE.
13. **Partially observable, interactive reformulation**
    - What it does: turn a one-shot classification into multi-turn evidence gathering against a hidden state, scoring success and efficiency.
    - Example: "which disease matches these findings?" becomes a KUMO game with 12 candidate diseases and 16 tests, where outcomes come from an SAT engine and the player must identify the truth with few tests.
    - Keep verifiable: a symbolic engine guarantees outcomes are consistent with exactly one truth; compare action counts to an optimal searcher.
    - Sources: KUMO.
14. **Adversarial filtering and model-in-the-loop selection**
    - What it does: keep items that a panel (or the current policy) fails; iterate against retrained models; or redesign until reference accuracy falls below a threshold.
    - Example: HLE (more than 70,000 attempts led to about 13,000 candidates for review and 2,500 public items); BBEH (both references below 70%); CHASE (discard about 75% of GPT-4o-mini-solvable items); MMLU-Pro (drop items more than 4 of 8 small models solve).
    - Keep verifiable: the filter enriches label errors (HLE-Verified; Phang et al.). Use ensembles; require independent verification; treat confident cross-model consensus against the reference as a probable label error.
    - Sources: AF, AFLite, ANLI, Dynabench, HLE, HLE-Verified, BBEH, CHASE, MMLU-Pro, Phang et al.
15. **Weakness-targeted resynthesis**
    - What it does: profile where the checkpoint fails (capability trees, persistent RL failures), extract concepts, and synthesize there.
    - Example: concepts from problems whose RL accuracy never exceeds 50% and is trending down, recombined and kept if policy accuracy is in [25%, 75%] (SwS); EvalTree nodes where accuracy is significantly below τ.
    - Keep verifiable: teacher majority ≥ 50% plus student agreement ≥ 25%; programmatic checkers where available; strict quality filters (SwS rejects 78.35% at that stage).
    - Sources: SwS, EvalTree, AutoDetect, DataEnvGym, EnvGen (2403.12014), Agent-World (2604.18292).
16. **Regret- or learning-progress-guided editing of frontier tasks**
    - What it does: select tasks with the highest learning potential and apply small edits so children stay at the frontier.
    - Example: uniform evolution (eva 57.5) versus advantage-weighted evolution (60.0–60.1); ACCEL edits admitted only if regret stays high; CompassPlay gradient-alignment reward; MAGELLAN (2502.07709) learns to predict learning progress over large goal spaces.
    - Keep verifiable: re-verify every edited child; small edits reduce the risk of breaking solvability; hide checkers from proposers.
    - Sources: ACCEL, eva, CompassPlay, OMNI, MAGELLAN.
17. **Quality-diversity archive over task descriptors with novelty filtering**
    - What it does: a MAP-Elites grid over descriptors (skill combination, format, length, risk, topic). Mutate elites toward empty or weak cells, keep the hardest verified task per cell, and reject near-duplicates.
    - Example: one "write a hard problem" prompt that yields near-duplicates becomes ACES's archive over 21,700 skill-combination niches, or Rainbow's 10×10 grid with a BLEU filter.
    - Keep verifiable: execution-based fitness; judges with position swapping and majority vote; discard never-solved items unless certified.
    - Sources: ACES, Rainbow Teaming, QDAIF, QDRT (2506.07121), AC/DC, ACD.
18. **Author-model hardening rounds and open or unbounded targets**
    - What it does: strong models author, then explicitly harden, with cross-model solving and IRT calibration; or target problems with no known ceiling.
    - Example: MathDuels meta-prompting + amplification raised solver error from 9.5% to 25.0% (4-author ablation); UQ's unsolved Stack Exchange questions; FrontierMath Erdős open conjectures in Lean; AlphaEvolve-style (2506.13131) evaluator-scored algorithm discovery.
    - Keep verifiable: an independent verifier ensemble that picks the answer from candidates and rejects ill-posed items; formal checkers or hidden evaluators for open problems. Checkers must stay hidden: in the Darwin Gödel Machine (2505.22954), an agent removed the logging of tool-use tokens that its hallucination detector relied on, and objective hacking was more frequent when checking functions were not hidden.
    - Sources: MathDuels, SKATE, UQ, FrontierMath Erdős, AlphaEvolve, DGM.

## Insights & pitfalls

- **An executable intermediate representation is the dominant design pattern.** Every method with near-zero label noise at scale (DyVal, DARG, GSM-Infinite, RLVE, SCALER, KUMO, DynaMath, Putnam-AXIOM variations) computes the label from code or a solver. Free-form hardening without it fails: in CHASE's baseline, 34 of 100 hard math problems had errors while frontier models still solved 82.5–88.9% of them.
- **Structural knobs give smooth, monotone difficulty curves, which makes them ideal curriculum dials.** GSM-Infinite shows a sigmoid decline with operation count; TRACE reports r ≈ −0.96 between its difficulty metric and accuracy, with per-model correlations from −0.85 to −0.99. Train where the policy sits on the steep part of the curve, and adapt online (RLVE promotion at τ_acc = 0.9; SCALER proportional control toward τ).
- **Adaptive environments beat static datasets for RL, and the number of environments matters.** RLVE: +3.37 versus +0.49 with more than 3× the compute on a saturated model. SCALER beat MATH-7.5k and DeepMath-103K at both 1.7B and 4B. Neither method dominates every benchmark (RLVE beat SCALER on BBEH at 4B), so mix sources.
- **Adversarial selection buys difficulty at the price of label noise.** HLE: 15.4% expert disagreement on the public set; HLE-Verified kept only 668 of 2,500 items unchanged; models gain 30–40 points on erroneous items once they are fixed. AFLite oversamples low-agreement items (Phang et al.). In RL a wrong reference actively penalizes correct reasoning, so never use model failure as the only acceptance criterion.
- **Single-adversary filtering creates adversary-specific artifacts.** Rankings become unstable and adversary-dependent, and the adversary is disproportionately hurt (Phang et al.). Adversarial-only QA training improved results on other adversarial sets but hurt diverse out-of-domain sets (Kaushik et al.). Filter against an ensemble that includes the current policy and mix adversarial with standard data.
- **Generators prefer their own outputs, so verify with a different model family.** GPT-4 scored 93.25% on GSM8K seeds, 87.36% on its own unrevised GSM-Plus variations, and 85.58% on the human-corrected versions; humans revised 18.85% of GPT-4's variations. SKATE finds models pose questions suited to their own strengths. MathDuels finds authoring and solving abilities partially decoupled, so several authors help.
- **Verification weakens exactly where tasks get hard.** ACD's judge-human F1 (0.86 overall) drops on "Very Difficult" tasks. UQ designs validators for precision because hard questions "may appear easy". Prefer operators that preserve programmatic checks; send judge-scored hard tasks to SFT or evaluation rather than to RL reward.
- **Distractor generation creates false negatives.** Expanding options produced 1,953 correct-but-labeled-wrong options in MMLU-Pro's MMLU part. Any operator that adds candidates (options, plausible alternatives, near-miss documents) needs a sweep that asks whether each added item could also be correct.
- **Lenient or unsolvable verifiers are the main RL failure mode, and cheap guards work.** Code-as-Task (known-good solution passes, known-bad candidates fail) sharply reduced both false positives and false negatives. The CHASE k-of-n held-out verifier rule and the SwS student-agreement rule (80.6% → 97.5% labeling accuracy) are cheap additions.
- **Checkers must be hidden from proposers and self-improvers.** In the DGM, objective hacking (disabling tool-use token logging to fool the hallucination detector) was more frequent when checking functions were visible. CompassPlay's proposer produced renamed copies of the seed when it could see the seed.
- **Difficulty is not training value.** CompassPlay beat a difficulty reward (+1.5 coding, +2.7 math, 40% fewer GPU-hours in Lean). DataEnvGym gains peak at mid difficulty. In eva, the choice of signal flips results: reward variance (54.8) and inverse advantage (52.3) underperformed uniform evolution (57.5), while best-vs-mean advantage reached 60.0.
- **Diversity is as important as hardness.** Self-Challenging: 200 tasks degraded test performance, 400 gave marginal gains, 800 gave steady improvement. KUMO: fine-tuning on one domain's optimal trajectories failed out of domain and across difficulty shifts, and graph topology predicts difficulty. ACD still found about 20% novel tasks after 5,000 generations. Use descriptor archives and diversify structure, not just surface topics.
- **Expect heavy over-generation and measure yield per operator.** SwS: the quality filter rejects 78.35% and only about 35% of synthesized problems land in the 25–75% band. CHASE discards at every stage (about 75% of GPT-4o-mini-solvable math items). UQ kept 500 of about 3 million candidates.
- **Hard perturbations expose method memorization, and in-context originals can hurt.** On MATH-P-Hard, o1-mini dropped 16.49%. Most errors ignore the modified assumption rather than copying the old answer. Showing the original problem and solution helps on simple perturbations, but on hard ones a misleading effect offsets the help. Contrastive seed/variant pairs reward checking whether a method applies.
- **Beware cheap hardness.** Context length, planted distractors, tail trivia (AutoBencher drifts there) and tool-computable load all inflate difficulty. Check whether difficulty survives tool access: BBEH's reference model solved Boolean Expressions by running Python, so operands were moved into world-knowledge statements. In MathDuels, code execution only reduced solver error from 25.0% to 21.1%, evidence of genuine reasoning difficulty.
- **Outcome-only rewards reward luck on hard tasks.** TRACE found about 28% spurious guessing (correct answers with invalid reasoning) in mid-sized models. Putnam-AXIOM notes that small answer spaces (true/false, modular arithmetic) invite lucky boxed answers. Worst-case-over-variants scoring (DynaMath), error-localization reframings (MR-GSM8K) and larger answer spaces all reduce it.
- **Operator claims need statistical care.** A 2026 re-analysis of GSM-Symbolic found only 8 of 20 open-weight models had significant variant effects under mixed-effects models, and the variants' integers were shifted larger (K-S 0.12, p < 0.001), which accounted for significance in half of the remaining cases. Control for distribution shift before attributing drops or gains to an operator.
- **Dynamic-benchmark data helps SFT only modestly; the payoff is in adaptive RL curricula.** CHASE fine-tuning on self-generated data gave Llama-3.1-8B 30 → 34.7; MPA augmentation gave about 2%; DataEnvGym gains were +1 to +8 points. RLVE, SCALER and SwS show larger RL gains when these generators are used adaptively.

## Open problems & research opportunities

- **Verifiable complexification beyond executable domains** (proofs, long-form analysis, research questions), where judge reliability collapses at the frontier. Candidate routes are autoformalization into Lean (FrontierMath Erdős), precision-first validator ensembles with abstention (UQ), and error-localization reframings (MR-GSM8K), but there is no scalable, trusted recipe.
- **Cheap estimation of a candidate task's training value** at the scale of millions of candidates. Gradient alignment (CompassPlay), learning-progress prediction (MAGELLAN, OMNI), IRT difficulty (MathDuels) and pass-rate bands (SwS) are each partial, and eva shows naive variance signals can underperform uniform selection. No method is validated for large-scale RLVR selection.
- **Separating genuine reasoning difficulty from cheap hardness** (length, distractors, trivia, arithmetic load, ambiguity, small answer spaces). No accepted difficulty metric predicts capability transfer. The GSM-Symbolic re-analysis shows even the attribution of drops is fragile.
- **Automatic separation of "hard" from "wrong"** in adversarially filtered or amplified synthetic sets. HLE-Verified shows model confidence is associated with item errors, but no automated pipeline reliably repairs labels at RL-dataset scale.
- **Stable long-horizon co-evolution of tasks and policies** without diversity collapse, proposer-solver collusion, verifier exploitation (DGM), seed-copying (CompassPlay) or novelty-filter saturation. ACCEL's minimax-regret guarantees have not been carried over to LLM task generation.
- **Controlled attribution of which operators transfer.** No large ablation compares DARG-style depth, hard perturbations, counterfactual rewrites, distractor injection, composition, reframing and functional variation as RL curricula under a fixed compute budget.
- **Realism versus difficulty.** Chains of lookups (TaskCraft) and templated DAG stories are artificial, while real user tasks skew easy (UQ's diagnosis). Generating tasks that are both hard and representative of deployment remains open. KUMO-style LLM-proposed domains with symbolic backends are one route.
- **Automating environment engineering beyond unique-output problems.** RLVE needed 400 hand-built environments; SCALER lifts only problems with unique, scalable outputs. Lifting arbitrary seeds (multi-answer, interactive, agentic, open-ended) into generators with calibrated knobs and correct checkers is unsolved.
- **Descriptor design for QD archives.** Current descriptors are hand-picked grids or LLM-assigned skill tags. Behavior-based descriptors, such as clusters of policy-failure embeddings or gradient signatures, are largely unexplored, even though KUMO suggests structural topology is what predicts difficulty.
- **Contamination hygiene when the same operators produce training data and dynamic benchmarks.** A policy trained on DARG- or GSM-Infinite-style generators can be evaluated on the same distribution. Held-out operators, seeds, renderers and topologies, and decontamination at the operator level, are needed.
- **Cost-effective human involvement at the frontier.** Human adversarial writing no longer scales against frontier models. Hybrid pipelines in which models author and harden while experts only adjudicate (MathDuels, UQ, HLE-Verified-style repair) lack cost and quality studies and standard protocols.

## References

1. Zhu, K., Chen, J., Wang, J., et al. (2023). *DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks*. ICLR 2024. arXiv:2309.17167. https://arxiv.org/abs/2309.17167
2. Zhu, K., Wang, J., Zhao, Q., et al. (2024). *Dynamic Evaluation of Large Language Models by Meta Probing Agents*. ICML 2024. arXiv:2402.14865. https://arxiv.org/abs/2402.14865
3. Zhang, Z., Chen, J., Yang, D. (2024). *DARG: Dynamic Evaluation of Large Language Models via Adaptive Reasoning Graph*. arXiv:2406.17271. https://arxiv.org/abs/2406.17271
4. Zhou, Y., Liu, H., Chen, Z., et al. (2025). *GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?* arXiv:2502.05252. https://arxiv.org/abs/2502.05252
5. Zeng, Z., Ivison, H., Wang, Y., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*. ICML 2026. arXiv:2511.07317. https://arxiv.org/abs/2511.07317
6. Xu, C., Xiao, C., Peng, Z., et al. (2026). *SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning*. arXiv:2601.04809. https://arxiv.org/abs/2601.04809
7. Lin, H., Wang, X., Yan, R., et al. (2025). *Generative Evaluation of Complex Reasoning in Large Language Models* (KUMO). arXiv:2504.02810. https://arxiv.org/abs/2504.02810
8. Zou, C., Guo, X., Yang, R., et al. (2024). *DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models*. ICLR 2025. arXiv:2411.00836. https://arxiv.org/abs/2411.00836
9. Gulati, A., Miranda, B., Chen, E., et al. (2025). *Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs*. arXiv:2508.08292. https://arxiv.org/abs/2508.08292
10. Wang, S., Long, Z., Fan, Z., et al. (2024). *Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation*. arXiv:2402.11443. https://arxiv.org/abs/2402.11443
11. Xia, C. S., Deng, Y., Zhang, L. (2024). *Top Leaderboard Ranking = Top Coding Proficiency, Always? EvoEval: Evolving Coding Benchmarks via LLM*. arXiv:2403.19114. https://arxiv.org/abs/2403.19114
12. Huang, K., Guo, J., Li, Z., et al. (2025). *MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations*. arXiv:2502.06453. https://arxiv.org/abs/2502.06453
13. Patel, A., Reddy, S., Bahdanau, D. (2025). *How to Get Your LLM to Generate Challenging Problems for Evaluation* (CHASE). arXiv:2502.14678. https://arxiv.org/abs/2502.14678
14. Shi, D., Cao, J., Chen, Q., et al. (2025). *TaskCraft: Automated Generation of Agentic Tasks*. arXiv:2506.10055. https://arxiv.org/abs/2506.10055
15. Li, X. L., Kaiyom, F., Liu, E. Z., et al. (2024). *AutoBencher: Towards Declarative Benchmark Construction*. ICLR 2025. arXiv:2407.08351. https://arxiv.org/abs/2407.08351
16. Zeng, Z., Chen, P., Liu, S., et al. (2023). *MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation*. arXiv:2312.17080. https://arxiv.org/abs/2312.17080
17. Wang, Y., Ma, X., Zhang, G., et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*. NeurIPS 2024 Datasets & Benchmarks. arXiv:2406.01574. https://arxiv.org/abs/2406.01574
18. Kazemi, M., Fatemi, B., Bansal, H., et al. (2025). *BIG-Bench Extra Hard*. arXiv:2502.19187. https://arxiv.org/abs/2502.19187
19. Zellers, R., Holtzman, A., Bisk, Y., et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* ACL 2019. arXiv:1905.07830. https://arxiv.org/abs/1905.07830
20. Le Bras, R., Swayamdipta, S., Bhagavatula, C., et al. (2020). *Adversarial Filters of Dataset Biases*. ICML 2020. arXiv:2002.04108. https://arxiv.org/abs/2002.04108
21. Phang, J., Chen, A., Huang, W., Bowman, S. R. (2021). *Adversarially Constructed Evaluation Sets Are More Challenging, but May Not Be Fair*. arXiv:2111.08181. https://arxiv.org/abs/2111.08181
22. Nie, Y., Williams, A., Dinan, E., et al. (2019). *Adversarial NLI: A New Benchmark for Natural Language Understanding*. ACL 2020. arXiv:1910.14599. https://arxiv.org/abs/1910.14599
23. Kiela, D., Bartolo, M., Nie, Y., et al. (2021). *Dynabench: Rethinking Benchmarking in NLP*. NAACL 2021. arXiv:2104.14337. https://arxiv.org/abs/2104.14337
24. Kaushik, D., Kiela, D., Lipton, Z. C., Yih, W. (2021). *On the Efficacy of Adversarial Data Collection for Question Answering: Results from a Large-Scale Randomized Study*. ACL-IJCNLP 2021. arXiv:2106.00872. https://arxiv.org/abs/2106.00872
25. Bartolo, M., Thrush, T., Jia, R., et al. (2021). *Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation*. EMNLP 2021. arXiv:2104.08678. https://arxiv.org/abs/2104.08678
26. Phan, L., Gatti, A., Han, Z., et al. (2025). *Humanity's Last Exam*. arXiv:2501.14249. https://arxiv.org/abs/2501.14249
27. Zhai, W., Wang, Z., Wang, J., et al. (2026). *HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam*. arXiv:2602.13964. https://arxiv.org/abs/2602.13964
28. Nie, F., Liu, K. Z., Wang, Z., et al. (2025). *UQ: Assessing Language Models on Unsolved Questions*. arXiv:2508.17580. https://arxiv.org/abs/2508.17580
29. Glazer, E., Erdil, E., Besiroglu, T., et al. (2024). *FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI*. arXiv:2411.04872. https://arxiv.org/abs/2411.04872
30. Adamczewski, T., Bloom, T. F. (2026). *FrontierMath Erdős*. arXiv:2609.25050. https://arxiv.org/abs/2609.25050
31. Xu, Z., Jin, S., Arya, S., Naik, M. (2026). *MathDuels: A Self-Play Benchmark That Grows*. arXiv:2604.21916. https://arxiv.org/abs/2604.21916
32. Gould, D. S. W., Mlodozeniec, B., Brown, S. F. (2025). *SKATE, a Scalable Tournament Eval: Weaker LLMs differentiate between stronger ones using verifiable challenges*. arXiv:2508.06111. https://arxiv.org/abs/2508.06111
33. Khan, Z., Stengel-Eskin, E., Cho, J., Bansal, M. (2024). *DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback*. ICLR 2025. arXiv:2410.06215. https://arxiv.org/abs/2410.06215
34. Zeng, Z., Wang, Y., Hajishirzi, H., Koh, P. W. (2025). *EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees*. COLM 2025. arXiv:2503.08893. https://arxiv.org/abs/2503.08893
35. Cheng, J., Lu, Y., Gu, X., et al. (2024). *AutoDetect: Towards a Unified Framework for Automated Weakness Detection in Large Language Models*. EMNLP 2024 Findings. arXiv:2406.16714. https://arxiv.org/abs/2406.16714
36. Liang, X., Li, Z.-Z., Gong, Y., et al. (2025). *SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning*. arXiv:2506.08989. https://arxiv.org/abs/2506.08989
37. Ye, Z., Agarwal, R., Liu, T., et al. (2024). *Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play* (eva). arXiv:2411.00062. https://arxiv.org/abs/2411.00062
38. Parker-Holder, J., Jiang, M., Dennis, M., et al. (2022). *Evolving Curricula with Regret-Based Environment Design* (ACCEL). arXiv:2203.01302. https://arxiv.org/abs/2203.01302
39. Zhang, J., Lehman, J., Stanley, K., Clune, J. (2023). *OMNI: Open-endedness via Models of human Notions of Interestingness*. arXiv:2306.01711. https://arxiv.org/abs/2306.01711
40. Faldor, M., Zhang, J., Cully, A., Clune, J. (2024). *OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code*. arXiv:2405.15568. https://arxiv.org/abs/2405.15568
41. Lu, C., Hu, S., Clune, J. (2025). *Automated Capability Discovery via Foundation Model Self-Exploration*. arXiv:2502.07577. https://arxiv.org/abs/2502.07577
42. Pourcel, J., Colas, C., Molinaro, G., et al. (2023). *ACES: Generating Diverse Programming Puzzles with Autotelic Generative Models*. arXiv:2310.10692. https://arxiv.org/abs/2310.10692
43. Samvelyan, M., Raparthy, S. C., Lupu, A., et al. (2024). *Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts*. arXiv:2402.16822. https://arxiv.org/abs/2402.16822
44. Bradley, H., Dai, A., Teufel, H., et al. (2023). *Quality-Diversity through AI Feedback*. arXiv:2310.13032. https://arxiv.org/abs/2310.13032
45. Dai, A., Meinardus, B., Regan, C., et al. (2026). *Discovering Novel LLM Experts via Task-Capability Coevolution* (AC/DC). ICLR 2026. arXiv:2604.14969. https://arxiv.org/abs/2604.14969
46. Zhou, Y., Levine, S., Weston, J., et al. (2025). *Self-Challenging Language Model Agents*. arXiv:2506.01716. https://arxiv.org/abs/2506.01716
47. Pu, S. X., Sun, X., Liu, J., et al. (2026). *CompassPlay: Rewarding the Proposer for Where It Moves the Solver*. arXiv:2609.32228. https://arxiv.org/abs/2609.32228
48. Zhao, A., Wu, Y., Yue, Y., et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335. https://arxiv.org/abs/2505.03335
49. Li, Q., Cui, L., Zhao, X., et al. (2024). *GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers*. ACL 2024. arXiv:2402.19255. https://arxiv.org/abs/2402.19255
50. Mirzadeh, I., Alizadeh, K., Shahrokhi, H., et al. (2024). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models*. ICLR 2025. arXiv:2410.05229. https://arxiv.org/abs/2410.05229
51. Długosz, D. A., Oliveira, A., Díaz-Rodríguez, N. (2026). *The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic*. EMNLP 2026. arXiv:2605.28700. https://arxiv.org/abs/2605.28700
52. Wu, Z., Qiu, L., Ross, A., et al. (2023). *Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks*. NAACL 2024. arXiv:2307.02477. https://arxiv.org/abs/2307.02477
53. O'Brien, D., Haddow, B., Allaway, E., Chen, P. (2025). *MatheMagic: Generating Dynamic Mathematics Benchmarks Robust to Memorization*. arXiv:2510.05962. https://arxiv.org/abs/2510.05962
54. Xu, X., Lawrence, R., Dubey, K., et al. (2025). *RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation*. ICML 2025. arXiv:2506.15455. https://arxiv.org/abs/2506.15455
55. Yang, C., Wang, R., Jiang, J., et al. (2025). *Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs* (NPPC). TMLR. arXiv:2504.11239. https://arxiv.org/abs/2504.11239
56. Lei, F., Liu, Q., Huang, Y., et al. (2023). *S3Eval: A Synthetic, Scalable, Systematic Evaluation Suite for Large Language Models*. NAACL 2024. arXiv:2310.15147. https://arxiv.org/abs/2310.15147
57. Lacombe, V., Quesnel, V., Sileo, D. (2026). *Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre-training and Post-Training*. arXiv:2603.02208. https://arxiv.org/abs/2603.02208
58. Zhou, S., Wang, K., Shi, L., Wang, H. (2026). *A Temporal Reasoning Benchmarking Framework for LRMs via Difficulty-controlled and Dynamic Test Generation* (TRACE). ISSTA 2026. arXiv:2607.04784. https://arxiv.org/abs/2607.04784
59. Xu, C., Sun, Q., Zheng, K., et al. (2023). *WizardLM: Empowering Large Pre-trained Language Models to Follow Complex Instructions*. arXiv:2304.12244. https://arxiv.org/abs/2304.12244
60. Zeng, W., Xu, C., Zhao, Y., et al. (2024). *Automatic Instruction Evolving for Large Language Models* (Auto Evol-Instruct). arXiv:2406.00770. https://arxiv.org/abs/2406.00770
61. Zala, A., Cho, J., Lin, H., et al. (2024). *EnvGen: Generating and Adapting Environments via LLMs for Training Embodied Agents*. COLM 2024. arXiv:2403.12014. https://arxiv.org/abs/2403.12014
62. Dong, G., Lu, J., Huang, J., et al. (2026). *Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence*. arXiv:2604.18292. https://arxiv.org/abs/2604.18292
63. Gaven, L., Carta, T., Romac, C., et al. (2025). *MAGELLAN: Metacognitive Predictions of Learning Progress Guide Autotelic LLM Agents in Large Goal Spaces*. arXiv:2502.07709. https://arxiv.org/abs/2502.07709
64. Wang, R.-J., Xue, K., Qin, Z., et al. (2025). *Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models*. arXiv:2506.07121. https://arxiv.org/abs/2506.07121
65. Abichandani, H., Chong, P., Shen, J., et al. (2026). *EDGEGEN: Improving Tool-Calling Agents Beyond Happy Paths with Synthetic Edge Case Generation*. arXiv:2609.24115. https://arxiv.org/abs/2609.24115
66. Novikov, A., Vũ, N., Eisenberger, M., et al. (2025). *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery*. arXiv:2506.13131. https://arxiv.org/abs/2506.13131
67. Zhang, J., Hu, S., Lu, C., et al. (2025). *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*. arXiv:2505.22954. https://arxiv.org/abs/2505.22954
