# Other domains: multimodal/visual reasoning, science, text-to-SQL/tables, and non-verifiable domains

*Scope: making SFT/RL tasks harder, with answers or rewards that stay correct, outside text-only math and code. Covers vision-language reasoning and visual generation, physics/chemistry/optimization/science corpora, text-to-SQL and tables, medicine, and rubric-graded open-ended tasks (writing, consultation, legal and moral reasoning). Compiled 2026-09-30. Verification: 28 entries checked against primary sources (full arXiv text, fetched and searched for every number); 10 corrected, 0 dropped, 5 added. Another 22 cited arXiv IDs and 6 insight-level claims were also resolved; 1 of those claims was corrected.*

## TL;DR

- **Build the latent state first, compute the answer from it, and only then render the image or write the question.** Nearly every sound generator here does this: a scene, a program, a solver model, a simulator trace, an SQL AST or a symbolic figure. Examples are TRON, Trace, VVR, ChartVerse, GeoSym127K, AutoOR, Sim2Reason and EvolSQL. Because the verifier never has to read pixels or generated prose, the reward is exact, the supply is unlimited, and difficulty becomes an explicit parameter: level, depth, number of constraints, number of variables, AST operators.
- **For seeds you already have and that are saturated, the cheapest fixes need no new generator:**
  - Answer-hidden rewriting gated by the policy's pass rate (SynthRL). Harden seeds the policy solves at 12/16 or better; accept a rewrite only if 4 ≤ passes ≤ original − 2.
  - Recompose several seeds' skill factors into one multi-hop question (COGS).
  - Inject one known error and ask the model to find it (ViCrit).
  - Mask a reasoning span and ask a 9-way multiple-choice question to fill it (Golden Goose).
- **Calibrate difficulty against the target policy, not against a generator's opinion.** Keep items with 0 < pass < 1, or inside a band: 0.375–0.875 (VisualSphinx), 20–50% (QUBRIC), at least 1 of 10 correct (Arctic-Text2SQL-R1).
  - Generic "make it harder" rewriting that ignores the policy often hurts. Evol-Instruct prompts trained worse than unchanged base prompts (50.24 vs 50.51, LLM-as-a-Tutor). LLM-generated "complex question" augmentation for SQL cut BIRD-dev from 64.9 to 62.5 (Arctic). QUBRIC's surface-only rewrite scored 67.1 on ArenaHard-hard, below its own SFT starting point of 71.0.
- **Make levels change the reasoning mechanism, and check this with a pass-rate curve on the base model.** TRON's angle-chase goes from 1 deduction step to 4; the base pass rate falls from 72.8% at level 0 to 41.3% at level 9.
  - Easy-to-hard transfer is real but fades with distance. VVR trained only on easy tasks and gained +17.16, +8.31 and +1.35 points on progressively harder bins.
  - So mix levels and keep replaying lower ones (TRON replays with probability 0.30). In Game-RL an Easy+Medium+Hard mix beat every single-tier set.
- **Use synthetic hard tasks through RL; supervised fine-tuning (SFT) on them often regresses.**
  - Sim2Reason: SFT on 200K teacher solutions scored −3.9 on IPhO, while RL scored +5.4.
  - AutoOR: SFT on its own synthetic data scored 26 on Hard-LP, versus 80 for RL (the base model scored 55).
  - Game-RL: SFT lost 3.9% on general benchmarks; RL gained 2.33%.
  - ViCrit: SFT on the captions (CapSFT) averaged 48.41; RL averaged 53.01; the base averaged 50.61.
  - Jigsaw-R1: an SFT cold start hurt the RL that followed.
  - Exceptions: GeoSym, ChartVerse and HuatuoGPT-o1, where the model lacks domain knowledge or output format, did better with SFT before RL.
- **Filter out shortcuts and degenerate items before scaling.**
  - Sim2Reason's component-ablation shortcut filter was the difference between 7.14% and 13.15% on IPhO at 3B.
  - Arctic drops SQL whose gold query returns an empty result or runs longer than 5 s.
  - EvolSQL requires queries that execute and return non-empty results.
- **Task classes the policy never solves give no gradient.**
  - AutoOR's non-linear class scored 0% even at pass@64. It was fixed by putting solver syntax in the prompt and then removing it in stages, reaching 48.98%.
  - ether0 saw groups where every rollout got the same reward reach 90% of the batch. It fixed this with a buffer of problems that were recently non-trivial.
- **Non-verifiable domains need two things hardened: the prompts and the rubrics.**
  - Prompts: append one atomic constraint to prompts that have saturated (LLM-as-a-Tutor), or ground vague queries in scenarios built from key points (QUBRIC).
  - Rubrics: harden them from pairs of top responses (RubricHub difficulty evolution), let them co-evolve with the policy (EvoRubrics), or randomly drop 30–50% of criteria each step (Rubric Dropout).
  - Fixed rubrics lose their power to separate good from bad responses and get exploited.
- **Conjunctions are the hard part.** VVR's easy-only training closed 79% and 64% of the partial-score gap on counts and relations, but only 55% and 46% of the all-satisfied gap.
  - Use partial credit multiplied by an all-constraints guard (VVR's ψ).
  - For very hard perception tasks, keep partial credit so learning can start. Visual Jigsaw needed a discounted partial reward (γ = 0.2) to learn 3×3 grids at all.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| VVR / RLVVR (VVRBench) | 2026 | [2609.35641](https://arxiv.org/abs/2609.35641) | Text-to-image instruction following | RL (Flow-GRPO) | Scene-first constraint stacking; instance scaling; group-level constraint types; complexity score C(s) | Constraints kept only if the realized scene satisfies them; deterministic pixel→object→program verifier (94.7% agreement with humans) |
| SynthRL | 2025 | [2506.02096](https://arxiv.org/abs/2506.02096) | Visual math | RL (GRPO) | Answer-preserving, answer-hidden question rewriting | Target-policy Monte Carlo gate: ≥4/16 still reach the original answer and ≥2 fewer passes than the seed |
| TRON | 2026 | [2606.01599](https://arxiv.org/abs/2606.01599) | Visual reasoning (5 ability buckets) | RL (DAPO) | 10-level generators where the level changes the mechanism; online fresh instances; promote at 0.80 accuracy | Answer computed from latent state before rendering; per-environment audit (502/520 grade A) |
| Trace | 2026 | [2607.19790](https://arxiv.org/abs/2607.19790) | 11 visual domains | RL (GRPO) | Semantic-parameter scaling vs render-only variation; scene grammar × task program | Task program executed on shared semantic state; typed answer normalization |
| VisualSphinx | 2025 | [2505.23977](https://arxiv.org/abs/2505.23977) | Visual logic puzzles | RL (GRPO) | Rule-level genetic mutation/crossover; 10-option distractors from related rules; pass-rate band | Correct and violating images rendered by separate LLM-written scripts; format/feasibility scoring |
| Game-RL / Code2Logic (GameQA) | 2025 | [2505.13886](https://arxiv.org/abs/2505.13886) | Game-state visual reasoning | RL (GRPO) | Game-code wrapping; QA level × plot (state-size) level | Answers computed by running game logic |
| Jigsaw-R1 (+ Visual Jigsaw) | 2025 | [2505.23590](https://arxiv.org/abs/2505.23590) | Vision-centric perception | RL | Shuffle-and-reconstruct; grid size; pair→full format; video clips/depth ordering (Visual Jigsaw) | Permutation known by construction |
| ViCrit | 2025 | [2506.10128](https://arxiv.org/abs/2506.10128) | Perception/hallucination on natural images | RL (GRPO) | Single-error injection into ~200-word captions → span localization | Injected span known; string-match reward |
| EVE | 2026 | [2604.18320](https://arxiv.org/abs/2604.18320) | Self-evolving VLM | RL (GRPO) | Challenger-written executable image transforms, composed over time; difficulty reward peaking at 50% | Answers from code execution (parameter↔image matching) |
| Vision-Zero | 2025 | [2509.25541](https://arxiv.org/abs/2509.25541) | Label-free VLM self-play | RL (self-play + RLVR) | "Who is the Spy" games on image pairs; alternating clue/decision stages | Spy identity known by construction (decision stage) |
| Active-Zero | 2026 | [2602.11241](https://arxiv.org/abs/2602.11241) | Self-evolving VLM with image retrieval | RL (GRPO) | Searcher retrieves frontier images; Questioner calibrated to ~50% | Majority-vote pseudo-labels (silver standard) + difficulty window |
| ChartVerse | 2026 | [2601.13606](https://arxiv.org/abs/2601.13606) | Chart reasoning | SFT + RL (GSPO) | Complexity-aware chart coder selected by Rollout Posterior Entropy (RPE); answer-first inverse QA; fail-rate filter | Answer from executing a Python script over chart source; teacher re-answer must match |
| COGS | 2025 | [2510.15040](https://arxiv.org/abs/2510.15040) | Charts, webpages | RL (GRPO) | Factor decomposition + recomposition on new images; sub-question process reward | LLM-generated sub-answers (metadata when available); LLM checks sub-answers in CoT |
| GeoSym127K | 2026 | [2605.16371](https://arxiv.org/abs/2605.16371) | Diagram geometry (lengths, angles, areas) | SFT + RL (GRPO) | Typed grammar recursion depth; constructive augmentation; 3 tiers | Exact symbolic solver (generalized shoelace); SymPy Simplify(pred − GT) ≡ 0 for CoT |
| TrustGeoGen | 2025 | [2504.15780](https://arxiv.org/abs/2504.15780) | Formal geometry proofs/solutions | SFT | Forward-chaining depth; premise bootstrapping of deepest graphs | Every step a rule application in a deduction graph; compiler-checked constructions |
| CoSyn *(added)* | 2025 | [2502.14846](https://arxiv.org/abs/2502.14846) | Text-rich images (charts, docs, tables, UI) | SFT | Code as intermediate: 20 pipelines × 11 renderers; persona-conditioned topics | QA generated from the rendering code (text-only LLM), not from pixels |
| Sim2Reason | 2026 | [2604.11805](https://arxiv.org/abs/2604.11805) | Physics (mechanics, orbits, EM) | RL (GSPO) | Random DSL scene composition; forward/inverse/symbolic question types; shortcut filtering | MuJoCo-recorded values; 5% relative tolerance; unstable-segment pruning |
| ether0 | 2025 | [2506.17238](https://arxiv.org/abs/2506.17238) | Chemistry | SFT + RL (GRPO) | Property-constrained design; templated tasks from experimental DBs; problem rewriting; curriculum buffer | RDKit checks, ML oracles, purchasability Bloom filter, "reasonable molecule" check |
| AutoOR | 2026 | [2604.16804](https://arxiv.org/abs/2604.16804) | Operations research (LP/MILP/NLP) | RL (Dr. GRPO) | Code-first → back-translation; variable scaling 4→12; family expansion; scaffold-then-fade curriculum | Solver-computed optimum; component-wise description check; tolerance-based reward |
| MegaScience / TextbookReasoning | 2025 | [2507.16812](https://arxiv.org/abs/2507.16812) | University science | SFT | High-standard multi-step extraction; self-containment rewriting; difficulty selection | Textbook reference answers; MinHash dedup; LLM decontamination |
| NaturalReasoning *(added)* | 2025 | [2502.13124](https://arxiv.org/abs/2502.13124) | Multi-domain STEM/econ/social | SFT, self-training | Compose novel, self-contained hard questions from reasoning-rich pretraining documents | Reference answer extracted from source document when derivable (81.68%) |
| Golden Goose (GooseReason) | 2026 | [2601.22975](https://arxiv.org/abs/2601.22975) | STEM text, code, cybersecurity | RL | Mask key reasoning span; 9-option distractors; drop 16/16-solved | Masked original text is the answer; exact option match |
| OmniSQL / SynSQL-2.5M | 2025 | [2503.02240](https://arxiv.org/abs/2503.02240) | Text-to-SQL | SFT | DB widening; 4 complexity tiers; SQL→question in 9 styles | Execution filtering; execution-grouped majority vote over CoTs |
| EvolSQL | 2026 | [2601.04875](https://arxiv.org/abs/2601.04875) | Text-to-SQL | SFT | 6 AST operators (wrap, mutate, clause, JOIN, nest, set); scarcity-weighted selection | Execution-grounded refinement; must execute and return non-empty |
| Arctic-Text2SQL-R1 *(added)* | 2025 | [2505.20315](https://arxiv.org/abs/2505.20315) | Text-to-SQL | RL (GRPO) | Distractor tables added to synthetic schemas; long-SQL filter; ≥1/10 solvability filter | Execution-only reward; drop empty-result / >5 s gold SQL |
| TableDreamer | 2025 | [2506.08646](https://arxiv.org/abs/2506.08646) | Table understanding | SFT | 7 instruction-complication strategies; table generalization; weakness-guided rounds | GPT-4o reference + LLM judge (not executable) |
| HuatuoGPT-o1 | 2024 | [2412.18925](https://arxiv.org/abs/2412.18925) | Medical exams | SFT + RL (PPO) | Small-model difficulty filter; MCQ → open-ended; answer-uniqueness filter | GPT-4o alias-aware verifier (96.5%/94.5% vs 70.5%/74.5% exact match) |
| Baichuan-M2 | 2025 | [2509.02208](https://arxiv.org/abs/2509.02208) | Clinical consultation | SFT + RL (GRPO variant) | Static QA → multi-turn simulated patient with persona; per-case rubrics | Rubric generator (92.7% consistency with experts on 100 cases) |
| LLM-as-a-Tutor | 2026 | [2607.04412](https://arxiv.org/abs/2607.04412) | Open-ended instruction following | RL (rubric) | Pairwise saturation detection → append one atomic constraint + rubric item | Rubric-based LLM judge; base rubric kept |
| QUBRIC | 2026 | [2606.03968](https://arxiv.org/abs/2606.03968) | Open-ended QA, creative, legal/moral | RL (GRPO, rubric) | Key-point-grounded scenario rewriting; contrastive rubrics; 20–50% band | "Constitutive" rubrics containing the grading knowledge |
| EvoRubrics *(added)* | 2026 | [2606.23038](https://arxiv.org/abs/2606.23038) | Open-ended (medical, IF, chat) | RL (GRPO) | Rubric generator co-trained adversarially to stay discriminative | Rubric-aggregated LLM judging; generator rewarded for discrimination, diversity, alignment, constructiveness |
| RubricHub *(added)* | 2026 | [2601.08430](https://arxiv.org/abs/2601.08430) | Multi-domain open-ended | Rejection-sampling SFT + RL | Difficulty evolution: add criteria separating "excellent" from "exceptional" | Multi-model aggregated, response-grounded rubrics |
| Writing-Zero (BRPO) | 2025 | [2506.00103](https://arxiv.org/abs/2506.00103) | Creative writing | RL | Relative bar: in-group bootstrapped reference; skew filter | Pairwise generative reward model (GenRM) with self-principled critique; no ground truth |

## Method notes

**— Group: Multimodal: answer-first generators and environments —**

### VVR / RLVVR — Verifiable Visual Rewards Transfer from Synthetic Scenes to Natural Prompts (Li et al., 2026)
[arXiv 2609.35641](https://arxiv.org/abs/2609.35641) · Shuyue Stella Li, Xiaochuang Han, Yulia Tsvetkov, Luke Zettlemoyer · code/data: github.com/stellalisy/VVRBench
- **Mechanism:**
  - Sample a 2D scene first: groups of objects with colour, shape, position and size.
  - Enumerate candidate constraints from a library of **46 constraint types** (count, attribute, spatial, size and topology families) whose arity matches the groups. Keep only those the realized scene satisfies.
  - Render each constraint with a fixed template phrase, so every requirement maps to a phrase and back ("expression faithfulness").
  - VVRBench uses 32 of the 46 types; the other 14 group-level types appear only in the Challenge split.
  - Each task is validated: the verifier must accept the reference image and reject a counterfactual (for example all_same_row → not_all_same_row).
- **How it makes tasks harder:** more constraints; more instances per group; count ratios ("times_as_many", k ∈ {2..5}); one-to-one containment ("each A inside a different B"); larger grids (3×4 in Challenge); group-level types.
- **Correctness / verification:**
  - Pixel pipeline: HSV masks with fixed, non-overlapping hue ranges for 8 colours and background-adaptive contrast, then morphological cleanup, then connected components, then a contour-based shape classifier. Program checks for each constraint follow.
  - About 3,947 human decisions set the predicate boundaries. The released verifier agrees with 481/508 human labels (94.7%, κ = 0.89).
  - Joint satisfiability is guaranteed because constraints are read off a realized scene. This matters: "A contains B, B contains C, C contains A" is pairwise satisfiable but jointly impossible.
- **Difficulty control:**
  - Structural complexity C(s) = Σ c(a;s) with L(n) = 1 + log2 n. Colour, shape and exact count cost L(n); most relations cost L(N); times_as_many adds log2 k; each_inside costs the number of required matches.
  - VVRBench: 10,000 tasks at complexity 3–48 in five bins (3–16, 16–21, 21–26, 26–31, 31–48). VVRBench-Fast: 820 tasks. VVRBench-Challenge: 720 tasks at complexity 45–80.
  - Training sets: VVR-Easy is 100K tasks at C ≤ 20; VVR-Matched is 100K tasks following the benchmark distribution.
- **Reward design:**
  - Exact reward: product of all per-constraint decisions.
  - Dense reward: ψ · Σ w_a q_a, where ψ ∈ [0,1] is a multiplicative penalty that stops the policy exploiting one easy constraint.
  - Per-group weights: 0.45 count + 0.20 shape + 0.20 layout + 0.15 size.
- **Reported results:**
  - SD3.5-Medium with Flow-GRPO on VVR-Easy: 2.81% → 28.27%. Bins C3–C5 (all harder than any training task) gain +17.16, +8.31 and +1.35. VVR-Matched reaches 46.60%.
  - Transfer: GenEval +0.113, OCR +0.111. Human preference on 160 natural prompts outside VVR: 71.6% win rate (83.8% pairwise agreement).
  - Best open model: FLUX.2-dev at 19.2% on VVRBench. Best overall: GPT-Image-2.5-Sunburst at 21.4% on Challenge.
  - Across API models, 95% of grounding constraints are satisfied but only 56% of topology constraints; same_count passes in 31% of checks and times_as_many in 33%.
- **Limitations / failure modes:**
  - Only 8 colours, 3 shapes, plain backgrounds, 2D.
  - Conjunction gap: easy-only training closes 79% and 64% of the partial-score gap on counts and relations, but only 55% and 46% of the all-satisfied gap.
  - Only SD3.5-M with Flow-GRPO was trained ("RL-Zero").
- **How to reuse with easy seed tasks:**
  - For any output a program can parse (SVG, HTML/CSS layout, charts, CAD, LaTeX, UI trees, JSON), sample the target state first and read the constraints off it.
  - Score complexity additively, and raise it by adding constraints, instances or group-level relations.
  - Reward partial credit multiplied by an all-satisfied guard.
  - Keep a reference solution and a counterfactual for every generated task, as a unit test of the verifier.

### TRON — TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL (Yang et al., 2026)
[arXiv 2606.01599](https://arxiv.org/abs/2606.01599) · Tianze Yang, Yucheng Shi, Ruitong Sun, Jingyuan Huang et al.
- **Mechanism:**
  - 520 Python generator–verifier environments in 5 ability buckets: spatial, mathematical, diagram, pattern/logic, counting.
  - An instance is made in five steps: (env, ℓ) fixes a problem type; a seed fixes the free variables; a formula or solver computes the answer; the image is rendered with the answer slot left blank; the wording is sampled from a paraphrase pool.
  - Every rollout draws a fresh instance.
- **How it makes tasks harder:**
  - The level ℓ ∈ 0..9 switches the mechanism, not just the surface. Angle-chase goes from a one-step triangle sum at ℓ = 0 to a four-step composite chain over parallel lines at ℓ = 9.
  - In chart aggregation, ℓ scales the number of series, the number of time points and whether series are stacked.
- **Correctness / verification:**
  - The answer is fixed before rendering, so the verifier never parses the image.
  - Audit: quality is the minimum of generation, image, QA and verifier rates, graded A/B/C at 0.98/0.90/0.75. Generation succeeds on 99.07% of probes; 502/520 environments get grade A and the other 18 were re-authored.
  - Diversity audit within a level: perceptual-hash spread, question-template fraction and answer entropy.
  - Diversity audit across levels: pHash distance, template-Jaccard distance and foreground-complexity shift, which flags difficulty knobs that change nothing.
  - Near-duplicate checks run across environments.
- **Difficulty control:**
  - 8 rollouts per prompt. Each environment keeps a window of its 4 most recent groups (32 scored trajectories) and is promoted to ℓ+1 once mean accuracy reaches 0.80; the window then resets.
  - Lower-level instances are mixed in with probability 0.30.
  - Groups that are all correct or all wrong are filtered out.
  - Base-model check: Qwen3-VL-4B-Instruct pass rate is 72.8% at ℓ0, 59.9% at ℓ3, 48.0% at ℓ6 and 41.3% at ℓ9.
- **Reported results:** mean over 10 external benchmarks: Qwen3-VL-4B 52.61 → 55.23; Qwen2.5-VL-7B 40.85 → 43.35; MiMo-VL-7B-SFT 63.37 → 66.50.
- **Limitations / failure modes:** the images look synthetic. The paper has no ablation section, so online vs. static data and curriculum vs. uniform sampling are untested. Authoring cost is high.
- **How to reuse with easy seed tasks:**
  - Turn each easy seed family into a generator whose level adds deduction steps, series, hops or entities.
  - Plot the base model's pass rate per level before training; a flat curve means the knob is cosmetic.
  - Promote at about 80% and keep replaying lower levels.

### Trace — Trace: A Taxonomy-Guided Environment for Multidomain Visual Reasoning (Alam, 2026)
[arXiv 2607.19790](https://arxiv.org/abs/2607.19790) · Md Tanvirul Alam (single author; also first author of SPHINX, 2511.20814)
- **Mechanism:**
  - 1,000 tasks over 277 scene grammars in 11 visual domains: charts, games, geometry, graphs, icons, illustrations, pages, physics, puzzles, symbolic notation and 3D scenes.
  - Each task is a scene grammar plus an executable task program plus a reward contract. One semantic state determines the image, prompt, typed answer, verifier state and a replayable trace.
  - 317 tasks expose several internal queries, giving 1,475 query variants.
  - Tasks are also tagged with 13 operation families, but only as an analytical view.
- **How it makes tasks harder:**
  - *Semantic* parameters (counts, values, thresholds, board sizes) change the state and may change the answer.
  - *Render-only* parameters (placement, layout, palette, typography, camera) must preserve every answer-relevant relation; the typed answer is re-checked after rendering.
- **Correctness / verification:** the task program runs on the semantic state; answers are compared with type-aware normalization (integers, canonical numerics, strings, option letters); samples from every task were inspected for clarity, prompt–image consistency and legibility.
- **Difficulty control:** set through generation parameters. The author states there is no common difficulty notion across domains. Training used uniform task sampling.
- **Reported results:**
  - Training: 64,000 instances (64 per task), GRPO with 8 responses per prompt, 500 updates.
  - Held-out Trace accuracy: 24.45 → 41.05 at 3B and 34.25 → 51.55 at 7B.
  - 24-benchmark macro-average: 39.34 → 42.85 (+3.51) at 3B and 47.93 → 51.99 (+4.06) at 7B. Positive on 21/24 benchmarks at 3B and 24/24 at 7B.
  - The largest category gain is visual math (+7.44 at 3B, +5.65 at 7B); WeMath rises about 11 points at both scales.
- **Limitations / failure modes:** no mixture ablations, so it is unknown which domain or operation family drives transfer; one run per scale; no curriculum.
- **How to reuse with easy seed tasks:** separate "what the answer depends on" from "how it looks". Randomize render-only parameters freely for robustness and scale semantic parameters for difficulty. Record a replayable trace per instance for auditing.

### Game-RL / Code2Logic — Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning (Tong et al., 2025)
[arXiv 2505.13886](https://arxiv.org/abs/2505.13886) · Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou et al.
- **Mechanism:**
  - An LLM writes the game code from a short prompt (Sokoban needs a one-line prompt).
  - QA templates are designed around the game's own functions; for example, the move function suggests "where does the player end up after these steps?"
  - A data engine initializes states, solves them with game code and emits QA pairs with reasoning chains.
  - Data filtering removes items by length, >70% 4-gram overlap and wrong answers (about 150K → 126,760).
- **How it makes tasks harder:** two independent code-parameter axes: QA level (question complexity) and plot level (grid size or state complexity), each with 3 settings.
- **Correctness / verification:** answers are produced by executing game logic.
- **Difficulty control:** 3 × 3 levels. For the difficulty study, tasks were bucketed by baseline accuracy: Easy 55–85%, Medium 30–55%, Hard 5–30%.
- **Reported results:**
  - GameQA has 30 games, 158 tasks, about 140K questions (126,760 train).
  - GRPO on 5K samples with Qwen2.5-VL-7B raises the 7-benchmark average from 49.94 to 52.27 (+2.33): MathVista +1.40, MathVerse +2.89, MMBench −0.14, MMMU +1.52, CharXiv +5.00, MathVision +2.27, MMMU-Pro +3.40.
  - Difficulty mix (5K each): Easy+Medium+Hard 52.74 is best, versus Easy 52.40, Medium 51.95, Hard 52.07.
  - SFT on GameQA lowers general benchmarks (−3.9% for SFT, −2.5% for SFT then RL), while pure RL gives +2.33%.
  - About 5K samples from 20 games beat 4 games (Qwen2.5-VL-3B).
  - Humans score 84.75% on GameQA.
- **Limitations / failure modes:** game visuals are far from natural images; modest general gains.
- **How to reuse with easy seed tasks:** wrap rule-governed seeds in a state machine and ask state-prediction or planning questions. Grow state size and step count, prefer many distinct games over many samples per game, and train on a mix of difficulties.

### VisualSphinx — VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL (Feng et al., 2025)
[arXiv 2505.23977](https://arxiv.org/abs/2505.23977) · Yichen Feng, Zhangchen Xu, Fengqing Jiang, Yuetai Li et al.
- **Mechanism:**
  - About 4K (3,904) visual logic questions with explanations from the Chinese Civil Service Examination were translated and rewritten to remove answer leakage.
  - They became 2,398 seed rules, each written as five bullet points and grouped into 8 classes.
  - A rule-level genetic algorithm runs on per-class islands: mutation rewrites, adds or deletes bullets; crossover interleaves bullets from two parents; rules migrate between islands every 3 generations.
  - After 10 generations: 60,339 candidate rules, of which 41,287 are kept after deduplication and LLM scoring.
  - An LLM writes correct_script.py (5 rule-following images) and incorrect_script.py (3 rule-violating images) in 3 styles, giving about 110K image groups.
- **How it makes tasks harder:** assembly variants: default 4-option, answer-shuffled, and a 10-option version that adds 6 distractors from image groups of genetically related (parent or ancestor) rules.
- **Correctness / verification:** labels come from construction (following vs. violating script). Rules are scored for format, content and code-feasibility; puzzles for readability and logical coherence (93.1% scored ≥ 4/5 on readability).
- **Difficulty control:**
  - Pass rate is measured with a GRPO-trained Qwen2.5-VL-7B annotation model; 14,000 puzzles are never solved.
  - RL set: 10K puzzles with pass rate 0.375–0.875 and readability + coherence ≥ 8, split 80% 4-option and 20% 10-option.
- **Reported results:**
  - Over 660K puzzles for under $1,000 (about $0.0015 each).
  - Qwen2.5-VL-7B with GRPO (256 steps) on the VisualSphinx test: 29.30 → 55.94 (GPT-4.1: 55.72).
  - MathVista-testmini: 59.4 → 64.0.
- **Limitations / failure modes:** LLM-written rendering code may not faithfully realize the rule, and a distractor could accidentally satisfy it; the never-solved items may partly be label noise.
- **How to reuse with easy seed tasks:** abstract seed items into short rule specs, evolve the specs, render with code, and harden options by adding distractors from neighbouring specs. Filter to a pass-rate band on your own policy.

### GeoSym127K — GeoSym127K: Scalable Symbolically-verifiable Synthesis for Multimodal Geometric Reasoning (Jing et al., 2026)
[arXiv 2605.16371](https://arxiv.org/abs/2605.16371) · Jinhao Jing, Zheng Ma, Jinwei Liang, Qiannian Zhao et al. · data: huggingface.co/datasets/Tomie0506/GeoSym127K
- **Mechanism:**
  - A type-conditional grammar starts from primitives with exact symbolic coordinates: circles, regular polygons, triangle and quadrilateral types.
  - It recursively applies evolutionary operators (concentric scaling, rigid transforms, inscription and circumscription, extension) and constructive augmentations (connect vertices or midpoints, perpendiculars, parallels, diameters). Intersections are computed algebraically.
  - Shaded regions are registered as entities, so a solver can compute their areas exactly.
- **How it makes tasks harder:** deeper recursion and more constructions produce nested topology and multi-hop computation.
- **Correctness / verification:**
  - The SymGT solver computes exact answers with a generalized symbolic shoelace algorithm for regions bounded by lines and arcs.
  - Teacher chains of thought are kept only if SymPy Simplify(A_pred − A_GT) ≡ 0.
  - Expert audit of 1,000 samples: 100% topological validity, 100% ground truth, 98.4% of chains of thought acceptable.
- **Difficulty control:** maximum recursion depth and other grammar settings define Entry, Hard and Expert tiers.

  | Tier | QA pairs | Teacher CoT verified pass rate |
  |---|---|---|
  | Entry | 41,844 | 56.09% |
  | Hard | 60,157 | 39.78% |
  | Expert | 25,363 | 32.73% |

  GeoSym-RL-20K is 10K Entry plus 10K Hard.
- **Reported results:**
  - 127,364 QA pairs over 51,480 images; 55,577 answer-verified chains of thought.
  - Qwen3-VL-8B: MathVerse Vision-Only +22.21; WeMath 61.52 (+6.19).
  - GeoSym-Bench (511 expert-curated items): 18.79 for the 8B model, versus Gemini-3-Pro 15.66, Qwen3-VL-235B 14.68 and Doubao-1.8 11.55.
  - Zero-shot GRPO-Entry on Qwen2.5-VL-7B: overall 39.19 → 42.60, WeMath 27.43 → 34.67.
  - SFT-Entry followed by GRPO-Entry reaches 44.51; SFT-Entry followed by GRPO-**Hard** reaches only 43.59.
  - Best setting: 3–5 SFT epochs and about 100 GRPO steps.
- **Limitations / failure modes:** only computable metric geometry (no proofs); captions and chains of thought still come from teacher MLLMs; harder RL data was not better after SFT.
- **How to reuse with easy seed tasks:** write a typed grammar plus an exact solver for your diagram family and use recursion depth as the difficulty dial. Accept teacher traces only by symbolic equality, and do not assume the hardest tier is the best RL tier.

### TrustGeoGen — TrustGeoGen: Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving (Fu et al., 2025)
[arXiv 2504.15780](https://arxiv.org/abs/2504.15780) · Daocheng Fu, Jianlong Chen, Renqiu Xia, Zijun Chen et al.
- **Mechanism:**
  - A Constructor builds scenes from 46 hand-crafted base scene functions. Each construction's preconditions are checked by a geometric compiler.
  - A Reasoner forward-chains theorems into a reasoning graph, and a Sampler traces back from target statements.
  - GeoExplore variants extract multiple solution paths (-M) and self-reflective traceback paths that pass through wrong statements (-T).
  - Formal steps are translated into natural language with bridging rationales ("connection thinking").
- **How it makes tasks harder:** keep paths with length ≥ τ_l = 5 and premise ratio ≥ τ_r = 0.5. For bootstrapping, the 226 longest-reasoning samples (out of 200K) were used as starting scenes with added premises, giving 376 enhanced scenes and more samples with ≥ 40 steps.
- **Correctness / verification:** every step is a rule application inside the formal graph.
- **Difficulty control:** path-length and premise thresholds. GeoTrust-test has 240 items in 4 tiers of 60.
- **Reported results:**
  - GeoTrust-test: o3 45.83%, Gemini-2.5-Pro 43.33%, GPT-4o 25.83%. Reasoning-trained models degrade more gracefully across tiers.
  - Connection thinking raises Qwen2-VL-7B on GeoTrust-test from 8.75% (plain natural-language translation) to 21.67% (4.58% before training).
  - **Correction to the researcher JSON:** the out-of-distribution gains come mainly from *mixing* GeoTrust-train with GeoQA, not from GeoTrust-train alone.
    - On GeoQA, LLaVA-1.5-13B gains +9.15 with GeoQA only, +4.00 with GeoTrust-train only and +13.80 with both. Qwen2-VL-7B *drops* 8.62 with GeoTrust-train alone and gains +6.50 with the mix.
    - On OlympiadBench-geo (112 items), LLaVA-1.5-13B's +4.46 comes equally from GeoQA alone or the mix.
- **Limitations / failure modes:** coverage is bounded by the theorem set and the 46 scene templates; formal traces need translation; used alone it can hurt models with strong priors.
- **How to reuse with easy seed tasks:** if your domain has a rule engine (geometry, logic, type systems, reaction rules), generate by forward chaining, filter by derivation depth, and seed the next round with the deepest graphs plus new premises. Mix the result with natural-distribution data.

### ChartVerse — ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch (Liu et al., 2026)
[arXiv 2601.13606](https://arxiv.org/abs/2601.13606) · Zheng Liu, Honglin Lin, Chonghan Qin, Xiaoyang Wang et al.
- **Mechanism:**
  - Rollout Posterior Entropy (RPE): Qwen3-VL-2B-Thinking regenerates plotting code 8 times at T = 1.0; the renders are CLIP-embedded, and RPE is the normalized spectral entropy of their Gram matrix. Unstable reconstruction means a complex chart.
  - Charts with RPE ≥ 0.4 from a pool of over 3M images are kept. Claude-4-Sonnet infers code for the hard real charts, which trains a cold-start chart coder.
  - The coder's outputs are filtered to RPE ≥ 0.4 and CLIP similarity ≤ 0.65 to the hard set, giving 200K charts, and the coder is retrained.
- **How it makes tasks harder:**
  - Charts: complexity is selected by reconstruction instability.
  - Questions: a teacher writes a Python analysis script over the chart source, executes it to get A_py, then writes the question *conditioned on* the code and the answer (answer-first).
- **Correctness / verification:**
  - A_py comes from deterministic execution. Samples are kept only when the teacher's solution from the chart matches A_py.
  - Decontamination drops any training chart with CLIP similarity > 0.65 to a benchmark image.
- **Difficulty control:** Qwen3-VL-30B-A3B-Thinking produces 3 chains of thought; fail rate r(Q) is computed, and only 0 < r(Q) < 1 is kept. The highest-r items form RL-40K; the rest go to SFT.
- **Reported results:**
  - ChartVerse-SFT-600K (412K charts, 603K QA) and RL-40K; RL uses GSPO.
  - 6-benchmark average: ChartVerse-8B 64.1 vs. its teacher Qwen3-VL-30B-A3B-Thinking 62.9 (Qwen3-VL-32B-Thinking 67.0); ChartVerse-4B 61.9 vs. Qwen3-VL-8B-Thinking 60.0.
  - SFT→RL: 8B goes 56.9 → 62.5 → 64.1; 2B goes 42.5 → 49.8 → 54.3.
  - Ablation at 100K samples on Qwen3-VL-4B (baseline 53.9): image-space generation 56.5; code-space 56.8; truth-anchored QA 57.4; plus fail-rate filter 57.8. CoSyn data at the same budget degraded the model to 51.0.
  - RPE-selected samples have a 27.6% fail rate, versus 21.1% for VLM-as-judge and 23.5% for code-complexity selection.
- **Limitations / failure modes:** RPE needs many forward passes; the 3-sample fail rate is coarse; the teacher family is used for both QA and checking; only short, deterministic answer types.
- **How to reuse with easy seed tasks:** for any rendered artifact (chart, table image, document, UI), measure difficulty by how unstable a small model's reconstruction is. Generate harder source programs, compute answers by executing code on the source, and write questions last.

### CoSyn *(added)* — Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation (Yang et al., 2025)
[arXiv 2502.14846](https://arxiv.org/abs/2502.14846) · Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta et al.
- **Mechanism:**
  - From a short query ("nutrition fact labels", "book covers"), CoSyn picks one of 20 pipelines built on 11 rendering tools (Python, HTML, LaTeX and others).
  - It samples a persona to diversify the topic, generates the data content, turns it into rendering code and executes it.
  - A text-only LLM then writes QA, including chain-of-thought explanations, **from the code**.
  - Claude-3.5-Sonnet generates data and code; GPT-4o-mini writes the instructions.
- **How it makes tasks harder:** complexity lives in the code: data size, layout, number of elements. Multi-hop CoT questions come from reading the source rather than the image. It also generates pointing and grounding data.
- **Correctness / verification:** answers are grounded in the code and data rather than in pixels, but they are still LLM-written and not executed.
- **Difficulty control:** none explicit; diversity comes from personas and pipelines.
- **Reported results:**
  - 400K images in 9 categories; 2.7M instruction rows.
  - A 7B model averages 80.9 on 7 text-rich benchmarks (Llama 3.2 11B 77.0; GPT-4V 72.8). The zero-shot variant (no academic training sets) averages 74.7.
  - 7K synthetic nutrition-label examples adapted the model to a new domain.
- **Limitations / failure modes:** no difficulty filter or executable answers. At a fixed 100K budget ChartVerse found CoSyn data degraded Qwen3-VL-4B chart reasoning (53.9 → 51.0).
- **How to reuse with easy seed tasks:** use it as the *rendering substrate* and add the missing pieces: answers computed by executing code over the data (ChartVerse-style), and policy fail-rate filtering.

**— Group: Multimodal: hardening existing seeds —**

### SynthRL — SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis (Wu et al., 2025)
[arXiv 2506.02096](https://arxiv.org/abs/2506.02096) · Zijian Wu, Jinjie Ni, Xiangyan Liu, Zichen Liu, Hang Yan, Michael Qizhe Shieh · code: github.com/NUS-TRAIL/SynthRL
- **Mechanism:**
  - Seeds: MMK12 (8,099 items, 8,072 after converting multiple-choice questions to open-ended for reliable checking).
  - The target policy (Qwen2.5-VL-7B-Instruct, which is also the verifier) runs 16 rollouts per seed. Seeds with ≥ 12/16 correct are selected.
  - Gemini-2.5-Flash-Preview-04-17 sees **only the image and question, not the answer**, and is prompted to "transform it into a significantly more challenging version that requires deeper reasoning but maintains the same answer". The image is unchanged.
- **How it makes tasks harder:** the question is rewritten to require more reasoning steps or information derived from the image.
- **Correctness / verification:** a linguistic-quality threshold, then Monte Carlo rollouts on the target policy. The correctness criterion is c_cand ≥ T_min = 4 (the original answer is still reached); the difficulty criterion is c_cand ≤ c_ori − 2.
- **Difficulty control:** the two thresholds above. Accepted items spread across 4–14 passes out of 16, instead of piling up at 0 and 16.
- **Reported results:**
  - 3,380 verified harder variants from 8,072 seeds (A-MMK12 has 11,452 items).
  - 8K setting, seed-only vs. seed plus synthetic: MathVerse 51.6 → 53.5, MathVision 30.0 → 29.6, MathVista 73.9 → 74.2, WeMath 70.6 → 72.6, DynaMath 58.8 → 60.1, average 57.0 → 58.0. Gains are larger on the hardest evaluation items.
- **Limitations / failure modes:** "correctness" means agreement with the policy, not proof, so a rewrite that changed the true answer but still elicits the old one can pass. Aggregate gains are modest, and visual difficulty is untouched.
- **How to reuse with easy seed tasks:** the most direct fix for zero-advantage batches in any domain with reference answers. Select items with a pass rate of at least 75%, rewrite them with the answer hidden, and accept only rewrites whose pass rate falls into a still-solvable band. Consider a second, independent solver to catch rewrites that changed the answer.

### COGS — Composition-Grounded Data Synthesis for Visual Reasoning (Gu et al., 2025)
[arXiv 2510.15040](https://arxiv.org/abs/2510.15040) · Xinyi Gu, Jiayuan Mao, Zhang-Wei Hong, Zhuoran Yu et al.
- **Mechanism:**
  - An MLLM decomposes each seed question into perception and reasoning factors (category plus sub-question).
  - New questions are made by sampling factors from different seeds and grounding new sub-questions on a *new* image, then chaining them.
  - Generation used Qwen2.5-VL-72B; the underlying data tables are used when the chart has metadata.
- **How it makes tasks harder:** multi-factor chains, for example identification → comparison → fact-check or lookup → aggregate → difference.
- **Correctness / verification:** sub-answers are produced first and combined. During RL an LLM checks whether the model's chain of thought produced each correct sub-answer (binary c_i).
- **Difficulty control:** number and type of composed factors; mixing factors across datasets ("factor-level mix").
- **Reward design:** ProcessRM-max r = max(r_final, λ·r_sub) is proven to preserve the ranking of policies by final-answer accuracy under noisy r_sub. ProcessRM-sum is not.
- **Reported results:**
  - Seeds are a random 33% of the ChartQAPro test set; about 10K examples; GRPO on Qwen2.5-VL-7B.
  - ChartQAPro held-out split: 47.36 → 52.02. VisualWebBench: 85.65 → 88.04. Factor-level mixing gives ChartQAPro 52.33 and MMC 87.55.
  - Reward ablation: StandardRM 50.96, ProcessRM-sum 50.35, ProcessRM-max 52.02. SFT followed by ProcessRM-max scores 46.62, below the base.
- **Limitations / failure modes:** sub-answers are generated by an LLM and decompositions are not unique, so the factor signal is noisy. Very small seed sets (1%) help little; single images only.
- **How to reuse with easy seed tasks:** decompose easy seeds into skill factors and recompose 2–4 factors on fresh inputs. Compute sub-answers from metadata or code where possible, and use max-shaped (not summed) process reward.

**— Group: Multimodal: perception proxies and self-play —**

### Jigsaw-R1 (+ Visual Jigsaw) — Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles (Wang et al., 2025)
[arXiv 2505.23590](https://arxiv.org/abs/2505.23590) · Zifu Wang, Junyi Zhu, Bo Tang, Zhiyu Li et al. Related independent work: Visual Jigsaw (Penghao Wu et al., [2509.25190](https://arxiv.org/abs/2509.25190)), by different authors, not a follow-up.
- **Mechanism:**
  - COCO images are split into an m×n grid and shuffled.
  - "Full" questions ask for every patch's position ((mn)! answers; reward is the fraction of positions correct). "Pair" questions ask for the relative position of two patches (binary reward).
  - Visual Jigsaw extends the idea to video (order 6 clips, trimming 5% at clip boundaries to stop frame matching) and 3D (order RGB-D points by depth).
- **How it makes tasks harder:** larger grids; the full rather than the pair format; for video, more clips.
- **Correctness / verification:** the permutation is known by construction. Visual Jigsaw gives 1 for an exact match, γ·(fraction correct) with γ = 0.2 for a valid partial permutation, and 0 for invalid output.
- **Difficulty control:** grid size and format. In Jigsaw-R1 a curriculum mixing sizes (3×1 → 4×1) generalized better than any single size. In Visual Jigsaw, easier settings (2×2 images, fewer clips) gave much smaller gains than 3×3, and at 3×3 the model failed to learn without partial credit.
- **Reported results:**
  - Jigsaw-R1, Qwen2.5-VL-3B, pair puzzles, trained on 2×1 only: 2×1 48.50 → 96.80. The *unseen* larger sizes moved little: 3×1 47.50 → 58.80, 4×1 48.80 → 52.20, 2×2 12.20 → 13.10.
  - Downstream (non-thinking 3B): CV-Bench +3.22, MMVP +4.00, SAT +0.15, Super-CLEVR +7.00, average 69.59 → 73.18.
  - SFT alone averages 69.48 and SFT followed by RL 69.92, versus 73.18 for RL alone. The researcher JSON's "71.04 vs 73.57" were CV-Bench columns, not averages.
- **Limitations / failure modes:** zero-shot extension to much larger grids stays near chance; transfer is mostly to perception benchmarks; SFT cold start hurts.
- **How to reuse with easy seed tasks:** cheap, label-free vision RL. Train *on* the harder grid sizes (with partial credit) rather than hoping for zero-shot extension, mix sizes, and skip SFT for such pretext tasks.

### ViCrit — ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs (Wang et al., 2025)
[arXiv 2506.10128](https://arxiv.org/abs/2506.10128) · Xiyao Wang, Zhengyuan Yang, Chao Feng, Yongyuan Liang et al.
- **Mechanism:** PixMo-Cap human captions average 196 words. GPT-4 picks one object description and perturbs it into a visually similar, plausible, unambiguous hallucination: an object, attribute, count, scene-text or spatial-relation change, often only a couple of words, guided by 2 in-context examples. The model must return the corrupted span.
- **How it makes tasks harder:** an easy description corpus becomes a whole-scene verification task. Grading still reduces to string matching.
- **Correctness / verification:** reward is 0.9·answer + 0.1·format. The answer matches if it equals the injected span; copying extra words from the original caption on either side is not penalized.
- **Difficulty control:** implicit, through edit subtlety, error type and caption length. The benchmark covers 8 hallucination types over 4 image domains.
- **Reported results:**
  - 875K training pairs from 384K valid image–caption pairs.
  - 7B: MathVista 67.8 → 70.7, MMMU 50.6 → 52.0, VLMsAreBlind 49.3 → 52.6, CharXiv-reasoning 41.4 → 47.8, average 50.61 → 53.01.
  - SFT on the same captions (CapSFT) averages 48.41, below the base.
  - 72B: MathVista 74.8 → 77.3, MMMU 63.4 → 66.0, CharXiv 45.5 → 49.4.
  - ViCrit-Bench: o3 47.7%, Gemini-2.5-Pro 45.2%.
- **Limitations / failure modes:** depends on the original caption being correct; one error per caption; spatial hallucinations remain the weakest category.
- **How to reuse with easy seed tasks:** turn any easy, correct artifact (caption, table, chart summary, solution, code) into "find the single injected error". Scale difficulty with length, subtlety and number of errors.

### EVE — EVE: Verifiable Self-Evolution of MLLMs via Executable Visual Transformations (Heng et al., 2026)
[arXiv 2604.18320](https://arxiv.org/abs/2604.18320) · Yongrui Heng, Chaoya Jiang, Han Yang, Shikun Zhang et al.
- **Mechanism:**
  - The Challenger writes `edit_image(img, **args)` plus 4 parameter sets for an image drawn from Vision-SR1-47K. Execution yields 4 edited images.
  - Two question types are built from them: Type-0 (parameter → image) and Type-1 (image → parameter).
  - A priority queue holds the top 50 programs by difficulty reward. It starts from 4 seed programs (jigsaw, rotation, crop, bounding-box drawing), and programs with BLEU similarity above 0.25 are deduplicated.
- **How it makes tasks harder:** the Challenger invents compound transforms (for example rotation plus mirroring) beyond the seeds.
- **Correctness / verification:** answers come from code execution, not pseudo-labels.
- **Difficulty control:** r_diff = 1 − 2|acc_solver − 0.5|, estimated from 6 solver samples. Weights: format 0.2, validity 0.4, difficulty 0.4, diversity 0.3 (BLEU-cluster density).
- **Reported results:**
  - Qwen3-VL-8B 8-benchmark average: 66.48 → 68.43 after 3 iterations → 68.69 at iteration 4 (reported as 68.7) → 68.68 at iteration 5.
  - Baselines as reported: VisPlay-iter3 67.7, MM-Zero-iter3 66.7, Jigsaw-R1-8B 67.5.
  - Largest gains: MMVet +5.1, VisuLogic +3.2, BLINK +2.3. Removing the diversity reward costs 1.23 at iteration 1.
- **Limitations / failure modes:** 2D pixel operations cannot express physical commonsense; gains plateau after 4–5 iterations.
- **How to reuse with easy seed tasks:** let a proposer write executable transforms of your existing images, documents or tables whose answers follow from execution. Reward about 50% solver accuracy plus diversity.

### Vision-Zero — Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play (Wang et al., 2025)
[arXiv 2509.25541](https://arxiv.org/abs/2509.25541) · Qinsi Wang, Bo Liu, Tianyi Zhou, Jing Shi et al.
- **Mechanism:**
  - "Who is the Spy" games over image pairs with subtle differences: CLEVR renders (4–6 objects), 1,000 ChartQA charts and 1,000 ImgEdit real images. Four civilians and one spy.
  - Clue stage: self-play with a zero-sum reward, inversely proportional to the votes a player receives.
  - Decision stage: RLVR (reinforcement learning with verifiable rewards) with +1 for identifying the spy, −0.5 for "n/a", −1 for a wrong vote.
  - Iterative-SPO alternates the two stages.
- **How it makes tasks harder:** as opponents improve, clues become subtler; subtler image differences make harder games.
- **Correctness / verification:** the spy's identity is known, so the decision reward is exact. Clue rewards are relative, not ground truth.
- **Difficulty control:** stage-switch thresholds are accuracy 0.9, error 0.4, n/a 0.5 (up) and n/a 0.1 (down); at least 5 rounds per stage; patience 20.
- **Reported results:**
  - Qwen2.5-VL-7B 6-benchmark average 41.1 → 44.3 (CLEVR), 43.3 (Chart), 44.5 (Real-World).
  - The Chart variant gives LogicVista 47.2 → 51.2 and MathVision 25.4 → 28.9.
  - About 127 A100-hours for the CLEVR variant versus about 700 reported for MM-Eureka-Qwen-7B.
  - Removing role-advantage estimation drops the average to 37.4, below the base.
- **Limitations / failure modes:** depends on the quality of image pairs and on several thresholds; zero-sum clue rewards can be gamed; the method is fragile without its advantage-estimation component.
- **How to reuse with easy seed tasks:** turn image or document sets into asymmetric-information games (spot the difference, odd one out). Alternate self-play with RLVR phases.

### Active-Zero — Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration (He et al., 2026)
[arXiv 2602.11241](https://arxiv.org/abs/2602.11241) · Jinghan He, Junfeng Fang, Feng Xiong, Zijun Yao et al.
- **Mechanism:**
  - Three agents: a Searcher writes queries to retrieve images, a Questioner synthesizes tasks, and a Solver trains with GRPO.
  - The "open world" is a 1.6M-image pool: The Cauldron (50 datasets) plus Geo3K, UniGeo and GeoQA+.
- **How it makes tasks harder:** the Searcher is rewarded by R_chal = E[1 − 2|Acc − 0.5|], computed over questions posed on the retrieved images, which targets the Solver's frontier.
- **Correctness / verification:** **corrected: the paper does specify it.** Labels are "silver-standard" majority votes over K Solver trajectories, so they are pseudo-labels, not verified answers.
- **Difficulty control:** only tasks whose empirical accuracy falls in a window (τ_low, τ_high) are trained on.
- **Reported results:**
  - Qwen2.5-VL-7B reasoning average 51.05 → 53.97 at iteration 2 (+2.92 absolute; the abstract's "5.7%" is relative); iteration 3 gives 53.47.
  - General understanding 57.51 → 59.77 (+2.26; "3.9%" relative).
  - VisPlay scores 52.02 and EvoLMM 51.39.
  - Qwen2.5-VL-3B reasoning 41.34 → 45.33.
- **Limitations / failure modes:** pseudo-label errors can accumulate; iteration 3 already regresses slightly; running three agents is costly.
- **How to reuse with easy seed tasks:** when seed inputs are exhausted, retrieve new inputs targeted at the current frontier. Pair this with executable answers (EVE-style) wherever possible instead of majority votes.

**— Group: Science, chemistry, optimization —**

### Sim2Reason — Solving Physics Olympiad via Reinforcement Learning on Physics Simulators (Prabhudesai et al., 2026)
[arXiv 2604.11805](https://arxiv.org/abs/2604.11805) · Mihir Prabhudesai, Aryan Satpathy, Yangmin Li, Zheyang Qin et al.
- **Mechanism:**
  - A YAML DSL composes random scenes from reusable entities and compiles them to MuJoCo. Entities include pulleys, movable prisms, collision planes, rolling bodies with cutouts, solar systems, variable-mass rockets, and charged particles in time-varying E/B fields.
  - The DSL separates physically meaningful axes (masses) from irrelevant ones (string length).
  - Traces are recorded, then turned into three question types:
    - numeric/forward: "v at t = 3 s?"
    - reverse: mask one parameter, "which mass gives 5 m/s at 3 s?"
    - symbolic: "v(t) = ?"
- **How it makes tasks harder:** more entities and connections; inverse and symbolic formats.
- **Correctness / verification:** reward when within 5% relative error of the simulator value. Unstable segments are pruned by an acceleration-spike rule (k = 5). Scene-ablation shortcut detection (the answer is unchanged when components are removed) drops about 15% of QA pairs. Items are deduplicated.
- **Difficulty control:** entity and connection count, parameter ranges and question type. The authors note that simulator questions tend to be degenerate (trivial or intractable) and prune both extremes.
- **Reported results:**
  - About 6,400 distinct QA pairs seen (200 GSPO steps × batch 32).
  - IPhO mechanics, zero-shot: Qwen2.5-32B 19.8 → 25.2; 14B 16.07 → 20.45; 7B 10.7 → 15.1; 3B 5.68 → 13.15; Qwen3-30B (100 steps) 35.6 → 40.0.
  - Qwen2.5-32B: JEEBench 34.38 → 52.28; PHYSICS 39.42 → 43.09; OlympiadBench 41.41 → 44.53.
  - SFT on 200K rejection-sampled solutions from GPT-4, o3 and o4-mini *lowered* IPhO from 19.8 to 15.9, while RL reached 25.2.
  - Ablations at 3B: training on numeric questions gives 13.15, versus reverse 5.84 and symbolic 7.46. Without the shortcut filter: 7.14 vs 13.15.
- **Limitations / failure modes:**
  - Classical mechanics and basic electromagnetism only.
  - The 5% tolerance could reward approximations.
  - **Correction:** "DSL 100% vs raw XML 33% success" is a 3-case pilot (1/3 vs 3/3).
  - The inverse and symbolic formats, though harder, transferred worse than forward questions in their ablation.
- **How to reuse with easy seed tasks:** wrap any simulator or solver (circuits, kinetics, cash flows, queues) in an LLM-writable DSL. Generate forward questions and filter shortcuts by ablating components. Treat inversion and symbolic forms as extra variety to be validated, not as automatically better signal.

### ether0 — Training a Scientific Reasoning Model for Chemistry (Narayanan et al., 2025)
[arXiv 2506.17238](https://arxiv.org/abs/2506.17238) · Siddharth M. Narayanan, James D. Braza, Ryan-Rhys Griffiths, Albert Bou et al. (FutureHouse)
- **Mechanism:**
  - 640,730 problems in 18 task categories and 375 subtasks, templated from experimental data only (ChEMBL, COCONUT, PubChem, Open Reaction Database and others; 81 templates in total).
  - Tasks include IUPAC/formula → structure, elucidation, functional groups, solubility editing under similarity constraints, reaction prediction, retrosynthesis from purchasable reagents, and multiple-choice property tasks.
  - The base model is Mistral-Small-24B. Specialists are trained per task cluster with GRPO, then 186,010 specialist sequences are distilled into one generalist, followed by further RL.
- **How it makes tasks harder:**
  - Constraint stacking: "raise log S while staying similar to the input".
  - Open-answer design instead of multiple choice.
  - Paraphrase and "distractor-inserting" rewrites by Gemini 2.5 Flash during RL.
  - Multiple-choice distractors chosen by Tanimoto and property similarity.
- **Correctness / verification:**
  - RDKit checks: validity, formula, groups, exact product.
  - KDESol for solubility; Molecular Transformer for forward-predicting the product of proposed retrosynthesis reactants.
  - Purchasability via catalogue Bloom filters.
  - Under-determined answers pass at ECFP4 Tanimoto ≥ 0.7.
  - A "reasonable molecule" check (rings and groups must occur in the source data) was essential: without it the policy inserted peroxides to satisfy oxygen counts or hydrazines to raise solubility.
- **Difficulty control:** groups where every rollout gets the same reward reached 90% of batches. A curriculum buffer adds any problem that produced a mixed-reward group, samples a fraction ε_cur of each batch from it, removes items once they become trivial, and is seeded from earlier runs. Specialists proved more robust than scheduling.
- **Reported results:** retrosynthesis 70% after 46,000 examples, versus 64.1% for Molecular Transformer trained on about 480K USPTO reactions (retrained on their 60K it stays below 30%). The paper reports beating frontier LLMs, specialists and human experts, especially on open-answer tasks.
- **Limitations / failure modes:** oracle rewards can be gamed; templated phrasing needs paraphrasing; the ML oracles (solubility, forward prediction) have their own error.
- **How to reuse with easy seed tasks:** convert experimental databases into constraint-satisfaction design tasks verified with domain toolkits. Add plausibility guards from the source data distribution, keep a buffer of non-trivial items, and paraphrase during RL.

### AutoOR — AutoOR: Scalably Post-training LLMs to Autoformulate Operations Research Problems (Motwani et al., 2026)
[arXiv 2604.16804](https://arxiv.org/abs/2604.16804) · Sumeet Ramesh Motwani, Chuan Du, Aleksander Petrov, Christopher Davis et al.
- **Mechanism:**
  - A standard-form template is instantiated into a world descriptor: objective, constraints, solver code (OR-Tools for LP/MILP, Gekko for NLP), the solution x* and metadata.
  - An LLM back-translates the descriptor into a natural-language problem (templates → world → description).
  - Gemini 2.5 Pro verifies each description with 5 component-wise checks: data values, all constraints, objective, governing equations and parameters, self-consistency. Failures are regenerated.
- **How it makes tasks harder:**
  - LP: 5 templates scaling decision variables from 4 to 12.
  - MILP: 9 families derived from one seed form.
  - NLP: pump-network variants (series–parallel levels, coefficients, costs, constraints) grown from one production seed problem.
  - A multi-turn variant requires asking clarifying questions.
- **Correctness / verification:**
  - Optimum from the solver. LP/MILP reward requires the objective to match to 2 decimals (LP also checks decision variables); NLP uses 2% relative tolerance on cost.
  - About 50% code acceptance means 7–8% of initial generations become training items. An audit of 50 accepted pairs found no errors.
- **Difficulty control:** the base model scored 0% on the NLP class even at pass@64. A curriculum fixes this: phase 1 puts Gekko syntax and mapped constraints in the prompt on easy problems; phase 2 removes them; phase 3 trains on hard problems. Each phase must reach nonzero solvability before the next.
- **Reported results:**
  - AutoOR-8B (Qwen3-8B, Dr. GRPO): NL4LP 97.22 (Gemini 3 Pro 91.67); MAMO-Complex 83 (82); Hard-MIP 94 (92); Pump-NLP 48.98 vs. about 0 for Gemini 3 Pro and Qwen3-8B.
  - SFT on the same synthetic data scores Hard-LP 26 (base 55, RL 80) and MAMO-Complex 74 (RL 83).
- **Limitations / failure modes:** low yield; tied to specific solver APIs; tolerance must allow for alternative local optima; the reward is coarse.
- **How to reuse with easy seed tasks:** generate the formal artifact (model, SQL, program) first, compute the answer by execution, back-translate, and verify each component. For classes with zero pass rate, put the scaffold in the prompt and remove it in stages.

### MegaScience / TextbookReasoning — MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning (Fan et al., 2025)
[arXiv 2507.16812](https://arxiv.org/abs/2507.16812) · Run-Ze Fan, Zengzhi Wang, Pengfei Liu
- **Mechanism:**
  - 12.8K university-level textbooks were OCR'd with olmOCR and split into 4,096-token chunks.
  - Llama3.3-70B-Instruct extracts QA pairs under two criteria. The high standard requires multi-step reasoning with a full solution in the source; the low standard requires only a complete question and answer.
  - DeepSeek-V3 refines items to be self-contained and adds explanations. Defective items (external references, contradictions, missing information) are filtered out.
- **How it makes tasks harder:** it *harvests* harder items (high-standard extraction) rather than transforming them.
- **Correctness / verification:** reference answers come from textbooks. MinHash deduplication at 0.6. LLM decontamination against benchmarks (top-k similar test items, then a paraphrase judgment).
- **Difficulty control:**
  - Difficulty selection: 16 Qwen2.5-7B-Instruct responses are scored 0–10 by Qwen2.5-32B-Instruct against the reference. Items averaging > 9 (too easy) or < 1 (likely noisy) are dropped.
  - **Correction:** this selection helped only Nemotron-Science (57.40 vs 56.39 by length and 56.04 random). For NaturalReasoning and TextbookReasoning no selection method beat the full set (57.44 and 58.33).
- **Reported results:**
  - 945,683 extracted → 767,893 after deduplication → 728,767 after filtering → 651,840 after decontamination, with an average response of 410 tokens.
  - MegaScience totals about 1.26M items.
  - **Correction:** Qwen2.5-7B *base* trained on MegaScience scores 61.01, versus 58.80 for the *official Qwen2.5-7B-Instruct*.
  - Gains grow with model size: at 1.5B and 3B the trained models score slightly below the official instruct models (41.19 vs 44.18; 51.35 vs 51.50).
- **Limitations / failure modes:** free-form answers need a judge for RL; small models do not benefit.
- **How to reuse with easy seed tasks:** mine course material with a strict "multi-step with full solution" extraction prompt, rewrite items to be self-contained, keep the source answer, then convert to verifiable formats (see Golden Goose and HuatuoGPT-o1).

### NaturalReasoning *(added)* — NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions (Yuan et al., 2025)
[arXiv 2502.13124](https://arxiv.org/abs/2502.13124) · Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe et al. · data: huggingface.co/datasets/facebook/natural_reasoning
- **Mechanism:**
  - Documents from DCLM-baseline and FineMath are scored by an LLM on several axes, including problem complexity and technical depth, and thinking and reasoning.
  - For documents scoring full marks, Llama-3.3-70B-Instruct *composes* a novel, self-contained, challenging question rather than extracting one. It then checks whether a reference answer can be derived from the document.
  - A Llama-3-70B-Instruct response is attached as a teacher signal.
- **How it makes tasks harder:** questions are synthesized beyond what appears verbatim in the text and are selected from reasoning-dense sources.
- **Correctness / verification:** 81.68% of the 2.8M questions carry a document-derived reference answer. Semantic deduplication plus 13-gram decontamination removed 0.026% of items.
- **Difficulty control:** implicit. Median Llama3.3-70B response length is used as a proxy; at 434 words it is the longest of the compared datasets.
- **Reported results:** 2.8M questions across STEM, economics and social sciences. The paper shows it works for distillation and for self-training with an external reward model or self-rewarding. MegaScience later reuses it.
- **Limitations / failure modes:** about 18% have no reference answer; free-form answers need model-based judging; no policy-calibrated difficulty.
- **How to reuse with easy seed tasks:** a science- and domain-agnostic way to get fresh, harder questions from raw corpora. Follow it with policy pass-rate filtering, or with a Golden Goose mask-and-choose conversion for exact rewards.

### Golden Goose — Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text (Lu et al., 2026)
[arXiv 2601.22975](https://arxiv.org/abs/2601.22975) · Ximing Lu, David Acuna, Jaehun Jung, Jian Hu et al.
- **Mechanism:**
  - GPT-5 finds a contiguous span of key reasoning steps in a source passage, replaces it with [MASK], and writes diverse, plausible, style-matched wrong options.
  - Sources: AoPS-Instruct, rStar-Coder (code without tests), MegaScience, and FineWeb for cybersecurity.
- **How it makes tasks harder:** the model must reconstruct the missing reasoning; more distractors make elimination less effective.
- **Correctness / verification:** the masked text is the answer; reward is exact option match. Reasoning-dense sources needed no post-processing. On noisy FineWeb, items the student solved in all 16 rollouts were removed.
- **Difficulty control:**
  - Option count: with 3 options most items were too easy (the policy eliminated options). With 9 options more than 70% fell in the medium band.
  - An open-ended variant failed: more than 83% of GooseReason-Math items got zero accuracy consistently for ProRL-1.5B-v2.
- **Reported results:**
  - GooseReason-0.7M. About 70% of items give useful signal, versus a small share of ProRL's data (13× more effective examples).
  - ProRL-1.5B-v2 continued RL over about 1,100 H100-hours: math +2.71 vs +0.63, code +2.12 vs +0.95, STEM +3.48 vs +0.13 (GooseReason vs. original data).
  - Qwen-4B-Instruct: original data after 300 steps gives −1.29, +0.43 and −1.52; GooseReason gives +2.18, +2.24 and +2.40.
  - GooseReason-Cyber (180K): +4.44 across 3 cybersecurity benchmarks after 100 steps.
- **Limitations / failure modes:** multiple choice allows elimination strategies; quality depends on the distractor generator; there is no human audit of distractor plausibility.
- **How to reuse with easy seed tasks:** when RL has plateaued, convert reasoning-dense corpora in new domains into mask-and-choose items and tune the option count until most items are medium difficulty.

**— Group: Text-to-SQL and tables —**

### OmniSQL / SynSQL-2.5M — OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale (Li et al., 2025)
[arXiv 2503.02240](https://arxiv.org/abs/2503.02240) · Haoyang Li, Shang Wu, Xiaokang Zhang, Xinmei Huang et al.
- **Mechanism:**
  - Web tables (19,935 kept) → LLM infers a business scenario → generates a relational database with K ~ N(10, 4²) tables.
  - An enhancement pass fixes simplistic first drafts (about 4 columns per table) by adding columns and completing primary and foreign keys.
  - SQL is generated conditioned on a sampled tier (Simple, Moderate, Complex, Highly complex; each with criteria and an example), on sampled database values and on a column count drawn from Geometric(p = 0.6).
  - Questions are back-translated in 9 styles (formal, colloquial, imperative, interrogative, descriptive, concise, vague, metaphorical, conversational). A selector picks the candidate question most semantically consistent with the others (all-mpnet embeddings).
  - Chains of thought are generated per item.
- **How it makes tasks harder:** complexity tiers, wider schemas, more joins and advanced features.
- **Correctness / verification:** SQL is executed to drop syntax errors and timeouts. For each item several CoT solutions are grouped by execution result and chosen by majority vote. Human audit of 1,000 samples: 97% of questions meaningful, 89% of SQL appropriate, 86% fully correct.
- **Difficulty control:** tiers are set only by the prompt, with no enforcement.
- **Reported results:**
  - 2,544,390 samples over 16,583 databases; more than 2M unique SQL skeletons; 83 functions.
  - 1.75 joins per query (BIRD 0.94, Spider 0.48); about 40% use CTEs.
  - OmniSQL-7B: Spider test 87.9 (greedy), BIRD dev 63.9.
  - SynSQL alone: +9.0 on BIRD dev and +12.9 on EHRSQL over the base.
- **Limitations / failure modes:** tiers are not guaranteed; about 14% of samples are imperfect; synthetic databases are clean.
- **How to reuse with easy seed tasks:** start from the executable object, condition on explicit complexity tiers, back-translate questions in varied styles, and vote over execution results.

### EvolSQL — EvolSQL: Structure-Aware Evolution for Scalable Text-to-SQL Data Synthesis (Pan et al., 2026)
[arXiv 2601.04875](https://arxiv.org/abs/2601.04875) · Xuanguang Pan, Chongyang Tao, Jiayuan Bai, Jianling Gao et al.
- **Mechanism:**
  - Exploratory Query-SQL Expansion (EQE) broadens intents and covers schema elements the seeds leave unused.
  - Operator-guided evolution (OGE) applies 6 atomic AST operators: functional wrapping, operator mutation, logical clause expansion, JOIN expansion, nesting (a value node becomes a subquery), and set composition.
  - Each operator is scored by feasibility (a strategy model rates how applicable it is to the current SQL and schema) and scarcity (W_div = P_target / (P_accum + ε)). The top-K operators are applied; the paper reports two OGE rounds.
- **How it makes tasks harder:** more relations, predicates, aggregations and nesting.
- **Correctness / verification:** candidate SQL is executed; a refinement model fixes it using execution feedback. Items are kept only if the SQL runs and returns **non-empty** results. Schema-aware deduplication removes items above cosine 0.9 (all-mpnet); chains of thought come from rejection sampling (n = 4).
- **Difficulty control:** operator choice and number of rounds. CTE frequency rises from 0.02 to 0.58 across rounds; average JOINs are 130% above BIRD.
- **Reported results:**
  - 129,268 synthesized items (about 140K training corpus including BIRD and Spider training data).
  - EvolSQL-Qwen-7B: BIRD dev 65.1, Spider test 86.1, versus OmniSQL-7B's 63.9 BIRD with about 1/18 of the data.
  - Ablations: without operators 64.5; without OGE 62.7; without OGE and EQE 57.4.
- **Limitations / failure modes:** a non-empty result does not prove question and SQL mean the same thing; evaluated with SFT only.
- **How to reuse with easy seed tasks:** for any code-like target (SQL, Pandas, regex, spreadsheet formulas, DSLs), define atomic AST mutations and pick operators that are both feasible and under-represented. Require non-degenerate execution, then back-translate the question.

### Arctic-Text2SQL-R1 *(added)* — Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL (Yao et al., 2025)
[arXiv 2505.20315](https://arxiv.org/abs/2505.20315) · Zhewei Yao, Guoheng Sun, Lukasz Borchmann, Gaurav Nuti et al. (Snowflake)
- **Mechanism:** GRPO with a reward based only on execution correctness (plus syntax), trained on filtered BIRD and Spider plus synthetic Gretel-Synth data. Batch 256 × 16 rollouts, KL 0.001; online RL starting from strong SFT checkpoints (OmniSQL-32B).
- **How it makes tasks harder:**
  - Gretel-Synth ships schemas without data. GPT-4o writes INSERT statements, resampled until the gold SQL returns rows.
  - **Distractor tables from related domains are added to raise schema complexity.**
  - Only SQL longer than 160 characters is kept.
- **Correctness / verification:** gold SQL that returns empty results or runs longer than 5 s is removed (about 1,400 BIRD and 1,700 Spider items), because empty or slow gold SQL gives spurious rewards.
- **Difficulty control:** model-based filter: a query is kept only if the best trained model solves it at least once in 10 samples at T = 1.0.
- **Reported results:**
  - 14B on BIRD-dev: 64.9 with BIRD+Spider only; 64.6 adding unfiltered Gretel; 66.5 adding filtered Gretel.
  - LLM "paraphrase or generate complex question" augmentation mirrored the originals and hurt: 62.5 unfiltered and 64.9 filtered, versus 64.9 without it.
  - Filtering SynSQL-2.5M was inconclusive.
  - BIRD test: 71.83% for the 32B model; the 14B model is also above 70%.
- **Limitations / failure modes:** filtering by solvability removes unsolvable-but-correct hard items; the gains from synthetic data are modest; the augmentation failure shows that naive "make it complex" rewriting is not enough.
- **How to reuse with easy seed tasks:** harden SQL by enlarging the *context* (distractor tables, wider schemas), not only the query. Always drop degenerate gold answers, and filter for "solvable at least once".

### TableDreamer — TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning (Zheng et al., 2025)
[arXiv 2506.08646](https://arxiv.org/abs/2506.08646) · Mingyu Zheng, Zhifan Feng, Jia Wang, Lanrui Wang et al.
- **Mechanism:**
  - Tables are synthesized from scratch (flat, horizontal and hierarchical; size, headers, format) with seed instructions for 20 table tasks.
  - Seeds evolve in three directions:
    - instruction complication (7 strategies, including added reasoning steps, sub-tasks, constraints, depth and context)
    - instruction generalization
    - table generalization or complication (format, headers, row and column edits)
- **How it makes tasks harder:** more complex instructions and messier or larger tables; each round is seeded only with items the target model failed.
- **Correctness / verification:** an LLM judge rates the target model's answer against a GPT-4o reference on a 5-point scale (structured answers are exact-matched from JSON). Correctness therefore rests on GPT-4o.
- **Difficulty control:** "weakness" means a judge score below 3; 2 rounds.
- **Reported results:**
  - 3,272 seeds over 1,541 tables → 27,083 samples over 7,950 tables.
  - Llama3.1-8B-Instruct average over 10 benchmarks: 49.07 → 60.69.
  - Without weakness filtering (34K samples): 56.28. With Llama3.1-70B as the synthesizer: 56.02.
- **Limitations / failure modes:** references are not executable, so hard items may carry wrong labels; unsuitable for exact RL rewards as is.
- **How to reuse with easy seed tasks:** evolve both the artifact (the table) and the instruction, and keep only failures. For RL, replace the GPT-4o reference with answers computed by SQL, Pandas or formula execution over the synthetic table.

**— Group: Medicine and non-verifiable domains —**

### HuatuoGPT-o1 — HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs (Chen et al., 2024)
[arXiv 2412.18925](https://arxiv.org/abs/2412.18925) · Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang et al.
- **Mechanism:**
  - Start from 192K MedQA-USMLE and MedMCQA exam questions.
  - Remove items that all three of Gemma2-9B, LLaMA-3.1-8B and Qwen2.5-7B answer correctly, and remove short items.
  - Remove items with non-unique answers or that ask for the incorrect option (GPT-4o-assisted).
  - Rewrite the rest as open-ended problems with a unique ground truth.
  - A strategy search (explore new paths, backtrack, verify, correct) builds long chains of thought for SFT; PPO follows.
- **How it makes tasks harder:** small-model difficulty filtering plus removing the answer options (format hardening).
- **Correctness / verification:** GPT-4o acts as an alias-aware verifier. On 200 manually checked cases it was 96.5% and 94.5% accurate (stages 1 and 2), versus 70.5% and 74.5% for regex exact match. A LLaMA-3.1-8B verifier was also fine-tuned on 20K scoring examples.
- **Difficulty control:** the filters above.
- **Reward design:** RL reward is 1 correct / 0.1 incorrect / 0 null; KL β = 0.03.
- **Reported results:**
  - 40K verifiable problems (20K SFT, 20K RL).
  - 8B model: +8.5 over LLaMA-3.1-8B-Instruct (55.4 → 63.9 average).
  - RL gain: +3.6 with complex CoT, +2.6 with simple CoT and +1.1 with no CoT. PPO beat DPO and RLOO.
- **Limitations / failure modes:** the verifier is an LLM (about 5% error); removing options can create ambiguity if the uniqueness filter misses items.
- **How to reuse with easy seed tasks:** for saturated multiple-choice banks (law, finance, medicine), drop items that small models solve, filter for answer uniqueness, remove the options, and grade with an alias-aware judge that has been audited on about 200 samples.

### Baichuan-M2 — Baichuan-M2: Scaling Medical Capability with Large Verifier System (Baichuan-M2 Team / Dou et al., 2025)
[arXiv 2509.02208](https://arxiv.org/abs/2509.02208)
- **Mechanism:**
  - A Patient Simulator is built from de-identified records.
    - Medical layer: profile.
    - Psychological layer: MBTI-inspired behaviour; for example, extroverts ask about treatments proactively while introverts accept information passively.
    - Three modules: a Termination Gate, an Affective Unit and a Factual Unit, which prevent early endings, drift and leakage.
  - A Clinical Rubrics Generator writes weighted, case-specific criteria.
  - Multi-stage RL: rule-based tasks, then rubric-based tasks, then multi-turn dialogue with the simulator using fragment-level sampling.
  - The model is trained from Qwen2.5-32B-Base.
- **How it makes tasks harder:** static QA becomes interactive consultation where facts must be elicited from a patient with a persona.
- **Correctness / verification:** rubric-based judging. Generated rubrics matched expert-selected ones at 92.7% on 100 cases (GPT-4.1 referee). Rule-based stages use only tasks with unique answers.
- **Difficulty control:** interactivity, persona and staging. A length reward applies only when the group's 80th-percentile rubric score exceeds a threshold *and* the response itself is in the top 80th percentile.
- **Reported results:** HealthBench Hard 34.7 (GPT-5 46.2, as reported by the paper).
- **Limitations / failure modes:** simulator realism; reward leakage in multi-turn rollouts; judge and rubric bias; cost.
- **How to reuse with easy seed tasks:** hide the facts of a saturated single-turn item inside a simulated user who reveals them only when asked, and grade with case-specific rubrics (legal intake, financial advice, support).

### LLM-as-a-Tutor — LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL (Kim et al., 2026)
[arXiv 2607.04412](https://arxiv.org/abs/2607.04412) · Yujin Kim, Namgyu Ho, Sangmin Hwang, Joonkee Kim et al.
- **Mechanism:** the tutor pairwise-compares two rollouts. If they are indistinguishable in quality, the prompt is "non-challenging": the tutor appends one atomic constraint on an unspecified dimension (x̃ = x ⊕ c) and adds a matching rubric criterion (R̃ = R ∪ R_c, weights renormalized).
- **How it makes tasks harder:** constraints are only ever appended, so difficulty rises monotonically with policy capability.
- **Correctness / verification:** a rubric-based LLM judge scores each criterion; the base rubric stays. There is no explicit check for contradictory constraints.
- **Difficulty control:**
  - Policy-driven. With an 8B tutor, the share of augmented prompts grows with policy size: 8.1% at 0.6B, 25.8% at 1.7B, 40.5% at 4B.
  - Flagged prompts have mean reward 90.76 (SD 12.96) versus 78.24 (SD 27.07) for the rest.
- **Reported results:**
  - Qwen3-1.7B-Thinking policy; Qwen3-8B-Thinking as tutor and judge; 4K WildChat prompts; 3 epochs.
  - Average over FollowBench, AdvancedIF and InfoBench: 51.96, versus 51.04 for policy-adaptive rubrics, 51.04 for EVA and 50.51 for base rubrics.
  - **Evol-Instruct rewriting that ignores the policy scores 50.24, below base rubrics.**
  - Appending beats rewriting; pairwise detection beats variance-based detection.
- **Limitations / failure modes:** extra tutor calls; one main policy–tutor pair; constraints can be unnatural or contradictory.
- **How to reuse with easy seed tasks:** when rubric-graded prompts stop producing reward variance, append one checkable constraint and its rubric item. Do not regenerate the prompt wholesale.

### QUBRIC — QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards (Zhang et al., 2026)
[arXiv 2606.03968](https://arxiv.org/abs/2606.03968) · Rongzhi Zhang, Rui Feng, Zhihan Zhang, Jingfeng Yang et al.
- **Mechanism:** teachers write reference responses, and key points (atomic knowledge units) are extracted where teachers disagree. Vague queries are rewritten into scenarios in which 1–2 key points must be *reasoned to*. For example, "how much testosterone is safe in men" becomes a sports-medicine consultant reviewing an Olympic powerlifter with morning testosterone of 1,450 ng/dL.
- **How it makes tasks harder:** scenario grounding plus contrastive rubrics built from what the teacher answer covers and the policy answer misses.
- **Correctness / verification:** "constitutive" rubrics write the grading knowledge into the criteria. Grounding in key points prevents references to guidelines that do not exist, which make every response fail (for example "Endocrine Society 2018 says 916 ng/dL" → all zeros).
- **Difficulty control:** keep only query–rubric pairs where the initial policy passes 20–50%. A band sweep shows that having *a* filter matters more than the exact band.
- **Reported results:**
  - ArenaHard hard prompts 71.0 → 76.5; creative writing 51.6 → 58.9; held-out MoReBench, PLawBench and MuSR +6.31 on average (PLawBench +8.85); IFEval −1.7.
  - A naive narrowing rewrite gave 0/6 discriminative rubrics, versus 10/13 for the co-designed rewrite.
  - Ablations on ArenaHard-hard:
    - surface rewrite only: 67.1
    - scenario rewrite without key points: 73.6
    - no learnability filter: 69.7
    - non-contrastive rubrics: 71.1
    - rubric RL with original queries: 68.6
    - AutoIF verifiable-constraint RLVR: 63.8
- **Limitations / failure modes:** needs strong teachers; changes the query distribution; the band must be recalibrated as the policy improves.
- **How to reuse with easy seed tasks:** add concrete scenario context anchored on key points present in teacher answers, generate rubrics from teacher–policy gaps, and train only inside a pass band.

### EvoRubrics *(added)* — EvoRubrics: Dynamic Rubrics as Rewards via Adversarial Co-Evolution for LLM Reinforcement Learning (Ding et al., 2026)
[arXiv 2606.23038](https://arxiv.org/abs/2606.23038) · Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao et al.
- **Mechanism:** a Policy LLM and a Rubric Generator share one backbone through two LoRA adapters and are both trained with GRPO *within each training step*. The Rubric Generator is rewarded for discriminativeness, semantic diversity, alignment with human preferences and constructiveness. A fully self-supervised variant also exists.
- **How it makes tasks harder:** as the policy improves, the generator writes finer-grained criteria that expose remaining gaps, which acts as an automatic curriculum over the grading standard.
- **Correctness / verification:** rubric-aggregated LLM judging, with no ground truth.
- **Difficulty control:** implicit, through the adversarial co-training.
- **Reported results:**
  - Qwen3-4B HealthBench score: 12.62 (vanilla) → 15.82 (static "GoldenRubrics") → 20.80 (EvoRubrics).
  - Static golden rubrics *hurt* out-of-distribution FollowBench HSR (47.76 → 32.50), while EvoRubrics reaches 53.92.
  - Qwen3-8B HealthBench: 14.40 → 20.47 (golden) → 22.97.
  - Removing any rubric-reward component lowers results.
- **Limitations / failure modes:** judge bias is still inherited; adversarial co-training adds cost and instability risk.
- **How to reuse with easy seed tasks:** if rubric RL has saturated, make the rubric generator a trained, policy-tracking component instead of a fixed file.

### RubricHub *(added)* — RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation (Li et al., 2026)
[arXiv 2601.08430](https://arxiv.org/abs/2601.08430) · Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren et al.
- **Mechanism:** three stages:
  1. Response-grounded, principle-guided candidate criteria.
  2. Multi-model aggregation.
  3. **Difficulty evolution:** take a pair of reference responses with high consensus scores and extract nuances that separate "excellent" from "exceptional" as additive criteria R_add; R_final = R_base ∪ R_add.
- **How it makes tasks harder:** generic checks become strict ones; the paper's example turns "Is the code correct?" into "Does it handle the edge case in O(n)?".
- **Correctness / verification:** rubric scores from an LLM grader.
- **Difficulty control:** criteria are added until top responses no longer saturate.
- **Reported results:** about 110K rubrics. Qwen3-14B-Base after rubric-based rejection-sampling fine-tuning (RuFT) and rubric-based RL (RuRL) scores 69.3 on HealthBench, versus 67.2 reported for GPT-5 (high), and 22.6 points above Qwen3-14B-Instruct (non-thinking). RubricHub is the training set used in the Rubric Dropout study.
- **Limitations / failure modes:** harder rubrics are also easier to game (see Rubric Dropout and ImpossibleRubrics under Insights).
- **How to reuse with easy seed tasks:** the rubric-side analogue of task hardening. When all rollouts pass the rubric, mine the difference between the two best rollouts and add it as new criteria.

### Writing-Zero — Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards (Jia et al., 2025)
[arXiv 2506.00103](https://arxiv.org/abs/2506.00103) · Ruipeng Jia, Yunyi Yang, Yongbo Gai, Kai Luo et al.
- **Mechanism:**
  - A pairwise writing GenRM writes principles specific to the query, critiques two responses and scores each 0–10.
  - BRPO (Bootstrapped Relative Policy Optimization) picks one rollout of the group at random as the reference. Every other rollout gets +1 if it beats the reference under all voting conditions and −1 otherwise, used directly as the advantage.
- **How it makes tasks harder:** the bar is the policy's own output, so it rises as the policy improves (no prompt changes).
- **Correctness / verification:** no ground truth; format and score-margin checks on the GenRM output.
- **Difficulty control:** drop queries where the sum of preference scores exceeds 9 of 16 (|ΣR|/G > 0.6): too easy or too hard to discriminate.
- **Reported results:**
  - WritingBench: Qwen3-32B-Base 6.89 → Writing-Zero 8.29; Writing-R1 (SFT then BRPO) 8.68; Qwen3-32B-Instruct 8.64.
  - **Missing from the JSON:** scalar-RM GRPO scored *higher* on WritingBench (8.87) but lower on the authors' writing test set (2.83 vs 3.84). The paper reads this as the benchmark judge rewarding hacked outputs.
  - Hacking indicators, scalar-RM GRPO vs Writing-Zero: mean response length 1,872 vs 1,292; redundant self-explanation 417 vs 58.
  - The pairwise GenRM is *less* accurate on static preference sets (57.5% vs 64.7% for the scalar RM on Cultural & Creative Writing) yet more robust in RL.
- **Limitations / failure modes:** inherits GenRM biases; modest judge accuracy; benchmark judges can disagree.
- **How to reuse with easy seed tasks:** for saturated open-ended prompts, switch to pairwise comparison against an in-group or earlier-checkpoint reference, and combine with prompt hardening (LLM-as-a-Tutor, QUBRIC).

## Complexification operators from this area

1. **Answer-preserving stem hardening**
   - *What it does:* a stronger model rewrites the question, without seeing the answer, so it needs more reasoning. The image and answer stay fixed.
   - *Easy → hard:* "Area of the shaded triangle?", where base and height are labelled, becomes "…given only the perimeter and one angle marked in the figure", so the model must derive the base and height first.
   - *Keep it verifiable:* accept only if the target policy still reaches the original answer in ≥ 4/16 rollouts and its pass count drops by ≥ 2. Optionally have an independent solver re-solve the item.
   - *Sources:* SynthRL.

2. **Procedural level scaling that changes the mechanism**
   - *What it does:* a generator with a level knob that adds deduction steps, hops, series, entities or grid cells, run online.
   - *Easy → hard:* one-step triangle sum becomes a four-step angle chase over parallels (TRON). A 1-series bar chart becomes stacked multi-series aggregation over many time points. A small Sokoban board becomes a large board with multi-step state prediction (Game-RL).
   - *Keep it verifiable:* compute the answer from the latent state before rendering. Show the base-model pass rate falls per level (TRON 72.8 → 41.3%). Promote at about 0.8 accuracy and replay lower levels (p = 0.3).
   - *Sources:* TRON, Trace, Game-RL, GeoSym127K, SPHINX (2511.20814).

3. **Atomic constraint stacking**
   - *What it does:* add independently checkable requirements one at a time: counts, relations, properties, formats, rubric items.
   - *Easy → hard:* "Draw 3 red circles" becomes "3 red circles left of 2 blue squares, twice as many circles as squares in the top half, no object touching the border". "Edit this molecule to raise solubility" becomes the same while keeping Tanimoto ≥ 0.7 to the input and only reasonable groups. "Write a memo" becomes "…and name the policy clause, ≤ 200 words".
   - *Keep it verifiable:* one checker per constraint. Read constraints off a realized solution (VVR) so the set is satisfiable. Reward partial credit × all-satisfied guard. Screen for contradictions.
   - *Sources:* VVR, LLM-as-a-Tutor, ether0, MIFS (2609.16059: generative constraint protocol + code verifiers, 90K samples, 8 constraint categories), VISA (2608.26013: agentic constraint discovery with target-model difficulty probing).

4. **Factor decomposition and recomposition (multi-hop)**
   - *What it does:* split seeds into primitive factors, or knowledge-graph (KG) edges, and chain several on a new input.
   - *Easy → hard:* "What was sector A's 2020 growth?" becomes "By how much does the fastest-growing sector's 2020 growth exceed the median of sectors whose 2019 value was below average?"
   - *Keep it verifiable:* compute sub-answers from metadata, code or KG facts, and combine them programmatically. Reward sub-answers with a max-shaped process reward.
   - *Sources:* COGS, SPARK (2605.05546: KG paths over multimodal scientific documents; its gains grow with hop count), MMKG-RDS (2602.23632).

5. **Answer-first inverse synthesis (execute, then phrase)**
   - *What it does:* create the executable artifact (chart code, SQL, solver model, construction, simulator trace), execute it for the answer, then write the question conditioned on artifact and answer.
   - *Easy → hard:* "Value of bar B?" becomes a multi-panel, high-RPE chart where a script computes a cross-series year-over-year ratio and the question is written to that value.
   - *Keep it verifiable:* use deterministic execution. Back-check by having the teacher re-answer from the artifact (ChartVerse), or verify the description component by component (AutoOR).
   - *Sources:* ChartVerse, AutoOR, OmniSQL, CoSyn, ReachQA/CIT (2410.18798), TR-CoT (2410.17885).

6. **Structural AST mutation of formal targets**
   - *What it does:* apply grammar-level operators to a seed program or query, favouring operators that are feasible and rare so far.
   - *Easy → hard:* `SELECT name FROM employees WHERE dept='Sales'` becomes a CTE joining employees, departments and salaries, with a CASE-wrapped aggregate, a HAVING subquery and an INTERSECT.
   - *Keep it verifiable:* the query must execute and return a non-empty result. Refine using execution feedback, back-translate the question afterwards, and deduplicate per schema.
   - *Sources:* EvolSQL, OmniSQL (tiers).

7. **Context and distractor enlargement without changing the answer**
   - *What it does:* enlarge the input around the answer-bearing part.
   - *Easy → hard:*
     - distractor tables from related domains in the schema (Arctic)
     - 4 → 10 options with distractors from genetically related rules (VisualSphinx)
     - molecule distractors chosen by Tanimoto and property similarity (ether0)
     - render-only clutter, layout and palette changes (Trace)
     - complex charts selected by RPE (ChartVerse)
   - *Keep it verifiable:* distractors must provably fail, for example because they come from rule-violating scripts. Re-check that answer-relevant relations are unchanged, and measure the difficulty change on the policy.
   - *Sources:* Arctic-Text2SQL-R1, VisualSphinx, ether0, Trace, ChartVerse, DynaMath (2411.00836: seeds written as programs that generate value, graph and structure variants; report worst-case accuracy across variants).

8. **Deduction-depth extension and premise bootstrapping**
   - *What it does:* forward-chain, select problems by derivation length, and reuse the deepest graphs with extra premises. For computed geometry, raise grammar recursion depth.
   - *Easy → hard:* find a base angle of an isosceles triangle (1–2 rules) becomes a target needing ≥ 40 rule applications after bootstrap rounds.
   - *Keep it verifiable:* formal rule application per step, or symbolic equality of the final value. Thresholds: path length ≥ 5, premise ratio ≥ 0.5. Mix with natural data, because alone it can hurt.
   - *Sources:* TrustGeoGen, GeoSym127K, MAVIS (2407.08739).

9. **Controlled error injection → critique/localization**
   - *What it does:* inject exactly one subtle, known error into a correct artifact; the task is to locate it.
   - *Easy → hard:* "Describe the image" becomes "In this ~200-word caption, which span is wrong?" (three mugs → two; left → right).
   - *Keep it verifiable:* exact span match with tolerance for extra copied context. The source must be correct. Scale up with length, subtlety and number of errors.
   - *Sources:* ViCrit.

10. **Self-supervised transform recovery (pretext tasks)**
    - *What it does:* apply known transforms (permute, rotate, crop, mirror, reorder clips, depth-order points) and ask the model to recover them. A proposer can compose new transforms.
    - *Easy → hard:* swap two halves (2×1) becomes a 3×3 full permutation; rotation becomes rotation + mirror + crop with 4 candidate parameter sets; ordering 6 video clips.
    - *Keep it verifiable:* the parameters are the label. Give partial credit (γ·fraction correct) for large permutations. Reward the proposer for about 50% solver accuracy plus diversity.
    - *Sources:* Jigsaw-R1, Visual Jigsaw (2509.25190), EVE, MM-Zero (2603.09206: Proposer/Coder/Solver generating images by code from zero data).

11. **Mask-and-distract (fill-in-the-middle multiple choice from unverifiable text)**
    - *What it does:* mask a key reasoning span in a passage and choose among style-matched options.
    - *Easy → hard:* 3-option cloze with weak distractors (solved by elimination) becomes 9 options with near-miss derivation steps (wrong sign, misapplied law).
    - *Keep it verifiable:* the masked text is ground truth. Drop 16/16-solved items and tune the option count until most items are medium. Open-ended conversion failed (> 83% zero accuracy).
    - *Sources:* Golden Goose.

12. **Format hardening and scaffold fading**
    - *What it does:* remove affordances (options, hints, syntax scaffolds). Run in reverse for classes with zero pass rate: add scaffolds, then fade them.
    - *Easy → hard:* a USMLE 5-option multiple-choice question becomes an open-ended diagnosis. An OR prompt with Gekko syntax and mapped constraints becomes the same prompt without them, then the hard non-linear distribution.
    - *Keep it verifiable:* filter for answer uniqueness before removing options, and use an alias-aware judge audited on about 200 samples. Advance a phase only when pass rate > 0. Short-form answers are safer for rewards: in Nemotron-CrossThink, a unified open-ended format beat mixed formats by 1.21%, and short answers beat long ones by 1.20%.
    - *Sources:* HuatuoGPT-o1, AutoOR, Golden Goose, Nemotron-CrossThink (2504.13941).

13. **Direction inversion (forward → inverse → symbolic)**
    - *What it does:* ask for a hidden parameter given the outcome, or for a closed form, instead of forward evaluation.
    - *Easy → hard:* "2 kg block: v at 3 s?" becomes "what mass gives 5 m/s at 3 s?" or "give v(t)".
    - *Keep it verifiable:* the simulator or solver value with a relative tolerance; symbolic-equivalence checks; shortcut ablation.
    - *Caveat:* in Sim2Reason's 3B ablation, training on reverse (5.84) or symbolic (7.46) questions transferred far less to IPhO than forward numeric questions (13.15). Treat inversion as extra variety, not a replacement.
    - *Sources:* Sim2Reason, AutoOR (code→description is itself an inversion).

14. **Moving information from text into the image (modality shift)**
    - *What it does:* move the givens from the question text into the diagram, so the model has to read the figure.
    - *Easy → hard:* "In right triangle ABC with AB = 3, BC = 4…" becomes "Using the figure, find…", with the lengths only drawn.
    - *Keep it verifiable:* the answer is unchanged. Confirm the question is still uniquely answerable from the image, ideally by rendering from the same latent state.
    - *Sources:* MathVerse (2403.14624: six versions per problem, from text-dominant to vision-only), GeoSym127K (+22.21 on MathVerse Vision-Only).

15. **Scenario grounding and asymmetric information**
    - *What it does:* embed vague prompts in concrete, key-point-anchored scenarios, or hide the facts in a simulated user or opponent who must be questioned.
    - *Easy → hard:* "How much testosterone is safe?" becomes a consultant reviewing an Olympic powerlifter at 1,450 ng/dL. A static medical question becomes a multi-turn consultation with an introverted simulated patient. "Describe the image" becomes "find the spy whose image differs" (Vision-Zero).
    - *Keep it verifiable:* anchor on key points actually in teacher answers, not invented guidelines. Use case-specific constitutive rubrics and a 20–50% pass band. For games, labels come from construction.
    - *Sources:* QUBRIC, Baichuan-M2, Vision-Zero.

16. **Rubric hardening and relative bars (non-verifiable analogue of task hardening)**
    - *What it does:* when prompts cannot be hardened further, harden the grader: add criteria that separate the two best responses, co-train a rubric generator, drop criteria at random, or grade against the policy's own output.
    - *Easy → hard:* "Is the code correct?" becomes "handles the edge case in O(n)?" (RubricHub). A fixed checklist becomes a checklist regenerated each step to separate current rollouts (EvoRubrics).
    - *Keep it verifiable:* check that each new criterion splits current rollouts (both passes and fails). Randomly drop 30–50% of criteria per step, with the same subset shared across the rollout group. Keep a stronger held-out "gold" judge to detect hacking.
    - *Sources:* RubricHub, EvoRubrics, Rubric Dropout (2608.11669), Writing-Zero, DR Tulu (2511.19399: rubrics that evolve with the policy for deep research).

## Insights & pitfalls

- **Answer-first is what separates exact rewards from noisy ones.**
  - The sound pipelines fix the answer before rendering or phrasing: TRON, Trace, VVR, ChartVerse, GeoSym, Game-RL, AutoOR, Sim2Reason, EvolSQL, EVE.
  - Where an LLM supplies the answer, the authors name label noise as the main limitation: COGS sub-answers, TableDreamer's GPT-4o references, Active-Zero's majority votes, CoSyn's code-read QA.
- **Measure difficulty with the target policy.**
  - Bands used: SynthRL (≥ 12/16 seeds; 4 ≤ c ≤ c_ori − 2), ChartVerse (0 < fail < 1 over 3 teacher samples), VisualSphinx (0.375–0.875), QUBRIC (20–50%), EVE and Active-Zero (about 50% target), Arctic (≥ 1/10), Golden Goose (drop 16/16).
  - Behaviour-based measures beat LLM opinion: ChartVerse's RPE picked items with a 27.6% fail rate versus 21.1% for VLM-as-judge.
- **Hardening that ignores the policy is a reliable way to waste compute.**
  - Evol-Instruct rewriting lost to unchanged prompts (50.24 vs 50.51).
  - Arctic's LLM "complex question" augmentation mirrored the originals and dropped BIRD-dev from 64.9 to 62.5.
  - QUBRIC's surface rewrite fell below its SFT starting point (67.1 vs 71.0 on ArenaHard-hard).
  - At a fixed budget, CoSyn chart data lowered Qwen3-VL-4B from 53.9 to 51.0 in ChartVerse's comparison.
- **Check that levels are real, and do not train only on the top tier.**
  - TRON's base pass rate falls 72.8 → 41.3% across levels.
  - Game-RL's Easy+Medium+Hard mix beat every single tier.
  - GeoSym's GRPO on Hard after Entry SFT (43.59) underperformed GRPO on Entry (44.51).
  - Easy-to-hard transfer decays with distance: VVR +17.16 → +8.31 → +1.35; Jigsaw-R1's unseen 2×2 moved only 12.20 → 13.10. Train on hard instances directly, with partial credit if needed (Visual Jigsaw γ = 0.2).
- **Consume synthetic hard tasks with RL; SFT on the same data often regresses.**
  - Regressions: Sim2Reason (SFT 15.9 vs base 19.8 vs RL 25.2 on IPhO), AutoOR (Hard-LP SFT 26 vs base 55 vs RL 80), Game-RL (SFT −3.9% on general benchmarks), ViCrit (CapSFT 48.41 < base 50.61), COGS (SFT → RL 46.62 < base 47.36), Jigsaw-R1 (cold start hurts).
  - Where SFT helped: GeoSym, ChartVerse, HuatuoGPT-o1, Baichuan-M2. The model lacked domain knowledge or format there. MegaScience SFT helps only at ≥ 7B.
  - For regulated verticals, the 2026 finance study (2609.10113) found ordinary SFT lowered FINESSE-Bench by 3.2–4.0 points, self-distilled SFT raised it by 1.0–2.8, merging recovered 3.0, and GRPO directly on verifiable tasks added 3.0.
- **Shortcut and degeneracy filters matter as much as generators.**
  - Sim2Reason: shortcut-filtered data gave 13.15 vs 7.14 unfiltered, with about 15% of items removed.
  - Arctic dropped about 3,100 gold queries that returned empty results or ran too long.
  - EvolSQL requires non-empty execution.
  - VVR requires each task to accept its reference and reject a counterfactual.
- **Conjunctions and composition lag single skills.** VVR closes 79/64% of the gap on single-constraint partial scores but 55/46% on all-satisfied. Use multiplicative guards (ψ), max-shaped process rewards (COGS), and schedules that grow the number of constraints.
- **Zero-signal classes need scaffolds and buffers, not just harder data.** AutoOR went from 0% at pass@64 to 48.98% via scaffold fading. ether0's same-reward groups reached 90% of batches and were fixed by a non-trivial buffer. TRON and Trace filter same-reward groups.
- **Harder formats are not always better training signal.** Open-ended Golden Goose items gave zero signal for > 83% of items; 9-option multiple choice worked. Sim2Reason's inverse and symbolic formats transferred worse than forward questions. Pick the format that puts most items in the medium band.
- **Diversity and weakness targeting beat volume.** TableDreamer's 27K weakness items beat 34K unfiltered ones (60.69 vs 56.28). EvolSQL beats a model trained on SynSQL with about 1/18 of the data. Game-RL does better with 20 games than 4 at a fixed 5K samples. Golden Goose gets 13× more effective examples than ProRL.
- **Perception proxy tasks transfer to reasoning.** ViCrit (CharXiv +6.4 at 7B), Jigsaw-R1 (Super-CLEVR +7.00), Trace (visual math +7.44 at 3B), EVE (MMVet +5.1). These cheap label-free tasks are an underused source of hard RL signal for VLMs.
- **Rubric rewards saturate and get exploited under optimization.**
  - Rubric Dropout (2608.11669): the training judge's score keeps rising while a gold judge peaks and falls (−3 on HealthBench-Hard, −22 on ResearchQA). Dropping 30–50% of criteria per step, with the subset shared across the rollout group, recovers +1 to +2 and +6 to +7.
  - EvoRubrics: static golden rubrics lowered out-of-distribution FollowBench HSR from 47.76 to 32.50.
  - **Corrected claim:** ImpossibleRubrics (2609.16816) reports generated rubrics exploited in 8–26% of an unbiased 150-environment cut, and 36–98% on a selected 45-environment stress cut. The 64% figure is a naive "be decisive" rubric *on the stress cut*, not on the 150-environment cut. Certificate-faithful rubrics were exploited 0/45.
- **Relative bars resist hacking better than scalar reward models, even when those models look more accurate offline.** Writing-Zero's scalar RM was more accurate on static preference sets (64.7 vs 57.5) but produced longer outputs with more self-justifying filler under RL (417 vs 58).
- **Self-evolving VLM loops plateau.** EVE plateaus after 4–5 iterations; Active-Zero regresses slightly at iteration 3; Vision-Zero alternates self-play with RLVR to avoid stalling. Methods based on pseudo-labels (Active-Zero, VisPlay and EvoLMM as reported) are exposed to drift.
- **Hide the answer from the model doing the hardening** (SynthRL), **and paraphrase templated items during RL** (ether0 uses Gemini 2.5 Flash rewrites) to avoid overfitting to the wording of a few templates.
- **Finance and legal are still mostly "distill, filter, split by teacher-judged difficulty".** Examples: Fin-R1 (2503.16252) with 60,091 distilled CoT samples, and LexPam (2504.02590) with a teacher-based basic/challenging split. Selection is a cheap complement: ThinkLite-VL (2504.07934) keeps 11K of 70K samples by MCTS iteration count and reaches 75.1 on MathVista at 7B.

## Open problems & research opportunities

- **Predicting difficulty before rollouts.** Trace states there is no shared difficulty notion across domains, and TRON's levels are defined per environment. A calibrated model that maps generator parameters to the expected pass rate would cut the cost of filtering by rollouts. Today's filters need 3–16 rollouts per item.
- **Programmatic verifiers for generated or rendered outputs beyond toy scenes.** VVR covers 8 colours, 3 shapes and 2D shapes only. VBVR-Pro (2608.26105) gives rule-based scorers for 300 procedural tasks in image/video *generation*, but compositional verifiers for 3D, natural-image and video outputs are open.
- **Curricula and reward shaping for conjunctions.** How fast to grow the number of stacked constraints, and how to interpolate between partial credit and all-or-nothing, is unstudied.
- **Answer-first environments for long documents, multi-image and video reasoning.** Visual Jigsaw is a first step for video; COGS names long documents as out of scope; most generators produce single static images.
- **Science generators beyond classical mechanics and templated chemistry.** Thermodynamics, statistical mechanics, kinetics, materials and biology simulators are missing. Whether tolerance-based numeric rewards (5% in Sim2Reason, 2% in AutoOR) are exploited by approximate heuristics is untested. Why inverse and symbolic formats transfer worse is unexplained.
- **Programmatic latent-state generators for finance and law are almost absent.** Candidates: synthetic ledgers → statements with computable ratios and restatements; synthetic statutes and tax codes with rule engines; contract-clause engines with deterministic outcomes. Current work distills and filters existing QA (Fin-R1, LexPam, 2609.10113).
- **Table and spreadsheet RL with executable ground truth at scale.** TableDreamer still relies on GPT-4o references. Spreadsheet-RL (2605.22642) shows the environment side (SpreadsheetBench 12.0 → 23.4% for Qwen3-4B-Thinking), but formula-verified, hardness-controlled synthetic multi-table tasks are missing.
- **Joint prompt + rubric adaptation.** LLM-as-a-Tutor adapts prompts; EvoRubrics and RubricHub adapt rubrics; QUBRIC co-designs them once offline. No method adapts both online with contradiction detection for appended constraints.
- **Rubrics that hold up under optimization.** ImpossibleRubrics shows generated rubrics can signal which claim to fabricate. Certificate-faithful or adversarially tested rubric generation *for training* is open.
- **Attribution and transfer theory for procedural training.** Neither Trace nor TRON has mixture ablations, so it is unknown which operation families or levels drive natural-benchmark gains, or how many environments are enough.
- **Diversity collapse in million-scale generators.** TRON's pHash and template-Jaccard audits and ChartVerse's CLIP limits exist but are not standard, and no novelty metric has been tied to downstream gains.
- **Open-ended self-play anchored in execution.** Execution-grounded challengers (EVE, MM-Zero) plateau and pseudo-label ones drift. Combining active retrieval (Active-Zero) with executable answers and difficulty calibration is untried.

## References

1. Li, S. S., Han, X., Tsvetkov, Y., Zettlemoyer, L. (2026). *Verifiable Visual Rewards Transfer from Synthetic Scenes to Natural Prompts*. arXiv:2609.35641. https://arxiv.org/abs/2609.35641
2. Wu, Z., Ni, J., Liu, X., Liu, Z., Yan, H., Shieh, M. Q. (2025). *SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis*. arXiv:2506.02096. https://arxiv.org/abs/2506.02096
3. Yang, T., Shi, Y., Sun, R., Huang, J., et al. (2026). *TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL*. arXiv:2606.01599. https://arxiv.org/abs/2606.01599
4. Alam, M. T. (2026). *Trace: A Taxonomy-Guided Environment for Multidomain Visual Reasoning*. arXiv:2607.19790. https://arxiv.org/abs/2607.19790
5. Feng, Y., Xu, Z., Jiang, F., Li, Y., et al. (2025). *VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL*. arXiv:2505.23977. https://arxiv.org/abs/2505.23977
6. Tong, J., Tang, J., Li, H., Mou, Y., et al. (2025). *Game-RL: Synthesizing Multimodal Verifiable Game Data to Boost VLMs' General Reasoning*. arXiv:2505.13886. https://arxiv.org/abs/2505.13886
7. Wang, Z., Zhu, J., Tang, B., Li, Z., et al. (2025). *Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles*. arXiv:2505.23590. https://arxiv.org/abs/2505.23590
8. Wu, P., Zhang, Y., Diao, H., Li, B., et al. (2025). *Visual Jigsaw Post-Training Improves MLLMs*. arXiv:2509.25190. https://arxiv.org/abs/2509.25190
9. Wang, X., Yang, Z., Feng, C., Liang, Y., et al. (2025). *ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs*. arXiv:2506.10128. https://arxiv.org/abs/2506.10128
10. Heng, Y., Jiang, C., Yang, H., Zhang, S., et al. (2026). *EVE: Verifiable Self-Evolution of MLLMs via Executable Visual Transformations*. arXiv:2604.18320. https://arxiv.org/abs/2604.18320
11. Wang, Q., Liu, B., Zhou, T., Shi, J., et al. (2025). *Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play*. arXiv:2509.25541. https://arxiv.org/abs/2509.25541
12. He, J., Fang, J., Xiong, F., Yao, Z., et al. (2026). *Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration*. arXiv:2602.11241. https://arxiv.org/abs/2602.11241
13. Liu, Z., Lin, H., Qin, C., Wang, X., et al. (2026). *ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch*. arXiv:2601.13606. https://arxiv.org/abs/2601.13606
14. Gu, X., Mao, J., Hong, Z.-W., Yu, Z., et al. (2025). *Composition-Grounded Data Synthesis for Visual Reasoning*. arXiv:2510.15040. https://arxiv.org/abs/2510.15040
15. Jing, J., Ma, Z., Liang, J., Zhao, Q., et al. (2026). *GeoSym127K: Scalable Symbolically-verifiable Synthesis for Multimodal Geometric Reasoning*. arXiv:2605.16371. https://arxiv.org/abs/2605.16371
16. Fu, D., Chen, J., Xia, R., Chen, Z., et al. (2025). *TrustGeoGen: Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving*. arXiv:2504.15780. https://arxiv.org/abs/2504.15780
17. Yang, Y., Patel, A., Deitke, M., Gupta, T., et al. (2025). *Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation* (CoSyn). arXiv:2502.14846. https://arxiv.org/abs/2502.14846
18. Prabhudesai, M., Satpathy, A., Li, Y., Qin, Z., et al. (2026). *Solving Physics Olympiad via Reinforcement Learning on Physics Simulators* (Sim2Reason). arXiv:2604.11805. https://arxiv.org/abs/2604.11805
19. Narayanan, S. M., Braza, J. D., Griffiths, R.-R., Bou, A., et al. (2025). *Training a Scientific Reasoning Model for Chemistry* (ether0). arXiv:2506.17238. https://arxiv.org/abs/2506.17238
20. Motwani, S. R., Du, C., Petrov, A., Davis, C., et al. (2026). *AutoOR: Scalably Post-training LLMs to Autoformulate Operations Research Problems*. arXiv:2604.16804. https://arxiv.org/abs/2604.16804
21. Fan, R.-Z., Wang, Z., Liu, P. (2025). *MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning*. arXiv:2507.16812. https://arxiv.org/abs/2507.16812
22. Yuan, W., Yu, J., Jiang, S., Padthe, K., et al. (2025). *NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions*. arXiv:2502.13124. https://arxiv.org/abs/2502.13124
23. Lu, X., Acuna, D., Jung, J., Hu, J., et al. (2026). *Golden Goose: A Simple Trick to Synthesize Unlimited RLVR Tasks from Unverifiable Internet Text*. arXiv:2601.22975. https://arxiv.org/abs/2601.22975
24. Li, H., Wu, S., Zhang, X., Huang, X., et al. (2025). *OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale*. arXiv:2503.02240. https://arxiv.org/abs/2503.02240
25. Pan, X., Tao, C., Bai, J., Gao, J., et al. (2026). *EvolSQL: Structure-Aware Evolution for Scalable Text-to-SQL Data Synthesis*. arXiv:2601.04875. https://arxiv.org/abs/2601.04875
26. Yao, Z., Sun, G., Borchmann, L., Nuti, G., et al. (2025). *Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL*. arXiv:2505.20315. https://arxiv.org/abs/2505.20315
27. Zheng, M., Feng, Z., Wang, J., Wang, L., et al. (2025). *TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning*. arXiv:2506.08646. https://arxiv.org/abs/2506.08646
28. Chen, J., Cai, Z., Ji, K., Wang, X., et al. (2024). *HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs*. arXiv:2412.18925. https://arxiv.org/abs/2412.18925
29. Baichuan-M2 Team (Dou, C., Liu, C., et al.) (2025). *Baichuan-M2: Scaling Medical Capability with Large Verifier System*. arXiv:2509.02208. https://arxiv.org/abs/2509.02208
30. Kim, Y., Ho, N., Hwang, S., Kim, J., et al. (2026). *LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL*. arXiv:2607.04412. https://arxiv.org/abs/2607.04412
31. Zhang, R., Feng, R., Zhang, Z., Yang, J., et al. (2026). *QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards*. arXiv:2606.03968. https://arxiv.org/abs/2606.03968
32. Ding, H., Huang, B., Fang, Y., Liao, W., et al. (2026). *EvoRubrics: Dynamic Rubrics as Rewards via Adversarial Co-Evolution for LLM Reinforcement Learning*. arXiv:2606.23038. https://arxiv.org/abs/2606.23038
33. Li, S., Zhao, J., Wei, M., Ren, H., et al. (2026). *RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation*. arXiv:2601.08430. https://arxiv.org/abs/2601.08430
34. Jia, R., Yang, Y., Gai, Y., Luo, K., et al. (2025). *Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards*. arXiv:2506.00103. https://arxiv.org/abs/2506.00103
35. Alam, M. T., Aggarwal, S., Chae, J. Y., Rastogi, N. (2025). *SPHINX: A Synthetic Environment for Visual Perception and Reasoning*. arXiv:2511.20814. https://arxiv.org/abs/2511.20814
36. Xu, J., Wang, R., Pu, F., Wang, M., et al. (2026). *VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning*. arXiv:2608.26105. https://arxiv.org/abs/2608.26105
37. Zeng, Y., Sai, Z., Wang, Y., Hou, Y., et al. (2026). *Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation* (MIFS). arXiv:2609.16059. https://arxiv.org/abs/2609.16059
38. Zeng, M., Tan, G., Cen, L., Wen, Y., et al. (2026). *VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following*. arXiv:2608.26013. https://arxiv.org/abs/2608.26013
39. Park, H., Kim, T., Choi, D.-G. (2026). *SPARK: Self-Play with Asymmetric Reward from Knowledge Graphs*. arXiv:2605.05546. https://arxiv.org/abs/2605.05546
40. Zhan, L., Xiong, F., Liu, H., Zhang, F., et al. (2026). *MMKG-RDS: Reasoning Data Synthesis via Deep Mining of Multimodal Knowledge Graphs*. arXiv:2602.23632. https://arxiv.org/abs/2602.23632
41. He, W., Xi, Z., Zhao, W., Fan, X., et al. (2024). *Distill Visual Chart Reasoning Ability from LLMs to MLLMs* (ReachQA / Code-as-Intermediary Translation). arXiv:2410.18798. https://arxiv.org/abs/2410.18798
42. Deng, L., Zhu, L., Liu, Y., Wang, Y., et al. (2024). *Theorem-Validated Reverse Chain-of-Thought Problem Generation for Geometric Reasoning* (TR-CoT). arXiv:2410.17885. https://arxiv.org/abs/2410.17885
43. Zhang, R., Wei, X., Jiang, D., Guo, Z., et al. (2024). *MAVIS: Mathematical Visual Instruction Tuning with an Automatic Data Engine*. arXiv:2407.08739. https://arxiv.org/abs/2407.08739
44. Li, Z., Du, H., Huang, C., Wu, X., et al. (2026). *MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data*. arXiv:2603.09206. https://arxiv.org/abs/2603.09206
45. Akter, S. N., Prabhumoye, S., Novikov, M., Han, S., et al. (2025). *Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning*. arXiv:2504.13941. https://arxiv.org/abs/2504.13941
46. Zou, C., Guo, X., Yang, R., Zhang, J., et al. (2024). *DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models*. arXiv:2411.00836. https://arxiv.org/abs/2411.00836
47. Zhang, R., Jiang, D., Zhang, Y., Lin, H., et al. (2024). *MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?* arXiv:2403.14624. https://arxiv.org/abs/2403.14624
48. Yang, M., Guo, X., Tyagi, U., Zhang, M., et al. (2026). *Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL*. arXiv:2608.11669. https://arxiv.org/abs/2608.11669
49. Qin, B., Xie, Y., Liu, Y., Yang, X. (2026). *ImpossibleRubrics: Stress-Testing Generated Rubrics as Reward Signals*. arXiv:2609.16816. https://arxiv.org/abs/2609.16816
50. Shao, R., et al. (2025). *DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research*. arXiv:2511.19399. https://arxiv.org/abs/2511.19399
51. Liu, Z., Guo, X., Yang, Z., Lou, F., et al. (2025). *Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning*. arXiv:2503.16252. https://arxiv.org/abs/2503.16252
52. Zhang, K., Xie, G., Yu, W., Xu, M., et al. (2025). *Legal Mathematical Reasoning with LLMs: Procedural Alignment through Two-Stage Reinforcement Learning* (LexPam). arXiv:2504.02590. https://arxiv.org/abs/2504.02590
53. Hayrapetyan, Z., Kalmykov, A., Kokosinskii, D., Stanishevskii, D., et al. (2026). *Data-Centric Post-Training for Financial Reasoning: Mining, Distillation, and Verifiable Learning*. arXiv:2609.10113. https://arxiv.org/abs/2609.10113
54. Wang, X., Yang, Z., Feng, C., Lu, H., et al. (2025). *SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement* (ThinkLite-VL). arXiv:2504.07934. https://arxiv.org/abs/2504.07934
55. Chi, B., et al. (2026). *Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning*. arXiv:2605.22642. https://arxiv.org/abs/2605.22642
