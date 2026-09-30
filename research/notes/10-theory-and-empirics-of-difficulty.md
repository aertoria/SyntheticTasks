# Theory & empirical science: easy-to-hard generalization, skill composition, and why difficulty matters for SFT/RL

*Scope: controlled studies and theory on (a) when training on composed or harder synthetic tasks teaches new skills rather than sharpening old ones, (b) how difficulty knobs (depth, horizon, width, expressiveness, topology) map to learnability and compute, (c) how to place data at the model's edge of competence for GRPO/PPO-style RL and for SFT, and (d) the confounds that fake these gains (contamination, spurious rewards, model collapse). Compiled 2026-09-30. Verification: 28 researcher entries checked against primary sources (arXiv abstract pages plus full-text HTML, with numbers read from the text, tables and LaTeX alt-text); 10 corrected; 0 dropped; 6 added. A further 46 secondary arXiv IDs cited in the draft were resolved, and their titles and authors matched.*

---

## TL;DR

- **If your tasks are about 100% solved, compose them. Paraphrasing will not help.** Build composed tasks (nest f(g(x)), chain answer→input, branch–merge over a dependency graph) from atomic seeds the model already solves, and run RL on the composites. RL on Level-2 string compositions lifted held-out Level-3 accuracy from about 5% to about 30% and Level-4 from about 1% to about 15%. RL on atomic (Level-1) data alone stayed below 25% on Level 2 and near 0 on Levels 3–6 [f(g(x)), 2509.25123]. Chaining GSM8K problems with a horizon curriculum doubled AIME24 for a 3B instruct model (5.10→10.52 avg@32) [h1, 2510.07312]. Training on decomposed skills does not transfer upward. Training on composed tasks does transfer back down to the parts [He et al. 2609.19465; OMEGA 2506.18880].
- **Target the edge of competence: pass@1 low, pass@k > 0.** In a fully controlled pipeline, RL on in-distribution difficulty gave no pass@128 gain, while RL on edge data (op 11–14) gave up to +42% pass@128 [Interplay, 2512.07783]. Filter online to 0 < pass-rate < 1 and wrap each seed family in a generator with a difficulty ratchet. RLVE promotes a level when accuracy exceeds 90%, with a window of 4 levels. On an already-saturated ProRL-1.5B-v2 it gave +3.37 points in about 1,100 H100-h, versus +0.49 for continuing the original RL with more than 3× the compute [2511.07317].
- **RL composes existing primitives; it does not create them.** With 0% or 0.1% pretraining exposure to a context, RL did not transfer to it. With 1% exposure it gave up to +60% pass@128 [2512.07783]. Install atoms with SFT or mid-training, then use RL on compositions [Kong et al. 2606.18089; Chu et al. 2501.17161].
- **Difficulty compounds multiplicatively, and so does compute.** Multi-step success tracks the product of atomic-step success rates (Pearson ρ 0.69–0.96; 0.3⁵ ≈ 0.0024) [Algebrarium, 2602.08281]. RL steps needed to reach 90% grow as depth^γ, with γ rising from 1.05 to 2.60 as the logic becomes more expressive. Depth generalization falls to chance at about 3× the training depth [ScaleLogic, 2605.06638]. Make primitives reliable before adding depth, and budget compute superlinearly for deeper tiers.
- **Add new operator types rather than only more steps.** In ScaleLogic, the most expressive logic (∧, ∨, ¬, ∀) raised an 8-benchmark average from 49.39% to 60.05% (+10.66). The implication-only and conjunction-only settings plateaued near 52%.
- **Difficulty coverage matters more than ordering, except when the top tier is sparse.** A controlled 2026 study found no robust gain from easy→hard ordering over random mixing, for SFT or RL (e.g., OOD 0.27 vs 0.28) [2603.27226]. Theory says mixed easy+hard data is enough [2505.23683]. Staging does matter when the hardest tier would otherwise get zero reward: in h1, a uniform mix gave no long-horizon gains, and in ScaleLogic the curriculum reached γ = 1.33 versus 2.36 for difficult-only training.
- **For families at pass@K = 0, use dense partial credit, then a binary reward, then wait.** DELTA-Code stayed below 1% full-pass for 450 steps and then "grokked" to near 100% on Manufactoria-HAS after a per-test-case reward warm-up [2509.21016]. The alternatives are learned stepping stones (SOAR: 4× pass@1 on fail@128 MATH [2601.18778]) or scaffolds.
- **Guarantee correctness by construction, not by asking an LLM to judge its own harder questions.** R-Zero's majority-vote pseudo-label accuracy fell from 79% to 69% to 63% as its Challenger made questions harder [2508.05004]. Compose programs, propagate answers through chains, build proofs backward and audit them with a solver (ScaleLogic: Z3 audit of 1,000 sampled items per configuration, all passed), and check solvability. Some complexity knobs silently create impossible instances, e.g., River Crossing with N ≥ 6 and boat capacity 3 [2506.09250].
- **Judge complexified data by boundary metrics and on clean data.** Use large-k pass@k, the null-set unlock rate, per-topology accuracy and Cover@τ. Freshly generated instances, plus at least two model families, are required: random rewards gave +21.4 MATH-500 points on Qwen2.5-Math-7B but did not work on Llama3 or OLMo2 [2506.10947, 2507.10532].
- **Engineer diversity into complexification.** Evol-Instruct-style rewriting pushed pass@8 below the untuned backbone (54.2 vs 55.1). Skill crossover plus parametric mutation plus a learnability filter raised both pass@1 (+4) and pass@8 (+7.4) [EvoTD, 2605.11666].

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| f(g(x)) compositional RL | 2025 | https://arxiv.org/abs/2509.25123 | Synthetic string functions; transfer to Countdown | SFT(RFT)+RL | Function nesting depth (L1–L6); hidden definitions; held-out function sets | Execute composed Python functions; exact match |
| Composed-vs-decomposed asymmetry (He et al.) + Kong et al. + Abdulsalam et al. | 2026 | https://arxiv.org/abs/2609.19465 | Data structures (DSR-Bench), BFCL tool calls; rewrite grammars | RL (+SFT) | Dependency-graph levels: single-skill chain → multi-skill chain → branch–merge; length extrapolation | Deterministic data-structure simulation with fixed tie-breaking; BFCL normalized exact match |
| Countdown skill-composition analysis | 2025 | https://arxiv.org/abs/2512.01775 | Countdown | RL analysis | More operands n; unseen tree shapes (balanced / left-heavy / right-heavy) | Evaluate expression; target sampled from a chosen pattern so a solution of that shape exists |
| OMEGA (added) | 2025 | https://arxiv.org/abs/2506.18880 | 40 templated math generators, 6 domains | RL eval/benchmark | Complexity-level ladder; skill A+B composition; transformative strategy shift | Programmatic generation; symbolic/numeric/graphical solution checks |
| DELTA-Code / RL grokking recipe | 2025 | https://arxiv.org/abs/2509.21016 | Synthetic program-synthesis families (Manufactoria, BouncingSim) | RL | Template knobs (discrete + numeric jitter); tiered difficulty; skill recomposition | Hidden unit tests from reference programs; per-test then full-pass reward |
| Algebrarium (sharper primitives) | 2026 | https://arxiv.org/abs/2602.08281 | Procedural algebraic systems (4 group types) | RL analysis | Chain 1-hop operations to depth 2–5 | Symbolic evaluation under declared axioms; exact match |
| Skill-Mix composition fine-tuning | 2024 | https://arxiv.org/abs/2409.19808 | Open-ended text with k named language skills | SFT | Raise k (number of simultaneously required skills); held-out skill categories | LLM grader rubric per skill (GPT-4; Claude 3 Opus cross-check) |
| Grokked transformers (added) | 2024 | https://arxiv.org/abs/2405.15071 | Synthetic knowledge graphs (2-hop composition, comparison) | Pretraining analysis | Ratio of inferred (composed) to atomic facts | Facts deduced from latent rules over a generated KG |
| Interplay of pre-/mid-training and RL | 2025 | https://arxiv.org/abs/2512.07783 | GSM-Infinite-style DAG math, multiple surface contexts | Analysis (pretrain→mid→RL) | DAG op-count scaling; surface re-rendering; context exposure ratio | Generator knows DAG; parsed traces checked step-by-step against gold graph |
| Yue et al. pass@k boundary | 2025 | https://arxiv.org/abs/2504.13837 | Math, code, visual reasoning | Analysis | None (diagnostic) | Standard benchmarks |
| ProRL | 2025 | https://arxiv.org/abs/2505.24864 | Math, code, STEM, Reasoning Gym, IF | RL | Task-family diversity (96 Reasoning Gym tasks); training duration | Rule-based verifiers; Reasoning Gym verifiers; test-case fraction for code |
| SFT memorizes, RL generalizes (added) | 2025 | https://arxiv.org/abs/2501.17161 | GeneralPoints (card arithmetic), V-IRL navigation | SFT vs RL | Rule variants (J/Q/K = 11/12/13 vs 10); visual variants | Programmatic game/navigation verifiers |
| h1 (serial composition) | 2025 | https://arxiv.org/abs/2510.07312 | Chained GSM8K; transfer to MATH/AIME/ReasoningGym | RL | Serial chaining with adapters; horizon curriculum h = 1…5 | Answers propagate deterministically from verified atomic answers; well-posedness filters |
| ScaleLogic | 2026 | https://arxiv.org/abs/2605.06638 | Synthetic deductive logic | RL | Proof depth D; logical expressiveness; candidate count; distractor rules | Backward construction; single-axiom corruption for negatives; Z3 audit |
| e3 | 2025 | https://arxiv.org/abs/2506.09026 | Competition math (DeepScaleR data), Countdown | RL | Asymmetric skill chaining (generate + verify); coupled difficulty × token-budget curriculum | Final-answer verification |
| Illusion of Diminishing Returns (added) | 2025 | https://arxiv.org/abs/2509.09677 | Key–value running-sum execution | Eval/analysis | Number of turns; keys per turn (turn complexity) | Exact arithmetic over an in-context dictionary |
| iGSM (Physics of LMs 2.1) | 2024 | https://arxiv.org/abs/2407.20311 | Synthetic grade-school math | Pretraining analysis | DAG op count; unnecessary parameters; template-hash splits | Mod-23 arithmetic computed from DAG |
| Faith and Fate | 2023 | https://arxiv.org/abs/2305.18654 | Multiplication, logic grid puzzles, DP | Eval + fine-tuning analysis | Computation-graph depth/width; problem size | Algorithmic ground truth |
| Illusion of Thinking (added) + Lawsen comment | 2025 | https://arxiv.org/abs/2506.06941 | Tower of Hanoi, Checker Jumping, River Crossing, Blocks World | Eval | Scale N (disks, checkers, actors, blocks) | Puzzle simulators validate every move |
| RLVE | 2025 | https://arxiv.org/abs/2511.07317 | 400 procedural environments | RL | Integer difficulty ratchet per environment; environment-count scaling | Programmatic verifiers (exploit verify ≪ solve) |
| Online difficulty filtering (+ learnability sampling, DAPO, ScaleRL) | 2025 | https://arxiv.org/abs/2504.03380 | Math RLVR | RL | Pass-rate band selection (control, not generation) | Standard verifiable rewards |
| Hard Examples Are All You Need | 2025 | https://arxiv.org/abs/2508.14094 | GSM8K, BBH Tracking Shuffled Objects | RL | Select hardest 10% by base-model failure | Ground-truth answers |
| DARS / Depth-Breadth Synergy (added) | 2025 | https://arxiv.org/abs/2508.13755 | Math RLVR | RL | Extra rollouts for low-accuracy prompts (re-weights hard items) | Standard verifiable rewards |
| Easy-to-Hard Generalization (evaluators) | 2024 | https://arxiv.org/abs/2403.09472 | MATH | SFT + RL/rerank | Train evaluator on easy levels, apply to hard | PRM trained on easy human labels; final eval on ground truth |
| Unreasonable Effectiveness of Easy Data | 2024 | https://arxiv.org/abs/2401.06751 | QA (grade-school to college STEM, trivia) | Analysis (ICL, linear probe, QLoRA) | Easy-only vs hard-only training splits | Existing labels |
| Self-Improving Transformers | 2025 | https://arxiv.org/abs/2502.01612 | Arithmetic, string copy/reverse, mazes | SFT (self-training) | +1 size per round (or accelerated) | Self-labels filtered by length and multi-seed majority vote |
| s1K (+ LIMO, OpenThoughts) | 2025 | https://arxiv.org/abs/2501.19393 | Math/science reasoning SFT | SFT | Select questions unsolved by reference models; long traces; domain diversity | Teacher-distilled traces (no correctness filter needed for SFT per OpenThoughts) |
| Easy-to-hard theory (+ Curriculum I/II) | 2025–26 | https://arxiv.org/abs/2505.23683 | k-fold composition; semiautomata | Theory | Composition depth k; decomposition into blocks | Theoretical |
| Rethinking Easy-to-Hard | 2026 | https://arxiv.org/abs/2603.27226 | MathGAP arithmetic, Knights & Knaves | SFT + RL analysis | Inference depth / axiom count; number of characters | Unique-solution generators |
| SOAR (grounded teacher) | 2026 | https://arxiv.org/abs/2601.18778 | Math (fail@128 subsets) | RL (bilevel meta-RL) | Teacher-generated stepping-stone problems | Not verified per item; teacher rewarded by student gain on real hard problems |
| EvoTD | 2026 | https://arxiv.org/abs/2605.11666 | Code-grounded algorithmic reasoning → math/code benchmarks | RL | Skill crossover; attribute mutation (n = 10 variants); ZPD filter | Python executor computes outputs from proposer's program+input |
| SynthLLM | 2025 | https://arxiv.org/abs/2503.19551 | Synthetic math questions for (continued) training | SFT/pretraining-scale | Concept recombination; cross-document concept-graph random walks | LLM-generated answers, no verifier (filtering ablation gave marginal gain) |
| Spurious rewards (+ RandomCalculation) | 2025 | https://arxiv.org/abs/2506.10947 | Math RLVR on Qwen2.5-Math, Llama3, OLMo2 | Analysis | Reward corruption as a control; fresh arithmetic generator | RandomCalculation computes answers for never-published expressions |
| Model collapse (+ accumulate / verify / strong collapse) | 2023–24 | https://arxiv.org/abs/2305.17493 | LLMs, VAEs, GMMs, regression | Analysis | Recursive self-generated data (failure mode) | Mitigations: keep real data, verifier selection |

---

## Method notes

### f(g(x)) compositional RL — From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones (Lifan Yuan et al., 2025)
Link: https://arxiv.org/abs/2509.25123 (v3). Authors: Yuan, Chen, Zhang, Cui, Wang, You, Ding, Liu, Sun, Peng.
- **Mechanism**: The task is to predict the output of named string-transformation functions. There are 25 atomic functions covering character manipulation, reordering, filtering and structural edits. Stage 1 installs atoms with rejection fine-tuning (RFT). Correct traces are collected with function definitions in the prompt, and the definitions are removed at training time so the behavior is internalized under the function identifier. Stage 2 runs RL (or iterative RFT as a baseline) with definitions hidden, on Level-1 only, Level-2 only, or a uniform L1+2 mix. Functions are split into two disjoint sets: RL compositions use one set, and evaluation uses compositions of the held-out set. Base model: Llama-3.1-8B-Instruct.
- **How it makes tasks harder**: Increase the nesting depth of known atomic functions, e.g., func_2(func_16(x)), up to Level 6. Transfer is tested on Countdown Levels 3–5, which were never RL-trained.
- **Correctness / verification**: The ground truth comes from executing the composed Python functions on x, and the reward is exact match. Every depth is verifiable by construction.
- **Difficulty control**: A single knob, nesting depth (Level 1–6).
- **Reported results**:
  - Level 1 (atomic) RL peaks near 90% on L1, stays below 25% on L2, and is near 0 on L3–6.
  - RL on L2 (or L1+2) lifts L3 from about 5% to about 30% and L4 from about 1% to about 15%, and the gain continues at L5.
  - Head-to-head on the same L2 data: RL reaches 64% on L2 and 27% on L3. Iterative RFT reaches only 15% on L2 and never exceeds 2.6% on L3.
  - At L5 the gap between RL L1+2 and the RFT base grows from 4% at pass@1 to about 25% at pass@1024, i.e., the boundary expands.
  - Cross-task: a model that has Countdown atoms (via RFT) plus string-composition RL reaches 35% at Countdown Level 3 (Avg@32), more than 18 points over the atom-only baseline, and about 6% at Level 4. The same composition RL without Countdown atoms does not transfer.
  - Failure analysis at L3: RFT-base, RFT-L2 and RL-L1 models fail mainly by ignoring the composition (about 50%) or misreading the nesting (about 35%). RL-L2 eliminates "ignores composition" errors, solves 28.1%, and its remaining failures are mostly atomic errors.
- **Limitations / failure modes**: Fully synthetic. Atoms must be installed first. The authors claim atoms are sufficient, not strictly necessary. Next-token training (RFT) on the same composed data yields none of the gains.
- **How to reuse with easy seed tasks**: Treat each saturated seed family as an atom library. Programmatically generate depth-2 compositions whose ground truth is computed by executing the composed program. Run RL on the composites, not the atoms. Hold out whole atom subsets to test compositional transfer, and evaluate at depth ≥ 3 with large-k pass@k.

### Composed-vs-decomposed asymmetry — Compositional Reasoning in Language Models under Reinforcement Learning Post-Training (Yu He et al., 2026)
Link: https://arxiv.org/abs/2609.19465. Related verified works bundled here: Kong et al., *From Reasoning Traces to Reusable Modules* (arXiv 2606.18089, ICML 2026); Abdulsalam, Patel & Saxe, *RL Post-Training Builds Compositional Reasoning Strategies* (arXiv 2607.07646, ICML 2026 Compositional Learning workshop).
- **Mechanism**: A dependency-graph formalism defines three levels of compositionality:
  - I: single-skill chain, e.g., repeated inserts; this tests horizon.
  - II: multi-skill chain, e.g., interleaved inserts and deletes; this tests skill switching.
  - III: branch–merge graph; this tests non-local coordination.

  The tasks are DSR-Bench data structures (Array, BST, Bloom filter, Hashmap, Heap). RL-Atomic (decomposed skills only) is compared with RL-Compound (composed tasks only), using GRPO with binary final rewards and equal step counts, on Qwen3-4B-Instruct, OLMo-3-7B-Instruct and Llama-3.1-8B-Instruct.
- **How it makes tasks harder**: More operations (training lengths 5–10, evaluation up to 30), heterogeneous skill switches, branch–merge dependencies, structural distribution shift (skewed vs balanced BSTs, sparse vs dense graphs), and unseen skills.
- **Correctness / verification**: Deterministic simulation. Implementation ambiguities are fixed so that each prompt has a single valid output: neighbors are visited in increasing order, heap ties use a fixed rule, and every deletion is guaranteed valid. Outputs are serialized (e.g., preorder). Structured JSON output reduced accuracy, so it was dropped. The BFCL pilot uses normalized exact-match of function names, arguments and call sets.
- **Difficulty control**: Compositional level (I/II/III) × length × structural distribution.
- **Reported results**: The asymmetry is consistent across levels and model families. Atomic training improves atomic tasks but gives little or no compound gain; at Level III it collapses toward zero on compounds. Compound training improves compounds and preserves or improves atomics. RL-Compound also degrades sharply with length at Level III, so non-local coordination is the hardest bottleneck. The BFCL pilot (simple_python as atomic; parallel and parallel_multiple as compound) shows the same asymmetry. The theoretical explanation is error compounding plus coverage shift over induced intermediate states.
- **Kong et al.**: In theory and in controlled experiments, SFT supplies raw "modules" inside compositional traces, and RL decomposes those traces into atomic modules and recombines them. Training on compound traces beats training on isolated modules. The recommended protocol: SFT should cover all atomic modules via compositional traces, and RL should focus on novel compositions outside SFT support.
- **Abdulsalam et al.**: In a fully auditable rewrite-grammar world, RL solves held-out problems that the pretrained model rarely solves even at large sampling budgets. RL first sharpens primitive reductions, then discovers reusable sequential and parallel compositions. RFT improves early, plateaus, and produces many invalid shortcut rewrites. Compositional strategies emerge only if pretraining organized primitives into reduction procedures.
- **Limitations / failure modes**: Data-structure domain; the tool-calling evidence is a pilot. Intermediate-state rewards were not used because requiring intermediate outputs hurt final accuracy.
- **How to reuse with easy seed tasks**: Spend the RL budget on composites built from skills you can already verify (chains, switches, branch–merge). Keep atomic items only as regression probes. For agents, compose single-call tool tasks into parallel and multi-tool workflows with normalized exact-match checking.

### Countdown skill composition — How Does RL Post-training Induce Skill Composition? A Case Study on Countdown (Simon Park et al., 2025)
Link: https://arxiv.org/abs/2512.01775. Authors: Park, Kaur, Arora.
- **Mechanism**: Countdown solutions are parsed into canonical expression trees, where each subtree is a reusable "skill". Patterns are grouped by root operator and by the operand split between subtrees into left-heavy, balanced and right-heavy shapes. The authors track the discovery (coverage) and mastery (precision) of each pattern during GRPO on Qwen2.5-1.5B/3B/7B. The reward is 0.1 for format plus 0.9 for a correct, valid expression.
- **How it makes tasks harder**: Larger n (more operands) and held-out tree shapes. An entire family of patterns (a base structure and all its extensions) is removed from training.
- **Correctness / verification**: Evaluate the expression, check it hits the target, and check number usage. The balanced dataset is built pattern-first: pick a pattern, plug in uniform random numbers, and keep the item if the result is a valid integer. This gives up to 4,000 examples per pattern, 418,619 training and 1,140 test items. A solution of the intended shape is therefore guaranteed to exist.
- **Difficulty control**: n and tree topology. Within a fixed n, balanced (shallow) patterns are mastered before unbalanced (deep) ones. Right-heavy patterns stay hardest even at the same depth as left-heavy ones, a "lookahead bottleneck": the model must commit to the root operator before generating the complex subroutine.
- **Reported results**: OOD generalization to larger n and to fully held-out pattern families. Coverage of a held-out sub-pattern emerges before its dependents. The standard Countdown dataset (Pan et al.) lacks 36 possible patterns, and training on it leaves most patterns unlearned. The 7B model eventually improves on right-heavy patterns at the cost of balanced and left-heavy ones. SFT without chain-of-thought averages below 10% at larger n. Llama models do not learn properly.
- **Limitations / failure modes**: Single task family. Aggregate pass@k hides the structure-dependent hierarchy.
- **How to reuse with easy seed tasks**: When complexifying arithmetic, planning or program seeds, sample the solution topology explicitly (generate from a target tree shape) and balance the dataset across shapes. Oversample right-heavy and deep patterns, and report per-shape accuracy, not just size-binned accuracy.

### OMEGA (added) — OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization (Yiyou Sun et al., 2025)
Link: https://arxiv.org/abs/2506.18880. Authors: Sun, Hu, Zhou, Zheng, Hajishirzi, Dziri, Song.
- **Mechanism**: 40 templated generators span arithmetic, algebra, combinatorics, number theory, geometry, and logic & puzzles. Each template is "single-scope" (one strategy) with a complexity parameter. Matched train/test splits probe three axes:
  - exploratory: the same template at higher complexity;
  - compositional: train on skill A and skill B separately, test on problems that require both;
  - transformative: test problems need an unconventional strategy.

  RL is GRPO on 1k problems with Qwen2.5-7B-Instruct and Qwen2.5-Math-7B.
- **How it makes tasks harder**: Complexity levels per template. The exploratory cutoff is chosen so the base model scores below 50% on the training range. Composition pairs skills, e.g., GCD with polynomial roots. Transformative splits require a reverse or clever insight.
- **Correctness / verification**: Programmatic generation, with solutions verified by symbolic, numerical or graphical methods.
- **Difficulty control**: The per-template complexity level. Frontier models' accuracy falls to zero as the level rises.
- **Reported results**:
  - Exploratory: RL on levels 1–2 lifts level 3; e.g., Zebra logic from a 30% base gains +61 points ID and +53 OOD. Geometry gains less (+31 ID, +8 OOD). Arithmetic-GCD level 3 goes from 6% to 10%, while level 5 stays at 3% even after training on levels 1–4, so the gains plateau.
  - Compositional: strong gains on the isolated skills (e.g., polygon rotation +~70 points) but little or no gain on their composition (e.g., about +6%; none for GCD+roots). Gains depend on choosing conceptually aligned skill pairs.
  - Transformative: out-of-distribution accuracy is usually 0% after RL. In matrix rank, RL dropped a 70% OOD base by 30 points.
- **Limitations / failure modes**: Composition is trained as isolated skills, never jointly. This is exactly the "decomposed" regime that He et al. show does not transfer. The study uses a single 7B model family.
- **How to reuse with easy seed tasks**: OMEGA's generators are a ready source of complexity ladders. The main lesson is to include composed A+B items in RL data, not A and B separately. Use its three split types as acceptance tests for any complexification pipeline.

### DELTA-Code — RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? (Yiyou Sun et al., 2025)
Link: https://arxiv.org/abs/2509.21016 (code: https://github.com/sunblaze-ucb/rl-grok-recipe). Authors: Sun, Cao, Huang, Bai, Hajishirzi, Dziri, Song.
- **Mechanism**: A benchmark of synthetic program-synthesis families. Manufactoria uses a novel custom syntax: build finite-state and tag-system-like "factories" that accept or reject colored tapes. It has 14 families in Basic, Easy, Medium and Hard tiers. BouncingSim asks for elastic-collision simulators in polygonal containers, with families ROT_OBJ, ROT_BOX, MOV_BOX, GRAVITY, MULTI_BOX and MULTI_OBJ. The learnability study uses a staged reward: warm up with the per-test-case pass rate (dense), then switch to a binary full-pass reward. Experience replay and feedback-in-the-loop are also tested.
- **How it makes tasks harder**: Parameterized templates with discrete knobs (start/end substrings, regex subpatterns, token rewrites), toggled or swapped with near neighbors. Numeric knobs (bitwise constants, thresholds, offsets, division factors) are jittered within valid sets, with guards for well-posedness. BouncingSim tiers vary polygon vertex count, speeds, box motion, gravity and object/box counts, from Basic to Extreme. Single-skill families are composed into multi-skill families (ROT_BOX + ROT_OBJ).
- **Correctness / verification**: Hidden unit tests from reference implementations. The reward is the per-test pass rate, then full pass.
- **Difficulty control**: Template knobs define the tiers. The OOD splits are exploratory (e.g., smaller containers, denser collisions), compositional (unseen skill combinations) and transformative (e.g., initial states that guarantee periodic trajectories).
- **Reported results**:
  - Manufactoria-HAS (742 train / 100 test): Qwen3-4B-Instruct-2507 has 0% full-pass at pass@128. After the warm-up, binary-reward RL stays below 1% full-pass for about 450 steps, then "groks" to near 100%.
  - Replay (up to 3 recent successful traces appended) groks earlier but converges more slowly. Feedback-in-the-loop groks earlier but is less stable.
  - The curriculum must be structurally aligned: training on REGEX before HAS transfers, while COMPR (similar difficulty, different structure) does not. Warm-up fails to escape zero on the harder PREPEND family.
  - BouncingSim (Qwen3-4B-Instruct, binary reward, 6k Basic items): the model groks near step 200 to about 0.7 full-pass. Exploratory: 70–85% on Basic (ID), 50–75% Easy, 15–50% Medium, single digits on Hard. Compositional (e.g., ROT_BOX+MOV_BOX, MULTI_BOX+MULTI_OBJ): 60–70% full-pass, versus near zero before RL. Transformative: near zero.
- **Limitations / failure modes**: Transformative generalization fails. Grokking needs long runs with near-zero reward. Curricula help only when they share structure with the target.
- **How to reuse with easy seed tasks**: For generated tasks pushed to pass@K = 0, give partial credit per sub-check (tests, subgoals), then switch to a strict reward. Log and replay rare successes. Isolate and track the hard subset separately so aggregate averages do not wash it out.

### Algebrarium — New Skills or Sharper Primitives? A Probabilistic Perspective on the Emergence of Reasoning in RLVR (Zhilin Wang et al., 2026)
Link: https://arxiv.org/abs/2602.08281. Authors: Wang, Li, Zhang, Wang, Zhang, Qu, Cheng.
- **Mechanism**: Algebrarium declares formal systems as (carrier set, operation signature, axioms). It instantiates four group types: an infinite Abelian group ("encrypted history navigation"), a finite Abelian group (Enigma 3-rotor, ℤ₂₆³), an infinite non-Abelian group (knitting instructions) and a finite non-Abelian group (Rubik's cube sequences). Training is GRPO for 100 steps on 1-hop atomic operations only. Tests chain 2–5 operations, with 200 balanced items per depth, on Llama, Qwen and Gemma models.
- **How it makes tasks harder**: Chain more atomic operations; switch to a novel custom system that forces zero-shot rule use.
- **Correctness / verification**: Symbolic evaluation from the declared axioms, with canonical forms (e.g., lexicographic ordering for commuting Rubik's faces). Exact match on the boxed answer.
- **Difficulty control**: Operation depth.
- **Reported results**: The model is P(q) ≈ ∏ P(s_j); e.g., 0.3⁵ ≈ 0.0024, which falls below the "Null Set" threshold ε ≈ 0.023. The "Feasible Set" threshold is Avg@128 ≥ 0.125. RLVR moved 22.6% of Null-Set tasks into feasibility, raising their mean success from 0 to about 0.60. Composite accuracy correlates with joint atomic accuracy (ρ ∈ [0.69, 0.96]). RLVR can sacrifice already-mastered skills: regressions are severe for Llama and Gemma and mild for Qwen.
- **Limitations / failure modes**: Framed as sharpening. It is an algebraic domain with an independence assumption. Global reward maximization erodes some skills.
- **How to reuse with easy seed tasks**: Measure per-step reliability on atoms. If composites are at 0%, first push atoms toward 1 (cheap 1-hop RL), then compose. Report the "null-set unlock rate", and watch for regressions on mastered atoms.

### Skill-Mix composition — Can Models Learn Skill Composition from Examples? (Haoyu Zhao et al., 2024)
Link: https://arxiv.org/abs/2409.19808 (NeurIPS 2024). Builds on Skill-Mix (Yu et al., arXiv 2310.17567).
- **Mechanism**: GPT-4 writes 13,957 short texts, each exhibiting a random subset of k language skills (rhetorical, literary, reasoning, theory of mind, common sense) on a random topic. LLaMA-2-13B-Chat and Mistral-7B-Instruct-v0.2 are fine-tuned on k ≤ 3 and tested at k = 4 and 5, with skills and topics split by category into training and held-out groups.
- **How it makes tasks harder**: Raise k. The space of skill combinations grows roughly as N^k.
- **Correctness / verification**: An LLM grader scores each required skill and topic adherence ("ratio of full marks"). GPT-4 is the main grader, and results are re-checked with Claude 3 Opus to rule out GPT-4 self-preference. *(Corrected: the draft attributed grading to GPT-4/LLaMA-2-70B, which was the original Skill-Mix setup.)*
- **Difficulty control**: k, and the held-out skill categories.
- **Reported results**: Training on k = 2 and 3 improves composition at k = 4 and 5, including for held-out skills. Training on texts with larger k is more data-efficient for learning composition.
- **Limitations / failure modes**: Judge-based reward is not verifiable and is hackable. The study is SFT-only.
- **How to reuse with easy seed tasks**: For non-verifiable domains (writing, instruction following), complexify by stacking simultaneous constraints or skills. Train at k = 2–3, evaluate at k = 4–5, and use per-constraint programmatic checks wherever a constraint is checkable.

### Grokked transformers (added) — Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization (Boshi Wang et al., 2024)
Link: https://arxiv.org/abs/2405.15071 (NeurIPS 2024).
- **Mechanism**: Transformers are trained from scratch on a random knowledge graph of atomic facts plus a fraction of "inferred" facts (two-hop compositions or comparisons deduced from them). The authors test in-distribution (ID) generalization (unseen inferred facts over seen atoms) and OOD generalization (inferred facts over atoms never seen in inferred form).
- **How it makes tasks harder**: Two-hop composition versus comparison; OOD bridging entities.
- **Correctness / verification**: Facts are deduced by construction from latent rules.
- **Difficulty control**: The ratio φ of inferred to atomic facts, and the total entity count.
- **Reported results**: Implicit reasoning emerges only via grokking (training far beyond overfitting). The speed of generalization tracks the inferred/atomic ratio, not absolute data size. OOD composition fails, while OOD comparison succeeds; this is explained by circuit configuration (the need for cross-layer knowledge sharing).
- **Limitations / failure modes**: Small models, synthetic knowledge, implicit (no chain-of-thought) reasoning.
- **How to reuse with easy seed tasks**: In SFT or mid-training mixes, the share of composed examples relative to atomic ones is a first-class knob. More composed data per atom speeds up generalization. Do not expect OOD composition of parametric facts without explicit chain-of-thought.

### Interplay of Pre-/Mid-Training and RL — On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models (Charlie Zhang et al., 2025)
Link: https://arxiv.org/abs/2512.07783. Authors: Zhang, Neubig, Yue.
- **Mechanism**: 100M-parameter Qwen2.5-style models are trained from scratch on a 30B-token GSM-Infinite-style corpus, split disjointly into pretraining, mid-training and RL portions. Problems are dependency DAGs of arithmetic relations, rendered into several contextual templates (animals–zoo, teachers–school, movie-festival). Pretraining uses 10B tokens at op 2–10. RL is GRPO on 200K samples from one of four regimes: op 7–10 (ID), 9–12 (mixed), 11–14 (edge) or 17–20 (hard).
- **How it makes tasks harder**: More operations (op = number of edges) and new surface contexts for the same abstract graph.
- **Correctness / verification**: The generator evaluates all nodes in topological order. Model traces are parsed into predicted graphs. For evaluation, a sample counts as correct only if every gold step and the final answer are correct. A process-aware reward variant is also used during RL.
- **Difficulty control**: ID op 2–10 (base pass@128 near saturated), OOD-edge op 11–14 (nonzero pass@128, low pass@1), OOD-hard op 15–20 (near-zero pass@128).
- **Reported results**:
  - RL on ID data never improves pass@128 beyond the base.
  - RL on edge data gives up to +42% pass@128. Training reward also stagnates when the data is too easy (op 7–10) or too hard (op 17–20).
  - Contextual transfer: with 0% or 0.1% context-B exposure in pretraining, RL does not transfer; 1% exposure (atomic op = 2 examples only) yields up to +60% pass@128.
  - Under fixed compute, mid-training plus heavy RL beats RL-only by +10.8% on OOD-hard.
  - Process-verified rewards add 4–5% pass@1 at op 15–20 and reduce shortcut solutions.
- **Limitations / failure modes**: Small from-scratch models. The ~1% threshold may not carry over to real pretraining mixtures.
- **How to reuse with easy seed tasks**: Estimate pass@1 and pass@k per seed family. Complexify (add ops or edges) until pass@1 is low but pass@k > 0, and re-evaluate the pool periodically as the policy improves (a self-paced curriculum). Seed about 1% atomic exposure for any new context in mid-training before RL. Where the generator exposes intermediate values, add step-level checks to the reward.

### Pass@k boundary critique — Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yang Yue et al., 2025)
Link: https://arxiv.org/abs/2504.13837 (NeurIPS 2025; the project page lists a Best Paper Runner-Up award).
- **Mechanism**: Base and RLVR models are compared with pass@k up to k = 128–1024, plus coverage and perplexity analysis, across several model families, six RL algorithms, and math, code and visual benchmarks.
- **How it makes tasks harder**: It does not; this is diagnostic only.
- **Correctness / verification**: Standard benchmarks.
- **Difficulty control**: None.
- **Reported results**:
  - Minerva with a 32B model: the base solves about 9% more problems than RL at k = 128.
  - Solvable-problem breakdown, base vs SimpleRLZoo: AIME24 63.3% both / 13.3% base-only / 0.0% RL-only / 23.3% neither; MATH500 92.4 / 3.6 / 1.0 / 3.0.
  - The sampling-efficiency gap (RL pass@1 vs base pass@k) stays above 40 points for all six algorithms (GRPO 43.9, RLOO 42.6).
  - Distillation lifts the whole pass@k curve.
- **Limitations / failure modes**: Uses standard, largely in-distribution data with relatively short training. Later work (ProRL, Interplay, f(g(x)), h1, and Curriculum RL 2606.22317 at +9.8 pass@256 over base) shows expansion when data is composed or at the edge.
- **How to reuse with easy seed tasks**: Use large-k pass@k and solvable-set Venn diagrams as the acceptance test for a complexified dataset. If RL-only coverage is ~0%, the data is too easy or too far out of reach.

### ProRL — ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models (Mingjie Liu et al., 2025)
Link: https://arxiv.org/abs/2505.24864 (model: https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B).
- **Mechanism**: More than 2k RL steps from DeepSeek-R1-Distill-Qwen-1.5B, using KL control plus periodic hard resets of the reference policy and optimizer state. The data is 136K verifiable problems: math 40K, code 24K, STEM 25K, Reasoning Gym logic 37K across 96 tasks, and instruction following 10K.
- **How it makes tasks harder**: Breadth of procedurally generated puzzle families plus training duration. Reasoning Gym generators have difficulty parameters.
- **Correctness / verification**: Reasoning Gym's own verifiers (some tasks accept multiple correct answers). Code reward is the fraction of tests passed, with all tests run.
- **Difficulty control**: Domain mix and duration.
- **Reported results**: Reported gains over the base model differ between sections of the paper. The intro gives math +14.7%, code +13.9%, logic +54.8%, STEM +25.1% and IF +18.1%. Section 3 gives +15.7 / +14.4 / +25.9 (GPQA) / +22.0 (IFEval) / +54.8. There is a significant negative correlation between base pass@128 and the pass@128 gain after RL: tasks where the base is strong show minimal or negative breadth gains, a narrowing boundary. The creativity index rises with training. The model solves the unseen Reasoning Gym task boxnet, where the base has zero capability.
- **Limitations / failure modes**: One 1.5B model and heavy compute. Familiar tasks can narrow.
- **How to reuse with easy seed tasks**: Aim long RL at families where the base is weak but nonzero, and keep adding procedural families so data does not saturate. Use reference resets for long runs.

### SFT Memorizes, RL Generalizes (added) — SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training (Tianzhe Chu et al., 2025)
Link: https://arxiv.org/abs/2501.17161.
- **Mechanism**: Two environments, each with a text (-L) and vision-language (-VL) version. GeneralPoints is a 24-style card arithmetic game. V-IRL is real-world visual navigation. RL is multi-turn with sequential revision against a verifier, starting from an SFT-initialized Llama-3.2-Vision-11B. Training uses one rule, and testing uses unseen rules or visuals.
- **How it makes tasks harder**: Rule variants (J/Q/K read as 11/12/13 versus all as 10), different textual action spaces, different card colors, and different cities.
- **Correctness / verification**: A programmatic game checker and navigation success.
- **Difficulty control**: Rule and visual shift, and the number of verification iterations.
- **Reported results**:
  - OOD rule variants: RL gives +3.5 on GP-L (11.5→15.0), +11.0 on V-IRL-L (80.8→91.8), +3.0 on GP-VL and +9.3 on V-IRL-VL.
  - SFT degrades OOD: −8.1 on GP-L (to 3.4) and −79.5 on V-IRL-L (to 1.3).
  - Visual OOD: RL gives +17.6 on GP-VL and +61.1 on V-IRL-VL, while SFT drops.
  - RL without SFT fails, because the base model cannot follow the format. More verification iterations improve generalization.
- **Limitations / failure modes**: A 2-environment study.
- **How to reuse with easy seed tasks**: Use SFT only to install format and primitives. Put the complexified and rule-shifted variants in RL, not in SFT. Hold out rule variants as OOD probes.

### h1 — h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning (Sumeet Ramesh Motwani et al., 2025)
Link: https://arxiv.org/abs/2510.07312. Authors: Motwani, Ivanova, Cai, Torr, Islam, Shah, Schroeder de Witt, London. *(Corrected: the draft omitted Charles London.)*
- **Mechanism**: Atomic GSM8K problems are chained so that sub-problem i+1 takes an adapted form of answer i. Adapters are identity or deterministic transforms such as scaling, affine maps and unit conversion. The whole chain is rendered as one prompt, and only the final answer is rewarded. GRPO on Qwen-2.5-3B-Instruct runs through stages h = 1…5, with 200 steps and 200 samples per horizon per stage. Replications use Qwen-2.5-7B-Instruct on composed MATH and Llama-3.2-3B-Instruct.
- **How it makes tasks harder**: Serial composition at arbitrary horizon.
- **Correctness / verification**: The final answer is propagated deterministically from verified atomic answers. Well-posedness filters check types, unit compatibility and numeric range, and deduplicate.
- **Difficulty control**: Horizon h. Theory: with outcome-only reward, full-horizon training needs N = Θ(α^−H) samples for a useful gradient, while the curriculum needs Θ(H).
- **Reported results**:
  - In-domain GSM8K chains: h = 2 rises from 35.06% (instruct) to 58.51% with a curriculum up to Len-5. Only-L1, Uniform-Mix and Only-Long baselines give no long-horizon gains at equal compute.
  - Transfer: AIME24 5.10→10.52 (+106%, avg@32), MATH-500 69.20, plus ReasoningGym and long-context gains.
  - The curriculum-trained model beats RLVR on standard GSM8K even at pass@128.
  - Long-horizon accuracy falls faster than independent per-step compounding predicts, so horizon-specific training is needed.
- **Limitations / failure modes**: Adapter chaining is somewhat artificial. The main results use a 3B instruct model.
- **How to reuse with easy seed tasks**: Any pool of easy verified problems becomes an unlimited horizon generator for free. Train with staged horizons, not a uniform mix. Skewing data toward short chains is acceptable if compute is added.

### ScaleLogic — How Deep Can LLMs Learn to Reason? Expressiveness Is Key (Tianle Wang et al., 2026)
Link: https://arxiv.org/abs/2605.06638 (v4). Authors: Wang, Wang, Lan, Wei, Zhang, Qiu, Saparov.
- **Mechanism**: For each of B candidate conclusions (default B = 4), a proof tree is grown backward from the conclusion down to depth D. One candidate stays provable. For each other candidate, a single axiom of its proof is removed or has one literal's polarity flipped, which severs the unique proof. Distracting rules are inserted without changing derivability. Items are rendered with random entity names and fake predicate words. Expressiveness climbs from implication-only through +∧, +¬, +∨ to +∀. RL uses DAPO on non-thinking Qwen3-4B, with replications on Qwen3-8B and Ministral3-3B and with GRPO and GSPO.
- **How it makes tasks harder**: Deeper proofs, richer logic, more candidates.
- **Correctness / verification**: Correct by construction, plus a Z3 audit: for 1,000 sampled items per configuration, Z3 confirms the marked candidate is entailed and all others are not. All passed.
- **Difficulty control**: D, expressiveness level, and B. *(Corrected: the default training distribution is uniform over depths. The threshold-triggered depth curriculum is a compared variant, not the default.)*
- **Reported results**:
  - Steps to reach 90% held-out pass@1 follow T ∝ D^γ (R² > 0.99). γ rises monotonically with expressiveness from 1.05 to 2.60.
  - Under +∧, the curriculum gives γ = 1.33 and difficult-only gives 2.36, with uniform in between. Under +∀ the curriculum also lowers γ. GRPO is steeper than DAPO or GSPO.
  - OOD depth: for deeper training depths (D_train = 12, 14), accuracy approaches random at about 3·D_train.
  - Transfer (8-benchmark mean): base 49.39%. Implication-only and +∧ plateau around 52%. +∀ reaches 60.05% at 414 steps (+10.66).
- **Limitations / failure modes**: The generator is logic-only and depth extrapolation is bounded. Results are shown up to 8B.
- **How to reuse with easy seed tasks**: For logic or rule seeds, generate backward from goals, create hard negatives with single-edit corruptions, add distractors, and audit with a solver. Raise expressiveness (new connective or constraint types), not only depth. Use a depth curriculum and budget compute superlinearly in depth.

### e3 — e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs (Amrith Setlur et al., 2025)
Link: https://arxiv.org/abs/2506.09026.
- **Mechanism**: The recipe has three parts:
  1. chain skills the base model is asymmetrically good at, e.g., verification (easier) followed by generation (harder), which implements in-context search;
  2. keep negative gradients from incorrect traces, which lengthen exploration (masking them, as in "GRPOMask", kills this);
  3. couple problem difficulty with the RL token budget in a stage-wise curriculum.
- **How it makes tasks harder**: Stages move from easy to medium/hard problems, with budgets from 8k to 16k tokens. Difficulty bins (DMath, from DeepScaleR) are set by Qwen-R1-Distilled-32B accuracy.
- **Correctness / verification**: Final-answer checking.
- **Difficulty control**: For each stage, choose the smallest budget B ≥ B₀ such that performance at 2B is at most κ times performance at B (κ = 1.2).
- **Reported results**: e3-1.7B from Qwen3-1.7B scores pass@1 43.8 on AIME'25 (base 35.5) and 24.7 on HMMT'25 (base 22.2). Pass@32 is 67.2 on AIME'25 (base 65.2) and 56.1 on HMMT'25 (base 54.9). *(Corrected: the draft reported 67.2/56.1 as accuracy at a 32k budget; they are pass@32 values from Table 1.)* The model extrapolates to 2× the training budget (32k). Training on hard problems at an 8k budget suppresses the long traces needed for exploration. Easy-only training at 8k gave the best OOD AIME'25 at 32k.
- **Limitations / failure modes**: Small-model study; the budget heuristic needs tuning.
- **How to reuse with easy seed tasks**: When you raise difficulty, raise the token budget with it. Keep a self-check or verification sub-skill inside composed tasks. Do not train the hardest tier under a tight budget.

### Illusion of Diminishing Returns (added) — The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs (Akshit Sinha et al., 2025)
Link: https://arxiv.org/abs/2509.09677 (ICLR 2026).
- **Mechanism**: The task isolates execution from planning. The model is given an in-context dictionary (five-letter words mapped to integers). Each turn it receives keys (the plan) and must retrieve their values and keep a running sum. The knobs are the number of turns and the turn complexity (keys per turn).
- **How it makes tasks harder**: More turns and more keys per turn.
- **Correctness / verification**: Exact arithmetic.
- **Difficulty control**: Horizon length = the first step where mean accuracy falls below a threshold. Under constant independent per-step accuracy p, the horizon grows hyperbolically in p once p is high.
- **Reported results**: Per-step accuracy degrades as the task progresses ("self-conditioning": errors in context raise the chance of further errors), and scale alone does not fix it. Thinking models fix self-conditioning. Without chain-of-thought, DeepSeek-V3 fails within 4 steps, while R1 executes over 100. GPT-5 thinking executes over 2,100 steps and Claude-4 Sonnet 432.
- **Limitations / failure modes**: Execution only, with a synthetic task.
- **How to reuse with easy seed tasks**: Horizon operators (chaining, more turns) test execution reliability. Include error-in-context variants, e.g., prefill a trajectory with an injected earlier mistake, to train robustness to self-conditioning.

### iGSM — Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process (Tian Ye et al., 2024)
Link: https://arxiv.org/abs/2407.20311 (ICML 2024 tutorial).
- **Mechanism**: A structure graph (4 hierarchical categorizations, each with 4 layers of 100 items) plus a random dependency DAG over instance and abstract parameters. A parameter depending on t others costs max{1, t−1} operations. Problem difficulty op is the sum over necessary parameters. Unnecessary parameters and edges are added as distractors. GPT-2-style models are pretrained from scratch.
- **How it makes tasks harder**: Larger op and more irrelevant parameters.
- **Correctness / verification**: Arithmetic mod 23 on the DAG. Solution-template hashes strictly separate train and test.
- **Difficulty control**: iGSM-med trains on op ≤ 15 and tests on op 20–23. iGSM-hard trains on op ≤ 21 and tests on op 28–32.
- **Reported results**: 99% in-distribution accuracy, and OOD generalization to longer op than any seen in training. The model mostly produces shortest solutions ("level-1" reasoning: only necessary parameters). Probing shows it pre-computes all-pair dependencies ("level-2"). Errors correlate with planning mistakes. Depth matters: a 16-layer, 576-dim model beats a 4-layer, 1920-dim model on harder problems.
- **Limitations / failure modes**: Small from-scratch models, modular arithmetic.
- **How to reuse with easy seed tasks**: A DAG generator with an op knob plus distractor parameters turns any word-problem seed into a difficulty ladder. Hash solution templates to prevent train/test leakage.

### Faith and Fate — Faith and Fate: Limits of Transformers on Compositionality (Nouha Dziri et al., 2023)
Link: https://arxiv.org/abs/2305.18654 (NeurIPS 2023).
- **Mechanism**: Tasks (multi-digit multiplication, logic grid puzzles, a dynamic-programming problem) are formalized as computation graphs. Complexity is measured by reasoning depth (the longest path), reasoning width, and average parallelism (nodes divided by depth). The authors test zero-shot, few-shot and exhaustive fine-tuning of GPT-3 (text-davinci-003).
- **How it makes tasks harder**: Larger operands, grids and sequence lengths, i.e., deeper and wider graphs.
- **Correctness / verification**: Algorithmic ground truth.
- **Difficulty control**: Graph depth, width and parallelism.
- **Reported results**: Off-the-shelf ChatGPT and GPT-4 reach 55% and 59% on 3-digit × 3-digit multiplication. Accuracy decays to near zero with problem size and correlates negatively with average parallelism. Exhaustive fine-tuning gives high in-domain accuracy but a sharp OOD drop. The multiplication QA fine-tune covered up to 4-digit × 2-digit (1.8M pairs), and extended training for grokking did not help. GPT-2-XL trained from scratch on up to 4×4 (90M examples) with digit tokenization still fails 3×3 test items. Errors are mostly propagation errors at deeper graph layers. *(Corrected: the draft's "~90%+ in range, ~0% at 5×5 for GPT-3 trained to 4×4" figures could not be confirmed and were removed.)*
- **Limitations / failure modes**: Pre-reasoning-model era.
- **How to reuse with easy seed tasks**: Parameterize complexity by computation-graph depth and width, not by surface length, and hold out larger graphs.

### Illusion of Thinking (added) — The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity (Parshin Shojaee et al., 2025); with Lawsen's comment (arXiv 2506.09250)
Link: https://arxiv.org/abs/2506.06941 (NeurIPS 2025); comment: https://arxiv.org/abs/2506.09250.
- **Mechanism**: Four puzzle environments with simulators: Tower of Hanoi, Checker Jumping, River Crossing and Blocks World. Complexity is scaled by N (disks, checkers, actor/agent pairs, blocks) while the logic is held fixed. Reasoning and non-reasoning models are compared under matched token budgets.
- **How it makes tasks harder**: Increase N, which increases compositional depth (the minimum number of moves).
- **Correctness / verification**: The simulator checks every move.
- **Difficulty control**: N.
- **Reported results**: Three regimes: non-thinking models win at low complexity, thinking models win at medium, and both collapse at high complexity. Reasoning effort rises and then falls near collapse despite remaining budget. Supplying the algorithm (Hanoi, Checker) does not move the collapse point.
- **Lawsen's rebuttal**:
  - Some River Crossing instances are unsolvable: with boat capacity b = 3, no solution exists for N ≥ 6.
  - Tower of Hanoi move lists can exceed output-token limits, and models explicitly say they are truncating.
  - Asking for a generating function or program instead of an exhaustive move list changes conclusions.
- **Limitations / failure modes**: Evaluation artifacts can masquerade as capability limits.
- **How to reuse with easy seed tasks**: Every complexity knob needs (a) a solvability check (solver or construction), (b) an output-length cap below the context limit, and (c) optionally a compressed answer format (program or final state) so that difficulty measures reasoning rather than typing.

### RLVE — RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments (Zhiyuan Zeng et al., 2025)
Link: https://arxiv.org/abs/2511.07317 (ICML 2026).
- **Mechanism**: 400 hand-engineered environments (RLVE-Gym). Each has an input template, a generator indexed by an integer difficulty d, and an algorithmic verifier. Each environment keeps a window [ℓ, h], starting at ℓ = h = 0. Once the number of samples at level h reaches the minimum (8 × rollouts per problem) and accuracy exceeds τ_acc = 0.9, h is incremented. A sliding window d_Δ = 4 keeps ℓ ≥ h − d_Δ + 1.
- **How it makes tasks harder**: The generator's integer difficulty ratchet, plus adding environments.
- **Correctness / verification**: Verifiers are programs. The environment verifies rather than solves, which exploits cases where verification is far cheaper than solving (e.g., NP-complete problems).
- **Difficulty control**: The accuracy-triggered promotion above.
- **Reported results**:
  - From ProRL-1.5B-v2 (already trained for more than 20k H100-h to saturation), joint training on all 400 environments gives +3.37 absolute over 6 benchmarks in about 1,100 H100-h. Continuing the original RL gives +0.49 with 3,600 H100-h.
  - Static low-ceiling difficulty drives the effective-prompt ratio (non-identical rewards within a group) to zero. A static high ceiling keeps it nonzero but low.
  - Environment scaling: nested collections of 1, 4, 16 and 256 environments, evaluated on 50 held-out environments, improve monotonically.
  - From OpenThinker3-1.5B, RLVE beats RLVR on DeepMath-103K by about 2 points absolute. DeepMath-103K cost about $138K and 127K GPU-h to build. *(Corrected: the draft's "1→4→16→256→400" environment-scaling series; the scaling study used 1/4/16/256, and 400 was used for the ProRL continuation.)*
- **Limitations / failure modes**: Needs a monotone difficulty parameter and hand-built environments.
- **How to reuse with easy seed tasks**: Wrap each easy seed family in a generator with an integer knob. Apply the 90%-promotion rule with a window, and invest in many families rather than more items from one.

### Online difficulty filtering — Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning (Sanghwan Bae et al., 2025)
Link: https://arxiv.org/abs/2504.03380 (EACL 2026). Related: Foster et al., *Learning to Reason at the Frontier of Learnability* (arXiv 2502.12272); DAPO (arXiv 2503.14476); ScaleRL, *The Art of Scaling RL Compute for LLMs* (arXiv 2510.13786).
- **Mechanism**: The expected policy improvement is lower-bounded by the variance of task-level success probabilities, so intermediate-difficulty prompts maximize learning. Balanced online filtering maximizes this bound.
  - Foster et al. show that under PPO and VinePPO many questions are all-solved or none-solved, and they prioritize high success-variance p(1−p) ("sampling for learnability").
  - DAPO resamples to replace zero-variance groups.
  - ScaleRL drops zero-variance prompts from the loss, and with "No-Positive-Resampling" permanently retires prompts with historical pass rate ≥ 0.9.
- **How it makes tasks harder**: It does not generate tasks; it selects them.
- **Correctness / verification**: Standard verifiable rewards.
- **Difficulty control**: The online empirical pass rate.
- **Reported results**: Balanced filtering gives up to +12% in less than half the GRPO training steps. In ScaleRL, both zero-variance filtering and No-Positive-Resampling raise the asymptotic pass rate. ScaleRL also finds that curriculum-type choices mostly change compute efficiency, not the ceiling.
- **Limitations / failure modes**: If most seeds are saturated, filtering leaves too little data. This is the reason to complexify.
- **How to reuse with easy seed tasks**: Log pass@G for each seed. Items at 100% go to the complexification operators; items at 0% go to scaffolding; train on the 0 < p < 1 band.

### Hard Examples Are All You Need — Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets (Benjamin Pikus et al., 2025)
Link: https://arxiv.org/abs/2508.14094.
- **Mechanism**: Under a fixed budget, compare GRPO on the easiest, medium, hardest or random 10% of prompts, ranked by base-model failure rate. Data: GSM8K and BIG-Bench Hard Tracking Shuffled Objects. Models: Qwen3-4B, Qwen3-14B and Phi-4, plus Llama3.1-8B. OOD test: AIME2025-I. *(Corrected domain: the draft said "math reasoning" only.)*
- **How it makes tasks harder**: Selection of the hard tail.
- **Correctness / verification**: Ground-truth answers.
- **Difficulty control**: Base-model success rate.
- **Reported results**: The hardest 10% gives gains of up to 47%, versus 3–15% for easy subsets. Hard examples keep mixed outcomes, and therefore advantage, throughout training. Only hard-trained models make meaningful AIME2025 gains.
- **Limitations / failure modes**: Selection assumes a hard tail exists; small benchmarks.
- **How to reuse with easy seed tasks**: If you have no hard tail, generating one through composition is the highest-value use of budget.

### DARS (added) — Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration (Zhicheng Yang et al., 2025)
Link: https://arxiv.org/abs/2508.13755.
- **Mechanism**: The "cumulative advantage" (the sum of |advantages| in a group) under GRPO-family estimators is maximal at medium accuracy and small for low-accuracy prompts, so hard prompts are under-weighted. Difficulty-Adaptive Rollout Sampling runs a cheap pre-rollout to estimate per-prompt accuracy, then allocates extra rollouts to low-accuracy prompts in multiple stages so that their cumulative advantage increases with difficulty. DARS-Breadth adds large full-batch updates.
- **How it makes tasks harder**: Not a generator. It changes how much gradient mass hard items receive.
- **Correctness / verification**: Standard verifiers.
- **Difficulty control**: Per-prompt accuracy estimates.
- **Reported results** (Qwen2.5-Math-1.5B/7B): Naively scaling rollout size helps pass@1 but not reliably pass@K. DARS improves pass@K. Scaling batch size ("breadth", e.g., 3072 with full-batch updates) raises pass@1 via higher token entropy. The two effects are complementary.
- **Limitations / failure modes**: Extra rollout compute; math only.
- **How to reuse with easy seed tasks**: After complexifying, still re-weight. Newly hard items with low accuracy get little gradient under vanilla GRPO unless you give them more rollouts.

### Easy-to-Hard Generalization — Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision (Zhiqing Sun et al., 2024)
Link: https://arxiv.org/abs/2403.09472 (NeurIPS 2024).
- **Mechanism**: Train process-supervised reward models (PRMs) using human supervision only on easy MATH problems (levels 1–3). Use them to score policy solutions on levels 4–5, via reranking or as the RL reward.
- **How it makes tasks harder**: The evaluator transfers from easy to hard problems.
- **Correctness / verification**: The PRM estimates correctness on hard items; final evaluation uses ground truth.
- **Difficulty control**: MATH levels.
- **Reported results**: A process-supervised 7B RL model reaches 34.0% on MATH500, and 34B with rerank@1024 reaches 52.5%, with human supervision only on easy problems.
- **Limitations / failure modes**: PRM reliability degrades further from the training distribution, and PRMs are hackable under RL.
- **How to reuse with easy seed tasks**: When complexified items lack cheap exact answers, train verifiers or PRMs on seeds you can label, score the hard variants, and spot-check with an oracle.

### Unreasonable Effectiveness of Easy Data — The Unreasonable Effectiveness of Easy Training Data for Hard Tasks (Peter Hase et al., 2024)
Link: https://arxiv.org/abs/2401.06751 (ACL 2024; code https://github.com/allenai/easy-to-hard-generalization).
- **Mechanism**: ICL, linear probes and QLoRA on models up to 70B, trained on easy-only versus hard-only data. Four QA datasets (grade-3 science to college STEM, and trivia). Hardness uses seven measures: six human (e.g., grade level) and one loss-based.
- **How it makes tasks harder**: Split by difficulty.
- **Correctness / verification**: Existing labels.
- **Difficulty control**: Hardness metrics.
- **Reported results**: Easy-to-hard generalization is often as good as a hard-data oracle. Because hard labels are noisier and costlier, collecting easy data can be preferable even when the target is hard data.
- **Limitations / failure modes**: This elicits knowledge the model already has; it does not build multi-step procedures. It contrasts with RL findings favoring hard or edge data.
- **How to reuse with easy seed tasks**: For eliciting knowledge with SFT, easy data may suffice. For building procedures with RL, generate edge or composed data.

### Self-Improving Transformers — Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges (Nayoung Lee et al., 2025)
Link: https://arxiv.org/abs/2502.01612.
- **Mechanism**: Each round, the model labels problems slightly harder than its current range (+1 digit, length, hop, or 3 nodes). Labels are filtered and added to training, and the model retrains. The filters are (i) relative length filtering, which drops outputs more than a threshold shorter than the longest in the batch (wrong OOD answers tend to be short), and (ii) majority voting across models trained with different seeds.
- **How it makes tasks harder**: A size ratchet; an accelerated schedule samples several levels per round.
- **Correctness / verification**: No oracle. Consensus plus length filtering; a verifier variant is also tested for mazes.
- **Difficulty control**: Increment size per round.
- **Reported results**:
  - Reverse addition: trained on 1–16 digits, near-perfect to 100+ digits.
  - Copy and reverse: from length 10 to over 120 after about 100 rounds.
  - Forward addition with length filtering: from 10 digits to about 70 at >98% over 60 rounds.
  - CoT multiplication: without filtering, only 13.7% at round 7 on the next size. With majority vote plus length filtering, near-perfect up to 9×9 at round 31. The accelerated schedule reaches 10×10 in 19 rounds.
  - Majority voting raises 5-by-6 label accuracy from 31% to 93.3%.
  - Mazes: from 9 to 30 hops with majority voting.

  *(Corrected: the draft's "5×5→10×10 in 41 rounds" is not in the paper.)*
- **Limitations / failure modes**: Unfiltered errors cause "error avalanches". Small algorithmic transformers.
- **How to reuse with easy seed tasks**: If no oracle exists, raise difficulty in small steps, label with multi-seed or multi-model consensus plus cheap sanity filters, and monitor label accuracy on a gold subsample every round.

### s1K difficulty-driven SFT — s1: Simple test-time scaling (Niklas Muennighoff et al., 2025); with LIMO (arXiv 2502.03387) and OpenThoughts (arXiv 2506.04178)
Link: https://arxiv.org/abs/2501.19393 (code https://github.com/simplescaling/s1).
- **Mechanism**: Start from 59,029 questions from 16 sources, with traces from Gemini Flash Thinking. Filter in order:
  1. quality;
  2. difficulty: drop questions solved by Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct (Claude 3.5 Sonnet grades against the reference), with longer traces treated as harder;
  3. diversity: stratify by domain.

  This yields 1,000 examples; the model is SFT'd Qwen2.5-32B-Instruct.
- **How it makes tasks harder**: Selection of hard items.
- **Correctness / verification**: Distilled teacher traces.
- **Difficulty control**: The reference-model failure filter plus trace length.
- **Reported results**:
  - AIME24 (budget forcing): 1K-random 36.7, 1K-diverse 26.7, 1K-longest 33.3, s1K 50.0, full 59K 53.3. The 59K-vs-s1K difference is not significant per their bootstrap CI (−13.3% to +20.0%).
  - LIMO: 63.3% AIME24 and 95.6% MATH500, using 1% of the training data of prior approaches.
  - OpenThoughts (1,000+ ablations): sampling multiple answers per question (up to 16×) is an effective scale lever; no verification or answer-filtering method helped significantly; 1–2 high-quality question sources beat 8–16 diverse ones; filtering questions by LLM-labeled difficulty or response length beats embedding or fastText filters.
- **Limitations / failure modes**: Needs a teacher that can solve the hard items. SFT on small hard sets can hurt OOD on narrow synthetic tasks (Rethinking Easy-to-Hard found SFT sometimes drops below zero-shot on PartWhole and Knights & Knaves).
- **How to reuse with easy seed tasks**: For SFT on complexified items, keep only items your model fails, sample several teacher traces per item, and prioritize difficulty over breadth of sources.

### Easy-to-hard theory — Learning Compositional Functions with Transformers from Easy-to-Hard Data (Zixuan Wang et al., 2025); with Rajaraman et al., Curriculum I (arXiv 2603.18325) and II (arXiv 2606.27721)
Link: https://arxiv.org/abs/2505.23683 (COLT 2025).
- **Mechanism**: The k-fold composition task interleaves k input permutations and k hidden permutations and is expressible by an O(log k)-depth transformer. Any SQ learner with polynomially many queries needs sample size exponential in k. Gradient descent on an O(log k)-depth transformer learns it with poly(k) samples if the data contains k′ ≤ k compositions, either as an increasing-difficulty curriculum or all mixed together.
  - Curriculum I: autocurriculum (the model's own performance chooses the prompts) needs exponentially fewer SFT demonstrations than non-adaptive training. For RL it decouples compute from reference-model quality, reducing it to a burn-in cost.
  - Curriculum II (semiautomata / T-step state tracking): recursive decomposition into sub-problems gives SFT with 2^O(√log T) supervision tokens versus the Ω(T) barrier. For RLVR it weakens the requirement from reference-model coverage at length T to coverage at block length B ≪ T.
- **How it makes tasks harder**: Composition depth k and sequence length T.
- **Correctness / verification**: Theoretical.
- **Difficulty control**: k, T and B.
- **Reported results**: The bounds above.
- **Limitations / failure modes**: Stylized tasks and architectures.
- **How to reuse with easy seed tasks**: Never train only on the deepest composites; always include shallower instances, in a mix or a schedule. For RL, make sure the base covers the short blocks that long tasks decompose into.

### Rethinking Easy-to-Hard — Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning (Maximilian Mordig et al., 2026)
Link: https://arxiv.org/abs/2603.27226. Authors: Mordig, Opedal, Liu, Schölkopf.
- **Mechanism**: Five sampling strategies under fixed compute (uniform, easy→hard variants, hard→easy variants, mixed-range), with SFT on CoT traces and with GRPO/PPO. Models: Llama3.2-1B/3B, Qwen3-0.6B/1.7B/4B and Gemma2-9B-it.
- **How it makes tasks harder**: MathGAP LinearDepth (axioms and inference depth), MathGAP PartWhole, and Knights & Knaves (number of characters, unique-solution instances only).
- **Correctness / verification**: Unique-solution generators.
- **Difficulty control**: ID/OOD bands: LinearDepth 1–5 / 6–18; PartWhole 2–10 / 11–19; K&K 3–6 / 7–10. *(Corrected: the draft said "LinearDepth ≥ 11".)*
- **Reported results**: No robust gain from difficulty ordering in accuracy or response length. Example (Llama3.2-1B, GRPO): LinearDepth 0.27 standard vs 0.28 curriculum; PartWhole 0.53 vs 0.52. There is a persistent ID/OOD gap. RL improves OOD when the base has some initial capability. SFT sometimes degrades OOD below zero-shot.
- **Limitations / failure modes**: Small models and deductive tasks. Every level carried learnable signal, which is exactly the regime where ordering should not matter.
- **How to reuse with easy seed tasks**: Spend effort on the difficulty band and on composition types, not on ordering. Stage only the tiers that would otherwise get zero reward.

### SOAR — Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability (Shobhita Sundaram et al., 2026)
Link: https://arxiv.org/abs/2601.18778 (ICML 2026).
- **Mechanism**: Asymmetric teacher–student self-play with bilevel meta-RL. The student is Llama-3.2-3B-Instruct, with Llama-3.1-8B-Instruct in ablations. Hard sets are MATH, HARP and OlympiadBench items at 0/128 success ("fail@128"), split 50/50 into train and test.
  - Each outer iteration, the teacher samples n = 64 synthetic question–answer pairs.
  - The student trains on them.
  - The teacher's reward is the improvement in the student's greedy success on 64 real fail@128 "reward questions".
  - A student is promoted when it improves, and the resulting "promotion questions" (PQ) are kept.
- **How it makes tasks harder**: It generates stepping stones that make hard problems learnable, which is complexification in reverse.
- **Correctness / verification**: Synthetic answers are not verified. Usefulness is measured end-to-end against real ground truth.
- **Difficulty control**: Implicit, through the grounded reward.
- **Reported results**:
  - About 4× pass@1 and 2× pass@32 on fail@128 MATH; about 2× pass@1 and 1.5× pass@32 on HARP.
  - PQ gives +9.3 pass@32 on MATH and +4.2 on HARP over Hard-Only. Hard-Only with 4× compute (group 128) gives only +2.8, and extending Hard-Only from 1,500 to 6,500 steps does not help.
  - Transfers to OlympiadBench.
  - Only 32.8% of PQ problems have fully correct solutions and 63% are well-posed. Adding well-posed items with wrong answers still improves results.
  - Grounded rewards beat intrinsic learnability rewards, which are unstable and prone to diversity collapse.
- **Limitations / failure modes**: An expensive bilevel loop that needs a labeled hard set.
- **How to reuse with easy seed tasks**: For the hard end (0% items), train a generator whose reward is downstream student gain. For the final RL objective, keep exact verification on real targets.

### EvoTD — Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling (Liqin Ye et al., 2026)
Link: https://arxiv.org/abs/2605.11666 (code https://github.com/liqinye/EvoTD).
- **Mechanism**: Tasks are program–input–output triplets in deduction, abduction and induction formats (following Absolute Zero). A skill bank of algorithmic patterns (e.g., two pointers, Dijkstra) is extracted from seed data, and one baseline program is seeded per skill.
  - Attribute Mutation: the proposer (o4-mini) audits which complexity attributes apply (input size, tree depth, structural constraints) and produces n = 10 variants per parent, intensifying them while keeping the logic backbone.
  - Skill Crossover: synthesizes tasks where the selected skills are essential and interdependent, not merely concatenated.
  - Fitness check: executability, skill alignment, and a zone-of-proximal-development (ZPD) learnability filter on the current solver's avg@k (non-trivial yet solvable).
- **How it makes tasks harder**: Mutation (vertical) and crossover (horizontal).
- **Correctness / verification**: A Python executor runs the proposer's program on its input to get the deterministic output o, and checks syntax and termination.
- **Difficulty control**: Mutation plus the ZPD filter, which moves with the solver.
- **Reported results**:
  - Qwen3-4B (thinking): math average pass@1 33.8 → 40.2 at iteration 3; AIME24 34.0 → 42.2 (+8.2); AIME25 21.8 → 29.0 (+7.2).
  - Largest relative gain: 90% on LiveCodeBench v6 Hard.
  - Evol-Instruct raises pass@1 but its pass@8 average (54.2) falls below the untuned backbone (55.1). EvoTD gains +4 pass@1 and +7.4 pass@8.
  - Leads SPIRAL and Agent0 by +0.4 to +3.7 across four backbones.
  - Mutation lowers solver mean@10 from 56.7% to 51.4%. Synthesis is 13.5–16.1% of runtime.
- **Limitations / failure modes**: Code-expressible tasks only; relies on an LLM proposer.
- **How to reuse with easy seed tasks**: Tag seeds with skills, express them as executable programs, mutate attributes, cross over skills, and gate each round by ZPD. Monitor pass@k for homogeneity collapse.

### SynthLLM — Scaling Laws of Synthetic Data for Language Models (Zeyu Qin et al., 2025)
Link: https://arxiv.org/abs/2503.19551 (COLM 2025).
- **Mechanism**: Filter math documents from Fineweb-Edu, then generate at three levels:
  - Level 1: extract or rephrase questions.
  - Level 2: extract topics and key concepts from one document and recombine them.
  - Level 3: build a global concept graph weighted by co-occurrence, sample a topic, take 1–2 random-walk steps to related topics, then 3–4 steps in the key-concept subgraph, and write questions over the sampled concept set.

  Answers come from an open LLM (e.g., Qwen2.5-Math-72B-Instruct).
- **How it makes tasks harder**: Multi-concept, cross-document recombination.
- **Correctness / verification**: No answer verification. An ablation that filtered 8 candidate answers per question with a Llama-3.1-70B judge moved MATH only from 42.0 to 42.2.
- **Difficulty control**: Recombination level. Median question lengths are 32/66/80 tokens and median answer lengths 358/497/545 tokens for L1/L2/L3.
- **Reported results**: Follows a rectified scaling law; gains plateau near 300B tokens; an 8B model peaks near 1T tokens and a 3B model needs 4T. At 150 questions per document, L2 and L3 keep improving while rephrasing and persona augmentation saturate.
- **Limitations / failure modes**: Answers are unverified, which makes it unsuitable as-is for RLVR.
- **How to reuse with easy seed tasks**: For open-domain seeds, recombine distant concepts to raise difficulty and diversity. Add solver consensus or verification before RL use.

### Spurious rewards — Spurious Rewards: Rethinking Training Signals in RLVR (Rulin Shao et al., 2025); with Wu et al., RandomCalculation (arXiv 2507.10532, AAAI 2026)
Link: https://arxiv.org/abs/2506.10947.
- **Mechanism**: GRPO is run with random, incorrect or format-only rewards. Gains arise from a clipping bias that amplifies high-prior pretrained behaviors such as "code reasoning". Wu et al. trace the Qwen2.5 anomaly to benchmark contamination and build RandomCalculation, a generator of fresh arithmetic of arbitrary length and difficulty.
- **How it makes tasks harder**: RandomCalculation's expression length and complexity.
- **Correctness / verification**: Answers are computed for never-published expressions.
- **Difficulty control**: Procedural.
- **Reported results**: Random rewards give +21.4 MATH-500 points on Qwen2.5-Math-7B, versus +29.1 with ground truth. Code-reasoning frequency rises from 65% to over 90%. The effect fails on Llama3 and OLMo2. On RandomCalculation only accurate rewards give steady gains beyond the base boundary.
- **Limitations / failure modes**: Model-specific; the mechanism is debated.
- **How to reuse with easy seed tasks**: Validate every complexification pipeline on freshly generated items, with a random-reward control and at least two model families.

### Model collapse — The Curse of Recursion: Training on Generated Data Makes Models Forget (Ilia Shumailov et al., 2023; Nature 2024 as "AI models collapse when trained on recursively generated data")
Link: https://arxiv.org/abs/2305.17493. Related: Gerstgrasser et al. (arXiv 2404.01413); Feng et al. (arXiv 2406.07515); Dohmatob et al., *Strong Model Collapse* (arXiv 2410.04840).
- **Mechanism**: Recursive training on the previous model's samples, with real data replaced, irreversibly loses the tails (shown for LLMs, VAEs and GMMs).
  - Gerstgrasser: accumulating synthetic data alongside the real data keeps test error bounded, independent of the number of iterations.
  - Feng: verifier-based selection, even with imperfect verifiers, prevents collapse (matrix eigenvalues, news summarization).
  - Dohmatob: in regression, even about 1% synthetic data can cause collapse, and larger models can amplify it below the interpolation threshold.
- **How it makes tasks harder**: Not applicable (failure mode).
- **Correctness / verification**: Mitigations are keeping real data and verifying.
- **Difficulty control**: n/a.
- **Reported results**: As above.
- **Limitations / failure modes**: Mostly pure self-consumption. RLVR with exact verifiers is partly protected but still loses diversity.
- **How to reuse with easy seed tasks**: In iterative complexify→train loops, never replace seeds (accumulate them), gate items with verifiers, and track diversity (pass@k, distinct strategies, answer entropy) each round.

---

## Complexification operators from this area

1. **Functional composition / nesting depth**
   - What it does: Feeds the output of one known skill into another, h = g∘f, with depth as the knob.
   - Example: easy is `reverse_words(x)`; hard is `func_3(func_16(func_15(x)))` with definitions hidden.
   - How to keep it verifiable: Execute the composed program. Split atoms into train and held-out sets to test transfer. Atoms must already be reliable.
   - Sources: 2509.25123, 2602.08281, 2311.12997.

2. **Serial chaining with answer passing (horizon extension)**
   - What it does: Problem i+1 consumes an adapted answer i; only the final answer is rewarded.
   - Example: easy is one GSM8K problem; hard is a 5-link chain (×3 scaling, then unit conversion, then an affine map…).
   - How to keep it verifiable: Propagate ground truth through the adapters. Check types, units and ranges, and deduplicate. Use a staged horizon curriculum so reward is not about 0. Add error-in-context variants to fight self-conditioning.
   - Sources: 2510.07312, 2509.09677.

3. **Dependency-graph topology escalation (chain → multi-skill chain → branch–merge)**
   - What it does: Composes skills over increasingly non-local dependency graphs.
   - Example: easy is 8 BST inserts; hard is inserts and deletes on two structures whose results merge into a third query.
   - How to keep it verifiable: Deterministic simulators with fixed tie-breaking so the output is unique. Serialize outputs canonically.
   - Sources: 2609.19465, 2606.18089.

4. **DAG op-count scaling with distractors and surface re-rendering**
   - What it does: More edges and deeper dependencies, irrelevant parameters, and the same abstract graph rendered in new contexts.
   - Example: easy is a 3-op zoo problem; hard is a 14-op school problem with 6 unnecessary parameters.
   - How to keep it verifiable: Compute the answer from the DAG (mod-p arithmetic avoids big numbers). Hash solution templates to separate train and test. Parse traces against the DAG for process reward.
   - Sources: 2407.20311, 2512.07783, 2305.18654.

5. **Proof-depth scaling plus expressiveness escalation**
   - What it does: Builds proofs backward to depth D, and separately enriches the logic (→, then ∧, ¬, ∨, ∀).
   - Example: easy is a depth-2 implication chain; hard is depth 12 with disjunction elimination and universal instantiation, among 4 candidates.
   - How to keep it verifiable: Backward construction with a unique proof per candidate. Audit with Z3. Prefer adding operator types over adding depth alone.
   - Sources: 2605.06638.

6. **Single-edit corruption for hard negatives, plus candidate-set growth**
   - What it does: Makes near-miss distractors by removing or flipping one premise, and raises the number of candidates B.
   - Example: easy is a binary provable/unprovable question; hard is 4+ candidates where every false one differs by a single corrupted axiom.
   - How to keep it verifiable: Construction guarantees non-derivability because the proof was unique. Solver audit.
   - Sources: 2605.06638.

7. **Instance-size ratchet**
   - What it does: Raises a scalar size (digits, length, nodes, hops, variables) step by step.
   - Example: easy is 8-digit addition or a 9-hop maze; hard is 100-digit addition or a 30-hop maze.
   - How to keep it verifiable: Use a program oracle where possible. Otherwise use multi-seed majority vote plus length filters, and track label accuracy per round. Cap the output length the verifier expects.
   - Sources: 2502.01612, 2511.07317, 2505.16368.

8. **Solution-topology targeting**
   - What it does: Fixes the primitives and changes the required solution-tree shape (balanced → left-heavy → right-heavy, deeper).
   - Example: easy is Countdown with 3 numbers and a balanced tree; hard is 6 numbers whose only solutions are right-heavy.
   - How to keep it verifiable: Generate the target from a sampled tree of the desired shape. Balance the dataset across patterns.
   - Sources: 2512.01775.

9. **Skill crossover (k-skill combination)**
   - What it does: Requires several skills at once, each essential and interdependent.
   - Example: easy is a binary-search task, or text showing 1 rhetorical skill; hard is Dijkstra over a graph built by a two-pointer sweep, or text exhibiting k = 5 skills.
   - How to keep it verifiable: For code, the proposer writes the program and input and the executor produces the output. For text, use per-skill rubric grading with cross-grader checks. Include composed A+B items in training; A and B separately is not enough (OMEGA).
   - Sources: 2605.11666, 2409.19808, 2506.18880, 2503.19551.

10. **Parametric mutation of constraints (template knobs)**
    - What it does: Keeps the logical backbone and tightens or jitters discrete and numeric knobs.
    - Example: easy is "accept tapes starting with R"; hard is "accept tapes containing the subsequence GGRBB and whose binary value exceeds 27".
    - How to keep it verifiable: Reference implementation plus hidden tests. Guards keep the task well-posed and nontrivial. For pass@K = 0 families, use a per-test partial reward first.
    - Sources: 2509.21016, 2605.11666.

11. **Asymmetric skill chaining with a coupled budget**
    - What it does: Rewards chains of an easier skill (verify) with a harder one (generate), and grows the token budget with difficulty.
    - Example: easy is an easy problem at 8k tokens; hard is an AIME-level problem at 16k tokens, solved by propose→check→revise.
    - How to keep it verifiable: Final-answer check. Choose the budget as the smallest B for which doubling it gains ≤ κ = 1.2×.
    - Sources: 2506.09026, 2412.01951.

12. **Equivalence-preserving variation (re-rendering, rephrasing, variational problems)**
    - What it does: Same answer, new wording, format, order or context. It restores reward variance on saturated prompts and tests contextual transfer.
    - Example: easy is the zoo-context DAG; hard is the same DAG in a school context with shuffled sentences, or a variational problem synthesized from the policy's own correct solution.
    - How to keep it verifiable: The answer is invariant by construction (same DAG or same reference answer). Check equivalence with the generator, not with an LLM paraphraser alone.
    - Sources: 2512.07783, 2601.22478, 2508.14029.

13. **Adaptive difficulty ratchet / ZPD band (control operator)**
    - What it does: Pairs any generator with online control. Promote a level at ≥ 90% accuracy (window 4), train on 0 < p < 1, retire items at p ≥ 0.9, and give low-p items extra rollouts.
    - Example: easy is a static set with 95% of prompts solved 8/8 (zero advantage); hard is per-environment windows sliding upward so every batch has mixed outcomes.
    - How to keep it verifiable: Selection does not change labels. Estimate p with enough rollouts. Use zero-rollout priors at cold start. Note that about 39% of rollouts go to silent groups under uniform sampling (ThinkPrior, 2609.09075).
    - Sources: 2511.07317, 2504.03380, 2502.12272, 2503.14476, 2510.13786, 2508.13755, 2609.09075.

14. **Primitive seeding / atomic sharpening (enabling operator)**
    - What it does: Before composing, make atoms present (about 1% exposure) and reliable (1-hop RL or SFT), because composite success is roughly the product of atom successes.
    - Example: easy is a 30%-reliable atom; hard is a 5-step composite at 0.3⁵ ≈ 0.2% success, which becomes feasible once atoms are sharpened.
    - How to keep it verifiable: Atom-level exact checks. Track regressions on mastered atoms, since RL can sacrifice them.
    - Sources: 2512.07783, 2602.08281, 2606.18089, 2501.17161.

15. **Scaffolded decomposition (inverse operator for 0% items)**
    - What it does: When complexification overshoots, supply partial solutions or hints and fade them out, or turn reference chains into verifiable subproblems whose last one is the original.
    - Example: easy is a hard problem prefixed with 80% of a teacher solution; hard is the same problem with the hint withdrawn step by step to 0%.
    - How to keep it verifiable: Intermediate answers come from reference or teacher chains. The final subproblem is the original, so its ground truth is unchanged. Normalize rewards per subproblem position.
    - Sources: 2507.13266, 2605.22074, 2609.13997, 2606.22317.

16. **Learned stepping-stone generation**
    - What it does: A teacher writes intermediate problems and is rewarded by the student's gain on real hard problems.
    - Example: easy is a teacher prompted to "write a harder problem"; hard is a meta-RL teacher optimized for greedy-success gain on fail@128 items.
    - How to keep it verifiable: Stepping stones need to be well-posed rather than correct. Ground truth lives in the held-out hard set, which is kept out of the teacher's context. Contrast with R-Zero and Absolute Zero, which use intrinsic or learnability rewards.
    - Sources: 2601.18778, 2505.03335, 2508.05004.

17. **Held-out strategy and combination splits (evaluation operator)**
    - What it does: Test splits for exploratory (larger instances), compositional (unseen skill combinations) and transformative (new strategy) generalization. They tell you what kind of hardness training produced.
    - Example: easy is the same family with larger parameters; hard is a family needing an unconventional insight (e.g., periodic initial states).
    - How to keep it verifiable: Programmatic generation with symbolic, numeric or graphical validation. Keep generator code for held-out splits separate.
    - Sources: 2506.18880, 2509.21016.

18. **Solvability and length audit (guard operator)**
    - What it does: Every complexity knob gets a solvability check and an output-length cap, or asks for a compact answer (program, final state).
    - Example: easy is Hanoi with 5 disks; the naive hard version, 15 disks with a full move list, overflows the output. Hard done right is 15 disks answering with a generating program or the move count.
    - How to keep it verifiable: Solver or construction proves that a solution exists, e.g., River Crossing with b = 3 has no solution for N ≥ 6.
    - Sources: 2506.06941, 2506.09250.

---

## Insights & pitfalls

- **Composition is the unit of new skill.** Across string functions, Countdown, rewrite grammars, data structures, GSM chains and algebra, RL on composed tasks built new composite skills that extrapolate further; RL on atoms did not. The pattern is always the same: install atoms (SFT, RFT, mid-training, about 1% exposure), then RL on composites [2509.25123, 2512.01775, 2607.07646, 2609.19465, 2510.07312, 2602.08281].
- **Train on the composed form.** Decomposed training does not transfer up; composed training transfers down. OMEGA's "skills A and B separately" regime shows little or no compositional gain. Kong et al. find that compound traces beat isolated modules [2609.19465, 2506.18880, 2606.18089].
- **Grokking-like dynamics are real under RL for new procedures.** DELTA shows a long plateau (<1% for 450 steps) and then near-100%. Implicit composition in small transformers also appears only via grokking. Do not kill hard-tier runs early; log hard-subset metrics separately [2509.21016, 2405.15071].
- **Edge-of-competence is operational, not metaphorical.** Filter for items that fail at pass@1 but succeed at pass@k, and periodically re-scan as the model improves [2512.07783]. Practical filters include 0 < p < 1 (DAPO), p(1−p) (Foster et al.), R-Zero's 3–7 of 10 majority-agreeing, and retiring p ≥ 0.9 (ScaleRL).
- **Under vanilla GRPO, hard items are under-weighted even when they are present.** Cumulative advantage peaks at medium accuracy, so give low-accuracy prompts more rollouts [2508.13755].
- **Static datasets saturate even at scale.** A saturated 1.5B model gained 7× more from adaptive environments than from 3× more compute on its original data. More environment families beat more items from one family [2511.07317].
- **Difficulty knobs are not interchangeable.**
  - Depth and horizon costs grow superlinearly and extrapolate only about 3× [2605.06638].
  - Topology (right-heavy) is harder than size at equal depth [2512.01775].
  - Non-local branch–merge dependencies are harder than long chains [2609.19465].
  - Expressiveness transfers best [2605.06638].
- **Error compounding is worse than independent.** h1 observed long-horizon accuracy below the p^h prediction, and Sinha et al. document self-conditioning on one's own earlier errors. Horizon-specific training is needed, and thinking mitigates self-conditioning [2510.07312, 2509.09677].
- **Curricula: coverage beats ordering; staging matters only for sparse tiers.** Evidence: [2603.27226, 2505.23683]. Staging helps in h1 (uniform mix fails), ScaleLogic (γ 1.33 vs 2.36) and e3 (coupled budget), and DELTA's curriculum must be structurally aligned (REGEX works, COMPR does not).
- **Over-training on saturated items shrinks the boundary.** Diagnoses: support shrinkage outweighs expansion [2507.14843]; winner-take-all toward high-likelihood problems [2510.02230]; pass@k inversion on rare-path boundary prompts [2607.20543]; most standard RLVR updates are "overtraining" from the pass@k view [2606.15455]. Mitigations:
  - Per-Problem Base Anchoring (anchor risky prompts to the base) [2607.20543].
  - Bayesian Boundary Gating [2606.15455]; restricting updates to zero-success problems lifted pass@256 above the base.
  - Recycling saturated data by injecting high-quality incorrect rollouts: +6.4 to +9.0 on Qwen3-1.7B/4B [2609.33126].
  - Constrained uniform top-k exploration (Mixed-CUTS): up to +15.1 AIME25 pass@1 over GRPO [2604.18493].
- **SFT and RL play different roles.** RL "squeezes" (compresses incorrect trajectories) and SFT "expands" (adds correct ones) [2509.21128]. RL generalizes to rule variants where SFT memorizes, though RL needs SFT for format [2501.17161]. RL forgets less (KL-minimal updates, "RL's Razor") [2509.04259]. Distillation lifts the whole pass@k curve [2504.13837]. Use SFT to inject primitives and strategies, and RL on complexified composites.
- **Hard data beats easy data for RL, but not necessarily for SFT knowledge elicitation.** For RL: hardest 10% up to 47% versus 3–15% for easy [2508.14094]. For SFT reasoning, s1K beats random 1K (50.0 vs 36.7) [2501.19393], while easy data can match hard for knowledge QA [2401.06751].
- **Self-labeled harder data decays.** R-Zero's pseudo-label accuracy went 79→69→63% [2508.05004]. Self-improving transformers avalanche without filtering; majority voting took label accuracy from 31% to 93.3% [2502.01612]. Prefer answers that are known by design.
- **Stepping stones do not need correct answers, but final targets do.** Only 32.8% of SOAR's stepping stones were fully correct, yet well-posedness produced a 4× pass@1 gain [2601.18778]. Do not extend this to the final RL objective.
- **Diversity must be engineered.** Unstructured "make it harder" rewriting collapses pass@k [2605.11666]. Concept-graph recombination keeps scaling where rephrasing and personas saturate [2503.19551]. Problem augmentation and variational synthesis sustain entropy [2508.14029, 2601.22478].
- **Evaluation hygiene.** Use fresh procedurally generated items, a random-reward control, and at least two model families [2506.10947, 2507.10532]. Large-k pass@k is inflated on low-entropy answer spaces, so add Cover@τ [2510.08325]. Audit solvability and output lengths [2506.09250].
- **Collapse hygiene for iterative loops.** Accumulate, don't replace [2404.01413]. Verify [2406.07515]. Watch for competence polarization, where strong skills strengthen and weak ones decay [2607.17043]. About 1% synthetic data can already hurt in pure self-consumption [2410.04840].
- **RL can erase useful non-standard behaviors.** OMEGA: RL dropped a 70% matrix-rank OOD base by 30 points. Algebrarium: RL sacrifices some mastered skills. Keep regression probes for the atoms and for alternative strategies [2506.18880, 2602.08281].

---

## Open problems & research opportunities

- **Transformative generalization is still about zero** (OMEGA, DELTA). No known operator reliably yields verifiable tasks that force a *new* strategy rather than a recombination. Candidate direction: generators whose natural solution is intractable but which have a hidden short invariant (periodicity, symmetry, reverse reasoning), paired with DELTA-style dense warm-ups.
- **Composition beyond chains.** Branch–merge and non-local dependencies degrade sharply even after composed training [2609.19465]. Which data (graph-shaped traces, process rewards on merge nodes) fixes non-local coordination?
- **A shared "new skill vs. sharpening" acceptance protocol.** Candidates include large-k pass@k, null-set unlock rate, per-topology accuracy, Cover@τ and CoT-pass@k. There is no standard, and pass@k at large k can reward guessing [2510.08325].
- **Predicting learnability from generator parameters.** The edge-of-competence band shifts per model, per domain and per step. Zero-rollout priors [2609.09075] and cumulative-advantage re-weighting [2508.13755] are first steps. What is missing is a model of pass rate as a function of (depth, width, topology, expressiveness).
- **RL scaling laws over task complexity beyond logic.** ScaleLogic gives T ∝ D^γ for logic. Analogous exponents for code, math horizons and agent turns would let teams budget per tier and predict the ~3× extrapolation ceiling.
- **When does staging matter?** Theory (mixed data suffices [2505.23683]; autocurriculum helps [2603.18325, 2606.27721]) and experiments (ordering-neutral [2603.27226] vs staged wins [2510.07312, 2605.06638]) disagree. A predictive criterion, e.g., top-tier reward sparsity or SNR, would save compute.
- **Atomic-to-composite ratio and mid-training exposure at scale.** The 1% threshold (100M models) and the inferred/atomic-ratio effect (grokked transformers) are untested in frontier pretraining mixes.
- **Verifying complexified natural-language and agentic tasks.** LLM-written hard questions lose label accuracy with difficulty. Scalable certification is open: solver-backed formalization, executable world models, heterogeneous-solver consensus. CompoWorld composes 448 verified services (10,130 tools) with dependency-graph random walks [2609.33665]. The Agentic Compositional Generalization hypothesis says RL mainly routes pre-existing skills, and verifier quality matters more than environment count [2608.22631].
- **Robustness to self-conditioning inside long composed tasks.** Training signals that teach recovery from one's own earlier mistakes (error-injected contexts) are underexplored [2509.09677].
- **Unifying complexification with boundary-preserving updates.** Base anchoring, saturation gating, negative-sample injection and difficulty re-weighting are separate patches. A joint scheduler that decides which items to complexify, which to freeze and which to anchor does not exist.
- **Collapse guarantees for iterative self-complexification.** Minimum human-to-synthetic ratios are only beginning to be characterized (Fisher–Rao analysis, 2609.18878), and competence polarization [2607.17043] needs per-skill monitoring tools.

---

## References

1. Yuan, L., Chen, W., Zhang, Y., et al. (2025). *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*. arXiv 2509.25123. https://arxiv.org/abs/2509.25123
2. He, Y., Li, Y., Wang, Y., Vitercik, E. (2026). *Compositional Reasoning in Language Models under Reinforcement Learning Post-Training*. arXiv 2609.19465. https://arxiv.org/abs/2609.19465
3. Kong, L., Liu, X., Chen, G., et al. (2026). *From Reasoning Traces to Reusable Modules: Understanding Compositional Generalization in Language Model Reasoning*. ICML 2026; arXiv 2606.18089. https://arxiv.org/abs/2606.18089
4. Abdulsalam, A., Patel, N., Saxe, A. (2026). *RL Post-Training Builds Compositional Reasoning Strategies*. ICML 2026 Workshop on Compositional Learning; arXiv 2607.07646. https://arxiv.org/abs/2607.07646
5. Park, S., Kaur, S., Arora, S. (2025). *How Does RL Post-training Induce Skill Composition? A Case Study on Countdown*. arXiv 2512.01775. https://arxiv.org/abs/2512.01775
6. Sun, Y., Hu, S., Zhou, G., et al. (2025). *OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization*. arXiv 2506.18880. https://arxiv.org/abs/2506.18880
7. Sun, Y., Cao, Y., Huang, P., et al. (2025). *RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?* (DELTA-Code). arXiv 2509.21016. https://arxiv.org/abs/2509.21016
8. Wang, Z., Li, Y., Zhang, S., et al. (2026). *New Skills or Sharper Primitives? A Probabilistic Perspective on the Emergence of Reasoning in RLVR*. arXiv 2602.08281. https://arxiv.org/abs/2602.08281
9. Zhao, H., Kaur, S., Yu, D., Goyal, A., Arora, S. (2024). *Can Models Learn Skill Composition from Examples?* NeurIPS 2024; arXiv 2409.19808. https://arxiv.org/abs/2409.19808
10. Yu, D., Kaur, S., Gupta, A., et al. (2023). *Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models*. arXiv 2310.17567. https://arxiv.org/abs/2310.17567
11. Wang, B., Yue, X., Su, Y., Sun, H. (2024). *Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization*. NeurIPS 2024; arXiv 2405.15071. https://arxiv.org/abs/2405.15071
12. Zhang, C., Neubig, G., Yue, X. (2025). *On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models*. arXiv 2512.07783. https://arxiv.org/abs/2512.07783
13. Yue, Y., Chen, Z., Lu, R., et al. (2025). *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* NeurIPS 2025; arXiv 2504.13837. https://arxiv.org/abs/2504.13837
14. Liu, M., Diao, S., Lu, X., et al. (2025). *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models*. arXiv 2505.24864. https://arxiv.org/abs/2505.24864
15. Chu, T., Zhai, Y., Yang, J., et al. (2025). *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*. arXiv 2501.17161. https://arxiv.org/abs/2501.17161
16. Motwani, S. R., Ivanova, A., Cai, Z., et al. (2025). *h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning*. arXiv 2510.07312. https://arxiv.org/abs/2510.07312
17. Wang, T., Wang, Z., Lan, G., et al. (2026). *How Deep Can LLMs Learn to Reason? Expressiveness Is Key* (ScaleLogic). arXiv 2605.06638. https://arxiv.org/abs/2605.06638
18. Setlur, A., Yang, M. Y. R., Snell, C., et al. (2025). *e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs*. arXiv 2506.09026. https://arxiv.org/abs/2506.09026
19. Sinha, A., Arun, A., Goel, S., et al. (2025). *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs*. ICLR 2026; arXiv 2509.09677. https://arxiv.org/abs/2509.09677
20. Ye, T., Xu, Z., Li, Y., Allen-Zhu, Z. (2024). *Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process*. arXiv 2407.20311. https://arxiv.org/abs/2407.20311
21. Dziri, N., Lu, X., Sclar, M., et al. (2023). *Faith and Fate: Limits of Transformers on Compositionality*. NeurIPS 2023; arXiv 2305.18654. https://arxiv.org/abs/2305.18654
22. Shojaee, P., Mirzadeh, I., Alizadeh, K., et al. (2025). *The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity*. NeurIPS 2025; arXiv 2506.06941. https://arxiv.org/abs/2506.06941
23. Lawsen, A. (2025). *Comment on The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity*. arXiv 2506.09250. https://arxiv.org/abs/2506.09250
24. Zeng, Z., Ivison, H., Wang, Y., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*. ICML 2026; arXiv 2511.07317. https://arxiv.org/abs/2511.07317
25. Bae, S., Hong, J., Lee, M. Y., et al. (2025). *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning*. EACL 2026; arXiv 2504.03380. https://arxiv.org/abs/2504.03380
26. Foster, T., Sims, A., Forkel, J., et al. (2025). *Learning to Reason at the Frontier of Learnability*. arXiv 2502.12272. https://arxiv.org/abs/2502.12272
27. Yu, Q., Zhang, Z., Zhu, R., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv 2503.14476. https://arxiv.org/abs/2503.14476
28. Khatri, D., Madaan, L., Tiwari, R., et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs* (ScaleRL). arXiv 2510.13786. https://arxiv.org/abs/2510.13786
29. Pikus, B., Tiwari, P. R., Ye, B. (2025). *Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets*. arXiv 2508.14094. https://arxiv.org/abs/2508.14094
30. Yang, Z., Guo, Z., Huang, Y., et al. (2025). *Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration*. arXiv 2508.13755. https://arxiv.org/abs/2508.13755
31. Sun, Z., Yu, L., Shen, Y., et al. (2024). *Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision*. NeurIPS 2024; arXiv 2403.09472. https://arxiv.org/abs/2403.09472
32. Hase, P., Bansal, M., Clark, P., Wiegreffe, S. (2024). *The Unreasonable Effectiveness of Easy Training Data for Hard Tasks*. ACL 2024; arXiv 2401.06751. https://arxiv.org/abs/2401.06751
33. Lee, N., Cai, Z., Schwarzschild, A., Lee, K., Papailiopoulos, D. (2025). *Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges*. arXiv 2502.01612. https://arxiv.org/abs/2502.01612
34. Muennighoff, N., Yang, Z., Shi, W., et al. (2025). *s1: Simple test-time scaling*. arXiv 2501.19393. https://arxiv.org/abs/2501.19393
35. Ye, Y., Huang, Z., Xiao, Y., et al. (2025). *LIMO: Less is More for Reasoning*. COLM 2025; arXiv 2502.03387. https://arxiv.org/abs/2502.03387
36. Guha, E., Marten, R., Keh, S., et al. (2025). *OpenThoughts: Data Recipes for Reasoning Models*. arXiv 2506.04178. https://arxiv.org/abs/2506.04178
37. Wang, Z., Nichani, E., Bietti, A., et al. (2025). *Learning Compositional Functions with Transformers from Easy-to-Hard Data*. COLT 2025; arXiv 2505.23683. https://arxiv.org/abs/2505.23683
38. Rajaraman, N., Huang, A., Dudik, M., et al. (2026). *Learning to Reason with Curriculum I: Provable Benefits of Autocurriculum*. arXiv 2603.18325. https://arxiv.org/abs/2603.18325
39. Rajaraman, N., Huang, A., Dudik, M., et al. (2026). *Learning to Reason with Curriculum II: Compositional Generalization*. arXiv 2606.27721. https://arxiv.org/abs/2606.27721
40. Mordig, M., Opedal, A., Liu, W., Schölkopf, B. (2026). *Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning*. arXiv 2603.27226. https://arxiv.org/abs/2603.27226
41. Sundaram, S., Quan, J., Kwiatkowski, A., et al. (2026). *Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability* (SOAR). ICML 2026; arXiv 2601.18778. https://arxiv.org/abs/2601.18778
42. Ye, L., Yin, Y., Galarnyk, M., et al. (2026). *Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling* (EvoTD). arXiv 2605.11666. https://arxiv.org/abs/2605.11666
43. Qin, Z., Dong, Q., Zhang, X., et al. (2025). *Scaling Laws of Synthetic Data for Language Models* (SynthLLM). COLM 2025; arXiv 2503.19551. https://arxiv.org/abs/2503.19551
44. Shao, R., Li, S. S., Xin, R., et al. (2025). *Spurious Rewards: Rethinking Training Signals in RLVR*. arXiv 2506.10947. https://arxiv.org/abs/2506.10947
45. Wu, M., Zhang, Z., Dong, Q., et al. (2025). *Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination* (RandomCalculation). AAAI 2026; arXiv 2507.10532. https://arxiv.org/abs/2507.10532
46. Shumailov, I., Shumaylov, Z., Zhao, Y., et al. (2023). *The Curse of Recursion: Training on Generated Data Makes Models Forget*. arXiv 2305.17493 (Nature 2024 version: "AI models collapse when trained on recursively generated data"). https://arxiv.org/abs/2305.17493
47. Gerstgrasser, M., Schaeffer, R., Dey, A., et al. (2024). *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. arXiv 2404.01413. https://arxiv.org/abs/2404.01413
48. Feng, Y., Dohmatob, E., Yang, P., Charton, F., Kempe, J. (2024). *Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification*. arXiv 2406.07515. https://arxiv.org/abs/2406.07515
49. Dohmatob, E., Feng, Y., Subramonian, A., Kempe, J. (2024). *Strong Model Collapse*. arXiv 2410.04840. https://arxiv.org/abs/2410.04840
50. Ramesh, R., Lubana, E. S., Khona, M., et al. (2023). *Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks*. arXiv 2311.12997. https://arxiv.org/abs/2311.12997
51. Liu, H., Li, G., Li, J., et al. (2025). *SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning*. NeurIPS 2025; arXiv 2505.16368. https://arxiv.org/abs/2505.16368
52. Huang, A., Block, A., Foster, D. J., et al. (2024). *Self-Improvement in Language Models: The Sharpening Mechanism*. arXiv 2412.01951. https://arxiv.org/abs/2412.01951
53. Le, K., Nguyen, P., Mroueh, Y., et al. (2026). *Transformation-Augmented GRPO for Enhancing Exploration in Reasoning of Large Language Models*. arXiv 2601.22478. https://arxiv.org/abs/2601.22478
54. Liang, X., Li, Z., Gong, Y., et al. (2025). *Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR* (SvS). arXiv 2508.14029. https://arxiv.org/abs/2508.14029
55. Sha, T., Zhai, S., Zhao, S. (2026). *ThinkPrior: Zero-Rollout Difficulty Priors for Cold-Start Prompt Selection in RLVR*. arXiv 2609.09075. https://arxiv.org/abs/2609.09075
56. Li, J., Lin, H., Lu, H., et al. (2025). *QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation*. ICLR 2026; arXiv 2507.13266. https://arxiv.org/abs/2507.13266
57. Jiang, X., Tang, Z., Lin, W., et al. (2026). *From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning* (SCRL). arXiv 2605.22074. https://arxiv.org/abs/2605.22074
58. Zhu, Y., Han, Z. (2026). *Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR*. Findings of EMNLP 2026; arXiv 2609.13997. https://arxiv.org/abs/2609.13997
59. Cai, P., Fang, T., Li, X., et al. (2026). *Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model*. arXiv 2606.22317. https://arxiv.org/abs/2606.22317
60. Zhao, A., Wu, Y., Yue, Y., et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv 2505.03335. https://arxiv.org/abs/2505.03335
61. Huang, C., Yu, W., Wang, X., et al. (2025). *R-Zero: Self-Evolving Reasoning LLM from Zero Data*. arXiv 2508.05004. https://arxiv.org/abs/2508.05004
62. Zhou, T. (2026). *When RLVR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion*. arXiv 2607.20543. https://arxiv.org/abs/2607.20543
63. Yuan, S., Chen, J., Zheng, J., et al. (2026). *Understanding Diversity Collapse in RLVR via the Lens of Overtraining*. arXiv 2606.15455. https://arxiv.org/abs/2606.15455
64. Liang, Z., Zhou, Y., Lu, S., et al. (2026). *Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data* (Mixed-CUTS). ACL 2026; arXiv 2604.18493. https://arxiv.org/abs/2604.18493
65. Yang, Z., Wang, Y., Feng, S., Tsvetkov, Y. (2026). *Save Your Saturated Data: Learning Beyond Reward Saturation in Group-Based RL*. arXiv 2609.33126. https://arxiv.org/abs/2609.33126
66. Wu, F., Xuan, W., Lu, X., et al. (2025). *The Invisible Leash: Why RLVR May or May Not Escape Its Origin*. arXiv 2507.14843. https://arxiv.org/abs/2507.14843
67. Nguyen, P. M., La, C. D., Nguyen, D. M. H., et al. (2025). *The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models*. arXiv 2510.02230. https://arxiv.org/abs/2510.02230
68. Matsutani, K., Takashiro, S., Minegishi, G., et al. (2025). *RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs*. ICLR 2026; arXiv 2509.21128. https://arxiv.org/abs/2509.21128
69. Shenfeld, I., Pari, J., Agrawal, P. (2025). *RL's Razor: Why Online Reinforcement Learning Forgets Less*. arXiv 2509.04259. https://arxiv.org/abs/2509.04259
70. Luo, X., Huang, Y., Guo, K., et al. (2026). *Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning* (KITE). arXiv 2607.17043. https://arxiv.org/abs/2607.17043
71. Dragoi, M., Pintilie, I., Gogianu, F., Brad, F. (2025). *Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries* (Cover@τ). arXiv 2510.08325. https://arxiv.org/abs/2510.08325
72. Yang, X.-W., Xu, W., Da, W., et al. (2026). *CompoWorld: Compositional Environment Scaling for General Agents*. arXiv 2609.33665. https://arxiv.org/abs/2609.33665
73. Yao, Y., Pang, B., Nguyen, X. P., et al. (2026). *Learning Generalizable Behaviors for Terminal Agents* (River). arXiv 2608.22631. https://arxiv.org/abs/2608.22631
74. Marchi, M., Silvestre, J. P., Gharesifard, B., Tabuada, P. (2026). *Preventing Model Collapse: A Fisher-Rao Perspective on the Dynamics of Training with Synthetic Data*. CDC 2026 (extended); arXiv 2609.18878. https://arxiv.org/abs/2609.18878
