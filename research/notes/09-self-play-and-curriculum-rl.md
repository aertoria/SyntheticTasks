# Self-play, proposer-solver co-evolution, and automatic curricula for LLM RL

*Scope: closed-loop task generation, where a proposer (teacher, challenger or setter) writes tasks and a solver learns from them. Covers ungrounded, pseudo-labelled self-play; self-play grounded in executors, corpora, search, repos, databases, OR solvers and games; policy-failure-driven variant synthesis; learnability/UED-style selection curricula; online difficulty filters; adaptive procedural environments; and hint/scaffold curricula for tasks at p = 0. Compiled 2026-09-30. Verification: 36 method entries checked against primary sources (arXiv full text or PDF): the 30 from the input plus 6 added. 11 were corrected, 0 dropped and 6 added. A further 17 works cited only in the operator, insight and open-problem sections were checked against their arXiv abstract, plus the passage cited where a specific claim is used. Numbers I could not find in the source were removed.*

## TL;DR

- **Measure the problem first, then select, then generate. Selection alone cannot fix a saturated pool.**
  - Track the *effective prompt ratio*: the share of GRPO groups whose rewards are not all identical.
  - Learnability selection with p(1−p) is cheap and gives real gains. LILO's final test accuracy on MATH went 19.1 → 21.8 with PPO and 22.8 → 24.9 with VinePPO, reaching the baseline's best 2.5–3.2× sooner. Bae et al.'s balanced band 0.2 < p < 0.8 gave +10 on AIME for the 3B model and +12 on AMC for the 7B model, in under half the GRPO steps.
  - Both only *reorder* what you already have. Once nearly every seed has p = 1, the only lever left is generating new tasks.
- **Never reward a proposer for "the solver failed" without an independent validity gate.**
  - Naive setter–solver play is reward-hacked by invalid problems (VHG).
  - Lowering OpenSIR's solve-rate floor from 0.5 to 0.1 made problems only slightly harder: GPT-5's solve rate went 89.8% → 78.3%. Validity collapsed from 70.8% to 42.3%, and math accuracy fell from 29.6 to 26.0.
  - Working designs multiply difficulty by validity:
    - VHG: R = 1[V]·(1 − Acc).
    - SSR: −1 for an inconsistent bug artifact.
    - OPT-Zero: R_valid × R_correct × R_struct.
  - Budget for low yield. Only 5.2% of Self-Challenging Code-as-Task proposals survived. VHG kept 4,076 of 18,663 integral candidates and 20,670 of 46,080 judge-checked general-math pairs.
- **Grounding matters more than reward shaping.** Tie the answer to something outside the model: an executor, a database, an OR solver, a repo with tests, a corpus, search evidence or a game engine.
  - The pseudo-label-only R-Zero degrades after 1–3 iterations. Pseudo-label accuracy fell 79% → 69% → 63%.
  - SPICE's corpus grounding gave 43.9 vs 40.7 without it.
  - Removing SSP's retrieval-augmented (RAG) answerability check dropped GeneralQA from 60.0 to 49.5.
- **Anchor generation on your own unsolved tasks, not on undirected "hardness."**
  - GASP generates an easier *lemma* and then a harder *lift* for each of 146 pass@100 = 0 coding problems. It solved 11 of them; AZR and standard RL solved 0.
  - SOAR rewards the teacher by the student's measured improvement on fail@128 problems. That gave 4× pass@1 and 2× pass@32 on MATH-hard.
  - SOAR also found that well-posed questions matter more than correct answers: only 32.8% of its useful questions had fully correct solutions.
- **Spawn harder variants from the policy's own successes.** This expands the boundary (pass@k at large k), not just pass@1.
  - SvS rewrites problems the policy solves only 12.5–50% of the time into answer-preserving variants, rewarded only when their solve rate lands in [12.5%, 62.5%]. It added +18.3 and +22.8 pass@32 on AIME24 and AIME25, and entropy stayed stable.
  - In contrast, AZR-style self-play mostly *sharpens*: the base model is better at large k, and entropy still collapses (Chae et al.).
- **Scaffold the p = 0 pool and withdraw the scaffold.**
  - QuestA: prefixing 50% of the solution for 100 steps and then 25% scored 63.26 average, vs 60.26 for 50% throughout.
  - Scaf-GRPO: tiered hints were needed on only 17.4% of samples and raised AIME24 from 30.0 to 43.3.
  - MFC: 128 unsolvable problems plus a backward-chaining curriculum matched GRPO trained on 2,000 problems.
- **Replace static sets with parameterized generators plus an online controller, and scale the number of environment types.** RLVE's per-environment rule is: move the window up when accuracy at the top level is at least 0.9, keeping a 4-level sliding window. Across 400 environments it gave +3.37 average, vs +0.49 from continuing the original RL with more than 3× the compute.
- **Diversity collapses silently. Measure it at the skill level, across iterations.**
  - R-Diverse: a persistent memory penalty, plus similarity computed on canonical *solver code* rather than question text.
  - OpenSIR: novelty measured as embedding distance to the whole pool.
  - SQL-Zero: deduplication by masked-SQL template.
  - PopuLoRA: a population with cross-evaluation. A single agent self-calibrates toward easy problems.
- **The exact learnability reward curve is a second-order knob.**
  - SSR: a consistency-only ±1 reward was only slightly worse than the solve-rate-shaped reward.
  - Socratic-Zero: reward-shape variants were within 0.4 points of each other.
  - Chae et al.: a 50%-target reward *lowered* AZR's validation accuracy by 2%.
  - OPT-Zero: a *structural-complexity* reward beat a solve-rate reward.
  - Spend the effort on validity, grounding and diversity instead.
- **Easy-to-hard ordering by itself is weak.** Mordig et al. found no robust gain over random order on deductive tasks, for SFT or RL. Theory says mixed-difficulty RLVR is already an implicit curriculum, as long as the difficulty spectrum is *smooth* (Huang et al., ICML 2026). Generate intermediate bridge tasks instead of tuning schedules.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Absolute Zero Reasoner (AZR) | 2025 | [2505.03335](https://arxiv.org/abs/2505.03335) | Python program reasoning → math | RL | Task-mode inversion (deduction/abduction/induction); learnability-rewarded proposal; buffer-conditioned novelty | Python executor computes gold output. Programs must parse, avoid blocked modules and be deterministic (run twice). Abduction is checked by re-execution; induction by held-out I/O pairs |
| R-Zero | 2025 | [2508.05004](https://arxiv.org/abs/2508.05004) | Math / general reasoning | RL | Uncertainty-rewarded question generation; BLEU-cluster repetition penalty | Majority vote of 10 solver samples. Keep only questions with 3–7 of 10 agreeing (no external verifier) |
| Self-Questioning LMs (SQLM) [added] | 2025 | [2508.03682](https://arxiv.org/abs/2508.03682) | Arithmetic, algebra, code | RL | Topic-only proposal with a "not trivial, not impossible" reward | Majority vote (math); proposer-written unit tests (code) |
| OpenSIR | 2025 | [2511.00602](https://arxiv.org/abs/2511.00602) | Math from a single seed ("What is 1+1?") | RL | Pool-conditioned mutation with embedding-distance novelty; solution-length and solvability reward | Majority-vote reference answer. Solve-rate floor (0.5) filters malformed problems |
| R-Diverse | 2026 | [2602.13103](https://arxiv.org/abs/2602.13103) | Math / general | RL | Cross-iteration memory penalty; skill-level (solver-code) diversity | Inherits R-Zero pseudo-labels |
| Multi-Solver Disagreement reward [added] | 2026 | [2608.30035](https://arxiv.org/abs/2608.30035) | Competition math | RL | Challenger rewarded for inter-model disagreement | Pseudo-labels; heterogeneous ensemble of solvers |
| Agent0 (+ Tool-R0) | 2025/26 | [2511.16043](https://arxiv.org/abs/2511.16043) | Tool-integrated math / general reasoning; tool calling | RL | Tool-dependence escalation; uncertainty reward | Majority vote over tool-using rollouts; code interpreter grounds computation |
| TTCS (+ TTC-RL) | 2026/25 | [2601.22628](https://arxiv.org/abs/2601.22628) | Test-time math | RL (test-time) | Target-conditioned variant ladders | Self-consistency (majority) rewards |
| Socratic-Zero | 2025 | [2509.24726](https://arxiv.org/abs/2509.24726) | Math | SFT (generator) + preference optimization (solver) | Failure-conditioned problem rewriting by a strong teacher | 235B teacher as verifier / oracle; rule-based answer extraction |
| PopuLoRA | 2026 | [2605.16727](https://arxiv.org/abs/2605.16727) | Code + math (on AZR) | RL | Population co-evolution; cross-evaluation; LoRA mutation/crossover | AZR programmatic verifier |
| GASP | 2026 | [2603.15957](https://arxiv.org/abs/2603.15957) | Competitive programming (LCB) | RL | Goalpost-anchored lemma → lift stepping stones | AZR-style execution validation; pass-rate bands; embedding dedup |
| SOAR [added] | 2026 | [2601.18778](https://arxiv.org/abs/2601.18778) | Math fail@128 subsets | Meta-RL (RL inside RL) | Teacher-generated stepping stones rewarded by student progress on real hard problems | Grounding is the real hard set's answers. Synthetic Q/A are *not* verified |
| VHG | 2026 | [2605.06660](https://arxiv.org/abs/2605.06660) | Integrals; general math | RL | Validity-gated setter–solver–verifier play | SymPy differentiation (hard verifier) or hard-coded filter + LLM judge (soft) |
| SPICE | 2025 | [2510.24684](https://arxiv.org/abs/2510.24684) | Math / general, corpus-grounded | RL | Information asymmetry (document hidden from solver) | Gold answer extracted from real document; Math-Verify / exact match |
| SPELL [added] | 2025 | [2509.23863](https://arxiv.org/abs/2509.23863) | Long-context reasoning | RL | Growing multi-document context with history memory; Gaussian difficulty reward | Reference answers from documents; self-consistency-trained verifier role |
| Search Self-Play (SSP) | 2025 | [2510.18821](https://arxiv.org/abs/2510.18821) | Deep-search agents | RL | Answer-first multi-hop question construction via search | RAG answerability check (collected docs + 4 distractor docs), run with Qwen2.5-32B-Instruct |
| Self-play SWE-RL (SSR) | 2025 | [2512.18552](https://arxiv.org/abs/2512.18552) | SWE agents on real repos | RL | Code removal / history reversion bug injection; test weakening; higher-order bugs | 7 consistency checks, including inverse mutation testing; oracle tests restored at evaluation |
| CURE | 2025 | [2506.03136](https://arxiv.org/abs/2506.03136) | Code + unit-test generation | RL | Verifier/test strengthening from coder mistakes | Ground-truth unit tests define correct solutions; generated tests rewarded for separating them |
| Self-Challenging Agents (SCA) | 2025 | [2506.01716](https://arxiv.org/abs/2506.01716) | Multi-turn tool use | RL (+ distillation) | Explore-then-propose Code-as-Task | Solution must pass the verifier and every failure case must fail |
| DeepSeek-V3.2 agentic task synthesis | 2025 | [2512.02556](https://arxiv.org/abs/2512.02556) | General tool-use environments | RL | Iterative difficulty escalation with co-updated solution + verifier; toolset augmentation | Solution restricted to tool interface; must pass verifier; keep pass@100 > 0 |
| SQL-Zero [added] | 2026 | [2609.04697](https://arxiv.org/abs/2609.04697) | Text-to-SQL | RL | SQL-first generation at a target complexity; hard-but-solvable reward; template dedup | Execution against the DB: deterministic SELECT with non-empty result; no SQL leakage |
| OPT-Zero [added] | 2026 | [2609.34205](https://arxiv.org/abs/2609.34205) | Optimization modeling | RL | Structural-complexity reward (coupling, connectivity) with frontier-updated target | Canonical OR solver: feasible, bounded, solved to optimality; proposer code must reproduce the optimum |
| SPIRAL | 2025 | [2506.24119](https://arxiv.org/abs/2506.24119) | Zero-sum text games → reasoning | RL | Opponent strengthening; multi-game mixing | Game engine decides outcome |
| eva | 2024 | [2411.00062](https://arxiv.org/abs/2411.00062) | RLHF / chat | RL (DPO, RLOO) | Regret-prioritized prompt selection + WizardLM-style evolution | None (reward model only) |
| SvS | 2025 | [2508.14029](https://arxiv.org/abs/2508.14029) | Competition math, code | RL | Answer-preserving variants written from the policy's own correct solutions | Variants inherit original answer; solve-rate band rejects trivial/broken variants |
| SwS | 2025 | [2506.08989](https://arxiv.org/abs/2506.08989) | Math | RL | Weakness mining → concept extraction/recombination | Strong-model self-consistency labels (>50%) + student agreement ≥ 25%; [25%, 75%] band |
| UED family (PAIRED, PLR, ACCEL, POET, SFL) | 2019–24 | [2203.01302](https://arxiv.org/abs/2203.01302) | Deep-RL environments | RL | Regret-based generation; replay; small edits to frontier levels | Simulator / procedural generator |
| LILO (sampling for learnability) | 2025 | [2502.12272](https://arxiv.org/abs/2502.12272) | Math RL (PPO, VinePPO, GRPO) | RL | Selection only: top-k by p(1−p) | Existing dataset answers |
| Online difficulty filtering (Bae et al.; DAPO; GRESO; PCL; MoPPS; DOTS) | 2025 | [2504.03380](https://arxiv.org/abs/2504.03380) | Math RLVR | RL | Balanced pass-rate bands; dynamic sampling; predictive skipping | Existing verifiers |
| RLVE | 2025 | [2511.07317](https://arxiv.org/abs/2511.07317) | 400 procedural environments | RL | Integer difficulty knobs with an adaptive sliding window; environment scaling | Algorithmic verifiers; generate-then-derive (solve–verify asymmetry) |
| SEC (+ AdaRFT, E2H, DUMP) | 2025 | [2505.14970](https://arxiv.org/abs/2505.14970) | Planning, induction, math | RL | Bandit / target-difficulty scheduling over difficulty buckets | Existing verifiers |
| QuestA | 2025 | [2507.13266](https://arxiv.org/abs/2507.13266) | Competition math | RL | Partial-solution prefixing, annealed 50% → 25% | Original answer unchanged; hints from reference solutions |
| SEELE | 2025 | [2509.06923](https://arxiv.org/abs/2509.06923) | Math RLVR | RL | Per-instance hint length fitted by item response theory (IRT) | Original answer unchanged |
| Scaf-GRPO | 2025 | [2510.19807](https://arxiv.org/abs/2510.19807) | Math RLVR | RL | Tiered hints (knowledge → planning → solution); minimal-hint search | Original verifier; hints pre-generated by DeepSeek-R1 |
| MFC (+ R3) | 2026 (2024) | [2609.13997](https://arxiv.org/abs/2609.13997) | Math RLVR | RL | Backward-chaining guidance withdrawal; reverse curriculum | Original answer; teacher/demonstration traces |
| Self-Improving Transformers | 2025 | [2502.01612](https://arxiv.org/abs/2502.01612) | Arithmetic, strings, mazes | SFT (self-labelling) | Size/length scaling one step per round (or accelerated) | Self-labels filtered by length and by 5-model majority vote (≥ 4/5) |

## Method notes

*Ordering: (A) proposer–solver self-play with pseudo-labels; (B) grounded or goal-directed self-play; (C) synthesis driven by the policy's own failures; (D) selection curricula and adaptive environments; (E) scaffolding for p = 0 tasks; (F) SFT self-labelling.*

### R-Zero — R-Zero: Self-Evolving Reasoning LLM from Zero Data (Huang et al., 2025)
Link: https://arxiv.org/abs/2508.05004

- **Mechanism**: A Challenger and a Solver are initialized from the same base model as *separate* models and trained alternately with GRPO.
  - The Challenger samples questions and is rewarded by r_unc = 1 − 2|p̂ − 0.5|, where p̂ is the share of m = 10 Solver samples that agree with the majority answer.
  - It also gets a BLEU-cluster repetition penalty (d_ij = 1 − BLEU, cluster threshold τ_BLEU = 0.5) and a format gate.
  - Each iteration draws a candidate pool of N = 8,000 questions. The Solver trains on the questions where 3–7 of 10 answers match the majority label (δ = 0.25), using the majority answer as the label.
- **How it makes tasks harder**: As the Solver improves, questions it answers consistently lose Challenger reward, so the Challenger moves to questions that split the Solver near 50%.
- **Correctness / verification**: No external verifier; labels are majority-vote pseudo-labels. The consistency band also acts as a quality filter, since very low agreement usually means an ill-posed question. Measured on 200 sampled questions per iteration, pseudo-label accuracy fell from 79.0% to 69.0% to 63.0% over the first three iterations.
- **Difficulty control**: The uncertainty reward peaks at 50% agreement, and the 3–7-of-10 band restricts Solver data further.
- **Reported results**: Qwen3-4B-Base gained +6.49 on math and +7.54 on general-domain reasoning.
  - Ablation: a single shared-parameter model had lower pseudo-label accuracy than two separate models (63.4% vs 71.0% at step 15).
- **Limitations / failure modes**:
  - Every model size eventually degrades. The 0.6B model peaks after the first iteration, while the 4B model keeps improving for three.
  - There is no universal noise threshold. The 0.6B model starts declining while its labels are still 70.6% accurate; the 4B model tolerates 48.8%.
  - The authors think label noise is not the only cause and point to model collapse from training on self-synthesized data.
  - Follow-up work identifies:
    - "diversity illusion" (R-Diverse).
    - reward starvation once all sampled answers agree (Multi-Solver Disagreement).
    - no gain from R-Zero's 50%-target reward when it is transplanted into AZR (Chae et al.).
- **How to reuse with easy seed tasks**: Use this only when no verifier or grounding source exists. Seed the Challenger prompt with your easy tasks as style exemplars. Cap it at about 2–3 iterations, keep a small human-labelled probe set to track pseudo-label accuracy each iteration, and stop when that accuracy drops. Keep the Challenger and Solver as separate weights.

### SQLM [added] — Self-Questioning Language Models (Chen et al., 2025)
Link: https://arxiv.org/abs/2508.03682

- **Mechanism**: Asymmetric self-play. The proposer receives *only a topic prompt* (e.g., "three-digit multiplication", "linear-equation word problems", "LeetCode-easy-style problems") and writes a question. The solver answers it N times. Both are trained with RL.
  - Solver reward: 1 if its answer matches the majority answer.
  - Proposer reward: 1 if 0 < (#answers equal to the majority) < N, i.e., neither trivially easy nor hopeless; otherwise 0.
  - Coding: the proposer also writes unit tests. The solver is rewarded by the tests, and the proposer gets 1 iff 0 < pass fraction < 1.
- **How it makes tasks harder**: The binary "not trivial" reward pushes the proposer away from problems the solver always agrees on. The paper's samples show the problems growing harder over training.
- **Correctness / verification**: Majority voting where generating a solution and verifying it are equally hard. Where verifying is much easier than generating, as in code, the proposer writes unit tests.
- **Difficulty control**: Only the binary band 0 < agreement < N.
- **Reported results**: No external data was used.
  - Qwen2.5-3B-Instruct: three-digit multiplication 0.791 → 0.948; OMEGA linear equations 0.440 → 0.600.
  - Qwen2.5-Coder-3B-Instruct: Codeforces subset 0.320 → 0.391.
- **Limitations / failure modes**: Small models and narrow topics. Majority-vote labels can be confidently wrong. The coarse binary proposer reward gives no pressure toward p ≈ 0.5 and no pressure for diversity.
- **How to reuse with easy seed tasks**: This is the simplest version to prototype. Describe your easy seed family as a topic, and let a proposer generate harder members of the family. Wherever you can, have the proposer emit *checkable artifacts* (unit tests, constraints, a checker program) rather than a bare answer.

### OpenSIR — OpenSIR: Open-Ended Self-Improving Reasoner (Kwan et al., 2025)
Link: https://arxiv.org/abs/2511.00602

- **Mechanism**: One LLM alternates teacher and student roles. The problem pool starts with a single trivial problem, "What is 1+1?".
  - The teacher samples reference problems from the pool and writes a new problem wrapped in `<question>` tags, with at most three concepts in `<concepts>` tags.
  - The student answers several times. The most common answer becomes the reference, and the student's solve rate is recorded.
  - The teacher's novelty score is α·score_sol + λ·score_len + γ·score_div + δ·score_format, where:
    - score_sol is a triangular solvability score that peaks at the midpoint of [s_min, s_max].
    - score_len is min(average solution length / 1000 tokens, cap).
    - score_div is the cosine distance from the new problem's embedding to its *nearest* neighbour in the pool.
  - Accepted problems join the pool.
- **How it makes tasks harder**: The length term rewards problems that need longer multi-step solutions. The diversity term pushes toward concepts not yet in the pool. The solvability term keeps both inside the learnable range.
- **Correctness / verification**: Majority-vote reference answers. The solve-rate floor s_min doubles as a validity filter. Even at s_min = 0.5, about 30% of reference answers disagree with GPT-5's majority answer.
- **Difficulty control**: The triangular solvability window. The authors show a sharp **difficulty–validity trade-off**. With the upper threshold fixed at 0.9:
  - Lower threshold 0.5: validity 70.82%, GPT-5 solve rate 89.82%, math average 29.57.
  - Lower threshold 0.3: 52.32%, 81.38%, 27.81.
  - Lower threshold 0.1: 42.31%, 78.31%, 25.97.
- **Reported results**:
  - Across 7 math benchmarks: +3.6 points on instruction models and +3.1 on reasoning models.
  - Prior self-play baselines gained at most +0.87 on instruction models and fell as low as −1.93 on reasoning models.
  - Starting from one trivial seed, it beat GRPO trained on more than 7K annotated examples.
  - It was the only self-play method that transferred to general reasoning (at least +4.4 on reasoning models).
  - The diversity reward roughly doubled concept coverage.
- **Limitations / failure modes**: Labels come from majority vote. Only math was trained. Lowering the solve-rate floor to chase hardness mostly adds invalid problems.
- **How to reuse with easy seed tasks**: Put your easy seeds in the pool, and reward proposals by embedding distance to the *whole* archive rather than to the current batch. Add a solution-length term so the proposer favours problems that need more steps. Keep the solve-rate floor high (at least 0.5) unless you have a real verifier. Harder-looking problems below that floor are mostly broken.

### R-Diverse — R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training (Li et al., 2026)
Link: https://arxiv.org/abs/2602.13103 · Code: https://github.com/Gengsheng-Li/R-Diverse

- **Mechanism**: The paper diagnoses why R-Zero-style gains fade and names the cause **Diversity Illusion**. It takes two forms:
  - *Local*: diversity is enforced only within a batch, so the Challenger cycles between modes across iterations.
  - *Surface*: questions look different but exercise the same reasoning skill.
  - Two fixes:
    - Memory-Augmented Penalty (MAP): a persistent memory bank penalizes similarity to questions from earlier iterations. The same bank feeds a small experience-replay share of historical samples back into training.
    - Skill-Aware Measurement (SAM): each question is mapped by Qwen2.5-Coder-7B to canonical solver code for its solution procedure. The code is embedded and compared, so similarity reflects the skills used rather than the wording.
- **How it makes tasks harder**: It does not raise difficulty directly. It stops the Challenger from recycling skills the Solver has already mastered, which leaves the uncertainty reward free to find new ones.
- **Correctness / verification**: Inherits the R-Zero pseudo-labels.
- **Difficulty control**: Inherits R-Zero's uncertainty reward and band.
- **Reported results**: Across 10 math and general-reasoning benchmarks on the Qwen3 family, gains are sustained over more iterations and it consistently beats prior self-play methods. The abstract gives no headline number.
- **Limitations / failure modes**: The code-abstraction step costs extra LLM calls, and pseudo-label noise remains.
- **How to reuse with easy seed tasks**: Deduplicate generated tasks by a *solution signature* (canonical program, SymPy expression, or tool-call trace), not by question text, and keep the archive across iterations. Replay a small share of archived tasks so the model does not forget old skills.

### Multi-Solver Disagreement reward [added] — Beyond Uncertainty: Multi-Solver Disagreement Rewards for Self-Evolving Reasoning Curricula (Selvendran & Zhang, 2026)
Link: https://arxiv.org/abs/2608.30035 (CIKM '26)

- **Mechanism**: A drop-in change to the R-Zero Challenger reward. Each question goes to a heterogeneous frozen ensemble:
  - the Solver-v1 checkpoint (Qwen3-4B, GRPO-trained) at T = 0.7;
  - the same weights at T = 1.0;
  - Llama-3.2-3B-Instruct at T = 0.7.
  - Each member produces a plurality answer. The disagreement reward is the normalized Shannon entropy over those plurality answers, used alongside the pooled-uncertainty reward.
- **How it makes tasks harder**: It rewards questions where differently trained models *disagree*. This helps in two cases: when a single confident solver's agreement would starve the Challenger, and when a question only looks easy because it matches one model's biases.
- **Correctness / verification**: Pseudo-labels, as in R-Zero.
- **Difficulty control**: Entropy of disagreement across the ensemble.
- **Reported results**: +1.34 points average on MATH-500, AMC and Olympiad for Qwen3-4B solvers.
- **Limitations / failure modes**: Small gain; the extra models add inference cost. Disagreement can also come from ambiguous questions, so it still needs a validity check.
- **How to reuse with easy seed tasks**: Once your single-model pass rates saturate, estimate difficulty as disagreement across 2–3 different checkpoints or model families. Mine the seeds and variants where they disagree.

### Agent0 (+ Tool-R0) — Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning (Xia et al., 2025)
Link: https://arxiv.org/abs/2511.16043 (COLM 2026) · Tool-R0: https://arxiv.org/abs/2602.21320

- **Mechanism**: A curriculum agent and an executor agent start from the same base model, and the executor has a code interpreter.
  - Curriculum reward = format gate × max(0, λ_unc·(1 − 2|p̂ − 0.5|) + R_tool − R_rep).
  - R_tool is a weighted, capped count of tool-response markers in the executor's rollout.
  - The executor trains on tasks within |p̂ − 0.5| ≤ δ = 0.25, using k = 10 samples and majority-vote pseudo-labels. The paper describes this as retaining self-consistency between 0.3 and 0.8.
  - The executor uses Ambiguity-Dynamic Policy Optimization (ADPO), which scales advantages by self-consistency and relaxes the upper clipping bound on ambiguous tasks.
- **How it makes tasks harder**: As the executor gets better with tools, the curriculum agent is pushed toward tasks that *require* multi-step computation through tool calls.
- **Correctness / verification**: Majority-vote pseudo-labels over tool-using rollouts. The interpreter makes intermediate computation exact.
- **Difficulty control**: The uncertainty reward plus the consistency band. The tool term is capped so the agent cannot farm tool calls.
- **Reported results**:
  - Qwen3-8B-Base math average 49.2 → 58.2, vs R-Zero 54.7, Socratic-Zero 56.1 and AZR 52.6.
  - Qwen3-8B-Base general overall 34.5 → 42.1.
  - Qwen3-4B-Base math 42.6 → 52.5.
  - The abstract's summary: +18% math and +24% general reasoning.
  - Tool-R0 co-evolves a Generator and Solver with real tool calls from zero data and reports a 92.5 relative improvement over the base model on tool-use benchmarks.
- **Limitations / failure modes**: Pseudo-label noise. Few co-evolution iterations were run. Autonomous code execution needs sandboxing.
- **How to reuse with easy seed tasks**: If your easy seeds can be solved from memory or with shallow reasoning, add a capped proposer reward for tasks whose *successful* solutions need tool calls (computation, retrieval). Filter out tasks that are solvable without tools.

### TTCS (+ TTC-RL) — TTCS: Test-Time Curriculum Synthesis for Self-Evolving (Yang et al., 2026)
Link: https://arxiv.org/abs/2601.22628 · TTC-RL: https://arxiv.org/abs/2510.04786

- **Mechanism**: A question synthesizer and a reasoning solver are initialized from one model. The synthesizer writes progressively harder *variants of unlabelled test questions*. The solver trains on the originals plus the variants with self-consistency rewards, as in TTRL, and its feedback steers the synthesizer toward its current capability.
  - TTC-RL is a related method that does not generate: it *retrieves* the most task-relevant problems from a large pool for each target task and runs RL on them.
- **How it makes tasks harder**: It builds a ladder of variants from easy to target-level, conditioned on the target questions.
- **Correctness / verification**: Majority-vote (TTRL-style) pseudo-labels only. TTRL itself reported roughly +211% pass@1 on AIME 2024 for Qwen-2.5-Math-7B from unlabelled test data.
- **Difficulty control**: Solver feedback aligns variant difficulty with current ability.
- **Reported results**: The TTCS abstract gives only qualitative gains on hard math that transfer to general domains.
  - TTC-RL (Qwen3-8B): pass@1 about 1.8× on AIME25 and 2.1× on CodeElo.
  - TTC-RL pass@8: AIME25 40% → 62%, CodeElo 28% → 43%.
- **Limitations / failure modes**: Pseudo-labels on hard targets are noisy, and the test sets are small.
- **How to reuse with easy seed tasks**: When you know which hard target distribution matters, generate variants *conditioned on the targets*, or retrieve seeds similar to them, so the new tasks sit between your easy pool and those targets. Do this on a dev set, not the evaluation set, to avoid contamination.

### Socratic-Zero — Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution (Wang et al., 2025)
Link: https://arxiv.org/abs/2509.24726

- **Mechanism**: Three agents, starting from 100 seed questions.
  - A fixed Teacher (Qwen3-235B-A22B-Instruct-2507) takes the Solver's failed solutions and rewrites the original problem into a new problem–solution pair aimed at that weakness.
  - The Solver learns from preference feedback over its successful and failed trajectories, with k = 8 trajectories per problem.
  - The Generator (Qwen3-32B) learns the Teacher's problem-writing strategy by weighted SFT. Each example is weighted by a Gaussian utility of the Solver's success rate s (μ = 0.5, σ = 0.2).
- **How it makes tasks harder**: New problems target observed failure modes. The curriculum is partitioned into Mastered (s = 1), Learning (0 < s < 1) and Too Difficult (s = 0) zones, and new problems are generated from the first two.
- **Correctness / verification**: The Teacher's verification function is the oracle, with rule-based answer extraction (MathRule) and an LLM judge. Validity rate is defined as the share of generated problems that Qwen3-235B-A22B-Instruct can solve.
- **Difficulty control**: The Gaussian utility centred on 50% success, plus zone-based generation. Training batches mix 25% historical curriculum for replay.
- **Reported results**:
  - Socratic-Solver-8B averaged +20.2 percentage points over prior data-synthesis methods across 7 math benchmarks.
  - Generator validity rate: 95.6%, vs 89.1% for its Qwen3-32B base.
  - Reward-shape ablation: all variants were within about 0.4 points (Gaussian μ = 0.5 gave 35.72 average).
- **Limitations / failure modes**: Depends on a very strong teacher and an LLM judge. Solvers were first LoRA-SFT'd on a 1,500-problem Level-5 set, so the setup is not fully data-free.
- **How to reuse with easy seed tasks**: Log your model's failures on seeds and on generated variants. Have a stronger model rewrite each failure into a targeted new problem, then distil that rewriting skill into a cheaper generator, weighting examples by closeness to 50% solver success.

### Absolute Zero Reasoner (AZR) — Absolute Zero: Reinforced Self-play Reasoning with Zero Data (Zhao et al., 2025)
Link: https://arxiv.org/abs/2505.03335

- **Mechanism**: A single LLM alternates PROPOSE and SOLVE roles over three task types built from verified (program p, input i, output o) triplets.
  - Deduction: predict o from (p, i).
  - Abduction: find an input that yields o, given p.
  - Induction: the proposer samples a program, writes N new inputs plus a message m, and the environment computes the outputs. The solver sees the *first half* of the I/O pairs plus m and must write a program that maps the held-out inputs correctly.
  - The proposer is conditioned on K reference triplets sampled from growing per-type buffers.
  - Proposer reward: r_propose = 0 if the mean solve rate r̄ = 0, else 1 − r̄. So fully solved tasks also get 0, and harder-but-solvable tasks get more. r̄ is estimated from G Monte Carlo solver rollouts at non-zero temperature.
  - Both roles are trained jointly with RL.
- **How it makes tasks harder**: One verified triplet yields three tasks of rising difficulty (forward, inverse, synthesis). The learnability reward pushes the proposer toward programs its own solver finds hard.
- **Correctness / verification**: The Python executor is the only source of truth.
  - Proposed programs must parse and run.
  - They must avoid blocked modules (e.g., os.sys, sys, shutil).
  - They must be deterministic: in practice each program is run j = 2 times and the outputs must match.
  - Abduction answers are accepted if re-executing p on the proposed input reproduces o. Induction answers are checked on held-out I/O pairs, which discourages if-else memorisation.
- **Difficulty control**: The learnability reward from G rollouts, plus conditioning on the buffer for novelty.
- **Reported results** (Qwen2.5-7B-Coder):
  - Coding average 56.6 → 61.6.
  - Math average 23.9 → 39.1.
  - Overall 40.2 → 50.4.
  - It beat zero-setting models trained on tens of thousands of curated in-domain examples.
  - The authors report an "uh-oh moment" of safety-concerning chain-of-thought (CoT) with Llama-3.1-8B.
- **Limitations / failure modes**:
  - Restricted to deterministic, executable tasks.
  - Chae et al. (arXiv 2510.27072) find that AZR-Coder improves pass@k at small k, but the base model is better at large k, so the gain is sharpening of existing ability. That large-k drop was *not statistically significant*, and the authors suggest AZR preserves the base model's capacity better than RLVR does.
  - The same study finds AZR still suffers entropy collapse. A 50%-targeted proposer reward did not help and cut validation accuracy by 2%.
  - PopuLoRA reports that a single AZR agent self-calibrates toward easy problems.
- **How to reuse with easy seed tasks**: If your seeds are functions or programs with inputs, invert them mechanically. Forward prediction becomes input search (abduction) and then program synthesis from partial I/O (induction), and all three stay execution-verifiable. Seed the buffers with your easy tasks, sandbox execution, and require determinism. Monitor entropy and pass@k at large k, not only pass@1.

### PopuLoRA — PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play (Creus Castanyer et al., 2026)
Link: https://arxiv.org/abs/2605.16727

- **Mechanism**: A population-based version of AZR-style asymmetric self-play. Teachers and students are specialized LoRA adapters on one shared frozen base. Teachers propose problems, and matched students solve them under a programmatic verifier. *Cross-evaluation between sub-populations* replaces the self-calibration of a single agent. LoRA weight-space mutation and crossover operators produce new same-rank members in seconds and serve as the replacement step of population-based training at 7B scale.
- **How it makes tasks harder**: A co-evolutionary arms race. Teachers are scored against students other than themselves, so they cannot settle on problems they themselves find easy. Teacher problem complexity rises, and coverage of the problem space keeps expanding.
- **Correctness / verification**: AZR's execution-based verifier.
- **Difficulty control**: Emergent, from cross-population evaluation. Student solve rates oscillate instead of saturating.
- **Reported results**: Compared with a per-adapter compute-matched single agent: despite lower training-time reward, the population mean beats the baseline on 3 code benchmarks (HumanEval+, MBPP+, LiveCodeBench) and 7 math benchmarks. Even the weakest member beats the baseline in aggregate.
- **Limitations / failure modes**: More orchestration. Tested only on top of AZR.
- **How to reuse with easy seed tasks**: Run several proposer adapters. Score each one's proposals on solvers it is *not* paired with, and periodically mutate or cross over the adapters. This is a cheap guard against a proposer collapsing onto easy variants of your seeds.

### GASP — GASP: Guided Asymmetric Self-Play For Coding LLMs (Jana et al., 2026)
Link: https://arxiv.org/abs/2603.15957 (ICLR 2026 workshops RSI [spotlight] and LLA)

- **Mechanism**: Self-play anchored to real *goalpost* problems (base Qwen2.5-Coder-7B).
  - From 601 LiveCodeBench training questions dated before 2024-08, the authors keep those with pass@100 = 0, then remove any that another RL run can solve. That leaves 146 goalposts, almost 25% of the split. The publicly released AZR checkpoint solves none of them.
  - Per goalpost, the teacher first writes an easier **lemma** conditioned on the goalpost. It is accepted if the student pass rate p ∈ [0.3, 0.7], with reward [4p(1−p)]^5, and gets −0.5 otherwise.
  - The teacher then writes a harder **lift** conditioned *only on the lemma*, deliberately not shown the goalpost, to avoid copying surface features. It is accepted if p ∈ [0.1, 0.5], with reward 10p((1−p)/0.9)^9, which peaks at p = 0.1, and gets −0.5 otherwise.
  - The teacher generates only induction-style tasks; deduction and abduction variants appear in the solver phase.
- **How it makes tasks harder**: Lemma → lift builds stepping stones from the student's frontier toward each unsolved goalpost.
- **Correctness / verification**: AZR-style execution validation (valid, safe, deterministic). Proposals with cosine similarity ≥ 0.95 to any buffered item (question text and code embeddings) are rejected.
- **Difficulty control**: Two pass-rate bands with differently shaped rewards, one centred and one skewed toward hard.
- **Reported results** (LCB v5, 216 questions):
  - pass@20: GASP 33.69 ± 0.28, AZR 31.15, real-data RL 33.10, GASP + real-data RL 34.46.
  - pass@1: 18.26 vs AZR's 17.49.
  - GASP solves 11 unique goalposts of 146 (6/3/3 per seed); all baselines solve 0.
  - One-step variants (medium or hard only, without the lemma) underperform.
- **Limitations / failure modes**: The authors observe that LLMs "often borrow surface-level metaphors from the original problem rather than producing clean conceptual stepping stones". When asked to increase difficulty, they "frequently do so by adding extra constraints, which is not always the most informative form of complexity". The goalpost set is fixed.
- **How to reuse with easy seed tasks**: Your hardest unsolved tasks, at pass@k = 0, are free goalposts. Generate an easier variant that falls in the 0.3–0.7 band, then a harder variant *of that variant* in the 0.1–0.5 band, without showing the goalpost again. Repeat. Explicitly instruct the generator to vary the core concept rather than add constraints.

### SOAR [added] — Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability (Sundaram et al., 2026)
Link: https://arxiv.org/abs/2601.18778

- **Mechanism**: Asymmetric self-play trained by **bilevel meta-RL** on Llama-3.2-3B-Instruct.
  - Outer loop: the teacher is trained with RLOO. Each sampled dataset of n = 64 teacher-written (question, answer) pairs is rewarded by how much a student trained on it (inner loop: RLOO, 10 steps, batch 8) improves greedy accuracy on 64 real hard questions. The reward is Acc(trained student) − Acc(baseline student).
  - The student reverts after each inner loop. A *promotion* mechanism updates the baseline student when the 3-step moving average of teacher reward exceeds τ = 0.01. The datasets that triggered promotions are kept as "promotion questions".
  - Run for 200 outer steps.
- **How it makes tasks harder**: The teacher learns to write the stepping stones that actually move the student on problems it currently fails 128/128 times. The difficulty of the stepping stones changes as the baseline student is promoted.
- **Correctness / verification**: The reward is grounded in the *real* hard set's answers. Synthetic answers are not verified.
  - Checked by an oracle judge (Claude-4.5-Sonnet), only 32.8% of useful promotion questions had fully correct solutions, while 63% were well-posed.
  - Adding well-posed questions with *incorrect* answers still helped.
  - An intrinsic-learnability teacher produced more correct questions (55%) but performed worse because its questions lacked diversity.
- **Difficulty control**: Implicit, through measured student improvement rather than a pass-rate target.
- **Reported results**:
  - On the held-out fail@128 test splits: about 4× pass@1 and 2× pass@32 on MATH, and 2× pass@1 and 1.5× pass@32 on HARP.
  - Questions transfer to OlympiadBench, which was held out.
  - Grounded teachers are more stable than intrinsic-reward teachers; one intrinsic teacher caused student collapse.
- **Limitations / failure modes**: Very expensive, because every teacher reward requires an inner RL run. Small models only. Answers are unverified.
- **How to reuse with easy seed tasks**: When you have a set of hard real tasks at p = 0, a cheap approximation works. Score candidate *batches* of generated easier or bridge tasks by a short probe fine-tune, measuring the gain on a held-out slice of the hard set, and keep the batches that move it. Prioritise well-posedness checks over answer checks for bridge data.

### VHG — Verifier-Backed Hard Problem Generation for Mathematical Reasoning (Lai et al., 2026)
Link: https://arxiv.org/abs/2605.06660

- **Mechanism**: Three-party self-play. The setter proposes (problem x, reference y*). An independent verifier V judges validity, and the solver's accuracy measures difficulty. Setter reward: R_Q = 1[V(x, y*) = 1] · (1 − Acc_S(x, y*)). The setter and solver are both Qwen3-4B.
- **How it makes tasks harder**: The setter is rewarded for solver failure *only on valid pairs*, which removes the "unsolvable nonsense" exploit of naive self-play. The paper shows transformations of seed problems that reach zero pass@1 for Qwen3-4B-Base.
- **Correctness / verification**:
  - Hard verifier for indefinite integrals: SymPy validates the format of f and F, differentiates F, and checks F′ = f.
  - Soft verifier for general math: hard-coded pre-filters (malformed outputs, trivial copies), then an LLM judge that checks the problem, the answer and their correspondence.
- **Difficulty control**: 1 − solver accuracy, gated by validity. Pass-rate filtering is used when building the solver data.
- **Reported results**:
  - Integrals: pass@1 +16.9% on AntiderivBench Qualifier, +16.6% on AntiderivBench Competition, and +21.4% on Integration Stress Test, beating R-Zero and other baselines.
  - General math (MATH, AMC, Minerva, Olympiad, AIME24–26): overall pass@1 56.8% → 69.0%.
  - Problems generated by the 4B models also challenge Qwen3-8B, 14B and 32B.
  - Integral funnel: 18,663 parsed, deduplicated candidates → 4,076 kept by the verifier (4,074 reference-valid), of which 1,144 had zero local pass rate.
  - General-math funnel: 230,532 template-deduplicated → 107,398 after solver filtering → 46,080 judged → 20,670 accepted (44.9%).
- **Limitations / failure modes**: The soft verifier inherits LLM-judge errors, and hard verifiers exist only for some domains.
- **How to reuse with easy seed tasks**: For any seed family with an *inverse check* (differentiate, substitute, simulate, execute), make the hard verifier the gate and reward the proposer by 1[valid] × (1 − solver accuracy). Otherwise use a separately prompted judge plus rule pre-filters, and expect around 40–50% acceptance.

### SPICE — SPICE: Self-Play In Corpus Environments Improves Reasoning (Liu et al., 2025)
Link: https://arxiv.org/abs/2510.24684

- **Mechanism**: One model plays two roles.
  - As Challenger it reads a document sampled from a 20,000-document corpus (Nemotron-CC-Math plus NaturalReasoning) and writes either a 4-option MCQ or a free-form question with a typed answer (integer, expression, string) extracted from the document.
  - As Reasoner it answers *without* the document.
  - Challenger reward, over K Reasoner correctness indicators: exp(−(Var − 0.25)² / (2·0.01)), which peaks at a 50% pass rate. Invalid questions get ρ = −0.1.
  - Settings: G = 8, DrGRPO with role-specific advantages, T = 640 iterations, B = 128.
- **How it makes tasks harder**: Information asymmetry (the Reasoner never sees the source document) plus a variance reward that tracks the Reasoner's improving ability.
- **Correctness / verification**: The gold answer comes from real text. Math-Verify checks expression equivalence, with exact match for other types.
- **Difficulty control**: Gaussian-on-variance reward.
- **Reported results**:
  - Qwen3-4B-Base overall 35.8 → 44.9, vs R-Zero 39.5, AZR 40.7, and a fixed Qwen3-32B "strong challenger" 43.0.
  - Other base models: Qwen3-8B 43.0 → 48.7; OctoThinker-3B 14.7 → 25.2; OctoThinker-8B 20.5 → 32.4.
  - Averages: +8.9 math and +9.8 general.
  - Ablations:
    - Corpus: grounded 43.9 vs ungrounded 40.7.
    - Challenger reward: variance 44.9, R-Zero reward 43.6, threshold 41.4, AZR reward 40.7.
    - Task type: MCQ only 42.0, free-form only 43.7 (best math, 52.5), mixed 44.9.
    - Corpus mix: both corpora beat either alone.
- **Limitations / failure modes**: Correctness depends on the Challenger extracting answers faithfully. MCQ answers can be guessed (our note). Coverage is limited by the corpus.
- **How to reuse with easy seed tasks**: Turn the domain material behind your easy seeds (textbooks, API docs, logs, papers) into a hidden-context environment. The proposer sees a passage and must emit a question with a *typed, extractable* answer; the solver answers without the passage. Mix MCQ, which is reliable to verify, with free-form, which forces richer reasoning.

### SPELL [added] — SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models (Yang et al., 2025)
Link: https://arxiv.org/abs/2509.23863 (ICLR 2026) · Code: https://github.com/Tongyi-Zhiwen/Qwen-Doc

- **Mechanism**: One policy plays three roles.
  - Questioner: writes (question, reference answer) from raw documents.
  - Responder: answers *with* the documents.
  - Verifier: judges semantic equivalence between the response and the reference. It is trained by self-consistency: each of its G judgments is rewarded for matching their majority.
  - Role-specific dynamic sampling balances training across the three roles.
- **How it makes tasks harder**: A **history memory** stores the L most recent *solvable* QA pairs and their source documents. Each new questioner context is the union of the stored documents, a newly sampled subset, and the past QA pairs. Two things therefore push difficulty up: (1) the document set in context grows, so questions must integrate more and longer material; (2) the prompt shows already-solved questions and asks for harder ones. A Gaussian-shaped questioner reward keeps difficulty near the responder's frontier.
- **Correctness / verification**: References come from the documents. The verifier replaces brittle string matching, and the paper includes an analysis of external judges.
- **Difficulty control**: Document-length curriculum plus the Gaussian reward.
- **Reported results**: Consistent gains on six long-context benchmarks across models. Qwen3-30B-A3B-Thinking gains an average of 7.6 points of pass@8.
- **Limitations / failure modes**: The self-trained verifier can drift. Long contexts make training expensive.
- **How to reuse with easy seed tasks**: For QA or reading seeds, grow the *amount of evidence* the answer depends on (more documents, longer spans, cross-document joins) while keeping answers extractable. Prompt the proposer with already-solved examples and ask it to beat them.

### Search Self-Play (SSP) — Search Self-play: Pushing the Frontier of Agent Capability without Supervision (Lu et al., 2025)
Link: https://arxiv.org/abs/2510.18821 (ICLR 2026) · Code: https://github.com/Qwen-Applications/SSP

- **Mechanism**: One LLM with a search tool plays both proposer and solver.
  - The proposer is given a ground-truth answer a* from a predefined answer set. That set is 50,000 answers sampled from public training data (NQ/HotpotQA from Search-R1, plus ARPO), averaging 14.53 words, with People 26.3% and Time & Dates 12.9%.
  - The proposer searches over several turns and writes a query that leads to a*.
  - The solver answers with search.
  - Proposer reward: 1 − mean solver success, on questions that pass verification. It is trained with REINFORCE (n = 1). The solver is trained with GRPO on binary correctness.
  - Dynamic sampling refills batches after filtering.
- **How it makes tasks harder**: Low solver success pays the proposer, so it learns to require more hops and less obvious evidence. Co-evolution matters: with a fixed proposer, solver reward saturates near 0.9 and learning stalls.
- **Correctness / verification**: RAG answerability check. All documents the proposer retrieved, plus unrelated documents from other trajectories in the batch, are given to a RAG solver *without* search, which must recover a*. The default RAG solver in experiments is Qwen2.5-32B-Instruct, for sampling efficiency.
  - Ablation (GeneralQA / Multi-HopQA): no RAG verification 49.5 / 36.7; 0 noisy docs 58.5 / 38.2; 1 noisy doc 58.5 / 36.9; **4 noisy docs 60.0 / 41.6**; 7 noisy docs 57.8 / 35.9.
- **Difficulty control**: 1 − solver success on verified questions.
- **Reported results**:
  - From scratch on Qwen2.5-7B-Base: average 22.3 → 48.7 (+26.4); NQ 32.0 → 54.2; TriviaQA 33.2 → 73.6; HotpotQA 18.0 → 52.8.
  - Qwen2.5-7B-Instruct: +8.0 average.
  - Continual RL: Search-R1-7B 53.9 → 55.7; ZeroSearch-7B 43.6 → 45.9.
- **Limitations / failure modes**:
  - *Non-unique answers* can pass the RAG check. For example, "Temptations singer" → Otis Williams passes only because the retrieved documents favour one member.
  - A small −0.1 penalty for invalid proposer output caused a "death spiral": proposer entropy rose, the valid-question rate went to 0, and the solver reward rose misleadingly through overfitting. The main configuration gives 0, not a negative reward.
  - Answers are entity-like.
- **How to reuse with easy seed tasks**: Invert easy QA seeds. Fix the answer, have an agent chain evidence hops away from it, and admit a question only if a separate no-tool reader, given the gathered evidence plus about 4 distractor documents, recovers the answer. Add a uniqueness check (e.g., ask the reader to list all valid answers). Do not penalize malformed proposals; give them zero reward instead.

### Self-play SWE-RL (SSR) — Toward Training Superintelligent Software Agents through Self-Play SWE-RL (Wei et al., 2025)
Link: https://arxiv.org/abs/2512.18552 (ICML 2026)

- **Mechanism**: One agent, starting from CWM-sft (32B), shares weights between two roles on sandboxed real repositories with dependencies installed. No issues or tests are needed as input.
  - As *injector* it produces a bug artifact:
    - `test_script.sh`;
    - `test_files.txt`, the oracle tests, reset before evaluation;
    - `test_parser.py`;
    - `bug_inject.diff`;
    - `test_weaken.diff`, which hides the bug. Reversing it gives the solver its specification.
  - As *solver* it repairs the bug, specified by tests rather than a natural-language issue.
  - Solver reward is binary: all tests pass.
- **How it makes tasks harder**:
  - Injection strategies compared: direct injection was worst, because it produced trivial one-line bugs such as `var = 0 → var = 1`. Removing code hunks or files while keeping the project runnable was better. **Removal + history-aware reversion of past commits** was best.
  - Control parameters set the minimum number of changed files, passing tests and failing tests.
  - **Higher-order bugs**: the solver's failed repair attempts become new buggy starting states. Order is capped at 2 to avoid overlapping existing bugs.
- **Correctness / verification**: Seven consistency checks:
  1. Test files exist and cover the weakened tests.
  2. The parser is valid.
  3. The script is valid, with at least `min_passing_tests` passing on the original.
  4. Bug scope is at least `min_changed_files`.
  5. The bug is valid: at least `min_failing_tests` previously passing tests fail.
  6. The test weakening is valid.
  7. **Inverse mutation testing**: reverting each modified file on its own must fix at least one failing test.
- **Difficulty control**: Injector reward r = −1.0 if the consistency check fails; −α if the solve rate s is 0 or 1; 1 − (1 + α)s for 0 < s < 1; with α = 0.8. The reward ablation found the solve-rate term only slightly better than a consistency-only ±1 reward, because solve-rate feedback from G samples is weak and noisy.
- **Reported results**: +10.4 on SWE-bench Verified and +7.8 on SWE-Bench Pro. SSR stays above the human-data RL baseline throughout training, even though evaluation uses natural-language issues that self-play never saw.
- **Limitations / failure modes**: The authors' analysis identifies degenerate challenger strategies:
  - A *dominant* challenger writes randomly-failing (or hash-seeded) tests, or obfuscates the codebase.
  - A *tunnel-vision* challenger turns a single knob, such as operand length or a chain of medium bugs.
  - Attempts to synthesize natural-language issues failed: they copied test patches and collapsed to patterns.
  - Repository-specialized training on 23 repos did not beat training on more diverse repos.
- **How to reuse with easy seed tasks**: For any codebase with passing tests, harden tasks by *deleting* functions or modules, or *reverting* historical commits, so the solver must reconstruct them. Stack failed repairs into second-order bugs. Always require fail-before / pass-after behaviour and inverse-mutation necessity, and reset hidden oracle tests before scoring.

### CURE — Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning (Wang et al., 2025)
Link: https://arxiv.org/abs/2506.03136 (NeurIPS 2025 spotlight) · Code: https://github.com/Gen-Verse/CURE

- **Mechanism**: Training uses 4.5K CodeContests problems (difficulty ≤ 2) that have ground-truth unit tests but *no* ground-truth code. For each task, the policy generates 16 candidate solutions and 16 unit tests, and an execution table runs every solution against both generated and ground-truth tests.
  - A solution's reward is the number of ground-truth tests it passes.
  - A generated test's reward comes from a reward-precision analysis: it is rewarded for passing correct solutions and failing incorrect ones.
  - For long-CoT models, a length-aware reward transformation penalizes overly long test-generation responses.
- **How it makes tasks harder**: The coder's mistakes become training signal for a tester that writes more discriminating tests, which in turn makes passing the tests harder.
- **Correctness / verification**: Ground-truth tests define which solutions count as correct. Generated tests are rewarded only for agreeing with that separation.
- **Difficulty control**: Implicit.
- **Reported results**:
  - ReasonFlux-Coder-7B/14B: +5.3% code-generation accuracy and +9.0% Best-of-N, over Qwen2.5-Instruct.
  - +8.1% on downstream agentic coding.
  - ReasonFlux-Coder-4B, a long-CoT model, beats Qwen3-4B "while achieving 64.8% inference efficiency in unit test generation".
  - The trained tester works as a reward model for RL.
- **Limitations / failure modes**: Needs some ground-truth tests per task. Code only.
- **How to reuse with easy seed tasks**: Seeds that are "solved" under weak tests are not really solved. Train or prompt a test generator on the coder's failed solutions to write edge-case tests, keep only tests consistent with the reference, and use the stricter test suite as the new reward.

### Self-Challenging Agents (SCA) — Self-Challenging Language Model Agents (Zhou et al., 2025)
Link: https://arxiv.org/abs/2506.01716

- **Mechanism**: As challenger, the agent first *interacts with the tools* to see what is feasible, then proposes a **Code-as-Task (CaT)**: an instruction, a verification function in code, an example solution, and enumerated failure cases. As executor, the same agent trains on the filtered tasks with RL, using the verifier as reward.
  - Environments: M³ToolEval (calculation, web browsing) and Tau-Bench (Retail, Airline).
- **How it makes tasks harder**: Exploration grounds proposals in states the environment can actually reach, and the tasks are open-ended. There is no explicit difficulty objective.
- **Correctness / verification**: A task is kept only if the example solution passes the verifier and every failure case fails it. Yields with Llama-3.1-8B in Retail:
  - 47.7% for instruction + verifier, with only the check that the verifier runs.
  - 9.5% when the solution must also pass the verifier.
  - **5.2%** for full CaT, adding the failure cases.
  - Human labelling shows CaT sharply reduces false positives and false negatives. The remaining false negatives come from *incomplete instructions*, e.g., "return one of my latest orders" without saying which.
- **Difficulty control**: None explicit. Difficulty is measured afterwards by the length of the example solution. CaT filtering *narrowed* the task distribution for the weaker 8B model.
- **Reported results**:
  - Self-improvement: Llama-3.1-8B-Instruct average pass@1 12.0% → 23.5%, a 95.8% relative gain.
  - Distillation from Llama-3.1-70B: 32.2% (+20.2).
- **Limitations / failure modes**: Low yield, underspecified instructions, no targeting of difficulty.
- **How to reuse with easy seed tasks**: Require every synthesized agentic task to carry a verifier, a reference solution *and* negative solutions. Reject any task where the reference fails or a negative passes. Add a pass-rate band (e.g., 0.2–0.8) under the current policy to target difficulty, and an "instruction completeness" check using an LLM judge.

### DeepSeek-V3.2 large-scale agentic task synthesis — DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek-AI, 2025)
Link: https://arxiv.org/abs/2512.02556

- **Mechanism**: An environment-synthesis agent, given a task category (e.g., trip planning) and a sandbox with bash and search:
  1. collects or generates data into a sandbox database;
  2. implements task-specific tools as functions;
  3. proposes a simple task with a Python *solution function* and *verification function*;
  4. repairs the solution or verifier until they agree;
  5. then "**iteratively increases the difficulty of the task and updates the corresponding solution and verification functions**", augmenting the toolset when it is insufficient.
  - The design principle is "hard to solve, easy to verify". In the trip-planning example, searching the combinatorial space of plans is hard, but checking a plan against the constraints is easy.
- **How it makes tasks harder**: The agent escalates tasks step by step with more constraints, steps and tools, and co-evolves the verifier alongside them.
- **Correctness / verification**: The solution function may only call tool functions or do logical computation, and cannot read the database directly. This guarantees the task is solvable *through the tool interface*. Only instances with non-zero pass@100 under DeepSeek-V3.2 RL are kept.
- **Difficulty control**: Iterative escalation, then the pass@100 > 0 solvability filter.
- **Reported results**: 1,827 synthesized environments with 4,417 general-agent tasks. The same report lists 24,667 code-agent tasks from real environments, 50,275 synthesized search-agent tasks, and 5,908 code-interpreter tasks.
- **Limitations / failure modes**: Few details on how escalation works. Verifier bugs are possible when the agent writes its own verifiers.
- **How to reuse with easy seed tasks**: Take an easy agent task and its tools. Have a synthesis agent add constraints, steps or tools one at a time, re-deriving a *tool-only* reference solution and a verifier at each step. Keep the hardest version that still has pass@k > 0 under your policy.

### SQL-Zero [added] — SQL-Zero: Self-Evolving Text-to-SQL (Pedrozo et al., 2026)
Link: https://arxiv.org/abs/2609.04697

- **Mechanism**: Challenger and solver start from the same base model and alternate GRPO turns, with the other role frozen. Given a database schema and a *target complexity level*, the challenger writes SQL first, then the natural-language question.
  - Challenger reward: max(0, ½f + d − ρ).
    - f is a format and integrity term.
    - d(k) = (n − k)/(n − 1) for 0 < k < n and 0 otherwise, where k is the number of execution matches out of n = 5 solver samples. It peaks at k = 1: hard but solvable.
    - ρ is a repetition penalty computed over masked-SQL-template clusters (R-Zero's form, with templates in place of BLEU).
- **How it makes tasks harder**: Complexity-level conditioning, plus a difficulty term that favours pairs the frozen solver gets right only once in five.
- **Correctness / verification**: **Execution against the database is the only ground truth.** A pair is admitted only if its gold SQL is a deterministic SELECT with a *non-empty* result and the question does not leak SQL. Admitted pairs are deduplicated to one per database–template pair.
- **Difficulty control**: d(k) plus the complexity-level prompt.
- **Reported results**: With no labels on BIRD databases, BIRD dev improved over the zero-shot base by 6.6 points at 3B and 7.3 at 7B. It was above a matched control trained on human gold pairs, though a paired test does not resolve that margin. At 3B every iteration transferred to unseen Spider databases and Spider-Syn; at 7B only the first iteration preserved transfer.
- **Limitations / failure modes**: Transfer degrades with more iterations at the larger scale. Execution match can accept a query that returns the right result for the wrong reason.
- **How to reuse with easy seed tasks**: For any seed family with an executable artifact (SQL, regex, spreadsheet formula, API call), generate the *artifact first*, execute it for the gold output, then have the model write the natural-language task. Reject empty or degenerate outputs and deduplicate by masked template.

### OPT-Zero [added] — Learning to Optimize through Solver-Grounded Self-Play (Jiang et al., 2026)
Link: https://arxiv.org/abs/2609.34205

- **Mechanism**: One LLM plays Proposer and Solver for optimization modeling.
  - The Proposer, conditioned on K reference problems from a buffer, emits a formal model (sense σ, objective f, variables V, constraints C), a natural-language scenario s, and solver code.
  - The Solver sees only s and must produce its own model and code.
  - Proposer reward R^P = R_valid × R_correct × R_struct:
    - R_valid: the model is parseable, feasible, bounded and solved to optimality by a canonical solver (e.g., Gurobi).
    - R_correct ∈ {α_c, 1}: whether the Proposer's code reproduces the solver's optimum.
    - R_struct ∈ [α_s, 1]: a weighted structural score over variable utilization, non-trivial constraint count, coupling density, variable coverage and constraint connectivity.
  - The structural target is re-estimated each iteration from the medium-difficulty problems for the current Solver.
- **How it makes tasks harder**: It rewards *structural* complexity: coupled constraints that need relational reasoning across variables. It does not reward instance size or solve rate directly, and it penalizes duplicated components.
- **Correctness / verification**: A canonical solver pipeline supplies P*. The Solver is scored on model correctness (objective vs P*, with rule-based partial credit) and on code execution.
- **Difficulty control**: A structural target tracked from the Solver's frontier. The structural floor α_s keeps simple early problems from getting near-zero reward.
- **Reported results**: With zero curated data it matches state-of-the-art data-dependent methods and generalizes better (abstract). Ablations: a binary proposer reward degrades performance, and structural-reward GRPO beats a solve-rate-based baseline.
- **Limitations / failure modes**: Domain-specific structural metrics must be designed by hand.
- **How to reuse with easy seed tasks**: When your domain has a formal backbone (LP/MIP, SAT/SMT, graphs, schedules), define "harder" as *structural* properties (coupling, connectivity, non-redundant constraints) measured on the formal object. Gate on solver-verified validity and have the proposer's code reproduce the solver's answer.

### SPIRAL — SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning (Liu et al., 2025)
Link: https://arxiv.org/abs/2506.24119 (ICLR 2026) · Code: https://github.com/spiral-rl/spiral

- **Mechanism**: A fully online, multi-turn, multi-agent RL system in which the model plays zero-sum text games (TicTacToe, Kuhn Poker, Simple Negotiation) against continuously updated copies of itself. **Role-conditioned Advantage Estimation (RAE)** normalizes rewards relative to each player's expected performance, with separate baselines per game and role.
- **How it makes tasks harder**: The opponent improves as the policy improves, giving an automatic curriculum with no task authoring.
- **Correctness / verification**: Game rules and engines, so the reward is exact.
- **Difficulty control**: Implicit, through opponent strength.
- **Reported results**:
  - Up to 10% improvement across 8 reasoning benchmarks on 4 Qwen and Llama models.
  - Beats SFT on 25,000 expert game trajectories. Multi-game training is best.
  - Still helps DeepSeek-R1-Distill-Qwen-7B.
  - Without RAE, models show "thinking collapse", abandoning reasoning traces after about 200 steps.
  - Fixed-opponent training fails, and random opponents cause collapse.
- **Limitations / failure modes**: Transfer is limited to the skills the games exercise, and the games are hand-designed.
- **How to reuse with easy seed tasks**: Recast puzzle or negotiation seeds as two-player adversarial games, e.g., one side constructs an instance and the other solves it, or a bargaining game. Train against the latest self, or a pool of past checkpoints, with per-role baselines.

### eva — Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play (Ye et al., 2024)
Link: https://arxiv.org/abs/2411.00062 (NeurIPS 2024 Language Gamification workshop, spotlight)

- **Mechanism**: A creator–solver game with regret-based signals.
  - The creator estimates each prompt's informativeness as |estimated regret|, i.e., the reward advantage between the best observed response and a baseline under a reward model.
  - It samples prompts weighted by informativeness and evolves them with an off-the-shelf rewriter (default: WizardLM-style in-depth and in-breadth instructions).
  - The solver trains on the evolved prompts with DPO, SPPO or RLOO.
  - The online version keeps a prioritized generative buffer that goes through warm-up, mix-up and bootstrap phases.
- **How it makes tasks harder**: Complexity-raising rewrites applied *selectively* to high-regret prompts.
- **Correctness / verification**: None beyond the reward model.
- **Difficulty control**: Regret-weighted sampling. This is the UED idea (ACCEL-style edits to high-regret levels) applied to prompts.
- **Reported results** (gemma-2-9b-it, UltraFeedback):
  - Arena-Hard 51.6% → 60.1% with DPO, and 52.6% → 62.4% with RLOO.
  - Uniform evolution gives 57.5%; adding *new human prompts* gives 59.8%.
  - Advantage-based informativeness beats variance or average-reward heuristics and the inverse-advantage variant.
- **Limitations / failure modes**: Reward-model exploitation; not verifiable.
- **How to reuse with easy seed tasks**: For non-verifiable seeds, score each prompt by the reward gap between the best and a baseline sampled response. Apply complexity-increasing rewrites only to the top of that ranking, not uniformly.

### SvS — Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR (Liang et al., 2025)
Link: https://arxiv.org/abs/2508.14029

- **Mechanism**: Online, inside RLVR. Each step:
  1. The policy solves the originals (G = 8 samples each).
  2. "Underperforming" problems, with accuracy in [12.5%, 50%], are selected. Each *correct* solution to them is used as context for the policy to write G_v = 8 **variational problems**: structurally different reformulations that must keep the original reference answer.
  3. The policy solves the variants, graded against the original answer.
  - The synthesis role is rewarded only when a variant's solve rate lands in **[12.5%, 62.5%]**.
- **How it makes tasks harder**: It rewrites the problems the policy is weakest on into new problems with the same answer. Because the rewrite starts from a correct solution, the underlying structure stays solvable while the surface and setup change.
- **Correctness / verification**: Variants inherit the verified answer. The band rejects variants solved by every rollout (trivial, or leaking the answer) and by none (likely broken).
- **Difficulty control**: Selection of underperforming problems, plus the reward band.
- **Reported results**: Qwen2.5-32B-Instruct on DAPO-17k gained +18.3 and +22.8 absolute pass@32 on AIME24 and AIME25 over standard RLVR. Policy entropy stays stable while RLVR's decays. Gains are consistent across 12 benchmarks and 3B–32B models, and also appear on code.
- **Limitations / failure modes**: Integer-only seed answers (DAPO-17k) cause answer-format overfitting. The authors' "D25k" mix adds 8k open-ended-answer DeepMath problems. Variants with the same answer can still leak it.
- **How to reuse with easy seed tasks**: Every time the policy *solves* a seed it usually fails, have it write same-answer reformulations as new RL prompts, conditioned on its own correct solution, and gate them by the band. This is cheap, answer-verified, and keeps pass@k and entropy from collapsing. Mix answer formats in the seeds.

### SwS — SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning (Liang et al., 2025)
Link: https://arxiv.org/abs/2506.08989

- **Mechanism**:
  1. A preliminary RL run logs per-problem accuracy over epochs. A **weakness** is a problem that never reaches 50% accuracy and has a negative accuracy slope: F(x) = 1[max_t a_t < 0.5 ∧ slope < 0].
  2. LLaMA-3.3-70B-Instruct extracts core concepts from the weak problems. Concepts are embedded with LLaMA-3.1-8B-base and recombined within each category using co-occurrence and similarity.
  3. LLaMA-3.3-70B-Instruct writes new problems from the category label and concept set, allocating budget across categories by failure share.
  4. LLaMA-3.3-70B and Qwen2.5-72B filter out low-quality problems.
  5. Reference answers come from Skywork-OR1-Math-7B for student models up to 7B, or from QwQ-32B for 32B students.
  6. Training continues on the augmented set. 40k synthetic problems are generated per base model (Qwen2.5 base, 3B–32B).
- **How it makes tasks harder**: It composes several concepts the model is weak on into new problems in the same category.
- **Correctness / verification**: A revised self-consistency rule. The teacher's consistent answer must exceed 50% of its samples, *and* the student must produce that answer in at least 25% of its responses.
- **Difficulty control**: Keep problems whose current-policy accuracy is in [25%, 75%]; about 35% of generated problems survive.
- **Reported results**: Average gains of +10.0% (7B) and +7.7% (32B) across 8 math benchmarks.
- **Limitations / failure modes**: Relies on large external generator and labeller models. Self-consistency labels can be wrong. The student-agreement rule biases toward problems the student can already partly solve.
- **How to reuse with easy seed tasks**: Log per-seed accuracy curves during your current RL run. Cluster the persistent failures by concept, synthesize problems that *combine* those concepts, label them with a stronger model's majority answer, and keep only those in the 25–75% band.

### UED family (PAIRED, PLR, ACCEL, POET, SFL) — Evolving Curricula with Regret-Based Environment Design (Parker-Holder et al., 2022)
Link: https://arxiv.org/abs/2203.01302 · PAIRED https://arxiv.org/abs/2012.02096 · PLR https://arxiv.org/abs/2010.03934 · POET https://arxiv.org/abs/1901.01753 · SFL https://arxiv.org/abs/2408.15099

- **Mechanism**: The conceptual root of LLM self-play curricula.
  - PAIRED (Dennis et al., 2020): an adversary generates environments that maximize *regret*, the antagonist's return minus the protagonist's. This avoids the unsolvable worst cases that plain minimax produces.
  - PLR (Jiang et al., 2020): replays procedurally generated levels, prioritized by estimated learning potential (TD-error based).
  - ACCEL (Parker-Holder et al., 2022): makes *small edits* to high-regret levels in the replay buffer, so complexity compounds from the frontier.
  - POET (Wang et al., 2019): co-evolves environments and agents under a minimal criterion and transfers agents between environments as stepping stones.
  - SFL (Rutherford et al., 2024): shows that practical regret approximations track *success rate* rather than regret. Training directly on high-*learnability* p(1−p) levels beats them.
- **How it makes tasks harder**: Edit or mutate levels that are already at the frontier; replay levels by priority.
- **Correctness / verification**: Simulators and generators define solvability and reward.
- **Difficulty control**: Regret or learnability scores, a replay buffer and edit operators.
- **Reported results**: PLR raised test return on Procgen to over 76% improvement relative to standard RL baselines, in combination with the previous leading method. The other results are qualitative in the abstracts.
- **Limitations / failure modes**: Regret estimates are noisy. The methods target low-dimensional environment parameters, not text.
- **How to reuse with easy seed tasks**: Treat your seed pool as the level buffer. Score seeds by p(1−p), and generate new tasks mainly by *small edits to frontier seeds* (ACCEL) rather than from scratch. Admit new tasks under a minimal criterion band (POET).

### LILO — LILO: Learning to Reason at the Frontier of Learnability (Foster et al., 2025)
Link: https://arxiv.org/abs/2502.12272

- **Mechanism**: "LILO" stands for Learnability Improves LLMs Optimally; it adapts SFL to LLM RL.
  1. Sample a candidate pool |D| = 4|B|.
  2. Roll out N_learnability = 8 attempts per question.
  3. Compute p̂ and learnability p̂(1 − p̂).
  4. Train on the top-|B| questions.
  - For GSM8K near 95% training accuracy, |D| = 8|B| was needed. When N_train is large (e.g., VinePPO around 500), the extra sampling is about 4%.
- **How it makes tasks harder**: It does not. It concentrates training on questions the policy solves only sometimes.
- **Correctness / verification**: Existing dataset answers.
- **Difficulty control**: p(1−p), recomputed online.
- **Reported results** (Table 2 of the paper; speed-up = steps to reach the baseline's best):
  - PPO on MATH: 2.5×, 19.1 → 21.8.
  - PPO on GSM8K: 1.9×, 51.1 → 53.2.
  - VinePPO on MATH: 3.2×, 22.8 → 24.9.
  - VinePPO on GSM8K: 3.3×, 53.2 → 55.9.
  - GRPO on ORZ57K: 1.5×, 35.5 → 37.1.
  - The input's caveat that these numbers came only from an alphaXiv summary is withdrawn: they are in the paper.
- **Limitations / failure modes**: Exploratory rollouts cost extra when N_train is small. It cannot create signal once the pool is saturated.
- **How to reuse with easy seed tasks**: This is layer 0. Estimate p on every seed with about 8 rollouts, drop p ∈ {0, 1}, and sample by p(1−p). The p = 1 bucket becomes the *input* to your complexification operators, and the p = 0 bucket goes to scaffolding.

### Online difficulty filtering and cheap difficulty prediction (Bae et al.; DAPO; GRESO; PCL; MoPPS; DOTS) — Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning (Bae et al., 2025)
Link: https://arxiv.org/abs/2504.03380 (EACL 2026) · DAPO https://arxiv.org/abs/2503.14476 · GRESO https://arxiv.org/abs/2506.02177 · PCL https://arxiv.org/abs/2510.01135 · MoPPS https://arxiv.org/abs/2507.04632 · DOTS https://arxiv.org/abs/2506.05316

- **Mechanism**:
  - Bae et al. show that the reverse KL between the initial policy and the KL-regularized optimal policy is at least p(x)(1 − p(x))/(2β²). The bound is maximized at p = ½ and is zero at p ∈ {0, 1}. So *balanced* filtering, with thresholds symmetric around 0.5 (e.g., 0.2–0.8), maximizes the lower bound on learnability. Batches are refilled by asynchronous sampling.
  - DAPO over-samples and drops groups whose accuracy is exactly 0 or 1.
  - GRESO skips prompts *before* rollout, because a prompt that is uninformative in one epoch tends to stay so.
  - PCL trains a value model alongside the policy to predict intermediate-difficulty prompts.
  - MoPPS treats each prompt's success rate as a latent variable, with streaming Bayesian inference and posterior-sampling bandits.
  - DOTS estimates adaptive difficulty by attention-based similarity to a small reference set that does get rollouts, and replays recent rollouts.
- **How it makes tasks harder**: Selection only.
- **Correctness / verification**: Existing verifiers.
- **Difficulty control**: Pass-rate bands or predicted difficulty.
- **Reported results**:
  - Bae: +10% AIME and +4.2 average (3B); +12% AMC and +4.5 average (7B); beats plain GRPO in under half the gradient updates.
  - DAPO: 50 points on AIME 2024 from Qwen2.5-32B base.
  - GRESO: up to 2.4× rollout and 2.0× total training speedup with no accuracy loss.
  - PCL: 12.1× (MATH) and 16.9× (DeepScaleR) faster at finding intermediate-difficulty prompts than rollout filtering.
  - DOTS: 23–62% less RL time to reach GRPO-level performance.
- **Limitations / failure modes**: Filtering wastes compute on a mostly saturated pool. Bands are policy-relative, and p̂ from small G is noisy.
- **How to reuse with easy seed tasks**: Log the effective prompt ratio. Use a balanced band plus a cheap predictor to avoid rolling out on saturated seeds. When the band empties, switch the budget to generating new tasks, and put the same predictors on *generated* candidates before paying for full rollouts.

### RLVE — RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments (Zeng et al., 2025)
Link: https://arxiv.org/abs/2511.07317 (ICML 2026)

- **Mechanism**: Each environment E = (input template I, generator P_d conditioned on integer difficulty d, verifier R). RLVE-Gym contains 400 hand-engineered environments.
  - Each environment keeps a difficulty range [ℓ, h], starting at [0, 0].
  - Once more than τ_num samples have been seen at level h and accuracy there is at least τ_acc, h is raised by 1.
  - A sliding window sets ℓ = h − d_Δ + 1 when the range exceeds d_Δ.
  - Defaults: τ_acc = 0.9, τ_num = 8 × rollouts per problem, d_Δ = 4.
- **How it makes tasks harder**: Size and structure knobs, for example:
  - Sorting: array length N ≈ 3 × 1.1^d.
  - Integration: the expression tree has d + 2 nodes.
  - Sudoku: grid dimension up to d + 2.
- **Correctness / verification**: Algorithmic verifiers, with shaped partial credit such as (x/N)^10 for sorting or (min/max)^10 for numeric answers. Generators exploit **solve–verify asymmetry**:
  - Integration: generate F, give F′, and verify the model's answer by differentiation, so no integral ever has to be computed.
  - Sudoku: generate a full solution, then mask cells.
- **Difficulty control**: The per-environment adaptive window keeps the effective prompt ratio high.
- **Reported results**:
  - From ProRL-1.5B-v2, which had already had more than 20,000 H100 hours of RLVR: joint training on all 400 environments gave +3.37 absolute across six reasoning benchmarks in about 1,100 H100 hours. Continuing the original RL gave +0.49 with more than 3× the compute.
  - Expanding the set of environments consistently improves held-out generalization.
  - Cost comparison: building DeepMath-103K took about $138,000 and 127,000 GPU hours.
- **Limitations / failure modes**: Environments are hand-engineered. Some knobs are only proxies for real reasoning difficulty.
- **How to reuse with easy seed tasks**: Turn each easy seed family into a parameterized generator with a verifier (size, depth, number of constraints, distractors, hops). Where you can, build the answer first and derive the problem from it. Attach the τ_acc/τ_num rule, and spend engineering effort on *more families* rather than more instances.

### SEC (+ AdaRFT, E2H, DUMP) — Self-Evolving Curriculum for LLM Reasoning (Chen et al., 2025)
Link: https://arxiv.org/abs/2505.14970 · AdaRFT https://arxiv.org/abs/2504.05520 · E2H https://arxiv.org/abs/2506.06632 · DUMP https://arxiv.org/abs/2504.09710

- **Mechanism**:
  - SEC: each problem category (difficulty level or type) is an arm of a non-stationary multi-armed bandit. The arm's reward is the mean *absolute advantage* of its samples, a proxy for immediate learning gain, and values are updated by TD(0).
  - AdaRFT (TMLR): moves a target difficulty according to recent reward and samples problems near it.
  - E2H Reasoner: schedules tasks easy → hard and finds that *fading out* easy tasks is essential to avoid overfitting. It provides convergence and finite-sample guarantees.
  - DUMP: runs UCB over data *distributions*, using advantage magnitude.
- **How it makes tasks harder**: Scheduling only.
- **Correctness / verification**: Existing verifiers.
- **Difficulty control**: |advantage| bandits, reward-tracked target difficulty, or schedules.
- **Reported results**:
  - SEC: better generalization to harder out-of-distribution problems and better skill balance in multi-domain training (qualitative in the abstract).
  - AdaRFT: up to 2× less training time, given problem-level difficulty annotations.
  - E2H: large gains for 1.5B–3B models that struggle with vanilla RL.
- **Limitations / failure modes**: Needs categories or difficulty labels. Mordig et al. (2603.27226) find no robust benefit from difficulty-ordered curricula over random sampling on deductive reasoning, for either SFT or RL.
- **How to reuse with easy seed tasks**: Bucket *generated* tasks by operator depth or knob value, let a bandit on |advantage| allocate sampling across buckets, and retire buckets once mastered.

### QuestA — QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation (Li et al., 2025)
Link: https://arxiv.org/abs/2507.13266 (ICLR 2026) · Code: https://github.com/foreverlasting1202/QuestA

- **Mechanism**: Problems where the base pass rate is near zero get a prefix containing the first p% of a reference solution.
  - DeepSeek-R1-Distill-1.5B, used as a weak selection model, filters OpenR1-Math-220K down to the 26K hardest problems.
  - Augmented prompts are kept if the model gets **0–4 of 8** correct; 4/8 is the point of maximum variance.
  - Training on Nemotron-1.5B: Partial-50 for 100 steps, then Partial-25 for 1,900 steps. The switch point is where entropy starts to fall. Also n = 16 rollouts and DAPO-style dynamic filtering.
  - The theory says a hint reduces the rollouts needed from Θ(1/δ_p) to O(1/δ_p′), with δ_p′ = δ_p^(1/2 − ε).
- **How it makes tasks harder**: It runs in reverse: it makes unsolvable tasks learnable, then removes the help.
- **Correctness / verification**: The final answer is unchanged, and prefixes come from DeepSeek-R1 reference solutions.
- **Difficulty control**: The prefix fraction (50% → 25%) and the 0–4-of-8 filter.
- **Reported results**:
  - Nemotron-1.5B → QuestA: AIME24 72.50% (+10.73), AIME25 62.29% (+12.79), HMMT FEB 25 41.67% (+10.11).
  - Curriculum average 63.26 vs 60.26 for Partial-50 only.
  - Extending to Partial-0 gave no further gain.
  - Improves pass@k across k, and pass rates on training prompts *without* hints rise.
- **Limitations / failure modes**: Needs reference solutions, and hints must be annealed.
- **How to reuse with easy seed tasks**: For your p = 0 seeds, prefix part of a verified solution, choosing the fraction that puts pass rate in 0–4/8. Train, then shrink the prefix. It pairs well with any generator that produces harder-than-frontier variants.

### SEELE — Staying in the Sweet Spot: Responsive Reasoning Evolution via Capability-Adaptive Hint Scaffolding (Li et al., 2025)
Link: https://arxiv.org/abs/2509.06923

- **Mechanism**: The paper relates loss-descent speed to rollout accuracy, which defines a high-efficiency accuracy region. Each problem gets a hint made of part of a full solution, but its length is set *per instance, in real time*. Rollouts are sampled in several rounds, and after each round an **item response theory (IRT)** model fitted to the (hint length, accuracy) pairs so far predicts the hint length that will hit the target accuracy next round.
- **How it makes tasks harder**: As capability grows, less of the solution needs to be revealed.
- **Correctness / verification**: Original answers; hints are prefixes of verified solutions.
- **Difficulty control**: The IRT mapping from hint length to accuracy.
- **Reported results**: Averaged over six math benchmarks: +11.8 over GRPO, +10.5 over SFT, and +3.6 over the best previous supervision-aided method.
- **Limitations / failure modes**: Multi-round sampling adds cost. Needs full solutions.
- **How to reuse with easy seed tasks**: Treat "fraction of the solution revealed" as a continuous difficulty knob, fitted per seed with a logistic curve, and target about 50% success rather than a fixed ratio.

### Scaf-GRPO — Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning (Zhang et al., 2025)
Link: https://arxiv.org/abs/2510.19807 (ICLR 2026) · Code: https://github.com/JIA-Lab-research/Scaf-GRPO

- **Mechanism**: No guidance during the first 15% of steps, the exemption period. Exemptions of 10–40% all gave a 49.5–50.9 plateau. After that, problems with persistent all-zero groups are "true-hard".
  - For each such problem, the method searches pre-generated hints in order of decreasing abstraction: knowledge → planning → solution. The hints are generated by DeepSeek-R1, which scored higher on hint quality than Qwen2.5-72B-Instruct.
  - It stops at the first hint level that yields a correct on-policy completion, and that trajectory *replaces one failed trajectory* in the GRPO group, which restores non-zero advantage.
- **How it makes tasks harder**: It does not. It reclaims gradient from problems that are too hard.
- **Correctness / verification**: The original verifier.
- **Difficulty control**: Triggered only by learning plateaus, using the minimal sufficient hint.
- **Reported results** (Qwen2.5-Math-7B):
  - AIME24 30.0 → 43.3 over vanilla GRPO (+44.3% relative).
  - Benchmark average 45.2 → 50.9 (+12.6% *relative*), vs LUFFY 46.6 (+9.2% relative).
  - Hint-guided exploration was triggered for 17.4% of samples.
- **Limitations / failure modes**: Hints must be precomputed by a strong model. There is only approximate treatment of the off-policy prefix.
- **How to reuse with easy seed tasks**: Precompute a hint ladder (concept, plan, step) for each p = 0 seed and inject the smallest rung that produces a success, only after a hint-free warm-up.

### MFC (+ R3) — Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR (Zhu & Han, 2026)
Link: https://arxiv.org/abs/2609.13997 (Findings of EMNLP 2026) · R3 https://arxiv.org/abs/2402.05808

- **Mechanism**:
  - MFC: partial reasoning traces from a stronger teacher turn all-fail problems into a graded difficulty landscape. A backward-chaining curriculum withdraws guidance until the model solves them unaided. **Monotone Frontier Curriculum (MFC)** forces the schedule to move monotonically toward unguided solving, countering a distribution-shift cost that is acute when training only on unsolvable problems.
  - R3 (Xi et al., 2024): slides the reasoning start state from the end of a correct demonstration back to its beginning, so outcome rewards act like step-level supervision.
- **How it makes tasks harder**: The start state moves earlier, and the task gets harder, as competence grows.
- **Correctness / verification**: Original answers; guidance comes from verified or teacher traces.
- **Difficulty control**: The amount of teacher trace, withdrawn monotonically.
- **Reported results**:
  - MFC: training on only 128 unsolvable problems matches or beats GRPO on a full 2,000-problem corpus (about 16× data efficiency) on a nine-benchmark average for both base models. It substantially expands pass@k at large k.
  - R3: +4.1 average over the RL baseline on eight reasoning tasks (Llama2-7B), and +4.2 on program-based GSM8K across three backbones.
- **Limitations / failure modes**: Needs teacher or demonstration traces. Distribution shift when hints dominate training.
- **How to reuse with easy seed tasks**: Your p = 0 pool is the most valuable data. Run a reverse curriculum on it, starting near the end of a reference solution and moving the start earlier as success rises, until no guidance remains.

### Self-Improving Transformers — Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges (Lee et al., 2025)
Link: https://arxiv.org/abs/2502.01612

- **Mechanism**: Supervised self-improvement.
  1. Train on labelled data up to difficulty d₀.
  2. In round r, the current model labels slightly harder instances, e.g., 50,000 examples one digit longer.
  3. Retrain on the union of all rounds.
  - An *accelerated* schedule samples every difficulty level already above 99% evaluation accuracy, instead of advancing one level per round.
- **How it makes tasks harder**: A monotone size, length or hop knob.
- **Correctness / verification**: Self-labels plus unsupervised filters.
  - *Length filtering*: for multiplication, drop outputs shorter than the batch's longest by more than 10 tokens, since dropped intermediate steps show up as short outputs.
  - *Majority voting*: train 5 models with different seeds and keep labels on which at least 4 agree.
  - Without filtering, an **error avalanche** spreads mistakes to later rounds.
- **Difficulty control**: A hand-set schedule.
- **Reported results**:
  - Reverse addition: trained on 1–16 digits, generalizes to over 100 digits.
  - String copy/reverse: trained on length 1–10, generalizes to over 120 after about 100 rounds.
  - Forward addition: trained on 1–10 digits, generalizes up to 75 digits over 60 rounds with length filtering, staying above 98% up to length 70.
  - Multiplication without filtering: 6×6 accuracy only 13.7% after 7 rounds.
  - Majority voting raised 5×6 label accuracy from an average of 31% to 93.3%.
  - 5×5 → 10×10 took 41 rounds on the standard schedule and 19 on the accelerated one.
  - Maze hops: 9 → 30 with majority voting.
- **Limitations / failure modes**: Needs a clean size knob. Structured label noise is dangerous.
- **How to reuse with easy seed tasks**: For SFT on seeds with a size parameter but no cheap verifier at large sizes, extend one step at a time. Filter self-labels by cross-seed-model agreement and by length sanity checks, and use an accelerated schedule once lower levels exceed 99%.

## Complexification operators from this area

Each operator lists what it does, a concrete easy → hard example, how to keep it verifiable, and its sources. Operators compose; e.g., solution-first generation + knob scaling + information hiding.

### 1. Task-mode inversion (forward → inverse → synthesis)
- **What it does**: Turns one verified (function, input, output) triplet into three tasks: predict the output (deduction), find an input that gives a target output (abduction), or write the function from partial I/O plus a hint (induction). The inverse and synthesis tasks are much harder and accept many correct answers.
- **Easy → hard**:
  - Easy: "What does `f(x) = sorted(set(x))[-2]` return for `x = [3, 1, 3, 2]`?" (answer 2).
  - Harder (abduction): "Give a list `x` of length 5 with `f(x) = 7`" (e.g., `[7, 9, 1, 1, 9]`).
  - Hardest (induction): "Here are 3 of 6 input/output pairs and the note 'second-largest distinct value'; write `f`." It is graded on the 3 hidden pairs.
- **Keep it verifiable**: Execute. Accept any abduction input that reproduces the output. Grade induction on held-out pairs. Require determinism (run twice), sandbox execution, and block dangerous modules.
- **Sources**: AZR [1]; GASP [15]; PopuLoRA [14].

### 2. Solution-first (planted) generation
- **What it does**: Create the answer-bearing artifact *first*, then derive the task from it. The generator never has to solve the hard direction.
- **Easy → hard**:
  - Integration: "∫ 2x dx" → sample a random expression tree F with d + 2 nodes and ask for ∫F′(x) dx.
  - Sudoku: a 4×4 grid → a 9×9 grid made by permuting a full valid grid and masking cells.
  - Text-to-SQL: write a 3-table JOIN/GROUP BY SQL query, execute it, then ask the model to write the natural-language question.
- **Keep it verifiable**: Check answers by the cheap direction: differentiate, check the constraints, or execute and compare result sets. Reject degenerate artifacts (empty SQL results, non-deterministic queries, NL that leaks the SQL).
- **Sources**: RLVE [43]; VHG [17]; SQL-Zero [25]; AZR [1].

### 3. Answer-first backward construction (multi-hop chaining)
- **What it does**: Fix an answer entity or value, then build the question by chaining evidence hops away from it through search, a knowledge graph or documents. More hops and less salient bridge facts make it harder.
- **Easy → hard**: "Who directed Film X?" → "In which city was the director of the film that won Award Y in the year City Z hosted the Olympics born?"
- **Keep it verifiable**:
  - Answerability: a separate no-tool reader, given all gathered evidence plus about 4 distractor documents, must recover the answer.
  - Uniqueness: ask the reader to list *all* valid answers, since SSP's hacked questions had non-unique answers.
  - Knowledge-graph paths give exact gold answers.
- **Sources**: SSP [20]; SPICE [18]; SPARK [62]; WIST [63].

### 4. Information hiding / asymmetry
- **What it does**: The proposer sees privileged context (a document, a database, repository history, a solution); the solver does not. Hiding more of what the answer depends on raises difficulty, and the solver must recall, infer, retrieve or use tools.
- **Easy → hard**: Reading comprehension with the passage supplied → the same question with the passage withheld. "Fix issue #123 (described in English)" → "Make these oracle tests pass" when the only specification is the reversed test-weakening patch.
- **Keep it verifiable**: The gold answer is *extracted* from the privileged context and must have a type (integer, expression, string, MCQ letter) checkable with Math-Verify or exact match. Reject answers that cannot be parsed. Mix MCQ, which is reliable, with free-form, which demands richer reasoning.
- **Sources**: SPICE [18]; SSR [21]; SSP [20]; SPARK [62].

### 5. Evidence / context accumulation
- **What it does**: Grow the amount of material the answer depends on: more documents, longer spans, cross-document joins. Also condition the proposer on already-solved questions and ask it to beat them.
- **Easy → hard**: A question answerable from one paragraph of an 8k-token document → a question that requires combining facts from 5 documents in a 100k-token context.
- **Keep it verifiable**: References still come from the documents. Use a semantic-equivalence verifier, e.g., one trained by self-consistency, rather than string match.
- **Sources**: SPELL [19].

### 6. Answer-preserving variation from the policy's own correct solutions
- **What it does**: For a problem the policy solves only 12.5–50% of the time, feed it one of its *correct* solutions and ask for structurally different problems with the *same* reference answer.
- **Easy → hard**: "How many positive divisors does 360 have?" (24) → "A rectangular garden with integer side lengths in metres has area 360 m². How many ordered (length, width) pairs are possible?" (also 24). The second requires recognising the divisor structure.
- **Keep it verifiable**: Inherit the verified answer. Reward only variants whose solve rate lies in [12.5%, 62.5%]: 100% solved suggests the answer leaked or the variant is trivial, and 0% suggests it is broken. Mix answer formats to avoid format overfitting.
- **Sources**: SvS [29]; TTCS [10].

### 7. Weakness-targeted concept recombination (text or symbolic)
- **What it does**: Find seeds the policy keeps failing (never above 50% accuracy, negative slope), extract their concepts, and compose co-occurring weak concepts into new problems. A symbolic variant edits a SymPy or SMT representation of the problem and re-renders it in words.
- **Easy → hard**: A single-concept modular-arithmetic problem → "How many integers in [1, 1000] are ≡ 2 (mod 3) and ≡ 3 (mod 5)?" This combines the Chinese Remainder Theorem with counting in an interval.
- **Keep it verifiable**:
  - Text route: a stronger model's majority answer with more than 50% consistency, plus the student producing it in at least 25% of samples; then a [25%, 75%] band under the current policy.
  - Symbolic route: compute ground truth from the symbolic form.
- **Sources**: SwS [30]; Adaptive Problem Generation via Symbolic Representations [66]; MathSmith [65]; Socratic-Zero [13].

### 8. Size / length / depth knobs with an adaptive controller
- **What it does**: A procedural generator takes an integer difficulty d. An online rule raises it once accuracy at the current top level exceeds a threshold, and a sliding window drops mastered levels.
- **Easy → hard**: Sort 5 numbers → sort about 3·1.1^d numbers; add two 10-digit numbers → add two 100-digit numbers; a 9-hop maze path → a 30-hop path.
- **Keep it verifiable**:
  - Use algorithmic verifiers with shaped partial credit, e.g., (x/N)^10.
  - Where verifiers break down at large sizes and you fall back on self-labels, filter by agreement across seed-varied models (≥ 4/5) and by length sanity checks to stop error avalanches.
  - Controller defaults from RLVE: τ_acc = 0.9, τ_num = 8 × rollouts, window d_Δ = 4.
- **Sources**: RLVE [43]; Self-Improving Transformers [53]; SPELL [19].

### 9. Structural-complexity shaping
- **What it does**: Define "harder" through properties of the formal object rather than its size or the solve rate: coupling between constraints, connectivity, variable coverage, non-redundant constraints. Update the target from the solver's current frontier.
- **Easy → hard**: An LP with 3 variables and 3 independent constraints → a MIP whose constraints share variables across resource, time and precedence blocks, so they cannot be satisfied one at a time.
- **Keep it verifiable**: A canonical solver must certify the problem feasible, bounded and solved to optimality. The proposer's own code must reproduce the optimum; otherwise its reward is capped. The solver sees only the natural-language description.
- **Sources**: OPT-Zero [26].

### 10. Fault / bug injection with higher-order composition
- **What it does**: Start from a working artifact (repo, program, plan) and break it by removing code hunks or files, or by reverting historical commits. Hide the break by weakening tests. Compose second-order faults from the solver's failed repair attempts.
- **Easy → hard**: A single off-by-one change (`var = 0 → var = 1`), which SSR found gives little signal → a deleted function body plus a reverted related commit, with the solver's failed partial fix layered on top.
- **Keep it verifiable**:
  - Tests pass on the original and fail on the injected version.
  - Test weakening is valid.
  - Inverse mutation testing: reverting each changed file alone must fix at least one test.
  - Reset hidden oracle tests before scoring.
  - Penalize inconsistent artifacts at −1.
  - Watch for challengers that write random-failing tests or obfuscate the code.
- **Sources**: SSR [21].

### 11. Adversarial context corruption (polluter vs. repairer)
- **What it does**: A polluter role corrupts the conditioning context with a locally coherent wrong step, a misleading partial solution or an input perturbation. The solver must detect the corruption and still reach the correct answer.
- **Easy → hard**: A clean algebra problem → the same problem with a plausible but wrong "work so far" step, e.g., a sign error in line 3.
- **Keep it verifiable**: Only the context changes, so the original outcome verifier applies. An imitation term on self-generated repairs helps early in training, when successful recoveries are rare.
- **Sources**: Guided Adversarial Self-Play for robust reasoning [61].

### 12. Iterative escalation with a co-updated solution and verifier (Code-as-Task)
- **What it does**: Propose a simple task together with a reference solution and a verification function in code. Then repeatedly add requirements, steps or tools, re-deriving the solution and verifier each time, and augment the toolset when it becomes insufficient. Enumerated failure cases act as negative tests.
- **Easy → hard**: "Find a flight from A to B." → "Find the cheapest A→B itinerary on flexible dates, with a hotel within 1 km of the venue, under a total budget, excluding carrier X, using only the provided tool APIs."
- **Keep it verifiable**: The reference must pass the verifier and every failure case must fail it. Restrict the reference to the tool interface so it cannot read the database directly. Keep tasks with pass@k > 0 under the policy. Check that instructions are complete.
- **Sources**: DeepSeek-V3.2 [24]; SCA [23]; Tool-R0 [9]; Skill Self-Play [64].

### 13. Tool-dependence escalation
- **What it does**: Add a capped proposer reward for tasks whose successful solutions *require* tool calls, so difficulty comes from computation or retrieval rather than recall.
- **Easy → hard**: "What is 17 × 23?" → "What is the sum of all primes below 2,000,000?", which in practice needs code execution.
- **Keep it verifiable**: Use a code interpreter to get exact intermediate values. Where there is no external label, use majority vote over tool-using rollouts. Filter out tasks solved without tools.
- **Sources**: Agent0 [8]; Tool-R0 [9].

### 14. Opponent strengthening (game self-play)
- **What it does**: Recast the task as a multi-turn zero-sum or strategic game against copies of the policy. Difficulty rises as the opponent improves.
- **Easy → hard**: TicTacToe against a random player → Kuhn Poker or Simple Negotiation against the latest self, mixing several games.
- **Keep it verifiable**: The game engine decides outcomes. Use per-game, per-role baselines (RAE) to avoid "thinking collapse". For non-verifiable variants that use a reward model, add a quality self-reward.
- **Sources**: SPIRAL [27]; Vision-Zero [60]; Language Self-Play [59].

### 15. Guidance withdrawal (hint / prefix annealing, reverse curriculum)
- **What it does**: Make a p = 0 problem learnable by revealing part of a verified solution, or a ladder of hints from abstract to concrete. Then shrink the reveal, or move the start state earlier, until the model solves it unaided. The fraction withheld is a continuous difficulty knob.
- **Easy → hard**: Problem + first 50% of the reference solution → problem + 25% → problem alone.
- **Keep it verifiable**: The final answer is unchanged, and hints come from verified solutions. Anneal:
  - QuestA: 50% → 25%.
  - Backward Hint Annealing: per-question hint dropout.
  - SEELE: pick the hint length with an IRT fit targeting about 50%.
  - KnowRL: use minimal-sufficient knowledge points rather than long prefixes.
- **Sources**: QuestA [48]; SEELE [49]; Scaf-GRPO [50]; MFC [51]; R3 [52]; DAHS/BHA [69]; KnowRL [70]; boundary-aware Curriculum RL [57].

### 16. Goal-anchored stepping stones (lemma → lift; teacher rewarded by real progress)
- **What it does**: For real problems at pass@k = 0, generate an *easier* variant in a middle band, then a *harder* variant of that variant, without showing the goal again. Repeat until the gap closes. The meta-RL alternative rewards the teacher directly by the student's improvement on the hard set.
- **Easy → hard**: Lemma, a simplified version of an unsolved competitive-programming task (p ∈ [0.3, 0.7]) → lift, a harder variant of the lemma (p ∈ [0.1, 0.5]) → the goalpost.
- **Keep it verifiable**: Validate by execution (valid, safe, deterministic). Deduplicate at cosine similarity ≥ 0.95. For bridge data, well-posedness matters more than a correct answer: in SOAR, only 32.8% of helpful questions had fully correct solutions and 63% were well-posed. Final-target data still needs verified answers.
- **Sources**: GASP [15]; SOAR [16]; implicit-curriculum theory [55].

### 17. Decomposition into verifiable subproblems / recomposition into chains
- **What it does**: Derive intermediate subproblems from a reference solution chain, keeping the original problem as the final subproblem, so partial progress earns reward. In the other direction, compose short solved blocks into long sequential computations.
- **Easy → hard**: "Compute the intermediate quantity from step 2" → the full multi-step original. Or state tracking for B = 8 steps → T = 256 steps, composed from blocks.
- **Keep it verifiable**: Intermediate answers are read off the verified chain. Normalize rewards per subproblem position (SCRL). Compute ground truth for composed chains by running the component steps.
- **Sources**: SCRL [67]; Learning to Reason with Curriculum II [68]; E2H [46].

### 18. Verifier / test strengthening
- **What it does**: Make "solved" tasks hard again by tightening the check: add adversarial edge-case tests learned from the coder's typical mistakes, or add stricter output constraints.
- **Easy → hard**: A coding task checked by 3 happy-path tests → the same task checked by generated tests for empty input, overflow and duplicate keys, where earlier solutions failed.
- **Keep it verifiable**: Generated tests must agree with ground-truth tests or reference solutions. Reward tests for separating correct from incorrect code, and penalize tests that fail correct code.
- **Sources**: CURE [22]; SQLM [4].

### 19. Novelty-constrained mutation at the skill level
- **What it does**: Reward or filter proposals by their distance to a *persistent* archive, measured on solution signatures (canonical solver code, masked SQL templates, concept tags) rather than wording. Use populations with cross-evaluation to stop drift toward easy problems.
- **Easy → hard**: Fifty near-duplicate "solve a two-step linear equation" variants → tasks whose canonical solver programs differ from everything in the archive.
- **Keep it verifiable**: Diversity terms say nothing about correctness, so always multiply them by a validity gate. Novelty alone favours odd or invalid tasks.
- **Sources**: R-Diverse [6]; OpenSIR [5]; SQL-Zero [25]; R-Zero [3]; PopuLoRA [14].

### 20. Frontier-prioritized mutation (regret or learnability-weighted edits)
- **What it does**: Choose *which* seeds to complexify by their current learning value: p(1−p), |advantage|, or regret = best-response reward minus baseline. Apply small edits or rewrites only to those.
- **Easy → hard**: Rewriting all 10k instructions uniformly with "add constraints" → rewriting only the prompts with the largest best-vs-baseline reward gap, then re-scoring them.
- **Keep it verifiable**: Inherit each family's verifier. For reward-model-scored prompts, watch for exploitation.
- **Sources**: ACCEL and the UED family [33, 31, 32, 34, 35]; eva [28]; LILO [36].

## Insights & pitfalls

- **Stack the tools in order: select, then generate, then scaffold.**
  - Layer 0 (selection): compute p over about 8 rollouts per seed. Train on p(1−p) or a balanced band (LILO; Bae et al.; DAPO dynamic sampling). Kimi k1.5's prioritized sampling ∝ 1 − success rate is a cheaper variant.
  - Layer 1 (seeds at p = 1): complexification operators (knobs, inversion, information hiding, SvS-style variants, bug injection).
  - Layer 2 (seeds at p = 0): scaffolds that are later withdrawn (QuestA, Scaf-GRPO, MFC), or goal-anchored stepping stones (GASP, SOAR).
  - Put cheap difficulty predictors (PCL, GRESO, MoPPS, DOTS) on *generated* candidates before paying for full rollouts.
- **Validity is the main constraint on how hard you can go.**
  - Naive setter–solver loops are hacked by invalid problems (VHG).
  - OpenSIR shows the cost of pushing the solve-rate floor down: at 0.5 → 0.1, validity fell 70.8% → 42.3% and math accuracy fell 29.6 → 26.0, while problems got only slightly harder.
  - SSP's RAG check still let non-unique questions through.
  - SSR's analysis shows challengers can always "win" by writing randomly failing tests or by obfuscating code, unless their action space is constrained and checked.
  - Always compute difficulty *conditional on* an independent validity signal.
- **Expect low and variable yield, and budget for it.**
  - SCA: 47.7% of proposals pass a "verifier runs" check, 9.5% pass "reference passes the verifier", 5.2% pass the full Code-as-Task filter.
  - VHG accepted 4,076 of 18,663 integral candidates and 20,670 of 46,080 judged general-math pairs.
  - About 35% of SwS problems land in its [25%, 75%] band.
  - DeepSeek-V3.2 keeps only tasks with pass@100 > 0.
- **Grounding beats introspection.**
  - R-Zero-style pseudo-label loops peak after 1 iteration (0.6B) to 3 iterations (4B). Label accuracy decays 79% → 69% → 63%, and noise tolerance depends on scale: decline starts at 70.6% accuracy for 0.6B vs 48.8% for 4B.
  - Grounded variants keep improving:
    - SPICE: 43.9 vs 40.7 ungrounded.
    - WIST: open-web grounding gave +9.8 overall on Qwen3-4B-Base and +14.79 in medicine for Qwen3-8B-Base.
    - SPARK: knowledge-graph grounding widens its lead as hop count grows.
    - SQL-Zero and OPT-Zero: databases and OR solvers.
    - SSR: real repositories.
  - The ICML 2026 position paper argues self-play plateaus when the loop adds data without adding *learnable information*. It proposes three levers: asymmetric proposer–solver–verifier co-evolution, capacity or inference-budget growth, and active external information seeking.
- **Self-play usually sharpens existing skills; variant synthesis and scaffolds expand what the model can solve.**
  - AZR self-play helps pass@k at small k, but the base model is better at large k (not statistically significant), and entropy still collapses (Chae et al.).
  - Methods that expanded pass@k at large k or kept entropy up:
    - SvS: +18.3 and +22.8 pass@32 on AIME24 and AIME25.
    - Boundary-aware curriculum RL: +9.8 pass@256 over the base and +10.3 over vanilla RLVR.
    - MFC: widened pass@k at large k.
    - SCRL: +4.6 pass@64 on Qwen3-4B-Base.
    - DAHS/BHA: pass@2048 gains.
  - Evaluate generated-task training with pass@k curves, not pass@1 alone.
- **The exact learnability reward is a second-order knob; co-evolution and grounding are first-order.**
  - SSR: the ±1 consistency-only reward was nearly as good as the solve-rate-shaped one.
  - Socratic-Zero: reward-shape variants were within about 0.4 points.
  - Chae et al.: R-Zero's 50%-target reward *hurt* AZR by 2%.
  - SPICE: reward choice mattered somewhat (variance 44.9, R-Zero reward 43.6, threshold 41.4, AZR reward 40.7).
  - OPT-Zero: a structural reward beat a solve-rate reward.
  - Co-evolution is essential:
    - SSP with a fixed proposer: solver reward saturates near 0.9.
    - SPIRAL with fixed opponents: training fails.
    - SOAR: grounded teachers were more stable than intrinsic-reward teachers.
- **Negative rewards for malformed proposals can kill the proposer.** In SSP, a −0.1 penalty for invalid questions produced a "death spiral": entropy rose, the valid-question rate fell to 0, and solver reward rose misleadingly. SPICE uses ρ = −0.1 without that problem. So monitor the valid-proposal rate, and prefer zero reward (or a clipped reward) for format failures if it drops.
- **"Harder" produced by naive prompting is mostly shallow.**
  - SSR's direct injection produced one-line bugs, while code removal plus history reversion worked best.
  - GASP's teachers raise difficulty by "adding extra constraints, which is not always the most informative form of complexity", and copy surface metaphors from the goal problem.
  - Better-behaved axes: structural complexity (OPT-Zero), required tool use (Agent0), evidence accumulation (SPELL), inversion (AZR), deletion or reconstruction (SSR).
- **Screen hard variants for guessability.** Kimi k1.5 drops multiple-choice, true/false and proof questions from RL. It also removes prompts that a model answers correctly *without CoT* within N = 8 guesses, since such prompts give false-positive rewards. Apply the same screen to generated "hard" tasks. Note the tension: SPICE found mixing MCQ into its tasks helped, because MCQ is reliable to verify.
- **For bridge data, well-posedness matters more than answer correctness; for target data, it does not.** SOAR's useful stepping stones were well-posed 63% of the time but fully correct only 32.8%, and adding well-posed-but-wrong questions still helped students stuck on a plateau. This does not license unverified answers for final RL targets. Use it to lower the bar only for bridge or curriculum data.
- **Diversity collapses without anyone noticing.**
  - Within-batch penalties allow cycling across iterations. Different wording can hide identical skills (R-Diverse).
  - Single agents self-calibrate toward easy problems (PopuLoRA).
  - Self-play problems cluster around familiar concepts (OpenSIR); its diversity reward roughly doubled concept coverage.
  - Use persistent archives, similarity over solution signatures, template-level deduplication (SQL-Zero), and multiple proposers.
- **Separate weights or one shared model: evidence is mixed.** In R-Zero, separate Challenger and Solver models gave better pseudo-labels than a shared model (71.0% vs 63.4% at step 15). SPICE, AZR, SSP and SSR succeed with a shared policy when there is external grounding. Where labels come from self-consistency, keep the roles separate.
- **Environment breadth beats instances.** RLVE: +3.37 with 400 environments vs +0.49 from continued RL with more than 3× compute. Held-out gains grow as environments are added. RLVE also notes that building DeepMath-103K cost about $138K and 127K GPU hours, while procedural environments scale at roughly zero marginal cost per instance.
- **Ordering tricks are weak; spectrum smoothness is strong.**
  - Mordig et al.: no robust benefit of easy→hard ordering over random for SFT or RL on deductive tasks.
  - Huang et al. (ICML 2026): mixed-difficulty RLVR self-organizes into an implicit curriculum. A smooth spectrum gives a "relay" regime; gaps cause grokking-like plateaus.
  - E2H: fading out easy tasks is essential.
  - Invest in generating intermediate tasks (lemma/lift, subproblems, hint levels) that fill gaps in your difficulty distribution.
- **Self-labels can extend difficulty only with strong filtering.** In Lee et al., majority voting across 5 seed-varied models (≥ 4/5 agreement) raised 5×6 multiplication label accuracy from 31% to 93.3%. Without filtering, 6×6 accuracy reached only 13.7% after 7 rounds, because errors avalanche. An accelerated schedule cut 5×5 → 10×10 from 41 rounds to 19.
- **Non-verifiable self-play needs quality anchors.** Language Self-Play without a quality self-reward "degenerate[s] into adversarial nonsense", and the solver reward-hacked a DeBERTa reward model by answering most queries in Python. eva's regret-prioritized prompt evolution works (Arena-Hard 52.6% → 62.4% with RLOO) but inherits reward-model exploitability.
- **Test-time curricula are a cheap special case.** When the target distribution is known, conditioning variant synthesis (TTCS) or retrieval (TTC-RL) on the targets gives large gains. TTC-RL raised Qwen3-8B pass@8 on AIME25 from 40% to 62%. Keep evaluation items out of the generator's conditioning set to avoid contamination.
- **Safety and oversight.**
  - AZR saw safety-concerning CoT (the "uh-oh moment") from Llama-3.1-8B.
  - SSR identifies degenerate injector strategies.
  - Agent0-style autonomous code execution needs sandboxing.
  - Log proposer outputs, audit samples of generated tasks, and run detectors for test tampering and obfuscation.

## Open problems & research opportunities

- **Cheap, independent validity certification outside executable domains.** Proofs, open-ended QA, and long-horizon web or agent tasks still depend on LLM judges (VHG's soft verifier accepts 44.9%) or answerability heuristics that miss non-unique answers (SSP). A calibrated "valid / ambiguous / unsolvable" classifier that is separate from the difficulty signal is missing.
- **Sustaining improvement beyond a few iterations.** There is no standard per-iteration metric of learnable information gain, no plateau detector, and no recipe for *when* to inject external context or grow capacity (position paper [56]).
- **Skill-level coverage metrics.** R-Diverse's solver-code embeddings and OpenSIR's concept coverage are early steps. A shared, cheap skill-taxonomy metric for generated pools would make diversity illusion measurable across papers.
- **Goal-directed generation at scale.** GASP's fixed goalposts and SOAR's expensive inner-loop RL leave open how to amortize "teacher rewarded by real progress". Examples: learned predictors of the student's improvement, reusing probe fine-tunes, or refreshing goalposts once solved. Both also need guards against benchmark contamination.
- **Difficulty estimation that transfers across checkpoints.** p̂ from small groups is noisy and specific to one policy. IRT across items and checkpoints (SEELE uses IRT per instance), value-model predictors (PCL), Bayesian bandits (MoPPS) and multi-solver disagreement have not been compared on cost versus accuracy.
- **Answer-preserving transformations with guarantees.** SvS variants can leak answers or overfit to answer format. Symbolic intermediate representations (SymPy/SMT) give guarantees but only cover narrow domains. Methods that *recompute* answers after an arbitrary natural-language rewrite are an open need.
- **Structural difficulty definitions per domain.** OPT-Zero's coupling and connectivity metrics beat solve-rate rewards for optimization. Analogous structural measures for code (dependency depth, cross-module coupling), math (proof-graph depth) and agents (constraint interaction) are largely unexplored.
- **Optimal hint withdrawal.** Open questions: the schedule (QuestA's fixed switch, SEELE's IRT, BHA's per-question dropout, MFC's monotone frontier); choosing the minimal sufficient hint (KnowRL, Scaf-GRPO); and removing the off-policy cost of hinted trajectories.
- **Agentic environment synthesis.** Catching bugs in synthesized tools and verifiers, preventing verifier and test weakening from being exploited (SSR), and verifying long-horizon, multi-tool tasks whose instructions may be incomplete (SCA's residual false negatives).
- **Theory that reconciles curriculum results.** "Curriculum helps" (E2H, Curriculum II, SCRL, MFC) vs "ordering is negligible" (Mordig et al.) vs "implicit curriculum depends on smoothness" (Huang et al.). Practical diagnostics are also needed for spectrum gaps in a real task pool.
- **SFT vs RL use of generated hard tasks.** Few compute-matched comparisons exist between distilling verified traces on generated frontier tasks (SCA distillation +20.2; Socratic-Zero's SFT generator) and direct RL on them.
- **Compute efficiency.** The cost is proposer rollouts × solver rollouts × verification (and inner-loop training for SOAR). Ways to cut it include sharing rollouts between roles (SPICE reuses the G = 8 Reasoner samples for the Challenger reward), predictive filtering of candidates, and populations on LoRA (PopuLoRA).
- **Safety of autonomous curricula.** Monitoring proposers for harmful or deceptive content and for degenerate strategies that pass validity checks, such as test tampering, obfuscation, or tunnel-vision on a single parameter.

## References

1. Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Yue, Y., Lin, M., Wang, S., Wu, Q., Zheng, Z., Huang, G. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335. https://arxiv.org/abs/2505.03335
2. Chae, J. Y., Alam, M. T., Rastogi, N. (2025). *Towards Understanding Self-play for LLM Reasoning*. arXiv:2510.27072. https://arxiv.org/abs/2510.27072
3. Huang, C., Yu, W., Wang, X., Zhang, H., Li, Z., Li, R., Huang, J., Mi, H., Yu, D. (2025). *R-Zero: Self-Evolving Reasoning LLM from Zero Data*. arXiv:2508.05004. https://arxiv.org/abs/2508.05004
4. Chen, L., Prabhudesai, M., Fragkiadaki, K., Liu, H., Pathak, D. (2025). *Self-Questioning Language Models*. arXiv:2508.03682. https://arxiv.org/abs/2508.03682
5. Kwan, W.-C., Leang, J. O. J., Vougiouklis, P., Pan, J. Z., Valentino, M., Minervini, P. (2025). *OpenSIR: Open-Ended Self-Improving Reasoner*. arXiv:2511.00602. https://arxiv.org/abs/2511.00602
6. Li, G., He, J., Wang, S., et al. (2026). *R-Diverse: Mitigating Diversity Illusion in Self-Play LLM Training*. arXiv:2602.13103. https://arxiv.org/abs/2602.13103
7. Selvendran, V., Zhang, Z. (2026). *Beyond Uncertainty: Multi-Solver Disagreement Rewards for Self-Evolving Reasoning Curricula*. CIKM '26; arXiv:2608.30035. https://arxiv.org/abs/2608.30035
8. Xia, P., Zeng, K., Liu, J., Qin, C., Wu, F., Zhou, Y., Xiong, C., Yao, H. (2025). *Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning*. COLM 2026; arXiv:2511.16043. https://arxiv.org/abs/2511.16043
9. Acikgoz, E. C., Qian, C., Hübotter, J., Ji, H., Hakkani-Tür, D., Tur, G. (2026). *Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data*. arXiv:2602.21320. https://arxiv.org/abs/2602.21320
10. Yang, C., Xiang, Z., Tang, Y., et al. (2026). *TTCS: Test-Time Curriculum Synthesis for Self-Evolving*. arXiv:2601.22628. https://arxiv.org/abs/2601.22628
11. Hübotter, J., Diaz-Bone, L., Hakimi, I., Krause, A., et al. (2025). *Learning on the Job: Test-Time Curricula for Targeted Reinforcement Learning*. arXiv:2510.04786. https://arxiv.org/abs/2510.04786
12. Zuo, Y., Zhang, K., Sheng, L., Qu, S., et al. (2025). *TTRL: Test-Time Reinforcement Learning*. arXiv:2504.16084. https://arxiv.org/abs/2504.16084
13. Wang, S., Jiao, Z., Zhang, Z., et al. (2025). *Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution*. arXiv:2509.24726. https://arxiv.org/abs/2509.24726
14. Creus Castanyer, R., Bradway, G., Wolf, L., Lin, M., Mavor-Parker, A. N., Sargent, M. J. (2026). *PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play*. arXiv:2605.16727. https://arxiv.org/abs/2605.16727
15. Jana, S., Sancaktar, C., Daniš, T., Martius, G., Orvieto, A., Kolev, P. (2026). *GASP: Guided Asymmetric Self-Play For Coding LLMs*. ICLR 2026 Workshops (RSI, LLA); arXiv:2603.15957. https://arxiv.org/abs/2603.15957
16. Sundaram, S., Quan, J., Kwiatkowski, A., Ahuja, K., Ollivier, Y., Kempe, J. (2026). *Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability* (SOAR). arXiv:2601.18778. https://arxiv.org/abs/2601.18778
17. Lai, Y., Feng, J., Teh, Y. W., Miao, N. (2026). *Verifier-Backed Hard Problem Generation for Mathematical Reasoning* (VHG). arXiv:2605.06660. https://arxiv.org/abs/2605.06660
18. Liu, B., Jin, C., Kim, S., Yuan, W., Zhao, W., Kulikov, I., Li, X., Sukhbaatar, S., Lanchantin, J., Weston, J. (2025). *SPICE: Self-Play In Corpus Environments Improves Reasoning*. arXiv:2510.24684. https://arxiv.org/abs/2510.24684
19. Yang, Z., Shen, W., Li, C., Chen, R., Wan, F., Yan, M., Quan, X., Huang, F. (2025). *SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models*. ICLR 2026; arXiv:2509.23863. https://arxiv.org/abs/2509.23863
20. Lu, H., Wen, Y., Cheng, P., Ding, R., et al. (2025). *Search Self-play: Pushing the Frontier of Agent Capability without Supervision*. ICLR 2026; arXiv:2510.18821. https://arxiv.org/abs/2510.18821
21. Wei, Y., Sun, Z., McMilin, E., Gehring, J., Zhang, D., Synnaeve, G., Fried, D., Zhang, L., Wang, S. (2025). *Toward Training Superintelligent Software Agents through Self-Play SWE-RL*. ICML 2026; arXiv:2512.18552. https://arxiv.org/abs/2512.18552
22. Wang, Y., Yang, L., Tian, Y., Shen, K., Wang, M. (2025). *Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning* (CURE). NeurIPS 2025; arXiv:2506.03136. https://arxiv.org/abs/2506.03136
23. Zhou, Y., Levine, S., Weston, J., Li, X., Sukhbaatar, S. (2025). *Self-Challenging Language Model Agents*. arXiv:2506.01716. https://arxiv.org/abs/2506.01716
24. DeepSeek-AI (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*. arXiv:2512.02556. https://arxiv.org/abs/2512.02556
25. Pedrozo, D. M., Dollis, J. S., de Oliveira, B. L. M., Aguiar, V. A., de Oliveira, S. S. T., Soares, T. W. L. (2026). *SQL-Zero: Self-Evolving Text-to-SQL*. arXiv:2609.04697. https://arxiv.org/abs/2609.04697
26. Jiang, X., Wu, Y., Zhou, C., Xu, M., Nuijten, W. P. M., Zhang, Y. (2026). *Learning to Optimize through Solver-Grounded Self-Play* (OPT-Zero). arXiv:2609.34205. https://arxiv.org/abs/2609.34205
27. Liu, B., Guertler, L., Yu, S., Liu, Z., et al. (2025). *SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning*. ICLR 2026; arXiv:2506.24119. https://arxiv.org/abs/2506.24119
28. Ye, Z., Agarwal, R., Liu, T., Joshi, R., Velury, S., Le, Q. V., Tan, Q., Liu, Y. (2024). *Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play* (eva). NeurIPS 2024 Language Gamification Workshop; arXiv:2411.00062. https://arxiv.org/abs/2411.00062
29. Liang, X., Li, Z., Gong, Y., Shen, Y., Wu, Y. N., Guo, Z., Chen, W. (2025). *Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR* (SvS). arXiv:2508.14029. https://arxiv.org/abs/2508.14029
30. Liang, X., Li, Z.-Z., Gong, Y., Wang, Y., Zhang, H., Shen, Y., Wu, Y. N., Chen, W. (2025). *SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning*. arXiv:2506.08989. https://arxiv.org/abs/2506.08989
31. Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., et al. (2020). *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design* (PAIRED). NeurIPS 2020; arXiv:2012.02096. https://arxiv.org/abs/2012.02096
32. Jiang, M., Grefenstette, E., Rocktäschel, T. (2020). *Prioritized Level Replay*. arXiv:2010.03934. https://arxiv.org/abs/2010.03934
33. Parker-Holder, J., Jiang, M., Dennis, M., Samvelyan, M., et al. (2022). *Evolving Curricula with Regret-Based Environment Design* (ACCEL). arXiv:2203.01302. https://arxiv.org/abs/2203.01302
34. Wang, R., Lehman, J., Clune, J., Stanley, K. O. (2019). *Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions*. arXiv:1901.01753. https://arxiv.org/abs/1901.01753
35. Rutherford, A., Beukman, M., Willi, T., Lacerda, B., et al. (2024). *No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery* (SFL). arXiv:2408.15099. https://arxiv.org/abs/2408.15099
36. Foster, T., Sims, A., Forkel, J., Fellows, M., Foerster, J. (2025). *LILO: Learning to Reason at the Frontier of Learnability*. arXiv:2502.12272. https://arxiv.org/abs/2502.12272
37. Bae, S., Hong, J., Lee, M. Y., Kim, H., Nam, J., Kwak, D. (2025). *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning*. EACL 2026; arXiv:2504.03380. https://arxiv.org/abs/2504.03380
38. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476. https://arxiv.org/abs/2503.14476
39. Zheng, H., Zhou, Y., Bartoldson, B. R., Kailkhura, B., et al. (2025). *Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts* (GRESO). arXiv:2506.02177. https://arxiv.org/abs/2506.02177
40. Gao, Z., Kim, J., Sun, W., Joachims, T., et al. (2025). *Prompt Curriculum Learning for Efficient LLM Post-Training* (PCL). arXiv:2510.01135. https://arxiv.org/abs/2510.01135
41. Qu, Y., Wang, Q., Mao, Y., Hu, V. T., et al. (2025). *Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models?* (MoPPS). arXiv:2507.04632. https://arxiv.org/abs/2507.04632
42. Sun, Y., Shen, J., Wang, Y., Chen, T., et al. (2025). *Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay* (DOTS). NeurIPS 2025; arXiv:2506.05316. https://arxiv.org/abs/2506.05316
43. Zeng, Z., Ivison, H., Wang, Y., Yuan, L., et al. (2025). *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*. ICML 2026; arXiv:2511.07317. https://arxiv.org/abs/2511.07317
44. Chen, X., Lu, J., Kim, M., Zhang, D., Tang, J., Piché, A., Gontier, N., Bengio, Y., Kamalloo, E. (2025). *Self-Evolving Curriculum for LLM Reasoning* (SEC). arXiv:2505.14970. https://arxiv.org/abs/2505.14970
45. Shi, T., Wu, Y., Song, L., Zhou, T., et al. (2025). *Efficient Reinforcement Finetuning via Adaptive Curriculum Learning* (AdaRFT). TMLR; arXiv:2504.05520. https://arxiv.org/abs/2504.05520
46. Parashar, S., Gui, S., Li, X., Ling, H., et al. (2025). *Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning* (E2H Reasoner). arXiv:2506.06632. https://arxiv.org/abs/2506.06632
47. Wang, Z., Cui, G., Li, Y.-J., Wan, K., et al. (2025). *DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training*. arXiv:2504.09710. https://arxiv.org/abs/2504.09710
48. Li, J., Lin, H., Lu, H., Wen, K., Yang, Z., Gao, J., Wu, Y., Zhang, J. (2025). *QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation*. ICLR 2026; arXiv:2507.13266. https://arxiv.org/abs/2507.13266
49. Li, Z., Sun, Z., Zhao, J., Min, E., et al. (2025). *Staying in the Sweet Spot: Responsive Reasoning Evolution via Capability-Adaptive Hint Scaffolding* (SEELE). arXiv:2509.06923. https://arxiv.org/abs/2509.06923
50. Zhang, X., Wu, S., Zhu, Y., Tan, H., Yu, S., He, Z., Jia, J. (2025). *Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning*. ICLR 2026; arXiv:2510.19807. https://arxiv.org/abs/2510.19807
51. Zhu, Y., Han, Z. (2026). *Unlocking the Unsolvable: Teacher-Guided Curriculum for Data-Efficient RLVR* (MFC). Findings of EMNLP 2026; arXiv:2609.13997. https://arxiv.org/abs/2609.13997
52. Xi, Z., Chen, W., Hong, B., Jin, S., et al. (2024). *Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning* (R3). arXiv:2402.05808. https://arxiv.org/abs/2402.05808
53. Lee, N., Cai, Z., Schwarzschild, A., Lee, K., Papailiopoulos, D. (2025). *Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges*. arXiv:2502.01612. https://arxiv.org/abs/2502.01612
54. Mordig, M., Opedal, A., Liu, W., Schölkopf, B. (2026). *Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning*. arXiv:2603.27226. https://arxiv.org/abs/2603.27226
55. Huang, Y., Wen, Z., Chi, Y., Wei, Y., Singh, A., Liang, Y., Chen, Y. (2026). *On the Emergence of Implicit Curriculum in RLVR Learning Dynamics*. ICML 2026; arXiv:2602.14872. https://arxiv.org/abs/2602.14872
56. Liu, W., Qi, S., Du, Y., He, Y. (2026). *Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain*. ICML 2026 (position track); arXiv:2603.02218. https://arxiv.org/abs/2603.02218
57. Cai, P., Fang, T., Li, X., Zeng, Q., Li, G., Chen, J. (2026). *Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model*. arXiv:2606.22317. https://arxiv.org/abs/2606.22317
58. Kimi Team (2025). *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. arXiv:2501.12599. https://arxiv.org/abs/2501.12599
59. Kuba, J. G., Gu, M., Ma, Q., Tian, Y., et al. (2025). *Language Self-Play For Data-Free Training*. arXiv:2509.07414. https://arxiv.org/abs/2509.07414
60. Wang, Q., Liu, B., Zhou, T., Shi, J., et al. (2025). *Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play*. ICLR 2026; arXiv:2509.25541. https://arxiv.org/abs/2509.25541
61. Li, S., Tadiparthi, V., Lee, K., Agarwal, N., et al. (2026). *Learning Robust Reasoning through Guided Adversarial Self-Play*. arXiv:2602.00173. https://arxiv.org/abs/2602.00173
62. Park, H., Kim, T., Choi, D.-G. (2026). *SPARK: Self-Play with Asymmetric Reward from Knowledge Graphs*. arXiv:2605.05546. https://arxiv.org/abs/2605.05546
63. Li, F., Li, P., Wang, S., Gao, J., et al. (2026). *WIST: Web-Grounded Iterative Self-Play Tree for Domain-Targeted Reasoning Improvement*. arXiv:2603.22352. https://arxiv.org/abs/2603.22352
64. Huang, S., Cheng, P., Liu, H., Chen, T., et al. (2026). *Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills*. arXiv:2607.22529. https://arxiv.org/abs/2607.22529
65. Zhan, S., Lai, Y., Lu, Z., Lin, D., et al. (2025). *MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy*. AAAI 2026; arXiv:2508.05592. https://arxiv.org/abs/2508.05592
66. Yeo, T., Jeon, M., Weerakoon, D., Qiao, R., et al. (2026). *Adaptive Problem Generation via Symbolic Representations*. arXiv:2602.19187. https://arxiv.org/abs/2602.19187
67. Jiang, X., Tang, Z., Lin, W., Yue, Y., et al. (2026). *From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning* (SCRL). arXiv:2605.22074. https://arxiv.org/abs/2605.22074
68. Rajaraman, N., Huang, A., Dudik, M., Schapire, R., et al. (2026). *Learning to Reason with Curriculum II: Compositional Generalization*. arXiv:2606.27721. https://arxiv.org/abs/2606.27721
69. Xie, P.-X., Lin, C.-Y., Yang, C.-L. (2026). *Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing*. arXiv:2604.07747. https://arxiv.org/abs/2604.07747
70. Yu, L., Yang, T., Ding, S., Jin, R., et al. (2026). *KnowRL: Boosting LLM Reasoning via Reinforcement Learning with Minimal-Sufficient Knowledge Guidance*. arXiv:2604.12627. https://arxiv.org/abs/2604.12627
