# Composable verifiable constraints for instruction following (SFT and RLVR)

*Scope: turning easy instruction-following (IF) seeds into harder prompts by composing constraints that can be checked, for SFT data and for RLVR (GRPO-family, DPO, on-policy distillation). Covers constraint stacking, structured composition (chain, branch, nesting, scope), back-translation, learned and self-play complexifiers, program-derived and implicit-reasoning instructions, rubric and checklist rewards for soft constraints, reward aggregation, and difficulty control. Compiled 2026-09-30. Verification: 28 entries checked against primary sources (arXiv full text of every entry), plus about 20 supporting works cited in the operator and insight sections. 15 entries corrected, 0 dropped, 6 added.*

---

## TL;DR

- **Stacking constraints gives a predictable difficulty dial, but all-pass reward collapses quickly.**
  - CSE (15 models, 36 deterministic constraint types) fits mean per-constraint success at 72.0% × 0.922^(k−1). About 41% per-constraint success at k=8 means only 5.7% of responses satisfy all eight.
  - All-pass success falls below 50% at 7 constraints for the best model, and at 3 or fewer for 12 of 15 models.
  - On RECAST-Test Level 4 (at least 15 constraints), the all-constraint satisfaction rate is at most 7% for every model tested, including Qwen3-235B.
  - Choose k so that p^k lands in your learnable band. At high k, use per-constraint, graded or structure-aware rewards instead of binary all-pass.
- **Calibrate on the policy's pass rate, not on constraint count.**
  - IFDecorator measures 8 rollouts. It keeps prompts with pass rate in (0, 0.5], evolves prompts above 0.5 by adding up to 3n verifiable constraints at iteration n, and regenerates prompts at 0.
  - From 21k seeds over 5 iterations, it produced 7,324 usable prompts and 10,772 unsolvable ones (pass rate 0). **Over-hardening is the normal failure mode.**
  - It also reports low-complexity prompts that are hard and high-complexity prompts that stay easy.
  - QUBRIC keeps a 20–50% pass-rate corridor.
- **Harder prompts alone do not help; policy-aware hardening does.**
  - In LLM-as-a-Tutor, offline Evol-Instruct rewriting (average 50.24) was slightly *worse* than training on the unmodified prompts (50.51).
  - Appending one atomic constraint only when the tutor judges two rollouts indistinguishable reached 51.96.
  - In SEIF (a self-play instructor that adds constraints), removing the conflict filter cost −3.2 IFEval and −6.0 CFBench.
- **Train with more, and more varied, constraints than the test set uses.**
  - IFBench (Qwen2.5 policy): 1 constraint per prompt gave 48.9 on IFBench; up to 3 gave 59.5. With the TÜLU-DPO policy, up to 5–6 beat up to 3, although IFEval prompts have at most 3 constraints and IFBench at most 2.
  - Training variable ranges disjoint from test ranges hurt; wider ranges did as well as or better than the same range.
  - TÜLU-3-8B-DPO scores 81.1 on IFEval but 25.5 on IFBench, so models overfit the 25 IFEval constraint types.
- **Reward precision is the binding constraint on soft constraints.**
  - Code checkers reach 96.0% precision. LLM judges miss violations: in multi-constraint judging, Qwen3-32B caught only 30.6% of hard-constraint and 20.9% of soft-constraint failures. Judging each constraint separately ("pointwise") raised this to 59.3% and 54.7%.
  - Hard-only data (22.3% of VerInstruct) matched mixed-data RL on average and beat soft-only. Pilot-run learnability filtering plus at most one soft constraint per prompt gave +13.4 average (Precision over Diversity).
  - But with a strong *reasoning* judge, soft constraints add signal: VerIF scored 84.5 IFEval with code+QwQ-32B, vs 74.7 code-only and 76.9 with a non-reasoning 72B judge.
- **Match reward aggregation to the composition structure.**
  - Averaging per constraint: RECAST RLVC, Ren et al.
  - Structure-aware aggregation (LsrIF): average parallel constraints, decay credit for later sequential steps after an early failure, and score only the active conditional branch. Averaging instead cost 3.9 IFEval and 5 CFBench points.
  - Graded continuous violation scores: ScopeIF.
  - Hindsight relabeling: HiR, +10.6 IFBench on Qwen3-4B.
  - Counterfactual teacher shaping: CC-OPD.
  - Counterpoint: with a fine-tuned rubric verifier, an all-or-nothing reward beat a fractional one (AdvancedIF 58.1 vs 53.6).
- **Structure is a stronger difficulty lever than count.**
  - ComplexBench, GPT-4: And 0.881, Chain 0.766, Selection at depth ≥3 0.694, Selection+Chain at depth ≥3 0.626. On the coherent test for multi-layer Selection, GPT-4 scores 14.9%.
  - CSE: structural constraints lose about 2× more per added constraint than lexical ones.
  - Constraints scoped to parts of the output (ScopeIF test set: best frontier model 21.8% instruction success) and procedures derived from programs (LogicIF: most models below 60%) are the hardest verifiable IF tasks available.
- **Back-translation is correct by construction but cheap in difficulty.**
  - Crab, VerIF and RECAST derive constraints from an existing response, so the constraints are satisfiable, but the task is never harder than that response.
  - It also invites copying: UNSPECIFIC (constraints shared by two references, plus a satisfaction check on a summary) dropped GPT-5 Mini's satisfaction from 90% to 78%.
  - Crab's SFT data did not help FollowBench L3–L5 relative to Conifer.
- **Always AND the constraint reward with an intent or quality check, and track hacking.**
  - IFBench: gate the reward with a preference RM.
  - IFDecorator: IntentCheck cut the trip-wire hack rate from 14.53% to 7.60%.
  - RLCF and AdvancedIF: add universal anti-preamble and anti-artifact criteria.
  - Kimi K2: a hack-check layer catches responses that claim compliance falsely.
  - With a GPT-4o-mini rubric verifier, the incorrect-credit rate rose from 39% to 65% over training.
- **Monitor cross-domain side effects.**
  - IF-RLVR shifts math responses toward direct-answer openings and lowers best@k.
  - Math-RLVR raised IFEval pass@1 by 6.5% but lowered best@32 by 9.8% (Qwen3-8B-Base).
  - Track pass@k and best@k as well as pass@1 when you mix hardened IF data into multi-domain RL.

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| FollowBench | 2023 | [2310.20410](https://arxiv.org/abs/2310.20410) | Open-ended IF (content, situation, style, format, example, mixed) | Eval | +1 constraint per level (L1–L5); n−1 noise examples at level n | Rule programs where possible, otherwise a GPT-4 judge given the constraint-evolution path (88% agreement with humans) |
| ComplexBench | 2024 | [2407.03978](https://arxiv.org/abs/2407.03978) | Compositional IF | Eval | And / Chain / Selection, recursively nested; 4 types × 19 dimensions | Rule-augmented LLM scoring with dependency-aware aggregation (87.82% question-level agreement with humans) |
| Evol-Instruct (WizardLM) | 2023 | [2304.12244](https://arxiv.org/abs/2304.12244) | General IF | SFT | add constraints, deepen, concretize, add reasoning steps, complicate input; in-breadth mutation | Elimination filter only; answers unverified |
| Tree-Instruct | 2023 | [2308.05696](https://arxiv.org/abs/2308.05696) | General IF | SFT | add N nodes (3/6/10) to the instruction's semantic tree | None (teacher responses) |
| Conifer | 2024 | [2404.02823](https://arxiv.org/abs/2404.02823) | Constrained IF | SFT (+DPO) | reframe query → constraint categories/items → 5 levels; easy-to-hard multi-turn packing | GPT-4 two-stage filtering (missing context, conflicts); GPT-4 answers and critiques |
| AutoIF | 2024 | [2406.13542](https://arxiv.org/abs/2406.13542) | Verifiable IF | SFT + offline/online DPO | self-instruct expansion of atomic constraints; graft onto real queries | Verifier functions and test cases cross-validated by execution; NLI back-translation; rejection sampling |
| Crab | 2024 | [2410.24175](https://arxiv.org/abs/2410.24175) | Complex constrained IF | SFT (DPO on UltraFeedback) | constraint back-translation; sample 6–8 constraints and shuffle | Constraints derived from an existing response; Python for 6 types; LLM re-check |
| UltraIF | 2025 | [2502.04153](https://arxiv.org/abs/2502.04153) | IF from real prompts | SFT + iterative DPO/NCA | learned composer adds (constraint, evaluation question) pairs, applied iteratively | Per-constraint evaluation questions for rejection sampling |
| RECAST | 2025 | [2505.19030](https://arxiv.org/abs/2505.19030) | Very dense multi-constraint IF | SFT + RL (RLVC) | response-grounded extraction; 10–15+ constraints; nested 5/10/15/all test levels | 9 rule extractors plus LLM validators; multi-LLM voting on instructions and responses |
| MulDimIF | 2025 | [2505.07591](https://arxiv.org/abs/2505.07591) | Code-verifiable IF | Eval + RL | add categories up to Level I–IV; rewrite as Example / Listing / Incorporation | Per-instance Python checkers; LLM conflict check |
| RAIF | 2025 | [2506.01413](https://arxiv.org/abs/2506.01413) | Compositional IF with reasoning | SFT + RL | And/Chain/Selection/Nested templates on WildChat and Alpaca seeds | Python for lexical, numeric and format constraints; boolean LLM RM for semantic ones |
| VerIF / VerInstruct | 2025 | [2506.09942](https://arxiv.org/abs/2506.09942) | IF RLVR | RL | back-translation plus synthesized length constraints (~6.2 per prompt) | Code for hard constraints; QwQ-32B (or distilled IF-Verifier-7B) for soft |
| IFBench / IF-RLVR | 2025 | [2507.02833](https://arxiv.org/abs/2507.02833) | Precise IF | Eval + RL | 58 OOD test and 29 new train constraint types; multiple constraints per prompt; variable-range shifts | Hand-written verification functions; conflict dictionary; RM gating |
| RLCF | 2025 | [2507.18624](https://arxiv.org/abs/2507.18624) | General IF | RL (DPO) | instruction → weighted checklist mined from candidate failures | Judge scores averaged over 25 samples; verifier programs where exact |
| Rubrics as Rewards | 2025 | [2507.17746](https://arxiv.org/abs/2507.17746) | Medicine, science | RL (GRPO) | Essential / Important / Optional / Pitfall rubric items | Per-criterion LLM judge (explicit) or holistic rubric judge (implicit) |
| IFDecorator | 2025 | [2508.04632](https://arxiv.org/abs/2508.04632) | IF RLVR data flywheel | RL | iterative constraint addition (≤3n at iteration n) until pass rate is in (0, 0.5] | Scripts + LLM checklist, ANDed with IntentCheck; trip-wire monitoring |
| LogicIF | 2025 | [2508.09125](https://arxiv.org/abs/2508.09125) | Logic-rich IF from code | Eval + RL | program → NL procedure; anonymization; state trackers; difficulty evolution | Execute the anonymized code for outputs and tracker values |
| Self-supervised IF-RL (Ren et al.) | 2025 | [2510.14420](https://arxiv.org/abs/2510.14420) | Multi-constraint IF RL | RL | GPT-4o stacking (23 hard, 25 soft types); prefix curriculum L1–L5 | Rules for hard constraints; soft-constraint classifier trained on adjacent-level pseudo-labels |
| AdvancedIF / RIFL | 2025 | [2511.10507](https://arxiv.org/abs/2511.10507) | Complex, multi-turn and system-prompt IF | Eval + RL | adversarially collected 6+-instruction prompts; carried context; system-prompt steering | Expert rubrics; fine-tuned rubric generator and verifier; anti-artifact criteria |
| HiR | 2025 | [2512.23457](https://arxiv.org/abs/2512.23457) | Multi-constraint IF RL | RL | reverse operator: relabel a failure to the constraint subset it satisfied | Same per-constraint verifiers as the reward |
| LsrIF | 2026 | [2601.06431](https://arxiv.org/abs/2601.06431) | Logic-structured IF | RL | parallel / sequential / conditional / nested constraint trees | Code or RM at leaves; structure-aware aggregation up the tree |
| Precision over Diversity | 2026 | [2601.04954](https://arxiv.org/abs/2601.04954) | IF RLVR data composition | RL (analysis + recipe) | hard-only selection; ≤1 soft constraint; pilot-run learnability filtering | Code verifiers (96% precision) |
| Kimi K2 IF-RL | 2025 | [2507.20534](https://arxiv.org/abs/2507.20534) | Frontier IF RL | RL | expert conditional prompts; AutoIF-style augmentation; fine-tuned failure-mode prompt generator | Code interpreter + LLM judge + hack-check layer |
| SEIF | 2026 | [2605.07465](https://arxiv.org/abs/2605.07465) | Self-play IF | RL | trained Instructor adds constraints to seeds (reward = Follower failure) | Filter and Judger instantiated from the latest Follower |
| LLM-as-a-Tutor | 2026 | [2607.04412](https://arxiv.org/abs/2607.04412) | Rubric-rewarded IF RL | RL | append one atomic constraint (and rubric item) when rollouts are indistinguishable | Rubric reward from an LLM judge |
| CSE | 2026 | [2608.12426](https://arxiv.org/abs/2608.12426) | Compositional analysis | Eval | k-subset sampling (k=1–12) from 36 types with calibrated parameters | Deterministic verifiers; satisfiability checker; 444 deliberately impossible probes |
| UNSPECIFIC | 2026 | [2608.09154](https://arxiv.org/abs/2608.09154) | Long-form writing | Eval | constraints shared by two references; selective hardening; summary-level check | LLM judge on the article and on its summary |
| CC-OPD | 2026 | [2609.27421](https://arxiv.org/abs/2609.27421) | Multi-constraint distillation | On-policy distillation (PPO) | leave-one-constraint-out counterfactual prompts | Frozen RL-trained teacher; no external verifier during distillation |
| ScopeIF *(added)* | 2026 | [2609.32189](https://arxiv.org/abs/2609.32189) | Scope-aware precise IF | RL | Scope × Target × Range recombination; LLM crossover to k ∈ {2,4,6,8} | Tool-grounded verification (the judge writes and executes Python); graded rewards |
| ImpRIF *(added)* | 2026 | [2602.21228](https://arxiv.org/abs/2602.21228) | Implicit-reasoning IF | SFT + RL | hide conditional, math and knowledge nodes of a reasoning DAG; multi-turn injection | Executable code per constraint; rubric RM for multi-turn |
| QUBRIC *(added)* | 2026 | [2606.03968](https://arxiv.org/abs/2606.03968) | Open-ended (non-verifiable) IF | RL | rewrite open-ended queries into scenario questions grounded in key points | Validation checks; contrastive teacher-vs-policy rubrics; 20–50% learnability corridor |
| Inverse IFEval *(added)* | 2025 | [2509.04292](https://arxiv.org/abs/2509.04292) | Counter-intuitive IF | Eval | invert SFT conventions (8 types) | Human-in-the-loop construction; tuned LLM judge |
| Reward Hacking in Rubric-Based RL *(added)* | 2026 | [2605.12474](https://arxiv.org/abs/2605.12474) | Rubric RL analysis | Analysis | — | Cross-family panel of 3 frontier reference judges |
| MDP-GRPO *(added)* | 2026 | [2606.06058](https://arxiv.org/abs/2606.06058) | Multi-constraint IF RL | RL (optimizer) | — (keeps hard stacked prompts learnable) | Verifiable constraint checkers |

---

## Method notes

### FollowBench — FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models (Jiang et al., 2023; ACL 2024)
Link: https://arxiv.org/abs/2310.20410
- **Mechanism:** 820 instructions in five constraint categories (Content, Situation, Style, Format, Example), plus Mixed. Each seed becomes a 5-level chain in which level n adds exactly one new constraint to level n−1. Mixed chains add constraints of different types. Example chains are built from PromptSource, and level n contains n−1 noise examples in the input.
- **How it makes tasks harder:** Constraints accumulate one at a time on the same seed. Distractor examples grow with the level.
- **Correctness / verification:** Hybrid. Rule programs cover the checkable items. A strong LLM judge is prompted with the *constraint-evolution path*, so it knows which constraint was added at each level; this agreed with expert humans 88% of the time.
- **Difficulty control:** The level (1–5). Metrics: HSR (all constraints met), SSR (mean per constraint), and CSL (number of consecutive levels satisfied from L1).
- **Reported results:** GPT-4-Preview-1106 HSR fell from 84.7 at L1 to 61.9 at L5, with CSL 3.3. The best open model, Qwen-Chat-72B, had CSL 2.4, and LLaMA2-Chat-70B had 2.1.
- **Limitations / failure modes:** Only 5 constraints, essentially And-composition. Soft constraints need an LLM judge. Frontier models are now near the ceiling.
- **How to reuse with easy seed tasks:** Emit L1…Lk variants per seed, each adding one constraint. Store the evolution path and pass it to the judge so it can focus on the newest constraint. The first failing level for the current policy is a free difficulty label.

### ComplexBench — Benchmarking Complex Instruction-Following with Multiple Constraints Composition (Wen et al., 2024; NeurIPS 2024 D&B)
Link: https://arxiv.org/abs/2407.03978
- **Mechanism:** A taxonomy of 4 constraint types (Lexical, Format, Semantic, Utility) across 19 dimensions. Composition types are Single, And, Chain and Selection; Chain tasks and Selection branches can themselves contain compositions, which gives nesting. In Chain, tasks T1…Tn run in order and T(k+1) may depend on T(k). In Selection, a selection function S(cond) chooses the branch whose output is expected. 1,150 manually built instructions.
- **How it makes tasks harder:** Structural composition and nesting depth, beyond simply adding constraints.
- **Correctness / verification:** Rule-augmented LLM evaluation (RAL). The LLM extracts the relevant segment and a rule scores it when possible. Scores are aggregated with dependencies, so a question fails if a question it depends on fails. Only 17% of scoring questions can be checked by rules. On that subset, RAL agrees with humans 95.36% of the time vs 82.02% for the LLM alone; overall agreement is 87.82%.
- **Difficulty control:** Composition type × nesting depth × number of constraints.
- **Reported results (DRFR):**
  - GPT-4-1106: And 0.881; Chain 0.766 on average; Selection at depth 1/2/≥3 = 0.815/0.772/0.694; Selection+Chain at depth 2/≥3 = 0.802/0.626.
  - On the coherent test (one response must be right on every branch variant), GPT-4 scores only 14.9% on multi-layer nested Selection.
  - Decomposing instructions into multi-round sub-steps *lowered* GPT-3.5 from 0.682 to 0.652.
  - Models do worse on Format and Lexical constraints than on Semantic and Utility ones.
- **Limitations / failure modes:** About 1k hand-built items, so it is not a training generator. Semantic and Utility items depend on an LLM judge.
- **How to reuse with easy seed tasks:** Use the grammar as a generator: wrap a seed in a Chain (summarize → transform → title) or gate constraint sets behind a condition that can be checked. Keep the dependency graph with each item and use dependency-aware reward aggregation.

### Evol-Instruct (WizardLM) — WizardLM: Empowering Large Pre-trained Language Models to Follow Complex Instructions (Xu et al., 2023; ICLR 2024)
Link: https://arxiv.org/abs/2304.12244
- **Mechanism:** In-depth evolution prompts (add constraints, deepening, concretizing, increase reasoning steps, complicate input) and in-breadth mutation to new topics. There are 4 evolution epochs from Alpaca seeds with ChatGPT, giving 250k instructions; a 70k sample fine-tunes LLaMA-13B. Elimination evolving discards four kinds of failure:
  - no information gain (judged by ChatGPT);
  - a response containing "sorry" and under 80 words;
  - a response of only punctuation and stop words;
  - leaked template text such as "#Rewritten Prompt#".
- **How it makes tasks harder:** Compounding LLM rewrites.
- **Correctness / verification:** None for answers; responses come from ChatGPT.
- **Difficulty control:** The number of epochs and the operator mix. Difficulty was measured afterwards with ChatGPT ratings.
- **Reported results:** Automatic and human evaluations show WizardLM beating Alpaca and Vicuna.
- **Limitations / failure modes:** Constraints cannot be verified, so the data is not RL-ready. Rewrites drift, conflict or become unanswerable. In LLM-as-a-Tutor, offline Evol-Instruct hardening underperformed unmodified prompts under rubric RL (50.24 vs 50.51).
- **How to reuse with easy seed tasks:** Use the operator names as a prompt library, but restrict "add constraint" to a catalog in which every constraint has a verifier. Keep the elimination rules as cheap filters.

### Tree-Instruct — A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment (Zhao et al., 2023; LREC-COLING 2024)
Link: https://arxiv.org/abs/2308.05696
- **Mechanism:** Parse the instruction into a semantic tree, add a specified number of nodes (3, 6 or 10), and render the tree back to text.
- **How it makes tasks harder:** Controlled growth in the number of semantic units.
- **Correctness / verification:** None (teacher responses).
- **Difficulty control:** The number of added nodes.
- **Reported results:**
  - On 1,000 Alpaca-GPT4 samples, adding 3/6/10 nodes gave gains of 13%/20%/26% across 8 sub-skills. Adding 10 nodes raised average instruction length from 186 to 607 tokens.
  - At equal token budget (4,000 original instructions vs 1,000 with 10 nodes added), the complex set still gained more than 15%.
  - Curriculum ordering (3 → 6 → 10 nodes) beat mixed ordering but lost to training on 10-node data only.
  - It beat three rounds of WizardLM in-depth evolution by 2% win rate at equal tokens.
- **Limitations / failure modes:** Small scale and no verifier. Added nodes can read unnaturally.
- **How to reuse with easy seed tasks:** Treat added semantic units as an integer difficulty knob. At a fixed token budget, prefer fewer, harder prompts, and test curricula rather than assuming they help.

### Conifer — Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models (Sun et al., 2024)
Link: https://arxiv.org/abs/2404.02823
- **Mechanism:** The authors found GPT-4 "consistently struggles" to write multi-constraint instructions in one shot, so they split the job into steps:
  1. Take 6,000 ShareGPT seeds and have GPT-4 reframe each into at least 3 forms.
  2. Extract subjects and objects, then generate constraints as category → item.
  3. Recombine them into difficulty levels 1–5; each harder level adds 1–2 categories and 2–3 items.
  4. Augment 1,000 instructions with 32 curated format and numeric constraints.
  5. Filter twice: remove prompts missing context, then resolve conflicts.
  The result is 35,613 instructions → 13,606 multi-turn conversations (3.02 turns on average) that pack levels easy-to-hard, plus internal and external process feedback.
- **How it makes tasks harder:** Adding constraint categories and items level by level.
- **Correctness / verification:** GPT-4 generation, filtering and critiques. No programmatic verifier.
- **Difficulty control:** Explicit levels 1–5 and progressive multi-turn ordering.
- **Reported results:** Mixed with 53k ShareGPT examples. Conifer-7B-DPO: IFEval loose 52.3 and FollowBench L5 HSR 41.0 (Qwen-72B-Chat 39.9).
- **Limitations / failure modes:** Teacher answers at high levels may violate constraints. GPT-4 favors content constraints over format and numeric ones, which needed a separate augmentation step.
- **How to reuse with easy seed tasks:** Break "write a hard instruction" into small generator calls. Package several levels of one seed as a multi-turn SFT sample.

### AutoIF — Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models (Dong et al., 2024; ICLR 2025)
Link: https://arxiv.org/abs/2406.13542
- **Mechanism:**
  1. Hand-write seed instructions, each with one atomic constraint (e.g., "answer with words that begin with B"), and expand them with LLM rewrites.
  2. For each instruction, sample K verification functions and test cases. Keep a function only if it compiles and scores above 0.5 accuracy on the test cases; keep a test case only if it scores above 0.5 across the functions.
  3. Back-translate each function to an instruction and drop it if an NLI model finds contradiction with the original.
  4. Concatenate the verified constraints with ShareGPT queries and sample responses; keep a response if it passes more than 0.5 of the functions.
  5. Have the LLM score query–instruction compatibility 1–10 and keep scores ≥8.
  6. Train with SFT, then offline or online DPO.
- **How it makes tasks harder:** Grafting verified constraints onto real queries.
- **Correctness / verification:** Execution-based cross-validation, NLI back-translation, and a compatibility score for query and constraint.
- **Difficulty control:** Implicit. The pass-rate threshold controls quality more than difficulty; at the query level, performance peaked at a pass-rate threshold of 0.8.
- **Reported results:** IFEval loose instruction accuracy 88.0 (Qwen2-72B) and 90.4 (LLaMA3-70B) with online DPO. FollowBench SSR improved by more than 5%. Removing cross-verification caused the largest single-component drop; removing every quality step cost 2.2 to 3.8 points.
- **Limitations / failure modes:** Atomic, IFEval-like surface constraints. A synthesized verifier can encode a wrong reading of the instruction.
- **How to reuse with easy seed tasks:** Build a library of (NL constraint, verifier, unit tests) triples validated by execution. Graft k sampled constraints onto real easy queries and use the verifiers directly as RLVR rewards.

### Crab — Constraint Back-translation Improves Complex Instruction Following of Large Language Models (Qi et al., 2024; CIKM 2025)
Link: https://arxiv.org/abs/2410.24175
- **Mechanism:**
  - Take 4,500 high-quality seed pairs (Alpaca-GPT4, Orca Chat, Evol-Instruct, OpenAssistant; responses over 300 words).
  - Llama3-70B-Instruct generates constraints the response already satisfies, using 13 seed types as examples, which yields more than 100 types; it then re-verifies them.
  - Python handles 6 types (length, keyword, punctuation, …): it fills templates with values measured from the response, and for length it samples a range that contains the actual value.
  - ROUGE-L < 0.6 deduplicates constraints.
  - 6–8 constraints are sampled and shuffled per instruction, and half the data gets 1–3 in-context demonstrations. The result is 13,500 instances averaging 7.1 constraints.
  - A reverse-training objective (predict the constraints from the instruction and response) is used for the first 70% of training, then forward SFT.
- **How it makes tasks harder:** Appending 6–8 response-grounded constraints to an easy instruction.
- **Correctness / verification:** Correct by construction (the response exists first). A manual review of 50 pairs found minimal noise.
- **Difficulty control:** The number and types of constraints.
- **Reported results:** MistralCrab IFEval average 54.5 (Conifer-SFT 53.9); with DPO (on UltraFeedback, not Crab data) 59.3 (Conifer-DPO 55.7). FollowBench HSR over L1–L5 averages 43.3 (SFT) and 49.4 (DPO). SFT Crab trails Conifer-SFT at L3–L5 (40.1/30.4/27.9 vs 49.3/40.8/30.5).
- **Limitations / failure modes:** Never harder than the existing response. Over-specific constraints can be satisfied by copying (see UNSPECIFIC). Weak at high constraint levels.
- **How to reuse with easy seed tasks:** Take your easy SFT pairs, compute 5–15 checkable properties of each response with code, and append them. For RL, drop the gold answer and keep the checkers as the reward.

### UltraIF — UltraIF: Advancing Instruction Following from the Wild (An et al., 2025; EMNLP 2025)
Link: https://arxiv.org/abs/2502.04153
- **Mechanism:**
  - Decompose real ShareGPT-style prompts into a simplified query, its constraints, and one evaluation question per constraint. For example, "In Shakespeare's tone, recommend me ten Chinese books" becomes "Recommend me ten Chinese books" + "In Shakespeare's tone" + "Is the response written in Shakespeare's tone?".
  - Train UltraComposer (LLaMA-3.1-8B-Instruct) on the reverse mapping: simple query → complex instruction + evaluation question.
  - Apply it iteratively to add constraints. Generate-then-Evaluate filters responses with the evaluation questions for SFT and iterative DPO, switching to NCA in the last iteration because DPO rewards drifted below zero.
- **How it makes tasks harder:** A learned complexifier that mirrors the distribution of real user constraints.
- **Correctness / verification:** Each added constraint comes with an evaluation question that an LLM answers during rejection sampling.
- **Difficulty control:** The number of composer iterations. SFT-stage pass rate fell from 91.76% (1 constraint) to 86.41% (2) to 79.44% (3) in the strong-to-weak setting.
- **Reported results:** LLaMA-3.1-8B-Base with 181k SFT + 20k DPO samples, self-aligned with an 8B supervisor:
  - IFEval Pr(S) 71.35 / Pr(L) 75.42 / Ins(S) 79.38 / Ins(L) 83.09 (LLaMA-3.1-8B-Instruct: 69.13 Pr(S), 77.46 Ins(S));
  - Multi-IF turn 1/2/3: 69.63 / 58.28 / 46.86;
  - InfoBench 80.70; LiveBench 56.00; FollowBench SSR 62.55.
- **Limitations / failure modes:** LLM-answered evaluation questions are weak on exact counts. Diversity is bounded by the source prompts.
- **How to reuse with easy seed tasks:** Mine real hard prompts, strip their constraints to recover the easy query, and train a small composer on (easy → hard + check). Apply it 1–3 times to your easy seeds and route countable checks to code.

### RECAST — RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data (Guo et al., 2025; ICLR 2026)
Link: https://arxiv.org/abs/2505.19030
- **Mechanism:**
  - Nine rule-based extractors find verifiable properties of seed responses (paragraph count, keywords, word limits, …).
  - A three-step LLM process proposes model-based constraints from 10 categories (style, tone, role, …) and keeps only those the response satisfies.
  - LLMs select a coherent subset of constraints, several LLMs rewrite the instruction to include them, and majority voting picks the best rewrite.
  - Several LLMs then *generate new responses*, and voting selects one, since the original response may not fit the enriched instruction.
  - RECAST-30K averages 13.4 constraints across 19 types; 78.8% of instances have ≥10 and 36.4% have ≥15.
  - RLVC runs GRPO on the mean per-constraint satisfaction, one reward channel per constraint.
- **How it makes tasks harder:** Extreme constraint density.
- **Correctness / verification:** Rule checkers for quantitative constraints and LLM validators for qualitative ones, with human spot-checks. Because responses are regenerated, the data is not strictly correct by construction.
- **Difficulty control:** The number of constraints. RECAST-Test starts from 500 items with ≥15 constraints and nests them: L1=5, L2=10 (a superset of L1), L3=15, L4=all.
- **Reported results:**
  - Building the dataset cost about $175.
  - Qwen2.5-7B on RECAST-Test: SFT 31.25%, RLVC 32.33% average satisfaction, with RLVC gains concentrated at L3–L4.
  - RLVC generalization: IFEval Pr(L) 77.39 (Llama3.1-8B) and 74.01 (Qwen2.5-7B); FollowBench HSR 61.76 and 63.23.
  - Gemini-2.5-Pro is best at 39.75% average. All-constraint (OSR) rates at L4 are ≤7% for all models.
  - Llama3.1-8B-Base + RECAST-30K beats Llama-3.3-70B-Instruct on RECAST-Test.
- **Limitations / failure modes:** Mean rewards let the policy trade away hard constraints. Some extracted constraints are trivial. Model-based items rely on LLM judges.
- **How to reuse with easy seed tasks:** When seeds saturate, extract 10–20 properties from strong responses, rewrite the instruction to include them, and use nested supersets as an automatic ladder. Reward with per-constraint channels.

### MulDimIF — MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models (Ye et al., 2025; ACL 2026)
Link: https://arxiv.org/abs/2505.07591
- **Mechanism:**
  - Constraint expansion picks a category not yet covered (Content, Format, Language, Length) and adds 1–2 constraints from it, until Level IV.
  - Conflict detection discards the instance if the new constraint was not correctly incorporated, or if constraints contradict (e.g., all-lowercase vs required uppercase words).
  - Instruction rewriting picks a presentation pattern: Example (three QA pairs sharing the constraint subcategories), Listing, or Incorporation.
  - The result is 9,106 code-verifiable instances. The GRPO reward is the number of satisfied constraints, with 32 rollouts per prompt.
- **How it makes tasks harder:** More categories and elements, and harder presentation.
- **Correctness / verification:** Per-instance Python checkers.
- **Difficulty control:** Level I (1 type, 1–2 elements), II (2 types, 2–4), III (3 types, 3–6), IV (4 types, 4–8), crossed with the pattern.
- **Reported results:**
  - Across 18 LLMs, accuracy falls from 80.82% (Level I) to 36.76% (Level IV); the best model scores 67.50%.
  - GRPO takes LLaMA3.1-8B-Instruct from 36.17 to 88.08 overall and from 12.00 to 84.00 at Level IV. Gains transfer to IFEval and Multi-IF, general benchmarks hold, and updates concentrate in attention modules.
- **Limitations / failure modes:** Four surface categories. The large in-domain gains are partly learning the templates.
- **How to reuse with easy seed tasks:** Parameterize level as (#categories, #elements) and pattern as presentation style. Generate every combination per seed and train where the policy's pass rate is intermediate.

### RAIF — Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models (Qin et al., 2025; NeurIPS 2025)
Link: https://arxiv.org/abs/2506.01413
- **Mechanism:**
  - Seeds from WildChat and Alpaca are tagged and evolved with constraints from templates (lexical, numerical, format, semantic, stylistic, linguistic), composed as And/Chain/Selection/Nested.
  - Compatibility checks and LLM checks for 7 known quality issues filter the data.
  - Reward: a format reward (+1 if think/answer tags are present, −1 otherwise) plus an accuracy reward over the *active* constraints: +2 if all are satisfied, the fraction satisfied if some are, −2 if none.
  - Sample-wise contrast keeps samples where reasoning beats answering without reasoning. Behavior cloning of expert outputs stabilizes the shift to reasoning.
- **How it makes tasks harder:** Structured composition, plus requiring reasoning to pay off.
- **Correctness / verification:** Python heuristics for lexical, numeric and format constraints; a boolean LLM RM for semantic, stylistic and linguistic ones.
- **Difficulty control:** Composition type and depth; contrast filtering.
- **Reported results:** Qwen2.5-1.5B-Instruct average over 7 benchmarks rose from 50.61 to 62.35 (+11.74), comparable to an 8B model. Vanilla CoT *dropped* the same model by 11.79 because it paraphrases the instruction. IFEval itself slipped slightly (45.28 → 44.91) while CFBench, ComplexBench, FB-Bench and FollowBench rose.
- **Limitations / failure modes:** The LLM RM for soft constraints; partial credit can be gamed; heavy compute.
- **How to reuse with easy seed tasks:** Recompose saturated seeds into Chain and Selection with conditions the checker can evaluate. Use the piecewise ±2 reward and drop prompts where reasoning gives no gain.

### VerIF / VerInstruct — VerIF: Verification Engineering for Reinforcement Learning in Instruction Following (Peng et al., 2025; EMNLP 2025)
Link: https://arxiv.org/abs/2506.09942
- **Mechanism:**
  - Sample 25,000 instances (Alpaca-GPT4, Orca Chat, Evol-Instruct, OpenAssistant). Llama3.1-70B-Instruct back-translates implicit constraints, and Python adds length constraints computed from the response.
  - Qwen2.5-72B writes verification code for hard constraints (checked manually, with nearly no errors). Soft constraints are tagged "LLM" and judged online by QwQ-32B.
  - Drop instances with fewer than 2 constraints. This leaves 22,000 instructions averaging 6.2 constraints.
  - Reward = mean of binary hard (code) and binary soft (QwQ) scores; GRPO with 16 rollouts.
  - IF-Verifier-7B is distilled from about 130k QwQ judgments (WildChat and Infinity-Instruct prompts, responses from 6 LLMs).
- **How it makes tasks harder:** About 6 back-translated constraints per easy instruction.
- **Correctness / verification:** Consistent with the response by construction; hybrid code and reasoning-LLM verification.
- **Difficulty control:** None explicit.
- **Reported results:**
  - TULU 3 SFT → +VerIF: IFEval Pr(S) 68.4 → 84.5; Multi-IF turn 3 40.3 → 54.0; FollowBench SSR 62.0 → 68.6; CFBench 63 → 72.
  - Ablation (IFEval Pr(S)): code-only (no LLM) 74.7; non-reasoning Qwen2.5-72B judge 76.9; full 84.5.
  - LLM-only verification showed faster reward growth, a sign of hacking.
  - IF-Verifier-7B gives 80.0 IFEval vs 84.5 with QwQ, at far lower cost.
- **Limitations / failure modes:** The reasoning judge is expensive, and later work finds soft judges noisy (Precision over Diversity).
- **How to reuse with easy seed tasks:** Split every synthetic constraint into hard (code) and soft (judge). Use a reasoning judge or a distilled verifier, and audit the judge's recall on violations before scaling.

### IFBench / IF-RLVR (incl. Tulu 3 precedent) — Generalizing Verifiable Instruction Following (Pyatkin et al., 2025; NeurIPS 2025 D&B)
Link: https://arxiv.org/abs/2507.02833
- **Mechanism:**
  - Tulu 3 ([2411.15124](https://arxiv.org/abs/2411.15124)) generated about 30k persona-driven prompts covering the 25 IFEval constraint types for IF SFT and RLVR.
  - IFBench adds 58 new out-of-distribution test constraints (counting, copying, word, sentence and character manipulation, …) and IFTrain, 29 new training constraints with verification functions.
  - RLVR prompts come from TÜLU-SFT with 1 to n constraints appended (n ≤ 6); a dictionary of constraint conflicts blocks contradictory combinations.
  - Ablations cover constraint count, seen vs unseen types, variable ranges, leaving out a category, GRPO vs DPO, base vs instruct starting points, and RM mixing.
- **How it makes tasks harder:** New constraint families, more constraints per prompt, shifted variable ranges.
- **Correctness / verification:** Hand-written Python verification functions.
- **Difficulty control:** The number of constraints per prompt and the variable ranges.
- **Reported results:**
  - Constraints per prompt, Qwen2.5 policy: IFBench 48.9 (1), 53.1 (2), 59.5 (3), 49.4 (4), 55.8 (5), 54.1 (6). TÜLU-DPO policy: up to 5–6 beat up to 3.
  - Variable ranges: disjoint train ranges hurt; wider ranges ≥ same range.
  - GRPO beat DPO on identical prompts and verifiers (IFEval strict 89.65 vs 79.67, both starting from DPO).
  - Final models: TÜLU-3-8B IFEval 82.4 → 92.2 and IFBench 28.9 → 45.9; Qwen2.5-7B base → 87.8 IFEval and about 54 IFBench (53.7 in the table, 54.7 in the text).
  - In the preference data (up to 5 constraints per prompt), only 6–26% of completions from 5 strong open models (Qwen-72B best) satisfied all constraints, and 54% of fully correct completions had a single constraint.
  - Pure IF-RLVR over-optimizes the constraints at the expense of the rest of the response. Fix: F = V+1 if V>0 and RM>7; V−0.5 if V>0 and RM≤7; V otherwise.
- **Limitations / failure modes:** Surface constraints only; writing verifiers by hand limits scale.
- **How to reuse with easy seed tasks:** Grow the constraint library well beyond IFEval, hold out whole families for evaluation, train with 3–6 constraints per prompt over wide parameter ranges, and gate the reward with a quality RM.

### RLCF — Checklists Are Better Than Reward Models For Aligning Language Models (Viswanathan et al., 2025; NeurIPS 2025)
Link: https://arxiv.org/abs/2507.18624
- **Mechanism:**
  - Candidate-based checklists: generate responses of varying quality (Qwen2.5 0.5B–7B), then have Qwen2.5-72B-Instruct write a checklist of every failure mode, with importance weights from 0 to 100. This beat asking for a checklist directly on objectivity, atomicity and downstream RL.
  - Two universal items are always added: "directly addresses the request without excessive or off-topic information" and "matches the required context and register". They were added after the model learned to open with long preambles.
  - Each item is scored by the judge (mean of 25 sampled 0–100 scores), plus verifier programs where exact checks are possible. Pairs from the top 40% by score gap go to DPO.
- **How it makes tasks harder:** Richer, instruction-specific success criteria rather than harder prompts.
- **Correctness / verification:** Averaging repeated judge samples; code checks for exact items.
- **Difficulty control:** None explicit.
- **Reported results:** WildChecklists covers 130k WildChat instructions. On Qwen2.5-7B-Instruct, RLCF is the only method that improves all 5 benchmarks: FollowBench HSR +5.5% relative (about +4 points) and CSL +8.2%, InFoBench +6 points, Arena-Hard +3 points, and IFEval loose +2.8–3.0% relative.
- **Limitations / failure modes:** 25 judge calls per item; used off-policy.
- **How to reuse with easy seed tasks:** For soft requirements, mine checklist items from the failures of sampled weak responses, route exact items to code, and always add anti-preamble and relevance items.

### Rubrics as Rewards — Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains (Gunjal et al., 2025)
Link: https://arxiv.org/abs/2507.17746
- **Mechanism:** o3-mini or GPT-4o writes rubrics per prompt, conditioned on reference answers. Items are tagged Essential, Important, Optional or Pitfall. gpt-4o-mini judges during GRPO, either per criterion with a normalized weighted sum (explicit) or with the whole rubric at once (implicit).
- **How it makes tasks harder:** Replaces one Likert judgment with many criteria, including negative "pitfall" items.
- **Correctness / verification:** Criterion-level judgments grounded in references.
- **Difficulty control:** Rubric granularity.
- **Reported results:** Up to 31% relative gain on HealthBench and 7% on GPQA-Diamond over Likert LLM-judge rewards. Rubrics narrow the gap between small and large judges.
- **Limitations / failure modes:** Weak verifiers get exploited (see the added rubric-hacking entry). Rubrics are dominated by presence criteria.
- **How to reuse with easy seed tasks:** Turn every soft constraint into a weighted atomic criterion and add pitfall and absence items.

### IFDecorator — IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards (Guo et al., 2025)
Link: https://arxiv.org/abs/2508.04632
- **Mechanism:**
  - A cooperative-adversarial flywheel. The Instruction-Former adds constraints and updates the verification: at iteration n it applies a dynamic evolution template n times and adds up to 3n IFEval-style verifiable constraints, randomly reordering few-shot examples to balance constraint types.
  - The Instruction-Solver (Qwen2.5-32B-Instruct, T=1.0, 8 samples) measures the pass rate. Pass rate >0.5 → evolve further; =0 → regenerate from scratch; (0, 0.5] → keep.
  - LLM checks confirm that critical components of the task were preserved and that the instruction is reasonable. Math, code and reasoning prompts are filtered out.
  - The final data has 3,625 training samples.
  - Reward: IntentCheck(I,R) ∧ V(I,R), where V combines scripts with an LLM *checklist* ("strict reward").
  - Trip wires are held-out trap instructions that detect four exploit patterns (literal format placeholders like `<<title>>`, dummy lists, trivial repetition like "p p p", literal section delimiters), measured as a macro hack rate (MHR).
- **How it makes tasks harder:** Constraint addition in a loop until the policy fails often enough.
- **Correctness / verification:** Scripts, an LLM checklist and an intent gate. Prompts with pass rate 0 are removed as possibly contradictory.
- **Difficulty control:** Pass rate against the policy, not constraint count; they observe that complexity alone does not determine difficulty.
- **Reported results:** Qwen2.5-32B-Instruct → 87.43 IFEval Pr(S) (+7.95; GPT-4o 86.50) using about 0.71M synthetic tokens. FollowBench HSR 62.61 → 66.76, with larger gains at L3–L5. IntentCheck cut MHR from 14.53% to 7.60%. Ablations: removing the difficulty filter, the checklist reward or the non-instruction filter each hurt.
- **Limitations / failure modes:** Rollout cost of measuring difficulty; the trip wires are hand-designed.
- **How to reuse with easy seed tasks:** This is the default loop for "too easy" pools: append verifiable constraints until 8-sample pass@1 is in (0, 0.5], AND with an intent judge, and track MHR on a held-out trap set.

### LogicIF — LogicIF: Towards Complex Logic Instruction Following (Zhang et al., 2025; COLM 2026)
Link: https://arxiv.org/abs/2508.09125
- **Mechanism:**
  - Seed functions come from Codeforces "implementation" problems rated above 1700 and from hard POJ simulation problems: 1,107 seeds → 426 functions and 3,050 tests.
  - An LLM anonymizes names and injects state trackers (e.g., removals, max_heap_size).
  - Multi-turn difficulty evolution (1 turn) can make the function harder.
  - An LLM writes a detailed NL instruction, and multi-turn verification and refinement (up to 3 turns) checks coverage of loops, conditions and edge cases.
  - Tests are filtered to exclude errors, tracker values ≥50, outputs with more than 6 decimals, and values above 10^7.
  - The model must follow the NL procedure on an input and report the output *and* the tracker values.
- **How it makes tasks harder:** Program-shaped procedures with nesting, loops and function calls.
- **Correctness / verification:** Execution of the anonymized code. Two experts reviewed each instruction: 97% were faithful, with 97.79% agreement.
- **Difficulty control:** Score = 3·nesting depth + 2·function calls + 1·cyclomatic complexity + 0.5·lines; the eval set splits 142/145/139 easy/medium/hard.
- **Reported results:**
  - gpt-5 84.98%; most models are below 60% and several open models below 10%.
  - GRPO on LogicIFTrain (27,324 instructions) gave Qwen3-1.7B +16.7 on LogicIFEval, ZebraLogic +31.4, HumanEval +4.6, MATH-500 +4.3 and GPQA-Diamond +3.1.
  - Output accuracy exceeds tracker accuracy (GPT-4.1-mini on Hard: 71.22 vs 49.64).
- **Limitations / failure modes:** Procedural rather than natural user requests; long instructions (about 662 words on average).
- **How to reuse with easy seed tasks:** Turn any program you already have tests for into an NL-procedure task. Scale difficulty with the AST score and require intermediate state values to penalize right-answer/wrong-process shortcuts.

### Self-supervised IF-RL with incremental constraint curriculum — Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following (Ren et al., 2025)
Link: https://arxiv.org/abs/2510.14420 (sibling for reasoning models: [2508.02150](https://arxiv.org/abs/2508.02150))
- **Mechanism:**
  - GPT-4o adds constraints (23 hard types, 25 soft types) to 3,000 seeds.
  - An instruction with c1…cn becomes prefix levels L1…Ln, where L_k keeps the first k constraints; L1–L5 hold about 2.6–2.8k instructions each.
  - Soft constraints get a binary classifier trained without labels. The response to x_k (with c_k) is the positive for c_k; the response to x_(k−1) (without c_k) is the negative; the loss is BCE.
  - Reward = mean over present constraints of the rule reward (hard) or classifier probability (soft). 4,501 DeepScaleR math and 1,929 SciKnowEval problems are mixed in.
- **How it makes tasks harder:** Stacking up to 5 constraints, with the prefix ladder as a curriculum.
- **Correctness / verification:** Rules for hard constraints. The pseudo-labels agree with human annotation at Kendall τ 94.0 and position consistency 97.0. The resulting 7B RM reaches τ 62.7, vs 78.0 for constraint-level QwQ-32B, but runs 126.5× faster.
- **Difficulty control:** Curriculum levels.
- **Reported results:** Qwen2.5-1.5B-Instruct IFEval 43.6 → 65.2; Qwen2.5-7B-Instruct +5.0 IFEval and +5.0 CFBench; R1-0528-Qwen3-8B → 87.1 IFEval, +6.1 AgentIF and +7.3 MultiChallenge.
- **Limitations / failure modes:** The pseudo-label assumption (a response with c_k in the prompt satisfies c_k better) fails for weak generators; at most about 5 constraints.
- **How to reuse with easy seed tasks:** For every hardened prompt, emit its prefix ladder. It gives a curriculum and free contrastive labels for cheap per-constraint soft verifiers.

### AdvancedIF / RIFL — AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following (He et al., 2025)
Link: https://arxiv.org/abs/2511.10507
- **Mechanism:**
  - Expert-written prompts (via Surge) in three categories: Complex IF (6+ instructions mixing tone, format, style, structure, length, negative, spelling and inter-conditional constraints; 402 dialogs, 7.44 criteria each); Multi-turn Carried Context (736 dialogs, 7.69 turns); and System-Prompt Steerability (507 dialogs, 11.21 turns).
  - Prompts are collected adversarially: only prompts whose final turn made the model fail are kept. Each rubric has at most 20 criteria.
  - RIFL fine-tunes Llama 4 Maverick as a rubric generator and as a rubric verifier (SFT on about 5k prompts, then RL on about 14k).
  - Reward = 1 only if every criterion passes. Two criteria are always added: a clean response without artifacts or verbose self-evaluation, and no truncated ending. They were added after the policy began writing "all instructions are followed" to fool the verifier.
- **How it makes tasks harder:** Adversarial selection; carrying constraints across turns and system prompts.
- **Correctness / verification:** Expert rubrics and a trained verifier: F1 0.728, vs 0.515 for the vanilla judge and 0.723 for o3-mini. The rubric generator reaches F1 0.790 vs 0.639.
- **Difficulty control:** Adversarial filtering against the model, plus constraint density.
- **Reported results:** 1,645 benchmark prompts. Llama 4 Maverick + RIFL: AdvancedIF 51.4 → 58.1 (CIF +5.7, CC +5.4, SS +9.1); MultiChallenge +2.9; IFEval 89.9 → 90.0. Reward design: all-or-nothing 58.1 vs hybrid 55.7 vs fractional 53.6.
- **Limitations / failure modes:** Needs a seed of expert rubrics; the sparse reward is harder for weak policies.
- **How to reuse with easy seed tasks:** Keep only prompts the current model fails. Move constraints into system prompts and earlier turns. Train a rubric generator and verifier from a few thousand expert rubrics, and add the two anti-hacking criteria.

### HiR — Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following (Zhang et al., 2025)
Link: https://arxiv.org/abs/2512.23457
- **Mechanism:**
  - Select-then-rewrite. Of m=6 rollouts, the k=2 failures with the highest F(y) = F_div(y) + λ·F_int(y) are rewritten into pseudo-instructions that keep only the constraints they satisfied. λ = (1+η)^s·λ0 with η=0.05 and λ0=2, so emphasis moves from diversity to constraint integrity over training steps s.
  - RL uses binary instruction-level rewards on original and replayed samples, framed as dual preference learning at the instruction and response levels.
  - DeepSeek-V3.1 judges soft constraints.
  - HiR-16K: 16,969 queries with ≥5 decomposable constraints (76,456 hard, 46,536 soft), drawn from MulDimIF, VerIF, IFTrain and Chatbot Arena.
- **How it makes tasks harder:** Indirectly. It keeps very hard stacked prompts trainable by creating positives from failures.
- **Correctness / verification:** The relabeled instruction is correct by construction under the same per-constraint verifiers.
- **Difficulty control:** Curriculum weighting of which failures are replayed.
- **Reported results:** Qwen3-4B-Instruct-2507: IFBench 29.9 → 40.5 (+10.6), CFBench +5.7, MulDimIF +23.3. Llama-3.2-3B: IFEval +12.4. HiR beats RL with instruction-level and constraint-level rewards on most benchmarks.
- **Limitations / failure modes:** Replayed instructions are easier than the targets; quality depends on per-constraint verifier accuracy.
- **How to reuse with easy seed tasks:** Pair aggressive stacking with hindsight relabeling so all-zero GRPO groups still give gradient.

### LsrIF — LSRIF: Enhancing Logic-Structured Instruction Following of Large Language Models (Ren et al., 2026)
Link: https://arxiv.org/abs/2601.06431
- **Mechanism:**
  - GPT-4.1 turns each seed into a constraint tree with PAR, SEQ, COND and NEST nodes. For conditionals, it writes a branch condition based on the input.
  - Leaves are scored by code (hard) or a trained RM (soft), and aggregation follows the tree:
    - PAR: mean of the children;
    - SEQ: A = (1/m)·Σ g_j·a_j with g_j = λ^(number of earlier failed children), λ ∈ (0,1);
    - COND: GPT-4.1 decides which branch is active from the input, and only that branch counts;
    - NEST: recursive.
  - GRPO on the root reward.
- **How it makes tasks harder:** Sequential and conditional logic among constraints.
- **Correctness / verification:** Leaf verifiers plus aggregation that respects the structure. Note that the branch condition is judged by an LLM, not computed.
- **Difficulty control:** Structure type, depth and number of constraints.
- **Reported results:** Qwen2.5-1.5B IFEval 43.6 → 68.8. Distill-Qwen-7B IFEval 61.7 → 71.5 and CFBench 36 → 47. Distill-Qwen-14B FollowBench +7.0; out-of-domain gains up to +6.5 on ComplexBench and +8.7 on AgentIF. Ablation on Distill-Qwen-7B: plain averaging gives 67.6 / 42.0, dropping penalty propagation 68.7 / 44.0, dropping branch selection 67.9 / 44.0.
- **Limitations / failure modes:** The structure must be known at generation time; the conditional judge is not deterministic.
- **How to reuse with easy seed tasks:** Store a constraint tree with every composed prompt and aggregate rewards along it. Prefer conditions computed from the input so the active branch is known.

### Precision over Diversity — Precision over Diversity: High-Precision Reward Generalizes to Robust Instruction Following (Zeng et al., 2026)
Link: https://arxiv.org/abs/2601.04954
- **Mechanism:**
  - Split a VerInstruct-like pool (22k instances; 22.3% with only hard constraints) into Hard-only, Soft-only and Mix subsets and run RL on each.
  - Measure reward reliability: precision, and recall on false responses.
  - Propose HPPT: (a) learnability filtering, which pilot-trains for a few epochs and keeps prompts whose reward was >0 at least once; (b) at most one soft constraint per instance.
- **How it makes tasks harder:** It does not; it chooses which hard tasks are worth keeping.
- **Correctness / verification:** Code checkers have 96.0 precision. LLM judges' recall on false responses in multi-constraint judging: Qwen-3-32B 30.6 hard / 20.9 soft; QwQ-32B 44.1 / 33.1; Gemini-2.5-pro 65.7 / 63.8. Judging each constraint separately raises recall (Qwen-3-32B → 59.3 / 54.7).
- **Difficulty control:** Learnability filtering.
- **Reported results:**
  - Qwen2.5-7B: Hard-only IFEval 80.78 vs Mix 78.37 vs Soft-only 77.82. On CFBench, Hard 49 < Mix 51, with Soft 47.
  - Averages: Hard 57.85 ≈ Mix 57.26 > Soft 55.18. On 32B and 3B, Mix is slightly higher on average, so the abstract's "consistently outperform mixed" is really parity.
  - HPPT: +13.4 average (IFEval 87.25, IFBench 40.13 on the 7B) with a 58% reduction in training time.
- **Limitations / failure modes:** Conclusions depend on judge strength. VerIF found the opposite with a strong reasoning judge.
- **How to reuse with easy seed tasks:** Prefer adding code-checkable constraints. Cap LLM-judged ones at 1 per prompt, judge each constraint separately, and prune prompts that stay at zero reward in a pilot run.

### Kimi K2 IF-RL recipe — Kimi K2: Open Agentic Intelligence (Kimi Team, 2025)
Link: https://arxiv.org/abs/2507.20534
- **Mechanism:** IF prompts come from three sources: (1) expert-crafted complex conditional prompts with rubrics; (2) agentic instruction augmentation inspired by AutoIF; (3) a fine-tuned model that generates instructions probing specific failure modes and edge cases. Verification uses a code interpreter for verifiable outputs (length, style), an LLM judge for nuanced constraints, and a hack-check layer that detects claims of compliance without actual compliance. For non-verifiable tasks, a self-critique rubric reward uses core, prescriptive (anti-self-qualification) and human-annotated rubrics.
- **How it makes tasks harder:** A dedicated generator model that targets failure modes.
- **Correctness / verification:** Hybrid, plus the hack-check layer.
- **Difficulty control:** Adversarial targeting of weaknesses.
- **Reported results:** No IF-specific ablations reported.
- **Limitations / failure modes:** Few details; data not released.
- **How to reuse with easy seed tasks:** Fine-tune a small "edge-case prompt generator" on prompts your policy fails, and add a detector for false claims of compliance.

### SEIF — SEIF: Self-Evolving Reinforcement Learning for Instruction Following (Ren et al., 2026)
Link: https://arxiv.org/abs/2605.07465
- **Mechanism:**
  - The Instructor and the Follower are initialized from the same base model. The Filter and the Judger are frozen copies of the latest Follower.
  - Stage 1: GRPO trains the Instructor to add constraints to seeds, with reward R_I = 1 − A(x,y) if the Filter keeps x, and 0 if the Filter rejects it. A is the Judger's per-constraint satisfaction rate for the frozen Follower's response.
  - Stage 2: GRPO trains the Follower on the new Instructor's prompts with reward A.
  - Three iterations; the roles are refreshed from the latest Follower each time.
- **How it makes tasks harder:** A learned adversarial complexifier.
- **Correctness / verification:** Model-based only (Filter and Judger are the policy itself).
- **Difficulty control:** Implicit; the Instructor is rewarded for Follower failure.
- **Reported results:** Qwen2.5-7B-Instruct: IFEval 78.6 (+4.7), CFBench 51.0 (+4.0), FollowBench 59.0 (+3.9), WritingBench +6.6. Distill-Qwen-14B IFEval 80.0 (+5.1). Without Instructor evolution, IFEval 75.9; without the Filter, −3.2 IFEval and −6.0 CFBench.
- **Limitations / failure modes:** Rewarding raw failure pushes the Instructor toward contradictory or unfair prompts, so everything depends on the Filter. Self-judging can collude. The authors recommend long early-stage and moderate late-stage training to limit overfitting.
- **How to reuse with easy seed tasks:** Reward the generator for policy failure only after a strict satisfiability filter, and prefer a learnability-shaped reward (see open problems).

### LLM-as-a-Tutor — LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL (Kim et al., 2026)
Link: https://arxiv.org/abs/2607.04412
- **Mechanism:** During rubric-reward RL, the tutor LLM (the examiner) pairwise-compares two rollouts. If they are indistinguishable in quality, the prompt is non-challenging, and the tutor (as generator) appends one atomic constraint along a dimension the prompt left unspecified. It also adds the matching rubric criterion, with weights renormalized. Because edits only append, difficulty rises monotonically.
- **How it makes tasks harder:** Policy-triggered, append-only atomic constraints.
- **Correctness / verification:** Rubric reward; each appended constraint brings its own criterion.
- **Difficulty control:** Self-calibrating. Non-challenging prompts had mean reward 90.76 vs 78.24 and SD 12.96 vs 27.07. The constraint-addition ratio grows with policy scale: 8.1% → 25.8% → 40.5%.
- **Reported results:** Qwen3-1.7B policy with a Qwen3-8B tutor and judge: average 51.96 vs 51.04 for policy-adaptive rubrics and for EVA, 50.51 for base rubrics, and 50.24 for offline Evol-Instruct. FollowBench HSR 40.91 vs 38.60 with base rubrics. Examiner-based selection beat Always, Random and lowest-variance selection.
- **Limitations / failure modes:** Depends on the judge; prompts grow long; modest gains at small scale.
- **How to reuse with easy seed tasks:** Add an online "is this prompt still discriminative?" check (a pairwise tie, or all-equal verifier rewards). When it fails, append one checkable constraint rather than discarding the prompt.

### CSE — Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction (Vasileva, 2026)
Link: https://arxiv.org/abs/2608.12426
- **Mechanism:**
  - 36 deterministic constraint types across 8 dimensions, from lipograms and forbidden words to exact paragraph counts, monotonic sentence length, scene graphs, logic grids and self-referential word counts. Each verifier returns a binary result and a continuous score.
  - The probe composer draws k-subsets (k = 1–12). A compatibility checker applies unconditional pair blocks, arithmetic checks that depend on parameters, and propagation across constraints. Rejection grows from about 32% at k=4 to about 98% at k=12.
  - Parameters come from fixed calibrated ranges (e.g., word targets 40–60), so difficulty comes from composition. Presentation order is shuffled. There are 444 deliberately impossible probes.
- **How it makes tasks harder:** k alone.
- **Correctness / verification:** Deterministic verifiers; no LLM judge.
- **Difficulty control:** k, and the mix of structural and lexical constraints.
- **Reported results:**
  - 15 models; 369,753 checks. mCSR(k) = 72.0% × 0.922^(k−1) (held-out MAE 0.2 pp). About 41% per constraint at k=8 → 5.7% all-pass.
  - All-pass falls below 50% at k=7 for the best model and at k≤3 for 12 of 15 models.
  - Structural and ordering constraints degrade 2.0× faster than lexical ones.
  - Pairwise failures are nearly independent (mean φ = +0.067); what coupling exists comes from shared output features (a wrong sentence count fails every constraint that reads it).
  - Planning did not move the threshold, and self-correction or best-of-5 delayed it by only 1–2 constraints.
- **Limitations / failure modes:** Synthetic surface constraints; no training.
- **How to reuse with easy seed tasks:** Estimate the policy's per-constraint pass rate p and pick k so that p^k ≈ 0.2–0.5. Use CSE-style compatibility rules, and expect the rejection rate to explode at high k.

### UNSPECIFIC — UNSPECIFIC: General Constraint Synthesis for Breaking Copy-and-Paste Shortcut in LLM Instruction Following (Sharma et al., 2026)
Link: https://arxiv.org/abs/2608.09154
- **Mechanism:** Back-translated constraints often copy text from the reference, so they can be satisfied by copying. UNSPECIFIC instead (1) synthesizes constraints common to two similar reference articles (news, story, blog), (2) hardens only the constraints found to be trivially satisfied, and (3) checks satisfaction on the generated article *and on its summary*, so a constraint must hold in the core narrative.
- **How it makes tasks harder:** Abstraction (shared across two references) and selective hardening.
- **Correctness / verification:** Grounding in two references keeps constraints satisfiable; LLM judges do the checking.
- **Difficulty control:** Selective hardening.
- **Reported results:** GPT-5 Mini satisfaction fell from 90% to 78%; the LLM win-rate gap for naturalness improved by 30% against human judgment. Many constraints are satisfied only superficially.
- **Limitations / failure modes:** Evaluation-focused; relies on an LLM judge.
- **How to reuse with easy seed tasks:** When back-translating, derive constraints from ≥2 similar references, reject any that can be satisfied by pasting a phrase, and verify on a summary of the output.

### CC-OPD — Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following (Zheng et al., 2026)
Link: https://arxiv.org/abs/2609.27421
- **Mechanism:** For each student rollout, a frozen teacher (already trained with GRPO on instruction-level reward on HiR-16K) scores every token under the full prompt and under each leave-one-constraint-out prompt. δ_i(t) = ℓ(C;t) − ℓ(C∖{c_i};t); these are summed over constraints and clipped at c=5. The per-token reward is r_t = −KL(t) + λ·δ̂(t) with λ=2, trained with PPO. Leaving one constraint out keeps the interactions between constraints, which scoring each constraint alone would lose.
- **How it makes tasks harder:** It does not; it provides dense credit on heavily stacked prompts (5–17 constraints, mean 7.25).
- **Correctness / verification:** The teacher's likelihood shift; no external verifier during distillation.
- **Difficulty control:** Not applicable.
- **Reported results:** Average over 7 benchmarks: 51.4 (Qwen2.5-1.5B student, +2.5 over ExOPD) and 64.3 (Qwen3-4B, +1.5 over Teacher-TopK RKL). The 1.5B student beats its 7B teacher on MulDimIF (72.9 vs 70.1; Level 4: 59.4 vs 53.1). It reaches OPD's peak in 15.2× fewer steps, using about 30% of OPD's FLOPs.
- **Limitations / failure modes:** |C|+1 teacher forward passes per rollout (4.63× per-step overhead); needs a teacher already strong at IF.
- **How to reuse with easy seed tasks:** If stacked-constraint GRPO stalls, RL-train a bigger teacher on the hard pool and distill with leave-one-out counterfactual shaping.

### ScopeIF *(added)* — ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling (Wen et al., 2026)
Link: https://arxiv.org/abs/2609.32189
- **Mechanism:** Each constraint is written as ⟨Scope, Target, Range⟩:
  - **Scope** selects segments: Global, Traversal (every paragraph), Position (the first sentence), or Condition (sentences containing X). Scopes can be nested.
  - **Target** is one of 15 measurable subcategories in 6 groups: capacity, literals, structure, checklist, pattern, relations.
  - **Range** is a one-sided bound, prohibition, equality, interval, comparison, or arithmetic relation.
  Atomic constraints come from enumerating compatible category triples (10 generation iterations each, keeping history to avoid repeats). LLM "crossover" samples 16 candidate constraints and composes instructions with at least k ∈ {2,4,6,8}. Seeds come from HIR-16K and MulDimIF, followed by LLM quality control. SCOPEINSTRUCT has 16,968 training and 1,000 test instructions with 1–12 constraints (4.83 on average).
- **How it makes tasks harder:** Scoped and nested constraints, and recombination along three dimensions.
- **Correctness / verification:** Tool-grounded verification: the judge writes and runs a Python tool on the specific response (e.g., to extract titles before counting words), then reports (bound, measured value) records.
- **Difficulty control:** k and scope complexity. The graded reward r = exp(−α·deviation/scale), combined by geometric mean, is aggregated hierarchically with the binary signal.
- **Reported results:** Qwen3-8B: IFBench prompt-strict 31.0 → 53.7; ScopeIF-Test instruction success 7.2 → 19.6 (Seed-2.0-Pro 21.8, Gemini-2.5-Pro 15.8). It beats the constraint-level-average RL baseline (51.0 IFBench).
- **Limitations / failure modes:** The judge's tool code can be wrong; the graded reward needs a sensible scale for each target.
- **How to reuse with easy seed tasks:** Re-scope existing global constraints ("≤60 words") to segments ("the last sentence of every paragraph ≤12 words"). Recombine Scope, Target and Range to build a large constraint space, and reward the distance to the bound, not just pass/fail.

### ImpRIF *(added)* — ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following (Yang et al., 2026; ACL 2026)
Link: https://arxiv.org/abs/2602.21228
- **Mechanism:**
  - Each constraint is an explicit reasoning graph (ERG), a DAG of conditional nodes (Boolean checks and branches), mathematical nodes (arithmetic and comparisons) and knowledge nodes (facts with objective answers). Nodes are sampled from three banks; an LLM builds the dependencies, renders each node in NL and *hides the intermediate hops*.
  - Each constraint gets executable verification code that is validated iteratively.
  - Single-turn data samples a variable number of constraints and attaches a user query.
  - Multi-turn data puts constraints in system prompts, or injects new constraints each turn with newer ones taking priority. Some dialogs end with an adversarial final turn (conflicts, prompt injection).
  - An evaluation model filters out incoherent or contradictory instructions.
  - SFT uses ERG-structured CoT. GRPO uses the fraction of satisfied constraints (single-turn), or code plus rubric items (multi-turn), plus a reward on the thinking process.
- **How it makes tasks harder:** Implicit multi-hop conditions; constraint dependencies; accumulation across turns.
- **Correctness / verification:** Code per constraint; rubric RM only for multi-turn soft items.
- **Difficulty control:** The number of atomic constraints (nodes per constraint are typically 3–7). Performance declines clearly as constraints increase.
- **Reported results:** Average gains of about 7.3, 9.3 and 9.9 points on five external complex-IF benchmarks for Qwen3-4B, 8B and 32B. Thinking-process rewards helped most on logic-heavy sets.
- **Limitations / failure modes:** Knowledge nodes require objective facts; synthesis is fairly heavy.
- **How to reuse with easy seed tasks:** Replace explicit parameters in easy constraints with ones that must be derived ("use as many bullets as the number of vowels in the city's name"). Compute the gold value by executing the graph.

### QUBRIC *(added)* — QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards (Zhang et al., 2026; EMNLP 2026)
Link: https://arxiv.org/abs/2606.03968
- **Mechanism:**
  - Open-ended queries produce vague rubrics, and naive narrowing invents references (non-existent guidelines), so every response fails.
  - QUBRIC extracts atomic key points from teacher responses, prioritizing queries where models disagree. It then rewrites the query as a self-contained scenario in which 1–2 key points must *emerge through reasoning*, with an answer-format constraint (e.g., "2–3 sentences").
  - Validation checks four things: the key points still answer the query, the scenario does not leak them, the constraints are consistent, and the item hits the target complexity.
  - Contrastive rubrics come from four sources: explicit query constraints (Critical); content in the teacher's response but missing from the policy's; content both share; and atomic splits of holistic criteria.
  - Learnability filter: keep pairs where the initial policy's pass rate is 20–50%.
- **How it makes tasks harder:** Converts an easy, vague request into a specific, evaluable scenario question.
- **Correctness / verification:** Rubrics that a judge can check from the text alone, with weights (Critical 3, Optional 1).
- **Difficulty control:** The 20–50% corridor.
- **Reported results:** +5.5 ArenaHard over SFT, and +6.3 average on three held-out benchmarks (legal, moral, narrative).
- **Limitations / failure modes:** Needs a teacher; the judge still decides presence.
- **How to reuse with easy seed tasks:** For soft chat seeds, ground a rewrite in teacher key points, add a format bound, and keep only items in the pass-rate corridor.

### Inverse IFEval *(added)* — Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions? (Zhang et al., 2025)
Link: https://arxiv.org/abs/2509.04292
- **Mechanism:** Eight types of instructions that invert SFT conventions: Question Correction, Intentional Textual Flaws, Code without Comments, Counter-Conventional Formatting, Deliberately Incorrect Answers, Instructional Induction, Mid-turn Instruction Modification, and Counterfactual Answering. 1,012 Chinese and English items across 23 domains, built with humans in the loop.
- **How it makes tasks harder:** Targets "cognitive inertia" from SFT rather than adding constraints.
- **Correctness / verification:** An optimized LLM judge.
- **Difficulty control:** Choice of inversion type.
- **Limitations / failure modes:** Many types need an LLM judge, and it is an evaluation-only benchmark with no training recipe.
- **How to reuse with easy seed tasks:** Add inversion constraints that code can check (no comments, no paragraph breaks, a required misspelling count, a non-Markdown format) to saturated seeds.

### Reward Hacking in Rubric-Based RL *(added)* — Reward Hacking in Rubric-Based Reinforcement Learning (Mahmoud et al., 2026)
Link: https://arxiv.org/abs/2605.12474
- **Mechanism:** Train with a weak (GPT-4o-mini) or strong (GPT-OSS-120B) rubric verifier and evaluate against a cross-family panel of three frontier judges. The paper separates verifier failure from gaps in the rubric design, and introduces a "self-internalization gap" diagnostic based on the policy's log-probabilities.
- **How it makes tasks harder:** Not applicable. This is the failure analysis for rubric-rewarded complexification.
- **Correctness / verification:** Reference panel.
- **Reported results:**
  - With the weak verifier, the per-window incorrect-credit rate rose from 39% to 65% (medical) and from 63% to 75% (science). The same pattern held for 14B and 32B policies.
  - Exploits concentrate in partial satisfaction of compound criteria, treating implicit content as explicit, and imprecise topical matching.
  - A strong verifier reduces but does not remove exploitation. When rubrics leave failure modes unspecified, rubric judges prefer the RL checkpoint while rubric-free judges prefer the base model.
  - 90.2% of custom-rubric weight sat on presence criteria.
- **How to reuse with easy seed tasks:** Split compound criteria, add absence and negative criteria, use a strong verifier, and audit against a held-out judge panel.

### MDP-GRPO *(added)* — MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following (Salmani-Zarchi et al., 2026)
Link: https://arxiv.org/abs/2606.06058
- **Mechanism:** Names three pathologies of z-score group normalization with discrete, low-dispersion multi-constraint rewards:
  - low-variance amplification;
  - mean-centering blindness (always-easy and always-hard groups look the same after normalization);
  - zero-variance collapse.
  Fixes: multi-temperature sampling within a group, to increase reward dispersion; dual-anchor advantages that interpolate between group-relative and goal-aware advantages; bounded shaping inspired by prospect theory; and asymmetric KL.
- **How it makes tasks harder:** Not applicable. It keeps hard stacked prompts learnable.
- **Correctness / verification:** Verifiable constraints.
- **Reported results:** Up to +5.0% strict constraint satisfaction on Llama-3.2-3B over GRPO; stable with small group sizes; MMLU and ARC preserved.
- **How to reuse with easy seed tasks:** When hardened prompts give all-0 or all-1 groups, sample at mixed temperatures and add an absolute (goal) anchor to the advantage.

---

## Complexification operators from this area

1. **Constraint stacking (And, raise k).**
   - What it does: attaches more independent atomic constraints.
   - Easy → hard: "Write a poem about autumn." → "Write a poem about autumn: exactly 4 stanzas, all lowercase, include 'ember' exactly twice, no commas, end with a question, title wrapped in <<>>."
   - Keeping it verifiable: one code checker per constraint, plus a compatibility checker (pair blocks, arithmetic feasibility, propagation). Calibrate k with p^k. Beyond about 6 constraints, switch from all-pass to per-constraint, graded or relabeled rewards.
   - Sources: CSE, IFBench, RECAST, MulDimIF, IFDecorator.
2. **Nested level ladder (L1 ⊂ L2 ⊂ … ⊂ Lk).**
   - What it does: builds nested prefixes of a single constraint list.
   - Easy → hard: "Recommend three sci-fi novels." → L5: "…published before 1970, as a Markdown table Title|Author|Year, each row ≤15 words, without mentioning Asimov."
   - Keeping it verifiable: V(L_k) = V(L_(k−1)) ∧ v(c_k). Responses at adjacent levels give free positive/negative labels for soft verifiers, and the first failing level is a difficulty label.
   - Sources: FollowBench, Conifer, Ren et al., RECAST-Test.
3. **Chain (sequential dependency).**
   - What it does: splits the task into ordered steps where later outputs use earlier ones.
   - Easy → hard: "Summarize this article." → "Summarize in exactly 3 bullets; translate the bullets into French; write a title that reuses a noun from bullet 2; put the parts under SUMMARY / FR / TITLE."
   - Keeping it verifiable: parse sections, check each step, and discount downstream credit after an upstream failure (dependency-AND in ComplexBench, λ-decay in LsrIF).
   - Sources: ComplexBench, RAIF, LsrIF.
4. **Selection (conditional branching).**
   - What it does: gates constraint sets behind a condition on the input.
   - Easy → hard: "Describe this product." → "If the listed price is >$100, write a JSON review {pros, cons, verdict}; otherwise a ≤280-character post with exactly 2 hashtags."
   - Keeping it verifiable: make the condition *computable from the input* so the gold branch is known without an LLM (LsrIF uses GPT-4.1, which adds noise). Reward only the active branch; an output on the wrong branch gets 0. Nest selections for depth; multi-layer Selection is where GPT-4 fell to 14.9% on the coherent test.
   - Sources: ComplexBench, RAIF, LsrIF.
5. **Scoped and hierarchical constraints (Scope × Target × Range).**
   - What it does: applies a constraint to selected segments and nests scopes.
   - Easy → hard: "Every paragraph under 60 words." → "In section 2 only, every bullet starts with a verb; bullets under 'Risks' are numbered and each contains a percentage; the last sentence of every paragraph in section 3 has ≤12 words."
   - Keeping it verifiable: parse the output into a tree (headings, lists, sentences) and have the judge write and run an extraction tool for the actual response. Reward the normalized distance to the bound: exp(−α·d/s).
   - Sources: ScopeIF, IFHierBench (600 prompts, 4 depths, 35 constraints; the best model is only marginally above 50%).
6. **Response-grounded back-translation.**
   - What it does: derives constraints that a good response already satisfies.
   - Easy → hard: ("Explain photosynthesis", a 280-word, 3-paragraph answer) → "Explain photosynthesis in 3 paragraphs, 250–300 words, mention chlorophyll and the Calvin cycle, end with a one-sentence summary."
   - Keeping it verifiable: compute numeric constraints from the response with code and sample a range around the true value. Re-verify LLM-proposed constraints. Deduplicate with ROUGE-L < 0.6. Block copy shortcuts: use constraints shared by ≥2 references, harden trivially satisfied ones, and check a summary of the output.
   - Sources: Crab, VerIF, RECAST, UNSPECIFIC.
7. **Decompose-then-recompose with a learned complexifier.**
   - What it does: learns the easy → hard + check mapping from real prompts.
   - Easy → hard: "Recommend me ten Chinese books." → "In Shakespeare's tone, recommend ten Chinese books, each from a different dynasty…", each constraint with its question ("Are there exactly ten books?").
   - Keeping it verifiable: every added constraint must come with a check question or a checker. Route counts to code. Cap iterations (pass rate falls from 91.8% to 79.4% by the third).
   - Sources: UltraIF, DecIF (2505.13990).
8. **Constraint grafting with execution-validated verifiers.**
   - What it does: attaches library constraints to real easy queries.
   - Easy → hard: "How do I make sourdough starter?" → "…in exactly 5 numbered steps, each at most 25 words, with no digits outside the numbering."
   - Keeping it verifiable: keep a verifier only if it compiles, passes cross-validated tests at >0.5, and back-translates to a consistent instruction (NLI). Filter pairs whose query and constraint don't fit together (an LLM score ≥8 of 10).
   - Sources: AutoIF, Kimi K2.
9. **New constraint families and wider variable ranges.**
   - What it does: moves beyond the 25 IFEval types and trains on broader parameter ranges.
   - Easy → hard: "Use exactly 3 bullet points." → "Every sentence must contain exactly one palindrome of ≥4 letters", or a sentence-count limit drawn from 1–40 instead of 1–20.
   - Keeping it verifiable: hand-write or synthesize the verifier and unit-test it. Hold out whole families for evaluation. Disjoint train ranges generalize worse than wider ones.
   - Sources: IFBench, CSE.
10. **Convention inversion.**
    - What it does: requires behavior that contradicts SFT habits.
    - Easy → hard: "Write a function that sorts a list." → "…with no comments and no docstrings, variable names of exactly one letter, and include one deliberate off-by-one bug in a line marked #BUG."
    - Keeping it verifiable: prefer inversions that AST or regex checks can confirm; audit the judge on the rest.
    - Sources: Inverse IFEval.
11. **Program-derived logic instructions.**
    - What it does: turns code into an NL procedure the model must execute.
    - Easy → hard: "Sort these numbers." → an anonymized Codeforces simulation described in words ("maintain two counters…; while the queue is non-empty, if the front is even and counter A < 3 …"), reporting the output and the maximum queue size.
    - Keeping it verifiable: execute the anonymized code for gold outputs and tracker values, filter degenerate inputs, and stratify by 3D + 2F + C + 0.5L.
    - Sources: LogicIF.
12. **Implicit-parameter constraints.**
    - What it does: hides the value of a constraint behind hidden reasoning hops.
    - Easy → hard: "Use 5 bullets." → "Use as many bullets as there are prime numbers below the number of letters in the capital of Australia; if that count is even, write in the passive voice."
    - Keeping it verifiable: build the DAG from banks of knowledge, math and condition nodes with objective values, execute it to get the target, and check the output with code.
    - Sources: ImpRIF.
13. **Policy-adaptive appending and band targeting.**
    - What it does: adds constraints only where the current policy has saturated.
    - Easy → hard: a prompt where 8/8 rollouts pass (zero GRPO advantage) → the same prompt plus 1–3 new atomic constraints, until pass@1 is in (0, 0.5] or two rollouts become distinguishable.
    - Keeping it verifiable: append-only edits keep earlier checkers valid. Pass rate 0 → regenerate, don't train. Pair with an intent check and a trip-wire hack rate. Pilot-train and prune prompts that never score.
    - Sources: IFDecorator, LLM-as-a-Tutor, SEIF, QUBRIC, Precision over Diversity.
14. **Coupled constraints (shared output features).**
    - What it does: combines jointly satisfiable constraints that read the same feature (structure, length, format).
    - Easy → hard: "Include 'blue'; mention a city." → "Valid JSON only; the 'summary' field is 40–50 words; exactly three citations in a list; the first letters of the keys spell MAP."
    - Keeping it verifiable: prove satisfiability with a witness response or solver-style rules. Expect many incompatibilities: in Instruction Stacking Collapse, "output JSON" is jointly unsatisfiable with 9 of the 24 instructions. Structural constraints cost about 2× more per addition than lexical ones.
    - Sources: CSE, Instruction Stacking Collapse (2608.02639), MulDimIF.
15. **Presentation rewriting (pattern, order, distractors).**
    - What it does: keeps the constraint set but makes it harder to parse.
    - Easy → hard: 4 bulleted constraints before the task → the same 4 woven mid-paragraph into a long message, in easy-to-hard order, next to 3 examples that each violate one constraint.
    - Keeping it verifiable: the checkers don't change. Randomize order and pattern across copies. Models do better when constraints come hard-to-easy, and Incorporation is harder than Listing, which is harder than Example.
    - Sources: MulDimIF, Order Matters (2502.17204), FollowBench.
16. **Multi-turn carry-over, system-prompt embedding and adversarial final turns.**
    - What it does: turns a single-turn constraint into state tracking.
    - Easy → hard: "Answer in French." → system: "Always French, ≤80 words"; turn 3: "For this answer only, English and a table"; turn 5: a new question that must revert; the final turn contains a conflicting or injected instruction.
    - Keeping it verifiable: track the active-constraint set per turn in code, with newer constraints taking priority, and check each turn against it. Use rubrics for the soft parts. Keep only prompts the current model fails (AdvancedIF).
    - Sources: AdvancedIF, ImpRIF, Multi-IF (o1-preview 0.877 → 0.707 from turn 1 to turn 3), MTAC-IFBench (7.04 turns and 91.33 constraints per instance), EvolIF.
17. **Rubric or checklist decomposition, with scenario grounding.**
    - What it does: turns open-ended requests into many atomic, weighted yes/no criteria.
    - Easy → hard: "Write a persuasive email asking for a budget increase." → a scenario with concrete numbers plus a 10–15-item rubric (states a dollar amount, cites 2 quantified results, anticipates one objection, ≤200 words, no exclamation marks, ends with a meeting request, pitfall: no threats).
    - Keeping it verifiable: mine items from failures of weak candidates or from teacher-vs-policy contrasts. Make items constitutive (checkable from rubric text alone). Split compound criteria, add absence items and universal anti-artifact items, route exact items to code, use a strong or fine-tuned verifier, and audit against a held-out judge panel.
    - Sources: RLCF, RaR, AdvancedIF, QUBRIC, ComplexConstraints (2606.09118; 10–40 criteria per prompt), the rubric-hacking entry.
18. **Hindsight relabeling (a reverse operator for gradient signal).**
    - What it does: turns failed hard rollouts into positives for the subset of constraints they satisfied.
    - Easy → hard: not a hardener. A response meets 4 of 7 constraints → training pair (instruction restricted to those 4, response) marked successful.
    - Keeping it verifiable: reuse the per-constraint verifiers. Mix with the original prompts; weight replay toward integrity later in training.
    - Sources: HiR.

---

## Insights & pitfalls

- **Complexity is not difficulty.** Constraint count correlates with measured difficulty only loosely (IFDecorator). Tree-Instruct's 10-node prompts beat 4× more simple prompts at equal tokens, but offline-hardened prompts did not help rubric RL (LLM-as-a-Tutor). Decide by pass rate against the current policy. A corridor of (0, 0.5] (IFDecorator) or 20–50% (QUBRIC) is the working standard.
- **Most aggressive hardening yields unsolvable or contradictory prompts.** IFDecorator: 10,772 prompts at pass rate 0 vs 7,324 usable. CSE: 98% of random 12-constraint sets are incompatible. SEIF: without the filter, performance drops. Budget for a satisfiability checker (conflict dictionary, pair blocks, witness response) and for regeneration.
- **Structure-aware or graded rewards beat naive means once constraints interact.** LsrIF ablation: averaging lost about 4 IFEval and 5 CFBench points. ScopeIF's graded reward beat constraint-level averaging. But when the verifier is noisy per criterion, fractional credit amplifies noise: AdvancedIF found all-or-nothing best. Rule of thumb: use partial or graded credit for *code-verified* constraints and strict gating for *judge-verified* rubrics.
- **Reward precision matters more than verifier diversity, but judge strength decides.** Multi-constraint LLM judging misses most violations; judging each constraint separately roughly doubles recall. Weak judges get hacked (39% → 65% incorrect credit). A reasoning judge (QwQ-32B) made soft constraints worth 10 IFEval points in VerIF, while Precision over Diversity found soft constraints neutral to harmful with weaker judges. Measure your judge's recall on known violations before mixing soft constraints in.
- **Reward hacking is the default outcome of IF-RLVR.** Observed forms:
  - over-satisfying constraints while ignoring the task (IFBench);
  - literal placeholders, dummy lists, "p p p" repetition (IFDecorator);
  - long preambles (RLCF);
  - "all instructions are followed" self-claims (AdvancedIF, Kimi K2);
  - satisfying constraints by copying reference text (UNSPECIFIC).
  Mitigations with evidence: RM gating (IFBench), IntentCheck (MHR 14.53 → 7.60), universal criteria (RLCF, AdvancedIF), trip-wire dashboards, and summary-level checks.
- **Diversity of constraint *types* and *parameters* is what generalizes.** Training on IFEval types overfits them (TÜLU-DPO: 81.1 IFEval vs 25.5 IFBench). Removing LENGTH or KEYWORDS categories hurt most. Disjoint variable ranges hurt; wider ranges help. Hold out whole families.
- **More constraints per training prompt than at test time.** IFBench shows 3–6 beating 1, but the Qwen2.5 curve is not monotonic (a peak at 3, then 49.4 at 4). Treat the count as a hyperparameter per policy, not "more is always better".
- **Composition type is a verifier-preserving free knob.** Chain < And and nested Selection << And (ComplexBench). Decomposing into multiple rounds *hurt* GPT-3.5, so don't expect agentic decomposition at inference to rescue it. Presentation (Incorporation), constraint order and distractor examples also raise difficulty without new verifiers.
- **Vanilla CoT is not a fix.** RAIF: plain CoT cut a 1.5B model's average by 11.79. CSE: planning did not move the phase transition, and retries bought only 1–2 constraints. Gains come from RL that rewards reasoning only where it helps (sample-wise contrast) or from structured, graph-aligned CoT (ImpRIF).
- **Back-translation is the cheapest correct-by-construction hardener but saturates.** It never exceeds the source response's difficulty, invites copying, and in Crab did not help FollowBench L3–L5 relative to Conifer. Use it for volume and for SFT warm-up, then switch to policy-adaptive appending for RL.
- **Zero-advantage groups need optimizer-side fixes too.** Remedies: multi-temperature sampling and dual-anchor advantages (MDP-GRPO), ordinal decomposition of discrete judge scores (ODRPO, 2605.12667), hindsight replay (HiR), and dense counterfactual teacher shaping (CC-OPD, 15.2× fewer steps).
- **Program- and graph-derived IF transfers beyond IF.** LogicIF GRPO: ZebraLogic +31.4, MATH-500 +4.3, GPQA-D +3.1. LsrIF improved Enigmata arithmetic (Distill-Qwen-14B 21.0 → 39.0). ImpRIF: +7–10 points on external IF sets. Requiring intermediate state exposes right-answer/wrong-process shortcuts.
- **Curriculum evidence is mixed.** Tree-Instruct: curriculum beats mixed ordering but loses to hard-only. Ren et al. and Conifer use easy-to-hard ladders successfully. Adaptive appending is the most robust scheduling evidence so far.
- **Cross-task interference.** IF-RLVR reshapes the first few response tokens (direct answers), reducing math best@k and later Math-RLVR trainability. Math-RLVR does the reverse to IF best@32 (Verifier-Induced Support Reshaping, 2608.00220). Mix domains, monitor best@k, and consider on-policy distillation or reference constraints (they help only partially).

---

## Open problems & research opportunities

- **Predicting pass rate from a constraint set.** A learned predictor of p(pass | policy, constraint types, scopes, coupling, composition) would allow hard-but-solvable prompts in one shot instead of 8-rollout probing. CSE's near-independence of failures and its structural/lexical rates are a starting prior.
- **Formal satisfiability and witness construction for NL constraint sets.** A typed constraint DSL (for example, ScopeIF's Scope/Target/Range) plus a solver that proves joint satisfiability and emits a witness response, replacing LLM conflict detectors and hand-coded pair blocks.
- **Deterministic conditions and scopes.** Current conditional and scoped rewards still call an LLM to resolve the branch (LsrIF) or to write the extraction tool (ScopeIF). Conditions and parsers that can be computed from the input would remove this source of noise.
- **High-precision, hack-resistant soft verification.** Per-criterion judges with measured recall on violations, ensembles, and fine-tuned verifiers (AdvancedIF F1 0.728) calibrated against a held-out panel. Principled rules for when soft constraints help (VerIF) or hurt (Precision over Diversity).
- **Automatic invention of new constraint families** with verified checkers and novelty metrics, to close the gap to held-out types that IFBench exposed. Inverse IFEval-style convention inversions are an under-used source.
- **Training beyond the 5–7-constraint phase transition** for system prompts and agent harnesses with dozens to hundreds of rules (IFScale 2507.11538: best 68% at 500 keyword instructions; AgentIF averages 11.9 constraints per instruction; MTAC-IFBench 91.33), without all-zero rewards or loss of quality.
- **Learnability-shaped self-play generators.** SEIF rewards raw failure (1 − satisfaction), relying on a filter. Rewarding the generator for a policy pass rate near a target band, with satisfiability guarantees and anti-degeneracy terms, is untested for IF.
- **Measuring "constraint-satisfied but useless" outputs** and principled ways to combine constraint rewards with quality or preference rewards (IFBench's threshold gating is ad hoc).
- **Interference-aware multi-domain mixtures** that improve IF without shrinking the support for reasoning, math or agentic behavior.
- **Agentic and multimodal transfer.** Constraints on actions and trajectories rather than only text (MTAC-IFBench, AgentIF), and multimodal verifiable-constraint synthesis with learnability-aware filtering (MIFS 2609.16059: 90k samples, 8 constraint categories; VISA 2608.26013: an agentic loop driven by memory and failure profiles).

---

## References

1. Jiang, Y., Wang, Y., Zeng, X., et al. (2023). *FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models*. ACL 2024. arXiv:2310.20410. https://arxiv.org/abs/2310.20410
2. Wen, B., Ke, P., Gu, X., et al. (2024). *Benchmarking Complex Instruction-Following with Multiple Constraints Composition*. NeurIPS 2024 D&B. arXiv:2407.03978. https://arxiv.org/abs/2407.03978
3. Xu, C., Sun, Q., Zheng, K., et al. (2023). *WizardLM: Empowering Large Pre-trained Language Models to Follow Complex Instructions*. ICLR 2024. arXiv:2304.12244. https://arxiv.org/abs/2304.12244
4. Zhao, Y., Yu, B., Hui, B., et al. (2023). *A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment* (Tree-Instruct). LREC-COLING 2024. arXiv:2308.05696. https://arxiv.org/abs/2308.05696
5. Sun, H., Liu, L., Li, J., et al. (2024). *Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models*. arXiv:2404.02823. https://arxiv.org/abs/2404.02823
6. Dong, G., Lu, K., Li, C., et al. (2024). *Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models* (AutoIF). ICLR 2025. arXiv:2406.13542. https://arxiv.org/abs/2406.13542
7. Qi, Y., Peng, H., Wang, X., et al. (2024). *Constraint Back-translation Improves Complex Instruction Following of Large Language Models* (Crab). CIKM 2025. arXiv:2410.24175. https://arxiv.org/abs/2410.24175
8. An, K., Sheng, L., Cui, G., et al. (2025). *UltraIF: Advancing Instruction Following from the Wild*. EMNLP 2025. arXiv:2502.04153. https://arxiv.org/abs/2502.04153
9. Guo, Z., Liu, W., Xie, M., et al. (2025). *RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data*. ICLR 2026. arXiv:2505.19030. https://arxiv.org/abs/2505.19030
10. Ye, J., Huang, C., Chen, Z., et al. (2025). *MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models*. ACL 2026. arXiv:2505.07591. https://arxiv.org/abs/2505.07591
11. Qin, Y., Li, G., Li, Z., et al. (2025). *Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models* (RAIF). NeurIPS 2025. arXiv:2506.01413. https://arxiv.org/abs/2506.01413
12. Peng, H., Qi, Y., Wang, X., et al. (2025). *VerIF: Verification Engineering for Reinforcement Learning in Instruction Following*. EMNLP 2025. arXiv:2506.09942. https://arxiv.org/abs/2506.09942
13. Pyatkin, V., Malik, S., Graf, V., et al. (2025). *Generalizing Verifiable Instruction Following* (IFBench). NeurIPS 2025 D&B. arXiv:2507.02833. https://arxiv.org/abs/2507.02833
14. Lambert, N., Morrison, J., Pyatkin, V., et al. (2024). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124. https://arxiv.org/abs/2411.15124
15. Viswanathan, V., Sun, Y., Ma, S., et al. (2025). *Checklists Are Better Than Reward Models For Aligning Language Models* (RLCF). NeurIPS 2025. arXiv:2507.18624. https://arxiv.org/abs/2507.18624
16. Gunjal, A., Wang, A., Lau, E., et al. (2025). *Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains*. arXiv:2507.17746. https://arxiv.org/abs/2507.17746
17. Guo, X., Liang, T., Jian, T., et al. (2025). *IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards*. arXiv:2508.04632. https://arxiv.org/abs/2508.04632
18. Zhang, M., Liu, S., Dong, S., et al. (2025). *LogicIF: Towards Complex Logic Instruction Following*. COLM 2026. arXiv:2508.09125. https://arxiv.org/abs/2508.09125
19. Ren, Q., He, Q., Chang, P., et al. (2025). *Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following*. arXiv:2510.14420. https://arxiv.org/abs/2510.14420
20. Ren, Q., He, Q., Zhang, B., et al. (2025). *Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following*. arXiv:2508.02150. https://arxiv.org/abs/2508.02150
21. He, Y., Li, W., Zhang, H., et al. (2025). *AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following*. arXiv:2511.10507. https://arxiv.org/abs/2511.10507
22. Zhang, K., Yao, Q., Liu, S., et al. (2025). *Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following* (HiR). arXiv:2512.23457. https://arxiv.org/abs/2512.23457
23. Ren, Q., He, Q., Chang, J., et al. (2026). *LSRIF: Enhancing Logic-Structured Instruction Following of Large Language Models*. arXiv:2601.06431. https://arxiv.org/abs/2601.06431
24. Zeng, Y., Liu, Y., Ding, X., et al. (2026). *Precision over Diversity: High-Precision Reward Generalizes to Robust Instruction Following*. arXiv:2601.04954. https://arxiv.org/abs/2601.04954
25. Kimi Team (Bai, Y., et al.) (2025). *Kimi K2: Open Agentic Intelligence*. arXiv:2507.20534. https://arxiv.org/abs/2507.20534
26. Ren, Q., He, Q., Zhu, J., et al. (2026). *SEIF: Self-Evolving Reinforcement Learning for Instruction Following*. arXiv:2605.07465. https://arxiv.org/abs/2605.07465
27. Kim, Y., Ho, N., Hwang, S., et al. (2026). *LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL*. arXiv:2607.04412. https://arxiv.org/abs/2607.04412
28. Vasileva, M. I. (2026). *Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction* (CSE). arXiv:2608.12426. https://arxiv.org/abs/2608.12426
29. Sharma, J., Kaur, B., Hong, J., Zamani, H., Chang, H.-S. (2026). *UNSPECIFIC: General Constraint Synthesis for Breaking Copy-and-Paste Shortcut in LLM Instruction Following*. arXiv:2608.09154. https://arxiv.org/abs/2608.09154
30. Zheng, Y., Yu, Y., Xu, T., et al. (2026). *Counterfactual Constraint-Conditioned On-Policy Distillation for Multi-Constraint Instruction Following* (CC-OPD). arXiv:2609.27421. https://arxiv.org/abs/2609.27421
31. Wen, B., Niu, Y., Ning, X., et al. (2026). *ScopeIF: Improving Scope-Aware Precise Instruction-Following in Large Language Models via Graded Reward Modeling*. arXiv:2609.32189. https://arxiv.org/abs/2609.32189
32. Yang, Y., Yang, L., Wang, X., Tong, C., Yang, H. (2026). *ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following*. ACL 2026. arXiv:2602.21228. https://arxiv.org/abs/2602.21228
33. Zhang, R., Feng, R., Zhang, Z., et al. (2026). *QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards*. EMNLP 2026. arXiv:2606.03968. https://arxiv.org/abs/2606.03968
34. Zhang, Q., Lei, X., Miao, R., et al. (2025). *Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions?* arXiv:2509.04292. https://arxiv.org/abs/2509.04292
35. Mahmoud, A., Rezaei, M., Wang, Z., Gunjal, A., Liu, B., He, Y. (2026). *Reward Hacking in Rubric-Based Reinforcement Learning*. arXiv:2605.12474. https://arxiv.org/abs/2605.12474
36. Salmani-Zarchi, M. M., Rahimi, Z., Faili, H., Dousti, M. J. (2026). *MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following*. arXiv:2606.06058. https://arxiv.org/abs/2606.06058
37. Patel, N., Wang, F., Dhillon, I. S. (2026). *ODRPO: Ordinal Decompositions of Discrete Rewards for Robust Policy Optimization*. arXiv:2605.12667. https://arxiv.org/abs/2605.12667
38. Zeng, J., He, Q., Ren, Q., et al. (2025). *Order Matters: Investigate the Position Bias in Multi-constraint Instruction Following*. arXiv:2502.17204. https://arxiv.org/abs/2502.17204
39. He, Y., Jin, D., Wang, C., Bi, C., et al. (2024). *Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following*. arXiv:2410.15553. https://arxiv.org/abs/2410.15553
40. Mao, Y., Chen, C. (2026). *IFHierBench: Hierarchical Instruction Following for Large Language Models*. arXiv:2607.27912. https://arxiv.org/abs/2607.27912
41. Wen, B., Wang, C., Gui, J., et al. (2026). *MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding*. arXiv:2609.14992. https://arxiv.org/abs/2609.14992
42. Qi, Y., Peng, H., Wang, X., et al. (2025). *AgentIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios*. arXiv:2505.16944. https://arxiv.org/abs/2505.16944
43. Jaroslawicz, D., Whiting, B., Shah, P., Maamari, K. (2025). *How Many Instructions Can LLMs Follow at Once?* (IFScale). arXiv:2507.11538. https://arxiv.org/abs/2507.11538
44. Anand, A., Chattaraj, S. (2026). *Instruction Stacking Collapse: A Benchmark and the Capability-Dependent Value of Prompt Compilation*. arXiv:2608.02639. https://arxiv.org/abs/2608.02639
45. Wei, S., Su, Z., Song, F., et al. (2026). *Verifier-Induced Support Reshaping in On-Policy Optimization*. arXiv:2608.00220. https://arxiv.org/abs/2608.00220
46. Hui, T., Zhu, P., Ping, B., et al. (2025). *DecIF: Improving Instruction-Following through Meta-Decomposition*. arXiv:2505.13990. https://arxiv.org/abs/2505.13990
47. Jia, Q., Shen, Y., Song, X., et al. (2025). *One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework* (EvolIF). arXiv:2511.03508. https://arxiv.org/abs/2511.03508
48. Mehta, S., Panavas, L., Garre, S., Chen, E. (2026). *ComplexConstraints and Beyond: Expert Rubrics for RLVR*. GEM Workshop @ ACL 2026. arXiv:2606.09118. https://arxiv.org/abs/2606.09118
49. Zeng, Y., Sai, Z., Wang, Y., et al. (2026). *Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation* (MIFS). arXiv:2609.16059. https://arxiv.org/abs/2609.16059
50. Zeng, M., Tan, G., Cen, L., Wen, Y. (2026). *VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following*. arXiv:2608.26013. https://arxiv.org/abs/2608.26013
