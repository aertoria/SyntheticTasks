# Math problem synthesis and difficulty escalation (informal math)

*Scope: generating harder informal-math problems (final-answer, not formal proofs) from easy seeds for SFT and RLVR. Covers question bootstrapping and evolution, concept and skill composition, structure-first generators that are correct by construction, composition and perturbation probes, teacher-driven hard-problem generators, policy-aware RL loops and self-play, and curated hard pools with format hardening. Compiled 2026-09-30. Verification: 37 entries checked against primary sources (arXiv abstract pages plus full-text HTML; venues from arXiv comments, citing papers or official repos). 27 corrected, 0 dropped, 5 added (each added entry verified the same way).*

---

## TL;DR

- **Paraphrasing does not make problems harder. Structural operators do.**
  - FLAMES compared synthesis agents and found that "data agents designed to increase problem complexity lead to best improvements on most math metrics".
  - MetaMath: adding 20K samples on top of 80K answer-augmentation data gave +0.1 (more answer augmentation), +0.4 (rephrasing), +2.3 (FOBAR inversion) and +2.6 (self-verification inversion) GSM8K points.
  - RV-Syn's executable function-graph composition produced problems Qwen2.5-Math-7B-Instruct solved only 55.6% of the time. On six other synthetic or human datasets it solved 70.6–97.2%.
- **Chaining already-labelled easy problems is the cheapest fix that stays exactly verifiable for RL.**
  - h1 chains GSM8K problems with deterministic adapters and trains with Dr. GRPO on a horizon curriculum. It needs no new labels. AIME24 avg@32 went 5.10 → 10.52, MATH-500 64.20 → 69.20, GSM-Symbolic P2 43.08 → 52.00, and gains held at pass@128.
  - Uniform mixing and long-only training at equal compute gave **no** long-horizon gains, so the curriculum is required.
  - The difficulty follows a product law. Compositional GSM measures the gap against S1·S2. On MATH², success is roughly the square of MATH success.
- **For saturated RL prompts, use answer-preserving hardening.**
  - MathForge MQR rewrites each MATH question three ways: irrelevant background, an invented abstract term, and a key number replaced by an independent sub-problem. The gold answer stays the same, so no new labels are needed. An o3 equivalence check found 99/97/97% of rewrites answer-equivalent.
  - A failed rewrite yields an all-zero GRPO group, so it gives no gradient and does no harm.
  - With the DGPO optimizer, the average rose from 37.61 (GRPO) to 42.17 on Qwen2.5-Math-7B.
  - SvS rewrites partially solved prompts from the policy's own correct solutions and keeps the answer fixed. It kept entropy from collapsing and gave +18.3 / +22.8 pass@32 on AIME24/25.
- **Seed complexification from problems the model mostly solves, and keep a pass-rate band.**
  - QbQ seeds variants from problems solved 8–15/16 times. This reached 16.46% AIME pass@1, vs 11.36% when seeding from the hardest failures.
  - Working bands in the literature: SwS keeps 25–75%. SvS rewards 12.5–62.5%. R-Zero keeps 3–7 majority matches out of 10. Socratic-Zero uses a Gaussian utility around 0.5 with σ=0.2.
  - Pikus et al.: training GRPO on the hardest 10% gave gains up to 47%, vs 3–15% for easy examples.
  - Route each prompt by pass rate: complexify above the band, train as-is inside it, scaffold below it (PrefixRL, ReGFT, backward hint annealing).
- **Get labels from construction wherever possible.**
  - Execution or symbolic solving: RV-Syn function graphs (0.9% problem and 1.4% solution error, vs 4.9–10.6% solution error for other synthetic sets); EFAGen programs with `single_valued`/`matches_original` tests; SymPy/Z3 mutation (Yeo et al.); GSM-Infinite computation graphs.
  - Inverse checks, such as differentiating a proposed antiderivative (VHG).
  - Cycle consistency (MathCAMPS: 97.7% of survivors judged faithful).
  - Majority vote is the weakest tier. Keep it out of RL rewards when you can.
- **Never reward "hard" without an independent validity gate.**
  - VHG's setter reward is 1[verifier accepts] × (1 − solver accuracy). Without the gate, setters hack difficulty by emitting invalid problems. With it, validity is learned first (30.6% → 75.5% valid), then difficulty (valid-and-hard share 27.5% → 58.5%).
  - R-Zero's majority-vote labels decayed from 79% to 63% accurate over three iterations. Math performance peaked, then fell (49.12 → 46.52).
  - J-Zero's judge co-adaptation kept improving for at least 10 iterations, while its baselines degraded after two.
- **Consensus filters remove exactly the frontier items. SFT and RL tolerate noise very differently.**
  - Removal examples: CoT-Self-Instruct Answer-Consistency drops items that are "incorrectly labeled or too difficult". SAND-Math all-k agreement dropped 23,437 → 17,578. MindLoom excludes all-wrong items.
  - SFT is forgiving: a GPT-4 validation pass did not help MathScale; OpenMathInstruct-2 found "SFT is robust to low-quality solutions"; FLAMES found coverage beats reliability under a fixed budget.
  - RLVR is not forgiving: a wrong reference is a wrong reward.
- **Train the generator (RL or DPO toward valid-and-hard) rather than only filtering.**
  - MathSmith's usable-problem ratio rose from 71.5% (SFT generator) to 95.4% (RL with a consistency reward).
  - Other working recipes: ScaleQuest's QPO (DPO toward solvable or harder); Learning-to-Pose's solver-feedback reward; Agentic Proposing's MGPO; PROPEL's activation-probe proxy, which cuts solver rollouts.
  - Without a trained generator, plan for heavy overgeneration. Code2Math needed 1.56–6.55 failed rollouts per accepted evolution.
- **Harden formats, decontaminate, and test on more than one model family.**
  - Big-Math-Reformulated turned MCQs into open-ended items and skews harder (>50% in the two hardest solve-rate quintiles). DAPO-Math-17K converts answers to integers.
  - DeepMath found raw pools 90% contaminated for AIME24/AMC23.
  - Spurious random rewards still give Qwen2.5-Math-7B +21.4 on MATH-500 (vs +29.1 with true rewards) but fail on Llama3/OLMo2. A synthetic-data gain seen only on Qwen may be an artifact.
- **Real (insight) difficulty differs from tedium, so mix both.**
  - Op-count or horizon scaling gives a smooth sigmoid difficulty knob (GSM-Infinite R²≈0.98), which is good for state-tracking robustness.
  - Insight hiding (Code2Math's "burden of discovery", with scores 1–2 rejected as tedium), hard perturbation (MATH-P-Hard: o1-mini −16.49%) and concept injection change which method is needed.

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| MetaMath (FOBAR/SV) | 2023 | [2309.12284](https://arxiv.org/abs/2309.12284) | GSM8K/MATH | SFT | inversion (mask a given, reveal answer), declarative self-verification, rephrase, answer augmentation | masked number is the free label; CoTs filtered by answer match |
| WizardMath Math Evol-Instruct | 2023 | [2308.09583](https://arxiv.org/abs/2308.09583) | GSM8K/MATH | SFT + PPO | upward evolution (constraints, concretize, more reasoning), downward evolution | GPT-4 answers; instruction RM + process RM; no execution |
| MathScale | 2024 | [2403.02884](https://arxiv.org/abs/2403.02884) | K-12→college | SFT | concept-graph random walk, multi-KP composition | none (GPT-4 validation tested, removed) |
| KPDDS / KPMath | 2024 | [2403.02333](https://arxiv.org/abs/2403.02333) | MATH/GSM8K | SFT | 2–3 co-occurring topics + key points per problem | 10 GPT-4 samples, SymPy equivalence, consensus voting |
| MathFusion | 2025 | [2503.16212](https://arxiv.org/abs/2503.16212) | GSM8K/MATH | SFT | sequential / parallel / conditional fusion of similar pairs | GPT-4o-mini solution + reasonableness check |
| MathMixup | 2026 | [2601.17006](https://arxiv.org/abs/2601.17006) | MATH, AMC-AIME | SFT (curriculum) | hybrid (harder) and decomposed (bridge) fusion of pairs | GPT-4o self-check + 10% manual review |
| DART-Math | 2024 | [2407.13690](https://arxiv.org/abs/2407.13690) | GSM8K/MATH | SFT | difficulty-proportional sampling budget (no new problems) | gold-answer match |
| ScaleQuest | 2024 | [2410.18693](https://arxiv.org/abs/2410.18693) | GSM8K/MATH-level | SFT | question fine-tuning + DPO toward solvable/harder; fail-rate scorer | RM picks best of 5 solutions |
| GSM-Symbolic / NoOp | 2024 | [2410.05229](https://arxiv.org/abs/2410.05229) | grade school | eval (operators reusable) | templating, +1/+2 clauses, irrelevant clause | answer computed from template |
| GSM-Infinite | 2025 | [2502.05252](https://arxiv.org/abs/2502.05252) | arithmetic word problems | eval / generator | computation-graph op scaling, reverse mode, spider-topology noise | graph evaluated before rendering |
| iGSM (Physics of LMs 2.1) | 2024 | [2407.20311](https://arxiv.org/abs/2407.20311) | synthetic GSM | pretraining testbed | dependency-graph op scaling, OOD op, re-ask | graph-computed answers + solution checker |
| MathCAMPS | 2024 | [2407.00900](https://arxiv.org/abs/2407.00900) | K-8 Common Core | eval | grammar sampling, counterfactual/incremental follow-ups | SymPy answer + cycle-consistency back-translation |
| EFAGen | 2025 | [2504.09763](https://arxiv.org/abs/2504.09763) | MATH/AMC/olympiad | SFT (+ variant mining) | lift seed into parameterized program; resample/extreme params | execute `solve()`; unit tests incl. `matches_original` |
| RV-Syn | 2025 | [2504.20426](https://arxiv.org/abs/2504.20426) | NuminaMath-derived | SFT | function-library graph composition + back-translation | graph execution; CoT must match executed answer |
| Adaptive symbolic generation (Yeo et al.) | 2026 | [2602.19187](https://arxiv.org/abs/2602.19187) | GSM8K/MATH seeds | RL (PPO/GRPO) | SymPy/SMT mutation (coupled, nested, nonlinear), closed-loop prompt optimization | SymPy / Z3 solve |
| MathAgent (added) | 2026 | [2604.11188](https://arxiv.org/abs/2604.11188) | general math | SFT | adversarial evolution of constraint-graph blueprints, then instantiation | multi-agent critic + external LLM judge |
| MATH² | 2024 | [2407.21009](https://arxiv.org/abs/2407.21009) | competition math | eval / ICL | skill-pair composition | LLM validation + majority vote + human edits |
| Compositional GSM | 2024 | [2410.01748](https://arxiv.org/abs/2410.01748) | grade school | eval | serial composition (Q1 answer → Q2 variable) | code execution + ≥4/16 model agreement + manual fix |
| CHASE-Math (added) | 2025 | [2502.14678](https://arxiv.org/abs/2502.14678) | grade school | eval (+ small FT) | iterative continuation (depth growth) | each easy hop verified by an ensemble of solvers |
| MATH-Perturb (added) | 2025 | [2502.06453](https://arxiv.org/abs/2502.06453) | MATH Level 5 | eval | hard perturbation (minimal edit that breaks the method) | PhD annotators + cross-validation |
| h1 | 2025 | [2510.07312](https://arxiv.org/abs/2510.07312) | GSM8K → long chains | RL (Dr. GRPO) | serial composition with deterministic adapters, horizon curriculum | atomic labels + deterministic adapters (exact) |
| PromptCoT / PromptCoT 2.0 | 2025 | [2503.02324](https://arxiv.org/abs/2503.02324), [2509.19894](https://arxiv.org/abs/2509.19894) | olympiad math, code | SFT + self-play RL | concept → rationale → problem; EM-trained rationale model | 1.0: two-LLM "perfect" rating; 2.0: majority vote of 8 |
| SAND-Math | 2025 | [2507.20527](https://arxiv.org/abs/2507.20527) | competition math | SFT | from-scratch generation; difficulty hiking (theorem + cross-domain concept) | all-k R1 answers agree; solver-failure + novelty filters |
| MathSmith | 2025 | [2508.05592](https://arxiv.org/abs/2508.05592) | PlanetMath concepts | SFT | 9 difficulty strategies, rationale-first, RL generator (trace-length reward) | majority exists among K=5 teacher answers |
| ScaleDiff | 2025 | [2509.21070](https://arxiv.org/abs/2509.21070) | competition math | SFT | adaptive-thinking difficulty detection; generator trained on hard slice only | rule filters + drop items base model already solves |
| MindLoom | 2026 | [2605.21630](https://arxiv.org/abs/2605.21630) | STEM + competition math | SFT | mined "thought modes" applied iteratively; scarcity reward | 3 rollouts judged by LLM; ≥1 correct required |
| CoT-Self-Instruct | 2025 | [2507.23751](https://arxiv.org/abs/2507.23751) | verifiable reasoning | RL (GRPO) | plan-then-generate from seeds | Answer-Consistency (generator answer = solver majority) |
| Hypothesis-driven errors (Fu et al.) | 2026 | [2604.04386](https://arxiv.org/abs/2604.04386) | MATH-level | eval | hypothesis-conditioned generation on target model's failures | R1 / o3 / GPT-5 agreement + manual validation |
| SwS | 2025 | [2506.08989](https://arxiv.org/abs/2506.08989) | competition math | RL | mine weak problems, recombine concepts | strong reasoner, ≥50% self-consistent answers |
| SvS | 2025 | [2508.14029](https://arxiv.org/abs/2508.14029) | competition math | RL (online) | answer-preserving variational rewriting from policy's correct solutions | same reference answer; accuracy band as validity |
| MathForge (MQR + DGPO) | 2026 | [2601.20614](https://arxiv.org/abs/2601.20614) | MATH (+GeoQA) | RL | background noise, abstract term, nested sub-problem; difficulty-weighted GRPO | answer preservation (o3 equivalence audit 97–99%) |
| QbQ | 2026 | [2608.01522](https://arxiv.org/abs/2608.01522) | AIME | RL (GRPO) | generalize-then-specialize, parametrize-and-sum, change queried quantity, inverse, constraint layer | teacher answers; variant answer must differ from parent |
| R-Zero | 2025 | [2508.05004](https://arxiv.org/abs/2508.05004) | math (from zero) | RL self-play | challenger rewarded 1−2\|p̂−½\|, BLEU repetition penalty | solver majority vote only |
| Socratic-Zero | 2025 | [2509.24726](https://arxiv.org/abs/2509.24726) | math | DPO solver + WSFT generator | failure-targeted refinement; harder variants of mastered items | teacher reference; rule + LLM judge |
| VHG | 2026 | [2605.06660](https://arxiv.org/abs/2605.06660) | integrals, general math | RL self-play | setter reward = validity × (1 − solver acc) | SymPy differentiation (hard) / LLM judge (soft) |
| Learning to Pose (added) | 2025 | [2511.09907](https://arxiv.org/abs/2511.09907) | math + general reasoning | RL generator → solver training | design-CoT cold start; solver-accuracy "complementary" reward | solver majority-vote pseudo-labels |
| PROPEL (added) | 2026 | [2606.18284](https://arxiv.org/abs/2606.18284) | math, code, SWE | generator RL | target-solve-rate generation with probe proxy | task validity predicate + oracle scoring |
| Agentic Proposing | 2026 | [2602.03279](https://arxiv.org/abs/2602.03279) | math, science, code | SFT + RL | skill-library composition by a trained 4B proposer agent | verifier ensemble + sandboxed code; pass@16 prober |
| Code2Math | 2026 | [2603.03202](https://arxiv.org/abs/2603.03202) | competition → IMO-level | eval + SFT | insight hiding via code exploration | solvability agent + external judge (99.2% precision vs humans) |
| Big-Math (+Reformulated) | 2025 | [2502.17387](https://arxiv.org/abs/2502.17387) | mixed | RL pool | MCQ → open-ended; remove guessable/unverifiable formats | human-in-loop filters ≥90% P/R; solvability rollouts |
| DeepMath-103K | 2025 | [2504.11456](https://arxiv.org/abs/2504.11456) | MSE/Numina | RL pool | hard-source mining, level ≥5 filter, standardization | 3 R1 solutions + source answer must agree |
| OpenMathReasoning | 2025 | [2504.16891](https://arxiv.org/abs/2504.16891) | AoPS olympiad | SFT | proof → answer conversion, remove MCQ/binary | forum answer, else most common candidate |

---

## Method notes

### A. Question bootstrapping, evolution and concept composition (SFT era)

### MetaMath — MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models (Yu et al., 2023)
Link: https://arxiv.org/abs/2309.12284 (ICLR 2024 spotlight)
- **Mechanism:** Bootstraps GSM8K/MATH training questions four ways:
  - AnsAug: new GPT-3.5 CoTs, kept if the answer matches gold.
  - Rephrasing.
  - FOBAR: mask a number with x and append "If we know the answer to the above question is A, what is the value of unknown variable x?".
  - Self-Verification (SV): rewrite question plus answer as a declarative statement and ask for the masked quantity.

  MetaMathQA has 395K samples (AnsAug 155K, Rephrasing 130K, SV 55K, FOBAR 55K).
- **How it makes tasks harder:** Backward questions reverse the reasoning direction. A forward multiply-and-add word problem becomes an equation to solve for an input.
- **Correctness / verification:** The label for a backward question is the masked number, so it costs nothing. Generated CoTs are kept only if their final answer matches.
- **Difficulty control:** No explicit knob, only the mix of augmentation types. The authors measure "diversity gain" instead.
- **Reported results:**
  - MetaMath-7B scores 66.5 GSM8K / 19.8 MATH (paper table; the v1 abstract says 66.4 / 19.4, +11.5 / +8.7 over same-size SOTA). MetaMath-70B scores 82.3 on GSM8K.
  - LLaMA-2-7B ablation: AnsAug only 59.7 → +Rephrasing 60.6 → all four 64.4 GSM8K.
  - Adding 20K samples to 80K AnsAug gives +0.1 (AnsAug), +0.4 (Rephrasing), +2.3 (FOBAR) and +2.6 (SV).
  - Diversity gain vs accuracy: Pearson 0.972.
- **Limitations / failure modes:** Inverses of easy seeds remain easy, so this adds diversity more than difficulty. Masking can create problems with several solutions or non-integer solutions. The mix is not calibrated to any policy.
- **How to reuse with easy seed tasks:** Apply inversion to any numeric seed to get a free label. Stack it on composed or chained problems to get real depth. Check uniqueness of the masked value over the intended domain by brute force or a solver.

### WizardMath (Math Evol-Instruct + RLEIF) — WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct (Luo et al., 2023)
Link: https://arxiv.org/abs/2308.09583 (ICLR 2025 oral)
- **Mechanism:**
  - Seeds: the GSM8K and MATH training sets (the "original manually annotated 7.5k data of GSM8k and MATH").
  - GPT-4 evolves each instruction for 5 rounds, 2 downward and 3 upward, each round building on the previous one. It makes 6 evolutions per round at T=0.7.
  - Downward evolution makes the question easier or moves it to an easier topic. Upward evolution adds "more real-world constraints and dependencies between variables", concretizes, and increases reasoning.
  - Result: 448K unique instructions (17K duplicates removed), 418K after removing 30K for contamination. GPT-4-0613 writes step-by-step answers.
  - RLEIF adds an Instruction Reward Model, trained on GPT-4 rankings of evolved instructions by difficulty and definition, and a process-supervised RM with GPT-4 step labels. Both feed step-by-step PPO.
- **How it makes tasks harder:** Upward evolution adds constraints, variable dependencies and reasoning steps.
- **Correctness / verification:** Weak. GPT-4 answers its own evolved questions, and the PRM scores steps. There is no execution check.
- **Difficulty control:** Evolution direction and number of rounds, plus the IRM's difficulty/definition score.
- **Reported results:**
  - Mistral-7B SFT: original data 59.7 GSM8K / 15.1 MATH. Two downward rounds: 74.5 / 34.7. Three upward rounds: 78.6 / 42.5. Upward plus downward: 81.2 GSM8K.
  - WizardMath-Mistral-7B: 90.7 / 55.4. WizardMath-70B: 92.8 / 58.6.
- **Limitations / failure modes:** Upward evolution can create ill-posed questions, and GPT-4's answers to its own hardest questions may be wrong. Difficulty is asserted by the prompt, not measured by solve rate.
- **How to reuse with easy seed tasks:** Build per-seed difficulty ladders. Note that easier (downward) variants also helped, which supports adding "bridge" steps. Before RL, gate each rung with an independent answer check and a policy pass rate.

### MathScale — MathScale: Scaling Instruction Tuning for Mathematical Reasoning (Tang et al., 2024)
Link: https://arxiv.org/abs/2403.02884
- **Mechanism:**
  - GPT-3.5 extracts topics and knowledge points (KPs) from about 20K MwpBench training questions, giving 2,018 topics and 8,892 KPs.
  - These form a concept graph with co-occurrence-weighted edges.
  - A random walk takes 1–2 steps in the topic subgraph, then expands on the KP graph for 0–4 steps.
  - About 1K epochs give 2M unique concept compositions. GPT-3.5 turns each into a QA pair.
- **How it makes tasks harder:** Longer walks require more concepts per problem.
- **Correctness / verification:** None in the final pipeline. On 100 hand-annotated items GPT-4 was 87% accurate vs 69% for GPT-3.5, but a GPT-4 validate-and-replace step "does not improve the results", so it was removed.
- **Difficulty control:** Implicit, through walk length and number of concepts.
- **Reported results:** MathScale-7B scores 35.0% micro / 37.5% macro accuracy on MwpBench. That is +42.9% / +43.7% *relative* over the best same-size peers.
- **Limitations / failure modes:** Unverified labels are tolerable for SFT distillation but not as RL rewards.
- **How to reuse with easy seed tasks:** Build a concept co-occurrence graph from your seeds and sample longer walks to force multi-concept items. For RL, attach an executable or consensus check.

### KPDDS / KPMath — Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning (Huang et al., 2024)
Link: https://arxiv.org/abs/2403.02333 (AAAI 2025, vol. 39, per citing papers)
- **Mechanism:**
  - GPT-4 extracts topics and key points from seed problems, producing the MPKP set, and builds a Topic-level Co-occurrence Probability Matrix (TCPM).
  - The pipeline samples 2–3 co-occurring topics. For each topic it takes one practice problem with its key points, and GPT-4 writes a new problem from this set.
  - A quality scorer (0–1) with a 0.85 threshold keeps about 51%.
- **How it makes tasks harder:** Each problem must integrate key points from several topics.
- **Correctness / verification:** GPT-4 resamples 10 responses per sub-question (T=0.75, top-p 0.95). SymPy checks answer equivalence, and a consensus score vector is computed. "Semi-voting" with a CSV threshold of 0.1 was best: it cut 46.7% of the data vs no voting with no loss in performance.
- **Difficulty control:** Number of co-sampled topics. TCPM keeps combinations natural.
- **Reported results:** KPMath has over 800K pairs; KPMath-Plus has 1,576K. Qwen1.5-72B scores 87.0 GSM8K and 58.3 MATH, averaging 81.5 over six datasets.
- **Limitations / failure modes:** Needs a strong teacher. Consensus biases the set toward items the teacher finds easy.
- **How to reuse with easy seed tasks:** Sample key points from 2–3 different seeds using a co-occurrence prior, and keep SymPy-equivalence voting as the gate.

### MathFusion — MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion (Pei et al., 2025)
Link: https://arxiv.org/abs/2503.16212 (ACL 2025) · code: github.com/QizhiPei/MathFusion
- **Mechanism:**
  - For each seed, the most similar problem is retrieved by text-embedding-3-large inner product. 83% of MATH pairs share a category.
  - GPT-4o-mini fuses each pair three ways:
    - Sequential: A's answer becomes an input of B.
    - Parallel: one problem needs both solution strategies.
    - Conditional: solve both and compare or select.
  - GPT-4o-mini also writes the solutions.
- **How it makes tasks harder:** Multi-problem dependency, strategy integration, and extra comparison steps.
- **Correctness / verification:** A GPT-4o-mini reasonableness check flagged 5.6% of fused problems, which were removed. There is no hard answer check.
- **Difficulty control:** Fusion type, and fusions can be stacked.
- **Reported results:** MathFusionQA has 60K items (15K original GSM8K+MATH plus 45K fused). That gives +18.0 average accuracy over standard training "while requiring only 45K additional synthetic instructions". Scaled to 195K (top-2 to top-4 neighbours) and combined with DART-Math, it surpasses DART-Math on average with less than a third of its data.
- **Limitations / failure modes:** Small teacher, no consensus. Similar pairs limit how far difficulty can rise.
- **How to reuse with easy seed tasks:** If seeds have code solutions, sequential fusion is exactly verifiable, because it is the Compositional GSM construction. Use conditional fusion to add compare/select steps.

### MathMixup — MathMixup: Boosting LLM Mathematical Reasoning with Difficulty-Controllable Data Synthesis and Curriculum Learning (Li et al., 2026)
Link: https://arxiv.org/abs/2601.17006
- **Mechanism:** BGE embeddings pair similar questions (similarity 0.75–0.9) that have *different* difficulty levels. GPT-4o builds two kinds of question from each pair:
  - Hybrid: "more difficult than the harder of the two originals".
  - Decomposed: simplified dependencies, intermediate difficulty.

  QwQ-32B writes long-CoT responses. Training follows a decomposed → original → hybrid curriculum.
- **How it makes tasks harder:** Integrates the structures of two problems into one scenario with extra constraints.
- **Correctness / verification:** GPT-4o self-check and correction, plus manual review of a random 10%. The authors admit subtle errors can remain.
- **Difficulty control:** Three explicit tiers plus the curriculum order.
- **Reported results:** MathMixupQA has 34.5K items (11.5K per tier) built from MATH and AMC-AIME seeds. Qwen2.5-7B averages 47.6% over 7 benchmarks, and 52.6% when mixed with MathFusionQA under the curriculum.
- **Limitations / failure modes:** LLM-judged correctness; only two-problem fusion.
- **How to reuse with easy seed tasks:** Generate *both* harder hybrids and bridging decompositions, so an RL curriculum has intermediate steps.

### DART-Math — DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving (Tong et al., 2024)
Link: https://arxiv.org/abs/2407.13690 (NeurIPS 2024) · code: github.com/hkust-nlp/dart-math
- **Mechanism:** Vanilla rejection sampling gives every query the same budget, so hard queries end up with few or no correct traces. DART measures difficulty as the fail rate of DeepSeekMath-7B-RL samples. Two sampling schemes:
  - Uniform: keeps sampling until k_u = 40 correct responses per query.
  - Prop2Diff: allocates correct responses in proportion to difficulty (k_u = 192).
- **How it makes tasks harder:** It does not create problems. It reweights SFT data toward hard queries (Table 3: DART-Hard averages 8.49 responses per query at Level 1 and 107.06 at Level 5).
- **Correctness / verification:** Final answer must match gold.
- **Difficulty control:** Per-query sampling budget from fail rate. With a maximum of 2048 raw samples, over 90% of queries reach their quota under Uniform (about 5M raw samples in total).
- **Reported results:** 585K (Uniform) and 590K (Hard) examples. Llama3-8B improves MATH 21.2 → 46.6 and GSM8K 51.0 → 82.5. ToRA-Corpus-16k covers only 68% of Level-5 MATH queries, while DART datasets cover 99.6%.
- **Limitations / failure modes:** Needs gold answers and adds no new problems.
- **How to reuse with easy seed tasks:** After complexifying, collect SFT traces with difficulty-proportional budgets. Otherwise the new hard items silently vanish from the SFT mix.

### ScaleQuest — Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch (Ding et al., 2024)
Link: https://arxiv.org/abs/2410.18693 (ACL 2025)
- **Mechanism:**
  - Question Fine-Tuning (QFT) turns small math solvers (DeepSeekMath-7B-RL and Qwen2-Math-7B-Instruct) into question generators, using about 15K GSM8K/MATH questions.
  - Question Preference Optimization (QPO): for each sample, an LLM rewrites the generated question toward either solvability or difficulty (direction chosen at random). The rewrite is preferred over the original in DPO.
  - Filters: language, solvability, and "difficulty sampling". The difficulty scorer is DeepSeekMath-7B-Base regressed (MSE) on DeepSeekMath-7B-RL fail rates over GSM8K/MATH train, and part of the "overly simple" questions is dropped.
- **How it makes tasks harder:** DPO toward harder-but-solvable rewrites, plus removal of easy questions.
- **Correctness / verification:** 5 sampled solutions; InternLM2-7B-Reward picks the best. Reward-model selection, not verification.
- **Difficulty control:** The learned fail-rate regressor plus QPO direction.
- **Reported results:** 2M generated questions yield 1M QA pairs for 522.9 GPU-hours (about $680.8). Qwen2-Math-7B-ScaleQuest reaches 73.4 MATH. Average gains of 5.6–11.5% over prior SOTA synthetic sets.
- **Limitations / failure modes:** RM-selected labels are unsuitable as RL rewards.
- **How to reuse with easy seed tasks:** A cheap in-house recipe: fine-tune your own solver into a question generator, DPO it toward "harder but solvable", and use a fail-rate regressor as a cheap difficulty prior.

### B. Structure-first generators (labels correct by construction)

### GSM-Symbolic / GSM-NoOp — GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models (Mirzadeh et al., 2024)
Link: https://arxiv.org/abs/2410.05229 (ICLR 2025)
- **Mechanism:** GSM8K problems become symbolic templates with variables and constraints. The study uses 100 templates with 50 samples each. Difficulty variants: GSM-M1 removes a clause, GSM-P1 and GSM-P2 add one or two clauses, and GSM-NoOp adds "seemingly relevant but ultimately irrelevant" clauses.
- **How it makes tasks harder:** More reasoning clauses and inert distractors. The paper's example: "…On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average."
- **Correctness / verification:** The answer is computed from the instantiated template.
- **Difficulty control:** M1 < base < P1 < P2, plus NoOp.
- **Reported results:** Accuracy falls as clauses are added and varies when only the numbers change. NoOp causes drops of up to 65% (Phi-3-mini over 65%), and o1-preview also declines significantly.
- **Limitations / failure modes:** Templates are hand-built; the work is evaluation-oriented.
- **How to reuse with easy seed tasks:** Template your seeds. Treat +1/+2 clauses and NoOp distractors as graded difficulty levels for RL, with labels recomputed exactly.

### GSM-Infinite — GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity? (Zhou et al., 2025)
Link: https://arxiv.org/abs/2502.05252 · code: github.com/Infini-AI-Lab/gsm_infinite
- **Mechanism:**
  - Problems are computation graphs: nodes are variables, and edges are explicit or implicit operations. "Three-entity" variables allow implicit ×/÷.
  - "op" is the number of operations needed to reach the answer. GSM8K test problems span op 2–12, mostly 3–4.
  - **Reverse mode** makes the query an initially assigned variable, so the solver must set up and solve an equation. This generates implicit subtraction and division. Generations that would need quadratic or higher-order equations are discarded.
  - **Spider-topology noise**: noise nodes attach with edges pointing *out of* core nodes, so they cannot affect the answer. They are semantically close, which defeats RAG retrieval.
  - Three levels: Symbolic, Medium and Hard, rendered through three templates.
- **How it makes tasks harder:** More ops, backward direction, and more noise and context length.
- **Correctness / verification:** The graph is evaluated before rendering.
- **Difficulty control:** op (an integer knob), forward vs reverse, and amount of noise. AUC is a Riemann sum of accuracy vs op, from op=2 until accuracy falls below 5%.
- **Reported results:**
  - Accuracy decays as a sigmoid in op (R²≈0.98 on Medium for o1-mini, Qwen2.5-72B and Qwen2.5-7B). Reverse problems follow a left-shifted sigmoid, so they are harder.
  - DeepSeek-R1's zero-noise AUC is nearly 4× the previous SOTA.
  - Repeated sampling gives linear gains for exponentially growing compute.
- **Limitations / failure modes:** Templated language and arithmetic only.
- **How to reuse with easy seed tasks:** Use it as an unlimited, exactly labelled curriculum: raise op, switch to reverse mode, then add spider noise when the pass rate saturates.

### iGSM — Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process (Ye et al., 2024)
Link: https://arxiv.org/abs/2407.20311 · code: github.com/facebookresearch/iGSM (generator `IdGen`, full-solution checker `true_correct`)
- **Mechanism:** Samples a structure graph and a parameter dependency graph, then renders a problem and a step-by-step solution. "Re-ask" resamples the query parameter, which changes the number of operations needed.
- **How it makes tasks harder:** Larger op, more parameters and dependencies.
- **Correctness / verification:** Answers and full solutions are checked against the graph.
- **Difficulty control:** iGSM-med trains with op ≤ 15 and evaluates out of distribution at op ∈ {20…23}. iGSM-hard trains with op ≤ 21 and evaluates at op ∈ {28…32}.
- **Reported results:** A GPT-2-size model pretrained on iGSM reaches 99% in distribution and generalizes to longer op counts than it saw in training.
- **Limitations / failure modes:** Fully synthetic language. Best used as a controlled testbed or midtraining data.
- **How to reuse with easy seed tasks:** Controlled studies of which operators (depth, dependency, re-ask) break your model before you spend on LLM-based generation.

### MathCAMPS — MathCAMPS: Fine-grained Synthesis of Mathematical Problems From Human Curricula (Mishra et al., 2024)
Link: https://arxiv.org/abs/2407.00900 (latest version retitled "From Next-Token to Mathematics: The Learning Dynamics of Mathematical Reasoning in Language Models", COLM 2025) · code: github.com/gpoesia/mathcamps
- **Mechanism:** 44 of the 229 K–8 Common Core standards are encoded as attribute grammars. Symbolic problems are sampled and solved with SymPy, and GPT-4 renders them as word problems. Follow-up questions are derived symbolically: counterfactual (change a constant) or incremental (add information).
- **How it makes tasks harder:** Follow-ups turn a single-turn item into multi-turn mathematical dialogue.
- **Correctness / verification:** **Cycle consistency.** GPT-4 back-translates the word problem to symbolic form without seeing the original, and the problem is kept only if the answers match.
  - In a manual check of 245 problems, 30 (12.2%) were unfaithful. Cycle consistency discarded 25 of those, plus 7 faithful ones.
  - 210 of the 215 survivors (97.7%) were judged faithful.
- **Difficulty control:** Grammar productions per standard (grade level) and follow-up turns.
- **Reported results:** 4,900 problems, or 9,607 including follow-ups, for about $330 (about $0.034 each) with gpt-4-0613. The v1 paper evaluated 23 LLMs and found "surprising failures even in the strongest models", especially on simple follow-ups.
- **Limitations / failure modes:** Elementary level only; grammars are written by hand.
- **How to reuse with easy seed tasks:** Cycle consistency is a cheap general faithfulness gate for any symbolic → text complexification. Counterfactual and incremental follow-ups are cheap operators that keep labels exact.

### EFAGen — Executable Functional Abstractions: Inferring Generative Programs for Advanced Math Problems (Khan et al., 2025)
Link: https://arxiv.org/abs/2504.09763
- **Mechanism:** An LLM turns a seed problem and its solution into a Python class with three methods: `sample()` draws valid parameters, `solve(params)` computes the generalized solution, and `render(params)` writes the problem text. It over-generates candidates and filters them with unit tests:
  - `is_extractable`
  - `is_executable`
  - `has_dof` (parameters actually vary)
  - `single_valued`
  - `matches_original` (the EFA reproduces the original problem and answer, a soundness or cycle check)
- **How it makes tasks harder:** Sampling many instances surfaces hard variants. The authors took MATH problems (Level 1 and Level 5) that GPT-4o solved, sampled 50 variants from each EFA, and found variants GPT-4o fails with non-zero probability *even for Level-1 seeds*.
- **Correctness / verification:** Answers come from executing `solve()`; the tests reject unsound abstractions.
- **Difficulty control:** Parameter ranges. Test outcomes also serve as rewards for self-training the EFA generator.
- **Reported results:**
  - On 10K NuminaMath problems, EFAs were inferred for 38.4% (Olympiads), 50.9% (Synthetic AMC) and 40.6% (AMC-AIME).
  - EFA augmentation (33% seed setting): +1.9 MATH-500 pass@1 and +2.2 on both FnEval splits.
  - One correctly solved variant in context raises pass rate on other variants by 16.65%.
- **Limitations / failure modes:** Parameter changes often preserve the solution template. About half of competition problems resist abstraction.
- **How to reuse with easy seed tasks:** Lift seeds into EFAs. Mine the variants your policy fails, which gives exactly labelled RL items. Compose EFAs by feeding one's output into another's parameter.

### RV-Syn — RV-Syn: Rational and Verifiable Mathematical Reasoning Data Synthesis based on Structured Function Library (Wang et al., 2025)
Link: https://arxiv.org/abs/2504.20426 (EACL 2026)
- **Mechanism:**
  - NuminaMath solutions are decomposed into executable Python functions (skills). AST- and semantics-aware merging removes about 28% redundancy.
  - Functions are linked by co-occurrence edges and topic edges.
  - New problems: sample function nodes (co-occurrence, topic, or "edgeless" long-tail combinations), compose them into a computational graph, execute it, then back-translate to a problem. A fine-tuned Qwen2.5-7B-Instruct generates the graphs and Qwen2.5-72B-Instruct back-translates them.
- **How it makes tasks harder:** Multi-function, cross-topic solution graphs designed "solution-first".
- **Correctness / verification:** 16% of graphs fail to execute and are dropped. A further 37% of problem–solution pairs are dropped because no CoT matched the executed ground truth. On 1,000 audited samples per method (Table 4):

  | Source | Problem error | Solution error |
  |---|---|---|
  | RV-Syn | 0.9% | 1.4% |
  | NuminaMath | 0.6% | 5.5% |
  | ScaleQuest | 5.6% | 8.0% |
  | MathScale | 1.6% | 4.9% |
- **Difficulty control:** Number of function nodes (the ablation uses 1, 2 or 3). Sampling strategy trades off: co-occurrence is best on GSM8K, edgeless on MATH-500, topic on GSM-Hard and OlympiadBench.
- **Reported results:** 50K samples beat prior SOTA synthetic sets that use 100K, a +6.3% gain on LLaMA-3-8B-Instruct. Difficulty (Table 3), measured as Qwen2.5-Math-7B-Instruct solve rate on each dataset's problems:

  | Dataset | Solve rate | Avg R1-Distill-7B tokens |
  |---|---|---|
  | RV-Syn | 55.6% | 4,431 |
  | MetaMath | 97.2% | 1,339 |
  | NuminaMath | 72.9% | 3,858 |
  | MAmmoTH2 | 70.6% | 3,272 |
- **Limitations / failure modes:** Only computable numeric answers. Back-translation may leak solution structure.
- **How to reuse with easy seed tasks:** Mine a function library from your easy seeds' solutions and compose 2–4 functions across topics. The executed value is the RL label, and graph size is the difficulty knob.

### Adaptive Problem Generation via Symbolic Representations (Yeo et al., 2026)
Link: https://arxiv.org/abs/2602.19187 · code: github.com/aserety/adaptive-problem-generation
- **Mechanism:**
  - GSM8K/MATH seeds are lifted into SymPy code or SMT-LIB (variables plus constraints).
  - An LLM modifies the symbolic program, for example by adding coupled equations, nested substitutions, or "at least one nonlinear or chained constraint (such as a product, ratio, or sum-of-squares)".
  - SymPy or Z3 solves for the new answer, and the result is rendered to natural language.
  - TextGrad optimizes the modification prompt in closed loop from the student model's performance.
- **How it makes tasks harder:** Structural edits in symbolic space.
- **Correctness / verification:** Solver-computed ground truth; no LLM-written answers.
- **Difficulty control:** Closed-loop prompt optimization against student accuracy.
- **Reported results:**
  - Students: Qwen2.5-1.5B/3B-Instruct trained with PPO or GRPO. In the main PPO setting the average went from 51.14 (seed data) to 54.32 (optimized symbolic prompts), vs 53.16 for optimized natural-language prompts.
  - About 8% improvement with only 100 seed problems (baseline 4%).
  - Diversity (average pairwise cosine distance): 0.37 for symbolic vs 0.07 for natural-language generation.
- **Limitations / failure modes:** Only problems expressible in SymPy/SMT. Rendered text can be stilted.
- **How to reuse with easy seed tasks:** Complexify in symbolic space: lift, mutate, solve, render, then cycle-check. Adapt the mutation prompt, or train a mutation policy, against your policy's pass rate.

### MathAgent (added) — MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis (Yu et al., 2026)
Link: https://arxiv.org/abs/2604.11188
- **Mechanism:** Treats synthesis as "an unsupervised optimization problem over a constraint graph followed by semantic instantiation", with two levels:
  - **Legislator** (meta-level): a tri-agent system of Proposer, Critic and Moderator iteratively evolves a constraint-graph blueprint conditioned on style tokens. The Critic checks internal consistency, specification alignment, and "optimization potential".
  - **Executor** (base-level): grounds the blueprint in diverse natural-language scenarios.
- **How it makes tasks harder:** Adversarial evolution of the logical skeleton, separate from wording.
- **Correctness / verification:** Critic consistency checks plus an external model judge of question/answer logic and consistency. It is LLM-based, not executed.
- **Difficulty control:** Rounds of adversarial evolution on the graph.
- **Reported results:** Across 10 models (Qwen, Llama, Mistral, Gemma), fine-tuning on 1K synthesized samples beats LIMO and s1K on eight math benchmarks, with better out-of-distribution generalization.
- **Limitations / failure modes:** Judge-only correctness. SFT-scale evidence.
- **How to reuse with easy seed tasks:** Separate "design the constraint structure" from "write the story". Evolve the structure adversarially and render many surface stories per structure.

### C. Composition and perturbation probes (evaluation work that yields operators)

### MATH² — AI-Assisted Generation of Difficult Math Questions (Shah et al., 2024)
Link: https://arxiv.org/abs/2407.21009 · code: github.com/veds12/ai_assisted_questions
- **Mechanism:** An LLM extracts 114 skills from MATH. Random pairs of skills go through a five-step pipeline:
  1. Skill-pair validation (are the two skills distinct?).
  2. Question generation requiring both skills.
  3. An attempted solution with a "defeatist" approach.
  4. Question validation for correctness, rigor and clarity.
  5. A final solution using in-context prompting and majority voting (maj@4).

  Humans then verify. 130 of 210 questions (61.9%) were modified, to fix errors and ambiguity, to make unsolvable questions solvable, or to raise difficulty.
- **How it makes tasks harder:** Forces two distinct skills that rarely co-occur.
- **Correctness / verification:** LLM validation, majority vote and human editing.
- **Difficulty control:** The number of skills composed.
- **Reported results:** All models score lower on MATH² than on MATH, and "the success rate on MATH² is the square on MATH". MATH² questions also work better than MATH questions as in-context exemplars.
- **Limitations / failure modes:** Heavy human effort, 210 items only.
- **How to reuse with easy seed tasks:** Use skill-pair (or triple) composition as a difficulty multiplier. The p² law lets you predict landing difficulty. Replace humans with executable pieces plus cross-family consensus.

### Compositional GSM — Not All LLM Reasoners Are Created Equal (Hosseini et al., 2024)
Link: https://arxiv.org/abs/2410.01748
- **Mechanism:** Two GSM8K test questions are chained. A number in Q2 is replaced by X, defined as the answer to Q1, and the new answer comes from executing Q2's code solution. Each split has 1,200 examples.
- **How it makes tasks harder:** Two-hop dependency plus more context.
- **Correctness / verification:** Code execution. 16 candidate solutions come from GPT-4o and Gemini 1.5 Pro, and questions where fewer than 4 agree with the executed answer were fixed by hand (about 25%).
- **Difficulty control:** Number of hops. The reasoning gap is Δ = S_comp − S1·S2.
- **Reported results:**
  - Most models fall below the product expectation. The gap is larger for small, cheaper and math-specialized models.
  - Qwen2.5-Math-7B-IT scores above 80% on MATH but solves under 60% of compositional grade-school problems.
  - Fine-tuning Gemma2-27B on GSM8K improves compositional accuracy for about 100 steps, after which it drops while GSM8K accuracy keeps rising (task overfitting). Neither improves after 400 steps.
  - The gaps come from distraction and poor second-hop reasoning, not leakage.
- **Limitations / failure modes:** Evaluation only.
- **How to reuse with easy seed tasks:** The cheapest exactly verifiable complexifier for word problems. Also use S_comp − ∏S_i as a diagnostic of where the policy breaks.

### CHASE-Math (added) — How to Get Your LLM to Generate Challenging Problems for Evaluation (Patel et al., 2025)
Link: https://arxiv.org/abs/2502.14678 · code: github.com/McGill-NLP/CHASE
- **Mechanism:** Builds hard problems bottom-up from simple components and splits generation into independently verifiable sub-tasks. For math:
  - Split a seed word problem into context and question.
  - Prompt a generator for a *continuation* that assumes the seed's answer as given information, adds new context, and asks a new question answered by one arithmetic operation on it.
  - Concatenate the contexts and iterate, raising reasoning depth each time.
- **How it makes tasks harder:** Iterative depth growth, a form of serial composition.
- **Correctness / verification:** Each new hop is as easy as the seed, so it is checked by an ensemble of non-identical verifier models that "perform well on the seed dataset". If any verifier disagrees, the hop is discarded and restarted. Post-hoc filters use held-out verifiers.
- **Difficulty control:** Number of continuation iterations.
- **Reported results:** Frontier models score 40–60% across the three CHASE benchmarks. Fine-tuning 7–8B models on CHASE-Math data they generated themselves gave only marginal gains: Llama-3.1-8B 30 → 34.7, Mistral-7B 3.3 → 4.7, Qwen2-7B 12.7 → 15.3.
- **Limitations / failure modes:** Arithmetic-level depth; built for evaluation.
- **How to reuse with easy seed tasks:** The key idea is to verify each easy increment, not the hard composite. This keeps verification cost flat as depth grows.

### MATH-Perturb (added) — MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations (Huang et al., 2025)
Link: https://arxiv.org/abs/2502.06453
- **Mechanism:** 12 PhD-student annotators make *minimal* edits to 279 Level-5 MATH problems. MATH-P-Simple keeps the solution path. MATH-P-Hard "fundamentally change[s] the nature of the problem so that the original solution steps do not apply".
- **How it makes tasks harder:** Breaks the memorized method while staying textually close to the original.
- **Correctness / verification:** Annotators double-check their work, and an independent annotator cross-validates each answer.
- **Difficulty control:** Simple vs hard perturbation.
- **Reported results:** Large drops on MATH-P-Hard, e.g. o1-mini −16.49% and gemini-2.0-flash-thinking −12.9%. Models "blindly apply learned problem-solving skills without assessing their applicability". This is amplified when original problems are given in context.
- **Limitations / failure modes:** Human-made; small.
- **How to reuse with easy seed tasks:** Ask an LLM for minimal edits that invalidate the standard method, and require the answer to change (a leak check). Verify by brute force or search where possible.

### h1 — h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning (Motwani et al., 2025)
Link: https://arxiv.org/abs/2510.07312 · code: github.com/AlesyaIvanova/h1
- **Mechanism:**
  - Atomic GSM8K problems are chained: y1 = f1(x1), x2 = φ1(y1), y2 = f2(x2), and so on. The adapters φ are identity or simple deterministic transforms such as scaling or unit conversion.
  - All sub-problems appear in one prompt, and only the final answer is rewarded.
  - Training uses Dr. GRPO with a curriculum over horizon h: 200 steps per horizon, up to Len-5. Evaluation goes up to length 8.
- **How it makes tasks harder:** Longer dependency chains that require state tracking.
- **Correctness / verification:** Atomic answers are known labels and adapters are deterministic, so the final answer is exact by construction.
- **Difficulty control:** Horizon h and the curriculum schedule. The theory shows curriculum RL with outcome rewards has exponentially better sample complexity than training directly at full horizon.
- **Reported results:**
  - In-domain (current version, Len-5 curriculum vs instruct model): L-2 58.51 (+66.9%), L-3 36.39 (+81.3%), L-5 9.82 (+175.1%).
  - Only-L1, Uniform-Mix and Only-Long baselines, at equal compute, give no long-horizon improvement.
  - Out of distribution: MATH-500 69.20 (+7.8%), GSM-Symbolic P1 73.28, P2 52.00 (+20.7%), AIME24 avg@32 10.52 (+106.3%, from 5.10), AIME25 3.02.
  - Beats standard GSM8K RLVR even at pass@128. Also transfers to LongBench-v2 and Hash-hop.
- **Limitations / failure modes:** Problems get longer rather than conceptually deeper; adapters are simple.
- **How to reuse with easy seed tasks:** The most direct fix for "seeds too easy": chain k labelled seeds and increase k when the pass rate crosses a threshold. No new labels are needed.

### D. Teacher-driven hard-problem generators

### PromptCoT and PromptCoT 2.0 — PromptCoT 2.0: Scaling Prompt Synthesis for Large Language Model Reasoning (Zhao et al., 2025); builds on PromptCoT: Synthesizing Olympiad-level Problems for Mathematical Reasoning in Large Language Models (Zhao et al., 2025)
Links: https://arxiv.org/abs/2509.19894 · https://arxiv.org/abs/2503.02324 · code: github.com/inclusionAI/PromptCoT (2.0), github.com/zhaoxlpku/PromptCoT (1.0)
- **Mechanism:**
  - **PromptCoT 1.0:** Llama-3.1-70B-Instruct extracts concepts from seed prompts. A "designer rationale" is generated for each problem, and a generator is trained on concept–rationale–problem triples. Only pairs rated "perfect" by two independent LLM evaluators are kept (factual accuracy and solvability Perfect, no Bad ratings).
  - The theory: an optimal rationale maximizes both p(rationale | concepts) and p(problem | rationale, concepts).
  - **PromptCoT 2.0** replaces the heuristics with EM. In the E-step, the rationale model draws 8 candidates per (concepts, prompt) pair, scores them with R = log p(x|z,c) + log p(z|c), and is SFT-updated on the best one (best-of-8 selection, not policy-gradient RL). In the M-step, the prompt generator trains on triples using the latest rationales.
- **How it makes tasks harder:** Rationale-first design produces deliberately structured problems. Harder prompts emerge without explicit difficulty filters.
- **Correctness / verification:** For Qwen3-30B-A3B self-play, math labels are a majority vote over 8 generations of Qwen3-30B-A3B-Thinking-2507. Code gets 3–4 Qwen3-32B unit tests. Self-play excludes problems the model solved in at least 4 of 8 attempts. SFT uses GPT-OSS-120B trajectories.
- **Difficulty control:** Implicit, measured post hoc (Table 5): Qwen2.5-72B-Instruct solves 18.5% of PromptCoT 2.0 prompts and needs 37,373 GPT-OSS-120B tokens on average.

  | Dataset | Qwen2.5-72B-Instruct accuracy |
  |---|---|
  | PromptCoT 2.0 | 18.5% |
  | OpenThoughts3 | 21.3% |
  | PromptCoT 1.0 | 24.7% |
  | OpenMathReasoning | 28.9% |
  | OpenR1 | 32.3% |
- **Reported results:**
  - PromptCoT 1.0 scaled to 905K problems. PromptCoT-DS-7B reaches 60.0 AIME24, vs 43.3 for R1-Distill-Qwen-7B.
  - PromptCoT 2.0 self-play on Qwen3-30B-A3B-Thinking-2507: AIME24 87.7 → 92.1, AIME25 85.0 → 89.8, HMMT Feb25 71.4 → 76.7, LiveCodeBench v5/v6 +6.1/+5.0.
  - SFT of Qwen2.5-7B-Instruct on synthetic prompts only (4.77M trajectories): AIME24 12.8 → 73.1, AIME25 8.0 → 65.6, HMMT Feb25 2.7 → 46.5.
- **Limitations / failure modes:** Majority-vote labels are weakest at the hardest items. Needs a strong teacher.
- **How to reuse with easy seed tasks:** Make the generator write a design rationale (concepts, intended difficulty sources, solution path) before the problem. Train it on rationale-annotated seeds, or reuse the released prompts as a hard pool.

### SAND-Math — SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers (Manem et al., 2025)
Link: https://arxiv.org/abs/2507.20527 (NeurIPS 2025 MATH-AI workshop)
- **Mechanism:** DeepSeek-R1 generates new problems from scratch together with independent solutions. **Difficulty Hiking** rewrites a problem to require a same-branch theorem combined with a cross-domain concept drawn from a math taxonomy.
- **How it makes tasks harder:** Theorem and concept injection.
- **Correctness / verification and filtering:**
  - Self-consistency: all k answers must agree (23,437 → 17,578).
  - Performance filter: keep only questions a solver (Qwen2.5-32B-Instruct) gets wrong (9,211, 56.6% retained).
  - Llama-3.3-70B judge: 1–10 difficulty ratings (AoPS-guided rubric, averaged over 3 runs) and decontamination.
  - semhash dedup at 0.99.
  - Novelty filter: each question is used as a web-search query (SearXNG), and questions with similarity >0.85 to the top 10 results are removed (4%). Final set: 8,842.
- **Difficulty control:** Solver failure plus the judge's rating.
- **Reported results:** 500 SAND-Math samples added to a strong SFT baseline beat the next-best synthetic dataset by 17.85 absolute points on AIME25. One hiking step raised mean difficulty from 5.02 to 5.98, AIME25 from 46.38 to 49.23, and the overall average from 72.94 to 74.39.
- **Limitations / failure modes:** All-k agreement removes hard-but-correct items. Labels come from a single teacher family.
- **How to reuse with easy seed tasks:** Apply one "hiking" rewrite per seed (same-branch theorem plus cross-domain concept). Keep items your policy fails and multiple independent teachers agree on.

### MathSmith — MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy (Zhan et al., 2025)
Link: https://arxiv.org/abs/2508.05592 (AAAI 2026) · code: github.com/Jasaxion/MathSmith
- **Mechanism:**
  - About 11,000 concept–explanation pairs are summarized from PlanetMath. Each input samples five concepts, so problems are built from scratch, which limits contamination.
  - The generator (Qwen3-8B, cold-started on GPT-4o data) writes an exactly five-step construction rationale, then the problem. It must use at least two of nine strategies: multi-step, cross-topic, implicit/reverse logic, distractors, abstract modeling, multiple solution paths, advanced manipulation, extreme/boundary conditions, non-standard representation.
  - It is then trained with GRPO on a composite reward: structural validity + α·complexity (the reasoning-trace length of teacher Qwen3-30B-A3B) + β·consistency (1 if a majority exists among K=5 teacher answers).
- **How it makes tasks harder:** Strategy constraints plus RL toward longer teacher reasoning.
- **Correctness / verification:** The majority-exists consistency reward. MathSmith-Hard (complexity only) vs MathSmith-HC (with consistency) isolates its effect.
- **Difficulty control:** Trace-length reward and strategy choice. A weakness-focused module generates 10 concept-linked variants per failed question.
- **Reported results:**
  - Hard-benchmark relative improvements of 9.8–18.1%: Qwen3-8B short-CoT 36.6 (+12.3%), R1-Distill-Qwen-7B long-CoT 51.0 (+15.6%), Qwen3-8B long-CoT 71.8 (+9.8%).
  - Available ratio (well formatted and solvable): SFT 71.50%, Hard 84.92%, HC 95.38%.
  - Weakness loop on Qwen2.5-Math-1.5B (easy+medium / hard): original 38.2 / 14.5; epoch 1 69.9 / 18.8 (random baseline 69.4 / 15.6); epoch 3 77.6 / 21.6.
- **Limitations / failure modes:** Trace length can be gamed by verbose or ambiguous problems. A majority is not the same as correctness.
- **How to reuse with easy seed tasks:** Train a small generator with reward = validity × consensus-exists × normalized solver trace length, require explicit difficulty strategies, and keep the consistency term. It is what lifts usability.

### ScaleDiff — ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning (Pei et al., 2025)
Link: https://arxiv.org/abs/2509.21070 · code: github.com/QizhiPei/ScaleDiff
- **Mechanism:**
  - AdaptThink (THU-KEG/AdaptThink-7B-delta0.05) is an RL-trained model that chooses between Thinking and NoThinking. A problem counts as "difficult" if its first generated token is not `</think>`: one forward pass, no solving. On the 558K AM-Qwen3-Distilled problems this gives a 192K difficult subset.
  - DiffGen-8B is trained only on difficult problems and samples new ones. Qwen3-8B (thinking mode) distills solutions.
- **How it makes tasks harder:** The generator imitates the distribution of hard problems.
- **Correctness / verification:** Rule filters, plus removal of problems the base model already solves (its answer matches the teacher's). About 43% are filtered. Labels are single-sample teacher answers.
- **Difficulty control:** About 88% of DiffGen outputs are classified as difficult.
- **Reported results:**
  - 1.15M generated pairs; ScaleDiff-Math totals 1.7M.
  - Qwen2.5-Math-7B-Instruct averages 65.9% on AIME24/25, HMMT-Feb25, BRUMO25 and MATH500, +11.3% relative over training on the original data.
  - Ablation: the 192K difficult subset scores 56.6 vs 59.2 for all 558K, while a random 192K subset scores 45.1.
- **Limitations / failure modes:** Unverified single-sample labels are acceptable for SFT only.
- **How to reuse with easy seed tasks:** Label your pool cheaply (think-mode trigger, response length, fail-rate regressor). Train a generator on the hard slice only, then filter with your policy.

### MindLoom — MindLoom: Composing Thought Modes for Frontier-Level Reasoning Data Synthesis (Shen et al., 2026)
Link: https://arxiv.org/abs/2605.21630 · code: github.com/EachSheep/MindLoom
- **Mechanism:**
  - Views difficulty as an accumulation of atomic knowledge-reasoning transformations ("thought modes"). A thought mode has a summary, detail, general knowledge and specific knowledge, e.g. "Introduce definite integral computation over a bounded domain".
  - Hard problems with verified solutions are decomposed by an "operational inverse curriculum" into thought-mode chains.
  - A retrieval model (from Qwen3-Embedding-0.6B, trained with margin ranking) matches problem states to compatible modes.
  - New problems grow by iteratively applying retrieved modes to seed questions, with a similarity–scarcity trade-off to cover rare modes.
- **How it makes tasks harder:** Each applied mode adds a reasoning transformation, so difficulty accumulates.
- **Correctness / verification:** Three rollouts per question, each judged by an LLM (all-correct / partial / all-wrong). SFT uses only questions with at least one judged-correct response. All-wrong questions are excluded unless a thinking-mode rollout rescues them.
- **Difficulty control:** Number of modes applied; rollout-based difficulty labels.
- **Reported results:** Qwen3-4B-2507 pass@1 / pass@3, base → MindLoom:

  | Benchmark | Base | MindLoom |
  |---|---|---|
  | MATH-500 | 84.40 / 90.20 | 85.20 / 98.60 |
  | HMMT-Feb | 20.00 / 26.67 | 33.33 / 43.33 |
  | AIME25 | 30.00 / 33.33 | 40.00 / 53.33 |

  "Favorable" across 9 benchmarks, though the OpenThought baseline is higher on some math cells (e.g. AIME25 pass@1 43.33).
- **Limitations / failure modes:** LLM-judged labels. The exclusion rules drop the hardest items.
- **How to reuse with easy seed tasks:** Mine "what makes hard problems hard" from your best verified hard problems, then apply 1–k of these transformations to easy seeds. k is the knob.

### CoT-Self-Instruct — CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks (Yu et al., 2025)
Link: https://arxiv.org/abs/2507.23751
- **Mechanism:** Given few-shot seeds (893 verifiable s1k tasks), the LLM reasons about their domain, complexity and quality, plans a new task, writes it, and produces its own answer. Surviving prompts go to GRPO.
- **How it makes tasks harder:** Matches seed complexity through explicit planning. It is not explicitly a hardener.
- **Correctness / verification:** **Answer-Consistency:** discard an item if the solver's majority answer differs from the answer produced at generation time. The assumption is that such items are "either incorrectly labeled or too difficult". RIP is used for non-verifiable tasks.
- **Difficulty control:** Implicit, since the filter caps difficulty at the solver frontier.
- **Reported results:**
  - 2,926 filtered items average 57.2% on MATH500, AMC23, AIME24 and GPQA-Diamond. Compare s1k (R1 labels) 44.6%, 10K OpenMathReasoning 47.5%, and a Self-Consistency filter 55.1%.
  - Scaling to 10K gives 58.7%.
- **Limitations / failure modes:** Removes hard-but-correct items by design.
- **How to reuse with easy seed tasks:** Have generators emit answers and require agreement with an independent solver majority. This is cheap and strong for RL prompts at or below the frontier. Route disagreements to a separate relabelling pool rather than dropping them.

### Hypothesis-driven hard problem generation — Automatically Generating Hard Math Problems from Hypothesis-Driven Error Analysis (Fu et al., 2026)
Link: https://arxiv.org/abs/2604.04386 (ICLR 2026 workshop on logical reasoning)
- **Mechanism:**
  - Hypogenic (backbones GPT-4o-mini, GPT-4.1-mini or Qwen3-14B) generates natural-language hypotheses about which concepts drive a target model's (Llama-3.3-70B-Instruct) MATH failures, scored by predictive accuracy.
  - Hypotheses are built on concept taxonomies at extremely low, low, mid and high granularity, plus a baseline skill list from Shah et al.
  - Accurate hypotheses condition problem generation.
- **How it makes tasks harder:** Targets model-specific weak concept combinations.
- **Correctness / verification:** Filters remove invalid, incorrect, incomplete and ambiguous items; proofs are excluded. Answer keys come from DeepSeek-R1, o3 and GPT-5, checked by cross-model agreement and manual validation.
- **Difficulty control:** Hypothesis accuracy and taxonomy granularity.
- **Reported results:** Hypothesis accuracy correlates with difficulty. Problems from the most accurate low-granularity hypotheses cut the target's solve rate to as low as 45%, vs 77% on MATH. The effect is non-monotonic in granularity: baseline and mid taxonomies stayed at or above 77%, and redundant categories hurt.
- **Limitations / failure modes:** Evaluated as a difficulty generator, not as large-scale training data.
- **How to reuse with easy seed tasks:** Periodically fit readable failure hypotheses on your policy's rollouts and condition RL item generation on them.

### E. Policy-aware RL loops, self-play and trained generators

### SwS — SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning (Liang et al., 2025)
Link: https://arxiv.org/abs/2506.08989 (NeurIPS 2025, per official repo) · code: github.com/MasterVito/SwS
- **Mechanism:** A preliminary RL run records per-problem accuracy. A problem counts as a weakness if the model **never reaches 50% accuracy in any epoch *and* its accuracy trend has negative slope**. Concepts from failures are extracted per category and recombined (co-occurrence and embedding similarity). An LLM then synthesizes new problems, which are added to RL.
- **How it makes tasks harder:** New problems target the concept clusters where the policy stagnates.
- **Correctness / verification:** Skywork-OR1-Math-7B (up to 7B) or QwQ-32B (32B) labels answers by self-consistency. Only problems with at least 50% consistent answers are kept.
- **Difficulty control:** The trained policy's accuracy on each synthetic problem must fall in [25%, 75%].
- **Reported results:** 40K synthetic problems per model. Average gains of +10.0% (7B) and +7.7% (32B) across 8 benchmarks. SwS-7B AIME24 reaches 26.7 Avg@1 vs 10.0 for the same RL without synthesis (Avg@32 18.3 vs 14.5).
- **Limitations / failure modes:** Needs a preliminary run; majority labels.
- **How to reuse with easy seed tasks:** Log per-prompt accuracy during your RL run. The stagnating prompts become the concept seeds; synthesize and band-filter at 25–75%.

### SvS — Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR (Liang et al., 2025)
Link: https://arxiv.org/abs/2508.14029 · code: github.com/MasterVito/SvS
- **Mechanism:** During RLVR, underperforming problems (group accuracy 12.5–50%) are selected. The policy uses its own correct solution to write "variational problems" that keep the reference answer. The policy is also trained on the synthesis task, with a positive reward only if a variant's accuracy falls in 12.5–62.5%. Only variants with mixed correct and incorrect solutions enter training.
- **How it makes tasks harder:** Mainly diversification at the frontier, with structure and description changed and the answer fixed.
- **Correctness / verification:** The answer is invariant, so no new labels are needed. The accuracy band serves as the validity proxy.
- **Difficulty control:** Band selection of seeds and a band reward for synthesis.
- **Reported results:** +18.3 and +22.8 absolute pass@32 on AIME24/25, where standard RLVR shows little improvement. Entropy is preserved. Tested at 3B–32B on 12 benchmarks.
- **Limitations / failure modes:** Answer invariance limits how far a variant can move from its seed.
- **How to reuse with easy seed tasks:** An online self-augmentation step inside GRPO when prompts start saturating. It keeps pass@k and entropy from collapsing.

### MathForge (MQR + DGPO) — Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation (Dai et al., 2026)
Link: https://arxiv.org/abs/2601.20614 (ICLR 2026) · code: github.com/AMAP-ML/MathForge
- **Mechanism:** MQR has a reformulator (default o3) rewrite each training question in three ways under the rule "The final answer MUST remain unchanged":
  - **Background:** add a story that seems related but is not.
  - **Term:** invent an abstract term for a central concept and restate the problem with it.
  - **Sub-Problem:** turn a key numerical condition into an independent sub-problem from any branch whose answer is that value.

  DGPO changes the optimizer. DGAE normalizes group advantages by mean absolute deviation instead of standard deviation. DQW weights questions by λ_s = B_v·softmax(D_s/T), where D_s = −mean reward and T = 2.0.
- **How it makes tasks harder:** Noise filtering, abstraction, and nested computation. The label is unchanged.
- **Correctness / verification:** Answer preservation. An o3 equivalence audit on 100 samples per type found 99% (Background), 97% (Term) and 97% (Sub-Problem) equivalent. A failed rewrite yields all-incorrect groups, so under GRPO it produces no gradient rather than a harmful one.
- **Difficulty control:** Aspects can be stacked, and DQW focuses updates on hard questions.
- **Reported results:**
  - Qwen2.5-Math-7B on MATH: GRPO 37.61 → DGPO 39.79 → MQR 41.04 → MathForge 42.17.
  - Qwen2.5-Math-1.5B: 29.39 → 33.84. DeepSeek-Math-7B: 14.91 → 17.77.
  - Weaker reformulators (Qwen2.5-7B-Instruct, Qwen3-30B-A3B-Thinking) also help.
- **Limitations / failure modes:** Preservation is checked by an LLM. Nested sub-problems must evaluate exactly to the replaced constant.
- **How to reuse with easy seed tasks:** The cheapest RL-ready hardener for saturated seeds. Nest each given as a sub-problem, verify the sub-answer by execution, and add narrative noise and new terminology. The gold label is unchanged.

### QbQ — Question Begets Question: Self-Evolving Curriculum for Reinforcement Fine-Tuning on Competition Mathematics (Bao et al., 2026)
Link: https://arxiv.org/abs/2608.01522
- **Mechanism:** A teacher transforms seeds with five structural operators:
  - **generalize-then-specialize**, e.g. trailing zeros in another base;
  - **parametrize-and-sum**;
  - **change the queried quantity**, e.g. inradius instead of area, which still needs Heron's formula;
  - **inverse problem**, fixing the target and solving for a parameter;
  - **add a constraint layer**, one extra routine reduction.

  A planning call picks the three most suitable operators per seed. Each round evaluates the checkpoint with k=16 rollouts and sorts problems into mastered (16), mostly-right (8–15), sometimes-right (1–7) and never (0). Only the mostly-right band seeds new variants. Training is GRPO on statements plus final answers only, never teacher traces.
- **How it makes tasks harder:** Operators "probe complementary aspects" of a skill. They do *not* systematically increase difficulty; the curriculum drives difficulty.
- **Correctness / verification:** Teacher-generated answers. Every variant must have an answer different from its parent, which rejects copies and surface rewrites.
- **Difficulty control:** Seed band and per-round re-evaluation. Mastered problems leave the curriculum.
- **Reported results:** On Qwen2.5-Math-7B (AIME 2025/2026 pass@1):

  | Setting | Pass@1 |
  |---|---|
  | Base | 5.6% |
  | SFT | 9.5% |
  | GRPO | 11.5% |
  | Static real+synthetic | 12.5% (cap) |
  | Non-curriculum QbQ | 14.5% (cap) |
  | Curriculum QbQ | 16.5% (16.46±0.45), no saturation after 20 rounds |
  | Seeding from hardest failures | 11.36% |
- **Limitations / failure modes:** Small absolute numbers; teacher labels are not audited.
- **How to reuse with easy seed tasks:** Seed complexification from prompts the policy *mostly* solves, re-bucket every round, and use the five operators as the prompt menu.

### R-Zero — R-Zero: Self-Evolving Reasoning LLM from Zero Data (Huang et al., 2025)
Link: https://arxiv.org/abs/2508.05004 (ICLR 2026, per official repo) · code: github.com/Chengsong-Huang/R-Zero
- **Mechanism:** One base model initializes a Challenger and a Solver.
  - The Challenger is trained with GRPO on r = 1 − 2|p̂ − ½|, where p̂ is the Solver's self-consistency over m=10 samples, minus a BLEU-clustered repetition penalty (τ_BLEU = 0.5).
  - The Solver trains with GRPO on Challenger questions whose majority count is 3–7 out of 10 (|p̂ − ½| ≤ 0.25), using the majority as the pseudo-label.
- **How it makes tasks harder:** The Challenger chases the Solver's 50% boundary.
- **Correctness / verification:** Solver majority vote only.
- **Difficulty control:** Uncertainty reward plus band filter.
- **Reported results:**
  - Abstract: Qwen3-4B-Base math +6.49 (general +7.54). The current main table shows 42.57 → 49.93.
  - Oracle check (GPT-4o on 200 sampled questions): pseudo-label accuracy fell from 79.0% to 69.0% to 63.0% over iterations 1–3.
  - Iteration table: 48.06 → 48.44 → 49.12 → 46.52 at steps 15/30/45/60, with pseudo-label accuracy 71.0 → 56.2 → 48.8 → 42.2%. Smaller models peak earlier (0.6B at iteration 1).
- **Limitations / failure modes:** Label drift and collapse. No validity gate.
- **How to reuse with easy seed tasks:** Reuse the uncertainty reward and repetition penalty. Add an independent validity check and monitor label accuracy on a gold slice.

### Socratic-Zero — Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution (Wang et al., 2025)
Link: https://arxiv.org/abs/2509.24726
- **Mechanism:** Starts from 100 seed questions (7 MATH subjects, levels 2–4) with three agents:
  - A fixed Teacher (Qwen3-235B-A22B) evaluates and refines problems. For learning-zone problems it targets the specific error in a failed solution. For mastered problems it builds harder variants.
  - A Solver trains with DPO on its own successful vs failed trajectories.
  - A Generator (Qwen3-32B) distills the Teacher's refinements by utility-weighted SFT, with a Gaussian utility U = exp(−(s−μ)²/2σ²) over Solver success s, σ = 0.2, peaked near 50%.
- **How it makes tasks harder:** Failure-conditioned refinement plus harder variants of mastered items.
- **Correctness / verification:** Teacher reference solutions, rule-based answer extraction (MathRule) and an LLM judge.
- **Difficulty control:** Zone routing and the Gaussian utility.
- **Reported results:**
  - Socratic-Solver-8B averages 56.1% on 7 benchmarks, +20.2 points over the baseline and +15.4 over static augmentation. AIME24 +19.1, AIME25 +16.5.
  - Socratic-Generator-32B has a 95.6% validity rate (solvable by Qwen3-235B-A22B-Instruct-2507). A student trained on its data reaches 37.72%, vs 37.13% on data from Qwen3-235B-A22B itself.
- **Limitations / failure modes:** Correctness depends on the Teacher.
- **How to reuse with easy seed tasks:** Route by pass rate. Mastered seeds get "harder variant" prompts and failing seeds get "target this error" prompts. Then distill the teacher's edits into a cheap generator.

### VHG — Verifier-Backed Hard Problem Generation for Mathematical Reasoning (Lai et al., 2026)
Link: https://arxiv.org/abs/2605.06660
- **Mechanism:** Three-party self-play. The setter's reward is R_Q = 1[V(x,y*)=1]·(1 − Acc_S(x,y*)), and the solver's reward is its accuracy on verifier-accepted pairs. Two verifiers:
  - Hard verifier: SymPy checks that the proposed antiderivative differentiates to the integrand.
  - Soft verifier (general math): rule filters plus an LLM judge.
- **How it makes tasks harder:** The setter earns difficulty credit only for valid problems.
- **Correctness / verification:** Independent verifier gate before any difficulty credit.
- **Difficulty control:** Solver accuracy, applied after validity.
- **Reported results:**
  - Integrals, pass@1 (base → VHG):

    | Benchmark | Base | VHG | Vanilla GRPO | R-Zero (iterations 1–3) |
    |---|---|---|---|---|
    | Competition | 28.8 | 45.4 | 38.8 | 30.5–31.9 |
    | Qualifier | 52.5 | 69.4 | 66.5 | – |
    | Stress | 43.3 | 64.7 | 60.3 | 51.0–52.9 |

  - General math overall: 69.01, vs vanilla GRPO 67.62, R-Zero 65.61–66.20, base 56.79.
  - Setter dynamics: over steps 0→50 the valid rate rises 30.6% → 65.2%. Then the solver pass rate on valid samples falls to 17.6%, the valid rate ends at 75.5%, and the valid-and-hard share rises 27.5% → 58.5%.
- **Limitations / failure modes:** Needs a trustworthy verifier; the soft verifier is weaker.
- **How to reuse with easy seed tasks:** Multiply any "make it harder" reward by an independent validity indicator. Prefer domains with inverse-operation checks: integrals by differentiation, equations by substitution, factorizations by expansion.

### Learning to Pose Problems (added) — Learning to Pose Problems: Reasoning-Driven and Solver-Adaptive Data Synthesis (Wei et al., 2025)
Link: https://arxiv.org/abs/2511.09907 · code: github.com/WalkerWorldPeace/Reasoning-Synthesis
- **Mechanism:**
  - **Cold start:** mine related problem pairs from human multi-part questions, where a later sub-question is the concrete target. A reasoning model reconstructs the latent "problem-design CoT" that links the pair, and the generator learns to plan before posing.
  - **RL:** the generator proposes problems from seeds. The solver answers each several times, and majority voting gives pseudo-labels and accuracy a_new. Reward: R_acc = 1 − |a_new − (1 − a_ori)| + min(a_new, 1 − a_new).
- **How it makes tasks harder:** The reward pushes toward *complementary* difficulty: harder variants of seeds the solver finds easy, easier ones for seeds it fails. The second term favours about 50%.
- **Correctness / verification:** Solver majority-vote pseudo-labels (weak).
- **Difficulty control:** Seed-relative accuracy reward.
- **Reported results:** +3.4% cumulative average over 10 math and general reasoning benchmarks, and +2.17% over preference-reward-model methods. The full method gives the best average on Qwen3-4B (49.23%) and Qwen3-8B (52.45%).
- **Limitations / failure modes:** Pseudo-label noise.
- **How to reuse with easy seed tasks:** Condition the generator's reward on the *seed's* pass rate, so saturated seeds automatically get hardened.

### PROPEL (added) — Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier (Wolf et al., 2026)
Link: https://arxiv.org/abs/2606.18284
- **Mechanism:** Training a generator with RL toward a target solve rate needs many solver rollouts per candidate. PROPEL trains a lightweight **activation probe** once, on a labelled corpus of generated tasks and solver outcomes. The probe predicts target-solver pass rate from a frozen reference generator's activations and replaces live solver trials as the reward proxy.
- **How it makes tasks harder:** Moves the generator's output distribution toward the learnable frontier.
- **Correctness / verification:** A task validity predicate (task-specification rules for math) and oracle scoring of outputs.
- **Difficulty control:** Target solve rate via the probe. KL regularization trades utility against topic collapse; the math run fully collapsed at the lowest KL.
- **Reported results:** Roughly doubles the rate of frontier tasks, with fewer than half the solver trials. For coding: 10.1% → 20.0% (Qwen2.5-3B-Instruct solver) and 5.3% → 12.6% (7B). For SWE: 9.8% → 19.6%. Math also shows gains in post-oracle conditional yield.
- **Limitations / failure modes:** The probe drifts as the policy moves. Diversity loss needs mitigation.
- **How to reuse with easy seed tasks:** When solver rollouts dominate generator-training cost, fit a probe on logged (task, pass-rate) pairs and use it as a cheap difficulty reward, re-anchoring it periodically with real rollouts.

### Agentic Proposing — Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis (Jiao et al., 2026)
Link: https://arxiv.org/abs/2602.03279 · code: github.com/Frostlinx/Agentic-Proposing
- **Mechanism:**
  - A teacher (Qwen3-235B-Instruct-2507) extracts atomic problem-construction skills into a library.
  - Agentic-Proposer-4B composes skills in a Draft → Check → Refine → Finalize loop with sandboxed tool calls. It is trained by agentic SFT and then Multi-Granularity Policy Optimization (MGPO), which combines trajectory-level validity/difficulty rewards with step-level rewards for successful tool executions and coherent reflections.
- **How it makes tasks harder:** Sequential skill composition. Weak skill categories are sampled with probability inversely proportional to verifier-validated proficiency.
- **Correctness / verification:** A verifier ensemble (Qwen3-235B-Thinking, DeepSeek-V3.2, GPT-OSS-120B) plus code checks. Invalid problems get zero total reward.
- **Difficulty control:** A prober estimates ρ(q) = Pass@16, and the difficulty bonus is paid only for solvable instances.
- **Reported results:** A 4B solver trained on 10,000 synthetic trajectories beats established reasoning datasets across math, science and coding. A 30B solver trained on 11,000 trajectories reaches 91.6% on AIME25.
- **Limitations / failure modes:** Ensemble cost; depends on the quality of the skill library.
- **How to reuse with easy seed tasks:** Annotate skills with their "difficulty effect", then RL-train a small proposer to compose 2–4 skills under a validity gate and a pass@k band.

### Code2Math — Code2Math: Can Your Code Agent Evolve Math Problems Through Exploration? (Guo et al., 2026)
Link: https://arxiv.org/abs/2603.03202 · code: github.com/TarferSoul/Code2Math
- **Mechanism:**
  - An Evolution Agent finds a seed's key idea and uses code (construction search, counterexample pruning, parameter search: 43.3% of executions; verification and edge cases: 34.2%) to design a variant whose entry point is "deliberately obscured".
  - A Solvability Verification Agent checks statement and solution logic.
  - A Difficulty Verification Agent scores 1–5. Score 1 means an unchanged solution path; score 2 means difficulty "caused merely by computational tedium". Both are rejected.
- **How it makes tasks harder:** Raises the "burden of discovery" rather than the amount of computation.
- **Correctness / verification:** Code-verified constructions, a solvability agent, and a GPT-5.2-High external judge. Against humans the judge had 99.2% validity precision (126/127), agreed on 1,134 of 1,226 answer judgments, and matched difficulty scores exactly 79.3% of the time.
- **Difficulty control:** Rubric threshold plus solve-rate drops.
- **Reported results:**
  - Acceptance by evolver: Gemini-3-Pro 98/98, DeepSeek-Reasoner 94/98, Seed-2.0-Pro 83/97, DeepSeek-Chat 83/94, Kimi-K2-Thinking 74/90.
  - GPT-5.2-High falls from 70% to 64–61% on evolved problems; Gemini-3-Flash-Thinking falls by up to 32 points.
  - Cost: 1.56 (Gemini) to 6.55 (Kimi) failed rollouts per success.
  - SFT with GLM-5.3-Flash evolving 2,500 problems: Qwen3-8B 39.9 → 44.4 and Qwen3-14B 55.6 → 60.2 average on AIME24/25/26.
- **Limitations / failure modes:** Expensive; needs frontier agents.
- **How to reuse with easy seed tasks:** Reserve it for the top slice of seeds where cheap operators stop adding difficulty. Keep the anti-tedium rubric as a filter for every other operator too.

### F. Curated hard pools and format hardening

### Big-Math — Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models (Albalak et al., 2025)
Link: https://arxiv.org/abs/2502.17387 · code: github.com/SynthLabsAI/Big-Math
- **Mechanism:**
  - 643,374 source problems are deduplicated and filtered with regex plus Llama-3.1-70B filters developed iteratively with humans until at least 90% precision and recall (MCQ filter over 98% recall).
  - Removed: MCQ (the largest category, over 80% from cn_k12), yes/no, true/false, multi-part and proof items.
  - Solvability: at least one correct answer from 64 Llama-3.1-8B rollouts or 5–8 405B rollouts.
  - **Big-Math-Reformulated:** a 4-step Llama-3.1-405B process rewrites MCQs as open-ended questions (88,983). Items are kept only if solved at least once but not 100% by 8B (48,698), leaving 47,010 after all filters.
- **How it makes tasks harder:** Removing options and guessable formats raises effective difficulty.
- **Correctness / verification:** Closed-form, uniquely verifiable answers; human-validated filters.
- **Difficulty control:** Solve-rate quintiles from 64 rollouts.
- **Reported results:** 251,122 problems, 36.50% in the hardest quintile (under 20% solve rate). The reformulated set is harder: 34.44% in the hardest quintile plus 16.42% in the second, over 50% combined. Reformulation recovered 63.4% of amc_aime MCQs, which made up 72.7% of that subset.
- **Limitations / failure modes:** Difficulty is relative to Llama-3.1-8B; no new problems.
- **How to reuse with easy seed tasks:** Before any complexification, remove guessable formats and convert MCQs to open answers. Pass rates that look saturated may be inflated by guessing.

### DeepMath-103K — DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning (He et al., 2025)
Link: https://arxiv.org/abs/2504.11456 · code: github.com/zwhe99/DeepMath
- **Mechanism:**
  - A 2,869K raw pool: MathStackExchange subsets chosen for harder distributions, plus NuminaMath-CoT.
  - Decontamination: paraphrase-multilingual-MiniLM retrieves the top-5 nearest benchmark items, and a Llama-3.3-70B judge checks them, against a suite including MATH, AIME, AMC, Minerva, OlympiadBench, Omni-MATH, GAOKAO, JEEBench, MMLU-STEM, GSM8K and GPQA.
  - Difficulty: GPT-4o rates each problem six times (AoPS-guided) and level ≥5 is kept.
  - GPT-4o standardizes questions to a single numeric or symbolic answer.
- **How it makes tasks harder:** Mining and filtering, not complexification.
- **Correctness / verification:** Three DeepSeek-R1 solutions plus the source answer (when available) must all agree under a rule-based extractor.
- **Difficulty control:** The rating filter. 95K items at level ≥5 plus 8K SimpleRL items give 103K.
- **Reported results:** Raw-pool contamination: 90% (AIME24, AMC23), 76.6% (MATH500), 35.7% (Minerva), 33.6% (OlympiadBench). RL-Zero on Qwen2.5-Math-7B: AIME24 11.2 → 34.2 and AIME25 4.4 → 23.5 (pass@1 averaged over 16).
- **Limitations / failure modes:** Agreement filtering drops items R1 cannot solve.
- **How to reuse with easy seed tasks:** Use it as a hard seed pool. Copy the embedding + LLM-judge decontamination and the multi-solution agreement gate for your own synthetic sets.

### OpenMathReasoning (AIMO-2) — AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset (Moshkov et al., 2025)
Link: https://arxiv.org/abs/2504.16891 · code: NeMo-Skills
- **Mechanism:**
  - Qwen2.5-32B-Instruct extracts problems from AoPS forums (all except Middle School Math) and classifies them as proof, MCQ, binary or invalid. MCQ, binary and invalid are removed.
  - **Proofs are converted into answer-based questions "that require similar problem-solving techniques."**
  - Answers are extracted from forum posts where possible. LLM-based decontamination follows.
  - Pipeline: 620K discussions → 580K extracted → 550K → 540K (260K converted proofs, 190K with an extracted answer).
  - Solutions: 3.2M long CoT (DeepSeek-R1, QwQ-32B) and 1.7M TIR, built iteratively (700K → 260K in the first TIR round). Harder problems, by Qwen2.5-Math-72B-Instruct pass rate over 32 samples, get more solutions.
- **How it makes tasks harder:** Turns unverifiable olympiad proofs into verifiable RL/SFT items, and a harder second SFT stage drops problems with pass rate above 0.3.
- **Correctness / verification:** Forum answers where present; otherwise the most common candidate answer.
- **Difficulty control:** Pass-rate prioritization.
- **Reported results:** First place in AIMO-2 (34/50 on the private set). Successor Nemotron-Math (arXiv 2512.15489) combines 85K AoPS problems with 262K StackExchange-Math problems (filtered from 651K) and provides 7.5M traces across high, medium and low reasoning modes, with and without Python TIR.
- **Limitations / failure modes:** Conversion can lose difficulty. Majority answers can be wrong.
- **How to reuse with easy seed tasks:** Convert "show that" seeds into answer-bearing questions (an extremal value, count or parameter implied by the proof) to make them RLVR items.

---

## Complexification operators from this area

Every worked answer below was checked by brute-force computation.

### 1. Serial composition / chaining
- **What it does:** Feeds the answer of problem i into problem i+1, directly or through a deterministic adapter. Success roughly multiplies across hops (minus a reasoning gap), and horizon length is an explicit curriculum knob.
- **Easy → hard:**
  - Easy: "A bakery sells 12 muffins per hour for 5 hours. How many muffins?" (60)
  - Hard: "…Let X be that number. Muffins are packed 4 per box; let Y be the number of boxes. Each box costs $8; let Z be the total cost. A 15% discount applies to Z. What is paid?" (15 boxes, $120, **$102**)
- **Keep it verifiable:** Keep each seed's code-form solution, substitute the upstream value, and execute (Compositional GSM). Keep adapters deterministic and integral (h1). Alternatively, verify each easy hop with a solver ensemble (CHASE-Math). Reward only the final answer.
- **Sources:** Compositional GSM, h1, CHASE-Math, MathFusion (sequential).

### 2. Inversion / backward reasoning
- **What it does:** Masks a given, reveals the original answer, and asks for the given (FOBAR/SV). Alternatively, fixes the target and asks for the parameter that achieves it (QbQ "inverse problem"; GSM-Infinite reverse mode). This turns forward evaluation into equation solving.
- **Easy → hard:**
  - Easy: "James buys 5 packs of beef, 4 lb each, at $5.50/lb. How much does he pay?" ($110) → "…buys x packs… pays $110. Find x." (5)
  - Competition: "How many trailing zeros does 100! have?" (24) → "Find the smallest n such that n! ends in exactly 24 zeros." (**100**)
- **Keep it verifiable:** The masked value is the label, but **check uniqueness**. "Find n such that n! ends in exactly 24 zeros" has five solutions (100–104), so the question must ask for the smallest one or a count. Brute-force or solve over the stated domain. Discard nonlinear blow-ups (GSM-Infinite discards quadratic reverse cases).
- **Sources:** MetaMath, GSM-Infinite, QbQ.

### 3. Skill / concept-pair composition (parallel or conditional fusion)
- **What it does:** Requires 2–3 skills, key points or concepts in one problem, chosen by a co-occurrence prior or an embedding pair. Or it fuses two analogous problems (parallel) or asks for a comparison or selection (conditional). Difficulty follows roughly p^k (MATH² square law).
- **Easy → hard:**
  - Easy: "How many positive divisors does 360 have?" (24)
  - Hard: "A triangle has vertices (0,0), (d,0), (0,k), where d is the number of positive divisors of 360. Find the smallest positive integer k making its area a perfect square." (area 12k, **k = 3**)
- **Keep it verifiable:** Compose executable pieces (RV-Syn function graphs, EFAs) so the answer is executed. Otherwise use SymPy-equivalence voting (KPDDS) and cross-family agreement. Use a co-occurrence prior (TCPM) to avoid incoherent pairs.
- **Sources:** MATH², KPDDS, MathScale, MathFusion, MathMixup, SwS.

### 4. Computation-graph / constraint expansion (symbolic mutation)
- **What it does:** Represents the problem as an executable graph or symbolic program and grows it: more ops or nodes, coupled or nested equations, nonlinear constraints. Then renders it to text. Difficulty is an integer knob.
- **Easy → hard:**
  - Easy: "Ann has 5 apples; Tom has 3 more. How many does Tom have?" (8)
  - Hard: "Tom has 3 more than twice Ann's apples. Ben has 4 times as many pears as Ann has apples. Cara has 2 fewer plums than Tom has apples. Together, Ann's apples, Tom's apples, Ben's pears and Cara's plums total 58. How many apples does Tom have?" (a = 6, **t = 15**)
- **Keep it verifiable:** Solve with SymPy/Z3 or execute before rendering. Check the text's faithfulness by cycle consistency (MathCAMPS) or by requiring a CoT to reproduce the executed answer (RV-Syn). Enforce a unique integer solution.
- **Sources:** GSM-Infinite, iGSM, RV-Syn, Yeo et al., MathCAMPS, MathAgent.

### 5. Distractor / irrelevant-information injection
- **What it does:** Adds clauses, narrative or noise nodes that look relevant but cannot change the answer. This trains information selection.
- **Easy → hard:**
  - Easy: "Oliver picks 44 kiwis on Friday, 58 on Saturday, and on Sunday double Friday's number. How many kiwis?" (190)
  - Hard: the GSM-NoOp version adds "…but five of them were a bit smaller than average." (still **190**)
- **Keep it verifiable:** Only attach noise with edges pointing *away* from the query's ancestor set (GSM-Infinite spider topology), then re-solve to confirm the answer is unchanged. Screen for clauses that genuinely create ambiguity; models do subtract the 5.
- **Sources:** GSM-Symbolic/NoOp, GSM-Infinite, MathForge (Background), MathSmith (distractor strategy).

### 6. Nesting a given as a sub-problem / abstract terminology (answer-preserving)
- **What it does:** Replaces a numeric given with an independent sub-problem whose answer is that number, and/or defines a new abstract term for a central concept. The gold answer is unchanged.
- **Easy → hard:**
  - Easy: "A rectangle has length 12 and width 5. Find its diagonal." (13)
  - Hard: "A rectangle's length equals the number of positive divisors of 60, and its width equals the number of primes less than 12. Call its diagonal the *span*. Find the span." (12 and 5, **13**)
- **Keep it verifiable:** Execute every nested sub-problem and require exact equality with the replaced constant; reject mismatches. Keep the original label. Audit equivalence (MathForge's o3 audit: 97–99%). Under GRPO a corrupted item gives an all-zero group, so it is inert rather than harmful.
- **Sources:** MathForge MQR, MathSmith (non-standard representation), SvS (answer-invariant variation).

### 7. Parametric lifting (generalize-then-specialize, parametrize-and-sum, change the queried quantity)
- **What it does:** Lifts a problem into a parameterized family (EFA, template, meta-template), then (a) moves parameters to regimes where the same lemma applies differently, (b) asks for an aggregate over parameters, or (c) asks for a different unknown in the same configuration.
- **Easy → hard:** "Trailing zeros of 100!" (24) →
  - "…in base 12" (min(⌊97/2⌋, 48) = **48**)
  - "Σ_{n=1}^{50} Z(n!)" (**262**)
  - For a triangle with given sides, ask for the inradius instead of the area.
- **Keep it verifiable:** `solve(params)` with the EFAGen unit tests (`matches_original`, `single_valued`, `has_dof`). Compute aggregates by brute force. QbQ also requires the variant's answer to differ from the parent's.
- **Sources:** EFAGen, QbQ, TemplateGSM (arXiv 2411.18104), InfinityMATH (arXiv 2408.07089).

### 8. Concept / theorem injection ("difficulty hiking", upward evolution, thought modes)
- **What it does:** Rewrites a problem so it also needs a theorem from the same branch plus a cross-domain concept, or applies k mined "thought modes". Difficulty accumulates with each injection.
- **Easy → hard:**
  - Easy: "Find 2^10 mod 7." (2)
  - Hard: "Let N be the number of lattice points strictly inside the triangle with vertices (0,0), (20,0), (0,10). Find 2^N mod 7." (Pick's theorem gives N = 81; the order of 2 mod 7 is 3; answer **1**)
- **Keep it verifiable:** Compute the injected piece programmatically where possible. Otherwise use k-way self-consistency (SAND-Math) plus a different-family verifier. Confirm the difficulty increase with the policy's solve rate, not only an LLM rating.
- **Sources:** SAND-Math, MindLoom, WizardMath, MathSmith.

### 9. Insight hiding ("burden of discovery")
- **What it does:** Finds the key step that makes a seed easy and restructures the problem so that step must be discovered: ask for the extremal object, remove the hint, require a construction.
- **Easy → hard:**
  - Easy: "Show that 1 + 2 + … + 5 is divisible by 5."
  - Hard: "How many n ≤ 1000 satisfy n | 1^k + 2^k + … + n^k for every odd k?" (pairing i ↔ n − i works exactly for odd n; answer **500**)
- **Keep it verifiable:** Verify by exhaustive or programmatic search. Use separate solvability and difficulty critics, and reject "artificial complexity" (Code2Math rubric: scores 1–2 rejected).
- **Sources:** Code2Math, VHG (validity gating of hard items).

### 10. Hard perturbation
- **What it does:** Makes a minimal edit so the memorized method no longer applies.
- **Easy → hard:**
  - Easy: "Minimize x² − 6x + 13 over real x." (vertex, 4)
  - Hard: "…over integers x not divisible by 3." (x = 3 is excluded; x = 2 or 4 gives **5**)
- **Keep it verifiable:** Require the new answer to differ from the original, which also checks for leakage. Verify by search, or by independent agreement plus a judge confirming the old method fails.
- **Sources:** MATH-Perturb, QbQ (answer-must-differ rule).

### 11. Answer-format hardening
- **What it does:** Removes guessable or unverifiable formats: MCQ → open-ended (Big-Math-Reformulated); answers transformed to integers (DAPO-Math-17K); proofs → answer questions (OpenMathReasoning); yes/no and multi-part items dropped.
- **Easy → hard:**
  - Easy: "What is the smallest x on the circle x² + y² − 22x − 16y + 113 = 0? (A)…(D)"
  - Hard: "The smallest x-coordinate on the circle has the form k − m√n with n squarefree. Find k + m + n." (radius √72 = 6√2, x = 11 − 6√2, answer **19**)
- **Keep it verifiable:** The mapping from the original answer must be deterministic and canonical (e.g. squarefree n). Use rule-based exact match.
- **Sources:** Big-Math, DAPO (arXiv 2503.14476), OpenMathReasoning, DeepMath (standardization).

### 12. Follow-up / multi-turn extension
- **What it does:** Derives counterfactual follow-ups (change a constant) or incremental ones (add a fact) from the symbolic structure. A single-turn seed becomes a dialogue where the model must update its answer.
- **Easy → hard:**
  - Seed: "Ann has 5 apples; Tom has 3 more."
  - Follow-up 1: "Suppose instead Ann had 9."
  - Follow-up 2: "Tom then gives away half. How many are left?"
- **Keep it verifiable:** Re-solve the modified symbolic program; cycle-check the rendering.
- **Sources:** MathCAMPS.

### 13. Weakness-targeted synthesis
- **What it does:** Mines the policy's failures (stagnating RL prompts, error hypotheses, failed trajectories), extracts concepts or error patterns, and generates targeted items kept in a learnable band.
- **Easy → hard:** The policy saturates combinatorics seeds but stagnates on inclusion–exclusion with restrictions. Recombine that concept cluster (e.g. with modular constraints) and keep items at 25–75% policy accuracy.
- **Keep it verifiable:** Label with a stronger reasoner requiring at least 50% agreement (SwS), cross-model agreement plus manual checks (Fu et al.), or a teacher reference (Socratic-Zero). Re-measure with the current policy.
- **Sources:** SwS, Fu et al., Socratic-Zero, MathSmith (weakness-focused variants).

### 14. Rationale-first (design-plan-first) generation
- **What it does:** The generator writes an explicit construction plan (concepts, difficulty strategies, intended solution path) before the problem. The plan can be optimized (PromptCoT 2.0 EM; MathSmith RL; Learning-to-Pose design-CoT cold start).
- **Easy → hard:**
  - Easy: "Write a hard algebra problem." This yields generic, often easy or ill-posed items.
  - Hard: Concepts {Vieta, AM-GM, integrality} → rationale "bound the root sum by AM-GM, then use integrality to force a unique case" → the problem is written to realize that path.
- **Keep it verifiable:** Treat the rationale as a reference skeleton and keep an item only if an independent solver majority matches the rationale's answer (Answer-Consistency).
- **Sources:** PromptCoT / 2.0, MathSmith, CoT-Self-Instruct, Learning to Pose.

### 15. Validity-gated difficulty optimization of a generator (setter RL)
- **What it does:** Trains the generator with RL on reward = validity × difficulty. Difficulty can be 1 − accuracy, an uncertainty peak at ½, a Gaussian utility, or a seed-relative complementary target. Add diversity terms.
- **Easy → hard:**
  - Ungated: a setter rewarded only by solver failure learns to emit invalid "hard" problems.
  - Gated: VHG's setter learns validity first, then difficulty. Competition integrals: 45.4% vs about 31% for R-Zero.
- **Keep it verifiable:** The gate must be independent of the solver: symbolic checks, execution, or a different-family verifier. Add BLEU/scarcity penalties, and track pseudo-label accuracy on a gold holdout. Probes can amortize the difficulty estimate (PROPEL).
- **Sources:** VHG, R-Zero, Socratic-Zero, Agentic Proposing, Learning to Pose, MathSmith, PROPEL.

---

## Insights & pitfalls

- **Operator choice matters more than volume.**
  - MetaMath's controlled additions: inversion gives +2.3/+2.6, rephrasing +0.4, more answer augmentation +0.1.
  - FLAMES: complexity-increasing agents gave the best improvements on most math metrics.
  - OpenR1-Math's card: adding the easier cn_k12 source gave lower SFT performance than the curated default subset, "likely because the questions from cn_k12 are less difficult".
- **Composition is dependable, but only if composed items are in the RL mix.**
  - OMEGA: RL on isolated skills gave large in-skill gains (e.g. a polygon-rotation family improved by nearly 70 points), but "these improvements do not reliably carry over to the composed setting".
  - Yuan et al. (string-function compositions): RL on f∘g compositions *does* teach composition, which extrapolates to more than 2 functions and transfers across tasks, while next-token training on the same data does not.
  - h1: only a curriculum over horizon works; uniform mixes do not.
- **Easy seeds can still yield hard training items once they are parameterized.** EFAGen found GPT-4o-failing variants even for Level-1 MATH seeds GPT-4o had solved. Before writing new problems, try lifting and resampling existing ones.
- **The learnable band is the target, and it moves.** QbQ, SwS, SvS, R-Zero, Socratic-Zero and Learning-to-Pose all re-measure accuracy with the *current* policy each round. Pikus et al.: GRPO needs outcome variance, so easy items "quickly converge to consistent success, eliminating learning opportunities".
- **Pair complexification with scaffolding.** Near-zero prompts give no GRPO signal either. The literature restores signal and then removes the help:
  - PrefixRL conditions on successful off-policy prefixes and modulates difficulty by prefix length.
  - ReGFT fine-tunes on reference-guided self-generated traces before RL.
  - Backward Hint Annealing removes hints across difficulty buckets.

  A practical controller: complexify above the band, train inside it, scaffold below it.
- **Answer-preserving edits fail safe under GRPO.** A variant with a wrong inherited label produces all-zero rewards, hence zero advantage and no update (MathForge's argument). Wrong *new* labels from majority vote are not safe: they reward wrong answers. Prefer answer-preserving operators whenever labels are uncertain.
- **Label drift is the main failure of self-play.** R-Zero's pseudo-label accuracy fell from 79% to 63% as questions got harder, and performance collapsed after about 3 iterations (earlier for smaller models). VHG's hard verifier avoids this. J-Zero reports co-adapting the judge sustains at least 10 iterations where baselines degrade after 2. Always keep a gold-labelled monitor slice.
- **The verification budget should rise with RL stakes.** SFT tolerates noise (MathScale: GPT-4 validation did not help; OpenMathInstruct-2: "SFT is robust to low-quality solutions"; FLAMES: coverage beats reliability). RL does not: a wrong reference is a wrong reward. Use tier-1 (construction, execution) or tier-2 (answer-preserving, inverse-checked) labels for RL pools. Use consensus only when needed, and never a single-sample teacher.
- **Consensus and agreement filters cap difficulty at the verifier's frontier.** CoT-Self-Instruct, SAND-Math all-k agreement, DeepMath 3-way agreement and MindLoom's all-wrong exclusion all remove frontier items. Mitigations:
  - construct the answer first (graph or program);
  - verify easy increments rather than the composite (CHASE);
  - use a stronger or different-family verifier;
  - pool disagreements for tool-based relabelling.
- **Training the generator is what raises yield.** MathSmith's available ratio went 71.5% → 95.4% once the consistency term was in the RL reward. Socratic-Generator-32B reached 95.6% validity. VHG's valid rate went 30.6% → 75.5%. Without generator training, expect large rejection: KPDDS kept about 51% at the quality gate; SAND-Math kept 8,842 of 23,437; Code2Math needs 1.56–6.55 failed rollouts per accepted problem.
- **Cheap difficulty proxies need calibration against the current policy.**
  - Teacher trace length (MathSmith).
  - Adaptive-thinking trigger (ScaleDiff: about 88% of generated items flagged difficult).
  - A regressor on fail rates (ScaleQuest).
  - GPT-4o ratings averaged over six queries (DeepMath), or Llama-3.3-70B 1–10 ratings averaged over three (SAND-Math).
  - Activation probes (PROPEL).
- **Separate real difficulty from tedium.** Op or horizon scaling gives smooth, predictable decay (GSM-Infinite sigmoid), which is useful for robustness and state tracking. Code2Math explicitly rejects difficulty from "computational tedium". Hard perturbations and insight hiding change *which* method is needed (MATH-P-Hard: o1-mini −16.49%). Use both families.
- **Small, hard, verified sets beat large easy ones.** SAND-Math: 500 samples gave +17.85 AIME25 over the next-best synthetic set. MathAgent: 1K samples beat LIMO and s1K. Agentic Proposing: 11K trajectories → 91.6% AIME25 at 30B.
- **Decontaminate both seeds and outputs.** DeepMath's raw pool was 90% contaminated for AIME24/AMC23 and 76.6% for MATH500. Teachers asked for "hard, novel" problems can regurgitate known ones; SAND-Math web-searches each question (threshold 0.85). LiveAoPSBench (arXiv 2501.14275) shows performance declining on newer problems, suggesting older ones were seen in pretraining.
- **Validate on more than one model family.** Spurious rewards: random rewards give Qwen2.5-Math-7B +21.4 on MATH-500 (vs +29.1 with ground truth), and such gains "often fail" for Llama3 or OLMo2. Report synthetic-data gains on at least one non-Qwen base and on fresh contests (AIME 2025/2026, HMMT).
- **Protect diversity explicitly.**
  - BLEU-cluster repetition penalty (R-Zero).
  - Scarcity reward over thought modes (MindLoom).
  - Inverse-proficiency skill sampling (Agentic Proposing).
  - Symbolic-space mutation: cosine diversity 0.37 vs 0.07 (Yeo et al.).
  - KL regularization; PROPEL's math run collapsed at the lowest KL.
  - SvS keeps policy entropy from collapsing.
- **Build bridge difficulties, not only harder items.** WizardMath's *downward* evolution rounds added large gains (GSM8K 59.7 → 74.5). MathMixup's decomposed → original → hybrid curriculum and h1's stagewise horizons both rely on intermediate rungs.

---

## Open problems & research opportunities

- **Labels beyond the solver frontier.** Majority vote, consensus and judges fail exactly where you need them. Answer-first construction covers computable problems. It does not yet cover olympiad inequalities, combinatorial constructions or geometry, where scalable certificates (search witnesses, SMT or CAS checks, partial formalization) are missing.
- **Operator ablations under RLVR at matched compute.** Most operator comparisons are SFT-based (FLAMES, MetaMath). A GRPO/DAPO study on identical seeds, comparing inversion, chaining, nesting, distractors, concept injection, parametric lifting, hard perturbation and insight hiding, measured on held-out contests and pass@k, would settle budget allocation.
- **A transferable difficulty ruler.** Pass rate depends on the policy and drifts during training. Trace length, think-mode triggers, LLM ratings and probes are uncalibrated across families. A shared, model-independent difficulty scale for synthetic math does not exist.
- **Automatic detection of tedium vs insight.** Code2Math relies on an LLM rubric (79.3% exact agreement with humans). Generators optimized on solve rate or trace length drift toward verbosity or ambiguity. A validated metric is open.
- **Well-posedness and uniqueness checks.** Multi-solution and ambiguous items are the main quality leak (see inversion's 100–104 example). Symbolic uniqueness checks cover only computable cases, and LLM judges are unreliable on hard items.
- **Provable answer preservation.** MQR and SvS assume the label survives a rewrite and audit this with LLMs (97–99%). Lifting to a symbolic program, editing there and re-rendering with cycle consistency could make preservation checkable.
- **Stable long-run self-play with verified labels.** R-Zero-style loops peak within 3–4 iterations. VHG and J-Zero improve stability, but more than 20 rounds at frontier scale with verified labels has not been shown.
- **Unified per-prompt controllers.** No public system jointly schedules complexification (pass rate too high), plain training (in band) and scaffolding (too low), with operators chosen by measured learning progress.
- **Cost-efficient closed loops.** Generator RL needs many solver rollouts per candidate. Probes (PROPEL), learned fail-rate regressors and reuse of policy-training rollouts are promising but under-explored for math.
- **Contamination at pretraining scale.** Web-search novelty checks (SAND-Math) and benchmark-embedding checks (DeepMath) do not rule out teachers paraphrasing forum problems. Scalable novelty detection against pretraining-scale corpora remains an engineering gap.
- **From proofs to verifiable hardness.** Proof → answer conversion (OpenMathReasoning) can lose difficulty. Generating harder proof-style problems with reliable graders, or coupling informal generation with formal checking, is still largely open.

---

## References

1. Yu, L., Jiang, W., Shi, H., et al. (2023). *MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models*. ICLR 2024. https://arxiv.org/abs/2309.12284
2. Luo, H., Sun, Q., Xu, C., et al. (2023). *WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct*. ICLR 2025. https://arxiv.org/abs/2308.09583
3. Tang, Z., Zhang, X., Wang, B., Wei, F. (2024). *MathScale: Scaling Instruction Tuning for Mathematical Reasoning*. arXiv 2403.02884. https://arxiv.org/abs/2403.02884
4. Huang, Y., Liu, X., Gong, Y., et al. (2024). *Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning*. AAAI 2025. https://arxiv.org/abs/2403.02333
5. Pei, Q., Wu, L., Pan, Z., et al. (2025). *MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion*. ACL 2025. https://arxiv.org/abs/2503.16212
6. Li, X., Chen, J., Li, X., et al. (2026). *MathMixup: Boosting LLM Mathematical Reasoning with Difficulty-Controllable Data Synthesis and Curriculum Learning*. arXiv 2601.17006. https://arxiv.org/abs/2601.17006
7. Tong, Y., Zhang, X., Wang, R., Wu, R., He, J. (2024). *DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving*. NeurIPS 2024. https://arxiv.org/abs/2407.13690
8. Ding, Y., Shi, X., Liang, X., et al. (2024). *Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch*. ACL 2025. https://arxiv.org/abs/2410.18693
9. Mirzadeh, I., Alizadeh, K., Shahrokhi, H., et al. (2024). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models*. ICLR 2025. https://arxiv.org/abs/2410.05229
10. Zhou, Y., Liu, H., Chen, Z., Tian, Y., Chen, B. (2025). *GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?* arXiv 2502.05252. https://arxiv.org/abs/2502.05252
11. Ye, T., Xu, Z., Li, Y., Allen-Zhu, Z. (2024). *Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process*. arXiv 2407.20311. https://arxiv.org/abs/2407.20311
12. Mishra, S., Poesia, G., Goodman, N. D. (2024). *MathCAMPS: Fine-grained Synthesis of Mathematical Problems From Human Curricula* (later: *From Next-Token to Mathematics: The Learning Dynamics of Mathematical Reasoning in Language Models*). COLM 2025. https://arxiv.org/abs/2407.00900
13. Khan, Z., Stengel-Eskin, E., Prasad, A., Cho, J., Bansal, M. (2025). *Executable Functional Abstractions: Inferring Generative Programs for Advanced Math Problems*. arXiv 2504.09763. https://arxiv.org/abs/2504.09763
14. Wang, J., Jiang, J., Zhang, Z., Zhou, J., Zhao, W. X. (2025). *RV-Syn: Rational and Verifiable Mathematical Reasoning Data Synthesis based on Structured Function Library*. EACL 2026. https://arxiv.org/abs/2504.20426
15. Yeo, T., Jeon, M., Weerakoon, D., Qiao, R., et al. (2026). *Adaptive Problem Generation via Symbolic Representations*. arXiv 2602.19187. https://arxiv.org/abs/2602.19187
16. Yu, Z., Rao, J., Chen, G., et al. (2026). *MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis*. arXiv 2604.11188. https://arxiv.org/abs/2604.11188
17. Shah, V., Yu, D., Lyu, K., Park, S., et al. (2024). *AI-Assisted Generation of Difficult Math Questions*. arXiv 2407.21009. https://arxiv.org/abs/2407.21009
18. Hosseini, A., Sordoni, A., Toyama, D., Courville, A., Agarwal, R. (2024). *Not All LLM Reasoners Are Created Equal*. arXiv 2410.01748. https://arxiv.org/abs/2410.01748
19. Patel, A., Reddy, S., Bahdanau, D. (2025). *How to Get Your LLM to Generate Challenging Problems for Evaluation* (CHASE). arXiv 2502.14678. https://arxiv.org/abs/2502.14678
20. Huang, K., Guo, J., Li, Z., et al. (2025). *MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations*. arXiv 2502.06453. https://arxiv.org/abs/2502.06453
21. Motwani, S. R., Ivanova, A., Cai, Z., Torr, P., et al. (2025). *h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning*. arXiv 2510.07312. https://arxiv.org/abs/2510.07312
22. Zhao, X., Wu, W., Guan, J., Kong, L. (2025). *PromptCoT: Synthesizing Olympiad-level Problems for Mathematical Reasoning in Large Language Models*. arXiv 2503.02324. https://arxiv.org/abs/2503.02324
23. Zhao, X., Wu, W., Guan, J., Gong, Z., Kong, L. (2025). *PromptCoT 2.0: Scaling Prompt Synthesis for Large Language Model Reasoning*. arXiv 2509.19894. https://arxiv.org/abs/2509.19894
24. Manem, C., Brahma, P. P., Mishra, P., Liu, Z., Barsoum, E. (2025). *SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers*. NeurIPS 2025 MATH-AI Workshop. https://arxiv.org/abs/2507.20527
25. Zhan, S., Lai, Y., Lu, Z., Lin, D., et al. (2025). *MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy*. AAAI 2026. https://arxiv.org/abs/2508.05592
26. Pei, Q., Pan, Z., Lin, H., Gao, X., et al. (2025). *ScaleDiff: Scaling Difficult Problems for Advanced Mathematical Reasoning*. arXiv 2509.21070. https://arxiv.org/abs/2509.21070
27. Shen, H., Guo, T., Chen, X., Liu, M., et al. (2026). *MindLoom: Composing Thought Modes for Frontier-Level Reasoning Data Synthesis*. arXiv 2605.21630. https://arxiv.org/abs/2605.21630
28. Yu, P., Lanchantin, J., Wang, T., Yuan, W., et al. (2025). *CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks*. arXiv 2507.23751. https://arxiv.org/abs/2507.23751
29. Fu, J., Heddaya, M., Tan, C. (2026). *Automatically Generating Hard Math Problems from Hypothesis-Driven Error Analysis*. ICLR 2026 Workshop on Logical Reasoning of LLMs. https://arxiv.org/abs/2604.04386
30. Liang, X., Li, Z.-Z., Gong, Y., Wang, Y., et al. (2025). *SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning*. NeurIPS 2025. https://arxiv.org/abs/2506.08989
31. Liang, X., Li, Z., Gong, Y., Shen, Y., et al. (2025). *Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR*. arXiv 2508.14029. https://arxiv.org/abs/2508.14029
32. Dai, Y., Ji, Y., Zhang, X., Wang, Y., et al. (2026). *Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation*. ICLR 2026. https://arxiv.org/abs/2601.20614
33. Bao, L., Wang, J., Zhang, Y., Zheng, Y., et al. (2026). *Question Begets Question: Self-Evolving Curriculum for Reinforcement Fine-Tuning on Competition Mathematics*. arXiv 2608.01522. https://arxiv.org/abs/2608.01522
34. Huang, C., Yu, W., Wang, X., Zhang, H., et al. (2025). *R-Zero: Self-Evolving Reasoning LLM from Zero Data*. ICLR 2026. https://arxiv.org/abs/2508.05004
35. Wang, S., Jiao, Z., Zhang, Z., Peng, Y., et al. (2025). *Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution*. arXiv 2509.24726. https://arxiv.org/abs/2509.24726
36. Lai, Y., Feng, J., Teh, Y. W., Miao, N. (2026). *Verifier-Backed Hard Problem Generation for Mathematical Reasoning*. arXiv 2605.06660. https://arxiv.org/abs/2605.06660
37. Wei, Y., Zhao, Y., Hu, Z., et al. (2025). *Learning to Pose Problems: Reasoning-Driven and Solver-Adaptive Data Synthesis*. arXiv 2511.09907. https://arxiv.org/abs/2511.09907
38. Wolf, L., Watts, C., Castanyer, R. C., et al. (2026). *Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier* (PROPEL). arXiv 2606.18284. https://arxiv.org/abs/2606.18284
39. Jiao, Z., Wang, S., Zhang, Z., Ren, X., et al. (2026). *Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis*. arXiv 2602.03279. https://arxiv.org/abs/2602.03279
40. Guo, D., Xie, Y., Liu, Q., Huang, W., et al. (2026). *Code2Math: Can Your Code Agent Evolve Math Problems Through Exploration?* arXiv 2603.03202. https://arxiv.org/abs/2603.03202
41. Albalak, A., Phung, D., Lile, N., Rafailov, R., et al. (2025). *Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models*. arXiv 2502.17387. https://arxiv.org/abs/2502.17387
42. He, Z., Liang, T., Xu, J., Liu, Q., et al. (2025). *DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning*. arXiv 2504.11456. https://arxiv.org/abs/2504.11456
43. Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., et al. (2025). *AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset*. arXiv 2504.16891. https://arxiv.org/abs/2504.16891
44. Du, W., Toshniwal, S., Kisacanin, B., et al. (2025). *Nemotron-Math: Efficient Long-Context Distillation of Mathematical Reasoning from Multi-Mode Supervision*. arXiv 2512.15489. https://arxiv.org/abs/2512.15489
45. Seegmiller, P., Mehta, K., Saha, S., et al. (2025). *FLAMES: Improving LLM Math Reasoning via a Fine-Grained Analysis of the Data Synthesis Pipeline*. arXiv 2508.16514. https://arxiv.org/abs/2508.16514
46. Sun, Y., Hu, S., Zhou, G., et al. (2025). *OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization*. arXiv 2506.18880. https://arxiv.org/abs/2506.18880
47. Yuan, L., Chen, W., Zhang, Y., et al. (2025). *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*. arXiv 2509.25123. https://arxiv.org/abs/2509.25123
48. Pikus, B., Tiwari, P. R., Ye, B. (2025). *Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets*. arXiv 2508.14094. https://arxiv.org/abs/2508.14094
49. Toshniwal, S., Du, W., Moshkov, I., et al. (2024). *OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data*. arXiv 2410.01560. https://arxiv.org/abs/2410.01560
50. Shao, R., Li, S. S., Xin, R., et al. (2025). *Spurious Rewards: Rethinking Training Signals in RLVR*. arXiv 2506.10947. https://arxiv.org/abs/2506.10947
51. Mahdavi, S., Li, M., Liu, K., et al. (2025). *Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation* (AoPS-Instruct / LiveAoPSBench). arXiv 2501.14275. https://arxiv.org/abs/2501.14275
52. Setlur, A., Wang, Z., Cohen, A., et al. (2026). *Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes* (PrefixRL). arXiv 2601.18795. https://arxiv.org/abs/2601.18795
53. Wu, Y., Li, S., Wen, Z., et al. (2026). *Learn Hard Problems During RL with Reference Guided Fine-tuning* (ReGFT). arXiv 2603.01223. https://arxiv.org/abs/2603.01223
54. Xie, P.-X., Lin, C.-Y., Yang, C.-L. (2026). *Mitigating Distribution Sharpening in Math RLVR via Distribution-Aligned Hint Synthesis and Backward Hint Annealing*. arXiv 2604.07747. https://arxiv.org/abs/2604.07747
55. Chu, G., Jeon, M., Yeo, T., et al. (2026). *J-Zero: Unified Challenger–Solver–Judge Self-Evolution from Zero Data*. arXiv 2608.26582. https://arxiv.org/abs/2608.26582
56. Yu, Q., Zhang, Z., Zhu, R., et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv 2503.14476. https://arxiv.org/abs/2503.14476
57. Zhang, Y. (2024). *Training and Evaluating Language Models with Template-based Data Generation* (TemplateGSM). arXiv 2411.18104. https://arxiv.org/abs/2411.18104
58. Zhang, B.-W., Yan, Y., Li, L., et al. (2024). *InfinityMATH: A Scalable Instruction Tuning Dataset in Programmatic Mathematical Reasoning*. arXiv 2408.07089. https://arxiv.org/abs/2408.07089
59. Open-R1 team (2025). *OpenR1-Math-220k* (dataset card). Hugging Face. https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
