# Instruction evolution & complexification for SFT data (Evol-Instruct family and successors)

*Scope: LLM-driven rewriting, composition and constraint-stacking methods that turn easy seed prompts into harder SFT/RL prompts (Evol-Instruct lineage, constraint-based instruction following, fusion/crossover, learned evolvers, and target-aware difficulty filtering), 2022 to Sept 2026. Compiled 2026-09-30. Verification: 29 researcher entries checked against primary sources (full-text arXiv PDFs), 14 corrected (the RECAST/Crab entry is split into two notes), 0 dropped, 6 added. A further 12 supporting citations were checked.*

---

## TL;DR

- **"Complex" is not "hard for your model".** Classic evolvers (Evol-Instruct, Tree-Instruct, Deita, IDEA-MCTS, TAG-INSTRUCT) raise complexity as an LLM judges it. When GRPO gets zero advantage, the signal you need is difficulty measured on your own policy: keep items whose pass rate over k samples falls in a band. IFDecorator uses (0, 0.5] over 8 samples. EvoTD uses (0, 1). The D2Evo questioner targets [0.4, 0.6]. QbQ seeds from problems solved 8–15 times out of 16. IFDecorator states directly that "complexity alone does not determine difficulty".
- **Naive evolution mostly produces broken tasks, not hard ones.** After 5 iterations of adding constraints in IFDecorator, 10,772 instructions had pass rate 0 and only 7,324 landed in the target band. Also: GPT-4o intensified complexity in only 1 of 3 depth-evolution attempts (TaCIE). Code Evol-Instruct grew a prompt from 14 to 325 tokens with up to 8 constraints in 4 rounds (Instruction Fusion). 5.6% of fused math problems stayed unreasonable after 5 regenerations (MathFusion). Budget 2–3 rounds, stop on a dev set (WizardCoder was best after 3), and gate every item with an answerability check.
- **Prefer operators that keep correctness by construction.** These include constraint back-translation (Crab: state constraints the reference answer already satisfies), programs executed for ground truth (EvoTD), code↔instruction round-trips (Phi-4), and verifier-per-constraint generation (AutoIF, IFBench). For free-form math or science, use answer consistency: CoT-Self-Instruct raised a GRPO average from 53.0 to 57.2 with 2,926 filtered prompts, versus 5,000 unfiltered.
- **Train the evolver against the learner.** Several recent works replace a fixed rewrite prompt with a model trained on the learner's failures. SEIF's instructor reward is 1 − the follower's constraint satisfaction. Removing instructor evolution cost 2.7 IFEval points (78.6 vs 75.9). The D2Evo questioner is trained with GRPO toward [0.4, 0.6] accuracy, anchored on real mid-difficulty problems. On Qwen3-8B-Base it beat full-data RL (19K real) on math using 1.7K real samples (55.32 vs 52.70). Lion and CodecLM are the cheap, non-RL version: keep or expand only the prompts where teacher and student are judged far apart.
- **For instruction-following RL, stack more constraints than the eval uses, and randomize their parameters.** IFBench found training on 5–6 constraints per prompt beats training on up to 3, even though IFEval tests at most 3 and IFBench at most 2. Wider variable ranges help, and disjoint ranges hurt. Crab samples 6–8 constraints per prompt and RECAST averages 13.4. Filter conflicts: removing SEIF's filter cost 3.2 IFEval points and 6.0 CFBench points.
- **Constraint rewards get hacked. Budget for it from the start.** IFDecorator lists four exploit patterns (copying format markers, dummy list items, trivial repetition, copying delimiters), and its intent check cut the hack rate from 14.53% to 7.60%. IFBench adds a preference-RM bonus or penalty only when the verifiable reward is positive. LsrIF aggregates rewards along the logic structure instead of averaging. UNSPECIFIC shows that back-translated constraints can often be satisfied by copy-paste, and hardens only the constraints that are satisfied trivially.
- **Blind Evol-Instruct can hurt.** Reimplemented in UltraIF's comparison, 10K Evol-Instruct samples scored below plain ShareGPT on IFEval (Pr(S) 41.96 vs 43.99). In EvoTD, Evol-Instruct raised pass@1 but pushed pass@8 below the untuned backbone (54.2 vs 55.1), which is mode collapse. Tulu 3 found 70.7% of HumanEval test items overlapped with Evol-CodeAlpaca. Always decontaminate, and track pass@k, not just pass@1.
- **Seed from "almost solved", not from "never solved".** QbQ took Qwen2.5-Math-7B from 5.6% to 16.5% AIME pass@1 over 20 rounds, while static augmentation plateaued at 12.5% and 14.5%. Seeding from the harder 1–7/16 band did worse. Phi-4 drops seeds where all sampled answers agree and seeds where none do.
- **Iterative self-synthesis drifts.** KITE reports "polarization of competence": strong skills get stronger and weak ones degrade. D2Evo reports that an R-Zero-style questioner with no anchor suffers entropy collapse. Keep real seeds in every round. Add diversity pressure: Deita's Repr Filter, VISA's farthest-neighbor selection and coverage down-weighting, QbQ's forced structural operators. Stop after about 3 rounds unless a held-out metric keeps improving (SEIF, UltraIF, WizardCoder).

---

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Evol-Instruct / WizardLM (+ Code Evol-Instruct / WizardCoder) | 2023 | [2304.12244](https://arxiv.org/abs/2304.12244), [2306.08568](https://arxiv.org/abs/2306.08568) | General chat; code | SFT | Add constraints, deepen, concretize, more reasoning steps, complicate input, in-breadth mutation. Code adds rarer requirements, erroneous reference code, time/space complexity | Weak: ChatGPT answers are not verified. An elimination filter handles no-information-gain, "sorry" + <80 words, and template leakage. Code uses a dev-set "Evol stop" |
| WizardMath (RLEIF) | 2023 | [2308.09583](https://arxiv.org/abs/2308.09583) | Math | SFT + PPO | Upward evolution (constraints, concretize, reasoning) and downward evolution (easier variants) | IRM (instruction reward model on difficulty and definition) plus a GPT-4-labelled PRM in PPO. No symbolic answer check |
| Tree-Instruct | 2023 | [2308.05696](https://arxiv.org/abs/2308.05696) | General | SFT | Add N nodes (3/6/10) to the instruction's semantic tree | None beyond the LLM answer |
| Deita (EVOL COMPLEXITY / QUALITY) | 2023 | [2312.15685](https://arxiv.org/abs/2312.15685) | General chat, data selection | SFT (+DPO) | Evolve-then-rank chains; learned complexity and quality scorers | Quality scorer; no answer verification |
| Lion *(added)* | 2023 | [2305.12870](https://arxiv.org/abs/2305.12870) | General | SFT (adversarial distillation) | Referee-identified "hard" prompts → generate new prompts of the same domain/task type | Teacher (ChatGPT) answers; referee score gap |
| Instruction Fusion *(added)* | 2023 | [2312.15692](https://arxiv.org/abs/2312.15692) | Code | SFT | Fuse two random seed prompts into one | "INVALID PROMPT" escape; teacher answers |
| CodecLM | 2024 | [2404.05875](https://arxiv.org/abs/2404.05875) | General IF | SFT | Metadata → Self-Rubrics actions (≤4 iterations) | Strong LLM answers; contrastive filtering (judge gap > 3/10) |
| Conifer | 2024 | [2404.02823](https://arxiv.org/abs/2404.02823) | Constrained IF | SFT (+DPO) | Reframe query; two-level constraint lists; recombine into levels 1–5; easy→hard multi-turn | GPT-4 answers; context and conflict filters; GPT-4 process feedback |
| Auto Evol-Instruct | 2024 | [2406.00770](https://arxiv.org/abs/2406.00770) | Chat, math, code | SFT | Optimizer LLM rewrites the evolving method from failure trajectories | Rule-based failure detection on responses; no answer verification |
| AutoIF | 2024 | [2406.13542](https://arxiv.org/abs/2406.13542) | Verifiable IF | SFT + DPO | Self-instructed atomic constraints appended to ShareGPT queries | Executable verifiers cross-validated with test cases, NLI back-translation, response pass >0.5, LLM match score ≥8 |
| Persona Hub | 2024 | [2406.20094](https://arxiv.org/abs/2406.20094) | Math, reasoning, instructions | SFT | Persona conditioning plus stated difficulty/focus | Training answers unverified (GPT-4o); test set needs 2 of 3 solutions to agree |
| AgentInstruct (Orca-3) | 2024 | [2407.03502](https://arxiv.org/abs/2407.03502) | 17 skills | SFT | Content transformation; taxonomy seeds; Suggester→Editor refinement | Strong-LLM answers grounded in source text; no formal verifier |
| Genetic-Instruct | 2024 | [2407.21077](https://arxiv.org/abs/2407.21077) | Code | SFT | Evol mutation plus few-shot crossover in parallel colonies | Parser/AST check plus CoT LLM-judge fitness; no execution |
| Instruct-SkillMix | 2024 | [2408.14774](https://arxiv.org/abs/2408.14774) | General IF | SFT | Random k-skill (k=2) plus query-type composition | Frontier teacher answers only |
| IDEA-MCTS | 2024 | [2410.10392](https://arxiv.org/abs/2410.10392) | General IF | SFT | MCTS over a 12-action evolution space | Learned quality/complexity/diversity value; no answer verification |
| Crab (constraint back-translation) | 2024 | [2410.24175](https://arxiv.org/abs/2410.24175) | Complex IF | SFT (+reverse training) | Add 6–8 constraints the existing response already satisfies | By construction; LLM re-verification; Python-filled templates for 6 types |
| Tulu 3 Persona-IF / IFBench-IFTrain | 2024–25 | [2411.15124](https://arxiv.org/abs/2411.15124), [2507.02833](https://arxiv.org/abs/2507.02833) | Verifiable IF | SFT, DPO, RLVR | Persona + constraint; stacking up to 6; wide variable ranges; 29 new train constraint types | Python verifier per constraint; RM bonus/penalty against over-optimization |
| Smaller LMs as evolvers + IC-IFD | 2024 | [2412.11231](https://arxiv.org/abs/2412.11231) | IF, math, code | SFT | Evolver choice (8B vs 70B); IC-IFD selection | IC-IFD down-weights incomprehensible prompts |
| Phi-4 synthetic pipeline | 2024 | [2412.08905](https://arxiv.org/abs/2412.08905) | Reasoning, math, code | Pre/mid-training + post-training | Plurality-banded seeds; rewrite into exercises; self-revision; instruction reversal | Majority vote; code execution/tests; reversal-fidelity filter |
| UltraIF (UltraComposer) | 2025 | [2502.04153](https://arxiv.org/abs/2502.04153) | IF | SFT + iterative DPO/NCA | Learned composer adds constraint plus eval question; iterative | LLM-answered per-constraint evaluation questions |
| MathFusion | 2025 | [2503.16212](https://arxiv.org/abs/2503.16212) | Math | SFT | Sequential / parallel / conditional fusion of nearest-neighbor pairs | Weak (GPT-4o-mini answers); 5.6% unreasonable after regeneration |
| TAG-INSTRUCT | 2025 | [2505.18557](https://arxiv.org/abs/2505.18557) | General IF | SFT | Compress to tags; DPO-trained tag-expansion policy; regenerate | Teacher answers; coherence criteria |
| RECAST | 2025 (ICLR 2026) | [2505.19030](https://arxiv.org/abs/2505.19030) | Complex IF (>10 constraints) | SFT + RL (RLVC/GRPO) | Extract rule/model constraints from real responses; integrate ~13 per prompt | Rule validators plus LLM validators; multi-LLM majority-voted rewrite and response |
| Tag-Evol | 2025 | [2505.24165](https://arxiv.org/abs/2505.24165) | General, math, code | SFT | Inject k knowledge tags (budget) in one shot | Teacher answers; rewrite step to remove hallucinations |
| SynthQuestions | 2025 | [2506.03968](https://arxiv.org/abs/2506.03968) | General user instructions | SFT (+DPO) | Attributed grounding (doc → user → motivation → instruction) | LLM 7-criterion score ≥3; no answer verification |
| Infinity Instruct (+ Subject report) | 2025 | [2506.11116](https://arxiv.org/abs/2506.11116), [2507.06968](https://arxiv.org/abs/2507.06968) | General chat | SFT | Informative seed selection → Evol rewriting → deficiency diagnosis → re-evolve | Semantic-equivalence/harm check on rewrites; GPT-4 judged diagnosis |
| SAND-Math (Difficulty Hiking) | 2025 | [2507.20527](https://arxiv.org/abs/2507.20527) | Competition math | SFT | Rewrite with same-branch theorem plus cross-domain concept, conditioned on rating | k=2 teacher solutions must agree; solver-failure filter; dedup/decontam/novelty |
| CoT-Self-Instruct | 2025 | [2507.23751](https://arxiv.org/abs/2507.23751) | Math/science; general | RL (GRPO, online DPO) | Plan-then-generate new prompt of similar/higher complexity | Answer-Consistency (majority vote = generated answer); RIP for non-verifiable |
| IFDecorator | 2025 | [2508.04632](https://arxiv.org/abs/2508.04632) | IF RLVR | RL | Add up to 3n verifiable constraints per iteration until pass rate ∈ (0, 0.5] | Script ∧ LLM-criteria verification ∧ IntentCheck; trip wires |
| LsrIF *(added)* | 2026 | [2601.06431](https://arxiv.org/abs/2601.06431) | Logic-structured IF | RL | Parallel / sequential / conditional / nested constraint trees | Code verifier (hard) plus RM (soft); structure-aware reward aggregation |
| SEIF *(added)* | 2026 | [2605.07465](https://arxiv.org/abs/2605.07465) | IF | RL (GRPO, self-evolving) | RL-trained Instructor adds constraints; reward = 1 − follower satisfaction | Filter (conflict/invalid) and constraint-level Judger from latest Follower |
| EvoTD | 2026 | [2605.11666](https://arxiv.org/abs/2605.11666) | Program-grounded reasoning | RL | Skill crossover; attribute mutation (size/depth/range); ZPD filter | Python execution; skill audit; 0 < pass rate < 1 |
| D2Evo | 2026 (ICML 2026) | [2605.17037](https://arxiv.org/abs/2605.17037) | Math → general reasoning | RL | GRPO-trained anchor-conditioned Questioner | Majority-vote pseudo-labels double-checked by GPT-5.2; real anchors |
| Question Begets Question (QbQ) *(added)* | 2026 | [2608.01522](https://arxiv.org/abs/2608.01522) | Competition math | RL (GRPO) | 5 skill-preserving structural operators applied to "mostly-right" seeds | Teacher answers; answer must differ from parent; solve-count selection |
| UNSPECIFIC *(added)* | 2026 | [2608.09154](https://arxiv.org/abs/2608.09154) | Long-form constrained generation | Eval (and data synthesis) | Common constraints from two similar articles; selective hardening of trivially satisfied constraints | Satisfaction checked on the article and its summary |
| VISA | 2026 | [2608.26013](https://arxiv.org/abs/2608.26013) | Multimodal IF | SFT + RL | Image-aware constraint discovery; difficulty- and coverage-weighted sampling; diversity selection | Code tools plus structured LLM judges; diagnostic repair; target-model probing |

---

## Method notes

### Evol-Instruct (WizardLM) + Code Evol-Instruct (WizardCoder) — WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions (Xu et al., 2023); WizardCoder: Empowering Code Large Language Models with Evol-Instruct (Luo et al., 2023)
Link: https://arxiv.org/abs/2304.12244 (ICLR 2024) · https://arxiv.org/abs/2306.08568 (ICLR 2024)
- **Mechanism:** Each round, an "Instruction Evolver" LLM (gpt-3.5-turbo) picks one of six prompts per instruction. Five are **in-depth**: add constraints, deepening, concretizing, increased reasoning steps, and complicating input (the last uses in-context examples, e.g. adding XML data as input). One is **in-breadth**: "create a brand new prompt in the same domain but even more rare", of similar length and difficulty. The same LLM writes the responses. Failed evolutions go back into the pool to be retried next round. Starting from the 52K Alpaca instructions, the authors ran M=4 rounds and got 250K instructions. All rounds were merged and 70K were sampled to match Vicuna's data size.
- **How it makes tasks harder:** The core prompt says "rewrite a given prompt into a more complex version to make those famous AI systems … a bit harder to handle … must be reasonable, understood, and responded to by humans". Each rewrite may add at most 10–20 words, a deliberate "gradual difficulty increase". WizardCoder's code heuristics: add new constraints (about 10 extra words), replace a common requirement with a less common and more specific one, add reasoning steps, "provide a piece of erroneous code as a reference to increase misdirection", and "propose higher time or space complexity requirements, but … refrain from doing so frequently".
- **Correctness / verification:** None for answers. "Elimination evolving" removes four failure cases: no information gain over the parent (judged by ChatGPT); the response contains "sorry" and is under 80 words; the response is only punctuation or stop words; the prompt copies template words such as "#Rewritten Prompt#".
- **Difficulty control:** Number of rounds plus the word budget. ChatGPT's post-hoc difficulty ratings on a 1–10 scale were Alpaca 3.00 → C1 5.48 → C2 6.35 → C3 6.84 → C4 7.08, showing diminishing increments. GPT-4 rated 2.69 → 5.54 and humans 3.15 → 6.82. On 300 pairwise difficulty judgments, Kappa was 0.68 between humans and 0.66 between ChatGPT and the human majority. The ratings were **not** used to guide generation. WizardCoder evolved about 20K Code Alpaca seeds into about 78K samples and used an external dev set (MBPP-400) as an "Evol stop". Performance peaked after 3 rounds.
- **Reported results:** WizardLM-13B vs Vicuna-13B: HumanEval 24.0 vs 12.5, GSM8K 37.15 vs 24.34, MT-Bench 6.35 vs 6.21. Using all 250K rows gave HumanEval 25.6 and GSM8K 37.46. Evolving with LLaMA-2-70B-Chat instead of ChatGPT gave worse results (HumanEval 19.5, GSM8K 33.83). WizardCoder-15B: HumanEval pass@1 57.3.
- **Limitations / failure modes:** Answers are unverified. What rises is LLM-judged complexity, not target difficulty. Later work documents drift (lost information, unrealistic math) and "over-escalating constraints": 14 → 325 tokens with up to 8 constraints after 4 code rounds (Instruction Fusion). Gains saturate at about 3 code rounds.
- **How to reuse with easy seeds:** Use it as the baseline operator library, with domain-specific heuristic lists as WizardCoder did. Evolve 2–3 rounds, merge all rounds, and stop on a dev set. Swap ChatGPT answering for a strong reasoner plus verification. Maintained open implementations: distilabel's `EvolInstruct`, `EvolComplexity`, `EvolQuality`, and `ComplexityScorer` tasks (argilla-io/distilabel).

### WizardMath (RLEIF) — WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct (Luo et al., 2023)
Link: https://arxiv.org/abs/2308.09583 (ICLR 2025 oral)
- **Mechanism:** Math Evol-Instruct runs on GSM8K and MATH training seeds. GPT-4 evolves each seed for 5 rounds, 2 downward and 3 upward, each built on the previous round, with 6 evolutions per seed per round at temperature 0.7. That gave 448K unique instructions (17K duplicates removed) and 418K after decontamination (30K removed). GPT-4-0613 wrote step-by-step answers for SFT. Two reward models are then trained. The **IRM** ranks evolved instructions by *difficulty* and *definition*, trained on GPT-4 rankings (1–6) of 90K extra evolutions. The **PRM** gives step labels from GPT-4 on 5 SFT-model answers per instruction. Step-wise PPO then uses reward r = r_IRM · min_step r_PRM.
- **How it makes tasks harder:** Upward evolution adds constraints, concretizes, or increases reasoning. Downward evolution makes a hard question easier or writes a new easier one on another topic.
- **Correctness / verification:** Only the IRM (instruction quality) and PRM (step correctness). There is no symbolic answer check on evolved problems. The PRM is also used at inference for PRM-weighted majority voting.
- **Difficulty control:** A two-sided ladder (downward and upward). The IRM keeps evolution from "spiraling out of control".
- **Reported results:** WizardMath-Mistral-7B: GSM8K 90.7, MATH 55.4 (MetaMath-Mistral-7B: 77.9 / 28.6). WizardMath-70B: 92.8 / 58.6. Ablation on Mistral-7B: SFT 82.8 / 48.1 → +PRM 87.2 / 52.7 → +PRM+IRM 90.7 / 55.4.
- **Limitations / failure modes:** Both RMs are GPT-4-labelled proxies and can be exploited. Answers to evolved problems are not verified.
- **How to reuse with easy seeds:** Generate both harder and easier siblings of each seed to fill the difficulty spectrum. Train a cheap instruction-quality ranker, scoring difficulty × well-definedness, to gate evolutions. Before RLVR, replace GPT-4 answer trust with multi-sample agreement or a checker.

### Tree-Instruct — Tree-Instruct: A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment (Zhao et al., 2023)
Link: https://arxiv.org/abs/2308.05696 (LREC-COLING 2024)
- **Mechanism:** One prompt asks the LLM to (1) parse the instruction into a semantic tree, (2) add a specified number of new nodes, only nouns or verbs, in depth or width, and (3) "sentenceize" the tree back into a fluent instruction.
- **How it makes tasks harder:** More entities, relations, and sub-requirements. Adding 10 nodes grew the average instruction length from 186 to 607 tokens.
- **Correctness / verification:** None beyond the LLM's answer.
- **Difficulty control:** `your_added_number` = 3, 6, or 10.
- **Reported results:** On 1,000 Alpaca-GPT-4 samples, +3/6/10 nodes gave win-rate gains of 13%, 20%, and 26% across eight sub-skills. Matching tokens with 4,000 original samples, the 10-node set still gained more than 15%. At equal tokens it beat 3 rounds of WizardLM in-depth evolution by 2% win rate. On about 6K OpenChat-filtered ShareGPT conversations, AlpacaEval rose from 84.56% to 86.19%. Curriculum training (3 → 6 → 10 nodes) beat mixed-difficulty training but fell short of training on 10-node data alone.
- **Limitations / failure modes:** Small scale. Structural complexity is a proxy. Chat benchmarks only.
- **How to reuse with easy seeds:** Make "add k sub-requirements/entities/relations" an explicit knob and sweep k. Under a fixed token budget, prefer fewer, richer prompts.

### Deita — What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning (Liu et al., 2023)
Link: https://arxiv.org/abs/2312.15685 (ICLR 2024; code github.com/hkust-nlp/deita)
- **Mechanism:** EVOL COMPLEXITY evolves each of 2K Alpaca seeds M=5 times with in-depth prompts, giving 6 variants. ChatGPT then ranks and scores all 6 **in one prompt**, and a LLaMA-1-7B complexity scorer is trained on those scores. EVOL QUALITY does the same for responses. Selection sorts by the product of complexity and quality scores, then greedily adds a sample only if its embedding distance to its nearest selected neighbor passes threshold τ (0.9). This is the "Repr Filter".
- **How it makes tasks harder:** It does not generate data for training. It creates complexity ladders only to train a scorer, then selects the hardest high-quality slice of an existing pool.
- **Correctness / verification:** Quality scorer only.
- **Difficulty control:** A continuous learned complexity score. Scoring variants together is key: scored separately, "ChatGPT tends to assign similar scores to most examples".
- **Reported results:** DEITA-Mistral-7B with 6K SFT reached MT-Bench 7.22 and AlpacaEval 80.78%. Adding 10K DPO gave MT-Bench 7.55 and AlpacaEval 90.06%.
- **Limitations / failure modes:** Scores judged complexity, not policy difficulty. Chat only.
- **How to reuse with easy seeds:** After evolving, train a cheap ranker from *within-chain* comparisons and keep the top slice with an embedding-diversity gate. Better still, swap the ChatGPT label for your policy's failure rate.

### Lion (added) — Lion: Adversarial Distillation of Proprietary Large Language Models (Jiang et al., 2023)
Link: https://arxiv.org/abs/2305.12870 (EMNLP 2023)
- **Mechanism:** ChatGPT plays teacher, referee, and generator, and each iteration runs imitation → discrimination → generation. In **Imitation**, the student is fine-tuned on teacher responses in a Train Pool. In **Discrimination**, both models answer every prompt in a growing Cache Pool. The referee scores both answers twice with positions swapped and averages. Prompts where teacher − student ≥ τ = 1.0 are "hard". In **Generation**, the model samples a hard prompt and generates a new one "in the same domain and matching the task type". It also samples an easy prompt and generates a more long-tailed one, keeping hard:easy at 1:1 to limit forgetting. A new prompt is kept only if its ROUGE-L against the cache pool is below 0.7.
- **How it makes tasks harder:** New prompts are drawn toward the student's measured weak spots. Hard prompts skew toward math and coding.
- **Correctness / verification:** Teacher answers are trusted, and the referee is an LLM judge.
- **Difficulty control:** The referee-gap threshold τ. It can be read as a min-max game that ends when the referee can no longer separate student from teacher.
- **Reported results:** Alpaca seeds, 3 iterations with 6K new prompts each, 70K data in total. Lion-13B beat Vicuna-13B by 55.4% (relative) on BBH and 16.7% on AGIEval.
- **Limitations / failure modes:** Judge noise; the student cannot surpass the teacher.
- **How to reuse with easy seeds:** This is the simplest target-aware loop. Periodically score your policy against a stronger reference, or with a verifier, on a growing prompt cache. Generate new prompts only from the failing clusters, and keep an easy-to-long-tail stream alongside.

### Instruction Fusion (added) — Instruction Fusion: Advancing Prompt Evolution through Hybridization (Guo et al., 2023)
Link: https://arxiv.org/abs/2312.15692
- **Mechanism:** GPT-4 Turbo acts as a "Prompt Fusion Specialist". It merges two randomly selected seed prompts from evol-codealpaca into one prompt that integrates both, keeps similar length and complexity, and is "coherent and solvable". If fusion is impossible it outputs "INVALID PROMPT", and the pair is resampled. GPT-4 Turbo also writes the responses.
- **How it makes tasks harder:** It increases the number and diversity of objectives, not the number of constraints. The motivation is that Code Evol-Instruct keeps the primary objective and piles on constraints: one 4th-round example grew from 14 to 325 tokens with up to 8 constraints. Code evolution "reaches its capacity (3 rounds)".
- **Correctness / verification:** Teacher answers; the validity escape hatch.
- **Difficulty control:** Implicit, since the prompt asks for complexity similar to the parents.
- **Reported results:** IF-CL-34B HumanEval 78.7; IF-CLP-34B 75.6 vs WizardCoder-CLP-34B 73.2. More evolved data does not scale linearly: 13B was best with 30K evolved samples plus fused data.
- **Limitations / failure modes:** No execution-based check. Code only.
- **How to reuse with easy seeds:** Once depth evolution saturates, fuse pairs of easy seeds to create multi-objective tasks. Use an explicit "INVALID" escape. For code, require generated tests to pass.

### CodecLM — CodecLM: Aligning Language Models with Tailored Synthetic Data (Wang et al., 2024)
Link: https://arxiv.org/abs/2404.05875 (Findings of NAACL 2024)
- **Mechanism:** A strong LLM (Gemini-Pro for the LLaMA experiments) encodes seed instructions into metadata (use case plus skills) and decodes it into base instructions. **Self-Rubrics** then generates 4 metadata-specific rubrics and actions, for example "add SWOT analysis" for business plans. Each iteration applies one random action, for up to 4 iterations. **Contrastive Filtering** has the strong LLM (also acting as scorer) and the target model both answer, scored on a 10-point scale. If |gap| > 3, the pair is kept: the strong answer normally, or rarely the target's answer as regularization. Otherwise the prompt goes back to Self-Rubrics to be made harder.
- **How it makes tasks harder:** Domain-specific rubric actions rather than generic rewrites.
- **Correctness / verification:** Strong-LLM responses; the LLM judge.
- **Difficulty control:** Explicitly target-aware through the teacher–student quality gap, with a threshold of 3 tuned on AlpacaEval.
- **Reported results:** CRR (wins+ties vs Gemini-Pro), LLaMA-7B: Evol-Ins. set 79.82 vs WizardLM 74.31; Vicuna 88.75 vs 76.25; Koala 74.44 vs 65.56; Self-Ins. 78.17 vs 71.43. LLaMA-13B Evol-Ins. set: 86.70 vs 82.11.
- **Limitations / failure modes:** Metadata was extracted from 20% validation splits of the evaluation benchmarks, a tailoring that also risks benchmark fit. The judge is noisy.
- **How to reuse with easy seeds:** For verifiable tasks, replace the judge gap with a pass-rate gap between reference solver and policy. Recycle small-gap prompts into another harden step instead of discarding them.

### Conifer — Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models (Sun et al., 2024)
Link: https://arxiv.org/abs/2404.02823
- **Mechanism:** Start with 6,000 random ShareGPT queries. GPT-4 reframes each into at least 3 variants, then generates a **two-level** constraint list (category → item, e.g. Cultural Background → Chinese) with in-context examples. Recombination builds levels 1–5, each adding 1–2 categories and 2–3 items over the previous level. GPT-4 prefers content constraints, so 1,000 instructions were also given format or numerical constraints from a curated set of 32. A two-stage GPT-4 filter drops prompts lacking context (after reframing) and prompts with conflicting constraints (after recombination). The result is 35,613 instructions with GPT-4 answers.
- **How it makes tasks harder:** Constraint count and variety per level.
- **Correctness / verification:** GPT-4 answers and conflict filtering. "Internal" process feedback has GPT-4 explain how each format or number constraint is met. "External" feedback builds {hard instruction, model response, GPT-4 critique, GPT-4 response} turns from level-5 items.
- **Difficulty control:** Levels 1–5, packaged easy→hard as multi-turn conversations: 13,606 conversations mixed with 53K ShareGPT.
- **Reported results:** Conifer-7B-DPO: IFEval (loose prompt) 52.3, FollowBench level-5 41.0, InFoBench 82.3 (ShareGPT-DPO baseline: 48.2 / 38.6 / 82.0). In the ablation (Mistral-7B SFT), full Conifer scored IFEval 50.8 and FollowBench average 44.9, against random shuffle 48.6 / 44.1, hard-to-easy 49.9 / 41.7, and hard-only 48.2 / 43.4.
- **Limitations / failure modes:** No programmatic verification. Curriculum gains are 1–2 points on a single run.
- **How to reuse with easy seeds:** Produce a level-1…5 ladder per seed, with more constraint categories per level. Package the levels as multi-turn easy→hard conversations. Add critique-and-revise turns for the hardest level.

### Auto Evol-Instruct — Automatic Instruction Evolving for Large Language Models (Zeng et al., 2024)
Link: https://arxiv.org/abs/2406.00770 (EMNLP 2024, 2024.emnlp-main.397)
- **Mechanism:** The initial universal evolving method is a 4-step prompt: list methods to make the instruction more complex; plan using several of them; rewrite, adding only 10–20 words; review and fix unreasonable parts. At each optimization step, the evol LLM evolves a mini-batch of 10 instructions for l rounds. An optimizer LLM (GPT-4) reads the trajectory and names failures, then proposes 5 improved methods. The method with the lowest failure rate λ on a 50-example dev set is kept. Optimization runs for 10 steps on about 2,000 sampled instructions, then the method evolves the full dataset. By default this is **one** round.
- **How it makes tasks harder:** Domain-adapted methods discovered automatically. Code methods, for example, involve time/space complexity, which the paper notes does not suit chat.
- **Correctness / verification:** The failure function F is rule-based on the evol LLM's *response* to the evolved prompt. Stagnant complexity: the response starts "Understood"/"Thank you" and ends with "?". Insufficient qualification: starts "Sure" and ends with "?". Loss of key information: contains "please provide". On 200 GSM8K evolutions, the initial method also produced changed problem nature, irrelevant details, contradictions, and "incorrect or unrealistic mathematical calculations". Answers are not verified.
- **Difficulty control:** Implicit.
- **Reported results:** Large-model row, seed → Evol-Instruct → Auto Evol-Instruct: MT-Bench 7.65 → 7.76 → 8.09 and AlpacaEval 87.98 → 89.50 → 91.37 (Mixtral-8x7B, 10K ShareGPT); GSM8K 70.60 → 79.15 → 82.49 (Mixtral-8x7B, 7K); HumanEval 72.0 → 73.2 → 77.4 (DeepSeek-Coder-33B, 20K Code Alpaca). GPT-4 was both evol and optimizer LLM in the main runs.
- **Limitations / failure modes:** The failure detectors are surface heuristics. The objective is still complexity.
- **How to reuse with easy seeds:** Run a small prompt-optimization loop per task family. Add *policy pass rate in band* and *verifier success* to λ so the optimizer tunes for usable difficulty.

### AutoIF — Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models (Dong et al., 2024)
Link: https://arxiv.org/abs/2406.13542 (ICLR 2025; code github.com/QwenLM/AutoIF)
- **Mechanism:** Humans write seed instructions containing a single atomic constraint each. An LLM, "not necessarily advanced", self-instructs more of them and writes verification functions plus test cases. Cross-validation keeps an instruction only if each test case gets >0.5 accuracy across functions and each function gets >0.5 across test cases. The verification function is then back-translated to an instruction and checked against the original with NLI (mDeBERTa), dropping contradictions. Verified constraints are concatenated with K ShareGPT queries. K responses are sampled; each must pass >0.5 of the verification functions, and an LLM query–instruction match score ≥ 8 is required, which removes conflicts such as "write a news article" plus "two words". The data serves SFT, offline DPO, and online DPO with execution feedback.
- **How it makes tasks harder:** Attaches verifiable constraints to real queries.
- **Correctness / verification:** Executable, cross-validated verifiers. The same functions give DPO pairs and RL rewards.
- **Difficulty control:** Number and type of constraints. No explicit pass-rate band.
- **Reported results:** LLaMA3-70B-Instruct with online DPO: IFEval loose-instruction 90.4 (first above 90), FollowBench SSR average 66.5. Qwen2-72B: 88.0 and 67.5.
- **Limitations / failure modes:** Mostly format and lexical constraints. Verifier bugs become reward bugs.
- **How to reuse with easy seeds:** Wrap any easy prompt with LLM-written, cross-tested checkers and reuse them as RLVR rewards. Keep the query–constraint compatibility judge.

### Persona Hub — Scaling Synthetic Data Creation with 1,000,000,000 Personas (Ge et al., 2024)
Link: https://arxiv.org/abs/2406.20094 (code github.com/tencent-ailab/persona-hub)
- **Mechanism:** Text-to-Persona ("who would read or write this?") runs on RedPajama v2. Persona-to-Persona then expands by interpersonal relations for 6 iterations. Dedup uses MinHash (1-gram, signature 128, threshold 0.9) and embedding cosine at 0.9. Prompts combine a persona with an optional focus or difficulty, e.g. "Olympiad-level".
- **How it makes tasks harder:** Mainly breadth and realism. Difficulty comes only from the explicit difficulty phrase.
- **Correctness / verification:** Problems were written 0-shot by GPT-4, and GPT-4o wrote the training solutions **without verification**. For the 20K held-out synthetic test set, answers were kept only where at least 2 of 3 solutions agreed (gpt-4o, gpt-4o PoT, gpt-4-turbo), leaving 11.6K.
- **Difficulty control:** Prompt wording only.
- **Reported results:** 1.07M training problems from 1.09M personas. The fine-tuned Qwen2-7B scored 64.9% on MATH with greedy decoding (gpt-4-turbo-0125-preview: 64.5) and 79.4 on the in-distribution synthetic test.
- **Limitations / failure modes:** The authors warn that the in-distribution answers "are not absolutely reliable". Persona diversity is not difficulty.
- **How to reuse with easy seeds:** Use personas as a *breadth* operator stacked with depth operators ("rewrite this seed as a problem this persona would face at level X"). Always add answer agreement and dedup.

### AgentInstruct (Orca-3) — AgentInstruct: Toward Generative Teaching with Agentic Flows (Mitra et al., 2024)
Link: https://arxiv.org/abs/2407.03502
- **Mechanism:** Three flows run over raw documents and code. **Content Transformation** uses, for reading comprehension, nine agents that produce argument passages, debates, meeting transcripts, poems, satire, and so on. **Seed Instruction Generation** is taxonomy-driven: more than 100 subcategories overall, and 43 reading-comprehension question types. **Instruction Refinement** uses Suggester→Editor agents to make tasks "more complex, unsolvable, or tricky". Agents may use tools such as search, a code interpreter, or a calculator. Flows cover 17 skills.
- **How it makes tasks harder:** Refinement goals for reading comprehension: (1) modify the passage so the question becomes unanswerable; (2) modify the passage to flip the answer; (3) make the question or options more complex. A verified example: "Include a distractor option that seems to strengthen the argument but … does not directly relate to the causal relationship" (uric acid → cardiovascular disease).
- **Correctness / verification:** Strong-LLM answers grounded in the transformed passage; no formal verifier.
- **Difficulty control:** Number and type of Suggester–Editor passes.
- **Reported results:** About 22M AgentInstruct instructions plus about 3.8M others, 25.8M total, used to fine-tune Mistral-7B into Orca-3. Against Mistral-7B-Instruct: +40% AGIEval, +19% MMLU, +54% GSM8K, +38% BBH, +45% AlpacaEval, and 31.34% less hallucination in summarization.
- **Limitations / failure modes:** Cost; answers to hardened items are unverified; hand-built taxonomy.
- **How to reuse with easy seeds:** Add a Suggester→Editor pass with a per-task-type menu (distractor, answer flip, unanswerable variant, format shift). Ground each task in a document so answer invariance can be checked.

### Genetic-Instruct — Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models (Majumdar et al., 2024)
Link: https://arxiv.org/abs/2407.21077 (ACL 2025)
- **Mechanism:** A genetic algorithm with three roles: an Instructor-LLM (temperature 1.2), a Coder-LLM, and a Judge-LLM (temperature 1.0). **Mutation** randomly picks one of WizardCoder's 5 Evol heuristics (batch 100). **Crossover** is Self-Instruct-style few-shot generation of several new prompts from several parents (batch 10). Mutation probability Mp is 0.5. 20 colonies run in parallel from the same seeds and merge into a "generation", which becomes the next seed population.
- **How it makes tasks harder:** Mutation adds depth; crossover adds novelty.
- **Correctness / verification:** Solutions that fail to parse or compile are dropped (AST check), and a few-shot CoT Judge-LLM gives a binary pass/fail fitness. The authors call execution "the ideal case" but did not use it.
- **Difficulty control:** Generations; operator mix.
- **Reported results:** More than 7.5M pairs from 512 seeds. Llama-3.1-8B-Base on 7.5M: MBPP 79.9, MBPP+ 69.1, HumanEval 66.5, HumanEval+ 63.4 (average 69.7; Llama-3.1-8B-Instruct 65.9). At equal size (4M) and generator: Genetic 68.0 vs Self-Instruct 66.8 vs WizardCoder-style 65.7. Gains plateau after about 6M samples.
- **Limitations / failure modes:** A judge-only fitness lets subtly wrong code through.
- **How to reuse with easy seeds:** Combine mutation and crossover over a population, and parallelize by colonies. Replace judge fitness with execution against generated tests and with the target model's failure rate.

### Instruct-SkillMix — Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning (Kaur et al., 2024)
Link: https://arxiv.org/abs/2408.14774 (ICLR 2025)
- **Mechanism:** GPT-4-Turbo extracts instruction-following topics, skills, and query types ("LLM metacognition"). Each example samples k random skills (mainly k=2) plus a query type, and the teacher writes a Q&A pair that exhibits them. No seed dataset is needed.
- **How it makes tasks harder:** Random skill combinations are meant to add diversity and difficulty.
- **Correctness / verification:** Teacher quality only.
- **Difficulty control:** k.
- **Reported results:** 4K examples: LLaMA-3-8B-Base reached 42.76% LC win rate on AlpacaEval 2.0, at a cost under $600. Replacing 20% of responses with short "shirker" answers (Mistral-7B-Base, ISM-D k=2, 2K) dropped LC win rate from 31.57% to 23.93%. k=2 was only marginally better than k=1. Performance saturated early, around 4K examples.
- **Limitations / failure modes:** Chat benchmarks only; limited composition gains.
- **How to reuse with easy seeds:** Extract a skill inventory from your easy tasks and generate tasks requiring random pairs or triples. Treat response quality as the binding constraint.

### IDEA-MCTS — Optimizing Instruction Synthesis: Effective Exploration of Evolutionary Space with Tree Search (Li et al., 2024)
Link: https://arxiv.org/abs/2410.10392
- **Mechanism:** Each seed is the root. Actions come from a **12-action** evolution space: add global/local goals, key constraints, task requirements, problem-solving skills, reasoning complexity, domain knowledge, life topics, real-world applications, emotional expression; format input style; format output style; create a new one. More task-specific actions are extracted by a meta-prompt from benchmark-related instructions (AlpacaEval). The evolver is GPT-3.5-turbo-0125 at temperature 0.7. MCTS uses UCT with C=1, expands 5 nodes per step, and runs 3 iterations. A path terminates at depth > 4 or reward > 10. Node value is the sum of quality + complexity + diversity scores from three LLaMA2-7B scorers trained on ChatGPT scores; diversity is the number of distinct intents via InsTagger. **1,000 training samples were drawn at random from all nodes on root-to-terminal paths**, not only best leaves.
- **How it makes tasks harder:** Searched multi-step rewrite sequences.
- **Correctness / verification:** The quality term in the value; no answer verification.
- **Difficulty control:** Depth and the complexity term.
- **Reported results:** Average data score (quality/InsTag/complexity) rose from 2.19 for the seed to 3.81; complexity alone from 1.40 to 3.62. AlpacaEval, 1K-sample fine-tuning, **seed → MCTS+**: LLaMA2 45.59 → 51.61; LLaMA3 56.27 → 60.37; Phi-3 55.22 → 62.36; Mistral 57.52 → 62.80. For comparison, Evol-Instruct on LLaMA3 scored 57.08. MT-Bench average 5.90 → 6.94.
- **Limitations / failure modes:** Many LLM calls per seed; proxy value models; QLoRA low-resource setting only.
- **How to reuse with easy seeds:** Search short action sequences per seed, with policy failure rate and verifier pass as extra value terms.

### Crab — Constraint Back-translation Improves Complex Instruction Following of Large Language Models (Qi et al., 2024)
Link: https://arxiv.org/abs/2410.24175 (code github.com/THU-KEG/Crab)
- **Mechanism:** Seeds are high-quality pairs from Alpaca-GPT4, Orca-Chat, Evol-Instruct, and OpenAssistant, with responses over 300 words (4,500 raw instances). Llama3-70B-Instruct generates constraints **already satisfied by the response**, using 13 manually collected constraint types as examples, which yields over 100 types. It then re-verifies each and drops unmet ones. Six types (length, keywords, punctuation, etc.) are filled by Python scripts that read the actual value from the response. For length, a random range containing the true value is sampled. Similar constraints are removed at ROUGE-L 0.6. Each instruction gets **6–8** sampled constraints in shuffled order. Half get 1–3 in-context demos. "Reverse training", generating the constraints from (x, y), is an auxiliary objective.
- **How it makes tasks harder:** Dense multi-constraint prompts from easy prompts, without a new answer.
- **Correctness / verification:** By construction, the original response is the reference, plus LLM re-verification and scripted values. A manual review of 50 pairs found "minimal noise".
- **Difficulty control:** Number of constraints (6–8).
- **Reported results:** In RECAST's comparison (Llama3.1-8B), Crab scored IFEval Pr(L) 42.14 and FollowBench HSR 41.90. That is below ShareGPT (46.40 / 45.41) in that setup, so the gain is benchmark- and mix-dependent.
- **Limitations / failure modes:** Constraints are read off one response, so they can be idiosyncratic and copyable. See UNSPECIFIC.
- **How to reuse with easy seeds:** This is the cheapest correctness-preserving hardening. For each easy (prompt, verified answer), extract properties the answer has and state them as explicit, script-checkable constraints. Use value *ranges* rather than exact copies.

### Tulu 3 Persona-IF / IFBench-IFTrain — Tulu 3: Pushing Frontiers in Open Language Model Post-Training (Lambert et al., 2024); Generalizing Verifiable Instruction Following (Pyatkin et al., 2025)
Link: https://arxiv.org/abs/2411.15124 · https://arxiv.org/abs/2507.02833 (NeurIPS 2025 D&B; code github.com/allenai/IFBench)
- **Mechanism:** *Tulu 3.* 1–2 hand-written example instructions for each of the 25 IFEval constraint types (33 seeds). GPT-4o with a persona and one example constraint generates new prompts and responses, giving 29,980 "IF-Persona-SFT" pairs. Preference pairs come from GPT-4o rewriting the prompt to **relax one constraint**. The response to the relaxed prompt fails the original, so it becomes the rejected sample (about 20K). "IF-augmented" prompts (Tulu 2 SFT prompts plus IFEval-taxonomy constraints) are used for DPO and RLVR. Of about 66K generated, the roughly 26K whose chosen responses actually passed the verifiers were kept. *IFBench/IFTrain.* 58 new out-of-domain test constraints and 29 new training constraints, each with a Python verifier. GRPO (IF-RLVR) on prompts with 1–6 stacked constraints and varied variable ranges.
- **How it makes tasks harder:** More constraints per prompt, wider or unseen constraint families, and wider parameter ranges.
- **Correctness / verification:** A Python verifier per constraint. Against over-optimization, final reward F = V+1 if V>0 and RM score S>α; F = V−0.5 if V>0 and S≤α; else V. Here α=7 and the RM is Llama-3.1-Tulu-3-8B-RM.
- **Difficulty control:** Constraints per prompt and variable ranges. Wider ranges performed comparably to or better than matched ranges; disjoint ranges were worse.
- **Reported results:** Training on up to 5–6 constraints per prompt beat training on up to 3, even though IFEval has at most 3 and IFBench at most 2. Tulu-3-8B: IFEval 82.4 → 92.2 and IFBench 28.9 → 45.9 (as stated in the intro; Table 3 lists 44.6 for the DPO-initialized run). Qwen2.5-7B base: 87.8 / 54.7 (table: 53.7). **Caveat from the same table:** the IF-only-RLVR Qwen2.5-7B base scored AlpacaEval 1.1 and GSM8K 15.3, and the run initialized from Tulu-3-8B-DPO fell from AlpacaEval 33.5 to 21.3. Adding the RM term gave IFEval 86.1 and IFBench 30 after 1,100 steps. Tulu 3 removed 3.5% of Evol-CodeAlpaca and 11.3% of NuminaMath-TIR by 8-gram decontamination. Evol-CodeAlpaca overlapped **70.7%** of HumanEval.
- **Limitations / failure modes:** Models overfit to IFEval-style families. Constraint-only RL damages general quality unless it is mixed.
- **How to reuse with easy seeds:** Compose 3–6 verifiable constraints per prompt from a library larger than the eval's, and randomize parameters widely. Use "relax one constraint" rewrites to mine hard negatives for DPO. Mix IF-RLVR with general data, or add an RM gate. Ready seeds: `allenai/tulu-3-sft-personas-instruction-following`, `allenai/tulu-3-pref-personas-instruction-following`.

### Smaller LMs as instruction evolvers + IC-IFD — Smaller Language Models Are Better Instruction Evolvers (Hui et al., 2024)
Link: https://arxiv.org/abs/2412.11231 (work in progress)
- **Mechanism:** Compares evolvers (Llama-3.1-8B/70B-Instruct, Qwen-2-7B/72B-Instruct) under Evol-Instruct, AutoIF, and Auto Evol-Instruct. Large models put more mass on top-1 tokens ("overconfidence"), so their outputs span a narrower space. Proposes **IC-IFD = L(A|Q) / (L(Q)·L(A))**, which adds instruction perplexity to IFD to penalize prompts that are hard to *understand* rather than hard to *answer*.
- **How it makes tasks harder:** The small evolvers produce more complex and diverse variants, as the paper claims.
- **Correctness / verification:** Not addressed.
- **Difficulty control:** Evolver choice and IC-IFD thresholding.
- **Reported results:** Evol-Instruct, Llama-3.1-8B base model trained on data evolved by **8B vs 70B** evolvers: IFEval Ins.(S) 50.96 vs 46.04, GSM8K 67.10 vs 64.22, MATH 13.12 vs 11.32, MBPP 41.60 vs 40.60, but HumanEval **48.78 vs 51.22 (70B better)**. A third round of SLM evolution degraded performance. Keeping the top 25% by IC-IFD recovered it (IFEval Pr.(S) 34.01 vs 33.09 on the full set), while length, perplexity, and IFD filters did worse.
- **Limitations / failure modes:** Not uniform across benchmarks. The opposite evolver finding appears in WizardLM, where LLaMA-2-70B-Chat was worse than ChatGPT. Answers still need a strong model.
- **How to reuse with easy seeds:** Decouple a cheap, high-temperature proposer from a strong answerer and verifier. Filter out incomprehensible prompts with IC-IFD or perplexity-aware scores.

### Phi-4 synthetic data pipeline — Phi-4 Technical Report (Abdin et al., 2024)
Link: https://arxiv.org/abs/2412.08905
- **Mechanism:** About 50 synthetic dataset types, roughly 400B unweighted tokens. Seeds are web, book, and code passages scored for reasoning and educational value, plus collected questions. Q&A pairs are also extracted from deduction chains in organic text. Seeds are rewritten into exercises, discussions, and structured reasoning tasks through multi-step prompting. Answers go through rubric-guided self-revision. For code, **instruction reversal** generates the instruction from existing code.
- **How it makes tasks harder:** Question seeds are **plurality-banded**: "We discarded questions where all answers agreed (… too easy) or where answers were entirely inconsistent (… too difficult or ambiguous)". The plurality answer then serves as ground truth for rejection sampling. For DPO pivotal-token search, questions were filtered to 0.2 ≤ p(success) ≤ 0.8.
- **Correctness / verification:** Majority vote; code validated "through execution loops and tests"; reversal kept only pairs with "high fidelity between the original and regenerated code".
- **Difficulty control:** Agreement bands.
- **Reported results:** Scale figures above; per-operator ablations are limited.
- **Limitations / failure modes:** Prompts are not released.
- **How to reuse with easy seeds:** Band seeds by agreement before evolving anything. For code, round-trip instruction↔code as a correctness filter.

### UltraIF (UltraComposer) — UltraIF: Advancing Instruction Following from the Wild (An et al., 2025)
Link: https://arxiv.org/abs/2502.04153 (EMNLP 2025; code github.com/kkk-an/UltraIF)
- **Mechanism:** Real user prompts (ShareGPT, OpenHermes, No Robots) are decomposed into a simplified query, constraints, and one yes/no evaluation question per constraint. For example, "In Shakespeare's tone, recommend me ten Chinese books" becomes query "recommend me ten Chinese books", constraint "in Shakespeare's tone", and question "Is the response in Shakespeare's tone?". **UltraComposer**, fine-tuned from LLaMA-3.1-8B-Instruct, learns simple → (complex prompt, evaluation question) and is applied iteratively. K responses are sampled, and the evaluation questions pick chosen and rejected. The whole pipeline needs 3–4 LLM calls per sample.
- **How it makes tasks harder:** Adds realistic constraints learned from the wild, stacked over iterations.
- **Correctness / verification:** An LLM answers the per-constraint evaluation questions. Removing this filter cost 3.35–5.36 points, and the gap widened with complexity.
- **Difficulty control:** Composition iterations. Performance rose from 1 to 3 iterations (max tested). In iterative DPO, iteration 3 diverged, with both chosen and rejected rewards going negative, and was fixed by switching to NCA.
- **Reported results:** Self-alignment with only an 8B supervisor, LLaMA-3.1-8B-Base, 181K SFT + 20K DPO: IFEval Pr(S) 71.35 and Ins(S) 79.38, InfoBench DRFR 80.70, FollowBench SSR 62.55, Multi-IF turn-1 69.63, LiveBench 56.00. LLaMA-3.1-8B-Instruct: 69.13 / 77.46. In strong-to-weak runs (70B supervisor, 10K), reimplemented **Evol-Instruct scored below the ShareGPT baseline** on IFEval Pr(S) (41.96 vs 43.99).
- **Limitations / failure modes:** Evaluation questions are LLM-judged. Realism depends on the source corpus.
- **How to reuse with easy seeds:** Mine your hard real prompts, decompose them, and train a small composer. Use its evaluation questions as rubric rewards. Switch to a more stable preference loss (NCA) for later rounds.

### MathFusion — MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion (Pei et al., 2025)
Link: https://arxiv.org/abs/2503.16212 (ACL 2025; code github.com/QizhiPei/MathFusion)
- **Mechanism:** Each GSM8K or MATH problem is paired with its most similar problem by inner product of text-embedding-3-large vectors. GPT-4o-mini fuses the pair: **sequential** (A's answer becomes an input to B), **parallel** (one problem covering the shared concept), or **conditional** (compare or choose between the outcomes). GPT-4o-mini also writes the solutions.
- **How it makes tasks harder:** Multi-step dependency and cross-problem reasoning.
- **Correctness / verification:** In an error analysis, GPT-4o-mini checked fused problems and regenerated failures up to 5 times at temperature 1.0. 5.6% stayed unreasonable. Training only on the filtered set gave essentially the same score (39.1 vs 39.0, Llama3-8B).
- **Difficulty control:** Fusion type.
- **Reported results:** +18.0 points average accuracy across benchmarks with only 45K extra instructions (MathFusionQA = 15K originals + 3×15K fused). Sequential and parallel fusion generally beat conditional.
- **Limitations / failure modes:** Answers unverified; the "noise doesn't matter" result is for SFT, not RLVR rewards.
- **How to reuse with easy seeds:** When both seeds have verified numeric answers, sequential fusion can be *checked*: substitute A's answer into B and recompute. Otherwise require k-sample agreement before RL.

### TAG-INSTRUCT — TAG-INSTRUCT: Controlled Instruction Complexity Enhancement through Structure-based Augmentation (Zhu et al., 2025)
Link: https://arxiv.org/abs/2505.18557
- **Mechanism:** Each instruction is compressed into semantic tags; for example, "Create an HTML page that displays a list of books and their authors" becomes {book_listing, author_info, web_develop}. A **policy model**, initialized from the teacher, is DPO-trained to propose new tags. Chosen and rejected pairs come from a utility score: a Shapley-inspired value approximated as the **average response length** of instructions containing the tag in a reference dataset, plus embedding similarity to high- vs low-utility tags. Semantic alignment and global coherence are prompted criteria. The teacher regenerates the instruction from the expanded tags over several rounds.
- **How it makes tasks harder:** Controlled addition of coherent sub-requirements in tag space.
- **Correctness / verification:** Coherence criteria only; teacher answers.
- **Difficulty control:** Expansion rounds and tags per round.
- **Reported results:** LLaMA3-8B base with Ministral-8B teacher on Alpaca-5K, gains over baseline: AlpacaEval 2 LC +10.17 (Evol-Instruct +2.96, Tree-Instruct +1.56, CodecLM +5.40), Arena-Hard +19.0 (+13.7 / +13.5 / +16.1), MT-Bench +0.95 (+0.48 / +0.74 / +0.80).
- **Limitations / failure modes:** The length-based utility rewards verbosity. Chat benchmarks only.
- **How to reuse with easy seeds:** Train a small expansion policy, but define tag or constraint utility from **your policy's failure rate** or downstream deltas, not response length.

### RECAST — RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data (Guo et al., 2025)
Link: https://arxiv.org/abs/2505.19030 (ICLR 2026; code github.com/Thekey756/RECAST). An earlier arXiv version was titled differently.
- **Mechanism:** Seeds are Tulu 3 Persona IF. **Nine** rule-based extractors read checkable properties from seed responses: paragraph count, keyword inclusion or exclusion, word limits, and so on. For model-based constraints, an LLM picks applicable types from a **10-category** taxonomy (19 constraint types in total), generates instances, and keeps only those the seed response satisfies, which was human-validated. An LLM selects a coherent subset per instruction. Several LLMs integrate the subset into rewrites, and majority voting across LLMs picks the best rewrite for fluency, coherence, and completeness. **New responses are then generated by several LLMs and majority-voted**, so the original response is *not* reused as the reference. **RLVC** runs GRPO with reward = mean satisfaction over constraints: rule validators for quantitative constraints, LLM validators for qualitative ones.
- **How it makes tasks harder:** Very dense constraint sets. The authors note existing datasets "do not exceed 10 constraints per instance".
- **Correctness / verification:** Constraints are guaranteed *achievable*, because a real response satisfied them. Responses are selected by multi-LLM voting and scored by validators.
- **Difficulty control:** Constraint count. RECAST-Test has 4 difficulty levels by constraint count.
- **Reported results:** RECAST-30K averages 13.4 constraints per instruction (8.6 model-based, 4.8 rule-based), at a total cost of about $175. Llama3.1-8B: IFEval Pr(L) 76.34 (SFT) / 77.39 (RLVC) vs 73.94 for Tulu 3 Persona IF; FollowBench HSR 57.10 / 61.76 vs 55.44. Qwen2.5-7B RECAST-Test average: 31.25 (SFT) → 32.33 (RLVC). Gemini-2.5-Pro, the best frontier model tested, scored 39.75.
- **Limitations / failure modes:** LLM validators are noisy. All frontier models degrade steeply from level 1 to level 4.
- **How to reuse with easy seeds:** Mine constraints from existing good answers, add 10+ per prompt, then use mean-satisfaction rewards with GRPO. Keep the "satisfied by a real response" test as a feasibility guarantee.

### Tag-Evol — Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection (Wang et al., 2025)
Link: https://arxiv.org/abs/2505.24165 (Findings of ACL 2025; code github.com/fghccv/TagEvol)
- **Mechanism:** Two-phase tagging: first abstract **aspects** (task type, skills, arithmetic type), then concrete knowledge **tags** per aspect. This gave about **20×** more tags than the original tagging and **2×** more than single-step tagging. For each seed, a difficulty budget b is set, and the evolver receives a batch of candidate tags. It then (1) selects b fitting tags, (2) plans the injection, (3) executes the rewrite, and (4) rewrites again to remove hallucinations.
- **How it makes tasks harder:** Specific knowledge requirements, e.g. "exponential correlation", instead of a generic "add a constraint".
- **Correctness / verification:** Teacher answers; the hallucination-removal rewrite.
- **Difficulty control:** Budget b, with 1/3/5 tags for math and 3/5/7 for code. Any difficulty can be produced directly without chaining. The three budgets were run as three "rounds" only for comparison with Evol-Instruct. Even 7B evolvers executed it well.
- **Reported results:** Mistral-7B-base average over MT-Bench, IFEval, GSM8K, MATH, HumanEval, and MBPP: seed 33.9, Evol-Instruct 40.0, Auto Evol-Instruct 41.4, Tag-Evol 43.7. GSM8K rose from 67.0 to 69.3 as the tag pool grew.
- **Limitations / failure modes:** Incompatible tag combinations; answers unverified.
- **How to reuse with easy seeds:** Build a concept or tag pool from your corpus and generate difficulty-k variants in one shot, avoiding error accumulation from chaining. Add execution or agreement checks for math and code.

### SynthQuestions — From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding (Zhu et al., 2025)
Link: https://arxiv.org/abs/2506.03968 (ACL 2025; code github.com/Ignoramus0817/SynthQuestions)
- **Mechanism:** Real instructions are scored by LLaMA-3-70B-Instruct on 7 Arena-Hard-style criteria, one point each. The 29K full-score items form REAL QUESTIONS. *Top-down:* each is attributed to a web document (Google top-1 on key concepts) plus a simulated user and motivation. *Bottom-up:* for FineWeb documents, with PILE and MathPILE mixed in for reasoning, LLaMA-3-8B-Instruct builds a grounded user and motivation, then utters an instruction, using attributed examples as demonstrations.
- **How it makes tasks harder:** Realistic situational grounding plus score thresholds.
- **Correctness / verification:** None for answers. Instructions scoring below 3 are dropped, and the highest-scoring items per BERTopic topic are kept, giving 1M.
- **Difficulty control:** Criteria scores; math and code document mixing.
- **Reported results:** LLaMA-3-8B SFT on 1M: Arena-Hard 15.4, AlpacaEval 2 LC 18.87 and WR 19.15. Comparisons: OpenHermes2.5 4.4 / 9.94 LC; GenQA 3.0 / 9.05 LC; MAmmoTH2 (10M) 16.6 / 18.5 LC.
- **Limitations / failure modes:** LLM-judged complexity only.
- **How to reuse with easy seeds:** Make easy seeds realistic and harder by grounding them in a document plus a user persona and motivation. Then apply verifiable depth operators.

### Infinity Instruct (InfInstruct-G) + Subject report — Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models (Li et al., 2025); Scaling Towards the Information Boundary of Instruction Sets: The Infinity Instruct Subject Technical Report (Du et al., 2025)
Link: https://arxiv.org/abs/2506.11116 · https://arxiv.org/abs/2507.06968 (data: huggingface.co/datasets/BAAI/Infinity-Instruct)
- **Mechanism:** Phase 2 starts from about 9M open instructions. Qwen1.5-72B produces labels: 26 first-level and more than 15K second-level. Seed selection yields 1.2M: long-tail labels (frequency 20–200 kept in full, 200–500 at one third), multi-capability prompts, high answer loss under Qwen1.5-7B, and removal of samples whose loss changes a lot after fine-tuning (overfit-prone). Seeds are evolved with Evol-Instruct strategies, and the rewriter checks semantic identity and harmful content. **Diagnosis:** GPT-4 grades evolved-instruction responses from Mistral-7B and Llama3-8B, and prompts with poor responses go into the next evolution round. The Subject report formalizes four seed criteria: the 50K "hard-to-follow" prompts with the smallest loss reduction after fine-tuning; long-tail tags; more than 4 fine-grained tags; and "undertrained" prompts with loss above mean + 1.96σ under Llama-2-7B. It defines **depth = log(label count) × token-level log loss**.
- **How it makes tasks harder:** Choose hard or rare seeds first, evolve them, then re-target where models fail.
- **Correctness / verification:** Rewrite equivalence and harm checks; no answer verification. Decontamination uses BGE cosine similarity against benchmarks.
- **Difficulty control:** Loss-based seed selection and the deficiency loop.
- **Reported results (corrected):** InfInstruct-G is about 1.5M evolved instructions. AlpacaEval 2.0 / Arena-Hard: InfInstruct-Llama3.1-8B 33.9 / 33.7; InfInstruct-Mistral-7B 40.0 / 26.9; InfInstruct-Llama3.1-70B 46.1 / 66.0 vs GPT-4-0314 35.3 / 50.0, which is the source of the "+8.6% over GPT-4-0314" claim. In the Subject report, AlpacaEval and Arena-Hard rose with both depth and coverage at a fixed 20K samples.
- **Limitations / failure modes:** Chat-oriented; heuristic depth metric.
- **How to reuse with easy seeds:** Rank seeds by loss and tag rarity *before* evolving. After each training round, diagnose failing capability tags and re-seed evolution from them.

### SAND-Math (Difficulty Hiking) — SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers (Manem et al., 2025)
Link: https://arxiv.org/abs/2507.20527 (NeurIPS 2025 MATH-AI workshop; data huggingface.co/datasets/amd/SAND-MATH)
- **Mechanism:** DeepSeek-R1 generates questions with co-generated solutions (temperature 0.8), then k=2 fresh solutions (temperature 0.6): 23,437 items. **Self-consistency** keeps a question only if all k answers agree: 17,578, a 74% yield. Semantic dedup (semhash, 0.99) removed 1,293 (7.3%). Decontamination (retrieval top-5 plus a Llama-3.3-70B judge) removed 4. A **performance filter keeps only questions that Qwen2.5-32B-Instruct answers incorrectly**: 9,211, 56.6% retained. A web-search novelty filter (τ=0.85) removed 4%, leaving 8,842. Llama-3.3-70B rates difficulty 1–10 against an AoPS-referenced rubric, averaged over 3 runs.
- **How it makes tasks harder:** **Difficulty Hiking** re-prompts the teacher with the question, its difficulty rating, and a **forced synthesis of a same-branch theorem with a cross-domain concept**. The prompt appendix also lists adding new constraints. Only one hiking step was evaluated.
- **Correctness / verification:** 2-sample agreement plus decontamination.
- **Difficulty control:** Solver-failure filter and rating-conditioned hiking.
- **Reported results:** Hiking moved mean rated difficulty from 5.02 to 5.98. Qwen2.5-32B with LIMO + 1,500 samples: AIME25 46.38 → 49.23, average 72.94 → 74.39. The "+17.85 AIME25 over next-best synthetic dataset" figure is LIMO + SAND-Math 500 (48.89) vs LIMO + MetaMathQA 500 (31.04). Against LIMO + OpenR1-Math 500 (47.71) the margin is only about 1.2 points.
- **Limitations / failure modes:** k=2 agreement is weak for Olympiad items. Keeping only items a 32B solver fails also keeps wrong-key items, which a same-model-family agreement check may miss.
- **How to reuse with easy seeds:** Keep only items a reference solver fails and that k ≥ 4 strong samples agree on. Then "hike" them by injecting an advanced lemma plus a cross-domain concept, conditioned on the current rating.

### CoT-Self-Instruct — CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks (Yu et al., 2025)
Link: https://arxiv.org/abs/2507.23751
- **Mechanism:** Few-shot seeds are s1k for reasoning (893 items with verifiable closed-form answers) and WildChat-RIP for general prompts. The generator (Qwen3-4B family, Think/NoThink) first reasons about the seeds' domain, complexity, and purpose, plans a new task "of similar quality and complexity", then outputs the prompt and, for reasoning, a target answer. **Answer-Consistency:** sample K solutions and drop the item if the majority answer differs from the generated target. **RIP** (non-verifiable): score K responses with an RM, use the *minimum* as prompt quality, and keep the best of 32 prompts per few-shot context.
- **How it makes tasks harder:** Plan-guided generation at seed-level complexity; filtering removes unsolvable or ill-posed items.
- **Correctness / verification:** Answer-Consistency beats plain self-consistency because the target was produced with the creation-time reasoning.
- **Difficulty control:** Implicit.
- **Reported results:** GRPO on Qwen3-4B-Base, average of MATH500/AIME24/AMC23/GPQA-D: CoT-Self-Instruct 53.0 unfiltered → 55.1 with self-consistency filter → 56.2 with RIP → **57.2 with Answer-Consistency** (2,926 prompts) → 58.7 with 10K. Self-Instruct: 42.7; Self-Instruct + CoT targets + self-consistency: 53.6; s1k: 44.6; OpenMathReasoning 10K: 47.5. Non-reasoning, Llama-3.1-8B online DPO (GPT-4-Turbo judge): AlpacaEval 2 LC 83.2 and Arena-Hard 67.3, vs 80.1 and 64.4 with human WildChat prompts.
- **Limitations / failure modes:** Same-family majority votes can share errors. No explicit difficulty targeting.
- **How to reuse with easy seeds:** A strong default for turning seeds into RLVR prompts. Plan a harder variant, generate prompt plus answer, keep only answer-consistent items, then band by policy pass rate.

### IFDecorator — IFDecorator: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards (Guo et al., 2025)
Link: https://arxiv.org/abs/2508.04632 (code github.com/guox18/IFDecorator)
- **Mechanism:** A "cooperative-adversarial data flywheel" between an Instruction-Former and an Instruction-Solver. In iteration n, a dynamic template is applied n times; from iteration 2 it reorders few-shot examples and tracks already-used constraint types to avoid bias. Up to **3n** programmatically verifiable IFEval-type constraints are added. Each instruction is re-evaluated **8 times**. Items with pass rate above 0.5 (too easy) or equal to 0 (too hard or contradictory) go back for further evolution. LLM checks confirm evolved prompts keep all critical components of the original and stay reasonable. Math, code, and reasoning prompts are filtered out because the verifier cannot score them.
- **How it makes tasks harder:** Keep adding verifiable constraints until the task is *difficult for the solver*, not just longer.
- **Correctness / verification:** Reward = IntentCheck ∧ (all scripts ∧ all LLM criteria). IntentCheck is a strict binary judge of whether the response fulfills the extracted user intent. **Trip wires**, trap instructions invisible to training, measure a Macro Hack Rate over four exploit patterns: format-marker copying, dummy list entries, trivial repetition, and structural-delimiter copying.
- **Difficulty control:** Acceptance band (0, 0.5], reported in bins (0, .125], (.125, .25], (.25, .375], (.375, .5].
- **Reported results:** After 5 iterations: **7,324** in-band samples and **10,772** unsolvable ones (pass = 0), which were dropped. The final set is 3,625 train and 200 val, about 0.71M tokens. Qwen2.5-32B-Instruct-IFDecorator: IFEval 87.43% (GPT-4o 86.50%), FollowBench +4.20%. IntentCheck cut hack rate from 14.53% to 7.60%. Removing the LLM criteria increased hacking ("LLM-based criteria may be more difficult to exploit than script-based verification").
- **Limitations / failure modes:** Mostly format and length constraints; hand-set band and trip wires.
- **How to reuse with easy seeds:** This is the reference recipe for zero-advantage IF prompts. Add verifiable constraints until pass@8 ∈ (0, 0.5], recycle 0/8 items, gate rewards with an intent judge, and monitor with trap prompts.

### LsrIF (added) — LsrIF: Enhancing Logic-Structured Instruction Following of Large Language Models (Ren et al., 2026)
Link: https://arxiv.org/abs/2601.06431
- **Mechanism:** GPT-4.1 takes a seed (instruction, input), selects compatible atomic constraints (23 hard types, 21 soft types; 1–5 per seed), and organizes them into a **constraint tree** of Parallel (And), Sequential (First–Then–Finally), Conditional (If–Else, with the condition generated from the input), and Nested nodes. The tree is rendered into natural language. The dataset has 17,510 parallel, 10,435 sequential, and 10,574 conditional structures. RL uses **structure-aware reward aggregation**: average over parallel children; penalty propagation in sequential nodes, where an early failure discounts later rewards; only the active branch counts for conditional nodes; recursion for nested nodes. Hard constraints are scored by a code verifier and soft ones by an RM (Qwen2.5-7B-Instruct fine-tuned for constraint judging).
- **How it makes tasks harder:** The model must track control flow among constraints rather than satisfy a flat checklist.
- **Correctness / verification:** Per-atomic evaluators plus tree-semantics aggregation.
- **Difficulty control:** Tree depth and composition types.
- **Reported results:** IFEval +2.4 to +25.2, CFBench +2.0 to +11.0, FollowBench +1.7 to +7.0 across models. Qwen2.5-1.5B-Instruct went from 43.6 to 68.8 on IFEval, and Qwen3-8B reached 90.2 IFEval. Out-of-domain gains: ComplexBench up to +6.5, AgentIF up to +8.7. Ablation (Distill-Qwen-7B), IFEval / CFBench / Enigmata-Arith: full 71.5 / 47.0 / 14.3; w/o structure-aware reward 67.6 / 42.0 / 10.7; w/o logic-structured data 69.1 / 45.0 / 7.7.
- **Limitations / failure modes:** Soft constraints depend on the RM; GPT-4.1 builds the data.
- **How to reuse with easy seeds:** Turn flat constraint lists into conditional and sequential trees whose condition depends on the input. Aggregate rewards by tree semantics instead of the mean.

### SEIF (added) — SEIF: Self-Evolving Reinforcement Learning for Instruction Following (Ren et al., 2026)
Link: https://arxiv.org/abs/2605.07465 (code github.com/Rainier-rq1/SEIF)
- **Mechanism:** An Instructor and a Follower are initialized from the same base. Each iteration first trains the Instructor with GRPO, keeping the Follower, **Filter**, and **Judger** frozen; Filter and Judger are instantiated from the latest Follower. Instructor reward is **R_I = 1 − A(x, y)** if the Filter accepts the evolved instruction (A = the Judger's constraint-level satisfaction rate of the frozen Follower's response), and 0 if it is conflicting or invalid. The Follower is then trained with GRPO on the new Instructor's prompts, with reward = constraint-level satisfaction.
- **How it makes tasks harder:** The Instructor learns to add constraints the current Follower fails. There is no band; difficulty is capped by the validity filter.
- **Correctness / verification:** Self-judging (Judger); Filter for conflicts. The constraint-level reward outperformed binary all-or-nothing.
- **Difficulty control:** The adversarial reward plus filter; 3 iterations. Gains saturate after turn 3, and a front-loaded training schedule beat a late-intensive one.
- **Reported results:** Qwen2.5-7B-Instruct: IFEval +4.7, CFBench +4.0, WritingBench +6.6. Distill-Qwen-14B: IFEval 80.0 (+5.1). On SEIF-7B: 78.6 IFEval vs 76.6 for the best static self-play baseline (Meta-Rewarding), and **75.9 without Instructor evolution**. Without the Filter: −3.2 IFEval, −6.0 CFBench.
- **Limitations / failure modes:** The self-judge can drift, and there is no external verifier. Over-training on late rounds hurts.
- **How to reuse with easy seeds:** Train your own "hardener" with reward = 1 − policy success, gated by a validity filter. Refresh the judge and filter from the current policy, and stop at about 3 rounds.

### EvoTD — Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling (Ye et al., 2026)
Link: https://arxiv.org/abs/2605.11666 (code github.com/liqinye/EvoTD)
- **Mechanism:** A task is a (program, input) pair posed as deduction (predict the output), abduction (infer the input), or induction (write the program from I/O). Tasks are abstracted onto two axes: **Algorithmic Skills** (a skill bank extracted from a seed corpus) and **Complexity Attributes** (input size, value range, tree depth, graph size). **Attribute Mutation:** the proposer (o4-mini) audits which attributes apply, then produces n=10 variants that intensify them while keeping the algorithm. **Skill Crossover:** tasks with skill combinations not yet in the population, with all skills "essential and deeply interconnected … rather than appearing sequentially". Fitness = executability × skill alignment × learnability. Induction tasks with avg@k = 0 get natural-language hints.
- **How it makes tasks harder:** Vertical scaling by attributes; horizontal composition by skills.
- **Correctness / verification:** Ground truth by execution. Induction I/O is checked against the reference program. A skill audit rejects drift.
- **Difficulty control:** ZPD filter 0 < avg@k < 1 against the *current* solver, a dynamic curriculum, over 3 iterations.
- **Reported results:** Qwen3-4B (thinking): AIME24 +8.2, AIME25 +7.2, in-domain code average +5.7, out-of-domain math average +6.4; paper-stated +4% pass@1 and +7.4% pass@8. Evol-Instruct raised pass@1 but its pass@8 average (54.2) fell below the untuned backbone (55.1), which the paper calls homogeneity collapse. Removing skill crossover lowered the average (61.1 vs 61.6).
- **Limitations / failure modes:** Requires executable task formats and a strong proposer.
- **How to reuse with easy seeds:** If seeds can be expressed as code, get answers for free. Scale size, depth, and range for depth, cross skills for breadth, filter to 0 < pass < 1 each round, and give hints to all-fail items.

### D2Evo — D²Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning (Zhang et al., 2026)
Link: https://arxiv.org/abs/2605.17037 (ICML 2026)
- **Mechanism:** Difficulty = (1 − correct/N) × 100 with N = 32 rollouts. Each iteration mines **mid-difficulty real anchors** (difficulty band 0.4–0.8) from a small labeled pool: 3,180 OpenRs-7K items with GPT-5.2 CoT solutions matching the reference. A **Questioner** receives (anchor, CoT solution, subject), generates a new problem, and is trained with GRPO. Its reward is r = 1 if solver accuracy (N_v = 10, majority-vote pseudo-label) ∈ [0.4, 0.6], (x/0.4)^2 below, and ((1−x)/0.4)^2 above; the output must use `<question>` tags. The **Solver** trains with GRPO on real anchors plus generated questions re-checked in the band, with pseudo-labels double-checked by GPT-5.2.
- **How it makes tasks harder:** The anchors get harder as the solver improves, and the questioner tracks them.
- **Correctness / verification:** Majority vote plus an external strong-model check; real anchors keep the distribution grounded.
- **Difficulty control:** Two bands: anchor selection and questioner reward.
- **Reported results:** Qwen3-8B-Base (1.7K real samples): 7-benchmark math average 55.32 vs full data (19K) 52.70 vs SPICE (20K) 54.34; MMLU-Pro 62.95 (base 58.8); SuperGPQA 33.71 (base 31.06). Qwen3-4B-Base (1.4K real): +7.48 over base, +4.44 over R-Zero, +4.99 over AZR. Compute 8×10 H20-hours vs 8×7.5 for solver-only and 8×160 for AZR.
- **Limitations / failure modes:** Pseudo-labels are least reliable in the band targeted. Needs an external checker. The paper attributes R-Zero's degradation (no anchors) to entropy collapse and limited diversity.
- **How to reuse with easy seeds:** Train your evolver as an RL policy conditioned on real seeds with solutions. Reward band-hitting on the current policy, and check labels with majority vote plus a stronger model.

### Question Begets Question (added) — Question Begets Question: Self-Evolving Curriculum for Reinforcement Fine-Tuning on Competition Mathematics (Bao et al., 2026)
Link: https://arxiv.org/abs/2608.01522
- **Mechanism:** Qwen2.5-Math-7B on a pool of 1,005 AIME problems (1983–2024; the latest two years held out). Each round: (1) self-evaluate with k=16 samples and bin problems into mastered (16), mostly-right (8–15), sometimes (1–7), and never (0); (2) a teacher (GPT-5-mini) expands **only the mostly-right band** into variants; (3) GRPO (m=8) on an RL set of N=300 variants with 1 ≤ n ≤ 15, picking per seed the variant with solve count closest to k/2, then filling by |n − k/2| under a per-seed cap. Training uses only statement plus answer, never teacher traces.
- **How it makes tasks harder:** Five human-designed **skill-preserving structural operators**, 3 chosen per seed by a planning call:
  - **Generalize, then specialize.** "trailing zeros of 100! in base 10" becomes "in base 12".
  - **Parametrize and sum.** "ordered sums of 4 positive integers equal to 20" becomes Σ_{j=1..5} f(j).
  - **Change the queried quantity.** Triangle 13-14-15: ask for the inradius instead of the area.
  - **Inverse problem.** "how many positive solutions does 20m + 12n = 2012 have?" becomes "find the least c for which 20m + 12n = c has exactly 5".
  - **Add a constraint layer.** "7^2024 mod 100" becomes "(7^2024 + 3^2024) mod 100".
- **Correctness / verification:** Teacher-written answers. Each variant must have an answer different from its parent, which rejects copies and surface rewrites, and changing only numbers is disallowed. No independent re-solve is described in the method section.
- **Difficulty control:** Seed band plus closest-to-k/2 selection; variants that become mostly-right seed the next round.
- **Reported results:** pass@1 5.6% → **16.5%** after 20 rounds with no saturation. Static real + synthetic augmentation capped at 12.5%, and non-curriculum QbQ at 14.5%. Seeding from the harder 1–7 band underperformed. QbQ variants were more diverse (Vendi 1.59 vs 1.25).
- **Limitations / failure modes:** One model and one domain. Teacher answer errors propagate into rewards.
- **How to reuse with easy seeds:** Keep the few seeds your policy *mostly* solves and apply structural operators with a changed-answer check. Select variants nearest 50% solve rate and iterate.

### UNSPECIFIC (added) — UNSPECIFIC: General Constraint Synthesis for Breaking Copy-and-Paste Shortcut in LLM Instruction Following (Sharma et al., 2026)
Link: https://arxiv.org/abs/2608.09154 (code github.com/JeetDSharma/UNSPECIFIC)
- **Mechanism:** Diagnoses a loophole in reference-based constraint synthesis (back-translation): the synthesizer copies specific text from the reference into constraints, and the evaluated model satisfies them by copying. The fix has three parts. (1) Retrieve a similar second article and synthesize only constraints **common to both**, which are more general. (2) Generate an article *without* seeing the constraints, find the constraints it already satisfies, and **harden only those**. (3) Score satisfaction on both the article and **its summary**, to catch constraints met only superficially. Instructions follow a format of one main task plus 39 constraints.
- **How it makes tasks harder:** Selective hardening of trivially satisfied constraints. Hardening everything produces unnatural, over-detailed constraints that invite more copying.
- **Correctness / verification:** Satisfaction judged on the article and the summary.
- **Difficulty control:** Only the "easy" constraints are revised.
- **Reported results:** GPT-5 Mini satisfaction rate dropped from 90% to 78%. The human-judged naturalness win-rate gap improved by 30%. A large share of constraints were satisfied only superficially.
- **Limitations / failure modes:** Primarily a benchmark (news, story, and blog domains).
- **How to reuse with easy seeds:** When back-translating constraints (Crab, RECAST), generalize them across two references, test which ones an unconstrained generation already satisfies, and harden only those.

### VISA — VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following (Zeng et al., 2026)
Link: https://arxiv.org/abs/2608.26013 (vivo AI Lab)
- **Mechanism:** A closed loop. **Perception** filters incompatible constraints for the image and proposes new verifiable types. **Planning & Execution** samples constraint sets weighted by a *coverage* signal (down-weight well-represented types) and a *difficulty* signal (up-weight types the target model historically fails). It writes k=4 candidate instructions and keeps the one whose nearest neighbor in the accepted pool is farthest away, and only then generates a response. **Reflection** verifies with code tools (deterministic constraints) and structured LLM judges (semantic or visual constraints), applies diagnostic repair (local response edits, or instruction rewrite and constraint simplification), discards samples that stop improving, and probes the target model to assign a difficulty d ∈ [0, 1]. **Memory** persists the samples, embeddings, per-constraint difficulty, and the registry.
- **How it makes tasks harder:** Sampling is biased toward under-covered, target-failed constraint types, and the registry keeps growing.
- **Correctness / verification:** Executable verifiers plus LLM judges; bounded recovery.
- **Difficulty control:** Target-model failure profile per constraint type.
- **Reported results:** The registry grew from 76 hand-written types to 310 over 126 rounds. Of 15,668 accepted samples, 65.3% passed without recovery and 14.3% needed one attempt, at about 20 s per accepted sample. On Qwen3.5-4B, MM-IFEval average went from 60.8 to 63.9 (SFT-15k) and 64.9 (RL-15k); the 9B reference scored 63.0. General 7-benchmark average: 70.5 → 71.0 (SFT) and 72.9 (RL).
- **Limitations / failure modes:** Multimodal-specific; multi-agent cost.
- **How to reuse with easy seeds:** Keep a growing operator or constraint registry with per-type failure statistics. Sample in proportion to failure × under-coverage, select candidates by farthest-neighbor, and let the LLM propose new verifiable types when the registry saturates.

---

## Complexification operators from this area

Examples marked *(paper)* are quoted from the cited paper. The others are illustrative.

1. **Constraint addition / stacking.** Attach format, length, lexical, content, or style requirements, several at once. Difficulty grows with their count and how they interact.
   - Easy → hard: "Write a short poem about autumn." → "…exactly 4 stanzas of 3 lines; use 'ember' and 'ledger' exactly once each; no letter 'z'; each stanza ends with a question; title in <<double angle brackets>>."
   - Keep verifiable: one Python checker per constraint (AutoIF, IFBench). Filter conflicts (Conifer, SEIF Filter). Make the reward the conjunction or the mean (IFDecorator, RECAST) plus an intent or quality gate. Train with more constraints than the eval uses (5–6 in IFBench; 6–8 in Crab; about 13 in RECAST), with wide parameter ranges.
   - Sources: 2304.12244, 2406.13542, 2507.02833, 2508.04632, 2404.02823, 2505.19030.
2. **Constraint back-translation / answer-first inversion.** Read properties off a known-good answer and state them as requirements, so the reference stays correct by construction.
   - Easy → hard: "Explain photosynthesis." plus a 180-word, 3-paragraph answer → "Explain photosynthesis in 150–200 words across exactly 3 paragraphs; mention chlorophyll in the first paragraph and the Calvin cycle in the last."
   - Keep verifiable: script-fill values as ranges (Crab); keep LLM-proposed constraints only if the answer satisfies them (Crab, RECAST). For code, round-trip code → instruction → code and keep high-fidelity pairs (Phi-4). Beware copy-paste satisfiability (UNSPECIFIC).
   - Sources: 2410.24175, 2505.19030, 2412.08905, 2608.09154.
3. **Selective hardening.** Generate once without constraints and find which constraints are already satisfied trivially. Revise only those, or keep adding constraints only while pass rate > 0.5.
   - Easy → hard: "The story should have some emotional moments" → "Their relationship is tested by some difficult moments" *(paper, UNSPECIFIC)*.
   - Keep verifiable: re-run the checker after revision. Evaluate on a summary as well, to catch superficial satisfaction.
   - Sources: 2608.09154, 2508.04632.
4. **Logic-structured constraint composition.** Arrange atomic constraints into parallel, sequential, conditional, and nested trees whose conditions depend on the input.
   - Easy → hard: "Answer in English using bullet points." → "If the question concerns health, first give a one-sentence disclaimer, then answer in bullets; otherwise answer in one paragraph; in both cases end with a line starting 'Summary:'."
   - Keep verifiable: evaluate the condition first, score only the active branch, discount later steps after an early failure, and average parallel children (LsrIF).
   - Sources: 2601.06431.
5. **Deepening / knowledge-concept injection ("difficulty hiking").** Require more advanced or cross-domain knowledge, e.g. k knowledge tags, or a same-branch theorem plus a cross-domain concept.
   - Easy → hard: "A rectangle has perimeter 20 and width 4; find its area." → "Find the maximum area of an axis-parallel rectangle inscribed in x²/25 + y²/9 = 1, and the probability that a uniform random point in the ellipse lies inside it."
   - Keep verifiable: k ≥ 4 strong-solver agreement (SAND-Math used k=2), or a symbolic or CAS check. Rate difficulty before and after. Drop unsolvable items.
   - Sources: 2507.20527, 2505.24165, 2304.12244.
6. **Skill-preserving structural variation.** Change the problem's structure while keeping its core lemma: generalize-then-specialize, parametrize-and-sum, change the queried quantity, inverse problem, add a constraint layer.
   - Easy → hard: "7^2024 mod 100" → "(7^2024 + 3^2024) mod 100" *(paper, QbQ)*. "How many positive solutions does 20m + 12n = 2012 have?" → "Find the least c for which 20m + 12n = c has exactly 5 solutions" *(paper, QbQ)*.
   - Keep verifiable: the answer must differ from the parent's, with no number-only edits; verify the answer by an independent re-solve or a program. Seed only from problems the policy solves 8–15 times out of 16.
   - Sources: 2608.01522.
7. **Concretizing / situational and persona grounding.** Replace generic asks with specific entities, users, motivations, or documents.
   - Easy → hard: "Give tips for saving money." → "A night-shift ICU nurse in Lisbon nets €1,900/month, pays €750 rent, and has €4,000 of card debt at 18% APR: build a 12-month payoff plan that keeps a €1,000 buffer, with the monthly cash flow."
   - Keep verifiable: compute any numeric parts programmatically. Otherwise use per-constraint evaluation questions (UltraIF) or document-grounded rubric checks. Dedup personas and prompts (MinHash plus embeddings).
   - Sources: 2406.20094, 2506.03968, 2407.03502, 2304.12244.
8. **Sequential composition / fusion.** Chain problems so one output becomes the next input, or fuse two prompts into one multi-objective task.
   - Easy → hard: "(A) 3 boxes × 12 apples; how many? (B) 36 apples shared by 4 kids" → "Tom has 3 boxes of 12 apples, gives away a third, and shares the rest equally among as many kids as there are primes below 10. How many does each kid get?"
   - Keep verifiable: substitute A's verified answer into B and recompute. Reject ill-posed substitutions, such as a non-integer number of people. Provide an "INVALID" escape (Instruction Fusion).
   - Sources: 2503.16212, 2312.15692, 2410.02795.
9. **Skill crossover / composition.** Require two or more skills to be used *jointly*, not one after the other.
   - Easy → hard: "reverse a linked list" + "detect a cycle" → "Given a singly linked list that may contain a cycle, reverse the acyclic prefix in place, leave the cycle intact, and return the new head and the cycle entry, using O(1) extra space."
   - Keep verifiable: a reference solution plus randomized tests and a brute-force oracle. A skill audit rejects cases where one skill is merely appended (EvoTD). Judge-only fitness (Genetic-Instruct) is weaker.
   - Sources: 2605.11666, 2407.21077, 2408.14774.
10. **Parametric scaling (attribute mutation).** Keep the algorithm fixed and scale the input size, depth, range, or the required time or space complexity.
    - Easy → hard: "shortest path in a 5-node graph" → the same with 10⁵ nodes, negative edges, no negative cycles, under a time limit.
    - Keep verifiable: execute the reference implementation, and use timed hidden tests for efficiency. Keep items with 0 < pass < 1.
    - Sources: 2605.11666, 2306.08568.
11. **Complicate input / distractor, misdirection, unanswerability.** Add plausible but irrelevant information, erroneous reference code, trap options, or edit the passage so the question becomes unanswerable or the answer flips.
    - Easy → hard: "Add a distractor option that seems to strengthen the argument but does not relate to the causal relationship" *(paper, AgentInstruct)*. Code tasks can include "a piece of erroneous code as a reference" *(paper, WizardCoder)*.
    - Keep verifiable: check answer invariance under the distractor by re-solving and comparing. Confirm the injected code fails its tests. For deliberate answer flips or unanswerable edits, the new gold answer must be re-derived.
    - Sources: 2407.03502, 2306.08568, 2304.12244.
12. **Structured complexity budget.** Represent the prompt as a tree or tag set and add exactly k nodes or tags, in one shot or through a learned policy.
    - Easy → hard: tags {book_listing, author_info, web_develop} *(paper, TAG-INSTRUCT)* plus {pagination, accessibility, sort_by_rating} gives an accessible, paginated, sortable HTML book list.
    - Keep verifiable: tie each added node or tag to a checkable requirement, e.g. DOM or accessibility tests. Drop incoherent outputs with IC-IFD or a coherence judge.
    - Sources: 2308.05696, 2505.24165, 2505.18557, 2412.11231.
13. **Decompose-and-recompose.** Split real complex prompts into a base query plus components with evaluation questions, and train a composer to add them to easy queries. Alternatively, decompose prompts into Background/Objective/Constraints and add exactly one element per step (TaCIE).
    - Easy → hard: "recommend me ten Chinese books" → "In Shakespeare's tone, recommend me ten Chinese books" plus the evaluation question "Is it in Shakespeare's tone?" *(paper, UltraIF)*.
    - Keep verifiable: one evaluation question or checker per added component, used for rejection sampling and as a rubric reward.
    - Sources: 2502.04153, 2410.02795.
14. **Target-aware selection and hardening.** Measure difficulty on the model being trained and keep, harden, or regenerate accordingly. Signals include a pass-rate band (IFDecorator, EvoTD, QbQ, D2Evo), a teacher–student gap (Lion τ=1.0; CodecLM >3/10), plurality agreement (Phi-4), solver failure (SAND-Math), and per-type failure weighting (VISA).
    - Easy → hard: a prompt solved 8/8 times is hardened until pass@8 ∈ (0, 0.5]; at 0/8 it is sent back for regeneration rather than kept.
    - Keep verifiable: use N ≥ 8–16 samples. For hard or unsolved items, confirm solvability with a stronger model or a reference answer, so that "hard" does not really mean "broken".
    - Sources: 2508.04632, 2404.05875, 2305.12870, 2412.08905, 2605.17037, 2608.01522, 2608.26013.
15. **Downward evolution / difficulty ladder and hints.** Also produce easier siblings, or add hints, so that all-fail items still give signal.
    - Easy → hard: a competition problem at 0/16 becomes a ladder of 3–5 variants (simpler numbers or fewer conditions → full problem), or gets a natural-language hint (EvoTD for avg@k = 0).
    - Keep verifiable: a verified answer for each rung. Where rungs are related, their answers can be cross-checked.
    - Sources: 2308.09583, 2404.02823, 2605.11666.
16. **Meta-operator: learned or optimized evolvers.** Optimize the evolution procedure itself: prompt optimization from failure trajectories, MCTS over action sequences, DPO-trained tag expanders, a trained composer, or an RL-trained questioner or instructor rewarded for band-hitting (D2Evo) or for follower failure (SEIF).
    - Easy → hard: a fixed "add one more constraint" prompt is replaced by an instructor whose reward is 1 − follower satisfaction, gated by a validity filter.
    - Keep verifiable: put validity, verifier pass, and label agreement *inside* the reward or selection objective. Keep real anchors to avoid entropy collapse.
    - Sources: 2406.00770, 2410.10392, 2505.18557, 2502.04153, 2605.17037, 2605.07465.

---

## Insights & pitfalls

- **Overshoot is the default outcome.** Constraint addition made more instructions unsolvable (10,772) than well-banded (7,324) in IFDecorator. Plan for a regeneration path and never keep 0-pass items without a solvability certificate.
- **Evolver quality is task-dependent.** WizardLM found a LLaMA-2-70B-Chat evolver worse than ChatGPT. Hui et al. found 8B evolvers beat 70B on IFEval and GSM8K but not on HumanEval. TaCIE found GPT-4o intensified complexity in only 1 of 3 depth attempts. Measure evolver yield on your own seeds.
- **Rounds: 2–3, then stop on data.** WizardCoder peaked at 3 rounds. Instruction Fusion reports code evolution "reaches its capacity (3 rounds)". Hui et al. saw a third SLM round degrade performance. SEIF saturated after turn 3. UltraIF's third DPO round diverged. Genetic-Instruct plateaued after about 6M samples.
- **Verification dominates.** Answer-Consistency beat self-consistency (57.2 vs 55.1) and no filter (53.0) in CoT-Self-Instruct. Removing UltraIF's evaluation-question filter cost 3.35–5.36 points. Shirker responses cut Instruct-SkillMix LC by about 7.6 points. MathFusion's result that 5.6% bad problems had no effect on SFT should not be carried over to RLVR, where each wrong key is a wrong reward.
- **Generic evolution can reduce diversity.** EvoTD's Evol-Instruct baseline lowered pass@8 below the backbone. KITE reports "polarization of competence" in iterative synthetic instruction tuning. D2Evo reports entropy collapse for anchor-free questioners. Counter-measures: real anchors every round; farthest-neighbor or embedding-distance selection (VISA, Deita τ=0.9); forced structural diversity (QbQ's 3 of 5 operators per seed; Vendi 1.59 vs 1.25); coverage down-weighting (VISA); gradient-space diversity (Prismatic Synthesis, G-Vendi).
- **Constraint-only RL is a specialist trap.** The IF-only-RLVR Qwen2.5-7B base in IFBench scored AlpacaEval 1.1 and GSM8K 15.3. Mix general data, add an RM gate (IFBench), or add an intent check (IFDecorator). Constraint-level partial credit beat binary reward in SEIF. Structure-aware aggregation beat averaging in LsrIF.
- **Reward hacking of stacked constraints is systematic.** The patterns are copying placeholders, dummy list items, repetition to hit counts, copying delimiters (IFDecorator), and copy-paste from reference-derived constraints (UNSPECIFIC). Track a hidden trip-wire set during training.
- **Curriculum evidence is small and mixed.** In Tree-Instruct, hard-only beat easy→hard. Conifer's easy→hard multi-turn packaging beat shuffled or reversed orders by about 1–2 points. SEIF preferred front-loaded training on early rounds. What consistently helps is a *dense* difficulty ladder plus band selection, not a strict ordering.
- **Decontaminate evolved data.** Evol-CodeAlpaca overlapped 70.7% of HumanEval test items. Tulu 3 removed 3.5% of it by 8-gram matching and 11.3% of NuminaMath-TIR. SAND-Math combines retrieval with an LLM judge and a web-search novelty filter (τ=0.85). Infinity Instruct uses BGE similarity to benchmarks.
- **Benchmark-derived metadata leaks.** CodecLM's metadata came from 20% validation splits of its evaluation benchmarks, and IDEA-MCTS mined task-specific actions from AlpacaEval-related instructions. Keep operator discovery on data disjoint from your eval sets.
- **Measure the right thing.** Report pass@1 *and* pass@k, IF *and* general benchmarks, and per-difficulty-level scores. RECAST and FollowBench show that frontier models collapse at high constraint counts (Gemini-2.5-Pro averaged 39.75 on RECAST-Test), so aggregate IFEval can hide saturation.
- **The researcher JSON contained corrections worth remembering.** SEIF does *not* use a 0.4–0.6 band; its instructor reward is 1 − satisfaction. D2Evo's 0.4–0.6 band is for the questioner reward; anchors use 0.4–0.8. The system in 2601.22607 is AReaL-SEA; "Eigen" is an affiliation. UNSPECIFIC's 90% → 78% figure is for GPT-5 Mini.

---

## Open problems & research opportunities

- **Cheap difficulty prediction.** Predicting policy pass rate for a candidate prompt without 8–32 rollouts would make evolve-and-band loops much cheaper. Deita and IDEA-MCTS scorers predict LLM-judged complexity, not policy difficulty. IFDecorator notes that complexity is a poor proxy.
- **Hard vs broken at scale.** In the 0.4–0.6 band, majority-vote labels are least reliable, and an RL-trained questioner can learn to produce ambiguous items that land in the band through noise. Solvability certificates are rarely required: executable derivations, reference solutions checked by a stronger model, or answer-first construction.
- **Verifiable hardening for open-ended tasks.** Evaluation questions (UltraIF), rubrics (CodecLM), model-based constraints (RECAST), and intent judges (IFDecorator) are all LLM-judged and hackable. UNSPECIFIC shows that even rule-verifiable constraints can be satisfied superficially.
- **Operator-level credit assignment.** Few works measure which operator transfers to which capability. VISA's per-constraint-type failure weighting and QbQ's operator planning are early steps. A bandit over (operator × domain) driven by held-out deltas is open.
- **Multi-round stability.** Preventing polarization (KITE), entropy collapse (D2Evo on R-Zero), homogeneity collapse (EvoTD on Evol-Instruct), and late-round overfitting (SEIF, UltraIF) needs principled diversity-plus-difficulty objectives and stopping rules.
- **Hardening operators for agentic and multi-turn tasks** that keep per-instance executable checkers valid as tasks grow: more tools, longer horizons, distractor documents. AReaL-SEA (2601.22607; synthetic tool-use dialogues with executable per-instance checkers) and Skill Self-Play (2607.22529; proposer–solver–skill-controller co-evolution) are early examples.
- **Beyond the teacher ceiling.** Evolved tasks can exceed what the answerer can reliably solve. Answer-first construction (back-translation, programs), tool-verified solutions, and proof checking are needed to push past the teacher. QbQ and D2Evo still rely on GPT-5-class answer checks.
- **Constraint-family overfitting.** Models trained on IFEval-style families generalize poorly (IFBench). Automatically discovering natural, verifiable new constraint types (VISA grew from 76 to 310) and measuring out-of-distribution transfer remain open.
- **Standardized, open, trained evolvers.** Learned evolvers exist (TAG-INSTRUCT policy, UltraComposer, D2Evo Questioner, SEIF Instructor), but there is no general, open, verifier-aware "hardener" model and benchmark for comparing hardening operators on a fixed policy.

---

## References

1. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., Hajishirzi, H. (2022). *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. ACL 2023, arXiv:2212.10560. https://arxiv.org/abs/2212.10560
2. Honovich, O., Scialom, T., Levy, O., Schick, T. (2022). *Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor*. arXiv:2212.09689. https://arxiv.org/abs/2212.09689
3. Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., Jiang, D. (2023). *WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions*. ICLR 2024, arXiv:2304.12244. https://arxiv.org/abs/2304.12244
4. Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., Jiang, D. (2023). *WizardCoder: Empowering Code Large Language Models with Evol-Instruct*. ICLR 2024, arXiv:2306.08568. https://arxiv.org/abs/2306.08568
5. Luo, H., Sun, Q., Xu, C., Zhao, P., Lou, J., Tao, C., Geng, X., Lin, Q., Chen, S., Tang, Y., Zhang, D. (2023). *WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct*. ICLR 2025, arXiv:2308.09583. https://arxiv.org/abs/2308.09583
6. Zhao, Y., Yu, B., Hui, B., Yu, H., Li, M., Huang, F., Zhang, N. L., Li, Y. (2023). *Tree-Instruct: A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment*. LREC-COLING 2024, arXiv:2308.05696. https://arxiv.org/abs/2308.05696
7. Liu, W., Zeng, W., He, K., Jiang, Y., He, J. (2023). *What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning*. ICLR 2024, arXiv:2312.15685. https://arxiv.org/abs/2312.15685
8. Jiang, Y., Chan, C., Chen, M., Wang, W. (2023). *Lion: Adversarial Distillation of Proprietary Large Language Models*. EMNLP 2023, arXiv:2305.12870. https://arxiv.org/abs/2305.12870
9. Guo, W., Yang, J., Yang, K., Li, X., Rao, Z., Xu, Y., Niu, D. (2023). *Instruction Fusion: Advancing Prompt Evolution through Hybridization*. arXiv:2312.15692. https://arxiv.org/abs/2312.15692
10. Wang, Z., Li, C.-L., Perot, V., Le, L. T., Miao, J., Zhang, Z., Lee, C.-Y., Pfister, T. (2024). *CodecLM: Aligning Language Models with Tailored Synthetic Data*. Findings of NAACL 2024, arXiv:2404.05875. https://arxiv.org/abs/2404.05875
11. Sun, H., Liu, L., Li, J., Wang, F., Dong, B., Lin, R., Huang, R. (2024). *Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models*. arXiv:2404.02823. https://arxiv.org/abs/2404.02823
12. Zeng, W., Xu, C., Zhao, Y., Lou, J.-G., Chen, W. (2024). *Automatic Instruction Evolving for Large Language Models*. EMNLP 2024, arXiv:2406.00770. https://arxiv.org/abs/2406.00770
13. Dong, G., Lu, K., Li, C., Xia, T., Yu, B., Zhou, C., Zhou, J. (2024). *Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models*. ICLR 2025, arXiv:2406.13542. https://arxiv.org/abs/2406.13542
14. Ge, T., Chan, X., Wang, X., Yu, D., Mi, H., Yu, D. (2024). *Scaling Synthetic Data Creation with 1,000,000,000 Personas*. arXiv:2406.20094. https://arxiv.org/abs/2406.20094
15. Mitra, A., Del Corro, L., Zheng, G., Mahajan, S., Rouhana, D., Codas, A., et al. (2024). *AgentInstruct: Toward Generative Teaching with Agentic Flows*. arXiv:2407.03502. https://arxiv.org/abs/2407.03502
16. Majumdar, S., Noroozi, V., Samadi, M., Narenthiran, S., Ficek, A., Ahmad, W. U., Huang, J., Balam, J., Ginsburg, B. (2024). *Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models*. ACL 2025, arXiv:2407.21077. https://arxiv.org/abs/2407.21077
17. Kaur, S., Park, S., Goyal, A., Arora, S. (2024). *Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning*. ICLR 2025, arXiv:2408.14774. https://arxiv.org/abs/2408.14774
18. Li, C., Chen, Q., Li, Z., Tao, F., Li, Y., Chen, H., Yu, F., Zhang, Y. (2024). *Optimizing Instruction Synthesis: Effective Exploration of Evolutionary Space with Tree Search*. arXiv:2410.10392. https://arxiv.org/abs/2410.10392
19. Yang, J., Lu, S., Guo, W., Li, X., Yang, K., Xu, Y., Niu, D. (2024). *TaCIE: Enhancing Instruction Comprehension in Large Language Models through Task-Centred Instruction Evolution*. arXiv:2410.02795. https://arxiv.org/abs/2410.02795
20. Qi, Y., Peng, H., Wang, X., Xu, B., Hou, L., Li, J. (2024). *Constraint Back-translation Improves Complex Instruction Following of Large Language Models* (Crab). arXiv:2410.24175. https://arxiv.org/abs/2410.24175
21. Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., et al. (2024). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124. https://arxiv.org/abs/2411.15124
22. Hui, T., Zhao, L., Dong, G., Zhang, Y., Zhou, H., Su, S. (2024). *Smaller Language Models Are Better Instruction Evolvers*. arXiv:2412.11231. https://arxiv.org/abs/2412.11231
23. Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., et al. (2024). *Phi-4 Technical Report*. arXiv:2412.08905. https://arxiv.org/abs/2412.08905
24. An, K., Sheng, L., Cui, G., Si, S., Ding, N., Cheng, Y., Chang, B. (2025). *UltraIF: Advancing Instruction Following from the Wild*. EMNLP 2025, arXiv:2502.04153. https://arxiv.org/abs/2502.04153
25. Pei, Q., Wu, L., Pan, Z., Li, Y., Lin, H., Ming, C., Gao, X., He, C., Yan, R. (2025). *MathFusion: Enhancing Mathematical Problem-solving of LLM through Instruction Fusion*. ACL 2025, arXiv:2503.16212. https://arxiv.org/abs/2503.16212
26. Zhu, H., Ruan, Z., Su, J., He, X., Chen, Y., Zhang, W., Chen, G. (2025). *TAG-INSTRUCT: Controlled Instruction Complexity Enhancement through Structure-based Augmentation*. arXiv:2505.18557. https://arxiv.org/abs/2505.18557
27. Guo, Z., Liu, W., Xie, M., Xu, J., Huang, Z., Tian, M., et al. (2025). *RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data*. ICLR 2026, arXiv:2505.19030. https://arxiv.org/abs/2505.19030
28. Wang, Y., Zhou, S., Guo, C., Zhu, Q. (2025). *Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection*. Findings of ACL 2025, arXiv:2505.24165. https://arxiv.org/abs/2505.24165
29. Jung, J., Han, S., Lu, X., Hallinan, S., Acuna, D., Prabhumoye, S., Patwary, M., Shoeybi, M., Catanzaro, B., Choi, Y. (2025). *Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning*. arXiv:2505.20161. https://arxiv.org/abs/2505.20161
30. Zhu, C., Xu, B., Wang, X., Mao, Z. (2025). *From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding*. ACL 2025, arXiv:2506.03968. https://arxiv.org/abs/2506.03968
31. Li, J., Du, L., Zhao, H., Zhang, B.-w., Wang, L., Gao, B., Liu, G., Lin, Y. (2025). *Infinity Instruct: Scaling Instruction Selection and Synthesis to Enhance Language Models*. arXiv:2506.11116. https://arxiv.org/abs/2506.11116
32. Du, L., Zhao, H., Ju, Y., Wu, C., Pan, T. (2025). *Scaling Towards the Information Boundary of Instruction Sets: The Infinity Instruct Subject Technical Report*. arXiv:2507.06968. https://arxiv.org/abs/2507.06968
33. Pyatkin, V., Malik, S., Graf, V., Ivison, H., Huang, S., Dasigi, P., Lambert, N., Hajishirzi, H. (2025). *Generalizing Verifiable Instruction Following* (IFBench). NeurIPS 2025 Datasets & Benchmarks, arXiv:2507.02833. https://arxiv.org/abs/2507.02833
34. Manem, C., Brahma, P. P., Mishra, P., Liu, Z., Barsoum, E. (2025). *SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers*. NeurIPS 2025 MATH-AI Workshop, arXiv:2507.20527. https://arxiv.org/abs/2507.20527
35. Yu, P., Lanchantin, J., Wang, T., Yuan, W., Golovneva, O., Kulikov, I., Sukhbaatar, S., Weston, J., Xu, J. (2025). *CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks*. arXiv:2507.23751. https://arxiv.org/abs/2507.23751
36. Guo, X., Liang, T., Jian, T., Yang, X., Wu, L.-I., Li, C., Lu, Z., Guo, Q., Chen, K. (2025). *IFDecorator: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards*. arXiv:2508.04632. https://arxiv.org/abs/2508.04632
37. Ren, Q., He, Q., Chang, J., Zhang, G., Zhu, J., Chen, X., Shi, Z., Liang, J., Xiao, Y., Xia, H., Sun, Z., Yu, F. (2026). *LsrIF: Enhancing Logic-Structured Instruction Following of Large Language Models*. arXiv:2601.06431. https://arxiv.org/abs/2601.06431
38. Gao, J., Chen, J., He, C., Xu, S., Jin, D., Wu, Y. (2026). *From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents* (AReaL-SEA). arXiv:2601.22607. https://arxiv.org/abs/2601.22607
39. Ren, Q., He, Q., Zhu, J., Chen, X., Chang, J., Sun, Z., Xia, H., Yu, F., Liang, J., Xiao, Y. (2026). *SEIF: Self-Evolving Reinforcement Learning for Instruction Following*. arXiv:2605.07465. https://arxiv.org/abs/2605.07465
40. Ye, L., Yin, Y., Galarnyk, M., Heng, Y., Chava, S., Zhang, C. (2026). *Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling*. arXiv:2605.11666. https://arxiv.org/abs/2605.11666
41. Zhang, R., Li, R., Ma, Z., Qiu, W., Tao, C., Wang, Y., Chu, X. (2026). *D²Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning*. ICML 2026, arXiv:2605.17037. https://arxiv.org/abs/2605.17037
42. Luo, X., Huang, Y., Guo, K., He, P., Zou, C., Hua, T., Zhang, X. (2026). *Learning from Synthetic Data without Model Collapse in Iterative Instruction Tuning* (KITE). arXiv:2607.17043. https://arxiv.org/abs/2607.17043
43. Huang, S., Cheng, P., Liu, H., Chen, T., Liu, Y., Ni, J., et al. (2026). *Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills*. arXiv:2607.22529. https://arxiv.org/abs/2607.22529
44. Bao, L., Wang, J., Zhang, Y., Zheng, Y., Paturi, R. (2026). *Question Begets Question: Self-Evolving Curriculum for Reinforcement Fine-Tuning on Competition Mathematics*. arXiv:2608.01522. https://arxiv.org/abs/2608.01522
45. Sharma, J., Kaur, B., Hong, J., Zamani, H., Chang, H.-S. (2026). *UNSPECIFIC: General Constraint Synthesis for Breaking Copy-and-Paste Shortcut in LLM Instruction Following*. arXiv:2608.09154. https://arxiv.org/abs/2608.09154
46. Zeng, M., Tan, G., Cen, L., Wen, Y., Hu, R., Bian, L., Chen, X., Chen, X. (2026). *VISA: Agentic Self-Evolving Data Synthesis for Multimodal Instruction Following*. arXiv:2608.26013. https://arxiv.org/abs/2608.26013
