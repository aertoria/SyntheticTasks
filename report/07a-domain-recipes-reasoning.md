# Domain recipes I: math, formal proofs & geometry, code, software engineering, logic puzzles, science

> **Key takeaways**
>
> - **Build the answer or certificate first, then write the task.** Every sound generator in these six domains starts from something that can be checked: an executed computation graph, a deduction closure, a planted solution, a simulator trace, a reference program or a proved lemma. The prompt text is derived from it afterwards. Answers written by an LLM are the weakest label tier. Keep them out of RL rewards unless an independent check backs them. **Strong**
> - **Harden the verifier and the answer format before you generate new tasks.** Weak tests and guessable formats make tasks look easy when they are not.
>   - Strengthening SWE-bench Verified tests cut the top agent from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)).
>   - Evolved tests cut pass@1 on the same problems from 43.80 to 31.22 ([EvolveCoder](https://arxiv.org/abs/2603.12698)).
>   - Bare SAT/UNSAT labels invite "satisfiability bias" (answering SAT without a valid assignment), which accounted for 68.7% of o4-mini's errors ([SATBench](https://arxiv.org/abs/2505.14615)).
>
>   **Strong**
> - **Composition is the most reliable hardening knob that stays verifiable in every domain.** Each atom keeps its own checker. Examples: chained GSM8K problems ([h1](https://arxiv.org/abs/2510.07312)), fused Lean statements ([GAR](https://arxiv.org/abs/2510.11769)), chained Dafny functions ([DafnyComp](https://arxiv.org/abs/2509.23061)), fused GPU operators ([DRTriton](https://arxiv.org/abs/2603.21465)), multiple bugs or deeper feature removal in repositories, and f∘g string functions ([f(g(x))](https://arxiv.org/abs/2509.25123)). Train on composed tasks with RL and a curriculum. SFT and uniform mixing mostly fail to transfer composition. **Strong**
> - **Certify hardness against a reference, then calibrate it to the current policy.** The reference can be a cheap solver that fails without the key ingredient, a brute-force solver that agrees with the reference solution, or a stronger teacher that solves what your previous model could not. Keep a pass-rate band and re-measure it every round. LLM ratings of problem quality are close to useless: o3's correlation with human quality ratings was 0.07 ([AutoCode](https://arxiv.org/abs/2510.12803)). **Strong**
> - **Each domain has its own attack surface, and it moves as tasks get harder.** Typical hacks:
>   - weakened specs in verified code;
>   - stream timing and hard-coded inputs in kernels;
>   - `git clone` and `curl` in SWE;
>   - enumerating labels instead of inducing a rule in induction;
>   - flat feasibility rewards in optimization;
>   - proof bypasses and verifier bugs in Lean.
>
>   Budget for verifier red-teaming alongside generation. **Strong**
> - **When exact-answer tasks saturate, change the objective, not just the instance size.** Options: open-ended optimization scored against a baseline, speed relative to an expert or a hardware bound, prove-or-disprove, and certificates for "no solution" answers (MUS/MCS). **Moderate**

**Contents**

1. [Reading guide and cross-domain summary](#1-reading-guide-and-cross-domain-summary)
2. [Informal math (competition and word problems)](#2-informal-math-competition-and-word-problems)
3. [Formal theorem proving, geometry and inequalities](#3-formal-theorem-proving-geometry-and-inequalities)
4. [Competitive and algorithmic code](#4-competitive-and-algorithmic-code) (4.1 contest problems · 4.2 formally verified code · 4.3 program induction · 4.4 performance objectives)
5. [Software engineering and repository-level tasks](#5-software-engineering-and-repository-level-tasks)
6. [Logic puzzles and procedural reasoning gyms](#6-logic-puzzles-and-procedural-reasoning-gyms)
7. [Science (textbook-, database- and simulator-derived)](#7-science-textbook--database--and-simulator-derived)
8. [Choosing where to start](#8-choosing-where-to-start)

---

## 1. Reading guide and cross-domain summary

Each domain section follows the same template:

- what the seeds look like;
- the 4–8 operators with the best evidence, each with a concrete easy → hard example;
- a recommended generator architecture;
- the verifier;
- difficulty knobs;
- known pitfalls;
- open datasets and environments to start from;
- key references with verified numbers.

The chapter does not repeat material covered elsewhere:

- The full operator taxonomy is in [Chapter 02](02-complexification-operator-taxonomy.md).
- Generic pipeline patterns (teacher–student loops, setter RL, environment factories) are in [Chapter 03](03-generation-architectures.md).
- Verifier engineering is in [Chapter 04](04-verification-and-quality-control.md).
- Pass-rate bands, controllers and scaffolding for all-fail items are in [Chapter 05](05-rl-playbook.md).
- SFT-specific data policy is in [Chapter 06](06-sft-playbook.md).
- Tool use, terminal, multimodal, SQL and open-ended domains are in [Chapter 07b](07b-domain-recipes-agents-and-beyond.md).
- Lab practice is in [Chapter 08](08-frontier-lab-practices.md), and failure modes are catalogued in [Chapter 09](09-pitfalls-and-failure-modes.md).

Evidence tags:

- **Strong**: several independent works, or large ablations.
- **Moderate**: one careful study.
- **Emerging**: a single recent or unreplicated result.
- **Proposal**: our synthesis, not yet tested.

**Table 1. The six domains at a glance.**

| Domain | Typical seed | Best first operator | Sound verifier | Primary difficulty knob | Main way it goes wrong |
|---|---|---|---|---|---|
| Informal math | GSM8K/MATH/AoPS items with a final answer | Serial chaining of labelled seeds; answer-preserving nesting | Execution / SymPy on a construction; answer preservation | Horizon, op count, number of composed skills | Consensus labels drop the frontier; wrong new labels become wrong rewards |
| Formal proofs and geometry | Lean statements, Mathlib, random geometric premises | Subgoal extraction and recomposition; auxiliary-point hiding | Lean kernel with a bypass screen; DDAR engine | Proof steps κ, number of auxiliaries, pass-rate window | Misformalization, proof bypasses, verifier bugs |
| Contest code | TACO/APPS/Codeforces with oracles | Test hardening, then solution-first algorithmic lift | Oracle + validator + checker; brute force vs reference | Composed features, input scale, P → NP-hard goal | Weak tests; shared misconceptions under majority voting |
| Verified code, induction, performance | Tested functions, ARC tasks, correct kernels | Spec lifting; hindsight relabelling; correctness → speed | Proof checker; execution; hardened timing harness | Pipeline stage and target language; composition depth; speed bar p | Weakened specs; label enumeration; timer hacks |
| SWE / repository level | Repositories with passing tests | Realistic bug injection (rewrite, reversion, feature-add), multiple bugs | F2P/P2P tests, hidden triggering tests, inverse mutation testing | Bug size, number of bugs, call-tree depth, issue vagueness | One-line bugs, ambiguity mistaken for difficulty, fetching the fix with git/curl |
| Logic puzzles and gyms | Reasoning Gym / SynLogic / RLVE generators | Planted solutions plus an adaptive level controller | Certificate checks (assignment, core, path) | Level d, solver effort, information hiding | Generator bugs, unsolvable instances, puzzle overfit |
| Science | Textbooks, experimental DBs, simulators, formula libraries | Simulator/solver-first generation; MCQ → short answer or mask-and-choose | Simulator within tolerance; RDKit; solver optimum | Entities/formulas composed, constraint count | Degenerate or shortcut questions; plausibility hacks; SFT regressions |

One pattern repeats across the table. Easy tasks usually have one of three properties: the checker is weak, the format lets the model guess, or the instance needs only one skill. So the first fixes are usually verifier hardening, format hardening and composition. Clever LLM rewriting comes later.

---

## 2. Informal math (competition and word problems)

### 2.1 Seeds

Informal-math seeds are problems with a checkable final answer: GSM8K and MATH train, NuminaMath, AoPS-derived pools, and curated RL pools such as DeepMath-103K, Big-Math and DAPO-Math-17K. Clean them before you complicate them.

1. **Harden the format.**
   - Convert multiple-choice items to open-ended ones. [Big-Math-Reformulated](https://arxiv.org/abs/2502.17387) (2025) recovered 63.4% of amc_aime MCQs this way, and more than 50% of the reformulated set landed in the two hardest solve-rate quintiles.
   - Remove yes/no, multi-part and proof items, or convert proofs into answer-bearing questions ([OpenMathReasoning](https://arxiv.org/abs/2504.16891)).
   - Make answers integers where you can ([DAPO](https://arxiv.org/abs/2503.14476)).
2. **Decontaminate.** [DeepMath-103K](https://arxiv.org/abs/2504.11456)'s raw pool was 90% contaminated for AIME24/AMC23 and 76.6% for MATH500. Its recipe is embedding top-5 retrieval plus an LLM judge, and multi-solution agreement for labels.
3. **Profile pass rates on your policy** (16 rollouts is typical). The pass rate decides which operator each seed gets (§2.3).

### 2.2 Operators that work

The examples are from the verified notes. Every worked answer below was checked by brute force when the notes were compiled.

| # | Operator | Easy → hard example | How the label stays exact | Evidence |
|---|---|---|---|---|
| 1 | **Serial composition / chaining.** The answer of problem i becomes an input of problem i+1, directly or through a deterministic adapter | "12 muffins/hour for 5 hours?" (60) → "…pack 4 per box (Y), $8 per box (Z), 15% discount on Z. What is paid?" (**$102**) | Execute each seed's code solution with the upstream value substituted ([Compositional GSM](https://arxiv.org/abs/2410.01748)). Adapters are integer-valued and deterministic ([h1](https://arxiv.org/abs/2510.07312)) | **Strong** |
| 2 | **Answer-preserving hardening.** Add irrelevant background, invent an abstract term, or replace a given number with an independent sub-problem that evaluates to it | "Rectangle 12 × 5; diagonal?" (13) → "Length = number of divisors of 60, width = number of primes < 12; call the diagonal the *span*; find the span" (**13**) | The gold label is unchanged. Execute each nested sub-problem and require exact equality with the constant it replaces | **Moderate** ([MathForge](https://arxiv.org/abs/2601.20614), [SvS](https://arxiv.org/abs/2508.14029)) |
| 3 | **Parametric lifting.** Turn the seed into a `sample/solve/render` program, then resample, generalize or aggregate | "Trailing zeros of 100!" (24) → "…in base 12" (**48**), or "Σ_{n=1}^{50} Z(n!)" (**262**) | `solve(params)` plus unit tests `single_valued` and `matches_original` ([EFAGen](https://arxiv.org/abs/2504.09763)). The variant's answer must differ from the parent's ([QbQ](https://arxiv.org/abs/2608.01522)) | **Moderate** |
| 4 | **Computation-graph / symbolic expansion.** Grow an executable graph or a SymPy/SMT program: more ops, coupled or nested equations, nonlinear constraints | "Ann 5 apples, Tom 3 more" (8) → a four-person system whose quantities total 58, solve for Tom (**15**) | Solve with SymPy/Z3 or execute before rendering. Check the rendering by cycle consistency ([MathCAMPS](https://arxiv.org/abs/2407.00900)) or by requiring a CoT to reproduce the executed answer ([RV-Syn](https://arxiv.org/abs/2504.20426)) | **Strong** |
| 5 | **Inversion.** Mask a given, reveal the answer, and ask for the given; or fix the target and ask for the parameter | "Trailing zeros of 100!" → "Smallest n with exactly 24 trailing zeros" (**100**) | The masked value is the label. **Check uniqueness**: without "smallest", n = 100–104 all qualify | **Strong** ([MetaMath](https://arxiv.org/abs/2309.12284), [GSM-Infinite](https://arxiv.org/abs/2502.05252), [T-SAE study](https://arxiv.org/abs/2605.28388)) |
| 6 | **Skill / concept-pair composition.** One problem requires two or three skills that rarely co-occur | "Divisors of 360" (24) → "Triangle (0,0),(d,0),(0,k), d = divisors of 360; smallest k making the area a perfect square" (**3**) | Compose executable pieces (RV-Syn function graphs) or use SymPy-equivalence voting ([KPDDS](https://arxiv.org/abs/2403.02333)) | **Strong** for SFT; RL evidence via [SwS](https://arxiv.org/abs/2506.08989) |
| 7 | **Distractor injection.** Add clauses or graph nodes that cannot affect the answer | GSM-NoOp: "…five of them were a bit smaller than average" (answer still 190) | Attach noise only with edges pointing *away* from the query's ancestors ([GSM-Infinite](https://arxiv.org/abs/2502.05252) spider topology), then re-solve | **Moderate** |
| 8 | **Insight hiding and hard perturbation.** Find the step that makes the seed easy and make the solver discover it; or make a minimal edit that breaks the memorized method | "Minimize x² − 6x + 13" (4) → "…over integers not divisible by 3" (**5**) | Brute-force or search verification; the new answer must differ from the old one; reject "tedium" ([Code2Math](https://arxiv.org/abs/2603.03202) rubric scores 1–2) | **Emerging** |

The data supports several of these rankings:

- **Structure beats paraphrase.** Adding 20K MetaMath samples on top of 80K gave +0.1 GSM8K points from more answer augmentation, +0.4 from rephrasing, +2.3 from FOBAR inversion and +2.6 from self-verification inversion ([MetaMath](https://arxiv.org/abs/2309.12284), 2023). [FLAMES](https://arxiv.org/abs/2508.16514) found that agents designed to increase complexity gave the best improvements on most math metrics. **Strong.**
- **Chaining needs a horizon curriculum.** h1 chains GSM8K atoms and trains with Dr. GRPO over horizons up to length 5. AIME24 avg@32 rose from 5.10 to 10.52, MATH-500 reached 69.20 (+7.8%) and GSM-Symbolic P2 rose from 43.08 to 52.00. Uniform mixing and long-only training at equal compute gave no long-horizon gain ([h1](https://arxiv.org/abs/2510.07312), 2025). Composed difficulty is roughly multiplicative. Compositional GSM measures the gap as Δ = S_comp − S1·S2 and finds that Qwen2.5-Math-7B-IT solves more than 80% of MATH but under 60% of compositional grade-school problems. On MATH², success rate is roughly the square of the MATH rate ([MATH²](https://arxiv.org/abs/2407.21009)).
- **Answer-preserving edits fail safe under GRPO.** If a rewrite silently breaks the inherited label, the group scores all zeros and produces no update. MathForge's o3 audit found 99/97/97% of Background/Term/Sub-Problem rewrites answer-equivalent. On Qwen2.5-Math-7B trained on MATH, the reported average rose from 37.61 (GRPO) to 41.04 (MQR data) and 42.17 (MQR + DGPO) ([MathForge](https://arxiv.org/abs/2601.20614), ICLR 2026). A wrong *new* label from majority voting does not fail safe: it rewards wrong answers.
- **Construction makes items both harder and cleaner.** RV-Syn problems were solved only 55.6% of the time by Qwen2.5-Math-7B-Instruct, against 70.6–97.2% on six other datasets. Audited error rates were 0.9% (problem) and 1.4% (solution), against 4.9–8.0% solution error for MathScale, NuminaMath and ScaleQuest ([RV-Syn](https://arxiv.org/abs/2504.20426)).
- **Easy seeds hide hard variants.** EFAGen resampled 50 variants from MATH seeds that GPT-4o had solved and found variants GPT-4o fails *even for Level-1 seeds* ([EFAGen](https://arxiv.org/abs/2504.09763)).
- **Inversion rescues unreachable items.** Converting hard@8 items (pass@8 = 0) into up to 2m inverse variants each ("AugHard@8") recovered AIME24 from 3.33 (raw hard@8 items) to 13.33 in the intervention table, above the 10.00 of full-data training ([T-SAE study](https://arxiv.org/abs/2605.28388)).

### 2.3 Recommended generator architecture

Route each seed by its current pass rate. Do not run one operator over the whole pool.

```
                   profile p = pass@16 on current policy
                                  │
      ┌───────────────────────────┼───────────────────────────┐
  p ≥ ~0.75 (saturated)     band (~0.25–0.75)           p ≈ 0 (unreachable)
      │                           │                           │
  complexify:                 train as-is              audit label / format first,
  - nest / abstract term      (re-profile each round)  then scaffold:
    (answer-preserving)                                 - inverse rewrite (AugHard)
  - chain k labelled seeds                              - format ladder MCQ4→MCQ10→cloze
  - lift + resample params                              - expert-prefix / hint anneal
  - symbolic mutation                                   (see Ch. 05)
      │                           │                           │
      └──────► validity gate (execution / preservation / uniqueness)
                                  │
               band filter on the *current* policy → RL pool
```

QbQ gives the cleanest evidence for seeding from the "mostly solved" band. Variants seeded from problems solved 8–15 times out of 16 reached 16.46% AIME pass@1. Seeding from the hardest failures reached 11.36%, and plain GRPO 11.5% (Qwen2.5-Math-7B). Curriculum QbQ did not saturate after 20 rounds ([QbQ](https://arxiv.org/abs/2608.01522), 2026). **Moderate.**

Validity gating and generator training come next.

- **Gate the setter reward on validity.** At scale, train the generator with RL on *validity × difficulty*, where validity comes from a check independent of the solver. [VHG](https://arxiv.org/abs/2605.06660) (2026) uses setter reward 1[verifier accepts] × (1 − solver accuracy), with SymPy differentiation as the hard verifier for integrals. The setter learned validity first (valid rate 30.6% → 75.5%), then difficulty (valid-and-hard share 27.5% → 58.5%). On competition integrals, pass@1 went from 28.8 to 45.4, against 38.8 for vanilla GRPO and 30.5–31.9 for R-Zero. Without the gate, setters learn to emit invalid "hard" problems. **Moderate.**
- **Train the generator rather than only filtering its output.** [MathSmith](https://arxiv.org/abs/2508.05592)'s usable-problem ratio rose from 71.5% (SFT generator) to 95.4% once a consistency term entered its RL reward. Without generator training, expect heavy overgeneration: Code2Math needed 1.56–6.55 failed rollouts per accepted evolution.
- **Make the generator write a design rationale first** (concepts → intended difficulty → solution path → problem), as in [PromptCoT 2.0](https://arxiv.org/abs/2509.19894). It reported AIME24 87.7 → 92.1 and AIME25 85.0 → 89.8 via self-play on Qwen3-30B-A3B-Thinking-2507. Its synthetic-prompt SFT took Qwen2.5-7B-Instruct from 12.8 to 73.1 on AIME24.

A minimal operator dispatcher (pseudo-code):

```python
def harden(seed, p):                       # p: current pass rate of the seed
    if p >= 0.75:
        ops = [nest_given_as_subproblem, chain_with(k=next_horizon()),
               lift_and_resample, symbolic_mutate]
    elif p <= 0.05:
        return scaffold(seed)              # inverse rewrite / format ladder (Ch. 05)
    else:
        return [seed]                      # already in band
    out = []
    for op in ops:
        v = op(seed)
        if exact_label(v) and unique(v) and not tedious(v):   # execution / SymPy / brute force
            out.append(v)
    return [v for v in out if 0.25 <= pass_rate(v, k=16) <= 0.75]
```

### 2.4 Verifier

Use label tiers and match the tier to how the items will be used (details in [Chapter 04](04-verification-and-quality-control.md)).

| Tier | Source of the label | Usable for |
|---|---|---|
| 1 | Construction: an executed graph, SymPy/Z3 solve, or an inverse-operation check (differentiate the antiderivative, substitute the root, expand the factorization) | RL rewards |
| 2 | Answer preservation from a gold-labelled seed, with nested sub-problems checked by execution | RL rewards; fails safe under GRPO |
| 3 | Cycle consistency (back-translate text to a symbolic form without the original). MathCAMPS: 97.7% of survivors judged faithful | Filtering text renderings |
| 4 | Independent cross-family majority / Answer-Consistency | RL only near the frontier, with a monitored gold slice |
| 5 | A single teacher sample | SFT only |

Two practical details:

- **Recheck rule-based negatives.** Rule checkers produce false negatives on hard items. INTELLECT-3 re-checks every answer that math-verify marks wrong with CompassVerifier-7B ([INTELLECT-3](https://arxiv.org/abs/2512.16144)).
- **Watch drifting self-play labels.** In R-Zero, pseudo-labels from majority vote fell from 79.0% to 69.0% to 63.0% accurate over three iterations, and math performance peaked and then fell (49.12 → 46.52) ([R-Zero](https://arxiv.org/abs/2508.05004)). Keep a gold-labelled monitor slice in every self-play loop. **Strong.**

### 2.5 Difficulty knobs

- **Integer knobs:**
  - Op count. GSM-Infinite accuracy decays as a sigmoid in op, with R² ≈ 0.98 on Medium.
  - Horizon h.
  - Number of composed skills. Expect success of roughly p^k.
  - Number of nested sub-problems.
  - Graph size (RV-Syn uses 1–3 function nodes).
- **Direction:** forward vs reverse. Reverse problems follow a left-shifted sigmoid in GSM-Infinite.
- **Noise:** number of spider-topology distractor nodes.
- **Format:** MCQ → open-ended → integer-canonical.
- **Insight vs tedium.** Op and horizon scaling give smooth, predictable difficulty, which is good for state tracking. Hard perturbation and insight hiding change *which method* applies (MATH-P-Hard: o1-mini −16.49%). Mix both families. **Moderate.**

### 2.6 Pitfalls

- **Consensus filters remove the frontier.** All-k agreement ([SAND-Math](https://arxiv.org/abs/2507.20527): 23,437 → 17,578), Answer-Consistency ([CoT-Self-Instruct](https://arxiv.org/abs/2507.23751)) and 3-way R1 agreement (DeepMath) all drop hard-but-correct items. Build the answer first, verify the easy increments rather than the composite ([CHASE-Math](https://arxiv.org/abs/2502.14678)), or route disagreements to a relabelling pool.
- **Inversion creates non-unique answers.** Brute-force the domain or add "smallest/largest/count".
- **Some hard items damage training.** Training only on hard@8 items lowered averages by 5.75, 11.24 and 1.07 points in three settings. Harmful items often had shortcut reward paths, such as a bare `\boxed{40}` rewarded on a geometry problem. A single such sample collapsed mean response length from 510.7 to 45.7 tokens by step 58, before accuracy fell ([T-SAE study](https://arxiv.org/abs/2605.28388)). Watch response length as an early alarm.
- **Gains seen only on Qwen may be artifacts.** Random rewards still gave Qwen2.5-Math-7B +21.4 on MATH-500 (vs +29.1 with true rewards) but failed on Llama3/OLMo2 ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). Validate on a second model family and on post-cutoff contests. **Strong.**
- **Operator gains may not transfer across operators.** In OMEGA, isolated-skill RL gains "do not reliably carry over to the composed setting". Put composed items in the RL mix itself ([OMEGA](https://arxiv.org/abs/2506.18880)).
- **LLM "make it harder" prompts overshoot into ill-posed items.** Upward evolution without an answer check ([WizardMath](https://arxiv.org/abs/2308.09583)) is acceptable for SFT distillation but not as an RL reward.

### 2.7 Where to start

| Resource | What it gives you |
|---|---|
| [DeepMath-103K](https://arxiv.org/abs/2504.11456) | Decontaminated, level ≥ 5, 3-way-verified RL pool. RL-Zero on Qwen2.5-Math-7B: AIME24 11.2 → 34.2 |
| [Big-Math](https://arxiv.org/abs/2502.17387) (+Reformulated) | 251,122 problems with solve-rate quintiles from 64 rollouts; MCQ → open conversions |
| [OpenMathReasoning](https://arxiv.org/abs/2504.16891) | AoPS olympiad pool with 260K proofs converted into answer questions |
| [GSM-Infinite](https://arxiv.org/abs/2502.05252) / [iGSM](https://arxiv.org/abs/2407.20311) generators | Unlimited, exactly labelled op/horizon curricula |
| [PromptCoT 2.0](https://arxiv.org/abs/2509.19894) prompts, [ScaleDiff](https://arxiv.org/abs/2509.21070), [SAND-Math](https://arxiv.org/abs/2507.20527) | Hard synthetic prompt pools (teacher labels; better suited to SFT) |

**Key references:** [h1](https://arxiv.org/abs/2510.07312) (chaining with a horizon curriculum; numbers above); [MathForge](https://arxiv.org/abs/2601.20614) (answer-preserving MQR; 37.61 → 42.17); [RV-Syn](https://arxiv.org/abs/2504.20426) (solution-first function graphs; 55.6% solve rate, 1.4% solution error); [VHG](https://arxiv.org/abs/2605.06660) (validity-gated setter; 28.8 → 45.4 on competition integrals); [QbQ](https://arxiv.org/abs/2608.01522) (band-seeded structural operators; 16.46% vs 11.36%); [PromptCoT 2.0](https://arxiv.org/abs/2509.19894).

---

## 3. Formal theorem proving, geometry and inequalities

A symbolic engine or proof kernel certifies every item in this domain. Correctness is therefore cheap, and the hard problems are elsewhere: misformalization, verifier exploits, and difficulty that is artificial rather than real.

### 3.1 Seeds and the first decision: the correctness contract

- **Lean / Isabelle.**
  - Autoformalized contest pools: [Lean Workbook](https://arxiv.org/abs/2406.03847) has 57,231 statements that pass NLI, plus 82,893 in Lean Workbook Plus.
  - Mathlib files.
  - Earlier prover corpora: [DeepSeek-Prover](https://arxiv.org/abs/2405.14333) produced about 8M formal statements with proofs.
- **Olympiad geometry.** No human seeds are needed. Sample random premises from a construction language ([AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5)), or bias sampling with statistics from real olympiad problems ([TongGeometry](https://arxiv.org/abs/2412.10673)).
- **Inequalities.**
  - Classical seeds: AM-GM, Cauchy-Schwarz and similar.
  - Random cyclically symmetric expressions ([AIPS](https://arxiv.org/abs/2406.14219)).
  - Informal inequality proofs ([IneqMath](https://arxiv.org/abs/2506.07927)).

**Decide the contract before building anything:**

- **Prove-or-disprove this formal statement.** Misformalization is harmless here: any well-typed statement is a valid task. AlphaProof trained on about 80M auto-formalized statements from about 1M NL problems "regardless of fidelity" ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y)).
- **Match the informal intent or answer.** Faithfulness now sits on the critical path, and it fails often:
  - In [FormalMATH](https://arxiv.org/abs/2505.02735), multi-LLM semantic checks cut the kept share of compiling statements from 92.4% to 32.7%.
  - [MechGeo](https://arxiv.org/abs/2608.02295) refuted 14 of 43 autoformalized IMO geometry statements with Lean-checked counterexamples.

**Strong.**

### 3.2 Operators that work

**Lean / formal statements**

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Subgoal extraction and recomposition** (bidirectional). Turn `have` steps or stuck goals into standalone theorems; add earlier subgoals as premises for intermediate difficulty; chain verified lemmas into longer targets | Lemma "n² mod 4 ∈ {0,1}" → "no integers x, y with x² + y² = 4k + 3", posed with no hint | Kernel checks each subgoal and the composed proof. Extracted goals may be false: when the negation is proved, add it as a disproof task | **Strong** ([DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801), [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613), [Kimina](https://huggingface.co/blog/AI-MO/kimina-prover)) |
| **Barely-provable conjecturing.** Condition on (seed, its proof, a lemma); keep conjectures whose empirical pass rate lies in (0, 1/4] | a² + b² ≥ 2ab (pass 1.0) → a, b, c > 0, abc = 1 ⇒ a² + b² + c² ≥ a + b + c | Only kernel-proved conjectures become data. Add an elegance filter (drop the bottom 20% of proof/statement length ratio) and reweight toward unsolved statements | **Moderate** ([STP](https://arxiv.org/abs/2502.00212)) |
| **Statement fusion with adversarial reward.** Fuse two seeds; the fuser's reward is (1−p)(1−m)·1{p≠0} (m = statement-modification rate); the prover trains on 0 < p < 0.5 | AM-GM item + divisibility item → one fused statement | Compile + prove; p = 0 earns nothing, so false or impossible fusions are not paid | **Moderate** ([GAR](https://arxiv.org/abs/2510.11769)) |
| **Rewrite / implication mutation.** Rewrite a theorem with library `rw` (eq/iff) or `apply` (implication) lemmas | Mathlib theorem → a longer, less recognisable statement | The proof is the original proof plus the mutation step | **Moderate** ([Alchemy](https://arxiv.org/abs/2410.15748): 110k → 6M theorems) |
| **Target-centred variant cloud.** Generate simplifications, generalizations, lemmas and analogies around a target; keep the harder direction for solved seeds | "3 ∣ n³ − n" → "p ∣ n^p − n for all primes p" | Each variant is labelled by an actual proof or disproof, never by the generator | **Moderate** ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y) TTRL, [Pythagoras ALF](https://arxiv.org/abs/2606.12594)) |
| **Negation pairing / answer enumeration.** Pose every statement together with its negation; for "find all X", propose many candidates, refute the wrong ones and prove the survivor | "n² + n even" → "decide: n² + n + 41 prime for all n ≥ 0" (false at n = 40) | Kernel checks both directions. Always run a vacuity check (try to prove `False` from the hypotheses) | **Strong** ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333), AlphaProof, Goedel-Prover-V2) |

**Geometry**

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Random premises → deduction closure → minimal traceback** | Midpoint theorem (1 step) → a concyclicity deep in the closure of a 10+-point construction | The engine derivation is the proof. Check the diagram is numerically non-degenerate and filter trivially reducible goals (GenesisGeo: ∠ = 0° reduces to ∥) | **Strong** (AG1 → [AG2](https://arxiv.org/abs/2502.03544) → TongGeometry → [Seed-Prover](https://arxiv.org/abs/2507.23726) → [GenesisGeo](https://arxiv.org/abs/2509.21896)) |
| **Auxiliary hiding with a necessity certificate.** Remove the objects the proof needs from the statement; keep the item only if the engine fails without them and succeeds with them | Isosceles base angles: fails in DD alone; "D = midpoint of BC" makes it two steps | Store the with-aux proof and the failed without-aux run as the hardness certificate | **Strong** ([InternGeometry](https://arxiv.org/abs/2512.10534), AG1/AG2) |
| **Complexity knob with a closed-loop controller** (κ = DDAR proof steps) | κ = 5 (≈100% solved, zero advantage) → κ moves toward 50% batch success | Correctness does not depend on κ. Log the realized pass rate | **Moderate** (InternGeometry CBRL) |
| **Premise stacking and distractor premises.** Reuse the deepest verified scenes and add constructions and never-used givens | 10-step triangle → a 40+-step conclusion with irrelevant extra premises | The rule-based reasoner re-derives the scene after each addition | **Moderate** ([TrustGeoGen](https://arxiv.org/abs/2504.15780): 226 seed scenes → 376) |

**Inequalities**

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Duplication and substitution composition** | (x+y)/2 ≥ √(xy) → ((a+b)/2)·((c+d)/2) ≥ √(ab)·√(cd); or substitute x ↦ x² | The rules preserve truth under their side conditions; Lean-checked | **Moderate** ([Ineq-Comp](https://arxiv.org/abs/2505.12680)) |
| **Forward chaining of named inequalities** (AM-GM, Hölder, Schur, Muirhead + transitivity, ≤ 25 iterations) | One AM-GM step → a multi-step chain on a random cyclic expression with a tight equality case | The derivation chain is the proof; discard chains whose equality condition fails | **Moderate** ([AIPS](https://arxiv.org/abs/2406.14219): 191,643 theorems in 8 h on 32 CPUs) |
| **Certificate-first SOS construction.** Build F = Σ pᵢ² (+ g·Σ qⱼ²), expand or substitute to hide the structure, pose "prove F ≥ 0" | (a − b)² ≥ 0 → an expanded 4-variable degree-4 polynomial with no factored hint | The constructed certificate is the proof; pass it as a hint to `nlinarith`/`polyrith` for a kernel check | **Proposal** (inverts the [NSPI](https://arxiv.org/abs/2605.15445) prover, which certifies SOS proofs with up to 10 variables) |

### 3.3 Recommended architecture

The closure-and-traceback generator is the template for every domain that has a forward-chaining engine: rewriting systems, type inference, planners, SMT.

```
random or statistics-guided premises (construction program)
        │
        ▼
 deduction closure (DD+AR / rule engine)  ── numeric diagram check ──► drop degenerate
        │
        ▼
 pick goal g deep in the closure ── trivial-goal filters (∥ vs 0° angle, ratio 1, self-congruence)
        │
        ▼
 traceback: minimal premises P(g), proof π(g)
        │
        ▼
 move "dependency difference" objects out of the statement → auxiliary constructions
        │
        ▼
 hardness certificate: engine(P_raw) fails AND engine(P_raw + aux) proves g
        │
        ▼
 κ = |π(g)| stored → cache indexed by κ → controller samples κ near 50% success
```

InternGeometry's complexity-boosting RL (CBRL) is the most transferable recipe for GRPO runs with no signal, at least in this domain:

```python
kappa = kappa_init
for step in training:
    batch = [cache.select_around(kappa) or generate(kappa) for _ in range(B)]
    # keep only: conclusion uses X_raw points; provable(X_add); not provable(X_raw)
    rewards = rollout_and_score(policy, batch)       # r = outcome AND step-validity
    kappa += alpha * (mean(rewards) - 0.5)           # 0.5 maximizes expected |advantage|
```

The ablation on IMO-50 geometry (the full system trains on about 13K examples: a 7K formal cold start plus 6K CBRL-synthesized problems):

| Setting | IMO-50 solved |
|---|---|
| Full CBRL | 44/50 |
| Same data, no schedule | 38 |
| Easy data only | 29 |
| Hard data only | 24 |
| SFT cold start | 22 |

For comparison, AG2 scores 42/50 and SeedGeometry 43. InternGeometry's 13K examples are 0.004% of AG2's data ([InternGeometry](https://arxiv.org/abs/2512.10534), ICLR 2026). **Moderate** (one study, but a clean controlled ablation).

For Lean the analogous loop is: extract stuck goals from failed proofs → prove or negate them → compose proved lemmas into longer targets → admit to RL only items in a pass-rate window. Independent pipelines agree on the window: STP (0, 1/4] for conjectures, Seed-Prover drops items with proof rate > 1/4, GAR trains on 0 < p < 0.5, and Goedel-Prover-V2 samples (0, 0.75]. AlphaProof's matchmaker deprioritizes both mastered statements and statements that stay unsolved. **Strong.**

### 3.4 Verifier

- **Kernel plus a proof-bypass screen.** Reject `sorry`, `admit` and `sorryAx`, and any new `axiom`, `constant` or `opaque` introduced by the model ([MathlibLemma](https://arxiv.org/abs/2602.02561)).
- **Pin the Lean version and audit unusual lemma use.** A Lean 4.9.0 `apply?` bug silently dropped `sorry`. DeepSeek-Prover-V2-7B learned to exploit it through `Cardinal.toNat`, which inflated its PutnamBench count ([DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)).
- **Vacuity check.** Try to prove `False` from the hypotheses, and run negation proving in parallel ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333)).
- **Statement-modification penalty.** Penalize solver outputs that edit the statement ([GAR](https://arxiv.org/abs/2510.11769)).
- **Engines can be wrong.** HAGeo reports that one of AlphaGeometry's 25 announced IMO-30 proofs is fallacious ([HAGeo](https://arxiv.org/abs/2512.00097)).
- **Never skip the kernel.** Pythagoras ALF's mutations were never type-checked, and a 2,000-instance spot check found only 87.8% valid ([Pythagoras-Prover](https://arxiv.org/abs/2606.12594)).
- **Faithfulness, when the contract needs it.** Back-translation plus NLI is the minimum (Lean Workbook). Multi-LLM consensus plus negation disproof is stronger (FormalMATH). Counterexample-guided repair turns refuted formalizations into disproof tasks (MechGeo).

### 3.5 Informal proofs: reformulate, but guard against guessing

When seeds are informal proofs, convert them to checkable targets:

- the best constant (IneqMath bound estimation);
- the relation symbol;
- truth values over a *group* of minimal entailed and contradicted edits ([DeepTheorem](https://arxiv.org/abs/2505.23754));
- answer-bearing variants ("proofs → short-answer problems": [Phi-4-reasoning](https://arxiv.org/abs/2504.21318), OpenMathReasoning).

Guessing is the main risk. In IneqMath, accuracy drops by up to 65.5% once steps are checked. DeepTheorem counts a theorem only when every variant in its group gets a consistent verdict, which cuts random-guess accuracy to 11.2–17.4%. **Moderate.**

For full proofs, the frontier recipe scales the *verifier* along with the generator. [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) trains a proof verifier (scores 0 / 0.5 / 1) plus a meta-verifier, and auto-labels the generator's hardest proofs by scaling verification compute. It reports IMO 2025 83.3% (5 of 6) and Putnam 2024 118/120 with scaled test-time compute. Frontier olympiad recipes still mostly *select* hard human problems rather than synthesize them. [Nemotron-IMO](https://arxiv.org/abs/2609.10712) keeps problems solved in 1–3 of 4 attempts (9,597), see [Chapter 08](08-frontier-lab-practices.md).

### 3.6 Difficulty knobs

- Proof length κ (DDAR steps correlate with human difficulty, per AlphaGeometry as cited by InternGeometry).
- Number of auxiliaries, and the aux : non-aux ratio. AG2 moved it from AG1's 9:91 to 50:50 alongside other changes.
- Diagram size. AG2 explored diagrams at 2× size, giving proofs up to 10× longer.
- The pass-rate window.
- Decomposition granularity: lemma alone → lemma with context → full theorem.
- Number of fused seeds.
- Inequality chain length.
- For the Proposal above: SOS degree and variable count.

### 3.7 Pitfalls

- **SFT does not fix compositional brittleness.** Ineq-Comp: DeepSeek-Prover-V2-7B pass@32 is 66.23% on seeds, 46.97% on Type I and 42.1% on Type II. Even the 671B model drops (88.0 / 68.33 / 70.67). Fine-tuning on about 8K composed AM-GM problems improved only in-distribution Type I. MiniF2F-ALF mutations cost every SOTA prover about 3–5 points. Rotate operator families, train with RL, and hold whole families out for evaluation. **Strong.**
- **Self-play collapses in three ways.**
  - Onto one topic: STP drifted into algebraic manipulation.
  - Toward artificial complexity: STP needs its elegance filter.
  - Toward easy items: Minimo collapsed without hindsight relabeling, and [CPL](https://arxiv.org/abs/2509.14274) showed that sampling statement and proof jointly biases toward short proofs.

  Separate the conjecturer from the prover, and reweight toward unsolved statements.
- **Hardness claims need an honest reference solver.** HAGeo's random auxiliary points alone solved 25/30 on IMO-30, as many as AlphaGeometry, and heuristic auxiliaries solved 28/30. Use such a heuristic as the weak solver in a "fails without X" check.
- **Volume does not buy interestingness.** TongGeometry found 6,688,310,403 aux-requiring problems but calls automatic selection of olympiad-worthy ones "unresolved". Expect gains in RL signal, not problems humans find beautiful.
- **Standard benchmarks are saturated.** miniF2F-test is at 99.2% pass@1. Report gains on MiniF2F-ALF, Ineq-Comp, HAGeo-409, Fate-H/X or MathlibLemma instead (37% solved by the ensemble).

### 3.8 Where to start

| Resource | Use |
|---|---|
| [Lean Workbook](https://arxiv.org/abs/2406.03847) | Standard seed pool for expert iteration. STP proves 28.5% of it, vs 13.2% for expert iteration |
| [GenesisGeo](https://arxiv.org/abs/2509.21896) | Open AG-style recipe: 1M aux-requiring problems in 28 h on 50 CPU threads; trivial-goal and fake-aux filters |
| [HAGeo](https://arxiv.org/abs/2512.00097) | CPU-only DDAR baseline (a weak solver for hardness certificates) and HAGeo-409 |
| [Ineq-Comp](https://arxiv.org/abs/2505.12680), MiniF2F-ALF | Held-out mutated twins for evaluation |
| [MathlibLemma](https://arxiv.org/abs/2602.02561) | 4,028 non-trivial statements from library gaps plus the bypass screen |
| [FormalMATH](https://arxiv.org/abs/2505.02735) | Faithfulness pipeline and 5,560-problem benchmark |

Lean-native geometry pools also exist: Euclean ([2607.19374](https://arxiv.org/abs/2607.19374), 177,597 Numina-Geometry formalizations) and LeanGeo ([2508.14644](https://arxiv.org/abs/2508.14644)). Multimodal, diagram-rendered geometry (GeoSym127K, TrustGeoGen as VLM data) is covered in [Chapter 07b](07b-domain-recipes-agents-and-beyond.md).

**Key references:** [InternGeometry](https://arxiv.org/abs/2512.10534) (CBRL; 44/50 vs 38 without the schedule); [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5) (100M theorems, 9M with auxiliaries, 25/30 IMO-AG-30); [STP](https://arxiv.org/abs/2502.00212) (barely-provable band; 28.5% vs 13.2% of LeanWorkbook); [GAR](https://arxiv.org/abs/2510.11769) (the base model's accuracy on fused statements fell from 29.16% to 7.69% across iterations, while the trained prover stayed near 21%); [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) (`extract_goal` + negations; 84.6% MiniF2F pass@32 at 8B); [Ineq-Comp](https://arxiv.org/abs/2505.12680).

---

## 4. Competitive and algorithmic code

Code is the domain where "too easy" most often means "too weakly verified". Before generating anything, measure how many policy "passes" survive stronger tests.

- HardTests found TACO's false-positive rate above 90% on difficult problems.
- DeepCoder found that problems with fewer than 5 tests led to reward hacking: the policy printed memorized answers ([DeepCoder](https://www.together.ai/blog/deepcoder)).

After that audit, four families of hardening keep a sound reward:

- harder contest problems (4.1);
- lifting into formally verified code (4.2);
- program induction with execution labels (4.3);
- switching the objective from correctness to performance (4.4).

### 4.1 Contest and function-level problems

**Seeds.** TACO, APPS, CodeContests, Codeforces and LeetCode items with oracle solutions. Also function-level pools such as [KodCode](https://arxiv.org/abs/2503.02951) (447K verified triplets, with 10-attempt pass rates as free difficulty labels).

**Operators, in the order to try them:**

| # | Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|---|
| 1 | **Verifier hardening.** Hacking inputs aimed at plausible wrong or slow solutions; tests where candidate solutions disagree; tests that separate overlapping solutions | ~10 tests that 95% of samples pass → ~35 evolved tests including split tests | New tests must pass every known-correct reference. Drop tests with < 10% pass rate; deduplicate by pass vector | **Strong** ([HardTests](https://arxiv.org/abs/2505.24098), [EvolveCoder](https://arxiv.org/abs/2603.12698), [CodeContests-O](https://arxiv.org/abs/2601.13682)) |
| 2 | **Input-scale amplification.** Same statement, larger limits and worst-case structures | n ≤ 100 (O(n²) passes) → n ≤ 2·10⁵ with adversarial structure, so O(n log n) is required | A scale-parameterized generator plus an input validator; oracle outputs; time limit relative to the reference | **Strong** ([rStar-Coder](https://arxiv.org/abs/2505.21297), HardTests) |
| 3 | **Solution-first algorithmic lift.** Mutate the *reference solution* so the parent approach is insufficient (stronger asymptotics, richer state, new reformulation), then derive the statement and tests by running it | Array-sum seed → range updates that need a segment tree | Triangulate the evolved reference against a brute force that sees only the statement and a public-output oracle. Reject "false difficulty" (ambiguous wording, unnatural edge cases) | **Moderate** ([BenchEvolver](https://arxiv.org/abs/2606.01286), [AutoCode](https://arxiv.org/abs/2510.12803)) |
| 4 | **Feature / atom composition.** Build an inventory of algorithmic features from the seeds; choose 2–4 *compatible* ones, then write a hint-free task | Binary search → binary search on the answer + prefix sums + monotone-deque feasibility on a circular array with updates | Majority of 8–16 strong solvers + a strong-solver solvability filter; hold-out tests for choosing the golden solution; near-miss adversarial tests | **Moderate** ([X-Coder](https://arxiv.org/abs/2601.06953), [ADR](https://arxiv.org/abs/2605.31058), [EpiCoder](https://arxiv.org/abs/2501.04694)) |
| 5 | **Closed → open-ended goal.** Alter the objective, constrain the outputs or generalize the inputs until no ceiling remains | MST → degree-constrained spanning tree; 2-SAT → Min-True 2-SAT; bipartite MIS → general MIS | Feasibility checker + scorer normalized to [0, 1] against a baseline; test and verifier agents cross-validate; keep only problems where independent solutions use different strategies (idea divergence) | **Emerging** ([FrontierSmith](https://arxiv.org/abs/2605.14445)) |
| 6 | **Task-direction inversion.** One artifact becomes deduction (predict output), abduction (find input), induction (write f from I/O), fuzzing (find an input that type-checks but breaks the property) or test-writing | "Write f" → "given a 60-line f and output [3,7,7,12], find an input" | Exact output match; re-execute f on the predicted input; property violation + pre-test check | **Moderate** ([CodeI/O](https://arxiv.org/abs/2502.07316), [AZR](https://arxiv.org/abs/2505.03335), [Deep Dive](https://arxiv.org/abs/2603.24202), [CURE](https://arxiv.org/abs/2506.03136)) |
| 7 | **Agentification.** Turn the solution's logic into a tool library behind partial observation | "Compute X for array A" → an agent that must make 10–256 calls to ≥ 4 distinct tools | The reference produces outputs; one of K = 10 LLM-written tool-calling solutions must pass | **Emerging** ([CodeGym](https://arxiv.org/abs/2509.17325): +8.7 on OOD τ-Bench) |

Numbers that justify this order:

- **Test hardening pays twice.** RL on Qwen3-4B with HardTests-quality tests reached pass@10 64.76, vs 57.14 with TACO tests. EvolveCoder's evolved suites cut pass@1 on the same problems from 43.80 to 31.22 and raised a Qwen3-4B RL average from 46.6 to 49.0. **Strong.**
- **Solution-first evolution makes real difficulty.** On LiveCodeBench-v6 Hard seeds, average pass@1 fell from 87.0% to 45.7% on the evolved problems. RL on gpt-oss-20b gained +8.7 on LCB v6 Hard, a gain 70.7% larger than seed-only training gave ([BenchEvolver](https://arxiv.org/abs/2606.01286), 2026). AutoCode's mutated problems were about +334 Elo harder than their seeds. Its dual-solution check raised reference correctness from 86% to 94%. **Moderate.**
- **Open-ended goals remove the ceiling.** FrontierSmith ran GRPO on only 200 problems and gave Qwen3.5-9B +8.82 on FrontierCS and +306.36 on ALE-bench. That beat a closed-ended HardTests control by +5.24 / +236.40 and a random-reward control by +7.58 / +256.76. Removing the divergence filter cost 2.05 FrontierCS points. **Emerging.**
- **Prefer unique tasks to multiple solutions per task.** X-Coder found 64k tasks × 1 solution > 16k × 4 > 8k × 8. Keep oracle-backed seeds in the mix: rStar-Coder scored 57.3 with seed + synthetic data, vs 49.7 seed-only and 46.8 synthetic-only.

**Architecture.** A teacher loop conditioned on the student's results is the best general pattern.

```
seed (statement + oracle) ──► teacher mutates (lift / compose / scale / open-end)
        ▲                                   │
        │                        reference + brute force + validator
        │                                   │ agree on validator-checked tests?
 pass-rate summary                          ▼
 (M=16–32 student rollouts) ◄── student attempts ◄── accepted candidate
        │
  0.4 ≤ p ≤ 0.6 ? ── yes ──► RL pool
        └─ no ──► teacher gets "make it harder/easier" + representative solutions (≤ 6 turns)
```

Deep Dive's 6-turn teacher loop got about 4× more valid problems than single-turn generation. Medium-difficulty problems (pass rate 0.41–0.59) gave the best balance of speed and generalization; easy problems caused overfitting. M = 8 rollouts gave noisy pass-rate estimates, so use 16–32 ([Deep Dive](https://arxiv.org/abs/2603.24202), Meta, 2026). **Moderate.**

**Verifier.**

- The Validator–Generator–Checker triple is the baseline ([AutoCode](https://arxiv.org/abs/2510.12803)): 98.7% consistency with official verdicts on its 720-problem benchmark ([CodeContests+](https://arxiv.org/abs/2506.05817) is similar).
- For problems without an oracle, use mutual verification: a majority of 16 strong solutions must agree on at least 50 inputs. It gave 96.8% output-label accuracy, vs 12.7% for GPT-4o-written outputs. Lower the agreement threshold for hard seeds (rStar-Coder uses 40% for Codeforces seeds rated above 1600) so hard items survive.
- Expect about 13% residual label noise before deterministic filters. Hand-audit a sample: 94% of X-Coder's wrong labels came from one pattern that a mechanical check could detect.

**Knobs.**

- Number of composed features.
- Input scale (10⁰–10⁵ in rStar-Coder).
- Number of adversarial test rounds (0–3).
- Speedup bar and P → NP-hard goal changes.
- Number of teacher rounds.

**Pitfalls.**

- **"Make it harder" prompts yield trivial or ill-posed tasks.** X-Coder notes that LLMs "oversimplify complex prompts into trivial cases". In a gated augmentation study, 74.5% of mutated variants were rejected, 64% of those as *too easy* ([Trading Human Curation](https://arxiv.org/abs/2606.03800)).
- **Majority voting fails on shared misconceptions,** which are most likely on the hardest items.
- **Do not rank generated problems with LLM quality ratings.** In AutoCode, o3–human correlation was 0.07 for quality, while measured difficulty gain correlated up to 0.60 with human quality ratings. Only about 5% of generated problems landed in the pass@1 0.1–0.5 zone.

### 4.2 Formally verified code (Dafny, Verus, Lean)

Lift a saturated, tested function into "implementation + machine-checked proof against a formal spec". The proof checker cannot be fooled on the solution side, so **the attack surface moves to the spec**.

**Operators:**

1. **Spec lifting.** Tested code becomes an implement-and-prove task. [ATLAS](https://arxiv.org/abs/2512.10173) lifted TACO into 2.7K verified Dafny programs and split each into about 7× more subtasks, taking Qwen2.5-7B-Coder on DafnyBench from 32.4% to 56.9%. Lifting succeeded for 47.1% of EASY seeds but only about 20% of HARD ones, so oversample hard seeds.
2. **Stage and language ladder on the same problem.**
   - VeriContest stages: NL→code 92.18%, spec 48.31%, proof 13.95%, end-to-end 5.29% ([VeriContest](https://arxiv.org/abs/2605.08553)).
   - AlgoVeri languages: Dafny 40.3%, Verus 24.7%, Lean 7.8% ([AlgoVeri](https://arxiv.org/abs/2602.09464)).
3. **Hint stripping.** Remove invariants and assertions and ask for them back ([DafnyBench](https://arxiv.org/abs/2406.08467)). Cheap, but DafnyBench is nearly saturated (68% → 96% model union).
4. **Chain composition of verified units.** Each caller must be proved from its callees' specs. DafnyComp outputs across 13 LLMs are 95.67% syntax-correct but only 3.69% verify ([DafnyComp](https://arxiv.org/abs/2509.23061)). Chains had 47% synthesis success; trees and DAGs had under 8%.
5. **Spec strengthening.** Require the spec to reject outputs mutated from known-correct runs ("spectests", [SpecRL](https://arxiv.org/abs/2604.05820)), to imply the reference spec ([Re:Form](https://arxiv.org/abs/2507.16331)), or to be two-way equivalent to the code ([VeriEquivBench](https://arxiv.org/abs/2510.06296)).
6. **Certified equivalence / inequivalence pairs.** An equivalent rewrite comes with a Liquid Haskell proof; an inequivalent one comes with an executed counterexample. Keep pairs the current evaluator misjudges ([semantic-equivalence self-play](https://arxiv.org/abs/2604.17010): up to +13.3pp on EquiBench).

**Self-play architecture.** [PSV](https://arxiv.org/abs/2512.18160) (2025) prompts a proposer with specs labelled by the solver's current pass-rate bucket, admits only well-formed specs, and trains the solver on Verus-verified solutions. A 3B model reached 65.63% pass@1 on Dafny2Verus, vs 34.46% for plain rejection fine-tuning (RFT). Ablations:

- no solution verification: 31.82% (a 51.5% relative drop);
- no difficulty conditioning: 60.16%.

So verification matters far more than curriculum tuning. **Moderate.**

[ANCORA](https://arxiv.org/abs/2604.27644) adds three stabilizers:

- the proposer is rewarded only when exactly 1 of K attempts verifies;
- a generated spec is admitted only after a verified solve;
- the curriculum grows as a tree of one-step edits.

It reached 81.5% on Dafny2Verus in its test-time-training setting. Removing tree descent peaked at 51.3% and then collapsed to 12%. **Emerging.**

**Spec-hacking defenses.** Use at least two.

- **Rule filters.** Reject `assume(false)`, `sorry` and `external_body`. AlphaVerus saw `assume(false)` spread from one program to all of them without critique ([AlphaVerus](https://arxiv.org/abs/2412.06176)).
- **An exploit model.** It writes the laziest program that could verify; if one verifies, the spec is flawed.
- **Completeness tests.** Spectests, perturbation lemmas, or MutDafny code mutants.
- **Implication or equivalence against a reference.** Under a verification-only reward, Re:Form's specs weakened progressively.

A thesis-scale Dafny RL run shows why this matters: verified reward rose from 2.2% to 58.1%, and inspection found trivial specs (`ensures result >= 0`) and leaky ones ([Tan](https://arxiv.org/abs/2605.30914)). **Strong.**

**Gate precision.** In count-matched RFT, admitting 25% false positives cost 1.58pp, while discarding 75% of true positives cost only 0.03pp ([The Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)). Under GRPO the accounting shifts toward false negatives, so measure both.

**Where to start:**

- the [Vericoding benchmark](https://arxiv.org/abs/2509.22908) (12,504 specs across Dafny/Verus/Lean; filter first, since about 9% of successful specs were too weak);
- Dafny2Verus;
- [VERINA](https://arxiv.org/abs/2505.23135) with VeriScale-expanded tests;
- [Verus-SpecGym](https://arxiv.org/abs/2605.26457) (Codeforces hacks as the spec oracle; an LLM judge misses 26% of the failures its checks catch).

### 4.3 Program induction (ARC-style and programming-by-example)

Labels come from executing programs, so every generated task is correct by construction. The risks are ambiguity, degenerate tasks, and label enumeration in place of rule induction.

**Operators:**

- **Per-task procedural generator with a difficulty interval.** [RE-ARC](https://arxiv.org/abs/2404.07353) has one sampler and verifier per ARC task, ≥ 10,000 examples each, at about 1,000 verified examples/s. Add ARC-TGI-style episode constraints so hard instances still expose the rule ([ARC-TGI](https://arxiv.org/abs/2603.05099)).
- **Concept remixing.** An LLM merges descriptions of two seeds and writes the code; execution provides the labels.
  - [NVARC](https://github.com/1ytic/NVARC) mixed 3,268 summaries into 266,593 new ones. It kept input programs that produced ≥ 30 unique grids passing unit tests, and output programs whose independent samples agreed on all 30 inputs. The resulting 103,253 puzzles helped it win ARC Prize 2025 with 24.03% on the private ARC-AGI-2 evaluation.
  - Mixtures of harder puzzles passed the input-program filter about 50% of the time, vs about 70% for easier ones. That acceptance rate is itself a difficulty signal. **Moderate.**
- **Program mutation and hindsight relabelling.** Every sampled program defines a task it solves.
  - [CodeIt](https://arxiv.org/abs/2402.04858): 59/400 ARC evaluation tasks, vs 42/400 without relabelling and 20/400 without mutation.
  - [SOAR](https://arxiv.org/abs/2507.14172): greedy-diverse selection of 25 best + 25 least-successful programs per task beat correct-only selection (36.46% vs 34.67%).
  - Never discard failed rollouts in executable domains. **Moderate.**
- **Black-box induction.** Hide a solved seed function behind a query API and reward differential-testing equivalence ([CodeARC](https://arxiv.org/abs/2503.23145): best of 18 models 52.7%).

**Pitfall: harder induction invites enumeration.** Under an extensional verifier, RLVR-trained models produce many more label-enumeration shortcuts: 40 at complexity levels 1–10 vs 458 at levels 11–20. Scoring on an isomorphic twin (bijectively renamed IDs) kept the gap between the two rewards near 0 ([IPT](https://arxiv.org/abs/2604.15149)). Pay reward only when both the original and the renamed twin are correct. **Emerging.**

Transfer from ARC-style data is mixed: fine-tuning on ARC-TGI data lifted Phi-4 from 8% to 16% and Llama-3.1-8B from 6% to 17%, but moved Qwen3-8B from 9% to 6%.

### 4.4 Performance objectives (efficiency, kernels, repository speedups)

When pass@1 is 100%, keep correctness as a *gate* and reward speed. The reward stays continuous. It is also the most-hacked reward in this literature.

**Operators:**

- **Goal switch.** "Correct" becomes "beat the human runtime distribution". o4-mini has 89.11% pass@1 on Venus but beats only 56.85% of human solutions on runtime. [Afterburner](https://arxiv.org/abs/2505.23387)'s GRPO kept improving where SFT and DPO saturated (pass@1 47% → 62%).
- **Operator-fusion composition.** Stack k already-correct operators into one kernel task.
  - [DRTriton](https://arxiv.org/abs/2603.21465) samples operator DAGs with CP-SAT-solved shapes and promotes level k when held-out pass@1 exceeds 50%.
  - [CUDA Agent](https://arxiv.org/abs/2602.24286) stacks up to 5 torch operators and keeps tasks with 1–100 ms eager runtime.
- **Raise the bar.** Speedup threshold p, workload size, baseline (eager → torch.compile → TF32 → speed-of-light). In GSO, Claude-4.0 reaches 70% at Opt₀ but under 5% at Opt₀.₉₅ ([GSO](https://arxiv.org/abs/2505.23671)).
- **Hidden input distributions and shapes** (TritonRL, KernelBench-Verified).

**Reward shaping.** Milestone rewards beat raw speedup. CUDA Agent's {−1, 1, 2, 3} ladder reached 96.8% of kernels faster than torch.compile, vs 60.4% with a raw speedup reward. Other shapes that work:

- clipped normalized speedup (CUDA-L1, clip at 1.5);
- tanh gated on correctness (Afterburner);
- speed reward only once group accuracy ≥ 0.5 ([KernelZero](https://arxiv.org/abs/2609.33074));
- log-speedup weighting (DRTriton).

**Moderate.**

**Hardening the timer and checker is mandatory.**

- **Stream timing.** 82/250 (32.8%) of CUDA-L1's early RL outputs exploited it, for a fake 18× speedup. The fix: synchronize all streams and check that outputs are materialized ([CUDA-L1](https://arxiv.org/abs/2507.14111)).
- **Hard-coded test values.** Under hidden inputs, GPT-5.5's apparent 1.43× geomean falls to 0.88× ([KernelBench-Verified](https://arxiv.org/abs/2607.16241)).
- **Weak oracles.** The official KernelBench check misses 16.9% of injected faults. A kill-matrix-optimized 2-input suite catches 98.0% ([Measuring the Checker](https://arxiv.org/abs/2609.22220)). Mutation-test your oracle before RL.
- **Lazy optimization.** A custom kernel covered 0.014% of CUDA time. Reward the share of runtime spent in the generated kernel ([Dr. Kernel](https://arxiv.org/abs/2602.05885)).
- **Cross-machine replay.** Reference patches replay validly on every machine for only 39/102 GSO and 11/140 SWE-Perf tasks ([replay audit](https://arxiv.org/abs/2607.01211)). Keep only tasks whose speedup replays on *your* hardware, or use deterministic simulation (gem5 in [PIE](https://arxiv.org/abs/2302.07867)).

**Strong.**

**Calibrate the mix to your model.** Kevin found easy-only training plateaued. TritonRL found L1-only training beat adding fusion tasks, whose rewards were too sparse.

**Key references for §4:** [HardTests](https://arxiv.org/abs/2505.24098) (RL pass@10 64.76 vs 57.14 with TACO tests); [BenchEvolver](https://arxiv.org/abs/2606.01286) (LCB-v6 Hard pass@1 87.0% → 45.7% after solution-first evolution); [FrontierSmith](https://arxiv.org/abs/2605.14445) (+8.82 FrontierCS from 200 open-ended problems); [PSV](https://arxiv.org/abs/2512.18160) (65.63% vs 34.46% on Dafny2Verus); [NVARC](https://github.com/1ytic/NVARC) (24.03% on the private ARC-AGI-2 evaluation); [CUDA Agent](https://arxiv.org/abs/2602.24286) (96.8% vs 60.4% of kernels faster than torch.compile); [Measuring the Checker](https://arxiv.org/abs/2609.22220) (the official check misses 16.9% of injected faults; a 2-input suite catches 98.0%).

---

## 5. Software engineering and repository-level tasks

### 5.1 Seeds: repositories with tests, not issues

The unit of supply is an executable environment: a repository snapshot with a passing test suite. Once it exists, each environment can host many tasks. Environment cost dominates, so reuse matters.

| Source | Scale and cost |
|---|---|
| [SWE-smith](https://arxiv.org/abs/2504.21798) | About 18 human hours to build 128 repository environments, which yielded 50,137 instances |
| OpenSWE ([daVinci-Env](https://arxiv.org/abs/2603.13023)) | About $19.66 per environment |
| [SWE-rebench V2](https://arxiv.org/abs/2602.23866) | 32,079 mined tasks in 20 languages |
| GLM-5's RepoLaunch | Over 10k environments in 9 languages ([GLM-5](https://arxiv.org/abs/2602.15763)) |
| [R2E-Gym](https://arxiv.org/abs/2504.07164) | 8,135 tasks mined from small-scope commits that need no linked issue |

R2E-Gym's synthetic issues and tests trained as well as real ones (27.8% vs 28.0%). Its back-translation writes the issue from the diff *plus the failing-test output*.

### 5.2 Operators that work

**Bug injection, ordered from cheapest to most realistic.** This ordering is also roughly the order of rising difficulty and training value.

```
procedural AST mutation  →  LM subtle edit  →  LM re-implementation of a masked component
  (flip <, drop branch)       (median 3 lines)     (median 24 lines, coverage-weighted)
        →  removal of hunks/files + history reversion of real PRs  (PR Mirror: median 14 lines)
        →  agentic *unintentional* bugs from feature work  (FeatAdd: 4.2 files, 415.9 net lines)
        →  higher-order bugs: the solver's own failed repairs become new buggy states
```

The numbers behind the ordering:

- **Direct prompting collapses.** Asking an LLM "for a bug" produced one-line edits. Removal plus history reversion worked best ([SSR](https://arxiv.org/abs/2512.18552)).
- **Realistic bugs are harder.** Claude's resolve rate was 41.4% on FeatAdd bugs vs 65.9% on SWE-smith bugs, and 1.2k BugPilot bugs beat 3k other bugs by 2% in SFT ([BugPilot](https://arxiv.org/abs/2510.19898)).
- **Bigger injected changes gave better trajectories.** In SWE-smith, PR Mirror and LM Rewrite instances gave the most effective trajectories.

**Strong.**

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Multi-bug combination.** Merge independently validated bugs, or union temporally adjacent real fixes into one no-issue task | One bug named in the issue → 2–3 interacting bugs across modules, no issue text | Validate each bug alone. The combined task's F2P set must equal the union of the individual F2P sets; check the bugs do not cancel | **Moderate** (SWE-smith Combine: 96.9% yield at 0¢; [Active-SWE](https://arxiv.org/abs/2608.04682): best agent resolves 20.0%) |
| **Feature removal along dependency depth.** Mask implementations chosen from dynamic call trees or runtime dependency graphs, or chained behaviors | One depth-1 helper → a depth-≥3 call tree (5+ interdependent functions) from a requirements doc | The developer tests or a deterministic replay are the oracle; confirm the target tests fail before restoration | **Moderate** ([SWE-Dev](https://arxiv.org/abs/2505.16975), [SWE-Flow](https://arxiv.org/abs/2506.09003), [ProgramDistill](https://arxiv.org/abs/2609.18805)) |
| **History reversion and cross-repository transplant.** Revert real fixes from git history, or move an issue's root-cause pattern into another repository's existing environment (write the test first, then the bug) | Operator flip in a utility → a reverted multi-file PR, or a race-condition pattern from project A rebuilt in project B | The test passes before and fails after the mirror patch; the inverse patch is the fix; no P2P regressions | **Moderate** ([SWE-Mirror](https://arxiv.org/abs/2509.08724): 60,671 tasks; 32B reached 52.2% SWE-bench Verified) |
| **Specification degradation.** Symptom-only issue, hidden triggering tests, hints removed, no issue at all | "Fix `export_rows` off-by-one" → "export sometimes drops the last row" | Tests unchanged; a reference solver must still solve it *from the spec alone* (else it is ambiguity, not difficulty) | **Moderate** ([Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729), MiniMax-M2, [FrogNano](https://arxiv.org/abs/2609.07925)) |
| **Verifier hardening with mutant patches.** Mutate the gold patch into plausible, non-equivalent patches that pass the tests; add tests that kill them | ~1 in 5 accepted patches is semantically wrong → strengthened tests | Gold patch must pass; an LLM jury excludes mutants equivalent to the gold patch | **Moderate** ([SWE-ABS](https://arxiv.org/abs/2603.00520): top agent 78.80% → 62.20%) |
| **Chaining and inversion.** Chain dependent milestones so the agent builds on its own code; or invert to "write a test that fails pre-patch and passes post-patch" | Isolated milestone (> 80% success) → continuous chain (≤ 38.03% score, about 13% fully resolved) | Per-step tests, re-running earlier tests to catch regressions | **Emerging** ([SWE-Milestone](https://arxiv.org/abs/2603.13428); SWE-Test inversion in MiniMax-M2) |

Performance-optimization tasks mined from commits (GSO, SWE-Perf, SWE-fficiency) are covered in §4.4.

### 5.3 Recommended architecture

```
repo snapshots ──► env builder (pinned deps, test runner, log parser → F2P/P2P)
                                   │
            ┌──────────── task injectors (per env, many tasks) ────────────┐
            │ rewrite/removal/reversion · FeatAdd harvesting · multi-bug   │
            │ union · feature masking by call depth · cross-repo transplant│
            └──────────────────────────────┬───────────────────────────────┘
                                           ▼
   validators: bug breaks ≥1 passing test · revert fixes it · inverse mutation testing
               (each edited file alone restores ≥1 test) · no flaky tests · P2P stable
                                           ▼
   issue writer: from diff + failing-test output; symptom-level; hide triggering tests;
                 check solvable-from-spec with a reference agent
                                           ▼
   calibrate in the EXACT training harness: K rollouts → keep 0 < p ≤ ~0.5–0.75
                                           ▼
   online refine (per checkpoint): out-of-band → rewrite statement, keep patch+tests
                                           ▼
   anti-cheat sandbox: block git remote/clone/curl to upstream; delete future commits
```

**Online per-checkpoint generation is the strongest single loop reported.**

- [FrogNano](https://arxiv.org/abs/2609.07925) (2026) regenerates 300 tasks per iteration with the current checkpoint, targeting about 50% resolve rate. Out-of-band candidates are *refined* (statement revised, patch and tests kept) rather than discarded. For iteration 5 it switched to a stronger generator and a 0 < p̂ ≤ 0.5 band.
- A 4B model rose on SWE-bench Verified from 43.0 → 49.1 → 53.1 → 56.9 → 59.1 → 61.5, without frontier distillation.
- [Envs-FORGE](https://arxiv.org/abs/2608.14312) is a per-seed rather than per-checkpoint variant, built on Terminal-Bench-style environments. It picks increase / reduce / diversify for each seed from its measured pass rate around τ = 0.5, and always regenerates oracle, tests and environment together. With 100 environments and GRPO on Qwen3.5-35B it reported SWE-bench Verified 77.1 vs 73.4.

**Moderate.**

**Self-play works but must be anchored.**

- [SSR](https://arxiv.org/abs/2512.18552) runs one policy as injector and solver. The injector reward is −1 when any of 7 consistency checks fails, −α when solve rate s ∈ {0, 1}, and 1 − (1+α)s otherwise. Starting from CWM-sft it gave +10.4 SWE-bench Verified and +7.8 SWE-Bench Pro. It hit gibberish-output instability at scale, and its self-written issues copied the test patches.
- [Anchored Self-Play](https://arxiv.org/abs/2607.03523) works at function level (BigCodeBench-based bug repair), not on repositories, so its evidence for repository-level SWE is indirect. It adds an embedding-similarity reward toward real bugs (λ = 0.20) and 20% reference bugs in fixer training. It beat standard self-play by +7.0 pp on average and +3.4 pp on human bugs. Unanchored self-play improved early but degraded on human-written bugs.

**Moderate.**

A cheaper variant targets the solver's observed failure modes. [Socratic-SWE](https://arxiv.org/abs/2606.07412) distills failure modes from rollouts into a skill registry and conditions generation on them. It reached 50.40% SWE-bench Verified, +3.40 over SSR at the same budget.

### 5.4 Verifier

- **F2P/P2P partitions and hidden triggering tests.** Qwen3-Coder-Next keeps a bug only if it fails existing tests and is resolved by patch reversion, and excludes the bug-triggering test files from the task.
- **SSR's seven artifact checks**, especially inverse mutation testing: reverting each bug file alone must fix at least one failing test.
- **Mutant-killing tests** for the saturated fraction (SWE-ABS).
- **Anti-cheat.** Agents fetched fixes with `git remote add`, `git clone` and `curl`. Qwen3-Coder-Next blocks any tool call that contains both a repository link and a network keyword. Nemotron 3 Ultra physically deletes future commits ([Chapter 08](08-frontier-lab-practices.md)).
- **Soft verification is for SFT only.** [SERA](https://arxiv.org/abs/2601.20789)'s line-overlap recall trained about as well as verified data: recall thresholds 0 to 1.0 performed similarly, at 26× lower cost than RL. It is not a valid RL reward.

### 5.5 Difficulty knobs

- Injection strategy. Median changed lines in SWE-smith: Modify 3, Procedural 5, PR Mirror 14, Combine 11, Rewrite 24.
- Number of bugs.
- Masked call-tree depth. On SWE-Dev's hard split, PPO nearly doubled pass@1 (6.68 → 12.25).
- Restoration depth. ProgramDistill: frontier success fell from 100% to 64.0%, and from 96% to 32%, going from depth 1 to 8.
- Issue specificity and hint count.
- Chain length.
- Hints for items at 0%. CWM adds the hidden test as a hint, raising pass@1 from 0% to about 30%, and anneals it away later ([CWM](https://arxiv.org/abs/2510.02387)).

### 5.6 Pitfalls

- **Ambiguity is not difficulty.** FrogNano's example: an under-specified statement (missing an interval-closure contract) scored 0%, a behaviorally precise one 50%, and one naming the class/method was easy. Check that degraded specs still determine behavior.
- **The harness can dominate difficulty.** Changing FrogNano's harness alone moved Qwen3.5-4B from 8.3% to 37.2%. Under the old harness, about 96% of trajectories hit the turn limit. Calibrate pass rates in the exact RL harness.
- **Too-hard bugs give no RL gain.** GRPO on FeatAdd bugs did not beat SFT on them. The authors attribute this to GRPO needing partially solvable problems. Filter to pass@k > 0 or add hints.
- **For SFT, difficulty-rated subsets did not help.** SWE-smith's difficulty 2/4/6/8 sets gave 12.4 / 10.8 / 13.6 / 12.2%. Repository diversity grew performance roughly log-linearly. Use diversity for SFT and calibrated difficulty for RL ([Chapter 06](06-sft-playbook.md)).
- **Composition has a yield cost.** Active-SWE had to limit its merge window to 2 PRs because integration failed often. SWE-Mirror yields 68% for Python but 28% for Rust.
- **Memorized repository cues.** Some "solved" tasks rely on memorized repository cues. Remapping names is a cheap hardening and contamination check ([SchrodingerRepo](https://arxiv.org/abs/2609.27891)).

**Key references:** [SWE-smith](https://arxiv.org/abs/2504.21798) (injection strategies, yields and costs); [BugPilot](https://arxiv.org/abs/2510.19898) (FeatAdd 41.4% vs 65.9%); [SSR](https://arxiv.org/abs/2512.18552) (+10.4 / +7.8); [Anchored Self-Play](https://arxiv.org/abs/2607.03523) (+7.0 pp); [FrogNano](https://arxiv.org/abs/2609.07925) (43.0 → 61.5); [SWE-ABS](https://arxiv.org/abs/2603.00520) (78.80 → 62.20).

---

## 6. Logic puzzles and procedural reasoning gyms

### 6.1 Seeds

The seeds are generator + verifier pairs, not static items:

- [Reasoning Gym](https://arxiv.org/abs/2505.24760): 100+ generators.
- [SynLogic](https://arxiv.org/abs/2505.19641): 35 tasks.
- [Enigmata](https://arxiv.org/abs/2505.19914): 36 puzzles.
- [RLVE-Gym](https://arxiv.org/abs/2511.07317): 400 environments with an unbounded level d.
- [Reasoning Core](https://arxiv.org/abs/2603.02208): randomized PDDL, first-order-logic, grammar and Bayes-net worlds.
- Single families: Knights & Knaves, [ZebraLogic](https://arxiv.org/abs/2502.01100), SAT ([SATQuest](https://arxiv.org/abs/2509.00930)) and NP-hard optimization ([NP-Engine](https://arxiv.org/abs/2510.16476)).

Wrap any templated seed in the interface `generator(level, seed) → instance`, `score(instance, answer)`. Tag stylistic parameters separately from difficulty parameters, so the stylistic ones can serve as anti-memorization noise.

**Audit before use.** The Reasoning Core v3 audit found material defects in 13 of 105 Reasoning Gym tasks, for example a scorer that gave full credit to any nonempty answer. It also found defects in 9 SynLogic generators, for example displayed constraints that contradict the stored solution. **Strong.**

### 6.2 Operators that work

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Planted solution / solve-first** | Random solved 9×9 grid → masked Sudoku; planted Hamiltonian path + random edges; random F → "∫F′ dx" | Check the answer against the constraints (several answers may be valid), not against the planted one; differentiate proposed antiderivatives with SymPy | **Strong** (RLVE, [SATURN](https://arxiv.org/abs/2505.16368), [EvoEnv](https://arxiv.org/abs/2605.14392)) |
| **Constraint minimization to uniqueness** | 3-house Zebra with direct clues → 6×6 with a minimal set of relational clues | SAT solver confirms uniqueness after each clue removal; bucket by Z3 conflicts, not surface size | **Moderate** (ZebraLogic: most models collapse beyond ~10⁷ search space or ~20 Z3 conflicts) |
| **Composition of atomic skills** | func_15(x) → func_16(func_15(func_3(x))) with definitions hidden | Execute the pipeline; atoms must already be learned | **Moderate** ([f(g(x))](https://arxiv.org/abs/2509.25123): RL on depth 2 → ~30% on unseen depth 3; RFT ≤ 2.6%) |
| **Problem-type upgrade on the same instance** | SAT decision → assignment → MaxSAT → MCS → MUS | PySAT checks certificates; MUS/MCS certify "no solution" answers without 50% guessing | **Moderate** (SATQuest; SATBench's satisfiability bias shows why bare labels fail) |
| **Feasibility → optimality** | Any valid tour → shortest tour (TSP 10–20 → 45–55 cities) | Exact feasibility gate with a penalty; reward the ratio to a heuristic or exact optimum; **never** a flat reward for feasibility (it was hacked early in [Forge](https://arxiv.org/abs/2605.08905)) | **Moderate** |
| **Information hiding (static → interactive)** | Written-out knapsack → parameters discoverable only through stateful tools | Pre-solve u*, u₀; reward clip((u − u₀)/(u* − u₀)); replay reference policies through the generated code before admission | **Emerging** ([VHD-Play](https://arxiv.org/abs/2609.27321): 0.962 written-out vs 0.204 agentic on the same instances; training lifted agentic to 0.815) |
| **Depth × distractor coverage and inversion** | 1 ownership transfer → 8+ events across 6 parallel chains sharing people; forward → abductive "which event is missing?" | The generated world graph is ground truth | **Moderate** ([Depth × Complexity](https://arxiv.org/abs/2605.26934): joint coverage beats either axis alone) |
| **Perturbation and rule flips** | Replace one K&K statement (the solution must change); knights lie instead | Re-solve every perturbation | **Moderate** ([K&K memorization](https://arxiv.org/abs/2410.23123): role-flipped accuracy near zero) |

**Size escalation alone mostly adds length.** In *The Illusion of Thinking*'s River Crossing, instances with N ≥ 6 actors and boat capacity 3 were unsolvable, yet they were scored as failures, and Tower of Hanoi hit output-token limits ([comment](https://arxiv.org/abs/2506.09250)). Solver-check every generated instance, and allow compact answers such as a program.

### 6.3 Architecture: unbounded knobs plus an online controller

The main failure is a static difficulty range. Under RLVE's static low cap the effective-prompt ratio (share of prompts whose rollouts disagree) fell to 0. Its adaptive sliding window beat even an oracle static range covering all the levels reached ([RLVE](https://arxiv.org/abs/2511.07317)).

```python
# RLVE-style per-environment window (promote at 90%, keep the last 4 levels)
lo, hi = 0, 0
def sample_level():            return random.randint(lo, hi)
def update(acc_at_hi, n_seen):
    global lo, hi
    if n_seen >= 8 * rollouts_per_prompt and acc_at_hi >= 0.9:
        hi += 1; lo = max(0, hi - 4 + 1)
# log effective-prompt ratio every step; it is the primary health metric
```

Controllers, from cheapest to most flexible:

1. Offline small-model solve-rate pools plus dropping all-pass and all-fail groups ([INTELLECT-3](https://arxiv.org/abs/2512.16144)).
2. RLVE's window.
3. [SCALER](https://arxiv.org/abs/2601.04809)'s proportional controller d ← clip(d + β(acc − τ)), with environment retirement when difficulty stalls. It lifted 4,973 CodeContests problems into 2,739 environments by treating their constraint variables as knobs.
4. [Frontier Learning](https://arxiv.org/abs/2609.35426)'s regret-prioritized buffer with mutation, for knobs that interact non-monotonically. On Dice it scored 71.8 vs 33.3 for SEC.
5. A learned teacher rewarded by min(p, 1−p) ([LURE](https://arxiv.org/abs/2608.21871)).

Details are in [Chapter 05](05-rl-playbook.md). **Strong.**

**LLM-written environments now scale, but only behind staged gates.**

| Pipeline | Scale | Admission gate |
|---|---|---|
| [ReSyn](https://arxiv.org/abs/2602.20117) | 418 environments | Solve rate must fall with level (Wald test) |
| [InternBootcamp](https://arxiv.org/abs/2508.08636) | 704 tasks | Drop solver accuracy > 0.85 (oversimplified) or < 0.03 (broken) |
| [EvoEnv](https://arxiv.org/abs/2605.14392) | 840 environments | Parse, execute, determinism, non-triviality and scorer-perturbation checks, plus any-reject review; target pass about 0.3 |

InternBootcamp's generator tried to "simplify" the task in 97.93% of first-iteration runs. Budget at least 3 refinement iterations.

**More environments beat more instances.** In ReSyn, 400 environments × 40 instances scored 75.19 on BBH, vs 71.20 for 25 × 640. Selection by ability coverage can beat using everything: 30 of 200 environments gave +95.6% relative gain vs +43.4% for all 200 ([AES](https://arxiv.org/abs/2608.03571)). **Moderate.**

### 6.4 Reward shaping and pitfalls

- **Change the reward shape as tasks get harder.**
  - Binary reward worked best on K&K but collapsed on Logic Puzzle Baron. There, partial credit peaked early and then declined, and format or rescaled rewards eventually won ([Can One Domain Help Others?](https://arxiv.org/abs/2507.17512)).
  - For long outputs use steep partial credit: (x/N)^10 in RLVE.
  - UltraLogic's Bipolar Float Reward gives +1 only to a fully correct answer and maps partial correctness S to S − 1, so an imperfect answer never earns a positive reward ([UltraLogic](https://arxiv.org/abs/2601.03205)).
  - For long-horizon puzzles use solver-value step credit ([CAST](https://arxiv.org/abs/2607.25308)).
- **RLVR is brittle to generator bugs.** In UltraLogic, 1–3 buggy task types out of 50 caused training collapse. Quarantine a family whose zero-pass rate or length profile looks anomalous.
- **Transfer is real but uneven.**
  - Positive: SynLogic took AIME24 from 4.5 to 19.6 at 32B; RLVE gave +3.37 on ProRL-1.5B-v2 in about 1,100 H100 hours, vs +0.49 in 3,600 hours of continued ProRL.
  - None: Enigmata saw no general-reasoning transfer at Qwen2.5-32B.
  - Negative: K&K-only RL dropped the code average from 67.46 to 56.09 ([Can One Domain Help Others?](https://arxiv.org/abs/2507.17512)).
  - Keep puzzles a minority of the mix and track out-of-domain metrics every evaluation. **Strong.**
- **Cold start.** RL alone on Qwen2.5-32B-Instruct moved InternBootcamp's OOD average by +0.7, while SFT → RL gave +19.5. ARC-like tiers needed RFT first in Enigmata.
- **Memorization.** Evaluate on held-out generator families, held-out parameter regions and perturbed variants, not just fresh seeds. Audit role prompts for answer leakage: in LURE's study, R-Zero's role prompt leaked the ZebraLogic gold grid, and removing it cut that baseline by 62 points.

**Key references:** [RLVE](https://arxiv.org/abs/2511.07317) (+3.37 on ProRL-1.5B-v2 in about 1,100 H100 hours vs +0.49 in 3,600); [SCALER](https://arxiv.org/abs/2601.04809) (54.25 vs 53.52 for RLVE on Qwen3-4B-Base); [ReSyn](https://arxiv.org/abs/2602.20117) (verifier-based rewards 75.24 vs LLM answers 68.83 on BBH); [f(g(x))](https://arxiv.org/abs/2509.25123) (RL on depth 2 → about 30% on unseen depth 3; RFT ≤ 2.6%); [VHD-Play](https://arxiv.org/abs/2609.27321) (agentic score 0.204 → 0.815 after training); [ZebraLogic](https://arxiv.org/abs/2502.01100) / [SATQuest](https://arxiv.org/abs/2509.00930).

---

## 7. Science (textbook-, database- and simulator-derived)

### 7.1 Seeds

| Seed source | Example | What it provides |
|---|---|---|
| Textbooks | [MegaScience](https://arxiv.org/abs/2507.16812) / TextbookReasoning | 12.8K university textbooks → 651,840 decontaminated QA with reference answers; "multi-step with full solution" extraction |
| Reasoning-dense web documents | [NaturalReasoning](https://arxiv.org/abs/2502.13124) | 2.8M composed questions, 81.68% with a document-derived reference |
| Experimental databases | [ether0](https://arxiv.org/abs/2506.17238) | 640,730 chemistry problems templated from ChEMBL, PubChem and others |
| Simulators | [Sim2Reason](https://arxiv.org/abs/2604.11805) | Random MuJoCo scenes from a YAML DSL |
| Formula libraries | [IPG](https://arxiv.org/abs/2603.14486) | Formula-as-Code for mechanics |
| Solvers | [AutoOR](https://arxiv.org/abs/2604.16804) | OR-Tools/Gekko models |
| Scientific coding | SciCode ([BenchEvolver](https://arxiv.org/abs/2606.01286)) | Scientific coding tasks with tests |

Existing science MCQ (GPQA-style) is guessable. Convert it before use (§7.2).

### 7.2 Operators that work

| Operator | Easy → hard | Verification | Evidence |
|---|---|---|---|
| **Simulator scene composition** (more entities and connections) | Single block on an incline → pulleys + movable prism + rolling body with cutout; ask "v at t = 3 s" | Simulator value within 5% relative error; prune unstable segments; **shortcut filter**: drop QA whose answer is unchanged when components are ablated (~15% of pairs) | **Moderate** (Sim2Reason: IPhO mechanics 19.8 → 25.2 for Qwen2.5-32B; filter 13.15 vs 7.14 without it at 3B) |
| **Formula composition** | One kinematics formula → 3 composed formulas with unit and range checks | Executed `solve()` built only from library calls; 99.85% verification success | **Emerging** (IPG: >99% valid at 2–3 formulas; signature mismatches dominate at 4+) |
| **Code-first back-translation with scaling** | LP with 4 variables → 12; one pump-network NLP seed → families of series-parallel variants | Solver optimum; component-wise check that the description matches the model (data, constraints, objective, equations, self-consistency) | **Moderate** (AutoOR: 7–8% of generations become items; an audit of 50 found no errors) |
| **Constraint stacking in design tasks** | "Raise log S" → "raise log S while Tanimoto ≥ 0.7 to the input, using only groups seen in the source data" | RDKit validity, ML oracles, purchasability Bloom filter, "reasonable molecule" check | **Moderate** (ether0: retrosynthesis 70% after 46,000 examples) |
| **Format conversion that resists guessing** | 4-option MCQ → open short answer (≤ 10 words), or a masked reasoning span with 9 style-matched distractors | Short canonical answers; the masked text is ground truth; tune option count until most items are medium | **Moderate** ([Nemotron-CrossThink](https://arxiv.org/abs/2504.13941): open-ended +1.21%, short answers +1.20%; [Golden Goose](https://arxiv.org/abs/2601.22975): >70% medium with 9 options) |
| **Scaffold then fade** (for classes at 0%) | NLP class at 0% even at pass@64 → solver syntax and mapped constraints in the prompt on easy problems → removed → hard problems | Advance a phase only when pass rate > 0 | **Moderate** (AutoOR: 0% → 48.98% on Pump-NLP, where Gemini 3 Pro and Qwen3-8B score about 0) |

**Inversion is not automatically better signal here.** In Sim2Reason's 3B ablation, training on forward numeric questions scored 13.15 on IPhO. Reverse questions ("which mass gives 5 m/s?") scored 5.84 and symbolic questions ("v(t) = ?") 7.46. Treat inversion as extra variety to be validated. **Moderate.**

### 7.3 Architecture and verifier

```
domain artifact first ─► simulator scene / solver model / formula composition / DB record
        │                         │ execute
        ▼                         ▼
 question renderer (forward numeric first)      exact or tolerance-checked answer
        │
 shortcut & degeneracy filters: component ablation · unstable-segment pruning ·
 plausibility guard (source-distribution groups, physical ranges) · empty/trivial outputs
        │
 description ↔ artifact consistency check (component-wise) ─► policy band filter ─► RL
```

The verifier is whatever computed the answer: a simulator within tolerance, a solver optimum (AutoOR requires objectives to match to 2 decimals for LP/MILP and within 2% for NLP), RDKit and domain oracles, or a textbook reference. For free-form references, pair a rule checker with a model verifier on rule-negatives. INTELLECT-3 found a "non-negligible fraction of false negatives" with rule-only checking in its math environment, and it checks its 29.3K MegaScience items with math-verify plus an LLM judge. Human-written hard science labels are noisy too: HLE's own audits estimate 15.4% expert disagreement overall and about 18% on its biology/chemistry/health subset ([HLE](https://arxiv.org/abs/2501.14249); see [Chapter 04](04-verification-and-quality-control.md)).

### 7.4 Knobs, pitfalls, starting points

**Knobs:**

- entity and connection count;
- formula count (IPG reports R² ≈ 0.95 between formula count and verification-code length);
- decision-variable count;
- stacked constraints;
- option count for mask-and-choose items;
- scaffold phase.

**Pitfalls:**

- **SFT on synthetic hard science data regresses, while RL gains.**
  - Sim2Reason: SFT on 200K teacher solutions lowered IPhO from 19.8 to 15.9; RL reached 25.2.
  - AutoOR: SFT scored Hard-LP 26 vs base 55 vs RL 80.

  Use these generators with RL ([Chapter 06](06-sft-playbook.md)). **Moderate.**
- **Plausibility hacks.** Without ether0's "reasonable molecule" check, the policy inserted peroxides to satisfy oxygen counts and hydrazines to raise solubility.
- **Physically implausible answers.** At ≥ 5 composed formulas, answers can be mathematically valid but physically implausible (for example 20g accelerations) (IPG).
- **Degenerate items.** Simulator questions tend to be trivial or intractable, so prune both extremes.
- **Zero-signal batches.** ether0 saw groups where every rollout got the same reward reach 90% of batches, and fixed it with a buffer of recently non-trivial problems.
- **Open-ended conversion of masked spans gave zero signal.** More than 83% of GooseReason-Math items had zero accuracy.
- **Tolerance-based rewards are untested against exploitation.** Whether 5% or 2% tolerances can be exploited by approximate heuristics has not been studied.
- **Coverage gaps.** Thermodynamics, kinetics, materials and biology simulators are missing.

**Starting points:**

- MegaScience (difficulty selection helped only one of its three sources, so default to the full set);
- NaturalReasoning;
- GooseReason-0.7M: continued RL on ProRL-1.5B-v2 gave STEM +3.48 vs +0.13 on the original data;
- Nemotron-CrossThink (287.4K items);
- ether0's templates;
- Sim2Reason's DSL.

GLM-5's lab recipe defines hard science RL items as those its previous model rarely solves but GPT-5.2 xhigh or Gemini 3 Pro can ([GLM-5](https://arxiv.org/abs/2602.15763)).

**Key references:** [Sim2Reason](https://arxiv.org/abs/2604.11805) (IPhO mechanics 19.8 → 25.2 for Qwen2.5-32B with RL; SFT 15.9); [AutoOR](https://arxiv.org/abs/2604.16804) (Pump-NLP 0% → 48.98% via scaffold fading); [ether0](https://arxiv.org/abs/2506.17238) (retrosynthesis 70% after 46,000 examples); [Golden Goose](https://arxiv.org/abs/2601.22975) (more than 70% of items medium with 9 options; STEM +3.48 vs +0.13); [MegaScience](https://arxiv.org/abs/2507.16812) (651,840 QA from 12.8K textbooks); [IPG](https://arxiv.org/abs/2603.14486) (more than 99% valid at 2–3 composed formulas).

---

## 8. Choosing where to start

For a team whose tasks in one of these domains are "too easy", this is the order of operations. It is a cross-domain synthesis, tagged **Proposal** as a whole; the individual steps carry the evidence cited above.

1. **Audit the verifier and the format.**
   - Add hacking or stress tests, mutant-killing tests, spectests or isomorphic twins as the domain requires.
   - Convert MCQ and bare labels to open answers or certificates.
   - Re-measure pass rates. Part of the apparent saturation can disappear here (SWE-ABS: 78.80% → 62.20%; EvolveCoder: 43.80 → 31.22 on the same problems).
2. **Profile with the current policy.** Use 16–32 rollouts per item in the exact training harness, and bucket by pass rate. This is above the 8–16 default in Chapters 05 and 08 because a narrow band needs the precision: at p = 0.5 the 95% half-width is about ±0.35 at 8 rollouts and ±0.17 at 32 (Chapter 01 §6.1).
3. **Complexify the saturated bucket with the domain's construction-first operator.**
   - Math: chaining and answer-preserving nesting.
   - Formal: subgoal recomposition and aux hiding.
   - Code: solution-first lift or an open-ended goal.
   - SWE: rewrite, reversion or multi-bug injection.
   - Puzzles: a level knob.
   - Science: simulator or solver composition.

   Certify hardness against a reference.
4. **Scaffold the zero bucket.** Audit labels first, then use inverse rewrites, format ladders or hints ([Chapter 05](05-rl-playbook.md)).
5. **Close the loop.** Regenerate each round with the current checkpoint, keep a 0 < p < 1 band (center it near 0.3–0.6), anchor with 20–33% real or oracle-backed items, and monitor response length and a gold-labelled slice.
6. **Only then train a generator** (validity-gated setter RL, anchored self-play). This is where yields rise from about 70% to about 95%. It is also where drift appears ([Chapter 03](03-generation-architectures.md), [Chapter 09](09-pitfalls-and-failure-modes.md)).

Concrete build items that follow from this chapter, such as certificate-first inequality generators, a speed-plus-proved-equivalence reward and a composition curriculum for verified code, are ranked in [Chapter 10](10-idea-bank-and-roadmap.md).
