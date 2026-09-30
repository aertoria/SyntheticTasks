# Domain recipes I: math, formal proofs & geometry, code, software engineering, logic puzzles, science

> **Key takeaways**
>
> - **Build the answer or certificate first, then write the task.** Every sound generator in these six domains starts from something that can be checked: an executed computation graph, a deduction closure, a planted solution, a simulator trace, a reference program or a proved lemma. The prompt text is derived from it afterwards. Answers written by an LLM are the weakest label tier. Keep them out of RL rewards unless an independent check backs them. **Strong**
> - **Harden the verifier and the answer format before you generate new tasks.** Weak tests and guessable formats make tasks look easy when they are not.
>   - Strengthening SWE-bench Verified tests cut the top agent from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)).
>   - Evolved tests cut pass@1 on the same problems from 43.80 to 31.22 ([EvolveCoder](https://arxiv.org/abs/2603.12698)).
>   - Bare SAT/UNSAT labels invite "satisfiability bias", which caused 68.7% of o4-mini's errors ([SATBench](https://arxiv.org/abs/2505.14615)).
>
>   **Strong**
> - **Composition is the most reliable hardening knob that stays verifiable in every domain.** Each atom keeps its own checker. Examples: chained GSM8K problems ([h1](https://arxiv.org/abs/2510.07312)), fused Lean statements ([GAR](https://arxiv.org/abs/2510.11769)), chained Dafny functions ([DafnyComp](https://arxiv.org/abs/2509.23061)), fused GPU operators ([DRTriton](https://arxiv.org/abs/2603.21465)), multiple bugs or deeper feature removal in repositories, and f∘g string functions ([f(g(x))](https://arxiv.org/abs/2509.25123)). Train on composed tasks with RL and a curriculum. SFT and uniform mixing mostly fail to transfer composition. **Strong**
> - **Certify hardness against a reference, then calibrate it to the current policy.** The reference can be a cheap solver that fails without the key ingredient, a brute-force solver that agrees with the reference solution, or a stronger teacher that solves what your previous model could not. Keep a pass-rate band and re-measure it every round. LLM ratings of problem quality are close to useless: o3's agreement with humans was 0.07 ([AutoCode](https://arxiv.org/abs/2510.12803)). **Strong**
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

- **Structure beats paraphrase.** Adding 20K MetaMath samples on top of 80K gave +0.1 GSM8K points from more answer augmentation, +0.4 from rephrasing, +2.3 from FOBAR inversion and +2.6 from self-verification inversion ([MetaMath](https://arxiv.org/abs/2309.12284), 2023). FLAMES found that agents designed to increase complexity gave the best improvements on most math metrics. **Strong.**
- **Chaining needs a horizon curriculum.** h1 chains GSM8K atoms and trains with Dr. GRPO over horizons up to length 5. AIME24 avg@32 rose from 5.10 to 10.52, MATH-500 reached 69.20 (+7.8%) and GSM-Symbolic P2 rose from 43.08 to 52.00. Uniform mixing and long-only training at equal compute gave no long-horizon gain ([h1](https://arxiv.org/abs/2510.07312), 2025). Composed difficulty is roughly multiplicative. Compositional GSM measures the gap as Δ = S_comp − S1·S2 and finds that Qwen2.5-Math-7B-IT solves more than 80% of MATH but under 60% of compositional grade-school problems. On MATH², success rate is roughly the square of the MATH rate ([MATH²](https://arxiv.org/abs/2407.21009)).
- **Answer-preserving edits fail safe under GRPO.** If a rewrite silently breaks the inherited label, the group scores all zeros and produces no update. MathForge's o3 audit found 99/97/97% of Background/Term/Sub-Problem rewrites answer-equivalent. On Qwen2.5-Math-7B trained on MATH, the reported average rose from 37.61 (GRPO) to 41.04 (MQR data) and 42.17 (MQR + DGPO) ([MathForge](https://arxiv.org/abs/2601.20614), ICLR 2026). A wrong *new* label from majority voting does not fail safe: it rewards wrong answers.
- **Construction makes items both harder and cleaner.** RV-Syn problems were solved only 55.6% of the time by Qwen2.5-Math-7B-Instruct, against 70.6–97.2% on six other datasets. Audited error rates were 0.9% (problem) and 1.4% (solution), against 4.9–8.0% solution error for MathScale, NuminaMath and ScaleQuest ([RV-Syn](https://arxiv.org/abs/2504.20426)).
- **Easy seeds hide hard variants.** EFAGen resampled 50 variants from MATH seeds that GPT-4o had solved and found variants GPT-4o fails *even for Level-1 seeds* ([EFAGen](https://arxiv.org/abs/2504.09763)).
- **Inversion rescues unreachable items.** Converting hard@8 items (pass@8 = 0) into up to 2m inverse variants each ("AugHard@8") recovered AIME24 from 3.33 to 13.33 in the intervention table ([T-SAE study](https://arxiv.org/abs/2605.28388)).

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
- **Some hard items damage training.** Training only on hard@8 items lowered averages by 5.75, 11.24 and 1.07 points in three settings. A single sample whose reward accepted a bare `\boxed{}` collapsed mean response length from 510.7 to 45.7 tokens within 58 steps, before accuracy fell ([T-SAE study](https://arxiv.org/abs/2605.28388)). Watch response length as an early alarm.
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

The ablation on IMO-50 geometry, all at about 13K training examples:

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

