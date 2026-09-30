# Generation architectures: pipelines that produce hard, verified tasks at scale

> **Key takeaways**
>
> - **One skeleton, nine variants.** Every working pipeline runs seed → operator → validity gate → difficulty gate measured on the target policy → managed pool. The generator is usually the cheapest stage. Validity checks, profiling rollouts and environment builds dominate cost, and they decide whether a task can be learned from. Track **cost per accepted in-band task** rather than tasks generated. This is note 18's worksheet, and [Trading Human Curation](https://arxiv.org/abs/2606.03800) is the one controlled yield study behind it. **Moderate.**
> - **If an executable or formal backbone exists, build the task correct by construction** (fix the answer, solution, state or certificate first; write the text last) **and put its difficulty behind a parametric knob with an online controller.** This is the best-supported way out of zero-advantage GRPO groups: [RLVE](https://arxiv.org/abs/2511.07317), [SCALER](https://arxiv.org/abs/2601.04809), [InternGeometry](https://arxiv.org/abs/2512.10534), [BenchEvolver](https://arxiv.org/abs/2606.01286). **Strong.**
> - **Free-form LLM rewriting and trained proposers pay off only behind an independent validity gate.** A difficulty reward without one gets hacked by invalid or ambiguous tasks ([VHG](https://arxiv.org/abs/2605.06660), [OpenSIR](https://arxiv.org/abs/2511.00602), [SSP](https://arxiv.org/abs/2510.18821)). At a fixed data size, prose-only "make it harder" rewrites often fail to beat the unmodified seeds ([OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)). **Strong.**
> - **Composition of verified atoms is the most dependable verifiable difficulty multiplier.** Success falls roughly as the product of atom success rates, and RL (not SFT) teaches the composed skill, provided the atoms are already mastered ([f(g(x))](https://arxiv.org/abs/2509.25123), [h1](https://arxiv.org/abs/2510.07312)). **Strong.**
> - **For agents, synthesize executable environments with state kept in code or a database.** Use LLM simulators only for surface text and user behavior, steer them explicitly, and never let a trainable component grade ([EnvSimBench](https://arxiv.org/abs/2605.07247), [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597), [SEAD](https://arxiv.org/abs/2602.03548)). **Moderate.**
> - **Run generation as a factory, not a one-shot dataset build.** Route each seed by its pass rate, log yield per operator, defer p̂ = 0 items instead of deleting them, cap injection, retire mastered tasks and re-profile every stage (§11). **Proposal**, assembled from components with Moderate-to-Strong individual evidence.

**Contents**

- [0. The common skeleton and the vocabulary used here](#0-the-common-skeleton-and-the-vocabulary-used-here)
- [1. Parametric procedural generators with difficulty knobs and adaptive controllers](#1-parametric-procedural-generators-with-difficulty-knobs-and-adaptive-controllers)
- [2. Correct-by-construction forward generation](#2-correct-by-construction-forward-generation)
- [3. LLM rewrite/evolve with an independent verification gate](#3-llm-rewriteevolve-with-an-independent-verification-gate)
- [4. Compositional assembly from verified atoms](#4-compositional-assembly-from-verified-atoms)
- [5. Solver-in-the-loop hardening](#5-solver-in-the-loop-hardening)
- [6. Trained task generators](#6-trained-task-generators)
- [7. Failure-mining loops](#7-failure-mining-loops)
- [8. Environment synthesis: executable and LLM-simulated](#8-environment-synthesis-executable-and-llm-simulated)
- [9. Open-ended quality-diversity archives](#9-open-ended-quality-diversity-archives)
- [10. Comparing and combining archetypes](#10-comparing-and-combining-archetypes)
- [11. Reference design: the hardening factory](#11-reference-design-the-hardening-factory)
- [12. Open problems specific to generation architectures](#12-open-problems-specific-to-generation-architectures)

This chapter covers *pipelines*, meaning how the pieces are wired. The individual easy → hard transformations are catalogued in [02-complexification-operator-taxonomy.md](02-complexification-operator-taxonomy.md). How verifiers are built and audited is in [04-verification-and-quality-control.md](04-verification-and-quality-control.md). How to spend rollouts on the resulting pool (allocation, reward shaping, scaffolds for p = 0) is in [05-rl-playbook.md](05-rl-playbook.md). Why the target is a policy-relative pass-rate band is argued in [01-diagnosis-difficulty-and-learning-signal.md](01-diagnosis-difficulty-and-learning-signal.md).

Evidence tags follow Chapter 01: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (a single recent or unreplicated result), **Proposal** (our synthesis, not yet tested).

---

## 0. The common skeleton and the vocabulary used here

```
 seed pool ──► operator / generator ──► validity gate ──► difficulty gate ──► dedup/decontam ──► pool ──► trainer
    ▲              (archetypes 1-9            (is it correct,      (measured on the           │       │
    │               differ mostly here)        solvable, unique,   CURRENT policy:            │       │
    │                                          unhackable?)        0 < p < 1, target band)    │       │
    └──────────── retire / re-profile / mine failures / reseed ◄────────────────────────────────┴───────┘
```

Terms used throughout:

- **p, p̂**: the pass rate of the *target policy* on a task, estimated from k rollouts. "In band" means 0 < p̂ < 1, and usually a narrower target. Pipelines that report a target converge on roughly 0.3–0.6: EvoEnv aims at 0.3, STRETCH at 0.5, and UltraLogic trained best at 40–60% (see [01](01-diagnosis-difficulty-and-learning-signal.md) for why).
- **y_valid, y_band**: the fraction of candidates passing the validity gate, and the fraction of valid candidates landing in band.
- **C_acc**, cost per accepted in-band task, from note 18's worksheet:

  C_acc = (c_gen + c_validate + c_env + k·c_roll) / (y_valid × y_band)

  It needs about 4 candidates per accepted task at 25% yield and about 44 at 2.25%.
- **Reference solution / oracle**: an artifact that provably passes the verifier (a planted answer, an executed program, a proof, a solution script). An archetype's correctness guarantee is set mostly by where the oracle comes from.

**The two gates that recur in every archetype.**

- A **two-sided validity gate**: the task must be *solvable* when given the evidence, the tools or the oracle, and must *fail* through shortcuts (no tools, no data, closed-book, bridge masked, no-op agent). Examples: InfoSeek, OpenSeeker, QwenLong-L1.5, DSGym's no-data filter and Terminal-Bench 2.0's no-op CI check (notes 06, 07, 16, 18).
- A **policy-relative difficulty gate** with k rollouts of the current checkpoint. No archetype below gets by without both.

---

## 1. Parametric procedural generators with difficulty knobs and adaptive controllers

```
 seed family ──lift──► generator G(θ, rng) ──► (instance x, checker V)        θ = size, depth, #constraints,
                           ▲                                                      distractors, hidden info, ...
                           │ θ_next
            controller ◄───┴── per-level accuracy / regret, taken from the GRPO groups already collected
   (sliding window │ proportional control │ regret buffer + mutation │ learned teacher)
```

**What it is.** One program per task family samples instances at a chosen difficulty and ships an exact checker. It usually exploits *solve–verify asymmetry*: the answer is cheap to check even when expensive to find. An online controller moves θ so that each family stays in band. For how to write the operators behind θ, see [02](02-complexification-operator-taxonomy.md).

**When to use.**

- Your seeds have an executable core: arithmetic, algorithms, logic, graphs, SAT/CSP, planning, OR, geometry DSLs, kernels, grid puzzles.
- You can name the dominant variable (length, depth, node count, clause count, operator count).
- This is the default first build for "our RLVR prompts are saturated". If the seeds are fixed problems, *lift* them into generators instead of writing new ones:
  - SCALER extracts scale parameters from contest constraints;
  - EFAGen infers `sample()/solve()/render()` classes;
  - DARG extracts a reasoning graph and perturbs it.

**Exemplar systems.**

| System | Knob(s) | Controller | Correctness | Reported result |
|---|---|---|---|---|
| [RLVE](https://arxiv.org/abs/2511.07317) (400 envs) | Integer level d, e.g. sorting length ≈ 3·1.1^d; Sudoku max(N,M) ≤ d+2 | Window [ℓ, h]; promote when accuracy at h ≥ 0.9 over ≥ 8 × rollouts; keep 4 levels | Algorithmic verifiers; planted solutions; checking by differentiation | +3.37 average over 6 benchmarks from saturated ProRL-1.5B-v2 in about 1,100 H100-h, vs +0.49 in 3,600 h for continuing the original RL |
| [SCALER](https://arxiv.org/abs/2601.04809) | Scale parameters lifted from CodeContests constraints | d_{t+1} = clip(d_t + β(acc_t − τ), 0, D); retire when slope ≤ 0 over 10 steps, accuracy is 0 for 5 steps, or the env sits at max level for 5 | ≥2 independent reference solutions must agree; output diversity at a fixed scale | 4,973 problems → 2,739 envs for about 70M synthesis tokens; Qwen3-4B-Base 54.25 vs RLVE 53.52 and DeepMath-103K 51.08 |
| [Frontier Learning](https://arxiv.org/abs/2609.35426) | Multi-attribute, non-monotone level space | Regret-prioritized buffer with staleness bonus, mutation and exploration | Inherited from the generator | Dice 71.8 vs SEC 33.3; ablation PLR+Explore 33.5 vs PLR+Mutation 62.5 |
| [InternGeometry CBRL](https://arxiv.org/abs/2512.10534) | κ = DDAR proof-step count | Raise κ if mean batch reward > 0.5, else lower it | Engine proves the item with auxiliaries and fails without them | 44/50 IMO geometry with 13K examples; same data unscheduled 38, easy-only 29, hard-only 24 |
| [SATURN](https://arxiv.org/abs/2505.16368) | (n, k, l)-SAT; D(n,k,l) regressed on the LLM's pass@3 (R² ≈ 0.71) | Advance when validation pass@1 > ε | Satisfiable-only instances; the full assignment is checked | Unseen harder levels, pass@3 (7B): 36.1 → 64.2 |
| [TRON](https://arxiv.org/abs/2606.01599) | 10 levels, each changing the *mechanism* | Promote at 0.80 accuracy; replay lower levels with probability 0.30 | Answer computed from latent state before rendering | Base-model check (Qwen3-VL-4B): pass rate 72.8% at level 0 → 41.3% at level 9 |
| [INTELLECT-3 pools](https://arxiv.org/abs/2512.16144) | None (selection only) | Easy/normal/hard pools by solve rate; drop all-pass and all-fail groups | Environment verifiers | Day-1 baseline; saturates once the hard pool is solved |

**Controller menu, cheapest first** (note 05):

1. Solve-rate pools with all-pass/all-fail dropping (INTELLECT-3).
2. A pass@1 gate (SATURN).
3. RLVE's window (τ_acc = 0.9, d_Δ = 4).
4. SCALER's proportional controller with slope/zero/saturation retirement (K = 10/5/5).
5. A regret buffer with mutation, for interacting knobs (Frontier Learning).
6. A learned teacher that places tasks along the generator's axis at p ≈ 0.5, rewarded by min(p, 1−p) minus a signature-repetition penalty ([LURE](https://arxiv.org/abs/2608.21871)).

The effective-prompt ratio (the share of groups whose rewards are not all identical) is the health metric. RLVE showed a static low cap drives it to 0, and even an oracle static range spanning every level the adaptive run reached did worse. **Strong.**

**Cost and yield.**

- The marginal cost per instance is near zero once the generator exists. RLVE contrasts this with DeepMath-103K's roughly $138K and 127K GPU-hours.
- The fixed cost is engineering and auditing the generators. LLM-written generators cut it:
  - [ReSyn](https://arxiv.org/abs/2602.20117): 418 environments.
  - [InternBootcamp](https://arxiv.org/abs/2508.08636): 704 retained from about 100 hand-built seeds.
  - [VHD-Play](https://arxiv.org/abs/2609.27321): 3,300 at about $0.01–0.03 each.

  They need the gates in §8.
- Environment interaction was 16.85% of SCALER's step time.
- Breadth beats depth at fixed instance count. On BBH at about 16K instances, ReSyn scored 75.19 with 400 envs × 40 instances and 71.20 with 25 × 640. SCALER improved monotonically from 8 to 2,739 environments.

**Failure modes.**

- **Size is not reasoning.** Scaling N mostly adds length and bookkeeping. GSM-Infinite's accuracy decays as a sigmoid in operation count. Reasoning Core v3 found grid tasks formed a negative-transfer cluster, largely because of long prompts and answers.
- **Generators silently produce impossible instances.** River Crossing with N ≥ 6 actors and boat capacity 3 is unsolvable ([critique](https://arxiv.org/abs/2506.09250)).
- **Gyms ship with bugs.** An audit found material defects in 13 of 105 Reasoning Gym tasks and 9 SynLogic generators, e.g. a scorer that gives full credit to any nonempty answer ([Reasoning Core v3](https://arxiv.org/abs/2608.05148)).
- **Offline calibration goes stale.** SynLogic's one-shot bounds are an example.
- **Transfer shrinks with scale.** Enigmata saw no general-reasoning transfer at Qwen2.5-32B. K&K-only RL dropped the code average from 67.46 to 56.09.
- **Memorization of the generator family.** K&K fine-tuned models are brittle under one-statement perturbations.
- **Reward shape must change with level.** Partial credit such as (x/N)^10 is needed for long outputs, and binary reward collapsed on Logic Puzzle Baron.

Mitigations are in [09](09-pitfalls-and-failure-modes.md): solver-checked solvability, a monotonicity test of level against solve rate, held-out generator families.

---

## 2. Correct-by-construction forward generation

The shared principle is to **fix the ground truth before the problem text exists**. Difficulty then comes from hiding, deepening or obfuscating around a fixed point, and correctness is inherited. It is not asked of an LLM. Five sub-patterns recur:

```
 (a) answer-first / planted    sample y* ──► build x that y* solves (+ distractors) ──► check by the cheap direction
 (b) execution-backed          write program / SQL / tool chain ──► execute ──► gold = output ──► write NL task LAST
 (c) closure + traceback       random premises ──► engine closure (DAG of facts) ──► pick a deep fact
                               ──► minimal premises = statement, traceback = proof ──► hide auxiliaries
 (d) certificate-first         build witness (assignment, KKT point, SOS decomposition, optimum u*)
                               ──► derive and obfuscate the statement ──► the witness is the verifier
 (e) back-translation gate     formal object ──► LLM renders NL ──► independent parse back ──► equivalence check
```

**Exemplars and numbers.**

- **(a) Planted.**
  - RLVE plants Hamiltonian paths, transforms canonical Sudoku solutions and masks cells, and checks an antiderivative of F′ by differentiating it.
  - [ZebraLogic](https://arxiv.org/abs/2502.01100) samples a solution grid, enumerates every true clue, then SAT-prunes to a minimal unique set.
  - [A²utoLPBench](https://arxiv.org/abs/2607.02141) builds an LP around a KKT-certified optimum. DeepSeek-V4's solve rate fell from 100% on the smallest strata to 8.3% at 40×40.
- **(b) Execution-backed.**
  - [SQL-Zero](https://arxiv.org/abs/2609.04697) writes SQL first, executes it, and admits only deterministic, non-empty SELECTs whose question does not leak the SQL.
  - [AZR](https://arxiv.org/abs/2505.03335) gets deduction, abduction and induction tasks from executed (program, input, output) triplets.
  - [RV-Syn](https://arxiv.org/abs/2504.20426) composes a function library into a graph, executes it, and back-translates. 16% of graphs fail to execute and a further 37% of pairs are dropped because no chain of thought matches the executed answer. The audited error rate is 0.9% for problems and 1.4% for solutions, against 5.5–8.0% solution error for NuminaMath and ScaleQuest. Its problems are harder too: Qwen2.5-Math-7B-Instruct solves 55.6% vs 97.2% for MetaMath.
  - Agentic analogues: [DIVE](https://arxiv.org/abs/2603.11076) executes real tools first and derives tasks the trace entails. [OS-Genesis](https://arxiv.org/abs/2412.19723) explores GUIs first and lifts observed transitions into goals.
- **(c) Closure + traceback.** [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5) sampled nearly 1B random premise sets. That gave more than 100M theorem–proof pairs, 9M of which need auxiliary constructions. Successors scaled the same recipe:

  | System | Output | Compute or notes |
  |---|---|---|
  | [AG2](https://arxiv.org/abs/2502.03544) | about 300M | aux:non-aux rebalanced from 9:91 to 50:50 |
  | [TongGeometry](https://arxiv.org/abs/2412.10673) | 6.7B aux-requiring problems | 10,368 cores for 30 days |
  | [GenesisGeo](https://arxiv.org/abs/2509.21896) | 1M | 28 h on 50 CPU threads (open recipe) |
  | [AIPS](https://arxiv.org/abs/2406.14219) (inequalities) | 191,643 theorems | 8 h on 32 CPUs |
- **(d) Certificate-first.**
  - VHD-Play solves the OR mechanism for u*(θ) and u₀(θ) *before* any environment code exists. Its reward is clip[0,1]((u − u₀)/(u* − u₀)), and admission replays optimal, greedy and idle policies through the generated code.
  - [NSPI](https://arxiv.org/abs/2605.15445) is a prover whose exact SOS certificates can be inverted into a generator (note 03).
- **(e) Back-translation gate.**
  - [SATBench](https://arxiv.org/abs/2505.14615) renders CNF as a story, parses it back to a formula, and discards puzzles that are not solver-equivalent (clause faithfulness 97% on a 100-puzzle audit).
  - [MathCAMPS](https://arxiv.org/abs/2407.00900)'s cycle consistency discarded 25 of 30 unfaithful renderings, and 97.7% of survivors were faithful, at about $0.034 per problem.
  - [Lean Workbook](https://arxiv.org/abs/2406.03847) uses NLI back-translation for autoformalized statements (the funnel keeps 57,231 of 205,079 compiled statements).

**Why it wins.** Solution-first beat problem-first in a head-to-head with the same evolver: [BenchEvolver](https://arxiv.org/abs/2606.01286) reached 97.7% validity against 79.3%, while cutting LCB-v6 Hard pass@1 from 87.0% to 45.7%. In [CHASE](https://arxiv.org/abs/2502.14678)'s direct "generate a hard problem" baseline, 34 of 100 Evol-Instruct-style math problems had errors, and frontier models still solved 82.5–88.9% of them. **Strong.**

**When to use.**

- Whenever a formal object sits under the task: program, SQL, graph, formula, game state, database, proof, geometry diagram, OR model.
- It is also the right design for agent tasks. Fix the gold tool trace or final database state first, then write the user request, as in APIGen-MT, AgentScaler, CoVe and tau2's init/solve/assert triples (note 06).

**Cost and yield.**

- Symbolic-engine generation runs on CPUs (see the table above).
- LLM-assisted execution-backed pipelines pay mainly for rejection: RV-Syn loses about half of its candidates.
- Rendering plus back-translation roughly doubles LLM calls per item, but it removes the dominant error source.

**Failure modes.**

- **The natural distribution skews easy.** AG1's synthetic set was 91% auxiliary-free, and only about 0.05% of proofs were longer than the test-set average. Rebalance deliberately toward the hard subtype, as AG2 did.
- **DSL coverage caps reach.** AG2's language covers 88% of IMO 2000–2024 geometry, up from 66%.
- **Engines have soundness bugs.** [HAGeo](https://arxiv.org/abs/2512.00097) reports one of AlphaGeometry's 25 announced IMO-30 proofs is fallacious.
- **"Hard" may be solvable by a cheap baseline.** In HAGeo, random auxiliary points alone matched AG's 25/30. Certify hardness against a weak reference solver ("must fail without ingredient X").
- **Planted answers are not always unique.** Verify against constraints, or require uniqueness by solver. Inversion is the classic trap: "n! ends in exactly 24 zeros" has five solutions, 100–104 (note 02).
- **The rendering step is unverified** unless back-translated (TrustGeoGen's NL translation is not formally verified). Back-translation can also leak solution structure into the text (RV-Syn).
- **Tedium masquerades as difficulty.** Size-scaled constructions raise op-count, not insight. Code2Math rejects items whose added difficulty is "merely computational tedium".

---

## 3. LLM rewrite/evolve with an independent verification gate

```
 seed (x, y, V) ──► rewriter LLM  [operator prompt; optionally the policy's own correct solution,
      │                            a near-miss summary, or a failure diagnosis]
      │                 │
      │                 ▼
      │           candidate x'   ── y' = y (answer-preserving)  or  y' = new label (label-changing)
      │                 │
      │                 ▼
      │   INDEPENDENT gate: answer-equivalence audit │ execution │ solver/CAS │ cross-family verifier
      │                 │          (never the rewriter grading itself)
      │                 ▼
      │   policy gate: e.g. 4 ≤ passes ≤ (orig − 2) of 16  │  0 < p < 1  │  directional acceptance
      │                 │
      └──── log (operator, reason) ◄── accept │ bounded repair │ reject
```

**What it is.** An LLM applies an operator to a seed: add constraints, nest a sub-problem, invent a term, change the queried quantity, perturb minimally, stack instructions, or remove clues. It works where no generator program exists. The design choice that matters most is *answer-preserving* vs *label-changing*:

- **Answer-preserving rewrites fail safe under GRPO.** A variant whose inherited label is wrong produces an all-incorrect group, so it has zero advantage and makes no update. This is MathForge's argument. A *new* label from majority vote is not safe: it actively rewards wrong answers (note 02). Prefer answer-preserving operators whenever labels are uncertain. **Moderate.**

**Exemplars.**

| System | Operator | Gate | Reported |
|---|---|---|---|
| [MathForge MQR](https://arxiv.org/abs/2601.20614) | Background noise; invented abstract term; a given replaced by a sub-problem (answer unchanged) | o3 equivalence audit (99/97/97% equivalent); nested sub-answers must equal the replaced constant | Qwen2.5-Math-7B MATH: GRPO 37.61 → MathForge 42.17 |
| [SvS](https://arxiv.org/abs/2508.14029) | Policy rewrites problems it solves 12.5–50% of the time, conditioned on its own correct solution, keeping the answer | Band reward: variant solve rate in [12.5%, 62.5%] | +18.3 / +22.8 pass@32 on AIME24/25; entropy stays stable |
| [SynthRL](https://arxiv.org/abs/2506.02096) | Answer-hidden rewriting of visual-math seeds solved ≥12/16 | ≥4/16 still reach the original answer, and ≥2 fewer passes than the seed | 3,380 accepted from 8,072 seeds; 8K-setting average 57.0 → 58.0 |
| [QbQ](https://arxiv.org/abs/2608.01522) | Five structural operators (generalize-then-specialize, parametrize-and-sum, change the queried quantity, inverse, constraint layer) | Variant answer must *differ* from the parent; seeds only from the 8–15/16 band | AIME pass@1 16.46%, vs 11.36% when seeding from hardest failures |
| [IFDecorator](https://arxiv.org/abs/2508.04632) | Add ≤ 3n verifiable constraints at iteration n until p ∈ (0, 0.5] | Script ∧ LLM checklist ∧ IntentCheck | From 21k seeds: 7,324 usable vs 10,772 at pass rate 0 |
| [Code2Math](https://arxiv.org/abs/2603.03202) | Code-exploring agent hides the key insight | Separate solvability and difficulty agents; anti-tedium rubric | 1.56–6.55 failed rollouts per accepted evolution |
| [COVERT](https://arxiv.org/abs/2604.09813) | Distractor tools, indirect queries, noisy/erroneous outputs; oracle call and answer preserved | Reference matching (exact families) or a light judge (tagged families) | BFCL v3 56.5 → 59.9 with RL alone (Qwen2.5-14B) |
| [RLAnything](https://arxiv.org/abs/2602.02488) | Harder rewrite when accuracy > 0.8, guided by near-miss steps | Accept iff 0.2 < acc(q′) < acc(q) | OSWorld in-domain 49.6 → 52.1 from environment adaptation |

**When to use.**

- Seeds that already carry labels but no generator program: competition math, instructions, code problems, tool calls, QA.
- Also for SFT breadth.
- For RL, use it only with a gate that is independent of the rewriter and a band measured on your policy.

**Cost and yield.** Rejection is the dominant cost, and it grows with the difficulty asked for:

- [Loong](https://arxiv.org/abs/2509.03059), GPT-4.1-mini as generator: few-shot variants pass 92.6% (logic). Evol-Instruct code is 55.0% non-executable (logic); in physics, 29.8% of Evol-Instruct variants are judged incorrect and 14.0% are non-executable.
- The one controlled RLVR mutation study ([Trading Human Curation](https://arxiv.org/abs/2606.03800)): 25.5% of 930 attempts passed a five-factor gate, at about $0.05 API cost per accepted variant.
  - 64% of rejections were *too easy*.
  - Yield per mutation axis ranged 7.7–45.5%.
  - Information-removal operators cut solve rate by 70–100 points, and structural coordination by about 60.
- [SAND-Math](https://arxiv.org/abs/2507.20527) kept 8,842 of 23,437.
- SwS's quality filter alone removes 78.35% (note 14).

Log yield *per operator* and move budget toward the operators that produce in-band items. **Moderate** (one controlled study plus consistent funnels).

**Failure modes.**

- **Validity collapses as you push difficulty.** OpenSIR lowered its solve-rate floor from 0.5 to 0.1 (note 09):
  - validity fell from 70.8% to 42.3%;
  - problems got only slightly harder (GPT-5 solve rate 89.8% → 78.3%);
  - math accuracy fell from 29.6 to 26.0.
- **"Harder" usually means "more constraints".** GASP observed that LLMs asked to increase difficulty "frequently do so by adding extra constraints, which is not always the most informative form of complexity" ([GASP](https://arxiv.org/abs/2603.15957)).
- **Prose-only hardening does not pay at fixed size.**
  - OpenThoughts-Agent: LLM rewrites that combine tasks, add constraints or "harden" descriptions did not beat untouched descriptions.
  - LLM-as-a-Tutor: offline Evol-Instruct scored 50.24 vs 50.51 for the unmodified prompts ([LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412)).
  - For terminal tasks, harden the executable solution and environment instead (§5). **Strong.**
- **Answer leakage and gaming.** SvS showed a naive "variant is solvable" reward is gamed by leaking hints or the answer into the variant.
- **Mode collapse.** In EvoTD, Evol-Instruct raised pass@1 but pushed pass@8 below the untuned backbone (54.2 vs 55.1) (note 10).
- **Consensus gates cap difficulty at the verifier's frontier.** CoT-Self-Instruct's Answer-Consistency and SAND-Math's all-k agreement drop hard-but-correct items by design. Route disagreements to relabelling rather than deleting them.
- **Skipping the formal check.** Pythagoras-Prover's mutation data, never kernel-checked, was 87.8% valid on a 2,000-instance spot check ([ALF](https://arxiv.org/abs/2606.12594)).
- **Contamination.** Tulu 3 found 70.7% of HumanEval test items overlapped with Evol-CodeAlpaca (note 01).

---

## 4. Compositional assembly from verified atoms

**Pattern A: typed chaining or DAG composition.**

```
 atom library  {(x_i, y_i, V_i, in_type_i, out_type_i, p_i)}   all verified; policy p_i high (atoms mastered)
      │  select a compatible chain/DAG: out_type_i ⊑ in_type_j via typed adapters φ
      │  (identity, scale, unit conversion, key lookup, precondition→postcondition match)
      ▼
 x_1 ─► y_1 ─φ₁─► x_2 ─► y_2 ─φ₂─► … ─► y_k          gold = executed chain (no new labels needed)
      │  render ONE prompt; hide intermediates; optionally add distractor atoms
      ▼
 gates: each hop checked where it is added │ bridge-masked / closed-book must FAIL │ with-evidence must PASS
```

**Pattern B: KG and web-graph walk variant for search tasks.**

```
 KG / hyperlink graph ──► walk k steps (degree window, add cycles / raise treewidth, disperse sources)
   ──► answer = attribute of end node ──► fuzz / obfuscate clues ──► uniqueness check (enumerate, alt-answer probe)
   ──► no-tool solver must fail ──► agent with retrieval must succeed (≥1 rollout)
```

**Why it works.** Difficulty compounds multiplicatively while verification stays additive:

- **Product law.** Compositional GSM measures the reasoning gap against S1·S2. On MATH² success is roughly the square of MATH success. In Algebrarium, multi-step success tracks the product of atomic success rates (ρ 0.69–0.96) ([Algebrarium](https://arxiv.org/abs/2602.08281)).
- **RL teaches the composed skill; SFT does not.** RL on depth-2 string compositions lifted unseen depth-3 accuracy to about 30%, while RFT on the same data stayed at or below 2.6% ([f(g(x))](https://arxiv.org/abs/2509.25123)).
- **No new labels are needed.** [h1](https://arxiv.org/abs/2510.07312) chains labelled GSM8K problems with deterministic adapters under a horizon curriculum. AIME24 avg@32 went 5.10 → 10.52 and MATH-500 64.20 → 69.20, with no new labels.
- **Strong** across independent works.

**Exemplars by domain.**

- **Math and logic.**
  - [Compositional GSM](https://arxiv.org/abs/2410.01748): Q1's answer becomes a variable of Q2, and the result is executed.
  - [MATH²](https://arxiv.org/abs/2407.21009): skill pairs.
  - RV-Syn: function graphs.
  - [ScaleLogic](https://arxiv.org/abs/2605.06638): RL steps to reach 90% grow as depth^γ, with γ from 1.05 to 2.60, and depth generalization falls to chance at about 3× the training depth.
  - [Ineq-Comp](https://arxiv.org/abs/2505.12680): rule-based duplication and substitution of Lean inequalities.
  - [GAR](https://arxiv.org/abs/2510.11769): LLM fusion of two statements with reward (1−p)(1−m)·1{p≠0}.
- **Multi-hop QA and long context.**
  - [MuSiQue](https://arxiv.org/abs/2108.00573): bottom-up single-hop composition with a disconnection filter. DiRe cheatability (answer) is 37.8, vs 68.8 for HotpotQA.
  - [PhantomWiki](https://arxiv.org/abs/2502.20377): grammar-generated fictional universes, with Prolog computing the answer. RL gave Qwen3-0.6B 56–131% relative F1 gains on five real benchmarks.
  - [QwenLong-L1.5](https://arxiv.org/abs/2512.12967): KG random walks over sparse cross-document evidence, a SQL track for aggregation, a remove-the-source filter and a distractor-robustness filter. 42.7K synthesized, 14.1K kept.
- **Search agents.**
  - [DeepDive](https://arxiv.org/abs/2509.10446): KG walks with k ∈ [5, 9] and an out-degree window [4, 8].
  - [WebSailor-V2](https://arxiv.org/abs/2509.13305): cycles, and WL-distinct subgraphs.
  - [REDSearcher](https://arxiv.org/abs/2602.14234): explicit treewidth and minimum source dispersion targets.
  - [TaskCraft](https://arxiv.org/abs/2506.10055): depth and width extension, verifying only the increment, so cost is roughly linear in hops.
  - [WebShaper](https://arxiv.org/abs/2507.15061): expand only *leaf* constants to avoid shortcuts.
- **Tool use and agents.**
  - [BUTTON](https://arxiv.org/abs/2410.12952): function outputs type-matched to downstream inputs.
  - [APIGen-MT](https://arxiv.org/abs/2504.03601): reverse recombination of validated blueprints, with policy re-checks for conflicts. Phase-1 success was 70% with agentic feedback vs 28% without.
  - [tau2](https://arxiv.org/abs/2506.07982): (init, solve, assert) triples with at most one per mutually exclusive group.
  - [Agent-World](https://arxiv.org/abs/2604.18292) and [AgentScaler](https://arxiv.org/abs/2509.13311): walks on tool graphs executed on a database.
  - [SAP](https://arxiv.org/abs/2609.06124): per-argument provenance tags. Removing them costs 6.4 points.
  - [HardGen](https://arxiv.org/abs/2601.01498): collapses a trace into an "advanced tool" query with implicit intermediate calls.
- **Terminal, GUI and data.**
  - [SkillSynth](https://arxiv.org/abs/2604.25727): precondition → postcondition skill-graph paths of length 1–7. 38% of tasks were unsolved in 3 tries, vs 16% for single-skill tasks.
  - [ChainWorld](https://arxiv.org/abs/2606.21654): directional-compatibility chaining of verified desktop tasks. The best agent completes 31% of chains.
  - [UltraCUA](https://arxiv.org/abs/2510.17790): evaluator-first composition. 29% rollout success vs 45% for instruction-first tasks.
  - [DataMind](https://arxiv.org/abs/2509.25084): chains 2–5 analytic task types.
  - [Terminal-Universe](https://arxiv.org/abs/2609.04148): cross-workspace breadth. Teacher pass@1 72.3% → 49.2%.

**When to use.** Atoms are saturated and individually verifiable, and you want exact labels at higher difficulty. This is the first thing to try for word problems, single-hop QA, single tool calls and single-skill terminal tasks.

**Cost and yield.**

- The cheapest variants (h1, Compositional GSM, tau2 triples) need no LLM labelling at all.
- LLM-rendered compositions cost one rendering plus a check on each new hop.
- [InfoSeek](https://arxiv.org/abs/2509.00375) built 52,138 deep-research samples for $571.8.
- Agentic composition with full execution is dearer: SkillSynth spends $27.3 per verified task.

**Failure modes.**

- **Atoms must already be mastered.** RL on atoms alone does not transfer upward, and RL cannot compose primitives the model lacks (notes 07, 10). Install atoms with SFT or mid-training first; see [06](06-sft-playbook.md).
- **Structural depth is not realized difficulty.** [FORT](https://arxiv.org/abs/2606.12087) measured, on 200 questions per dataset:

  | Dataset | Retrieval calls to solve | Answer first seen at step |
  |---|---|---|
  | InfoSeek | 20.6 | 5.7 |
  | FORT | 141.0 | 46.9 |

  Removing FORT's anti-shortcut controls cumulatively raised a strong agent's accuracy from 29.0% to 81.6%, with fuzzing the most important single control. Accept items on *trajectory signatures*, not graph depth.
- **Obfuscation trades uniqueness for difficulty.** SSP's retrieval-based answerability check still passed non-unique answers. A "fails GPT-4o 4 of 4" filter also keeps broken items (DeepDive). Run an alternative-answer check after every fuzz.
- **Composition without ordering is incoherent.** Graph-ordered skill composition beat random multi-skill composition by 3.8 points on TB2.0 (SkillSynth). Width merges can be shallow concatenation (TaskCraft).
- **Curriculum is required when the top tier gives zero reward.** In h1, uniform mixing and long-only training at equal compute gave no long-horizon gains.
- **Agreement labels bias toward easy compositions** (DataMind, in the authors' own words). Compute the gold with an executable reference pipeline.
- **Length rather than depth.** h1's problems get longer rather than conceptually deeper. Add new operator *types*, not only more steps: ScaleLogic's most expressive logic gave +10.66 on an 8-benchmark average.
- **SFT on composed items transfers narrowly.** SFT on about 8K composed AM-GM problems fixed only the trained operator type (Ineq-Comp). Hold out whole operator families for evaluation.

---

## 5. Solver-in-the-loop hardening

The rule: **harden until the policy (or a frontier probe) fails, while the reference still passes.**

```
 task t₀ = (instruction, env, reference solution, verifier)
 for round r = 1..R:
     probe:  solver(s) S attempt t_r  (k tries; S = current policy │ frontier agent │ strong/weak pair)
     if S passes too often ─► diagnose the shortcut / easy entry ─► harden:
            extend the solution │ add a named difficulty pattern │ fuzz or remove a clue │ add tools/constraints
     if nobody passes      ─► is it solvable? (reference passes in a FRESH sandbox; evidence sufficient; unique)
            ─► repair or narrow, don't just keep it
     co-update solution + environment + verifier + instruction (in that order); re-validate from scratch
 stop when p_S is in the target band, or the budget runs out
```

**What it is.** A closed loop in which an agent edits a *bundle* (task, oracle, verifier, environment) against a live solver signal. It differs from §3 in two ways: the solver's success is the stopping rule, and the oracle and verifier are rewritten in lockstep with the task.

**Exemplars and numbers.**

- **[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556).**
  - An agent builds a database and tools, then proposes a simple task with Python *solution* and *verification* functions. It "iteratively increases the difficulty", updates both functions, and adds tools when needed.
  - The solution may only use the tool interface, so it cannot read the database directly.
  - Kept only if pass@100 > 0: 1,827 environments and 4,417 tasks.
  - On 50 sampled tasks, pass@1 was 12% for V3.2-Exp and 62% for GPT-5-Thinking.
  - RL on these tasks alone improved Tau2Bench, MCP-Mark and MCP-Universe; RL on code and search alone did not.
- **[RST](https://arxiv.org/abs/2608.05466)** (recursive, solution-first, terminal).
  - Each round, extend `solve.sh` first, then align the environment, then extend the verifier, then rewrite the instruction.
  - Gates: a fresh-sandbox oracle, *contract validity* (every checked requirement is stated or discoverable), and minimum deltas (≥3 files, ≥8 solution lines, ≥12 verifier lines).
  - Over 15 rounds from 639 seeds, DeepSeek-V4-Pro pass@4 fell from 90% to 2.5%. Yield stayed at 498–572 accepted per 1,000 attempts, about $0.05 per task, 37,484 tasks in all.
  - The hardness came from work, not prose: median solution lines 67 → 374; instruction words only 85 → 122.
- **[Skill2Env](https://arxiv.org/abs/2609.33772).** If reward exceeds 0.7, diagnose the shortcut and strengthen or add one of 100 named difficulty patterns, each with a control knob. Full-credit rate fell 48.40% → 24.80% → 15.40% over two rounds.
- **[CalibForge](https://arxiv.org/abs/2608.06352).** Author–solver revision until a strong solver passes and a weak one fails.
  - Only 19% of *already validated* candidates showed that pattern on first probe; after revision, 96% did.
  - At 1,300 tasks each on TB2.0: no solver 22.47, contrastive pair 31.09.
- **[Envs-FORGE](https://arxiv.org/abs/2608.14312)** and **[SETA](https://arxiv.org/abs/2607.10891).**
  - Choose the operator from the policy's pass rate: increase / reduce / diversify in Envs-FORGE (chosen by a per-seed MILP); increase / context-shift / decrease in SETA.
  - Envs-FORGE: tb-core 40.0 → 49.2 (Qwen 3.5 35B).
  - SETA's "increase" moved difficulty in the declared direction on only 60% of tasks.
- **Search QA.**
  - [ProgSearch](https://arxiv.org/abs/2510.13913) adds fact branches until a baseline agent fails, with an explicit alternative-answer filter.
  - [SAGE](https://arxiv.org/abs/2601.18202) rewrites until the agent needs at least S search steps and still reaches the gold answer.
  - [WebExplorer](https://arxiv.org/abs/2509.06501)'s long-to-short evolution cut Claude-4-Sonnet from 86.6% to 67.1%.
  - FORT's adversarial refinement repairs shortcut-prone drafts *and* over-fuzzed, unsolved ones.
- **Math and code.**
  - BenchEvolver accepts an evolved solution only if the target-model pass rate beats the seed's.
  - [MathDuels](https://arxiv.org/abs/2604.21916) hardening rounds raised solver error from 9.5% to 25.0% (4-author ablation).
  - [BBEH](https://arxiv.org/abs/2502.19187) iterated each task until both reference models scored below 70%.

**When to use.**

- Agentic tasks with executable bundles (terminal, SWE, tool/DB, office).
- Search QA with evidence graphs.
- Any seed set where you can afford a solver probe per revision.
- It is the natural successor to §2 once a correct-by-construction pool saturates. **Moderate** (several independent works, but few ablations against simpler alternatives).

**Cost and yield.**

- RST's roughly $0.05 per accepted task is the low anchor.
- CalibForge allows up to 50 revision rounds, each with several solver attempts of up to 30 minutes.
- Human-authored long-horizon terminal tasks cost "hundreds to thousands of dollars" each (RST).
- Probing dominates the bill. Use the pilot-then-commit profiling of §11.

**Failure modes.**

- **"Fails" because ambiguous, not hard.** DeepDive's frontier-fail filter keeps broken items unless solvability is checked separately. Every loop needs a "nobody solves → check solvability → repair" branch.
- **Solver-specific artifacts.** BBEH was hardened against specific reference models and will saturate. FORT's signatures depend on the solver. CalibForge is calibrated to fixed solvers, not to the policy, so make the weak solver your current checkpoint.
- **Shared blind spots.** When the same agent authors the solution and the verifier, they share assumptions. SETA's judge of all-fail tasks found about 2% design flaws. RST's contract-validity gate plus its instruction audit cut weakly grounded tasks from 32.8% to 1.2%.
- **Verifier weakening and hackability.** 16% of 1,968 terminal-benchmark tasks were hackable from the description alone ([Hacker–Fixer loop](https://arxiv.org/abs/2606.08960)). Run a hacker–fixer pass after every verifier edit.
- **Drift and homogenization.** Over RST's rounds:
  - within-round nearest-neighbour similarity rose 0.223 → 0.464;
  - lexical distance from human benchmarks grew (unigram JSD 0.358 → 0.433).
  Anchor to real seeds (TerminalWorld recordings) and cap lineages.
- **The binary reward signal vanishes at late rounds.** On T1's hard pool, GRPO was flat at 51.7%. A per-assertion count reward r = P/20 with a warm-started critic reached 64.0% on TB2.1 ([T1](https://arxiv.org/abs/2609.11042)). Make each hardening round *add* verifier assertions, and see [05](05-rl-playbook.md) for estimators.

---

## 6. Trained task generators

```
 proposer π_G ──emits──► task (x, [y*, tests, env, hint])
      ▲                          │
      │                          ▼
      │            INDEPENDENT validity gate V  (SymPy/inverse check │ executor │ Lean │ OR solver │
      │                          │               RAG over proposer's evidence + distractors │ hidden source doc)
      │                          ▼ valid only
      │   p̂ from K solver rollouts ◄── solver π_S (trained on accepted tasks; may share weights)
      │                          │
      └── R_G = 1[valid] · f(p̂)  (+ novelty/repetition terms; invalid → 0, NOT negative)
```

**What it is.** The generator itself is optimized, by RL, DPO, RWR or weighted SFT, to emit tasks that are valid and near the solver's frontier. It comes in four families.

1. **Setter–solver self-play**: [R-Zero](https://arxiv.org/abs/2508.05004), [AZR](https://arxiv.org/abs/2505.03335), [OpenSIR](https://arxiv.org/abs/2511.00602), [VHG](https://arxiv.org/abs/2605.06660), [SPICE](https://arxiv.org/abs/2510.24684), [SSP](https://arxiv.org/abs/2510.18821), [SSR](https://arxiv.org/abs/2512.18552), [SQL-Zero](https://arxiv.org/abs/2609.04697), [OPT-Zero](https://arxiv.org/abs/2609.34205), [GAR](https://arxiv.org/abs/2510.11769), [STP](https://arxiv.org/abs/2502.00212).
2. **Offline-trained generators**:
   - [MathSmith](https://arxiv.org/abs/2508.05592) (GRPO on validity + complexity + consistency);
   - [ScaleQuest](https://arxiv.org/abs/2410.18693) (DPO toward "harder but solvable");
   - [PROPEL](https://arxiv.org/abs/2606.18284) (an activation probe replaces solver rollouts as the reward);
   - [Agentic Proposing](https://arxiv.org/abs/2602.03279);
   - [Learning to Pose](https://arxiv.org/abs/2511.09907);
   - the [Socratic-Zero](https://arxiv.org/abs/2509.24726) Generator.
3. **Goal-anchored teachers**:
   - [GASP](https://arxiv.org/abs/2603.15957): lemma in p ∈ [0.3, 0.7], then lift in [0.1, 0.5], toward pass@100 = 0 goalposts;
   - [SOAR](https://arxiv.org/abs/2601.18778): meta-RL rewarded by the student's gain on fail@128 problems;
   - [CompassPlay](https://arxiv.org/abs/2609.32228): the cosine between the task gradient and the target-task gradient.
4. **Learned environment designers**:
   - [SPADE](https://arxiv.org/abs/2608.19197): hint-based regret, i.e. return with a privileged hint minus return without;
   - [EvoEnv](https://arxiv.org/abs/2605.14392): the policy writes verifiable environment classes;
   - [GenEnv](https://arxiv.org/abs/2512.19682): exp(−β(p̂ − 0.5)²) with RWR;
   - [STRETCH](https://arxiv.org/abs/2609.18642): a 50%-peaked scaffolder reward, where 0.5 beat 0.2 and 0.8;
   - LURE: min(p, 1−p) over generator knobs;
   - [From Trainee to Trainer](https://arxiv.org/abs/2606.17682): the policy writes the next generator configuration;
   - [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969): the model is trained as a task constructor on difficulty and correctness rewards. No formula or ablation is given.

**The design rule that matters: difficulty × independent validity.** **Strong.**

- **Without a validity gate, setters win by emitting invalid problems** (VHG).
- **With the gate, VHG's setter learns validity first, then difficulty.**
  - The valid rate rose 30.6% → 75.5%, and the valid-and-hard share 27.5% → 58.5%.
  - Competition integrals scored 45.4 pass@1, vs 38.8 for vanilla GRPO and 30.5–31.9 for R-Zero.
- **Other working products:**
  - SSR: −1 for an inconsistent bug artifact.
  - OPT-Zero: R_valid × R_correct × R_struct.
  - GAR: (1−p)(1−m)·1{p≠0}.
- **Grounding beats introspection:**
  - R-Zero's pseudo-label accuracy fell 79% → 69% → 63% over three iterations, and every model size eventually degraded.
  - SPICE with corpus grounding scored 43.9 vs 40.7 without.
  - Removing SSP's RAG answerability check dropped GeneralQA from 60.0 to 49.5.
- **The exact shape of f(p̂) is second-order.**
  - SSR's consistency-only ±1 reward was nearly as good as its solve-rate-shaped one.
  - Socratic-Zero's reward-shape variants were within 0.4 points of each other.
  - A 50%-target reward *lowered* AZR's validation accuracy by 2% ([Chae et al.](https://arxiv.org/abs/2510.27072)).
  - OPT-Zero's structural-complexity reward beat a solve-rate reward.

**Yield and cost.**

- *Training the generator is what raises yield.* MathSmith's usable-problem ratio went from 71.5% (SFT generator) to 95.4% (RL with a consistency reward). Socratic-Generator-32B reached 95.6% validity.
- *Untrained proposers have low yield.* Only 5.2% of Self-Challenging Code-as-Task proposals survive. VHG kept 4,076 of 18,663 integral candidates and 20,670 of 46,080 judged general-math pairs.
- *Probes cut the cost of the reward.* PROPEL roughly doubles the rate of frontier tasks with fewer than half the solver trials.
- *A trained generator beats more teacher data.* GenEnv at 1× data beat offline Gemini-2.5-Pro augmentation at 3.3× (0.458 vs 0.438 on BFCL validation).
- *SOAR is the expensive end.* It needs an inner RL run per teacher reward, but it gave about 4× pass@1 on fail@128 MATH splits.
- **Moderate.**

**When to use.**

- Hand-written operators and one-shot rewrites have saturated.
- You have an independent validity check (inverse operation, executor, kernel, solver, hidden grounding document).
- You need the task distribution to track the policy online.
- Without a verifier, prefer §2–§5. Ungrounded self-play peaks within about 1–3 iterations.

**Failure modes and stabilizers.**

- **Death spiral from penalizing invalid output.** A −0.1 penalty in SSP made the valid-question rate collapse toward 0. Give invalid proposals 0 reward, not a negative one.
- **Diversity illusion.** Diversity was enforced within the batch only, and questions differed in wording but not in skill ([R-Diverse](https://arxiv.org/abs/2602.13103)). Use a persistent archive and similarity over solver-code signatures.
- **Drift toward easy.** A single agent self-calibrates toward easy problems. A population with cross-evaluation counters this ([PopuLoRA](https://arxiv.org/abs/2605.16727)).
- **Proposer collapse.** PROPEL's math run collapsed at the lowest KL. Use KL or RWR anchoring.
- **Copying the seed.** CompassPlay's proposer emitted renamed copies of the seed when it could see it. Add a novelty weight.
- **Sharpening rather than expanding.** AZR-style self-play mostly *sharpens* (better pass@k at small k, base model better at large k, though that large-k gap was not statistically significant; Chae et al.), while SvS-style variant synthesis expanded pass@32. Evaluate with pass@k curves.
- **Degenerate challengers.** SSR found challengers that write randomly failing tests, obfuscate the codebase, or turn a single knob ("tunnel vision").
- **A trained grader collapses.** SEAD's trained user simulator forced success rates to about 50% by accepting or rejecting regardless of agent performance. **Never train the component that decides the outcome.**
- **Keep separate weights when labels come from self-consistency.** Pseudo-label accuracy was 71.0% with separate models vs 63.4% with a shared model at step 15 (R-Zero).
- **Stabilizers that worked:**
  - golden replay (STRETCH's formatting collapsed by epoch 4 without it);
  - 1–5% human anchors (R-Few);
  - a capped synthetic fraction (DreamGym);
  - a gold-labelled monitor slice for pseudo-label accuracy.

---

## 7. Failure-mining loops

```
 RL/eval logs ──► per-task history (acc_t, slope, forgetting, boundary), failed trajectories, stuck states
    │
    ├─(a) weakness-driven synthesis: cluster failures → concepts / error patterns / capability-tree nodes
    │        → generate targeted tasks → validity gate → band gate (e.g. 25–75%)
    ├─(b) stuck-state extraction: unsolved goals, failed search nodes, failed repairs → standalone tasks
    │        (+ negations if refutable; + higher-order bugs from failed fixes)
    ├─(c) hindsight relabeling: goal := what the failed trajectory actually achieved → verified task
    └─(d) failure-prefix conditioning: start rollouts from a truncated rare failure (accuracy ≈ 0.5)
```

**What it is.** The policy's own errors become the seed distribution. This places new tasks where the gradient is, rather than where a generator guesses difficulty lies.

**Exemplars and numbers.**

- **(a) Weakness-driven synthesis.**
  - [SwS](https://arxiv.org/abs/2506.08989) defines a weakness as a problem that never reaches 50% and has a negative accuracy slope. It recombines the concepts behind weaknesses, labels them with a stronger model's majority plus ≥25% student agreement, and keeps [25%, 75%]. About 35% land in band. Average gains: +10.0% (7B) and +7.7% (32B).
  - [SENTINEL](https://arxiv.org/abs/2606.12908): a controller summarizes error patterns and a proposer writes executable tau2 tasks. Retail pass^1 66.4 → 74.9.
  - [HardGen](https://arxiv.org/abs/2601.01498) builds its tool graph from 1,204 of 2,095 tools that two small models got wrong. Qwen3-4B BFCLv3 62.13 → 79.14 after SFT+RL.
  - [EvalTree](https://arxiv.org/abs/2503.08893): weakness-guided synthesis gave an accuracy gain 2.5× larger than generic-capability guidance.
  - [Hypothesis-driven errors](https://arxiv.org/abs/2604.04386): the target's solve rate fell to 45% vs 77% on MATH.
- **(b) Stuck-state extraction.**
  - [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613) turns `extract_goal` on failed proof states, plus their negations, into new statements.
  - [Kimina-Prover](https://huggingface.co/blog/AI-MO/kimina-prover) generates sublemmas after 128 failures.
  - [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) composes subgoal proofs for problems it cannot solve end to end.
  - SSR stacks failed repairs into second-order bugs, capped at order 2.
- **(c) Hindsight relabeling.**
  - [Minimo](https://arxiv.org/abs/2407.00695) collapses to easy conjectures without it.
  - [CodeIt](https://arxiv.org/abs/2402.04858) and ARC [SOAR](https://arxiv.org/abs/2507.14172) turn every failed program into a valid task.
  - [AgentHER](https://arxiv.org/abs/2603.21357) relabels failed agent trajectories with 97.1% human-rated precision under two cross-family judges.
- **(d) Failure-prefix conditioning** ([2601.20829](https://arxiv.org/abs/2601.20829)): MATH items solved 121–127 of 128 times, restarted from a truncated rare failure, raised a 5-benchmark average from 40.6 to 44.5. Plain RLVR on the same items gave 40.7, and fresh medium-difficulty problems gave 44.0.
- **Agent-side loops.**
  - [CoEvolve](https://arxiv.org/abs/2604.15840) mines forgetting, boundary and rare signals and re-explores: average gains of +15.58 to +19.43.
  - [SCALECUA](https://arxiv.org/abs/2607.11185) writes harder extensions of solved GUI tasks and simpler variants of failed ones: OSWorld 64.6 → 68.7.
  - [WebRL](https://arxiv.org/abs/2411.02337) evolves tasks from failures and keeps critic scores in [0.05, 0.75].

**Cost and yield.**

- Targeting is nearly free: per-task accuracy histories and failed trajectories come from logs the trainer already writes.
- Weakness-driven synthesis inherits the rejection costs of §3. SwS's quality filter removes 78.35% of candidates, and only about 35% of synthesized problems land in its [25%, 75%] band (note 14).
- Agentic loops cost more per task. SCALECUA, whose pipeline includes the failure-driven augmentation above, spends $0.93–1.01 per accepted GUI task, and about 3K of its 24K+ candidates reach the final RL pool (note 15).
- Failure-prefix conditioning needs no new tasks: it reuses items the policy already solves 121–127 times out of 128.

**When to use.**

- Always, as a *targeting* layer on top of another archetype. It says where to generate, and §1–§6 say how.
- Stuck-state extraction and hindsight relabeling are nearly free wherever an interactive verifier exists: Lean, program execution, environment state.
- **Moderate.**

**Failure modes.**

- **Seeding from the hardest failures underperforms seeding from mostly-solved items.** QbQ: 11.36% vs 16.46%. The best seeds sit just below the band: "almost solved", not "never solved".
- **The generator must be correct exactly where the policy is weak** (EvalTree's stated limitation). Keep an independent label source.
- **SwS's student-agreement rule biases toward partly-solvable items.**
- **Acceptance loopholes.** CoEvolve accepts a task if "execution fails but the environment returns a positive reward". Close such loopholes.
- **Deployment failure replay is not checkpoint failure replay.** DeepSeek-V4.1's replayed failures come from employees and deployment, not from the current checkpoint.
- **Off-policy injected errors do not teach self-correction.** The model often repeats the injected mistake (note 20). Use on-policy failures.

---

## 8. Environment synthesis: executable and LLM-simulated

**Executable synthesis (preferred).**

```
 theme / tool docs / real traces / PRDs
   ─► scenario + task spec ─► state backend (SQLite / files / Docker image) + tools as code + unit tests for tools
   ─► task + reference solution (tool-interface only) + state checker (DB diff / assertions / reward.txt)
   ─► validate: tools unit-tested │ oracle passes in fresh sandbox │ no-op fails │ known-bad fails │
               hacker probe │ weak-vs-strong differential │ builder ≠ runtime account, residue scrubbed
   ─► profile on policy ─► pool
```

**LLM-simulated.**

```
 agent action ─► world model LLM   [prompt: initial state, hidden-state summary from a VERIFIED trajectory,
            ◄── observation         tool code, control directives: errors / pagination / withheld answers]
 state of record: DB or code (safe)  │  LLM memory (breaks on multi-variable updates)
 reward: executable check on state   │  rule-anchored rubric   (never a grader that is being trained)
```

**What it is.** Difficulty moves out of the prompt and into the world: richer state, more tables and near-duplicate records, cross-app coupling, dynamics, hidden information, dual control. The task stays checkable on final state. Domain recipes are in [07b](07b-domain-recipes-agents-and-beyond.md), and verifier construction is in [04](04-verification-and-quality-control.md).

**Executable exemplars and numbers.**

- **Tool and database environments.**
  - [AWM](https://arxiv.org/abs/2602.10090): 1,000 SQL-backed MCP environments with 35,062 tools, averaging 18.5 tables, 35.1 tools and about 1,985 lines of code each.
    - Over 85% of syntheses succeed on the first attempt, with 1.13 repair iterations on average.
    - Reward ablation (8B, BFCLv3): LLM-only 55.46, code-only 60.00, code-augmented judge 65.94.
  - [Agent-World](https://arxiv.org/abs/2604.18292): the four-domain average rose from 18.4% to 38.5% as training environments grew from 0 to 1,978, with smaller gains past 500.
  - [ScaleEnv](https://arxiv.org/abs/2602.06820): zero-shot scores kept rising from 2 to 16 domains at a fixed 1,024 tasks.
  - [EnvFactory](https://arxiv.org/abs/2605.18703): unit-tested stateful tools, plus implicit-reference, action-compression, ambiguity and goal-expansion refinements.
- **LLM-written reasoning environments**, which need staged gates.
  - EvoEnv: 840 environments from 10 seeds, admitted through five mechanical layers (parse, execute, determinism, non-triviality, scorer contract) plus 3 any-reject reviews.
    - Review F1 was 87.0% against a GPT-5.4 audit.
    - Of 79 environments that passed all mechanical layers, GPT-5.4 judged 35 buggy (note 11).
  - InternBootcamp: its generator tried to "simplify" in 97.93% of first-iteration runs, and 54.39% at iteration 2. It keeps a 0.03–0.85 solver pass band.
  - ReSyn: a Wald test requires solve rate to fall with level.
- **GUI and web.**
  - [CUA-Gym](https://arxiv.org/abs/2605.25624): 32,112 tuples, requiring `reward(golden)=1`, `reward(initial)=0`, a checker written behind an information barrier, and a forbidden-pattern scan.
  - [Verified synthetic web environments](https://arxiv.org/abs/2608.21898): raw LLM-built sites averaged 12.4 defects, and only 48.6% of tasks admitted a bounded executable trace. Verify-and-repair raised that to 94.8% (note 15).
  - [AutoWebWorld](https://arxiv.org/abs/2602.14296): finite-state-machine sites at about $0.04 per verified trajectory.
  - [GUI-Genesis](https://arxiv.org/abs/2602.14093): the same policy scores 63.76% by VLM judge and 38.93% by code assertions.
- **Terminal and SWE.**
  - [SWE-Factory](https://arxiv.org/abs/2506.10954): 50.2% valid at $0.047 per processed issue (GPT-4.1-mini); exit-code fail-to-pass has F1 0.99.
  - [SWE-Next](https://arxiv.org/abs/2603.20691): 2.25% yield from commit pairs.
  - [OpenSWE](https://arxiv.org/abs/2603.13023): $19.66 per built environment, about $99 per retained one (derived in note 18).
  - [CLI-Gym](https://arxiv.org/abs/2602.10999): agents break healthy environments and the original tests become the verifier.
  - Packaging: [Harbor](https://docs.harborframework.com/core-concepts/tasks/overview) directories (`instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh` → `/logs/verifier/reward.txt`).
- **Partially observable and adversarial settings.**
  - [KUMO](https://arxiv.org/abs/2504.02810): an SAT engine keeps outcomes consistent with a hidden truth.
  - [AutoEnv](https://arxiv.org/abs/2511.19304): about $4.12 per environment; 65 of 100 themes pass. It discards environments where a weaker model consistently beats a stronger one.
  - [Kimi K3](https://arxiv.org/abs/2607.24653): Autonomous Execution Tasks pair public diagnostic verifiers with hidden scoring verifiers under a submission budget.

**LLM-simulated environments: what they buy and what they cost.**

| Axis | Executable (code/DB state) | LLM-simulated |
|---|---|---|
| Transition correctness | Exact | Collapses when several state variables change at once: the "state-change cliff", present in every frontier model tested ([EnvSimBench](https://arxiv.org/abs/2605.07247)) |
| Realized difficulty | Faithful | Often *easier*: sycophantic outcomes ([WebWorld](https://arxiv.org/abs/2602.14721)); explanatory errors are part of why sim-RL beat real RL on OfficeBench ([Simia](https://arxiv.org/abs/2511.01824)); simulated non-buyers' resistance 25.1% → 13.5% ([Simulated Customers](https://arxiv.org/abs/2606.20708)) |
| Cost and coverage | Build and inspect per environment (ToolHazard: $0.59, 78% of it quality inspection) | [DreamGym](https://arxiv.org/abs/2511.03773): about 1/3–1/5 of real-environment RL; WebArena-Lite 13.3 vs 7.3 for GRPO on 80K real transitions (Llama-3.2-3B) |
| Controllability | Knobs in code | Natural-language directives. Uncontrolled sim RL did nothing (Tool Decathlon 32.4 → 31.5); controlled perturbations gave 36.1 and MCPMark 21.5 → 33.8 ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)) |
| Reward hacking | Sandbox exploits: forged RPC to scheduler sockets, `XFS_IOC_SWAPEXT` ([DSec](https://arxiv.org/abs/2609.22978)) | Judge self-praise (Qwen-AgentWorld); a non-CoT reward model hacked within about 20 steps ([SWE-World](https://arxiv.org/abs/2602.03419)) |

**Rules of thumb.** **Moderate.**

- Keep state transitions in code or a database, and let the LLM render only surface text or user behavior.
- Give the simulator the before-state and the tool code, or a hidden-state summary of a verified trajectory. A 4B constraint-driven simulator beat frontier LLMs on configuration match and cut cost by more than 90% (EnvSimBench).
- Use simulators to *remove shortcuts real environments allow*. Qwen-AgentWorld's answer-withholding snippets raised page-extraction calls from 2.5 to 4.0, while real-search RL lowered them to 1.5. Fictional SQL worlds raised WideSearch item-F1 from 34.02 to 50.31.
- Validate simulated difficulty with real-environment spot checks. Compare against a small state-grounded environment such as [EnterpriseBench CoreCraft](https://arxiv.org/abs/2602.16179) at equal compute.

**When to use.**

- Agentic seeds (tool calls, GUI, terminal, SWE, office) whose difficulty should come from state, tools and dynamics rather than from prompt text.
- Saturated single-call or single-app tasks: enrich the world (more tables, near-duplicate records, cross-app coupling, hidden information) while the check stays on final state.
- LLM simulation only for surface text, user behavior, or perturbations real deployments rarely produce (errors, pagination, withheld answers), and always behind a state-grounded check.

**Cost and yield.**

- LLM-built environments are cheap per unit: about $0.04 per verified trajectory (AutoWebWorld), $0.047 per processed issue at 50.2% validity (SWE-Factory, GPT-4.1-mini), and about $4.12 per environment with 65 of 100 themes passing (AutoEnv).
- SWE environments mined from real repositories are expensive or low-yield: OpenSWE costs $19.66 per built and about $99 per retained environment (derived, note 18), and SWE-Next keeps 2.25% of commit pairs.
- Validation dominates the bill: 78% of ToolHazard's $0.59 per environment is quality inspection (note 18).
- Simulation is cheaper still, at about 1/3–1/5 of real-environment RL (DreamGym), but see the fidelity rows above.

**Failure modes (executable).**

- Environment bugs dominate: on raw LLM-built websites about half the tasks were infeasible before repair (48.6% feasible).
- Checkers are too lenient (VLM judges over-accept) or too strict (scripts reject valid alternative paths).
- Answer leakage through build residue or package mirrors (DSec, note 17) and through git history (note 12).
- Shortcut channels: about 15% of OSWorld tasks need only a terminal (note 15).
- Mock-tool fidelity is rarely measured.

---

## 9. Open-ended quality-diversity archives

```
 archive A: cells = descriptor tuples (skill-combination × format × length × topic × ...)
            each cell: elite task(s) + fitness + stats
 loop:
   target ← empty / weak / under-covered cell      parents ← elites of nearby cells
   child  ← LLM mutation toward target (harder │ easier │ novel)
   gate: validity (execution / solver / judge with position swaps) ─► fitness = 1 − p_policy (or learnability)
   novelty filter vs archive (embedding / BLEU / solution signature) ─► replace incumbent iff fitter
 training sampler draws ACROSS cells (coverage), not only from the fittest
```

**What it is.** A standing, descriptor-indexed population of tasks, grown by mutating elites toward under-covered regions. It targets the failure every other archetype eventually hits: collapse onto a few templates that are hard for easy reasons.

**Exemplars and numbers.**

- [ACES](https://arxiv.org/abs/2310.10692): MAP-Elites over 20-skill combinations (21,700 niches) of Python puzzles. Fitness is the negative solver success rate; puzzles never solved in 50 attempts are discarded. On average across 11 code LLMs, the puzzles were three times more challenging than existing benchmarks.
- [Rainbow Teaming](https://arxiv.org/abs/2402.16822): a 10 × 10 risk × style grid with targeted mutation, a BLEU similarity filter and pairwise judging with position swaps. [QDAIF](https://arxiv.org/abs/2310.13032) generalizes the idea to quality and diversity axes defined by AI feedback.
- [ACD](https://arxiv.org/abs/2502.07577): tasks as code with `score()`. 1,330 "interestingly new" tasks in 25 clusters came from 5,000 generations, and about 20% of proposals were still novel after 5,000. Judge–human F1 was 0.86 but dropped on "Very Difficult" tasks.
- [OMNI-EPIC](https://arxiv.org/abs/2405.15568): an archive of learned and failed tasks. Failures route to simpler variants and successes to elaborations. Human agreement with the success detector was 72.7%.
- [AC/DC](https://arxiv.org/abs/2604.14969): tasks co-evolve with merged models under a minimal-criterion filter; 97.8% of tasks were human-rated correct.
- Regret-driven editing:
  - [ACCEL](https://arxiv.org/abs/2203.01302) reached POET-comparable complexity with under 0.05% of [POET](https://arxiv.org/abs/1901.01753)'s environment samples.
  - Frontier Learning (§1): mutation of high-regret levels was the key ingredient.
  - [COvolve](https://arxiv.org/abs/2603.28386) mutates tasks against the *equilibrium mixture* of past policies (small code domains), so the generator does not chase only the latest one.
  - [eva](https://arxiv.org/abs/2411.00062): advantage-weighted evolution reached 60.0 on Arena-Hard vs 57.5 for uniform, but variance-based selection (54.8) did *worse* than uniform.

**Cost and yield.**

- The archive adds novelty checks and cell bookkeeping on top of the base generator's validity gate; fitness still needs policy or solver rollouts per child (ACES spends 50 solver attempts on every new puzzle).
- Yield of genuinely new tasks is modest: ACD kept 1,330 "interestingly new" tasks from 5,000 generations (about 27%), and about 20% of proposals were still novel at the end.
- Regret-driven editing can be very sample-efficient: ACCEL matched POET's complexity with under 0.05% of POET's environment samples.

**When to use.**

- Long-running generation where diversity collapse is the main risk.
- Multi-skill coverage requirements.
- Red-teaming.
- As the pool data structure in the factory of §11: the "archive" *is* the managed pool.
- For LLM RL training specifically, evidence is thin, since most QD work above is evaluation-oriented. **Emerging.**
- Diversity itself is first-order:
  - Self-Challenging: 200 training tasks slightly degraded test performance, while 800 gave steady gains.
  - R-Diverse's skill-level similarity sustained self-play gains over more iterations.
  - **Moderate.**

**Failure modes.**

- Descriptor grids are hand-designed, and LLM-assigned skill labels are noisy.
- Judge fitness can be gamed.
- Discarding never-solved items caps difficulty at the solver's reach (ACES).
- Novelty filters saturate.
- Interestingness judgments are least reliable exactly on the hardest tasks (ACD).
- Checkers visible to the proposer invite objective hacking ([Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)).

---

## 10. Comparing and combining archetypes

| Archetype | Correctness guarantee | Scalability | Diversity | Cost profile | Best domain fit | Dominant failure |
|---|---|---|---|---|---|---|
| 1 Procedural + controller | Exact checker by construction, *if* the generator is audited | Unbounded instances; limited by number of families | Low within a family; rises with family count | High fixed engineering, ~0 marginal; LLM-written envs $0.01–0.03 (VHD-Play) | Algorithms, logic, SAT/CSP, OR, arithmetic, kernels | Size ≠ reasoning; generator bugs; family memorization |
| 2 Correct-by-construction | Exact up to engine soundness and rendering faithfulness | Very high, CPU-bound (1M geometry problems in 28 h on 50 threads) | Bounded by DSL; natural distribution skews easy | Low; plus back-translation calls | Anything with a formal backbone: geometry, inequalities, SQL, code, OR, DB-backed agents | Easy skew; DSL coverage; unverified NL |
| 3 Rewrite + gate | As strong as the gate: answer-preserving ≈ inherited label; new labels ≈ gate quality | High, but yields around 25–50% | Medium; mode-collapse risk | About $0.05 API per accepted variant, plus rejection | Labelled seeds without a generator: math, IF, code, QA, tool calls | Invalidity grows with difficulty; "add constraints"; leakage |
| 4 Composition | Exact if atoms are verified and adapters deterministic | High, combinatorial | High in combinations, but can be concatenative | Near zero (h1) to $27.3 per task (SkillSynth) | Word problems, multi-hop QA, tool chains, terminal skills, Lean | Shortcuts; non-uniqueness; unmastered atoms |
| 5 Solver-in-the-loop | Oracle re-verified each round; shared-author blind spots | Medium: one probe per revision | Drifts and homogenizes without caps | $0.05 per task (RST) up to 50 probe rounds (CalibForge) | Terminal, SWE, tool/DB bundles, search QA | Ambiguity mistaken for hardness; solver artifacts; hackable verifiers |
| 6 Trained generator | Only as strong as the independent gate; pseudo-labels decay | High once trained; tracks the policy | Collapse-prone without archive or population | Generator training + K solver rollouts per candidate (a probe halves it) | Domains with inverse checks, executors, kernels, grounding docs | Invalid-but-hard hacking; diversity illusion; label drift |
| 7 Failure mining | Inherits the base generator; relabeling is exact | Bounded by failure volume | Narrow by design | Low marginal (logs exist) | All; nearly free in Lean, code, env-state domains | Wrong labels exactly where the model is weak; loopholes |
| 8a Executable envs | Exact on state, once the environment is verified | Medium: build + sandbox minutes | High with many environments | $0.04/trajectory (FSM web) to $19.66/env (OpenSWE) + sandboxes | Tool use, GUI, terminal, SWE, office | Env bugs; answer leakage; sandbox exploits |
| 8b LLM-simulated envs | Weak; state-change cliff | Very high | High surface diversity | About 1/3–1/5 of real-env RL | User behavior, surface perturbation, SFT breadth | Easier than reality; hacked graders |
| 9 QD archive | Inherits the gate | Medium | Highest by design | Archive plus novelty checks | Long runs, multi-skill coverage, red-teaming | Descriptor design; judge gaming |

**Composition of archetypes (Proposal).** In practice the archetypes stack rather than compete:

```
 formal/executable core?  ──yes──► §2 construct ─► §1 knob + controller ─► saturated? ─► §4 compose atoms ─► §5 harden bundle
        │ no
        ├── labelled seeds?  ──► §3 answer-preserving rewrite + independent gate + policy band
        └── agentic task?    ──► §8a executable env (state in code/DB) ─► §4 / §5 inside it; §8b only to render
 always on top:  §7 decides WHERE to generate · §9 archive keeps coverage · §6 once operators are exhausted AND a gate exists
```

Frontier labs combine several of these (note 12, [08](08-frontier-lab-practices.md)). Examples: DeepSeek-V3.2 (§5 + §8a), GLM-5 and MiMo-V2-Flash (§4 knowledge- or fact-graph composition, with detail obfuscation in MiMo), and DeepSeek-V4.1-Flash (§6 constructor + §7 failure replay + §8a).

---

## 11. Reference design: the hardening factory

A **Proposal** that assembles components with Moderate-to-Strong individual evidence into one loop. Most defaults below are copied from a cited system; the few that are ours are marked "tune" in the code. Tune all of them on your own pilot.

```
 seed pool  (each task packaged as: prompt, env, oracle, verifier, known-bad, lineage, p̂-posterior)
   │
   ▼
 0 intake + verifier audit ──► 1 profile & route by p̂ ──┬── p̂ ≈ 1  ──► 2 choose operator ──► 3 generate
                                                       ├── in band ─► trainer                     │
                                                       └── p̂ ≈ 0  ──► scaffold (see 05) / defer    ▼
 7 pool management ◄── 6 dedup / decontam ◄── 5 difficulty calibration ◄──────────── 4 validity cascade
   (caps, retire, recycle,                      (pilot → commit)                          │
    re-profile, anchors, re-audit)                                                        │
   │                                   reject reasons + yield per operator ───────────────┘──► feeds step 2
   └──► back to seed pool (accepted children become next-round seeds, under lineage caps)
```

| Stage | What it does | Defaults (source) |
|---|---|---|
| 0 Intake and verifier audit | Package every seed as (prompt, env, oracle, verifier, known-bad solutions). Require oracle passes, no-op fails, known-bad fails. Add special judges for multi-answer items | 14.57% of code problems need special judges, and exact match rejects 59.01% of correct solutions to them ([ScaleBox](https://arxiv.org/abs/2604.27467)); Code-as-Task; Terminal-Bench 2.0 CI |
| 1 Profile and route | Pilot k = 8–16. Complexify above the band, train inside it, scaffold or defer below it | [LILO](https://arxiv.org/abs/2502.12272) p(1−p); [Bae et al.](https://arxiv.org/abs/2504.03380) balanced band |
| 2 Choose operator | Band-conditioned (increase / context-shift / decrease), plus a bandit over each operator's measured y_valid × y_band × Δp. Prior p̂ from parent seed + operator's mean drop | SETA; Envs-FORGE; per-axis yield 7.7–45.5% ([2606.03800](https://arxiv.org/abs/2606.03800)); [BOTS](https://arxiv.org/abs/2510.26374) |
| 3 Generate | Pick the generator per operator by C_acc on a pilot; prefer many cheap samples plus a strict verifier | [AgoraBench](https://arxiv.org/abs/2412.03679): 50K GPT-4o-mini beat 10K GPT-4o at 3.4× lower cost; solving skill does not predict generating skill (R² < 0.1) |
| 4 Validity cascade | Cheapest check first, abort on the first failure: schema → partial-solution checks → oracle in a fresh sandbox → shortcut probes (no-op / no-data / no-tool / closed-book) → uniqueness → hacker probe → contract validity | [MSIFR](https://arxiv.org/abs/2605.14062) (11–77% fewer tokens); RST; CUA-Gym; [DSGym](https://arxiv.org/abs/2601.16344) no-data filter |
| 5 Difficulty calibration | Pilot 16 rollouts: skip if p̂ > 0.75, defer if < 0.125, else commit 48. Directional acceptance. Hardness certificate against a weak reference solver | [Pilot-Commit](https://arxiv.org/abs/2605.26606); RLAnything; InternGeometry |
| 6 Dedup / decontam | Dedup on *solution signatures* (canonical solver code, masked SQL template, embedding ≥ 0.95). 13-gram overlap vs evals; embedding + LLM-judge decontam; web-search novelty | R-Diverse; SQL-Zero; GASP; RST; [DeepMath-103K](https://arxiv.org/abs/2504.11456) (90% of AIME24 items found in its raw pool); SAND-Math (0.85) |
| 7 Pool management | Cap new tasks at 20% of the pool per epoch; caps per lineage, category and operator; retire on slope ≤ 0, saturation or long zero; park 0/k items in a monitor pool; recycle rather than delete; re-profile every stage; anchor 20–33% real items + 2–10% easy replay; re-audit on each RL run | [RODS](https://arxiv.org/abs/2606.19047); RST caps (≤4/parent); SCALER K = 10/5/5; [recycling](https://arxiv.org/abs/2606.10709); note 19; DeepSeek-V4.1 |

```python
# Hardening factory: one generation cycle between RL stages (or run asynchronously beside the trainer).
BAND = (0.125, 0.75)          # pilot skip/defer thresholds (Pilot-Commit); train target ≈ 0.3-0.6
CAPS = dict(per_parent=4, per_category=160, per_operator=320)   # RST-style lineage caps
INJECT_FRAC = 0.20            # max share of the pool that is new per epoch (RODS)
K_STALE = 2                   # re-profile tasks whose p̂ was measured > K_STALE policy versions ago (tune)

def cycle(pool, policy, ver, ops, gen_for, budget):
    stats = pool.operator_stats                      # y_valid, y_band, mean Δp, reject reasons per operator
    seeds = [t for t in pool.active if t.p.mean() > 0.9 and not t.retired]   # near-saturated seeds (cf. RLVE's 0.9; tune)
    for seed in sample_under_caps(seeds, CAPS):
        op = ops.select(seed, stats, rule="band")    # increase if p>0.5; bandit over yield × gain
        for cand in gen_for(op).propose(seed, n=op.fanout):
            ok, why = validity_cascade(cand)        # cheapest first; returns first failing check
            stats[op].log("valid", ok, why)
            if not ok:
                continue
            cand.p = prior(seed.p, op.mean_drop)     # parent-seed prior (BOTS-style), then pilot
            p_hat = mean(policy.rollout(cand, n=16))
            if p_hat > BAND[1]: stats[op].log("too_easy"); continue      # optionally re-harden
            if p_hat < BAND[0]: pool.defer(cand, until=ver + 1); continue  # never delete on one pilot
            if op.direction == "harder" and p_hat >= seed.p.mean():         # directional acceptance
                continue
            if pool.near_duplicate(solution_signature(cand)) or contaminated(cand):
                continue
            pool.admit(cand, profiled_at=ver, lineage=seed.id, cap=INJECT_FRAC)
            stats[op].log("in_band", True)
    pool.retire(lambda t: t.slope(10) <= 0 or t.saturated(5) or t.zero_for(5), to="monitor")
    pool.repilot(pool.deferred + pool.monitor, policy)       # ~20% flip back to signal (recycling)
    pool.reprofile(older_than=ver - K_STALE)                  # labels go stale as the policy moves
    pool.hacker_fixer(new_since=ver)                          # re-harden verifiers edited this cycle

def validity_cascade(t):
    for check in (schema_ok, partial_checks, oracle_passes_fresh_sandbox,
                  noop_fails, known_bad_fails, no_tool_or_no_data_fails,
                  unique_or_certificate_ok, contract_valid, hack_probe_fails):
        if not check(t):
            return False, check.__name__
    return True, None
```

**Monitoring.** Watch these on every cycle:

- **Effective-prompt ratio** (RLVE).
- **Refill batches per step.** In verl, the inverse of `num_gen_batches` is the live in-band yield, and running into its cap raises an error.
- **C_acc and y_valid × y_band per operator.** 64% of mutation rejections in the controlled study were "too easy".
- **Pass@k at large k and policy entropy.** These separate expansion from sharpening.
- **Pseudo-label accuracy on a gold slice**, if any labels are self-generated.
- **Nearest-neighbour similarity and descriptor coverage.** RST's rose from 0.223 to 0.464.
- **Valid-proposal rate** for any trained proposer (the SSP death spiral).
- **Turn and length drift under count rewards** (T1).

**Budget arithmetic** (note 18; derived, so substitute your own numbers):

- **Candidates per accepted task** = 1/(y_valid·y_band): about 4 at 25% and about 19 at 5.2%.
- **Profiling can rival generation.** Profiling one candidate at k = 8 × 8K tokens costs about $0.006–0.011 on a 7B policy and $0.03–0.06 on a 32B policy. At y_band = 0.25 that becomes $0.12–0.24 per accepted task on 32B. In the worked example, profiling is about 36% of all rollout tokens.
- **Size the unique pool.** Reuse up to about 25× showed no significant degradation ([Tan et al.](https://arxiv.org/abs/2509.25300)), so S steps × B prompts need about S·B/25 unique verified tasks.
- **Reallocate rollouts before buying tasks.** [Knapsack RL](https://arxiv.org/abs/2509.25849) is worth about 2× compute; Pilot-Commit needs up to 1.9× fewer rollouts than GRPO. Buy new tasks when the pool's in-band fraction drops ([05](05-rl-playbook.md)).

**Framework hooks.** No major open RL framework regenerates or re-levels tasks online. They do provide the hooks:

- slime's `--custom-generate-function-path`, where RLVE runs;
- verl's `filter_groups`;
- NeMo Gym resource servers;
- the verifiers/Environments Hub.

Build the factory as a service beside the trainer, and version-tag each task with the policy version that profiled it. Groups that survive asymmetric filtering are systematically older than the buffer average (note 18).

---

## 12. Open problems specific to generation architectures

- **Learning which operator to apply to which seed at which step.** Envs-FORGE solves a per-seed MILP over 6 hand-defined actions, and SSP and SPADE train proposers end to end. No method learns operator choice from measured learning progress without collapse.
- **Validity certification outside executable domains.** Proofs, open-ended research and long-horizon agents still rely on judges that fail most on the hardest items (ACD, UQ). No calibrated "valid / ambiguous / unsolvable" classifier exists that is independent of the difficulty signal.
- **Realism under recursion.** Escalated tasks drift lexically and homogenize (RST). Whether late-round long-pipeline tasks transfer as well as realistic hard tasks is untested.
- **Simulator fidelity standards.** EnvSimBench, WebWorld-Bench and AgentWorldBench now measure simulators intrinsically. There is still no standard test that *simulated* hardness matches *real* hardness.
- **The compute-optimal split between new tasks and more rollouts.** IsoCompute allocates across rollouts, problems and steps. Nothing yet says when the next dollar should buy fresh hard tasks instead (note 18).
- **Operator ablations under RLVR at matched compute.** Most operator comparisons are SFT-based. A GRPO study on identical seeds comparing composition, inversion, hiding, perturbation and environment enrichment, measured on held-out generators and pass@k, would settle budget allocation (notes 02, 14). Candidate experiments are in [10](10-idea-bank-and-roadmap.md).
