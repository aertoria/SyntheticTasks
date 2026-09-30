# Generation architectures: pipelines that produce hard, verified tasks at scale

> **Key takeaways**
>
> - **One skeleton, nine variants.** Every working pipeline runs seed → operator → validity gate → difficulty gate measured on the target policy → managed pool. The generator is usually the cheapest stage. Validity checks, profiling rollouts and environment builds dominate cost, and they decide whether a task can be learned from. Track **cost per accepted in-band task** rather than tasks generated. This is note 18's worksheet, and [Trading Human Curation](https://arxiv.org/abs/2606.03800) is the one controlled yield study behind it. **Moderate.**
> - **If an executable or formal backbone exists, build the task correct by construction** (fix the answer, solution, state or certificate first; write the text last) **and put its difficulty behind a parametric knob with an online controller.** This is the best-supported way out of zero-advantage GRPO groups: [RLVE](https://arxiv.org/abs/2511.07317), [SCALER](https://arxiv.org/abs/2601.04809), [InternGeometry](https://arxiv.org/abs/2512.10534), [BenchEvolver](https://arxiv.org/abs/2606.01286). **Strong.**
> - **Free-form LLM rewriting and trained proposers pay off only behind an independent validity gate.** A difficulty reward without one gets hacked by invalid or ambiguous tasks ([VHG](https://arxiv.org/abs/2605.06660), [OpenSIR](https://arxiv.org/abs/2511.00602), [SSP](https://arxiv.org/abs/2510.18821)). At a fixed data size, prose-only "make it harder" rewrites often fail to beat the unmodified seeds ([OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)). **Strong.**
> - **Composition of verified atoms is the most dependable verifiable difficulty multiplier.** Success falls roughly as the product of atom success rates, and RL (not SFT) teaches the composed skill, provided the atoms are already mastered ([f(g(x))](https://arxiv.org/abs/2509.25123), [h1](https://arxiv.org/abs/2510.07312)). **Strong.**
> - **For agents, synthesize executable environments with state kept in code or a database.** Use LLM simulators only for surface text and user behavior, steer them explicitly, and never let a trainable component grade ([EnvSimBench](https://arxiv.org/abs/2605.07247), [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597), [SEAD](https://arxiv.org/abs/2602.03548)). **Moderate.**
> - **Run generation as a factory, not a one-shot dataset build.** Route each seed by its pass rate, log yield per operator, defer p̂ = 0 items instead of deleting them, cap injection, retire mastered tasks and re-profile every stage (§11). **Proposal**, assembled from Moderate-evidence components.

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
| [TRON](https://arxiv.org/abs/2606.01599) | 10 levels, each changing the *mechanism* | Promote at 0.80 accuracy; replay lower levels with probability 0.30 | Answer computed from latent state before rendering | Angle-chase base pass rate 72.8% at level 0 → 41.3% at level 9 |
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
  - [A²utoLPBench](https://arxiv.org/abs/2607.02141) builds an LP around a KKT-certified optimum. Solve rate fell from 100% to 8.3% as the LP grew to 40×40.
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

**Why it wins.** Solution-first beat problem-first in a head-to-head with the same evolver: [BenchEvolver](https://arxiv.org/abs/2606.01286) reached 97.7% validity against 79.3%, while cutting LCB-v6 Hard pass@1 from 87.0% to 45.7%. [CHASE](https://arxiv.org/abs/2502.14678)'s direct "generate a hard problem" baseline had 34 errors per 100 math problems, and frontier models still solved 82.5–88.9% of them. **Strong.**

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
