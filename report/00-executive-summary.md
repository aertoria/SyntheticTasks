# Executive summary

*Synthesizing hard training tasks from easy ones for LLM SFT and RL, as of 2026-09-30. Written for a post-training team whose tasks are saturated: pass rates near 100% and no GRPO signal.*

Evidence tags: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (one recent or unreplicated result), **Proposal** (our untested synthesis). Numbers are the cited papers' own and hold only under their model, benchmark and metric.

**How do we synthetically generate complicated tasks from easy ones?**

First confirm that the tasks really are easy. Lenient verifiers, contamination, guessable formats and harness limits often fake saturation, so read the per-prompt pass-rate histogram under the exact policy, harness and verifier.

Then transform *verified artifacts, not prose*:
- Lift each seed into an executable representation that carries its checker: a program, computation graph, environment state, constraint list or formal statement.
- Apply operators that recompute, inherit or construct the label. Compose mastered atoms that share real data dependencies, build the answer or certificate first, hide information behind fuzzed descriptions or tools, invert the direction, turn parametric knobs, or break working artifacts.
- Render the text last.

Admit a candidate only if it passes two gates:
- **Two-sided validity.** It is solvable with privileged information but not without it; the oracle passes and a no-op fails.
- **Pass-rate band.** It lands in a band measured on the *current* policy (0 < p < 1, centred near 0.3–0.6), and the band is re-measured as the policy moves.

Run this as a factory: complexify above the band, train inside it, scaffold below it, and log yield and cost per operator. Use SFT to install atoms, formats and behaviours, and RL to learn their compositions. Adopt a data policy only after paired multi-seed wins on held-out families and a non-Qwen model.

---

## If you only do five things

1. **Diagnose before you synthesize.** Audit the verifier (TPR/TNR on known-good and known-bad solutions) and read the pass-rate histogram in the exact training harness. *Why:* TACO's tests had a false-positive rate above 90% on difficult problems ([HardTests](https://arxiv.org/abs/2505.24098)), and stronger tests cut the SWE-bench Verified top score from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)). **Strong.** → [Ch. 01 §2](01-diagnosis-difficulty-and-learning-signal.md#2-a-diagnostic-checklist-to-run-before-synthesizing-anything), [Ch. 04 §5](04-verification-and-quality-control.md#5-testing-the-tester-mutation-hardened-and-adversarial-verifier-qa)
2. **Harvest the signal you already have before buying tasks.** Filter zero-variance groups, reallocate rollouts, recycle deferred items, keep 1–2% review. *Why:* reallocation alone bought about 2× compute-equivalent at no generation cost ([Knapsack RL](https://arxiv.org/abs/2509.25849)). Late in training Qwen2.5-Math-7B on DAPO-Math-17K (N = 8), about 40% of prompts were all-correct and about 20% all-wrong. **Moderate.** → [Ch. 05 §3.3](05-rl-playbook.md#33-reallocate-rollouts-before-buying-new-tasks)
3. **Make existing seeds hard again with label-preserving operators that reuse your checker.** Test hardening, shortcut and no-CoT filters, answer-preserving stem rewrites, oracle-preserving tool perturbations, hint stripping, failure-prefix and verdict meta-tasks. *Why:* cheapest, and fail-safe (a corrupted answer-preserving item yields an inert all-zero GRPO group); 12 of the Top-15 ideas take under an engineer-week. **Strong/Moderate.** → [Ch. 10 §2](10-idea-bank-and-roadmap.md#2-theme-a-quick-wins-on-existing-verified-seeds), [Ch. 02 §17](02-complexification-operator-taxonomy.md#17-selection-guide-which-operators-to-try-first)
4. **Build new difficulty construction-first from verified parts.** Answer or certificate first, composition of mastered atoms, a parametric knob with a per-family online controller, many families mixed equally; train compositions with RL, not SFT. *Why:* RL on depth-2 compositions reached 27% at unseen depth 3, while RFT on the same data stayed at ≤ 2.6% ([f(g(x))](https://arxiv.org/abs/2509.25123)). On a saturated 1.5B model, 400 adaptive environments gave +3.37, against +0.49 from over 3× more compute on the original data ([RLVE](https://arxiv.org/abs/2511.07317)). **Strong.** → [Ch. 03 §11](03-generation-architectures.md#11-reference-design-the-hardening-factory), [Ch. 05 §4.2](05-rl-playbook.md#42-breadth-the-strongest-lever)
5. **Gate both sides, and decide with statistics.** Bundle contract (oracle passes, no-op fails, known-bad fails), shortcut probes, a pilot band on the current policy. Adopt a policy only if a paired 95% CI excludes zero across 8–12 seeds and the gain holds on post-cutoff items, held-out families and a non-Qwen model. *Why:* IFDecorator ended with 10,772 unsolvable prompts against 7,324 usable ones ([IFDecorator](https://arxiv.org/abs/2508.04632)), and random rewards gave Qwen2.5-Math-7B +21.4 on MATH-500 ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). **Strong.** → [Ch. 04 §6](04-verification-and-quality-control.md#6-two-sided-validity-gates), [Ch. 10 §14.5](10-idea-bank-and-roadmap.md#145-statistics-and-decision-rules)

---

## 1. First diagnose: is it really too easy?

A GRPO group of N rollouts carries gradient with probability 1 − pᴺ − (1 − p)ᴺ ([Knapsack RL](https://arxiv.org/abs/2509.25849)). At N = 8 that is 0.99 at p = 0.5 and 0.08 at p = 0.01 or 0.99 (derived).

Difficulty belongs to the whole (task, policy, verifier, budget) tuple, not to the task:
- Profile with k ≥ 16 rollouts per prompt, and report per family and per operator.
- Treat p̂ = 0 as "unknown". At p = 0.5, an 8-rollout estimate has a 95% half-width of about ±0.35 (derived).
- If only the eval is saturated, build fresher evals rather than harder training data.

Run the checks in order ([Ch. 01 §7](01-diagnosis-difficulty-and-learning-signal.md#7-symptom--remedy-decision-table)):

| Symptom | Likely cause | Check | Remedy (owner) |
|---|---|---|---|
| Most groups all-correct; effective-prompt ratio falling | Saturated pool | Per-family p̂ histogram in the exact harness | Reallocate rollouts; failure-prefix and verdict meta-tasks; then compose and add families (05, 02, 03) |
| Known-bad or near-miss candidates pass | Lenient verifier | TPR/TNR audit (≥ 0.9) | Mutant, hacking-input and near-miss tests; hidden or read-only tests (04) |
| Valid items stuck at 0; a stronger model's answers are also rejected | Verifier false negatives (rule-based math checkers average 86% recall) | Equivalent-rewrite tests | Rule-first cascade with a discriminative second stage; special judges (04) |
| Gains vanish on renamed or post-cutoff items, or on another model family | Contamination or spurious reward | Partial-prompt completion; random-reward arm | Decontaminate *outputs*; procedural held-out families (04, 09) |
| Right answers without valid reasoning | Format leakage | No-CoT guess within 8 tries; question-only and choices-only baselines | MCQ → open-ended; code-RNG answer marginals (02, 04) |
| Mass only at 0 and 1 | Generator steps too coarse, or a broken sub-family | Per-operator histograms | Intermediate levels; audit the 0-spike (02, 03) |
| Failures hit turn or token caps | Budget, not capability (a harness change alone took a 4B SWE agent from 8.3% to 37.2%; [FrogNano](https://arxiv.org/abs/2609.07925)) | Cap-hit share | Scale budget with tier; compact answer formats (05) |
| pass@1 up, large-k pass@k flat or down | Sharpening on saturated data | Paired pass@k vs base | Edge data (low pass@1, pass@k > 0); composed tasks (05) |
| Rollouts never verify or backtrack | Missing behaviours | Behaviour counts on the hard tail | Priming SFT (wrong answers are fine); mid-training (06) |
| Composites near 0 although the atoms look "solved" | Atoms too unreliable for the product law | Per-step accuracy | Sharpen atoms, then compose at depth 2 with a staged horizon (02, 05) |

---

## 2. The operator families at a glance

An operator maps a verified seed (x, y*, V) to (x′, y′*, V′) that the policy solves less often *for a learnable reason*. Its label is **INV** (invariant), **REC** (recomputed), **CON** (constructed, answer first), **CERT** (certificate-graded) or **RED** (re-derived by a teacher, vote or judge). **Use INV, REC, CON and CERT for RL; use RED only for SFT or behind an independent verifier** ([Ch. 02 §1](02-complexification-operator-taxonomy.md#1-the-frame-what-an-operator-is-and-how-the-label-survives)).

| Family | What it does | Easy → hard | How the label stays correct | Evidence |
|---|---|---|---|---|
| [A Compose](02-complexification-operator-taxonomy.md#2-family-a--compose) | Joins verified subtasks that share real data dependencies | "12 muffins/h for 5 h?" → "…4 per box, $8 per box, 15% off: what is paid?" (illustrative, after [Compositional GSM](https://arxiv.org/abs/2410.01748)) | REC/CON: execute the chain | **Strong** (under RL, atoms first) |
| [B Constrain](02-complexification-operator-taxonomy.md#3-family-b--constrain) | Stacks requirements that must all hold | "Write a poem about autumn" → "4 stanzas, all lowercase, 'ember' exactly twice, no commas…" ([IFBench](https://arxiv.org/abs/2507.02833)) | A code checker per constraint; a compatibility witness; choose k so p^k ≈ 0.2–0.5 | **Strong** (count ≠ difficulty) |
| [C Conceal](02-complexification-operator-taxonomy.md#4-family-c--conceal-and-obfuscate) | Fuzzes values, removes clues, degrades specs, moves information behind tools | IMF → "an international financial institution" ([FORT](https://arxiv.org/abs/2606.12087)) | INV, plus a uniqueness, CONCAT or oracle-evidence check after every step | **Strong**; trades difficulty for ambiguity |
| [D Invert](02-complexification-operator-taxonomy.md#5-family-d--invert) | Asks the harder direction; builds the answer first | "Trailing zeros of 100!" → "smallest n whose factorial ends in exactly 24 zeros" ([QbQ](https://arxiv.org/abs/2608.01522)) | CON/CERT; check uniqueness | **Strong** (answer-first); **Moderate** (inversion) |
| [E Scale](02-complexification-operator-taxonomy.md#6-family-e--scale) | Turns size, depth, horizon or context knobs | 4×4 → 9×9 Sudoku; TSP from 10–20 to 45–55 cities | REC: a programmatic checker | **Strong** with a controller and a length audit |
| [F Perturb](02-complexification-operator-taxonomy.md#7-family-f--perturb-and-distract) | Adds off-path noise or makes minimal method-breaking edits | "…but five of them were a bit smaller than average" (answer still 190) ([GSM-Symbolic](https://arxiv.org/abs/2410.05229)) | INV (re-solve to confirm); REC for answer-changing edits | **Moderate** |
| [G Abstract](02-complexification-operator-taxonomy.md#8-family-g--abstract-and-generalize) | Lifts an instance to a parametric family | A Putnam problem with 2011 → 4680 ([Putnam-AXIOM](https://arxiv.org/abs/2508.08292)) | REC; two references agree across scales | **Strong** |
| [H Upgrade type](02-complexification-operator-taxonomy.md#9-family-h--upgrade-the-task-type) | Decision → optimization, compute → prove, static → environment, single → multi-turn | SAT → MaxSAT → minimal unsatisfiable core ([SATQuest](https://arxiv.org/abs/2509.00930)) | CERT behind a feasibility gate; a CONCAT control for sharding | Uneven: many ladders are evaluation-only |
| [I Break](02-complexification-operator-taxonomy.md#10-family-i--break-and-inject-errors) | Breaks a working artifact; the original is the fix | One flipped `<` → PR-mirror or rewrite bugs ([SWE-smith](https://arxiv.org/abs/2504.21798)) | CON: fail-to-pass tests; inverse mutation testing | **Moderate** |
| [J Target learner](02-complexification-operator-taxonomy.md#11-family-j--target-the-learner) | Picks where, which and how far to push from the policy's behaviour | DeepSeek-V4-Pro pass@4 90% → 2.5% over 15 rounds, about $0.05 per task ([RST](https://arxiv.org/abs/2608.05466)) | Accept iff α_low < acc(q′) < acc(q) | **Strong** (seed from "almost solved") |
| [K Scaffolding](02-complexification-operator-taxonomy.md#12-family-k--remove-scaffolding-and-add-it-back) | Strips hints and affordances; the inverse rescues p = 0 items | A task naming the file and format → the abstract goal, with no visible failing tests | INV: the verifier is unchanged | **Strong** |
| [L Representation](02-complexification-operator-taxonomy.md#13-family-l--change-representation) | Re-renders the latent instance | Givens moved into the figure (+22.21 MathVerse Vision-Only; [GeoSym127K](https://arxiv.org/abs/2605.16371)) | INV; back-translation plus a solver check | **Moderate**: mostly diversity |

**Match the family to the missing kind of difficulty.** Scale adds volume. Coupling and optimization add search. Perturbation, insight hiding and inversion force a method change. Tools, turns and memory expose interaction gaps: one base model scored 0.962 when the mechanism was written out and 0.204 when it had to query for parameters ([VHD-Play](https://arxiv.org/abs/2609.27321)).

**Avoid anti-operators** ([Ch. 02 §14](02-complexification-operator-taxonomy.md#14-anti-operators-fake-hardness-and-how-to-detect-it)): paraphrase, padding, ambiguity, contradiction and unrelated concatenation. Rephrasing gave +0.4 GSM8K points where inversion gave +2.3/+2.6 ([MetaMath](https://arxiv.org/abs/2309.12284)). o3's problem-quality ratings correlated 0.07 with human ratings ([AutoCode](https://arxiv.org/abs/2510.12803)). **Strong:** accept outputs only on a measured pass-rate drop plus a validity control.

**Compose operators in a fixed order** (**Proposal**; [Ch. 02 §16](02-complexification-operator-taxonomy.md#16-composing-operators-algebra-order-budgets-stopping-rules)): lift → structure → constrain → tighten → render → conceal → distract → deliver → gate. Verify only the increment, and stop on measured pass rate. Cap free-form evolution at 2–3 rounds unless every round is solution-first and fully re-validated, as in RST, which sustained 15 rounds.

---

## 3. The recommended pipeline: a hardening factory

Generation is usually the cheapest stage; validity checks, profiling and environment builds dominate cost. The design below is a **Proposal** assembled from components with Moderate-to-Strong evidence ([Ch. 03 §11](03-generation-architectures.md#11-reference-design-the-hardening-factory)):

```
 seed = (prompt, env, oracle, verifier, known-bad, lineage, p̂)
   │ 0 intake + verifier audit: oracle passes, no-op fails, known-bad fails
   │ 1 profile on the CURRENT policy (pilot k = 8–16), route by p̂
   ├─ p̂ ≈ 1 ─► 2 choose operator (bandit on yield × Δp) ─► 3 generate (cheap, many)
   │             ─► 4 validity cascade, cheapest first: schema → oracle in fresh sandbox
   │                  → no-op / no-data / no-tool probes → uniqueness → hacker probe
   │             ─► 5 calibrate: skip p̂ > 0.75, defer p̂ < 0.125, else commit 48;
   │                  keep only if harder than the parent and still solved
   │             ─► 6 dedup on solution signatures; decontaminate outputs ─┐
   ├─ in band ─────────────────────────────────────────────────────────────┴► 7 pool ─► trainer
   │    (≤ 20% new per epoch, ≤ 4 children/parent, 20–33% real anchors, 2–10% review,
   │     retire at slope ≤ 0, re-profile every stage)
   └─ p̂ ≈ 0 ─► audit → scaffold or defer (never delete after one pilot)
 per-operator yield and reject reasons feed step 2; accepted children become seeds
```

**Budget** (note 18; derived). The cost per accepted in-band task is C_acc = (c_gen + c_validate + c_env + k·c_roll) / (y_valid·y_band). That means about 4 candidates per accepted task at a 25% yield and about 19 at 5.2%. Profiling took about 36% of rollout tokens in the worked example. Size the unique pool at about S·B/25 (S steps × B prompts per step), since reuse up to about 25× showed no significant degradation ([Tan et al.](https://arxiv.org/abs/2509.25300)).

**Which archetype to use when** ([Ch. 03 §10](03-generation-architectures.md#10-comparing-and-combining-archetypes)):

| Situation | Archetype | Watch for |
|---|---|---|
| A formal or executable backbone exists | Correct-by-construction generation plus a knob and an online controller (**Strong**) | Easy skew; size ≠ reasoning; generator bugs |
| Labelled seeds but no generator | Answer-preserving rewrite behind an independent gate; yields around 25–50% | Invalidity grows with difficulty; prose-only rewrites often fail to beat the seeds |
| Saturated single-skill items with verified atoms | Composition, costing from near zero to $27.3 per task | Unrelated concatenation; non-uniqueness |
| Executable agent tasks | Solver-in-the-loop escalation inside environments whose state lives in code or a DB (**Moderate**) | Ambiguity mistaken for hardness; hackable verifiers |
| Static operators stop filling the band, *and* an independent gate exists | A trained generator (validity-gated setter RL, anchored self-play) | Invalid-but-hard hacking; pseudo-label decay |

Throughout, failure mining decides *where* to generate, a quality-diversity archive keeps coverage, and LLM simulators render only surface text and users.

---

## 4. RL in one page

**Band.** Keep pass rates roughly within 0.2–0.8, centred at about 0.3–0.6. Measure on the model being trained and re-measure every stage. **Strong** ([Ch. 05 §1](05-rl-playbook.md#1-the-signal-budget-why-the-pass-rate-band-matters)).

Training at the edge of competence beats both easier and harder data:
- In a controlled pipeline (100M models trained from scratch), edge data (low pass@1, pass@k > 0) gave up to +42% pass@128; in-distribution data gave none ([Interplay](https://arxiv.org/abs/2512.07783)).
- GRPO on MATH items with pass@8 = 0 *lowered* math-benchmark averages by 5.75, 11.24 and 1.07 points across three model settings ([T-SAE](https://arxiv.org/abs/2605.28388)).

Lab cutoffs vary (keep 2–5 of 8, drop > 62.5%, retire ≥ 0.9) and none is ablated; invest in re-profiling cadence ([Ch. 08 §10.2](08-frontier-lab-practices.md#102-where-labs-disagree)).

**Recipe** (**Proposal** as a whole; per-step tags from [Ch. 05 §12](05-rl-playbook.md#12-step-by-step-recipe)):
1. **Prepare the pool.** Instrument; audit verifiers (known-bad answers, no-op agents, isomorphic variants); profile with 8–16 rollouts; drop no-CoT-guessable items; defer p = 0. **Strong.**
2. **Harvest cheap signal.** Use zero-variance filtering, pilot–commit or Knapsack allocation, recycling and 1–2% review. **Moderate.**
3. **Wrap seed families in parameterized generators.** Give each family its own controller, mix families equally, and keep adding families. **Strong.**
4. **Harden p = 1 items** through gated acceptance and failure-prefix and verdict meta-tasks; add SvS variants of weakly solved items. **Moderate.**
5. **Rescue p = 0 items in order:** audit → behaviours → format ladder or inverse rewrite → teacher answer in the prompt → annealed prefix → off-policy trace. Withdraw the help as the policy improves. **Moderate.**
6. **Shape the reward** as a validity gate × the outcome, with structure-aware partial credit only where success is sparse. Probe it with invariance checks and trivial agents. **Moderate.**
7. **Add a proposer only when steps 3–4 run dry**, with a solver-independent validity gate, external grounding, a solution-signature archive, a never-trained grader and stabilizers (replay, real anchors, KL). **Strong** for the first four.
8. **Re-profile the whole pool each stage.** Stage only tiers that would otherwise sit at zero, resetting the reference model and optimizer. Decide with paired CIs over 8–12 seeds. **Strong.**

| Lever | Reported setting | Source |
|---|---|---|
| Profiling | 16 pilot + 48 commit rollouts; skip p̂ > 0.75; defer p̂ < 0.125 | [Pilot-Commit](https://arxiv.org/abs/2605.26606) |
| Controller | Promote when accuracy at the top level reaches τ_acc = 0.9 over ≥ 8 × (rollouts per problem) samples; sliding window of at most 4 levels | [RLVE](https://arxiv.org/abs/2511.07317) |
| Anchors, review | 20–33% oracle-backed real items (our starting default, **Proposal**; 20% real bugs gave +7.0 pp over unanchored self-play); 2–10% easy or review items | [Anchored Self-Play](https://arxiv.org/abs/2607.03523); [ReMind](https://arxiv.org/abs/2606.03087) |
| Family mix | Equal: none of 3 adaptive mixtures beat it over 12 seeds | [DataFlex-RL](https://arxiv.org/abs/2609.06107) |
| Unique items | ≈ S·B/25; clear overfitting at 100 reuses | [Tan et al.](https://arxiv.org/abs/2509.25300) |
| Scaffolds | Prefix 50% for 100 steps, then 25%; 1 teacher trace + 7 on-policy rollouts (+6.4 average on six math benchmarks over prior RLVR, Qwen2.5-Math-7B) | [QuestA](https://arxiv.org/abs/2507.13266); [LUFFY](https://arxiv.org/abs/2504.14945) |
| Saturated items | Failure-prefix conditioning: 44.5 vs 40.7 for plain RLVR on items solved 121–127 times out of 128 | [Failure-prefix](https://arxiv.org/abs/2601.20829) |

**Ordering and dose.** Stage only when the top tier would otherwise get zero reward. A staged horizon curriculum lifted AIME24 from 5.10 to 10.52 (avg@32, 3B), where a uniform mix gave no long-horizon gains ([h1](https://arxiv.org/abs/2510.07312)). One prompt took Qwen2.5-Math-1.5B from 36.0 to 73.6 on MATH500 ([1-shot RLVR](https://arxiv.org/abs/2504.20571)), but Qwen2.5-Math gains even under random rewards. Before scaling, run a dose ladder (16 / 128 / 1k / 10k) with a random-reward arm and a non-Qwen model. **Moderate.**

**What to watch** ([Ch. 05 §9](05-rl-playbook.md#9-monitoring-the-run-dashboard)): effective-prompt ratio, entropy, length (one sample rewarding a bare `\boxed{40}` collapsed mean length from 510.7 to 45.7 tokens in 58 steps, before accuracy fell; [T-SAE](https://arxiv.org/abs/2605.28388)), valid-proposal rate, large-k pass@k against the base, and a regression suite (math RLVR raised IFEval pass@1 by 6.5% but lowered best@32 by 9.8% on Qwen3-8B-Base; [Support Reshaping](https://arxiv.org/abs/2608.00220)).

---

## 5. SFT in one page

**Division of labour (Strong).** SFT installs atoms, formats, long-CoT behaviours and knowledge. It does not extrapolate composition, horizon or new rules, and it can regress on composites ([Ch. 06 §6](06-sft-playbook.md#6-what-sft-fails-to-teach-and-how-to-sequence-sft-and-rl)):
- RFT on depth-2 compositions stayed at ≤ 2.6% at depth 3, against 27% for RL ([f(g(x))](https://arxiv.org/abs/2509.25123)).
- On simulator-generated physics, SFT on 200K rejection-sampled teacher solutions lowered Qwen2.5-32B's IPhO mechanics score from 19.8 to 15.9, while RL reached 25.2 ([Sim2Reason](https://arxiv.org/abs/2604.11805)).

SFT gets atoms, moderate composites, behaviour traces and knowledge- or format-heavy items; RL alone gets held-out composites, deeper levels and the hardest verifiable band.

**Recipe** ([Ch. 06 §9](06-sft-playbook.md#9-step-by-step-recipe)):
1. **Scope and provenance.** Write down SFT's job and the SFT/RL split. License-screen seeds, evolvers and teachers; log generator, rewriter, verifier and judge per row.
2. **Profile with the student** (k = 8–16). Remove no-CoT-solvable items, convert MCQ to free-form, and decontaminate.
3. **Harden for 2–3 rounds** with correctness-preserving operators (≤ 20 new words per step, an "INVALID" escape, ≤ 10 rewrites per seed); stop on a dev set; hand-audit 100–200 items (0 errors in 200 puts the 95% Wilson upper bound on the error rate at 1.9%).
4. **Select questions** by student failure, teacher trace length or LLM difficulty. Pick the teacher by a student pilot: 2–3 teachers × 1–2K questions.
5. **Allocate traces.** One per unique question first; once unique prompts run out, 4–16 per question in proportion to the fail rate, chosen for route diversity. Privileged hints go in the teacher's context only.
6. **Assemble the mix:** 10–20% meta-tasks, equal operator families, a 20–33% real anchor, enough general data to hold the regression suite, and 2–3 generator families. Run a dose ladder at 1k/4k/16k/64k items with ≥ 3 seeds each. The shares and the ladder are **Proposal** starting points to ablate.
7. **Pick the checkpoint with a short RL probe.** After RL, rejection-sample from the RL policy and distil back.

**Key numbers.**
- **Question selection beats answer checking. Strong.**
  - s1K scored 50.0 vs 36.7 for a random 1K on AIME24 (Qwen2.5-32B-Instruct; [s1K](https://arxiv.org/abs/2501.19393)).
  - Length and difficulty filters beat random selection by +4% (math/science) and +6% (code).
  - Answer filtering did not help: 41.9 unfiltered vs 40.0 GPT-verified ([OpenThoughts](https://arxiv.org/abs/2506.04178)).
- **Choose teachers by the student's result.** QwQ-32B beat DeepSeek-R1 as a teacher (+1.9% code, +2.6% math; OpenThoughts). **Strong.**
- **Unique prompts before extra traces.** 64k×1 > 16k×4 > 8k×8 ([X-Coder](https://arxiv.org/abs/2601.06953)). SFT on about 590K difficulty-proportional traces took Llama3-8B from 21.2 (base) to 46.6 on MATH ([DART-Math](https://arxiv.org/abs/2407.13690)).
- **Size by goal.** 10³–10⁴ examples elicit behaviours in a strong base or cold-start RL; distilling a new domain into a mid-size student takes 10⁵–10⁶+ unique prompts.
- **SFT accuracy does not predict RL outcome.** SkillFactory trailed R1 distillation before RL (2.8% vs 11.7%) and led after GRPO (25.1% vs 21.2%) ([SkillFactory](https://arxiv.org/abs/2512.04072)).
- **Critiques and behaviour traces are cheap wins.**
  - Critique fine-tuning beat plain SFT by 4–10% on six math benchmarks ([CFT](https://arxiv.org/abs/2501.17703)).
  - Behaviour-rich traces with wrong answers primed RL as well as correct ones ([Cognitive Behaviors](https://arxiv.org/abs/2503.01307)).
  - Injected-error traces do not teach self-correction.

---

## 6. Domain quick-picks

| Domain | Best operators | Verifier | Start from |
|---|---|---|---|
| [Informal math](07a-domain-recipes-reasoning.md#2-informal-math-competition-and-word-problems) | Chaining + horizon curriculum; answer-preserving variants (SvS, QbQ); inversion | Execution or SymPy on the construction | [DeepMath-103K](https://arxiv.org/abs/2504.11456), [Big-Math](https://arxiv.org/abs/2502.17387), GSM-Infinite generators |
| [Formal, geometry](07a-domain-recipes-reasoning.md#3-formal-theorem-proving-geometry-and-inequalities) | Subgoal recomposition; auxiliary-point hiding; prove-or-disprove | Lean kernel with a bypass screen; DDAR | [Lean Workbook](https://arxiv.org/abs/2406.03847), [GenesisGeo](https://arxiv.org/abs/2509.21896), [MathlibLemma](https://arxiv.org/abs/2602.02561) |
| [Contest code](07a-domain-recipes-reasoning.md#41-contest-and-function-level-problems) | Test hardening first; solution-first evolution; feature composition | Oracle + validator + checker; brute force vs reference | TACO/CodeContests with [HardTests](https://arxiv.org/abs/2505.24098); [KodCode](https://arxiv.org/abs/2503.02951) |
| [Verified code, kernels](07a-domain-recipes-reasoning.md#42-formally-verified-code-dafny-verus-lean) | Spec lifting; composition; correctness → speed | Proof checker with `sorry`/`assume(false)` scans; hidden-input timing | [Vericoding](https://arxiv.org/abs/2509.22908) (filter weak specs), [VERINA](https://arxiv.org/abs/2505.23135), [KernelBench-Verified](https://arxiv.org/abs/2607.16241) |
| [SWE](07a-domain-recipes-reasoning.md#5-software-engineering-and-repository-level-tasks) | Realistic bug injection (rewrite, PR mirror, feature-add); multi-bug; symptom-only issues | F2P/P2P tests; inverse mutation testing; git/network blocked | [SWE-smith](https://arxiv.org/abs/2504.21798), [R2E-Gym](https://arxiv.org/abs/2504.07164), [SWE-rebench V2](https://arxiv.org/abs/2602.23866) |
| [Puzzles](07a-domain-recipes-reasoning.md#6-logic-puzzles-and-procedural-reasoning-gyms) | Planted solutions + level controller; counterfactual rules | Certificate checks (assignment, core, path) | [Reasoning Gym](https://arxiv.org/abs/2505.24760) (audit first), [SynLogic](https://arxiv.org/abs/2505.19641), RLVE-Gym |
| [Science](07a-domain-recipes-reasoning.md#7-science-textbook--database--and-simulator-derived) | Simulator- or solver-first generation; MCQ → short answer or mask-and-choose | Simulator within tolerance; RDKit | [MegaScience](https://arxiv.org/abs/2507.16812), NaturalReasoning, Sim2Reason DSL (train with RL) |
| [Tool use](07b-domain-recipes-agents-and-beyond.md#1-function-calling-and-multi-turn-tool-use) | Oracle-preserving perturbation; identifier fuzzing; implicit steps | Final DB state, any valid path | [tau2-bench](https://arxiv.org/abs/2506.07982), [AWM](https://arxiv.org/abs/2602.10090), [Agent-World](https://arxiv.org/abs/2604.18292) |
| [Web, deep research](07b-domain-recipes-agents-and-beyond.md#2-web-search-and-deep-research-browsecomp-style) | Fuzz constants; clue removal; leaf-first intersection | Two-sided gate + uniqueness | [WebShaper](https://arxiv.org/abs/2507.15061), [OpenSeeker](https://arxiv.org/abs/2603.15594), [ORBIT](https://arxiv.org/abs/2604.01195) |
| [GUI](07b-domain-recipes-agents-and-beyond.md#3-gui-computer-use-and-mobile-agents) | Evaluator-first composition; start-state regression; explicit → implicit goals | Code state checker: checker(golden) = 1, checker(initial) = 0; no VLM judge | OSWorld-Verified, AndroidWorld, [CUA-Gym](https://arxiv.org/abs/2605.25624) |
| [Terminal, office](07b-domain-recipes-agents-and-beyond.md#4-terminalcli-data-analysis-spreadsheet-and-office-agents) | Solution-first escalation (RST); workbook inversion; lower specificity | Fresh-sandbox oracle; real-engine recompute on perturbed inputs | [RST](https://arxiv.org/abs/2608.05466) (37,484 tasks), [CalibForge](https://arxiv.org/abs/2608.06352), Spreadsheet-RL |
| [Instruction following](07b-domain-recipes-agents-and-beyond.md#5-instruction-following-with-composable-constraints) | p^k-calibrated stacking; append until pass@1 ∈ (0, 0.5] | Code check per constraint; intent check | IFTrain/[IFBench](https://arxiv.org/abs/2507.02833), VerInstruct, RECAST-30K |
| [Long context](07b-domain-recipes-agents-and-beyond.md#6-multi-hop-and-long-context-reasoning) | Premise dispersal; lexical-overlap removal; mined distractors; length last | Construction labels; set-F1; evidence F-beta | [MuSiQue](https://arxiv.org/abs/2108.00573), 2Wiki, PhantomWiki, RULER |
| [Interaction, memory](07b-domain-recipes-agents-and-beyond.md#7-memory-and-multi-turn-interaction-structure) | Sharding; objective stacking; hidden-spec user | Original verifier + CONCAT control | [Lost in Conversation](https://arxiv.org/abs/2505.06120), MultiChallenge, LongMemEval |
| [Multimodal](07b-domain-recipes-agents-and-beyond.md#8-multimodal-and-visual-reasoning) | Answer-hidden stem rewrites; mechanism-changing levels | Answer from the latent state, never pixels | [TRON](https://arxiv.org/abs/2606.01599), [ChartVerse](https://arxiv.org/abs/2601.13606), [GeoSym127K](https://arxiv.org/abs/2605.16371) |
| [SQL, tables](07b-domain-recipes-agents-and-beyond.md#9-text-to-sql-and-tables) | AST mutation; distractor tables | Execution; non-empty result; gold < 5 s; ≥ 1/10 solvable | BIRD/Spider (cleaned gold), [OmniSQL](https://arxiv.org/abs/2503.02240), [EvolSQL](https://arxiv.org/abs/2601.04875) |
| [Open-ended](07b-domain-recipes-agents-and-beyond.md#10-open-ended-and-non-verifiable-tasks) | Convert to verifiable (mask-and-choose); append one constraint; rubric evolution | Constitutive rubrics + held-out judge panel | [Golden Goose](https://arxiv.org/abs/2601.22975), [HuatuoGPT-o1](https://arxiv.org/abs/2412.18925) |

The per-seed-type selection guide is in [Ch. 02 §17](02-complexification-operator-taxonomy.md#17-selection-guide-which-operators-to-try-first), and the operator × domain matrix is in [Ch. 02 §15](02-complexification-operator-taxonomy.md#15-operator--domain-matrix).

---

## 7. Top pitfalls

- **[Complexity ≠ difficulty](09-pitfalls-and-failure-modes.md#f1-complexity--difficulty).** In a gated RLVR study, 64% of rejected mutations were too easy. Accept a mutation only if the measured pass rate falls.
- **[Overshoot](09-pitfalls-and-failure-modes.md#f2-overshoot-to-unsolvable) and [ambiguity](09-pitfalls-and-failure-modes.md#f3-ambiguity-masquerading-as-difficulty).** Treat p = 0 as a bug report until certified solvable (teacher solve, pass@large-k > 0, co-built solution); check uniqueness after every obfuscation step; defer rather than delete.
- **[Consensus labels on the hard tail](09-pitfalls-and-failure-modes.md#f4-wrong-labels-and-wrong-verifiers-on-hard-items).** R-Zero's pseudo-label accuracy fell from 79% to 63%. The unverified majority is wrong on 73.33% of AIME 2024 questions ([T³RL](https://arxiv.org/abs/2603.02203)). Construct labels instead, and use decorrelated evidence.
- **[Hacking grows with difficulty](09-pitfalls-and-failure-modes.md#f7-reward-hacking-of-verifiers-tests-rubrics-leandafny-constraint-checkers).** Verifier shortcuts rose from 40 at complexity levels 1–10 to 458 at levels 11–20 ([IPT](https://arxiv.org/abs/2604.15149)). Guard with read-only, out-of-process grading, trip-wires, trivial-agent baselines and hacker–fixer passes.
- **[Generator signatures](09-pitfalls-and-failure-modes.md#f6-generator-artifacts-signatures-and-shortcut-learning).** LLM-written NLI is 86–96% solvable from the hypothesis alone. Use a code RNG for answer positions, run partial-input baselines, and draw on 2–3 generator families.
- **[SFT for composition](09-pitfalls-and-failure-modes.md#f8-sft-only-composition-failures).** SFT scored 0.000 at 16 objectives, where RL scored 1.900 ([MEM1](https://arxiv.org/abs/2506.15841)). Teach atoms with SFT and composites with RL.
- **[Collateral damage](09-pitfalls-and-failure-modes.md#f9-collateral-damage-specialist-traps-and-forgetting).** RL on Knights-and-Knaves alone dropped the code average from 67.46 to 56.09. Keep every target domain in the mix and run a regression suite.
- **[Loops degrade](09-pitfalls-and-failure-modes.md#f12-proposer-death-spirals).** One probe-based generator reward put about 74% of tasks on one topic, and a −0.1 penalty for invalid proposals caused a death spiral. Zero reward for invalid proposals, real anchors, a persistent archive; stop after about 3 rounds unless held-out metrics still rise.
- **[Contamination recreated by complexification](09-pitfalls-and-failure-modes.md#f14-contamination-including-contamination-recreated-by-complexification).** DeepMath's raw pool contained 90% of AIME24. Decontaminate the outputs, not only the seeds.
- **[False wins](09-pitfalls-and-failure-modes.md#f16-eval-noise-and-false-conclusions).** The seed-to-seed SD on AIME/AMC is 5–15 points. Use a random-reward arm, a non-Qwen model and 8–12 paired seeds.
- **[Cost through yield](09-pitfalls-and-failure-modes.md#f18-cost-blowups-from-overgeneration-and-profiling).** At 5% yield, each kept task costs about 20 candidates plus their profiling. Audit, reallocate and recycle first, and track C_acc per operator.

Run the [pre-flight checklist](09-pitfalls-and-failure-modes.md#8-pre-flight-checklist) before scaling any generator.

---

## 8. Top-15 ideas and the 30/60/90-day roadmap

Score = Impact × Confidence ÷ Effort ([Ch. 10 §12](10-idea-bank-and-roadmap.md#12-top-15-ranked)). Effort S means under about one engineer-week, reusing the existing verifier; M means 2–4 engineer-weeks.

| # | Idea | Evidence | Effort | First deliverable |
|---|---|---|---|---|
| 1 | [A1](10-idea-bank-and-roadmap.md#a1--re-audit-the-verifier-then-harden-the-tests) Verifier audit + test hardening | Strong | S | TPR/TNR and false-negative report on both tails |
| 2 | [F1](10-idea-bank-and-roadmap.md#f1--a-task-bundle-contract-with-ci-gates) Task-bundle contract + CI | Strong | S | Oracle-pass / no-op-fail / known-bad-fail over the pool |
| 3 | [A2](10-idea-bank-and-roadmap.md#a2--answer-preserving-stem-hardening) Answer-preserving stem hardening | Moderate | S | 3 rewrites per saturated seed, SynthRL gate |
| 4 | [A5](10-idea-bank-and-roadmap.md#a5--oracle-preserving-perturbations-for-tool-use-seeds) Oracle-preserving tool perturbations | Moderate | S | 4 perturbation families, gold traces replayed |
| 5 | [A4](10-idea-bank-and-roadmap.md#a4--shortcut-and-no-context-filters-as-a-pre-pass) Shortcut / no-context filters | Strong | S | Shortcut rate per family |
| 6 | [B3](10-idea-bank-and-roadmap.md#b3--constraint-stacking-calibrated-by-pk-with-structure-aware-rewards) p^k-calibrated constraint stacking | Strong | S | Per-constraint success fit on your policy |
| 7 | [F4](10-idea-bank-and-roadmap.md#f4--two-sided-privileged-information-solvability-gates) Two-sided solvability gates | Strong | S | p = 0 tail split into hard vs broken |
| 8 | [I1](10-idea-bank-and-roadmap.md#i1--mutated-twin-and-held-out-family-evaluations-built-before-training) Mutated-twin / held-out-family evals | Strong | S | Frozen evaluation suite |
| 9 | [G1](10-idea-bank-and-roadmap.md#g1--a-per-family-online-difficulty-controller) Per-family online controller | Strong | M | 10 families with knobs + sliding window |
| 10 | [A6](10-idea-bank-and-roadmap.md#a6--strip-hints-scaffolds-and-specifics-keep-the-checker) Strip hints, scaffolds, specifics | Moderate | S | 3-rung ladders with contract validity |
| 11 | [G4](10-idea-bank-and-roadmap.md#g4--pilotcommit-profiling-rollout-reallocation-and-zero-variance-recycling) Pilot–commit, reallocation, recycling | Moderate | S | Rollouts saved at equal accuracy |
| 12 | [H1](10-idea-bank-and-roadmap.md#h1--behavior-priming-including-wrong-answer-traces) Behaviour priming | Moderate | S | Primed vs unprimed RL probe |
| 13 | [I3](10-idea-bank-and-roadmap.md#i3--a-dashboard-of-per-operator-yield-and-cost-per-accepted-in-band-task) Per-operator yield/cost dashboard | Moderate | S | C_acc by operator |
| 14 | [B1](10-idea-bank-and-roadmap.md#b1--serial-chaining-with-deterministic-adapters-and-a-horizon-curriculum) Serial chaining + horizon curriculum | Strong | M | Chains of length 2–5, staged vs uniform |
| 15 | [D1](10-idea-bank-and-roadmap.md#d1--recursive-solution-first-escalation-rst-for-executable-tasks) RST recursive escalation | Moderate | M | 3 rounds on 200 executable seeds |

All fifteen are established practice. The best-scoring novel bets are D2 (RST for workbooks and notebooks), J4 (online re-complexification) and B8 (branch–merge curricula), all **Proposal**.

**Roadmap** ([Ch. 10 §13](10-idea-bank-and-roadmap.md#13-a-306090-day-roadmap-for-a-team-whose-tasks-are-too-easy)):
- **Days 1–30: measure, then re-harden the existing pool (label-preserving only).** Week 1: task bundles, CI gates, yield dashboard, frozen eval suite with a non-Qwen model, pilot–commit profiling. Week 2: verifier audit, shortcut probes, two-sided gates. Weeks 3–4: rewrites, perturbations, hint stripping, p^k stacking, failure-prefix at p̂ ≥ 0.95, controllers, priming. *Exit:* effective-prompt ratio back up, ≥ S·B/25 in-band items, a held-out CI excluding zero.
- **Days 31–60: construction engines.** Chaining and composition (reasoning), RST and evaluator-first chaining (agents), fuzz-and-disperse (search), red-team harnesses, the p ≈ 0 ladder, SvS. *Exit:* every operator has a logged yield, exploit rate and held-out delta; the two best policies are run through the 8–12-seed decision protocol.
- **Days 61–90: learn the generator.** A validity-gated setter or a trained/prompted/static constructor ablation; an operator × domain bandit vs an equal mix; one novel bet. *Exit:* promote only if the pipeline passes [§14.5](10-idea-bank-and-roadmap.md#145-statistics-and-decision-rules) and its C_acc is known.

---

## 9. How this report was built

1. **Coverage.** 14 subtopic research sweeps produced notes 01–14; a critic review identified 9 gap subtopics, swept the same way (notes 15–23). The notes hold 817 method entries; [Ch. 11](11-bibliography.md) merges their references into 1,134 unique works.
2. **Adversarial citation verification.** Every entry was re-checked against its primary source (mostly arXiv full text), with numbers re-read from text and tables; 315 entries were corrected (6–27 per file) and 3 dropped.
3. **Deterministic arXiv checks.** Every arXiv ID was resolved on arxiv.org and its title matched to the cited work.
4. **Chapter reviews.** Chapters were written only from the verified notes. Each then got a fact-check review of numbers, conditions, links and evidence tags.

**Limits.**
- Many 2026 results are single preprints (**Emerging**).
- Much RL evidence comes from Qwen models.
- Most operator comparisons use SFT; no matched-compute RLVR ablation of operators exists.
- Every **Proposal** is untested.

## 10. How to read this report

| Chapter | What it gives you |
|---|---|
| [01 Diagnosis](01-diagnosis-difficulty-and-learning-signal.md) | Signal math, bands, diagnostic checklist, symptom → remedy |
| [02 Operators](02-complexification-operator-taxonomy.md) | 12 families, label modes, anti-operators, domain matrix, composition order, selection guide |
| [03 Architectures](03-generation-architectures.md) | Nine archetypes, comparison, hardening factory, cost model |
| [04 Verification](04-verification-and-quality-control.md) | Verifier failure rates, two-sided gates, hard-vs-broken triage, hacking guards, QC checklist |
| [05 RL playbook](05-rl-playbook.md) | Bands, pool management, controllers, self-play, scaffolds, rewards, dose, dashboard, recipe |
| [06 SFT playbook](06-sft-playbook.md) | Evolution, distillation, quality vs quantity, meta-tasks, SFT → RL sequencing, licensing, recipe |
| [07a Reasoning domains](07a-domain-recipes-reasoning.md) | Math, formal proofs and geometry, code, SWE, puzzles, science |
| [07b Agents and beyond](07b-domain-recipes-agents-and-beyond.md) | Tool use, web, GUI, terminal/office, instruction following, long context, memory, multimodal, SQL, open-ended |
| [08 Lab practices](08-frontier-lab-practices.md) | What labs report, consensus recipe, disagreements |
| [09 Pitfalls](09-pitfalls-and-failure-modes.md) | 18 failure modes, alarms, pre-flight checklist |
| [10 Idea bank](10-idea-bank-and-roadmap.md) | 65 ideas in ten themes, Top-15, roadmap, experiment protocol |
| [11 Bibliography](11-bibliography.md) | All works, mapped to the notes that discuss them |

Evidence base: [`research/notes/`](../research/notes/) (each file: TL;DR, methods table, method notes, operators, pitfalls, open problems, references).

**Reading paths:**
- **One hour:** this summary, then [Ch. 01 §7](01-diagnosis-difficulty-and-learning-signal.md#7-symptom--remedy-decision-table), [Ch. 09 §8](09-pitfalls-and-failure-modes.md#8-pre-flight-checklist) and [Ch. 10 §13](10-idea-bank-and-roadmap.md#13-a-306090-day-roadmap-for-a-team-whose-tasks-are-too-easy).
- **Building an RL pipeline:** chapters 01 → 02 → 03 → 04 → 05.
- **SFT:** chapter 06.
- **One domain:** chapter 07a or 07b, then the Ch. 02 matrix.
