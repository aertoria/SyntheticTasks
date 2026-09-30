# Idea bank and roadmap: concrete things to build, ranked

> **Key takeaways**
>
> - **First make the seeds you already have hard again; build generators second.** The cheapest high-confidence moves reuse the existing verifier: audit and harden tests, strip shortcuts, answer-preserving stem rewrites, oracle-preserving tool perturbations, hint stripping. All are established practice (**Strong/Moderate**).
> - **New difficulty is most reliable when built from verified parts:** composition of mastered atoms (chaining with a horizon curriculum, p^k-calibrated constraint stacking, evaluator-first chaining), solution-first recursive escalation of executable artifacts (RST), and planted or answer-first construction. Train compositions with RL, not SFT (**Strong**).
> - **"Hard" usually means "broken" unless every candidate passes the same gates:** bundle contract (oracle passes, no-op fails, known-bad fails), two-sided solvability, shortcut probes, a pilot band on *your* policy, and a red-team pass. Log yield and cost per operator: in the one controlled RLVR study, 64% of rejected mutations were too easy ([Trading Human Curation](https://arxiv.org/abs/2606.03800)).
> - **Trained generators are the 60–90-day bet** (validity-gated setter RL, SvS, stepping stones, a learned constructor of (problem, environment, verifier) triplets). They pay off once static operators stop filling the band, and the loop must never train its own grader (**Moderate/Emerging**).
> - **About a quarter of the ideas are novel proposals** built from the notes' open problems (operator-conditioned priors, online re-complexification, Lean-checked "harder-than" variants, RST for workbooks and notebooks). Each is tagged **Proposal** and carries a first experiment and a kill criterion.
> - **Adopt a data policy only if its paired 95% CI excludes zero over 8–12 matched seeds** and the gain holds on held-out operator families, post-cutoff items and a non-Qwen model, with no regression-suite drop ([§14](#14-experiment-protocol)).

## Contents

1. [How to use this chapter](#1-how-to-use-this-chapter)
2. [Theme A: Quick wins on existing verified seeds](#2-theme-a-quick-wins-on-existing-verified-seeds)
3. [Theme B: Composition engines](#3-theme-b-composition-engines)
4. [Theme C: Obfuscation and inversion engines](#4-theme-c-obfuscation-and-inversion-engines)
5. [Theme D: Environment and agentic hardening](#5-theme-d-environment-and-agentic-hardening)
6. [Theme E: Trained generators and self-play](#6-theme-e-trained-generators-and-self-play)
7. [Theme F: Verification infrastructure](#7-theme-f-verification-infrastructure)
8. [Theme G: Difficulty calibration and curriculum](#8-theme-g-difficulty-calibration-and-curriculum)
9. [Theme H: SFT-specific ideas](#9-theme-h-sft-specific-ideas)
10. [Theme I: Evaluating the synthetic pipeline itself](#10-theme-i-evaluating-the-synthetic-pipeline-itself)
11. [Theme J: Novel combinations drawn from the open problems](#11-theme-j-novel-combinations-drawn-from-the-open-problems)
12. [Top-15, ranked](#12-top-15-ranked)
13. [A 30/60/90-day roadmap for a team whose tasks are too easy](#13-a-306090-day-roadmap-for-a-team-whose-tasks-are-too-easy)
14. [Experiment protocol](#14-experiment-protocol)

---

## 1. How to use this chapter

This chapter turns the report into build items and does not repeat the background: diagnosis ([01](01-diagnosis-difficulty-and-learning-signal.md)), operators ([02](02-complexification-operator-taxonomy.md)), pipelines ([03](03-generation-architectures.md)), verifiers ([04](04-verification-and-quality-control.md)), RL ([05](05-rl-playbook.md)), SFT ([06](06-sft-playbook.md)), domains ([07a](07a-domain-recipes-reasoning.md), [07b](07b-domain-recipes-agents-and-beyond.md)), labs ([08](08-frontier-lab-practices.md)) and failure modes ([09](09-pitfalls-and-failure-modes.md)).

**Tags used on every idea.**

| Tag | Meaning |
|---|---|
| `Established` | Used for this purpose in at least one published pipeline or lab report in the notes. |
| `Extension` | Every component is published, but not in this domain or combination. |
| `Novel` | Our proposal. The notes contain no test of it; most come from an "open problems" section. |
| **Strong / Moderate / Emerging / Proposal** | Evidence strength: several independent works or large ablations / one careful study / a single recent or unreplicated work / untested synthesis. |
| Effort **S / M / L** | S: under about one engineer-week and reuses the existing verifier. M: 2–4 engineer-weeks and new generator or environment code. L: more than an engineer-month, or a trained generator model. |

**Shared admission pipeline.** Every idea assumes that its output passes through the gates below. Each idea's "seeds / verifier" line lists only what it needs beyond them.

```
verified seed ──► operator(s) ──► candidate task bundle
                                     │ G0 contract: oracle passes, no-op fails, known-bad fails  (§7 F1)
                                     │ G1 two-sided gate: solvable with privileged info, not without (F4)
                                     │ G2 shortcut probes: no-CoT / no-context / tool-free / trivial agent (A4)
                                     │ G3 pilot band on the CURRENT policy: keep 0<p<1, aim ≈0.3–0.6 (G4)
                                     │ G4 dedup + skill-level signature, lineage log, per-operator yield (I3, I4)
                                     ▼
                      accepted ──► pool (+20–33% real anchors, 2–10% easy replay) ──► RL / SFT
                      rejected ──► reason logged: too easy | broken | ambiguous | hackable | duplicate
```

The band target of about 0.3–0.6 follows the convergence of calibrated generators reported in [UltraLogic](https://arxiv.org/abs/2601.03205), Knapsack RL and ZPPO (note 20). The anchor shares come from note 19; see [Chapter 05 §8](05-rl-playbook.md).

---

## 2. Theme A: Quick wins on existing verified seeds

These nine ideas reuse the checker you already have, so labels are free and the downside is small. Run them before building anything new.

### A1 · Re-audit the verifier, then harden the tests
`Established` · **Strong** · Effort **S**

- **Pitch.** Many "saturated" items are verifier artifacts; tightening the check is the cheapest, safest way to make them hard again.
- **Mechanism / seeds / verifier.** Add near-miss and hack inputs, mutant-killing and disagreement-driven tests; convert exact match to special judges; add a model fallback for rule-negative answers. Needs any executable checker plus your own passing and failing rollouts as near-miss candidates.
- **Why it should work.**
  - [SWE-ABS](https://arxiv.org/abs/2603.00520): about 1 in 5 "solved" SWE-bench Verified patches were semantically wrong; the top score fell from 78.80% to 62.20% under strengthened tests.
  - [EvolveCoder](https://arxiv.org/abs/2603.12698)'s evolved tests cut pass@1 from 43.80 to 31.22 *on the same problems*.
  - [HardTests](https://arxiv.org/abs/2505.24098) raised precision on AtCoder 4+ from 21.67 (TACO tests) to 60.00, and RL on them reached pass@10 64.76 vs 57.14.
  - The error also runs the other way: rule-based math checkers average 86% recall ([Rule vs model verifiers](https://arxiv.org/abs/2505.22203)).
- **Effect / risk.** Restores variance on items passed by exploit. Risk: tests that reject correct code; keep a test only if every known-correct reference passes, and track TPR/TNR ≥ 0.9 (note 11).
- **First experiment.** Harden tests on 200 items at p̂ = 1; re-check 200 at p̂ = 0 with a model fallback. *Success:* report the share of "saturated" items that re-enter (0, 1) and of "impossible" items that were false negatives, with new tests at TPR ≥ 0.9.

### A2 · Answer-preserving stem hardening
`Established` · **Moderate** · Effort **S**

- **Pitch.** Rewrite saturated questions so the *same* gold answer takes more reasoning; no new labels.
- **Mechanism / seeds / verifier.** [MathForge](https://arxiv.org/abs/2601.20614) MQR's three rewrites: irrelevant background, an invented abstract term, and a key number replaced by an independent sub-problem (execute it; it must equal the constant it replaces). Or an answer-hidden rewrite gated by pass rate ([SynthRL](https://arxiv.org/abs/2506.02096)). Exact-answer seeds and a rule checker.
- **Why it should work.** An o3 audit found 99/97/97% of MQR rewrites answer-equivalent, and a broken rewrite yields an all-zero GRPO group, so it is inert. With the DGPO optimizer, the average rose from 37.61 (GRPO) to 42.17 on Qwen2.5-Math-7B (data and optimizer confounded). SynthRL hardens only seeds at ≥ 12/16 and accepts a rewrite only if 4 ≤ passes ≤ original − 2.
- **Effect / risk.** Refills the band from the easy end. Risk: answer or hint leakage; screen with a model that sees only the rewrite.
- **First experiment.** 2k seeds at p̂ ≥ 0.9, 3 rewrites each, SynthRL gate; GRPO on seeds + rewrites vs seeds only at equal steps. *Success:* higher effective-prompt ratio and a held-out paired CI excluding zero.

### A3 · Answer-space hardening and special-judge conversion
`Established` · **Strong** · Effort **S**

- **Pitch.** Remove guessable formats and make multi-answer problems trainable.
- **Mechanism / seeds / verifier.** MCQ → open-ended; canonical answers (integers, squarefree forms); drop yes/no and multi-part items; replace exact match with generated special judges, each validated on known-correct and known-incorrect submissions.
- **Why it should work.** [Big-Math](https://arxiv.org/abs/2502.17387)-Reformulated puts more than 50% of items in the two hardest solve-rate quintiles, and labs routinely convert or drop MCQ (note 12). [ScaleBox](https://arxiv.org/abs/2604.27467): 14.57% of 34,757 code problems need special judges, exact match rejects 59.01% of correct solutions to them, and the best generated judges reach TPR/TNR 96.3/88.5.
- **Effect / risk.** Removes guessing and false negatives. Risk: free-form answers need a stronger verifier; a Qwen2.5-72B judge gave up to 67% false positives on "master key" answers ([Master-RM](https://arxiv.org/abs/2507.08794)). For masked-span items, open-ended conversion failed (>83% zero accuracy, [Golden Goose](https://arxiv.org/abs/2601.22975)).
- **First experiment.** Convert the MCQ slice, re-verify uniqueness, and test the checker metamorphically on equivalent answer rewrites. *Success:* option-only baselines at chance; verifier false positives below 1% on the metamorphic set.

### A4 · Shortcut and no-context filters as a pre-pass
`Established` · **Strong** · Effort **S**

- **Pitch.** Delete items solvable without the target skill *before* hardening them, or the hardening sits on a shortcut.
- **Mechanism / seeds / verifier.** Five probes: no-CoT guess, dropped if right within 8 tries ([Kimi k1.5](https://arxiv.org/abs/2501.12599)); tool-free model, dropped if right in ≥ 1 of 8 ([GLM-5](https://arxiv.org/abs/2602.15763)); closed-book ([MemAgent](https://arxiv.org/abs/2507.02259) dropped about 50% of 80K HotpotQA questions); no-data, dropped if ≥ 3 of 5 LLMs answer without the files ([DSGym](https://arxiv.org/abs/2601.16344)); and component ablation.
- **Why it should work.** [Sim2Reason](https://arxiv.org/abs/2604.11805)'s component-ablation filter was the difference between 7.14% and 13.15% on IPhO at 3B. LLM-written NLI is 86–96% solvable from the hypothesis alone (note 19).
- **Effect / risk.** Fewer items, each carrying real signal. Risk: filters are policy-specific; re-run each stage.
- **First experiment.** Report the shortcut rate per family, then train filtered vs unfiltered at equal steps. *Success:* equal or better held-out accuracy with fewer items.

### A5 · Oracle-preserving perturbations for tool-use seeds
`Established` · **Moderate** · Effort **S**

- **Pitch.** Keep the gold call trace and answer; make the inputs harder.
- **Mechanism / seeds / verifier.** Distractor and look-alike tools, indirect phrasing, noisy or erroneous outputs ([COVERT](https://arxiv.org/abs/2604.09813)); IDs → unique descriptions ([CoVe](https://arxiv.org/abs/2603.01940)); a multi-call trace collapsed into one high-level request ([HardGen](https://arxiv.org/abs/2601.01498)); arguments that must come from earlier turns ([SAP](https://arxiv.org/abs/2609.06124)). The gold trace must still replay.
- **Why it should work.** COVERT: BFCL v3 56.5 → 59.9 with RL alone (Qwen2.5-14B). HardGen: Qwen3-4B BFCLv3 62.13 → 79.14 after SFT+RL.
- **Effect / risk.** Saturated tool sets become learnable at near-zero verification cost. Risk: an ID-to-description swap can be ambiguous; check uniqueness with a DB query, as CoVe does, and tag judge-assisted families.
- **First experiment.** Apply the four families to 1k saturated tool tasks. *Success:* ≥ 1 family per task lands in band, and held-out tool benchmarks beat a seeds-only arm.

### A6 · Strip hints, scaffolds and specifics (keep the checker)
`Established` · **Moderate** · Effort **S**

- **Pitch.** Many tasks are easy because the prompt does part of the work; remove that part and leave the verifier alone.
- **Mechanism / seeds / verifier.** Explicit → implicit goals ([WorkArena++](https://arxiv.org/abs/2407.05291)); lower instruction specificity ([WTM](https://arxiv.org/abs/2608.07873)); terse errors and no interface cues ([Environment Tuning](https://arxiv.org/abs/2510.10197)); start from the site root ([Go-Browse](https://arxiv.org/abs/2506.03533)); redact names and examples, vaguify references ([Trading Human Curation](https://arxiv.org/abs/2606.03800)).
- **Why it should work.** WorkArena++ L2 → L3 takes GPT-4o from 3.0% to 0%; WTM scores fall monotonically from Level 1 to Level 3; information removal cut solve rates by 70–100 points.
- **Effect / risk.** Large, cheap pass-rate drops. Risk: underspecification is ambiguity, not difficulty ([FrogNano](https://arxiv.org/abs/2609.07925) got 0% from a missing contract). Enforce [RST](https://arxiv.org/abs/2608.05466)'s *contract validity* (every checked property stated or discoverable) and keep the detailed twin to separate spec gaps from capability gaps.
- **First experiment.** Build full / abstract / minimal rungs for 300 saturated agent tasks. *Success:* ≥ 1 rung per task in band, while a strong reference agent stays within a few points of its full-rung success on the minimal rung.

### A7 · Failure-prefix conditioning for saturated items
`Established` (one study) · **Emerging** · Effort **S**

- **Pitch.** An item solved 127 times out of 128 still teaches something: start rollouts from its one failure.
- **Mechanism / seeds / verifier.** Truncate a rare incorrect rollout, choosing the prefix length so accuracy is about 0.5; refresh prefixes as the policy moves. Unchanged checker.
- **Why it should work.** On MATH items solved 121–127/128, the 5-benchmark average rose from 40.6 to 44.5, vs 40.7 for plain RLVR on the same items and 44.0 for freshly collected medium problems ([Failure-prefix conditioning](https://arxiv.org/abs/2601.20829)).
- **Effect / risk.** Signal from dead weight. Risk: prefixes drift off-policy; measure the cost to adherence to correct prefixes.
- **First experiment.** Replicate on your p̂ ≥ 0.95 pool with a non-Qwen model. *Success:* matches a "fresh medium problems" arm at equal compute.

### A8 · Meta-task relabeling: verdict, critique, error location
`Established` · **Moderate** · Effort **S**

- **Pitch.** Turn your own rollouts into judge, locate and repair tasks labelled by the existing verifier.
- **Mechanism / seeds / verifier.** Solution → verdict items (swapping 20% of RL prompts for them helped [Critique-Coder](https://arxiv.org/abs/2509.22824)); inject an error, recompute downstream steps and ask for the first wrong step ([2605.02395](https://arxiv.org/abs/2605.02395)); locate one injected error ([ViCrit](https://arxiv.org/abs/2506.10128)); an on-policy self-correction turn ([SCoRe](https://arxiv.org/abs/2409.12917)).
- **Why it should work.** GPT-4-injected errors gave diagnostic data more than 90% accurate even though GPT-4 detects incorrect solutions only about 40% of the time ([MR-GSM8K](https://arxiv.org/abs/2312.17080)). ViCrit RL averaged 53.01, vs 48.41 for caption SFT and 50.61 for the base.
- **Effect / risk.** Hard tasks with exact labels. Risk: off-policy injected errors do **not** teach self-correction; the model repeats them (note 20). Balance labels and draw candidates from near-misses.
- **First experiment.** Replace 20% of RL prompts with verdict items from your rollouts. *Success:* verdict accuracy rises and the main task does not regress.

### A9 · Worst-case variant scoring and method-breaking edits
`Established` (evaluation) / `Extension` (training) · **Moderate** · Effort **S–M**

- **Pitch.** Turn each seed into a small family and score the worst case; add minimal edits that break the memorized method.
- **Mechanism / seeds / verifier.** Variabilization (a script with free constants; reward only if all k variants are solved) and hard perturbation (a minimal edit whose answer must differ from the original, per [QbQ](https://arxiv.org/abs/2608.01522)). Recompute every answer by program or CAS.
- **Why it should work.** [DynaMath](https://arxiv.org/abs/2411.00836): worst-case accuracy is at most about 50% of average-case for all 14 VLMs tested. [Putnam-AXIOM](https://arxiv.org/abs/2508.08292) variations cost o1-preview 19.6 points; [MATH-Perturb](https://arxiv.org/abs/2502.06453) hard perturbations cost o1-mini 16.49%. [MathConstruct](https://arxiv.org/abs/2502.10197): o3-mini's robust accuracy was 42.2% vs 60.5% average.
- **Effect / risk.** Punishes template matching. Risk: seeds whose solution depends on the specific constant; exclude them.
- **First experiment.** Variabilize 500 seeds; train with an all-k-variants reward vs a single-instance reward. *Success:* worst-case accuracy on held-out variants improves more than average-case.

---

## 3. Theme B: Composition engines

Once atoms are saturated, composition is the most reliable source of new difficulty that stays verifiable ([Chapter 02](02-complexification-operator-taxonomy.md)). Two rules apply throughout: make the atoms reliable first, and train compositions with RL rather than SFT.

### B1 · Serial chaining with deterministic adapters and a horizon curriculum
`Established` · **Strong** · Effort **M**

- **Pitch.** Chain already-labelled problems so answer i feeds problem i+1; no new labels.
- **Mechanism / seeds / verifier.** Serial composition with typed, integral adapters; reward only the final answer; stage the horizon (2 → 3 → 5 links). Needs code-form solutions or exact answers and an executor that propagates values.
- **Why it should work.** [h1](https://arxiv.org/abs/2510.07312) (Dr. GRPO, 3B instruct): AIME24 avg@32 5.10 → 10.52, MATH-500 64.20 → 69.20, GSM-Symbolic P2 43.08 → 52.00, with gains holding at pass@128. At equal compute, uniform mixing and long-only training gave **no** long-horizon gains. Multi-step success tracks the product of per-step success (Pearson ρ 0.69–0.96, [Algebrarium](https://arxiv.org/abs/2602.08281)); [Compositional GSM](https://arxiv.org/abs/2410.01748) measures the gap against S1·S2.
- **Effect / risk.** Monotone difficulty with exact labels. Risk: under the product law long chains fail every rollout; sharpen atoms and stage the horizon.
- **First experiment.** Chains of length 2–5 from 5k solved seeds; staged curriculum vs uniform mix at equal compute. *Success:* held-out competition gains with pass@128 not below the base.

### B2 · Functional composition of mastered atoms, evaluated on held-out atoms
`Established` (controlled studies) · **Strong** in toy settings, **Emerging** for transfer · Effort **M**

- **Pitch.** Compose functions, tools or transforms the model already executes reliably (h = g∘f), and test on atoms held out of training.
- **Mechanism / seeds / verifier.** Nesting depth is the dial; add new *operator types*, not only depth. Execution gives the label.
- **Why it should work.** In [f(g(x))](https://arxiv.org/abs/2509.25123), RL on depth-2 compositions lifted unseen depth-3 accuracy from about 5% to about 30%, while RFT stayed ≤ 2.6% and atom-only RL stayed below 25% on Level 2. RL did not transfer to contexts with 0% or 0.1% pretraining exposure, but 1% exposure gave up to +60% pass@128 ([Interplay](https://arxiv.org/abs/2512.07783)). Adding operator types raised an 8-benchmark average by +10.66, and depth generalization fell to chance at about 3× the training depth ([ScaleLogic](https://arxiv.org/abs/2605.06638)). Composed training transfers down to the parts; the reverse does not ([He et al.](https://arxiv.org/abs/2609.19465)).
- **Effect / risk.** New compositional skill. Risk: transfer beyond string or DAG worlds is unproven; missing atoms need SFT or mid-training first.
- **First experiment.** 20 atoms at ≥ 90%; RL on depth-2 compositions of 15; evaluate depth 3–4 on the other 5. *Success:* held-out depth-3 accuracy clearly above an atoms-only RL control.

### B3 · Constraint stacking calibrated by p^k, with structure-aware rewards
`Established` · **Strong** · Effort **S**

- **Pitch.** Choose the number of constraints k so all-pass probability lands in the band, then add structure rather than count.
- **Mechanism / seeds / verifier.** Stacking plus chain and selection structure (conditions computable from the input), one checker per constraint, and an intent gate.
- **Why it should work.** Per-constraint success follows 72.0% × 0.922^(k−1) across 15 models ([CSE](https://arxiv.org/abs/2608.12426)). Training on 5–6 constraints beat up to 3 ([IFBench](https://arxiv.org/abs/2507.02833)). GPT-4 scores 0.626 on Selection+Chain at depth ≥ 3 and 14.9% on the coherent multi-layer Selection test ([ComplexBench](https://arxiv.org/abs/2407.03978)). Averaging instead of structure-aware aggregation cost 3.9 IFEval and 5 CFBench points ([LsrIF](https://arxiv.org/abs/2601.06431)).
- **Effect / risk.** A predictable dial. Risks: over-hardening is the default (10,772 prompts at pass rate 0 vs 7,324 in band, [IFDecorator](https://arxiv.org/abs/2508.04632)); conflicts ("output JSON" is jointly unsatisfiable with 9 of 24 instructions, note 08); hacking (IFDecorator's intent check cut the trip-wire hack rate from 14.53% to 7.60%).
- **First experiment.** Fit per-constraint success on your policy; generate at k with p^k ≈ 0.4; per-constraint reward with an all-pass guard vs binary. *Success:* higher in-band yield and gains on held-out constraint families.

### B4 · Evaluator-first and compatibility-checked chaining for agent tasks
`Established` · **Moderate** · Effort **M**

- **Pitch.** Compose trusted checkers, not instructions, and prove joint satisfiability before writing text.
- **Mechanism / seeds / verifier.** Sample 2–3 atomic checkers, reparameterize, build a golden state that satisfies all, then write the instruction ([UltraCUA](https://arxiv.org/abs/2510.17790)). For chains, score each ordered pair A → B for executability and beam-search ([ChainWorld](https://arxiv.org/abs/2606.21654)). Atoms need checker(golden) = 1 and checker(initial) = 0 ([CUA-Gym](https://arxiv.org/abs/2605.25624)).
- **Why it should work.** Evaluator-first tasks had 29% rollout success vs 45% for instruction-first; the best agent completes 31% of ChainWorld chains. Require data dependencies: chains of unrelated steps are "artificially hard" and transfer poorly (note 15).
- **Effect / risk.** Harder tasks with trusted checkers. Risk: checker interference; reject such pairs with hard rules before any LLM coherence judge.
- **First experiment.** 500 compositions of 2–3 verified atoms; RL on composed vs atoms only. *Success:* held-out app gains, reported per step and per chain.

### B5 · Feature-tree composition for code with dual-solution cross-verification
`Established` · **Moderate** · Effort **M**

- **Pitch.** Pick k compatible algorithmic features, then write the task; label with an efficient and a brute-force solution that must agree.
- **Mechanism / seeds / verifier.** Specification-space atom composition ([X-Coder](https://arxiv.org/abs/2601.06953)); differential testing on validator-checked inputs ([AutoCode](https://arxiv.org/abs/2510.12803)).
- **Why it should work.** X-Coder: 64k tasks × 1 solution beat 16k × 4; residual label error was 12.7% before a deterministic filter, 94% of it from one detectable pattern. [rStar-Coder](https://arxiv.org/abs/2505.21297) mutual verification: 96.8% label accuracy vs 12.7% for GPT-4o-written outputs. In AutoCode, measured difficulty gain tracked human-rated quality (up to 0.60), while o3–human agreement on quality was 0.07.
- **Effect / risk.** Diverse, verifiable hard code tasks. Risk: consensus labels share misconceptions; keep the brute-force channel independent.
- **First experiment.** Compose k = 2–4 features from your inventory; hand-audit 100 labels. *Success:* low audited label error and higher in-band yield than single-feature mutation.

### B6 · Realistic multi-bug and feature-addition SWE tasks
`Established` · **Moderate** · Effort **M**

- **Pitch.** Replace one-line synthetic bugs with bugs an agent introduces while adding features, and with combinations of individually validated bugs.
- **Mechanism / seeds / verifier.** FeatAdd agentic bugs ([BugPilot](https://arxiv.org/abs/2510.19898)); bug combination ([SWE-smith](https://arxiv.org/abs/2504.21798)) where the combined fail-to-pass set equals the union and bugs do not cancel.
- **Why it should work.** FeatAdd bugs touch 4.2 files and 415.9 net lines on average; Claude resolved 41.4% of them vs 65.9% of SWE-smith bugs; 1.2k FeatAdd/BugInstruct bugs beat 3k others by 2%. Asking an LLM "for a bug" collapses to one-line edits ([SSR](https://arxiv.org/abs/2512.18552)).
- **Effect / risk.** Harder, more realistic SWE data. Risk: GRPO on hard FeatAdd bugs did not beat SFT (it needs partial solvability); use SFT on the hardest tier and RL on the in-band subset.
- **First experiment.** FeatAdd bugs in 20 repositories vs SWE-smith-only at equal task count. *Success:* SWE-bench Verified and Pro gains.

### B7 · Heterogeneous chains: document hop → SQL hop → code hop
`Novel` · **Proposal** · Effort **L**

- **Pitch.** One composed task across modalities with end-to-end executable ground truth.
- **Mechanism / seeds / verifier.** Typed hops, each with an executor (retrieved span → SQL filter → Python aggregation); gold from executing the composite program; per-hop two-sided gate (masking any bridge must cause failure).
- **Why it is novel.** Note 07 calls such chains "barely explored". [DataMind](https://arxiv.org/abs/2509.25084) chains 2–5 analytic task types but labels by self-consistency, which its authors acknowledge biases toward easy compositions. QwenLong-L1.5's KG/SQL items and [Spreadsheet-RL](https://arxiv.org/abs/2605.22642) supply the parts.
- **Effect / risk.** Should transfer to data-analysis and research agents. Risks: realism and parametric leakage in the document hop (use post-cutoff or fictional documents, C6).
- **First experiment.** 300 three-hop items with per-hop masking ablations. *Success:* masking any hop defeats a strong solver on ≥ 90% of items, and RL on them beats single-modality chains on held-out heterogeneous tasks.

### B8 · Branch–merge topology curricula with merge-node rewards
`Novel` · **Proposal** · Effort **M**

- **Pitch.** Move beyond chains to diamonds and non-local dependencies, with exact checks at merge nodes.
- **Mechanism / seeds / verifier.** A DAG generator whose topology knob runs chain → tree → diamond → coupled; exact merge-node values supply partial or process reward.
- **Why it is novel.** Branch–merge and non-local dependencies still degrade sharply after composed training, and which data fixes this is open ([He et al.](https://arxiv.org/abs/2609.19465); note 10).
- **Effect / risk.** Targets the non-local coordination that chains never exercise. Risk: merge-node rewards invite partial-completion hacking; keep the final answer as the gate.
- **First experiment.** Chain-only vs a topology-mixed curriculum with merge-node rewards at matched compute. *Success:* diamond and coupled accuracy on held-out topologies rises without losing chain accuracy.

---

## 4. Theme C: Obfuscation and inversion engines

These keep the answer and change what the model is shown or asked. Their main risk is ambiguity, so each needs a uniqueness or sufficiency check.

### C1 · Fuzz, densify and disperse search questions, then repair uniqueness
`Established` · **Strong** · Effort **M**

*Pitch:* turn 1–2-hop QA into questions that need long, real search.
- **Mechanism / seeds / verifier.** On graph-derived QA with a stored gold chain: fuzz exact values (dates to ranges, numbers to arithmetic constraints); densify topology (cycles, higher treewidth; [WebSailor-V2](https://arxiv.org/abs/2509.13305)); disperse evidence so no page covers several constraints ([REDSearcher](https://arxiv.org/abs/2602.14234)). After every edit, enumerate candidates in the KB or by search to confirm uniqueness.
- **Evidence.** In [FORT](https://arxiv.org/abs/2606.12087)'s cumulative ablation, removing shortcut controls raised a strong agent's accuracy from 29.0% to 81.6%, with fuzzing the most important. FORT questions needed 141.0 retrieval calls vs 20.6 for InfoSeek's "deep" trees, so accept by trajectory signature, not graph depth. FORT-Searcher (SFT-only) reaches BrowseComp 72.2. The labs converge on the same operators (note 12).
- **Effect / risk.** Much higher realized search cost. Risks: uniqueness checks are "not a formal guarantee" (REDSearcher), and live-web answers drift; snapshot the evidence.
- **First experiment.** Harden 2k seeds; each must fail closed-book, be solvable from gold pages, and pass uniqueness enumeration. *Success:* calls-to-answer and answer-hit time rise without a rise in audited ambiguity.

### C2 · Direction inversion: backward math and abductive/inductive code
`Established` · **Moderate** · Effort **S–M**

*Pitch:* one verified artifact yields harder inverse tasks.
- **Mechanism / seeds / verifier.** *Math:* mask a given and ask for it from the answer ([MetaMath](https://arxiv.org/abs/2309.12284) FOBAR/SV; "if the answer is 42, what is z?", [2605.28388](https://arxiv.org/abs/2605.28388)); brute-force the domain for uniqueness (the trailing-zeros inversion has five solutions, 100–104, so ask for the smallest). *Code:* abduction (input for a given output), induction (function from I/O) ([Absolute Zero](https://arxiv.org/abs/2505.03335)), or tests that separate right from wrong code ([CURE](https://arxiv.org/abs/2506.03136)); execution grades.
- **Evidence.** 20K added MetaMath samples gave +2.3 (FOBAR) and +2.6 (SV) GSM8K points vs +0.4 for rephrasing. Caveats: inverse physics questions transferred less than forward ones (5.84 vs 13.15 IPhO, [Sim2Reason](https://arxiv.org/abs/2604.11805)), and AZR-style self-play mostly sharpens (note 09).
- **Effect / risk.** Backward reasoning and variety at no labelling cost. Risk: non-unique inverses and weaker transfer than forward items; keep inversions a minority.
- **First experiment.** Invert 1k seeds with uniqueness enforcement as ≤ 30% of the mix. *Success:* higher in-band yield, no loss in forward accuracy.

### C3 · Static → interactive: put the parameters behind tools
`Established` (recent) · **Emerging** · Effort **M**

*Pitch:* keep a solved instance but make the model acquire its facts.
- **Mechanism / seeds / verifier.** Pre-solve, expose parameters only through query tools, grade arithmetically ([VHD-Play](https://arxiv.org/abs/2609.27321); [CodeGym](https://arxiv.org/abs/2509.17325), 10–256 tool calls; [KUMO](https://arxiv.org/abs/2504.02810), SAT-backed hidden-state games).
- **Evidence.** The same solved instances score 0.962 written out and 0.204 behind tools (VHD-Play).
- **Effect / risk.** Large pass-rate drop at zero labelling cost. Risk: tedium rather than difficulty; also score efficiency against an optimal searcher.
- **First experiment.** Agentify 5 saturated procedural families. *Success:* band restored and tool-use benchmarks improve.

### C4 · Shard the instruction or hide the spec behind a user, with a CONCAT control
`Established` · **Moderate** · Effort **M**

*Pitch:* same label, harder delivery.
- **Mechanism / seeds / verifier.** Reveal a fully specified task one shard per turn, or move information into a simulated user who answers only from the spec; reuse the original verifier; drop items whose CONCAT view (all shards in one turn) fails; add an explicit reward for asking or abstaining.
- **Evidence.** SHARDED loses 39% on average across 15 LLMs while CONCAT keeps 95.1% of FULL ([Lost in Conversation](https://arxiv.org/abs/2505.06120)). A shard curriculum gated at ρ = 0.8 reached 71.9 LiC in 90 steps, vs 63.2 without curriculum and 40.9 at ρ = 0.6 ([RLAAR](https://arxiv.org/abs/2510.18731)). Removing information categories from SWE-bench Verified issues drops resolution from 43.8% to 23.7%; learned clarification recovers 36.8% with 3.0 questions ([CLARITI](https://arxiv.org/abs/2604.14624)).
- **Effect / risk.** Restores GRPO variance on saturated seeds. Risks: simulator leakage makes the task easy again; degenerate always-ask or always-abstain policies; cap unsolvable variants (RLAAR m = 0.1).
- **First experiment.** Shard 2k saturated tasks; train with the ρ = 0.8 curriculum. *Success:* sharded/full ratio improves with no single-turn regression.

### C5 · Counterfactual rules and worlds
`Established` (evaluation) / `Extension` (training) · **Emerging** · Effort **S**

*Pitch:* change the world's semantics so priors mislead.
- **Mechanism / seeds / verifier.** Represent the seed as a symbolic program, mutate a rule in code, execute for the new answer: base-9 arithmetic ([Reasoning or Reciting](https://arxiv.org/abs/2307.02477)), reinterpreted operators ([MatheMagic](https://arxiv.org/abs/2510.05962)), knights who lie, new Sudoku interactions ([Sudoku-Bench](https://arxiv.org/abs/2505.16135): SOTA solves < 15% unaided).
- **Evidence / risk.** Strong as an evaluation operator; little evidence as training data. Risk: overfitting to a "counterfactual" style.
- **First experiment.** Counterfactual twins of 3 procedural families. *Success:* held-out counterfactual accuracy rises with no drop on standard versions.

### C6 · Fictional-world grounding and answer-source removal
`Established` · **Moderate** · Effort **M**

*Pitch:* the answer exists only in a world the model cannot have memorized.
- **Mechanism / seeds / verifier.** A self-consistent fictional DB or wiki with answers computed by SQL or Prolog ([PhantomWiki](https://arxiv.org/abs/2502.20377)); or delete the answer's source page ([LiteResearcher](https://arxiv.org/abs/2604.17931)) and re-verify reachability (solver pass rate between 1/8 and 7/8).
- **Evidence.** Fictional SQL-backed worlds gave +16.29 WideSearch item-F1 ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)); GRPO on PhantomWiki gave Qwen3-0.6B relative F1 gains of 56–131% on five real multi-hop benchmarks. [SynthWorlds](https://arxiv.org/abs/2510.24427) documents the parametric "knowledge advantage gap" this removes.
- **Effect / risk.** Contamination-free, regenerable RL data. Risk: templated worlds are repetitive; measure diversity (I4).
- **First experiment.** Regenerate the world every epoch. *Success:* gains on real multi-hop and search benchmarks.

---

## 5. Theme D: Environment and agentic hardening

For agentic seeds, much of the difficulty lives in the environment and the verifier rather than the prompt. [Chapter 07b](07b-domain-recipes-agents-and-beyond.md) has the per-domain recipes.

### D1 · Recursive solution-first escalation (RST) for executable tasks
`Established` · **Moderate** · Effort **M**

*Pitch:* grow the reference solution first, then realign the environment, the verifier and, last, the instruction; reseed with the accepted children.
- **Mechanism / seeds / verifier.** Each round extends `solve.sh`, realigns the environment, extends the verifier, then rewrites the instruction. Gates: fresh-sandbox oracle pass; contract validity; minimum deltas (≥ 3 files, ≥ 8 solution lines, ≥ 12 verifier lines); instruction ≤ 180 words and ≤ 1.6× the seed. Reseed under diversity caps.
- **Evidence.** Over 15 rounds from 639 seeds, DeepSeek-V4-Pro pass@4 fell from 90% to 2.5%, with about 50% of attempts accepted each round at about $0.05 per task and median assertions growing from 17 to 57 ([RST](https://arxiv.org/abs/2608.05466)). On this pool GRPO stayed flat at 51.7%, while PPO with a warm-started critic and reward r = P/20 reached 64.0% ([T1](https://arxiv.org/abs/2609.11042)). LLM rewrites of task *descriptions* did not beat untouched ones (OpenThoughts-Agent, note 16).
- **Effect / risk.** Compounding verified difficulty at low cost. Risks: realism drift (nearest-neighbour similarity 0.22 → 0.46 over rounds) and unknown exploitability by round (J5). Plan for dense rewards.
- **First experiment.** 3 rounds on 200 seeds; assertion-count reward on round-2/3 tasks vs the original seeds. *Success:* held-out terminal-benchmark gains, pass rate falling per round, acceptance around 50%.

### D2 · RST for workbooks, notebooks and documents
`Novel` · **Proposal** · Effort **M–L**

*Pitch:* the published gap most directly exploitable by a team with easy, execution-verified office seeds.
- **Mechanism / seeds / verifier.** Each round grows the reference edit program (add a sheet, deepen formula chains, add a pivot or chart), recomputes in real Excel, verifies on 3–5 perturbed input variants, checks that formulas rather than pasted values exist, applies preservation predicates, and reseeds. For notebooks: extend the executed code, re-execute for new gold, then abstract the instruction.
- **Why novel.** Note 16 finds no such pipeline, though every part exists: [Spreadsheet-RL](https://arxiv.org/abs/2605.22642) (SpreadsheetBench 12.0% → 23.4% for Qwen3-4B-Thinking), multi-test-case [SpreadsheetBench](https://arxiv.org/abs/2406.14991), [WTM](https://arxiv.org/abs/2608.07873) inversion and specificity levels (scores collapse at ≥ 5 transformations), and [DocOps](https://arxiv.org/abs/2607.19865) preservation predicates. [LongDS](https://arxiv.org/abs/2605.30434)'s 47-point late-turn drop motivates multi-turn counterfactual and rollback sessions with per-turn replayed gold.
- **Risk.** LibreOffice and headless engines diverge from Excel, and 88.5% of WTM failures are placement and shape; check them explicitly.
- **First experiment.** 100 workbook seeds × 4 rounds. *Success:* monotone pass-rate decline at ≥ ~50% acceptance and zero passes from hard-coded values on perturbed variants.

### D3 · Logged dirty-data injection with gold computed on the clean source
`Novel` (for RL) · **Proposal** · Effort **M**

*Pitch:* same question, messier data, exact gold.
- **Mechanism / seeds / verifier.** Corrupt a clean table in logged ways (duplicates, mixed date formats, cents vs dollars, nulls) and add genuine outliers that must *not* be removed; compute gold on the clean source under stated, discoverable cleaning rules.
- **Why novel.** Note 16 finds label-preserving corruption only in evaluation and cleaning studies, where profiling baselines beat LLM agents at detection (F1 0.561 vs 0.421). Data tasks do respond to synthetic training: [DSGym](https://arxiv.org/abs/2601.16344)-SFT (2k execution-verified queries) took Qwen3-4B from 2.9% to 33.07% on DABStep-hard. [SandMLE](https://arxiv.org/abs/2604.04872)-style micro-scale data keeps rollouts more than 13× faster.
- **Effect / risk.** Robustness to messy data, where agents trail simple profilers. Risk: unstated cleaning conventions turn difficulty into ambiguity.
- **First experiment.** 5 corruption types × 3 intensities on 300 analysis tasks. *Success:* pass rate falls monotonically with intensity, and real dirty-data evaluations improve.

### D4 · Steered simulator perturbations, noise curricula and infeasible variants
`Established` · **Moderate** · Effort **M**

*Pitch:* inject realistic adversity while the goal and end-state checker stay fixed.
- **Mechanism / seeds / verifier.** Intermittent errors, pagination, partial results and withheld answers ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)); a tool-failure budget including silent value corruption, with a "stop" reward when every path is blocked ([BENCH2ROBUST](https://arxiv.org/abs/2608.11977)); GUI pop-ups mid-task ([AnTrap](https://arxiv.org/abs/2608.24099)); infeasible tasks true by construction ([ZeroGUI](https://arxiv.org/abs/2505.23762)); randomized agent harnesses ([Kimi K3](https://arxiv.org/abs/2607.24653)). Keep state in code or a database and let the LLM render only surface text; EnvSimBench found a cliff once several state variables change at once (note 17).
- **Evidence.** Uncontrolled simulated RL gave nothing (Tool Decathlon 32.4 → 31.5); instructed perturbations gave +3.7 Tool Decathlon and +12.3 MCPMark. LongCat-2601's noise curriculum raised noisy benchmark versions without hurting clean ones. Removing ZeroGUI's infeasible tasks dropped infeasible-subset success from 41.3 to 22.1.
- **Risk.** Perturbations can make tasks unsolvable (AnTrap: 91% passed its solvability audit), and policies can learn to refuse by default; balance the infeasible fraction.
- **First experiment.** 4 perturbation types × 3 levels on 500 tasks, clean and noisy scores side by side. *Success:* noisy scores rise, clean scores do not fall.

### D5 · State inversion: break a healthy environment
`Established` · **Moderate** · Effort **M**

*Pitch:* the original good state is the oracle.
- **Mechanism / seeds / verifier.** An agent degrades a working environment (mis-pinned dependency, missing environment variable, broken permissions) until tests fail ([CLI-Gym](https://arxiv.org/abs/2602.10999)); for workbooks, strip derived artifacts in dependency order ([WTM](https://arxiv.org/abs/2608.07873)). Expose only symptoms, forbid leftover backups, and filter recoveries that use cached git or conda state.
- **Why it should work.** CLI-Gym's degraded-environment tasks (1,655 from 29 repositories) gave LiberCoder +21.1 on TB1.0 and +12.9 on TB2.0 from 291 curated trajectories, and WTM scores collapse once ≥ 5 derived artifacts must be rebuilt.
- **Effect / risk.** Diagnosis-heavy tasks with free labels. Risk: degradation residue reveals the fix.
- **First experiment.** Degrade 200 containers with 1–3 faults each. *Success:* no-op fails, oracle passes, in-band yield, no recovery through a cache.

### D6 · Phase-state chaining for long workflows
`Established` (recent) · **Emerging** · Effort **L**

*Pitch:* build long, coupled workflows from validated phases.
- **Mechanism / seeds / verifier.** Each phase has a checker; phase k's validated end state becomes phase k+1's start state ([Qwen-CUA](https://arxiv.org/abs/2608.02352)); screen chains with ChainWorld's compatibility rules; train end to end on the sum or conjunction of phase checkers. Later checkers must not depend on artifact IDs created by one particular earlier solution.
- **Evidence.** Qwen-CUA reaches 86.2 on OSWorld-Verified with about 40,000 tuple-format tasks. Note 15 names phase-state chaining plus compatibility checks as the most promising route to verified 100–500+-step RL tasks, which no open pipeline yet produces at scale. Lengthening horizons alone destabilizes RL; reduce them first with macro-actions and subgoals ([2605.02572](https://arxiv.org/abs/2605.02572)).
- **Effect / risk.** Long, coupled workflows with state-grounded partial credit. Risks: long horizons amplify checker false positives, so strengthen checkers (note 15).
- **First experiment.** 4-phase chains in 3 applications. *Success:* full-chain success on held-out applications; compare sum vs conjunction rewards.

### D7 · Co-harden capability and security, with benign twins
`Established` (recent) · **Emerging** · Effort **M**

*Pitch:* adversarial content in the untrusted slots of capability environments is a durable difficulty source.
- **Mechanism / seeds / verifier.** Inject attacker text into tool outputs, emails and files at randomized reachable positions; reward = task completed AND injection resisted, from state (e.g. R_task − R_injected, [ToolHazard](https://arxiv.org/abs/2608.11878)); a population of adaptive attackers ([GPT-Red](https://arxiv.org/abs/2607.26115)); benign twins against over-refusal.
- **Evidence.** [CoER](https://arxiv.org/abs/2609.07529) raised utility from 63.2% to 76.3% while attack success fell from 38.5% to 0.2% (its own evaluation). Removing [IH-Challenge](https://arxiv.org/abs/2603.10521)'s anti-over-refusal split dropped the over-refusal score from 0.950 to 0.831 and helpfulness from 0.773 to 0.613. Adaptive attacks broke all 8 evaluated defenses at > 50% attack success.
- **Effect / risk.** Durable difficulty that does not saturate like static attack templates. Risk: over-refusal without benign twins.
- **First experiment.** Inject into 500 capability tasks. *Success:* utility holds, attack success falls against a *held-out* adaptive attacker, over-refusal stays flat.

### D8 · Goal switch: closed → open-ended optimization, and correctness-gated performance
`Established` · **Moderate** · Effort **M**

*Pitch:* when pass@1 is 100%, keep correctness as a gate and reward quality continuously.
- **Mechanism / seeds / verifier.** Alter the goal, constrain outputs or generalize inputs (MST → degree-constrained spanning tree) and score against a baseline in [0, 1] ([FrontierSmith](https://arxiv.org/abs/2605.14445)); for code, reward speed behind a correctness gate with milestone shaping ([CUDA Agent](https://arxiv.org/abs/2602.24286)), hidden multi-distribution inputs, synchronized timing and replay on your hardware.
- **Evidence.** With 200 problems, GRPO gave +8.82 FrontierCS and +306 ALE-bench rating on Qwen3.5-9B, beating a closed-ended HardTests control by +5.24 / +236.40. CUDA Agent's milestone reward {−1, 1, 2, 3} made 96.8% of kernels faster than torch.compile vs 60.4% with raw speedup.
- **Risk.** The most-hacked reward in the literature: under hidden inputs GPT-5.5's apparent 1.43× speedup is 0.88×, and the official KernelBench check misses 16.9% of injected faults ([Measuring the Checker](https://arxiv.org/abs/2609.22220)). *Novel extension:* reward only "faster **and** provably equivalent", via differential binary verification ([AsmEvo](https://arxiv.org/abs/2608.20711)) or refinement-type proofs ([Semantic-equivalence self-play](https://arxiv.org/abs/2604.17010)); note 22 finds this nearly absent.
- **First experiment.** 200 saturated algorithm problems → open-ended variants. *Success:* reward variance persists and held-out optimization benchmarks improve.

### D9 · Formal-verification lift, the stage/language ladder, and band proposers
`Established` · **Moderate** · Effort **M–L**

*Pitch:* turn saturated tested code into "implement + prove", graded soundly on the solution side.
- **Mechanism / seeds / verifier.** Lift the seed into a spec validated by its tests (soundness and completeness lemmas, mutated-output spectests); climb the stage ladder (code → spec → proof → end-to-end) and the language ladder (Dafny → Verus → Lean); use band-rewarded proposers (ANCORA: reward when 1 of K attempts verifies; KernelZero: 1 − 2|p − 0.5|).
- **Evidence.** The [ATLAS](https://arxiv.org/abs/2512.10173) TACO lift succeeds on 47% of EASY seeds and about 20% of HARD ones. [VeriContest](https://arxiv.org/abs/2605.08553): NL→code 92.18% vs end-to-end 5.29%; [AlgoVeri](https://arxiv.org/abs/2602.09464): Dafny 40.3% vs Lean 7.8%. [PSV](https://arxiv.org/abs/2512.18160) reaches 65.63% vs 34.46% for plain RFT, and dropping solution verification costs 51.5% relative. Specs get gamed (`assume(false)` spread from one program to all in [AlphaVerus](https://arxiv.org/abs/2412.06176)); use at least two spec defenses.
- **Effect / risk.** A sound, non-saturating check on the solution side. Risk: the attack surface moves to the spec; budget for spec hardening.
- **First experiment.** Lift 1k saturated functions; train along the stage ladder. *Success:* higher verified rate on held-out specs with a clean spec-strength audit.

---

## 6. Theme E: Trained generators and self-play

Build these once static operators stop filling the band. They follow the stability rules in [Chapter 05 §5](05-rl-playbook.md): a validity gate independent of the solver, external grounding, a persistent diversity archive, and a grader the loop never trains.

### E1 · Validity-gated, band-rewarded setter RL with real anchors
`Established` · **Moderate** · Effort **L**

*Pitch:* train the generator on "valid × in band", never on "the solver failed".
- **Mechanism / seeds / verifier.**
  - Setter reward: 1[V]·(1 − Acc) ([VHG](https://arxiv.org/abs/2605.06660)), or a band reward targeting [0.4, 0.6] ([D2Evo](https://arxiv.org/abs/2605.17037)).
  - V is independent of the solver: execution, a symbolic check, or a verifier from a different model family.
  - Mix real mid-difficulty anchors into every round, and keep a novelty archive keyed on solution signatures.
- **Evidence.**
  - VHG: validity is learned first (30.6% → 75.5%), then difficulty (the valid-and-hard share rises from 27.5% to 58.5%).
  - Lowering [OpenSIR](https://arxiv.org/abs/2511.00602)'s solve-rate floor from 0.5 to 0.1 collapsed validity from 70.8% to 42.3%.
  - D2Evo, with 1.7K real samples, beat full-data RL on 19K (55.32 vs 52.70, Qwen3-8B-Base).
  - [R-Zero](https://arxiv.org/abs/2508.05004)'s pseudo-label accuracy fell from 79% to 63%, and its math score peaked then fell (49.12 → 46.52).
- **Risk.** Collapse after 3–4 rounds. Ambiguous items can land in the band through noise.
- **First experiment.** At the same token budget, compare a prompted generator with a trained 7B setter. Log valid yield, band yield and label accuracy on a gold holdout every round. *Success:* more valid in-band tasks per GPU-hour, with label accuracy stable for ≥ 3 rounds.

### E2 · SvS: variants built from the policy's own correct solutions
`Established` · **Moderate** · Effort **M**

*Pitch:* spawn harder, answer-preserving variants from what the policy barely solves.
- **Mechanism / seeds / verifier.** For items solved 12.5–50% of the time, give the policy one of its correct solutions and ask for structurally different problems with the same answer. Reward only variants whose solve rate lies in [12.5%, 62.5%], and screen for leaked answers.
- **Evidence.** +18.3 and +22.8 pass@32 on AIME24 and AIME25, with entropy stable ([SvS](https://arxiv.org/abs/2508.14029)). The naive reward "the variant is solvable" gets gamed by leaking hints.
- **Effect / risk.** Expands the boundary (large-k pass@k), not only pass@1. Risk: answer or hint leakage into variants.
- **First experiment.** Add SvS to an existing GRPO run. *Success:* large-k pass@k beats the baseline, and entropy does not collapse.

### E3 · Goal-anchored stepping stones for pass@k = 0 targets
`Established` (recent) · **Emerging** · Effort **L**

*Pitch:* bridge to real problems the policy cannot yet solve.
- **Mechanism / seeds / verifier.** [GASP](https://arxiv.org/abs/2603.15957) generates an easier lemma (p ∈ [0.3, 0.7]), then a harder lift of it (p ∈ [0.1, 0.5]), then returns to the goal. [SOAR](https://arxiv.org/abs/2601.18778) instead rewards a teacher by the student's measured improvement on a fail@128 set, which the teacher never sees.
- **Evidence.** GASP solved 11 of 146 coding problems at pass@100 = 0, where AZR and standard RL solved none. SOAR gave 4× pass@1 and 2× pass@32 on MATH-hard. Only 32.8% of SOAR's useful questions had fully correct solutions: bridge tasks need to be well-posed more than correct.
- **Effect / risk.** The only route here aimed directly at pass@k = 0 targets. Risks: inner-loop cost, and contamination if the teacher sees the goals.
- **First experiment.** Take 100 real items at pass@64 = 0. *Success:* more items unlocked (pass@k > 0) than standard RL at equal compute.

### E4 · Anchored bug-injection self-play for SWE
`Established` · **Emerging** · Effort **L**

*Pitch:* one policy both breaks and repairs repositories, anchored to real bugs.
- **Mechanism / seeds / verifier.** Inject by removing hunks, reverting commits, or layering higher-order faults from failed repairs. Check with inverse mutation testing and give −1 for inconsistent artifacts ([SSR](https://arxiv.org/abs/2512.18552)). Anchor with an embedding-similarity reward toward real bugs (λ = 0.20) and 20% reference bugs in the fixer's training data ([Anchored Self-Play](https://arxiv.org/abs/2607.03523)).
- **Evidence.** SSR gained +10.4 on SWE-bench Verified and +7.8 on SWE-Bench Pro, but showed gibberish instability, and its generated issues copied the test patches. Unanchored self-play later degraded on human bugs; anchoring gave +7.0 pp on average (+3.4 pp on human bugs).
- **Effect / risk.** Unlimited SWE tasks without human issues. Risks: gibberish instability, issue texts that leak the test patch, drift from human bugs without anchoring.
- **First experiment.** Anchored vs unanchored self-play on 50 repositories. *Success:* the fix rate on human bugs rises round after round.

### E5 · Open replication of a learned (problem, environment, verifier) constructor
`Novel` (replication) · **Proposal** · Effort **L**

*Pitch:* test the frontier claim that a trained task constructor drives the gains.
- **Mechanism / seeds / verifier.** Train the constructor on reward = difficulty × correctness of the whole triplet. Use the pipeline build → self-test → leak-scrub → distinct solvers → inspector → repair, and re-audit the tasks at every RL run.
- **Why novel.** [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) (§5.1.1) attributes "essentially all of the observed gains" to data and environment pipelines, but gives no reward formula and no ablation. The cheapest prompted baseline is [RLAnything](https://arxiv.org/abs/2602.02488): when accuracy exceeds 0.8, ask for a harder rewrite and accept it iff 0.2 < acc(q′) < acc(q).
- **Effect / risk.** Potentially the largest lever if the frontier claim holds. Risk: the constructor games its own correctness scorer; keep an independent inspector.
- **First experiment.** A three-arm matched-compute ablation: trained constructor, RLAnything-style prompted constructor, static pool. *Success:* the trained arm wins on validated in-band triplets per GPU-hour *and* on held-out gains. Kill it if it does not beat the prompted arm.

---

## 7. Theme F: Verification infrastructure

Harder tasks raise both label error and hackability ([Chapter 04](04-verification-and-quality-control.md), [Chapter 09](09-pitfalls-and-failure-modes.md)); this infrastructure makes Themes A–E safe.

### F1 · A task-bundle contract with CI gates
`Established` · **Strong** · Effort **S**

*Pitch:* no task enters a pool without an oracle, a self-scoring test and known-bad candidates.
- **Mechanism.** The [Harbor](https://docs.harborframework.com/core-concepts/tasks/overview) layout (`instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh` → `/logs/verifier/reward.txt`). CI: oracle passes in a fresh sandbox; no-op fails; known-bad candidates fail; checkers hidden from proposers and agents; clean forbidden-pattern scan.
- **Evidence.** [CUA-Gym](https://arxiv.org/abs/2605.25624) requires reward(golden) = 1 and reward(initial) = 0 with the checker written behind an information barrier; its models reach 62.1 / 72.6 on OSWorld-Verified. [Self-Challenging](https://arxiv.org/abs/2506.01716) ships a known-good solution and failure cases with every task. An empty-response agent passes 38% of τ-bench-Airline ([ABC](https://arxiv.org/abs/2507.02825)). In the Darwin Gödel Machine, objective hacking was more frequent when checkers were visible (note 14).
- **Effect / risk.** Makes every later operator cheap to validate. Risk: endpoint tests do not prove that alternative valid solutions pass (note 15); add alternative-solution probes.
- **First experiment.** Convert the pool; report failure rate per gate. *Success:* every accepted task passes the contract, and the shift in the pass-rate profile is documented.

### F2 · A pre-RL red-team harness, including a public/hidden verifier split
`Established` (components) · **Moderate** · Effort **M**

*Pitch:* attack every operator family before training on it.
- **Mechanism.** Hacker–fixer–solver loop ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)); trivial-agent baselines and planted canaries; impossible variants with an abort option ([ImpossibleBench](https://arxiv.org/abs/2510.20270)); isomorphic-renamed twins rewarded only if both are right ([IPT](https://arxiv.org/abs/2604.15149)); truncated "master key" negatives for judges; read-only checksummed tests; a public diagnostic verifier with hidden scoring under a submission budget ([Kimi K3](https://arxiv.org/abs/2607.24653)).
- **Evidence.** 323 of 1,968 terminal tasks (16%) were hackable from the description alone, and the loop cut attack success on KernelBench from 76% and 61% to 0%. GPT-5 exploited tests in 54–76% of impossible SWE-bench variants; an abort option cut this from 54% to 9%. RLVR models found 40 shortcuts at complexity levels 1–10 vs 458 at 11–20. Monitors still miss about 35–58% of cheating on complex SWE variants.
- **Effect / risk.** Stops exploits before they become learned behavior; hacks learned in coding RL generalized to broad misalignment (note 11). Risk: overfitting to known exploit classes; rotate hackers.
- **First experiment.** Harness 200 tasks per operator family. *Success:* post-fix exploit rate near zero with the solver still passing.

### F3 · A decorrelated verification portfolio with J tracking
`Established` · **Moderate** · Effort **M**

*Pitch:* hard items need evidence from channels that fail differently.
- **Mechanism.** Execution, small-input brute force, a verifier from another model family and, where possible, a formal kernel; measure TPR and FPR per channel per difficulty decile on a gold-labelled subset.
- **Evidence.** With J = TPR − FPR > 0 incorrect modes die out; with J < 0 they grow ([RLVεR](https://arxiv.org/abs/2601.04411)). Within-group verifier errors correlate at 0.53, so 8 completions are worth about 1.70 independent votes; cross-family channels add 0.0913 marginal information vs 0.0126 for same-model repeats ([VStress](https://arxiv.org/abs/2609.36958)). Unverified majority answers are wrong on 25.85% of MATH-500, 46.07% of AMC and 73.33% of AIME 2024 ([T³RL](https://arxiv.org/abs/2603.02203)). Under count-matched RFT, admitting 25% false positives costs 1.58 pp while discarding 75% of true positives costs 0.03 pp (note 22): tune gates for precision.
- **Effect / risk.** Keeps J > 0 on the hardest items. Risk: cost grows with channels; add one only while hardest-decile J improves.
- **First experiment.** 300 gold-labelled hard items stratified by difficulty. *Success:* portfolio J on the hardest decile clearly exceeds the best single channel.

### F4 · Two-sided (privileged-information) solvability gates
`Established` · **Strong** · Effort **S**

*Pitch:* separate "hard" from "broken" before a p ≈ 0 item enters training.
- **Mechanism.** Admit only if solvable with privileged information (hint, gold evidence, stronger teacher, large k) and unsolved without it.
- **Evidence.** Hint-conditioned admission ([CLI-Universe](https://arxiv.org/abs/2606.22883)); pass@100 > 0 ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)); pass@8 = 0 with pass@512 ≫ 0 from a verified-answer pool ([GLM-4.5](https://arxiv.org/abs/2508.06471)). Of 79 [EvoEnv](https://arxiv.org/abs/2605.14392) environments passing all mechanical layers, a stronger auditor judged 35 buggy; [HLE-Verified](https://arxiv.org/abs/2602.13964) kept only 668 of 2,500 items unchanged.
- **Effect / risk.** Keeps label errors out of the p ≈ 0 tail, where they concentrate. Risk: difficulty is bounded by the privileged solver's ability.
- **First experiment.** Gate the current p = 0 tail; hand-audit 100 accepts and 100 rejects. *Success:* rejects mostly broken, accepts low in label error.

### F5 · Solver-backed label-preservation certificates outside math
`Novel` · **Proposal** · Effort **M**

*Pitch:* certify that a rewrite kept the label by editing in an executable IR, not by asking an LLM.
- **Mechanism.** Lift the seed into a program, SQL query, state machine or tool trace; apply the operator there (shard, obfuscate, abstract); re-render; certify by re-execution equivalence plus cycle consistency; check sufficiency with a QuestBench-style solver.
- **Why novel.** [Lost in Conversation](https://arxiv.org/abs/2505.06120) needed 1–4 hours of manual work per task, CLARITI validated 50 of 1,500 rewrites, and MQR and SvS audit preservation with LLMs. Note 21 lists solver-backed CONCAT equivalence and shard-necessity checks for code, SQL and tool tasks as missing; the math analogue works (MathCAMPS cycle consistency: 97.7% of survivors faithful).
- **Effect / risk.** Removes the manual or LLM-audit bottleneck for interaction and obfuscation operators. Risk: lifting natural text into an IR can itself lose information.
- **First experiment.** IR-certified vs LLM-audited rewrites on 500 SQL-backed tool tasks, hand audit as ground truth. *Success:* lower label error at equal yield.

---

## 8. Theme G: Difficulty calibration and curriculum

[Chapter 05](05-rl-playbook.md) covers the mechanics. The ideas here are the build items.

### G1 · A per-family online difficulty controller
`Established` · **Strong** · Effort **S–M**

*Pitch:* replace static "hard sets" with generators whose knob follows the policy.
- **Mechanism.** Give each family an integer knob and a sliding window: promote when top-level accuracy is ≥ 0.9, with τ_num = 8 × rollouts and a window of 4 ([RLVE](https://arxiv.org/abs/2511.07317)). Alternatives are a proportional controller ([SCALER](https://arxiv.org/abs/2601.04809)) or "raise κ if mean reward > 0.5" ([InternGeometry](https://arxiv.org/abs/2512.10534)).
- **Evidence.** RLVE gave +3.37 on an already-saturated model vs +0.49 from more than 3× the compute of continued RL; a static low-cap range drove the effective-prompt ratio to 0. SCALER reached 54.25 vs 53.52 for RLVE. CBRL scored 44/50 on IMO-50 vs 38 without the schedule. Breadth pays: 400 environments × 40 instances beat 25 × 640 (75.19 vs 71.20, [ReSyn](https://arxiv.org/abs/2602.20117)), and selecting environments by ability coverage beat using all of them ([AES](https://arxiv.org/abs/2608.03571)).
- **Effect / risk.** Keeps every family at the edge of competence all run. Risk: size knobs mostly add length (note 05); pair them with structural knobs (A9, B2).
- **First experiment.** Wrap your 10 largest seed families. *Success:* effective-prompt ratio stays up and held-out scores improve.

### G2 · Route every prompt by pass rate: complexify, train or scaffold
`Established` (components) / `Novel` (unified router) · **Moderate** · Effort **M**

*Pitch:* one controller decides, per prompt, which operator to apply next.

- **Mechanism.** Route each prompt on its current p̂: harden when saturated, train in band, gate then scaffold at zero. Every child re-enters through the admission gates.

```python
def route(item, p_hat, history):
    if p_hat >= 0.9:            # saturated: harden with the operator that has the best
        op = pick_operator(item, yield_log, priors)   # logged yield for this family (I3, G3)
        enqueue_variant(op(item))                     # child goes through the admission gates
        return train_with_failure_prefix(item)        # meanwhile, extract residual signal (A7)
    if 0 < p_hat < 0.9:
        return train(item)       # in band
    if p_hat == 0:
        if not two_sided_gate(item): return quarantine(item)   # probably broken (F4)
        return scaffold(item, ladder=["format", "prefix", "teacher_in_prompt"])  # G6
```

- **Evidence.**
  - [SETA](https://arxiv.org/abs/2607.10891) increases difficulty when r > 0.5, shifts context when 0 < r ≤ 0.5 and decreases it when r = 0. 77% of decreases and 60% of increases moved the pass rate as declared.
  - [QbQ](https://arxiv.org/abs/2608.01522) seeded from 8–15/16 and reached 16.46% AIME pass@1, against 11.36% when seeding from the hardest items.
  - Note 02 records that no public system unifies these three routes.
- **Effect / risk.** Keeps the pool in band without manual passes. Risk: p̂ from small groups is noisy; use pilots and deferral (G4).
- **First experiment.** Compare the router with "filter only" at equal compute. *Success:* a higher in-band share per batch and a held-out gain.

### G3 · Operator-conditioned difficulty priors
`Novel` · **Proposal** · Effort **M**

*Pitch:* predict a child's pass rate from its parent's pass rate and the operator's measured effect, and skip most profiling.
- **Mechanism.** Maintain a Beta prior per (operator, parent band, domain), update it with a 4–8-rollout pilot, and commit rollouts only when the posterior is plausibly in band.
- **Why.** Profiling cost about 36% of rollout tokens in note 18's worked example, and existing predictors are weak: pre-rollout agentic predictors reach ρ = 0.399 in distribution and 0.225 on unseen benchmarks ([2608.05797](https://arxiv.org/abs/2608.05797)); [PROPEL](https://arxiv.org/abs/2606.18284) probes reach 0.59–0.66 balanced accuracy; similarity-based priors break for adversarial variants. Operator effects, by contrast, are large (information removal: −70 to −100 points). Note 18 proposes exactly this.
- **Effect / risk.** Could remove most profiling rollouts. Risk: adversarial or insight-hiding operators break smooth priors; always confirm with a small pilot.
- **First experiment.** Log 5k (parent p̂, operator, child p̂) triples and fit the prior. *Success:* fewer rollouts at a fixed misfiling rate, with calibration error reported.

### G4 · Pilot–commit profiling, rollout reallocation and zero-variance recycling
`Established` · **Moderate** · Effort **S**

*Pitch:* spend rollouts where the signal is before you buy new tasks.
- **Mechanism.** Pilot 16 rollouts per item, skip items with p̂ > 0.75, defer those with p̂ < 0.125, and commit 48 to the rest ([Pilot-Commit](https://arxiv.org/abs/2605.26606)). Allocate rollouts by knapsack ([Knapsack RL](https://arxiv.org/abs/2509.25849)). Recycle zero-variance items instead of deleting them ([query recycling](https://arxiv.org/abs/2606.10709)).
- **Evidence.** Pilot-Commit needs 1.9× fewer rollouts than GRPO and 4.0× fewer than DAPO. Knapsack allocation is worth about 2× compute, and 577 prompts labelled "extremely hard" produced positives during training. About 20% of recycled queries later carried signal, supplying about three-quarters of accepted groups late in training. A pass@6 = 0 label is noisy: 10–29% of such items are reachable under perturbed decoding.
- **Effect / risk.** About 2× compute-equivalent before buying any new task. Risk: a rare success on a low-p item may be a verifier false positive; spot-check it with a second verifier (note 18).
- **First experiment.** A/B test on the current run. *Success:* equal accuracy with fewer rollouts.

### G5 · A bandit over (operator × domain), driven by held-out deltas
`Novel` · **Proposal** · Effort **M**

*Pitch:* let measured transfer, not generator opinion, decide which operator gets the budget.
- **Mechanism.** Each arm is an operator family in a domain. Its reward is the held-out delta from a short probe run, or learning progress (TSCL |slope|). Keep a floor for exploration.
- **Why.** Note 01 lists this as open. Yield varies 6× across mutation axes (7.7–45.5%, [Trading Human Curation](https://arxiv.org/abs/2606.03800)). The bar is high: in [DataFlex-RL](https://arxiv.org/abs/2609.06107), none of 8 selection methods and none of 3 adaptive mixtures beat uniform sampling or a fixed equal mix.
- **Effect / risk.** Moves budget to operators that transfer. Risk: noisy short-probe deltas against a high equal-mix bar.
- **First experiment.** Run a 6-arm bandit against an equal mix over 8 seeds. *Kill* it if the paired CI includes zero.

### G6 · Make p ≈ 0 families learnable, in a fixed order
`Established` · **Moderate** · Effort **S–M**

*Pitch:* over-hardened families produce no gradient, so make them learnable before discarding them.
- **Mechanism, in order.**
  1. Audit the family (F4).
  2. Answer-preserving format ladder: 4-choice → 10-choice → cloze → open, promoting at ≥ 0.5 ([Cog-DRIFT](https://arxiv.org/abs/2604.04767)).
  3. A teacher candidate in the prompt, with graduation ([ZPPO](https://arxiv.org/abs/2606.18216)).
  4. Annealed solution prefixes ([QuestA](https://arxiv.org/abs/2507.13266)).
  5. One off-policy trace per group ([LUFFY](https://arxiv.org/abs/2504.14945)).
  6. A dense per-test reward, then binary ([DELTA-Code](https://arxiv.org/abs/2509.21016)).
- **Evidence.** Cog-DRIFT: +10.11 and +8.64 on items at pass@64 = 0. QuestA's annealed prefixes scored 63.26 vs 60.26 for a fixed 50% prefix; [Scaf-GRPO](https://arxiv.org/abs/2510.19807) needed hints on only 17.4% of samples and raised AIME24 from 30.0 to 43.3. DELTA-Code stayed below 1% for 450 steps, then "grokked". AutoOR went from 0% at pass@64 to 48.98% by fading syntax scaffolds. Caution: items at pass@8 = 0 lowered averages by 5.75, 11.24 and 1.07 points, and one harmful sample collapsed response length from 510.7 to 45.7 tokens in 58 steps ([2605.28388](https://arxiv.org/abs/2605.28388)).
- **Effect / risk.** Recovers families that would otherwise be deleted. Risk: scaffolds are off-policy; withdraw them fully and watch for answer-parroting (note 20).
- **First experiment.** On 3 audited families at p ≈ 0, compare the format ladder with prefixes. *Success:* the families enter the band, and the scaffolds are fully withdrawn by the end.

---

## 9. Theme H: SFT-specific ideas

SFT tolerates label noise that RL does not, and benefits more from prompt difficulty and diversity than from answer filtering ([Chapter 06](06-sft-playbook.md)).

### H1 · Behavior priming, including wrong-answer traces
`Established` · **Moderate** · Effort **S**

*Pitch:* if hard variants stall, the model may be missing a behavior (verify, backtrack), not the knowledge.
- **Mechanism.** Before RL, run a brief SFT on traces that show try → check → backtrack → retry, filtered by whether the behavior is present rather than by the answer. Pick the RL starting checkpoint with a short RL probe, not by SFT accuracy.
- **Evidence.**
  - Under identical PPO, Llama-3.2-3B plateaued at about 30% on Countdown while Qwen-2.5-3B reached about 60%. Priming closed the gap, and traces with **wrong** final answers worked as well as correct ones ([Cognitive behaviors](https://arxiv.org/abs/2503.01307)).
  - For agentic search, behavior-filtered trajectories beat outcome-filtered ones ([Behavior Priming](https://arxiv.org/abs/2510.06534)).
  - [SkillFactory](https://arxiv.org/abs/2512.04072) scored 2.8% vs 11.7% after SFT, but 25.1% vs 21.2% after GRPO.
- **Effect / risk.** Cheaply unblocks RL on hard variants. Risk: priming can teach reflection for its own sake; judge by an RL probe.
- **First experiment.** Prime on the teacher's *failed* attempts at your hardest synthetic items. *Success:* a 200-step RL probe beats the unprimed checkpoint.

### H2 · Distill verified trajectories on hardened agentic tasks
`Established` · **Moderate** · Effort **M**

*Pitch:* for search and terminal agents, SFT on hard verified trajectories goes a long way.
- **Mechanism / seeds / verifier.** Harden tasks with C1 or D1, roll out a strong teacher, keep trajectories that pass the verifier (or behavior filters), then SFT.
- **Evidence.**
  - FORT-Searcher is SFT-only and reaches BrowseComp 72.2 ([FORT](https://arxiv.org/abs/2606.12087)).
  - [OpenSeeker](https://arxiv.org/abs/2603.15594)-v2 used 10.6k SFT samples to beat Tongyi DeepResearch's CPT+SFT+RL on BrowseComp (46.0 vs 43.4).
  - [SkillSynth](https://arxiv.org/abs/2604.25727)'s skill-graph data beat single-skill SFT by 8.3 points on TB2.0.
  - Counter-evidence for procedural and scientific tasks: SFT scored −3.9 against +5.4 for RL on IPhO (Sim2Reason), and 26 against 80 on Hard-LP for [AutoOR](https://arxiv.org/abs/2604.16804).
- **Rule.** Use SFT for breadth of agentic behavior and formats. Use RL for composition (B1, B2).
- **Effect / risk.** Strong gains for search and terminal agents. Risk: SFT on hard procedural or scientific tasks often regresses; prefer RL there.
- **First experiment.** Compute-matched SFT vs RL on the same hardened pool; note 09 lists this comparison as open. *Success:* a documented crossover point per domain.

### H3 · Select SFT prompts by difficulty and diversity, and don't filter answers
`Established` · **Moderate** · Effort **S**

*Pitch:* spend the SFT budget on hard, diverse prompts.
- **Mechanism.** Rank prompts by LLM difficulty rating or teacher response length, keep many seeds with few rewrites each, and select traces for route diversity.
- **Evidence.**
  - [OpenThoughts](https://arxiv.org/abs/2506.04178): the best question filters were LLM difficulty ratings for code and response length for math and science (+6% and +4% over random). No answer filtering averaged 41.9, against 40.0 with GPT verification.
  - [GLM-4.5](https://arxiv.org/abs/2508.06471): removing the bottom 50% of prompts by response length gave +2–4%.
  - X-Coder: 64k tasks × 1 solution beat 16k × 4.
  - Many seeds with few rewrites each beat the reverse ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)).
  - Route-diverse trace selection raised post-RL pass@8 by 16.9 points ([2609.33780](https://arxiv.org/abs/2609.33780)).
- **Effect / risk.** More headroom per SFT token. Risk: length-based selection also favors verbosity; cap it and check held-out quality.
- **First experiment.** At equal tokens, compare a response-length/LLM-difficulty selector with random selection. *Success:* held-out gain, and higher downstream RL headroom (large-k pass@k).

### H4 · Fault-injected and recovery-spliced trajectories
`Established` (recent) · **Emerging** · Effort **M**

*Pitch:* hard tasks need recovery skills that clean demonstrations never show.
- **Mechanism.** Inject faults at action steps while collecting expert trajectories ([TermiGen](https://arxiv.org/abs/2602.07274)). Splice failing prefixes onto verified sibling successes, and mask the loss on the erroneous turns.
- **Why it should work.** TermiGen-Qwen2.5-Coder-32B reached a 31.3% average pass rate on TerminalBench, above o4-mini with Codex CLI; recovery splicing gave +5.59% over baselines across WebShop, SciWorld and TextCraft ([Agent-R](https://arxiv.org/abs/2501.11425)).
- **Effect / risk.** Teaches diagnosis and recovery. Risk: off-policy injected errors do not teach self-correction (note 20). For that goal, use an on-policy correction turn ([SCoRe](https://arxiv.org/abs/2409.12917)).
- **First experiment.** Inject 1–2 faults into 1k expert terminal trajectories and SFT on the recovered versions against clean-only traces. *Success:* recovery rate on held-out faulty environments rises with no drop in clean success.

---

## 10. Theme I: Evaluating the synthetic pipeline itself

### I1 · Mutated-twin and held-out-family evaluations, built before training
`Established` · **Strong** · Effort **S**

*Pitch:* if the same operators produce both your training data and your evaluation, you are measuring memorization.
- **Mechanism.** For every operator family used in training, hold out whole families, generators, renderers and topologies. Build mutated twins of every target benchmark.
- **Evidence.**
  - [Pythagoras ALF](https://arxiv.org/abs/2606.12594) (MiniF2F-ALF) mutations cost every SOTA prover about 3–5 points.
  - [Ineq-Comp](https://arxiv.org/abs/2505.12680): pass@32 falls from 66.2% on seeds to 47.0% and 42.1% on the composed types, and SFT on about 8K composed problems improved only the operator type it was trained on.
  - TÜLU-3-8B-DPO scores 81.1 on IFEval but 25.5 on IFBench.
  - [K&K](https://arxiv.org/abs/2410.23123) fine-tuning is near-perfect on training puzzles and brittle under one-statement perturbations.
- **Effect / risk.** Separates skill from template learning. Risk: held-out families written by the same generator drift toward training; use another generator family.
- **First experiment.** Freeze the suite before the first training run (§14.1). *Success:* each synthetic gain is reported alongside its held-out-family counterpart.

### I2 · Audit the generator's signature
`Established` · **Moderate** · Effort **S**

*Pitch:* find what your generator leaks before the policy learns to exploit it.
- **Mechanism.** Chi-square tests on answer-position and answer-value marginals; partial-input baselines; an LLM-ID classifier; n-gram and embedding decontamination, including variant contamination. Let an external RNG make every structural random choice.
- **Evidence.** A five-way LLM-ID classifier reaches 97.1% accuracy, and students trained on two different teachers are 98.9% separable ([Idiosyncrasies](https://arxiv.org/abs/2502.12150)). LLM MCQ generators put the answer first 47.9–57.9% of the time ([2605.01846](https://arxiv.org/abs/2605.01846)). Preference leakage is 23.6% when the judge is also the generator, against 2.8% across model series (note 19).
- **Effect / risk.** Removes exploitable regularities before RL. Risk: fixing one signature can create another; re-run after every generator change.
- **First experiment.** Run the audit per generator and operator. *Success:* partial-input baselines are at chance and marginals are flat.

### I3 · A dashboard of per-operator yield and cost per accepted in-band task
`Established` · **Moderate** · Effort **S**

*Pitch:* one number, cost / (validity yield × in-band yield), decides where the next dollar goes.
- **Mechanism.** Log every candidate's lineage, operator, gate outcome, rejection reason and pilot p̂.
- **Evidence.**
  - Costs: about $0.05 per accepted RST task; $19.66 per built OpenSWE environment and about $163 per retained one.
  - Yields range from 2.25% (SWE-Next) to about 50% (SWE-Factory), and vary 7.7–45.5% across mutation axes.
  - Profiling alone was about 36% of rollout tokens in note 18's worked example.
  - Training compute dwarfed generation spend: about $20K per arm, against about $16 for 319 variants ([Trading Human Curation](https://arxiv.org/abs/2606.03800)).
- **Effect / risk.** Budget follows measured yield instead of intuition. Risk: yield alone favors easy-to-accept operators; also weight by held-out delta (G5).
- **First experiment.** Instrument the pipeline. *Success:* budget moves weekly toward the operators with the best cost per accepted in-band task.

### I4 · Skill-level diversity telemetry and failure-cluster descriptors
`Established` (components) / `Novel` (descriptors) · **Emerging** · Effort **S–M**

*Pitch:* diversity collapses silently. Measure it on solutions, not on wording.
- **Mechanism.** Similarity over canonical solver code ([R-Diverse](https://arxiv.org/abs/2602.13103)) or masked SQL templates; nearest-neighbour similarity per round; pass@k tracking. *Novel extension:* index a QD archive by clusters of policy-failure embeddings instead of hand-picked skill tags (note 14 lists this as unexplored).
- **Evidence.** RST's nearest-neighbour similarity rose from 0.22 to 0.46 across rounds. Evol-Instruct pushed pass@8 below the untuned backbone (54.2 vs 55.1, [EvoTD](https://arxiv.org/abs/2605.11666)). [ACES](https://arxiv.org/abs/2310.10692) keeps 21,700 skill-combination niches. No diversity metric has yet been shown to predict RL gains (note 19).
- **Effect / risk.** Catches collapse before downstream damage. Risk: surface-embedding metrics cannot tell reskins from new skills (note 11).
- **First experiment.** Correlate 3 diversity metrics with the held-out RL deltas of 10 pools. *Success:* one metric ranks the pools correctly.

### I5 · A benchmark calibrating simulated hardness against real hardness
`Novel` · **Proposal** · Effort **M**

*Pitch:* check that a task hard in your simulator is also hard in the real environment.
- **Mechanism.** Run the same tasks in a database-backed environment and in an LLM simulator, with and without perturbations. Report pass-rate correlation per band, stratified by the number of state changes (0 / 1–2 / 3–6 / 7–12).
- **Why novel.** Note 17 lists it as open. EnvSimBench found a cliff once several variables change at once. On GUI-Genesis a VLM judge scores 63.76% where code assertions give 38.93%. PhoneWorld's functional-page coverage is 51–80% even though rendered-page coverage is above 96%.
- **Effect / risk.** Tells you how far simulator-hardened tasks can be trusted. Risk: needs a real backend for the paired tasks, which is the expensive part.
- **First experiment.** Build 300 paired tasks. *Success:* a per-stratum correlation table that sets how far you trust the simulator.

---

## 11. Theme J: Novel combinations drawn from the open problems

All eight ideas are `Novel` with **Proposal** evidence. Each is paired with the open problem it addresses and a kill criterion.

### J1 · One seed, four interaction operators (effort M)
- **Mechanism.** Deliver one verifiable seed as sharded turns (N), across a horizon beyond the context window (H), with K injected tool outputs and a partner who withholds one needed fact. The label is preserved: the CONCAT or oracle view must still pass.
- **Why.** Note 21: no open pipeline composes these operators or measures how their knobs interact. Each operator works alone: CONCAT keeps 95.1% of FULL; MemAgent trained at 32K–60K tokens keeps 71.09% at 3.5M; AgentDojo supplies the injection; CLARITI supplies the withholding.
- **Effect / risk.** Realistic multi-turn difficulty from single-turn seeds. Risks: degenerate ask/abstain policies and simulator leakage (C4).
- **First experiment.** A 2⁴ factorial design on 500 seeds. *Success:* identify which knobs interact super-additively. *Kill* if training on mixed cells transfers no better than training on single operators.

### J2 · "Harder-than" variants checked by the Lean kernel (effort M)
- **Mechanism.** Accept a formal variant only with a kernel-checked proof that it implies the seed (or generalizes it), plus a certificate that a fixed automation portfolio (`aesop`, `exact?`) fails on it. Hardness is then at least the seed's by construction.
- **Why.** Note 03: no pipeline enforces a verified "harder-than" relation at scale. Related certificates: InternGeometry's X_raw vs X_add, and LeanConjecturer's automation floor. Ineq-Comp shows that simple composition already breaks provers.
- **Effect / risk.** Hardness by construction rather than by sampling. Risk: drift toward trivially stronger statements; keep a triviality/elegance filter.
- **First experiment.** 1k Lean seeds, compared against STP-style conjecturing. *Success:* higher in-band yield with zero false statements.

### J3 · Hidden-invariant generators for transformative difficulty (effort M)
- **Mechanism.** Generate instances whose natural solution is intractable at the chosen size but which have a short hidden invariant (periodicity, symmetry, reversibility). Verify by brute force at small n and with an invariant-based checker at large n. Warm up with dense rewards (DELTA-style).
- **Why.** Transformative generalization is still about zero ([OMEGA](https://arxiv.org/abs/2506.18880), DELTA), and note 10 names exactly this direction. Horizon generalization did not transfer to harder Sudoku techniques (note 20).
- **Effect / risk.** Targets new-strategy difficulty, which no current operator reliably produces. Risk: small-n pattern matching instead of insight; vary surface form.
- **First experiment.** Train on 3 families at small n. *Kill* if nothing at large n is unlocked.

### J4 · An online re-complexification service inside RL (effort M–L)
- **Mechanism.** A generator service reads the trainer's per-item p̂ stream. When an item saturates, it applies one operator (hop, fuzz, distractor, constraint), chosen by G3 priors and I3 yields. It gates the child and injects it with a version tag under the same domain quota. The parent moves to replay.
- **Why.** Notes 07 and 18: online re-complexification is rare, and no framework generates or re-levels tasks online (SkyRL marks dropped groups as consumed). The closest published mechanisms are [RODS](https://arxiv.org/abs/2606.19047), which resamples isomorphic variants in the [0.20, 0.85] band and retires tasks above 0.95, and the RLAnything acceptance rule.
- **Effect / risk.** Keeps the band full without stage-level regeneration. Risks: stale difficulty labels in async training and mix drift; replace within domain quotas.
- **First experiment.** Compare against offline regeneration between stages, at equal compute. *Success:* a higher effective-prompt ratio across the run and a held-out gain. Report the staleness of difficulty labels.

### J5 · A hacker–fixer loop inside recursive escalation (effort S–M)
- **Mechanism.** Run a hacker agent on every accepted child in each RST round. Fix, re-validate, and log the exploit rate by round and by operator.
- **Why.** Note 16 asks whether hackability grows with escalation; nobody reports exploit rates by round. The shortcut counts (40 at complexity levels 1–10 vs 458 at 11–20) suggest it does.
- **Effect / risk.** Keeps recursive escalation safe to train on. Risk: extra cost per round.
- **First experiment.** 5 rounds on 200 seeds, with and without the loop. *Success:* an exploit-by-round curve, and fewer reward-hacking incidents in downstream RL.

### J6 · A typed constraint DSL with SMT witnesses for IF self-play (effort M)
- **Mechanism.** Write constraints in a typed Scope/Target/Range DSL ([ScopeIF](https://arxiv.org/abs/2609.32189)). A solver proves the constraint set jointly satisfiable and emits a witness skeleton. The self-play generator is rewarded for hitting the pass-rate band, not for raw failure as in [SEIF](https://arxiv.org/abs/2605.07465), with anti-degeneracy terms. Push past the 5–7-constraint phase transition toward agent system prompts.
- **Why.** Note 08 lists both formal satisfiability and learnability-shaped IF generators as open. "Output JSON" is jointly unsatisfiable with 9 of 24 instructions, and AgentIF averages 11.9 constraints per instruction.
- **Effect / risk.** No unsatisfiable prompts, and direct band targeting. Risk: DSL constraints may be less natural; hold out natural constraint families.
- **First experiment.** A DSL with 30 constraint types, compared against an LLM conflict filter. *Success:* zero unsatisfiable prompts and a higher in-band yield.

### J7 · First-error localization for agent trajectories, labelled by replay (effort M)
- **Mechanism.** In a replayable environment, corrupt one action of a successful trajectory and re-simulate. If the checker then fails, label that step as the first error. Use the labels for verifier and process-reward training, and for "resume from state k" tasks.
- **Why.** Note 20: exact first-error labels exist for logic, math and code ([2605.02395](https://arxiv.org/abs/2605.02395)) but are largely missing for multi-turn agents.
- **Effect / risk.** Exact process labels for agents. Risk: constructed errors may not look on-policy (note 20); select corruptions by policy likelihood.
- **First experiment.** Collect 5k labels and train an agent process reward model (PRM). *Success:* better accuracy and best-of-N gains than a PRM trained on judge labels.

### J8 · Research-mined monthly tasks with checkpoint rewards (effort M)
- **Mechanism.** Every month, harvest post-cutoff statements and PRs, rewrite them to be self-contained, filter to the policy's band, and decompose them into checkpoints with machine-verifiable answers.
- **Why.** Note 23. [RealMath](https://arxiv.org/abs/2505.12575) turned 14,747 theorems into 280 usable items, about 94% valid. [LemmaBench](https://arxiv.org/abs/2602.24173) accuracy rose from 12.3% to 40.8% in nine months, so freshness decays. [CritPt](https://arxiv.org/abs/2509.26574) splits 71 challenges into 190 checkpoints. [ResearchMath](https://arxiv.org/abs/2605.28003) SFT gave +2.1 even though only 3.7% and 4.3% of trajectories were judged correct.
- **Effect / risk.** A self-refreshing supply of hard, uncontaminated tasks. Risks: low yield, unrefereed results, and share-alike license obligations (note 23).
- **First experiment.** One month of harvest, with checkpoint rewards vs final-answer-only. *Success:* a non-zero gradient share and gains on held-out research-level items.

---

## 12. Top-15, ranked

**Scoring.** Score = Impact × Confidence ÷ Effort.

- **Impact (I, 1–5):** expected recovery of learning signal and held-out gain on a typical saturated pool.
- **Confidence (C):** Strong = 5, Moderate = 4, Emerging = 3, Proposal = 2.
- **Effort (E):** S = 1, M = 2, L = 4.

The formula favors cheap, label-preserving work by design. The highest-*impact* items are G1, D1 and B1.

| # | Idea | Status | Evidence | I | C | E | Score | First deliverable |
|---|---|---|---|---|---|---|---|---|
| 1 | [A1](#a1--re-audit-the-verifier-then-harden-the-tests) Verifier audit + test hardening | Established | Strong | 4 | 5 | S | 20 | TPR/TNR and false-negative report on the p̂ = 0 and p̂ = 1 tails |
| 2 | [F1](#f1--a-task-bundle-contract-with-ci-gates) Task-bundle contract + CI | Established | Strong | 4 | 5 | S | 20 | Oracle-pass / no-op-fail / known-bad-fail on the whole pool |
| 3 | [A2](#a2--answer-preserving-stem-hardening) Answer-preserving stem hardening | Established | Moderate | 4 | 4 | S | 16 | 3 rewrites per saturated seed, SynthRL gate |
| 4 | [A5](#a5--oracle-preserving-perturbations-for-tool-use-seeds) Oracle-preserving tool perturbations | Established | Moderate | 4 | 4 | S | 16 | 4 perturbation families with replayed gold traces |
| 5 | [A4](#a4--shortcut-and-no-context-filters-as-a-pre-pass) Shortcut / no-context filters | Established | Strong | 3 | 5 | S | 15 | Shortcut rate per family |
| 6 | [B3](#b3--constraint-stacking-calibrated-by-pk-with-structure-aware-rewards) p^k-calibrated constraint stacking | Established | Strong | 3 | 5 | S | 15 | Per-constraint success fit on your policy |
| 7 | [F4](#f4--two-sided-privileged-information-solvability-gates) Two-sided solvability gates | Established | Strong | 3 | 5 | S | 15 | p = 0 tail split into hard vs broken |
| 8 | [I1](#i1--mutated-twin-and-held-out-family-evaluations-built-before-training) Mutated-twin / held-out-family evaluations | Established | Strong | 3 | 5 | S | 15 | Frozen evaluation suite (§14.1) |
| 9 | [G1](#g1--a-per-family-online-difficulty-controller) Per-family online controller | Established | Strong | 5 | 5 | M | 12.5 | 10 families with knobs + sliding window |
| 10 | [A6](#a6--strip-hints-scaffolds-and-specifics-keep-the-checker) Strip hints, scaffolds and specifics | Established | Moderate | 3 | 4 | S | 12 | 3-rung ladders with contract validity |
| 11 | [G4](#g4--pilotcommit-profiling-rollout-reallocation-and-zero-variance-recycling) Pilot–commit, reallocation, recycling | Established | Moderate | 3 | 4 | S | 12 | Rollouts saved at equal accuracy |
| 12 | [H1](#h1--behavior-priming-including-wrong-answer-traces) Behavior priming (incl. wrong-answer traces) | Established | Moderate | 3 | 4 | S | 12 | Primed vs unprimed RL probe |
| 13 | [I3](#i3--a-dashboard-of-per-operator-yield-and-cost-per-accepted-in-band-task) Per-operator yield and cost dashboard | Established | Moderate | 3 | 4 | S | 12 | Cost per accepted in-band task, by operator |
| 14 | [B1](#b1--serial-chaining-with-deterministic-adapters-and-a-horizon-curriculum) Serial chaining + horizon curriculum | Established | Strong | 4 | 5 | M | 10 | Chains of length 2–5, staged vs uniform |
| 15 | [D1](#d1--recursive-solution-first-escalation-rst-for-executable-tasks) RST recursive escalation | Established | Moderate | 5 | 4 | M | 10 | 3 rounds on 200 executable seeds |

**Runners-up (score 8–10).** C1 and A3 score 10 but apply only to search/QA pools and to MCQ or exact-match pools respectively, so they rank below B1 and D1. A7 scores 9. At 8: A8, B4, B6, D4, D8, E2, F2, F3, G2, G6, H2, H3 and I2.

**Best-scoring novel bets (score 4):** D2 (RST for workbooks and notebooks), J4 (online re-complexification) and B8 (branch–merge curricula). Run one of them in days 61–90, in the domain you care about most.

---

## 13. A 30/60/90-day roadmap for a team whose tasks are too easy

The roadmap assumes one post-training team with an RL stack, a pool of verified but saturated seeds, and one or two target domains. It is ordered so that each phase's exit gate uses measurements the previous phase built.

**Days 1–30: measure, then make the existing pool hard again (label-preserving only).**

| Week | Build | Ideas |
|---|---|---|
| 1 | Convert the pool to task bundles and run the CI gates. Build the yield and cost dashboard. Freeze the evaluation suite, including held-out families and a non-Qwen model. Profile the whole pool on the current checkpoint with pilot–commit. | F1, I3, I1, G4 |
| 2 | Audit the verifier on both tails. Run shortcut probes. Run the two-sided gate on the p = 0 tail. Audit generator signatures. | A1, A4, F4, I2 |
| 3–4 | Label-preserving hardening on saturated items: answer-preserving rewrites, oracle-preserving perturbations, hint stripping, p^k constraint stacking for IF, failure-prefix on items at p̂ ≥ 0.95. Wrap procedural families with online controllers. Prime behaviors if p ≈ 0 families lack verify or backtrack. | A2, A5, A6, B3, A7, G1, H1 |

*Exit gate (day 30).* The effective-prompt ratio is back up. The in-band pool holds at least about S·B/25 unique items for the next stage (the reuse rule in note 19: S steps × B prompts per step, reusing items up to about 25×). A screening run shows a held-out gain whose paired CI excludes zero. If the gain appears only on in-distribution synthetic evaluations, stop and revisit A4 and I1.

**Days 31–60: build construction engines for new difficulty.**

| Track | Build | Ideas |
|---|---|---|
| Reasoning seeds | Serial chaining with a horizon curriculum; functional composition tested on held-out atoms; worst-case variant rewards | B1, B2, A9 |
| Agentic / executable seeds | Recursive solution-first escalation; evaluator-first chaining; steered perturbations | D1, B4, D4 |
| Search / QA seeds | Fuzz + densify + disperse with uniqueness repair; fictional worlds | C1, C6 |
| All | Red-team harness per operator family; p ≈ 0 ladder; SvS inside math RL; start logging (parent, operator, child) triples for priors | F2, G6, E2, G3 |

*Exit gate (day 60).* Every operator has a logged yield, a rejection-reason breakdown, an exploit rate and a held-out delta. Drop operators whose output is mostly "too easy" or whose exploit rate stays non-zero after fixes. Run the §14.5 decision protocol on the two best data policies with 8–12 matched seeds.

**Days 61–90: learn the generator and test one novel bet.**

| Build | Ideas |
|---|---|
| Either train a validity-gated setter, or run a three-arm constructor ablation (trained / prompted / static) | E1 or E5 |
| Bandit over operator × domain vs an equal mix; operator-conditioned priors in the profiler | G5, G3 |
| One novel bet in your main domain: online re-complexification (J4), workbook/notebook RST (D2), or branch–merge curricula (B8) | J4 / D2 / B8 |
| Dose ladder, cross-family replication, and a lifecycle re-audit of the pool (sudden 0 → 100% families, reward/length anomalies) | §14 |

*Exit gate (day 90).* Promote a pipeline to production only if it passes §14.5 and its cost per accepted in-band task is known. Keep the kill criteria from Theme J.

---

## 14. Experiment protocol

### 14.1 The evaluation suite (freeze before training)

| Component | What it guards against | Source of the rule |
|---|---|---|
| Natural target benchmarks, plus post-cutoff items (new contests, new PRs) | Contamination; gains that exist only on synthetic data | Notes 12, 19, 23 |
| Held-out operator families, held-out generator family, mutated twins | Learning templates instead of skills | I1; notes 03, 05, 14 |
| Large-k pass@k (up to 128), null-set unlock rate, per-topology accuracy | Sharpening mistaken for new capability | Note 10 |
| Regression suite: other domains, IF, safety, over-refusal, best@k | Untargeted regressions. Math RLVR raised IFEval pass@1 by 6.5% but lowered best@32 by 9.8% on Qwen3-8B-Base. | Note 08 |
| A second, non-Qwen model family | Base-model artifacts: random rewards gave +21.4 on MATH-500 for Qwen2.5-Math-7B but did not work on Llama3 or OLMo2 ([Spurious rewards](https://arxiv.org/abs/2506.10947)) | Notes 02, 19 |
| Trivial agents (empty, no-op, random) on agentic evaluations | Invalid benchmarks; an empty response passes 38% of τ-bench-Airline | Note 19 |

Use a domain-balanced aggregate and pre-register it. Swapping a math-heavy aggregate for a domain-balanced one gave a ranking correlation of ρ = −0.33 (note 19).

### 14.2 Controls every data policy must beat

1. **Seeds only**, at matched rollout compute.
2. **Hard real data selected by pass rate**, at matched compute. The equal-compute comparison of synthetic hardening against selecting hard real data is still open (notes 12, 19), so run it yourself.
3. **Random-reward control** on the same base.
4. **Operator leave-one-out** for mixtures of operators.
5. **Dose ladder.** Vary the number of unique items and the reuse factor. There was no significant degradation up to τ = 25 reuses and clear overfitting at τ = 100 ([RL scaling behaviors](https://arxiv.org/abs/2509.25300)).
6. **Anchoring.** Default to 20–33% oracle-backed real or seed items plus 2–10% easy or mastered replay. Justify any deviation (note 19; [Chapter 05 §8](05-rl-playbook.md)).
7. **SFT vs RL** on the same pool, wherever the idea could serve either.

### 14.3 Band monitoring during training

| Signal | Alarm and action |
|---|---|
| Effective-prompt ratio; in-band share of the pool; p̂ histogram per family | Sustained decline → trigger re-complexification (G2/J4) or promote the controller level (G1) |
| All-zero share per family | Rising → run the two-sided gate, then scaffold (G6); do not delete items after a single pilot (G4) |
| Entropy, response length | Length collapse is an early warning: 510.7 → 45.7 tokens in 58 steps from one harmful sample → audit the items at zero |
| Per-skill accuracy | Strong skills rising while weak ones fall ("polarization of competence", KITE) → re-weight toward the weak skills and add anchors |
| Label accuracy of generated items on a gold holdout | Decaying, as R-Zero's did (79% → 63%) → strengthen the validity gate |
| Judge incorrect-credit rate; divergence of verifier reward from the oracle | Incorrect credit went from 39% to 65% with a GPT-4o-mini rubric verifier. A fine-tuned verifier's reward diverged from the oracle after about 450 iterations. → re-audit against a held-out judge panel |
| Sudden 0 → 100% jumps; reward–length anomalies; canary triggers | Presumed exploit → quarantine the family and run F2 (lifecycle re-audit, note 17) |

### 14.4 Signature and contamination audits (before and after training)

- Run the I2 audit per generator and operator: marginals, partial-input baselines, and LLM-ID separability.
- Decontaminate with n-grams and embeddings, and include *variant* contamination. A hierarchical detector reports F1 0.76, against 0.17–0.49 for baselines (note 11).
- Repeat the audit after post-training, because GRPO can spread leaked information to related benchmarks.
- Log the generator on every row for provenance and license obligations ([Chapter 08](08-frontier-lab-practices.md); note 23).

### 14.5 Statistics and decision rules

```text
SCREEN   short runs; fit the early reward curve (ScaleRL-style sigmoid fits reproduced the
         asymptote within ±0.02 across 3 runs) and drop clear losers.
POWER    paired per-item analysis with many samples per item; add prompts before samples;
         cluster SEs by seed family. At p≈0.5 an unpaired 30-item comparison needs ~25 points
         for p<0.05 (~36 for 80% power). Seed SD on AIME/AMC is 5–15 points; hardware, batch
         size and BF16 alone moved accuracy by up to 9%.
ADOPT    iff  paired 95% CI excludes 0 across 8–12 matched seeds
          and no regression-suite metric drops
          and the gain holds on post-cutoff items, a held-out generator/operator family,
              and a non-Qwen model.
STOP     generator/self-play loops after ~3 rounds unless the held-out metric is still rising
         (WizardCoder best after 3; SEIF, UltraIF; R-Zero peaked within 3–4 iterations).
KILL     any Proposal whose first experiment misses its success metric; any operator whose
         post-fix exploit rate stays > 0 or whose accepted items fail hand audit.
```

Adaptive mixtures and selection policies must beat a fixed equal mix under this rule before they replace it. In DataFlex-RL, none of 13 policies tested with 12 matched seeds did ([DataFlex-RL](https://arxiv.org/abs/2609.06107)).
