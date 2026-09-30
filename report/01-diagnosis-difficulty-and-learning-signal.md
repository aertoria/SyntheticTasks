# Diagnosis: why tasks look "too easy", what "hard" should mean, and where learning signal comes from

> **Key takeaways**
>
> - **"Too easy" is often a measurement artifact, so diagnose before you synthesize.** Look at the per-prompt pass-rate histogram under the exact policy, harness and verifier, not at the mean. Then rule out lenient verifiers, contamination and format shortcuts. Weak tests alone can make hard tasks look solved: TACO's tests had a false-positive rate above 90% on difficult problems, and strengthening SWE-bench Verified tests cut the top score from 78.80% to 62.20%.
> - **Difficulty is a property of a (task, policy, verifier, budget) tuple, not of the task.** GRPO-style RL learns only from prompts with 0 < p < 1. The chance that a group of N rollouts carries any gradient is 1 − pᴺ − (1 − p)ᴺ. Across more than 30 papers and lab reports, nearly every pipeline drops or specially handles p = 0 and p = 1, centres on roughly 0.3–0.6, and re-profiles as the policy moves.
> - **SFT and RL need different things from "hard".** SFT needs questions the teacher can solve that elicit long, diverse reasoning, and it tolerates wrong answers. RL needs in-band items with a precise verifier. The best-supported division of labour is to install atoms, formats and behaviours with SFT or mid-training, then run RL on compositions of them.
> - **Controlled studies agree on three things.** (1) Composition beats paraphrase: RL on composed tasks builds new skills and RL on atoms does not. (2) Edge-of-competence data expands pass@k, while in-distribution data only sharpens pass@1. (3) Difficulty coverage and diversity matter more than ordering, except when the top tier would otherwise get zero reward.
> - **Measure difficulty with enough samples and treat p̂ = 0 as "unknown".** At p = 0.5, an 8-rollout estimate has a 95% half-width of about ±0.35 (derived). Between 10% and 29% of a pass@6 = 0 stratum was reachable under perturbed decoding. Use Bayesian trackers, cheap priors and trajectory signatures, and validate every gain against a random-reward control, freshly generated items and a second model family.

## Contents

1. [Where learning signal comes from](#1-where-learning-signal-comes-from)
2. [A diagnostic checklist to run before synthesizing anything](#2-a-diagnostic-checklist-to-run-before-synthesizing-anything)
3. [Difficulty as a policy-relative quantity](#3-difficulty-as-a-policy-relative-quantity)
4. [What SFT needs from hard tasks vs what RL needs](#4-what-sft-needs-from-hard-tasks-vs-what-rl-needs)
5. [What theory and controlled studies say](#5-what-theory-and-controlled-studies-say)
6. [How to measure difficulty](#6-how-to-measure-difficulty)
7. [Symptom → remedy decision table](#7-symptom--remedy-decision-table)

Evidence tags used for recommendations: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (a single recent or unreplicated result), **Proposal** (our synthesis, not yet tested).

---

## 1. Where learning signal comes from

In group-based RL (GRPO, DAPO, RLOO), each prompt is sampled N times and each rollout's advantage is its reward minus the group mean, often divided by the group standard deviation. If all N rollouts receive the same reward, every advantage is zero and the prompt contributes no gradient. For a prompt with true success rate p, the probability that a group is informative is

```
P(non-zero gradient) = 1 − pᴺ − (1 − p)ᴺ          (Knapsack RL, 2509.25849)
```

| p | N = 4 | N = 8 | N = 16 | N = 64 | rollouts for a 90% chance to see the minority outcome |
|---|---|---|---|---|---|
| 0.01 or 0.99 | 0.04 | 0.08 | 0.15 | 0.47 | 229 |
| 0.05 or 0.95 | 0.19 | 0.34 | 0.56 | 0.96 | 45 |
| 0.125 or 0.875 | 0.41 | 0.66 | 0.88 | 1.00 | 17 |
| 0.25 or 0.75 | 0.68 | 0.90 | 0.99 | 1.00 | 8 |
| 0.5 | 0.88 | 0.99 | 1.00 | 1.00 | 3 |

*Derived from the formula above. The 229-rollout figure for p = 0.01 matches the worked example in [Knapsack RL](https://arxiv.org/abs/2509.25849).*

A non-zero group is necessary but not sufficient. The theory and allocation papers agree on *how much* signal each band carries:

- Expected policy improvement under KL-regularized RL is lower-bounded by p(1 − p)/(2β²). The bound is zero at p ∈ {0, 1} and maximal at p = ½ ([Bae et al., 2025](https://arxiv.org/abs/2504.03380)). This is the basis of "learnability" selection by p(1 − p) ([LILO](https://arxiv.org/abs/2502.12272)).
- An information-gain proxy p(1 − p)² peaks at p = 1/3, i.e. it favours the harder side of the middle ([Knapsack RL](https://arxiv.org/abs/2509.25849)).
- Under vanilla GRPO the cumulative absolute advantage peaks at medium accuracy, so low-accuracy prompts are under-weighted *even when present* ([DARS](https://arxiv.org/abs/2508.13755)). Non-linear objectives such as log-likelihood of success weight hard prompts by about 1/√p ([Reinforce-Ada](https://arxiv.org/abs/2510.04996)).

How much of a real pool is silent? Late in training on DAPO-Math-17K with N = 8, about 40% of Qwen2.5-Math-7B's prompts were all-correct and about 20% all-wrong ([Knapsack RL](https://arxiv.org/abs/2509.25849)). About 39% of rollouts go to silent groups under uniform sampling ([ThinkPrior](https://arxiv.org/abs/2609.09075)), and zero-variance groups are "typically about 40%" in agentic RL ([Selective Rollout](https://arxiv.org/abs/2605.05802)). A static pool with a low difficulty ceiling drives the effective-prompt ratio (share of groups with non-identical rewards) to zero ([RLVE](https://arxiv.org/abs/2511.07317)).

```
 pass rate p under the CURRENT policy, verifier and budget
 0 ─────┬──────────┬──────────────────────────┬──────────┬───── 1
   p = 0│ rare     │   learnable band          │ brittle  │p = 1
  (silent│ success  │   ~0.2–0.7, peak ~1/3–½   │ mastery  │(silent)
  unless │ (needs   │   -> train here           │ -> replay│ -> complexify,
  audited│ many     │                           │  lightly │    meta-task,
  +      │ rollouts,│                           │          │    or retire
  scaffold)│ reweight)│                         │          │
```

For SFT the signal is different: it comes from teacher tokens, so "too easy" means the question elicits a short, low-information trace from the teacher, or the student already reproduces it. Section 4 treats the two regimes separately.

---

## 2. A diagnostic checklist to run before synthesizing anything

A team that sees "95% train accuracy" or "most GRPO groups are all-correct" usually jumps to generating harder tasks. Several different conditions produce that symptom, and only some of them are cured by complexification. Run the checks below first; each row points to the chapter that owns the fix.

| # | Check | How | Red flag | Owner chapter |
|---|---|---|---|---|
| 1 | Per-prompt pass-rate histogram | k ≥ 16 rollouts per prompt with the training policy, harness, sampling temperature and token budget | Mass at p = 1 (saturated); mass at 0 *and* 1 (bimodal, coarse generator); long 0-tail | This chapter, Ch. 5 |
| 2 | pass@1 vs pass@k | Unbiased pass@k from n ≥ k samples, k up to 64–1,024 on a slice | pass@1 high but pass@k flat (saturation); pass@1 low but pass@k high (reliability, not capability, is the gap) | §2.2, §5.3 |
| 3 | Verifier false positives | Known-bad candidates (near-miss programs, truncated answers, no-op agents) scored by the verifier | Any known-bad passing; TPR or TNR < 0.9 | Ch. 4 |
| 4 | Verifier false negatives | Known-good rewrites (equivalent forms, alternative valid outputs) | Rule checker rejecting correct answers; many 0-pass items solved by a stronger model | Ch. 4 |
| 5 | Contamination and memorization | Partial-prompt completion probe; fresh procedurally generated control; post-cutoff items; renamed or perturbed variants | Model completes prompts verbatim; accuracy drops on renamed or perturbed variants | §2.4, Ch. 9 |
| 6 | Format leakage and shortcuts | No-CoT guessing (N = 8), tool-free solver, question-only / choices-only baselines, answer-marginal histogram | Above-chance partial-input accuracy; correct guesses without reasoning | §2.5, Ch. 4 |
| 7 | Eval saturation vs training saturation | Compare train-pool histogram with held-out, post-cutoff, second-family evals | Train pool saturated but eval not (or the reverse) | §2.6 |
| 8 | Harness and budget | Same tasks under the RL harness, turn limit and max tokens | Most failures hit the turn or length cap | §2.6, Ch. 5 |
| 9 | Behaviours | Rate of verification, backtracking and retry in rollouts on the hard tail | Behaviours absent: harder data will not help until they are installed | §4, Ch. 6 |

### 2.1 Look at the distribution, not the mean

A pool at 80% mean accuracy can be 80% saturated items plus 20% impossible ones (no signal at all), or it can be uniformly at p ≈ 0.8 (plenty of signal). Only the histogram tells these apart. Typical shapes and readings:

| Shape of per-prompt p̂ histogram | Reading | First move |
|---|---|---|
| Spike at 1 | Pool saturated for this policy | Complexify (Ch. 2), meta-task transforms or failure-prefix conditioning (Ch. 5) |
| Spikes at 0 and 1, little in between | Generator step size too coarse, or a broken subset | Add intermediate levels per seed (e.g. [UltraLogic](https://arxiv.org/abs/2601.03205)-style calibrated ladders); audit the 0-spike |
| Long tail at 0 | Items hard, broken or mislabelled | Audit before scaffolding (§2.3–2.4) |
| Mass in 0.2–0.8 | Healthy; "too easy" is probably an eval or verifier problem | Check rows 3–7 |

Report the histogram per task family and per operator, not pooled. Pooled averages wash out a hard subset; DELTA's grokking run stayed below 1% full-pass for about 450 steps on its hard family before jumping to near 100% ([DELTA-Code](https://arxiv.org/abs/2509.21016)), which a pooled curve would hide. **Strong.**

### 2.2 pass@1 vs pass@k

pass@1 measures reliability; pass@k at large k measures coverage (whether the policy can reach a correct answer at all). They diagnose different problems:

- **High pass@1, flat pass@k.** The pool is saturated. More RL on it tends to *shrink* coverage: [ProRL](https://arxiv.org/abs/2505.24864) found a significant negative correlation between base pass@128 and the pass@128 gain after RL, and other work documents support shrinkage and pass@k inversion ([Invisible Leash](https://arxiv.org/abs/2507.14843); [pass@k inversion](https://arxiv.org/abs/2607.20543)).
- **Low pass@1, pass@k > 0.** This is the edge of competence where RL expands the boundary: in a fully controlled pipeline, RL on such "edge" data gave up to +42% pass@128, while RL on in-distribution data gave none ([Interplay](https://arxiv.org/abs/2512.07783)). **Strong** (see §5.3).
- **pass@k = 0 at large k.** Either the task is out of reach, or it is invalid, or the verifier rejects correct answers. Audit first (§2.3), then scaffold (Ch. 5).

Two cautions. Large-k pass@k is inflated on low-entropy answer spaces, where guessing eventually hits the answer; add a breadth–depth metric such as Cover@τ ([Cover@τ](https://arxiv.org/abs/2510.08325)). And estimate pass@k with the unbiased estimator (§6.1), not with 1 − (1 − p̂)ᵏ, which is biased ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)).

### 2.3 Is the verifier making tasks look easy (or hard)?

Verifiers fail in both directions, and both failures look like difficulty.

**False positives make tasks look easy.**
- TACO's tests had a false-positive rate above 90% on difficult problems. On AtCoder difficulty 4+, precision for Qwen2.5-Coder-7B was 21.67 with TACO tests and 60.00 with HardTests ([HardTests](https://arxiv.org/abs/2505.24098)).
- About 1 in 5 "solved" SWE-bench Verified patches from the top-30 agents were semantically wrong; strengthening tests cut the top score from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)). Evolved tests cut pass@1 from 43.80 to 31.22 *on the same problems* ([EvolveCoder](https://arxiv.org/abs/2603.12698)).
- Problems with fewer than 5 tests led to reward hacking by printing memorized answers ([DeepCoder](https://www.together.ai/blog/deepcoder)).
- Reference-based LLM judges accept content-free "master keys" such as "Thought process:"; the false-positive rate was 67.0% for Qwen2.5-72B, and up to 80% across settings ([Master-RM](https://arxiv.org/abs/2507.08794)).
- An audit found material defects in 13 of 105 Reasoning Gym tasks, including a scorer that gives full credit to any nonempty answer, and in 9 SynLogic generators ([Reasoning Core](https://arxiv.org/abs/2603.02208)).
- In agentic benchmarks an empty-response agent passes 38% of τ-bench-Airline tasks, and outcome-validity flaws were found in 7 of 10 benchmarks ([ABC](https://arxiv.org/abs/2507.02825)).

**False negatives make tasks look hard (and push valid hard items to p = 0).**
- Rule-based math checkers average 86% recall, and their false negatives rise as the policy gets stronger ([Huang et al.](https://arxiv.org/abs/2505.22203)). TinyV found false negatives in more than 38% of responses on Big-Math-RL-Verified ([TinyV](https://arxiv.org/abs/2505.14625)).
- More than 4,000 original CodeContests problems had TPR ≤ 0.1, i.e. tests that reject nearly all correct code ([CodeContests+](https://arxiv.org/abs/2506.05817)). ScaleBox found 14.57% of 34,757 code problems need special judges, and exact match rejects 59.01% of correct solutions to them ([ScaleBox](https://arxiv.org/abs/2604.27467)).

**Why it matters for training.** A verifier with Youden's J = TPR − FPR > 0 still drives incorrect modes extinct; noise mostly sets the *rate*. At J < 0 incorrect modes grow until they dominate ([RLVεR](https://arxiv.org/abs/2601.04411)). Precision matters more for rejection fine-tuning: admitting 25% false positives cost −1.58pp of transfer, while masking 75% of correct candidates cost −0.03pp. Under GRPO at matched TPR − FPR the false-negative arm drew more gradient ([The Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)). And the gap grows with difficulty: RLVR-trained models showed 40 verifier shortcuts at complexity levels 1–10 but 458 at levels 11–20 ([IPT](https://arxiv.org/abs/2604.15149)).

**Recommendation.** Before any complexification, score every verifier on known-good and known-bad candidates and log TPR, TNR and J per family and difficulty bucket. Hardening the verifier is often the cheapest and safest way to make an item hard again (Ch. 2 operator "verifier/test strengthening"; Ch. 4). **Strong.**

### 2.4 Contamination and memorization

A model that has memorized its eval or training items shows high accuracy that no amount of "harder" data will explain. Evidence that this is common, not exotic:

- From 60% of each MATH-500 prompt, Qwen2.5-Math-7B reproduced the remaining 40% exactly 54.6% of the time; Llama3.1-8B did so 3.8% of the time. On a later benchmark (LiveMathBench 202505) the completion rate fell to 0.0% ([RandomCalculation](https://arxiv.org/abs/2507.10532)).
- DeepMath's raw candidate pool contained 90% of AIME24 and AMC23 and 76.6% of MATH500 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)). Evol-CodeAlpaca overlapped 70.7% of HumanEval test items (note 01).
- n-gram checks miss paraphrases; a 13B model trained on rephrased test data reached GPT-4-level scores on MMLU, GSM8K and HumanEval ([LLM Decontaminator](https://arxiv.org/abs/2311.04850)). After GRPO, contamination inflated scores on related *uncontaminated* benchmarks too ([Kocyigit & Yildirim](https://arxiv.org/abs/2601.06103)).
- Fine-tuned models reach near-perfect accuracy on training Knights-and-Knaves puzzles but fail slight perturbations; role-flipped accuracy is near zero ([Xie et al.](https://arxiv.org/abs/2410.23123)). MathArena found strong signs of contamination in AIME 2024 ([MathArena](https://arxiv.org/abs/2505.23281)).

**Diagnostics.** (a) Partial-prompt completion on eval and training items. (b) A freshly generated procedural control whose answers were never published (RandomCalculation-style). (c) Renamed, reordered or minimally perturbed variants (the K&K perturber; isomorphic relabelling from [IPT](https://arxiv.org/abs/2604.15149)). (d) Post-cutoff items for both the student and the generator. Decontaminate the *complexified outputs* as well as the seeds, because a generator asked for a "hard, novel" problem can regurgitate a known one ([LLM Decontaminator](https://arxiv.org/abs/2311.04850); note 02). **Strong.**

### 2.5 Format leakage and shortcut solvability

A task can be "solved" without the capability it is meant to train. Known leak channels and the screens labs use:

| Leak | Evidence | Screen |
|---|---|---|
| Guessable answer formats (MCQ, T/F, small answer spaces) | Kimi k1.5 removes a prompt if a no-CoT guess is right within 8 tries ([Kimi k1.5](https://arxiv.org/abs/2501.12599)); TRACE found about 28% spurious guessing (right answer, invalid reasoning) in mid-sized models ([TRACE](https://arxiv.org/abs/2607.04784)) | No-CoT guessing filter; convert MCQ to open-ended with canonical answers ([Big-Math](https://arxiv.org/abs/2502.17387); [DAPO](https://arxiv.org/abs/2503.14476)) |
| Generator answer marginals | LLM MCQ generators put the answer first 47.9–57.9% of the time vs 25% uniform ([Tang et al.](https://arxiv.org/abs/2605.01846)) | Code RNG for positions and magnitudes; chi-square test on marginals |
| Partial-input solvability | LLM-elicited NLI is 86–96% solvable from the hypothesis alone ([Proebsting & Poliak](https://arxiv.org/abs/2410.08996)); choices-only prompting beats the majority baseline in 11 of 12 settings ([Balepur et al.](https://arxiv.org/abs/2402.12483)) | Question-only, choices-only and answer-prior baselines per generator family |
| Tool-free or prior-knowledge answers (search/agent tasks) | GLM-5 drops questions a tool-free model answers in ≥ 1 of 8 tries ([GLM-5](https://arxiv.org/abs/2602.15763)); in one deep-search dataset the answer appeared at step 3.4 and 27.2% were answerable from priors ([FORT](https://arxiv.org/abs/2606.12087)) | Tool-free and shallow-agent filters; answer-hit-time logging |
| Answer-format overfitting | Integer-only seed answers caused format overfitting in SvS ([SvS](https://arxiv.org/abs/2508.14029)) | Mix answer formats |
| Reward-format shortcuts | A single sample whose reward accepted a bare `\boxed{40}` collapsed mean response length from 510.7 to 45.7 tokens within 58 steps, before accuracy fell ([Cheng et al.](https://arxiv.org/abs/2605.28388)) | Audit 0-pass and anomalous items; monitor length online |

Remove shortcuts *before* hardening, or the hardening adds guessability rather than reasoning (note 12). **Strong** (every major lab report reviewed does some version of this).

### 2.6 Eval saturation is not training saturation

Two different things get called "too easy", and they have different cures.

| | Eval not saturated | Eval saturated |
|---|---|---|
| **Training pool not saturated** | Normal regime. If gains are small, look at noise (below), verifier, or data mix. | Eval too easy or contaminated: build harder, fresher evals (post-cutoff, perturbed, held-out generator families). Do not harden training data because of it. |
| **Training pool saturated** | The classic case this report addresses: complexify, re-profile, add families. RL on a saturated fixed pool can *hurt*: fixed-data DAPO took Qwen3-4B-Thinking from 72.4 to 64.8, while self-evolving environments raised it to 74.8 ([EvoEnv](https://arxiv.org/abs/2605.14392)). | Both saturated: need new families or a new capability axis (Ch. 2, Ch. 10). |

Further cautions:
- **Training accuracy is not a stopping signal.** In 1-shot RLVR, training accuracy saturated before step 100 while test accuracy kept rising ("post-saturation generalization") ([Wang et al.](https://arxiv.org/abs/2504.20571)). Mixing RL domains gave lower training reward but not lower downstream scores ([Olmo 3](https://arxiv.org/abs/2512.13961)).
- **Saturated evals are noisy evals.** Pass@1 SD across seeds is 5–15 points on AIME/AMC-sized sets, one question moves AIME24 by 2.5–3.3 points, and hardware or batch size alone moved BF16 greedy accuracy by up to 9% ([Sober Look](https://arxiv.org/abs/2504.07086); [Yuan et al.](https://arxiv.org/abs/2506.09501)). At p ≈ 0.5 an unpaired 30-item comparison needs about a 25-point gap for p < 0.05 (derived in note 19 from p(1 − p)/N). A "no gain from harder data" pilot on a 30-item eval says almost nothing.
- **Harness can dominate difficulty.** A 4B SWE agent went from 8.3% to 37.2% by changing harness alone; 96% of runs in the original harness hit the turn limit ([FrogNano](https://arxiv.org/abs/2609.07925)). Tower-of-Hanoi move lists can exceed output-token limits, which makes a capability limit out of a budget limit ([Lawsen](https://arxiv.org/abs/2506.09250)). Profile difficulty inside the exact RL harness and budget.
- **Retention is part of saturation.** About 21% of review records for mastered prompts contained both correct and incorrect rollouts under GRPO ([ReMind](https://arxiv.org/abs/2606.03087)). "Solved" is not permanent; keep a small review queue (Ch. 5).

### 2.7 The checklist as code

```python
def diagnose_pool(pool, policy, verifier, harness, k=16):
    """Run before generating a single new task. Returns per-family findings."""
    report = {}
    for fam, items in group_by(pool, "family"):
        R = rollouts(policy, items, n=k, harness=harness)          # exact training config
        p = [mean(verifier(x, y) for y in R[x]) for x in items]
        report[fam] = dict(
            hist        = histogram(p, bins=[0, 1e-9, .125, .25, .5, .75, .875, 1-1e-9, 1]),
            eff_ratio   = mean(0 < pi < 1 for pi in p),             # effective-prompt ratio
            passk       = unbiased_pass_at_k(R, ks=[1, 8, 64]),     # §6.1
            # verifier audit (§2.3): known-bad must fail, known-good must pass
            fpr         = mean(verifier(x, bad) for x, bad in near_misses(items)),
            fnr         = 1 - mean(verifier(x, g) for x, g in equivalent_rewrites(items)),
            # contamination (§2.4) and shortcuts (§2.5)
            completion  = partial_prompt_completion_rate(policy, items, frac=0.6),
            no_cot_hit  = mean(any(guess(policy, x, cot=False) == ans(x) for _ in range(8))
                               for x in items),
            fresh_ctrl  = pass_rate(policy, regenerate_fresh(items)),   # new params / names
            cap_hits    = mean(hit_turn_or_token_cap(r) for x in items for r in R[x]),
        )
    return report
# Complexify only families whose failure is "high p̂ on a sound verifier, clean items,
# no shortcut". Everything else goes to the owner chapter in the table above.
```

---
