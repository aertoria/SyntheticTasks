# Diagnosis: why tasks look "too easy", what "hard" should mean, and where learning signal comes from

> **Key takeaways**
>
> - **"Too easy" is often a measurement artifact, so diagnose before you synthesize.** Look at the per-prompt pass-rate histogram under the exact policy, harness and verifier, not at the mean. Then rule out lenient verifiers, contamination and format shortcuts. Weak tests alone can make hard tasks look solved: TACO's tests had a false-positive rate above 90% on difficult problems, and strengthening SWE-bench Verified tests cut the top score from 78.80% to 62.20%.
> - **Difficulty is a property of a (task, policy, verifier, budget) tuple, not of the task.** GRPO-style RL learns only from prompts with 0 < p < 1. The chance that a group of N rollouts carries any gradient is 1 − pᴺ − (1 − p)ᴺ. Across more than 30 papers and lab reports, nearly every pipeline drops or specially handles p = 0 and p = 1, centres on roughly 0.3–0.6, and re-profiles as the policy moves.
> - **SFT and RL need different things from "hard".** SFT needs questions the teacher can solve that elicit long, diverse reasoning, and it tolerates wrong answers. RL needs in-band items with a precise verifier. The best-supported division of labour is to install atoms, formats and behaviours with SFT or mid-training, then run RL on compositions of them.
> - **Controlled studies agree on three things.** (1) Composition beats paraphrase: RL on composed tasks builds new skills and RL on atoms does not. (2) Edge-of-competence data expands pass@k, while in-distribution data only sharpens pass@1. (3) Difficulty coverage and diversity matter more than ordering, except when the top tier would otherwise get zero reward.
> - **Measure difficulty with enough samples and treat p̂ = 0 as "unknown".** At p = 0.5, an 8-rollout estimate has a 95% half-width of about ±0.35 (derived). Greedy decoding plus activation perturbations recovered 10–29% of a pass@6 = 0 stratum, so a small-k zero is weak evidence that an item is truly out of reach (though, the authors caution, not proof that ordinary sampling would reach it). Use Bayesian trackers, cheap priors and trajectory signatures, and validate every gain against a random-reward control, freshly generated items and a second model family.

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
| 0.125 or 0.875 | 0.41 | 0.66 | 0.88 | 1.00 | 18 |
| 0.25 or 0.75 | 0.68 | 0.90 | 0.99 | 1.00 | 8 |
| 0.5 | 0.88 | 0.99 | 1.00 | 1.00 | 4 |

*Derived from the formula above. The last column is the smallest n with 1 − (1 − p_min)ⁿ ≥ 0.9, to within rounding (229 and 8 rollouts reach 89.99%). The 229-rollout figure for p = 0.01 matches the worked example in [Knapsack RL](https://arxiv.org/abs/2509.25849).*

A non-zero group is necessary but not sufficient. The theory and allocation papers agree on *how much* signal each band carries:

- Under KL-regularized RL, the reverse KL between the initial policy and the optimal regularized policy (a measure of how far a prompt can move the policy) is lower-bounded by p(1 − p)/(2β²). The bound is zero at p ∈ {0, 1} and maximal at p = ½ ([Bae et al., 2025](https://arxiv.org/abs/2504.03380)). "Learnability" selection ranks prompts by the same quantity, p(1 − p) ([LILO](https://arxiv.org/abs/2502.12272)).
- An information-gain proxy p(1 − p)² peaks at p = 1/3, i.e. it favours the harder side of the middle ([Knapsack RL](https://arxiv.org/abs/2509.25849)).
- Under vanilla GRPO the cumulative absolute advantage peaks at medium accuracy, so low-accuracy prompts are under-weighted *even when present* ([DARS](https://arxiv.org/abs/2508.13755)). Non-linear objectives such as the log-likelihood of success weight each prompt's gradient by 1/p, which can be realized by sampling n ∝ 1/√p rollouts with a residual 1/√p weight ([Reinforce-Ada](https://arxiv.org/abs/2510.04996)).

How much of a real pool is silent? Late in training on DAPO-Math-17K with N = 8, about 40% of Qwen2.5-Math-7B's prompts were all-correct and about 20% all-wrong ([Knapsack RL](https://arxiv.org/abs/2509.25849)). About 39% of rollouts go to silent groups under uniform sampling ([ThinkPrior](https://arxiv.org/abs/2609.09075)), and zero-variance groups are "typically about 40%" in agentic RL ([Selective Rollout](https://arxiv.org/abs/2605.05802)). A static pool with a low difficulty ceiling drives the effective-prompt ratio (share of groups with non-identical rewards) to zero ([RLVE](https://arxiv.org/abs/2511.07317)).

```
 pass rate p (current policy, verifier, budget)
 0 ───────┬─────────────┬───────────────────────┬─────────────┬─────── 1
  p = 0   │ rare        │ learnable band        │ brittle     │ p = 1
  silent: │ success:    │ ~0.2–0.7, signal      │ mastery:    │ silent:
  audit,  │ more        │ peaks near 1/3–1/2:   │ light       │ complexify,
  certify,│ rollouts,   │ TRAIN HERE            │ review      │ meta-task
  scaffold│ reweight    │                       │ queue       │ or retire
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

Report the histogram per task family and per operator, not pooled. Pooled averages wash out a hard subset; DELTA's grokking run stayed below 1% full-pass for about 450 steps on Manufactoria-HAS (0% full-pass at pass@128 for the base model) before jumping to near 100% ([DELTA-Code](https://arxiv.org/abs/2509.21016)), which a pooled curve would hide. **Moderate.**

### 2.2 pass@1 vs pass@k

pass@1 measures reliability; pass@k at large k measures coverage (whether the policy can reach a correct answer at all). They diagnose different problems:

- **High pass@1, flat pass@k.** The pool is saturated. More RL on it tends to *shrink* coverage: [ProRL](https://arxiv.org/abs/2505.24864) found a significant negative correlation between base pass@128 and the pass@128 gain after RL, and other work documents support shrinkage and pass@k inversion ([Invisible Leash](https://arxiv.org/abs/2507.14843); [pass@k inversion](https://arxiv.org/abs/2607.20543)).
- **Low pass@1, pass@k > 0.** This is the edge of competence where RL expands the boundary: in a fully controlled pipeline, RL on such "edge" data gave up to +42% pass@128, while RL on in-distribution data gave none ([Interplay](https://arxiv.org/abs/2512.07783)). **Strong** (see §5.3).
- **pass@k = 0 at large k.** Either the task is out of reach, or it is invalid, or the verifier rejects correct answers. Audit first (§2.3), then scaffold (Ch. 5).

Two cautions. Large-k pass@k is inflated on low-entropy answer spaces, where guessing eventually hits the answer; add a breadth–depth metric such as Cover@τ ([Cover@τ](https://arxiv.org/abs/2510.08325)). And estimate pass@k with the unbiased estimator (§6.1), not with 1 − (1 − p̂)ᵏ, which is biased ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)).

### 2.3 Is the verifier making tasks look easy (or hard)?

Verifiers fail in both directions, and both failures look like difficulty.

**False positives make tasks look easy.**
- TACO's tests had a false-positive rate above 90% on difficult problems. On AtCoder difficulty 4+, test precision on Qwen2.5-Coder-7B's programs was 21.67 with TACO tests and 60.00 with HardTests ([HardTests](https://arxiv.org/abs/2505.24098)).
- About 1 in 5 "solved" SWE-bench Verified patches from the top-30 agents were semantically wrong; strengthening tests cut the top score from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)). Evolved tests cut pass@1 from 43.80 to 31.22 *on the same problems* ([EvolveCoder](https://arxiv.org/abs/2603.12698)).
- Problems with fewer than 5 tests led to reward hacking by printing memorized answers ([DeepCoder](https://www.together.ai/blog/deepcoder)).
- Reference-based LLM judges accept content-free "master keys" such as "Thought process:"; the false-positive rate was 67.0% for Qwen2.5-72B, and up to 80% across settings ([Master-RM](https://arxiv.org/abs/2507.08794)).
- An audit found material defects in 13 of 105 Reasoning Gym tasks, including a scorer that gives full credit to any nonempty answer, and in 9 SynLogic generators ([Reasoning Core v3](https://arxiv.org/abs/2608.05148)).
- In agentic benchmarks an empty-response agent passes 38% of τ-bench-Airline tasks, and outcome-validity flaws were found in 7 of 10 benchmarks ([ABC](https://arxiv.org/abs/2507.02825)).

**False negatives make tasks look hard (and push valid hard items to p = 0).**
- Rule-based math checkers average 86% recall, and their false negatives rise as the policy gets stronger ([Huang et al.](https://arxiv.org/abs/2505.22203)). TinyV found false negatives in more than 38% of responses on Big-Math-RL-Verified ([TinyV](https://arxiv.org/abs/2505.14625)).
- More than 4,000 original CodeContests problems had TPR ≤ 0.1, i.e. tests that reject nearly all correct code ([CodeContests+](https://arxiv.org/abs/2506.05817)). ScaleBox found 14.57% of 34,757 code problems need special judges, and exact match rejects 59.01% of correct solutions to them ([ScaleBox](https://arxiv.org/abs/2604.27467)).

**Why it matters for training.** A verifier with Youden's J = TPR − FPR > 0 still drives incorrect modes extinct; noise mostly sets the *rate*. At J < 0 incorrect modes grow until they dominate ([RLVεR](https://arxiv.org/abs/2601.04411)). Precision matters more for rejection fine-tuning: admitting 25% false positives cost −1.58pp of transfer, while masking 75% of correct candidates cost −0.03pp. Under GRPO at matched TPR − FPR the false-negative arm drew more gradient ([The Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)). And the gap grows with difficulty: RLVR-trained models showed 40 verifier shortcuts at complexity levels 1–10 but 458 at levels 11–20 ([IPT](https://arxiv.org/abs/2604.15149)).

**Recommendation.** Before any complexification, score every verifier on known-good and known-bad candidates and log TPR, TNR and J per family and difficulty bucket. Hardening the verifier is often the cheapest and safest way to make an item hard again (Ch. 2 operator B8, "verifier and rubric hardening"; Ch. 4). **Strong.**

### 2.4 Contamination and memorization

A model that has memorized its eval or training items shows high accuracy that no amount of "harder" data will explain. Evidence that this is common, not exotic:

- From 60% of each MATH-500 prompt, Qwen2.5-Math-7B reproduced the remaining 40% exactly 54.6% of the time; Llama3.1-8B did so 3.8% of the time. On a later benchmark (LiveMathBench 202505) the completion rate fell to 0.0% ([RandomCalculation](https://arxiv.org/abs/2507.10532)).
- DeepMath's raw candidate pool contained 90% of AIME24 and AMC23 and 76.6% of MATH500 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)). Evol-CodeAlpaca overlapped 70.7% of HumanEval test items ([Tülu 3](https://arxiv.org/abs/2411.15124)).
- n-gram checks miss paraphrases; a 13B model trained on rephrased test data reached GPT-4-level scores on MMLU, GSM8K and HumanEval ([LLM Decontaminator](https://arxiv.org/abs/2311.04850)). After GRPO, contamination inflated scores on related *uncontaminated* benchmarks too ([Kocyigit & Yildirim](https://arxiv.org/abs/2601.06103)).
- Fine-tuned models reach near-perfect accuracy on training Knights-and-Knaves puzzles but fail slight perturbations; role-flipped accuracy is near zero ([Xie et al.](https://arxiv.org/abs/2410.23123)). MathArena found strong signs of contamination in AIME 2024 ([MathArena](https://arxiv.org/abs/2505.23281)).

**Diagnostics.** (a) Partial-prompt completion on eval and training items. (b) A freshly generated procedural control whose answers were never published (RandomCalculation-style). (c) Renamed, reordered or minimally perturbed variants (the K&K perturber; isomorphic relabelling from [IPT](https://arxiv.org/abs/2604.15149)). (d) Post-cutoff items for both the student and the generator. Decontaminate the *complexified outputs* as well as the seeds, because a generator asked for a "hard, novel" problem can regurgitate a known one ([LLM Decontaminator](https://arxiv.org/abs/2311.04850); note 02). **Strong.**

### 2.5 Format leakage and shortcut solvability

A task can be "solved" without the capability it is meant to train. Known leak channels and the screens labs use:

| Leak | Evidence | Screen |
|---|---|---|
| Guessable answer formats (MCQ, T/F, small answer spaces) | Kimi k1.5 removes a prompt if a no-CoT guess is right within 8 tries ([Kimi k1.5](https://arxiv.org/abs/2501.12599)); TRACE found about 28% spurious guessing (right answer, invalid reasoning) in mid-sized models ([TRACE](https://arxiv.org/abs/2607.04784)) | No-CoT guessing filter; convert MCQ to open-ended with canonical answers ([Big-Math](https://arxiv.org/abs/2502.17387); [DAPO](https://arxiv.org/abs/2503.14476)) |
| Generator answer marginals | Two Llama MCQ generators put the answer first 47.1–57.9% of the time vs 25% uniform ([Tang et al.](https://arxiv.org/abs/2605.01846)) | Code RNG for positions and magnitudes; chi-square test on marginals |
| Partial-input solvability | LLM-elicited NLI is 86–96% solvable from the hypothesis alone ([Proebsting & Poliak](https://arxiv.org/abs/2410.08996)); choices-only prompting beats the majority baseline in 11 of 12 settings ([Balepur et al.](https://arxiv.org/abs/2402.12483)) | Question-only, choices-only and answer-prior baselines per generator family |
| Tool-free or prior-knowledge answers (search/agent tasks) | GLM-5 drops questions a tool-free model answers in ≥ 1 of 8 tries ([GLM-5](https://arxiv.org/abs/2602.15763)); in one deep-search dataset the answer appeared at step 3.4 and 27.2% were answerable from priors ([FORT](https://arxiv.org/abs/2606.12087)) | Tool-free and shallow-agent filters; answer-hit-time logging |
| Answer-format overfitting | Integer-only seed answers caused format overfitting in SvS ([SvS](https://arxiv.org/abs/2508.14029)) | Mix answer formats |
| Reward-format shortcuts | A single sample whose reward accepted a bare `\boxed{40}` collapsed mean response length from 510.7 to 45.7 tokens within 58 steps, before accuracy fell ([Cheng et al.](https://arxiv.org/abs/2605.28388)) | Audit 0-pass and anomalous items; monitor length online |

Remove shortcuts *before* hardening, or the hardening adds guessability rather than reasoning (note 12). **Strong** (Qwen3, Kimi k1.5, GLM-5, LongCat, Seed1.5-Thinking and Magistral all report some version of this).

### 2.6 Eval saturation is not training saturation

Two different things get called "too easy", and they have different cures.

| | Eval not saturated | Eval saturated |
|---|---|---|
| **Training pool not saturated** | Normal regime. If gains are small, look at noise (below), verifier, or data mix. | Eval too easy or contaminated: build harder, fresher evals (post-cutoff, perturbed, held-out generator families). Do not harden training data because of it. |
| **Training pool saturated** | The classic case this report addresses: complexify, re-profile, add families. RL on a saturated fixed pool can *hurt*: fixed-data DAPO took Qwen3-4B-Thinking from 72.4 to 64.8, while self-evolving environments raised it to 74.8 ([EvoEnv](https://arxiv.org/abs/2605.14392)). | Both saturated: need new families or a new capability axis (Ch. 2, Ch. 10). |

Further cautions:
- **Training accuracy is not a stopping signal.** In 1-shot RLVR, training accuracy saturated before step 100 while test accuracy kept rising ("post-saturation generalization") ([Wang et al.](https://arxiv.org/abs/2504.20571)). Mixing RL domains gave lower training reward but not lower downstream scores ([Olmo 3](https://arxiv.org/abs/2512.13961)).
- **Saturated evals are noisy evals.** Pass@1 SD across seeds is 5–15 points on AIME/AMC-sized sets, one question moves AIME24 or AMC23 by 3.3 or 2.5 points, and hardware or batch size alone moved BF16 greedy accuracy by up to 9% ([Sober Look](https://arxiv.org/abs/2504.07086); [Yuan et al.](https://arxiv.org/abs/2506.09501)). At p ≈ 0.5 an unpaired 30-item comparison needs about a 25-point gap for p < 0.05 (derived in note 19 from p(1 − p)/N). A "no gain from harder data" pilot on a 30-item eval says almost nothing.
- **Harness can dominate difficulty.** A 4B SWE agent went from 8.3% to 37.2% by changing harness alone; 96% of runs in the original harness hit the turn limit ([FrogNano](https://arxiv.org/abs/2609.07925)). Tower-of-Hanoi move lists can exceed output-token limits, which makes a capability limit out of a budget limit ([Lawsen](https://arxiv.org/abs/2506.09250)). Profile difficulty inside the exact RL harness and budget.
- **Retention is part of saturation.** About 21% of review records for mastered prompts (269 of 1,275) contained both correct and incorrect rollouts ([ReMind](https://arxiv.org/abs/2606.03087)). "Solved" is not permanent; keep a small review queue (Ch. 5).

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

## 3. Difficulty as a policy-relative quantity

"Difficulty" in this report means the success probability p(x; π, V, B) of task x under policy π, verifier V and budget B (tokens, turns, tools). Change any one of the four and p changes: a stronger checkpoint raises it, a stricter verifier lowers it (§2.3), a different harness or turn limit can change it several-fold (§2.6). Human difficulty labels (AoPS level, Codeforces rating) and LLM ratings are only priors for p (§6.4).

### 3.1 Pass-rate bands reported across papers

The tables group the explicit bands found in the notes by what each band controls. k is the number of rollouts used to estimate p.

**(a) Offline or online selection filters on existing prompts**

| Source | k | Rule |
|---|---|---|
| [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) | 8 | keep 2–5 of 8 correct |
| [Olmo 3](https://arxiv.org/abs/2512.13961) | 8 | drop p > 62.5% (initial checkpoint, T = 1.0); 32B fills batches with non-zero-gradient groups only |
| [Llama-Nemotron](https://arxiv.org/abs/2505.00949) | 8 | drop p ≥ 0.75; Gaussian batches whose mean moves from easy to hard |
| [MiniMax-M1](https://arxiv.org/abs/2506.13585) | 10 | keep 0 < p < 0.9 (strong reasoning model's pass@10) |
| [MiMo](https://arxiv.org/abs/2505.07608) | 16 | drop p > 90% (removed about 50% of math); resample an easy pool 10% of the time |
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | — | drop p ∈ {0, 1} offline; drop items solved at p = 1 in the previous stage |
| [POLARIS](https://hkunlp.github.io/blog/2025/Polaris/) | 8 | drop perfectly solved; drop p > 0.9 after each phase ("mirrored-J" distribution) |
| [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) | 16 | later stages keep p ≤ 6/16 |
| [rStar2-Agent](https://arxiv.org/abs/2508.20722) | 8 | before stage 3, drop 8/8 on the *original* 42K set → 17.3K |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 10 | sample ∝ (1 − success); drop if a no-CoT guess is right within 8 tries |
| [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) | 8/16 | drop if GPT-OSS-120B solves 8/8; keep 10% of 0-pass SWE items; mask groups with no reward > 0.5 |
| [InternBootcamp](https://arxiv.org/abs/2508.08636) | — | keep 3–85% accuracy |
| [DAPO](https://arxiv.org/abs/2503.14476) | G | drop groups with accuracy exactly 0 or 1, oversample to refill |
| [Bae et al.](https://arxiv.org/abs/2504.03380) | 16 | balanced band symmetric about 0.5; (0.3, 0.7) best |
| [LILO](https://arxiv.org/abs/2502.12272) | 8 | top-\|B\| of a 4\|B\| pool by p̂(1 − p̂) |
| [ScaleRL](https://arxiv.org/abs/2510.13786) | — | drop zero-variance groups; permanently retire p ≥ 0.9 |
| [Prompt Replay](https://arxiv.org/abs/2603.21177) | 16 | replay p ∈ [0.25, 0.75], priority by closeness to 0.5 |
| [Pilot-Commit](https://arxiv.org/abs/2605.26606) | 16 pilot | skip p̂ > 0.75, defer p̂ < 0.125, commit 48 more rollouts to the rest |
| [Trading Human Curation](https://arxiv.org/abs/2606.03800) | 16 | admit variants with pass@8 ∈ [0.05, 0.95] |

**(b) Generator rewards and admission gates for newly synthesized tasks**

| Source | k | Rule |
|---|---|---|
| [R-Zero](https://arxiv.org/abs/2508.05004) | 10 | challenger reward 1 − 2\|p̂ − 0.5\|; train on items with 3–7 of 10 agreeing |
| [Agent0](https://arxiv.org/abs/2511.16043) | 10 | train within \|p̂ − 0.5\| ≤ 0.25 |
| [OpenSIR](https://arxiv.org/abs/2511.00602) | — | triangular solvability window, floor 0.5 and ceiling 0.9 |
| [SPICE](https://arxiv.org/abs/2510.24684) / [Socratic-Zero](https://arxiv.org/abs/2509.24726) | 8 | Gaussian reward peaking at 50% (Socratic-Zero σ = 0.2) |
| [SvS](https://arxiv.org/abs/2508.14029) | 8 | seeds at 12.5–50%; variants rewarded only in [12.5%, 62.5%] |
| [SwS](https://arxiv.org/abs/2506.08989) | — | keep [25%, 75%] (about 35% of generated problems survive) |
| [GASP](https://arxiv.org/abs/2603.15957) | — | easier "lemma" at p ∈ [0.3, 0.7], harder "lift" at p ∈ [0.1, 0.5] |
| [PROPEL](https://arxiv.org/abs/2606.18284) | 8 / 3 | 1/8 ≤ p ≤ 3/8 (math, code); 1/3–2/3 (SWE) |
| [AZR](https://arxiv.org/abs/2505.03335) | G | proposer reward 1 − r̄, and 0 when r̄ = 0 |
| [EvoEnv](https://arxiv.org/abs/2605.14392) | 8 | admit 0 < p < 1, target 0.3 |
| [SQL-Zero](https://arxiv.org/abs/2609.04697) | 5 | difficulty reward peaks at 1 correct of 5 |
| [IFDecorator](https://arxiv.org/abs/2508.04632) | 8 | keep (0, 0.5]; send prompts above 0.5 or at 0 back for further evolution (0-pass leftovers dropped) |
| [QuestA](https://arxiv.org/abs/2507.13266) | 8 | keep hinted prompts at 0–4 of 8 |

**(c) Promotion, graduation and "edge" definitions**

| Source | Rule |
|---|---|
| [RLVE](https://arxiv.org/abs/2511.07317) | raise an environment's difficulty level when accuracy ≥ 0.9 at the top level (window of 4 levels) |
| [Cog-DRIFT](https://arxiv.org/abs/2604.04767) / [ZPPO](https://arxiv.org/abs/2606.18216) | promote or graduate an item at accuracy ≥ 0.5 |
| [Failure-prefix conditioning](https://arxiv.org/abs/2601.20829) | choose prefix length so accuracy ≈ 0.5 (τ = 0.25–0.75 gave 44.2–43.9 vs 44.5 at 0.5) |
| [UltraLogic](https://arxiv.org/abs/2601.03205) | best training when success, minus about 0.1 for formatting, is 40–60% |
| [Knapsack RL](https://arxiv.org/abs/2509.25849) | information-gain proxy peaks at p = 1/3 |
| [Deep Dive](https://arxiv.org/abs/2603.24202) | medium-only training (0.41–0.59 over 32 attempts) gave the best speed/generalization balance; easy items overfit |
| [Interplay](https://arxiv.org/abs/2512.07783) | "edge": low pass@1 but pass@128 > 0 |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | stage-2 tier: pass@8 = 0 and pass@512 ≫ 0, verified-answer pool only |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | keep synthesized agent tasks with pass@100 > 0 |
| [GLM-5](https://arxiv.org/abs/2602.15763) | failed or rarely solved by the previous model, solvable by stronger teachers |

### 3.2 Reading the bands

- **The centre is 0.3–0.6, with a slight lean to the hard side.** p(1 − p)² peaks at 1/3; EvoEnv targets 0.3 because near-half environments saturate quickly as the solver improves; UltraLogic's sweet spot was 40–60%; Deep Dive's best tier was 0.41–0.59. **Strong.**
- **The best band depends on model size.** In UltraLogic, Qwen3-8B gained most from Easy data and Qwen3-14B from Medium. Calibrate on the model you train, not a proxy (POLARIS profiles with "the specific model being trained"; INTELLECT-3's 4B proxy is cheaper but less faithful). **Moderate.**
- **The exact reward curve inside the band is second order.** SSR's consistency-only ±1 reward was only slightly worse than a solve-rate-shaped one; Socratic-Zero's reward-shape variants were within about 0.4 points; a 50%-target reward *lowered* AZR's validation accuracy by 2% ([Chae et al.](https://arxiv.org/abs/2510.27072)). Spend the effort on validity, grounding and diversity (Ch. 3). **Moderate.**
- **The low edge is where risk concentrates.** "Hard examples" help while they keep mixed outcomes: GRPO on the hardest 10% by base failure gave gains of up to 47% versus 3–15% for easy subsets ([Pikus et al.](https://arxiv.org/abs/2508.14094)). Items with *zero* successes behave differently: hard@8 items (pass@8 = 0) lowered averages by 5.75, 11.24 and 1.07 points across three model settings ([Cheng et al.](https://arxiv.org/abs/2605.28388)). Admit p = 0 items only with a solvability certificate and a plan to make them non-silent (scaffolds, extra rollouts; Ch. 5). **Strong.**
- **The high edge should retire, not delete.** Retiring p ≥ 0.9 raises ScaleRL's fitted asymptote, but mastered items regress (about 21% of review records for mastered prompts had mixed outcomes, [ReMind](https://arxiv.org/abs/2606.03087)); in ReMind even a 1% review budget avoided GRPO's mid-training plateau, and 4–10% gave no further gain. **Moderate.**
- **The band moves, so re-profile the whole original pool.** [Magistral](https://arxiv.org/abs/2506.10910) re-grades the entire original set with its RL model; [rStar2-Agent](https://arxiv.org/abs/2508.20722) re-filters the original 42K set; [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) runs a background process that rescans the full pool with intermediate checkpoints; [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) re-profiles at plateaus; [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) alternated training with re-filtering to "medium-to-hard" prompts. Items that looked impossible can come back into the band. **Strong.**

### 3.3 What "hard" should mean

**Definition (Proposal).** A task x is *useful-hard* for policy π at time t if all four hold:

1. **Valid.** Well-posed and uniquely answerable; the verifier accepts known-good and rejects known-bad outputs; solvability is certified by construction, a stronger solver, or large-k success.
2. **Learnable now.** For RL: p̂_π(x) in the band, or low pass@1 with pass@k > 0. For SFT: a teacher solves it and the trace is long and structured.
3. **Capability-bearing.** Not solvable by guessing, partial input, priors or tool-free lookup; the difficulty comes from the target capability (composition, structure, information that must be acquired), not from tedium, ambiguity, output length or format.
4. **New.** Decontaminated against training and eval sets, and not a reskin of items already in the pool.

Criterion 3 is where naive complexification fails. "Structure is not difficulty": across seven deep-search datasets run with the same agent and budget, solving cost Ω ranged from 20.6 to 141.0 and the answer appeared at step 3.4 to 46.9 ([FORT](https://arxiv.org/abs/2606.12087)). LLMs asked to "make it harder" mostly add constraints or produce trivial variants: 64% of rejected mutations in a gated study were "too easy" ([Trading Human Curation](https://arxiv.org/abs/2606.03800)), and SSR's direct prompt produced one-line bugs ([SSR](https://arxiv.org/abs/2512.18552)). Criterion 1 is where aggressive hardening fails: lowering OpenSIR's solve-rate floor from 0.5 to 0.1 dropped validity from 70.8% to 42.3% and math accuracy from 29.6 to 26.0, while problems got only slightly harder ([OpenSIR](https://arxiv.org/abs/2511.00602)); <3% accuracy on a new InternBootcamp environment mostly flagged semantic errors ([InternBootcamp](https://arxiv.org/abs/2508.08636)).

**Difficulty compounds, and each axis scales differently.** Useful when predicting where an operator will land (Ch. 2 has the operators):

| Axis | Scaling evidence |
|---|---|
| Serial depth / horizon | Composite success ≈ product of atom successes (Pearson ρ 0.69–0.96; 0.3⁵ ≈ 0.0024) ([Algebrarium](https://arxiv.org/abs/2602.08281)); long-horizon accuracy falls *faster* than independent compounding predicts ([h1](https://arxiv.org/abs/2510.07312)); self-conditioning on earlier errors ([Sinha et al.](https://arxiv.org/abs/2509.09677)) |
| Width / constraint count | Mean per-constraint success 72.0% × 0.922^(k−1); about 41% per constraint at k = 8 gives 5.7% all-pass ([CSE](https://arxiv.org/abs/2608.12426)); two-skill composition success ≈ the square of single-skill success ([MATH²](https://arxiv.org/abs/2407.21009)) |
| Topology | Right-heavy solution trees stay hardest at equal depth ([Park et al.](https://arxiv.org/abs/2512.01775)); branch–merge dependencies are harder than long chains ([He et al.](https://arxiv.org/abs/2609.19465)) |
| Expressiveness | RL steps to 90% scale as depth^γ with γ from 1.05 to 2.60 as logic gets richer; the richest logic gave the best transfer (+10.66 on an 8-benchmark mean) ([ScaleLogic](https://arxiv.org/abs/2605.06638)) |
| Information hiding | Removing information cut solve rate by 70–100 points ([Trading Human Curation](https://arxiv.org/abs/2606.03800)); the same instances scored 0.962 written out vs 0.204 with parameters behind tools ([VHD-Play](https://arxiv.org/abs/2609.27321)) |
| Tedium (op count, length) | Smooth sigmoid decay with operation count ([GSM-Infinite](https://arxiv.org/abs/2502.05252)); useful for robustness, but difficulty from "computational tedium" is not the same as needing a new method (note 02) |

---

## 4. What SFT needs from hard tasks vs what RL needs

The two regimes extract signal differently, so "make the data harder" means different things for each. Ch. 6 (SFT) and Ch. 5 (RL) give the playbooks; this table is the diagnosis.

| | SFT / distillation | RL (RLVR, GRPO-family) |
|---|---|---|
| Signal source | Teacher tokens on questions the teacher can solve | Contrast between the policy's own successes and failures; needs 0 < p < 1 |
| What "hard" should mean | Hard *question* with a long, structured teacher trace. s1K's difficulty + diversity + length filter gave 50.0 on AIME24 vs 36.7 for a random 1K ([s1](https://arxiv.org/abs/2501.19393)); dropping the bottom 50% of prompts by response length gave +2–4% on math and science ([GLM-4.5](https://arxiv.org/abs/2508.06471)) | In-band for the *current* policy; composed or edge items (§5) |
| Answer correctness | Tolerant. No answer filtering 41.9 vs GPT-verification filtering 40.0 ([OpenThoughts](https://arxiv.org/abs/2506.04178)); long-CoT samples with wrong answers cost only 3.2% ([Li et al.](https://arxiv.org/abs/2502.07374)); priming on incorrect-answer traces matched correct ones ([Gandhi et al.](https://arxiv.org/abs/2503.01307)) | Intolerant: a wrong key is a wrong reward. A guessable wrong label gave 57.0 on MATH500 vs 73.4 with the correct label ([1-shot RLVR](https://arxiv.org/abs/2504.20571)); 1–3 buggy task types out of 50 collapsed training ([UltraLogic](https://arxiv.org/abs/2601.03205)) |
| Easy data | Can suffice to *elicit knowledge*: easy-to-hard often matched a hard-data oracle ([Hase et al.](https://arxiv.org/abs/2401.06751)) | Produces zero advantage; hardest 10% gave up to 47% vs 3–15% for easy subsets ([Pikus et al.](https://arxiv.org/abs/2508.14094)) |
| Generalization and forgetting | Memorizes rules: −8.1 (GP-L) and −79.5 (V-IRL-L) on OOD rule variants, while RL gained +3.5 and +11.0; but RL without SFT failed because the base could not follow the format ([Chu et al.](https://arxiv.org/abs/2501.17161)). Math-only SFT dropped IFEval 69.2 → 42.3; RL kept 70.0 ([Huan et al.](https://arxiv.org/abs/2507.00432)) | Forgets less because its data is on-policy ([Chen et al.](https://arxiv.org/abs/2510.18874)) |
| Composition | Next-token training on composed data gives little: iterative RFT reached 15% on Level 2 and ≤ 2.6% on Level 3 vs RL's 64% and 27% ([f(g(x))](https://arxiv.org/abs/2509.25123)) | Learns new composite skills from atoms (§5.1) |
| Small students | Over-complex teacher traces can *lower* accuracy for 1.5B/3B models ([BREAD](https://arxiv.org/abs/2506.17211)) | Needs some successes; scaffold rather than imitate |
| Diversity | Unique prompts matter more than responses per prompt (regression coefficients 4.831 vs 2.635), though more responses still lifted AIME25 41.3 → 49.3 ([AceReason 1.1](https://arxiv.org/abs/2506.13284)); route-diverse SFT traces raised post-RL pass@8 by 16.9 points ([Zhang et al.](https://arxiv.org/abs/2609.33780)) | Family and environment breadth: more environments improved held-out results monotonically ([RLVE](https://arxiv.org/abs/2511.07317)); task diversity was UltraLogic's main driver |

Recommendations:

- **Install with SFT or mid-training, compose with RL.** SFT should cover atoms, output formats and behaviours (verify, backtrack, retry) through compositional traces; RL should target novel compositions outside SFT support ([Kong et al.](https://arxiv.org/abs/2606.18089)). RL did not transfer to a context with 0% or 0.1% pretraining exposure but gave up to +60% pass@128 at 1% ([Interplay](https://arxiv.org/abs/2512.07783)). **Strong.**
- **If behaviours are missing, prime before hardening.** Llama-3.2-3B plateaued near 30% on Countdown where Qwen-2.5-3B reached about 60% under identical PPO; brief SFT on backtracking/verification traces closed the gap ([Gandhi et al.](https://arxiv.org/abs/2503.01307)). **Moderate.**
- **For SFT spend on question selection and teacher choice; for RL spend on verifiers.** **Strong** (OpenThoughts, HardTests, SWE-smith, note 11).
- **Do not pick the RL starting checkpoint by SFT accuracy.** SkillFactory scored 2.8% vs 11.7% for R1-distillation on hard Countdown before GRPO, and 25.1% vs 21.2% after ([SkillFactory](https://arxiv.org/abs/2512.04072)); Mixed SFT had the lowest pre-RLVR accuracy but the highest post-RLVR ceiling, using over 60× less GPU time than next-chunk RL ([Tang et al.](https://arxiv.org/abs/2608.23256)). Run a short RL probe. **Moderate.**
- **If the gap is coverage (large-k pass@k) and a stronger teacher exists, distillation is the direct lever.** RL "squeezes" and SFT "expands" ([Matsutani et al.](https://arxiv.org/abs/2509.21128)); distillation lifts the whole pass@k curve ([Yue et al.](https://arxiv.org/abs/2504.13837)); by contrast, consolidating same-base RL domain experts by merging, mixed RL or multi-teacher on-policy distillation left AIME pass@32 indistinguishable from the base; there the "teachers" were the same-base RL experts, not a stronger model ([Wu et al.](https://arxiv.org/abs/2608.27409)). **Moderate.**

---

## 5. What theory and controlled studies say

### 5.1 Composition is what RL learns

- On string-function nesting, RL on Level-1 atoms stayed below 25% on Level 2 and near 0 on Levels 3–6. RL on Level-2 compositions lifted held-out Level 3 from about 5% to about 30% and Level 4 from about 1% to about 15%. At Level 5 the gap over the RFT base grew from 4% at pass@1 to about 25% at pass@1024 ([f(g(x))](https://arxiv.org/abs/2509.25123)).
- Training on decomposed skills does not transfer up; training on composed tasks transfers down to the parts. This held across three compositionality levels and three model families, and in a BFCL tool-calling pilot ([He et al.](https://arxiv.org/abs/2609.19465)). OMEGA's "A and B trained separately" regime gave little or no compositional gain ([OMEGA](https://arxiv.org/abs/2506.18880)).
- Chaining GSM8K problems with a horizon curriculum doubled AIME24 for a 3B instruct model (5.10 → 10.52 avg@32) ([h1](https://arxiv.org/abs/2510.07312)). In a rewrite-grammar world, RL first sharpened primitives and then discovered reusable compositions, while RFT plateaued with invalid shortcuts ([Abdulsalam et al.](https://arxiv.org/abs/2607.07646)).
- Composition needs reliable atoms. RLVR moved 22.6% of null-set tasks into feasibility, and composite accuracy tracked joint atomic accuracy ([Algebrarium](https://arxiv.org/abs/2602.08281)).

**Implication.** If seeds are about 100% solved, compose them; paraphrasing will not add signal. **Strong** (six independent controlled studies).

### 5.2 Easy-to-hard generalization: what transfers and what does not

- *Transfers:* larger instances of the same procedure (exploratory generalization; e.g. OMEGA's Zebra logic +61 ID and +53 OOD from a 30% base), longer unseen horizons after horizon-reduced training ([Kim et al.](https://arxiv.org/abs/2605.02572)), self-labelled length extension with strong filtering (reverse addition from 16 to 100+ digits; [Lee et al.](https://arxiv.org/abs/2502.01612)), and evaluators trained on easy MATH levels applied to hard ones ([Sun et al.](https://arxiv.org/abs/2403.09472)).
- *Plateaus:* depth generalization falls to chance at about 3× the training depth ([ScaleLogic](https://arxiv.org/abs/2605.06638)); OMEGA's GCD level 5 stayed at 3% after training on levels 1–4.
- *Does not transfer:* new strategies. Transformative generalization was usually 0% after RL (OMEGA, [DELTA-Code](https://arxiv.org/abs/2509.21016)), and horizon-trained Sudoku models did not generalize to harder solving techniques ([Kim et al.](https://arxiv.org/abs/2605.02572)).
- *Theory:* a k-fold composition task needs samples exponential in k for any SQ learner with polynomially many queries, yet gradient descent on an O(log k)-depth transformer learns it with poly(k) samples if the data also contains shallower compositions, as a curriculum *or all mixed together* ([Wang et al.](https://arxiv.org/abs/2505.23683)). For RLVR, recursive decomposition weakens the needed coverage from length T to block length B ≪ T ([Curriculum II](https://arxiv.org/abs/2606.27721)).

**Implication.** Never train only on the deepest composites; keep shallower instances in the mix. Budget separate operators for "longer" and "conceptually harder". **Strong.**

### 5.3 pass@k: sharpening vs boundary expansion

The 2025 critique: on standard, largely in-distribution data, base models solve more at large k. For a 32B model on Minerva, the base solved about 9% more at k = 128; on AIME24 no problem was solved only by the RL model ([Yue et al.](https://arxiv.org/abs/2504.13837)). Self-play without new information behaves the same way: AZR mostly sharpens and entropy still collapses, although its large-k deficit was not statistically significant ([Chae et al.](https://arxiv.org/abs/2510.27072)).

The follow-up evidence: expansion appears when the data is composed or at the edge of competence.

| Setting | Large-k result |
|---|---|
| Edge data (op 11–14) vs in-distribution | up to +42% pass@128 vs none ([Interplay](https://arxiv.org/abs/2512.07783)) |
| Answer-preserving variants of the policy's own weak problems | +18.3 / +22.8 pass@32 on AIME24 / AIME25 over RLVR ([SvS](https://arxiv.org/abs/2508.14029)) |
| Boundary-aware curriculum RL | +9.8 pass@256 over the base ([Cai et al.](https://arxiv.org/abs/2606.22317)) |
| Prolonged, diverse RL | solves an unseen Reasoning Gym task where the base has zero capability ([ProRL](https://arxiv.org/abs/2505.24864)) |
| Restricting updates to zero-success problems | pass@256 above the base ([Yuan et al.](https://arxiv.org/abs/2606.15455)) |

Over-training on saturated items shrinks the boundary (support shrinkage, [Wu et al.](https://arxiv.org/abs/2507.14843); winner-take-all toward high-likelihood problems, [Nguyen et al.](https://arxiv.org/abs/2510.02230); pass@k inversion on rare-path prompts, [Zhou](https://arxiv.org/abs/2607.20543)). Claims about crossovers need statistics: across five public RLVR pairs no crossover was statistically established until fresh 32k-token prompts were used ([Yang & Chen](https://arxiv.org/abs/2609.22547)).

**Implication.** Use large-k pass@k, the null-set unlock rate and Cover@τ, with paired confidence bands, as the acceptance test for a complexified dataset. If RL-only coverage stays about 0%, the data is too easy or too far out of reach. **Strong.**

### 5.4 Prolonged RL and saturated models

- More than 2k RL steps with reference and optimizer resets expanded boundaries for a 1.5B model, but gains were largest where the base was weak and could be negative where it was strong ([ProRL](https://arxiv.org/abs/2505.24864)).
- On an already-saturated ProRL-1.5B-v2 (more than 20k H100-h of RL), adaptive environments gave +3.37 in about 1,100 H100-h; continuing the original RL gave +0.49 with 3,600 H100-h ([RLVE](https://arxiv.org/abs/2511.07317)). **Moderate.**
- New procedures can need long, near-zero-reward phases: DELTA-Code stayed below 1% full-pass for about 450 steps after a partial-credit warm-up, then reached near 100% ([DELTA-Code](https://arxiv.org/abs/2509.21016)). Do not kill hard-tier runs on a pooled curve. **Emerging.**
- Compute choices mostly change efficiency, not the ceiling: loss aggregation, normalization and curriculum mainly changed ScaleRL's efficiency B, not its asymptote A ([ScaleRL](https://arxiv.org/abs/2510.13786)); performance was governed mainly by total optimization steps, with reuse up to 25× costing nothing significant ([Tan et al.](https://arxiv.org/abs/2509.25300)).
- Difficulty and budget interact: training on hard problems at an 8k-token budget suppressed the long traces needed for exploration ([e3](https://arxiv.org/abs/2506.09026)). Raise the budget with the tier.

### 5.5 Curricula: coverage and diversity beat ordering, except for sparse tiers

- No robust gain from easy→hard ordering over random mixing under fixed compute, for SFT or RL (e.g. OOD 0.27 vs 0.28) ([Mordig et al.](https://arxiv.org/abs/2603.27226)). Mixed-difficulty RLVR self-organizes into an implicit curriculum if the difficulty spectrum is smooth; gaps cause plateaus ([Huang et al.](https://arxiv.org/abs/2602.14872)).
- Of 13 data policies tested with 12 matched seeds, none of 8 selection or reweighting methods beat uniform sampling and none of 3 adaptive mixtures beat a fixed equal mix; signs flipped across model scale ([DataFlex-RL](https://arxiv.org/abs/2609.06107)).
- Staging matters when the hardest tier would get zero reward: a uniform mix gave no long-horizon gains in h1; in ScaleLogic the curriculum gave γ = 1.33 vs 2.36 for difficult-only; Nemotron 3 Nano's Gaussian curriculum kept multi-domain learning stable where random sampling biased the model toward easy tasks ([Nemotron 3 Nano](https://arxiv.org/abs/2512.20848)).
- Diversity must be engineered: Evol-Instruct-style rewriting pushed pass@8 *below* the untuned backbone (54.2 vs 55.1), while skill crossover plus mutation plus a learnability filter raised pass@1 by 4 and pass@8 by 7.4 ([EvoTD](https://arxiv.org/abs/2605.11666)).

**Implication.** Spend effort on the band, on filling gaps in the difficulty spectrum and on family breadth; stage only tiers that would otherwise be silent. **Strong.**

### 5.6 Confounds that fake "harder data works"

| Confound | Evidence | Control |
|---|---|---|
| Spurious rewards | Random rewards gave +21.4 MATH-500 on Qwen2.5-Math-7B vs +29.1 with ground truth; failed on Llama3 and OLMo2 ([Shao et al.](https://arxiv.org/abs/2506.10947)). Training on the same 32 prompts every step matched a 13k-prompt baseline on Qwen2.5-Math-1.5B ([Prompt Replay](https://arxiv.org/abs/2603.21177)) | Random-reward arm; a second model family |
| Contamination | §2.4 | Fresh procedural control; post-cutoff evals |
| Format correction | Part of 1-shot RLVR's gain is format correction ([1-shot RLVR](https://arxiv.org/abs/2504.20571)) | Format-only reward arm |
| Eval noise | Seed SD 5–15 points; 30 seeds optimal for small sets ([Sober Look](https://arxiv.org/abs/2504.07086)); clustered SEs can be 3× naive ([Miller](https://arxiv.org/abs/2411.00640)) | Paired per-item tests; SEs clustered by seed family |
| Verifier hacking that grows with difficulty | 40 vs 458 shortcuts in low vs high complexity levels ([IPT](https://arxiv.org/abs/2604.15149)) | Invariance-checked verification |
| Label decay on self-generated hard items | R-Zero pseudo-label accuracy 79% → 69% → 63% as questions got harder ([R-Zero](https://arxiv.org/abs/2508.05004)); unverified majority wrong on 73.33% of AIME 2024 ([T³RL](https://arxiv.org/abs/2603.02203)) | Gold audit slice bucketed by difficulty |

---

## 6. How to measure difficulty

| Method | Cost per item | Tied to your policy? | Best use |
|---|---|---|---|
| Empirical pass rate, k rollouts (§6.1) | k full rollouts | Yes, but stale after each checkpoint | Ground truth for bands; final admission |
| Bayesian / Kalman trackers over training history (§6.1) | ≈ 0 extra | Yes | Online selection of existing items |
| IRT over a model zoo (§6.2) | Rollouts from many models | No (population) | Label audits; generator rewards; eval design |
| Learned probes and value models (§6.3) | One forward pass | Partly (needs periodic re-anchoring) | Pre-filtering generated candidates |
| Rollout-free text features and LLM ratings (§6.4) | Cheap | No | Triage only |
| Trajectory signatures and solver effort (§6.5) | Logged anyway | Yes | Separating real difficulty from shortcuts, and early alarms |

### 6.1 Empirical pass rates: how many samples, and what p̂ = 0 means

- **Estimate pass@k without bias.** Draw n ≥ k samples, count c correct, and use pass@k = E[1 − C(n−c, k)/C(n, k)]; the naive 1 − (1 − p̂)ᵏ is biased ([Chen et al., 2021](https://arxiv.org/abs/2107.03374), who used n = 200 for k ≤ 100).
- **Small k is noisy.** At p = 0.5 the 95% half-width of p̂ is about ±0.35 with 8 rollouts, ±0.25 with 16 and ±0.17 with 32 (derived, normal approximation). A "2–5 of 8" band is therefore a coarse filter, not a measurement. Deep Dive found M = 8 noisy and used 32 ([Deep Dive](https://arxiv.org/abs/2603.24202)).
- **p̂ = 0 from small k means "unknown".** The pass@6 = 0 stratum was 5.1–8.3% of GSM8K and 28.7–43.5% of MATH across four models. At matched compute, a deterministic regime (greedy decoding plus five residual-stream activation perturbations) recovered 10–29% of it, versus at most 6% for greedy alone on the math cells. The authors read this as the stratum being "structurally identifiable", not as proof that ordinary sampling reaches it ([Zhou et al.](https://arxiv.org/abs/2606.19636)). In Knapsack RL, 577 prompts still labelled "extremely hard" after 1,000 iterations had produced at least one positive trajectory during training ([Knapsack RL](https://arxiv.org/abs/2509.25849), via note 18). Defer and re-pilot rather than delete: Pilot-Commit defers p̂ < 0.125 and evicts permanently only when every pilot rollout is correct, yet about 19% of prompts fully solved in epoch 2 were not solved in epoch 3 ([Pilot-Commit](https://arxiv.org/abs/2605.26606)).
- **Track rather than re-measure.** Beta-posterior bandits ([MoPPS](https://arxiv.org/abs/2507.04632)), Kalman filters over logit success (83% fewer rollouts than dynamic sampling, [KGPS](https://arxiv.org/abs/2607.27610)) and HMM models of solving progress ([DPS](https://arxiv.org/abs/2603.10887)) reuse training rollouts. For evaluation, Bayes@N reached Kendall τ > 0.90 against the gold ranking by N = 10 samples ([Don't Pass@k](https://arxiv.org/abs/2510.04265)).

```python
from math import comb
def pass_at_k(n, c, k):                     # unbiased (Chen et al., 2021)
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)

def band_decision(c, n, lo=0.125, hi=0.75, a0=1, b0=1):
    a, b = a0 + c, b0 + n - c               # Beta posterior; a0,b0 = operator/parent prior
    p_mean = a / (a + b)
    if c == 0:   return "defer_or_audit"    # unknown, not 'too hard' (§6.1)
    if c == n:   return "retire_to_review"  # keep 1-2% review queue (§3.2)
    return "train" if lo <= p_mean <= hi else ("complexify" if p_mean > hi else "scaffold")
```

### 6.2 Item response theory

A 1PL (Rasch) model fits a difficulty β per item from many models' responses. It is good for three jobs: flagging mislabels (an IRT indicator reached 95% precision in its top 200 flags across seven benchmarks, [Land & Bikel](https://arxiv.org/abs/2605.30504)); rewarding a rewriter toward harder items (RIDE's perturbations cost 26 models 21.73% on average, [RIDE](https://arxiv.org/abs/2511.04120)); and fitting per-instance hint length to target accuracy ([SEELE](https://arxiv.org/abs/2509.06923)). Limits: it measures difficulty for a population, not your policy; it needs models that can solve the items; and AUC is a misleading score for it, since a constant predictor already reached 0.715 vs 0.937 for the fitted model ([Krsteski & Meyer](https://arxiv.org/abs/2608.05797)). **Moderate.**

### 6.3 Learned and probe predictors

- **Activation probes.** PROPEL predicts whether a generated task lands in the solver's band from a frozen model's activations. Balanced accuracy was only 0.594–0.660, yet it doubled the frontier share of generated code tasks (10.1% → 20.0% for a 3B solver); RL gain tracked the probe's reward variance under the base policy better than its accuracy ([PROPEL](https://arxiv.org/abs/2606.18284)). Select probes by the reward variance they induce. **Emerging.**
- **Value models and skip rules.** PCL found intermediate-difficulty prompts 12.1× (MATH) and 16.9× (DeepScaleR) faster than rollout filtering ([PCL](https://arxiv.org/abs/2510.01135)); GRESO skips prompts that were uninformative last epoch, for up to 2.4× rollout speed-up ([GRESO](https://arxiv.org/abs/2506.02177)); DOTS cut RL time by 23–62% ([DOTS](https://arxiv.org/abs/2506.05316)); GPS gave a 1.4–2.0× step speed-up ([GPS](https://arxiv.org/abs/2602.01970)).
- **Disagreement.** Disagreement across differently trained solvers flags items that look easy only because they match one model's biases, which a single confident solver would miss (+1.34 points as a challenger reward, [Multi-Solver Disagreement](https://arxiv.org/abs/2608.30035)); a three-forward-pass disagreement probe gave a 3–5× lift at ranking recoverable p̂ = 0 items ([Zhou et al.](https://arxiv.org/abs/2606.19636)).
- **Operator priors (Proposal, from note 18).** Give each new variant a prior from its parent seed's p̂ shifted by the operator's measured mean solve-rate drop (e.g. −70 to −100 points for information removal, [Trading Human Curation](https://arxiv.org/abs/2606.03800)), then update online. This removes most cold-start profiling.

### 6.4 Rollout-free predictors

- **Text features.** On 5,230 agentic tasks, a regressor on token-entropy trajectories, embeddings and length reached Spearman ρ = 0.399 in-distribution but 0.225 on unseen benchmarks; length alone gave 0.086 / 0.101. Large residuals flagged contaminated or infeasible tasks ([Krsteski & Meyer](https://arxiv.org/abs/2608.05797)). Use for triage and audit routing, not admission. **Moderate.**
- **LLM ratings and teacher length.** Good enough to rank SFT questions: LLM difficulty (code) and response length (math, science) beat embedding and fastText filters, and beat random selection by 6% (code) and 4% (math) ([OpenThoughts](https://arxiv.org/abs/2506.04178)). Poor as a quality judge: o3–human correlation on problem quality was 0.07, while measured difficulty gain correlated up to 0.60 with human quality ([AutoCode](https://arxiv.org/abs/2510.12803)). A human-label-trained SWE difficulty rater reached 75.3% accuracy, but SFT on its difficulty bins showed no trend (12.4 / 10.8 / 13.6 / 12.2%) ([SWE-smith](https://arxiv.org/abs/2504.21798)).
- **Structural parameters.** Op count, depth, constraint count and topology predict pass rate smoothly within a family (§3.3; TRACE reports r ≈ −0.96 between its difficulty metric and accuracy, [TRACE](https://arxiv.org/abs/2607.04784)). Calibrate the curve once per family on your policy, then pick parameters that land in the band.

### 6.5 Trajectory signatures and solver effort

- **Effort and shortcut signatures.** Log solving cost (tool calls, tokens, turns), the step at which the answer first appears, and a no-evidence answer probe; FORT's datasets ranged from Ω 20.6 with the answer at step 5.7 to Ω 141.0 at step 46.9 ([FORT](https://arxiv.org/abs/2606.12087)). CLI-Universe's evidence-guided refinement took 3.45× more solver turns and lowered pass rate by 13.3 points ([CLI-Universe](https://arxiv.org/abs/2606.22883)). Effort is not monotone in difficulty: reasoning effort rose and then fell near collapse ([Illusion of Thinking](https://arxiv.org/abs/2506.06941)).
- **Strong/weak solver pairs.** Only 19% of validated terminal tasks separated a strong from a weak solver on first probe; solver-guided revision raised this to 96%. The diagnostic rules are reusable: all pass → shortcut or too easy; all fail → check solvability; weak passes but strong fails → leakage or a misleading statement ([CalibForge](https://arxiv.org/abs/2608.06352)).
- **Worst case over variants.** Worst-case accuracy over programmatic variants was at most about 50% of average-case for all 14 VLMs tested ([DynaMath](https://arxiv.org/abs/2411.00836)); average accuracy overstates mastery.
- **Online alarms.** Mean response length collapsing (510.7 → 45.7 tokens) preceded accuracy loss from one shortcut-rewarded sample ([Cheng et al.](https://arxiv.org/abs/2605.28388)); a jump in the share of maximum-length responses accompanied long-horizon RL collapse ([Kim et al.](https://arxiv.org/abs/2605.02572)); faster entropy collapse correlated with worse test performance ([Skywork-OR1](https://arxiv.org/abs/2505.22312)). Monitor these per family, especially right after adding a new generator. **Moderate.**

---

## 7. Symptom → remedy decision table

Run the checks in order; most "too easy" pools fail an early check and never need a generator.

```
pool looks saturated
  │
  ├─ verifier sound? (TPR, TNR ≥ 0.9 on known-good / known-bad) ──no──► fix verifier, re-profile   [Ch. 4]
  ├─ items clean? (no completion leak, gains hold on fresh/renamed items) ──no──► decontaminate  [Ch. 4, 9]
  ├─ no shortcuts? (no-CoT, tool-free, partial-input at chance) ──no──► reformat / filter        [Ch. 2, 4]
  ├─ harness OK? (few turn/length-cap hits) ──no──► fix budget or answer format                 [Ch. 5]
  ├─ behaviours present? (verify / backtrack in rollouts) ──no──► prime or mid-train            [Ch. 6]
  ├─ atoms reliable? (per-step success near 1) ──no──► sharpen atoms first                      [Ch. 2]
  └─ yes to all ──► complexify (compose, add families) inside a calibrated band                  [Ch. 2, 3, 5]
```

| Symptom | Likely cause | Diagnostic | Remedy | Where |
|---|---|---|---|---|
| Most GRPO groups all-correct; effective-prompt ratio falling | Pool saturated for this policy | §2.1 histogram per family | Reallocate rollouts away from p̂ = 1; failure-prefix conditioning and meta-tasks on saturated items; then compose and add generator families with a difficulty ratchet | Ch. 5, Ch. 2, Ch. 3 |
| High train pass rate, but strengthened tests or near-miss candidates also pass | Lenient verifier (false positives) | Known-bad candidates; TPR/TNR audit | Harden tests (hacking inputs, near-miss programs, mutants); invariance checks; hidden or read-only tests | Ch. 4 |
| Valid hard items stuck at p = 0; stronger model's answers also rejected | Verifier false negatives | Equivalent-rewrite unit tests; rule-negative re-check | Rule-first cascade with a discriminative second stage; canonical answer formats | Ch. 4 |
| Gains vanish on fresh, renamed or post-cutoff items, or on a second model family | Contamination, memorization or spurious reward | Partial-prompt completion; random-reward arm; RandomCalculation-style control | Decontaminate outputs; procedural held-out families; report non-Qwen results | Ch. 4, Ch. 9 |
| Correct answers without valid reasoning; no-CoT guesses succeed | Format leakage, small answer space | No-CoT guess within 8; question-only / choices-only baselines | Convert MCQ to open-ended with canonical answers; code-RNG answer marginals; drop tool-free-solvable items | Ch. 2, Ch. 4 |
| Bimodal histogram (mass at 0 and 1) | Generator steps too coarse, or a broken sub-family | Per-operator histograms; inspect the 0-spike | Add intermediate levels; calibrate the ladder on your policy; answer-preserving format ladders for 0-pass items | Ch. 2, Ch. 3 |
| Long 0-tail after hardening | Over-hardening, invalid items or verifier FN | Solvability certificate (teacher solve, large-k, construction); MathQ-Verify-style well-posedness check | Repair or drop invalid items; scaffold valid ones (hints withdrawn over training, teacher-in-prompt, expert prefixes); more rollouts for rare successes | Ch. 4, Ch. 5 |
| pass@1 rising, pass@k flat or falling; entropy collapsing | Training on saturated or in-distribution data (sharpening) | Paired pass@k bands vs base; Cover@τ | Move data to the edge (low pass@1, pass@k > 0); composed tasks; answer-preserving variants of weak items; boundary-preserving updates | Ch. 5, Ch. 2 |
| RL stalls on hard variants; rollouts never verify or backtrack | Missing behaviours | Behaviour-frequency count on hard-tail rollouts | Priming SFT on behaviour-rich traces (wrong answers acceptable); mid-training; then RL with the unchanged verifier | Ch. 6 |
| Composites near 0 although each atom looks "solved" | Atom reliability too low for the product law | Per-step accuracy; ρ between composite and joint atom success | Sharpen atoms (cheap 1-hop RL or SFT), then compose at depth 2 with a staged horizon | Ch. 2, Ch. 5 |
| SFT on hard teacher traces lowers a small student's accuracy | Traces too complex to imitate | Compare with partial-prefix training | Expert-prefix anchors or in-prompt candidates instead of SFT on full traces | Ch. 6, Ch. 5 |
| Mean length collapses or max-length share spikes after adding a family | Shortcut reward in a few items; horizon too long | Per-family length and cap-hit monitors | Quarantine the family; audit rewards; horizon reduction (macro-actions, subgoals) and short-to-long curriculum | Ch. 9, Ch. 5 |
| Generated "hard" items look alike; pass@k below the backbone | Diversity collapse from generic "make it harder" rewriting | Seeds × rewrites ratio; solution-signature similarity | Skill crossover and parametric mutation with a learnability filter; many seeds, few rewrites each; persistent archives | Ch. 2, Ch. 3 |
| Many failures hit turn or token caps | Budget, not capability | Cap-hit share per tier | Raise budget with difficulty tier; compressed answer formats (program, final state) | Ch. 5 |
| Self-labelled hard items; audited label accuracy falling | Pseudo-labels decay with difficulty | Gold audit slice bucketed by difficulty | Stop raising difficulty; switch to construction-based or cross-family verification | Ch. 4 |
| Pilot shows "no difference" between data arms | Underpowered comparison | Power calculation; seed SD | Paired per-item tests, 10–30 samples per item, 8–12 matched seeds before deciding | Ch. 9 |

**Bottom line (Proposal, assembled from the Strong-rated findings above).** Complexify only after the pool passes the verifier, contamination, shortcut, harness and behaviour checks. Then compose from reliable atoms, keep training inside a policy-relative band centred near 0.3–0.6, treat p̂ = 0 as "unknown until certified", re-profile the whole pool as the policy moves, and judge success by paired large-k pass@k on fresh items across at least two model families.
