# RL playbook: feeding a policy hard synthetic tasks it can learn from

> **Key takeaways**
>
> - **Only mixed-outcome groups train, so manage the pool around a policy-relative pass-rate band.** Keep roughly 0.2–0.8 and aim the centre at about 0.3–0.5. Measure it with the model being trained and re-measure every stage. Once the band empties, stop filtering and start generating. **Strong.**
> - **Build more generator families rather than tuning the curriculum.** Give each family its own online difficulty controller and mix families equally. Across controlled studies, adding environments keeps paying off. Clever orderings and adaptive mixtures rarely beat an equal mix, except when the hardest tier would otherwise get zero reward. **Strong (breadth); Moderate (ordering null).**
> - **Proposer–solver loops that stay stable share five things:** a validity gate independent of the solver, external grounding, a persistent diversity archive keyed on solution signatures, a grader the loop never trains, and stabilizers (golden replay, real anchors, KL-anchored generator updates). The shape of the proposer's reward matters much less than these. **Strong** (rules 1–4); **Moderate** (stabilizers).
> - **For items at p ≈ 0, audit first, then add the most on-policy help that works.** The order is: answer-preserving reformulation, then a teacher answer in the prompt, then annealed solution prefixes, then an off-policy trace in the group. Install missing behaviours (verify, backtrack) by priming or mid-training before adding scaffolds. **Moderate.**
> - **Composite rewards need three layers:** a validity gate, the outcome reward, and structure-aware partial credit only where success is sparse. Every shaped term opens a hacking surface. Test it with invariance checks, trivial-agent baselines and a held-out judge panel. **Moderate.**
> - **Spend compute in this order:** audit the verifier, reallocate rollouts, recycle zero-variance items, and only then generate. Rollouts are over 90% of runtime, and in a derived worked example profiling used about a third of rollout tokens. Anchor hard synthetic data with 20–33% real items and 2–10% review or easy items. **Moderate/Proposal.**

**Contents**

1. [The signal budget: why the pass-rate band matters](#1-the-signal-budget-why-the-pass-rate-band-matters)
2. [Measuring pass rates cheaply](#2-measuring-pass-rates-cheaply)
3. [Managing the pool: filters, dynamic sampling, replay, retirement, refresh](#3-managing-the-pool-filters-dynamic-sampling-replay-retirement-refresh)
4. [Controllers and curricula: breadth beats ordering](#4-controllers-and-curricula-breadth-beats-ordering)
5. [Proposer–solver self-play and trained generators](#5-proposersolver-self-play-and-trained-generators)
6. [Making too-hard items learnable](#6-making-too-hard-items-learnable)
7. [Reward design for composite tasks](#7-reward-design-for-composite-tasks)
8. [Mixing synthetic-hard with real tasks, and dose](#8-mixing-synthetic-hard-with-real-tasks-and-dose)
9. [Monitoring: the run dashboard](#9-monitoring-the-run-dashboard)
10. [Compute and cost allocation](#10-compute-and-cost-allocation)
11. [Reference table: reported bands and hyperparameters](#11-reference-table-reported-bands-and-hyperparameters)
12. [Step-by-step recipe](#12-step-by-step-recipe)

Scope: RLVR with GRPO-family estimators and agentic RL on synthetic tasks harder than your seeds. Related chapters: why tasks look easy ([Ch. 01](01-diagnosis-difficulty-and-learning-signal.md)), hardening operators ([Ch. 02](02-complexification-operator-taxonomy.md)), generation pipelines ([Ch. 03](03-generation-architectures.md)), verifiers ([Ch. 04](04-verification-and-quality-control.md)), SFT and distillation ([Ch. 06](06-sft-playbook.md)).

---

## 1. The signal budget: why the pass-rate band matters

**The arithmetic.** Group-normalized estimators give zero advantage to a group whose N rollouts all receive the same reward. With per-item success rate p, the probability that a group carries a gradient is

```
P(non-zero gradient) = 1 − p^N − (1 − p)^N          (Knapsack RL)
```

At N = 8 this is about 0.99 at p = 0.5, 0.34 at p = 0.05 or 0.95, and 0.08 at p = 0.01 or 0.99 (derived). [Knapsack RL](https://arxiv.org/abs/2509.25849) works the tail case: a p = 0.01 item needs about 100 rollouts on average, and 229 for 90% confidence, before one success appears.

Three theory results point to the same middle band.

- [Bae et al.](https://arxiv.org/abs/2504.03380) lower-bound learnability by p(1−p)/(2β²), which peaks at p = ½.
- Knapsack's information-gain proxy p(1−p)² peaks at p = 1/3, which favours the harder side.
- [DARS](https://arxiv.org/abs/2508.13755) shows that GRPO's cumulative advantage is largest at medium accuracy. Hard items are therefore under-weighted even when they are in the batch.

**How much is wasted in practice.**

- Qwen2.5-Math-7B on DAPO-Math-17K with N = 8: late in training about 40% of prompts are all-correct and about 20% all-wrong, and the effective-gradient ratio (share of samples with a non-zero gradient) falls to about 20% by iteration 1,000 ([Knapsack RL](https://arxiv.org/abs/2509.25849)).
- Under uniform sampling about 39% of rollouts go to silent groups ([ThinkPrior](https://arxiv.org/abs/2609.09075)).
- In multi-turn agent RL, zero-variance groups are typically about 40% of groups ([Selective Rollout](https://arxiv.org/abs/2605.05802)).
- A static, low-cap difficulty range drives the effective-prompt ratio (share of groups with non-identical rewards) to zero ([RLVE](https://arxiv.org/abs/2511.07317)).

**Evidence that the edge beats both easy and too-hard data** (**Strong**: several independent controlled studies):

| Study | Setting | Result |
|---|---|---|
| [Interplay of pre/mid/RL](https://arxiv.org/abs/2512.07783) | From-scratch 100M models, DAG math | RL on in-distribution difficulty never raised pass@128. Edge data (low pass@1, pass@k > 0) gave up to +42% pass@128. Reward stagnated when data was too easy or too hard. |
| [Sample difficulty (T-SAE)](https://arxiv.org/abs/2605.28388) | MATH, 3 model settings | Medium@8 was best in all three. Hard@8 (pass@8 = 0) *lowered* averages by 5.75, 11.24 and 1.07 points. |
| [STRETCH](https://arxiv.org/abs/2609.18642) | Generator target ablation | A 0.5 target beat 0.2 (too few successes) and 0.8 (already solved). |
| [UltraLogic](https://arxiv.org/abs/2601.03205) | Calibrated logic generators | Best training band was 40–60% success after subtracting about 0.1 for formatting. The best tier depended on size: Qwen3-8B learned most from Easy, 14B from Medium. |
| [EvoEnv](https://arxiv.org/abs/2605.14392) | Self-written environments | Targets 0.3, because environments near 50% saturate quickly as the solver improves. |
| [Hard Examples](https://arxiv.org/abs/2508.14094) | Fixed annotation budget | Hardest 10% by base failure: gains up to 47% vs 3–15% for easy subsets; hard items keep mixed outcomes longer. |

**Triage by symptom** (details in §3–§6):

| Profile of the pool | Diagnosis | First moves |
|---|---|---|
| Mass at p ≈ 1 | Saturated seeds | Reallocate rollouts away from these items. Turn them into failure-prefix and meta-tasks (§6.4). Complexify with operators from [Ch. 02](02-complexification-operator-taxonomy.md). |
| Mass at p ≈ 0 | Too hard, or broken | Audit labels and verifiers. Check whether the needed behaviours are present. Scaffold (§6). Reduce horizon for agents. |
| Bimodal (0 and 1) | Generator step size too coarse | Add intermediate levels or bridge tasks. Graduate items at about 0.5. Route rollouts to the middle with predictors. |
| Wide middle band | Healthy | Keep the controller running, and add families before the band drifts. |

---

## 2. Measuring pass rates cheaply

**Who profiles.** Measure the band with the policy you are training. Use stronger models only to certify that an item is solvable.

- [POLARIS](https://hkunlp.github.io/blog/2025/Polaris/) uses "the specific model being trained".
- [Olmo 3](https://arxiv.org/abs/2512.13961) uses the initial checkpoint of each stage.
- [INTELLECT-3](https://arxiv.org/abs/2512.16144) annotates with a 4B proxy. That is cheaper but less faithful.
- Teacher-relative filters answer a different question: is the item solvable at all?
  - [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) drops prompts GPT-OSS-120B solves 8/8.
  - [GLM-5](https://arxiv.org/abs/2602.15763) keeps items the previous model rarely solves but stronger teachers can.
  - [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) keeps pass@100 > 0.

**How many samples.** Eight rollouts is the common default ([LILO](https://arxiv.org/abs/2502.12272), Olmo 3, [Qwen2.5-Math](https://arxiv.org/abs/2409.12122)). [Magistral](https://arxiv.org/abs/2506.10910) and [MiMo](https://arxiv.org/abs/2505.07608) use 16, [Kimi k1.5](https://arxiv.org/abs/2501.12599) uses 10.

A cost-effective reported design is [Pilot-Commit](https://arxiv.org/abs/2605.26606) (rollout-efficiency multipliers across papers are not directly comparable; see §10):

- Draw 3× the training batch as candidates.
- Run 16 pilot rollouts on each.
- Skip items with p̂ > 0.75. Defer items with p̂ < 0.125 to a later epoch.
- Commit 48 more rollouts to the rest, and train on pilot and commit rollouts together.

It reached target accuracy with up to 1.9× fewer cumulative rollouts than GRPO and 4.0× fewer than DAPO. **Moderate.**

**Small-k labels are noisy. Never delete an item on one pilot.**

- The pass@6 = 0 stratum is 28.7–43.5% of MATH across four models. Perturbed deterministic decoding reaches 10.3–22.9% of it on the math cells ([Hard or Just Unreached?](https://arxiv.org/abs/2606.19636)).
- In Knapsack RL, 577 prompts still labelled "extremely hard" after 1,000 iterations had produced at least one positive trajectory.
- Only 81.2% of prompts fully solved in epoch 2 were still solved in epoch 3 (Pilot-Commit).

Treat p̂ = 0 at small k as *unknown*. Defer the item and re-pilot it. **Strong** (three independent observations).

**Cheaper than rollouts: priors and predictors.**

| Tool | What it does | Reported effect |
|---|---|---|
| Parent-seed prior + operator offset | Start a variant at its parent's p̂ minus the operator's measured drop. Information removal drops solve rate by 70–100 points and structural coordination by about 60 ([Trading Human Curation](https://arxiv.org/abs/2606.03800)). | Removes most cold-start profiling (**Proposal**; untested as a whole) |
| [BOTS](https://arxiv.org/abs/2510.26374), [VADE](https://arxiv.org/abs/2511.18902), [graph estimator](https://arxiv.org/abs/2608.17941) | Bayesian posteriors shared across similar tasks; Thompson sampling | Better data efficiency with negligible extra rollouts |
| [GRESO](https://arxiv.org/abs/2506.02177) | Skip prompts that were uninformative last epoch | Up to 2.4× rollout and 2.0× total speedup |
| [PCL](https://arxiv.org/abs/2510.01135) | Value model predicts intermediate-difficulty prompts | 12.1× (MATH) and 16.9× (DeepScaleR) faster than rollout filtering at finding them |
| [PROPEL](https://arxiv.org/abs/2606.18284) | Activation probe predicts solver pass rate for generated tasks | About 2× the frontier-task rate with under half the solver trials. The probe drifts, so re-anchor it. |
| [Rollout-free agentic predictor](https://arxiv.org/abs/2608.05797) | Entropy features plus embeddings predict IRT difficulty | Spearman ρ = 0.399 in-distribution, 0.225 on unseen benchmarks. A constant predictor already scores AUC 0.715, so evaluate predictors by rank correlation. |

Use predictors to triage candidates and to schedule re-profiling, not as the final admission filter. **Moderate.**

**Budget the profiler.** In a worked example (7B or 32B policy, k = 8, 8K tokens, in-band yield 0.3, then 3 training epochs at G = 16), profiling was about 36% of all rollout tokens. This figure is derived arithmetic from the cost-model research notes, not a measured result; see §10. Profiling rollouts are stateless and tolerate latency, so run them on preemptible capacity ([RLBoost](https://arxiv.org/abs/2510.19225)). For agents, stop groups whose action prefixes have converged ([Selective Rollout](https://arxiv.org/abs/2605.05802): 10.7% wall-clock saved).

```python
def profile(item, policy, k_pilot=16, k_commit=48, hi=0.75, lo=0.125):
    p_prior = item.parent_p - OPERATOR_DROP[item.operator] if item.parent else None
    if p_prior is not None and not (0.05 < p_prior < 0.95) and item.age < MAX_DEFER:
        return "defer"                    # skip rollouts; re-check later
    r = policy.rollout(item, n=k_pilot)
    p = mean(r)
    if p > hi:  return "too_easy"         # → complexify (Ch. 02); evict only after 2 consecutive all-correct pilots
    if p < lo:  return "defer"            # → re-pilot next epoch, scaffold (§6), or disagreement probe
    return ("commit", r + policy.rollout(item, n=k_commit))
```

**Re-profile on a schedule.** [Magistral](https://arxiv.org/abs/2506.10910) re-grades the *whole original* pool with its RL model, not only the survivors; [rStar2-Agent](https://arxiv.org/abs/2508.20722) re-filters 42K problems to 17.3K harder ones before stage 3; [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) re-profiles at plateaus; [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) rescans the pool in the background with intermediate checkpoints; [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) alternated training and re-filtering. Treat the band as a schedule, not a constant (see [Ch. 08](08-frontier-lab-practices.md)). **Strong.**

---

## 3. Managing the pool: filters, dynamic sampling, replay, retirement, refresh

Treat the task pool as a state machine. Paid-for items move between states and are almost never deleted.

```
 generator ──► CANDIDATE ──pilot──► ACTIVE (0 < p < 1) ──hist. p ≥ 0.9──► RETIRED
                  │  │                 │   ▲                              │
     p̂ > 0.75 ◄──┘  └──► p̂ < 0.125      │   │ replay mid-band              │ 1–2% review queue
  (→ complexify)       (→ DEFERRED:      │   │ (p ∈ [0.25,0.75])            ▼
                        re-pilot,        ▼   │                         REVIEW ──regressed──► ACTIVE
                        scaffold §6)   zero-variance group:
                                       drop from loss, refill in-domain,
                                       keep item RECYCLABLE (weighted)
```

### 3.1 Offline filters (before a stage)

- Remove p ∈ {0, 1} against the base model ([Skywork-OR1](https://arxiv.org/abs/2505.22312)), then apply a band cut. Reported cutoffs are in §11.
- Before any hardening, remove items that can be solved without reasoning:
  - [Kimi k1.5](https://arxiv.org/abs/2501.12599) drops a prompt if the model guesses it without chain-of-thought within 8 tries.
  - [Qwen3](https://arxiv.org/abs/2505.09388) removes queries solvable without CoT.
  - [GLM-5](https://arxiv.org/abs/2602.15763) drops search questions that a tool-free model answers in at least 1 of 8 attempts.

  Otherwise hardening adds guessability rather than reasoning. **Strong.**

### 3.2 Online filtering and dynamic sampling (every step)

- **Drop zero-variance groups from the loss.** [ScaleRL](https://arxiv.org/abs/2510.13786) shows this raises the fitted asymptote.
- **Refill the batch**: oversample up front ([DAPO](https://arxiv.org/abs/2503.14476); 24,576 rollouts per step vs 8,192 for GRPO in Pilot-Commit's table), refill actively until enough non-zero-gradient groups exist ([Olmo 3](https://arxiv.org/abs/2512.13961)), filter-and-replace ([AReaL-SEA](https://arxiv.org/abs/2601.22607): disabling it cut Airline pass^1 from 70.5 to 65.0), or duplicate non-zero-variance samples ([WebSailor](https://arxiv.org/abs/2507.02592) DUPO: about 2–3× faster than DAPO's dynamic sampling).
- **Log the refill cost.** In verl, `filter_groups` raises an error after `max_num_gen_batches` (default 10) ([verl DAPO doc](https://github.com/volcengine/verl/blob/main/docs/algo/dapo.md)). The number of generation batches per step is the inverse of your live in-band yield. Use it to trigger the generator service before the run crashes.

### 3.3 Reallocate rollouts before buying new tasks

Uniform G wastes rollouts on both tails. Four ways to reallocate:

- **[Knapsack RL](https://arxiv.org/abs/2509.25849).** Solves a knapsack per epoch: N_low = 2 rollouts for saturated items, and up to 93 for the hardest.
  - The effective-gradient ratio rises 20–40%, average gains are 2–4 points, and matching it with uniform allocation would take about 2× compute.
- **[Reinforce-Ada](https://arxiv.org/abs/2510.04996).** Samples sequentially until an exit condition, e.g. K_pos = 16 positives or N_max = 128. Up to 2× faster convergence at equal budget.
- **[VIGOR](https://arxiv.org/abs/2607.22002).** Expands the highest-variance groups. Up to 2.3× fewer rollouts.
- **[IsoCompute](https://arxiv.org/abs/2603.12151).** The optimal number of rollouts per problem n rises with compute and then saturates:
  - about 512 on an easy split;
  - a moderate n = 64 on a pass@128 = 0 subset;
  - with only 500 problems, the frontier saturates at n = 256, and n = 512 overfits.

  Generate new tasks when n has saturated or the pool starts to overfit. **Moderate.**

A false positive is amplified when it is the only success in a large group. Spot-check rare successes on low-p items with a second verifier (see [Ch. 04](04-verification-and-quality-control.md)).

### 3.4 Replay and review

- **Mid-band prompt replay ([Prompt Replay](https://arxiv.org/abs/2603.21177)).** Buffer prompts with p ∈ [0.25, 0.75], prioritized by closeness to 0.5; replay up to ε = 0.75 of each batch with a 10-step cooldown and ≤ 15 reuses. It speeds early learning, then converges with the baseline, and the benefit disappears once rollouts are not the bottleneck: an efficiency lever, not a ceiling lever. **Moderate.**
- **Review mastered items ([ReMind](https://arxiv.org/abs/2606.03087)).** Solved prompts enter a queue with p_add = 0.25; every 5 steps, 10% of the batch is replaced with review prompts (about 2% of all prompts). About 21% of 1,275 review records showed partial regression. ReMind lifted GRPO from 60.25 to 64.06 on Qwen3-VL-8B (average over eight image-text benchmarks), and even a 1% review budget avoided GRPO's mid-training plateau; 4% or 10% gave no further gain. **Emerging.**
- **Easy pool.** [MiMo](https://arxiv.org/abs/2505.07608) samples a perfect-pass pool 10% of the time.
- **Moderately easy items as a length regularizer.** Keeping them cut Qwen3-4B-Thinking's mean AIME25 length by about 56%, at a cost of 73.33 → 70.00 pass@1 ([Frugal](https://arxiv.org/abs/2511.01937)).

### 3.5 Retirement and recycling

- **Retire at high pass rate, with a way back.**
  - [ScaleRL](https://arxiv.org/abs/2510.13786) "No-Positive-Resampling" permanently retires prompts with historical pass rate ≥ 0.9. This raises the asymptote.
  - [POLARIS](https://hkunlp.github.io/blog/2025/Polaris/) removes items above 0.9 after each phase.
  - [RODS](https://arxiv.org/abs/2606.19047) retires above 0.95 (mastered) and below 0.20 (unsolvable).
  - Pair retirement with a review queue, since retired skills decay.
- **Recycle zero-variance items instead of deleting them.**
  - In agentic search, about 20% of unique queries flipped from zero-variance to signal-bearing.
  - By the end of training, about three-quarters of accepted groups came from recycled queries.
  - Recycling beat epoch-based replay at matched rollouts ([Coelho et al.](https://arxiv.org/abs/2606.10709)).
  - SkyRL's fully-async mode marks dropped groups as consumed, which forfeits this within an epoch ([SkyRL docs](https://docs.skyrl.ai/docs/tutorials/fully_async)). **Moderate.**

### 3.6 Refresh

When the in-band fraction falls, you have three options:

1. Re-profile the *full original* pool with the current checkpoint.
2. Swap in newly moderate items from a backup pool ([Tongyi](https://arxiv.org/abs/2510.24701)).
3. Call the generator.

[LoongRL](https://arxiv.org/abs/2510.19363)'s last stage retrains only on the 30–40% of items not solved in all 8 rollouts. None of the major open frameworks re-levels or regenerates tasks on its own. Every one filters zero-variance groups, but the generator service, the difficulty-prior store and the per-domain quotas are yours to build ([Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape)).

### 3.7 Asynchrony pitfalls

- **Filtered groups are older than average**, because the easy prompts they solve were issued earlier ([Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape)). Tag tasks and tokens with the policy version *and* the profiling version, and bound staleness per sample. Replace dropped groups within the same domain, since refilling from any source drifts toward fast domains (an inference, not measured).
- **Keep S·η roughly constant**: the largest stable learning rate scales about as 1/S for sync interval S ([collapse laws](https://arxiv.org/abs/2607.01083)). If the pool changes mid-run, use stale-data-robust objectives ([M2PO](https://arxiv.org/abs/2510.01161) was stable with data ≥ 256 updates stale).

---

## 4. Controllers and curricula: breadth beats ordering

### 4.1 The controller menu

Every procedural or parameterized family needs its own controller, because each environment's "level" means something different ([RLVE](https://arxiv.org/abs/2511.07317)). From cheapest to most flexible:

| Controller | Rule | When to use | Source |
|---|---|---|---|
| Difficulty pools | Easy/normal/hard pools by observed solve rate. Drop all-pass and all-fail groups. A p = 1 prompt is never sampled again. | Day-1 baseline; no generator | [INTELLECT-3](https://arxiv.org/abs/2512.16144) |
| Gaussian pass-rate curriculum | Each batch's pass-rate distribution is a Gaussian whose mean moves from easy to hard linearly over training. Domain ratios fixed per batch. Re-profile at plateaus. | Multi-domain static pools | [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848), [Llama-Nemotron](https://arxiv.org/abs/2505.00949) |
| Pass@1 gate | Advance a level when validation pass@1 > ε (0.5 at 1.5B, 0.75 at 7B) | One scalar knob | [SATURN](https://arxiv.org/abs/2505.16368) |
| Sliding window | Promote the top level h when accuracy there is ≥ τ_acc = 0.9 over ≥ τ_num = 8 × rollouts samples. Keep a window of d_Δ = 4 levels. No upper cap. | Integer knob per environment; many environments | [RLVE](https://arxiv.org/abs/2511.07317) |
| Proportional controller + retirement | d ← clip(d + β(acc − τ)). Active set of 64 environments. Retire an environment when its difficulty slope over 10 steps is ≤ 0, it has had 0 accuracy for 5 steps, or it has sat at max level for 5 steps. Retired environments return to the pool. | Many lifted environments | [SCALER](https://arxiv.org/abs/2601.04809) |
| Regret buffer + mutation | Problem regret ρ = 1[s ≥ 1] − s/n. Zipf sampling by priority plus a staleness bonus. Mutate one attribute of high-regret levels. | Interacting, non-monotone knobs | [Frontier Learning](https://arxiv.org/abs/2609.35426) |
| Directional rewrite | When acc > 0.8, ask an LLM for a harder rewrite. Accept it only if 0.2 < acc(q′) < acc(q). | Non-parametric seeds (GUI, agents, code) | [RLAnything](https://arxiv.org/abs/2602.02488) |
| Learned teacher over knobs | A teacher picks generator parameters. Reward min(p, 1−p) minus a signature-repetition penalty. | Several generators; budget to train a teacher | [LURE](https://arxiv.org/abs/2608.21871) |

Frontier Learning's ablation shows mutation is the key ingredient (Dice: PLR + exploration 33.5, PLR + mutation 62.5, full method 71.8). The adaptive window matters too. In RLVE, it beat an *oracle* static range covering all the levels the adaptive run reached. **Strong** (controllers beat static ranges); **Moderate** (choice among controllers).

### 4.2 Breadth: the strongest lever

| Evidence | Result |
|---|---|
| [RLVE](https://arxiv.org/abs/2511.07317) | ProRL-1.5B-v2 had already had more than 20k H100-h of RLVR. Joint training on 400 environments added +3.37 in about 1,100 H100-h. Continuing the original RL added +0.49 in 3,600 H100-h. Held-out accuracy rose over nested sets of 1, 4, 16 and 256 environments. |
| [ReSyn](https://arxiv.org/abs/2602.20117) | At about 16K instances, BBH was 75.19 for 400 envs × 40 instances, 69.85 for 100 × 160 and 71.20 for 25 × 640. |
| [SCALER](https://arxiv.org/abs/2601.04809) | Gains rose monotonically over 8, 64, 512 and 2,739 environments. |
| [InternBootcamp](https://arxiv.org/abs/2508.08636) | 512 > 128 > 32 > 8 tasks. Three tasks unlearnable in isolation became solvable after about 300 steps in the 512-task mix. |
| [UltraLogic](https://arxiv.org/abs/2601.03205) | Task-type diversity was the primary driver of gains. |
| [Forge](https://arxiv.org/abs/2605.08905) | 10 tasks beat 3 on out-of-domain average (53.6 vs 53.0) with 3.3× less data per task. |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | RL on synthetic general-agent environments improved Tau2Bench, MCP-Mark and MCP-Universe. RL on code and search alone did not. |

**Strong.** Spend engineering on more families, not more instances per family.

**Breadth is not indiscriminate.**

- [AES](https://arxiv.org/abs/2608.03571): 30 of 200 environments, selected for ability coverage, gave +95.6% relative gain; all 200 gave +43.4%.
- A 50-task Reasoning Gym subset selected by BBH-dev NLL degraded most other metrics ([Reasoning Core v3](https://arxiv.org/abs/2608.05148)).
- 1–3 buggy task types out of 50 collapsed UltraLogic training.
- [Banyan](https://arxiv.org/abs/2606.00880): too much diversity inhibits continued optimization on later distributions.

Select families by skill coverage and gate each one for bugs (see [Ch. 04](04-verification-and-quality-control.md)).

### 4.3 Ordering: usually second-order

Default to an equal mix across families plus per-family controllers. **Moderate** (null results on ordering replicate across several controlled studies, but mostly with small models or short budgets).

- No robust gain from easy→hard ordering over random mixing, for SFT or RL ([Mordig et al.](https://arxiv.org/abs/2603.27226); e.g. 0.27 standard vs 0.28 curriculum on LinearDepth, Llama3.2-1B GRPO).
- With 12 matched seeds, none of 8 selection or reweighting methods beat uniform sampling and none of 3 adaptive mixtures beat a fixed equal mix; method signs flipped across scale ([DataFlex-RL](https://arxiv.org/abs/2609.06107)).
- Curriculum choices mostly change efficiency, not the asymptote ([ScaleRL](https://arxiv.org/abs/2510.13786)); uniform mixing beat a staged 3-block curriculum ([Depth × Complexity](https://arxiv.org/abs/2605.26934)); curriculum benefit was negligible in [Logic-RL](https://arxiv.org/abs/2502.14768).
- Theory: mixed easy+hard data suffices ([Wang et al.](https://arxiv.org/abs/2505.23683)); mixed-difficulty RLVR self-organizes into a curriculum when the spectrum is *smooth*, and gaps cause plateaus ([Huang et al.](https://arxiv.org/abs/2602.14872)).

**When staging does matter: the top tier would otherwise get zero reward.**

- [h1](https://arxiv.org/abs/2510.07312): a uniform mix gave no long-horizon gains; a staged horizon curriculum lifted AIME24 from 5.10 to 10.52 (avg@32, 3B).
- [ScaleLogic](https://arxiv.org/abs/2605.06638): steps ∝ depth^γ with γ = 1.33 under a curriculum vs 2.36 for difficult-only data.
- [Horizon study](https://arxiv.org/abs/2605.02572): training directly at goal distance 10–12 barely improved; short-to-long gave marked gains. [e3](https://arxiv.org/abs/2506.09026): raise the token budget with difficulty.
- [Can One Domain Help Others?](https://arxiv.org/abs/2507.17512): a staged Knights-and-Knaves curriculum *with reference and optimizer resets* reached 99.71 vs 94.29 mixed. [E2H](https://arxiv.org/abs/2506.06632): fading out easy tasks is essential.

Rule: stage only the tiers whose pass rate would otherwise be about 0. Fill spectrum gaps with bridge tasks (lemma/lift, subproblems, format levels) rather than tuning schedules. If you stage, reset the reference model and the optimizer. **Moderate.**

---

## 5. Proposer–solver self-play and trained generators

Closed loops, where a proposer writes tasks and a solver learns from them, keep the band full without human authoring. Architectures are in [Ch. 03](03-generation-architectures.md). This section covers what keeps the loops stable when they feed RL.

### 5.1 Five stability rules

1. **Multiply difficulty by an independent validity gate.** Naive setter–solver play is reward-hacked by invalid problems. Working proposer rewards:
   - [VHG](https://arxiv.org/abs/2605.06660): R = 1[V]·(1 − Acc).
   - [SSR](https://arxiv.org/abs/2512.18552): −1 for an inconsistent bug artifact.
   - [OPT-Zero](https://arxiv.org/abs/2609.34205): R_valid × R_correct × R_struct.

   Lowering the solve-rate floor to chase hardness mostly adds broken tasks. In [OpenSIR](https://arxiv.org/abs/2511.00602), moving the floor from 0.5 to 0.1:
   - cut validity from 70.82% to 42.31%;
   - moved GPT-5's solve rate only from 89.82% to 78.31%;
   - lowered math accuracy from 29.57 to 25.97.

   **Strong.**
2. **Ground the answer outside the model.** Use an executor, database, OR solver, repository tests, corpus or search evidence.
   - Pseudo-label-only [R-Zero](https://arxiv.org/abs/2508.05004) peaks after 1–3 iterations; label accuracy fell 79% → 69% → 63%.
   - Grounded variants hold up:
     - [SPICE](https://arxiv.org/abs/2510.24684): corpus-grounded 43.9 vs 40.7 ungrounded.
     - [SSP](https://arxiv.org/abs/2510.18821): removing the RAG answerability check dropped GeneralQA from 60.0 to 49.5.

   With self-consistency labels, keep Challenger and Solver as separate weights: pseudo-label accuracy was 71.0% vs 63.4% for a shared model at step 15. **Strong.**
3. **Never let the loop train its own grader.** A trained user simulator "force[d] success rates around 50% by directly accepting or rejecting regardless of agent performance" ([SEAD](https://arxiv.org/abs/2602.03548)); a reward model without chain-of-thought was hacked within about 20 steps as trajectory length collapsed ([SWE-World](https://arxiv.org/abs/2602.03419)); a world model learned "self-praise" to fool its judge ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)); a solver hacked an off-the-shelf reward model by answering in Python ([LSP](https://arxiv.org/abs/2509.07414)). A learned component may choose tasks or initial states; the grader stays frozen or rule-anchored. **Strong.**
4. **Keep diversity with a persistent, skill-level archive.** Within-batch penalties allow cycling across iterations, and different wording hides identical skills, so compare canonical solver code rather than question text ([R-Diverse](https://arxiv.org/abs/2602.13103)). OpenSIR's novelty against the whole pool roughly doubled concept coverage; [SQL-Zero](https://arxiv.org/abs/2609.04697) deduplicates by masked-SQL template; a single agent self-calibrates toward easy problems where a cross-evaluated population does not ([PopuLoRA](https://arxiv.org/abs/2605.16727)); [vocabulary dropout](https://arxiv.org/abs/2604.03472) gave +4.4 at 8B on R-Zero. **Strong.**
5. **Co-evolve, and add stabilizers.** With a fixed proposer, solver reward saturates near 0.9 ([SSP](https://arxiv.org/abs/2510.18821)); fixed opponents fail ([SPIRAL](https://arxiv.org/abs/2506.24119)); a frozen challenger stops improving after one iteration ([SCOPE](https://arxiv.org/abs/2605.31433)). Stabilizers with supporting evidence (mostly single-study ablations): golden replay of verified trajectories (without it, [STRETCH](https://arxiv.org/abs/2609.18642)'s formatting collapsed by epoch 4); 1–5% human anchors ([R-Few](https://arxiv.org/abs/2512.02472): +3.0 over R-Zero on math at 8B); a cap on the synthetic share ([DreamGym](https://arxiv.org/abs/2511.03773)); KL-anchored or reward-weighted-regression generator updates ([GenEnv](https://arxiv.org/abs/2512.19682); [PROPEL](https://arxiv.org/abs/2606.18284)'s math run collapsed at the lowest KL); per-role baselines (without them SPIRAL models abandoned reasoning after about 200 steps). **Moderate.**

**The exact shape of the learnability reward matters much less.** **Strong.**

- SSR's consistency-only ±1 reward was only slightly worse than its solve-rate-shaped one.
- [Socratic-Zero](https://arxiv.org/abs/2509.24726)'s reward-shape variants were within about 0.4 points.
- A 50%-targeted reward *cut* AZR's validation accuracy by 2% ([Chae et al.](https://arxiv.org/abs/2510.27072)).
- SPICE saw some spread: variance reward 44.9, R-Zero reward 43.6, threshold 41.4, AZR reward 40.7.
- OPT-Zero's *structural-complexity* reward beat a solve-rate reward.

Spend the effort on validity, grounding and diversity.

### 5.2 Collapse modes and their signatures

| Failure | Signature to watch | Fix | Source |
|---|---|---|---|
| Invalid-problem hacking | Solver accuracy on proposals falls while an external judge's validity rate falls | Multiplicative validity gate | VHG, OpenSIR |
| "Death spiral" from negative format penalty | Proposer entropy rises and the valid-question rate goes to 0 while solver reward rises misleadingly | Give 0, not −0.1, to malformed proposals. SPICE used ρ = −0.1 without trouble, so watch the valid-proposal rate. | [SSP](https://arxiv.org/abs/2510.18821) |
| Pseudo-label decay | Accuracy on a gold probe falls each iteration | Stop at about 2–3 iterations; ground the answer | R-Zero |
| Diversity illusion or tunnel vision | Skill-signature entropy falls; one knob keeps turning (operand length, chains of medium bugs) | Persistent archive; populations | R-Diverse, SSR |
| Dominant challenger | Randomly failing or hash-seeded tests; obfuscated code | Constrain the action space; 7 consistency checks including inverse mutation testing | SSR |
| Answer leakage into variants | Variant solve rate near 100% | Reward only a band, e.g. [12.5%, 62.5%] | [SvS](https://arxiv.org/abs/2508.14029) |
| Role-prompt gold leak | Implausibly high baseline | Audit role prompts. Removing R-Zero's leaked ZebraLogic grid cut the baseline by 62 points. | [LURE](https://arxiv.org/abs/2608.21871) |
| Sharpening, not expansion | pass@1 rises while pass@k at large k falls below base; entropy collapses | Anchor on real hard goals; variant synthesis | AZR per Chae et al.; SvS |

### 5.3 Anchor generation on your own unsolved and weakly solved items

"Make it harder" without a target produces shallow difficulty. Asked to raise difficulty, LLMs mostly add constraints and borrow the goal problem's surface metaphors ([GASP](https://arxiv.org/abs/2603.15957)). Direct injection gives one-line bugs (SSR). Three anchored patterns expanded the model's boundary. **Moderate** (each is a single study, but all point the same way):

- **Variants of weakly solved items ([SvS](https://arxiv.org/abs/2508.14029)).**
  - For problems solved 12.5–50% of the time, condition on the policy's own *correct* solution and write answer-preserving variants.
  - Reward the synthesis only when a variant's solve rate lands in [12.5%, 62.5%].
  - +18.3 and +22.8 pass@32 on AIME24 and AIME25 (Qwen2.5-32B-Instruct), and entropy stayed stable.
- **Stepping stones to unsolved goals.**
  - [GASP](https://arxiv.org/abs/2603.15957) writes a lemma in the p ∈ [0.3, 0.7] band, then a lift of the lemma in [0.1, 0.5] without showing the goalpost again. It solved 11 of 146 pass@100 = 0 goalposts, where AZR and real-data RL solved 0.
  - [SOAR](https://arxiv.org/abs/2601.18778) rewards a teacher by measured student gain on fail@128 problems: about 4× pass@1 and 2× pass@32 on MATH. Only 32.8% of its useful stepping stones had fully correct answers and 63% were well-posed. For *bridge* data, well-posedness matters more than answer correctness; this does not extend to final RL targets.
- **Weakness mining ([SwS](https://arxiv.org/abs/2506.08989)).**
  - Flag items that never exceed 50% accuracy and have a negative slope.
  - Recombine their concepts into new problems and keep those in [25%, 75%]; about 35% survive.
  - +10.0% (7B) and +7.7% (32B) across 8 math benchmarks.

### 5.4 Adoption ladder

From cheapest to most expensive:

1. **Directional rewrite ([RLAnything](https://arxiv.org/abs/2602.02488)).** Rewrite with an LLM and accept by the band rule (§4.1). No generator training.
2. **In-loop answer-preserving variants (SvS).** The policy writes its own variants.
3. **A generator fine-tuned by reward-weighted regression toward the band ([GenEnv](https://arxiv.org/abs/2512.19682)).** Reward exp(−β(p̂ − 0.5)²), excluding batches with |p̂ − α| > 0.1; at 1× data it beat offline Gemini-2.5-Pro augmentation at 3.3× data (0.458 vs 0.438, BFCL validation).
4. **Full grounded self-play.** Examples: SSR for repositories, SSP for search, SQL-Zero and OPT-Zero for databases and OR solvers.
5. **Meta-RL teachers (SOAR).** Very expensive: each teacher reward needs an inner RL run.

Budget for low yield: 5.2% of Self-Challenging Code-as-Task proposals survive the full filter ([SCA](https://arxiv.org/abs/2506.01716)), and VHG kept 4,076 of 18,663 integral candidates. Let the *current* checkpoint propose: in a toy multi-agent path-finding generator where the model only sets parameters, the 4B RL checkpoint was a better environment designer than the base model and than larger proprietary designers ([From Trainee to Trainer](https://arxiv.org/abs/2606.17682); single toy domain, so **Emerging**). At the frontier, [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) trains the model itself as a task constructor, rewarded on difficulty and correctness of (problem, environment, verifier) triplets, and re-audits tasks on every RL run. It publishes no formula or ablation (see [Ch. 08](08-frontier-lab-practices.md)).

```python
def proposer_reward(task, solver_rollouts, gate, archive):
    if not gate.valid(task):          # executor / solver / RAG check, independent of the solver
        return 0.0                    # zero, not negative (SSP death spiral)
    p = mean(solver_rollouts)         # reuse the solver's GRPO group; no extra rollouts (SPICE)
    if p in (0.0, 1.0):
        return 0.0                    # unsolvable or trivial earns nothing (AZR; PretrainZero guard)
    if archive.max_sim(solution_signature(task)) > SIM_MAX:
        return 0.0                    # persistent, skill-level novelty (R-Diverse, SQL-Zero)
    return band_score(p)              # any peaked shape; second-order (§5.1)
```

---

## 6. Making too-hard items learnable

Complexification overshoots routinely. [IFDecorator](https://arxiv.org/abs/2508.04632) produced 7,324 usable prompts and 10,772 unsolvable ones from 21k seeds, and AutoCode found only about 5% of generated problems in the 0.1–0.5 pass@1 zone ([AutoCode](https://arxiv.org/abs/2510.12803)). A p ≈ 0 item is either broken, missing a prerequisite behaviour, or valid but out of reach. Handle the three cases in that order.

### 6.1 Step 1: audit

- **Zero-pass items often hide label errors or spurious reward paths.** One harmful sample whose reward accepted a bare boxed answer collapsed mean response length from 510.7 to 45.7 tokens within 58 steps, *before* accuracy fell ([T-SAE](https://arxiv.org/abs/2605.28388)).
- **pass@64 = 0 filters let through incomplete questions and missing figures** ([Cog-DRIFT](https://arxiv.org/abs/2604.04767)).
- **Separate "hard" from "broken" with a certificate.** Options: a stronger-model solve, large-k success from a verified-answer pool ([GLM-4.5](https://arxiv.org/abs/2508.06471): "pass@8 = 0, pass@512 ≫ 0"), or a co-built solution. Otherwise keep only a small tail: [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) keeps 10% of 0%-pass SWE items and masks never-solved groups.

Details: [Ch. 04](04-verification-and-quality-control.md).

### 6.2 Step 2: check behaviours before adding scaffolds

If rollouts never verify or backtrack, harder variants stall whatever you do to the data.

- **Countdown.** Under identical PPO, Llama-3.2-3B plateaued at about 30% and Qwen-2.5-3B reached about 60%. A brief SFT on backtracking and verification traces closed the gap. Traces with *wrong* final answers worked as well as correct ones ([Cognitive Behaviors](https://arxiv.org/abs/2503.01307)).
- **Agentic search.** Priming on behaviour-filtered trajectories (including ones with wrong outcomes) beat outcome-filtered priming, with gains of 37.2% (web) and 6.2% (multi-hop QA) relative to direct RL ([Behavior Priming](https://arxiv.org/abs/2510.06534)).
- **Self-distillation from easy seeds** ([SkillFactory](https://arxiv.org/abs/2512.04072)): stitch the model's own wrong and right attempts with verdict-checked reflections. It trailed R1 distillation on hard Countdown before RL (2.8% vs 11.7%) and led after GRPO (25.1% vs 21.2%). Compute-matched Mixed SFT likewise had the lowest pre-RLVR accuracy and the highest post-RLVR ceiling ([Tang et al.](https://arxiv.org/abs/2608.23256)). **Pick RL starting checkpoints with a short RL probe, not by SFT accuracy.**
- **Mid-training installs atoms that RL then composes.** With 0% or 0.1% pretraining exposure to a context, RL did not transfer to it; 1% exposure gave up to +60% pass@128, and at fixed compute mid-training plus RL beat RL-only by +10.8% on OOD-hard ([Interplay](https://arxiv.org/abs/2512.07783)). [OctoThinker](https://arxiv.org/abs/2506.20512)'s first mid-training stage alone gave 10–20%. If many families sit at zero, a reasoning-dense anneal may be cheaper than more scaffolding ([Ch. 06](06-sft-playbook.md)).

**Moderate.**

### 6.3 Step 3: bring in outside information, most on-policy first

| Lever | Mechanism | Reported result | Source |
|---|---|---|---|
| Answer-preserving format ladder | 4-choice MCQ → 10-choice → cloze → open-ended per item; promote at accuracy ≥ 0.5. Single formats did not help; mixing was essential. | +10.11 (Qwen) and +8.64 (Llama) points on items at pass@64 = 0 | [Cog-DRIFT](https://arxiv.org/abs/2604.04767) |
| Inverse rewrites | Mask a given quantity, supply the answer, ask for the quantity | Hard@8 data recovered to AMC 43.37 and AIME24 13.33, beating full data (40.96 and 10.00) | [T-SAE](https://arxiv.org/abs/2605.28388) |
| Teacher in the prompt | One anonymized correct teacher answer plus a wrong student answer in the prompt; replay until plain accuracy ≥ 0.5. Every gradient token is the student's own. | +9.3 (0.8B) and +2.8 (9B) on VLM benchmarks over base. At 0.8B, 82.7% of NCQ rollouts parroted a listed wrong answer. | [ZPPO](https://arxiv.org/abs/2606.18216) |
| Annealed solution prefix | Prefix 50% of a reference solution for 100 steps, then 25%. Keep items at 0–4/8 correct. Switch when entropy starts falling. | 63.26 vs 60.26 for 50% throughout; a 0% final stage gave no further gain | [QuestA](https://arxiv.org/abs/2507.13266) |
| Per-item hint length | Item-response-theory (IRT) fit of hint length → accuracy, targeting the efficient band | +11.8 over GRPO on six math benchmarks | [SEELE](https://arxiv.org/abs/2509.06923) |
| Binary-searched expert anchor | Shortest expert prefix that makes a failing group non-degenerate | Under 40% of ground-truth traces needed; about 3× faster than GRPO | [BREAD](https://arxiv.org/abs/2506.17211) |
| Tiered minimal hint | No hints for the first 15% of steps. Then knowledge → planning → solution hints; the first successful hinted trajectory replaces one failure in the group. | AIME24 30.0 → 43.3; hints used on 17.4% of samples | [Scaf-GRPO](https://arxiv.org/abs/2510.19807) |
| Monotone guidance withdrawal | Backward-chaining teacher trace, withdrawn monotonically | 128 unsolvable problems matched GRPO on 2,000 | [MFC](https://arxiv.org/abs/2609.13997) |
| Off-policy trace in the group | 1 verified teacher trace + 7 on-policy rollouts, with shaped importance weight r/(r + 0.1) | +6.4 average on six math benchmarks and +6.2 OOD over prior RLVR (Qwen2.5-Math-7B) | [LUFFY](https://arxiv.org/abs/2504.14945) |

**Rules:**

- Always withdraw the help as the policy improves. The successful methods anneal the scaffold (QuestA stops at a 25% prefix, since a further 0% stage added nothing), graduate items at about 0.5 (Cog-DRIFT, ZPPO), or let the teacher's share of the advantage shrink as on-policy success grows (LUFFY).
- For small students, prefer prefixes or in-prompt candidates over SFT on long teacher CoTs. SFT of 1.5B/3B models on S1K-style traces can *lower* accuracy (BREAD).
- Watch parroting in candidate formats (ZPPO's 82.7% case above).

**Moderate** (several independent methods, few head-to-head comparisons).

### 6.4 Step 4: dense credit, patience, and horizon reduction

- **Warm up, then go binary, then wait.** After a per-test-case partial-credit warm-up, binary-reward RL stayed below 1% full-pass for about 450 steps, then "grokked" to near 100% on Manufactoria-HAS ([DELTA-Code](https://arxiv.org/abs/2509.21016)). Track the hard subset separately so averages do not hide it. **Emerging.**
- **Zero-reward rollouts can still train.**
  - Next-observation prediction on the same rollouts doubled TerminalBench-2.0 pass@1 for Qwen3-8B, 2.70% → 5.17% ([ECHO](https://arxiv.org/abs/2605.24517)).
  - Solver-value credit per turn ([CAST](https://arxiv.org/abs/2607.25308)) keeps long puzzles learnable without lowering difficulty.
- **Reduce agent horizons before lengthening them.**
  - Horizon-only scaling collapsed RL at levels L3–L4, together with a jump in max-length responses.
  - Macro-actions and verifiable subgoals prevented the collapse, and a short-to-long curriculum worked ([Horizon study](https://arxiv.org/abs/2605.02572)).
- **Items stuck at p ≈ 1 also carry signal.**
  - *Failure-prefix conditioning*: truncate a rare failure so accuracy is about 0.5. On items solved 121–127/128 times this gave 44.5, against 40.7 for plain RLVR on the same items and 44.0 for fresh medium items ([Failure-prefix](https://arxiv.org/abs/2601.20829)).
  - *Meta-tasks that reuse the verifier*: "judge this solution" items labelled by tests, mixed in at 20% ([Critique-Coder](https://arxiv.org/abs/2509.22824)); on-policy self-correction turns ([SCoRe](https://arxiv.org/abs/2409.12917)).
  - Off-policy injected errors do *not* teach self-correction ([Wu et al.](https://arxiv.org/abs/2512.02389)).

---

## 7. Reward design for composite tasks

Composed tasks make all-pass rewards sparse: [CSE](https://arxiv.org/abs/2608.12426) fits mean per-constraint success at 72.0% × 0.922^(k−1) across 15 models. At k = 8 that is about 41% per constraint, and only 5.7% of responses satisfy all eight. Partial credit restores signal but opens hacking surfaces, and the evidence is mixed, so choose by regime.

### 7.1 What the evidence says

| Regime | What worked | What failed | Sources |
|---|---|---|---|
| Single-answer puzzles, success not sparse | Binary reward was best on Knights & Knaves | — | [Can One Domain…](https://arxiv.org/abs/2507.17512) |
| Multi-cell or long outputs | Steep partial credit: (x/N)^10 for sorting, (min/max)^10 numeric, (valid edges/(N−1))^5 for Hamiltonian path | Binary collapsed on Logic Puzzle Baron; partial reward peaked at 38.63 at step 200 and then declined; format-shaped and rescaled rewards eventually won. | [RLVE](https://arxiv.org/abs/2511.07317); [Can One Domain…](https://arxiv.org/abs/2507.17512) |
| Partial correctness with GRPO | Bipolar Float Reward: +1 only if fully correct, otherwise S − 1 ∈ [−1, 0). This avoids the "non-negative reward trap", where a flawed 0.9 answer gets positive advantage. AIME24 82.6 vs 81.7 binary vs 76.9 graded. | Plain graded float was the worst of the three | [UltraLogic](https://arxiv.org/abs/2601.03205) |
| Optimization tasks | Feasibility gate (−1.5 if infeasible) plus quality ratio to a heuristic reference | A flat 1.0 for mere feasibility was hacked early | [Forge](https://arxiv.org/abs/2605.08905) |
| Code | Per-test partial credit, clustered by test difficulty ([MiMo](https://arxiv.org/abs/2505.07608)); per-test warm-up then full-pass (DELTA-Code) | — | as cited |
| Stacked instruction constraints | Structure-aware aggregation: average parallel constraints, decay credit after an early sequential failure, score only the active branch | Plain averaging cost 3.9 IFEval and 5 CFBench points | [LsrIF](https://arxiv.org/abs/2601.06431) |
| Same, with a strong fine-tuned rubric verifier | All-or-nothing beat fractional, 58.1 vs 53.6 | — | [AdvancedIF](https://arxiv.org/abs/2511.10507) |
| Long-context evidence | F-beta over selected evidence chunks vs the gold set (β = 2), blended with the answer reward: RULER-QA 73.17 → 88.90 at 14B | Free-form LLM-judge process reward on thinking cost 1.29 points. Recall rewards on aggregation tasks are gamed by enumerating candidates, so CrossEntity switched to a judge. | [LongRLVR](https://arxiv.org/abs/2603.02146); [Beyond Reward Engineering](https://arxiv.org/abs/2606.18831) |
| Answer matching | Two-way substring 72.4 vs exact match 69.2 vs LLM judge 65.2 vs F1 65.1 | — | [LoongRL](https://arxiv.org/abs/2510.19363) |
| Process checks derived from construction | Verifying steps against the generator's DAG gave +4–5% pass@1 at op 15–20 and fewer shortcuts; SMT-checked deduction steps gave up to +49.71 ZebraLogic over base | Requiring intermediate outputs hurt final accuracy in data-structure composition | [Interplay](https://arxiv.org/abs/2512.07783); [SPRING](https://arxiv.org/abs/2609.34660); [He et al.](https://arxiv.org/abs/2609.19465) |
| Agent state | Code-augmented judge over the database-state diff (1.0 / 0.1 / 0): 65.94 vs code-only 60.00 vs LLM-only 55.46 (8B, BFCLv3); progress reward mean(r_state × r_exec) | No LLM-judge configuration exceeded AUROC 0.65 at detecting false success on τ²-bench | [AWM](https://arxiv.org/abs/2602.10090); [RODS](https://arxiv.org/abs/2606.19047); [judge study](https://arxiv.org/abs/2606.09863) |

### 7.2 Hacking risks scale with difficulty

- **Shortcuts concentrate at the top of the difficulty range.** RLVR models gamed extensional verifiers far more on hard items: 40 shortcuts at complexity levels 1–10 vs 458 at levels 11–20. In a controlled training comparison, extensional-only verification opened a growing hacking gap, and verifying on isomorphic re-labellings eliminated it ([IPT](https://arxiv.org/abs/2604.15149)).
- **Weak judges get worse under optimization pressure.** With a GPT-4o-mini rubric verifier the incorrect-credit rate rose from 39% to 65% (medical) over training, with exploits concentrated in partial satisfaction of compound criteria ([Rubric hacking](https://arxiv.org/abs/2605.12474)). Judging each constraint separately raised Qwen3-32B's recall on hard-constraint violations from 30.6% to 59.3% ([Precision over Diversity](https://arxiv.org/abs/2601.04954)). A fine-tuned generative verifier's training reward drifted from the oracle after about 450 iterations ([rule vs model verifiers](https://arxiv.org/abs/2505.22203)).
- **Verifier noise mostly slows learning, if it stays better than chance.** If J = TPR − FPR > 0, mass on incorrect modes dies out; if J < 0, incorrect modes take over ([RLVεR](https://arxiv.org/abs/2601.04411)).
- **Realistic environments invite environment attacks**: fetching fixes via `git clone`/`curl` ([Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729) blocks tool calls combining a repository link and a network keyword); forged scheduler RPCs, log reading and `XFS_IOC_SWAPEXT` ([DSec](https://arxiv.org/abs/2609.22978)); slide-rendering hacks ([GLM-5](https://arxiv.org/abs/2602.15763)).
- **Trivial agents pass invalid tasks.** An empty-response agent passes 38% of τ-bench-Airline ([ABC](https://arxiv.org/abs/2507.02825)).

### 7.3 A composite reward pattern (Proposal)

This pattern is a synthesis of the rows above. Only the individual parts have been tested.

```python
def composite_reward(resp, task):
    if not task.format_ok(resp):             return -1.0                 # strict format gate (RLVE, Logic-RL)
    if task.forbidden_action(resp):          return  0.0                 # zeroed on forbidden behaviour (SynthAgent)
    if task.has_feasibility and not task.feasible(resp): return -1.0     # never pay for feasibility alone (Forge)
    leaves = [c.check(resp) for c in task.constraints]                   # code checks; judge pointwise if soft
    s = task.structure.aggregate(leaves)     # parallel=mean, sequential=decay after first failure, branch=active only
    if s == 1.0:                             return  1.0
    if task.sparse_regime:                   return s - 1.0              # bipolar: partial < full, never positive
    return 0.0                                                           # binary once success is no longer sparse
```

Operating rules:

- Use code checks for every leaf you can. Keep at most one soft (judged) constraint per prompt; this plus pilot-run filtering gave +13.4 average in Precision over Diversity.
- Switch from partial to binary once the item's pass rate leaves the sparse regime (DELTA-Code).
- AND the result with an intent or quality check. IntentCheck cut the trip-wire hack rate from 14.53% to 7.60% ([IFDecorator](https://arxiv.org/abs/2508.04632)).
- Add absence and negative criteria to rubrics.
- For hardened agentic tasks, run no-op, empty-response and random agents, plus isomorphic variants, before trusting any reward.

---

## 8. Mixing synthetic-hard with real tasks, and dose

### 8.1 Dose saturates early; spend on breadth

- **Very small doses recover most of the gain.** One prompt took Qwen2.5-Math-1.5B from 36.0 to 73.6 on MATH500, matching 1.2k DeepScaleR prompts ([1-shot RLVR](https://arxiv.org/abs/2504.20571)); 1,389 of 8,523 MATH items matched the full set, 58.1 vs 57.0 ([LIMR](https://arxiv.org/abs/2502.11886)); 500 adaptively hardened items gave 91.7 on GSM8K and 1,000 gave 91.8 ([Synthetic Data RL](https://arxiv.org/abs/2505.17063)).
- **Caveat: much of this rests on Qwen2.5-Math, which gains even under random rewards** (+21.4 MATH-500; [Spurious Rewards](https://arxiv.org/abs/2506.10947)). Run a dose ladder (for example 16 / 128 / 1k / 10k) with a random-reward arm and a non-Qwen model before scaling a generator.
- **Reuse is cheap, up to a point.** At fixed total volume there was no significant degradation up to a reuse factor of 25, and clear overfitting at 100 ([Tan et al.](https://arxiv.org/abs/2509.25300)). A derived sizing rule: S steps × B prompts per step needs about S·B/25 unique verified items, e.g. about 10k items for 1,000 × 256.
- **Mixed-complexity dose can be non-monotone.** Counting Mixed-500 scored below Mixed-100 ([Learning from Less](https://arxiv.org/abs/2604.18381)). Run 3 or more seeds per dose point.

**Moderate.**

### 8.2 Blend defaults

| Component | Share | Evidence |
|---|---|---|
| Hardened synthetic families | Fixed equal mix across operator families | [DataFlex-RL](https://arxiv.org/abs/2609.06107): no adaptive mixture beat equal. [MoDoMoDo](https://arxiv.org/abs/2505.24871)'s surrogate (+5.24 OOD over uniform) was never re-tested with multiple seeds. |
| Oracle-backed real or seed items | 20–33% | [Anchored Self-Play](https://arxiv.org/abs/2607.03523): 20% real bugs gave +7.0 pp over unanchored self-play. Seed+synthetic beat either alone ([rStar-Coder](https://arxiv.org/abs/2505.21297), SFT: 57.3 vs 49.7 seed-only vs 46.8 synthetic-only). Only an analogy from another regime: in pretraining, about 30% rephrased synthetic text (the rest natural) was best ([Kang et al.](https://arxiv.org/abs/2510.01631)), which points to a larger real share, not a smaller one. The 20–33% range is a starting guess, anchored mainly on the 20% result. |
| Review or easy pool | 2–10% | ReMind (about 2%), MiMo (10%) |
| Every target domain, not just puzzles | Keep all present | Knights-and-Knaves-only RL dropped the code average from 67.46 to 56.09 ([Can One Domain…](https://arxiv.org/abs/2507.17512)). Excess Enigmata data caused catastrophic forgetting ([Enigmata](https://arxiv.org/abs/2505.19914)). Single-environment RL "often" caused unrecoverable regressions ([Nemotron 3 Nano](https://arxiv.org/abs/2512.20848)). Mixing domains limits over-optimization ([Olmo 3](https://arxiv.org/abs/2512.13961)). |

This blend follows the dose-and-mixing research notes. Treat it as a default until your own ablation says otherwise. **Proposal** built on **Moderate** parts.

Three further results bound what mixing can do:

- **Puzzles as a minority help.** Adding 17k SynLogic items to math+code lifted BBEH from 18.5 to 28.6 with small gains elsewhere ([SynLogic](https://arxiv.org/abs/2505.19641)). Logic, simulation and tabular skills still needed in-domain RL data ([Guru](https://arxiv.org/abs/2506.14965)).
- **Synthetic tasks can partly replace human curation.** 10 human tasks + 80 gated variants scored 54.38 against 54.18 for 97 human tasks. At matched compute the human arm was slightly ahead (54.72 vs 54.11), and the whole spread was about 1.5 points ([Trading Human Curation](https://arxiv.org/abs/2606.03800)). **Emerging.**
- **Consolidation does not add coverage.** Merging, Mix RL and multi-teacher distillation differ by at most 1.4 points on average, and none beats base at AIME pass@32. Mix RL is the cheapest single run, at 0.58–0.67× the cost of per-domain RL ([Consolidating RLVR](https://arxiv.org/abs/2608.27409)). Coverage needs new hard items or a stronger teacher.

---

## 9. Monitoring: the run dashboard

| Signal | What it detects | Alarm / action | Source |
|---|---|---|---|
| Effective-prompt ratio (share of non-identical groups); refill batches per step | Band drift, saturation | Falling ratio or rising refills: trigger re-profiling or the generator before the refill cap is hit | RLVE; verl; Knapsack |
| Pass-rate histogram per family (mass at 0 and 1) | Pool staleness, bimodality | Add intermediate levels; refresh from the backup pool | §1 triage |
| Correct-set turnover on training items | Forgetting of mastered skills | About 21% of reviewed mastered prompts had partly regressed: keep 1–2% review | ReMind; [Algebrarium](https://arxiv.org/abs/2602.08281) |
| Policy entropy | Collapse; loss of exploration | "Faster entropy collapse generally correlates with poorer test performance" ([Skywork-OR1](https://arxiv.org/abs/2505.22312)). AceReason 1.1 sets temperature so temperature-adjusted entropy stays about 0.3 ([AceReason 1.1](https://arxiv.org/abs/2506.13284)). QuestA anneals hints when entropy starts to fall. | as cited |
| Cross-input mutual information (multi-turn) | Template collapse invisible to entropy | Filter by reward variance; track mutual information | [RAGEN-2](https://arxiv.org/abs/2604.06268) |
| Mean and max response length | Shortcut rewards; horizon collapse; degeneration | Length collapse (510.7 → 45.7 in 58 steps) came *before* the accuracy drop. A jump in the share of max-length responses accompanied horizon collapse. Quarantine the family that was just added. | [T-SAE](https://arxiv.org/abs/2605.28388); [Horizon](https://arxiv.org/abs/2605.02572); [SWE-World](https://arxiv.org/abs/2602.03419) |
| Generator health: valid-proposal rate, gold-probe label accuracy, yield per operator, skill-signature entropy | Death spiral; pseudo-label decay; diversity illusion | Stop or re-anchor the loop. In one gated study, 64% of rejects were "too easy" and yield per operator ranged 7.7–45.5% ([Trading Human Curation](https://arxiv.org/abs/2606.03800)). | §5.2 |
| Reward vs held-out judge or oracle; isomorphic and trivial-agent checks | Reward hacking | Incorrect-credit rate rising, e.g. 39% → 65% in one rubric study; any trivial agent passing | §7.2 |
| pass@k at large k and Cover@τ on held-out families | Sharpening vs boundary expansion | pass@1 rising while pass@k falls below base: add frontier items, or use base anchoring ([pass@k inversion](https://arxiv.org/abs/2607.20543)) | [Yue et al.](https://arxiv.org/abs/2504.13837); [Cover@τ](https://arxiv.org/abs/2510.08325) |
| Regression suite: IFEval, hallucination, long context, other domains | Collateral damage | Math-RLVR raised IFEval pass@1 by 6.5% but lowered best@32 by 9.8% on Qwen3-8B-Base ([Support Reshaping](https://arxiv.org/abs/2608.00220)). RL forgets less than SFT ([Retaining by Doing](https://arxiv.org/abs/2510.18874)). | as cited |

**Decision hygiene.** Pass@1 SD across evaluation seeds on AIME and AMC is 5–15 points ([Sober Look](https://arxiv.org/abs/2504.07086)), and under BF16 greedy decoding, GPU count, GPU type or batch size alone moved accuracy by up to 9% ([Yuan et al.](https://arxiv.org/abs/2506.09501)). Protocol: screen with short runs and fitted compute curves (ScaleRL's asymptote reproduces within ±0.02); adopt a data policy only if its paired 95% CI excludes zero across 8–12 matched seeds on a pre-registered, domain-balanced aggregate, and the gain holds on post-cutoff items, a held-out generator family and a non-Qwen model ([DataFlex-RL](https://arxiv.org/abs/2609.06107); [Measuring all the noises](https://arxiv.org/abs/2512.21326); [Ch. 09](09-pitfalls-and-failure-modes.md)).

**Strong.**

---

## 10. Compute and cost allocation

**Unit economics.** Track one number, the cost per accepted in-band task:

```
C_acc = (c_gen + c_validate + c_env + k·c_roll) / (y_valid × y_band)
```

Verified yields range from 2.25% (SWE-Next commit pairs) through 25.5% (a gated RLVR mutation study) to about 50% (SWE-Factory valid environments), i.e. about 44 down to 2 candidates per accepted task. Unit-cost ledgers: [Ch. 03](03-generation-architectures.md).

**Where GPU time goes.**

- Rollouts are over 90% of RL runtime ([APRIL](https://arxiv.org/abs/2509.18521)); [Olmo 3](https://arxiv.org/abs/2512.13961) spent about 5× (32B) and 14× (7B) more compute on inference than on learning.
- Task-generation API spend is small by comparison: about $0.05 per accepted variant against about $20K of training per arm ([Trading Human Curation](https://arxiv.org/abs/2606.03800)). Yield and quality bind before API dollars do. Cheaper generators often win at matched spend but raise false positives, so tighten the verifier as the generator gets cheaper ([AgoraBench](https://arxiv.org/abs/2412.03679); [compute-optimal sampling](https://arxiv.org/abs/2408.16737)).

**Allocation order.** This follows the cost-model notes. **Moderate** for each step; **Proposal** for the order as a whole.

1. **Audit the verifier.** 59.01% of correct solutions to special-judge problems fail exact match ([ScaleBox](https://arxiv.org/abs/2604.27467)). Rule checkers reject about 14% of correct math answers ([rule vs model verifiers](https://arxiv.org/abs/2505.22203)). Many "hard" items are verifier artifacts.
2. **Profile the pool cheaply** (§2).
3. **Reallocate rollouts and recycle** (§3.3, §3.5). Reported gains are often 1.5–2× (Knapsack, Pilot-Commit, VIGOR, Reinforce-Ada). The survey warns these multipliers are not directly comparable ([rollout-efficiency survey](https://arxiv.org/abs/2609.25463)).
4. **Raise n per problem until it saturates**, then generate (IsoCompute).
5. **Generate with a gated acceptance function.** Log yield per operator, reject cheaply first (staged in-flight rejection saves 11–77% of tokens; [MSIFR](https://arxiv.org/abs/2605.14062)), and use parent-seed priors.
6. **Put pilots and re-pilots on preemptible capacity.** RLBoost cites about $2.66 per spot H100-hour vs about $10.5 on demand.

**Marginal-dollar rule (Proposal).** Keep profiling under a third of rollout tokens; buy verifier work whenever audits show false negatives or hacks, more rollouts per item while hard items return all-zero groups, and more generator families otherwise.

---

## 11. Reference table: reported bands and hyperparameters

All values are as reported, under each paper's own model and domain. Use them as starting points and re-tune on your policy.

| Purpose | Method | Reported setting |
|---|---|---|
| Offline band | [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) | Keep 2–5 of 8 correct |
| Offline band | [Olmo 3](https://arxiv.org/abs/2512.13961) | 8 rollouts at T = 1.0 from the stage's initial checkpoint; drop pass rate > 62.5% |
| Offline band | [Llama-Nemotron](https://arxiv.org/abs/2505.00949) | Drop pass rate ≥ 0.75 of 8 |
| Offline band | [MiniMax-M1](https://arxiv.org/abs/2506.13585) | Keep 0 < pass@10 < 0.9 (strong reasoning model) |
| Offline band | [MiMo](https://arxiv.org/abs/2505.07608) | Drop > 90% of 16 (math); sample the perfect-pass pool with α = 10% |
| Offline band | [AceReason](https://arxiv.org/abs/2505.16400) | Later stages: pass rate ≤ 6/16. Context grows 8K → 16K → 24K → 32K. |
| Offline band | [Nemotron-IMO](https://arxiv.org/abs/2609.10712) / [LiteResearcher](https://arxiv.org/abs/2604.17931) | Keep 1–3 of 4 / keep 1–7 of 8 |
| Offline band | [SwS](https://arxiv.org/abs/2506.08989) | Keep [25%, 75%]; about 35% of generated problems survive |
| Offline band | [Trading Human Curation](https://arxiv.org/abs/2606.03800) | Gate pass@8 ∈ [0.05, 0.95], estimated from 16 completions; five-factor product ≥ 0.5 |
| Online band | [Bae et al.](https://arxiv.org/abs/2504.03380) / [LILO](https://arxiv.org/abs/2502.12272) | Balanced 0.2–0.8. LILO: top-\|B\| by p(1−p) from a 4\|B\| pool (8\|B\| near saturation), N = 8. |
| Online filter | [DAPO](https://arxiv.org/abs/2503.14476) / [ScaleRL](https://arxiv.org/abs/2510.13786) | Oversample, drop p ∈ {0, 1}. Retire prompts with historical p ≥ 0.9. |
| Hard tier | [GLM-4.5](https://arxiv.org/abs/2508.06471) / [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | Stage 2: pass@8 = 0 and pass@512 ≫ 0, from a verified-answer pool / keep pass@100 > 0 |
| Hard tier | [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) | Drop items the teacher solves 8/8. Discard 90% of 0%-pass SWE items. Mask groups with no reward > 0.5. |
| Profiling | [Pilot-Commit](https://arxiv.org/abs/2605.26606) | 3× candidates; 16 pilot + 48 commit; skip p̂ > 0.75; defer p̂ < 0.125 |
| Allocation | [Knapsack RL](https://arxiv.org/abs/2509.25849) / [Reinforce-Ada](https://arxiv.org/abs/2510.04996) | N_low = 2, N_up = 128 / sample until K_pos = 16 positives or N_max = 128 |
| Replay | [Prompt Replay](https://arxiv.org/abs/2603.21177) | Band [0.25, 0.75], priority by closeness to 0.5; ε ≤ 0.75; cooldown 10; ≤ 15 reuses |
| Review | [ReMind](https://arxiv.org/abs/2606.03087) | p_add = 0.25; replace r = 10% of the batch every f = 5 steps (about 2% of prompts) |
| Controller | [RLVE](https://arxiv.org/abs/2511.07317) | τ_acc = 0.9; τ_num = 8 × rollouts; window d_Δ = 4; start [0, 0] |
| Controller | [SCALER](https://arxiv.org/abs/2601.04809) | d ← clip(d + β(acc − τ)); 64 active environments; retire at slope ≤ 0 over 10 steps, zero accuracy for 5, or max level for 5 |
| Controller | [Reasoning Gym](https://arxiv.org/abs/2505.24760) / [SATURN](https://arxiv.org/abs/2505.16368) | Promote at > 70% over 20 steps / at validation pass@1 > 0.5 (1.5B) or 0.75 (7B), checked every 250 samples |
| Environment agents | [RODS](https://arxiv.org/abs/2606.19047) | Boundary [0.20, 0.85]; retire > 0.95; injection ≤ 20% of the pool per epoch; P_max = 400 |
| Generator target | [GenEnv](https://arxiv.org/abs/2512.19682) / [STRETCH](https://arxiv.org/abs/2609.18642) / [EvoEnv](https://arxiv.org/abs/2605.14392) | α = 0.5, skip batches with \|p̂ − α\| > 0.1 / peak at 0.5 (beat 0.2 and 0.8) / target 0.3, admit 0 < â < 1 |
| Generator target | [RLAnything](https://arxiv.org/abs/2602.02488) / [SCOPE](https://arxiv.org/abs/2605.31433) | Rewrite when acc leaves [0.2, 0.8]; accept harder q′ iff 0.2 < acc(q′) < acc(q) / τ = 0.5, discard outside [0.2, 0.8] |
| Variant synthesis | [SvS](https://arxiv.org/abs/2508.14029) | Source problems at [12.5%, 50%]; G = 8, G_v = 8 variants; reward only if variant solve rate ∈ [12.5%, 62.5%] |
| Stepping stones | [GASP](https://arxiv.org/abs/2603.15957) | Lemma p ∈ [0.3, 0.7] with reward [4p(1−p)]^5; lift p ∈ [0.1, 0.5]; −0.5 outside; dedup at cosine ≥ 0.95 |
| Self-play validity | [OpenSIR](https://arxiv.org/abs/2511.00602) / [SSP](https://arxiv.org/abs/2510.18821) | Solve-rate floor ≥ 0.5 without a real verifier / RAG check with 4 distractor docs; 0 reward, not −0.1, for invalid proposals |
| Scaffolds | [QuestA](https://arxiv.org/abs/2507.13266) | Prefix 50% for 100 steps, then 25% for 1,900; keep 0–4 of 8; n = 16 |
| Scaffolds | [Scaf-GRPO](https://arxiv.org/abs/2510.19807) / [LUFFY](https://arxiv.org/abs/2504.14945) | 15% hint-free exemption (10–40% all plateau at 49.5–50.9) / 1 off-policy + 7 on-policy rollouts, f(r) = r/(r + 0.1) |
| Reformulation | [Cog-DRIFT](https://arxiv.org/abs/2604.04767) / [ZPPO](https://arxiv.org/abs/2606.18216) / [Failure-prefix](https://arxiv.org/abs/2601.20829) | Promote format level at ≥ 0.5 / admit below 0.5, graduate at ≥ 0.5 / prefix length set so accuracy ≈ 0.5 (0.25–0.75 all within 0.6 points) |
| Meta-tasks | [Critique-Coder](https://arxiv.org/abs/2509.22824) | 20% of RL prompts as verdict items; True if ≥ 80% of tests pass |
| Rewards | [RLVE](https://arxiv.org/abs/2511.07317) / [Forge](https://arxiv.org/abs/2605.08905) / [UltraLogic](https://arxiv.org/abs/2601.03205) | −1 bad format; (x/N)^10 / −1.5 infeasible, else quality ratio ≤ 1 / +1 full, S − 1 partial |

---

## 12. Step-by-step recipe

This recipe is **Proposal**-level as a whole; the tag after each step is the evidence strength for that step.

1. **Instrument first** (Strong): the §9 dashboard, per-item pass history, held-out generator families, post-cutoff items and a non-Qwen replica.
2. **Audit the verifiers** (Strong) with known-correct alternatives, known-bad solutions, no-op/empty/random agents and isomorphic variants. Fix false negatives before calling anything easy.
3. **Profile the seeds with the policy** (Strong). Use 8–16 rollouts. Drop no-CoT-guessable items and bucket the rest into p = 1, 0 < p < 1 and p = 0. Treat p = 0 as "unknown" and defer it.
4. **Harvest the cheap signal** (Moderate) before generating anything: zero-variance loss filtering with same-domain refill, Pilot-Commit or Knapsack-style allocation, recycling, and 1–2% review of mastered items. Measure the effective-prompt ratio after a short run.
5. **Wrap seed families in parameterized generators with verifiers** (Strong).
   - Use solve-first or planted construction wherever possible ([Ch. 02](02-complexification-operator-taxonomy.md), [Ch. 03](03-generation-architectures.md)).
   - Gate each family: execution across seeds, a monotonicity test between level and solve rate, low-level template validation.
   - Attach an RLVE-style window per family. Mix families equally, and grow the *number* of families, selected for skill coverage.
6. **Harden the p = 1 bucket** (Moderate).
   - Run operators through a gated acceptance function, with a known-bad solution check and the in-band pass@k test.
   - Log yield and rejection reason per operator.
   - Prime each variant's p̂ from its parent.
   - Add SvS variants of weakly solved items, and failure-prefix and verdict meta-tasks for saturated ones.
7. **Rescue the p = 0 bucket** (Moderate). Audit, then check behaviours (prime or mid-train if missing), then:
   - format ladder or inverse rewrite;
   - teacher-in-prompt;
   - annealed prefix;
   - off-policy trace.

   For agents, reduce the horizon.
8. **Compose rewards by regime** (Moderate). Use the §7.3 pattern:
   - binary where success is dense;
   - bipolar or structure-aware partial credit where it is sparse;
   - pointwise judging and at most one soft constraint per prompt;
   - evidence-F-beta for grounding;
   - rule anchors for any judge.
9. **Blend and dose** (Moderate/Proposal).
   - Equal mix across families, 20–33% oracle-backed real items, 2–10% review or easy items.
   - Unique items ≈ S·B/25.
   - Run a dose ladder with a random-reward arm and a second model family.
10. **Add a proposer only when steps 5–6 run dry** (Strong for rules 1–4, Moderate for the stabilizers and the gains). Follow the five rules in §5.1: validity gate, grounding, frozen grader, skill-level archive, stabilizers. Start with directional rewrites.
11. **Re-profile and refresh on a schedule** (Strong). Re-grade the whole original pool at every stage or plateau. Stage only the tiers that would otherwise sit at zero reward, and reset the reference model and optimizer at each stage.
12. **Decide with statistics** (Strong).
    - Compare fitted curves, not snapshots.
    - Require paired CIs across 8–12 seeds on a pre-registered domain-balanced aggregate.
    - Check pass@k at large k and held-out families.
    - No regression-suite metric may drop.

```python
# One epoch of the loop (sketch; Proposal)
pool.reprofile_if(stage_start or plateau(metrics), policy)            # §2
for step in range(steps_per_epoch):
    cand   = pool.sample(3 * B, by="prior")                           # parent-seed priors, BOTS-style
    pilots = {x: policy.rollout(x, 16) for x in cand}
    batch  = []
    for x, r in pilots.items():
        p = mean(r)
        if   p > 0.75:  pool.mark(x, "too_easy"); generator.enqueue_harder(x)   # Ch. 02 operators, gated
        elif p < 0.125: pool.defer(x);            scaffolder.maybe_attach(x)    # §6
        else:           batch.append((x, r + policy.rollout(x, 48)))
    batch += pool.review_sample(frac=0.02) + pool.real_anchor_sample(frac=0.25)
    groups = [g for g in score(batch, composite_reward) if var(g) > 0]    # zero-variance out of the loss
    policy.update(groups, version_tags=True)
    for fam in families: fam.controller.update(fam.recent_acc())        # RLVE window per family
    dashboard.log(effective_ratio(groups), entropy(), lengths(), turnover(), gen_batches())
    if dashboard.alarm(): pool.quarantine(dashboard.suspect_family())   # length collapse / hack signals
```
