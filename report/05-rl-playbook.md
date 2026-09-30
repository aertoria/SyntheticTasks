# RL playbook: feeding a policy hard synthetic tasks it can learn from

> **Key takeaways**
>
> - **Only mixed-outcome groups train, so manage the pool around a policy-relative pass-rate band.** Keep roughly 0.2–0.8 and aim the centre at about 0.3–0.5. Measure it with the model being trained and re-measure every stage. Once the band empties, stop filtering and start generating. **Strong.**
> - **Build more generator families rather than tuning the curriculum.** Give each family its own online difficulty controller and mix families equally. Across controlled studies, adding environments keeps paying off. Clever orderings and adaptive mixtures rarely beat an equal mix, except when the hardest tier would otherwise get zero reward. **Strong (breadth); Moderate (ordering null).**
> - **Proposer–solver loops are stable only with four things:** a validity gate independent of the solver, external grounding, a persistent diversity archive keyed on solution signatures, and a grader the loop never trains. The shape of the proposer's reward matters much less than these four. **Strong.**
> - **For items at p ≈ 0, audit first, then add the most on-policy help that works.** The order is: answer-preserving reformulation, then a teacher answer in the prompt, then annealed solution prefixes, then an off-policy trace in the group. Install missing behaviours (verify, backtrack) by priming or mid-training before adding scaffolds. **Moderate.**
> - **Composite rewards need three layers:** a validity gate, the outcome reward, and structure-aware partial credit only where success is sparse. Every shaped term opens a hacking surface. Test it with invariance checks, trivial-agent baselines and a held-out judge panel. **Moderate.**
> - **Spend compute in this order:** audit the verifier, reallocate rollouts, recycle zero-variance items, and only then generate. Rollouts are over 90% of runtime, and profiling can use about a third of rollout tokens. Anchor hard synthetic data with 20–33% real items and 2–10% review or easy items. **Moderate/Proposal.**

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

Scope: this chapter covers RLVR with GRPO-family estimators and agentic RL on synthetic tasks that are harder than your seeds. Other chapters cover related ground:

- Why tasks look easy and what "hard" should mean: [Chapter 01](01-diagnosis-difficulty-and-learning-signal.md).
- The operators that make tasks harder: [Chapter 02](02-complexification-operator-taxonomy.md).
- Pipelines that generate hard tasks: [Chapter 03](03-generation-architectures.md).
- Verifier construction and auditing: [Chapter 04](04-verification-and-quality-control.md).
- Distillation and SFT on hard tasks: [Chapter 06](06-sft-playbook.md).

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

- Qwen2.5-Math-7B on DAPO-Math-17K with N = 8: by iteration 1,000, about 40% of prompts are all-correct and about 20% all-wrong, so only about 20% of gradients are effective ([Knapsack RL](https://arxiv.org/abs/2509.25849)).
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
| [Hard Examples](https://arxiv.org/abs/2508.14094) | Fixed annotation budget | The hardest 10% (by base-model failure) gave gains up to 47%, against 3–15% for easy subsets, because hard items keep mixed outcomes throughout training. |

**Triage by symptom.** Pass-rate profiles call for different fixes. This table condenses the decision guide in the learnability notes. Sections 3–6 give the details.

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

The most cost-effective design reported is [Pilot-Commit](https://arxiv.org/abs/2605.26606):

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
| [DOTS](https://arxiv.org/abs/2506.05316) | Similarity to a small reference set that gets rollouts | 23–62% less RL time |
| [PROPEL](https://arxiv.org/abs/2606.18284) | Activation probe predicts solver pass rate for generated tasks | About 2× the frontier-task rate with under half the solver trials. The probe drifts, so re-anchor it. |
| [Rollout-free agentic predictor](https://arxiv.org/abs/2608.05797) | Entropy features plus embeddings predict IRT difficulty | Spearman ρ = 0.399 in-distribution, 0.225 on unseen benchmarks. A constant predictor already scores AUC 0.715, so evaluate predictors by rank correlation. |
| [Multi-solver disagreement](https://arxiv.org/abs/2608.30035) | Entropy of answers across 2–3 checkpoints or families | Mines items once single-model pass rates saturate |

Use predictors to triage candidates and to schedule re-profiling, not as the final admission filter. **Moderate.**

**Budget the profiler.** In a worked example (7B or 32B policy, k = 8, 8K tokens, in-band yield 0.3), profiling was about 36% of all rollout tokens. This figure is derived arithmetic from the cost-model research notes, not a measured result; see §10. Profiling rollouts are stateless and tolerate latency, so run them on preemptible capacity ([RLBoost](https://arxiv.org/abs/2510.19225)). For agents, stop groups whose action prefixes have converged ([Selective Rollout](https://arxiv.org/abs/2605.05802): 10.7% wall-clock saved).

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

**Re-profile on a schedule.**

- [Magistral](https://arxiv.org/abs/2506.10910) re-grades the *whole original* pool with its RL model, not only the survivors.
- [rStar2-Agent](https://arxiv.org/abs/2508.20722) re-filters 42K problems down to 17.3K harder ones before stage 3.
- [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) re-profiles at plateaus.
- [Tongyi DeepResearch](https://arxiv.org/abs/2510.24701) runs a background process that rescans the pool with intermediate checkpoints.
- [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) alternated between training and re-filtering.

Treat the band as a schedule, not a constant (see [Ch. 08](08-frontier-lab-practices.md)). **Strong.**

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
- **Refill the batch.** There are four options, in rough order of cost:
  - [DAPO](https://arxiv.org/abs/2503.14476) oversamples up front. Pilot-Commit's table shows 24,576 rollouts per step for DAPO against 8,192 for GRPO.
  - Active refill only until enough non-zero-gradient groups exist ([Olmo 3](https://arxiv.org/abs/2512.13961) active sampling).
  - Filter-and-replace ([AReaL-SEA](https://arxiv.org/abs/2601.22607): disabling it cut Airline pass^1 from 70.5 to 65.0).
  - Duplicate the non-zero-variance samples ([WebSailor](https://arxiv.org/abs/2507.02592) DUPO: about 2–3× faster than DAPO's dynamic sampling).
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

- **Mid-band prompt replay ([Prompt Replay](https://arxiv.org/abs/2603.21177)).**
  - Buffer prompts with p ∈ [0.25, 0.75], prioritized by closeness to 0.5.
  - Replayed prompts fill up to ε = 0.75 of each batch, with a 10-step cooldown and at most 15 reuses per prompt.
  - It speeds up early learning, then converges with the baseline. The benefit disappears once rollouts are no longer the bottleneck.
  - This is an efficiency lever, not a ceiling lever. **Moderate.**
- **Review mastered items ([ReMind](https://arxiv.org/abs/2606.03087)).**
  - Solved prompts enter a queue with p_add = 0.25. Every 5 steps, 10% of the batch is replaced with review prompts, about 2% of all prompts.
  - Of 1,275 review records, about 21% had both correct and incorrect rollouts, i.e. the skill had partly regressed.
  - A 1% review budget already removed GRPO's mid-training plateau.
  - Qwen3-VL-8B: 60.25 → 64.06 averaged over eight image-text benchmarks. **Emerging.**
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

- **Filtered groups are older than average.** With asymmetric filtering, the groups that survive are systematically older than the buffer average, because the easy prompts they solve were issued earlier. Tag tasks and tokens with both the policy version and the *profiling* version, and bound staleness per sample.
- **Refills shift the domain mix.** Refilling from any source shifts the mix toward fast domains. Replace dropped groups within the same domain (a logical consequence noted in the cost-model research notes, not measured).
- **Keep S·η roughly constant.** The largest stable learning rate scales about as 1/S, where S is the sync interval ([collapse laws](https://arxiv.org/abs/2607.01083)).
- **Use stale-data-robust objectives if the pool changes mid-run.** [M2PO](https://arxiv.org/abs/2510.01161) was stable with data at least 256 updates stale.

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

Frontier Learning's ablation shows mutation is the key ingredient:

| Variant | Score |
|---|---|
| PLR + exploration | 33.5 |
| PLR + mutation | 62.5 |
| Full method | 71.8 |

The adaptive window matters too. In RLVE, it beat an *oracle* static range covering all the levels the adaptive run reached. **Strong** (controllers beat static ranges); **Moderate** (choice among controllers).

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

Default to an equal mix across families plus per-family controllers. **Strong** (null results on ordering replicate).

- [Mordig et al.](https://arxiv.org/abs/2603.27226) found no robust gain from easy→hard ordering over random mixing, for SFT or RL. Llama3.2-1B GRPO scored 0.27 standard vs 0.28 with a curriculum on LinearDepth.
- [DataFlex-RL](https://arxiv.org/abs/2609.06107) ran 12 matched seeds per configuration. None of 8 selection or reweighting methods had a paired 95% CI excluding zero against uniform sampling, and none of 3 adaptive mixtures beat a fixed equal mix. Method signs flipped across scale.
- [ScaleRL](https://arxiv.org/abs/2510.13786) found curriculum choices mostly change compute efficiency, not the asymptote.
- [Depth × Complexity](https://arxiv.org/abs/2605.26934) found uniform mixing beat a staged 3-block curriculum at fixed budget.
- [Logic-RL](https://arxiv.org/abs/2502.14768) saw a practically negligible curriculum benefit.
- Theory agrees:
  - Mixed easy+hard data suffices for compositional learning ([Wang et al.](https://arxiv.org/abs/2505.23683)).
  - Mixed-difficulty RLVR forms an implicit curriculum when the difficulty spectrum is *smooth*; gaps cause plateaus ([Huang et al.](https://arxiv.org/abs/2602.14872)).

**When staging does matter: the top tier would otherwise get zero reward.**

- [h1](https://arxiv.org/abs/2510.07312): a uniform mix gave no long-horizon gains at equal compute. A staged horizon curriculum lifted AIME24 from 5.10 to 10.52 (avg@32, 3B).
- [ScaleLogic](https://arxiv.org/abs/2605.06638): the exponent γ in steps ∝ depth^γ was 1.33 with a curriculum and 2.36 with difficult-only data.
- [Horizon study](https://arxiv.org/abs/2605.02572): direct training at goal distance 10–12 barely improved; short-to-long gave marked gains.
- [e3](https://arxiv.org/abs/2506.09026): couple difficulty with the token budget; hard problems at an 8k budget suppress exploration.
- [Can One Domain Help Others?](https://arxiv.org/abs/2507.17512): a sequential Knights-and-Knaves curriculum *with a reference and optimizer reset* at each stage reached 99.71, against 94.29 for mixed.
- [E2H](https://arxiv.org/abs/2506.06632): fading out easy tasks is essential.

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
3. **Never let the loop train its own grader.**
   - A trained user simulator "force[d] success rates around 50% by directly accepting or rejecting regardless of agent performance" ([SEAD](https://arxiv.org/abs/2602.03548)).
   - A reward model without chain-of-thought was hacked within about 20 steps, and trajectory length collapsed ([SWE-World](https://arxiv.org/abs/2602.03419)).
   - A world model learned "self-praise" phrases to fool its judge ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)).
   - A solver hacked an off-the-shelf reward model by answering in Python ([LSP](https://arxiv.org/abs/2509.07414)).

   A learned component may choose tasks or initial states; the grader stays frozen or rule-anchored. **Strong.**
4. **Keep diversity with a persistent archive at the skill level.**
   - Within-batch penalties allow cycling across iterations, and different wording hides identical skills ([R-Diverse](https://arxiv.org/abs/2602.13103)). Compare canonical solver code, not question text.
   - OpenSIR's embedding-distance novelty against the whole pool roughly doubled concept coverage.
   - [SQL-Zero](https://arxiv.org/abs/2609.04697) deduplicates by masked-SQL template.
   - A single agent self-calibrates toward easy problems; a population scored by cross-evaluation does not ([PopuLoRA](https://arxiv.org/abs/2605.16727)).
   - [Vocabulary dropout](https://arxiv.org/abs/2604.03472) gave +4.4 at 8B on R-Zero.

   **Strong.**
5. **Co-evolve, and add stabilizers.**
   - Without co-evolution, progress stops:
     - A fixed proposer lets solver reward saturate near 0.9 ([SSP](https://arxiv.org/abs/2510.18821)).
     - Fixed opponents fail ([SPIRAL](https://arxiv.org/abs/2506.24119)).
     - A frozen challenger stops improving after the first iteration ([SCOPE](https://arxiv.org/abs/2605.31433)).
   - Stabilizers that were needed:
     - Golden replay of verified good trajectories; without it, formatting collapsed by epoch 4 ([STRETCH](https://arxiv.org/abs/2609.18642)).
     - 1–5% human anchors ([R-Few](https://arxiv.org/abs/2512.02472)).
     - A cap on the synthetic share ([DreamGym](https://arxiv.org/abs/2511.03773)).
     - KL-anchored or reward-weighted-regression generator updates ([GenEnv](https://arxiv.org/abs/2512.19682)). [PROPEL](https://arxiv.org/abs/2606.18284)'s math run collapsed at the lowest KL.
     - Per-role baselines; without them, SPIRAL models abandoned reasoning after about 200 steps ("thinking collapse").

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
3. **A generator fine-tuned by reward-weighted regression toward the band ([GenEnv](https://arxiv.org/abs/2512.19682)).**
   - Reward exp(−β(p̂ − 0.5)²); exclude batches with |p̂ − α| > 0.1 from generator updates.
   - At 1× data it beat offline Gemini-2.5-Pro augmentation at 3.3× data (0.458 vs 0.438 on BFCL validation).
4. **Full grounded self-play.** Examples: SSR for repositories, SSP for search, SQL-Zero and OPT-Zero for databases and OR solvers.
5. **Meta-RL teachers (SOAR).** Very expensive: each teacher reward needs an inner RL run.

Budget for low yield:

- 5.2% of Self-Challenging Code-as-Task proposals survive the full filter ([SCA](https://arxiv.org/abs/2506.01716)).
- VHG accepted 4,076 of 18,663 integral candidates.

Let the *current* checkpoint propose: the RL checkpoint was a better environment designer than the base model and larger proprietary designers ([From Trainee to Trainer](https://arxiv.org/abs/2606.17682)). At the frontier, [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) trains the model itself as a task constructor, rewarded on difficulty and correctness of (problem, environment, verifier) triplets, and re-audits tasks on every RL run. It publishes no formula or ablation (see [Ch. 08](08-frontier-lab-practices.md)).

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
