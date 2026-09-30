# Dose, mixing and measurement: how many hard tasks, synthetic:real blends, eval noise and decision protocols, and generator-signature audits

*Scope: how many hardened items to generate and how often to reuse them; how to blend synthetic, real-seed and easy items; how large evaluation noise is and which decision rules separate real gains from it; and how to audit a generator's fingerprints (judge leakage, style signatures, answer marginals, partial-input shortcuts, trait transfer) before training. Compiled 2026-09-30. Verification: 28 researcher entries checked against primary sources (arXiv abstract pages plus full-text HTML/PDF, with numbers read from the text and tables; 27 related works and 9 operator sources were resolved as well); 12 corrected; 0 dropped; 4 added (ScaleRL, RegMix, Agentic Benchmark Checklist, Isomorphic Perturbation Testing).*

---

## TL;DR

- **In RLVR the dose saturates early, so spend generator budget on breadth, not volume.** One prompt took Qwen2.5-Math-1.5B from 36.0 to 73.6 on MATH500, the same as 1.2k DeepScaleR prompts. LIMR's 1,389 of 8,523 MATH items matched the full set (58.1 vs 57.0 average). Synthetic Data RL got 91.7 on GSM8K with 500 adaptively hardened items and 91.8 with 1,000. Most of this evidence comes from Qwen2.5-Math, which improves even under spurious rewards. Run a dose ladder with a random-reward control and a second model family before you scale a generator.
- **Reusing items is cheap and unique items are expensive.** Holding total volume fixed, Tan et al. saw no significant degradation up to a reuse factor of τ = 25 and clear overfitting at τ = 100. A derived sizing rule: for S steps × B prompts per step, about S·B/25 unique verified items is enough. For example, 1,000 × 256 needs about 10k items. Beyond that point, extra items pay off only through diversity.
- **Prefer many seeds with few rewrites each over a few seeds with many rewrites.** Fidelity–Diversity metrics detect the diversity loss from few-seed rewriting (from 999 rewrites of 0.1% of GSM8K down to 9 rewrites of 10%), and that loss tracks lower downstream SFT accuracy. Selecting SFT traces for route diversity raised OLMo3-7B's post-RL pass@8 by 16.9 points on held-out environments, because more prompts were left with non-zero GRPO advantage.
- **Use a fixed equal mix as the default and make adaptive methods earn their place.** DataFlex-RL tested 13 policies with 12 matched seeds each. None of 8 selection or reweighting methods had a paired 95% CI that excluded zero against uniform sampling, and none of 3 adaptive mixtures beat a fixed equal mix. Merge, Mix RL and multi-teacher on-policy distillation differ by at most 1.4 points on average, and none beats the base model at AIME pass@32. Surrogate mixture search (MoDoMoDo: +5.24 OOD over uniform; RegMix) helps only when confirmed under multi-seed paired tests.
- **Anchor hardened data with real and easy items.** The evidence points to 20–33% oracle-backed real or seed items plus 2–10% replay of mastered or easy items:
  - Anchored Self-Play mixes 20% real bugs.
  - About 30% rephrased synthetic data is best in pretraining.
  - rStar-Coder scored 57.3 with seed+synthetic data vs 49.7 seed-only and 46.8 synthetic-only.
  - ReMind's roughly 2% review budget added +3.81 over GRPO.
  - MiMo-7B samples an easy pool 10% of the time.
  - Keeping moderately easy items cut AIME25 output length by about 56%, with pass@1 moving 73.33 → 70.00.
- **Small pilots cannot see small effects.** At p ≈ 0.5, an unpaired comparison on 30 items needs about a 25-point gap to reach p < 0.05 (derived from the p(1−p)/N rule; about 36 points for 80% power). Seed SD on AIME/AMC is 5–15 points. Hardware, batch size and BF16 alone move accuracy by up to 9%. Use paired per-item analysis with many samples per item, add prompts before adding samples, cluster SEs by seed family, and pre-register a domain-balanced aggregate. Swapping a math-heavy aggregate for a domain-balanced one gave a ranking correlation of ρ = −0.33.
- **Decision protocol.** Screen with short runs and early-curve fits (ScaleRL's sigmoid fits reproduce the asymptote within ±0.02 across 3 runs). Adopt a data policy only if its paired 95% CI excludes zero across 8–12 matched seeds, no metric in the regression suite drops, and the gain holds on post-cutoff items, on a held-out generator family and on a non-Qwen model.
- **Audit the generator before training.** Preference leakage is 23.6% when the judge is the generator but 2.8% across model series. A five-way LLM-ID classifier reaches 97.1% accuracy, and students fine-tuned on two teachers' outputs are 98.9% separable. Traits pass through filtered data, including correct-filtered CoT, when teacher and student share a base model. Cross-model output similarity is already 0.71–0.82. LLM MCQ generators put the answer first 47.1–57.9% of the time, and LLM-written NLI is 86–96% solvable from the hypothesis alone.
- **Verifier and benchmark validity degrade as tasks get harder.** RLVR models game extensional verifiers far more on hard items (40 shortcuts in complexity levels 1–10 vs 458 in levels 11–20). Agentic benchmarks misestimate performance by up to 100% in relative terms; for example, an empty-response agent passes 38% of τ-bench-Airline. Hardened tasks need invariance-checked verifiers, or the measured gains are fake.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| 1-shot RLVR (+ Beyond Variance) | 2025 / 2026 | [2504.20571](https://arxiv.org/abs/2504.20571), [2602.03452](https://arxiv.org/abs/2602.03452) | Math RLVR | RL (dose floor) | Extreme dose reduction; variance or success-band prompt selection; duplication; +/− pairing with weighted GRPO | Ground-truth numeric label + exact-answer checker; label-error ablation |
| LIMR (Learning Impact Measurement) | 2025 | [2502.11886](https://arxiv.org/abs/2502.11886) | Math RLVR | RL (selection) | Training-dynamics subset selection (alignment score s_i > 0.6) | Existing MATH labels |
| Synthetic Data RL | 2025 | [2505.17063](https://arxiv.org/abs/2505.17063) | Math, science, medical, law, finance | RL | Harder-from-solved / easier-from-unsolved rewriting; lowest-positive-pass-rate selection | Instructor (GPT-4o) majority vote; no independent verifier |
| Prompt Replay | 2026 | [2603.21177](https://arxiv.org/abs/2603.21177) | Math RLVR | RL (online selection) | Pass-rate-band buffer [0.25, 0.75], priority near 0.5, cooldown and reuse cap | Unchanged labels; fresh on-policy rollouts |
| Learning from Less | 2026 | [2604.18381](https://arxiv.org/abs/2604.18381) | Procedural counting, graph, spatial | RL (dose/mix study) | Complexity knobs (depth 1–7, 5–25 nodes, moves); Easy vs Mixed tiers; 100/200/500 ladder | Generator-computed answers |
| RL post-training scaling behaviors | 2025 | [2509.25300](https://arxiv.org/abs/2509.25300) | Math RL, Qwen2.5 0.5–72B (+ Llama 3 fits) | Analysis | Reuse factor τ at fixed total volume | Verified math data; difficulty-matched subsets |
| ScaleRL *(added)* | 2025 | [2510.13786](https://arxiv.org/abs/2510.13786) | Math (+code) RL at scale | Analysis / decision protocol | Sigmoid compute-curve fitting; zero-variance filter; No-Positive-Resampling (retire p ≥ 0.9) | Verifiable rewards; held-out 1,000-prompt iid validation |
| Synthetic:natural ratio scaling | 2025 | [2510.01631](https://arxiv.org/abs/2510.01631) | Pretraining mixtures | Pretraining | Synthetic:natural ratio; rephrase vs textbook generation; generator size | n/a (loss); grounded rephrasing vs ungrounded generation |
| RegMix *(added)* | 2024 | [2407.01492](https://arxiv.org/abs/2407.01492) | Pretraining domain mixtures | Mixture search | Dirichlet-sampled proxy mixtures + regression surrogate | n/a (loss/downstream) |
| MoDoMoDo | 2025 | [2505.24871](https://arxiv.org/abs/2505.24871) | Multimodal RLVR | RL (mixture) | Quadratic surrogate over dataset weights; single / exclude-one / all seed mixtures | Per-dataset rule rewards (IoU, MC accuracy, format) |
| DataFlex-RL | 2026 | [2609.06107](https://arxiv.org/abs/2609.06107) | Math/logic/science RLVR | Eval platform | Rollout selection, reweighting, adaptive vs fixed mixtures; 12-seed paired tests | Rule verifiers; calibration smoke test; 13-gram decontamination audit |
| Consolidating RLVR (Merge / Mix RL / MOPD) | 2026 | [2608.27409](https://arxiv.org/abs/2608.27409) | Math, science, code, IF, agent | RL fusion | Task-vector merge; pooled Mix RL (25/22/22/19/12%); multi-teacher OPD | Per-domain verifiable rewards; RL experts as teachers |
| ReMind (+ Frugal Reasoning) | 2026 / 2025 | [2606.03087](https://arxiv.org/abs/2606.03087), [2511.01937](https://arxiv.org/abs/2511.01937) | Image/video/text RLVR; math RLVR | RL (retention) | Mastered-prompt review queue; pre-rollout batch replacement; retaining moderately easy items | Same verifiers; no new labels |
| Retaining by Doing (+ Transferability Index) | 2025 | [2510.18874](https://arxiv.org/abs/2510.18874), [2507.00432](https://arxiv.org/abs/2507.00432) | Forgetting / transfer | Analysis | SFT vs RL on identical data; on-policy vs off-policy data | Verified target tasks; regression suites |
| Fidelity–Diversity metrics | 2026 | [2607.04563](https://arxiv.org/abs/2607.04563) | Synthetic data auditing (M2D2, GSM8K-style) | Analysis (SFT) | Seeds × rewrites-per-seed trade-off at fixed size | Solutions by GPT-5.2; invalid answers repaired by GPT-5.4; metrics audit the distribution |
| Route-diverse SFT trace selection | 2026 | [2609.33780](https://arxiv.org/abs/2609.33780) | Puzzles, math; SFT→RL | SFT+RL | Rule-based route fingerprint; diversity-maximizing selection | Only verified solutions are candidates |
| A Sober Look (+ numerical nondeterminism) | 2025 | [2504.07086](https://arxiv.org/abs/2504.07086), [2506.09501](https://arxiv.org/abs/2506.09501) | Math reasoning eval | Eval protocol | Multi-seed eval; paired per-problem tests; fixed stack; FP32/batch-invariant inference | n/a |
| Measuring all the noises | 2025 | [2512.21326](https://arxiv.org/abs/2512.21326) | LLM eval statistics | Eval protocol | Prediction / data / total noise; all-pairs paired analysis | n/a |
| Adding Error Bars (+ Quantifying Variance) | 2024 | [2411.00640](https://arxiv.org/abs/2411.00640), [2406.10229](https://arxiv.org/abs/2406.10229) | LLM eval statistics | Eval protocol | CLT and clustered SEs; answer resampling; paired differences; power analysis | n/a |
| Signal and Noise | 2025 | [2508.13144](https://arxiv.org/abs/2508.13144) | Benchmark reliability for small-scale decisions | Eval protocol | SNR; noisy-subtask filtering; checkpoint averaging; BPB metrics | n/a |
| Don't Pass@k (+ Kernel crossovers, HiBayES) | 2025 / 2026 | [2510.04265](https://arxiv.org/abs/2510.04265), [2609.22547](https://arxiv.org/abs/2609.22547), [2505.05602](https://arxiv.org/abs/2505.05602) | Small-set reasoning eval | Eval protocol | Dirichlet posterior; credible-interval decision rule; paired pass@k bands; hierarchical GLMs | n/a |
| MathArena | 2025 | [2505.23281](https://arxiv.org/abs/2505.23281) | Math competitions, proofs | Held-out eval | Post-release time split | Official answers; human proof grading |
| Agentic Benchmark Checklist (ABC) *(added)* | 2025 | [2507.02825](https://arxiv.org/abs/2507.02825) | Agentic benchmarks | Eval validity | Outcome-validity and task-validity checks; trivial-agent baselines | Checklist-driven fixes; expert confirmation |
| Isomorphic Perturbation Testing (IPT) *(added)* | 2026 | [2604.15149](https://arxiv.org/abs/2604.15149) | Inductive rule learning RLVR | Verifier audit / reward | Logically isomorphic task variants; dual extensional + isomorphic verification | Program-constructed isomorphisms |
| Label-error audits (IRT indicator + MMLU-Redux + Platinum + test-oracle protocol) | 2024–2026 | [2605.30504](https://arxiv.org/abs/2605.30504), [2406.04127](https://arxiv.org/abs/2406.04127), [2502.03461](https://arxiv.org/abs/2502.03461), [2607.13707](https://arxiv.org/abs/2607.13707) | Benchmark and synthetic-label integrity | Analysis | IRT mislabel flagging; stratified re-annotation; gold-perturbation negatives | Manual adjudication; item-level string oracle |
| Preference Leakage (+ self-preference) | 2025 | [2502.01534](https://arxiv.org/abs/2502.01534), [2404.13076](https://arxiv.org/abs/2404.13076) | LLM-as-judge | Analysis | Generator–judge relatedness audit; PLS | n/a |
| Unveiling the Flaws (+ What Has Been Lost) | 2024 / 2025 | [2406.12397](https://arxiv.org/abs/2406.12397), [2505.22830](https://arxiv.org/abs/2505.22830) | Synthetic Q-A in CPT/SFT; synthetic benchmarks | Analysis | Distribution diagnostics; unlearning of format patterns; synthetic-vs-human difficulty | Human validity annotation (Gill) |
| MCQ answer-position bias (+ Bad Dice) | 2026 | [2605.01846](https://arxiv.org/abs/2605.01846), [2601.05414](https://arxiv.org/abs/2601.05414) | Synthetic MCQ / attribute-constrained generation | Analysis | Answer-marginal audit; external RNG; probing and steering | n/a |
| Idiosyncrasies in LLMs (+ linguistic features) | 2025 / 2026 | [2502.12150](https://arxiv.org/abs/2502.12150), [2606.04177](https://arxiv.org/abs/2606.04177) | Generator fingerprinting | Analysis | Generator-ID classifier; paraphrase/translate robustness | n/a |
| Subliminal Learning | 2025 (Nature 2026) | [2507.14805](https://arxiv.org/abs/2507.14805) | Distillation safety | Analysis | Teacher–student base-model matching audit | n/a (filtering shown insufficient) |
| Artificial Hivemind (+ NCD generative-process diversity) | 2025 / 2026 | [2510.22954](https://arxiv.org/abs/2510.22954), [2609.03422](https://arxiv.org/abs/2609.03422) | Output diversity across models | Analysis | Intra/inter-model similarity audit; compression-distance diversity | n/a |
| Partial-input and training-dynamics baselines | 2018–2024 | [1803.02324](https://arxiv.org/abs/1803.02324), [2410.08996](https://arxiv.org/abs/2410.08996), [2402.12483](https://arxiv.org/abs/2402.12483), [2009.10795](https://arxiv.org/abs/2009.10795) | Dataset shortcut audits | Analysis | Hypothesis-/question-only, choices-only, answer-prior baselines; data maps | Flagged items rewritten or sent to label audit |

## Method notes

**A. Dose: how many hard items, and how often to reuse them**

### 1-shot RLVR (+ Beyond Variance pairing) — Reinforcement Learning for Reasoning in Large Language Models with One Training Example (Yiping Wang et al., 2025)
Link: https://arxiv.org/abs/2504.20571 (NeurIPS 2025; code github.com/ypwang61/One-Shot-RLVR). Related: Pang et al., 2026, *Beyond Variance: Prompt-Efficient RLVR via Rare-Event Amplification and Bidirectional Pairing*, https://arxiv.org/abs/2602.03452.
- **Mechanism**: No new tasks; this sets the dose floor.
  - Examples in the 1,209-prompt DeepScaleR subset (DSR-sub) are ranked by the variance of their per-epoch training accuracy in a prior full-data run ("historical variance").
  - The chosen example is duplicated to 128 copies to fill verl's batch, with 8 rollouts each, so it is sampled 1,024 times per step. Training uses GRPO or PPO.
  - The policy-gradient loss drives the gain, which the authors distinguish from weight-decay-driven grokking. An entropy bonus adds more.
  - Beyond Variance probes two candidate pools with the current policy: an easy pool (DSR-sub) and a hard pool (AIME 2025).
    - It pairs a hard-but-solvable prompt q+ with an easy-but-brittle q−, where p(q−) ∈ [1 − c/G, 1 − 1/G].
    - Its Weighted GRPO maps outcomes to +1 or −λ_neg before group normalization, so rare successes on q+ and rare failures on q− both get large advantages.
    - Selection happens once, before training.
- **How it makes tasks harder**: It doesn't. It tells you how little hardened data is needed before diversity, rather than volume, becomes the constraint.
- **Correctness / verification**: A numeric label is checked with an exact-answer checker. The label ablation on Qwen2.5-Math-1.5B, MATH500 (same loss configuration):
  - Precise label 12.7: 73.4. Slightly imprecise 12.8: 74.8.
  - Unguessable wrong label 929725: 64.4. Guessable wrong label 4: 57.0.
  - A plausible wrong label is the worst case. π1207, which has a wrong label, fails to train.
- **Difficulty control**: The best example (π1) is one the base model nearly solves: 57.8% of 128 base samples already output 12.7 or 12.70. The "too difficult" π1208, whose ground truth is never sampled, reaches only 45.0% on MATH500. π1207 and π1208 are the only examples that fail to give a ≥30% gain.
- **Reported results**:
  - Qwen2.5-Math-1.5B: MATH500 36.0 → 73.6 and the 6-benchmark average 17.6 → 35.7. This matches DSR-sub (73.6 / 35.9). Two examples reach 74.8 / 36.6.
  - GRPO, MATH500, 1-shot vs 1,209-example DSR-sub: Qwen2.5-Math-7B 79.2 vs 78.6; Llama-3.2-3B-Instruct 45.8 vs 43.2; DeepSeek-R1-Distill-Qwen-1.5B (32k eval) 83.9 vs 84.5.
  - Training accuracy saturates before step 100 while test accuracy keeps rising ("post-saturation generalization").
  - The training example overfits into multilingual gibberish after about 1.4k steps (π1) or 1.8k steps (π13), yet test outputs stay interpretable.
  - Beyond Variance (Qwen2.5-Math-7B, 2 prompts): AIME25 pass@8 16.8 → 22.2 and AMC23 pass@64 94.0 → 97.0 vs the two highest-variance prompts. The AIME25 training prompt was removed from the eval, which leaves 29 problems; max generation length was 3,072 tokens.
- **Limitations / failure modes**:
  - The headline model, Qwen2.5-Math, is prone to spurious rewards (Spurious Rewards, note 10; Prompt Replay below).
  - Part of the gain is format correction (the abstract counts 8.6 points beyond format correction).
  - Variance ranking needs a full-data run.
  - The evaluations are tiny: one AIME item is worth 3.3 points.
  - *(Corrected: the researcher's "~1.8k training steps" and "~1.5k more steps of test improvement" are not stated as such. The paper reports saturation before step 100 and overfitting after 1.4k/1.8k steps. The "58% solve rate" refers to the base model's final-answer rate on π1.)*
- **How to reuse with easy seed tasks**: Before building a generator, run a dose ladder of 1 / 16 / 128 / 1k / 10k hardened prompts on your own model, with a random-reward arm and a non-Qwen model. If 16–128 prompts recover most of the gain, spend the budget on seed breadth and held-out coverage. For tiny budgets, pair one low-success hardened item with one high-but-imperfect-success item rather than picking by variance.

### LIMR — LIMR: Less is More for RL Scaling (Xuefeng Li et al., 2025)
Link: https://arxiv.org/abs/2502.11886 (code/data github.com/GAIR-NLP/LIMR)
- **Mechanism**:
  - Run RL on the full pool, log each sample's per-epoch reward r_i^k, and compute the average curve r_avg^k.
  - Score each sample by how closely its reward curve tracks the average learning curve: s_i = 1 − Σ_k (r_i^k − r_avg^k)² / Σ_k (1 − r_avg^k)².
  - Keep samples with s_i > θ. θ = 0.6 gives 1,389 samples.
- **How it makes tasks harder**: Indirectly. The score drops samples that stay at zero reward or saturate early, and keeps those whose rewards rise with the model.
- **Correctness / verification**: Existing MATH labels (levels 3–5), unchanged.
- **Difficulty control**: The threshold θ sets the dose.
- **Reported results**: Qwen2.5-Math-7B, scores for AIME24 / MATH500 / AMC23 (average):

  | Data | Samples | AIME24 / MATH500 / AMC23 (avg) |
  |---|---|---|
  | LIMR | 1,389 | 32.5 / 78.0 / 63.8 (58.1) |
  | Full MATH | 8,523 | 32.5 / 76.6 / 61.9 (57.0) |
  | Random subset | 1,389 | 25.8 / 66.0 / 56.3 (49.4) |
  | Linear-progress baseline | 1,138 | (54.9) |
  | LIMO SFT (7B) | 817 | (45.7) |
  | s1 SFT (7B) | 1,000 | (38.0) |
- **Limitations / failure modes**:
  - Scoring requires a full RL run.
  - Scores are specific to one model and checkpoint.
  - Only one domain and a spurious-reward-prone base model were tested.
- **How to reuse with easy seed tasks**: After a pilot on a complexified pool, compute LIM scores per item and aggregate them per operator (depth+k, obfuscation, distractors). Operators whose items rarely pass θ produce unlearnable or trivial items, so down-weight them. Keep the top-LIM core set (about 16% here) as a replay anchor.

### Synthetic Data RL — Synthetic Data RL: Task Definition Is All You Need (Yiduo Guo et al., 2025)
Link: https://arxiv.org/abs/2505.17063 (code github.com/gydpku/Data_Synthesis_RL)
- **Mechanism**:
  - An instructor (GPT-4o) extracts keywords from the task definition, retrieves passages, abstracts a pattern and generates N = 500 initial Q-A pairs.
  - The base model (Qwen2.5-7B) attempts each one. A rewriter turns solved items into harder variants and unsolved items into easier ones, giving S_synth = initial ∪ harder ∪ easier.
  - Each item is scored by its pass rate over L samples. Never-solved items get score 1, so they are ranked last.
  - The M = 500 items with the lowest positive pass rate are kept for GRPO.
- **How it makes tasks harder**: Solvability-conditioned rewriting, then selection of items that are rarely but sometimes solved.
- **Correctness / verification**: Consensus labels from multiple instructor samples (majority vote). There is no independent verifier.
- **Difficulty control**: The rewriting direction depends on solvability; selection takes the lowest non-zero pass rates.
- **Reported results**:
  - Absolute gains over Qwen2.5-7B: GSM8K +29.2, MATH +8.7, GPQA +13.1, MedQA +8.9, CQA (law) +17.7, CFA +13.7.
  - GSM8K: 91.7 vs 92.1 for RL on the whole human training set. 100 human demonstrations add only +0.4.
  - Dose on GSM8K: M = 100 85.5, 300 89.5, 500 91.7, 1,000 91.8.
  - Human-data RL at the same budget: 82.5 (M = 100), 89.5 (300), 91.7 (1,000).
  - Ablations at M = 500 (GSM8K): no difficulty adaptation 89.1; easy-only 90.5; hard-only 90.7; all synthetic items unselected 89.9.
- **Limitations / failure modes**:
  - Majority-vote labels become unreliable exactly where items become hard.
  - Benchmarks are close to the task definitions.
  - The gains are 1–3 points above ablations with 3-run SDs of 0.1–0.5.
- **How to reuse with easy seed tasks**: A minimal seed-to-hard loop: harden solved seeds, soften unsolved ones, keep the rarely-solved band, and stop at a few hundred to a thousand items per task. Swap in a programmatic verifier wherever one exists.

### Prompt Replay — Prompt replay: speeding up GRPO with on-policy reuse of high-signal prompts (Andrei Baroian et al., 2026)
Link: https://arxiv.org/abs/2603.21177
- **Mechanism**:
  - After each GRPO step, prompts with pass rate in [p_min, p_max] = [0.25, 0.75] enter a buffer, prioritized by closeness to 0.5.
  - Batches mix fresh prompts with up to ε = 0.75 replayed prompts. Each prompt has a cooldown of C = 10 steps and a maximum of R = 15 reuses.
  - Only prompts are reused, never trajectories, so updates stay on-policy.
  - Main runs: 32 prompts × 16 rollouts = 512 rollouts per step.
- **How it makes tasks harder**: Items don't change. The training stream gets harder because trivial and hopeless prompts stop taking rollouts.
- **Correctness / verification**: Same labels, with fresh verifier calls on every replay.
- **Difficulty control**: The pass-rate band plus priority by |p − 0.5|.
- **Reported results**:
  - On Llama-3.2-3B and Qwen3-8B (Dolci and Polaris data), replay reduces zero-variance prompts, raises mean absolute advantage and speeds early accuracy gains. Training then plateaus and converges with the baseline.
  - Cooldown values of 2, 5, 10 and 20 performed alike.
  - On Qwen3-8B/Polaris, doubling rollout GPUs removed the rollout bottleneck, and with it the speed benefit.
  - On Qwen2.5-Math-1.5B, training on the same 32 prompts at every step performed "similarly or better" than the 13k-prompt baseline. The authors attribute this to the Qwen2.5 spurious-reward effect reported by Shao et al. and declare their Qwen ablations invalid.
  - *(Corrected: the random-reward observation is Shao et al.'s. Prompt Replay's own finding is the 32-identical-prompt anomaly.)*
- **Limitations / failure modes**: A compute-efficiency gain, not a higher ceiling. It helps most when rollouts are the bottleneck and the data is hard for the model.
- **How to reuse with easy seed tasks**: Stretch a small hardened set by replaying mid-band prompts: ε ≤ 0.75, a cooldown and a reuse cap. Validate every data-policy ablation on a non-Qwen2.5 model.

### Learning from Less — Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes (Justin Bauer et al., 2026)
Link: https://arxiv.org/abs/2604.18381
- **Mechanism**: Three procedural generators with explicit knobs:
  - Counting: number range; operator family (conditional, threshold, arithmetic aggregation, extrema, bitwise); 1–4 conditional filters plus 0–3 transformations, i.e., 1–7 intermediate steps.
  - Graph: 5–25 nodes and several operator families.
  - Spatial: moves, rotations and query type.
  - Qwen3-4B with LoRA (r = 64, α = 16) is trained on 100, 200 or 500 items from the Easy tier only or from a Mixed-tier set. Counting and graph runs use 300 steps; spatial runs use 1,000.
- **How it makes tasks harder**: Parametric depth, size and operator knobs.
- **Correctness / verification**: The generators compute every answer.
- **Difficulty control**: Explicit parameters, Easy vs Mixed training tiers, and tier-stratified evaluation.
- **Reported results**: Mean test accuracy (%), with the base model at 31.3 (counting), 29.4 (graph) and 26.1 (spatial):

  | Task | Easy-100 | Easy-200 | Easy-500 | Mixed-100 | Mixed-200 | Mixed-500 |
  |---|---|---|---|---|---|---|
  | Counting | 21.9 | 40.0 | 44.2 | 44.2 | 43.4 | 35.5 |
  | Graph | 33.3 | 32.9 | 36.5 | 29.1 | 32.7 | 34.0 |
  | Spatial | 49.9 | 56.7 | 53.1 | 56.6 | 54.3 | 55.7 |

  The authors report "up to 5×" sample efficiency for mixed complexity.
- **Limitations / failure modes**:
  - One 4B model with LoRA and small test sets.
  - Counting Easy-100 collapsed after about step 150–200, finishing below the base model. That collapse inflates the "5×" figure.
  - Graph results are confounded by answer-extraction failures (59–73% of rollouts) under the token cap.
  - The dose response is non-monotone.
  - *(Corrected: the researcher's "graph did not generalize" is better stated as "graph gains were modest and confounded by extraction failures".)*
- **How to reuse with easy seed tasks**: Harden seeds with a knob and train on a smooth spread of tiers. Run 100 / 200 / 500 dose points per operator, with at least 3 seeds each, because a single run can collapse.

### RL post-training scaling behaviors — Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning (Zelin Tan et al., 2025)
Link: https://arxiv.org/abs/2509.25300 (ACL 2026 main, per arXiv v4 comment)
- **Mechanism**:
  - 63 RL runs on more than 50k math problems across Qwen2.5 0.5B–72B.
  - Fits log L = −k(N)·log X + E(N), where L is 1 − pass rate and X is compute or data, with k(N) = K_max / (1 + N₀/N).
  - Data-reuse experiment: D_unique × τ = D_total is held fixed. Subsets preserve the difficulty distribution and the curriculum ordering is kept.
  - v4 adds a Llama 3 (1B–70B) cross-architecture fit with R² > 0.99.
- **How it makes tasks harder**: Not a generator. It measures uniqueness vs repetition.
- **Correctness / verification**: Standard verified math data.
- **Difficulty control**: The difficulty distribution is matched across reuse conditions.
- **Reported results**:
  - Final test loss is insensitive to τ, with no significant degradation for τ ≤ 25. At τ = 100 there are clear signs of overfitting.
  - Performance is governed mainly by total optimization steps.
  - Larger models are more compute- and sample-efficient, but k(N) saturates.
  - Llama-70B peaks around 50% holdout accuracy vs about 59% for Qwen-72B.
- **Limitations / failure modes**:
  - Math only.
  - The reuse experiment is on Qwen2.5.
  - Loss is measured on an in-distribution holdout.
  - *(Corrected: the researcher's "Qwen2.5 family only" is outdated. v4 includes Llama 3 for the compute and data laws.)*
- **How to reuse with easy seed tasks**: Size the unique pool from the RL budget, e.g., about S·B/25 unique verified items (derived). Put the remaining budget into diversity and held-out families. Watch for degeneration if a small pool is cycled more than about 25 times.

### ScaleRL *(added)* — The Art of Scaling Reinforcement Learning Compute for LLMs (Devvrit Khatri et al., 2025)
Link: https://arxiv.org/abs/2510.13786 (Meta; more than 400,000 GPU-hours). Its filtering components are covered in note 10; this entry covers its use as a decision protocol.
- **Mechanism**:
  - Fits a sigmoid to held-out pass rate (mean@16 on 1,000 iid validation prompts, every 100 steps) vs compute: R_C − R₀ = (A − R₀) / (1 + (C_mid/C)^B).
  - A is the asymptote, B is the efficiency and C_mid is the half-gain compute. Fits start after about 1.5k GPU-hours.
  - Recipe changes are compared by their fitted (A, B), not by an early snapshot.
  - The final recipe was validated by leave-one-out ablations (16k GPU-hours each) and then extrapolated from 50k to 100k GPU-hours (8B dense) and from 16k to 45k (Llama-4 17B×16 MoE).
- **How it makes tasks harder**: Two dose-relevant components of the recipe:
  - Zero-variance filtering: drop all-0 or all-1 groups from the loss.
  - No-Positive-Resampling: permanently retire prompts once their historical pass rate is ≥ 0.9.
  - Both raise the asymptote.
- **Correctness / verification**: Verifiable rewards; iid held-out validation; AIME-24 as a downstream check.
- **Difficulty control**: Pass-rate-based retirement.
- **Reported results**:
  - Across 3 independent runs, the fitted asymptote A varies by ±0.02.
  - Loss aggregation, normalization, curriculum and the off-policy algorithm mainly change B, not A.
  - "Methods that appear superior at small compute budgets can be worse when extrapolated."
  - At a fixed total batch, sweeping generations per prompt over 8/16/24/32 left the fitted curves essentially unchanged.
  - Larger batches (2,048 prompts) looked worse early but won later.
- **Limitations / failure modes**:
  - A single-run fit carries ±0.02 uncertainty on A, so differences in asymptote smaller than that are not resolved.
  - Fits need to exclude the early regime.
  - Mostly math, with some code.
- **How to reuse with easy seed tasks**: Compare hardened-data arms by their fitted curves, not by accuracy at step N. Any claimed ceiling gain must exceed the ±0.02 run-to-run spread of A. Retire mastered prompts (p ≥ 0.9), but pair this with a review queue (ReMind) so retired skills don't decay.

**B. Mixing: synthetic:real blends, domain mixtures and consolidation**

### Synthetic:natural ratio scaling — Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls (Feiyang Kang et al., 2025)
Link: https://arxiv.org/abs/2510.01631 (EMNLP 2025 main)
- **Mechanism**: More than 1,000 models (over 100k GPU-hours) trained under one protocol. It compares natural web text, rephrased synthetic text, textbook-style generated text and their mixtures, and fits scaling laws per mixture.
- **How it makes tasks harder**: n/a. The levers are ratio, synthetic type and generator size.
- **Correctness / verification**: n/a (validation loss). The key distinction is grounded rephrasing vs ungrounded generation.
- **Difficulty control**: n/a.
- **Reported results**:
  - Rephrased synthetic data alone is not faster than natural text.
  - A 1/3 rephrased + 2/3 natural mix reaches the same validation loss 5–10× faster at larger budgets.
  - Textbook-style data alone gives notably higher loss on many downstream domains and shows collapse-like patterns.
  - Good rephrased ratios converge to about 30%.
  - Generators larger than about 8B do not necessarily give better data.
- **Limitations / failure modes**: Pretraining loss, not post-training RL; single-round (n = 1) synthetic use.
- **How to reuse with easy seed tasks**: A prior for SFT and midtraining blends: about 30% when hardened items are rewrites of real seeds, and less when they are generated from scratch. Don't assume a bigger generator gives better training data.

### RegMix *(added)* — RegMix: Data Mixture as Regression for Language Model Pre-training (Qian Liu et al., 2024)
Link: https://arxiv.org/abs/2407.01492 (ICLR 2025; code github.com/sail-sg/regmix)
- **Mechanism**:
  - Train many small proxy models on mixtures sampled from a Dirichlet distribution whose α is the token distribution scaled by 0.1–5.0, which covers extreme weights.
  - Fit a regressor (ridge or LightGBM) from mixture weights to a target metric, judged by Spearman ρ and MSE on unseen mixtures.
  - Train the large model on the predicted-best mixture.
  - The study used 17 available Pile domains.
- **How it makes tasks harder**: n/a. This is the mixture-search methodology that MoDoMoDo-style surrogates inherit.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - 512 models of 1M parameters on 1B tokens each fitted the regressor.
  - The predicted mixture, used to train a 1B model on 25B tokens (1,000× larger and 25× longer), ranked best among 64 candidate 1B mixtures.
  - It beat human selection up to 7B / 100B tokens and matched or exceeded DoReMi at 10% of the compute.
  - Web corpora correlated with downstream performance more strongly than "high-quality" sources such as Wikipedia, and domains interact in counter-intuitive ways.
- **Limitations / failure modes**:
  - Pretraining only.
  - Proxy rankings must transfer across scale, which it verifies there but which is untested for RL post-training, where DataFlex-RL shows method signs flipping across scale.
- **How to reuse with easy seed tasks**: Treat operator families as "domains". Sample mixtures from a Dirichlet, run short, cheap RL or SFT proxies, fit a regressor, then confirm the winner against an equal mix with matched seeds.

### MoDoMoDo — MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning (Yiqing Liang et al., 2025)
Link: https://arxiv.org/abs/2505.24871
- **Mechanism**:
  - Five verifiable vision-language datasets: COCO and LISA (IoU), GeoQAV and ScienceQA (MC accuracy), and SAT (spatial).
  - Seed mixtures: Single, Exclude-One and All (uniform).
  - A multivariate quadratic surrogate predicts post-RL test performance from the weights w. It fits better than linear models under cross-validation.
  - The surrogate's maximizer on the simplex is trained with GRPO on Qwen2-VL-2B-Instruct.
- **How it makes tasks harder**: n/a. It controls which skills get gradient, and the quadratic term captures interference.
- **Correctness / verification**: Per-dataset rule rewards plus a format reward.
- **Difficulty control**: n/a.
- **Reported results**:
  - The best predicted mixture improves out-of-distribution accuracy by 5.24% on average over uniform, and by 20.74% over the pre-RL model.
  - The All mixture underperforms two single-dataset mixtures on ChartQA and InfoVQA.
  - LISA-only helps InfoVQA but hurts ScienceQA, while ScienceQA-only is best in-domain but weak on InfoVQA.
- **Limitations / failure modes**: A 2B VLM on a 4×A100 budget. It was not re-tested under the multi-seed paired protocol that DataFlex-RL shows is needed.
- **How to reuse with easy seed tasks**: Treat each operator family (depth-scaled, obfuscated, distractor-injected, real anchors) as a dataset. Run 10–20 short pilots, fit the quadratic, and train w*. Adopt it only if it beats a fixed equal mix with 8–12 matched seeds.

### DataFlex-RL — DataFlex-RL: An Evaluation Platform for RLVR Data Policies (Hao Liang et al., 2026)
Link: https://arxiv.org/abs/2609.06107
- **Mechanism**:
  - One GRPO stack (verl v0.5+, 5 rollouts per prompt, KL 1e-3, 1,024/8,192-token prompt/response limits, 300 optimizer steps) in which only the data policy varies.
  - Data: math; Knights & Knaves logic puzzles with an assignment checker; SciQ science with exact letter matching.
  - 13 configurations, each with 12 matched seeds on Qwen2.5-7B-base: selection (difffilter, maxvar, gfpo, topk), reweighting, adaptive mixtures (e.g., DUMP-style, Teacher–Student Curriculum Learning) and controls.
  - Paired 95% CIs vs uniform sampling or a fixed equal mix. The full matrix is 591 runs.
- **How it makes tasks harder**: n/a. It tests the difficulty filters and curricula that hardening pipelines rely on.
- **Correctness / verification**: Rule verifiers, aligned train and eval formats, a calibration smoke test, and a 13-gram decontamination audit (no exact or normalized matches).
- **Difficulty control**: The policies under test include difficulty filters and adaptive curricula.
- **Reported results**:
  - Uniform GRPO vs the untrained checkpoint: +7.76 [7.28, 8.25] on Qwen2.5-7B-base and +10.25 [7.75, 12.33] on Llama-3.1-8B-base.
  - None of the 8 selection or reweighting methods had a paired 95% CI excluding zero vs uniform, and none of the 3 adaptive mixtures beat a fixed equal mix. A 12-seed Llama extension showed no consistent winner.
  - Signs flip across scale: difffilter is −0.83 and −0.79 at 1.5B and 3B but +0.08 and +1.20 at 7B and 14B, and maxvar is negative everywhere.
  - A math-heavy six-benchmark summary vs the domain-balanced 12-benchmark summary: ρ = −0.33, with the method spread changing from 0.90 to 3.31.
- **Limitations / failure modes**: 300-step budgets, few model families, and no data-generation policies tested.
- **How to reuse with easy seed tasks**: Default to a fixed equal mix across generator families, plus zero-variance filtering. Claim a win for any adaptive scheme only with at least 8 matched seeds, paired CIs and a pre-registered domain-balanced aggregate.

### Consolidating RLVR across domains — Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms (Siye Wu et al., 2026)
Link: https://arxiv.org/abs/2608.27409 (Tencent)
- **Mechanism**:
  - Five domain experts (math, science, code, instruction following, agent) are trained with RLVR, as full-parameter and LoRA experts, on Qwen3-4B-Instruct-2507 and Qwen3-8B (non-thinking).
  - They are consolidated three ways: Merge (task vectors), Mix RL (one pooled run of 87,699 examples: 25% math, 22% science, 22% code, 19% IF, 12% agent) and MOPD (multi-teacher on-policy distillation).
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: Per-domain verifiable rewards. MOPD teachers are the RL experts.
- **Difficulty control**: Domain proportions in Mix RL.
- **Reported results**:
  - Average performance differs by at most 1.4 points, and by up to 8.6 on a single benchmark.
  - Cost relative to per-domain RL: Mix RL 0.58× (4B) and 0.67× (8B); MOPD 1.14× and 1.19×; Merge about 0 extra.
  - At AIME pass@32 all three are indistinguishable from the base model.
  - No held-out capability loss on SimpleQA-Verified or AA-LCR.
- **Limitations / failure modes**: Qwen3 only; Mix RL proportions not ablated systematically.
- **How to reuse with easy seed tasks**: With several domain generators, Mix RL is the cheapest single run. Don't expect consolidation to expand coverage; that needs new hard items with p = 0 at the start, or a stronger teacher.

### ReMind (+ Frugal Reasoning) — Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR (Chuanyu Qin et al., 2026)
Link: https://arxiv.org/abs/2606.03087. Related: Bounhar et al., 2025, *Shorter but not Worse: Frugal Reasoning via Easy Samples as Length Regularizers in Math RLVR*, https://arxiv.org/abs/2511.01937.
- **Mechanism**:
  - Prompts that become fully solved enter a review queue with probability p_add = 0.25.
  - Every f = 5 steps, a fraction r = 0.10 of the batch is replaced with review prompts before rollout, so there is no extra rollout cost. Reviews make up about 2% of all prompts.
  - All-correct reviews "graduate" out of the queue; regressed prompts go back to its tail.
  - Frugal deliberately skews the dataset toward moderately easy problems (0 < p < 1) instead of filtering them, which acts as an implicit length regularizer. Its Stage 2 keeps difficulty levels starting where success is below 75%, giving 14.5k samples.
- **How it makes tasks harder**: Counter-operator. It keeps solved and easy items in the stream to protect retention and length.
- **Correctness / verification**: Same verifiers; no new labels.
- **Difficulty control**: r, f and p_add for ReMind; dataset imbalance toward moderately easy items for Frugal.
- **Reported results**:
  - Of 1,275 review records for mastered prompts, 269 (about 21%) contained both correct and incorrect rollouts.
  - Repair cost grows with review delay (the "repair window").
  - Qwen3-VL-8B-Instruct, average over **eight image-text benchmarks**: GRPO 60.25 → 64.06 with ReMind, DAPO 61.02 → 64.46, with the RePO replay baseline at 62.02.
  - Even a 1% review budget avoids GRPO's mid-training plateau; 4% or 10% gives no further gain. At a matched 2%, burst review (r = 0.10, f = 5) beat smooth review (r = 0.02, f = 1) by 0.36.
  - Frugal (42k-token eval budget, 16k training cap): Qwen3-4B-Thinking-2507 AIME25 73.33 → 70.00 (Stage 2), with mean AIME25 length 21,090 → 9,368 tokens (about 56% shorter).
  - *(Corrected: the 60.25 → 64.06 averages are over eight image-text benchmarks, not 20. Frugal's numbers are at a 42k eval budget; 16k was the training cap.)*
- **Limitations / failure modes**: Queue parameters are tuned per setup. Frugal trades about 3 AIME25 points for brevity.
- **How to reuse with easy seed tasks**: Never permanently retire solved seeds. Keep a review queue at 1–2% of prompts and 5–10% easy or mastered items, and log correct-set turnover on the training set as a first-class regression metric.

### Retaining by Doing (+ Transferability Index) — Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting (Howard Chen et al., 2025)
Link: https://arxiv.org/abs/2510.18874 (ICML 2026). Related: Huan et al., 2025, *Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning*, https://arxiv.org/abs/2507.00432.
- **Mechanism**:
  - Chen et al. compare SFT and RL forgetting on instruction following, general knowledge and arithmetic with Llama 3 and Qwen 2.5 models up to 8B.
  - A two-mode mixture model attributes RL's robustness to the mode-seeking behavior of on-policy data, not to the KL term or advantage estimation.
  - Huan et al. define a Transferability Index (TI) and fine-tune Qwen3-14B-Base on the same math-only data with SFT or RL.
- **How it makes tasks harder**: n/a. It measures collateral damage from narrow hardening.
- **Correctness / verification**: Identical training data under different learning rules.
- **Difficulty control**: n/a.
- **Reported results**:
  - RL forgets less at equal or higher target performance.
  - For SFT, data generated only from the initial policy is not enough, but approximately on-policy data regenerated at the start of each epoch can suffice.
  - Huan et al., non-reasoning TI: SFT (think) −104.1, with IFEval 69.2 → 42.3 and HaluEval 35.7 → 2.3. RL: +52.2, with IFEval 70.0.
- **Limitations / failure modes**: Few model families; TI depends on the chosen benchmark groups.
- **How to reuse with easy seed tasks**: Give every hardening run a regression suite for untargeted skills (IFEval, a hallucination benchmark, conversational QA, long context). When distilling hard synthetic traces with SFT, regenerate them near-on-policy each epoch or follow with RL.

### Fidelity–Diversity metrics — Fidelity-Diversity Metrics for Text (Amanda Wang et al., 2026)
Link: https://arxiv.org/abs/2607.04563
- **Mechanism**:
  - Texts are embedded and summarized as discrete measures. Fidelity is the cost of a greedy nearest-neighbor assignment between candidate and reference modes. Diversity is the gap between that greedy cost and the global optimal-transport cost, i.e., missing coverage of reference modes.
  - The two terms decompose the Wasserstein cost.
  - GSM8K test: seed sets are the first {0.1, 0.2, 0.5, 1, 2, 5, 10}% of the training set, with {999, 499, 199, 99, 49, 19, 9} rewrites per seed so that total size is fixed.
  - Questions are generated by GPT-4.1 at temperature 1.5 and solutions by GPT-5.2. Items with non-positive or non-integer answers are repaired by GPT-5.4.
  - Llama-3.2-1B and 3B are fine-tuned on each set.
- **How it makes tasks harder**: n/a. It measures the breadth vs depth of rewriting.
- **Correctness / verification**: Answer-format checks plus model repair. The metrics audit the distribution, not the labels.
- **Difficulty control**: n/a. The controlled variable is seeds × rewrites.
- **Reported results**: The diversity metric detects the deficit of few-seed datasets, and "diversity in these synthetic training datasets directly correlates with downstream accuracy". *(Corrected: GPT-5.4 repairs invalid answers; it does not validate. The researcher's approximate accuracy numbers read from figures could not be confirmed in the text and are omitted.)*
- **Limitations / failure modes**: SFT only, GSM8K-style data, small models; the metrics depend on the embedding model.
- **How to reuse with easy seed tasks**: Cap rewrites per seed (e.g., ≤ 10) and maximize distinct seeds. Before training, compute fidelity and diversity of each hardened batch against a real hard reference set, and reject batches whose diversity falls.

### Route-diverse SFT trace selection — Selecting Diverse SFT Traces Improves Post-RL Generalization (Dylan Zhang et al., 2026)
Link: https://arxiv.org/abs/2609.33780
- **Mechanism**:
  - Each verified solution gets a fixed-length topology "fingerprint" of its reasoning route. Steps are labeled by fixed rules, from step annotations where a domain provides them and from the trace text otherwise, and a fixed random projection shortens the vector.
  - From one pool at one budget, the selector picks diverse rather than similar routes. It runs on CPUs with no model calls.
  - Testbeds: RLVE, OMEGA, Sokoban, program simulation and Dolci-Think.
- **How it makes tasks harder**: Indirectly. Diverse SFT leaves more prompts in the learnable band before RL.
- **Correctness / verification**: Only verified solutions are candidates.
- **Difficulty control**: Indirect, through pre-RL success/failure mixing.
- **Reported results**:
  - Route-diverse SFT raises OLMo3-7B pass@8 by 16.9 points on environments held out from SFT.
  - With a single generator model, diverse selection gains up to 6.2 points of mean pass@8 across 10 math benchmarks.
  - The CPU selector beat more expensive alternatives in every mean post-RL comparison on 3 open corpora.
  - Multi-teacher SFT also improves post-RL coverage.
- **Limitations / failure modes**: Very recent (Sep 2026) and unreplicated; the fingerprints are domain-specific.
- **How to reuse with easy seed tasks**: When distilling hardened tasks before RL, choose traces by route diversity, not by shortest or most common route.

**C. Measurement: eval noise, statistics and decision rules**

### A Sober Look (+ numerical nondeterminism) — A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility (Andreas Hochlehnert et al., 2025)
Link: https://arxiv.org/abs/2504.07086 (COLM 2025). Related: Yuan et al., 2025, *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*, https://arxiv.org/abs/2506.09501; He et al. (Thinking Machines Lab), *Defeating Nondeterminism in LLM Inference*, blog, Sep 10 2025.
- **Mechanism**:
  - Measures each variance source on small math benchmarks: 20 eval seeds for 9 models, temperature, top_p, max tokens, prompt template, framework and hardware.
  - Re-evaluates RL and SFT methods on a standardized stack and uses paired per-problem tests, averaged over seeds.
  - Yuan et al. trace BF16 nondeterminism to non-associative floating-point arithmetic and propose LayerCast: 16-bit weights with FP32 compute.
  - Thinking Machines trace residual nondeterminism to kernels that are not batch-invariant and release batch-invariant ops.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - Pass@1 SD across seeds is 5–15 points, and one question moves AIME24 or AMC23 by 2.5–3.3 points.
  - Temperature changes swing accuracy by up to 15% and top_p by up to 8%.
  - Even with deterministic algorithms, OpenThinker2-7B scored 53.0 ± 4.6 on A100 vs 57.1 ± 5.2 on H100 (AIME24). Consecutive runs of Bespoke-Stratos-7B ranged from 19.3% to 23.0%.
  - Frameworks differ by 1–2 points, with outliers such as S1.1-7B at 22.2 vs 17.7.
  - Their analysis finds 30 seeds optimal; they used 10 for AIME/AMC as a compromise.
  - Most RL gains were modest and prone to overfitting on AIME24, while SFT generalized more consistently.
  - Yuan: under BF16 greedy decoding, DeepSeek-R1-Distill-Qwen-7B varies by up to 9% in accuracy and 9,000 tokens in length across GPU count, GPU type and batch size.
  - Thinking Machines: 1,000 temperature-0 completions from Qwen3-235B gave 80 unique outputs, which diverge from token 103.
- **Limitations / failure modes**: Math-centric; the recommendations are compute-heavy.
- **How to reuse with easy seed tasks**: For decision evals, fix the container, GPU type, batch size and decoding. Use FP32 compute or batch-invariant kernels, 10–30 samples per item, and mean ± SD with paired per-item tests.

### Measuring all the noises — Measuring all the noises of LLM Evals (Sida Wang, 2025)
Link: https://arxiv.org/abs/2512.21326 (interactive data at all-the-noises.github.io)
- **Mechanism**:
  - Splits eval noise into prediction noise (resampling answers to a fixed question), data noise (sampling questions) and total noise, via the law of total variance.
  - The all-pairs paired method runs paired analysis over every model pair, using millions of question-level predictions.
  - A Beta(p, 1 − p) model of per-question accuracy explains the observed Var[A − B] ≈ p_A(1 − p_A).
- **How it makes tasks harder**: n/a. The paper notes harder questions "cannot yet compensate for small sample sizes".
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - Each eval has a characteristic total noise level that holds across model pairs.
  - Paired prediction noise usually exceeds paired data noise, so averaging many samples per question raises power substantially.
  - HumanEval (N = 164, p = 0.5), SE of the difference and the difference needed for p < 0.05: unpaired 6% → 12%; unpaired + averaging 4% → 8%; paired 4% → 8%; paired + averaging 1–2% → 2–4%.
  - SWE-bench-Verified curves from one checkpoint: data SE is about 1/6 of prediction SE, so averaging 36 predictions detects effects about 0.24 as large.
  - *(Corrected: the researcher's "unpaired SE about 4%" is the per-model SE. The SE of the difference is 6%.)*
- **Limitations / failure modes**: Assumes i.i.d. questions (clustered items need Miller's corrections). Averaging can mislead when optimizing directly on the eval.
- **How to reuse with easy seed tasks**: A 30-item pilot at p ≈ 0.5 needs about a 25-point gap unpaired (derived). If the HumanEval 3–6× reduction carries over, paired analysis with averaging brings that to roughly 4–8 points (derived). Use paired per-item analysis with k ≥ 10–30 samples, and hundreds of items for 2–3-point claims.

### Adding Error Bars (+ Quantifying Variance) — Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (Evan Miller, 2024)
Link: https://arxiv.org/abs/2411.00640. Related: Madaan et al., 2024, *Quantifying Variance in Evaluation Benchmarks*, https://arxiv.org/abs/2406.10229.
- **Mechanism**:
  - Treats questions as draws from a super-population. Gives CLT standard errors and clustered SEs for related question groups.
  - Reduces variance by resampling answers or using next-token probabilities (for non-CoT evals), uses paired differences, and gives a sample-size formula for power analysis.
  - Madaan et al. measure seed variance across identically trained 7B models, bootstrap CIs and monotonicity during training.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - Clustered vs naive SE on real Anthropic-model data: DROP 1.34 vs 0.44 (3.05×), RACE-H 1.10×, MGSM 1.88×.
  - Madaan: 7B seed SDs are well below bootstrap 95% CIs (e.g., HumanEval 1.11 vs 3.98).
  - MMLU monotonicity is 0.09 in MC format vs 0.95 in cloze format.
  - IRT-based item analysis "struggle[s] to meaningfully reduce variance".
- **Limitations / failure modes**: Assumes independent clusters; the Madaan estimates are pretraining-oriented.
- **How to reuse with easy seed tasks**: Variants hardened from the same seed form a cluster. Compute SEs clustered by seed, and split train and eval by seed, not by variant. Run the power formula before a pilot.

### Signal and Noise — Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation (David Heineman et al., 2025)
Link: https://arxiv.org/abs/2508.13144 (NeurIPS 2025; AI2; 900K eval results)
- **Mechanism**:
  - Signal is the dispersion of scores across models. Noise is the relative SD over a run's final n checkpoints.
  - Decision accuracy asks whether small-scale rankings hold at large scale.
  - Covers 30 benchmarks and 375 models from 60M to 32B.
  - Three interventions: switch to a continuous metric (bits-per-byte), filter low-SNR subtasks, and average checkpoints.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - SNR correlates with decision accuracy (R = 0.791, R² = 0.626), while signal or noise alone do not.
  - Final-checkpoint noise tracks initialization, data-order and checkpoint noise (R² 0.82, 0.86, 0.95).
  - ARC-Challenge ranges over 1.7% within the final 30 checkpoints of a 1B model.
  - A 1,000-question ARC-Easy subset has higher decision accuracy than MMLU despite 90% fewer instances.
  - The synthetic AutoBencher (33K instances) shows an SNR inflection around 2K instances.
  - BPB raises ARC-Easy SNR from 21.0 to 64.6.
- **Limitations / failure modes**: Pretraining-scale decisions; RL noise is larger and step-dependent.
- **How to reuse with easy seed tasks**: Build pilot suites from high-SNR subsets and average several late checkpoints. Alongside accuracy, track continuous metrics such as mean pass rate over k samples or reference log-probability. Don't expect synthetic eval items beyond a few thousand to add SNR.

### Don't Pass@k (+ Kernel crossovers, HiBayES) — Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation (Mohsen Hariri et al., 2025)
Link: https://arxiv.org/abs/2510.04265 (ICLR 2026; code github.com/mohsenhariri/scorio). Related: Yang & Chen, 2026, *RLVR is a Kernel, Not a Function: Statistical Inference for pass@k Crossovers*, https://arxiv.org/abs/2609.22547; Luettgau et al., 2025, *HiBayES*, https://arxiv.org/abs/2505.05602.
- **Mechanism**:
  - Replaces pass@k and avg@N with a Dirichlet–categorical posterior over outcomes, including rubric grades. This gives closed-form means and credible intervals.
  - Decision rule: declare no winner while intervals overlap. Under a uniform prior, the posterior mean is order-equivalent to avg@N.
  - The Kernel paper builds paired confidence bands across k that require evidence of both an early gain and a later loss. It models base → post-RL per-prompt success as a Markov kernel.
  - HiBayES fits multilevel Bayesian GLMs to nested, low-data evals (fewer than 20 data points per evaluation).
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a. Kernel shows prompts with the same base success rate have reproducibly different post-RL success.
- **Reported results**:
  - Bayes@N reaches Kendall τ > 0.90 against the gold ranking by N = 10 on AIME24/25, HMMT25 and BrUMO25, and τ ≈ 1 by N ≈ 80 (AIME25 plateaus at about 0.95).
  - Mean trials to converge: HMMT 44.2 vs 69.5 for the pass family; BrUMO 27.1 vs 48.5.
  - Kernel: across five public RLVR pairs, no crossover was statistically established in the initial evals. A 32k-token eval on fresh prompts located a reversal with first loss between 11 and 61 samples.
  - Power analysis: more prompts can help more than more answers per prompt.
- **Limitations / failure modes**: Intervals assume the benchmark represents the target; kernel fits need many samples per prompt.
- **How to reuse with easy seed tasks**: Pre-register "the arm wins only if the 95% paired or credible interval on the pre-registered aggregate excludes zero". Spend extra budget on prompts before samples, and test pass@k-coverage claims with bands, not visual crossings.

### MathArena — MathArena: Evaluating LLMs on Uncontaminated Math Competitions (Mislav Balunović et al., 2025)
Link: https://arxiv.org/abs/2505.23281
- **Mechanism**: Evaluates models on recurring competitions as soon as problems are released, which rules out contamination by construction. It is a living benchmark and includes human-graded proof problems.
- **How it makes tasks harder**: n/a (natural difficulty).
- **Correctness / verification**: Official answers; human proof grading.
- **Difficulty control**: Competition choice; for example, CMIMC and the IMO separate top models.
- **Reported results**:
  - More than 50 models evaluated on 162 problems from seven competitions (at the time of the paper).
  - Strong signs of contamination in AIME 2024.
  - Top models score slightly under 40% on IMO 2025.
- **Limitations / failure modes**: 15–30 items per contest, which is high noise (see Sober Look). Time-split code and SWE suites are covered in notes 04 and 14.
- **How to reuse with easy seed tasks**: Anchor held-out evals on contests released after both the student's and the generator's cutoffs, pooled to hundreds of items. Evaluate each pilot on the next unseen window.

### Agentic Benchmark Checklist *(added)* — Establishing Best Practices for Building Rigorous Agentic Benchmarks (Yuxuan Zhu et al., 2025)
Link: https://arxiv.org/abs/2507.02825 (UIUC)
- **Mechanism**: Defines two validity conditions.
  - Outcome validity: the check truly indicates success.
  - Task validity: a task is solvable if and only if the agent has the target capability.
  - These are turned into a checklist (ABC) covering outcome checks, task setup and result reporting. ABC was applied to 10 popular agentic benchmarks and used to revise CVE-Bench.
- **How it makes tasks harder**: Indirectly. Trivial-agent and do-nothing baselines expose tasks that are "hard" only on paper.
- **Correctness / verification**: Checklist-driven fixes, confirmed by domain experts for CVE-Bench.
- **Difficulty control**: n/a.
- **Reported results**:
  - 7 of 10 benchmarks had outcome-validity flaws, 7 had task-validity issues, and all had reporting limitations.
  - Errors can reach up to 100% (relative) over- or under-estimation.
  - In τ-bench-Airline, an empty-response agent passes 38% of tasks and outperforms a GPT-4o agent.
  - An agent can score 100% on SWE-Lancer without resolving tasks.
  - KernelBench overestimates by 31 points (absolute) due to weak fuzzing; WebArena by 5.2% due to string matching.
  - ABC cut CVE-Bench overestimation by 33% (absolute).
- **Limitations / failure modes**: A checklist, not an automated tool; it covers benchmarks, not training environments.
- **How to reuse with easy seed tasks**: Before trusting pass rates on hardened agentic tasks, whether in training or eval, run a no-op agent, an empty-response agent and a random-action agent. Any task they pass is invalid.

### Isomorphic Perturbation Testing *(added)* — LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (Lukas Helff et al., 2026)
Link: https://arxiv.org/abs/2604.15149
- **Mechanism**:
  - On inductive rule-learning tasks (e.g., "trains carrying red cars go east"), each output is scored both extensionally (correct labels on the given instances) and isomorphically (the output must stay correct on a logically isomorphic relabeling of the task).
  - Genuine rules are invariant, while instance enumeration fails the isomorphic check.
  - A controlled test trains two identical models in Olmo 3's RLVR pipeline that differ only in the verifier.
- **How it makes tasks harder**: Isomorphic variants turn memorizable, enumerable targets into ones that require the actual rule.
- **Correctness / verification**: The isomorphism is constructed by program, so the gold rule maps across variants.
- **Difficulty control**: Complexity levels 1–20.
- **Reported results**:
  - Shortcuts appear in RLVR-trained models (GPT-5 family, Olmo 3) but not in non-RLVR models (GPT-4o, GPT-4.5, Ministral).
  - Shortcuts rise with task complexity and inference compute: 70% of gpt-5-mini-high's shortcuts fall in the top complexity quartile, and across all models there are 40 shortcuts in levels 1–10 vs 458 in levels 11–20.
  - Extensional-only verification induces a growing hacking gap during training; isomorphic verification eliminates it.
- **Limitations / failure modes**: Inductive-logic tasks only; requires a formal task representation.
- **How to reuse with easy seed tasks**: When complexifying seeds, add an invariance check (renamed entities, permuted structure) to the reward and the eval. Harder items raise the incentive to exploit verifier gaps, so unaudited accuracy gains on hardened sets overstate learning.

**D. Generator-signature and label audits**

### Label-error audits (IRT indicator + MMLU-Redux + Platinum + test-oracle protocol) — Auditing LLM Benchmarks with Item Response Theory (Sander Land et al., 2026)
Link: https://arxiv.org/abs/2605.30504 (EMNLP 2026). Related: Gema et al., 2024, *Are We Done with MMLU?*, https://arxiv.org/abs/2406.04127; Vendrow et al., 2025, *Do Large Language Model Benchmarks Test Reliability?*, https://arxiv.org/abs/2502.03461; Ballı, 2026, *The Test Oracle Problem in Synthetic LLM-as-Judge Corpora*, https://arxiv.org/abs/2607.13707.
- **Mechanism**:
  - Land & Bikel fit IRT to responses from 114 models and flag items that strong models "fail" in patterns inconsistent with their ability.
  - MMLU-Redux re-annotates 5,700 stratified items with an error taxonomy.
  - Platinum benchmarks revise 15 benchmarks to remove label errors and ambiguity.
  - Ballı shows that LLM-generated negatives carry no item-level oracle, whereas negatives built by deterministic perturbation of a gold answer can be checked for free by string comparison.
- **How it makes tasks harder**: n/a (label integrity).
- **Correctness / verification**: Manual adjudication of flagged items; perturbation-built items carry their own oracle.
- **Difficulty control**: n/a.
- **Reported results**:
  - The IRT indicator reaches 95% precision in its top 200 flags across seven benchmarks, beating a supervised classifier. One frontier reward model agreed with detected mislabels 78% of the time vs 38% for its peers.
  - MMLU-Redux estimates that 6.49% of MMLU questions contain errors, including 57% of the analysed Virology questions.
  - Frontier models still fail simple items on Platinum sets.
  - Ballı: a shared decoding-budget bug produced a 32-point "effect" that replicated from N = 50 to N = 500 and vanished when fixed. Only reading raw generations caught it. An analogous injected fault in a perturbation-built corpus was caught 100% of the time.
- **Limitations / failure modes**: IRT needs a population of models that can solve the items, which is hard beyond frontier level. Manual audits are expensive.
- **How to reuse with easy seed tasks**: Hand-audit 100–200 hardened items per generator family and report Wilson 95% CIs (derived examples below). Rank what to audit first with IRT-style flags from a model zoo. Build negatives and distractors by deterministic gold perturbation, and always read raw generations.

### Preference Leakage (+ self-preference) — Preference Leakage: A Contamination Problem in LLM-as-a-judge (Dawei Li et al., 2025)
Link: https://arxiv.org/abs/2502.01534 (ICLR 2026). Related: Panickssery et al., 2024, *LLM Evaluators Recognize and Favor Their Own Generations*, https://arxiv.org/abs/2404.13076.
- **Mechanism**:
  - Students are fine-tuned on synthetic data from GPT-4o, Gemini-1.5 or LLaMA-3.3. Related judges then rate them on Arena-Hard and AlpacaEval 2.0.
  - The preference leakage score (PLS) measures the judge's excess preference for its related student.
  - Panickssery et al. show self-recognition correlates linearly with self-preference.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - Average PLS by relatedness: same model 23.6% (Arena-Hard 28.7%, AlpacaEval 18.4%); inheritance 19.3% (same instructions) and 22.3% (different instructions); same family, same series 8.9%; same family, different series 2.8%.
  - By learning method: SFT 23.6%, DPO 5.2%, ICL −2.7%.
  - Leakage grows with the share of synthetic data, with no clear safe threshold.
  - The smallest students (LLaMA-3-1B, Qwen-2.5-3B, Qwen-3-1.7B) show the highest PLS.
- **Limitations / failure modes**: Subjective open-ended tasks. For RLVR it matters through rubric or LLM-judge rewards and judge-based evals.
- **How to reuse with easy seed tasks**: Keep generator, judge and reward model in different families and series. Never judge a student with the family that generated its hardened data, and run swapped-family judge controls.

### Unveiling the Flaws (+ What Has Been Lost) — Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models (Jie Chen et al., 2024)
Link: https://arxiv.org/abs/2406.12397 (Baichuan / RUC). Related: Gill et al., 2025, *What Has Been Lost with Synthetic Evaluation?*, https://arxiv.org/abs/2505.22830.
- **Mechanism**:
  - A Llama-like 2B model (1T tokens) is continued-pretrained on 300B tokens, 2% of them synthetic Q-A pairs, then SFT'd.
  - The synthetic signature: non-overlapping embedding clusters, token-frequency peaks on structural tokens, and a shifted, lower-variance perplexity distribution.
  - Mitigation: unlearning on 1B tokens of the synthetic data.
  - Gill et al. regenerate CondaQA and DROP with gpt-4-turbo and compare validity and difficulty with the crowdsourced originals across several model families.
- **How it makes tasks harder**: n/a (diagnostic).
- **Correctness / verification**: Human validity annotation (Gill).
- **Difficulty control**: n/a.
- **Reported results**:
  - Chen: MMLU 38.08 → 47.27 for the base model, but after SFT FollowBench HSR fell 27.58 → 24.00 and MT-Bench 5.45 → 5.39. Unlearning restored them to 27.87 and 5.85, while another 300B natural tokens only partly helped (25.22 / 5.43).
  - Gill: 72.8% of generated CondaQA questions and 88.7% of DROP questions were valid, but only 53.1% of DROP compositional edits by author assessment.
  - Generated data was "less challenging for LLMs", did not preserve model rankings, and the differences "may be imperceptible to researchers".
  - Generated questions averaged 18.35 words vs 27.17 for human-written ones.
- **Limitations / failure modes**: One 2B model; two reading-comprehension datasets.
- **How to reuse with easy seed tasks**: Before training, compare hardened items with real hard items on length, structural-token frequency, embedding overlap and reference-model perplexity. Vary templates across families, verify difficulty with a probe model, and keep natural-data replay in SFT.

### MCQ answer-position bias (+ Bad Dice) — Do Large Language Models Plan Answer Positions? Position Bias in Multiple-Choice Question Generation (Xuemei Tang et al., 2026)
Link: https://arxiv.org/abs/2605.01846. Related: Zhao et al., 2026, *Large Language Models Are Bad Dice Players*, https://arxiv.org/abs/2601.05414 (ACL 2026).
- **Mechanism**:
  - 10 LLMs and 5 VLMs generate MCQs on three tasks, and the correct-answer positions are tested against uniform.
  - Probing shows the question stem already encodes the planned position; mean-difference steering shifts it.
  - Bad Dice audits native sampling from 15 distributions across 11 models, with 1,000 samples per batch or 1,000 independent calls.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a (distributional audit).
- **Difficulty control**: n/a.
- **Reported results**:
  - First-position rate without / with option labels: Llama-3.1-8B-Instruct 56.8% / 47.9%; Llama-3.2-3B-Instruct 57.9% / 47.1% (uniform is 25%).
  - Removing option identifiers raises first-position odds (OR 1.76, 95% CI [1.653, 1.874]).
  - Steering shifts A → B by up to 22.2%, vs about 8–10% for random directions.
  - Bad Dice: batch generation has a 7% median pass rate on distribution tests. With independent requests, 10 of 11 models pass none. The failures propagate into MCQ answer-position constraints and demographic targets.
- **Limitations / failure modes**: MCQ-centric, but the lesson (LLMs are not samplers) applies to numeric parameters and entity choices.
- **How to reuse with easy seed tasks**: Never ask the generator to "randomize". Draw answer positions, magnitudes and entities with a code RNG, and chi-square-test the final marginals.

### Idiosyncrasies in LLMs (+ linguistic features) — Idiosyncrasies in Large Language Models (Mingjie Sun et al., 2025)
Link: https://arxiv.org/abs/2502.12150 (ICML 2025; code github.com/locuslab/llm-idiosyncrasies). Related: El Attar et al., 2026, https://arxiv.org/abs/2606.04177.
- **Mechanism**:
  - Fine-tuned text-embedding classifiers predict the source LLM of a response.
  - Robustness is tested by shuffling words, paraphrasing, translating and summarizing.
  - El Attar et al. evaluate 284 interpretable linguistic features across 27 LLMs and 10 domains.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - Five-way accuracy (ChatGPT, Claude, Grok, Gemini, DeepSeek) is 97.1%, and 59.8% across four Qwen-2.5 sizes.
  - Base vs instruct Qwen2.5-7B: 96.8%.
  - Paraphrased or translated text is still classified at over 90%; summarization drops accuracy to about 58%.
  - SFT'ing Llama3.1-8B and Gemma2-9B on the same ChatGPT data (UltraChat) cut their separability from 96.5% to 59.8%.
  - SFT'ing Qwen2.5-7B separately on Llama vs Gemma responses gave students separable at 98.9%: synthetic data propagates the source's idiosyncrasies.
  - El Attar: lexical richness is the only indicator robust across families and domains.
- **Limitations / failure modes**: Studied on responses; prompt-side signatures are less studied.
- **How to reuse with easy seed tasks**: Train a cheap synthetic-vs-real classifier and a generator-ID classifier on your hardened items. High separability means the student can learn the generator's signature. Diversify generators and templates, and add real items, until separability drops.

### Subliminal Learning — Subliminal Learning: Language models transmit behavioral traits via hidden signals in data (Alex Cloud et al., 2025)
Link: https://arxiv.org/abs/2507.14805. Published in Nature (2026) as "Language models transmit behavioural traits through hidden signals in data", doi:10.1038/s41586-026-10319-8.
- **Mechanism**:
  - A teacher with trait T (an animal preference or misalignment) generates number sequences, code or CoT unrelated to T, which is filtered to remove references to T.
  - A student fine-tuned on it acquires T when it shares the teacher's base or initialization.
  - A theorem shows this in general networks under certain conditions, and an MNIST MLP reproduces it.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a. Filtering, including filtering CoT to correct responses, is shown to be insufficient.
- **Difficulty control**: n/a.
- **Reported results**:
  - Traits transfer through numbers, code and correct-filtered CoT (a misaligned teacher yields a misaligned student).
  - No reliable transfer across different base models, except GPT-4.1 ↔ GPT-4o, which likely share an initialization.
  - Prompted classifiers cannot detect the hidden traits, and in-context learning does not transmit them.
- **Limitations / failure modes**: SFT on teacher outputs only; transfer through teacher-authored prompts under RL is untested.
- **How to reuse with easy seed tasks**: If a sibling or checkpoint of the student writes hard tasks or traces, it can pass on its quirks, including hacking tendencies, even through correct-filtered data. Prefer generators from a different base model, or add behavioral regression probes.

### Artificial Hivemind (+ NCD diversity) — Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) (Liwei Jiang et al., 2025)
Link: https://arxiv.org/abs/2510.22954 (NeurIPS 2025 D&B, oral). Related: Tieman & Markou, 2026, *Inferred Generative-Process Diversity Predicts Correlated Failure Across Language Models*, https://arxiv.org/abs/2609.03422.
- **Mechanism**:
  - Infinity-Chat has 26,070 open-ended real queries. The study covers 70+ LMs and measures intra- and inter-model embedding similarity; the intra-model test uses 50 responses per query on 100 queries (top-p 0.9, T = 1.0).
  - Tieman & Markou compute normalized compression distance (NCD) between raw outputs of 38 models, residualized against a permutation control, and relate it to chance-corrected correlated failure on 10 benchmark families.
- **How it makes tasks harder**: n/a.
- **Correctness / verification**: n/a.
- **Difficulty control**: n/a.
- **Reported results**:
  - For 79% of queries, a model's average self-similarity exceeds 0.8. Under min-p decoding, 61.2% of pairs still exceed 0.8.
  - Cross-model similarity is 0.71–0.82, e.g., DeepSeek-V3 vs qwen-max 0.82 and DeepSeek-V3 vs gpt-4o 0.81.
  - NCD diversity predicts less correlated failure (partial rank association −0.216 [−0.309, −0.122], negative on all 10 benchmarks), beyond semantic similarity and capability.
- **Limitations / failure modes**: Open-ended prompts; NCD is a proxy.
- **How to reuse with easy seed tasks**: Adding generator brands buys less diversity than expected. Choose generator and verifier ensembles that are far apart in NCD, and force structural diversity with code-driven operators rather than temperature.

### Partial-input and training-dynamics baselines — Annotation Artifacts in Natural Language Inference Data (Suchin Gururangan et al., 2018)
Link: https://arxiv.org/abs/1803.02324 (NAACL 2018). Related: Proebsting & Poliak, 2024, https://arxiv.org/abs/2410.08996; Balepur et al., 2024, *Artifacts or Abduction*, https://arxiv.org/abs/2402.12483 (ACL 2024); Swayamdipta et al., 2020, *Dataset Cartography*, https://arxiv.org/abs/2009.10795 (EMNLP 2020).
- **Mechanism**:
  - Give a model only part of the input: the hypothesis without the premise, the choices without the question, or the answer prior. Accuracy above the majority baseline means the label leaks through the generation process.
  - Dataset Cartography plots per-example confidence and variability over training epochs.
- **How it makes tasks harder**: Items solvable from partial input are "fake hard" and should be removed or rewritten.
- **Correctness / verification**: The hard-to-learn cartography region often holds label errors, so send it to audit.
- **Difficulty control**: Ambiguous-region items drive OOD generalization.
- **Reported results**:
  - Hypothesis-only accuracy is about 67% on SNLI and 53% on MultiNLI.
  - On LLM-elicited NLI (GPT-4, Llama-2, Mistral) it is 86–96%. "Swimming in a pool" appears in more than 10,000 GPT-4 contradictions.
  - Choices-only prompting beats the majority baseline in 11 of 12 settings, by up to 0.33.
- **Limitations / failure modes**: The classic results come from classification tasks. Item-level shortcut gates are in notes 07, 12, 13 and 14 (AFLite); here these serve as dataset-level audits.
- **How to reuse with easy seed tasks**: Per generator family, run a question-only solver (key givens stripped), an answer-prior guesser, and a choices-only solver for MCQs. Rewrite any family that scores above chance. Route hard-to-learn items from a pilot run to label audit.

## Complexification operators from this area

1. **Pass-rate-band targeting with prompt replay**
   - What it does: Keep reusing prompts whose current pass rate is in a middle band (priority near 0.5) and mix them with fresh prompts.
   - Easy → hard: A uniform sample from 10k synthetic prompts, most solved 8/8 → a buffer of p ∈ [0.25, 0.75] prompts filling up to 75% of each batch, with a 10-step cooldown and at most 15 reuses.
   - Keep it verifiable: Labels are unchanged. Re-estimate p on fresh rollouts, and run a random-reward control (the Qwen2.5 anomaly).
   - Sources: 2603.21177, 2505.17063, 2510.13786; cold-start priors 2609.09075 (note 10).
2. **Solvability-conditioned rewriting**
   - What it does: Rewrite solved seeds harder and unsolved seeds easier, then keep the lowest non-zero pass rates.
   - Easy → hard: A GSM8K-style item solved 8/8 → the writer adds a discount tier and a remainder condition until it is solved 1–3 times in 8. The top 500 are kept.
   - Keep it verifiable: Replace majority-vote labels with a programmatic solver or cross-family verifier, and re-solve every rewrite.
   - Sources: 2505.17063, 2603.24202 (note 04).
3. **Difficulty-spectrum smoothing (mixed-complexity blend)**
   - What it does: Sample generator knobs so that training covers a smooth difficulty range, which theory predicts puts training in a "relay" regime rather than grokking plateaus.
   - Easy → hard: 500 counting items at depth 1 → 100 items spread over depths 1–7 with mixed operators. Mixed-100 matched Easy-500 at 44.2, but Easy-100 collapsed and Mixed-500 fell to 35.5. Enigmata found easy:medium:hard 1:1:1 beat 2:6:2 (note 05).
   - Keep it verifiable: The generator computes answers. Report per-tier accuracy with seeds.
   - Sources: 2604.18381, 2602.14872, 2505.19914, 2605.26934.
4. **Seed breadth over rewrite depth**
   - What it does: At a fixed budget, spread rewrites over many seeds.
   - Easy → hard: about 7 seeds × 999 rewrites (0.1% of GSM8K train) → about 750 seeds × 9 rewrites (10%), filtered by OT diversity against a real reference set.
   - Keep it verifiable: Validate answers with a stronger or independent solver, and measure diversity (OT, route fingerprints, NCD) instead of assuming it.
   - Sources: 2607.04563, 2609.33780.
5. **Real-anchor blending**
   - What it does: Mix a fixed share of oracle-backed real or seed items into every hardened set.
   - Easy → hard: 100% LLM-hardened RL data → 20% real reference items (Anchored Self-Play: +7.0 pp average over unanchored self-play; note 04) or about 30% synthetic in SFT and midtraining blends. rStar-Coder: seed+synthetic 57.3 > seed-only 49.7 > synthetic-only 46.8 (note 04). Nemotron-CrossThink: 2:1 general:math was best (note 07).
   - Keep it verifiable: Real items carry trusted labels. Track synthetic-distribution and natural-distribution evals separately, and treat a widening gap as drift.
   - Sources: 2510.01631, 2607.03523, 2505.21297, 2504.13941.
6. **Mastered-item review and easy-pool replay**
   - What it does: Periodically reinsert mastered or easy items, and keep moderately easy items to regularize length.
   - Easy → hard: Retire every prompt at pass rate 1 → queue mastered prompts (p_add 0.25) and replace 10% of the batch every 5 steps, about 2% of prompts. Add a 10% easy pool (MiMo-7B, note 12) and up-weight moderately easy items (Frugal).
   - Keep it verifiable: Same verifier. Log per-item regressions (correct-set turnover).
   - Sources: 2606.03087, 2511.01937, 2505.07608.
7. **Positive–negative prompt pairing**
   - What it does: Pair a hard-but-solvable prompt with an easy-but-brittle prompt and reweight outcomes so both rare successes and rare failures give strong gradients.
   - Easy → hard: The two highest-variance prompts → q+ from a hard pool (low success) plus q− with p ∈ [1 − c/G, 1 − 1/G], trained with WGRPO.
   - Keep it verifiable: Only selection changes. Remove q+ from the eval set, and confirm on a non-Qwen-Math model.
   - Sources: 2602.03452.
8. **Controlled repetition budget**
   - What it does: Tie the number of unique hardened items to total optimization steps.
   - Easy → hard: 1M unique items for a 1,000-step run → about S·B/25 unique items (about 10k for 1,000 × 256) reused up to about 25×.
   - Keep it verifiable: Watch held-out accuracy and degeneration (gibberish, length blow-up).
   - Sources: 2509.25300, 2504.20571.
9. **Surrogate-optimized mixture over operator families**
   - What it does: Treat each operator family as a mixture component, fit a response surface on proxy runs, and take its argmax.
   - Easy → hard: Uniform mix → Dirichlet-sampled pilot weights (RegMix-style), a quadratic or LightGBM fit, then w* (MoDoMoDo +5.24 OOD).
   - Keep it verifiable: Accept w* only if it beats a fixed equal mix under 8–12 matched seeds and paired CIs (DataFlex-RL found none that did).
   - Sources: 2407.01492, 2505.24871, 2609.06107, 2608.27409.
10. **Deterministic gold perturbation**
    - What it does: Build negatives, distractors and variants by code edits of a gold item, so each item carries its own integrity check.
    - Easy → hard: "Write a plausible wrong answer" → swap one quantity or negate one clause in code, then assert the result differs from gold.
    - Keep it verifiable: A string or semantic diff against gold is free. Still read raw samples: a 32-point artifact replicated at N = 500.
    - Sources: 2607.13707.
11. **Isomorphic relabeling with an invariance-checked reward**
    - What it does: Present logically isomorphic variants (renamed predicates and objects, permuted structure) and credit only outputs correct under both.
    - Easy → hard: A rule-induction task verified only on its instances → the same task renamed. Instance enumeration fails, and the true rule passes.
    - Keep it verifiable: The isomorphism is constructed by program. Extensional-only rewards induce hacking that grows with complexity.
    - Sources: 2604.15149.
12. **External-sampler balancing of answer marginals**
    - What it does: A code RNG decides every structural random choice, and the LLM writes only content.
    - Easy → hard: "Randomize where the correct answer goes" (first position 47.1–57.9%) → the code draws the position ~ Uniform{A–D} and magnitudes from a target histogram. The LLM writes a stem and distractors conditioned on those draws.
    - Keep it verifiable: Re-verify the answer key after insertion, and chi-square-test the marginals.
    - Sources: 2605.01846, 2601.05414.
13. **Partial-input ablation as a dataset-level hardness audit**
    - What it does: Score each generator family with question-only, choices-only and answer-prior baselines.
    - Easy → hard: Keep all generated items (86–96% hypothesis-only solvable) → rewrite or drop families until partial-input accuracy is at chance.
    - Keep it verifiable: Report baseline accuracy with CIs next to full-input accuracy.
    - Sources: 1803.02324, 2410.08996, 2402.12483.
14. **Held-out generator-family, post-cutoff and seed-clustered evaluation split**
    - What it does: Evaluate only on items the generator could not have shaped, and split by seed cluster.
    - Easy → hard: A random 5% split of the same synthetic pool → post-cutoff contests, items from a held-out generator family and operator, real production tasks, and a regression suite. The split is by seed, with clustered SEs.
    - Keep it verifiable: Official answers for contests; the same verifier stack and audit for held-out families.
    - Sources: 2505.23281, 2411.00640, 2502.12150.
15. **Trivial-agent baselines for hardened agentic tasks**
    - What it does: Before counting a task as hard and valid, check that empty, no-op and random agents fail it.
    - Easy → hard: A generated airline-policy task that an empty response "passes" → a task whose checker requires the state change and rejects no-ops.
    - Keep it verifiable: Outcome-validity checks such as state diffs, adversarial tests and fuzzing.
    - Sources: 2507.02825.

## Insights & pitfalls

- **Dose, then diversity, then volume.** RLVR saturates at 1–2 prompts (Qwen-Math), about 16% of a pool (LIMR) or about 500 items per task (Synthetic Data RL). Mixed-complexity dose is non-monotone (Mixed-500 < Mixed-100 on counting). Run a dose ladder per operator before building a big generator.
- **Reuse is not the enemy; low diversity is.** Up to about 25× reuse is free (Tan et al.), and ScaleRL-style retirement of mastered prompts raises the asymptote. What hurts is a small number of seeds: rewriting a few seeds hundreds of times collapses diversity, and diversity tracks downstream accuracy (Fidelity–Diversity).
- **Blend recipe (default until your own ablation says otherwise).** This answers note 12's open problem "synthetic hardening vs selecting hard real data" with a blend, not a choice:
  1. A fixed equal mix across operator families.
  2. A 20–33% oracle-backed real or seed anchor.
  3. A 2–10% review or easy pool.
  4. Mid-band replay with reuse ≤ 25×.
  - To settle it for your model, run an equal-compute 2×2: {hardened-from-easy-seeds vs pass-rate-selected hard real} × {with vs without a 20% real anchor}, with 8–12 matched seeds and a pre-registered suite.
- **Most data-policy "wins" are noise or scale-specific.** In DataFlex-RL, signs flip between 3B and 7B (difffilter −0.79 vs +0.08), and a math-heavy aggregate reversed the ranking (ρ = −0.33). ScaleRL: early winners can lose at scale, and A has ±0.02 run-to-run error.
- **Pilot arithmetic (derived from p(1−p)/N at p = 0.5).** An unpaired comparison with 95% significance and 50% power needs a gap of 25.3 points at N = 30, 21.9 at N = 40, 13.9 at N = 100, 9.8 at N = 200 and 6.2 at N = 500. For 80% power these become 36.1, 31.3, 19.8, 14.0 and 8.9. Paired analysis plus averaging cut the HumanEval threshold from 12 to 2–4 points. Use small pilots to kill ideas, not to confirm them.
- **Recommended pilot protocol.**
  - Fix the container, GPU type, batch size and decoding. Use FP32 compute or batch-invariant kernels (BF16 alone moves accuracy up to 9% and length up to 9,000 tokens).
  - Use 10–30 samples per item: Bayes@N reaches τ > 0.9 at N = 10, and Sober Look finds 30 seeds optimal.
  - Analyze paired per-item results with SEs clustered by seed family.
  - Use 3 training seeds to screen and 8–12 to decide. Average the last few checkpoints (Signal and Noise).
  - Adopt only if the paired CI excludes zero on the pre-registered domain-balanced aggregate and no regression-suite metric drops.
- **Variants are clusters.** Clustered SEs can exceed naive SEs by 3× (DROP 1.34 vs 0.44). Splitting train and eval by variant instead of by seed gives both leakage and falsely tight CIs.
- **Pre-registered held-out suite.**
  1. Post-cutoff contests (MathArena; LiveCodeBench windows and SWE-rebench in notes 04 and 14), pooled to hundreds of items.
  2. Items from a generator family and operator never used in training.
  3. Natural-distribution targets.
  4. A regression suite for untargeted skills (IFEval, hallucination, conversational QA, long context). Math-only SFT dropped IFEval from 69.2 to 42.3 while RL kept 70.0.
  5. Correct-set turnover on the training set.
- **Qwen2.5-Math confounds small-data results.** 1-shot RLVR, LIMR and Beyond Variance report headline numbers on it, and Prompt Replay found 32 identical prompts per step matched full-data training on Qwen2.5-Math-1.5B. Every dose or mixing conclusion needs a second family plus a random-reward control (note 10, Spurious Rewards).
- **Consolidation doesn't buy coverage.** Merge, Mix RL and MOPD differ by at most 1.4 points on average, and all are indistinguishable from base at AIME pass@32. Reweighting reachable solutions won't raise pass@k. Coverage needs genuinely new hard items or a stronger teacher, and claims need paired bands (Kernel: no crossover established in 5 public pairs until fresh 32k-token prompts were used).
- **Judge and generator separation.** PLS is 23.6% for the same model, 19.3–22.3% for inheritance, 8.9% within a series and 2.8% across series. SFT leaks most (23.6% vs DPO 5.2%), and small students leak most.
- **Signatures persist, propagate and carry traits.**
  - A 97.1% five-way ID classifier survives paraphrase and translation at over 90%.
  - Students SFT'd on different teachers are 98.9% separable.
  - Subliminal transfer happens through correct-filtered CoT within a shared base.
  - Cross-model similarity is 0.71–0.82.
  - Rule: don't use the student's own checkpoints or siblings as the sole generator of hardened tasks or traces. Mix 2–3 generator families from different series, chosen by NCD distance.
- **Synthetic "hard" is often easier than it looks.** LLM-regenerated DROP/CondaQA were less challenging and didn't preserve rankings, and generated questions were shorter (18.35 vs 27.17 words). LLM NLI is 86–96% hypothesis-only solvable. Uniform-format Q-A (2% of 300B tokens) raised MMLU but cut FollowBench HSR (27.58 → 24.00) after SFT.
- **Verifier validity must scale with difficulty.** IPT shortcuts concentrate in high-complexity items (458 vs 40). Agentic checkers overestimate performance, e.g., KernelBench by 31 points and τ-bench's empty-response agent at 38%. Hardening without invariance checks and trivial-agent baselines inflates measured gains.
- **Cheap audit checklist per generator family before training.**
  - (a) A synthetic-vs-real classifier (embedding + logistic regression).
  - (b) A generator-ID classifier across your families.
  - (c) Question-only, choices-only and answer-prior baselines.
  - (d) Answer-marginal histograms with chi-square tests, and code-RNG positions and parameters.
  - (e) Diversity: OT fidelity/diversity, rewrites per seed, route fingerprints, NCD between families.
  - (f) Disjointness of generator, judge and student families.
  - (g) Trivial-agent and isomorphic-invariance checks.
  - (h) A hand audit of 100–200 items with Wilson CIs, prioritized by IRT flags.
  - (i) Reading raw generations.
- **Wilson intervals for hand audits (derived).** 5/100 errors → [2.2%, 11.2%]; 2/100 → [0.6%, 7.0%]; 10/200 → [2.7%, 9.0%]; 0/200 → < 1.9%; 0/100 → < 3.7%; 0/50 → < 7.1%. To certify a label-error rate below 2%, audit about 200 items and find none.
- **Retention is an active cost.** About 21% of reviewed mastered prompts partially regress under GRPO, repair cost grows with delay, and 1% review already removes GRPO's plateau. RL forgets less than SFT because its data is on-policy. When SFT-distilling hardened traces, regenerate them near-on-policy each epoch or follow with RL.

## Open problems & research opportunities

- **A dose-response law for synthetic hard tasks in RLVR.** Pretraining has ratio laws (about 30%) and RL has compute sigmoids (ScaleRL). Nobody has fitted held-out gain as a function of unique hardened items × diversity × reuse × steps, per operator family, across more than one model family.
- **Equal-compute ablation of synthetic hardening vs selecting hard real data** (note 12's open problem). No study holds rollout compute, verifier quality and seed count fixed while comparing hardened-from-easy seeds, pass-rate-selected hard real items and blends, with at least 2 families and at least 8 seeds.
- **Mixture search that reproduces.** MoDoMoDo's surrogate helped a 2B VLM, and RegMix transfers in pretraining, but DataFlex-RL found no adaptive mixture that beats equal at 12 seeds, with signs flipping across scale. Open questions: low-variance proxy objectives for RL mixtures, and whether surrogates transfer across scale in post-training.
- **Prompt-side generator signatures under RL.** The evidence on idiosyncrasies, preference leakage and subliminal transfer concerns responses, SFT or judges. Whether RL students learn or exploit prompt-side signatures (templates, answer marginals, phrasing), and whether traits pass through generator-authored prompts, is untested.
- **Statistics for agentic multi-turn RL pilots.** Tasks cluster by environment, seed and tool state, and item counts are small. HiBayES, clustered paired SEs and ABC exist, but there is no standard specifying seeds, samples per task, trivial-agent baselines and cluster-aware decision rules for agentic hardening pilots.
- **Certifying label error above verifier competence.** IRT flagging and hand audits need solvers or experts who can do the task. For items beyond frontier ability, only construction-based oracles (gold perturbation, generators with known answers, isomorphic checks) currently give CIs on label-error rates.
- **Predicting untargeted regression before training.** TI and correct-set turnover are measured after the fact. A cheap pre-training predictor, e.g., representation drift on a small probe run, would make regression suites targeted.
- **Which diversity measure predicts RL gain?** Embedding similarity, OT diversity, route fingerprints and NCD each correlate with outcomes in isolation. It is untested which one best predicts held-out RL improvement for a hardened task pool, and how much one more generator family adds given 0.71–0.82 inter-model similarity.
- **Cross-family replication harness for small-data RLVR.** 1-shot, LIMR and pairing rest mostly on Qwen2.5-Math. A shared, cheap multi-family harness with random-reward controls (DataFlex-RL is a start) would separate data-efficiency effects from base-model artifacts.

## References

1. Wang, Y., Yang, Q., Zeng, Z., Ren, L., et al. (2025). *Reinforcement Learning for Reasoning in Large Language Models with One Training Example*. NeurIPS 2025; arXiv:2504.20571. https://arxiv.org/abs/2504.20571
2. Pang, Y., Li, J., Sheng, X., Peng, R., et al. (2026). *Beyond Variance: Prompt-Efficient RLVR via Rare-Event Amplification and Bidirectional Pairing*. arXiv:2602.03452. https://arxiv.org/abs/2602.03452
3. Li, X., Zou, H., Liu, P. (2025). *LIMR: Less is More for RL Scaling*. arXiv:2502.11886. https://arxiv.org/abs/2502.11886
4. Guo, Y., Guo, Z., Huang, C., Wang, Z.-A., et al. (2025). *Synthetic Data RL: Task Definition Is All You Need*. arXiv:2505.17063. https://arxiv.org/abs/2505.17063
5. Baroian, A., Berger, R. (2026). *Prompt replay: speeding up GRPO with on-policy reuse of high-signal prompts*. arXiv:2603.21177. https://arxiv.org/abs/2603.21177
6. Bauer, J., Walshe, T., Pham, D., Vishwakarma, H., et al. (2026). *Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes*. arXiv:2604.18381. https://arxiv.org/abs/2604.18381
7. Tan, Z., Geng, H., Yu, X., Zhang, M., et al. (2025). *Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning*. ACL 2026; arXiv:2509.25300. https://arxiv.org/abs/2509.25300
8. Khatri, D., Madaan, L., Tiwari, R., Bansal, R., et al. (2025). *The Art of Scaling Reinforcement Learning Compute for LLMs*. arXiv:2510.13786. https://arxiv.org/abs/2510.13786
9. Kang, F., Ardalani, N., Kuchnik, M., Emad, Y., et al. (2025). *Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls*. EMNLP 2025; arXiv:2510.01631. https://arxiv.org/abs/2510.01631
10. Liu, Q., Zheng, X., Muennighoff, N., Zeng, G., et al. (2024). *RegMix: Data Mixture as Regression for Language Model Pre-training*. ICLR 2025; arXiv:2407.01492. https://arxiv.org/abs/2407.01492
11. Liang, Y., Qiu, J., Ding, W., Liu, Z., et al. (2025). *MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning*. arXiv:2505.24871. https://arxiv.org/abs/2505.24871
12. Liang, H., Chen, M., Feng, H., Qiang, M., et al. (2026). *DataFlex-RL: An Evaluation Platform for RLVR Data Policies*. arXiv:2609.06107. https://arxiv.org/abs/2609.06107
13. Wu, S., Yang, K., Cai, Y., Xu, X., et al. (2026). *Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms*. arXiv:2608.27409. https://arxiv.org/abs/2608.27409
14. Qin, C., Yang, C., Si, Q., Gu, N., et al. (2026). *Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR*. arXiv:2606.03087. https://arxiv.org/abs/2606.03087
15. Bounhar, A., Abdine, H., Dufraisse, E., Chamma, A., et al. (2025). *Shorter but not Worse: Frugal Reasoning via Easy Samples as Length Regularizers in Math RLVR*. arXiv:2511.01937. https://arxiv.org/abs/2511.01937
16. Chen, H., Razin, N., Narasimhan, K., Chen, D. (2025). *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting*. ICML 2026; arXiv:2510.18874. https://arxiv.org/abs/2510.18874
17. Huan, M., Li, Y., Zheng, T., Xu, X., et al. (2025). *Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning*. arXiv:2507.00432. https://arxiv.org/abs/2507.00432
18. Wang, A., Manole, T., Bunea, F., Thickstun, J. (2026). *Fidelity-Diversity Metrics for Text*. arXiv:2607.04563. https://arxiv.org/abs/2607.04563
19. Zhang, D., Wu, M., Li, J. (2026). *Selecting Diverse SFT Traces Improves Post-RL Generalization*. arXiv:2609.33780. https://arxiv.org/abs/2609.33780
20. Hochlehnert, A., Bhatnagar, H., Udandarao, V., Albanie, S., et al. (2025). *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility*. COLM 2025; arXiv:2504.07086. https://arxiv.org/abs/2504.07086
21. Yuan, J., Li, H., Ding, X., Xie, W., et al. (2025). *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference*. arXiv:2506.09501. https://arxiv.org/abs/2506.09501
22. He, H., and Thinking Machines Lab (2025). *Defeating Nondeterminism in LLM Inference*. Blog post, Sep 10 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
23. Wang, S. (2025). *Measuring all the noises of LLM Evals*. arXiv:2512.21326. https://arxiv.org/abs/2512.21326
24. Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*. arXiv:2411.00640. https://arxiv.org/abs/2411.00640
25. Madaan, L., Singh, A. K., Schaeffer, R., Poulton, A., et al. (2024). *Quantifying Variance in Evaluation Benchmarks*. arXiv:2406.10229. https://arxiv.org/abs/2406.10229
26. Heineman, D., Hofmann, V., Magnusson, I., Gu, Y., et al. (2025). *Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation*. NeurIPS 2025; arXiv:2508.13144. https://arxiv.org/abs/2508.13144
27. Hariri, M., Samandar, A., Hinczewski, M., Chaudhary, V. (2025). *Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation*. ICLR 2026; arXiv:2510.04265. https://arxiv.org/abs/2510.04265
28. Yang, C., Chen, J. (2026). *RLVR is a Kernel, Not a Function: Statistical Inference for pass@k Crossovers*. arXiv:2609.22547. https://arxiv.org/abs/2609.22547
29. Luettgau, L., Coppock, H., Dubois, M., Summerfield, C., et al. (2025). *HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics*. arXiv:2505.05602. https://arxiv.org/abs/2505.05602
30. Balunović, M., Dekoninck, J., Petrov, I., Jovanović, N., et al. (2025). *MathArena: Evaluating LLMs on Uncontaminated Math Competitions*. arXiv:2505.23281. https://arxiv.org/abs/2505.23281
31. Zhu, Y., Jin, T., Pruksachatkun, Y., Zhang, A., et al. (2025). *Establishing Best Practices for Building Rigorous Agentic Benchmarks*. arXiv:2507.02825. https://arxiv.org/abs/2507.02825
32. Helff, L., Delfosse, Q., Steinmann, D., Härle, R., et al. (2026). *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*. arXiv:2604.15149. https://arxiv.org/abs/2604.15149
33. Land, S., Bikel, D. M. (2026). *Auditing LLM Benchmarks with Item Response Theory*. EMNLP 2026; arXiv:2605.30504. https://arxiv.org/abs/2605.30504
34. Gema, A. P., Leang, J. O. J., Hong, G., Devoto, A., et al. (2024). *Are We Done with MMLU?* arXiv:2406.04127. https://arxiv.org/abs/2406.04127
35. Vendrow, J., Vendrow, E., Beery, S., Madry, A. (2025). *Do Large Language Model Benchmarks Test Reliability?* arXiv:2502.03461. https://arxiv.org/abs/2502.03461
36. Ballı, S. (2026). *The Test Oracle Problem in Synthetic LLM-as-Judge Corpora: Disappearance, Distortion and a Validation Protocol*. arXiv:2607.13707. https://arxiv.org/abs/2607.13707
37. Li, D., Sun, R., Huang, Y., Zhong, M., et al. (2025). *Preference Leakage: A Contamination Problem in LLM-as-a-judge*. ICLR 2026; arXiv:2502.01534. https://arxiv.org/abs/2502.01534
38. Panickssery, A., Bowman, S. R., Feng, S. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. arXiv:2404.13076. https://arxiv.org/abs/2404.13076
39. Chen, J., Zhang, Y., Wang, B., Zhao, W. X., et al. (2024). *Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models*. arXiv:2406.12397. https://arxiv.org/abs/2406.12397
40. Gill, A., Ravichander, A., Marasović, A. (2025). *What Has Been Lost with Synthetic Evaluation?* arXiv:2505.22830. https://arxiv.org/abs/2505.22830
41. Tang, X., Duan, X., Cai, Z. G. (2026). *Do Large Language Models Plan Answer Positions? Position Bias in Multiple-Choice Question Generation*. arXiv:2605.01846. https://arxiv.org/abs/2605.01846
42. Zhao, M., Du, Y., Wang, M. (2026). *Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions*. ACL 2026; arXiv:2601.05414. https://arxiv.org/abs/2601.05414
43. Sun, M., Yin, Y., Xu, Z., Kolter, J. Z., et al. (2025). *Idiosyncrasies in Large Language Models*. ICML 2025; arXiv:2502.12150. https://arxiv.org/abs/2502.12150
44. El Attar, Y., Dönmez, E., Maurer, M., Falenska, A. (2026). *A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models*. arXiv:2606.04177. https://arxiv.org/abs/2606.04177
45. Cloud, A., Le, M., Chua, J., Betley, J., et al. (2025). *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data*. arXiv:2507.14805; Nature (2026), doi:10.1038/s41586-026-10319-8. https://arxiv.org/abs/2507.14805
46. Jiang, L., Chai, Y., Li, M., Liu, M., et al. (2025). *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)*. NeurIPS 2025 Datasets & Benchmarks (oral); arXiv:2510.22954. https://arxiv.org/abs/2510.22954
47. Tieman, R., Markou, E. (2026). *Inferred Generative-Process Diversity Predicts Correlated Failure Across Language Models*. arXiv:2609.03422. https://arxiv.org/abs/2609.03422
48. Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., et al. (2018). *Annotation Artifacts in Natural Language Inference Data*. NAACL 2018; arXiv:1803.02324. https://arxiv.org/abs/1803.02324
49. Proebsting, G., Poliak, A. (2024). *Hypothesis-only Biases in Large Language Model-Elicited Natural Language Inference*. arXiv:2410.08996. https://arxiv.org/abs/2410.08996
50. Balepur, N., Ravichander, A., Rudinger, R. (2024). *Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?* ACL 2024; arXiv:2402.12483. https://arxiv.org/abs/2402.12483
51. Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., et al. (2020). *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*. EMNLP 2020; arXiv:2009.10795. https://arxiv.org/abs/2009.10795
52. Shao, R., Li, S. S., Xin, R., Geng, S., et al. (2025). *Spurious Rewards: Rethinking Training Signals in RLVR*. arXiv:2506.10947 (cross-reference, note 10). https://arxiv.org/abs/2506.10947
53. Choi, C., Kaya, Z., Wu, S., Ma, T., et al. (2026). *Anchored Self-Play for Code Repair*. arXiv:2607.03523 (cross-reference, note 04). https://arxiv.org/abs/2607.03523
54. Liu, Y., Zhang, L. L., Zhu, Y., Dong, B., et al. (2025). *rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset*. arXiv:2505.21297 (cross-reference, note 04). https://arxiv.org/abs/2505.21297
55. Akter, S. N., Prabhumoye, S., Novikov, M., Han, S., et al. (2025). *Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning*. arXiv:2504.13941 (cross-reference, note 07). https://arxiv.org/abs/2504.13941
56. Xiaomi LLM-Core Team (2025). *MiMo: Unlocking the Reasoning Potential of Language Model – From Pretraining to Posttraining*. arXiv:2505.07608 (cross-reference, note 12). https://arxiv.org/abs/2505.07608
57. Huang, Y., Wen, Z., Chi, Y., Wei, Y., et al. (2026). *On the Emergence of Implicit Curriculum in RLVR Learning Dynamics*. arXiv:2602.14872 (cross-reference, note 09). https://arxiv.org/abs/2602.14872
58. Chen, J., He, Q., Yuan, S., Chen, A., et al. (2025). *Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles*. arXiv:2505.19914 (cross-reference, note 05). https://arxiv.org/abs/2505.19914
59. Zhu, Y., Liu, Q., Cheng, F., Wang, J., et al. (2026). *Reasoning Depth and Environment Complexity: A Controlled Study of RLVR Data Allocation across Logical Reasoning Tasks*. arXiv:2605.26934 (cross-reference, note 05). https://arxiv.org/abs/2605.26934
60. Sancaktar, C., Zhang, D., Synnaeve, G., Cohen, T. (2026). *A Deep Dive into Scaling RL for Code Generation with Synthetic Data and Curricula*. arXiv:2603.24202 (cross-reference, note 04). https://arxiv.org/abs/2603.24202
61. Sha, T., Zhai, S., Zhao, S. (2026). *ThinkPrior: Zero-Rollout Difficulty Priors for Cold-Start Prompt Selection in RLVR*. arXiv:2609.09075 (cross-reference, note 10). https://arxiv.org/abs/2609.09075
