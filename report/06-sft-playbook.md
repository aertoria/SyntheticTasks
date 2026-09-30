# SFT playbook: using hard synthetic tasks for supervised fine-tuning and distillation

> **Key takeaways**
>
> - **Use SFT to install things, and use RL to compose them.** SFT on hard synthetic data reliably installs formats, long-CoT behaviours, domain knowledge and atomic skills. It does not teach composition or transfer to new rules, and SFT on the same composite pool RL would use often *regresses* below the base model. Split the hardened pool: atoms, moderate items, behaviour traces and meta-tasks go to SFT; held-out composites and the hardest band go to RL. **Strong.**
> - **Spend the SFT budget on question selection and teacher choice, not on answer verification.** Difficulty-based question filters beat random selection by 4–6%, and a 1K hard subset beat a random 1K by 13 AIME points. The best teacher is found empirically; it is often not the strongest model. Filtering teacher answers did not help (41.9 unfiltered vs 40.0 GPT-verified). The exception is self-distillation, which needs strict high-precision gates. **Strong.**
> - **Evolve for 2–3 rounds with operators that keep the answer correct, stop on a dev set, and gate well-posedness.** Blind multi-round evolution overshoots into broken items, stalls, bloats prompts and collapses diversity. It has scored below the unevolved baseline. **Strong.**
> - **Maximise unique prompts first, then add traces per prompt in proportion to difficulty.** 64k×1 beat 16k×4 and 8k×8. Extra traces pay off once unique prompts run out (AIME25 41.3 → 49.3), and should go to hard items and be chosen for route diversity. **Moderate–Strong.**
> - **Meta-task transforms (critique, verdict-checked reflection, first-error localisation, recovery) are the cheapest "harder" SFT data, but self-correction has to be learned on-policy.** Critique fine-tuning beat plain SFT by 4–10% on six math benchmarks. Offline or injected-error correction traces do not teach self-correction. **Moderate.**
> - **Teacher outputs are licensed inputs.** Log the generator, rewriter, verifier and judge on every row. Default to MIT/Apache teachers. Llama, Gemma ≤ 3 and Qwen2.5-72B outputs carry naming or derivative obligations, closed APIs forbid training competing models, and the EU AI Act template requires naming the models that generated distillation data. **Strong (as documented terms; not legal advice).**

**Contents**

1. [Where SFT fits in a hard-task pipeline](#1-where-sft-fits-in-a-hard-task-pipeline)
2. [Instruction evolution done right](#2-instruction-evolution-done-right)
3. [Distilling long reasoning traces on hard synthetic problems](#3-distilling-long-reasoning-traces-on-hard-synthetic-problems)
4. [Quality versus quantity](#4-quality-versus-quantity)
5. [Meta-task transforms for SFT](#5-meta-task-transforms-for-sft)
6. [What SFT fails to teach, and how to sequence SFT and RL](#6-what-sft-fails-to-teach-and-how-to-sequence-sft-and-rl)
7. [Mixing and dose](#7-mixing-and-dose)
8. [Licensing and provenance of teacher outputs](#8-licensing-and-provenance-of-teacher-outputs)
9. [Step-by-step recipe](#9-step-by-step-recipe)
10. [Open questions](#10-open-questions)

Evidence tags: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (one recent or unreplicated work), **Proposal** (our synthesis, untested). Related chapters: difficulty definitions and pass-rate profiling are in [Chapter 01](01-diagnosis-difficulty-and-learning-signal.md). The full operator catalogue is in [Chapter 02](02-complexification-operator-taxonomy.md) and generation pipelines in [Chapter 03](03-generation-architectures.md). Verifier engineering is in [Chapter 04](04-verification-and-quality-control.md). RL mechanics, including priming and scaffolds for items at p ≈ 0, are in [Chapter 05](05-rl-playbook.md). Per-domain seeds and operators are in [Chapters 07a](07a-domain-recipes-reasoning.md) and [07b](07b-domain-recipes-agents-and-beyond.md).

---

## 1. Where SFT fits in a hard-task pipeline

When your SFT data is "too easy", the student already produces the target outputs, so the loss is spent on tokens it already predicts. Hardening fixes this in two ways. It raises the per-example information (harder questions give longer, less predictable teacher traces). It also exposes behaviours the student lacks (verification, backtracking, recovery). But hard synthetic data plays different roles in SFT and in RL, and many of the failures reviewed below come from sending the wrong items to the wrong stage.

```
easy seeds ─► profile with the student (k = 8–16 samples) ─► bucket by pass rate p̂
                                                              │
   p̂ ≈ 1 (saturated) ─► harden: §2 and Ch 02 ─► hardened pool ┤   0 < p̂ < 1 ─► RL band (Ch 05)
                                                              │
                          ┌──────── split by role (§6.2) ─────┴──────────┐
                          ▼                                              ▼
     atoms, moderate items, behaviour traces,          held-out composites, hardest
     meta-tasks (§5), knowledge/format gaps            band, rule-shifted variants
                          │                                              │
                SFT / distillation (§3, §4, §7)                    RL (Ch 05)
                          │                                              │
                cold-start checkpoint ─► short RL probe (§6.3) ─► RL ─► rejection-sample
                                                                        or distil back (§6.4)
```

Three questions decide what SFT should get:

1. **Is the gap knowledge, format or behaviour, or is it composition?** SFT closes the first three and mostly fails at the fourth (§6.1).
2. **Is this teacher distillation or self-distillation?** Teacher distillation tolerates label noise; self-distillation does not (§2.4, §3.3).
3. **Will these prompts also be used as RL prompts?** If so, keep them out of SFT, or at least keep the RL composites disjoint from the SFT set (§6.2).

---

## 2. Instruction evolution done right

Instruction evolution (Evol-Instruct and its successors) rewrites easy seed prompts into harder ones with an LLM. The detailed operator menu is in [Chapter 02](02-complexification-operator-taxonomy.md). This section covers the SFT-specific engineering: which operators, how many rounds, when to stop, what to do with overshoot, and how much to verify.

### 2.1 "Complex" is not "hard", and naive evolution fails in five ways

Classic evolvers raise complexity *as an LLM judges it*. What an SFT run needs is prompts whose teacher answers the student cannot yet produce. The gap between the two shows up as five recurring failure modes:

| Failure | Evidence | Guard |
|---|---|---|
| **Overshoot into broken items** | Constraint addition in [IFDecorator](https://arxiv.org/abs/2508.04632) left 10,772 instructions at pass rate 0 and only 7,324 in the target band after 5 iterations. 5.6% of [MathFusion](https://arxiv.org/abs/2503.16212) fused problems stayed unreasonable after 5 regenerations. [Auto Evol-Instruct](https://arxiv.org/abs/2406.00770) found contradictions, changed problem nature and "incorrect or unrealistic mathematical calculations" in GSM8K evolutions | Well-posedness gate; an explicit "INVALID" escape ([Instruction Fusion](https://arxiv.org/abs/2312.15692)); a regeneration or downward path for 0-pass items |
| **Stalling** | GPT-4o intensified complexity in only 1 of 3 depth-evolution attempts ([TaCIE](https://arxiv.org/abs/2410.02795)). ChatGPT difficulty ratings rose 3.00 → 5.48 → 6.35 → 6.84 → 7.08 over four rounds, so gains shrink each round ([WizardLM](https://arxiv.org/abs/2304.12244)) | Measure difficulty on the student, not by LLM rating; drop no-information-gain rewrites |
| **Bloat** | One Code Evol-Instruct prompt grew from 14 to 325 tokens with up to 8 constraints in 4 rounds ([Instruction Fusion](https://arxiv.org/abs/2312.15692)) | A word budget per step (WizardLM allows 10–20 added words); a cap on constraints per prompt |
| **Diversity collapse** | Evol-Instruct data raised pass@1 but pushed pass@8 below the untuned backbone (54.2 vs 55.1) in [EvoTD](https://arxiv.org/abs/2605.11666)'s comparison. [KITE](https://arxiv.org/abs/2607.17043) reports "polarization of competence" in iterative synthetic instruction tuning. A third round of small-model evolution degraded performance ([Hui et al.](https://arxiv.org/abs/2412.11231)) | Keep real seeds in every round; embedding-distance selection ([Deita](https://arxiv.org/abs/2312.15685) τ = 0.9; [Lion](https://arxiv.org/abs/2305.12870) ROUGE-L < 0.7); track pass@k, not just pass@1 |
| **Contamination** | Evol-CodeAlpaca overlapped 70.7% of HumanEval test items; [Tulu 3](https://arxiv.org/abs/2411.15124) removed 3.5% of it and 11.3% of NuminaMath-TIR by 8-gram matching | Decontaminate every round's output, not only the seeds (Ch 04) |

The net effect can be negative. In [UltraIF](https://arxiv.org/abs/2502.04153)'s reimplementation, 10K Evol-Instruct samples scored below plain ShareGPT on IFEval Pr(S) (41.96 vs 43.99). At fixed dataset size, [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) found that LLM rewrites that "harden" task descriptions or add constraints failed to beat the unmodified descriptions. **Strong:** generic "make it harder" prompting is a weak baseline, not a method.

### 2.2 Prefer operators that keep the answer correct

For SFT the answer is usually written by a teacher, so an operator's main job is to produce a *well-posed* harder prompt. Operators that also fix the answer by construction give you free labels you can reuse later for RL or self-distillation.

| Operator | Answer after the edit | SFT evidence |
|---|---|---|
| Inversion (mask a given, reveal the answer; FOBAR / self-verification) | Free: the masked value | Adding 20K items to 80K: +2.3 (FOBAR) and +2.6 (SV) GSM8K points, vs +0.4 for rephrasing and +0.1 for more answer augmentation ([MetaMath](https://arxiv.org/abs/2309.12284)). Check uniqueness of the masked value |
| Sequential fusion (A's answer feeds B) | Checkable if both seeds have code solutions | +18.0 average accuracy with 45K extra fused instructions ([MathFusion](https://arxiv.org/abs/2503.16212)) |
| Program-grounded composition | Executed | [RV-Syn](https://arxiv.org/abs/2504.20426): 50K samples beat prior synthetic sets of 100K (+6.3% on LLaMA-3-8B-Instruct). Its problems were solved 55.6% of the time by Qwen2.5-Math-7B-Instruct, vs 97.2% for MetaMath |
| Constraint back-translation (state constraints the reference answer already meets) | By construction | [Crab](https://arxiv.org/abs/2410.24175); cheap, but constraints can be copied from the reference ([UNSPECIFIC](https://arxiv.org/abs/2608.09154)), and in [RECAST](https://arxiv.org/abs/2505.19030)'s comparison Crab scored below ShareGPT |
| Instruction reversal (code → instruction → code) | Keep only high-fidelity round trips | [Phi-4](https://arxiv.org/abs/2412.08905) |
| Knowledge-tag injection with a budget k, in one shot | Teacher-written | Mistral-7B average: seed 33.9, Evol-Instruct 40.0, Auto Evol-Instruct 41.4, [Tag-Evol](https://arxiv.org/abs/2505.24165) 43.7. One shot avoids errors accumulating across chained rounds |
| Downward evolution (easier siblings) | Teacher-written | Mistral-7B SFT: original 59.7 GSM8K; two downward rounds 74.5; three upward 78.6; both 81.2 ([WizardMath](https://arxiv.org/abs/2308.09583)) |
| Generic in-depth rewriting | Teacher-written, unverified | Baseline; see §2.1 |

Two design rules follow. **Harden through the solution and the environment, not only the prompt text** (OpenThoughts-Agent's negative result applies to text-only rewrites). **Build ladders, not single rungs:** WizardMath's easier siblings added nearly as much as its harder ones (+14.8 vs +18.9 GSM8K points), and [IDEA-MCTS](https://arxiv.org/abs/2410.10392) drew its training samples from every node on the search paths, not only the best leaves. **Moderate.**

### 2.3 Round budgets, stopping rules and overshoot

**Budget 2–3 rounds and stop on a dev set.** WizardCoder used an external dev set (MBPP-400) as an "Evol stop", and performance peaked after 3 rounds ([WizardCoder](https://arxiv.org/abs/2306.08568)). Instruction Fusion reports that code evolution "reaches its capacity (3 rounds)". The third round of small-model evolution degraded results in Hui et al., and [Auto Evol-Instruct](https://arxiv.org/abs/2406.00770) evolves for one round by default after optimising the evolution prompt. **Strong.**

**Keep the ladder, but weight it toward the top rung.** WizardLM merged all four rounds before sampling 70K. [Tree-Instruct](https://arxiv.org/abs/2308.05696) found an easy→hard curriculum (3 → 6 → 10 added nodes) beat a mixed-difficulty set but fell short of training on the 10-node data alone. So the top rung carries most of the value, provided it is well-posed; lower rungs add coverage and bridges (WizardMath's downward rounds).

**Route overshoot instead of discarding it.** A rewrite that no model can solve is either broken or a candidate for a hint or scaffold. Send it back one rung down, or regenerate it with a smaller budget. Recycle rewrites that stayed too easy into another hardening step ([CodecLM](https://arxiv.org/abs/2404.05875)). Keep an "INVALID" output option so the evolver can refuse impossible fusions.

```python
def evolve_for_sft(seeds, operators, dev_eval, max_rounds=3, words_per_step=20):
    pool, rungs, best = list(seeds), [list(seeds)], dev_eval(train_sft(seeds))
    for r in range(max_rounds):
        cand = [op.apply(x, max_new_words=words_per_step)
                for x in rungs[-1] for op in sample_ops(operators, x, k=2)]
        cand = [c for c in cand if c != "INVALID"]
        cand = well_posed(cand)               # LLM or program check; see §2.4
        cand = decontaminate(dedup(cand, tau=0.9))
        cand = diversity_select(cand, ref=pool)
        overshoot = [c for c in cand if teacher_fails(c)]   # route down, don't drop
        rungs.append([c for c in cand if c not in overshoot])
        pool += rungs[-1] + soften(overshoot)
        score = dev_eval(train_sft(pool))     # short SFT; dev set disjoint from final evals
        if score <= best + noise_margin: break  # the "Evol stop"
        best = score
    return pool                               # all rungs, not just the top one
```

### 2.4 How much answer verification SFT needs

SFT distillation is unusually tolerant of wrong teacher answers, and several independent results agree:

- [OpenThoughts](https://arxiv.org/abs/2506.04178) (compute-controlled, math): no filtering 41.9, random filtering 41.6, GPT verification 40.0. Majority-consensus and length-based answer filters also failed to beat training on all samples.
- A GPT-4 validate-and-replace step "does not improve the results" in [MathScale](https://arxiv.org/abs/2403.02884). [OpenMathInstruct-2](https://arxiv.org/abs/2410.01560) found "SFT is robust to low-quality solutions". [FLAMES](https://arxiv.org/abs/2508.16514) found coverage beats reliability at a fixed budget. MathFusion's filtered and unfiltered sets scored 39.1 vs 39.0. [SynthLLM](https://arxiv.org/abs/2503.19551)'s answer-filtering ablation moved MATH from 42.0 to 42.2.
- Code and SWE: training on *incorrect* R1 solutions beat training on correct ones in [OpenCodeReasoning](https://arxiv.org/abs/2504.01943), because the incorrect ones came from harder questions. [SERA](https://arxiv.org/abs/2601.20789)'s verification thresholds made no difference up to 7,400 samples. [HardTests](https://arxiv.org/abs/2505.24098) found test quality mattered less for teacher distillation than for RL or self-distillation.

What SFT does *not* tolerate:

- **Degenerate responses.** Replacing 20% of responses with short "shirker" answers cut [Instruct-SkillMix](https://arxiv.org/abs/2408.14774)'s AlpacaEval LC win rate from 31.57% to 23.93%.
- **Responses that violate the property being taught.** Removing UltraIF's per-constraint evaluation-question filter, which checks that responses satisfy each added constraint, cost 3.35–5.36 points. When the skill *is* satisfying checkable requirements (instruction following), response checks are not optional.
- **Broken structure.** Long-CoT samples with incorrect final answers cost only 3.2%, but shuffling or deleting steps degraded accuracy significantly ([Li et al.](https://arxiv.org/abs/2502.07374)).
- **Lenient gates in self-distillation.** In count-matched rejection fine-tuning, admitting 25% false positives cost −1.58pp, while masking 75% of correct candidates cost only −0.03pp. A strict gate at K = 8 beat a partial-credit gate at K = 32 ([The Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)).

| Data use | Verification to require | Tag |
|---|---|---|
| Teacher distillation (SFT on a stronger model's traces) | Well-posed prompt; complete, untruncated, non-degenerate response; final-answer check optional for reasoning traces, but required when the checked property is the skill (e.g., constraint satisfaction) | **Strong** |
| Self-distillation / RFT on the student's own samples | Strict, high-precision gate; accept low recall | **Moderate** |
| Behaviour-priming data | Filter by behaviour and coherent step structure, not by answer (§6.3) | **Moderate** |
| Prompts that will also feed RL | Verified answer keys; see [Chapter 04](04-verification-and-quality-control.md) | **Strong** |

### 2.5 Make the evolver target-aware

The strongest evolution loops choose *what* to harden from the student's failures:

- **Teacher–student gap.** [Lion](https://arxiv.org/abs/2305.12870) marks a prompt "hard" when the referee's score gap between teacher and student is ≥ 1.0. It generates new prompts from hard ones and keeps hard:easy at 1:1 to limit forgetting (3 iterations of 6K prompts). [CodecLM](https://arxiv.org/abs/2404.05875) keeps pairs with a judged gap > 3/10 and sends the rest back to be made harder.
- **Deficiency diagnosis.** [Infinity Instruct](https://arxiv.org/abs/2506.11116) grades student responses to evolved prompts and re-evolves the prompts with poor responses.
- **Agreement bands.** [Phi-4](https://arxiv.org/abs/2412.08905) discards seeds where all sampled answers agree (too easy) or none do (too hard or ambiguous). [Phi-4-reasoning](https://arxiv.org/abs/2504.21318) targets seeds "at the edge of Phi-4's current abilities", measured by weak-model agreement with a strong model's plurality answer.
- **Solver failure plus agreement.** [SAND-Math](https://arxiv.org/abs/2507.20527) keeps questions a 32B solver gets wrong *and* on which the teacher's samples agree. Note that failure filters also keep wrong-key items.

Replace LLM-judged gaps with measured ones where you can: the student's pass rate over k samples, or disagreement with a stronger model's plurality answer. **Moderate.**

---

## 3. Distilling long reasoning traces on hard synthetic problems

Distillation is where hard synthetic problems pay off most directly in SFT. Four decisions matter, in order: which questions, which teacher, which traces to keep, and how many traces per question.

### 3.1 Question selection comes first

| Filter | What it keeps | Evidence |
|---|---|---|
| Reference-model failure + domain stratification | Questions that Qwen2.5-7B/32B-Instruct fail, with longer traces treated as harder | [s1K](https://arxiv.org/abs/2501.19393), AIME24 on Qwen2.5-32B-Instruct: s1K 50.0 vs random-1K 36.7, diverse-1K 26.7, longest-1K 33.3; the full 59K scored 53.3, a difference that was not significant |
| Teacher response length | Long-trace questions | [OpenThoughts](https://arxiv.org/abs/2506.04178): best math/science filter, +4% over random. [GLM-4.5](https://arxiv.org/abs/2508.06471): dropping the bottom 50% by response length gave +2–4% on math and science with half the data. [AceReason-Nemotron](https://arxiv.org/abs/2505.16400) drops problems whose R1 responses are under 2,000 tokens and downsamples 2,000–4,000 |
| LLM difficulty rating | Items judged hard | OpenThoughts: best code filter, +6% over random. [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) removed more than 50% of SFT data tagged easy for Maverick and pruned 95% of SFT data for Behemoth |
| Student already solves | Items the base model gets wrong | [ScaleDiff](https://arxiv.org/abs/2509.21070) drops problems the base model already solves (about 43% filtered) and detects "difficult" items with one forward pass of an adaptive-thinking model |
| Weak–strong gap | Items at the edge of the student | [Phi-4-reasoning](https://arxiv.org/abs/2504.21318) |
| Shortcut removal | Items that need reasoning | [Qwen3](https://arxiv.org/abs/2505.09388) removes cold-start queries its 72B model solves without CoT and queries that are not easily verifiable |

Three caveats keep this from being "hardest is best". [MegaScience](https://arxiv.org/abs/2507.16812)'s difficulty selection helped only one of three sources; for the other two, no selection method beat the full set. [SWE-smith](https://arxiv.org/abs/2504.21798) SFT subsets by rated difficulty (2/4/6/8) gave 12.4/10.8/13.6/12.2% with no trend, while performance grew roughly log-linearly with the number of repositories. And OpenThoughts found 1–2 high-quality question sources beat 8–16 diverse ones. On the other side, the [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) card reports that adding the easier cn_k12 source lowered SFT performance, "likely because the questions from cn_k12 are less difficult". **Strong** for difficulty-based question selection; **Moderate** for the claim that source quality beats source count.

### 3.2 Teacher choice

The teacher's benchmark score is a poor guide to the student it produces:

| Finding | Source |
|---|---|
| QwQ-32B was a better teacher than DeepSeek-R1 (+1.9% code, +2.6% math) | [OpenThoughts](https://arxiv.org/abs/2506.04178) |
| GPT-5.3-Codex was about 5% worse on TB2.0 than GLM-4.7-AWQ as a teacher | [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) |
| The weakest teacher, Kimi-K2.5 (39.8), produced the best 2B and 3B students, better than Opus 4.5 (53.5) | [Gym-Anything](https://arxiv.org/abs/2604.06126) |
| A non-reasoning teacher (Qwen2.5-32B) overfit to Knights & Knaves during warm-up: MATH 11% vs 54% with a reasoning teacher | [Warm Up Before You Train](https://arxiv.org/abs/2505.13718) |
| SFT of 1.5B/3B students on S1K-style traces can *lower* accuracy, because the traces are too complex to imitate | [BREAD](https://arxiv.org/abs/2506.17211) |
| Multi-teacher SFT improves post-RL coverage | [Route-diverse SFT](https://arxiv.org/abs/2609.33780) |
| Evolver quality is task-dependent: 8B evolvers beat 70B on IFEval and GSM8K but not HumanEval, while an older 70B evolver was worse than ChatGPT | [Hui et al.](https://arxiv.org/abs/2412.11231); [WizardLM](https://arxiv.org/abs/2304.12244) |

Rules. **(1) Pick the teacher by the student's result:** run 2–3 license-compatible teachers on 1–2K of your selected questions, do a short SFT per teacher, and compare the students on a held-out set. **Strong** (three independent reports). **(2) The teacher must actually reason** on the target distribution. **Moderate.** **(3) For small students**, prefer shorter traces, a teacher closer in size, or scaffolded RL (expert prefixes, teacher answers in the prompt; [Chapter 05](05-rl-playbook.md) §6) over plain SFT on long frontier traces. **Moderate.** **(4) Decouple roles:** a cheap, high-temperature proposer for questions and a strong answerer for traces. **Moderate.** Teacher identity also leaks into the student ([§8](#8-licensing-and-provenance-of-teacher-outputs)).

### 3.3 Rejection sampling: what to filter and what to keep

Frontier cold-start pipelines filter for form rather than for difficulty. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) filtered its cold-start data "to retain only those with correct final answers and a readable format". Qwen3 removes responses with wrong final answers, heavy repetition, guesswork, thinking–summary inconsistency, language mixing, or similarity to validation items. [ResearchMath-14k](https://arxiv.org/abs/2605.28003) removes non-attempts and fabricated citations; a "committed-attempt" prompt cut non-attempts from 22% to 0%.

Keep these filters unconditionally: truncation and overflow, repetition, language mixing, format violations, non-attempts, fabricated references, and near-duplicates of evaluation items. Treat answer correctness as a cheap filter to apply when a checker exists, not as a reason to drop hard items (§2.4). The evidence on failed traces is mixed, so ablate:

| Keep failed or unverified traces? | Result | Source |
|---|---|---|
| Behaviour-rich traces with wrong answers, for priming | Matched correct-answer priming | [Cognitive Behaviors](https://arxiv.org/abs/2503.01307); [Behavior Priming](https://arxiv.org/abs/2510.06534) |
| Incorrect solutions to harder problems (code) | Beat correct solutions to easier ones | [OpenCodeReasoning](https://arxiv.org/abs/2504.01943) |
| Unsuccessful terminal trajectories | Helped: 12.4% vs 6.74% on complete-only | [Nemotron-Terminal](https://arxiv.org/abs/2602.21193) |
| Success-only terminal trajectories (2.3k) | Worse than a same-size mixed subset: 10.1 vs 12.4 (8B) | [Terminal-World](https://arxiv.org/abs/2605.20876) |
| Verifier-failed cross-workspace trajectories | Hurt: 53.2 vs 55.4; on single-workspace data filtering hardly mattered (56.0 vs 56.4) | [Terminal-Universe](https://arxiv.org/abs/2609.04148) |
| Research-frontier trajectories, 3.7–4.3% judged correct | Still +2.1 on graduate/research math, behaviourally filtered | [ResearchMath-14k](https://arxiv.org/abs/2605.28003) |

Two trace-construction rules have good support:

- **Re-solve; don't imitate raw trajectories.** SFT on public source trajectories scored *below the base model* (36.7 vs 47.0), while re-solving the recovered intent in reconstructed environments gave 52.1 ([Terminal-Universe](https://arxiv.org/abs/2609.04148)). **Moderate.**
- **Let the teacher see privileged information that the student does not.** Generate traces with hints, execution guidelines or the answer in the teacher's context, and train on the instruction without them. Keeping execution guidelines in the training instruction hurt (13.5 vs 15.7; [Terminal-World](https://arxiv.org/abs/2605.20876)). [STaR](https://arxiv.org/abs/2203.14465)'s rationalization (hint with the answer, train without it) is the same idea; without it the loop "eventually fails to solve any new problems". **Moderate.**

### 3.4 How many traces per problem

| Evidence | Implication |
|---|---|
| [X-Coder](https://arxiv.org/abs/2601.06953): 64k×1 > 16k×4 > 8k×8 at fixed count; 200k×1 scored 60.3 on LCB vs 53.6 for OpenCodeReasoning's 28k prompts × ~26 solutions | Unique prompts first |
| [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284): unique prompts mattered more than responses per prompt (regression coefficients 4.831 vs 2.635), yet more responses per prompt still raised AIME25 from 41.3 to 49.3 | Both help; prompts help more |
| [GLM-4.5](https://arxiv.org/abs/2508.06471): 4 responses per prompt added +1–2%. OpenThoughts: 16 answers per question was "an effective scale lever" | 4–16 once prompts run out |
| [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855): more trajectories per existing description plateaued from 31.6K to 100K, while new synthetic descriptions kept helping | Generate new prompts rather than re-rolling |
| [DART-Math](https://arxiv.org/abs/2407.13690): uniform rejection sampling leaves hard queries with few or no correct traces. Difficulty-proportional sampling averages 8.49 responses per Level-1 query and 107.06 per Level-5 query, covers 99.6% of Level-5 MATH queries (vs 68% for ToRA-Corpus-16k), and took Llama3-8B from 21.2 to 46.6 on MATH | Allocate traces in proportion to difficulty |
| [OpenMathReasoning](https://arxiv.org/abs/2504.16891) gives harder problems (by pass rate over 32 samples) more solutions; [KodCode](https://arxiv.org/abs/2503.02951) gives hard questions up to 10 generation attempts instead of discarding them | Give hard items more attempts |
| [Route-diverse SFT](https://arxiv.org/abs/2609.33780): choosing verified traces by reasoning-route diversity raised OLMo3-7B's post-RL pass@8 by 16.9 points on held-out environments | Pick traces for route diversity, not shortness |

Default policy (**Moderate–Strong**; route-diversity selection alone is **Emerging**): one trace per unique question while new questions can be generated cheaply. Beyond that, 4–16 traces per question, allocated in proportion to the student's fail rate, capped, and chosen for route diversity.

### 3.5 Pseudo-code: difficulty-aware distillation

```python
def build_distillation_set(questions, student, teachers, k_min=1, k_max=16, budget=N):
    # 1. question selection (§3.1)
    qs = [q for q in questions if well_posed(q) and not solvable_without_cot(q)]
    for q in qs:
        q.fail = 1 - pass_rate(student, q, k=8)       # student-relative difficulty
        q.len = median_len(teachers.pilot, q)          # cheap hardness proxy
    qs = [q for q in qs if q.fail > 0]                 # drop what the student already solves
    qs = stratify_by_domain(sorted(qs, key=lambda q: (q.fail, q.len), reverse=True))

    # 2. teacher chosen by a student pilot, not by teacher benchmarks (§3.2)
    teacher = argmax(teachers, key=lambda t: heldout_score(sft(student, sample(t, qs[:2000]))))

    # 3. traces: unique questions first, then extra traces ∝ difficulty (§3.4)
    data = []
    for q in qs:
        k = k_min + round((k_max - k_min) * q.fail) if unique_pool_exhausted() else k_min
        cands = teacher.generate(q, n=2 * k, context=q.privileged_hints)  # hints: teacher only
        cands = [c for c in cands if complete(c) and not repetitive(c)
                 and not lang_mixed(c) and not near_eval_duplicate(c)]
        cands = answer_filter(cands, q) if cheap_checker(q) else cands     # optional (§2.4)
        data += select_route_diverse(cands, k)          # diverse reasoning routes
        if len(data) >= budget: break
    return strip_hints(data)                            # the student never sees the hints
```

---

## 4. Quality versus quantity

The literature has well-known "small curated set" results and equally solid "scale keeps paying" results. They answer different questions.

| Regime | Result | Source |
|---|---|---|
| Small, hard, curated | 1K hard questions vs the full 59K: 50.0 vs 53.3 AIME24 (not significant) on Qwen2.5-32B-Instruct | [s1K](https://arxiv.org/abs/2501.19393) |
| | 63.3% AIME24 and 95.6% MATH500 using 1% of prior approaches' training data | [LIMO](https://arxiv.org/abs/2502.03387) |
| | 500 samples on top of LIMO: +17.85 AIME25 over MetaMathQA-500, but only about 1.2 over OpenR1-Math-500 | [SAND-Math](https://arxiv.org/abs/2507.20527) |
| | 1K synthesized samples beat LIMO and s1K on eight math benchmarks across 10 models | [MathAgent](https://arxiv.org/abs/2604.11188) |
| | 4K examples reach 42.76% AlpacaEval 2.0 LC on LLaMA-3-8B-Base; performance saturates around 4K | [Instruct-SkillMix](https://arxiv.org/abs/2408.14774) |
| | 1.2% of Nemotron-Terminal's data gave +4.5 on TB2.0; 11.7k SFT samples gave 29.5% on BrowseComp | [Terminal-World](https://arxiv.org/abs/2605.20876); [OpenSeeker](https://arxiv.org/abs/2603.15594) |
| | 1.2k realistic feature-addition bugs beat 3k other bugs by 2% | [BugPilot](https://arxiv.org/abs/2510.19898) |
| Large | SFT of Qwen2.5-7B-Instruct on 4.77M trajectories over synthetic prompts only: AIME24 12.8 → 73.1, AIME25 8.0 → 65.6 | [PromptCoT 2.0](https://arxiv.org/abs/2509.19894) |
| | SFT scaled from 36K to 2.2M examples | [AceReason-Nemotron 1.1](https://arxiv.org/abs/2506.13284) |
| | Difficult 192K subset 56.6 vs all 558K 59.2 vs a random 192K 45.1 | [ScaleDiff](https://arxiv.org/abs/2509.21070) |
| | 200k×1 unique tasks 60.3 vs 25k×8 52.5 | [X-Coder](https://arxiv.org/abs/2601.06953) |
| | 12× the terminal tasks gave +6.4 on TB2.0 (52.0 → 58.4) | [NexForge](https://arxiv.org/abs/2607.14186) |
| Saturation | Code instruction data plateaued after about 6M samples; synthetic math gains plateau near 300B tokens | [Genetic-Instruct](https://arxiv.org/abs/2407.21077); [SynthLLM](https://arxiv.org/abs/2503.19551) |

How to read these together:

1. **Small curated sets elicit; large sets teach.** s1K and SAND-Math fine-tune Qwen2.5-32B models that already hold the knowledge; the small set installs the long-CoT format and behaviours. [Hase et al.](https://arxiv.org/abs/2401.06751) likewise found easy data often as good as hard data for eliciting knowledge. At 7B the same small sets did much worse: in [LIMR](https://arxiv.org/abs/2502.11886)'s Qwen2.5-Math-7B comparison, LIMO SFT (817 items) averaged 45.7 and s1 SFT (1,000) 38.0, against 58.1 for RL on 1,389 prompts. Smaller students learning long-CoT reasoning keep improving into the millions of examples. **Moderate.**
2. **Hardness selection raises value per example but does not replace unique volume.** ScaleDiff's hard third recovered most of the full set's score, and far more than a random third, but the full set still won. **Strong.**
3. **Diversity is a separate axis from difficulty.** Seed breadth, repository count and route diversity track downstream gains more reliably than difficulty ratings (§3.1, §7). **Moderate.**
4. **Quality is a floor, not a dial.** Degenerate responses, broken step structure and responses that violate the property being taught cost points at any size (§2.4); beyond that floor, more answer verification buys little in teacher distillation.

| Goal of the SFT run | Size regime | Selection priority |
|---|---|---|
| Elicit long-CoT format and behaviours in a strong base, or cold-start RL | 10³–10⁴ | Hardness + behaviour richness + diversity |
| Distil a new domain or skill into a mid-size student | 10⁵–10⁶+ unique prompts | Unique prompts, then difficulty-proportional traces |
| Refresh a saturated SFT mix | Replace the easy half | Remove easy items (Llama 4, GLM-4.5) and add hardened, diverse ones |

---

## 5. Meta-task transforms for SFT

A meta-task turns an item the model already solves into a different, harder task, usually graded by the same checker: judge a solution, find its first error, repair it, or recover from a bad prefix. These are the cheapest "harder" SFT data because they need no new problems. The RL versions (verdict items in the RL mix, failure-prefix conditioning) are in [Chapter 05](05-rl-playbook.md) §6.

| Transform | Easy seed → new task | Label source | SFT evidence |
|---|---|---|---|
| **Critique fine-tuning** | (query, noisy response) → critique | Teacher critique; keep only critiques whose verdict matches your checker | Beat plain SFT by 4–10% on six math benchmarks. Qwen2.5-Math-CFT (50K examples, 1 hour on 8×H100) matched or beat Qwen2.5-Math-Instruct (over 2M samples) ([CFT](https://arxiv.org/abs/2501.17703)). Critiques of many solutions to *one* problem gave +15% math and +16% logic in 5 GPU hours ([one-shot CFT](https://arxiv.org/abs/2506.03295)) |
| **Verdict-checked retry traces** | Own wrong and right attempts stitched with reflections: attempt → reflection → retry → correct | Every attempt graded; reflections kept only if their verdict is right | [SkillFactory](https://arxiv.org/abs/2512.04072): trained on easy Countdown-3arg only, then GRPO, it reached 25.1% on harder Countdown vs 21.2% for R1 distillation. Removing sample ordering or reflections cut OOD accuracy to 24.1% and 23.3% (from 32.0%). [S²R](https://arxiv.org/abs/2502.12853): 3.1k trial-and-error trajectories then RL took Qwen2.5-Math-7B from 51.0% to 81.6% on MATH500 |
| **First-error localisation** | Correct chain → same chain with a template-compatible error at step k; output k | Prover recomputes downstream steps and checks the injected step is non-derivable | Improved Best-of-8 reranking for Llama-3.1-8B and Qwen-2.5-7B candidates ([Counterfactual PRM](https://arxiv.org/abs/2605.02395)). Use for verifiers and PRMs |
| **Critic against a fixed generator** | Failed attempt → critique that makes a frozen generator's revision pass | Sandbox tests on the revision | [CTRL](https://arxiv.org/abs/2502.03492): SFT on execution-informed critiques, then GRPO; up to 106.1% relative improvement with iterative critique–revision |
| **Recovery splicing (agents)** | Failing prefix → reflection → verified sibling success | Environment success | [Agent-R](https://arxiv.org/abs/2501.11425): +5.59% across three environments. [AgentRefine](https://arxiv.org/abs/2501.01702): mask the loss on the erroneous turns; masking the refinement tokens instead cut SciWorld by about 43% |
| **Fault injection during trace collection** | Unchanged task; the teacher's trajectory gets injected faults to diagnose and fix | Task verifier | [TermiGen](https://arxiv.org/abs/2602.07274): error-correction trajectories beat standard expert trajectories |
| **World-model prediction** | Rollout → "predict the next observation after action a" plus contrastive reflection | Real environment transitions | [Early Experience](https://arxiv.org/abs/2510.08558): consistently beat imitation learning across eight environments and gave a stronger RL start |
| **Inversion** | Forward problem → solve for a masked given | The masked value | +2.3 / +2.6 GSM8K points ([MetaMath](https://arxiv.org/abs/2309.12284)) |
| **Search serialisation** | Optimal solution → the solver's full search with dead ends | Deterministic solver | +25% search accuracy over optimal-path traces ([Stream of Search](https://arxiv.org/abs/2404.03683)) |
| **Reverse tasks** | (x, y) → the constraints y satisfies; code → instruction | By construction | Auxiliary objective in [Crab](https://arxiv.org/abs/2410.24175); fidelity-filtered reversal in [Phi-4](https://arxiv.org/abs/2412.08905) |

Rules:

- **Self-correction must be on-policy.** SFT on offline correction traces fails because the data-collection policy's mistakes differ from the model's own, and training collapses onto one correction mode ([SCoRe](https://arxiv.org/abs/2409.12917)). SFT with synthetic error injection failed even on simple tasks, and models that caught the error often repeated it ([Wu et al.](https://arxiv.org/abs/2512.02389)). Use constructed errors to train *verifiers*. For self-correction, use the model's own failures (SkillFactory, Agent-R) followed by RL ([SPOC](https://arxiv.org/abs/2506.06923), S²R). **Moderate.**
- **Take labels from execution, not from teacher prose.** A critique's verdict should agree with your checker; discard critiques that disagree. **Moderate.**
- **Draw candidates from near-misses and from other or weaker models, and balance verdict labels.** CFT reports robustness to the source of the noisy responses. Critique-Coder deliberately used a weaker generator's solutions ([Critique-Coder](https://arxiv.org/abs/2509.22824)). **Moderate.**
- **Dose.** No SFT study sets the meta-task share. Start at 10–20% of the SFT mix (Critique-Coder used 20% in RL) and ablate. **Proposal.**

---

## 6. What SFT fails to teach, and how to sequence SFT and RL

### 6.1 The evidence

| Capability | SFT result | RL result | Source |
|---|---|---|---|
| Deeper composition of known skills | Iterative RFT on depth-2 data: 15% at depth 2, ≤ 2.6% at depth 3 | 64% at depth 2, 27% at depth 3 on the same data | [f(g(x))](https://arxiv.org/abs/2509.25123) |
| Transfer of composition to real multi-hop QA | SFT on GSM-infinity traces improved GSM-infinity but not HotpotQA | RL improved both | [Kabra et al.](https://arxiv.org/abs/2603.02091) |
| Two-hop word problems | Fine-tuning Gemma2-27B on GSM8K helped compositional accuracy for about 100 steps, then hurt it while GSM8K kept rising | — | [Compositional GSM](https://arxiv.org/abs/2410.01748) |
| Composed inequalities | SFT on about 8,000 composed AM-GM problems raised in-distribution Type I to 56%, with minimal gains on Type II and unseen transformations | — | [Ineq-Comp](https://arxiv.org/abs/2505.12680) |
| Longer horizons (more stacked objectives) | SFT on GPT-4o trajectories: 0.088 at 6 objectives, 0.000 at 16 | 1.630 at 6, 1.900 at 16 | [MEM1](https://arxiv.org/abs/2506.15841) |
| New rule variants (OOD) | −8.1 (GeneralPoints-L), −79.5 (V-IRL-L) | +3.5, +11.0; but RL without SFT failed on format | [Chu et al.](https://arxiv.org/abs/2501.17161) |
| Simulator-generated physics | SFT on 200K rejection-sampled solutions: IPhO 19.8 → 15.9 | 25.2 | [Sim2Reason](https://arxiv.org/abs/2604.11805) |
| OR autoformulation | Hard-LP 26 (base 55) | 80 | [AutoOR](https://arxiv.org/abs/2604.16804) |
| Untargeted skills | Math-only SFT (think): IFEval 69.2 → 42.3, HaluEval 35.7 → 2.3 | IFEval 70.0 | [Huan et al.](https://arxiv.org/abs/2507.00432) |

Visual pretext tasks show the same pattern: SFT on the same captions fell below base in [ViCrit](https://arxiv.org/abs/2506.10128) (48.41 vs 50.61); SFT→RL fell below base in [COGS](https://arxiv.org/abs/2510.15040) (46.62 vs 47.36); and [Jigsaw-R1](https://arxiv.org/abs/2505.23590) scored 69.48 with SFT and 69.92 with SFT→RL, against 73.18 for RL alone. RL forgets less than SFT because its data is on-policy ([Retaining by Doing](https://arxiv.org/abs/2510.18874); [RL's Razor](https://arxiv.org/abs/2509.04259)).

SFT still does things RL cannot. Distillation lifts the whole pass@k curve, while RLVR mostly sharpens within the base model's coverage ([Yue et al.](https://arxiv.org/abs/2504.13837)); "RL squeezes, SFT expands" ([Matsutani et al.](https://arxiv.org/abs/2509.21128)). On InternBootcamp, Qwen2.5-32B-Instruct's OOD average went 42.3 → 43.0 with RL alone, 53.2 with SFT and 61.8 with SFT→RL ([InternBootcamp](https://arxiv.org/abs/2508.08636)). In a standardised re-evaluation, most RL gains were modest and prone to overfitting on AIME24, while SFT generalised more consistently ([A Sober Look](https://arxiv.org/abs/2504.07086)). SFT helps most where the model lacks knowledge or format: [ChartVerse](https://arxiv.org/abs/2601.13606) went 56.9 → 62.5 (SFT) → 64.1 (RL) at 8B, and [MegaScience](https://arxiv.org/abs/2507.16812) SFT beat the official instruct model at 7B but not at 1.5B or 3B.

**Summary (Strong):** SFT imports the teacher's solutions and behaviours, which expands coverage. It learns surface procedures for the distribution it sees and does not extrapolate composition, horizon or rules. SFT on the composite pool can even damage these skills. RL on composites is what extrapolates, but only on top of primitives and formats the model already has.

### 6.2 Division of labour

[Kong et al.](https://arxiv.org/abs/2606.18089) give the cleanest protocol: SFT should cover all atomic modules *through compositional traces*, and RL should focus on novel compositions outside SFT support. [Interplay](https://arxiv.org/abs/2512.07783) sets the floor: with 0% or 0.1% pretraining exposure to a context, RL did not transfer to it, while 1% exposure gave up to +60% pass@128. Practitioners split pools the same way. Qwen3 builds its RL set from query–verifier pairs "not used during the cold-start phase". ChartVerse sends the highest-fail-rate items to RL-40K and the rest to SFT.

| Item type (after hardening) | Send to | Why |
|---|---|---|
| Atomic skills the student fails or rarely passes | SFT (distillation) | Installs primitives; RL cannot compose missing atoms |
| Moderate composites and behaviour-rich traces | SFT (cold start) | Installs format, long-CoT, verify/backtrack |
| Knowledge-heavy or format-heavy items | SFT | Where SFT reliably helps |
| Held-out composites, deeper levels, rule-shifted variants | RL only | SFT does not extrapolate here and can regress |
| Hardest verifiable band (0 < p̂ small) | RL, with the scaffolds in [Chapter 05](05-rl-playbook.md) §6 | Needs on-policy exploration |
| Unverifiable but valuable hard items (research frontier) | SFT with behavioural filters | No reliable reward; SFT still helped (§3.3) |

**Moderate** (consistent across several controlled studies, few at frontier scale); the research-frontier row is **Emerging** (one study).

### 6.3 Building the cold start

**Content.**

- Traces that *show* verification, backtracking and retrying. Priming with wrong-answer traces matched correct-answer priming, and empty or placeholder CoT did not help ([Cognitive Behaviors](https://arxiv.org/abs/2503.01307)).
- Self-distilled retry traces from easy seeds (SkillFactory used 4,000 rows).
- Correct, readable teacher traces (DeepSeek-R1), with no-CoT-solvable queries removed (Qwen3).
- Traces chosen for route diversity: +16.9 post-RL pass@8 ([Route-diverse SFT](https://arxiv.org/abs/2609.33780)).
- A mix of short-answer and long-CoT data. Compute-matched Mixed SFT had the lowest accuracy before RLVR but the highest ceiling after, using over 60× less GPU time than next-chunk RL ([Tang et al.](https://arxiv.org/abs/2608.23256)).

**Size.** Keep it light. Llama 4 did "lightweight SFT" on the harder remainder before online RL. SkillFactory (4k rows), S²R (3.1k trajectories) and [Satori](https://arxiv.org/abs/2502.02508) (about 10K format demonstrations) are all small. **Moderate.**

**Checkpoint choice: run a short RL probe; do not pick by SFT accuracy.** SkillFactory scored 2.8% vs 11.7% for R1 distillation before RL, and 25.1% vs 21.2% after. The evidence on how strong the SFT should be conflicts. [X-Coder](https://arxiv.org/abs/2601.06953) found stronger SFT initialisations made RL more effective. [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) found "undertrained" SFT models benefited more from RL, and a Qwen3-8B that was weak in the harness could not benefit at all. Only a probe settles it for your setup. **Moderate.**

**Pitfalls.**

- Long-CoT data causes verbosity and RL instability. [OctoThinker](https://arxiv.org/abs/2506.20512) mitigates this with a progressive maximum-response-length schedule.
- Small students can get worse from frontier-length traces (BREAD).
- If the primitive is absent from pretraining, a cold start may not suffice. Mid-training plus RL beat RL-only by +10.8% on OOD-hard problems at fixed compute (Interplay). Reasoning data in pretraining gave a 19% gain that later SFT could not fully recover ([Front-Loading Reasoning](https://arxiv.org/abs/2510.03264)).

InternBootcamp is the sharpest single datapoint for why the cold start matters: RL alone moved OOD by +0.7, SFT→RL by +19.5. **Strong** that a cold start is needed for hard synthetic gyms on non-reasoning bases.

### 6.4 After RL: distil back

The loop does not end at RL. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) built about 800K SFT samples by rejection sampling from its RL checkpoint. [DeepSeek-V4](https://arxiv.org/abs/2606.19348) trains specialists with SFT and GRPO and then merges them by on-policy distillation into one model; [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) uses multi-domain on-policy distillation. Two cautions:

- Consolidation does not add coverage. Merging, pooled RL and multi-teacher on-policy distillation differed by at most 1.4 points on average, and none beat the base model at AIME pass@32 ([Consolidating RLVR](https://arxiv.org/abs/2608.27409)).
- Off-policy SFT forgets more. Approximately on-policy data regenerated at the start of each epoch can suffice ([Retaining by Doing](https://arxiv.org/abs/2510.18874)). In finance, ordinary SFT lowered FINESSE-Bench by 3.2–4.0 points, while self-distilled SFT raised it by 1.0–2.8 ([Hayrapetyan et al.](https://arxiv.org/abs/2609.10113)).

```
 cold start (small, behaviour-rich, route-diverse)       atoms + moderate items
        │                                                  (distillation SFT)
        ▼                                                          │
 RL probe on held-out composites ──► pick checkpoint ◄─────────────┘
        │
        ▼
 RL on composites / hardest band (Ch 05) ──► rejection-sample from the RL policy
        │                                         │ (near-on-policy, strict gate: §2.4)
        └──────────── regression suite ◄──────────┘──► next SFT round / on-policy distillation
```

---

## 7. Mixing and dose

Hardened synthetic data should not be the whole SFT mix. Most blend evidence comes from pretraining or RL, so the shares below are a starting point to ablate, not settled values.

| Component | Starting share | Evidence |
|---|---|---|
| Hardened synthetic items, equal across operator families | The remainder | In RL, no adaptive mixture beat a fixed equal mix across 12 matched seeds ([DataFlex-RL](https://arxiv.org/abs/2609.06107)); untested for SFT. **Proposal** |
| Real or seed items with trusted answers | 20–33% | Seed+synthetic 57.3 > seed-only 49.7 > synthetic-only 46.8 on LiveCodeBench at 7B ([rStar-Coder](https://arxiv.org/abs/2505.21297)). In pretraining, 1/3 rephrased + 2/3 natural text reached the same loss 5–10× faster, and good rephrased ratios converged to about 30% ([Kang et al.](https://arxiv.org/abs/2510.01631)). Mixing ResearchMath (77K) with DASD (50K) beat token-matched DASD by +0.8 to +5.7 across four benchmark groups. **Moderate** |
| General instruction data for untargeted skills | Enough to hold the regression suite flat | Math-only SFT cut IFEval from 69.2 to 42.3 (Huan et al.). Uniform-format synthetic QA (2% of 300B continued-pretraining tokens) lowered FollowBench HSR from 27.58 to 24.00 after SFT; unlearning restored 27.87 ([Chen et al.](https://arxiv.org/abs/2406.12397)). [Conifer](https://arxiv.org/abs/2404.02823) mixed 13,606 hard conversations with 53K ShareGPT. **Moderate** |
| Meta-task slices (§5) | 10–20% | **Proposal** |
| Rewrites per seed | ≤ 10 | At fixed size, few-seed rewriting (from 999 rewrites of 0.1% of GSM8K down to 9 rewrites of 10%) lost diversity, and diversity tracked downstream SFT accuracy ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)). **Moderate** |
| Generator and teacher families | 2–3, from different model series | Judge preference leakage: 23.6% when the judge generated the data vs 2.8% across series, and SFT leaks most (23.6% vs 5.2% for DPO) ([Preference Leakage](https://arxiv.org/abs/2502.01534)). Students SFT'd on different teachers are 98.9% separable ([Idiosyncrasies](https://arxiv.org/abs/2502.12150)). **Moderate** |

**Mix difficulties; don't stage them.** A mixed single-stage schedule beat two-stage curricula for terminal SFT ([Nemotron-Terminal](https://arxiv.org/abs/2602.21193)). [Mordig et al.](https://arxiv.org/abs/2603.27226) found no robust gain from easy→hard ordering for SFT or RL. Tree-Instruct's hard-only set beat its curriculum, and Conifer's easy→hard multi-turn packaging gained only 1–2 points in a single run. Keep the whole difficulty ladder in the mix (§2.3); ordering is second-order. **Moderate.**

**Run a dose ladder before scaling.** Size regimes are in §4. Per operator family, train at 1k / 4k / 16k / 64k items with at least 3 seeds each, and stop where the held-out curve flattens. Instruct-SkillMix saturated at about 4K and Genetic-Instruct at about 6M, so the knee differs by orders of magnitude between elicitation and distillation. **Proposal.**

**Measure with enough statistical power.** At p ≈ 0.5, an unpaired comparison on 30 items needs roughly a 25-point gap to reach significance. Use paired per-item analysis with 10–30 samples per item, cluster standard errors by seed family, split train and eval by seed rather than by variant, and pre-register a domain-balanced aggregate ([Chapter 05](05-rl-playbook.md); [Measuring all the noises](https://arxiv.org/abs/2512.21326); [Adding Error Bars](https://arxiv.org/abs/2411.00640)). **Strong.**

---

## 8. Licensing and provenance of teacher outputs

Distillation turns a teacher's license into your obligation. The terms below were read from primary files in September 2026 (note 23). They are a map for engineers, not legal advice.

| Teacher or generator | Terms for training on its outputs |
|---|---|
| DeepSeek-R1 / V3.2, GLM-4.5 / 4.7 / 5 | MIT. The R1 release says "Distill & commercialize freely!" ([DeepSeek-R1 release](https://api-docs.deepseek.com/news/news250120)) |
| Qwen3 / 3.5, gpt-oss, Olmo 3, Gemma 4 | Apache-2.0 |
| Kimi K2 / K2.5, MiniMax-M2 | Modified MIT: products above 100M MAU or a revenue threshold must display the model name in the UI ([Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/LICENSE); [MiniMax-M2](https://github.com/MiniMax-AI/MiniMax-M2/blob/main/LICENSE)) |
| Llama 3.1 / Llama 4 | A distributed model trained on Llama outputs must put "Llama" at the beginning of its name, plus "Built with Llama" and the 700M-MAU clause ([Llama 3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE); [Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)) |
| Gemma 1–3 | A model trained on Gemma synthetic outputs is a "Model Derivative": pass on the use restrictions, a copy of the terms and a Notice file ([Gemma Terms](https://ai.google.dev/gemma/terms)) |
| Qwen2.5-72B-Instruct / Qwen2.5-3B-Instruct | "Built with Qwen" or "Improved using Qwen" plus a 100M-MAU clause; the 3B model is research-only ([Qwen License](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE); [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)) |
| NVIDIA Nemotron models | No claim on outputs; derivatives carry a license notice ([Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/)) |
| Anthropic, Gemini API, OpenAI | Terms bar using the services to build or train competing models ([Anthropic D.4](https://www.anthropic.com/legal/commercial-terms); [Gemini API](https://ai.google.dev/gemini-api/terms)); the OpenAI clause was not re-verified ([OpenAI ToU](https://openai.com/policies/terms-of-use/)) |

**Dataset cards are hints, not lineage.**

- On 2026-09-30 the Hub API showed no license tag on Skywork-OR1-RL-Data, SYNTHETIC-2-RL and INTELLECT-3-RL, among others. KodCode-V1 and facebook/natural_reasoning are CC-BY-NC-4.0, and ResearchMath-14k is CC BY-SA 4.0 per its paper.
- The [Llama-Nemotron card](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) warns about Llama licenses but not about its 464,658 Qwen-2.5-72B-Instruct rows.
- Audits find hosting sites omit licenses 70%+ of the time and miscategorise them 50%+ of the time. The [Data Provenance Initiative](https://arxiv.org/abs/2310.16787) cites WizardCoder as licensed for commercial use while trained on commercially restricted OpenAI outputs. 35.4% of LLMware artifacts declare no license, and 52% of supply chains contain a conflict ([LLMware](https://arxiv.org/abs/2602.10758)).

**Regulation now covers distillation data.** Section 2.5 of the EU AI Act training-content template requires naming the GPAI models used to generate synthetic training data, explicitly including distillation and "AI feedback through reinforcement learning". AI Office enforcement started on 2 August 2026 ([EU template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models)).

**Teacher footprints are detectable and carry traits.** Part-of-speech templates identify a student's teacher where n-grams fail ([Who Taught You That?](https://arxiv.org/abs/2502.06659)). A teacher sharing the student's base model transmitted behavioural traits through correct-filtered CoT ([Subliminal Learning](https://arxiv.org/abs/2507.14805)), so avoid using the student's own siblings as the sole teacher.

**Exit route.** Regenerate a non-permissive pool with a permissive generator ("license transplant"), then re-verify it and keep per-category acceptance rates ([DFM Mimir](https://arxiv.org/abs/2608.13517)). Its performance-parity claim lacks a matched ablation, so run one.

Per-row schema, extending NVIDIA's `license` and `generator` fields ([Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)):

```json
{"id": "...", "seed_source": "gsm8k/train#1234", "seed_license": "MIT",
 "operators": ["sequential_fusion", "inversion"], "evolver": "Qwen3-32B@rev",
 "teacher": "DeepSeek-R1-0528", "verifier": "sympy-equiv-v2", "judge": null,
 "derived_license": "most-restrictive-parent", "decontam": {"13gram": "pass"},
 "student_pass_rate@8": 0.25, "created": "2026-09-30"}
```

Rules (**Strong** as compliance hygiene):

- Default to MIT/Apache teachers and evolvers.
- Keep closed-API models for evaluation unless a contract allows training.
- Never label derived data more permissively than its most restrictive parent.
- Log judges and reward models as well as generators.

---

## 9. Step-by-step recipe

Each step ends with a gate; do not proceed until it passes.

1. **Define SFT's job.** Is it a cold start for RL, a distillation product, or a meta-skill? This sets the size regime (§4) and the verification level (§2.4). *Gate:* a written split of which item types go to SFT and which to RL (§6.2).
2. **License-screen seeds, evolvers and teachers.** Adopt the row schema in §8. *Gate:* every source has a traced license, and no restricted teacher sits in the path of a commercial release.
3. **Profile seeds with the student** at k = 8–16 samples. Remove items solvable without CoT, convert MCQ to free-form, and decontaminate ([Chapter 01](01-diagnosis-difficulty-and-learning-signal.md), [Chapter 04](04-verification-and-quality-control.md)). *Gate:* a pass-rate histogram per seed family.
4. **Harden for 2–3 rounds** with correctness-preserving operators (§2.2): at most 20 new words per step, well-posedness checks, an "INVALID" escape, ≤ 10 rewrites per seed, and diversity selection. Keep every rung and route overshoot downward. *Gate:* the dev-set "Evol stop" (§2.3), plus a hand audit of 100–200 items. Zero errors in 200 puts the 95% Wilson upper bound on the error rate at 1.9%.
5. **Split the pool.** Held-out composites, deeper levels and the hardest verifiable band go to RL. Atoms, moderate items and knowledge- or format-heavy items go to SFT. *Gate:* no RL prompt appears in SFT.
6. **Select SFT questions** by student failure, teacher trace length or LLM difficulty, stratified by domain (§3.1). *Gate:* the selected set beats a random set of equal size in a short pilot.
7. **Choose the teacher by pilot:** 2–3 license-compatible teachers × 1–2K questions, a short SFT each, then a held-out student evaluation (§3.2). *Gate:* the choice rests on student scores, not teacher benchmarks.
8. **Generate traces.** One per unique question at first; add 4–16 per question in proportion to fail rate once unique prompts run out. Put privileged hints in the teacher's context only; filter truncation, repetition, language mixing and non-attempts; select for route diversity (§3.3–3.5). Answer-filter only where a checker is cheap; for self-distillation use strict gates. *Gate:* the format-violation rate is below your threshold on a 200-item read of raw generations.
9. **Add meta-task slices** (10–20%): verdict-checked critiques, retry traces, recovery splices and first-error items for verifier training (§5). *Gate:* verdict labels are balanced and agree with execution.
10. **Assemble the mix:** equal shares across operator families, a 20–33% real anchor, general data to protect untargeted skills, and 2–3 generator families (§7). *Gate:* a dose ladder with ≥ 3 seeds shows where gains flatten.
11. **Train, then pick the checkpoint with a short RL probe** on held-out composites, not by SFT accuracy (§6.3). *Gate:* the regression suite (IFEval, hallucination, long context, other domains) does not drop.
12. **Run RL on the RL split** ([Chapter 05](05-rl-playbook.md)). Then rejection-sample from the RL policy behind a strict gate and distil back, near-on-policy (§6.4). *Gate:* paired, multi-seed gains on post-cutoff and held-out-generator evaluations.

```python
pool   = harden(profile(license_ok(seeds), student, k=16), max_rounds=3, stop=dev_eval)
sft_q, rl_q = split_by_role(pool)                       # §6.2; disjoint
teacher = pick_by_student_pilot(license_ok(teachers), sft_q[:2000])
sft    = distill(sft_q, teacher, k=difficulty_prop, hints="teacher_only", gate="format")
sft   += meta_tasks(solved_items, frac=0.15) + real_anchor(frac=0.25) + general(frac=needed)
ckpt   = argmax(sft_ckpts(sft), key=lambda c: rl_probe(c, rl_q.heldout))
policy = rl(ckpt, rl_q)                                  # Ch 05
next_sft = rejection_sample(policy, sft_q + rl_q, gate="strict")   # distil back
```

---

## 10. Open questions

- **Operator ablations for SFT at matched compute.** Most comparisons (MetaMath, FLAMES, Tag-Evol) use different seeds, teachers and budgets. It remains open which hardening operators transfer to which downstream skills once the teacher is held fixed.
- **Learning from unverifiable frontier attempts.** ResearchMath SFT helped with 3.7–4.3% judged-correct traces. When does such data help or hurt RL initialisation, and can partial checks (checkpoints, lemmas) make it verifiable?
- **Making constructed errors look on-policy.** Injected errors train verifiers but not self-correction. Selecting injections by policy likelihood is untested.
- **How strong the cold start should be.** X-Coder and OpenThoughts-Agent disagree, and no predictor of RL-readiness short of an RL probe exists.
- **Teacher tracing under heavy rewriting.** Whether complexification or license transplants erase teacher footprints matters for both compliance and contamination analysis.
