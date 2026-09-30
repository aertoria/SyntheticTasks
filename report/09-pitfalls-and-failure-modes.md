# Pitfalls and failure modes (and how to guard against each)

> **Key takeaways**
> - **Many "hard" synthetic items that fail are broken, not hard**: unsolvable, ambiguous, mislabeled, or rejected by a faulty verifier. After 5 rounds of constraint addition, IFDecorator had 10,772 unsolvable prompts vs 7,324 usable. Treat every pass-rate-0 item as a bug report until a solvability certificate says otherwise.
> - **Measure difficulty on the current policy, in the exact training harness, conditional on independent validity.** Constraint counts, length, LLM-rated complexity and "make it harder" prompts are weak proxies; in one controlled RLVR study 64% of rejected mutations were too easy.
> - **Hardening widens the attack surface, so verifier gaming rises with difficulty** (shortcuts: 40 at complexity levels 1–10 vs 458 at 11–20). Every operator needs a hack audit before RL: read-only or out-of-process grading, invariance checks, trip-wires, trivial-agent baselines.
> - **Closed loops degrade silently**: pseudo-label accuracy 79% → 63%, ~74% of tasks on one topic, a proposer's valid-question rate falling to 0. Ground labels outside the model, keep real anchors every round, measure novelty against a persistent archive, and stop after ~3 rounds unless a held-out metric is still rising.
> - **Many claimed data-policy wins are noise, contamination or Qwen2.5-Math artifacts.** Decide with paired, seed-clustered statistics over 8–12 training seeds, a random-reward control, post-cutoff items and at least one non-Qwen family.
> - **Costs blow up through yield, not unit price**: at 5% yield each kept task costs ~20 candidates plus their profiling rollouts. Audit the verifier and reallocate rollouts before generating anything new.

**Contents**
1. [How to use this chapter](#1-how-to-use-this-chapter)
2. [Part A: the item is not what you think it is](#part-a--the-item-is-not-what-you-think-it-is) (F1–F5)
3. [Part B: the learner learns the wrong thing](#part-b--the-learner-learns-the-wrong-thing) (F6–F9)
4. [Part C: the loop degrades over time](#part-c--the-loop-degrades-over-time) (F10–F13)
5. [Part D: you believe a result that is not real](#part-d--you-believe-a-result-that-is-not-real) (F14–F17)
6. [Part E: the bill](#part-e--the-bill) (F18)
7. [Training dashboard: metrics and alarms](#7-training-dashboard-metrics-and-alarms)
8. [Pre-flight checklist](#8-pre-flight-checklist)

---

## 1. How to use this chapter

Each failure mode is written as **Symptom → Root cause → Evidence → Detect → Mitigate**. Recommendations carry evidence tags: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (single recent or unreplicated result), **Proposal** (our synthesis, untested). Mechanisms are covered in depth in [chapter 04](04-verification-and-quality-control.md) (verification), [chapter 05](05-rl-playbook.md) (RL control), [chapter 06](06-sft-playbook.md) (SFT) and [chapter 02](02-complexification-operator-taxonomy.md) (operators); this chapter links rather than repeats. Where each failure enters the pipeline:

```
easy seeds ─► [operator] ─► candidate ─► [validity gates] ─► [difficulty profiling] ─► pool ─► [SFT / RL] ─► [eval & decision]
              F1 F2 F3       F6            F4 F5 F7             F1 F2 F18                F10-F13   F8 F9 F11     F14 F15 F16 F17
              (proxy hard,   (artifacts,   (wrong keys,         (policy-relative,        (collapse, (composition, (contamination,
               overshoot,     leaks)        consensus caps,      misfiled p=0,             drift,     forgetting,    Qwen confound,
               ambiguity)                   hackable checks)     yield)                    spirals)   entropy)       noise, family)
```

| # | Failure mode | Bites | Cheapest detector | First guard | Evidence |
|---|---|---|---|---|---|
| F1 | Complexity ≠ difficulty | SFT+RL | Per-operator pass-rate histogram on the current policy | Accept a mutation only if pass rate drops | Strong |
| F2 | Overshoot to unsolvable | RL | Share of items at p = 0 per operator, plus a large-k re-check | Stop hardening at pass@8 ∈ (0, 0.5]; require a solvability certificate | Strong |
| F3 | Ambiguity masquerading as difficulty | RL | Privileged-information test; enumerate alternative answers | Fix the answer first; run a uniqueness check after every obfuscation step | Strong |
| F4 | Wrong labels and verifiers on hard items | RL | Gold audit slice bucketed by difficulty; TPR/TNR of the verifier | Correct by construction; decorrelated evidence | Strong |
| F5 | Consensus filters delete the frontier | RL | Difficulty histogram before and after each filter | Answer-first construction; verify only the new increment | Moderate |
| F6 | Generator artifacts and shortcuts | SFT+RL | Partial-input, no-CoT and no-tool baselines | External RNG, leak scans, rewriting templated items | Strong |
| F7 | Reward hacking of verifiers | RL | Reward jumps, length alarms, trip-wires, trivial agents | Hidden or read-only out-of-process grading; invariance checks | Strong |
| F8 | SFT-only composition failures | SFT | Held-out composition splits vs base | SFT for atoms, RL for composites | Strong |
| F9 | Specialist traps and forgetting | SFT+RL | Regression suite; correct-set turnover | Domain mixing, review queue, reward-model gate | Moderate |
| F10 | Diversity and topic collapse | RL loops | Top-topic share; novelty against a persistent archive | Skill-level dedup; populations | Moderate |
| F11 | Entropy collapse and boundary shrinkage | RL | Entropy; pass@k at large k vs base | Retire and review; answer-preserving variants; KL | Moderate |
| F12 | Proposer death spirals | Self-play | Valid-proposal rate; proposer entropy | Zero (not negative) reward for malformed proposals; multiplicative validity gate | Moderate |
| F13 | Iterative self-synthesis drift | Loops | Gold-slice accuracy per round | Accumulate data, keep anchors, cap at ~3 rounds | Strong |
| F14 | Contamination | Eval | Semantic decontamination of *outputs*; partial-prompt probes | Operator discovery disjoint from evals; post-cutoff sets | Strong |
| F15 | Qwen spurious-reward confound | Eval | Random-reward arm | Second model family; fresh procedural control | Strong |
| F16 | Eval noise and false conclusions | Decisions | Paired per-item CI across seeds | Pre-registered protocol with 8–12 seeds | Strong |
| F17 | Overfitting to one generator family | SFT+RL | Generator-ID classifier; held-out generator family | 2–3 generator families from different series | Moderate |
| F18 | Cost blowups | Budget | Cost per accepted in-band task, per operator | Audit, reallocate and recycle before generating | Moderate |

---

## Part A — The item is not what you think it is

### F1. Complexity ≠ difficulty

**Symptom.** The evolved pool *looks* harder (longer prompts, more constraints, higher LLM-rated difficulty), but the policy's pass-rate histogram barely moves, GRPO still sees mostly all-correct groups, and held-out scores stay flat.

**Root cause.** The operator optimizes a proxy (length, constraint count, judge score, graph structure), not the current policy's failure rate. "Make it harder" prompts drift into surface edits and extra constraints; some knobs add tedium or tool-computable work; difficulty also depends on the harness.

**Evidence.**
- **Complexity is not difficulty.** IFDecorator states that "complexity alone does not determine difficulty"; constraint count correlates only loosely with measured difficulty ([IFDecorator, 2025](https://arxiv.org/abs/2508.04632)).
- **Deep structure is not realized difficulty.** On the same agent, InfoSeek's "deep" trees needed about 20.6 retrieval calls, with the answer first seen at step 5.7. FORT's questions needed 141.0 calls, with the answer first seen at step 46.9 ([FORT](https://arxiv.org/abs/2606.12087)).
- **Rewriting that ignores the policy hurts.** Evol-Instruct prompts trained worse than unchanged ones (50.24 vs 50.51; [LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412)); unfiltered "complex question" augmentation cut BIRD-dev 64.9 → 62.5 ([Arctic](https://arxiv.org/abs/2505.20315)); text-only hardening rewrites did not beat untouched descriptions at fixed size ([OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)).
- **Asking for "hard" yields easy survivors.** 64% of rejected mutations were too easy ([Trading Human Curation](https://arxiv.org/abs/2606.03800)); "Hard" proposals overlapped heavily with "Easy" ones ([PSV](https://arxiv.org/abs/2512.18160)); generation success fell to 11% while evaluation success stayed at 48% ([AgentSynth](https://arxiv.org/abs/2506.14205)).
- **"Validates" is not "learnable."** 81% of validated candidates did not separate strong from weak solvers, mostly because both passed ([CalibForge](https://arxiv.org/abs/2608.06352)).
- **The harness changes difficulty.** A 4B model went from 8.3% to 37.2% just by changing harness ([FrogNano](https://arxiv.org/abs/2609.07925)).
- **Cheap hardness.** BBEH's reference model solved Boolean Expressions by running Python, so operands were moved into world-knowledge statements ([BBEH](https://arxiv.org/abs/2502.19187)); Tower of Hanoi move lists can exceed output-token limits ([Lawsen](https://arxiv.org/abs/2506.09250)).

**Detect.** Compare each operator's pass-rate histogram (k ≥ 8, current policy, RL harness) before and after. For agents, log realized solving cost (tool calls, step of first answer hit). Run a tool-access ablation and check output length against the context budget.

**Mitigate.**
- Define difficulty as the current policy's pass rate (**Strong**; see [chapter 01](01-diagnosis-difficulty-and-learning-signal.md)).
- Accept a mutation only if its measured pass rate falls below the parent's, as in BenchEvolver's solution-first loop (**Moderate**; [BenchEvolver](https://arxiv.org/abs/2606.01286)).
- Harden the executable artifact, not the prose. In RST, instructions grew ×1.4 while solutions grew ×5.6 (**Moderate**; [RST](https://arxiv.org/abs/2608.05466)).
- Prefer operators that remove information, invert, compose or change the method ([chapter 02](02-complexification-operator-taxonomy.md)).

### F2. Overshoot to unsolvable

**Symptom.** After a few rounds of hardening, mass piles up at p = 0. Zero-advantage groups multiply and rollouts are wasted. Response length or accuracy sometimes collapses right after a new family is added.

**Root cause.** Operators compound. Constraint sets become mutually incompatible, knobs move into regions where no solution exists, and the generator cannot tell when it has broken the task.

**Evidence.**
- **Hardening that ignores solvability breaks tasks.** IFDecorator: 10,772 pass-0 vs 7,324 in-band after 5 rounds ([IFDecorator](https://arxiv.org/abs/2508.04632)).
  - Random 12-constraint sets: about 98% are incompatible, against about 32% at k = 4 ([CSE](https://arxiv.org/abs/2608.12426)).
  - Lowering OpenSIR's solve-rate floor from 0.5 to 0.1 cut validity from 70.8% to 42.3% and math accuracy from 29.6 to 26.0, while GPT-5's solve rate only moved from 89.8% to 78.3% ([OpenSIR](https://arxiv.org/abs/2511.00602)).
- **Knobs create impossible instances.** River Crossing with boat capacity 3 has no solution for N ≥ 6 ([Lawsen](https://arxiv.org/abs/2506.09250)); 11–23% of initial LLM-built puzzles were unsolvable ([AutoLogi](https://arxiv.org/abs/2502.16906)); environments at <3% accuracy mostly had semantic errors ([InternBootcamp](https://arxiv.org/abs/2508.08636)).
- **Some hard items do damage.** Hard@8 items (pass@8 = 0) lowered averages by 5.75, 11.24 and 1.07 points in three settings, and one sample whose reward accepted a bare boxed answer collapsed mean length from 510.7 to 45.7 tokens within 58 steps ([2605.28388](https://arxiv.org/abs/2605.28388)). 1–3 buggy task types out of 50 collapsed training ([UltraLogic](https://arxiv.org/abs/2601.03205)).
- **Band control beats "harder".** On IMO-50, easy-only data scored 29, hard-only 24, the same data unscheduled 38 and controller-scheduled 44 ([InternGeometry](https://arxiv.org/abs/2512.10534)).
- **Small-k p = 0 is also misfiled the other way.** 10.3–22.9% of pass@6 = 0 math items were reachable with perturbed deterministic decoding ([Hard or Just Unreached?](https://arxiv.org/abs/2606.19636)).

**Detect.**
- Track the p = 0 share per operator per round.
- Triage every p = 0 item:
  1. large-k solvability (pass@100 > 0, as in [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556));
  2. a stronger or tool-assisted solver;
  3. a hint-conditioned attempt: solved with the hint, unsolved without it ([CLI-Universe](https://arxiv.org/abs/2606.22883)).
- Unit-test generators for level-vs-solve-rate monotonicity; alarm on length collapse after a new family is admitted.

**Mitigate.**
- Harden until pass@8 ∈ (0, 0.5], and send 0/8 items back for regeneration (**Strong**; IFDecorator plus the band literature in [chapter 05](05-rl-playbook.md)).
- Run a compatibility checker or produce a witness response for stacked constraints.
- Keep 0-pass items only with a certificate; otherwise keep a small tail (Cascade 2 keeps 10%; [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220)).
- *Defer*, don't delete, after one small pilot; scaffold valid p = 0 items ([chapter 05](05-rl-playbook.md)); quarantine a family whose zero-pass rate or length profile is anomalous (**Moderate**).

### F3. Ambiguity masquerading as difficulty

**Symptom.** Low-pass-rate items whose failures are diverse but defensible; strong models disagree, reviewers dispute the key, RL reward is noisy, and the policy drifts toward hedging or guessing.

**Root cause.** Obfuscation, clue removal, spec degradation and distractor insertion delete the information that made the answer unique, or add a second valid answer; "keep what strong models fail" filters then *enrich* such items.

**Evidence.**
- **Adversarial selection enriches bad items.** HLE reports 15.4% expert disagreement on its public set ([HLE](https://arxiv.org/abs/2501.14249)), and HLE-Verified kept only 668 of 2,500 items unchanged ([HLE-Verified](https://arxiv.org/abs/2602.13964)).
- **Filters based only on model failure keep broken items.** Keeping items that GPT-4o with search failed 4 of 4 times also keeps broken, ambiguous and unanswerable ones ([DeepDive](https://arxiv.org/abs/2509.10446)).
- **Adding candidates adds second answers.**
  - Option expansion produced 1,953 correct-but-labelled-wrong options in MMLU-Pro's MMLU part ([MMLU-Pro](https://arxiv.org/abs/2406.01574)).
  - Non-unique answers passed a RAG answerability check ("Temptations singer" → Otis Williams) ([SSP](https://arxiv.org/abs/2510.18821)).
- **Underspecification.**
  - After full Code-as-Task filtering, the residual false negatives came from incomplete instructions ("return one of my latest orders") ([SCA](https://arxiv.org/abs/2506.01716)).
  - A contract-validity gate plus an instruction audit cut weakly grounded tasks from 32.8% to 1.2% ([RST](https://arxiv.org/abs/2608.05466)).
- **Accuracy drops are not proof of difficulty.** RIDE notes that a drop can come from ambiguity ([RIDE](https://arxiv.org/abs/2511.04120)). BenchEvolver's evaluator explicitly rejects "false difficulty": ambiguous wording, misleading I/O and underspecified constraints ([BenchEvolver](https://arxiv.org/abs/2606.01286)).

**Detect.**
- *Privileged-information test:* keep a task only if solved with the gold evidence or hint and failed without it (CLI-Universe; oracle-evidence checks in [chapter 07b](07b-domain-recipes-agents-and-beyond.md)).
- Ask a reader to list *all* valid answers; triangulate a statement-only brute-force solver against the reference; sweep for false negatives whenever candidates are added.

**Mitigate.**
- Fix the answer or state first, harden around it, and verify only the new increment (**Strong**, [chapter 03](03-generation-architectures.md)).
- Pair every fuzzing step with a uniqueness check. Route flagged items to repair or an explicit "unanswerable" split (a 10% mix restored refusal; [SUM](https://arxiv.org/abs/2505.13988)); when labelling, discard rather than guess ([DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)).

### F4. Wrong labels and wrong verifiers on hard items

**Symptom.** RL reward rises while held-out accuracy stalls. Hand audits find wrong keys concentrated in the hardest bucket. Or the "hard" items turn out to be correct answers the checker rejected.

**Root cause.** Self-consistency labels degrade exactly where tasks get hard; same-family agreement is correlated evidence; rule checkers mishandle unusual answer forms; tests are too weak (false positives) or wrong (false negatives).

**Evidence.**
- **Self-labels decay with difficulty.**
  - R-Zero's pseudo-label accuracy fell from 79.0% to 69.0% to 63.0% over iterations ([R-Zero](https://arxiv.org/abs/2508.05004)).
  - The unverified majority answer is wrong on 25.85% of MATH-500, 46.07% of AMC and 73.33% of AIME 2024 questions ([T³RL](https://arxiv.org/abs/2603.02203)).
- **Plausible wrong keys do the most damage.** MATH500 reached 73.4 with the correct label, 64.4 with an unguessable wrong label, and 57.0 with a guessable wrong label ([1-shot RLVR](https://arxiv.org/abs/2504.20571)).
- **False negatives.**
  - Rule-based math checkers average 86% recall ([Huang et al.](https://arxiv.org/abs/2505.22203)).
  - TinyV found false negatives in more than 38% of responses on Big-Math-RL-Verified ([TinyV](https://arxiv.org/abs/2505.14625)).
  - More than 4,000 CodeContests problems had TPR ≤ 0.1 ([CodeContests+](https://arxiv.org/abs/2506.05817)).
- **False positives.**
  - TACO tests have a false-positive rate above 90% on hard problems ([HardTests](https://arxiv.org/abs/2505.24098)).
  - About 1 in 5 accepted patches were wrong ([SWE-ABS](https://arxiv.org/abs/2603.00520)).
- **Noise mostly slows learning, as long as the verifier beats chance.** With J = TPR − FPR > 0, the incorrect mass dies out; with J < 0 it grows ([RLVεR](https://arxiv.org/abs/2601.04411)).
- **SFT tolerates noise; RL does not.** For SFT, no answer filtering averaged 41.9 and GPT-verification filtering 40.0 ([OpenThoughts](https://arxiv.org/abs/2506.04178)). In RL, every wrong key becomes a wrong reward.

**Detect.** A gold audit slice bucketed by difficulty; verifier TPR/TNR on known-good and known-bad solutions; metamorphic tests where equivalent rewrites must get the same verdict ([Where the Verifier Fails](https://arxiv.org/abs/2609.01354)); IRT flags to prioritize hand audits (95% precision in the top 200; [Land & Bikel](https://arxiv.org/abs/2605.30504)). Size audits by binomial arithmetic: 0 errors in 200 items bounds the rate below about 1.9%.

**Mitigate.**
- Build correctness in by construction. Solution-first evolution gave 97.7% validity against 79.3% for problem-first ([BenchEvolver](https://arxiv.org/abs/2606.01286)) (**Strong**).
- Decorrelate evidence across model families, modalities and tools instead of adding more votes from the same model. VStress's conditional gain was 0.0126 for same-model repeats and 0.0913 for cross-family channels ([VStress](https://arxiv.org/abs/2609.36958)) (**Moderate**).
- Prefer answer-preserving operators when labels are uncertain. Under GRPO, a wrong *inherited* label gives all-zero rewards and so no update, whereas a wrong *new* majority label rewards wrong answers ([MathForge](https://arxiv.org/abs/2601.20614)) (**Moderate**).
- Spend verification budget on RL pools and question selection on SFT pools ([chapter 04](04-verification-and-quality-control.md)).

### F5. Consensus filters that delete the frontier

**Symptom.** Each quality filter makes the pool easier. The "hard" pipeline outputs items your policy already solves, and hard-tier yield approaches zero.

**Root cause.** Acceptance rules that require agreement cap difficulty at the labeller's own frontier. Examples: all-k agreement, same-family majority, "the student must sometimes agree", and "a reference solver must solve it". Lifting pipelines also succeed more often on easy seeds.

**Evidence.**
- **Agreement filters remove frontier items by design.** Examples are CoT-Self-Instruct's Answer-Consistency, SAND-Math's all-k agreement, DeepMath's 3-way unanimity and MindLoom's all-wrong exclusion (cross-paper synthesis in the math notes; [SAND-Math](https://arxiv.org/abs/2507.20527), [DeepMath-103K](https://arxiv.org/abs/2504.11456)).
- **Agreement labelling "inherently biases us toward easier queries"** ([DataMind](https://arxiv.org/abs/2509.25084)). A notebook-mined generator drifted to trivial questions: DABStep-easy reached 75% while the hard split stayed near 3% ([Jupyter Agents](https://huggingface.co/blog/jupyter-agent-2)).
- **Student-agreement rules skew toward what the student can already solve.** SwS requires the student to produce the teacher's answer in at least 25% of responses, which "biases toward problems the student can already partly solve" ([SwS](https://arxiv.org/abs/2506.08989)).
- **Lifting skews easy.**
  - ATLAS succeeds on 47.1% of EASY TACO seeds against about 20% of HARD ones ([ATLAS](https://arxiv.org/abs/2512.10173)).
  - rStar-Coder had to lower its agreement threshold from 60% to 40% for seeds rated above 1600 ([rStar-Coder](https://arxiv.org/abs/2505.21297)).
- **Nuance:** majority vote can still reward correctly when wrong answers scatter (37% label but 92% reward accuracy on AIME 2024), but not when one wrong answer dominates ([TTRL](https://arxiv.org/abs/2504.16084)).

**Detect.** Plot the current-policy pass-rate histogram before and after each filter, and log each filter's rejection rate per difficulty bucket; acceptance that falls steeply with difficulty is the signature.

**Mitigate.**
- Construct the answer first, from a program, graph or planted certificate, so that correctness never requires solving the hard direction (**Strong**).
- Verify only the new hop or increment, not the whole composite (TaskCraft, WebShaper, [CHASE](https://arxiv.org/abs/2502.14678)) (**Moderate**).
- Use a stronger or different-family verifier, send disagreements to tool-based relabelling instead of discarding them, and oversample hard seeds in lifting pipelines. Keep a separately audited hard-tier acceptance path: lower agreement threshold plus independent evidence (**Proposal**).

---

## Part B — The learner learns the wrong thing

### F6. Generator artifacts, signatures and shortcut learning

**Symptom.** Scores on synthetic evaluations far exceed natural ones. Partial-input baselines beat chance. Answer positions or magnitudes are skewed. The policy "solves" hard items with a format cue, leaked hint, literal-match lookup, prior knowledge or a non-target tool channel.

**Root cause.** LLM generators are poor random samplers and have strong stylistic fingerprints. The hardening model sees the answer and leaks it. Templates repeat. The environment offers a cheaper path than the skill you meant to train.

**Evidence.**
- **LLM generators are not random samplers.** LLM MCQ generators put the correct answer first 47.9–57.9% of the time vs 25% uniform ([Tang et al.](https://arxiv.org/abs/2605.01846)); LLM-written NLI is 86–96% solvable from the hypothesis alone ([Proebsting & Poliak](https://arxiv.org/abs/2410.08996)).
- **Leaks.** A naive "the variant is solvable" reward led the policy to embed hints or the answer in the variant ([SvS](https://arxiv.org/abs/2508.14029)); a role prompt leaked ZebraLogic gold grids, and removing it cut the baseline by 62 points ([LURE](https://arxiv.org/abs/2608.21871)).
- **Guessable or lookup-able items.** Kimi drops prompts answered correctly *without CoT* within 8 guesses, as they give false-positive rewards ([Kimi k1.5](https://arxiv.org/abs/2501.12599)). With a literal needle match Llama 3.3 70B scores 98.5 at 32K, vs 56.2 with one latent hop and 25.9 with two ([NoLiMa](https://arxiv.org/abs/2502.05167)). About 15% of OSWorld tasks need only a terminal ([Epoch AI](https://epoch.ai/blog/what-does-osworld-tell-us-about-ais-ability-to-use-computers)).
- **Right answer, wrong process.** About 28% spurious guessing in mid-sized models ([TRACE](https://arxiv.org/abs/2607.04784)); up to a 65.5% drop from answer-only to step-checked grading ([IneqMath](https://arxiv.org/abs/2506.07927)).
- **Shortcut filtering pays.** Shortcut-filtered data scored 13.15 against 7.14 unfiltered, at the cost of removing about 15% of items ([Sim2Reason](https://arxiv.org/abs/2604.11805)).

**Detect.** An audit battery per generator family (**Strong**; [Idiosyncrasies](https://arxiv.org/abs/2502.12150), [Gururangan et al.](https://arxiv.org/abs/1803.02324)): synthetic-vs-real classifier; question-only, choices-only and answer-prior baselines; no-CoT guess screen; no-tool/no-data/no-evidence ablations; chi-square on answer marginals; string search for the answer and intermediate values; renaming or isomorphic perturbation (remapping repo names also exposes memorized cues; [SchrodingerRepo](https://arxiv.org/abs/2609.27891)).

**Mitigate.** Let a code RNG draw every structural random choice; hide the answer from the hardening model ([SynthRL](https://arxiv.org/abs/2506.02096)); paraphrase templated items during RL ([ether0](https://arxiv.org/abs/2506.17238)); rewrite families until partial-input baselines are at chance; for agents, reward *how* the state was reached or block shortcut channels. Convert MCQ/true-false to open-ended for RL ([Big-Math](https://arxiv.org/abs/2502.17387)), but treat MCQ share as a knob: SPICE found mixing MCQ *helped* because it is reliable to verify ([SPICE](https://arxiv.org/abs/2510.24684)).

### F7. Reward hacking of verifiers (tests, rubrics, Lean/Dafny, constraint checkers)

**Symptom.** Sudden reward jumps; response length collapses or runs away; training reward diverges from an oracle or gold judge; trip-wires fire; outputs contain placeholders, `assume(false)`, `sorry` or edited test files.

**Root cause.** Hardening raises both the *incentive* to game the verifier (legitimate reward gets scarcer) and the *attack surface* (more files, tools and state). Most verifiers check outputs rather than intent. Scorers are often visible, and the same agent often writes both the solution and the reward.

**Evidence.**
- **Hacking scales with task complexity.** RLVR models enumerate instance labels instead of inducing rules: 40 shortcuts at complexity levels 1–10 against 458 at levels 11–20 ([Isomorphic Perturbation Testing](https://arxiv.org/abs/2604.15149)).
- **Scorer visibility matters.** o3 reward-hacked in 30.4% of RE-Bench runs against 0.7% on HCAST. METR suggests seeing the full scoring function made the difference ([METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/)).

| Verifier type | Observed exploits | Guards with evidence |
|---|---|---|
| Unit tests / SWE harnesses | GPT-5 cheated on 54–76% of impossible SWE variants. Production hacks: `__eq__` overrides, `sys.exit(0)`, patching pytest via `conftest.py`, and these generalized to misalignment ([ImpossibleBench](https://arxiv.org/abs/2510.20270); [MacDiarmid et al.](https://arxiv.org/abs/2511.18397)). Challengers write randomly failing tests or obfuscate code ([SSR](https://arxiv.org/abs/2512.18552)). Hidden tests led agents to fetch the fix over the network ([Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729)) | Read-only or hidden tests; out-of-process grading; detecting edits to test files; type-checking returned objects. An abort option cut GPT-5's cheating from 54% to 9%. Inverse-mutation checks. Network and git blocking |
| Terminal / sandbox | 16% of 1,968 terminal tasks were hackable from the description alone ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)). Agents attacked the sandbox itself: scheduler socket, logs, overwriting `/bin/bash` ([DSec](https://arxiv.org/abs/2609.22978)) | Hacker–fixer loop (KernelBench attack success 62% → 0%); checksummed mounts; separate build and run accounts |
| LLM judges / rubrics | "Master keys": up to 80% false-positive rate, 67.0% for Qwen2.5-72B on "Thought process:" ([Master-RM](https://arxiv.org/abs/2507.08794)). Generated rubrics exploited in 8–26% of the unbiased cut and 36–98% of the stress cut ([ImpossibleRubrics](https://arxiv.org/abs/2609.16816)). Training-judge score rises while the gold judge peaks and falls (−3 HealthBench-Hard, −22 ResearchQA) ([Rubric Dropout](https://arxiv.org/abs/2608.11669)). Fine-tuning a generative verifier *raised* its susceptibility from 21.7 to 35 ([Huang et al.](https://arxiv.org/abs/2505.22203)) | Rule check first, discriminative verifier second (xVerify ≤0.4% attack success); truncated-response negatives; certificate-faithful rubrics (0/45 exploited); dropping 30–50% of criteria per step; monitoring the train–oracle gap |
| Lean / Dafny / Verus | `assume(false)` spread from one program to all programs ([AlphaVerus](https://arxiv.org/abs/2412.06176)). Dafny RL reward rose from 2.2% to 58.1% via `ensures result >= 0` and leaky specs ([Tan](https://arxiv.org/abs/2605.30914)). A Lean 4.9.0 `apply?` bug silently dropped `sorry` and inflated PutnamBench counts ([DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)) | Screen for `sorry`, `admit`, new axioms and `assume(false)`; vacuity check (try to prove False); block spec edits; re-run completeness checks after any LLM spec repair |
| Constraint checkers (IF) | Copied placeholders, dummy list items, repetition to hit counts, copied delimiters ([IFDecorator](https://arxiv.org/abs/2508.04632)); copy-paste satisfaction ([UNSPECIFIC](https://arxiv.org/abs/2608.09154)); "all instructions followed" self-claims ([Kimi K2](https://arxiv.org/abs/2507.20534)) | IntentCheck cut hacking from 14.53% to 7.60%; hidden trip-wire prompts; reward-model gate ([IFBench](https://arxiv.org/abs/2507.02833)); satisfaction checked on a summary as well |
| Performance / timing | 32.8% of early CUDA-L1 outputs exploited stream timing, for a fake 18× speedup ([CUDA-L1](https://arxiv.org/abs/2507.14111)) | Stream sync; output-materialization checks; treat every sudden reward jump as a probable hack |
| Model-written checkers | When one agent writes both solution and reward, the reward re-checks the construction procedure, e.g. `chart_verified = True` ([CUA-Gym](https://arxiv.org/abs/2605.25624)) | Information barrier between solution author and reward author; forbidden-pattern scan |

- **Monitors and instructions are not enough.** LLM monitors caught 86–89% of cheating on LiveCodeBench but only 42–65% on SWE ([ImpossibleBench](https://arxiv.org/abs/2510.20270)); agents took a planted shortcut in over 50% of runs even when told not to ([BaitBench](https://arxiv.org/abs/2608.30724)).

**Detect.** Alarms on reward jumps and length; the train-vs-oracle reward gap; a hidden trip-wire set; planted hack-verifiable canaries ([HVE](https://arxiv.org/abs/2605.20744)); trivial-agent baselines (an empty-response agent passes 38% of τ-bench-Airline; [ABC](https://arxiv.org/abs/2507.02825)); invariance under isomorphic relabelling.

**Mitigate.** Hack-audit every new operator or family before RL (**Strong**); prefer certificate- or construction-based checks; never let one agent write both solution and reward. Full recipes: [chapter 04](04-verification-and-quality-control.md).

### F8. SFT-only composition failures

**Symptom.** SFT on composed or hardened synthetic data raises in-distribution scores but not compositional or out-of-distribution ones. Sometimes the model ends up below its base.

**Root cause.** Next-token imitation of composed traces memorizes surface patterns and does not build reusable composition. Off-policy traces also carry the teacher's style and errors.

**Evidence.**
- **RL composes where RFT does not.** On the same Level-2 string compositions, RL reached 64% on L2 and 27% on L3. Iterative RFT reached 15% on L2 and never exceeded 2.6% on L3 ([f(g(x))](https://arxiv.org/abs/2509.25123)).
- **Training on the parts separately does not compose.** Training skills A and B separately gave little or no compositional gain ([OMEGA](https://arxiv.org/abs/2506.18880)).
- **Composition robustness does not come from SFT.** DeepSeek-Prover-V2-7B drops from 66.2% to 47.0% (Type I) and 42.1% (Type II) on composed inequalities. SFT on about 8K composed items fixed only in-distribution Type I ([Ineq-Comp](https://arxiv.org/abs/2505.12680)).
- **SFT memorizes, RL generalizes.** On rule variants, SFT lost 8.1 points on GP-L (to 3.4) and 79.5 on V-IRL-L (to 1.3), while RL gained 3.5 and 11.0 ([Chu et al.](https://arxiv.org/abs/2501.17161)).
- **SFT can land below the base model.** IPhO: SFT 15.9, base 19.8, RL 25.2 ([Sim2Reason](https://arxiv.org/abs/2604.11805)); Hard-LP: SFT 26, base 55, RL 80 ([AutoOR](https://arxiv.org/abs/2604.16804)); raw-trajectory SFT 36.7 vs base 47.0, while re-solving gave 52.1 ([Terminal-Universe](https://arxiv.org/abs/2609.04148)).
- **Small students are fragile.** SFT on S1K-style traces lowered 1.5B and 3B accuracy ([2506.17211](https://arxiv.org/abs/2506.17211)).
- **The strongest teacher is not the best source of distillation data.** The weakest teacher (Kimi-K2.5, 39.8) produced the best 2B and 3B students, beating Opus 4.5 (53.5) ([Gym-Anything](https://arxiv.org/abs/2604.06126)).

**Detect.** Held-out compositional splits (unseen skill pairs, depth + 1) compared against the base. Choose checkpoints with a short RL probe: pre-RL scores of 2.8% vs 11.7% became 25.1% vs 21.2% after RL ([SkillFactory](https://arxiv.org/abs/2512.04072)).

**Mitigate.**
- Use SFT or mid-training to install atoms and format, then run RL on the composites (**Strong**). In a controlled 100M-model study, about 1% pretraining exposure to a context was enough for RL to transfer to it, and 0.1% was not ([Interplay](https://arxiv.org/abs/2512.07783)).
- Pick the teacher by measured student results and re-solve tasks rather than imitating raw trajectories ([chapter 06](06-sft-playbook.md)).

### F9. Collateral damage: specialist traps and forgetting

**Symptom.** The targeted metric rises while untargeted skills fall. Previously mastered training items quietly regress.

**Evidence.**
- **Narrow RL destroys general skills.** IF-only RLVR on the Qwen2.5-7B base scored AlpacaEval 1.1 and GSM8K 15.3 ([IFBench](https://arxiv.org/abs/2507.02833)).
- **SFT forgets more than RL.** Math-only SFT dropped IFEval from 69.2 to 42.3 and HaluEval from 35.7 to 2.3, while RL on the same data kept IFEval at 70.0 ([Huan et al.](https://arxiv.org/abs/2507.00432)).
- **Mastered items regress.** About 21% of reviewed mastered prompts partially regressed under GRPO, and a 1% review budget removed GRPO's plateau ([ReMind](https://arxiv.org/abs/2606.03087)).

**Detect.** Run a regression suite (IFEval, hallucination, conversational QA, long context) and track correct-set turnover on the training set.

**Mitigate.** Mix domains (**Strong**); gate constraint rewards with a reward model or intent check; keep a 1–2% review queue and a 5–10% easy pool (**Moderate**); regenerate distilled traces near-on-policy or follow SFT with RL.

---

## Part C — The loop degrades over time

### F10. Diversity and topic collapse in self-play

**Symptom.** Generated tasks concentrate on a few topics, skills or templates. Batch-level diversity looks fine, but the same skills recur across iterations. Pass@k at large k stops improving.

**Root cause.** Proposers exploit whichever region of task space maximizes their reward, and a single agent calibrates toward problems it finds easy. Diversity is measured on wording rather than skills, or only within a batch.

**Evidence.**
- **Proposers pile onto one topic.** A single activation probe used as the generator reward put about 74% of tasks on one topic (`sorting_order`). The auxiliary probe used alone drifted to trivial string tasks: 88.6% solved in 8 of 8 attempts, with a top-topic share of 99.9% ([PROPEL](https://arxiv.org/abs/2606.18284)).
- **Collapse onto a domain and onto easy items.** STP drifted into algebraic manipulation ([STP](https://arxiv.org/abs/2502.00212)); a single agent self-calibrates toward easy problems ([PopuLoRA](https://arxiv.org/abs/2605.16727)).
- **"Diversity illusion."** Within-batch penalties let the proposer cycle between modes across iterations, and different wording can hide identical skills ([R-Diverse](https://arxiv.org/abs/2602.13103)).
- **Unstructured rewriting homogenizes.** Evol-Instruct pushed pass@8 below the untuned backbone (54.2 vs 55.1) ([EvoTD](https://arxiv.org/abs/2605.11666)). Few seeds with many rewrites each lose diversity, and diversity tracks downstream accuracy ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)).

**Detect.** Top-topic share; novelty against a *persistent* archive computed on solution signatures (canonical solver code, masked templates), not wording; rewrites per seed; Vendi-style scores (QbQ variants 1.59 vs 1.25 for static augmentation; [QbQ](https://arxiv.org/abs/2608.01522)).

**Mitigate** (**Moderate**): persistent-memory penalties with skill-level similarity (R-Diverse); template dedup (SQL-Zero); worst-case probe ensembles (top-topic share 0.74 → 0.69 for a 3B solver, 0.67 → 0.54 for 7B); populations with cross-evaluation; forced structural operators (QbQ: 3 of 5 per seed); a mutation-family memory that demands larger gains for repeats ([BenchEvolver](https://arxiv.org/abs/2606.01286)); ≤ ~10 rewrites per seed. Multiply every diversity term by a validity gate, since novelty alone rewards odd or invalid tasks.

### F11. Entropy collapse and boundary shrinkage

**Symptom.** Pass@1 rises while large-k pass@k stays flat or drops below the base; entropy decays steadily; outputs converge on one template; all-correct groups keep growing.

**Root cause.** RL on saturated or narrow pools sharpens toward high-likelihood modes, shrinking support faster than it expands it. Self-play without anchors homogenizes both the tasks and the solutions.

**Evidence.**
- **Self-play mostly sharpens.** AZR self-play helps pass@k at small k, but the base model is better at large k (not statistically significant), and entropy still collapses ([Chae et al.](https://arxiv.org/abs/2510.27072)).
- **Answer-preserving variants keep entropy up.** SvS kept entropy stable while standard RLVR's decayed, and added +18.3 and +22.8 pass@32 on AIME24 and AIME25 ([SvS](https://arxiv.org/abs/2508.14029)).
- **Anchor-free questioners collapse.** An R-Zero-style questioner without anchors suffered entropy collapse ([D2Evo](https://arxiv.org/abs/2605.17037)).
- **Multi-turn collapse.** Game self-play without role-conditioned advantages showed "thinking collapse" after about 200 steps ([SPIRAL](https://arxiv.org/abs/2506.24119)); multi-turn RL shows the "Echo Trap" with entropy drop ([RAGEN](https://arxiv.org/abs/2504.20073)).
- **Saturated pools shrink the boundary.** Support shrinkage outweighs expansion ([2507.14843](https://arxiv.org/abs/2507.14843)). Fixed-data DAPO on Qwen3-4B-Thinking went from 72.4 to 64.8 ([EvoEnv](https://arxiv.org/abs/2605.14392)).

**Detect.** Log entropy, effective-prompt ratio and all-correct share; plot pass@k to k ≥ 64–256 against the base, adding Cover@τ because large-k pass@k is inflated on low-entropy answer spaces ([Cover@τ](https://arxiv.org/abs/2510.08325)).

**Mitigate** (**Moderate**): feed new items rather than reweighting old ones. Retire prompts at pass rate ≥ 0.9 ([ScaleRL](https://arxiv.org/abs/2510.13786)) with a review queue; refresh saturated items with answer-preserving variants; anchor self-play with real items; keep generator KL β ≥ 0.05 (PROPEL's math run collapsed at 0.02); consider boundary-preserving updates such as injecting incorrect rollouts into saturated data (+6.4 to +9.0 on Qwen3-1.7B/4B; [2609.33126](https://arxiv.org/abs/2609.33126)). See [chapter 05](05-rl-playbook.md).

### F12. Proposer death spirals

**Symptom.** The proposer's valid-output rate falls toward 0 while its entropy rises. Solver reward *rises* misleadingly as the solver overfits. Or the proposer settles on trivial or unsolvable tasks.

**Root cause.** Penalties for malformed outputs push the proposer away from the only region it can reach. A frozen partner or an intrinsic reward lets it game the objective. With no quality anchor, open-ended self-play degenerates.

**Evidence.**
- **A small penalty can start the spiral.** A −0.1 reward for invalid questions produced a death spiral: proposer entropy rose, the valid-question rate went to 0, and solver reward rose misleadingly. The main configuration uses 0 instead ([SSP](https://arxiv.org/abs/2510.18821)). SPICE uses ρ = −0.1 without the problem ([SPICE](https://arxiv.org/abs/2510.24684)), so the effect depends on the setup.
- **Missing stabilizers collapse the proposer.** A frozen solver drives problems to unsolvable or trivial. Removing the curriculum-DAG descent peaked at 51.3% and then collapsed to 12% ([ANCORA](https://arxiv.org/abs/2604.27644)).
- **Collapse without anchors or replay.** Formatting collapses by epoch 4 without golden replay ([STRETCH](https://arxiv.org/abs/2609.18642)); trained user simulators collapse to accept/reject ([SEAD](https://arxiv.org/abs/2602.03548)); self-play without a quality self-reward "degenerate[s] into adversarial nonsense" ([Language Self-Play](https://arxiv.org/abs/2509.07414)); an intrinsic-reward teacher caused student collapse ([SOAR](https://arxiv.org/abs/2601.18778)).
- **Naive setter–solver play is hacked by invalid problems.** VHG fixes it with R = 1[valid]·(1 − Acc) ([VHG](https://arxiv.org/abs/2605.06660)).

**Detect.** Put the valid-proposal rate, proposer entropy, the share of proposals at p ∈ {0, 1}, and solver reward *vs* held-out accuracy on one dashboard.

**Mitigate.** Gate difficulty multiplicatively by validity, and give malformed proposals zero or clipped reward whenever the valid rate drops (**Moderate**: SSP ablation vs SPICE). Co-evolve the solver rather than freezing it; add golden replay or human anchors; freeze or rule-anchor graders; exclude noisy batches from generator updates.

### F13. Iterative self-synthesis drift

**Symptom.** Rounds 1–3 improve, then gains plateau or reverse. Strong skills get stronger while weak ones decay. Accuracy on the gold audit slice falls. Generated tasks drift away from human task distributions.

**Root cause.** The loop recycles its own errors and narrows its own distribution, and it adds data without adding learnable information.

**Evidence.**
- **Self-labelled loops peak early.** R-Zero peaks after 1 iteration at 0.6B and after 3 at 4B. The 0.6B model starts declining at 70.6% label accuracy, while the 4B model tolerates 48.8%, so label noise alone does not explain collapse ([R-Zero](https://arxiv.org/abs/2508.05004)).
- **Round counts converge on about 3.** WizardCoder peaked at 3 rounds ([WizardCoder](https://arxiv.org/abs/2306.08568)); a third small-model evolution round degraded results ([Hui et al.](https://arxiv.org/abs/2412.11231)); SEIF saturated after round 3 ([SEIF](https://arxiv.org/abs/2605.07465)); UltraIF's third DPO round diverged ([UltraIF](https://arxiv.org/abs/2502.04153)).
- **Polarization and drift.** Iterative synthetic instruction tuning polarizes competence ([KITE](https://arxiv.org/abs/2607.17043)); unanchored self-play drifts toward LM-style bugs and regresses on human bugs, which a 20% real-bug mix fixed ([Anchored Self-Play](https://arxiv.org/abs/2607.03523)).
- **Unfiltered self-labels avalanche.** Without filtering, 6×6 multiplication accuracy was only 13.7% after 7 rounds. Five-model majority voting raised label accuracy from 31% to 93.3% ([Self-Improving Transformers](https://arxiv.org/abs/2502.01612)).
- **Model collapse.** Replacing real data loses the tails ([Shumailov et al.](https://arxiv.org/abs/2305.17493)); accumulating keeps error bounded ([Gerstgrasser et al.](https://arxiv.org/abs/2404.01413)); verifier selection prevents collapse ([Feng et al.](https://arxiv.org/abs/2406.07515)).

**Detect.** Measure accuracy on a gold slice every round, bucketed by difficulty, and keep a held-out human-task evaluation.

**Mitigate.** Accumulate, don't replace, and verify what you add (**Strong**); put 20–33% oracle-backed real items in every round (**Moderate**); ground labels in executors, corpora or solvers (SPICE: 43.9 grounded vs 40.7 ungrounded); stop after ~3 rounds unless a held-out metric still improves (**Strong**); with self-consistency labels, keep challenger and solver weights separate (71.0% vs 63.4% pseudo-label accuracy).

---

## Part D — You believe a result that is not real

### F14. Contamination (including contamination recreated by complexification)

**Symptom.** Large gains on public benchmarks that do not appear on post-cutoff items or on renamed or perturbed variants.

**Evidence.**
- **Evolved and raw pools overlap evaluations.** Evol-CodeAlpaca overlapped 70.7% of HumanEval ([Tulu 3](https://arxiv.org/abs/2411.15124)); DeepMath's raw pool contained 90% of AIME24/AMC23 and 76.6% of MATH500 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)).
- **Paraphrases evade n-grams.** A 13B model trained on rephrased test items reached GPT-4-level scores, and LLM-generated synthetic data was itself contaminated ([LLM Decontaminator](https://arxiv.org/abs/2311.04850)).
- **Memorization in the base model.** Given the first 60% of a MATH-500 prompt, Qwen2.5-Math-7B reproduced the rest 54.6% of the time, against 3.8% for Llama3.1-8B ([Wu et al.](https://arxiv.org/abs/2507.10532)).
- **GRPO spreads contamination.** After GRPO, contamination inflated scores on related *uncontaminated* benchmarks as well ([2601.06103](https://arxiv.org/abs/2601.06103)).
- **Leakage through the pipeline.** CodecLM's metadata came from validation splits of its evaluation benchmarks ([CodecLM](https://arxiv.org/abs/2404.05875)); ZeroGUI used OSWorld test tasks as generation exemplars ([ZeroGUI](https://arxiv.org/abs/2505.23762)).

**Mitigate** (**Strong**). Decontaminate complexified *outputs* semantically, not just seeds (embedding retrieval + LLM judge; web-search novelty filter, SAND-Math τ = 0.85); keep operator discovery and target conditioning disjoint from evals; remap names as a cheap memorization check; evaluate on post-cutoff contests ([MathArena](https://arxiv.org/abs/2505.23281)); hold out whole environments, not just instructions.

### F15. Qwen-specific spurious-reward confounds

**Symptom.** Tiny, odd or even random-label data gives large gains on Qwen2.5(-Math), and the effect vanishes on other model families.

**Evidence.**
- **Random rewards work on Qwen2.5-Math, and only there.** Random rewards gave +21.4 on MATH-500 against +29.1 with ground truth on Qwen2.5-Math-7B, through a clipping bias that amplifies "code reasoning" (65% → >90%). The gains did not appear on Llama3 or OLMo2 ([Spurious Rewards](https://arxiv.org/abs/2506.10947)).
- **Fresh data removes the effect.** On freshly generated arithmetic, only accurate rewards beat the base model ([RandomCalculation](https://arxiv.org/abs/2507.10532)).
- **Data-efficiency results are confounded.** 32 identical prompts per step matched full-data training on Qwen2.5-Math-1.5B, and the authors declared their Qwen ablations invalid ([Prompt Replay](https://arxiv.org/abs/2603.21177)); dose headlines such as 1-shot RLVR's MATH500 36.0 → 73.6 were measured on Qwen2.5-Math ([1-shot RLVR](https://arxiv.org/abs/2504.20571)).
- **Families respond differently.** RLVR's regressions on mastered skills were severe for Llama and Gemma and mild for Qwen ([Algebrarium](https://arxiv.org/abs/2602.08281)).

**Mitigate.** Every dose, mixing or operator claim needs a random-reward arm, a fresh procedural control and at least one non-Qwen family (**Strong**).

### F16. Eval noise and false conclusions

**Symptom.** A data-policy "win" disappears on rerun, at another scale, or under a different aggregate.

**Evidence.**
- **Seeds and hardware alone move scores.** Pass@1 SD across seeds is 5–15 points, one AIME question is worth 2.5–3.3 points, and OpenThinker2-7B scored 53.0 ± 4.6 on A100 vs 57.1 ± 5.2 on H100 ([Sober Look](https://arxiv.org/abs/2504.07086)); BF16 alone moves accuracy by up to 9% ([Yuan et al.](https://arxiv.org/abs/2506.09501)).
- **Small pilots cannot see small effects.** An unpaired comparison at p ≈ 0.5, with 95% significance and 50% power, needs a 25.3-point gap at N = 30, 13.9 at N = 100 and 6.2 at N = 500 (derived binomial arithmetic).
- **Most data-policy wins do not survive proper tests.** None of 8 selection or reweighting policies had a paired 95% CI excluding zero vs uniform sampling (12 matched seeds), none of 3 adaptive mixtures beat a fixed equal mix, signs flipped with scale (difffilter −0.79 at 3B, +0.08 at 7B), and swapping the aggregate gave ρ = −0.33 ([DataFlex-RL](https://arxiv.org/abs/2609.06107)).
- **Early winners and variance.** Early winners can lose at scale, with ±0.02 run-to-run error on the fitted asymptote ([ScaleRL](https://arxiv.org/abs/2510.13786)); in computer-use RL a published-size gain would have the wrong sign 33–44% of the time in the high-variance regime ([2607.17136](https://arxiv.org/abs/2607.17136)).
- **Naive SEs are too tight for clustered items.** Clustered SEs can be 3× naive ones (DROP 1.34 vs 0.44) ([Miller](https://arxiv.org/abs/2411.00640)).
- **Pipeline bugs replicate.** A decoding-budget bug produced a 32-point "effect" that replicated from N = 50 to N = 500. Only reading raw generations caught it ([Ballı](https://arxiv.org/abs/2607.13707)).

**Mitigate** (**Strong**). Pre-register the protocol: fixed stack with FP32 or batch-invariant kernels; 10–30 samples per item; paired per-item comparisons with SEs clustered by *seed family*; train/eval split by seed, not variant; 3 seeds to screen, 8–12 to decide; a domain-balanced aggregate plus regression suite; read raw generations.

### F17. Overfitting to one generator family

**Symptom.** Gains hold on items from your generator but not on items from other generators or from natural sources. A judge from the same family rates your model generously.

**Evidence.**
- **Generators leave detectable fingerprints.** A five-way generator-ID classifier reaches 97.1% accuracy and stays above 90% after paraphrase or translation. Students fine-tuned on Llama vs Gemma outputs were 98.9% separable ([Idiosyncrasies](https://arxiv.org/abs/2502.12150)).
- **Related judges favour related students.** Preference leakage is 23.6% for the same model, 8.9% within a model series and 2.8% across series ([Preference Leakage](https://arxiv.org/abs/2502.01534)).
- **Traits transfer through filtered data.** Traits pass through correct-filtered CoT when teacher and student share a base ([Subliminal Learning](https://arxiv.org/abs/2507.14805)).
- **Adding brands adds little diversity.** Cross-model output similarity is already 0.71–0.82 ([Artificial Hivemind](https://arxiv.org/abs/2510.22954)).
- **Generators find their own variants easier.** GPT-4 scored 87.36% on its own unrevised variations against 85.58% on human-corrected ones (humans revised 18.85% of them) ([GSM-Plus](https://arxiv.org/abs/2402.19255)).
- **Family-specific training overfits.** TÜLU-3-8B-DPO scores 81.1 on IFEval but 25.5 on IFBench ([IFBench](https://arxiv.org/abs/2507.02833)).

**Mitigate** (**Moderate**: mechanisms well documented, direct RL ablations scarce). Mix 2–3 generator families from different series, chosen by distance (e.g. NCD); keep generator, judge and student in different series; filter adversarially against an ensemble that includes the current policy, not one adversary ([Phang et al.](https://arxiv.org/abs/2111.08181)); evaluate on a held-out generator family and operator.

---

## Part E — The bill

### F18. Cost blowups from overgeneration and profiling

**Symptom.** Generation spend grows faster than the in-band pool. Profiling rollouts rival training rollouts. Environment builds dominate the budget.

**Root cause.** The cost per *accepted in-band* task is roughly

```
C_acc ≈ (c_gen + c_validate + c_env + k·c_roll) / (y_valid × y_band)
```

Low validity or band yield multiplies every other cost term.

**Evidence.**
- **Yields are low** (candidates needed = 1/yield): SWE-Next 2.25%, i.e. ~44 candidates per kept task ([SWE-Next](https://arxiv.org/abs/2603.20691)); Code-as-Task 5.2% ([SCA](https://arxiv.org/abs/2506.01716)); ~5% of AutoCode problems in the 0.1–0.5 zone ([AutoCode](https://arxiv.org/abs/2510.12803)); UQ kept 500 of ~3 million ([UQ](https://arxiv.org/abs/2508.17580)).
- **Profiling is expensive.** In a worked example (10K accepted math tasks at y = 0.3), profiling was about 36% of all rollout tokens (derived).
- **Absolute costs.** $891K for 45,320 SWE environments ([OpenSWE](https://arxiv.org/abs/2603.13023)); ~$138K and 127K GPU-hours for DeepMath-103K ([RLVE](https://arxiv.org/abs/2511.07317)); 50–500 TPU-days per problem for test-time variant curricula ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y)).
- **A stronger generator pays only if it raises band yield.** Most rejects are "too easy" (64%; [Trading Human Curation](https://arxiv.org/abs/2606.03800)), and solving skill does not predict generating skill (R² < 0.1; [AgoraBench](https://arxiv.org/abs/2412.03679)). Measure y_valid and y_band per generator.
- **Deleting after one pilot throws away value.** About 19% of epoch-2 solves were not solved in epoch 3 ([Pilot-Commit](https://arxiv.org/abs/2605.26606)).

**Mitigate.** Follow this order (**Moderate**; see [chapter 03](03-generation-architectures.md)):
1. Audit the verifier.
2. Measure the pool's p distribution.
3. Reallocate rollouts and recycle items, which is often a free 1.5–2×: Pilot-Commit uses up to 1.9× fewer rollouts than GRPO; [Knapsack RL](https://arxiv.org/abs/2509.25849) gives about 2× compute-equivalent.
4. Only then generate, with gated acceptance, yield logged per operator, and cheap difficulty predictors (KGPS used 83% fewer rollouts than dynamic sampling).
5. Reuse items up to about 25× before paying for new unique ones ([Tan et al.](https://arxiv.org/abs/2509.25300)).

---

## 7. Training dashboard: metrics and alarms

Most failures surface in cheap metrics before benchmarks. Log them per generator family and operator, not only in aggregate (alarm patterns are **Proposal**-level synthesis).

| Metric | Alarm pattern | Likely failure modes |
|---|---|---|
| Effective-prompt ratio (share of groups whose rewards are not all identical) | Falling | F1 or F11 (saturation); F2 (overshoot) |
| Share of all-correct groups | Rising; use it as the trigger to complexify | F1, F11 |
| Share of items at p = 0, per operator | Rising after a new operator or round | F2, F3, F4 |
| Valid-proposal rate / proposer entropy | Rate → 0 while entropy rises | F12 |
| Policy entropy | Steady decay | F11 |
| Mean and max response length | Abrupt collapse (510.7 → 45.7 tokens) or runaway | F7, F2 |
| Training reward minus oracle or gold-judge reward | Widening gap | F7 |
| Trip-wire / canary hit rate; trivial-agent pass rate | Any value above 0 | F7, F3 |
| Top-topic share; novelty against the persistent archive | Share rising, novelty falling | F10 |
| Gold-slice label accuracy by difficulty bucket | Falling across rounds | F4, F13 |
| pass@k at large k (plus Cover@τ) vs base | Converging to or below the base curve | F11 |
| Regression suite; correct-set turnover | Any drop | F9 |
| y_valid, y_band and cost per accepted in-band task, per operator | Rising cost or falling yield | F18, F1 |

---

## 8. Pre-flight checklist

Run this before scaling any generator of harder tasks. The items are ordered by when they apply.

**A. Before generating anything**
- [ ] **Verifier audited:** TPR and TNR on known-good/known-bad solutions (for code ≥ 0.9 on both, as in CodeContests+ HQ); metamorphic equivalent-rewrite tests; J = TPR − FPR > 0 per family and difficulty bucket.
- [ ] **Pool profiled.** Measure the current pool's pass-rate histogram with the *current* policy in the *exact* RL harness. Zero-variance filtering and rollout reallocation are already on.
- [ ] **Evaluation suite pre-registered, split by seed:** post-cutoff items; a held-out generator family and operator; natural-distribution targets; a regression suite; correct-set turnover.
- [ ] **Decontamination planned for outputs**, not just seeds, against every evaluation set. Operator discovery and target conditioning use no evaluation data.

**B. Per operator or generator family (pilot of a few hundred candidates)**
- [ ] **Yield logged:** y_valid, y_band and cost per accepted in-band task.
- [ ] **Mutations accepted only if** measured pass rate falls and the item stays valid.
- [ ] **p = 0 items triaged** with large-k solving, a stronger or tool-assisted solver, and a hint-conditioned attempt. They are *deferred*, not deleted.
- [ ] **Uniqueness checked.** Alternatives are enumerated, and the privileged-information test passes (solved with evidence, fails without).
- [ ] **Shortcut audit:** question-only, choices-only and no-CoT baselines; no-tool/no-data/no-evidence ablations; code-RNG answer marginals checked by chi-square; leak search; isomorphic renaming.
- [ ] **Hack audit:** hacker–fixer pass; empty/no-op/random-agent baselines; scans for `sorry`, `assume(false)` and test edits; read-only, out-of-process grading; information barrier between solution author and reward author.
- [ ] **Hand audit** of 100–200 items, reported with Wilson CIs (0 errors in 200 bounds the rate below about 1.9%), prioritized by IRT flags. Someone has read raw generations.
- [ ] **Generator signature checked.** Generator-ID and synthetic-vs-real classifiers are run. Generator, judge and student come from different series, and 2–3 generator families are used.
- [ ] **Diversity capped and measured:** ≤ ~10 rewrites per seed; solution-signature dedup against a persistent archive; top-topic share logged.

**C. Before RL**
- [ ] **Compositions go to RL**, not only to SFT. Atoms and format are installed first, and the checkpoint is chosen with a short RL probe.
- [ ] **Default blend** (**Moderate**) until your own ablation beats it: equal mix across operator families, a 20–33% oracle-backed real anchor, a 2–10% easy or review pool.
- [ ] **Self-play guards:** multiplicative validity gate; zero reward for malformed proposals; generator KL β ≥ 0.05; co-evolving (not frozen) solver; separate weights when labels come from self-consistency; ~3-iteration cap with a gold-slice stop rule.
- [ ] **Dashboard alarms** from section 7 are wired to page someone.

**D. Before claiming a win**
- [ ] **Statistics:** a paired 95% CI excluding zero on the pre-registered, domain-balanced aggregate, across 8–12 training seeds, with SEs clustered by seed family.
- [ ] **Confound controls:** a random-reward arm and at least one non-Qwen family.
- [ ] **Generality:** the gain holds on post-cutoff items and on the held-out generator family. The pass@k curve at large k is at or above the base. No regression-suite metric drops.
- [ ] **Post-training contamination audit** is done, including partial-prompt probes and renamed variants.
