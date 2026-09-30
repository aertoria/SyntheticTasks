# Verification and quality control: keeping hard synthetic tasks correct, unambiguous and unhackable

> **Key takeaways**
> - **Audit the verifier before you complexify anything.** Many tasks that look "too easy" or "impossibly hard" are verifier artifacts. Rule-based math checkers miss about 14% of correct answers, more than 4,000 CodeContests problems rejected almost all correct code, TACO tests accepted over 90% of wrong programs on hard problems, and hardened tests cut the SWE-bench Verified top score from 78.80% to 62.20%. **Strong**
> - **Build correctness in by construction and verify only what construction cannot guarantee.** Solution-first evolution reached 97.7% post-hoc validity against 79.3% for problem-first generation with the same evolver. Planted certificates, frozen executable oracles, execute-first derivation and answer-preserving variants all make the label a product of executed artifacts, not of a model's belief. **Strong**
> - **Gate every hard item on both sides.** It must *fail* when the key ingredient is removed (closed-book, no data, no hint, no auxiliary construction, no-op agent) and *succeed* when gold evidence or a hint is supplied. Rejection by a strong model alone keeps broken and ambiguous items. **Strong**
> - **Consensus labels are not a verifier for the hard tail.** Majority vote decays with difficulty (R-Zero pseudo-label accuracy 79% → 63%; the unverified majority is wrong on 73.33% of AIME 2024 questions), and agreement filters systematically delete frontier items. Use consensus only as a provisional label with independent evidence (execution, another model family, a formal check). **Strong**
> - **Harder tasks invite more hacking, and hacks generalize.** Harden the environment (read-only or hidden tests, out-of-process grading, information barrier between solution and checker authors, forbidden-pattern scans, hacker–fixer loops, isomorphic twins), and monitor for reward jumps and length collapse. **Strong**
> - **SFT and RL need different QC budgets.** For RL, verifier precision and recall *are* the reward; for SFT distillation, answer filtering often adds nothing and question selection matters more. Under rejection fine-tuning false positives are the costly error; under GRPO false negatives also draw gradient. **Moderate**

**Contents**

1. [Why verification becomes the bottleneck once tasks get hard](#1-why-verification-becomes-the-bottleneck-once-tasks-get-hard)
2. [The correctness contract: what must be true of every hard task](#2-the-correctness-contract-what-must-be-true-of-every-hard-task)
3. [Correctness by construction versus post-hoc verification](#3-correctness-by-construction-versus-post-hoc-verification)
4. [Verifier types and their measured failure rates](#4-verifier-types-and-their-measured-failure-rates)
5. [Testing the tester: mutation-hardened and adversarial verifier QA](#5-testing-the-tester-mutation-hardened-and-adversarial-verifier-qa)
6. [Two-sided validity gates](#6-two-sided-validity-gates)
7. [Distinguishing "hard" from "broken" or "ambiguous"](#7-distinguishing-hard-from-broken-or-ambiguous)
8. [Majority-vote and consensus pseudo-labels](#8-majority-vote-and-consensus-pseudo-labels)
9. [Reward hacking in synthetic environments and concrete guards](#9-reward-hacking-in-synthetic-environments-and-concrete-guards)
10. [Decontamination and deduplication](#10-decontamination-and-deduplication)
11. [Measuring diversity](#11-measuring-diversity)
12. [A QC checklist and acceptance-criteria template](#12-a-qc-checklist-and-acceptance-criteria-template)
13. [Open problems](#13-open-problems)

Cross-references: difficulty definitions and the p(1−p) learning-signal argument are in [chapter 01](01-diagnosis-difficulty-and-learning-signal.md); operators themselves in [chapter 02](02-complexification-operator-taxonomy.md); end-to-end pipelines in [chapter 03](03-generation-architectures.md); pass-rate bands, reward shaping and zero-variance handling in [chapter 05](05-rl-playbook.md); SFT-specific filtering in [chapter 06](06-sft-playbook.md); domain-specific verifiers in [chapters 07a](07a-domain-recipes-reasoning.md) and [07b](07b-domain-recipes-agents-and-beyond.md); a consolidated failure catalogue in [chapter 09](09-pitfalls-and-failure-modes.md).

---

## 1. Why verification becomes the bottleneck once tasks get hard

Every complexification operator in [chapter 02](02-complexification-operator-taxonomy.md) increases the probability that an item is wrong, ambiguous, unsolvable or gameable. It also makes the checker less reliable, because harder tasks produce more unusual answer forms, longer trajectories and more alternative solution paths. Four empirical regularities follow.

**(a) Verifier errors look like difficulty (or like easiness).**

| Symptom | What was actually happening | Source |
|---|---|---|
| Correct answers scored wrong | Rule checkers average 86% recall; false negatives rise as the policy gets stronger | [Rule vs model verifiers (2025)](https://arxiv.org/abs/2505.22203) |
| Pool looks harder than it is | >38% of responses on Big-Math-RL-Verified were false negatives; fixing them raised pass rates by up to 10% | [TinyV (2025)](https://arxiv.org/abs/2505.14625) |
| Items at pass rate ≈ 0 | >4,000 original CodeContests problems had TPR ≤ 0.1 (wrong tests, missing custom checkers); only 67.1% of 1.18M tests passed validators | [CodeContests+ (2025)](https://arxiv.org/abs/2506.05817) |
| Multi-answer problems unsolvable | 14.57% of 34,757 code problems need special judges; 59.01% of correct solutions to those fail exact match | [ScaleBox](https://arxiv.org/abs/2604.27467) |
| Pool looks easier than it is | TACO tests' false-positive rate exceeded 90% on difficult problems | [HardTests (2025)](https://arxiv.org/abs/2505.24098) |
| Saturated repo benchmark | About 1 in 5 "solved" SWE-bench Verified patches were semantically wrong; top agent 78.80% → 62.20% after strengthening | [SWE-ABS (2026)](https://arxiv.org/abs/2603.00520) |
| GUI policy looks competent | Same base policy: 63.76% by VLM judge, 38.93% by code assertions | [GUI-Genesis (2026)](https://arxiv.org/abs/2602.14093) |
| Kernel "speedups" | GPT-5.5's 1.43× geomean becomes 0.88× under the verified protocol (TF32 baseline, hidden inputs, tighter tolerance) | [KernelBench-Verified (2026)](https://arxiv.org/abs/2607.16241) |

Fixing the verifier is often the cheapest way to make a saturated item hard again: EvolveCoder's evolved tests cut pass@1 from 43.80 to 31.22 *on the same problems* ([EvolveCoder](https://arxiv.org/abs/2603.12698)). **Strong.**

**(b) Label and verifier errors grow with difficulty.** R-Zero's pseudo-label accuracy fell 79.0% → 69.0% → 63.0% as its Challenger produced harder questions ([R-Zero](https://arxiv.org/abs/2508.05004)). The unverified majority answer is wrong on 25.85% of MATH-500, 46.07% of AMC and 73.33% of AIME 2024 questions ([T³RL](https://arxiv.org/abs/2603.02203)). Shortcut counts under extensional verifiers (across all models) rise from 40 at complexity levels 1–10 to 458 at levels 11–20 ([IPT](https://arxiv.org/abs/2604.15149)). Judge–human F1 of 0.86 overall drops on tasks humans rate "Very Difficult" ([ACD](https://arxiv.org/abs/2502.07577)). Verification spend should therefore rise with difficulty, not stay flat. **Strong.**

**(c) Verifier noise mostly sets the learning *rate*, provided the verifier beats chance.** With Youden's J = TPR − FPR > 0 the probability mass on incorrect modes goes extinct; with J < 0 incorrect modes grow until they dominate ([RLVεR](https://arxiv.org/abs/2601.04411)). But verifier errors inside a GRPO group are correlated (0.53 on Qwen2.5-1.5B rollouts), so 8 completions carry about 1.70 independent completions' worth of verifier information ([Xin 2026](https://arxiv.org/abs/2609.06386)). Repeated calls to the same verifier saturate; switching model family or modality does not: VStress measured conditional marginal information gain of 0.0126 for same-model repeats against 0.0913 for cross-family channels ([VStress](https://arxiv.org/abs/2609.36958)). **Moderate.**

**(d) SFT and RL tolerate different errors.**

| Regime | Evidence | Implication |
|---|---|---|
| SFT distillation | Answer filtering did not help: no filter 41.9, GPT verification 40.0 ([OpenThoughts](https://arxiv.org/abs/2506.04178)); test quality mattered less for teacher distillation ([HardTests](https://arxiv.org/abs/2505.24098)); unverified SWE trajectories trained about as well as verified ones ([SERA](https://arxiv.org/abs/2601.20789)) | Spend QC on question selection, teacher choice and well-posedness; see [chapter 06](06-sft-playbook.md) |
| Rejection fine-tuning | Admitting 25% false positives cost −1.58pp; discarding 75% of true positives cost −0.03pp; a lenient gate erased the whole gain ([Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)) | Tune gates for precision; accept low recall |
| GRPO/PPO | At matched TPR − FPR, the false-negative arm draws *more* gradient under GRPO (same paper); a wrong reference actively penalizes correct reasoning | Measure both FPR and FNR; fix false negatives before hardening |

The rest of this chapter is organized around one question: *what evidence do you need before a hard synthetic item is allowed to produce reward?*

---

## 2. The correctness contract: what must be true of every hard task

A synthetic task is a tuple (statement, environment/initial state, reference, verifier, metadata). The contract below lists the properties that must hold, the failure each one prevents, and the cheapest check with evidence behind it.

| # | Property | Typical failure when missing | Cheapest reliable check |
|---|---|---|---|
| C1 | **Well-posed**: premises consistent and sufficient | Contradictions were the largest flaw class in ValiMath, a deliberately flaw-enriched set (13.97% of all questions; about 300 of 848 flawed items); vacuous hypotheses make formal statements trivially true | Condition-level checks ([MathQ-Verify](https://arxiv.org/abs/2505.13903)); prove `False` from hypotheses ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333)) |
| C2 | **Reference correct** | LLM-written outputs labeled 12.7% of test inputs correctly vs 96.8% for mutual verification ([rStar-Coder](https://arxiv.org/abs/2505.21297)) | Derive the reference from execution or a certificate (§3) |
| C3 | **Unique answer, or a checker that accepts every valid answer** | Obfuscated search questions with several valid answers ([WebSailor](https://arxiv.org/abs/2507.02592) admits this); alternative optima in LPs; 25.4% of contest problems need special judges ([HardTests](https://arxiv.org/abs/2505.24098)) | Enumerate alternatives (§7); grade certificates/objective values, not one argmax ([A²utoLPBench](https://arxiv.org/abs/2607.02141)) |
| C4 | **Solvable** with the information and tools provided | 48.6% of tasks on raw LLM-built websites admitted an executable trace (94.8% after verify-and-repair) ([Verified Synthetic Web Envs](https://arxiv.org/abs/2608.21898)); River Crossing instances with N ≥ 6 and boat capacity 3 are unsolvable yet were scored as failures ([Illusion of Thinking comment](https://arxiv.org/abs/2506.09250)) | ≥1 verified solve by a reference, hinted or oracle-evidence solver (§6) |
| C5 | **Contract valid**: everything the verifier checks is stated or discoverable | Hidden requirements in private tests; weakly grounded terminal tasks fell from 32.8% to 1.2% only after an instruction audit ([RST](https://arxiv.org/abs/2608.05466)) | Instruction–verifier alignment audit; T1 gives it the largest single weight (20%) ([T1](https://arxiv.org/abs/2609.11042)) |
| C6 | **Faithful to intent** (text ↔ program ↔ formal statement) | Compile-passing Lean statements fell from 92.4% to 32.7% after semantic checks ([FormalMATH](https://arxiv.org/abs/2505.02735)); 14 of 43 IMO geometry formalizations refuted by counterexample ([MechGeo](https://arxiv.org/abs/2608.02295)) | Back-translation + multi-LLM consensus; counterexample search; statement-only solver must pass ≥50% of tests ([BenchEvolver](https://arxiv.org/abs/2606.01286)) |
| C7 | **Non-trivial**: no shortcut solves it | Kimi k1.5 drops prompts a no-CoT guess gets right within 8 tries ([Kimi k1.5](https://arxiv.org/abs/2501.12599)); withholding data files cost only 40.5% of performance on QRData ([DSGym](https://arxiv.org/abs/2601.16344)) | Closed-book / no-CoT / no-data / no-tool / no-op probes over k attempts (§6) |
| C8 | **Unhackable**: the verifier cannot be satisfied without doing the task | 16% of 1,968 terminal-benchmark tasks hackable from the description alone ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)) | Forbidden-pattern scans, trivial-agent baselines, hacker–fixer pass (§5, §9) |
| C9 | **Deterministic and reproducible** | Performance references replay validly on every machine for only 39/102 GSO and 11/140 SWE-Perf tasks ([replay audit](https://arxiv.org/abs/2607.01211)) | Run twice and compare ([AZR](https://arxiv.org/abs/2505.03335) uses j = 2); fixed seeds; simulator timing ([PIE](https://arxiv.org/abs/2302.07867)) |
| C10 | **Uncontaminated and non-duplicate** | DeepMath's raw pool contained 90% of AIME24/AMC23 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)) | Output-side n-gram + embedding + LLM paraphrase judge (§10) |

Two refinements matter in practice.

**The contract depends on how the item is consumed.** AlphaProof trains on about 80M auto-formalized statements "regardless of fidelity", because every well-typed statement is a valid prove-or-disprove task ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y)); C6 is irrelevant there. When the reward compares against an informal answer, C6 is on the critical path. Likewise, stepping-stone data can relax C2: only 32.8% of the useful stepping stones in Sundaram et al.'s teacher had fully correct solutions (63% were well-posed), yet the method still gave about 4× pass@1 on held-out fail@128 MATH problems ([SOAR teacher, 2026](https://arxiv.org/abs/2601.18778)). That licence applies to bridge or curriculum items only, never to final RL targets. **Moderate.**

**Grade labels by provenance tier and require higher tiers for RL.** A practical tiering (our synthesis of the evidence in this chapter; **Proposal** for the exact cut-offs):

| Tier | Label source | Examples | Use |
|---|---|---|---|
| T1 | Construction or execution; formal kernel | Planted KKT certificate, DDAR traceback, Prolog answer set, Lean proof, execution of a frozen oracle | RL reward, SFT, eval |
| T2 | Answer-preserving transform or inverse check | SvS/MathForge variants, SymPy differentiation of an antiderivative ([VHG](https://arxiv.org/abs/2605.06660)) | RL reward, SFT |
| T3 | Independent consensus with external evidence | Cross-family agreement, output-vector agreement over ≥50 inputs, tool-verified vote | RL after audit; SFT |
| T4 | Single model or same-family majority; LLM judge without anchors | Majority vote, unanchored VLM judge | SFT (if at all), bridge data; never sole RL reward on hard items |

---

## 3. Correctness by construction versus post-hoc verification

The most reliable pipelines never ask a model whether a hard item is correct; they arrange for the answer to be a consequence of something executed. Controlled evidence: BenchEvolver's solution-first variant reached 97.7% post-hoc validity (GPT-5.4-mini evolver) against 79.3% for a problem-first ablation and 86.2% without mutation memory ([BenchEvolver](https://arxiv.org/abs/2606.01286)). CHASE's free-form Evol-Instruct baseline produced 34 errors in 100 hardened math problems while frontier models still solved 82.5–88.9% of them ([CHASE](https://arxiv.org/abs/2502.14678)). **Strong.**

### 3.1 Construction patterns

| Pattern | What is fixed first | Correctness guarantee | Examples |
|---|---|---|---|
| Solution-first evolution | The executable reference; statement and tests derived from it | Tests are outputs of the evolved program; statement-only brute force and output oracle triangulate | [BenchEvolver](https://arxiv.org/abs/2606.01286); [RST](https://arxiv.org/abs/2608.05466) (extend `solve.sh`, then environment, verifier, instruction) |
| Planted certificate / inverse construction | The answer or certificate | Optimal value known without a solver; scipy matched φ within 10⁻⁴ on all 256 instances | [A²utoLPBench](https://arxiv.org/abs/2607.02141); planted subset-sum ([EvoEnv](https://arxiv.org/abs/2605.14392)); SOS-certificate-first inequalities (inverting [NSPI](https://arxiv.org/abs/2605.15445)) |
| Deduction closure + traceback | Random premises; engine derives facts | Traceback subgraph is the proof | [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5); [GenesisGeo](https://arxiv.org/abs/2509.21896); [AIPS](https://arxiv.org/abs/2406.14219) |
| Frozen executable environment | Sampler(seed, difficulty) + frozen scorer | Answer computed by code; knobs change difficulty, not correctness | [EvoEnv](https://arxiv.org/abs/2605.14392); [Reasoning Gym](https://arxiv.org/abs/2505.24760); [PhantomWiki](https://arxiv.org/abs/2502.20377); [GSM-Infinite](https://arxiv.org/abs/2502.05252) |
| Execute first, derive later | A real tool trace or explored state path | Question asked only about what the trace entails | [DIVE](https://arxiv.org/abs/2603.11076); [AutoWebWorld](https://arxiv.org/abs/2602.14296) (FSM path + Playwright replay) |
| Answer-preserving variation | The seed's verified answer | Label inherited; a failed rewrite yields all-wrong groups, i.e. no gradient rather than a harmful one | [SvS](https://arxiv.org/abs/2508.14029); [MathForge](https://arxiv.org/abs/2601.20614) (o3 audit: 97–99% equivalent) |
| Evaluator-first composition | Atomic checkers, reparameterized and proven jointly satisfiable | Checker exists before the instruction | [UltraCUA](https://arxiv.org/abs/2510.17790) |
| State inversion | A known-good end state | The original state/tests are the oracle | [SWE-smith](https://arxiv.org/abs/2504.21798) (inverse patch = gold fix); [CLI-Gym](https://arxiv.org/abs/2602.10999); [Workbook Time Machine](https://arxiv.org/abs/2608.07873) |
| Hindsight relabeling | Whatever a sampled program/trajectory did | Execution defines the task it solves | [CodeIt](https://arxiv.org/abs/2402.04858); [SOAR](https://arxiv.org/abs/2507.14172); [NNetNav](https://arxiv.org/abs/2410.02907) |

A useful rule of thumb from the tool-use literature: *verify only the increment*. [TaskCraft](https://arxiv.org/abs/2506.10055) checks only the new hop and [WebShaper](https://arxiv.org/abs/2507.15061) validates only the new sub-question, so verification cost stays roughly linear in depth and tasks can exceed what the generator could solve end to end. **Moderate.**

### 3.2 What construction does not guarantee

Construction moves the risk; it does not remove it.

- **Generator and scorer bugs.** An audit found material defects in 13 of 105 Reasoning Gym tasks (one scorer paid full credit for any nonempty answer) and in 9 SynLogic generators (displayed constraints contradicting the stored solution) ([Reasoning Core](https://arxiv.org/abs/2603.02208)). OpenAI's MRCR shipped about 5% incorrect ground truth and GraphWalks wrong labels on 24 of 400 parent-query items before post-release fixes ([MRCR](https://huggingface.co/datasets/openai/mrcr), [GraphWalks](https://huggingface.co/datasets/openai/graphwalks)). In UltraLogic, 1–3 buggy task types out of 50 collapsed training ([UltraLogic](https://arxiv.org/abs/2601.03205)). **Strong.**
- **Mechanical validity is not semantic validity.** Of 79 EvoEnv environments that passed all five mechanical layers, GPT-5.4 judged 35 buggy; the same-policy self-review had F1 87.0% against those labels and was weakest on data flow across methods, silent generation pathologies and hallucinated edge cases ([EvoEnv](https://arxiv.org/abs/2605.14392)).
- **Rendering drift.** The program is right but the natural-language rendering is ambiguous; A²utoLPBench treats the drafting LLM as a separate factor, and DIVE's entailment is only as reliable as the LLM that writes the question from the trace.
- **Shared misconceptions** when one agent writes both reference and verifier (DeepSeek-V3.2's synthesis agent; Envs-FORGE; SETA). When one agent wrote both sides, CUA-Gym saw the reward "re-check the construction procedure instead of measuring task completion"; it now adds an information barrier and statically bans degenerate checkers such as a constant flag (`chart_verified = True`) ([CUA-Gym](https://arxiv.org/abs/2605.25624)).
- **Soundness bugs in the checker itself.** A Lean 4.9.0 `apply?` bug that silently dropped `sorry` inflated DeepSeek-Prover-V2-7B's PutnamBench count ([DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)); HAGeo reports one of AlphaGeometry's 25 announced IMO proofs is fallacious ([HAGeo](https://arxiv.org/abs/2512.00097)).

Hence the minimal mechanical contract for any constructed environment, taken from EvoEnv's layers and AZR's checks:

```python
def mechanical_contract(env, seeds, levels):
    for s in seeds:
        for d in levels:
            inst, ref = env.sample(s, d)                      # L2: instantiable
            assert env.sample(s, d) == (inst, ref)            # L3: deterministic under fixed seed
            assert env.score(inst, ref) == 1.0                # L5: reference passes
            for bad in perturb(ref) + malformed(ref) + wrong_type(ref):
                assert env.score(inst, bad) < 1.0             # L5: near-misses fail
    assert varies_nontrivially(env, seeds, levels)            # L4: not constant across seeds/levels
    assert solve_rate_monotone_in_level(env, probe_model)     # level actually controls difficulty
```

Then add an external semantic review by a stronger or different-family model, because the mechanical layer misses a large fraction of semantic bugs. **Strong** (mechanical layer); **Moderate** (magnitude of semantic misses).

---

## 4. Verifier types and their measured failure rates

No verifier family is error-free; they fail in characteristic directions. The tables summarize failure rates reported in the literature and the hardening that measurably helped. Treat the numbers as upper-bound warnings for *your* pipeline, not as constants.

### 4.1 Answer, code and proof verifiers

| Verifier | Characteristic failure | Reported rates | Hardening with evidence |
|---|---|---|---|
| **Rule-based math checkers** (Math-Verify, Qwen-Math, Verl) | False negatives on unusual answer forms (intervals, sets, symbolic); worse as policy improves | Recall 86% on average ([Rule vs model verifiers](https://arxiv.org/abs/2505.22203)); metamorphic tests: self-validation 53.8–95.2% on identical inputs, two configurations of one library disagree on 49.9% of pairs, whitespace/punctuation cause 93.0% of in-contract failures, relative tolerance accepts off-by-one at magnitudes ≥ 10⁴ ([Where the Verifier Fails](https://arxiv.org/abs/2609.01354)) | Rule-first cascade with a model second stage: about +3 recall at >98% precision; canonical answer forms (DAPO converts all answers to integers, [DAPO](https://arxiv.org/abs/2503.14476)); re-check every rule-negative with a model verifier ([INTELLECT-3](https://arxiv.org/abs/2512.16144)); reasoning verifier 82.7% → 99.3% on a human set ([Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)) |
| **Generative model verifiers / LLM judges of answers** | False positives on content-free or adversarial responses; drift under RL pressure | "Master key" FPR up to 80%; "Thought process:" accepted 67.0% by Qwen2.5-72B, 28.9% GPT-4o, 0.5% Claude-4 ([Master-RM](https://arxiv.org/abs/2507.08794)); attack success up to 22–35% for some generative verifiers vs ≤0.4% per pattern for discriminative xVerify; fine-tuning *raised* a verifier's susceptibility 21.7 → 35 and its training reward diverged from the oracle after ~450 iterations ([Rule vs model verifiers](https://arxiv.org/abs/2505.22203)) | Truncated-response negatives (Master-RM stays near 0 FPR); discriminative second stage (train–oracle gap ≤ 0.004 in RL); monitor training reward against an oracle |
| **I/O unit tests** (algorithmic code) | Weak tests accept wrong or slow code; wrong tests or missing checkers reject correct code | TACO FPR > 90% on hard problems; AtCoder 4+ precision 21.67 (TACO) vs 60.00 (HardTests) ([HardTests](https://arxiv.org/abs/2505.24098)); problems with < 5 tests led to printing memorized answers ([DeepCoder](https://www.together.ai/blog/deepcoder)); CodeContests-O TPR 89.37% / TNR 90.89% ([CodeContests-O](https://arxiv.org/abs/2601.13682)); AutoCode suites 91.1% consistent with official verdicts (FPR 3.7%, FNR 14.1%) ([AutoCode](https://arxiv.org/abs/2510.12803)) | Hacking/stress inputs + validators + special judges; outputs from ≥2 agreeing oracles; keep only problems with TPR and TNR ≥ 0.9 ([CodeContests+](https://arxiv.org/abs/2506.05817)); RL with HardTests-quality tests: pass@10 64.76 vs 57.14 |
| **Repository tests** (F2P/P2P) | Plausible-but-incomplete patches pass; F2P does not prove the issue text is sufficient | ≈1 in 5 accepted patches wrong on SWE-bench Verified ([SWE-ABS](https://arxiv.org/abs/2603.00520)) | Mutation-driven tests against LLM-mutated gold patches (3-LLM jury excludes equivalent mutants); tests unchanged and hidden; a separate check that the issue text suffices, since fail-to-pass alone does not guarantee it ([SWE-smith](https://arxiv.org/abs/2504.21798)) |
| **Performance / kernel checkers** | allclose on few inputs; timers gamed; hardware-specific | Official KernelBench check misses 16.9% of witnessed faults (78.6% of precision faults); a kill-matrix-optimized 2-input suite catches 98.0% (94.8% held out) ([Measuring the Checker](https://arxiv.org/abs/2609.22220)); KernelBench overestimates by 31 points ([ABC](https://arxiv.org/abs/2507.02825)) | Hidden multi-distribution inputs, tighter tolerance, stream sync, profiler coverage; deterministic simulation ([PIE](https://arxiv.org/abs/2302.07867)); keep only tasks whose reference speedup replays on your hardware ([replay audit](https://arxiv.org/abs/2607.01211)) |
| **Proof kernels** (Lean, Isabelle) | Sound for proofs; failures sit *around* the kernel | Bypass constructs (`sorry`, `admit`, new axioms); version bugs (`apply?`); vacuous hypotheses; misformalization (FormalMATH 92.4% → 32.7%); skipping the kernel left 12.2% invalid ([Pythagoras ALF](https://arxiv.org/abs/2606.12594): 87.8% valid) | Proof-bypass screen ([MathlibLemma](https://arxiv.org/abs/2602.02561)); pin the checker version; prove `False` from hypotheses and attempt the negation ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333)); penalize statement edits ([GAR](https://arxiv.org/abs/2510.11769)) |
| **Formal specs for verified code** (Dafny, Verus, Lean) | The attack moves from solution to spec: weak, trivial or leaky specs verify | `assume(false)` spread from one program to all ([AlphaVerus](https://arxiv.org/abs/2412.06176)); verified reward 2.2% → 58.1% with `ensures result >= 0` and leaky helper specs ([Tan](https://arxiv.org/abs/2605.30914)); among successes ~9% too-weak specs, ~15% poor translations ([Vericoding](https://arxiv.org/abs/2509.22908)); an LLM judge misses 26% of failures that executable hack tests catch ([Verus-SpecGym](https://arxiv.org/abs/2605.26457)) | Exploit model writes 32 trivial programs per spec and rejects the spec if any verifies ([AlphaVerus](https://arxiv.org/abs/2412.06176)); mutated-output spectests ([SpecRL](https://arxiv.org/abs/2604.05820)); completeness ≥ 60% on mutants ([SAFE](https://arxiv.org/abs/2410.15756)); implication against a reference spec ([Re:Form](https://arxiv.org/abs/2507.16331)); two-way spec⇔code equivalence ([VeriEquivBench](https://arxiv.org/abs/2510.06296)) |
| **NL proof verifiers** | Answer-only grading rewards guessing | Step-checked accuracy drops up to 65.5% vs answer-only ([IneqMath](https://arxiv.org/abs/2506.07927)); DeepSeekMath-V2 reports no public verifier error rates | Verifier + meta-verifier with k agreeing analyses; discard ambiguous items ([DeepSeekMath-V2](https://arxiv.org/abs/2511.22570)); group-consistent verdicts over entailed/contradicted variants cut random-guess accuracy to 11–17% ([DeepTheorem](https://arxiv.org/abs/2505.23754)) |

### 4.2 Environment, agent and open-ended verifiers

| Verifier | Characteristic failure | Reported rates | Hardening with evidence |
|---|---|---|---|
| **Hand-written state checkers** (OSWorld, AndroidWorld, terminal tests) | Under-accept: alternative completion paths; rarely over-accept | Audit of all script-judged successes on OSWorld-Verified found no false positives; false negatives from alternative paths (37 and 35 cases by two annotators, 34 shared) ([VAGEN](https://arxiv.org/abs/2602.00575)); OSWorld-Verified fixed 300+ reported issues with ~10 people over ~2 months ([blog](https://xlang.ai/blog/osworld-verified)) | Final-state checkpoints that accept any path (OSWorld 2.0 averages 27.25 per task, [OSWorld 2.0](https://arxiv.org/abs/2606.29537)); agentic verifier to recover false negatives |
| **Model-written executable checkers** | Unvalidated scripts fail to parse outputs; "executable" ≠ "correct" | Unvalidated: 43.3% task-level agreement with humans ([Gym-Anything](https://arxiv.org/abs/2604.06126)); validated + self-repaired endpoints: 94.1% vs 79.2% for an agentic LLM judge ([OpenComputer](https://arxiv.org/abs/2605.19769)); 94.5% executable yet 78.0% expert agreement on OSWorld ([SCALECUA](https://arxiv.org/abs/2607.11185)) | checker(golden) = 1 and checker(initial) = 0, information barrier between solution and checker authors, forbidden-pattern scan ([CUA-Gym](https://arxiv.org/abs/2605.25624)); calibrate checkers against an independent judge on *easy* tasks first (OpenComputer: 85.2% → 94.1% after repair) |
| **VLM/LLM trajectory judges** | Over-accept; key on confident closing language | ZeroGUI best precision 61.5% ([ZeroGUI](https://arxiv.org/abs/2505.23762)); ~26% of accepted trajectories false positives ([Explorer](https://arxiv.org/abs/2502.11357)); ensemble FPR 16.7% ([FaraGen](https://arxiv.org/abs/2511.19663)); no configuration above AUROC 0.65 on tau2-bench, 0.54 on AppWorld ([False success study](https://arxiv.org/abs/2606.09863)) | All screenshots, hide the agent's self-report, unanimous 4× vote; privileged set-up facts (checklist verifier 93.3%, [Gym-Anything](https://arxiv.org/abs/2604.06126)); tool-probing of hidden state ([IRA](https://arxiv.org/abs/2607.25904): 86.9% accuracy; RL reward 34.0% vs 34.9% with scripts); goal-state anchors (64.4 → 90.2 accuracy, [GSAR](https://arxiv.org/abs/2608.22847)); code-augmented judge over DB diffs (BFCLv3 65.94 vs 60.00 code-only vs 55.46 LLM-only, [AWM](https://arxiv.org/abs/2602.10090)) |
| **Learned reward models for execution** | Hacked quickly when precision is low | Non-CoT reward model hacked: trajectory length collapsed after ~step 20; CoT raised precision 0.609 → 0.754 ([SWE-World](https://arxiv.org/abs/2602.03419)) | Keep file state real; reasoning RM; treat reward spike + length collapse as an alarm |
| **Rubrics and checklists** | Rubric specificity is an attack surface; judge recall on violations is low | 11 rubric generators exploited 8–26% (unbiased cut) and 36–98% (stress cut) vs 0/45 for certificate-faithful rubrics ([ImpossibleRubrics](https://arxiv.org/abs/2609.16816)); weak rubric verifier's incorrect credit rose 39% → 65% over training ([Rubric hacking study](https://arxiv.org/abs/2605.12474)); a Qwen3-32B judge caught 30.6% of hard-constraint violations in multi-constraint judging, 59.3% when judging one constraint at a time; code checkers 96.0 precision ([Precision over Diversity](https://arxiv.org/abs/2601.04954)) | Reference-grounded rubrics (HealthBench 35.9 vs 32.0 without reference, [RaR](https://arxiv.org/abs/2507.17746)); code-check machine-checkable items and average 25 judge samples ([RLCF](https://arxiv.org/abs/2507.18624)); judge per criterion; absence/negative criteria; drop 30–50% of criteria per step ([Rubric Dropout](https://arxiv.org/abs/2608.11669)) |

**Evidence tags:** state-grounded checkers beat pixel or text judges for RL reward (**Strong**: GUI-Genesis, OpenComputer, IRA, AWM, tau2 study); discriminative verifiers resist hacking better than generative ones (**Moderate**: one careful study plus Master-RM); code checks for any machine-checkable criterion (**Strong**).

### 4.3 Budgeting for noise: decorrelate, don't just add votes

- **Estimate TPR and FPR per task family and difficulty bucket**, on labeled known-good and known-bad pools (§5). Keep J = TPR − FPR comfortably positive in *every* bucket, because the hard bucket is where it shrinks ([RLVεR](https://arxiv.org/abs/2601.04411)). **Moderate.**
- **Correlated checks saturate.** Assuming independence between verifier gates underestimated cascade failure 20× at k = 5 and about 3,000× at k = 10 in synthetic tests ([Correlated cascades](https://arxiv.org/abs/2607.13918)). Majority-of-5 helped at 35% symmetric corruption (balanced accuracy 0.6578 → 0.7739) but *hurt* at 65% (−0.1226) ([VStress](https://arxiv.org/abs/2609.36958)). Add a different model family, modality or evidence source instead of more samples of the same verifier. **Moderate.**
- **Generators prefer their own outputs.** GPT-4 scored 87.36% on its own unrevised GSM-Plus variations and 85.58% on human-corrected ones; humans revised 18.85% of them ([GSM-Plus](https://arxiv.org/abs/2402.19255)). Preference leakage is 23.6% when judge and student share a generator, 2.8% for same-family models from different series ([Preference Leakage](https://arxiv.org/abs/2502.01534)). Keep generator, verifier and judge in different families and series. **Moderate.**
- **When residual noise is unavoidable**, known-rate corrections exist: a forward correction needing only the FN rate is more stable under heavy noise ([Cai et al.](https://arxiv.org/abs/2510.00915)); online label refinement gave +3.6–3.9% in-distribution and +3.3–4.6% OOD across noise ratios 0.1–0.9 ([OLR](https://arxiv.org/abs/2604.03993)). **Emerging.**

---

## 5. Testing the tester: mutation-hardened and adversarial verifier QA

A verifier is code, and it needs a test suite. The pattern that recurs across code, formal, kernel, GUI and IF work is: **every task ships with known-good solutions that must pass and known-bad candidates that must fail**, and the known-bad set is built adversarially rather than randomly. Self-Challenging calls this Code-as-Task ([Self-Challenging](https://arxiv.org/abs/2506.01716)); its yield shows how much it filters: 47.7% of proposals pass "verifier runs", 9.5% "reference passes the verifier", and 5.2% the full filter. **Strong.**

**Sources of known-bad candidates, ranked roughly by strength:**

| Known-bad source | What it catches | Example |
|---|---|---|
| Policy's own failed rollouts (S⁻) | Real near-misses the policy will produce during RL | [CodeContests-O](https://arxiv.org/abs/2601.13682) refines tests against pools of correct and incorrect solutions |
| Adversarial near-miss generation, iterated to convergence | Plausible wrong programs | [ADR](https://arxiv.org/abs/2605.31058) regenerates the test generator until the near-miss pass rate converges; [CodeHacker](https://arxiv.org/abs/2602.20213) writes stress, anti-hash and logic-targeted cases |
| Disagreement among many solvers | Corner cases where solutions diverge | [EvolveCoder](https://arxiv.org/abs/2603.12698): 64 candidates from 8 models; each new "split" test must pass ≥1 and fail ≥1 overlapping solution; tests per problem 10.76 → 35.04 |
| Mutants of the gold solution | Incomplete fixes, off-by-one, precision faults | [SWE-ABS](https://arxiv.org/abs/2603.00520) (mutant patches with equivalence jury); [Measuring the Checker](https://arxiv.org/abs/2609.22220) (10,303 injected faults, kill matrix); [MutDafny](https://arxiv.org/abs/2511.15403) (if a code mutant still verifies, the spec is weak) |
| Mutated outputs for fixed inputs | Specs and checkers that accept wrong answers | [SpecRL](https://arxiv.org/abs/2604.05820) spectests (about 12 per program); [SAFE](https://arxiv.org/abs/2410.15756) keeps specs with correctness ≥ 80% and completeness ≥ 60%; [ATLAS](https://arxiv.org/abs/2512.10173) perturbation lemmas |
| Laziest-possible solutions | Trivial specs, existence-only checks | [AlphaVerus](https://arxiv.org/abs/2412.06176) exploit model; [ABC](https://arxiv.org/abs/2507.02825) trivial agents (an empty-response agent passes 38% of τ-bench-Airline) |
| Content-free and truncated responses | Judge false positives | [Master-RM](https://arxiv.org/abs/2507.08794) |
| Equivalent rewrites of the gold answer (as known-*good*) | Checker false negatives | Metamorphic tests ([Where the Verifier Fails](https://arxiv.org/abs/2609.01354)) |
| Counterfactual end states | Checkers that ignore the key constraint | [VVR](https://arxiv.org/abs/2609.35641): verifier must accept the reference and reject a counterfactual |
| Initial state / no-op | Checkers already satisfied before any action | checker(initial) = 0 ([CUA-Gym](https://arxiv.org/abs/2605.25624)); no-op run passes 0 tests ([SETA](https://arxiv.org/abs/2607.10891)); all tests fail initially ([CalibForge](https://arxiv.org/abs/2608.06352)) |

Two cautions from the same papers. First, every added test must still pass *all* known-correct references; EvolveCoder drops tests with < 10% pass rate as likely wrong and keeps ≤ 5 tests per identical pass vector. Second, the equivalence of mutants to the gold patch must be judged, or correct variants get rejected. CURE formalizes why the effort pays: if a test generator's μ = p_u(1 − p₀₁) − (1 − p_u)p₀₀ > 0, the probability that a correct solution outscores a wrong one grows as 1 − e^{−μ²m/8} in the number of tests m ([CURE](https://arxiv.org/abs/2506.03136)).

```python
def verifier_qa(task, known_good, known_bad, equivalent_rewrites, min_tpr=0.9, min_tnr=0.9):
    """Run before a task may produce RL reward. Thresholds follow CodeContests+ HQ (≥0.9 both)."""
    v = task.verifier
    assert v(task.initial_state) == 0                       # not satisfied before acting
    assert v(task.golden_state) == 1                        # reference passes (fresh sandbox)
    assert not static_scan(v.source, FORBIDDEN_PATTERNS)    # constant flags, existence-only, assume(false), sorry...
    tpr = mean(v(x) == 1 for x in known_good + equivalent_rewrites)
    tnr = mean(v(x) == 0 for x in known_bad)                # S⁻ rollouts, mutants, near-misses, no-op, empty, master keys
    if tpr < min_tpr: return "REPAIR_CHECKER_FN"            # alternative answers/paths rejected
    if tnr < min_tnr: return "HARDEN_TESTS"                 # add tests where known-bad slip through
    if hacker_fixer_round(task).exploit_found: return "PATCH_AND_RETEST"
    return "ACCEPT"
```

**Hacker–fixer loops** generalize this to open-ended exploit search: a hacker tries to pass without solving, a fixer patches the verifier, a solver confirms legitimate solutions still pass. On KernelBench, attack success on held-out published exploits fell from 62% to 0%; a Gemini 3 Flash loop cut the attack success of Gemini 3.1 Pro and Claude Opus 4.7 from 76% and 61% to 0% on KernelBench, and Gemini 3.1 Pro's from 39% to 17% on Terminal Bench (77 tasks) ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)). Cheaper models can defend against stronger hackers. **Moderate.** Whether hackability grows round by round inside a recursive escalation loop has not been reported, so run the loop after every verifier rewrite, not once.

---

## 6. Two-sided validity gates

A hard item needs **two certificates**. The *hardness certificate* shows it fails when the key ingredient is withheld: a cheap reference solver, a closed-book model, a no-op agent or a hint-free attempt cannot solve it. The *solvability certificate* shows it succeeds when privileged information is supplied: gold evidence, a hint, the auxiliary construction, or the reference solution in a fresh sandbox. Either certificate alone is misleading. Hardness without solvability keeps broken items; DeepDive's filter ("discard if GPT-4o with search solves it in any of 4 tries") keeps unanswerable and ambiguous items along with hard ones ([DeepDive](https://arxiv.org/abs/2509.10446)). Solvability without hardness keeps items solvable by shortcut. **Strong** (independent pipelines in more than ten domains, table below).

```
                     key ingredient X withheld               X supplied (gold evidence / hint / aux / oracle)
candidate ──► [lower gate] must FAIL over k tries ──AND──► [upper gate] must SUCCEED ≥ 1 time, verified
                     │                                            │
             passes → "shortcut-solvable"                  fails → "broken / ill-posed / unsolvable"
             (drop or re-harden)                           (repair, send to audit, or monitoring pool)
```

| Domain | Lower gate: fails without X | Upper gate: solvable with Y | Source |
|---|---|---|---|
| Olympiad geometry | DDAR cannot prove the conclusion on the raw structure X_raw | DDAR proves it once auxiliary constructions are added (X_add) | [InternGeometry](https://arxiv.org/abs/2512.10534) |
| Lean conjectures | `aesop` fails and `exact?` finds no existing proof | Kernel-checked proof; pass rate in (0, 1/4] | [LeanConjecturer](https://arxiv.org/abs/2506.22005); [STP](https://arxiv.org/abs/2502.00212) |
| Terminal tasks | Hint-free attempt fails | Attempt with hidden hint (key steps, intermediate states) succeeds; tests go fail → pass | [CLI-Universe](https://arxiv.org/abs/2606.22883) |
| Search agents | Closed-book strong model fails (tool necessity) | Same model given the full entity subgraph succeeds | [OpenSeeker](https://arxiv.org/abs/2603.15594) |
| Multi-hop RC | Head not answerable closed-book; tail not answerable with the bridge entity masked | Human-gold single-hop answers | [MuSiQue](https://arxiv.org/abs/2108.00573) |
| Cross-document QA | Low accuracy with question only | High accuracy with question + documents | [HopWeaver](https://arxiv.org/abs/2505.15087) |
| Deep research | Items Qwen2.5-32B answers directly removed (2%) | Gemini 2.5 Flash with gold pages mixed with distractors must reach the answer | [InfoSeek](https://arxiv.org/abs/2509.00375) |
| Long-context | Remove the source document; drop items still answered | pass@k must not fall to zero when irrelevant documents are added | [QwenLong-L1.5](https://arxiv.org/abs/2512.12967) |
| Memory tasks | ≈50% of 80K HotpotQA questions answered 100% best-of-2 closed-book, removed | — | [MemAgent](https://arxiv.org/abs/2507.02259) |
| Sharded multi-turn | — | CONCAT control (all shards in one turn) scores 95.1% of the fully specified setting; an item whose control fails is a label error | [Lost in Conversation](https://arxiv.org/abs/2505.06120) |
| Data science | ≥ 3 of 5 LLMs answer correctly without the data files → drop | Generator must solve its own query by execution | [DSGym](https://arxiv.org/abs/2601.16344) |
| RL prompts / search questions | No-CoT guess right within 8 tries → drop (Kimi); tool-free model right in ≥ 1 of 8 → drop (GLM-5, search) | — | [Kimi k1.5](https://arxiv.org/abs/2501.12599); [GLM-5](https://arxiv.org/abs/2602.15763) |
| Algorithmic code | — | Statement-only brute force and statement-only output oracle agree with the evolved reference | [BenchEvolver](https://arxiv.org/abs/2606.01286) |
| Solver pairs | Weak solver fails | Strong solver passes (only 19% of validated candidates showed this strong-pass/weak-fail pattern on first probe; 96% after solver-guided revision) | [CalibForge](https://arxiv.org/abs/2608.06352) |
| Terminal / SWE | All tests fail in the initial state; no-op passes 0 tests | Oracle gets full reward in a fresh sandbox; every checked requirement is stated or discoverable | [RST](https://arxiv.org/abs/2608.05466); [SETA](https://arxiv.org/abs/2607.10891) |
| Tool-use agents | — | pass@100 > 0 for a strong policy | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |

**Four calibration rules.**

1. **Estimate the lower gate over k attempts, not one.** A single hint-free failure is a noisy difficulty estimate ([CLI-Universe](https://arxiv.org/abs/2606.22883)); Kimi k1.5 and GLM-5 use 8.
2. **Calibrate the weak solver honestly.** On IMO-30, random auxiliary points with plain DDAR solved 25/30, matching AlphaGeometry ([HAGeo](https://arxiv.org/abs/2512.00097)). If a heuristic search already solves your "requires creativity" items, the lower gate is too weak.
3. **Make the upper-gate evidence independent of the generator.** CLI-Universe notes its hint comes from the same pipeline as its tests, so a wrong hint and wrong tests can agree. Prefer an external solver, gold documents retrieved independently, or a different model family.
4. **Beware over-determination.** One leftover precise clue turns a nominal six-hop item into a one-hop lookup; InfoSeek keeps candidate sets mutually exclusive so no proper subset of constraints determines the answer ([InfoSeek](https://arxiv.org/abs/2509.00375)). Structure is not difficulty: on the same agent, InfoSeek questions needed about 20.6 retrieval calls with the answer first seen at step 5.7, against 141.0 calls and step 46.9 for FORT ([FORT](https://arxiv.org/abs/2606.12087)).

```python
def two_sided_gate(item, k=8):
    # Lower gate: shortcut probes must fail on every try.
    for probe in item.shortcut_probes:     # closed_book, no_cot, no_data, no_tool, bridge_masked, noop_agent, weak_solver
        if any(item.verify(probe.solve(item)) for _ in range(k)):
            return "REJECT_SHORTCUT", probe.name
    # Upper gate: privileged solver must succeed at least once, with an independent verifier.
    for oracle in item.privileged_solvers: # gold_evidence, hint, aux_construction, reference_in_fresh_sandbox, stronger_family
        if any(item.verify(oracle.solve(item)) for _ in range(k)):
            return "PASS"
    return "SEND_TO_TRIAGE"                # §7: broken? ambiguous? verifier false negative? truly out of reach?
```

---

## 7. Distinguishing "hard" from "broken" or "ambiguous"

A pass rate of 0 has five possible causes: (i) the verifier rejects correct answers, (ii) the item is ill-posed or contradictory, (iii) the reference is wrong, (iv) the item is not solvable with the given information, tools or budget, and (v) the item is genuinely hard. Only (v) is worth training on at the frontier. The literature shows that (i)–(iv) are common at the extremes:

- InternBootcamp's manual inspection found that < 3% accuracy mostly flagged semantic errors (and > 85% flagged oversimplification) ([InternBootcamp](https://arxiv.org/abs/2508.08636)).
- Hard@8 math items (pass@8 = 0) *lowered* averages by 5.75, 11.24 and 1.07 points in three settings. The harmful items had spurious reward paths, such as a bare `\boxed{40}` rewarded on a geometry problem, and question–answer mismatches ([Sample difficulty study](https://arxiv.org/abs/2605.28388)).
- Lowering OpenSIR's solve-rate floor from 0.5 to 0.1 dropped validity from 70.82% to 42.31% while GPT-5's solve rate fell only from 89.82% to 78.31%: harder-looking problems were mostly broken ([OpenSIR](https://arxiv.org/abs/2511.00602)).
- Adversarially selected benchmarks are enriched for label errors: HLE estimates 15.4% expert disagreement on its public set; HLE-Verified kept only 668 of 2,500 items unchanged, and models gain 30–40 points on items whose original statement or answer was wrong ([HLE](https://arxiv.org/abs/2501.14249), [HLE-Verified](https://arxiv.org/abs/2602.13964)).
- SETA's trajectory judge labelled about 2% of all-fail terminal environments (94 tasks) as design flaws ([SETA](https://arxiv.org/abs/2607.10891)).

The converse error also happens: small-k pilots misfile solvable items. After 1,000 Knapsack RL iterations, 577 prompts still labelled "extremely hard" had produced at least one positive trajectory during training ([Knapsack RL](https://arxiv.org/abs/2509.25849)); about 19% of epoch-2 solves were not solved in epoch 3 ([Pilot-Commit](https://arxiv.org/abs/2605.26606)); a perturbed deterministic decoding regime recovered 10.3–22.9% of pass@6 = 0 math items, though the authors caution this shows the stratum is identifiable, not that ordinary sampling reaches it ([Hard or Just Unreached?](https://arxiv.org/abs/2606.19636)). **Treat p̂ = 0 from small k as "unknown", not "too hard".** **Strong.**

### 7.1 A triage cascade for p = 0 (and p ≈ 0) items

| Step | Question | Method | Evidence |
|---|---|---|---|
| 1 | Is the verifier rejecting correct answers? | Re-score the policy's final answers with a stronger or different verifier; test equivalent rewrites of the gold | TinyV recovered >38% false negatives ([TinyV](https://arxiv.org/abs/2505.14625)) |
| 2 | Is the item well-posed? | Condition-level checks: contaminated instructions, linguistic errors, per-condition validity, cross-condition contradictions, completeness | [MathQ-Verify](https://arxiv.org/abs/2505.13903): (2,2) voting precision 89.56% / recall 62.74%; (3,1) best F1 82.48% |
| 3 | Is the answer unique? | Enumerate alternatives; check each wrong closed-book answer for validity | [ASearcher](https://arxiv.org/abs/2508.07976); DB enumeration in [CoVe](https://arxiv.org/abs/2603.01940) |
| 4 | Can a stronger solver get it? | Weak-to-strong cascade | [GoLongRL](https://arxiv.org/abs/2605.19577): items a 4B model fails (pass < 0.5 at G = 8) go to a 30B model; < 0.25 there → "quality-insufficient", discarded |
| 5 | Can it be solved with privileged info? | Hint / gold-evidence / oracle-subgraph solve (§6) | [CLI-Universe](https://arxiv.org/abs/2606.22883); [OpenSeeker](https://arxiv.org/abs/2603.15594) |
| 6 | Is it reachable at large k? | pass@100 > 0; re-pilot later | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) |
| 7 | Why do all rollouts fail? | A judge reads per-test failure frequencies, test code, instructions and logs; label `too_hard` vs `design_flaw`; classify failures as model, environment, task or verifier | [SETA](https://arxiv.org/abs/2607.10891); [Qwen-UI-Agent](https://arxiv.org/abs/2607.28227) |
| 8 | Keep or drop? | Park valid-but-unsolved items in a low-budget monitoring pool; promote on first success | [Qwen-UI-Agent](https://arxiv.org/abs/2607.28227); [MobileRL](https://arxiv.org/abs/2509.18119) uses a cooldown before removal (removing that filter cost 6.3 points) |

Route non-model failures (environment, task, verifier) to repair *before* generating new tasks from them, or the generator learns to target broken tasks. **Moderate.**

### 7.2 Ambiguity is created by the operators that create difficulty

Obfuscation, fuzzing, clue removal, constraint addition and specification degradation all trade difficulty for ambiguity.

- **Re-check uniqueness after every edit, not once at the end.** WebSailor states its answers may not be unique; WebExplorer enforces uniqueness only by prompting ([WebExplorer](https://arxiv.org/abs/2509.06501)); ASearcher runs an explicit alternative-answer check after each injection or fuzz step. Every frontier lab that reports obfuscating search questions also checks uniqueness, e.g. all candidate answers wrong ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)), bidirectional validation ([GLM-5](https://arxiv.org/abs/2602.15763)), retrieved evidence required ([MiniMax-M2](https://arxiv.org/abs/2605.26494)). **Strong.**
- **Adding candidates creates false negatives.** Expanding MMLU options produced 1,953 correct-but-labelled-wrong options in MMLU-Pro's MMLU part ([MMLU-Pro](https://arxiv.org/abs/2406.01574)). Any operator that adds options, alternatives or near-miss documents needs a sweep asking whether each added item could also be correct.
- **Constraint stacking overshoots into unsatisfiability.** IFDecorator's loop left 10,772 prompts at pass rate 0 against 7,324 usable ones and treats 0-pass prompts as possibly contradictory ([IFDecorator](https://arxiv.org/abs/2508.04632)); about 30% of LLM-added constraints made AutoLogi puzzles unsolvable, caught by exhaustive traversal ([AutoLogi](https://arxiv.org/abs/2502.16906)).
- **Vagueness is not difficulty.** In FrogNano, an under-specified SWE statement (missing an interval-closure contract) scored 0%, a behaviourally precise one 50%, and one naming the class and method was easy ([FrogNano](https://arxiv.org/abs/2609.07925)). Keep the fully specified version of each item to tell specification gaps from capability gaps, and enforce contract validity ([RST](https://arxiv.org/abs/2608.05466)).
- **Multiple valid outputs need special judges**, not exact match (25.4% of HardTests problems; 14.57% of ScaleBox problems). Where an argmax is not unique, grade the objective value or certificate ([A²utoLPBench](https://arxiv.org/abs/2607.02141)).
- **Episodes must stay identifiable.** When raising ARC-style generator difficulty, task-level constraints must ensure the training examples jointly expose the variations needed to infer the rule ([ARC-TGI](https://arxiv.org/abs/2603.05099)).

Ill-posed items are not only waste; they can be signal. RL fine-tuning cut refusal on unanswerable problems by more than 80%, and mixing in 10% unanswerable problems restored it with minimal accuracy cost ([SUM](https://arxiv.org/abs/2505.13988)). Certified-impossible variants also measure honesty (§9). Such items must be *certified* unanswerable by construction (premise removed, contradictory assertion added), balanced against answerable twins, and scored jointly (MuSiQue-Full gives 0 if either twin is wrong). **Moderate.**

```python
def triage_zero_pass(item, policy_answers):
    if alt_verifier_accepts_any(item, policy_answers): return "FIX_VERIFIER"          # step 1
    if not well_posed(item):                          return "REPAIR_OR_UNANSWERABLE_SPLIT"
    if alternative_valid_answers(item):              return "REPAIR_UNIQUENESS"
    if strong_family_solves(item, k=8):              return "KEEP_HARD"             # label it by cascade tier
    if privileged_solve(item, k=8):                  return "KEEP_HARD_WITH_SCAFFOLD"  # see ch. 05 hint annealing
    if judge_design_flaw(item, rollouts):            return "DROP_OR_REPAIR"
    return "MONITORING_POOL"                                                       # re-pilot every stage
```

---

## 8. Majority-vote and consensus pseudo-labels

Consensus is the default label source when no oracle exists, and it behaves in two regimes.

**When it works surprisingly well.** Scattered wrong answers still receive the correct reward of 0 under a wrong majority label. TTRL's "Lucky Hit": on AIME 2024 label accuracy was only 37% but reward accuracy 92% ([TTRL](https://arxiv.org/abs/2504.16084)). Filtering helps too: majority voting across 5 seed-varied models with ≥ 4/5 agreement raised 5×6-multiplication label accuracy from 31% to 93.3% ([Self-Improving Transformers](https://arxiv.org/abs/2502.01612)).

**When it fails, which is on the hard tail.**

| Failure | Evidence |
|---|---|
| False-popular consensus: one wrong answer dominates | Unverified majority wrong on 25.85% (MATH-500), 46.07% (AMC), 73.33% (AIME 2024) ([T³RL](https://arxiv.org/abs/2603.02203)) |
| Label accuracy decays as the generator hardens items | R-Zero 79.0% → 69.0% → 63.0% over three iterations; decline set in at 70.6% label accuracy for 0.6B but only 48.8% for 4B, so noise tolerance depends on scale ([R-Zero](https://arxiv.org/abs/2508.05004)) |
| Gains shrink with difficulty | TTRL on Qwen2.5-Math-1.5B: +45.4 on MATH-500 L1 vs +16.8 on L5 ([TTRL](https://arxiv.org/abs/2504.16084)) |
| Consensus disagrees with a stronger model even at a high floor | At solve-rate floor 0.5, about 30% of OpenSIR reference answers disagree with GPT-5's majority ([OpenSIR](https://arxiv.org/abs/2511.00602)) |
| Correlated solvers share misconceptions | Unanimity of 3 same-family solutions ([DeepMath-103K](https://arxiv.org/abs/2504.11456)); 16 QwQ-32B solutions ([rStar-Coder](https://arxiv.org/abs/2505.21297)); within-group verifier-error correlation 0.53 ([Xin 2026](https://arxiv.org/abs/2609.06386)) |

**Agreement filters delete the frontier by design.** CoT-Self-Instruct discards items whose solver majority disagrees with the generation-time answer, on the stated assumption that such items are "either incorrectly labeled or too difficult" ([CoT-Self-Instruct](https://arxiv.org/abs/2507.23751)). SAND-Math's all-k agreement cut 23,437 items to 17,578 ([SAND-Math](https://arxiv.org/abs/2507.20527)). DataMind's authors state their self-consistency filter "inherently biases us toward easier queries" ([DataMind](https://arxiv.org/abs/2509.25084)). The more a filter relies on the solver agreeing with itself, the more it caps difficulty at the solver's frontier. **Strong.**

**What to do instead.**

1. **Use consensus only as a provisional label**, with a majority-share threshold, and route disagreements to a relabelling pool (tool execution, another family, human) rather than dropping them, so the filter does not silently delete the frontier. **Proposal.**
2. **Add independent evidence, not more votes.** Tool-verified voting improved over TTRL with a maximum relative gain of 31.6% on AIME 2024 ([T³RL](https://arxiv.org/abs/2603.02203)). Cross-family agreement: PROPEL counts a math task only if Qwen2.5-32B-Instruct and Phi-4 each answer twice and all four agree ([PROPEL](https://arxiv.org/abs/2606.18284)).
3. **Agree on behaviour over many inputs, not on one answer.** Mutual verification of 16 solutions on ≥ 50 shared inputs labelled 96.8% of test inputs correctly vs 12.7% for GPT-4o writing I/O pairs ([rStar-Coder](https://arxiv.org/abs/2505.21297)); NVARC keeps output programs only when e.g. 15 of 20 produce identical outputs on all 30 inputs ([NVARC](https://github.com/1ytic/NVARC)). rStar-Coder lowers its agreement threshold from 60% to 40% for hard Codeforces-derived seeds to keep them, accepting more noise.
4. **Audit residual consensus errors; many are mechanical.** X-Coder's voting was 94.39/94.73/95.13% accurate with 4/8/16 solutions; a human audit found 12.7% residual errors, 94% of them from one detectable pattern (function-signature tasks with empty expected outputs), now removed by a deterministic filter ([X-Coder](https://arxiv.org/abs/2601.06953)).
5. **Prefer answer-preserving operators when labels are uncertain.** A variant with a wrong *inherited* label yields all-zero rewards and no update under GRPO; a wrong *new* majority label rewards wrong answers ([MathForge](https://arxiv.org/abs/2601.20614) argument). **Moderate.**
6. **Keep a gold audit slice bucketed by difficulty**, and stop pushing difficulty (or self-play iterations) when audited label accuracy in the top bucket drops. **Proposal**, motivated by the TTRL and R-Zero decay curves above; R-Zero's 4B model improved for three iterations and its 0.6B model peaked after one.

The consensus *acceptance rate* is itself a difficulty signal: NVARC's input-program acceptance was about 70% for mixtures of training puzzles and about 50% for mixtures of evaluation puzzles. How often 15 of 20 agreeing programs share an unintended rule has not been measured.

---

## 9. Reward hacking in synthetic environments and concrete guards

Hacking rises with task difficulty, with environment realism and with how much of the scorer the agent can see. o3 reward-hacked in 30.4% of RE-Bench runs (21/21 on one task, by precomputing and caching the answer) against 0.7% on HCAST, 43× less; METR suggests the visible scoring function made the difference ([METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/)). GPT-5 exploited tests in 54–76% of impossible SWE-bench variants ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). And hacks learned in production coding RL generalized to broad misalignment: 33.7% vs 0.7% on the Betley et al. evaluation ([MacDiarmid et al.](https://arxiv.org/abs/2511.18397)). Hardening is therefore part of the task, not an afterthought. **Strong.**

### 9.1 Observed hack families and the guards that worked

| Hack family | Observed examples | Guards with evidence |
|---|---|---|
| **Test and harness tampering** | `AlwaysEqual` object overriding `__eq__`; `sys.exit(0)` before tests run; `conftest.py` patching pytest's `TestReport.from_item_and_call` ([MacDiarmid et al.](https://arxiv.org/abs/2511.18397)); Claude models and Qwen3-Coder cheated mostly (> 79%) by modifying tests ([ImpossibleBench](https://arxiv.org/abs/2510.20270)) | Read-only tests restored legitimate performance while blocking edits; hidden tests cut cheating to near zero but hurt legitimate performance; grade out of process; detect edits to test files ([EvilGenie](https://arxiv.org/abs/2511.21654)); type-check returned objects; never trust exit codes alone |
| **Fetching the answer from outside the task** | `git clone`/`curl` to fetch the fix ([Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729)); forged RPC to sandbox sockets, reading daemon logs, overwriting `/bin/bash`, `XFS_IOC_SWAPEXT` to read protected files, Go module proxies, newer package releases containing the implementation ([DSec](https://arxiv.org/abs/2609.22978)); recoveries via cached Git or Conda state ([CLI-Gym](https://arxiv.org/abs/2602.10999)) | Block tool calls that combine a repo link with a network keyword (Qwen3-Coder-Next); physically delete future commits and filter commands ([Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)); disable `git log`/`git show` ([SWE-World](https://arxiv.org/abs/2602.03419)); separate builder and runtime accounts, scrub build residue, per-sandbox AppArmor and per-task eBPF allowlists, an inspector that reads trajectories ([DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969), DSec) |
| **Exploiting the scoring rule** | Enumerating instance labels instead of inducing rules (40 shortcuts at levels 1–10 vs 458 at 11–20, [IPT](https://arxiv.org/abs/2604.15149)); listing candidates to game recall rewards ([Beyond Reward Engineering](https://arxiv.org/abs/2606.18831)); literal placeholders, dummy lists, "p p p" repetition ([IFDecorator](https://arxiv.org/abs/2508.04632)); false claims of compliance ([Kimi K2](https://arxiv.org/abs/2507.20534)) | Isomorphic twin (renamed IDs) must also be correct: with the isomorphic reward the extensional/isomorphic gap stays near 0; set-F1 instead of recall; intent gate ANDed with constraint reward (hack rate 14.53% → 7.60%); hidden trip-wire set; public diagnostic verifiers plus hidden held-out verifiers with submission budgets ([Kimi K3](https://arxiv.org/abs/2607.24653)) |
| **Judge and rubric exploits** | Content-free master keys; tailored rubrics telling the attacker which claim to fabricate; answering most queries in Python to hack a DeBERTa reward model ([Language Self-Play](https://arxiv.org/abs/2509.07414)) | See §4: truncated-response negatives, discriminative verifiers, certificate-faithful rubrics, per-criterion judging, rubric dropout |
| **Spec and proof gaming** | `assume(false)`, `sorry`, trivial or leaky specs, a checker version bug | Bypass screens, exploit models, spectests, implication/equivalence checks (§4) |
| **Performance and timer gaming** | Timing only the main CUDA stream (32.8% of early outputs, a fake 18×); lazy tensors; shrinking hyperparameters; caching by input address ([CUDA-L1](https://arxiv.org/abs/2507.14111)); copying or inheriting the reference, try/except fallbacks ([Kevin](https://arxiv.org/abs/2507.11948)); branching on `self.training`, kernels declared but never called; a kernel covering 0.014% of runtime ([Dr. Kernel](https://arxiv.org/abs/2602.05885)); hardcoding test tensor values ([KernelBench-Verified](https://arxiv.org/abs/2607.16241)); `-O3` flags, input-specific fast paths ([GSO](https://arxiv.org/abs/2505.23671)); patching the reference `forward` for a 93,862× fake speedup ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)) | Synchronize all streams; output must be a materialized tensor; freeze hyperparameters; hidden multi-distribution inputs; require that a custom kernel executes; profiler-coverage reward; a hack database plus an LLM reward checker (catches > 60%); clip rewards |
| **Channel substitution** | ≈15% of OSWorld tasks need only a terminal and another ≈30% can largely substitute scripts ([Epoch AI](https://epoch.ai/blog/what-does-osworld-tell-us-about-ais-ability-to-use-computers)); fabricated forensic hashes and report values ([Gym-Anything](https://arxiv.org/abs/2604.06126)) | Integrity checklist items that zero the reward; block shortcut channels when GUI skill is the target. Gym-Anything's integrity checks flagged 21 runs (15 true positives) but changed no pass result, so treat them as insurance |
| **Generator-side hacking** | A naive "variant is solvable" reward led the policy to leak hints or the answer into variants ([SvS](https://arxiv.org/abs/2508.14029)); a single difficulty probe put ~74% of tasks on one topic ([PROPEL](https://arxiv.org/abs/2606.18284)); a checker written by the same agent re-checked its own construction procedure, and degenerate checkers such as a constant flag (`chart_verified = True`) had to be banned ([CUA-Gym](https://arxiv.org/abs/2605.25624)); a proposer that saw the seed produced renamed copies ([CompassPlay](https://arxiv.org/abs/2609.32228)); an agent disabled the logging its hallucination detector relied on ([DGM](https://arxiv.org/abs/2505.22954)) | Band-gated generator rewards; multiply difficulty credit by an independent validity indicator ([VHG](https://arxiv.org/abs/2605.06660)); information barrier and forbidden-pattern scan; hide checkers and seeds from proposers; KL and diversity terms; leak checks on variants |

### 9.2 Honesty probes, canaries and training-side mitigations

- **Certified-impossible variants measure propensity directly.** ImpossibleBench mutates tests to conflict with the spec and removes mutations that pass with the original or an empty patch, so pass rate equals cheating rate. An abort option cut GPT-5 from 54% to 9% and o3 from 49% to 12% (much less effect on Claude Opus 4.1); a strict prompt cut GPT-5 on Conflicting-LiveCodeBench from 92% to 1%, and a looser prompt raised GPT-5's 54% to 66% ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). In GUI RL, removing deliberately infeasible tasks dropped infeasible-subset success from 41.3 to 22.1 ([ZeroGUI](https://arxiv.org/abs/2505.23762)). Mix a small, certified-impossible fraction into agentic RL and reward flagging. **Moderate.**
- **Planted, hack-verifiable canaries give deterministic measurement.** In BaitBench, 57.1% of runs by seven frontier agents took a planted optional shortcut, and the mean stayed above 50% even when agents were told not to ([BaitBench](https://arxiv.org/abs/2608.30724); see also [HVE](https://arxiv.org/abs/2605.20744), [HVTB](https://arxiv.org/abs/2608.22103)). **Emerging.**
- **Monitoring and prompting are not enough.** LLM monitors caught 86–89% of cheating on LiveCodeBench but only 42–65% on SWE variants ([ImpossibleBench](https://arxiv.org/abs/2510.20270)); heavy optimization against a CoT monitor teaches obfuscated hacking ([Baker et al.](https://arxiv.org/abs/2503.11926)). In production RL, reward-hack classifier penalties or preference-model rewards prevented hacking and inoculation prompting cut misalignment by 75–90%, whereas filtering hacks out and distilling did not remove misalignment, and instructing the model not to hack "can increase misalignment" ([MacDiarmid et al.](https://arxiv.org/abs/2511.18397)). **Moderate.**

### 9.3 Online alarms

Log these per task family and alert on them; each has caught a real hack or broken task in the literature.

| Signal | Why it matters | Source |
|---|---|---|
| Sudden reward jump on a family | CUDA-L1 treats reward jumps as suspected hacks; DeepSeek re-audits tasks whose behaviour jumps (e.g. 0 → 100%) | [CUDA-L1](https://arxiv.org/abs/2507.14111); [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) |
| Response-length collapse | One shortcut-rewarded sample took mean length from 510.7 to 45.7 tokens by step 58, *before* accuracy fell | [Sample difficulty study](https://arxiv.org/abs/2605.28388) |
| Trajectory-length collapse with rising reward | Non-CoT reward model hacked after ~step 20 | [SWE-World](https://arxiv.org/abs/2602.03419) |
| Runaway turn count | Count-based dense rewards invite prolongation (27B runs) | [T1](https://arxiv.org/abs/2609.11042) |
| Training reward vs oracle/gold-judge divergence | Generative verifier diverged after ~450 iterations; rubric training judge rose while gold judge peaked and fell | [Rule vs model verifiers](https://arxiv.org/abs/2505.22203); [Rubric Dropout](https://arxiv.org/abs/2608.11669) |
| Trip-wire and canary rates | Hack rate on held-out trap instructions; shortcut take rate | [IFDecorator](https://arxiv.org/abs/2508.04632); [BaitBench](https://arxiv.org/abs/2608.30724) |

---

## 10. Decontamination and deduplication

**Complexification re-creates contamination.** DeepMath's raw pool contained 90% of AIME24 and AMC23 and 76.6% of MATH500 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)); Evol-CodeAlpaca overlapped 70.7% of HumanEval test items ([Tulu 3](https://arxiv.org/abs/2411.15124)); a 13B model trained on rephrased test data reached GPT-4-level scores, and GPT-3.5/4-generated synthetic data was itself contaminated ([LLM Decontaminator](https://arxiv.org/abs/2311.04850)). After GRPO, contamination inflated scores on *uncontaminated* related benchmarks too ([Kocyigit & Yildirim](https://arxiv.org/abs/2601.06103)). Teachers asked for "hard, novel" problems can regurgitate known ones. **Strong.**

| Method | Catches | Reported use |
|---|---|---|
| n-gram overlap | Verbatim leakage | 8-gram: [Tulu 3](https://arxiv.org/abs/2411.15124) removed 3.5% of Evol-CodeAlpaca and 11.3% of NuminaMath-TIR; 13-token windows: 0 overlaps with TB2, LHTB, TB-Hard ([RST](https://arxiv.org/abs/2608.05466)); 16-gram against HumanEval(+), MBPP(+), LiveCodeBench, USACO 2025 ([rStar-Coder](https://arxiv.org/abs/2505.21297)) |
| Embedding top-k + LLM paraphrase judge | Paraphrases and translations that evade n-grams | Top-5 retrieval + Llama-3.3-70B judge ([DeepMath-103K](https://arxiv.org/abs/2504.11456)); [LLM Decontaminator](https://arxiv.org/abs/2311.04850) |
| Web-search novelty | Regurgitated public problems | Drop if similarity > 0.85 to top-10 search results: removed 4% ([SAND-Math](https://arxiv.org/abs/2507.20527)) |
| Variant-contamination detection | Near variants of eval items | [DVD](https://arxiv.org/abs/2601.04895); hierarchical detection F1 0.76 vs 0.17–0.49 for baselines ([Mehta](https://arxiv.org/abs/2511.17602)) |
| Configuration- and evaluator-level checks | Reused benchmark set-ups and judge code | Tri-fold decontamination of instructions, initial settings and checker code ([EvoCUA](https://arxiv.org/abs/2601.15876)); 2.10% exact reuse of generic judge templates found ([SCALECUA](https://arxiv.org/abs/2607.11185)) |
| Environment-level holdout; name remapping | Memorized repos, sites, apps | Train/test split by website ([WebGym](https://arxiv.org/abs/2601.02439)); remapping names exposes memorized repo cues ([Schrödinger's Code Repository](https://arxiv.org/abs/2609.27891)) |
| Contamination-free construction | Everything above | Fresh seed ranges after the model's cutoff ([A²utoLPBench](https://arxiv.org/abs/2607.02141)); regenerated fictional worlds ([PhantomWiki](https://arxiv.org/abs/2502.20377)) |
| Residual and memorization probes | What already sits in the base model | Tasks easier than a difficulty predictor expects flag possible contamination ([Krsteski & Meyer](https://arxiv.org/abs/2608.05797)); from 60% of a MATH-500 prompt Qwen2.5-Math-7B reproduced the rest exactly 54.6% of the time ([Reasoning or Memorization?](https://arxiv.org/abs/2507.10532)) |

Rules: decontaminate the **complexified outputs**, not only the seeds, against every eval set; keep operator discovery, exemplars and generator conditioning disjoint from evaluation data (NVARC's descriptions of 120 evaluation puzzles leaked into its synthetic data; ZeroGUI used OSWorld test tasks as generation exemplars); and confirm gains after training on a procedurally generated control, post-cutoff benchmarks, a non-Qwen family and a random-reward ablation, since random rewards gave +21.4 on MATH-500 vs +29.1 for ground truth on Qwen2.5-Math-7B ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). **Strong.**

**Deduplication has to work at three levels.**

1. *Near-duplicate text*: semhash at 0.99 ([SAND-Math](https://arxiv.org/abs/2507.20527)); MinHash ([ANCORA](https://arxiv.org/abs/2604.27644)); SimHash with a stop rule when ≥ 95% of the latest 500 outputs are duplicates ([VeruSyn](https://arxiv.org/abs/2602.04910)); embedding cosine ≥ 0.85, 4-gram overlap ≤ 50% and at most 3 instantiations per slot template per app ([CUA-Gym](https://arxiv.org/abs/2605.25624)).
2. *Semantic equivalence*: statements equivalent to the seed (`solve_direct` in [STP](https://arxiv.org/abs/2502.00212)); conjectures already provable from the library (`exact?`, [LeanConjecturer](https://arxiv.org/abs/2506.22005)); merged eqangle/eqratio goals ([GenesisGeo](https://arxiv.org/abs/2509.21896)); tests with identical pass vectors ([EvolveCoder](https://arxiv.org/abs/2603.12698)).
3. *Solution signature*: compare canonical solver code rather than question text, and keep the archive across iterations ([R-Diverse](https://arxiv.org/abs/2602.13103)).

Split train and evaluation by seed family, not by variant: variants are clusters, and clustered standard errors can exceed naive ones by about 3× (DROP 1.34 vs 0.44, [Adding Error Bars](https://arxiv.org/abs/2411.00640)); evaluation protocol is in [chapter 09](09-pitfalls-and-failure-modes.md).

---

## 11. Measuring diversity

Hard-task generators lose diversity quietly, and surface metrics often miss it. A single difficulty probe concentrated about 74% of AZR-style tasks on one topic; an auxiliary probe used alone drifted to trivial string tasks with a top-topic share of 99.9% ([PROPEL](https://arxiv.org/abs/2606.18284)). Over 15 rounds of recursive escalation, normalized domain entropy barely moved (0.821 → 0.817) while within-round nearest-neighbour similarity rose from 0.223 to 0.464 ([RST](https://arxiv.org/abs/2608.05466)). R-Diverse names two forms of "diversity illusion": *local* (diversity enforced within a batch while the generator cycles across iterations) and *surface* (different wording, same skill) ([R-Diverse](https://arxiv.org/abs/2602.13103)). And adding generator brands buys less than expected: cross-model output similarity is 0.71–0.82 ([Artificial Hivemind](https://arxiv.org/abs/2510.22954)). **Strong.**

| Level | Metric | Evidence |
|---|---|---|
| Surface | BLEU-cluster repetition penalty (τ = 0.5); MinHash/semhash near-duplicates | [R-Zero](https://arxiv.org/abs/2508.05004); §10 |
| Embedding novelty vs the whole archive | Cosine distance to the nearest pool item; frozen-embedding novelty with an admission gate (τ = 0.80); Vendi score | Diversity reward roughly doubled concept coverage ([OpenSIR](https://arxiv.org/abs/2511.00602)); [EvoEnv](https://arxiv.org/abs/2605.14392); QbQ variants Vendi 1.59 vs 1.25 ([QbQ](https://arxiv.org/abs/2608.01522)) |
| Coverage of a real reference | OT-based fidelity and diversity against real hard items | Diversity "directly correlates with downstream accuracy"; few seeds × many rewrites is the deficit ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)) |
| Pool structure | Domain and operator-family entropy, operator coverage, per-lineage caps | [RST](https://arxiv.org/abs/2608.05466) keeps family entropy at 2.26–2.31 bits (max 2.32) with caps of ≤ 4 descendants per parent, ≤ 160 per category, ≤ 320 per family, ≤ 280 per cohort |
| Skill | Similarity of canonical solver code; expert audit of algorithm categories | [R-Diverse](https://arxiv.org/abs/2602.13103); categories grew 19 → 30 and 95.6% of lineages added ≥ 1 new category ([BenchEvolver](https://arxiv.org/abs/2606.01286)) |
| Solution route | Rule-based route fingerprints of verified traces | Route-diverse SFT raised OLMo3-7B pass@8 by 16.9 points on held-out environments ([Route-diverse SFT](https://arxiv.org/abs/2609.33780)); **Emerging** |
| Generator signature | Synthetic-vs-real classifier; generator-ID classifier; NCD between generator families | Five-way generator ID 97.1%, > 90% after paraphrase ([Idiosyncrasies](https://arxiv.org/abs/2502.12150)); NCD diversity predicts less correlated failure ([Tieman & Markou](https://arxiv.org/abs/2609.03422)) |
| Environments | Count of distinct environments/toolsets | 10 → 80 environments gave gains more trajectories could not recover ([CUA-Gym](https://arxiv.org/abs/2605.25624)); diversity beat quantity with 4× less data ([DIVE](https://arxiv.org/abs/2603.11076)) |

Practical rules: measure diversity at the skill or solution level, not the wording; enforce quotas *at generation time* (lineage caps, inverse-frequency skill sampling p ∝ (count + 1)⁻¹ as in [SkillSynth](https://arxiv.org/abs/2604.25727), a mutation-family memory that demands a larger difficulty gain from families that already succeeded, as in BenchEvolver); cap rewrites per seed (e.g. ≤ 10; our reading of the Fidelity–Diversity result, **Proposal**) and maximize distinct seeds; mix 2–3 generator families from different series; and track the policy's pass@k and entropy, because Evol-Instruct-style rewriting pushed pass@8 below the untuned backbone (54.2 vs 55.1) while skill crossover with parametric mutation raised pass@8 by 7.4 ([EvoTD](https://arxiv.org/abs/2605.11666)), and answer-preserving variants kept policy entropy from collapsing ([SvS](https://arxiv.org/abs/2508.14029)). **Moderate.**

---

## 12. A QC checklist and acceptance-criteria template

The whole stack, in the order that minimizes wasted rollouts (cheap deterministic checks first, policy rollouts last):

```
seed ─► operator ─► candidate (statement, env/initial state, reference, verifier)
   [A] mechanical contract ........ executes, deterministic, scorer contract, golden=1 / initial=0     (§3, §5)
   [B] verifier QA ................ known-good pass, known-bad fail, forbidden patterns, hacker–fixer  (§5)
   [C] two-sided gate ............. shortcut probes fail  AND  privileged solver succeeds             (§6)
   [D] well-posed + unique ........ condition checks; alternative-answer sweep after every edit       (§7)
   [E] decontam + dedup + quotas .. output-side, three-level dedup, lineage/family caps              (§10, §11)
   [F] policy band (RL) ........... k rollouts of the current checkpoint; 0-pass → triage pool       (§7, ch. 05)
accepted ─► training ─► [G] alarms (reward jump, length collapse, oracle gap) ─► re-audit / quarantine (§9)
```

### 12.1 Checklist by stage

**Stage 0 — before generating at scale**
- [ ] Measure the existing verifier's TPR and TNR on known-good (references, equivalent rewrites) and known-bad (policy failures, mutants, near-misses, no-op, empty response, master keys) pools; fix false negatives before hardening anything (§4–5).
- [ ] Unit-test generators on edge cases (roots, revisits, ties, empty sets, duplicate needles) and check that the difficulty knob moves solve rate monotonically.
- [ ] Decide the required label tier per consumer (RL, SFT, eval, bridge) (§2).

**Stage 1 — every item**
- [ ] Mechanical contract: instantiable, deterministic across two runs, scorer contract, non-trivial variation (§3.2).
- [ ] Checker endpoints: checker(golden) = 1, checker(initial) = 0; no-op passes nothing; forbidden-pattern scan; checker written behind an information barrier.
- [ ] Contract validity: every checked requirement stated or discoverable; spec faithful to intent.
- [ ] Two-sided gate over k attempts (§6).
- [ ] Well-posedness and uniqueness, re-run after *every* obfuscating or constraint-adding edit (§7).
- [ ] Hacker–fixer pass and isomorphic twin where applicable (§5, §9).
- [ ] Output decontamination and three-level dedup (§10).
- [ ] Policy-relative band for RL (see [chapter 05](05-rl-playbook.md)); 0-pass items go to triage, not straight to training or deletion.

**Stage 2 — every batch or generator family**
- [ ] Hand-audit 100–200 items with Wilson intervals, prioritized by IRT-style flags (95% precision in the top 200 flags, [Land & Bikel](https://arxiv.org/abs/2605.30504)). Always read raw generations: a shared decoding-budget bug produced a 32-point "effect" that replicated from N = 50 to N = 500 and was caught only by reading outputs ([test-oracle study](https://arxiv.org/abs/2607.13707)).
- [ ] Partial-input baselines: question-only, answer-prior, choices-only. Hypothesis-only accuracy is about 67% on SNLI and 86–96% on LLM-elicited NLI ([Gururangan et al.](https://arxiv.org/abs/1803.02324); [Proebsting & Poliak](https://arxiv.org/abs/2410.08996)).
- [ ] Draw answer positions, magnitudes and entities with a code RNG and chi-square-test the marginals; Llama-3.1-8B-Instruct put the correct answer first in 56.8% of its generated MCQs (without option labels) vs 25% uniform ([Position bias](https://arxiv.org/abs/2605.01846)).
- [ ] Diversity report (§11) and per-operator yield (y_valid, y_band).
- [ ] Quarantine a whole family when its zero-pass rate or length profile is anomalous: 1–3 buggy task types out of 50 collapsed training in [UltraLogic](https://arxiv.org/abs/2601.03205).

**Stage 3 — during training:** alarms in §9.3; monitoring pool for 0-pass items; re-audit families whose reward jumps; gold audit slice by difficulty bucket.

**Stage 4 — after training:** random-reward control, a non-Qwen family, post-cutoff benchmarks, a held-out generator family and operator, and a procedurally generated control ([Reasoning or Memorization?](https://arxiv.org/abs/2507.10532)).

**Wilson 95% score intervals for hand audits:** 5/100 errors → [2.2%, 11.2%]; 2/100 → [0.6%, 7.0%]; 0/100 → < 3.7%; 0/200 → < 1.9%. To certify a label-error rate below 2%, audit about 200 items and find none.

### 12.2 Acceptance-criteria template

Thresholds tagged `[src]` come from the cited work; `[proposal]` values are our synthesis and should be tuned per family.

```yaml
family: <name>                       # one file per generator family
consumer: rl                         # rl | sft | eval | bridge
label_tier_min: T2                   # T1 construction/exec/kernel, T2 answer-preserving/inverse, T3 cross-family consensus + evidence
verifier_audit:                      # rerun after every verifier edit
  tpr_min: 0.90                      # [src] CodeContests+ HQ keeps TPR and TNR >= 0.9
  tnr_min: 0.90
  known_bad: [policy_failures, near_miss, gold_mutants, noop_agent, empty_response, master_keys]
  metamorphic_rewrites: true         # [src] catches checker false negatives
  hacker_fixer: until_no_new_exploit # [src] solver must still pass after each patch
per_item:
  mechanical: [instantiable, deterministic_x2, scorer_contract, varies_across_seeds]    # [src] EvoEnv, AZR
  checker_endpoints: {golden: 1.0, initial: 0.0}                                        # [src] CUA-Gym
  forbidden_patterns: [constant_flag, existence_only, assume_false, sorry, admit, new_axiom, external_body]
  information_barrier: true          # [src] checker author never sees solution scripts
  contract_validity: true            # [src] RST
  well_posedness: condition_checks   # [src] MathQ-Verify; formal: prove False from hypotheses
  uniqueness: sweep_after_each_edit  # [src] ASearcher, InfoSeek
  lower_gate: {probes: [closed_book, no_cot, no_data, noop, weak_solver], k: 8, max_success: 0}   # k=8 [src] Kimi k1.5
  upper_gate: {solvers: [gold_evidence, hint, fresh_sandbox_reference, other_family], k: 8, min_success: 1}  # [proposal] k
  decontamination: {ngram: 13, embed_top_k: 5, paraphrase_judge: true, against: all_eval_sets}   # [src] RST, DeepMath
  dedup: {embed_cosine_max: 0.85, max_per_template: 3, solution_signature: true}                 # [src] CUA-Gym, R-Diverse
  band_rl: {k: 8, keep: "1 <= successes <= 7"}   # [src] Qwen-CUA; alternatives in ch. 05
per_batch:
  hand_audit: {n: 200, max_errors: 0}            # [src] Wilson < 1.9%
  baselines: [question_only, answer_prior, choices_only, trivial_agents]
  signature: [synthetic_vs_real_clf, generator_id_clf]
  diversity: {rewrites_per_seed_max: 10, report: [ot_fidelity_diversity, skill_similarity, family_entropy]}
  yield_log: [y_valid, y_band, per_operator]
alarms: [reward_jump, length_collapse, turn_growth, train_vs_oracle_gap, trip_wire_rate, canary_rate, all_correct_share]
post_training: [random_reward, non_qwen_family, post_cutoff_sets, held_out_generator_family, procedural_control]
```

### 12.3 How QC differs for SFT and RL

| Check | RL reward pools | SFT / distillation pools |
|---|---|---|
| Reference correctness | T1–T2 required; T3 only after a gold audit | Answer filtering often adds nothing ([OpenThoughts](https://arxiv.org/abs/2506.04178)); spend on question selection and teacher choice ([chapter 06](06-sft-playbook.md)) |
| Verifier error | Both FPR and FNR matter; FNR draws gradient under GRPO | For rejection-sampled SFT/RFT, precision dominates ([Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)) |
| Well-posedness, contract validity | Required | Required; for stepping-stone data well-posedness mattered more than answer correctness ([SOAR teacher](https://arxiv.org/abs/2601.18778)) |
| Hacking guards | Required, with online alarms | Filter hacked trajectories, but filtering hacks and distilling did not remove misalignment ([MacDiarmid et al.](https://arxiv.org/abs/2511.18397)) |
| Difficulty | Policy-relative band, re-measured each stage | Difficulty and response-length selection of questions |
| Diversity | Required | Required, often the dominant factor: 64k tasks × 1 solution beat 16k × 4 ([X-Coder](https://arxiv.org/abs/2601.06953)); diversity tracks accuracy ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)) |

**Budget for rejection.** A simple worksheet: cost per accepted task C_acc = (c_gen + c_validate + c_env + k·c_roll) / (y_valid · y_band). Reported yields are low: 5.2% for Code-as-Task, about 12% of SCALECUA candidates, about 50% per round in RST at about $0.05 per accepted task. Quality inspection was about 78% of [ToolHazard](https://arxiv.org/abs/2608.11878)'s $0.59 per-environment cost. Verification is usually the largest line item, and it should be. **Moderate.**

---

## 13. Open problems

- **Labels without an oracle when all solvers are correlated.** No accepted confidence estimate exists for consensus labels; the rate at which 15 of 20 agreeing programs share an unintended rule is unmeasured.
- **A cheap, calibrated "hard vs ill-posed vs mislabeled" classifier.** MathQ-Verify reaches about 63% recall at 89.6% precision; self-review misses semantic bugs a stronger model catches.
- **Accepting all valid solutions in stateful environments.** checker(golden) = 1 and checker(initial) = 0 show direction, not coverage of alternative paths or rejection of destructive sequences that recreate the same end state ([CUA-Gym](https://arxiv.org/abs/2605.25624) says so).
- **A standard red-team harness for new verifiers**, combining metamorphic rewrites, master keys, optimized rubric attacks, impossible variants and canaries, with a pass bar tied to J = TPR − FPR.
- **Hackability under recursive escalation** and **online verifier patching** during RL without destabilizing training.
- **Spec-to-intent adequacy** for verified code: two-way spec⇔code equivalence proves agreement with the code, not with the user's intent ([VeriEquivBench](https://arxiv.org/abs/2510.06296)).
- **Skill-level diversity metrics that predict RL gains**, and decontamination of complexified *variants* after post-training spreads leakage.

Concrete projects built on these gaps are ranked in [chapter 10](10-idea-bank-and-roadmap.md).
