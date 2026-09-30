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
| Kernel "speedups" | GPT-5.5's 1.43× geomean becomes 0.88× under hidden inputs and tighter tolerance | [KernelBench-Verified (2026)](https://arxiv.org/abs/2607.16241) |

Fixing the verifier is often the cheapest way to make a saturated item hard again: EvolveCoder's evolved tests cut pass@1 from 43.80 to 31.22 *on the same problems* ([EvolveCoder](https://arxiv.org/abs/2603.12698)). **Strong.**

**(b) Label and verifier errors grow with difficulty.** R-Zero's pseudo-label accuracy fell 79.0% → 69.0% → 63.0% as its Challenger produced harder questions ([R-Zero](https://arxiv.org/abs/2508.05004)). The unverified majority answer is wrong on 25.85% of MATH-500, 46.07% of AMC and 73.33% of AIME 2024 questions ([T³RL](https://arxiv.org/abs/2603.02203)). Shortcut rates under extensional verifiers rise from 40 at complexity levels 1–10 to 458 at levels 11–20 ([IPT](https://arxiv.org/abs/2604.15149)). Judge–human F1 of 0.86 overall drops on tasks humans rate "Very Difficult" ([ACD](https://arxiv.org/abs/2502.07577)). Verification spend should therefore rise with difficulty, not stay flat. **Strong.**

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

A synthetic task is a tuple (statement, environment/initial state, reference, verifier, metadata). The contract below lists the properties that must hold, the failure each one prevents, and the cheapest check with evidence behind it. Use it as the specification that your generator and QC stack jointly enforce.

| # | Property | Typical failure when missing | Cheapest reliable check |
|---|---|---|---|
| C1 | **Well-posed**: premises consistent and sufficient | Contradictions were the largest flaw class in ValiMath (13.97% of all questions; 300 of 848 flawed items); vacuous hypotheses make formal statements trivially true | Condition-level checks ([MathQ-Verify](https://arxiv.org/abs/2505.13903)); prove `False` from hypotheses ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333)) |
| C2 | **Reference correct** | LLM-written outputs labeled 12.7% of test inputs correctly vs 96.8% for mutual verification ([rStar-Coder](https://arxiv.org/abs/2505.21297)) | Derive the reference from execution or a certificate (§3) |
| C3 | **Unique answer, or a checker that accepts every valid answer** | Obfuscated search questions with several valid answers ([WebSailor](https://arxiv.org/abs/2507.02592) admits this); alternative optima in LPs; 25.4% of contest problems need special judges ([HardTests](https://arxiv.org/abs/2505.24098)) | Enumerate alternatives (§7); grade certificates/objective values, not one argmax ([A²utoLPBench](https://arxiv.org/abs/2607.02141)) |
| C4 | **Solvable** with the information and tools provided | 48.6% of tasks on raw LLM-built websites admitted an executable trace (94.8% after verify-and-repair) ([Verified Synthetic Web Envs](https://arxiv.org/abs/2608.21898)); River Crossing instances with N ≥ 6 and boat capacity 3 are unsolvable yet were scored as failures ([Illusion of Thinking comment](https://arxiv.org/abs/2506.06941)) | ≥1 verified solve by a reference, hinted or oracle-evidence solver (§6) |
| C5 | **Contract valid**: everything the verifier checks is stated or discoverable | Hidden requirements in private tests; weakly grounded terminal tasks fell from 32.8% to 1.2% only after an instruction audit ([RST](https://arxiv.org/abs/2608.05466)) | Instruction–verifier alignment audit; T1 gives it the largest single weight (20%) ([T1](https://arxiv.org/abs/2609.11042)) |
| C6 | **Faithful to intent** (text ↔ program ↔ formal statement) | Compile-passing Lean statements fell from 92.4% to 32.7% after semantic checks ([FormalMATH](https://arxiv.org/abs/2505.02735)); 14 of 43 IMO geometry formalizations refuted by counterexample ([MechGeo](https://arxiv.org/abs/2608.02295)) | Back-translation + multi-LLM consensus; counterexample search; statement-only solver must pass ≥50% of tests ([BenchEvolver](https://arxiv.org/abs/2606.01286)) |
| C7 | **Non-trivial**: no shortcut solves it | Kimi k1.5 drops prompts a no-CoT guess gets right within 8 tries ([Kimi k1.5](https://arxiv.org/abs/2501.12599)); withholding data files cost only 40.5% of performance on QRData ([DSGym](https://arxiv.org/abs/2601.16344)) | Closed-book / no-CoT / no-data / no-tool / no-op probes over k attempts (§6) |
| C8 | **Unhackable**: the verifier cannot be satisfied without doing the task | 16% of 1,968 terminal-benchmark tasks hackable from the description alone ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)) | Forbidden-pattern scans, trivial-agent baselines, hacker–fixer pass (§5, §9) |
| C9 | **Deterministic and reproducible** | Performance references replay validly on every machine for only 39/102 GSO and 11/140 SWE-Perf tasks ([replay audit](https://arxiv.org/abs/2607.01211)) | Run twice and compare ([AZR](https://arxiv.org/abs/2505.03335) uses j = 2); fixed seeds; simulator timing ([PIE](https://arxiv.org/abs/2302.07867)) |
| C10 | **Uncontaminated and non-duplicate** | DeepMath's raw pool contained 90% of AIME24/AMC23 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)) | Output-side n-gram + embedding + LLM paraphrase judge (§10) |

Two refinements matter in practice.

**The contract depends on how the item is consumed.** AlphaProof trains on about 80M auto-formalized statements "regardless of fidelity", because every well-typed statement is a valid prove-or-disprove task ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y)); C6 is irrelevant there. When the reward compares against an informal answer, C6 is on the critical path. Likewise, stepping-stone data can relax C2: only 32.8% of the useful stepping stones in Sundaram et al.'s teacher were fully correct, yet well-posedness alone produced a 4× pass@1 gain ([SOAR teacher, 2026](https://arxiv.org/abs/2601.18778)). That licence applies to bridge or curriculum items only, never to final RL targets. **Moderate.**

**Grade labels by provenance tier and require higher tiers for RL.** A practical tiering (synthesized from the math-synthesis and verification notes):

| Tier | Label source | Examples | Use |
|---|---|---|---|
| T1 | Construction or execution; formal kernel | Planted KKT certificate, DDAR traceback, Prolog answer set, Lean proof, execution of a frozen oracle | RL reward, SFT, eval |
| T2 | Answer-preserving transform or inverse check | SvS/MathForge variants, SymPy differentiation of an antiderivative ([VHG](https://arxiv.org/abs/2605.06660)) | RL reward, SFT |
| T3 | Independent consensus with external evidence | Cross-family agreement, output-vector agreement over ≥50 inputs, tool-verified vote | RL after audit; SFT |
| T4 | Single model or same-family majority; LLM judge without anchors | Majority vote, unanchored VLM judge | SFT (if at all), bridge data; never sole RL reward on hard items |

---

## 3. Correctness by construction versus post-hoc verification

The most reliable pipelines never ask a model whether a hard item is correct; they arrange for the answer to be a consequence of something executed. The controlled evidence: BenchEvolver's solution-first variant reached 97.7% post-hoc validity (GPT-5.4-mini evolver) against 79.3% for a problem-first ablation and 86.2% without mutation memory ([BenchEvolver](https://arxiv.org/abs/2606.01286)). In the dynamic-evaluation literature, CHASE's free-form Evol-Instruct baseline produced 34 errors in 100 hardened math problems while frontier models still solved 82.5–88.9% of them ([CHASE](https://arxiv.org/abs/2502.14678)). **Strong.**

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
- **Shared misconceptions** when one agent writes both reference and verifier (DeepSeek-V3.2's synthesis agent; Envs-FORGE; SETA). CUA-Gym saw the reward "re-check the construction procedure instead of measuring task completion", e.g. `chart_verified = True` ([CUA-Gym](https://arxiv.org/abs/2605.25624)).
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

