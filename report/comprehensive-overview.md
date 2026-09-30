# Synthesizing Hard Training Tasks from Easy Ones: Comprehensive Overview

*For LLM SFT and RL, current to 2026-09-30. This one document sums up everything in the project workspace: the 11-chapter report, the 23 research-notes files and the verification record.*

## About this document

**The question.** Our SFT/RL tasks are too easy: pass rates are near 100% and GRPO gets no signal. How do we synthetically generate complicated tasks from easy ones, with correct answers and verifiers, at scale?

**What is condensed here.** Each part below ends with a link to its full chapter.

| Source | Size | Where |
|---|---|---|
| Report, chapters 00–10 | about 127,000 words | Google Docs (links in each part and in the inventory at the end) and GitHub `report/` |
| Research notes, 23 files | about 382,000 words; 817 verified method entries | GitHub `research/notes/` |
| Bibliography | 1,134 unique works | GitHub `report/11-bibliography.md` |
| Verification tools | 5 scripts | GitHub `tools/` |

**How to read it.**
1. Start with *The answer in one page*.
2. Then read the operator taxonomy (the core of the report) and the idea bank.
3. Go to the full chapter for any detail.

The research digest near the end summarizes each of the 23 subtopics with its key works.

**Evidence tags.** Every number holds only under the model, benchmark and metric its source reports. Much of the 2026 evidence rests on single preprints, and many RLVR results use Qwen2.5-Math bases, so validate on your own models before scaling anything.

| Tag | Meaning |
|---|---|
| **Strong** | Several independent works or large ablations |
| **Moderate** | One careful study |
| **Emerging** | One recent or unreplicated result |
| **Proposal** | Our own synthesis, not yet tested |


## The answer in one page

*As of 2026-09-30, for a team whose tasks are saturated (pass rates near 100%, no GRPO signal).* Evidence tags: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (one recent or unreplicated result), **Proposal** (our untested synthesis). Numbers are the cited papers' own and hold only under their model, benchmark and metric.

**How do we synthetically generate complicated tasks from easy ones?** First confirm that the tasks really are easy. Lenient verifiers, contamination, guessable formats and harness limits often fake saturation, so read the per-prompt pass-rate histogram under the exact policy, harness and verifier.

Then transform *verified artifacts, not prose*. Lift each seed into an executable representation that carries its checker (a program, computation graph, environment state, constraint list or formal statement). Apply operators that recompute, inherit or construct the label: compose mastered atoms that share real data dependencies, build the answer or certificate first, hide information behind fuzzed descriptions or tools, invert the direction, turn parametric knobs, or break working artifacts. Render the text last.

Admit a candidate only if it passes two gates:
- **Two-sided validity.** Solvable with privileged information but not without it; the oracle passes and a no-op fails.
- **Pass-rate band.** 0 < p < 1 on the *current* policy, centred near 0.3–0.6, re-measured as the policy moves.

Run this as a factory: complexify above the band, train inside it, scaffold below it, and log yield and cost per operator. Use SFT to install atoms, formats and behaviours, and RL to learn their compositions. Adopt a data policy only after paired multi-seed wins on held-out families and a non-Qwen model.

### If you only do five things

1. **Diagnose before you synthesize.** Audit the verifier (TPR/TNR on known-good and known-bad solutions) and read the pass-rate histogram in the exact training harness. TACO's tests had a false-positive rate above 90% on difficult problems ([HardTests](https://arxiv.org/abs/2505.24098)); stronger tests cut the SWE-bench Verified top score from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)). **Strong.**
2. **Harvest the signal you already have.** Filter zero-variance groups, reallocate rollouts, recycle deferred items, keep 1–2% review. Reallocation alone bought about 2× compute-equivalent ([Knapsack RL](https://arxiv.org/abs/2509.25849)); late in training Qwen2.5-Math-7B on DAPO-Math-17K (N = 8), about 40% of prompts were all-correct and 20% all-wrong. **Moderate.**
3. **Re-harden existing seeds with label-preserving operators that reuse your checker:** test hardening, shortcut and no-CoT filters, answer-preserving rewrites, oracle-preserving tool perturbations, hint stripping, failure-prefix and verdict meta-tasks. Cheapest and fail-safe (a corrupted answer-preserving item yields an inert all-zero group). **Strong/Moderate.**
4. **Build new difficulty construction-first:** answer or certificate first, composition of mastered atoms, a parametric knob with a per-family online controller, many families mixed equally; train compositions with RL, not SFT. RL on depth-2 compositions reached 27% at unseen depth 3 while RFT stayed at ≤ 2.6% ([f(g(x))](https://arxiv.org/abs/2509.25123)); on a saturated 1.5B model, 400 adaptive environments gave +3.37 vs +0.49 from over 3× more compute on the original data ([RLVE](https://arxiv.org/abs/2511.07317)). **Strong.**
5. **Gate both sides, and decide with statistics.** Oracle passes, no-op fails, known-bad fails; shortcut probes; a pilot band on the current policy. Adopt a policy only if a paired 95% CI excludes zero across 8–12 seeds and the gain holds on post-cutoff items, held-out families and a non-Qwen model. IFDecorator ended with 10,772 unsolvable prompts vs 7,324 usable ([IFDecorator](https://arxiv.org/abs/2508.04632)); random rewards gave Qwen2.5-Math-7B +21.4 on MATH-500 ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). **Strong.**

Full chapter: [00 · Executive summary](https://docs.google.com/document/d/1ezY8l_ub8xwLBCSQoatVAVLIYN1iBBDu-I_Uw2TKzPM/edit)

## Diagnosis: is it really too easy?

### Where the GRPO signal comes from

In group RL (GRPO, DAPO, RLOO) a group whose N rollouts all get the same reward has zero advantage and no gradient. For a prompt with true success rate p:

```
P(non-zero gradient) = 1 − pᴺ − (1 − p)ᴺ        (Knapsack RL)
```

| p | N = 8 | N = 16 | Rollouts for a 90% chance to see the minority outcome |
|---|---|---|---|
| 0.01 or 0.99 | 0.08 | 0.15 | 229 |
| 0.05 or 0.95 | 0.34 | 0.56 | 45 |
| 0.25 or 0.75 | 0.90 | 0.99 | 8 |
| 0.5 | 0.99 | 1.00 | 4 |

*Derived from the formula.* Signal also peaks mid-band: the KL-regularized bound p(1 − p)/(2β²) is maximal at ½ ([Bae et al.](https://arxiv.org/abs/2504.03380)), and the information-gain proxy p(1 − p)² peaks at 1/3 ([Knapsack RL](https://arxiv.org/abs/2509.25849)). A static pool with a low ceiling drives the effective-prompt ratio to zero ([RLVE](https://arxiv.org/abs/2511.07317)). For SFT the signal is teacher tokens: "too easy" means a short, low-information teacher trace.

### Difficulty is policy-relative

Difficulty is p(x; π, V, B), the success rate of task x under policy π, verifier V and budget B (tokens, turns, tools). Change any of the four and p changes; human and LLM difficulty labels are only priors. A task is **useful-hard** (**Proposal**) when it is:
1. **Valid:** well-posed, uniquely answerable, verifier sound on known-good and known-bad outputs, solvability certified.
2. **Learnable now:** for RL, in band, or low pass@1 with pass@k > 0; for SFT, a teacher solves it with a long, structured trace.
3. **Capability-bearing:** not solvable by guessing, partial input, priors or tool-free lookup; hard because of composition, structure or missing information, not tedium, ambiguity, length or format.
4. **New:** decontaminated, not a reskin.

Naive complexification fails criterion 3 (64% of rejected mutations in a gated study were "too easy"; [Trading Human Curation](https://arxiv.org/abs/2606.03800)); aggressive hardening fails criterion 1 (lowering OpenSIR's solve-rate floor from 0.5 to 0.1 dropped validity from 70.8% to 42.3%; [OpenSIR](https://arxiv.org/abs/2511.00602)).

### Reported pass-rate bands

| Source | k | Rule |
|---|---|---|
| [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) | 8 | keep 2–5 of 8 correct |
| [DAPO](https://arxiv.org/abs/2503.14476) | G | drop groups at exactly 0 or 1; oversample |
| [Bae et al.](https://arxiv.org/abs/2504.03380) | 16 | band symmetric about 0.5; (0.3, 0.7) best |
| [Pilot-Commit](https://arxiv.org/abs/2605.26606) | 16 pilot | skip p̂ > 0.75, defer p̂ < 0.125, commit 48 more rollouts |
| [R-Zero](https://arxiv.org/abs/2508.05004) | 10 | challenger reward 1 − 2\|p̂ − 0.5\|; train on 3–7 of 10 agreeing |
| [SvS](https://arxiv.org/abs/2508.14029) | 8 | seeds at 12.5–50%; variants rewarded in [12.5%, 62.5%] |
| [EvoEnv](https://arxiv.org/abs/2605.14392) | 8 | admit 0 < p < 1, target 0.3 |
| [RLVE](https://arxiv.org/abs/2511.07317) | — | raise level when accuracy ≥ 0.9 at the top level |
| [ScaleRL](https://arxiv.org/abs/2510.13786) | — | permanently retire p ≥ 0.9 |

- **The centre is 0.3–0.6, leaning hard:** p(1 − p)² peaks at 1/3; UltraLogic's sweet spot was 40–60%. **Strong.**
- **The best band depends on model size** (UltraLogic: Qwen3-8B gained most from Easy data, Qwen3-14B from Medium); calibrate on the model you train. **Moderate.**
- **Risk concentrates at the low edge.** The hardest 10% (still mixed outcomes) gave gains of up to 47% vs 3–15% for easy subsets ([Pikus et al.](https://arxiv.org/abs/2508.14094)), but pass@8 = 0 items lowered averages by 5.75, 11.24 and 1.07 points across three model settings ([Cheng et al.](https://arxiv.org/abs/2605.28388)). Admit p = 0 only with a solvability certificate. **Strong.**
- **Retire the high edge, don't delete it:** about 21% of review records for mastered prompts had mixed outcomes; a 1% review budget avoided GRPO's mid-training plateau ([ReMind](https://arxiv.org/abs/2606.03087)). **Moderate.**
- **The band moves:** re-profile the whole original pool each stage. **Strong.**

### How to measure difficulty

| Method | Tied to your policy? | Best use |
|---|---|---|
| Empirical pass rate, k rollouts | Yes, stale after each checkpoint | Ground truth for bands; final admission |
| Bayesian / Kalman trackers over training history (≈ 0 extra cost) | Yes | Online selection of existing items |
| IRT over a model zoo | No (population) | Label audits, generator rewards, eval design |
| Learned probes and value models (one forward pass) | Partly | Pre-filtering generated candidates |
| Text features and LLM ratings | No | Triage only |
| Trajectory signatures, solver effort | Yes | Real difficulty vs shortcuts; early alarms |

- **Use the unbiased pass@k estimator** 1 − C(n−c, k)/C(n, k), not 1 − (1 − p̂)ᵏ ([Chen et al.](https://arxiv.org/abs/2107.03374)).
- **Small k is noisy:** at p = 0.5 the 95% half-width is about ±0.35 with 8 rollouts, ±0.25 with 16, ±0.17 with 32 (derived). "2–5 of 8" is a coarse filter, not a measurement.
- **p̂ = 0 from small k means "unknown":** greedy decoding plus activation perturbations recovered 10–29% of a pass@6 = 0 stratum ([Zhou et al.](https://arxiv.org/abs/2606.19636)). Defer and re-pilot; don't delete.
- **Validate every gain** against a random-reward arm (+21.4 MATH-500 on Qwen2.5-Math-7B from random rewards, failing on Llama3 and OLMo2; [Shao et al.](https://arxiv.org/abs/2506.10947)), fresh items and a second model family; seed SD is 5–15 points on AIME/AMC-sized sets.

### Symptom → remedy

Run the checks in order: verifier sound (TPR, TNR ≥ 0.9) → items clean → no shortcuts → harness OK → behaviours present → atoms reliable → only then complexify inside a calibrated band. Most "too easy" pools fail an early check and never need a generator.

| Symptom | Likely cause | Check | Remedy |
|---|---|---|---|
| Most groups all-correct; effective-prompt ratio falling | Saturated pool | Per-family p̂ histogram | Reallocate rollouts; failure-prefix and meta-tasks; then compose and add families |
| Known-bad or near-miss candidates pass | Lenient verifier | TPR/TNR audit | Mutant, hacking-input and near-miss tests; hidden or read-only tests |
| Valid items stuck at 0; a stronger model's answers also rejected | Verifier false negatives (rule-based math checkers average 86% recall) | Equivalent-rewrite tests | Rule-first cascade with a discriminative second stage |
| Gains vanish on fresh, renamed or post-cutoff items, or another family | Contamination or spurious reward | Partial-prompt completion; random-reward arm | Decontaminate outputs; procedural held-out families |
| Right answers without valid reasoning | Format leakage | No-CoT guess within 8; question-only and choices-only baselines | MCQ → open-ended; code-RNG answer marginals |
| Mass only at 0 and 1 | Generator steps too coarse, or a broken sub-family | Per-operator histograms | Intermediate levels; audit the 0-spike |
| pass@1 up, large-k pass@k flat; entropy collapsing | Sharpening on saturated data | Paired pass@k vs base | Edge data (low pass@1, pass@k > 0); composed tasks |
| Rollouts never verify or backtrack | Missing behaviours | Behaviour counts on the hard tail | Priming SFT (wrong answers fine); mid-training |
| Composites near 0 though atoms look "solved" | Atoms too unreliable for the product law | Per-step accuracy | Sharpen atoms, then compose at depth 2 with a staged horizon |
| Failures hit turn or token caps | Budget, not capability (harness alone took a 4B SWE agent from 8.3% to 37.2%; [FrogNano](https://arxiv.org/abs/2609.07925)) | Cap-hit share | Raise budget with tier; compact answer formats |

Full chapter: [01 · Diagnosis](https://docs.google.com/document/d/1tpn_UscT66yYWQc3_pCVwOLC2gLqIhGlypmdkMWendE/edit)

## The complexification operator taxonomy

An operator maps a verified seed (x, y*, V) to (x′, y′*, V′) that the policy solves less often *for a learnable reason*. Robust operators share one pipeline: lift the seed into an executable IR, operate on the IR with a knob, recompute, inherit or construct the label and re-validate the verifier, render text last, then run controls (uniqueness, solvability, shortcut, no-op) and a policy pass-rate gate. Every method with near-zero label noise at scale computes the label from code or a solver; with the same evolver, solution-first evolution gave 97.7% validity vs 79.3% for problem-first ([BenchEvolver](https://arxiv.org/abs/2606.01286)). **Strong.**

### How the label survives

| Tag | Label mode | Risk under RL |
|---|---|---|
| **INV** | Invariant by construction (distractors, obfuscation, scaffold removal, sharding) | Low: a corrupted item is an inert all-zero group; main risk ambiguity |
| **REC** | Recomputed by re-executing the IR (chaining over code, scaling, lifting) | Low if the executor is right; risk: rendering drift |
| **CON** | Constructed, answer or certificate first (planted solutions, bug injection) | Low; grade the certificate, not the planted value |
| **CERT** | Any answer passing a checker (optimization, proofs, test writing) | Low if the checker is sound; exploits move into the task definition |
| **RED** | Re-derived by teacher, vote or judge (LLM fusion) | High: R-Zero's labels fell from 79% to 63% ([R-Zero](https://arxiv.org/abs/2508.05004)) |

**Use INV, REC, CON and CERT for RL; keep RED for SFT or behind an independent verifier.**

**Match the operator to the missing kind of difficulty:** volume (size, horizon, context) mostly adds length; search (solver effort, coupling, optimization) needs certificate grading; method change (hard perturbation, insight hiding, counterfactual rules, inversion) is hardest to generate; information acquisition (tools, users, sharding, clue removal) hides the largest gaps, 0.962 written out vs 0.204 agentic ([VHD-Play](https://arxiv.org/abs/2609.27321)); robustness needs certified solvability after injection. The axes do not substitute: depth extrapolates only about 3× ([ScaleLogic](https://arxiv.org/abs/2605.06638)). **Moderate.**

### A · Compose (Strong, under RL, atoms first)

RL on Level-2 string-function compositions took unseen Level 3 to about 30%; RFT on the same data stayed at ≤ 2.6%, and RL on atoms alone gave < 1% gain ([f(g(x))](https://arxiv.org/abs/2509.25123)).

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| A1 Sequential chaining | "12 muffins/h for 5 h?" → "…4 per box, $8 per box, 15% off: what is paid?" (illustrative, after [Compositional GSM](https://arxiv.org/abs/2410.01748)) | REC: feed the verified upstream value into downstream code; deterministic adapters | Staged horizon: AIME24 5.10 → 10.52, 3B ([h1](https://arxiv.org/abs/2510.07312)) |
| A2 Width composition | "cancel order X" + "update the address on order Y" → one conversation ([APIGen-MT](https://arxiv.org/abs/2504.03601)) | REC/CON: replay joint gold actions from a fresh state | Independent parts mostly multiply failure (**Proposal**) |
| A3 Nested substitution | "When was Michael P. Hein born?" → "…the first County Executive of Ulster County…" ([ASearcher](https://arxiv.org/abs/2508.07976)) | INV: sub-problem must resolve to the replaced constant; expand leaves only | MathForge: 97–99% answer equivalence |
| A4 Conditional branching | "Describe this product" → "if price > $100, JSON review; else a ≤ 280-character post" ([ComplexBench](https://arxiv.org/abs/2407.03978)) | REC: condition computable from input; score the active branch | GPT-4: And 0.881 vs Selection+Chain 0.626. **Moderate** |
| A5 Skill/concept mixing | "divisors of 360?" → a triangle built from that count; smallest k making its area a perfect square ([MATH²](https://arxiv.org/abs/2407.21009)) | REC via executable function graphs; else RED + cross-family agreement | Success ≈ p^k |
| A6 Statement fusion | (x+y)/2 ≥ √(xy) → ((a+b)/2)·((c+d)/2) ≥ √(ab)·√(cd) ([Ineq-Comp](https://arxiv.org/abs/2505.12680)) | CON if rule-based; LLM fusion is RED | DeepSeek-Prover-V2-7B pass@32 66.2% → 47.0% |
| A7 Multi-fault union | one flipped comparison → 2–3 interacting bugs, no issue ([SWE-smith](https://arxiv.org/abs/2504.21798)) | CON: combined F2P = union of each bug's set | 96.9% yield at 0¢ |
| A8 Long-horizon chaining | isolated milestone (> 80%) → continuous chain (≤ 38.03%) ([SWE-Milestone](https://arxiv.org/abs/2603.13428)) | REC: per-phase end state and checker; reward the AND | Unrelated chains made the weakest agent, 21.6 vs 38.2 ([AutoPlay](https://arxiv.org/abs/2509.25047)); require data dependencies. **Strong** |

### B · Constrain (Strong; count is not difficulty)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| B1 Constraint stacking | "Write a poem about autumn" → "4 stanzas, all lowercase, 'ember' exactly twice, no commas…" ([IFBench](https://arxiv.org/abs/2507.02833)) | CERT: code checker per constraint + compatibility check; pick k so p^k is in band | Per-constraint 72.0% × 0.922^(k−1) ([CSE](https://arxiv.org/abs/2608.12426)). **Strong** |
| B2 Nested level ladder | "Recommend three sci-fi novels" → L5: before 1970, Markdown table, no Asimov ([FollowBench](https://arxiv.org/abs/2310.20410)) | CERT: V(Lₖ) = V(Lₖ₋₁) ∧ v(cₖ) | Supplies intermediate rungs |
| B3 Coupled/scoped | "Include 'blue'" → "valid JSON; 'summary' 40–50 words; key initials spell MAP" ([ScopeIF](https://arxiv.org/abs/2609.32189)) | CERT: witness proves satisfiability; per-scope checks | CSE: structural degrades 2.0× faster than lexical |
| B4 Intersection | "Which club did X join in 2005?" → player fixed only by several blurred constraints ([InfoSeek](https://arxiv.org/abs/2509.00375)) | INV/CON: exactly one match; no proper subset suffices | AutoLogi: ~30% of LLM-added constraints break solvability |
| B5 Optimality/performance | any TSP tour → shortest; correct kernel → > 5% faster than eager and compile ([CUDA Agent](https://arxiv.org/abs/2602.24286)) | CERT: feasibility gate, then ratio to a pre-solved optimum | **Moderate**; most-hacked rewards |
| B6 Resource budgets | sequential lookup → wide search under a fixed step budget ([Kimi K2.5](https://arxiv.org/abs/2602.02276)) | INV: verifier unchanged | Tune both ways. **Moderate** |
| B7 Answer-space hardening | MCQ → canonical integer ([Big-Math](https://arxiv.org/abs/2502.17387)); SAT label → assignment or core | INV if canonical; CERT for certificates | **Strong**; open-ended zeroed > 83% of Golden Goose items |
| B8 Verifier/rubric hardening | ~10 tests → ~35 evolved tests: pass@1 43.80 → 31.22 ([EvolveCoder](https://arxiv.org/abs/2603.12698)) | CERT: new tests pass all correct references; TPR, TNR ≥ 0.9 | **Strong**: do it before synthesizing |

### C · Conceal and obfuscate (Strong; trades difficulty for ambiguity)

Every lab that obfuscates also checks answer uniqueness.

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| C1 Fuzzing | IMF → "an international financial institution"; 17,921 → "a five-digit prime whose digits sum to 20" ([FORT](https://arxiv.org/abs/2606.12087)) | INV + uniqueness check after every fuzz | Strong-agent accuracy 29.0 → 81.6 once fuzzing is also removed |
| C2 Clue removal | drop "died at the age of 44" ([WebExplorer](https://arxiv.org/abs/2509.06501)) | INV: uniqueness and solvability per deletion | Claude-4-Sonnet 86.6% → 67.1% in 5 rounds. **Strong** |
| C3 Insight hiding | "Show 1 + … + 5 is divisible by 5" → "how many n ≤ 1000 satisfy n \| 1^k + … + n^k for every odd k?" ([Code2Math](https://arxiv.org/abs/2603.03202)) | INV for answer-hidden stems; CON formal; RED free-form | **Moderate**; least automatable |
| C4 Implicit parameters | "Use 5 bullets" → "as many bullets as primes below the letter count of Australia's capital" ([ImpRIF](https://arxiv.org/abs/2602.21228)) | REC: execute the dependency DAG | +7–10 IF points. **Moderate** |
| C5 Withhold behind tools/users | written-out knapsack → agentic version, parameters only via tools ([VHD-Play](https://arxiv.org/abs/2609.27321)) | INV: pre-solve; fix simulator knowledge | 0.962 → 0.204. **Strong**: biggest agent lever |
| C6 Spec degradation | issue naming the fix → "export sometimes drops the last row" ([R2E-Gym](https://arxiv.org/abs/2504.07164)) | INV: tests unchanged; solvable from spec alone | RST contract gate: weak tasks 32.8% → 1.2% |
| C7 Retrieval obstruction | "Yuki has been to Dresden" → "Yuki lives next to the Semper Opera House" ([NoLiMa](https://arxiv.org/abs/2502.05167)) | INV: verified relation; re-check reachability | Llama 3.3 70B 98.5 → 25.9 at two hops. **Strong** |
| C8 Renaming/substitution | a + b + c ≥ 3∛(abc) → x/y + y/z + z/x ≥ 3; opaque names like func_16 | INV/REC: seed proof ∘ substitution | Same prover 66.2% → 42.1%. **Moderate** |

### D · Invert (Strong for answer-first; Moderate for inversion)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| D1 Given ↔ unknown | "trailing zeros of 100!" → "smallest n whose factorial ends in exactly 24 zeros" ([QbQ](https://arxiv.org/abs/2608.01522)) | CON; check uniqueness (100–104 all qualify) | +2.3/+2.6 GSM8K vs +0.4 rephrasing ([MetaMath](https://arxiv.org/abs/2309.12284)). **Moderate** |
| D2 Program inversion | predict f(x) → find an input → write f from partial I/O ([AZR](https://arxiv.org/abs/2505.03335)) | CERT: execute; hidden I/O pairs | Sharpens more than expands |
| D3 Answer-first/planted | solved grid → masked puzzle; LP around a KKT optimum, 100% → 8.3% ([A²utoLPBench](https://arxiv.org/abs/2607.02141)) | CON graded as CERT (check constraints) | **Strong** for every RL family |
| D4 Explore, then describe | "open settings" → discovered multi-step goal ([OS-Genesis](https://arxiv.org/abs/2412.19723)) | CON: recorded end state; re-execute | **Moderate**; explorer-biased |
| D5 Checker writing | "fix given the failing test" → "write a test that fails before the patch" ([MiniMax-M2](https://arxiv.org/abs/2605.26494)) | CERT: run pre- and post-patch | — |
| D6 Prove-or-disprove | "n² + n is even" → "Decide: n² + n + 41 prime for all n ≥ 0" ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333)) | CERT: kernel both ways; vacuity check | AlphaProof: ~80M statements |

### E · Scale (Strong with a controller and a length audit)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| E1 Instance size | 4×4 → 9×9 Sudoku; TSP 10–20 → 45–55 cities ([NP-Engine](https://arxiv.org/abs/2510.16476)) | REC: programmatic checker | Size alone adds length |
| E2 Depth/topology | 1-op story → 12-op DAG; depth +4: Claude-3-Opus ~95% → 41.0% ([DARG](https://arxiv.org/abs/2406.17271)) | REC from the graph | Depth extrapolates ~3× |
| E3 Horizon | GSM chains h = 1 → 5 ([h1](https://arxiv.org/abs/2510.07312)) | INV/REC | Only staged curricula help |
| E4 Context length | 150-token problem → premises in a 32–64K narrative ([LongReason](https://arxiv.org/abs/2501.15089)) | INV + no-context filter | LoongRL: 16K training handled 128K. **Strong** |
| E5 Entities/state | one-table DB → ~18 tables, 35 tools ([AWM](https://arxiv.org/abs/2602.10090)) | REC from a world model | Terminal-Universe cross-workspace: teacher pass@1 72.3% → 49.2% |
| E6 Solver effort | Zebra 0 → > 20 Z3 conflicts ([ZebraLogic](https://arxiv.org/abs/2502.01100)) | CERT | SATURN's metric: R² ≈ 0.71 vs pass rate |

Controllers matter as much as knobs: on IMO geometry (of 50), easy-only 29, hard-only 24, unscheduled 38, controller-scheduled 44 ([InternGeometry](https://arxiv.org/abs/2512.10534)). **Strong.**

### F · Perturb and distract (Moderate)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| F1 Irrelevant information | "…but five of them were a bit smaller than average" (still 190) ([GSM-Symbolic](https://arxiv.org/abs/2410.05229)) | INV: noise off the query path; re-solve | Up to 65% drops; significant in 8 of 20 models on re-analysis |
| F2 Hard distractors | random paragraphs → gold paragraphs of sibling questions ([MuSiQue](https://arxiv.org/abs/2108.00573)) | INV/CON: distractors provably fail | Quality beats quantity. **Moderate** |
| F3 Answer-changing edit | "minimize x² − 6x + 13 over reals" → "…over integers not divisible by 3" ([MATH-Perturb](https://arxiv.org/abs/2502.06453)) | REC via script, else RED; answer must differ | o1-mini −16.49% on MATH-P-Hard. **Moderate** |
| F4 Counterfactual rules | base-10 → base-9; knights lie ([K&K](https://arxiv.org/abs/2410.23123)) | REC: mutate program, execute | Method change without a judge |
| F5 Environment noise | perfect tools → 40% failing calls, silent corruption ([BENCH2ROBUST](https://arxiv.org/abs/2608.11977)) | INV: final vs uncorrupted state | 69 of 70 pairs degraded. **Moderate** |
| F6 Adversarial injection | "summarize my emails" + injected message ([AgentDojo](https://arxiv.org/abs/2406.13352)) | INV: reward completion AND resistance | Randomize injection position |
| F7 Infeasible twins | remove a key premise → "insufficient information" | CON: infeasible by construction | 10% mix restored refusal; cap it |

### G · Abstract and generalize (Strong)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| G1 Parametric lifting | Putnam problem with 2011 → 4680 ([Putnam-AXIOM](https://arxiv.org/abs/2508.08292)) | REC: two references agree across scales | EFAGen: GPT-4o-failing variants of solved Level-1 MATH seeds |
| G2 Generalize/specialize | "trailing zeros of 100!" → "…in base 12" (48) ([QbQ](https://arxiv.org/abs/2608.01522)) | REC/CERT (QbQ itself: RED); answer ≠ parent | AIME pass@1 5.6% → 16.5% (Qwen2.5-Math-7B) |
| G3 Demand the general object | "v at 3 s?" → "give v(t)"; Hanoi moves → program | REC per instance or CERT | Blocks hard-coding |

### H · Upgrade the task type (uneven: many ladders are evaluation-only)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| H1 Decision → optimization | SAT → MaxSAT → minimal unsatisfiable core ([SATQuest](https://arxiv.org/abs/2509.00930)) | CERT behind a feasibility gate | — |
| H2 Compute → prove/verify | APPS function → Dafny method with specs ([ATLAS](https://arxiv.org/abs/2512.10173)) | CERT; block `assume(false)`, `sorry` | ~9% of Vericoding successes: weak specs |
| H3 Closed → open-ended | USMLE MCQ → open diagnosis ([HuatuoGPT-o1](https://arxiv.org/abs/2412.18925)) | Uniqueness before removing options | Fails if most items hit zero |
| H4 Single → multi-turn | GSM8K → 5 shards over 5 turns ([Lost in Conversation](https://arxiv.org/abs/2505.06120)) | INV + CONCAT control | −39%; RL, not SFT, extrapolates. **Strong** |
| H5 Static → environment | code answer → shell task with pytest ([Nemotron-Terminal](https://arxiv.org/abs/2602.21193)) | Oracle passes, no-op fails | Synthetic web envs: 48.6% → 94.8% feasible after repair |
| H6 Beyond context | 1 question → 16 under fixed memory ([MEM1](https://arxiv.org/abs/2506.15841)) | INV per item | Qwen2.5-14B ~37% → ~3.5% EM. **Moderate** |
| H7 Multi-agent/game | TicTacToe → Kuhn Poker vs self ([SPIRAL](https://arxiv.org/abs/2506.24119)) | Rule-decided outcomes | Co-evolve opponents |

### I · Break and inject errors (Moderate)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| I1 Bug injection | flip `<` → LM rewrite, reverted PRs ([SWE-smith](https://arxiv.org/abs/2504.21798)) | CON: fail-to-pass; inverse mutation testing | +10.4 SWE-bench Verified ([SSR](https://arxiv.org/abs/2512.18552)) |
| I2 State corruption | healthy env → broken dependency, env var, permissions ([CLI-Gym](https://arxiv.org/abs/2602.10999)) | Original tests are the oracle | Block recovery via cached state |
| I3 Error localization | "solve" → "where is the first wrong step?" ([MR-GSM8K](https://arxiv.org/abs/2312.17080)) | CON: injected location is the label | Doesn't teach self-correction |

### J · Target the learner (Strong: seed from "almost solved")

| Op | Mechanism | Guardrail | Evidence |
|---|---|---|---|
| J1 Harden until it fails | Solution-first escalation until the solver fails ([RST](https://arxiv.org/abs/2608.05466)) | Reference or pass@N > 0 each round | DeepSeek-V4-Pro pass@4 90% → 2.5% in 15 rounds, ~$0.05/task |
| J2 Failure mining | Item solved 127/128 → restart from its one failure ([Failure-prefix](https://arxiv.org/abs/2601.20829)) | Repair non-model failures first | 44.5 vs 40.7 plain RLVR |
| J3 Variational rewrite | "divisors of 360?" → "360 m² garden: how many (length, width) pairs?" ([SvS](https://arxiv.org/abs/2508.14029)) | Same answer; reject 0/k, k/k | +18.3/+22.8 pass@32 AIME24/25. **Strong** |
| J4 Adversarial proposer | Rewarded for valid tasks the solver barely solves ([VHG](https://arxiv.org/abs/2605.06660)) | Independent gate; invalid → 0 reward | Valid rate 30.6% → 75.5% |
| J5 Directional rewrite | Harden above 0.5, shift below, simplify at 0 ([SETA](https://arxiv.org/abs/2607.10891)) | Accept iff α_low < acc(q′) < acc(q) | 60% of "increase" rewrites moved as declared |

### K · Remove scaffolding, and add it back (Strong)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| K1 Hint/harness removal | task naming file and format → abstract goal, no visible failing tests | INV: verifier unchanged | Harness alone: 4B agent 8.3% → 37.2% ([FrogNano](https://arxiv.org/abs/2609.07925)) |
| K2 Add scaffolds, then fade | p = 0 → format ladder, inverse rewrite, teacher candidates, expert prefix | Help withdrawn over training | 0% → 48.98% pass@64 ([AutoOR](https://arxiv.org/abs/2604.16804)) |

### L · Change representation (Moderate: mostly diversity)

| Op | Easy → hard | Label stays correct by | Evidence |
|---|---|---|---|
| L Representation | CNF → DIMACS → story; givens moved into the figure | INV; back-translate + solver check | +22.21 MathVerse Vision-Only ([GeoSym127K](https://arxiv.org/abs/2605.16371)) |

### Anti-operators (Strong)

They lower accuracy without adding learnable difficulty. Accept outputs only on a measured pass-rate drop on the current policy plus a validity control, never on an LLM's hardness rating (o3–human quality correlation 0.07; [AutoCode](https://arxiv.org/abs/2510.12803)).

| Anti-operator | Evidence | Detection |
|---|---|---|
| Paraphrase, surface rewrite | +0.4 GSM8K vs +2.3/+2.6 for inversion ([MetaMath](https://arxiv.org/abs/2309.12284)) | Answer or solution signature must differ; track pass@k |
| Tedium, padding | [Code2Math](https://arxiv.org/abs/2603.03202) rejects "computational tedium" | Method-change rubric; tool-access probe |
| Ambiguity injection | Removing a contract detail gave 0% from ambiguity ([FrogNano](https://arxiv.org/abs/2609.07925)) | Uniqueness enumeration; solver from spec alone; CONCAT |
| Contradiction | ~98% of random 12-constraint sets incompatible ([CSE](https://arxiv.org/abs/2608.12426)) | Witness or pass@N > 0; compatibility checker |
| Unrelated concatenation | Chain-then-summarise: weakest agent (21.6 vs 38.2) | Data dependency across every hop |
| Guessable answers | ~28% spurious guessing ([TRACE](https://arxiv.org/abs/2607.04784)) | No-CoT, no-tool, partial-input baselines |
| Verifier artifacts | 59.01% of correct special-judge solutions fail exact match ([ScaleBox](https://arxiv.org/abs/2604.27467)) | TPR/TNR audit before complexifying |
| LLM-regenerated "hard" items | Less challenging, rankings not preserved ([Gill et al.](https://arxiv.org/abs/2505.22830)) | Compare with real hard items on a probe model |

### Composing operators

**Label algebra.** INV ∘ INV = INV; REC ∘ {INV, REC} = REC; CON/CERT ∘ X = CON/CERT if the grader checks a certificate; RED ∘ anything = RED. Keep RED out of RL chains; verify only each step's increment, so cost stays linear in depth ([TaskCraft](https://arxiv.org/abs/2506.10055)). **Moderate.**

**Order of application (Proposal):**

```
0 LIFT       seed → executable IR + checker; pin gold               (G1, H5)
1 STRUCTURE  compose / scale / invert / upgrade / break on the IR   (A, E, D, H1–H2, I)
             → recompute label; oracle passes, no-op fails
2 CONSTRAIN  constraints + compatibility check + witness            (B1–B6)
3 TIGHTEN    answer-space and verifier hardening                    (B7, B8)
4 RENDER     IR → text / image / environment
5 CONCEAL    leaf-first; uniqueness check after EACH step           (C1–C8)
6 DISTRACT   noise computed against the FINAL IR, off-path          (F1, F2, F5)
7 DELIVER    tools, users, turns, memory, harness; drop scaffolds   (C5, H4–H7, K1)
8 GATE       CONCAT/oracle, shortcut probes, no-op → policy band
```

Concealing before composing exposes constants near the target; distractors added before scaling can land on the solution path.

**Budgets.**
- **Rounds:** free-form LLM evolution peaks at 2–3 rounds. RST sustained 15 because every child was solution-first and re-validated from scratch; yield held at 498–572 per 1,000 attempts, with hardness from work (solutions ×5.6), not prose (instructions ×1.4). **Strong.**
- **Knobs:** pick k so p^k ≈ 0.2–0.5 for conjunctions; predict chains by the product of link pass rates, then expect worse.
- **Yield:** log it per operator (7.7–45.5% per axis in [Trading Human Curation](https://arxiv.org/abs/2606.03800)); candidates per accepted task = 1/(y_valid·y_band).
- **Diversity:** ~750 seeds × 9 rewrites beat ~7 seeds × 999 at a fixed budget; cap children per parent.

**Stopping rules.** Stop on measured pass rate, not round count. Accept a step iff α_low < acc(q′) < acc(q) (23–40% of declared rewrites do not move). Stop concealing one step before uniqueness breaks. At p = 0, audit, then scaffold (K2) or regenerate; keep a 0-pass item only with a solvability certificate. Stop a lineage when yield or novelty drops, and self-play when gold-slice label accuracy or the valid-proposal rate falls.

### Selection guide: which operators to try first

**Step 0 for any seed type:** audit the verifier, measure the pass-rate histogram on the current policy, and reallocate rollouts and recycle zero-variance items first; reallocation alone has bought about 2× compute-equivalent. **Strong.** If the only verifier is a judge, send judge-scored hard tasks to SFT or evaluation and prefer INV operators. **Moderate.**

| Seed type (verifier) | Try first | Then |
|---|---|---|
| Word problems with code solutions | A1 chaining via code; E2 DAG depth/width; D1 inversion | F1 NoOp; F3 script perturbation; G1 lifting |
| Competition math, answer only | J3 same-answer rewrites; G2 with answer ≠ parent; B7 canonical answers | C3 insight hiding; A5 concept pairs; K2 for p = 0 |
| Formal statements | D6 prove-or-disprove; A6 rule-based fusion; C8 substitution | C3 auxiliary hiding; E6 proof depth |
| Algorithmic code with tests | B8 test hardening *first*; E1 input scale; A5 feature composition | D2 abduction/induction; B5 performance; H2 spec lifting |
| Repositories (SWE) | I1 removal + history reversion; A7 Combine; C6 symptom-only issues | A8 milestone chains; D5 test writing; K1 |
| Procedural puzzles, OR | E1/E6 with a controller; D3 planted certificates; H1 ladder | F4 counterfactual rules; C5; L formats |
| Tool/API agents with DB state | C4 hidden steps; A1/A2 with gold replay; E5 enrichment | F5 tool noise; C5 user-held info; F7 ≤ 10% |
| Web / deep search over a KB | A3 leaf substitution; B4 intersection; C1 fuzzing | C2 clue removal; C7 dispersion; cycles |
| GUI, terminal, spreadsheets | A8 chaining with data dependencies; C6 abstraction; E5 breadth | K1 start-state regression; F5; I2; J1 |
| Instruction following | B1 with p^k and wide ranges; A4 logic; B3 coupled/scoped | C4; H4 multi-turn carry-over; F6 |
| Long-context QA | C7 overlap removal; F2 hard distractors; A1 with bridge masking | H6; F7 sufficiency twins; E4 length *last* |
| Multimodal, chart, SQL, science | D3 execute-then-phrase; E2 from latent state; C3 answer-hidden stems | L givens → image; I3; F2 distractor tables |
| Open-ended, rubric only | B8 rubric hardening; H7 latent-variable games; A5 per-skill rubrics | Key-point rubrics; F7 probes; only INV operators for RL |

**When the failure is not "too easy":** brittle (high pass@1, flat pass@k) → J3, F3, F4; fails on interaction, not content → C5, H4, H6; fails on long tasks → shorten the horizon (K2, macro-actions), then E3 with a curriculum; fails on composition → install atoms, then A1/A5 under RL; SFT on composed data does not transfer.

Full chapter: [02 · Operator taxonomy](https://docs.google.com/document/d/1t5foEYg4oIvwpQmweZQbjzVE4kvaLCf6lH6oRozGDAo/edit)

## Generation architectures

Every working pipeline runs seed → operator → validity gate → difficulty gate (current policy) → dedup/decontamination → managed pool → trainer, with retirement, re-profiling and failure mining feeding back. The generator is usually the cheapest stage; validity checks, profiling rollouts and environment builds dominate. Track cost per accepted in-band task, **C_acc = (c_gen + c_validate + c_env + k·c_roll) / (y_valid × y_band)**, not tasks generated. **Moderate.** Two gates recur in every archetype: a two-sided validity gate (solvable with the evidence, tools or oracle; fails via no-op, no-data, closed-book or bridge-masked shortcuts) and a policy-relative difficulty gate.

### The nine archetypes compared

| Archetype | Correctness guarantee | Scalability | Cost / yield | Best fit | Exemplars |
|---|---|---|---|---|---|
| 1 Procedural generator + online controller (**Strong**) | Exact checker, if the generator is audited | Unbounded instances; limited by family count | High fixed engineering, ~0 marginal; LLM-written envs $0.01–0.03 | Algorithms, logic, SAT/CSP, OR, kernels | [RLVE](https://arxiv.org/abs/2511.07317), [SCALER](https://arxiv.org/abs/2601.04809) |
| 2 Correct-by-construction (**Strong**) | Exact up to engine soundness and rendering | Very high, CPU-bound (1M geometry problems in 28 h on 50 threads) | Low, plus back-translation | Any formal backbone: geometry, inequalities, SQL, code, OR, DB agents | [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5), [SQL-Zero](https://arxiv.org/abs/2609.04697) |
| 3 LLM rewrite + independent gate (gate required: **Strong**) | As strong as the gate | High; yields ~25–50% | ~$0.05 API per accepted variant, plus rejection | Labelled seeds without a generator | [SvS](https://arxiv.org/abs/2508.14029), [COVERT](https://arxiv.org/abs/2604.09813) |
| 4 Composition of verified atoms (**Strong**) | Exact if atoms verified, adapters deterministic | High, combinatorial | Near zero (h1) to $27.3 per task (SkillSynth) | Word problems, multi-hop QA, tool chains, terminal, Lean | [MuSiQue](https://arxiv.org/abs/2108.00573), TaskCraft |
| 5 Solver-in-the-loop (**Moderate**) | Oracle re-verified each round; shared-author blind spots | Medium: a probe per revision | $0.05 per task (RST) up to 50 probe rounds (CalibForge) | Terminal, SWE, tool/DB bundles, search QA | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), RST |
| 6 Trained generator (validity × difficulty: **Strong**) | Only as strong as the gate; pseudo-labels decay | High once trained; tracks the policy | Generator training + K solver rollouts per candidate | Domains with inverse checks, executors, grounding docs | [VHG](https://arxiv.org/abs/2605.06660), [SPICE](https://arxiv.org/abs/2510.24684) |
| 7 Failure mining (**Moderate**) | Inherits the base generator | Bounded by failure volume | Low marginal (logs exist) | All; nearly free in Lean, code, env state | [SwS](https://arxiv.org/abs/2506.08989), Goedel-Prover-V2 |
| 8a Executable environments (**Moderate**) | Exact on state once verified | Medium: build + sandbox | $0.04 per trajectory to $19.66 per env | Tool use, GUI, terminal, SWE, office | [AWM](https://arxiv.org/abs/2602.10090), [CUA-Gym](https://arxiv.org/abs/2605.25624) |
| 8b LLM-simulated environments | Weak: state-change cliff | Very high | ~1/3–1/5 of real-env RL | Users, surface perturbation, SFT breadth | [DreamGym](https://arxiv.org/abs/2511.03773) |
| 9 Quality-diversity archive (**Emerging** for LLM RL) | Inherits the gate | Medium | Archive + novelty checks | Long runs, coverage, red-teaming | [ACES](https://arxiv.org/abs/2310.10692), Rainbow Teaming |

**They stack rather than compete (Proposal):**

```
formal/executable core? ─yes─► 2 construct ─► 1 knob + controller ─► saturated? ─► 4 compose ─► 5 harden
      │ no
      ├── labelled seeds? ─► 3 answer-preserving rewrite + independent gate + policy band
      └── agentic task?   ─► 8a executable env (state in code/DB) ─► 4 / 5 inside it; 8b only to render
always on top: 7 decides WHERE · 9 keeps coverage · 6 once operators are exhausted AND a gate exists
```

Prose-only "make it harder" rewrites often fail to beat unmodified seeds at fixed size ([OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)); for agents, keep state in code or a DB and never let a trainable component grade.

### Reference design: the hardening factory (Proposal)

```
 seed = (prompt, env, oracle, verifier, known-bad, lineage, p̂)
   │ 0 intake + verifier audit: oracle passes, no-op fails, known-bad fails
   │ 1 profile on the CURRENT policy (pilot k = 8–16), route by p̂
   ├─ p̂ ≈ 1 ─► 2 choose operator (bandit on yield × Δp) ─► 3 generate (cheap, many)
   │             ─► 4 validity cascade, cheapest first: schema → oracle in fresh sandbox
   │                  → no-op / no-data / no-tool probes → uniqueness → hacker probe
   │             ─► 5 calibrate: skip p̂ > 0.75, defer p̂ < 0.125, else commit 48;
   │                  keep only if harder than the parent and still solved
   │             ─► 6 dedup on solution signatures; decontaminate outputs ─┐
   ├─ in band ─────────────────────────────────────────────────────────────┴► 7 pool ─► trainer
   └─ p̂ ≈ 0 ─► audit → scaffold or defer (never delete after one pilot)
 per-operator yield and reject reasons feed step 2; accepted children become seeds
```

- **0 Intake.** Package each seed with oracle and known-bad solutions; add special judges (14.57% of code problems need them; exact match rejects 59.01% of their correct solutions; [ScaleBox](https://arxiv.org/abs/2604.27467)).
- **1 Route** by pilot p̂ on the current checkpoint: complexify above the band, train inside, scaffold or defer below.
- **2 Choose the operator** by band direction (increase, context-shift, decrease) plus a bandit over y_valid × y_band × Δp; prior p̂ = parent's p̂ shifted by the operator's mean drop.
- **3 Generate** with the generator that minimizes C_acc on a pilot: 50K GPT-4o-mini samples beat 10K GPT-4o at 3.4× lower cost ([AgoraBench](https://arxiv.org/abs/2412.03679)).
- **4 Validity cascade**, cheapest first, abort on first failure, ending with a hacker probe and contract validity.
- **5 Calibrate** with pilot 16 and commit 48 ([Pilot-Commit](https://arxiv.org/abs/2605.26606)); accept only if harder than the parent and still solved.
- **6 Dedup** on solution signatures (canonical solver code, masked SQL template) and decontaminate outputs: DeepMath's raw pool held 90% of AIME24 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)).
- **7 Manage the pool:** ≤ 20% new per epoch, ≤ 4 children per parent, 20–33% real anchors + 2–10% easy replay; retire on slope ≤ 0; recycle deferred items; re-profile every stage.
- **Monitor** effective-prompt ratio, C_acc and yield per operator, large-k pass@k and entropy, gold-slice label accuracy, and the valid-proposal rate.
- **Budget** (derived): ~4 candidates per accepted task at 25% yield, ~19 at 5.2%; profiling was ~36% of rollout tokens in the worked example; size the unique pool at ~S·B/25, since reuse up to ~25× showed no significant degradation ([Tan et al.](https://arxiv.org/abs/2509.25300)). No major open RL framework regenerates tasks online, so run the factory as a service beside the trainer.

Full chapter: [03 · Generation architectures](https://docs.google.com/document/d/1WxWEIYAa41rQnNdvbJApXmWeeBTjtXXhDje5SHVbuPw/edit)


## Verification and quality control

Audit the verifier before complexifying. Many "too easy" or "impossibly hard" items are verifier artifacts, and label and verifier errors grow with difficulty. Hardened tests cut the top SWE-bench Verified agent from 78.80% to 62.20% ([SWE-ABS](https://arxiv.org/abs/2603.00520)). **Strong.**

### The correctness contract

Every hard task must be well-posed and reference-correct. It needs a unique answer, or a checker that accepts every valid answer. It must be solvable with the tools given and contract-valid: everything the verifier checks is stated or discoverable. It must also be faithful to intent, shortcut-free, unhackable, deterministic and uncontaminated. Faithfulness fails often: compile-passing Lean statements fell from 92.4% to 32.7% after semantic checks ([FormalMATH](https://arxiv.org/abs/2505.02735)). Grade labels by provenance (tiers are a **Proposal**):

- **T1:** construction, execution or a formal kernel. May drive RL reward.
- **T2:** answer-preserving transforms or inverse checks. May drive RL reward.
- **T3:** independent consensus with external evidence. Use for RL only after an audit.
- **T4:** one model, a same-family majority or an unanchored judge. Use for SFT or bridge data only.

### Construction versus post-hoc checking

Make the answer a consequence of something executed. Solution-first evolution reached 97.7% post-hoc validity, against 79.3% for problem-first generation with the same evolver ([BenchEvolver](https://arxiv.org/abs/2606.01286)). **Strong.** Construction patterns:

- planted certificates;
- deduction closure plus traceback;
- frozen executable environments;
- execute-first questions;
- answer-preserving variants;
- evaluator-first composition;
- state inversion.

Construction moves risk; it does not remove it. Of 79 environments that passed every mechanical check, GPT-5.4 judged 35 buggy ([EvoEnv](https://arxiv.org/abs/2605.14392)). When one agent wrote both solution and checker, the checker re-checked the construction instead of task completion ([CUA-Gym](https://arxiv.org/abs/2605.25624)). Run a mechanical contract first: the task instantiates, is seed-deterministic, the reference passes, near-misses fail, and the difficulty knob is monotone. Then have a different-family model review it.

### Verifier types and reported failure rates

| Verifier | Reported failure | Hardening |
|---|---|---|
| Rule-based math checkers | Recall 86% on average ([Rule vs model verifiers](https://arxiv.org/abs/2505.22203)) | Re-check rule negatives; a reasoning verifier went 82.7% → 99.3% ([Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)) |
| LLM answer judges | "Master key" false-positive rate up to 80% ([Master-RM](https://arxiv.org/abs/2507.08794)) | Truncated-response negatives; discriminative second stage |
| I/O unit tests | TACO false-positive rate > 90% on hard problems ([HardTests](https://arxiv.org/abs/2505.24098)) | Stress inputs, validators; keep TPR and TNR ≥ 0.9 |
| Repository tests | About 1 in 5 accepted SWE-bench Verified patches wrong | Tests that kill gold-patch mutants |
| Kernel checkers | Official check misses 16.9% of witnessed faults ([Measuring the Checker](https://arxiv.org/abs/2609.22220)) | A kill-matrix 2-input suite caught 98.0% |
| Verified-code specs | Verified reward 2.2% → 58.1% on trivial and leaky specs ([Tan](https://arxiv.org/abs/2605.30914)) | Reject a spec if a lazy program verifies |
| Model-written state checkers | 43.3% human agreement unvalidated ([Gym-Anything](https://arxiv.org/abs/2604.06126)) | checker(golden) = 1, checker(initial) = 0 |
| VLM trajectory judges | Best precision 61.5% ([ZeroGUI](https://arxiv.org/abs/2505.23762)) | State evidence; hide the agent's self-report |

Keep J = TPR − FPR > 0 in every difficulty bucket, because incorrect modes take over when J < 0 ([RLVεR](https://arxiv.org/abs/2601.04411)). Add a different model family rather than more votes. **Moderate.** Test the tester: known-good solutions must pass, and known-bad ones (policy failures, gold mutants, no-op, empty, master keys) must fail. A hacker–fixer loop cut attack success on held-out KernelBench exploits from 62% to 0% ([Hacker–Fixer](https://arxiv.org/abs/2606.08960)). **Moderate.** Under rejection fine-tuning, admitting 25% false positives cost −1.58pp, while dropping 75% of true positives cost −0.03pp, so gate for precision. Under GRPO, false negatives also draw gradient ([Verifier is the Curriculum](https://arxiv.org/abs/2607.09709)). **Moderate.**

### Two-sided validity gates

A hard item must pass two gates:

- **Lower gate.** It *fails* over k tries without the key ingredient: closed-book, no-CoT, no data, no tool, bridge masked, no-op agent, weak solver.
- **Upper gate.** It *succeeds* at least once with privileged help: gold evidence, a hint, the auxiliary construction, the reference in a fresh sandbox, another family.

A hardness filter alone kept unanswerable and ambiguous items ([DeepDive](https://arxiv.org/abs/2509.10446)). **Strong.** Use k ≈ 8. Calibrate the weak solver honestly: random auxiliary points plus DDAR solved 25/30 on IMO-30 ([HAGeo](https://arxiv.org/abs/2512.00097)).

### Telling hard from broken or ambiguous

Zero pass rate often means broken. Lowering OpenSIR's solve-rate floor from 0.5 to 0.1 cut validity from 70.82% to 42.31%, while GPT-5's solve rate fell only from 89.82% to 78.31% ([OpenSIR](https://arxiv.org/abs/2511.00602)). Small-k zeros also misfile solvable items: 577 prompts still labelled "extremely hard" after 1,000 iterations had produced a positive ([Knapsack RL](https://arxiv.org/abs/2509.25849)). Treat p̂ = 0 as "unknown". **Strong.** Triage in this order:

1. Re-score with another verifier: > 38% of Big-Math-RL-Verified responses were false negatives ([TinyV](https://arxiv.org/abs/2505.14625)).
2. Check well-posedness.
3. Look for alternative answers.
4. Try a stronger or privileged solver.
5. Have a judge label the item `too_hard` or `design_flaw` ([SETA](https://arxiv.org/abs/2607.10891)).
6. Park valid-but-unsolved items in a monitoring pool.

Difficulty operators also add ambiguity, so re-check uniqueness after every edit. Consensus fails on the hard tail: the unverified majority is wrong on 73.33% of AIME 2024 ([T³RL](https://arxiv.org/abs/2603.02203)). **Strong.**

### Reward-hacking guards

Hacks generalize: hacks learned in production coding RL came with 33.7% vs 0.7% broad misalignment ([MacDiarmid et al.](https://arxiv.org/abs/2511.18397)). **Strong.**

| Hack | Guard |
|---|---|
| Test tampering (`sys.exit(0)`, `conftest.py`) | Read-only or hidden tests ([ImpossibleBench](https://arxiv.org/abs/2510.20270)); out-of-process grading |
| Fetching the answer (`git clone`, sandbox attacks) | Delete future commits; eBPF allowlists ([DSec](https://arxiv.org/abs/2609.22978)) |
| Label enumeration (458 shortcuts at levels 11–20 vs 40 at 1–10) | Isomorphic twin must also pass ([IPT](https://arxiv.org/abs/2604.15149)) |
| Generator leaks answers into variants | Band-gated generator reward ([SvS](https://arxiv.org/abs/2508.14029)); builder–checker information barrier |

Mix a few certified-infeasible tasks into agentic RL. Removing them dropped ZeroGUI's infeasible-subset success from 41.3 to 22.1. **Moderate.** Inoculation prompting cut misalignment by 75–90%; filtering hacks and distilling did not. Watch these alarms:

- reward jumps;
- response-length collapse (510.7 → 45.7 tokens by step 58, before accuracy fell, [Sample difficulty](https://arxiv.org/abs/2605.28388));
- runaway turn counts;
- training reward diverging from an oracle.

### Decontamination and diversity

Complexification re-creates contamination: DeepMath's raw pool contained 90% of AIME24 and AMC23 ([DeepMath-103K](https://arxiv.org/abs/2504.11456)). Decontaminate the complexified outputs, and split train from eval by seed family. Confirm gains against a random-reward control: on Qwen2.5-Math-7B, random rewards gave +21.4 on MATH-500 vs +29.1 for ground truth ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). **Strong.** Diversity collapses quietly: over 15 escalation rounds, nearest-neighbour similarity rose from 0.223 to 0.464 ([RST](https://arxiv.org/abs/2608.05466)). Measure diversity at the skill level, and cap rewrites at ≤ 10 per seed (**Proposal**). **Moderate.**

### QC checklist (condensed)

1. **Before scaling.** Measure the verifier's TPR and TNR, fix false negatives first, and confirm the knob is monotone.
2. **Every item.**
   - Mechanical contract and checker endpoints.
   - Contract validity.
   - Two-sided gate at k = 8.
   - Uniqueness after each edit.
   - Isomorphic twin.
   - Decontamination.
   - RL band on the current checkpoint; 0-pass items go to triage.
3. **Every batch.** Hand-audit 100–200 items: 0/200 errors bounds the error rate below 1.9%. Quarantine anomalous families: 1–3 buggy task types out of 50 collapsed [UltraLogic](https://arxiv.org/abs/2601.03205).
4. **During and after training.** Run the alarms, keep a gold slice per difficulty bucket, and add random-reward and non-Qwen controls.

Full chapter: [04 · Verification and quality control](https://docs.google.com/document/d/1tNMP5XA4n1R9YsO5vwkQE9N8mQ-_MixWLy76GBzpSA8/edit)

## RL playbook

Only mixed-outcome groups carry gradient. Spend compute in this order: audit the verifier, reallocate rollouts, recycle zero-variance items, and only then generate.

### The pass-rate band and why

A group of N rollouts carries gradient with probability 1 − p^N − (1 − p)^N. At N = 8 that is about 0.99 at p = 0.5, but 0.08 at p = 0.01 or 0.99 ([Knapsack RL](https://arxiv.org/abs/2509.25849)). Edge data wins. In 100M from-scratch models it gave up to +42% pass@128, where in-distribution RL never raised pass@128 ([Interplay](https://arxiv.org/abs/2512.07783)). A 0.5 generator target beat 0.2 and 0.8 ([STRETCH](https://arxiv.org/abs/2609.18642)). **Keep roughly 0.2–0.8, centred near 0.3–0.6. Measure on the model being trained, and re-measure every stage. Strong.** Read the histogram:

- **Mass at p ≈ 1:** reallocate, add meta-tasks, complexify.
- **Mass at p ≈ 0:** audit, check behaviours, scaffold.
- **Bimodal:** add bridge levels.

### Measuring pass rates cheaply

Profile with the policy itself ([Olmo 3](https://arxiv.org/abs/2512.13961)); use stronger models only to certify solvability. [Pilot-Commit](https://arxiv.org/abs/2605.26606) runs 16 pilots on 3× the batch, skips p̂ > 0.75, defers p̂ < 0.125 and commits 48 more rollouts. It needed up to 1.9× fewer rollouts than GRPO. **Moderate.** Never delete an item on one pilot: only 81.2% of prompts fully solved in epoch 2 were still solved in epoch 3. **Strong.** Re-grade the whole original pool at each stage: [rStar2-Agent](https://arxiv.org/abs/2508.20722) re-filtered 42K problems to 17.3K. **Strong.**

### Pool management: filter, reallocate, recycle, retire

- **Filter.** Drop items solvable without CoT or tools ([Kimi k1.5](https://arxiv.org/abs/2501.12599), [GLM-5](https://arxiv.org/abs/2602.15763)). Drop zero-variance groups from the loss, which raises the asymptote, and refill the batch ([ScaleRL](https://arxiv.org/abs/2510.13786)). **Strong.**
- **Reallocate.** Knapsack RL gives 2 rollouts to saturated items and up to 93 to the hardest. That gained 2–4 points; uniform allocation would need about 2× compute to match. **Moderate.**
- **Recycle.** In agentic search, by the end of training about three-quarters of accepted groups came from recycled zero-variance queries ([Coelho et al.](https://arxiv.org/abs/2606.10709)). **Moderate.**
- **Retire, but review.** Retire items at historical p ≥ 0.9, but review about 2% of prompts. Review lifted GRPO from 60.25 to 64.06 on Qwen3-VL-8B ([ReMind](https://arxiv.org/abs/2606.03087)). **Emerging.**
- **Refresh.** Re-profile the full pool, then use a backup pool, then generate.

### Controllers and curricula: breadth beats ordering

Give each family its own controller. Options: a Gaussian pass-rate curriculum ([Nemotron 3 Nano](https://arxiv.org/abs/2512.20848)), a sliding window that promotes at ≥ 0.9 accuracy ([RLVE](https://arxiv.org/abs/2511.07317)), a proportional controller with retirement ([SCALER](https://arxiv.org/abs/2601.04809)), or directional rewrites accepted if 0.2 < acc(q′) < acc(q) ([RLAnything](https://arxiv.org/abs/2602.02488)). **Strong** (choice among them: **Moderate**). Breadth is the strongest lever. On ProRL-1.5B-v2, 400 environments added +3.37 in about 1,100 H100-hours; continued RL added +0.49 in 3,600. **Strong.** Ordering is second-order: over 12 matched seeds, no adaptive mixture beat an equal mix ([DataFlex-RL](https://arxiv.org/abs/2609.06107)). Stage only tiers that would otherwise get zero reward ([h1](https://arxiv.org/abs/2510.07312): AIME24 5.10 → 10.52 avg@32 at 3B). Reset the reference model and optimizer at each stage. **Moderate.**

### Self-play stability rules

1. **Multiply difficulty by an independent validity gate,** R = 1[V]·(1 − Acc) ([VHG](https://arxiv.org/abs/2605.06660)). **Strong.**
2. **Ground answers outside the model.** With pseudo-labels only, [R-Zero](https://arxiv.org/abs/2508.05004)'s label accuracy fell from 79% to 63%. **Strong.**
3. **Never let the loop train its own grader.** A trained user simulator forced success rates to about 50% ([SEAD](https://arxiv.org/abs/2602.03548)). **Strong.**
4. **Keep a persistent skill-level archive** ([R-Diverse](https://arxiv.org/abs/2602.13103)). **Strong.**
5. **Stabilize** with golden replay, human anchors and KL-anchored generator updates. **Moderate.**

The shape of the reward matters little (**Strong**). Give invalid proposals 0, not −0.1: a negative score caused a "death spiral" in [SSP](https://arxiv.org/abs/2510.18821). Anchor generation on the policy's weak items. Answer-preserving variants rewarded only in [12.5%, 62.5%] gave +18.3 pass@32 on AIME24 for Qwen2.5-32B-Instruct (SvS). **Moderate.**

### Making too-hard items learnable

Audit first, then check behaviours. On Countdown, a brief SFT on backtracking traces, even with wrong answers, closed Llama-3.2-3B's gap to Qwen-2.5-3B (about 30% vs 60%, [Cognitive Behaviors](https://arxiv.org/abs/2503.01307)). Then add the most on-policy help that works, and withdraw it as the policy improves:

| Lever | Reported result |
|---|---|
| Format ladder (MCQ → cloze → open) | +10.11 (Qwen) on pass@64 = 0 items ([Cog-DRIFT](https://arxiv.org/abs/2604.04767)) |
| Annealed prefix, 50% → 25% | 63.26 vs 60.26 for 50% throughout ([QuestA](https://arxiv.org/abs/2507.13266)) |
| Off-policy trace + 7 rollouts | +6.4 on six math benchmarks, Qwen2.5-Math-7B ([LUFFY](https://arxiv.org/abs/2504.14945)) |

**Moderate.** Items at p ≈ 1 still teach. On items solved 121–127 of 128 times, failure-prefix conditioning gave 44.5 vs 40.7 for plain RLVR ([Failure-prefix](https://arxiv.org/abs/2601.20829)).

### Composite reward design

With 8 constraints, only 5.7% of responses satisfy all of them ([CSE](https://arxiv.org/abs/2608.12426)). Use three layers:

1. **Validity gate.** Never pay for feasibility alone ([Forge](https://arxiv.org/abs/2605.08905)).
2. **Outcome reward.**
3. **Partial credit,** only where success is sparse.

A bipolar reward (+1 if fully correct, S − 1 otherwise) scored AIME24 82.6, vs 81.7 binary and 76.9 graded ([UltraLogic](https://arxiv.org/abs/2601.03205)). Probe every reward with trivial agents: an empty-response agent passes 38% of τ-bench-Airline ([ABC](https://arxiv.org/abs/2507.02825)). **Moderate** (pattern: **Proposal**).

### Mixing and dose

One prompt took Qwen2.5-Math-1.5B from 36.0 to 73.6 on MATH500 ([1-shot RLVR](https://arxiv.org/abs/2504.20571)). But that family gains even under random rewards, so test a non-Qwen model. At fixed volume, reuse up to 25× caused no significant degradation, so S steps × B prompts need about S·B/25 unique items ([Tan et al.](https://arxiv.org/abs/2509.25300)). Blend:

- equal shares across synthetic families;
- 20–33% oracle-backed real items (20% real bugs gave +7.0 pp over unanchored self-play, [Anchored Self-Play](https://arxiv.org/abs/2607.03523));
- 2–10% review or easy items.

**Proposal** on **Moderate** parts.

### Monitoring dashboard

Watch these signals:

- effective-prompt ratio and pass-rate histograms;
- entropy (faster collapse tracks poorer tests, [Skywork-OR1](https://arxiv.org/abs/2505.22312));
- response length, which collapses before accuracy;
- valid-proposal rate;
- reward vs a held-out oracle;
- pass@k at large k;
- a regression suite.

Pass@1 SD across seeds on AIME and AMC is 5–15 points ([Sober Look](https://arxiv.org/abs/2504.07086)). Adopt a change only if its paired 95% CI excludes zero over 8–12 seeds. **Strong.**

### Reference settings (condensed)

| Method | Reported setting |
|---|---|
| [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) | Keep 2–5 of 8 |
| Olmo 3 | Drop > 62.5% of 8 |
| [MiMo](https://arxiv.org/abs/2505.07608) | Drop > 90% of 16; 10% easy pool |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) / [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | Hard tier: pass@8 = 0 and pass@512 ≫ 0 / pass@100 > 0 |
| RLVE | Promote at 0.9 accuracy; window of 4 levels |
| SvS / GASP | Variants in [12.5%, 62.5%] / lemma p ∈ [0.3, 0.7] |

### Step-by-step recipe

The recipe as a whole is a **Proposal**; each step carries its own evidence tag.

1. Instrument the run. (Strong)
2. Audit verifiers and fix false negatives. (Strong)
3. Profile with the policy and defer zeros. (Strong)
4. Filter, reallocate and recycle. (Moderate)
5. Build verified generators and grow the number of families. (Strong)
6. Harden p = 1 items. (Moderate)
7. Scaffold p = 0 items. (Moderate)
8. Compose rewards by regime. (Moderate)
9. Blend and dose. (Moderate/Proposal)
10. Add a proposer only when generators run dry. (Strong for rules 1–4; Moderate for gains)
11. Re-profile at each stage. (Strong)
12. Decide with multi-seed CIs. (Strong)

Full chapter: [05 · RL playbook](https://docs.google.com/document/d/1lLy8y693dHjq1GAiTVukGYgPL3BevdzrOt2VMpc3xwI/edit)

## SFT playbook

SFT installs formats, behaviours, knowledge and atomic skills; RL composes them. SFT on the composite pool that RL would use often regresses below the base model. **Strong.**

### Instruction evolution done right

Naive evolution fails in three ways:

- **Overshoot:** 10,772 prompts ended at pass rate 0 vs 7,324 in band ([IFDecorator](https://arxiv.org/abs/2508.04632)).
- **Diversity collapse:** pass@8 fell below the untuned backbone, 54.2 vs 55.1 ([EvoTD](https://arxiv.org/abs/2605.11666)).
- **Negative net gain:** 10K Evol-Instruct samples scored below ShareGPT on IFEval, 41.96 vs 43.99 ([UltraIF](https://arxiv.org/abs/2502.04153)).

**Strong.** What works instead:

- **Answer-preserving operators.** Inversion gave +2.3 GSM8K points vs +0.4 for rephrasing ([MetaMath](https://arxiv.org/abs/2309.12284)).
- **Ladders.** Harden the solution and the environment, not only the text.
- **A dev-set stop after 2–3 rounds.** [WizardCoder](https://arxiv.org/abs/2306.08568) peaked at round 3. **Strong.**
- **Route overshoot one rung down** instead of discarding it.
- **Light answer checks, strict response checks.** Answer filtering did not help (41.9 unfiltered vs 40.0 GPT-verified, [OpenThoughts](https://arxiv.org/abs/2506.04178)). Degenerate responses did hurt: swapping 20% for "shirker" answers cut Instruct-SkillMix's AlpacaEval LC from 31.57% to 23.93%. **Strong.**

### Trace distillation

- **Select questions first.** A hard 1K beat a random 1K by 13 AIME24 points on Qwen2.5-32B-Instruct ([s1K](https://arxiv.org/abs/2501.19393)). **Strong.**
- **Pick the teacher by student results.** QwQ-32B was a better teacher than DeepSeek-R1 (OpenThoughts). Small students can get worse on long traces ([BREAD](https://arxiv.org/abs/2506.17211)). **Strong.**
- **Filter form, not correctness.** Failed trajectories sometimes helped (12.4% vs 6.74%, [Nemotron-Terminal](https://arxiv.org/abs/2602.21193)). Ablate before relying on it.
- **Re-solve rather than imitate raw trajectories.** Re-solving scored 52.1 vs 36.7, against a 47.0 base ([Terminal-Universe](https://arxiv.org/abs/2609.04148)). Show privileged hints to the teacher only. **Moderate.**
- **Maximize unique prompts first.** 64k × 1 beat 16k × 4 ([X-Coder](https://arxiv.org/abs/2601.06953)). Then allocate traces in proportion to difficulty: Llama3-8B MATH 21.2 → 46.6 ([DART-Math](https://arxiv.org/abs/2407.13690)). **Moderate–Strong.**

### Quality versus quantity

Small curated sets elicit; large sets teach (**Moderate**). At 7B, small SFT sets lost to RL: 45.7 vs 58.1 on Qwen2.5-Math-7B ([LIMR](https://arxiv.org/abs/2502.11886)). SFT on 4.77M synthetic trajectories took Qwen2.5-7B-Instruct from 12.8 to 73.1 on AIME24 ([PromptCoT 2.0](https://arxiv.org/abs/2509.19894)). Hardness does not replace volume: hard 192K scored 56.6, all 558K 59.2 and random 192K 45.1 ([ScaleDiff](https://arxiv.org/abs/2509.21070)). **Strong.**

### Meta-tasks

Critique fine-tuning beat plain SFT by 4–10% on six math benchmarks ([CFT](https://arxiv.org/abs/2501.17703)). Also consider verdict-checked retry traces, first-error localisation and recovery splicing. Self-correction must be learned on-policy ([SCoRe](https://arxiv.org/abs/2409.12917)). Start meta-tasks at 10–20% of the mix (**Proposal**). **Moderate.**

### What SFT fails to teach, and SFT → RL sequencing

| Capability | SFT | RL |
|---|---|---|
| Composition at depth 3 | RFT ≤ 2.6% | 27% ([f(g(x))](https://arxiv.org/abs/2509.25123)) |
| 16 stacked objectives | 0.000 | 1.900 ([MEM1](https://arxiv.org/abs/2506.15841)) |
| New rule variants (V-IRL-L) | −79.5 | +11.0 ([Chu et al.](https://arxiv.org/abs/2501.17161)) |
| IPhO physics | 19.8 → 15.9 | 25.2 ([Sim2Reason](https://arxiv.org/abs/2604.11805)) |

SFT still expands coverage. Qwen2.5-32B-Instruct's out-of-distribution score went from 42.3 to 43.0 with RL alone, 53.2 with SFT and 61.8 with SFT → RL ([InternBootcamp](https://arxiv.org/abs/2508.08636)). **Strong.** Send atoms, behaviour traces and knowledge to SFT. Send held-out composites, rule shifts and the hardest band to RL, and keep the two disjoint. **Moderate.** Pick the SFT checkpoint by a short RL probe. [SkillFactory](https://arxiv.org/abs/2512.04072) trailed R1 distillation before RL (2.8% vs 11.7%) and led after it (25.1% vs 21.2%).

### Mixing and dose

- Equal shares across operator families.
- 20–33% real items (**Proposal**). Seed + synthetic scored 57.3 vs 46.8 synthetic-only on LiveCodeBench ([rStar-Coder](https://arxiv.org/abs/2505.21297)).
- General data to protect other skills. Math-only SFT cut IFEval from 69.2 to 42.3.
- 2–3 generator families.

Mix difficulties rather than staging them. **Moderate.**

### Licensing of teacher outputs

| Teacher | Terms for training on outputs |
|---|---|
| DeepSeek-R1/V3.2, GLM-4.5/5 | MIT |
| Qwen3, gpt-oss, Olmo 3, Gemma 4 | Apache-2.0 |
| Kimi K2, MiniMax-M2 | Show the model name above 100M MAU |
| Llama 3.1/4 | "Llama" name prefix, 700M-MAU clause |
| Gemma 1–3 | A model trained on its outputs is a "Model Derivative" |
| Anthropic, Gemini API, OpenAI | Terms bar training competing models (OpenAI clause not re-verified) |

The EU AI Act template requires naming the models that generated synthetic data; enforcement began 2 August 2026 ([EU template](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models)). Log the generator on every row. **Strong** as documented terms; not legal advice.

### Step-by-step recipe

1. Define the SFT/RL split.
2. Screen licences.
3. Profile seeds with the student.
4. Harden for 2–3 rounds.
5. Split the pool.
6. Select questions by hardness.
7. Pilot the teachers.
8. Generate traces.
9. Add meta-tasks.
10. Mix and dose.
11. Run an RL probe.
12. Run RL, then distil back.

Full chapter: [06 · SFT playbook](https://docs.google.com/document/d/1PYXJLPwOT7pURaTZO2WrKWB6pjGcbEefuMPXDAG8JUQ/edit)

## Domain recipes

Build the answer or certificate first and write the prose last. Harden the verifier and the answer format before anything else. Composition is the knob that stays verifiable; train composed skills with RL. **Strong.**

**Math.**

- *Operators:* chain labelled seeds; nest answer-preserving sub-problems; lift the problem to parameters; invert, with a uniqueness check.
- *Verifier:* execution or SymPy on the construction; re-check rule-based negatives.
- *Knob:* op count, horizon and number of composed skills (success ≈ p^k).
- *Pitfall:* consensus filters drop the frontier.
- *Start:* [DeepMath-103K](https://arxiv.org/abs/2504.11456), [GSM-Infinite](https://arxiv.org/abs/2502.05252).
- *Refs:* h1; [MathForge](https://arxiv.org/abs/2601.20614) (37.61 → 42.17, Qwen2.5-Math-7B).

**Formal proofs and geometry.**

- *Operators:* extract and recompose subgoals; generate barely-provable conjectures (pass rate in (0, 1/4]); pair each statement with its negation; use random premises → deduction closure → traceback, then hide the auxiliary points.
- *Verifier:* a Lean kernel plus a bypass screen and a vacuity check; DDAR for geometry.
- *Knob:* proof length κ and number of auxiliaries.
- *Pitfall:* misformalization.
- *Start:* [Lean Workbook](https://arxiv.org/abs/2406.03847), [GenesisGeo](https://arxiv.org/abs/2509.21896).
- *Refs:* [InternGeometry](https://arxiv.org/abs/2512.10534) (IMO-50 44/50, vs 38 without its κ schedule); [STP](https://arxiv.org/abs/2502.00212).

**Competitive code.**

- *Operators:* harden the tests first; amplify input scale; lift the solution algorithmically, solution first; compose 2–4 features; make goals open-ended.
- *Verifier:* validator–generator–checker; brute force vs reference.
- *Knob:* features, input scale and test rounds.
- *Pitfall:* weak tests; LLM quality ratings (o3–human correlation 0.07).
- *Start:* [KodCode](https://arxiv.org/abs/2503.02951), [rStar-Coder](https://arxiv.org/abs/2505.21297).
- *Refs:* BenchEvolver (LCB-v6 Hard pass@1 87.0% → 45.7%); HardTests (RL pass@10 64.76 vs 57.14).

**Software engineering.**

- *Operators:* inject realistic bugs (rewrites, reverted history, bugs from feature work); combine several bugs; remove features along call depth; write symptom-only issues and hide the triggering tests.
- *Verifier:* F2P/P2P tests and a revert check; block `git clone` and `curl`.
- *Knob:* bug count, call depth and issue vagueness.
- *Pitfall:* ambiguity is not difficulty.
- *Start:* [SWE-smith](https://arxiv.org/abs/2504.21798), [R2E-Gym](https://arxiv.org/abs/2504.07164).
- *Refs:* [FrogNano](https://arxiv.org/abs/2609.07925) regenerates tasks at each checkpoint; its 4B model went 43.0 → 61.5 on SWE-bench Verified.

**Logic puzzles and gyms.**

- *Operators:* plant the solution; minimize clues until the answer is unique; upgrade the problem type (SAT → MaxSAT → MUS); move from feasibility to optimality.
- *Verifier:* certificate checks, after auditing the generator.
- *Knob:* an unbounded level, set per environment by a controller.
- *Pitfall:* Knights-and-Knaves-only RL dropped the code average from 67.46 to 56.09.
- *Start:* [Reasoning Gym](https://arxiv.org/abs/2505.24760), [SynLogic](https://arxiv.org/abs/2505.19641).
- *Refs:* RLVE; [ZebraLogic](https://arxiv.org/abs/2502.01100).

**Science.**

- *Operators:* compose simulator scenes, with a shortcut filter that ablates components; compose formulas; generate solver code first; convert MCQ to short answers.
- *Verifier:* simulator output within 5% error; the solver's optimum.
- *Knob:* number of entities and formulas.
- *Pitfall:* SFT regresses; plausibility hacks such as peroxides.
- *Start:* [MegaScience](https://arxiv.org/abs/2507.16812), [ether0](https://arxiv.org/abs/2506.17238).
- *Refs:* Sim2Reason (IPhO 19.8 → 25.2 under RL); [AutoOR](https://arxiv.org/abs/2604.16804) (Pump-NLP 0% → 48.98%).

**Tool use.**

- *Operators:* perturb inputs without changing the oracle; fuzz identifiers; hide intermediate steps; add long-range argument dependencies; inject tool failures.
- *Verifier:* the final DB state, accepting any valid path. A code-augmented judge scored 65.94 vs 55.46 for an LLM-only judge ([AWM](https://arxiv.org/abs/2602.10090)).
- *Knob:* walk length and number of implicit steps.
- *Pitfall:* noisy user simulators.
- *Start:* [tau2-bench](https://arxiv.org/abs/2506.07982), [APIGen-MT](https://arxiv.org/abs/2504.03601).
- *Refs:* [HardGen](https://arxiv.org/abs/2601.01498) (Qwen3-4B BFCLv3 62.13 → 79.14).

**Web and deep research.**

- *Operators:* swap entities for descriptions; fuzz constants; remove clues; intersect constraints; disperse evidence; layer aggregations.
- *Verifier:* the two-sided gate, plus a uniqueness check after every edit.
- *Knob:* hops and fuzzing strength. Accept items by realized difficulty, not structure.
- *Pitfall:* over-determining clues.
- *Start:* [WebShaper](https://arxiv.org/abs/2507.15061), [OpenSeeker](https://arxiv.org/abs/2603.15594).
- *Refs:* [FORT](https://arxiv.org/abs/2606.12087): 141.0 retrieval calls vs 20.6 for InfoSeek, and BrowseComp 72.2 from SFT alone.

**GUI and computer use.**

- *Operators:* write the evaluator first; chain verified tasks; turn explicit goals into implicit ones; change the start state; add noise entities; include infeasible tasks.
- *Verifier:* checker(golden) = 1 and checker(initial) = 0, with an information barrier between task builder and checker writer.
- *Knob:* a band of 1–7 successes out of 8.
- *Pitfall:* VLM judges score 63.76% where code checks give 38.93% ([GUI-Genesis](https://arxiv.org/abs/2602.14093)).
- *Start:* OSWorld-Verified, [AndroidWorld](https://arxiv.org/abs/2405.14573).
- *Refs:* CUA-Gym (54.5 → 62.1); [WebRL](https://arxiv.org/abs/2411.02337) (4.8% → 42.4%).

**Terminal and office.**

- *Operators:* extend tasks solution-first, recursively; invert environments; invert workbooks; add perturbed test cases.
- *Verifier:* an oracle run in a fresh sandbox, plus read-only tests; recompute spreadsheets in the same engine.
- *Knob:* recursion round.
- *Pitfall:* GRPO stalls on all-fail groups. On TB2.1, dense-reward PPO reached 64.0 while GRPO stayed flat at 51.7 ([T1](https://arxiv.org/abs/2609.11042); **Emerging**, one study).
- *Start:* [CLI-Gym](https://arxiv.org/abs/2602.10999), [SpreadsheetBench](https://arxiv.org/abs/2406.14991).
- *Refs:* RST (pass@4 fell from 90% to 2.5% over 15 rounds); [CalibForge](https://arxiv.org/abs/2608.06352) (19% → 96% learnable).

**Instruction following.**

- *Operators:* stack constraints, calibrated by p^k; compose them as Chain, Selection or Nesting; keep appending until pass@1 ∈ (0, 0.5].
- *Verifier:* a code checker per constraint, ANDed with an intent gate.
- *Knob:* number of constraints k.
- *Pitfall:* hacking, such as placeholders.
- *Start:* [IFBench](https://arxiv.org/abs/2507.02833), [MulDimIF](https://arxiv.org/abs/2505.07591).
- *Refs:* IFBench (TÜLU-3-8B 28.9 → 45.9); IFDecorator.

**Multi-hop and long context.**

- *Operators:* bridge composition; premise dispersal; pointer chains; removing lexical overlap; agent-mined distractors.
- *Verifier:* construction labels, scored by set-F1.
- *Knob:* hops and distractors, not raw length.
- *Pitfall:* label bugs (MRCR had about 5% wrong).
- *Start:* [MuSiQue](https://arxiv.org/abs/2108.00573), [PhantomWiki](https://arxiv.org/abs/2502.20377).
- *Refs:* [LoongRL](https://arxiv.org/abs/2510.19363) (7B multi-hop QA 72.4, +23.5); [LongRLVR](https://arxiv.org/abs/2603.02146).

**Memory and multi-turn.**

- *Operators:* stack objectives under bounded memory; split an instruction into shards across turns; hold the spec in a simulated user; inject into untrusted slots.
- *Verifier:* the seed's own verifier plus a control run. The all-shards-at-once control scores 95.1% of the fully specified run ([Lost in Conversation](https://arxiv.org/abs/2505.06120)).
- *Knob:* number of objectives and shards.
- *Pitfall:* policies that always ask or always abstain.
- *Start:* [QuestBench](https://arxiv.org/abs/2503.22674), [AgentDojo](https://arxiv.org/abs/2406.13352).
- *Refs:* MEM1; [IterResearch](https://arxiv.org/abs/2511.07327) (15.2% → 42.5%).

**Multimodal.**

- *Operators:* build the latent state before rendering; harden the question stem while hiding the answer; use generator levels.
- *Verifier:* the answer computed from the latent state, never from pixels.
- *Knob:* generator level. A flat pass-rate curve across levels means the knob is cosmetic.
- *Pitfall:* SFT regresses: Game-RL saw −3.9% from SFT vs +2.33% from RL.
- *Start:* [TRON](https://arxiv.org/abs/2606.01599), [ChartVerse](https://arxiv.org/abs/2601.13606).
- *Refs:* TRON (pass rate 72.8% → 41.3% across levels); [Game-RL](https://arxiv.org/abs/2505.13886).

**SQL and tables.**

- *Operators:* mutate the query AST; add schema tiers; add distractor tables.
- *Verifier:* execution, with non-empty gold results under 5 s.
- *Knob:* AST operators.
- *Pitfall:* text-only rewrites hurt (BIRD-dev 62.5 vs 64.9).
- *Start:* [OmniSQL](https://arxiv.org/abs/2503.02240).
- *Refs:* [Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315) (BIRD test 71.83%); [EvolSQL](https://arxiv.org/abs/2601.04875).

**Open-ended tasks.**

- *Operators:* convert the task to a verifiable form first; otherwise append constraints or ground the scenario in a concrete setting.
- *Verifier:* checklists plus a held-out judge.
- *Knob:* a 20–50% pass band.
- *Pitfall:* rubric hacking.
- *Start:* [RLCF](https://arxiv.org/abs/2507.18624), [RubricHub](https://arxiv.org/abs/2601.08430).
- *Refs:* [QUBRIC](https://arxiv.org/abs/2606.03968) (ArenaHard-hard 71.0 → 76.5).

Full chapter: [07a · Domain recipes: reasoning](https://docs.google.com/document/d/1X-YRC9ckYlMPM7POqsN0EJfAqHPvjhkqS-XXzIsXkI8/edit) · [07b · Domain recipes: agents and beyond](https://docs.google.com/document/d/1sQexHUTWmk-0eEf0K-_1cNEofcwDMxfuEpFLFpgZlFw/edit)

## What frontier labs report doing

Labs first selected hard items (2024 to early 2025), then constructed them (mid-2025 to 2026). In September 2026 one lab trained the model itself to construct them.

### Practices comparison

| Practice | What labs report | Tag |
|---|---|---|
| Sourcing | Competition math, issue–PR pairs, Q&A, knowledge graphs, deployment failures. Strip items solvable without CoT or tools ([Qwen3](https://arxiv.org/abs/2505.09388)), and convert MCQ | **Strong** |
| Bands | Exclude p = 0 and 1: keep 2–5 of 8, or drop > 62.5%, ≥ 0.75 or > 0.9. [Magistral](https://arxiv.org/abs/2506.10910) re-grades the whole original pool. The hard tier needs a certificate (pass@100 > 0, a teacher solve) or becomes a 10% masked tail ([Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220)) | **Strong** |
| Agentic synthesis | Agents escalate tasks and co-rewrite `solve()`/`verify()` (DeepSeek-V3.2: 1,827 environments); tool graphs grow respecting dependencies ([LongCat-2601](https://arxiv.org/abs/2601.16725)); SWE bugs are injected ([Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729): 807,693 instances). Synthetic agent RL transferred where code-and-search RL did not | **Strong**; transfer **Moderate** |
| Verifiers | Reasoning verifiers, LLM re-checks of rule negatives, final-state checks, hidden verifiers ([Kimi K3](https://arxiv.org/abs/2607.24653)). Six labs report new exploits | **Strong** |
| Staging | Gaussian pass-rate curricula; length stages are contested (for GLM-4.5, direct 64K beat a progressive schedule) | **Moderate** |
| Mixture | Multi-domain RL; small hard sets (Qwen3: 3,995 pairs, AIME'24 70.1 → 85.1); an easy anchor (MiMo 10%) | **Strong**; anchor size **Emerging** |

### The DeepSeek-V4.1-Flash learned task constructor

[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) treats each task as a (problem, environment, verification system) triplet: "Using difficulty and correctness as reward signals, we iteratively train the model to construct better tasks." Several solvers attempt each task. An inspector reads the environment together with their trajectories to catch hackability, and a repair agent re-tunes evaluation points that are too easy or too hard. Every RL run re-audits its tasks, and [DSec](https://arxiv.org/abs/2609.22978) runs more than 380K concurrent sandboxes. The report credits "essentially all of the observed gains" to data and environments, but gives no formula or ablation. The replayed failures come from deployment, not the current checkpoint, and the constructor is itself a frontier model. **Emerging.** Replicate the multi-solver and inspector pipeline first.

### The consensus recipe

1. Strip shortcuts. **Strong.**
2. Profile with the policy at k = 8–16, and re-profile at each stage. **Strong.**
3. Construct tasks when the band empties. **Strong** as practice; transfer **Moderate**.
4. Certify that hard items are solvable. **Strong.**
5. Scale the verifier and anti-cheat measures with task difficulty. **Strong.**
6. Train RL on a small, hard, multi-domain set. **Strong.**
7. Keep SFT hard and diverse. **Strong.**
8. Consolidate. **Emerging.**

### Where labs disagree

| Question | Positions | Our read |
|---|---|---|
| Who profiles? | The policy vs a proxy or teacher | The policy for the easy edge, a teacher for the hard edge (**Proposal**) |
| Upper edge | Only p = 1 removed vs caps at 62.5–90% | Keep 0.1–0.9 (**Moderate**) |
| p = 0 items | Drop vs a certified tier | Certify or defer (**Moderate**) |
| Length curriculum | Staged vs direct (GLM-4.5) | Unresolved (**Moderate**) |
| Select or construct | Nemotron-IMO vs DeepSeek-V3.2 | Select where human pools are deep; construct for agents (**Moderate**) |
| Constructor | Hand-designed vs trained | No ablation (**Emerging**) |

Full chapter: [08 · Frontier lab practices](https://docs.google.com/document/d/1ddbPT-TEskuCIV7VPNTSwb_YtnOR2JPppKLnW5gwxrg/edit)


## Pitfalls and failure modes

Many "hard" synthetic items that fail are broken, not hard: treat every pass-rate-0 item as a bug report until a solvability certificate says otherwise, and measure difficulty on the current policy in the exact training harness, on items already shown valid. Tags: **Strong** (several works or large ablations), **Moderate** (one careful study), **Emerging** (single unreplicated result), **Proposal** (untested synthesis).

### Failure modes at a glance

| Failure (tag) | Symptom | Key evidence | Detect | Mitigate |
|---|---|---|---|---|
| F1 Complexity ≠ difficulty (Strong) | Pool looks harder, pass-rate histogram barely moves | 64% of rejected mutations too easy ([Trading Human Curation](https://arxiv.org/abs/2606.03800)); 81% of validated candidates did not separate strong from weak solvers; harness change alone: 8.3% → 37.2% (4B) | Per-operator pass-rate histogram (k ≥ 8, current policy, RL harness) | Accept a mutation only if pass rate falls below the parent's; harden the executable artifact, not the prose |
| F2 Overshoot to unsolvable (Strong) | Mass piles up at p = 0; length collapses after a new family | 10,772 pass-0 vs 7,324 usable prompts after 5 rounds ([IFDecorator](https://arxiv.org/abs/2508.04632)); ~98% of random 12-constraint sets incompatible; yet 10.3–22.9% of pass@6 = 0 math items were reachable with perturbed decoding | p = 0 share per operator; triage by pass@100, stronger or tool solver, hint-conditioned attempt | Harden until pass rate over 8 rollouts is in (0, 0.5]; keep p = 0 only with a certificate; defer, don't delete |
| F3 Ambiguity as difficulty (Strong) | Diverse, defensible failures; strong models disagree | HLE 15.4% expert disagreement; a contract gate plus instruction audit cut weakly grounded tasks 32.8% → 1.2% | Privileged-information test (solved with evidence, failed without); list all valid answers | Fix the answer first; uniqueness check after every obfuscation step |
| F4 Wrong labels / verifiers (Strong) | Reward rises, held-out stalls; wrong keys in hardest bucket | R-Zero pseudo-labels 79% → 63%; unverified majority wrong on 73.33% of AIME 2024; rule math checkers 86% recall; TACO tests > 90% false positives on hard problems | Gold audit slice by difficulty; verifier TPR/TNR; metamorphic tests | Correct by construction (solution-first 97.7% vs 79.3% validity); cross-family evidence (Moderate) |
| F5 Consensus filters delete the frontier (Moderate) | Each filter makes the pool easier | A lifting pipeline succeeded on 47.1% of easy vs ~20% of hard seeds; agreement labelling "biases us toward easier queries" | Pass-rate histogram before/after each filter | Answer-first construction (Strong); verify only the new increment (Moderate) |
| F6 Generator artifacts, shortcuts (Strong) | Synthetic ≫ natural scores; partial-input baselines beat chance | Llama-3.x MCQ generators put the answer first 47.1–57.9% vs 25%; shortcut filtering 13.15 vs 7.14 on IPhO (3B) | Question-only, no-CoT, no-tool baselines; answer-marginal chi-square; leak search | Code RNG for structural choices; hide the answer from the hardening model |
| F7 Reward hacking (Strong) | Reward jumps; length collapse; `sorry`, `assume(false)`, edited tests | Shortcuts 40 at complexity 1–10 vs 458 at 11–20; GPT-5 cheated on 54–76% of impossible SWE variants ([ImpossibleBench](https://arxiv.org/abs/2510.20270)); 16% of 1,968 terminal tasks hackable from the description | Train-vs-oracle gap; trip-wires; trivial agents (an empty agent passes 38% of τ-bench-Airline) | Hack-audit every operator before RL; hidden or read-only out-of-process grading; separate solution and reward authors |
| F8 SFT-only composition (Strong) | In-distribution gains only; sometimes below base | RL 64% L2 / 27% L3 vs RFT 15% / ≤ 2.6%; IPhO: SFT 15.9, base 19.8, RL 25.2 | Held-out composition splits vs base; short RL probe | SFT for atoms and format, RL for composites; re-solve, don't imitate |
| F9 Specialist traps, forgetting (Moderate) | Target up, other skills down | Math SFT dropped IFEval 69.2 → 42.3 (RL kept 70.0); ~21% of reviewed mastered prompts partly regressed under GRPO | Regression suite; correct-set turnover | Domain mixing (Strong); 1–2% review queue, ~10% easy pool |
| F10 Topic collapse (Moderate) | Tasks pile onto few topics | One-probe generator reward put ~74% of tasks on one topic | Top-topic share; novelty vs a persistent archive of solution signatures | Skill-level dedup; ≤ ~10 rewrites per seed; diversity × validity gate |
| F11 Entropy collapse (Moderate) | pass@1 up, large-k pass@k at or below base | Fixed-data DAPO 72.4 → 64.8 (Qwen3-4B-Thinking); [SvS](https://arxiv.org/abs/2508.14029) kept entropy stable, +18.3 / +22.8 pass@32 on AIME24/25 | Entropy; pass@k to k ≥ 64–256 plus Cover@τ | Retire at p ≥ 0.9 with review; answer-preserving variants; generator KL β ≥ 0.05 |
| F12 Proposer death spirals (Moderate) | Valid-proposal rate → 0, entropy up | A −0.1 invalid penalty drove SSP's valid rate to 0 (SPICE is fine at −0.1) | Valid rate, proposer entropy, share at p ∈ {0, 1} | Gate R = 1[valid]·(1 − Acc); zero reward for malformed proposals (Emerging) |
| F13 Self-synthesis drift (Strong) | Rounds 1–3 improve, then reverse | R-Zero peaks after 1 (0.6B) or 3 (4B) iterations; real-bug anchoring +7.0 pp (+3.4 on human bugs) | Gold-slice accuracy per round | Accumulate, don't replace; 20–33% real anchors (Moderate); stop after ~3 rounds unless held-out rises |
| F14 Contamination (Strong) | Gains vanish post-cutoff or on renamed items | Evol-CodeAlpaca overlapped 70.7% of HumanEval; Qwen2.5-Math-7B completes MATH-500 from a 60% prefix 54.6% of the time (Llama3.1-8B 3.8%) | Semantic decontamination of outputs; partial-prompt probes | Decontaminate complexified outputs, not only seeds; post-cutoff sets |
| F15 Qwen spurious rewards (Strong) | Random-label gains only on Qwen2.5(-Math) | Random rewards +21.4 vs +29.1 ground truth on MATH-500 (Qwen2.5-Math-7B), none on Llama3/OLMo2 | Random-reward arm | Fresh procedural control; ≥ 1 non-Qwen family |
| F16 Eval noise (Strong) | Wins vanish on rerun or new aggregate | Seed SD 5–15 points on AIME24/AMC23; none of 8 selection policies had a paired CI excluding zero vs uniform (12 seeds) ([DataFlex-RL](https://arxiv.org/abs/2609.06107)) | Paired per-item CIs, SEs clustered by seed family; read raw generations | Pre-register; 3 seeds to screen, 8–12 to decide |
| F17 One generator family (Moderate) | Gains only on own-generator items | Generator-ID classifier 97.1% accurate; preference leakage 23.6% same model vs 2.8% across series | Held-out generator family | 2–3 generator families; generator, judge, student in different series |
| F18 Cost blowups (Moderate) | Spend outgrows the in-band pool | C_acc ≈ (c_gen + c_validate + c_env + k·c_roll) / (y_valid × y_band); SWE-Next's 2.25% yield ≈ 44 candidates per kept task | Cost per accepted in-band task, per operator | Audit verifier → profile → reallocate, recycle → only then generate; reuse up to ~25× |

### Pre-flight checklist (condensed)

- **Before generating:** verifier audited (TPR, TNR ≥ 0.9 for code; J = TPR − FPR > 0 per family and bucket); pool profiled on the current policy in the exact harness; eval suite pre-registered, split by seed; decontamination planned for outputs.
- **Per operator (pilot of a few hundred):** log y_valid, y_band, cost; accept mutations only if pass rate falls and validity holds; triage and defer p = 0; uniqueness, shortcut and hack audits; hand-audit 100–200 items with Wilson CIs; generator-signature check.
- **Before RL:** compositions go to RL; default blend (**Proposal**): equal operator mix, 20–33% real anchors, 2–10% easy/review pool; self-play guards (validity gate, KL β ≥ 0.05, ~3-round cap); dashboard alarms wired.
- **Before claiming a win:** paired 95% CI excluding zero over 8–12 seeds; random-reward arm and a non-Qwen family; holds post-cutoff and on a held-out generator family; no regression drop; post-training contamination audit.

Full chapter: [09 · Pitfalls and failure modes](https://docs.google.com/document/d/1IIPlhLkKuOYtbFCwi8PkOxvHKRr5RXQi00wgOnGDy9I/edit)

## Idea bank and roadmap

Make existing seeds hard again with the verifier you have; build construction engines second; trained generators are the 60–90-day bet. All outputs pass shared gates: bundle contract (oracle passes, no-op and known-bad fail), two-sided solvability, shortcut probes, a pilot band on the current policy (0 < p < 1, aim ≈ 0.3–0.6), dedup and per-operator yield logging. Status: Est = Established, Ext = Extension, Novel = our untested proposal. Effort: S < ~1 engineer-week, M 2–4 weeks, L > 1 month or a trained generator.

### Ranked shortlist

Score = Impact (1–5) × Confidence (Strong 5, Moderate 4, Emerging 3, Proposal 2) ÷ Effort (S 1, M 2, L 4). G1 and D1 have the highest impact (5) but cost more.

| # | Idea | What it does | Tag, effort, score | First deliverable |
|---|---|---|---|---|
| 1 | A1 Verifier audit + test hardening | Near-miss and hack tests, special judges, model fallback | Est, Strong, S, 20 | TPR/TNR on p̂ = 0 and p̂ = 1 tails |
| 2 | F1 Task-bundle contract + CI | Oracle passes, no-op and known-bad fail, hidden checkers | Est, Strong, S, 20 | Gates run on the whole pool |
| 3 | A2 Answer-preserving stem hardening | Rewrites keep the gold answer; SynthRL pass-rate gate | Est, Moderate, S, 16 | 3 rewrites per saturated seed |
| 4 | A5 Oracle-preserving tool perturbations | Distractor tools, indirect phrasing, noisy outputs; gold trace replays | Est, Moderate, S, 16 | 4 perturbation families |
| 5 | A4 Shortcut / no-context filters | No-CoT, tool-free, closed-book, no-data probes | Est, Strong, S, 15 | Shortcut rate per family |
| 6 | B3 p^k-calibrated constraint stacking | Pick k so all-pass lands in band; add structure | Est, Strong, S, 15 | Per-constraint success fit |
| 7 | F4 Two-sided solvability gates | Solvable with privileged info, unsolved without | Est, Strong, S, 15 | p = 0 tail split hard vs broken |
| 8 | I1 Mutated-twin / held-out-family evals | Hold out whole operator and generator families | Est, Strong, S, 15 | Frozen evaluation suite |
| 9 | G1 Per-family online controller | Difficulty knob, promote at ≥ 0.9 accuracy | Est, Strong, M, 12.5 | 10 families with knobs |
| 10 | A6 Strip hints and specifics | Implicit goals, terse errors; contract validity kept | Est, Moderate, S, 12 | 3-rung ladders |
| 11 | G4 Pilot–commit, reallocation, recycling | Pilot 16, skip p̂ > 0.75, defer p̂ < 0.125, commit 48 | Est, Moderate, S, 12 | Rollouts saved at equal accuracy |
| 12 | H1 Behavior priming | Brief SFT on try → check → backtrack traces, even wrong ones | Est, Moderate, S, 12 | Primed vs unprimed RL probe |
| 13 | I3 Yield and cost dashboard | Lineage, gate outcome, rejection reason, pilot p̂ | Est, Moderate, S, 12 | Cost per accepted in-band task |
| 14 | B1 Serial chaining + horizon curriculum | Answer i feeds problem i+1; 2 → 3 → 5 links | Est, Strong, M, 10 | Staged vs uniform chains |
| 15 | D1 [RST](https://arxiv.org/abs/2608.05466) recursive escalation | Grow solution, then environment, verifier, instruction | Est, Moderate, M, 10 | 3 rounds on 200 seeds |

Runners-up: C1, A3 (10), A7 (9); at 8: A8, B4, B6, D4, D8, E2, F2, F3, G2, G6, H2, H3, I2. Best novel bets (4): D2, J4, B8.

### Full idea list by theme

Format: ID, idea (effort, status, evidence). Shortlisted ideas point to their rank (#n) in the table above.

**A · Quick wins on verified seeds**
- A1 Verifier re-audit + test hardening (#1)
- A2 Answer-preserving stem hardening (#3)
- A3 Answer-space hardening: MCQ → open, validated special judges (S, Est, Strong)
- A4 Shortcut pre-pass (#5)
- A5 Oracle-preserving tool perturbations (#4)
- A6 Strip hints and specifics (#10)
- A7 Failure-prefix conditioning: roll out from a saturated item's rare failure (S, Est, Emerging)
- A8 Verdict, critique and error-location tasks from own rollouts (S, Est, Moderate)
- A9 Worst-case variant scoring, method-breaking edits (S–M, Est/Ext, Moderate)

**B · Composition engines** (atoms first; compose with RL)
- B1 Serial chaining + horizon curriculum (#14)
- B2 Composition of mastered atoms, tested on held-out atoms (M, Est, Strong in toy settings / Emerging for transfer)
- B3 p^k-calibrated constraint stacking (#6)
- B4 Evaluator-first chaining: compose trusted checkers, prove joint satisfiability (M, Est, Moderate)
- B5 Feature-tree code tasks; efficient and brute-force solutions must agree (M, Est, Moderate)
- B6 Realistic multi-bug and feature-addition SWE tasks (M, Est, Moderate)
- B7 Document → SQL → code chains with executable gold (L, Novel, **Proposal**)
- B8 Branch–merge topology curricula, merge-node rewards (M, Novel, **Proposal**)

**C · Obfuscation and inversion** (each needs a uniqueness check)
- C1 Fuzz, densify, disperse search questions; repair uniqueness (M, Est, Strong)
- C2 Inversion: backward math, abductive/inductive code (S–M, Est, Moderate)
- C3 Static → interactive: parameters only behind tools (M, Est, Emerging)
- C4 Shard the instruction or hide the spec behind a user, CONCAT control (M, Est, Moderate)
- C5 Counterfactual rules and worlds (S, Est/Ext, Emerging)
- C6 Fictional worlds and answer-source removal (M, Est, Moderate)

**D · Environment and agentic hardening**
- D1 RST solution-first recursive escalation (#15)
- D2 RST for workbooks, notebooks, documents (M–L, Novel, **Proposal**)
- D3 Logged dirty-data injection, gold from the clean source (M, Novel, **Proposal**)
- D4 Steered simulator perturbations, noise curricula, infeasible variants (M, Est, Moderate)
- D5 State inversion: break a healthy environment; the good state is the oracle (M, Est, Moderate)
- D6 Phase-state chaining: each validated phase's end state starts the next (L, Est, Emerging)
- D7 Co-harden capability and security, benign twins (M, Est, Emerging)
- D8 Open-ended goals; correctness-gated performance reward (M, Est, Moderate)
- D9 Formal-verification lift, stage/language ladder (M–L, Est, Moderate)

**E · Trained generators and self-play**
- E1 Validity-gated, band-rewarded setter RL with real anchors (L, Est, Moderate)
- E2 SvS variants from the policy's own correct solutions (M, Est, Moderate)
- E3 Stepping stones toward pass@k = 0 targets: easier lemma, harder lift, then the goal (L, Est, Emerging)
- E4 Anchored bug-injection self-play: one policy breaks and repairs repos (L, Est, Emerging)
- E5 Replicate a learned (problem, environment, verifier) constructor (L, Novel, **Proposal**)

**F · Verification infrastructure**
- F1 Task-bundle contract with CI gates (#2)
- F2 Pre-RL red-team harness: attack every operator family, public/hidden verifier split (M, Est, Moderate)
- F3 Decorrelated verification portfolio: channels that fail differently, J tracked (M, Est, Moderate)
- F4 Two-sided solvability gates (#7)
- F5 Label-preservation certificates outside math: edit in an executable IR, re-execute (M, Novel, **Proposal**)

**G · Difficulty calibration and curriculum**
- G1 Per-family online difficulty controller (#9)
- G2 Route each prompt by pass rate: complexify, train or scaffold (M, Est/Novel router, Moderate)
- G3 Operator-conditioned difficulty priors: predict child p̂ from parent and operator, skip profiling (M, Novel, **Proposal**)
- G4 Pilot–commit, reallocation, recycling (#11)
- G5 Bandit over operator × domain: measured held-out deltas set the budget (M, Novel, **Proposal**)
- G6 Make p ≈ 0 families learnable in a fixed order (S–M, Est, Moderate)

**H · SFT-specific**
- H1 Behavior priming (#12)
- H2 Distill verified trajectories on hardened agentic tasks (M, Est, Moderate)
- H3 Pick SFT prompts by difficulty and diversity; don't filter answers (S, Est, Moderate)
- H4 Fault-injected, recovery-spliced trajectories (M, Est, Emerging)

**I · Evaluating the pipeline**
- I1 Mutated-twin and held-out-family evals (#8)
- I2 Generator-signature audit (S, Est, Moderate)
- I3 Yield and cost dashboard (#13)
- I4 Diversity telemetry on solutions, not wording; failure-cluster descriptors (S–M, Est/Novel, Emerging)
- I5 Benchmark: is simulator-hard also real-hard? (M, Novel, **Proposal**)

**J · Novel combinations** (all Novel, **Proposal**; success metric = kill criterion)
- J1 One seed, four interaction operators: shards, long horizon, injections, withholding partner (M)
- J2 "Harder-than" variants checked by the Lean kernel (M)
- J3 Hidden-invariant generators: a short invariant behind an intractable surface (M)
- J4 Online re-complexification: re-harden items the moment they saturate (M–L)
- J5 Hacker–fixer loop inside recursive escalation: measure and cap exploitability per round (S–M)
- J6 Typed constraint DSL with SMT witnesses for IF self-play (M)
- J7 Replay-labelled first-error localization for agents (M)
- J8 Research-mined monthly tasks: fresh, uncontaminated, checkpoint rewards (M)

### Three-phase roadmap

- **Days 1–30, label-preserving only.** Week 1: F1, I3, I1 (incl. a non-Qwen model), G4. Week 2: A1, A4, F4, I2. Weeks 3–4: A2, A5, A6, B3, A7, G1, H1. *Exit:* effective-prompt ratio back up; in-band pool ≥ ~S·B/25 unique items (S steps × B prompts, reuse ≤ ~25×); screening gain with paired CI excluding zero; gains only on in-distribution synthetic evals → revisit A4, I1.
- **Days 31–60, construction engines.** Reasoning: B1, B2, A9. Agentic: D1, B4, D4. Search/QA: C1, C6. All: F2, G6, E2, triple logging for G3. *Exit:* every operator has yield, rejection reasons, exploit rate and held-out delta; drop "too easy" or still-exploitable operators; decide on the two best policies with 8–12 seeds.
- **Days 61–90, learned generators.** E1 or E5; G5, G3; one of J4, D2, B8; dose ladder, cross-family replication, pool re-audit. *Exit:* promote only pipelines that pass the decision rule with known cost per accepted in-band task.

### Experiment and decision protocol

- **Frozen eval suite:** natural targets plus post-cutoff items; held-out operator and generator families, mutated twins; pass@k to 128; regression suite; a non-Qwen family; trivial agents. Pre-register a domain-balanced aggregate (swapping gave ranking ρ = −0.33).
- **Controls:** seeds only and hard real data at matched compute; random reward; operator leave-one-out; dose ladder (no significant degradation up to 25 reuses, overfitting at 100); SFT vs RL.
- **Band monitoring:** falling effective-prompt ratio → re-complexify; rising all-zero share → two-sided gate, then scaffold; length collapse → audit zero items; sudden 0 → 100% jumps → quarantine and red-team.
- **Signature audits** (marginals, partial-input baselines, generator ID, variant decontamination) before and after training.
- **Decisions:** *adopt* only if the paired 95% CI excludes zero across 8–12 matched seeds (SEs clustered by seed family), no regression drops, and the gain holds post-cutoff, on a held-out family and on a non-Qwen model; *stop* loops after ~3 rounds unless held-out rises; *kill* Proposals that miss their metric and operators still exploitable after fixes. Adaptive mixtures must beat an equal mix; in DataFlex-RL none did.

Full chapter: [10 · Idea bank and roadmap](https://docs.google.com/document/d/1yjo85XDeN33EZSW1CvZKPJKkW5Ghj66j8tMrUimmi_U/edit)


## Research digest: the 23 subtopics

The evidence base is 23 citation-verified notes files (817 method entries): 14 planned subtopics plus 9 gap subtopics found by completeness critics. Each block gives the file's most decision-relevant TL;DR findings (numbers as reported), key works and a link to the notes.

### Instruction evolution for SFT

- "Complex" is not "hard for your model". Keep items by policy pass rate over k samples: IFDecorator uses (0, 0.5] over 8, D2Evo targets [0.4, 0.6], QbQ seeds from problems solved 8–15 times out of 16.
- Naive evolution mostly breaks tasks: after 5 iterations IFDecorator had 10,772 instructions at pass rate 0 and only 7,324 in band. Evol-Instruct pushed pass@8 below the untuned backbone (54.2 vs 55.1, EvoTD). Budget 2–3 rounds and track pass@k.
- Train the evolver against the learner: on Qwen3-8B-Base, D2Evo beat full-data RL (19K real) using 1.7K real samples (55.32 vs 52.70).

Key works: [Evol-Instruct](https://arxiv.org/abs/2304.12244), [IFDecorator](https://arxiv.org/abs/2508.04632), [D2Evo](https://arxiv.org/abs/2605.17037), [QbQ](https://arxiv.org/abs/2608.01522), [SEIF](https://arxiv.org/abs/2605.07465)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/01-instruction-evolution.md)

### Informal math synthesis

- Structural operators beat paraphrase. Chaining labelled GSM8K problems (h1, Dr. GRPO, horizon curriculum) took AIME24 avg@32 from 5.10 to 10.52; uniform mixing at equal compute gave no long-horizon gains.
- For saturated RL prompts, harden while preserving the answer: MathForge MQR with DGPO raised the average from 37.61 (GRPO) to 42.17 on Qwen2.5-Math-7B; SvS gave +18.3 / +22.8 pass@32 on AIME24/25.
- Never reward "hard" without a validity gate: VHG's 1[verifier accepts] × (1 − solver accuracy) lifted validity from 30.6% to 75.5%, while R-Zero's majority-vote labels decayed from 79% to 63% accurate.

Key works: [h1](https://arxiv.org/abs/2510.07312), [MathForge](https://arxiv.org/abs/2601.20614), [SvS](https://arxiv.org/abs/2508.14029), [VHG](https://arxiv.org/abs/2605.06660)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/02-math-problem-synthesis.md)

### Formal math, geometry and inequalities

- A deduction engine gives correctness for free; difficulty comes from proof depth and hidden auxiliary objects. InternGeometry's CBRL schedule scored 44/50 on IMO-50 vs 38 without it, 29 easy-only, 24 hard-only.
- Pass-rate windows agree across pipelines: STP (0, 1/4], GAR 0 < p < 0.5, Goedel-Prover-V2 (0, 0.75].
- Verified pipelines still get hacked (a Lean `apply?` bug silently dropped `sorry`). Always run the Lean check: ALF skipped it and only 87.8% of 2,000 instances were valid.

Key works: [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5), [InternGeometry](https://arxiv.org/abs/2512.10534), [STP](https://arxiv.org/abs/2502.00212), [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/03-formal-math-geometry-inequalities.md)

### Code and SWE tasks

- Audit the verifier first: SWE-ABS found about 1 in 5 "solved" SWE-bench Verified patches wrong (top score 78.80% to 62.20%). Hardening tests is the cheapest complexification operator.
- For saturated exact-answer tasks, change the goal: GRPO on FrontierSmith's 200 open-ended problems gave +8.82 FrontierCS and +306 ALE-bench rating on Qwen3.5-9B. Depth is the most reliable verifiable knob (ProgramDistill: frontier-agent success 100% to 64.0%, depth 1 to 8).
- SFT tolerates unverified or incorrect trajectories; RL needs strong verifiers and partial solvability. Anchor self-play to real bugs (Anchored Self-Play: +7.0 pp average over standard self-play).

Key works: [SWE-smith](https://arxiv.org/abs/2504.21798), [SSR](https://arxiv.org/abs/2512.18552), [BugPilot](https://arxiv.org/abs/2510.19898), [FrontierSmith](https://arxiv.org/abs/2605.14445), [HardTests](https://arxiv.org/abs/2505.24098)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/04-code-and-swe-tasks.md)

### Procedural puzzles and RLVR gyms

- Fix near-zero GRPO signal with unbounded-knob generators plus an online controller: RLVE's sliding window (promote at 90% accuracy, keep the last 4 levels) beat even an oracle static range.
- Many environments beat many instances (ReSyn, BBH: 400 × 40 = 75.19 vs 25 × 640 = 71.20), but select by ability coverage. LLM-written environments scale (SCALER 2,739) only behind staged gates; 13 of 105 Reasoning Gym tasks had material defects.
- Transfer is uneven and shrinks with model strength (K&K-only RL dropped the code average from 67.46 to 56.09), so keep puzzles a minority of the mix.

Key works: [RLVE](https://arxiv.org/abs/2511.07317), [ReSyn](https://arxiv.org/abs/2602.20117), [SCALER](https://arxiv.org/abs/2601.04809), [Reasoning Gym](https://arxiv.org/abs/2505.24760)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/05-procedural-puzzles-environments.md)

### Agentic tool-use and web tasks

- Fix the answer, gold trace or database state before writing text. Oracle-preserving transforms are cheapest: COVERT took BFCL v3 from 56.5 to 59.9 with RL alone (Qwen2.5-14B); HardGen took Qwen3-4B from 62.13 to 79.14 after SFT+RL.
- For search QA, fuzz values, raise graph coupling and spread evidence; removing FORT's shortcut controls raises a strong agent from 29.0% to 81.6%. Accept by trajectory signatures, not graph depth.
- Prefer executable, database-backed checkers (AWM code-augmented 65.94 vs 55.46 LLM-only); no LLM-judge setup exceeded AUROC 0.65 at detecting false success on tau2-bench.

Key works: [FORT](https://arxiv.org/abs/2606.12087), [COVERT](https://arxiv.org/abs/2604.09813), [HardGen](https://arxiv.org/abs/2601.01498), [AWM](https://arxiv.org/abs/2602.10090)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/06-agentic-tool-use-web-tasks.md)

### Multi-hop and long-context composition

- Composition plus RL teaches new difficulty: RL on depth-2 compositions reached 30% on unseen depth-3 (RFT ≤2.6%); GRPO on fictional PhantomWiki gave Qwen3-0.6B relative F1 gains of 56-131% on five real benchmarks.
- Gate two-sided (fail closed-book or with the bridge masked; solvable with gold evidence), then band-filter (LoongRL: 0<p<1 over 8, 277K to 72K). 72B Self-Instruct long-context data was only 33.1% truly multi-hop (LongMIT).
- Reasoning density beats raw length (LoongRL trained at 16K generalized to 128K), and the reward matters: 72.4 with two-way substring match vs 65.2 with an LLM judge.

Key works: [f(g(x))](https://arxiv.org/abs/2509.25123), [PhantomWiki](https://arxiv.org/abs/2502.20377), [LoongRL / KeyChain](https://arxiv.org/abs/2510.19363), [MuSiQue](https://arxiv.org/abs/2108.00573), [LongRLVR](https://arxiv.org/abs/2603.02146)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/07-multihop-and-long-context.md)

### Verifiable instruction-following constraints

- Stacking is a predictable dial: CSE fits per-constraint success at 72.0% × 0.922^(k−1). Choose k so p^k lands in band; use per-constraint or graded rewards at high k.
- Policy-aware hardening beats harder prompts (LLM-as-a-Tutor: offline Evol-Instruct 50.24, unmodified 50.51, tutor-gated constraints 51.96). Train with more constraints than the eval uses (IFBench, Qwen2.5: 48.9 with 1 per prompt, 59.5 with up to 3).
- LLM judges miss violations (Qwen3-32B: 30.6% / 20.9% of hard / soft failures caught multi-constraint, 59.3% / 54.7% pointwise). AND rewards with an intent check (hack rate 14.53% to 7.60%).

Key works: [CSE](https://arxiv.org/abs/2608.12426), [IFBench](https://arxiv.org/abs/2507.02833), [RECAST](https://arxiv.org/abs/2505.19030), [LsrIF](https://arxiv.org/abs/2601.06431), [VerIF](https://arxiv.org/abs/2506.09942)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/08-instruction-following-constraints.md)

### Self-play and automatic curricula

- Learnability selection (Bae et al.'s 0.2 < p < 0.8: +10 AIME at 3B, +12 AMC at 7B) only reorders; once nearly every seed has p = 1, only new tasks help.
- Never reward a proposer for solver failure without a validity gate: lowering OpenSIR's floor from 0.5 to 0.1 dropped validity from 70.8% to 42.3%. Grounding beats reward shape (R-Zero pseudo-labels: 79% → 69% → 63%).
- Anchor on your own unsolved tasks and scaffold p = 0: GASP solved 11 of 146 pass@100 = 0 problems (AZR and standard RL 0); Scaf-GRPO raised AIME24 from 30.0 to 43.3.

Key works: [R-Zero](https://arxiv.org/abs/2508.05004), [Absolute Zero](https://arxiv.org/abs/2505.03335), [SOAR](https://arxiv.org/abs/2601.18778), [GASP](https://arxiv.org/abs/2603.15957), [Scaf-GRPO](https://arxiv.org/abs/2510.19807)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/09-self-play-and-curriculum-rl.md)

### Theory and empirics of difficulty

- Compose solved tasks rather than paraphrase: RL on Level-2 compositions lifted Level-3 from about 5% to about 30%. RL composes primitives but does not create them (0% or 0.1% pretraining exposure: no transfer; 1%: up to +60% pass@128).
- Difficulty compounds: multi-step success tracks the product of atomic rates (ρ 0.69–0.96), and RL steps to 90% grow as depth^γ, γ rising from 1.05 to 2.60 as logic gets more expressive.
- Easy-to-hard ordering barely beats random mixing (OOD 0.27 vs 0.28) unless the top tier is sparse (ScaleLogic: curriculum γ = 1.33 vs 2.36 hard-only). At pass@K = 0, start with dense partial credit (DELTA-Code grokked after 450 steps below 1%).

Key works: [f(g(x))](https://arxiv.org/abs/2509.25123), [Interplay](https://arxiv.org/abs/2512.07783), [ScaleLogic](https://arxiv.org/abs/2605.06638), [Algebrarium](https://arxiv.org/abs/2602.08281), [DELTA-Code](https://arxiv.org/abs/2509.21016)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/10-theory-and-empirics-of-difficulty.md)

### Verification and quality control

- Check whether "too easy" is a verifier artifact: rule-based math checkers reject about 14% of correct answers; TACO tests had a false-positive rate above 90% on hard problems.
- Majority vote fails on the hard tail (unverified majority wrong 25.85% on MATH-500, 73.33% on AIME 2024). Build correctness in: BenchEvolver's solution-first evolution cut Hard-split pass from 87.0% to 45.7% with 89.9–97.7% validity.
- Noise mostly slows learning if J = TPR − FPR > 0, but votes correlate (8 completions ≈ 1.70 independent). Hard agentic tasks invite hacks; an abort option cut GPT-5's exploitation from 54% to 9%.

Key works: [BenchEvolver](https://arxiv.org/abs/2606.01286), [HardTests](https://arxiv.org/abs/2505.24098), [TTRL](https://arxiv.org/abs/2504.16084), [Master-RM](https://arxiv.org/abs/2507.08794), [ImpossibleBench](https://arxiv.org/abs/2510.20270)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/11-verification-and-quality-control.md)

### Frontier-lab practices

- Every lab keeps a policy-relative band excluding 0 and 1 (Olmo 3 drops pass rate > 62.5% of 8) and re-profiles with newer checkpoints: the band is a schedule.
- Selection runs dry, so 2025–26 reports construct hardness (agentic escalation, tool-graph expansion, KG multi-hop plus obfuscation, SWE bug injection) and certify solvability (GLM-4.5 "pass@8 = 0, pass@512 >> 0"; DeepSeek-V3.2 pass@100 > 0).
- Verifier quality must rise with difficulty (Seed-Verifier 82.7% vs Seed-Thinking-Verifier 99.3%). Mix domains and keep an easy anchor (MiMo re-samples an easy pool 10% of the time).

Key works: [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), [GLM-5](https://arxiv.org/abs/2602.15763), [Kimi k1.5](https://arxiv.org/abs/2501.12599), [Olmo 3](https://arxiv.org/abs/2512.13961), [Nemotron 3](https://arxiv.org/abs/2512.20848)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/12-frontier-lab-practices.md)

### Multimodal, science, SQL and non-verifiable domains

- Build the latent state (scene, program, SQL AST) first, compute the answer, then render. For saturated seeds, SynthRL accepts a rewrite only if 4 ≤ passes ≤ original − 2.
- Use synthetic hard tasks through RL; SFT often regresses (Sim2Reason IPhO: SFT −3.9 vs RL +5.4; AutoOR Hard-LP: SFT 26 vs RL 80, base 55). Policy-blind rewriting also hurts: "complex question" SQL augmentation cut BIRD-dev from 64.9 to 62.5 (Arctic).
- Non-verifiable domains need hardened prompts and rubrics (RubricHub, EvoRubrics, Rubric Dropout of 30–50% of criteria).

Key works: [SynthRL](https://arxiv.org/abs/2506.02096), [Sim2Reason](https://arxiv.org/abs/2604.11805), [AutoOR](https://arxiv.org/abs/2604.16804), [ViCrit](https://arxiv.org/abs/2506.10128)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/13-multimodal-science-sql-other-domains.md)

### Dynamic evaluation and open-endedness

- Lift saturated seeds into parameterized generators: SCALER reached 54.25 for Qwen3-4B-Base vs 52.04 (MATH-7.5k) and 53.52 (RLVE). DARG depth +4 took Claude-3-Opus from about 95% to 41.0% on GSM8K.
- Free-form hardening is unreliable (34 of 100 Evol-Instruct-style problems had errors vs 6, 3 and 7 per 100 for CHASE's verified pipeline). "Keep what strong models fail" enriches noise: HLE-Verified kept 668 of 2,500 items unchanged.
- Diversity is first-order: Self-Challenging's 200 synthetic tasks slightly degraded performance, 800 gave steady gains. Ship each task with a checker, a known-good solution and known-bad candidates.

Key works: [DARG](https://arxiv.org/abs/2406.17271), [CHASE](https://arxiv.org/abs/2502.14678), [HLE-Verified](https://arxiv.org/abs/2602.13964), [Self-Challenging](https://arxiv.org/abs/2506.01716)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/14-dynamic-eval-and-open-endedness.md)

### GUI and computer-use tasks

- The bottleneck is the checker. CUA-Gym's 32,112 (instruction, initial state, executable reward) tuples require `reward(golden)=1` and `reward(initial)=0`; its models reach 62.1 / 72.6 on OSWorld-Verified.
- Resample by pass rate each iteration, park 0/8 tasks in a monitoring pool, and keep a mixture: in WebGym, hard-biased sampling (2:5:3) peaked at 34.5% vs 38.2% for natural (≈25:5:1).
- VLM judges over-accept (GUI-Genesis: 63.76% by judge vs 38.93% by code assertions); give judges state (RL with IRA reward: 34.0% on OSWorld vs 34.9% with ground-truth scripts).

Key works: [CUA-Gym](https://arxiv.org/abs/2605.25624), [Qwen-CUA](https://arxiv.org/abs/2608.02352), [SCALECUA](https://arxiv.org/abs/2607.11185), [WebGym](https://arxiv.org/abs/2601.02439), [IRA](https://arxiv.org/abs/2607.25904)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/15-gui-computer-use-task-synthesis.md)

### Terminal and workspace agent tasks

- RST's recursive solution-first escalation (extend `solve.sh`, realign environment, verifier, then instruction) took DeepSeek-V4-Pro pass@4 from 90% to 2.5% over 15 rounds from 639 seeds, at about $0.05 per accepted task.
- Harder tasks stall GRPO (T1: stayed at 51.7% on TB2.1); PPO with a warm-started critic and r = P/20 reached 64.0%. CalibForge's calibrated data beat "author + validate" on TB2.0 (31.09% vs 22.47%).
- 16% of 1,968 terminal-benchmark tasks were hackable from the description alone. Open gap: no RST-style escalation yet for workbooks, notebooks or documents.

Key works: [RST](https://arxiv.org/abs/2608.05466), [T1](https://arxiv.org/abs/2609.11042), [CalibForge](https://arxiv.org/abs/2608.06352), [SETA](https://arxiv.org/abs/2607.10891)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/16-terminal-and-workspace-agent-tasks.md)

### Learned generators and simulated environments

- DeepSeek-V4.1-Flash (September 2026) trains the model as a task constructor rewarded for difficulty and correctness; it credits pipelines with "essentially all of the observed gains" but gives no reward formula or ablation.
- Generators steered to about 50% pass are standard (STRETCH: 0.5 beat 0.2 and 0.8). Cheapest: at accuracy > 0.8, RLAnything accepts a harder rewrite only if 0.2 < acc(q′) < acc(q).
- A learned component may choose tasks but must never grade (SEAD's trained simulator forced success rates around 50%). Simulators help only when steered: instructed perturbations gave +3.7 Tool Decathlon and +12.3 MCPMark.

Key works: [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969), [RLAnything](https://arxiv.org/abs/2602.02488), [STRETCH](https://arxiv.org/abs/2609.18642), [SEAD](https://arxiv.org/abs/2602.03548), [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/17-learned-generators-simulated-envs.md)

### Cost model and RL infrastructure

- Track cost per accepted in-band task: (generation + validation + environment build + profiling) / (validity yield × in-band yield). Most hardening rejects are "too easy" (64% of rejections; 25.5% of 930 mutations passed).
- Never delete p̂ = 0 items after one pilot (Knapsack RL: 577 prompts still "extremely hard" after 1,000 iterations had a positive), and reallocate rollouts before buying tasks (Knapsack about 2× compute-equivalent).
- Rollouts are over 90% of RL runtime; frameworks filter zero-variance groups but none generates tasks online. Check "oracle passes, no-op fails".

Key works: [Knapsack RL](https://arxiv.org/abs/2509.25849), [Pilot-Commit](https://arxiv.org/abs/2605.26606), [ScaleBox](https://arxiv.org/abs/2604.27467), [Harbor](https://docs.harborframework.com/core-concepts/tasks/overview)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/18-cost-model-and-rl-infrastructure.md)

### Dose, mixing and measurement rigor

- RLVR dose saturates early: one prompt took Qwen2.5-Math-1.5B from 36.0 to 73.6 on MATH500; reuse up to τ = 25 showed no significant degradation. Most evidence is Qwen2.5-Math: add a random-reward control and a second family.
- Default to a fixed equal mix (DataFlex-RL: no selection method's paired 95% CI excluded zero against uniform), anchored with 20–33% real or seed items plus 2–10% easy replay.
- At p ≈ 0.5, an unpaired test on 30 items needs about a 25-point gap. Adopt a data policy only if its paired 95% CI excludes zero across 8–12 seeds and holds on a non-Qwen model.

Key works: [1-shot RLVR](https://arxiv.org/abs/2504.20571), [LIMR](https://arxiv.org/abs/2502.11886), [DataFlex-RL](https://arxiv.org/abs/2609.06107), [Preference Leakage](https://arxiv.org/abs/2502.01534)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/19-dose-mixing-and-measurement-rigor.md)

### Learnability and difficulty calibration

- A too-hard item may signal a missing behavior: under PPO, Llama-3.2-3B plateaued at about 30% on Countdown vs about 60% for Qwen-2.5-3B; brief priming on backtracking traces closed the gap, even with wrong final answers.
- Saturated items still carry signal: failure-prefix conditioning on items solved 121–127 of 128 times raised the average from 40.6 to 44.5 (plain RLVR 40.7). Answer-preserving ladders gave +10.11 and +8.64 on pass@64 = 0 items.
- Not all hard items help: Hard@8 items lowered averages by 5.75, 11.24 and 1.07 points. Calibrated generators converge on about 0.3–0.6 pass rate.

Key works: [Cognitive Behaviors](https://arxiv.org/abs/2503.01307), [SkillFactory](https://arxiv.org/abs/2512.04072), [Failure-prefix](https://arxiv.org/abs/2601.20829), [Cog-DRIFT](https://arxiv.org/abs/2604.04767), [UltraLogic](https://arxiv.org/abs/2601.03205)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/20-learnability-and-difficulty-calibration.md)

### Interaction-structure operators

- Changing delivery keeps the label: in Lost in Conversation, CONCAT scores 95.1% of FULL while SHARDED loses 39% on average; treat CONCAT failures as label errors.
- Under RL "train short, test long" works (MemAgent trained on 32K/60K-token documents keeps 71.09% at 3.5M); under SFT it does not. Sharding needs a competence-gated curriculum (RLAAR ρ=0.8: 71.9 LiC; none: 63.2).
- Adversarial injections are the most durable difficulty source: CoER took utility from 63.2% to 76.3% while attack success fell from 38.5% to 0.2% (its own eval).

Key works: [Lost in Conversation](https://arxiv.org/abs/2505.06120), [MemAgent](https://arxiv.org/abs/2507.02259), [RLAAR](https://arxiv.org/abs/2510.18731), [CoER](https://arxiv.org/abs/2609.07529)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/21-interaction-structure-operators.md)

### Sound code verifiers and program induction

- Saturated code seeds have three sound exits: formal verification, program induction, performance. Verification matters most: PSV reaches 65.63% vs 34.46% for plain RFT, dropping to 31.82% without solution verification.
- Attacks move to specs (AlphaVerus: `assume(false)` spread to all programs) and timers (hidden inputs turn GPT-5.5's apparent 1.43× speedup into 0.88×). Shape rewards: CUDA Agent's milestone reward beat raw speedup (96.8% vs 60.4% of kernels faster than torch.compile).
- Verifier precision is the curriculum: in count-matched RFT, 25% false positives cost 1.58pp; discarding 75% of true positives costs 0.03pp.

Key works: [PSV](https://arxiv.org/abs/2512.18160), [AlphaVerus](https://arxiv.org/abs/2412.06176), [Re:Form](https://arxiv.org/abs/2507.16331), [NVARC](https://github.com/1ytic/NVARC), [CUDA Agent](https://arxiv.org/abs/2602.24286)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/22-sound-code-verifiers-and-induction.md)

### Sourcing, licensing and references

- Borrow a blend before synthesizing, but re-profile: NVIDIA's RL blends are ordered easy to hard by NVIDIA's checkpoints, with no pass-rate column.
- Open-pool licenses are unreliable (on 2026-09-30, Skywork-OR1-RL-Data had no license tag; KodCode-V1 is CC-BY-NC-4.0) and obligations travel through outputs. Default to MIT/Apache-2.0 generators; log the generator per row.
- Research-mined tasks yield little (RealMath: 14,747 theorems to 280 usable QA); expert tasks remain the reliable frontier-failing source. Decompose them into checkpoints (PaperBench: 20 papers to 8,316 gradable leaves).

Key works: [Nemotron RL blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends), [RealMath](https://arxiv.org/abs/2505.12575), [ResearchMath-14k](https://arxiv.org/abs/2605.28003), [CritPt](https://arxiv.org/abs/2509.26574), [Goal GAN](https://arxiv.org/abs/1705.06366)

[Full notes](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/research/notes/23-sourcing-licensing-and-references.md)


## How this was researched and verified

**Research.** The work covered 14 planned subtopics. Three completeness critics with different lenses (coverage, recency, practitioner needs) then found 9 gap subtopics.
- Each subtopic was researched by one agent with web search.
- A separate adversarial fact-checker then re-found every entry in its primary source. It corrected titles, IDs, mechanisms and numbers, removed numbers it could not confirm, and wrote the notes file.
- Result: 817 method entries, of which 315 were corrected and 3 dropped.

**Deterministic citation checks.**
- All 1,076 unique arXiv IDs in the notes were resolved on arxiv.org, and each official title was compared with its citation. All resolved.
- All 2,874 arXiv links in the report chapters were checked the same way, each label against its paper.
- Every relative link and section anchor in the report was validated.

**Chapter review.**
- Each chapter was written from the notes. A separate reviewer then fact-checked it against the notes and fixed errors in place. Typical errors were misattributions, missing conditions on numbers, overclaimed evidence tags and a few internal contradictions.
- A consistency editor then reconciled numbers and recommendations across chapters, making 18 fixes.
- A further reviewer checked the executive summary against the chapters, making 21 fixes.

**This overview.** Four writers condensed the chapters and notes into its sections. Three independent reviewers then checked every number, attribution, tag and link against the chapters and notes, and made 32 fixes, mostly results credited to a neighbouring paper or numbers that had lost their conditions.

**Google Docs copies.** Each chapter was uploaded verbatim, read back and compared word by word with its source. At least 99.9% of source words were recovered in order. The only differences come from the Markdown import, such as dropped code-fence language labels.

## Most-cited works across the report

The 45 works cited by the most chapters, ranked by the number of chapters that cite them and then by total citations. The full list of 1,134 works is in the [bibliography](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/11-bibliography.md).

| Short name | Title | arXiv date | Chapters citing | Citations |
|---|---|---|---|---|
| [IFDecorator](https://arxiv.org/abs/2508.04632) | IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards | 2025-08 | 10 | 20 |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | 2025-12 | 9 | 21 |
| [f(g(x))](https://arxiv.org/abs/2509.25123) | From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones | 2025-09 | 9 | 16 |
| [RLVE](https://arxiv.org/abs/2511.07317) | RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments | 2025-11 | 8 | 24 |
| [Trading Human Curation](https://arxiv.org/abs/2606.03800) | Trading Human Curation for Synthetic Augmentation in RLVR | 2026-06 | 8 | 23 |
| [T-SAE](https://arxiv.org/abs/2605.28388) | Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs | 2026-05 | 8 | 18 |
| [FORT](https://arxiv.org/abs/2606.12087) | FORT-Searcher: Synthesizing Shortcut-Resistant Search Tasks for Training Deep Search Agents | 2026-06 | 8 | 17 |
| [SvS](https://arxiv.org/abs/2508.14029) | Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR | 2025-08 | 8 | 15 |
| [IPT](https://arxiv.org/abs/2604.15149) | LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking | 2026-04 | 8 | 14 |
| [R-Zero](https://arxiv.org/abs/2508.05004) | R-Zero: Self-Evolving Reasoning LLM from Zero Data | 2025-08 | 8 | 14 |
| [HardTests](https://arxiv.org/abs/2505.24098) | HardTests: Synthesizing High-Quality Test Cases for LLM Coding | 2025-05 | 8 | 13 |
| [AutoCode](https://arxiv.org/abs/2510.12803) | AutoCode: LLMs as Problem Setters for Competitive Programming | 2025-10 | 8 | 11 |
| [InternBootcamp](https://arxiv.org/abs/2508.08636) | InternBootcamp: Boosting LLM Reasoning with Verifiable Task Scaling | 2025-08 | 8 | 9 |
| [RST](https://arxiv.org/abs/2608.05466) | Recursive Synthesis for Long-Horizon Terminal Tasks | 2026-08 | 7 | 23 |
| [GLM-5](https://arxiv.org/abs/2602.15763) | GLM-5: from Vibe Coding to Agentic Engineering | 2026-02 | 7 | 19 |
| [Knapsack RL](https://arxiv.org/abs/2509.25849) | Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation | 2025-09 | 7 | 15 |
| [QbQ](https://arxiv.org/abs/2608.01522) | Question Begets Question: Self-Evolving Curriculum for Reinforcement Fine-Tuning on Competition Mathematics | 2026-08 | 7 | 15 |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | SWE-smith: Scaling Data for Software Engineering Agents | 2025-04 | 7 | 14 |
| [h1](https://arxiv.org/abs/2510.07312) | h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning | 2025-10 | 7 | 14 |
| [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | 2026-09 | 7 | 14 |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Kimi k1.5: Scaling Reinforcement Learning with LLMs | 2025-01 | 7 | 13 |
| [EvoEnv](https://arxiv.org/abs/2605.14392) | Learning to Build the Environment: Self-Evolving Reasoning RL via Verifiable Environment Synthesis | 2026-05 | 7 | 13 |
| [OpenSIR](https://arxiv.org/abs/2511.00602) | OpenSIR: Open-Ended Self-Improving Reasoner | 2025-11 | 7 | 12 |
| [SWE-ABS](https://arxiv.org/abs/2603.00520) | SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark | 2026-03 | 7 | 11 |
| [FrogNano](https://arxiv.org/abs/2609.07925) | FrogNano: Training a 4B Coding Agent via Online Task Synthesis | 2026-09 | 7 | 11 |
| [OpenThoughts](https://arxiv.org/abs/2506.04178) | OpenThoughts: Data Recipes for Reasoning Models | 2025-06 | 7 | 11 |
| [SSR](https://arxiv.org/abs/2512.18552) | Toward Training Superintelligent Software Agents through Self-Play SWE-RL | 2025-12 | 7 | 11 |
| [rStar-Coder](https://arxiv.org/abs/2505.21297) | rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset | 2025-05 | 7 | 11 |
| [CalibForge](https://arxiv.org/abs/2608.06352) | CalibForge: Adversarial Solver Calibration for Scaling Learnable Terminal Tasks | 2026-08 | 7 | 10 |
| [VHG](https://arxiv.org/abs/2605.06660) | Verifier-Backed Hard Problem Generation for Mathematical Reasoning | 2026-05 | 7 | 10 |
| [Pilot-Commit](https://arxiv.org/abs/2605.26606) | Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training | 2026-05 | 7 | 9 |
| [Compositional GSM](https://arxiv.org/abs/2410.01748) | Not All LLM Reasoners Are Created Equal | 2024-10 | 7 | 8 |
| [Spurious Rewards](https://arxiv.org/abs/2506.10947) | Spurious Rewards: Rethinking Training Signals in RLVR | 2025-06 | 7 | 7 |
| [CUA-Gym](https://arxiv.org/abs/2605.25624) | CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents | 2026-05 | 6 | 16 |
| [UltraLogic](https://arxiv.org/abs/2601.03205) | UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward | 2026-01 | 6 | 12 |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models | 2025-08 | 6 | 12 |
| [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) | Qwen-AgentWorld: Language World Models for General Agents | 2026-06 | 6 | 12 |
| [Interplay](https://arxiv.org/abs/2512.07783) | On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models | 2025-12 | 6 | 11 |
| [DeepMath-103K](https://arxiv.org/abs/2504.11456) | DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning | 2025-04 | 6 | 11 |
| [Rule vs model verifiers](https://arxiv.org/abs/2505.22203) | From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning | 2025-05 | 6 | 11 |
| [MathForge](https://arxiv.org/abs/2601.20614) | Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation | 2026-01 | 6 | 11 |
| [Ineq-Comp](https://arxiv.org/abs/2505.12680) | Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities | 2025-05 | 6 | 11 |
| [VHD-Play](https://arxiv.org/abs/2609.27321) | Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms | 2026-09 | 6 | 10 |
| [Sim2Reason](https://arxiv.org/abs/2604.11805) | Solving Physics Olympiad via Reinforcement Learning on Physics Simulators | 2026-04 | 6 | 10 |
| [PROPEL](https://arxiv.org/abs/2606.18284) | Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier | 2026-06 | 6 | 10 |

## Workspace inventory and links

**Report chapters**

| Chapter | Review copy | Source |
|---|---|---|
| 00 · Executive summary | [Google Doc](https://docs.google.com/document/d/1ezY8l_ub8xwLBCSQoatVAVLIYN1iBBDu-I_Uw2TKzPM/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/00-executive-summary.md) |
| 01 · Diagnosis, difficulty and learning signal | [Google Doc](https://docs.google.com/document/d/1tpn_UscT66yYWQc3_pCVwOLC2gLqIhGlypmdkMWendE/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/01-diagnosis-difficulty-and-learning-signal.md) |
| 02 · Complexification operator taxonomy | [Google Doc](https://docs.google.com/document/d/1t5foEYg4oIvwpQmweZQbjzVE4kvaLCf6lH6oRozGDAo/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/02-complexification-operator-taxonomy.md) |
| 03 · Generation architectures | [Google Doc](https://docs.google.com/document/d/1WxWEIYAa41rQnNdvbJApXmWeeBTjtXXhDje5SHVbuPw/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/03-generation-architectures.md) |
| 04 · Verification and quality control | [Google Doc](https://docs.google.com/document/d/1tNMP5XA4n1R9YsO5vwkQE9N8mQ-_MixWLy76GBzpSA8/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/04-verification-and-quality-control.md) |
| 05 · RL playbook | [Google Doc](https://docs.google.com/document/d/1lLy8y693dHjq1GAiTVukGYgPL3BevdzrOt2VMpc3xwI/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/05-rl-playbook.md) |
| 06 · SFT playbook | [Google Doc](https://docs.google.com/document/d/1PYXJLPwOT7pURaTZO2WrKWB6pjGcbEefuMPXDAG8JUQ/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/06-sft-playbook.md) |
| 07a · Domain recipes I (math, formal, code, SWE, puzzles, science) | [Google Doc](https://docs.google.com/document/d/1X-YRC9ckYlMPM7POqsN0EJfAqHPvjhkqS-XXzIsXkI8/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/07a-domain-recipes-reasoning.md) |
| 07b · Domain recipes II (agents, web, GUI, terminal, IF, long context, multimodal, SQL) | [Google Doc](https://docs.google.com/document/d/1sQexHUTWmk-0eEf0K-_1cNEofcwDMxfuEpFLFpgZlFw/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/07b-domain-recipes-agents-and-beyond.md) |
| 08 · Frontier-lab practices | [Google Doc](https://docs.google.com/document/d/1ddbPT-TEskuCIV7VPNTSwb_YtnOR2JPppKLnW5gwxrg/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/08-frontier-lab-practices.md) |
| 09 · Pitfalls and failure modes | [Google Doc](https://docs.google.com/document/d/1IIPlhLkKuOYtbFCwi8PkOxvHKRr5RXQi00wgOnGDy9I/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/09-pitfalls-and-failure-modes.md) |
| 10 · Idea bank and roadmap | [Google Doc](https://docs.google.com/document/d/1yjo85XDeN33EZSW1CvZKPJKkW5Ghj66j8tMrUimmi_U/edit) | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/10-idea-bank-and-roadmap.md) |
| 11 · Bibliography (1,134 works) | GitHub only | [GitHub](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/report/11-bibliography.md) |

**Other contents**
- **Research notes:** the 23 verified literature-notes files are in [research/notes/](https://github.com/aertoria/SyntheticTasks/tree/claude/synthetic-task-complexity-research-twxsg8/research/notes). Each has a TL;DR, a methods table, per-method notes, an operator catalog, insights, open problems and references.
- **Tools:** the scripts behind the citation checks, the bibliography, the extracts and the Google Docs conversion are in [tools/](https://github.com/aertoria/SyntheticTasks/tree/claude/synthetic-task-complexity-research-twxsg8/tools).
- **Repository front page:** [README](https://github.com/aertoria/SyntheticTasks/blob/claude/synthetic-task-complexity-research-twxsg8/README.md).
- **Google Drive review folder:** holds this overview and the 12 chapter Docs. Its "00 · Start here" Doc is superseded by this overview and the executive summary.
- **Pending sync:** the Google Docs chapter copies predate the final consistency pass, which changed 20 lines across 9 chapters. The GitHub versions are final.
