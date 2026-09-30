# The complexification operator taxonomy: every known way to turn an easy task into a hard one

> **Key takeaways**
>
> - **Operate on an executable intermediate representation, not on prose.** Operators that *recompute* or *construct* the label (composition over code, graph and parameter scaling, answer-first or planted construction, bug injection checked by fail-to-pass tests) are the only ones that raise difficulty without raising label noise. Free-form "make it harder" rewriting mostly produces broken or easy items. In IFDecorator, 10,772 evolved prompts ended at pass rate 0 against 7,324 in band; in a gated RLVR study, 64% of rejected variants were "too easy". **Strong.**
> - **Composition is the most reliable source of new skill, but only under RL and only with real data dependencies.** RL on depth-2 compositions reached about 30% at unseen depth 3, while RFT on the same data stayed at ≤2.6%. Chains of unrelated steps are "artificially hard" and transfer poorly. **Strong.**
> - **Concealment is cheap and preserves the label, but it trades difficulty for ambiguity.** This covers fuzzing, clue removal, spec degradation and moving information behind tools. Every concealment step needs a control that proves the item is still uniquely solvable: a uniqueness enumeration, a CONCAT or oracle-evidence run, or a reference solver working from the degraded spec alone. **Strong.**
> - **Pick the operator family for the kind of difficulty you are missing.** Scale mostly adds length and tedium. Hard perturbations, insight hiding and counterfactual rules change *which method* applies. Information-acquisition operators (tools, turns, memory) expose interaction gaps that written-out variants hide: a base model scored 0.962 when the mechanism was written out and 0.204 when it had to query for parameters. Budget separate operators for length and for concept. **Moderate.**
> - **Paraphrase, verbosity, unrelated concatenation and ambiguity injection are anti-operators.** They lower accuracy without adding learnable difficulty. Adding rephrasings gave +0.4 GSM8K points where inversion gave +2.3/+2.6, and several "complex rewrite" augmentations scored *below* the untouched seeds. **Strong.**
> - **Compose operators in a fixed order.** Apply structure to the IR first, then constraints, then concealment and distraction, then delivery. Verify only the increment. Stop by measured pass rate, not by round count, and cap free-form LLM evolution at 2–3 rounds unless every round is solution-first and fully re-validated. **Moderate** (the order itself is **Proposal**).

**Contents**

1. [The frame: what an operator is and how the label survives](#1-the-frame-what-an-operator-is-and-how-the-label-survives)
2. [Family A — Compose](#2-family-a--compose)
3. [Family B — Constrain](#3-family-b--constrain)
4. [Family C — Conceal and obfuscate](#4-family-c--conceal-and-obfuscate)
5. [Family D — Invert](#5-family-d--invert)
6. [Family E — Scale](#6-family-e--scale)
7. [Family F — Perturb and distract](#7-family-f--perturb-and-distract)
8. [Family G — Abstract and generalize](#8-family-g--abstract-and-generalize)
9. [Family H — Upgrade the task type](#9-family-h--upgrade-the-task-type)
10. [Family I — Break and inject errors](#10-family-i--break-and-inject-errors)
11. [Family J — Target the learner](#11-family-j--target-the-learner)
12. [Family K — Remove scaffolding (and add it back)](#12-family-k--remove-scaffolding-and-add-it-back)
13. [Family L — Change representation](#13-family-l--change-representation)
14. [Anti-operators: fake hardness and how to detect it](#14-anti-operators-fake-hardness-and-how-to-detect-it)
15. [Operator × domain matrix](#15-operator--domain-matrix)
16. [Composing operators: algebra, order, budgets, stopping rules](#16-composing-operators-algebra-order-budgets-stopping-rules)
17. [Selection guide: which operators to try first](#17-selection-guide-which-operators-to-try-first)
18. [Gaps in the operator literature](#18-gaps-in-the-operator-literature)

---

## 1. The frame: what an operator is and how the label survives

A **complexification operator** maps a verified seed task `(x, y*, V)` (prompt, gold answer or end state, verifier) to a new task `(x′, y′*, V′)` that the target policy solves less often *for a learnable reason*. This chapter lists every operator family in the evidence base (22 topic notes, about 600 methods), with its definition, easy → hard examples in at least two domains, how the label and verifier survive, the evidence, the failure modes and the difficulty knobs. Chapter 01 covers why a pool looks "too easy" and why difficulty must be measured as the pass rate of the *current* policy. Chapter 03 covers the pipelines that run these operators at scale, and chapter 04 covers the gates.

### 1.1 The pipeline every robust operator shares

```
 seed (x, y*, V)
      │  lift: parse into an executable IR
      ▼        (program, computation graph, KG path, constraint list,
 IR  ─────────  environment state + checker, formal statement)
      │  operator(s) act on the IR, with knob κ
      ▼
 IR′ ──► label y′* recomputed / inherited / constructed ──► verifier V′ re-validated
      │  render IR′ to text / image / environment
      ▼
 x′ ──► controls (uniqueness, solvability, shortcut, no-op) ──► policy pass-rate gate ──► pool
```

Every method with near-zero label noise at scale computes the label from code or a solver: DyVal, DARG, GSM-Infinite, RLVE, SCALER, KUMO, DynaMath and the Putnam-AXIOM variations ([DARG](https://arxiv.org/abs/2406.17271); [GSM-Infinite](https://arxiv.org/abs/2502.05252); [RLVE](https://arxiv.org/abs/2511.07317)). Free-form hardening without an IR fails. In CHASE's baseline, 34 of 100 hard math problems had errors, yet frontier models still solved 82.5–88.9% of them ([CHASE](https://arxiv.org/abs/2502.14678)). With the same evolver, solution-first evolution gave 97.7% validity against 79.3% for problem-first ([BenchEvolver](https://arxiv.org/abs/2606.01286)). **Strong.**

### 1.2 Five ways a label survives an operator

The most useful property for choosing and composing operators is what happens to the label. Each operator card below carries one of these tags.

| Tag | Label mode | Examples | Risk under RL |
|---|---|---|---|
| **INV** | *Invariant*: the gold answer or end state is unchanged by construction | distractors off the solution path, nesting a given, obfuscation, scaffold removal, sharding, noise injection | Low. A corrupted item yields an all-zero GRPO group, so it is inert rather than harmful ([MathForge](https://arxiv.org/abs/2601.20614)). Main risk: ambiguity. |
| **REC** | *Recomputed*: the IR is re-executed | chaining over code, DAG scaling, parametric lifting, counterfactual rules | Low if the executor is right. Main risk: rendering drift (text no longer matches the IR). |
| **CON** | *Constructed*: answer or certificate first, task derived from it | planted solutions, back-translated constraints, bug injection (original code is the fix), explore-then-describe | Low for correctness, but several valid answers may exist, so grade the certificate, not the planted value. |
| **CERT** | *Certificate-graded*: any answer passing a checker is accepted | optimization with feasibility gate, proofs, test writing, abduction | Low if the checker is sound. The exploit moves into the task definition (weak specs, hackable timers). |
| **RED** | *Re-derived*: the new answer needs a fresh solve (teacher, vote, judge) | concept injection, LLM fusion, hard perturbation without a program | High. Majority-vote labels decay with difficulty; R-Zero's fell from 79% to 63% ([R-Zero](https://arxiv.org/abs/2508.05004)). A wrong key is a wrong reward. |

Rule of thumb: **prefer INV, REC, CON and CERT operators for RL pools; keep RED operators for SFT or put them behind an independent verifier** (chapter 04). SFT tolerates noisy labels much better than RL. Answer filtering did not beat no filtering in OpenThoughts' distillation study ([OpenThoughts](https://arxiv.org/abs/2506.04178)).

### 1.3 Five kinds of difficulty an operator can add

| Kind | What the solver must newly do | Typical operators | Caveat |
|---|---|---|---|
| **Volume / tedium** | carry more steps or state | scale size, op count, horizon, context | Smooth, predictable decay; mostly adds length ([Illusion of Thinking](https://arxiv.org/abs/2506.06941) critique; [Code2Math](https://arxiv.org/abs/2603.03202) rejects "computational tedium") |
| **Search** | explore a larger or more coupled space | solver-effort targeting, constraint coupling, optimization, topology densification | Needs certificate grading |
| **Method change / insight** | notice the memorized method no longer applies | hard perturbation, insight hiding, counterfactual rules, inversion | Hardest to generate; RED labels unless a program checks them |
| **Information acquisition** | find, ask for, or remember what is missing | move givens behind tools or users, sharding, beyond-context stacking, clue removal | The largest hidden gaps: 0.962 → 0.204 in [VHD-Play](https://arxiv.org/abs/2609.27321); −39% from sharding in [Lost in Conversation](https://arxiv.org/abs/2505.06120) |
| **Robustness** | ignore, resist or recover | distractors, noisy tools, injections, planted errors | Needs certified solvability after injection |

These axes do not substitute for one another. Horizon-reduced training generalized to longer horizons but *not* to harder Sudoku techniques ([Horizon-length study](https://arxiv.org/abs/2605.02572)). Depth extrapolates only about 3×, while expressiveness transfers best ([ScaleLogic](https://arxiv.org/abs/2605.06638)). **Moderate.**

### 1.4 Card format

Each operator below uses the same fields: **Def** (definition), **E→H** (easy → hard examples across domains), **Label** (tag plus how to keep V′ sound), **Knob**, **Evidence**, **Fails when**. Examples follow the verified notes. Many are short constructions that illustrate the cited method's operator rather than verbatim items from the paper, and "illustrative" flags the ones built only for exposition. Numbers are always the papers' own.

---

## 2. Family A — Compose

Composition joins verified pieces so that the new task needs all of them. It is the best-supported family for creating *new* capability. The canonical result: on 25 string functions, RL on Level-2 compositions took unseen Level-3 from near zero to about 30% and Level-4 to about 15%. RFT on the same data stayed at ≤2.6% at Level 3, and RL on Level-1 atoms alone gave <1% ([f(g(x))](https://arxiv.org/abs/2509.25123)). OMEGA found that large in-skill RL gains "do not reliably carry over to the composed setting" unless composed items are in the mix ([OMEGA](https://arxiv.org/abs/2506.18880)). **Strong:** install the atoms (SFT, RFT or about 1% exposure), then run RL on composites.

**A1. Sequential chaining (f∘g, answer passing, bridge composition)**
- **Def:** the output of subtask i becomes a required input of subtask i+1, directly or through a deterministic adapter.
- **E→H:** *Math* (illustrative, brute-force checked): "12 muffins/h for 5 h?" (60) → "…pack 4 per box, $8 per box, 15% off: what is paid?" ($102), the construction of [Compositional GSM](https://arxiv.org/abs/2410.01748). *Web QA*: "Who directed Inception?" + "When was Nolan born?" → "When was the director of the 2010 dream-heist film born?" ([MuSiQue](https://arxiv.org/abs/2108.00573)). *Tools*: create_order → get_order_details executed on a DB ([Agent-World](https://arxiv.org/abs/2604.18292)). *Data analysis*: a 4-hop pandas chain built as one executable reference program ([DataMind](https://arxiv.org/abs/2509.25084)).
- **Label:** REC. Substitute the verified upstream value into the downstream code solution and execute. Keep adapters deterministic and integral (types, units, ranges) ([h1](https://arxiv.org/abs/2510.07312)). For QA, require that the tail fails with the bridge masked and the head fails closed-book (MuSiQue's disconnection filter kept only 26.5% of candidate tails).
- **Knob:** hop count or horizon h.
- **Evidence:** Qwen2.5-Math-7B-IT scores above 80% on MATH but solves under 60% of two-hop compositional GSM problems, below the S1·S2 product prediction ([Compositional GSM](https://arxiv.org/abs/2410.01748)). h1's staged horizon curriculum took AIME24 from 5.10 to 10.52 on a 3B model, while only-short, uniform-mix and only-long baselines gave no long-horizon gain at equal compute. Long-horizon accuracy fell *faster* than independent per-step compounding predicts ([h1](https://arxiv.org/abs/2510.07312)).
- **Fails when:** adapters create ill-posed values (non-integer people), so provide an "INVALID" escape ([Instruction Fusion](https://arxiv.org/abs/2312.15692)). Labels come from self-consistency, which biases toward easy compositions ([DataMind](https://arxiv.org/abs/2509.25084)). The full horizon is trained without stages: outcome-only reward needs Θ(α^−H) samples for a useful gradient, against Θ(H) with a curriculum (h1 theory).

**A2. Width composition (parallel conjunction, bundling, recombining validated tasks)**
- **Def:** merge independent subtasks into one episode; every part must be satisfied.
- **E→H:** *Tool use*: "cancel order X" + "update the address on order Y" → one conversation for the same persona ([APIGen-MT](https://arxiv.org/abs/2504.03601); [TaskCraft](https://arxiv.org/abs/2506.10055)). *Long context*: 1 QA over 1 document → 5 QA embedded in concatenated documents, all required ([Beyond Reward Engineering](https://arxiv.org/abs/2606.18831)). *GUI*: `file_exists(tutorial.pdf) ∧ url_visited(docs.python.org/…)` ([UltraCUA](https://arxiv.org/abs/2510.17790)).
- **Label:** REC or CON. Replay the concatenated gold actions from a fresh state and re-run policy tests, which catches conflicts such as return and cancel on the same order. Choose at most one subtask per mutually exclusive group ([tau2 generator](https://arxiv.org/abs/2506.07982)). For evaluator-first GUI tasks, build a golden state proving the checkers are jointly satisfiable and check that checker(initial) = 0.
- **Knob:** width k.
- **Evidence:** evaluator-first composed GUI tasks had 29% rollout success against 45% for instruction-first tasks ([UltraCUA](https://arxiv.org/abs/2510.17790)).
- **Fails when:** subtasks interfere (shared state), or width is the only knob. Independent parts mostly multiply per-part failure (≈ p^k). New skill appears only when parts share state, dependencies or a bounded memory (A1, H6). **Proposal.**

**A3. Nested / recursive substitution (a constant becomes a sub-problem)**
- **Def:** replace a leaf constant or named entity with a description or sub-question that uniquely resolves to it. The final answer is unchanged.
- **E→H:** *Web*: "When was Michael P. Hein born?" → "When was the Eckerd College alumnus who served as the first County Executive of Ulster County, New York … born?" ([ASearcher](https://arxiv.org/abs/2508.07976)). *Math*: "rectangle 12 × 5, diagonal?" → "length = number of divisors of 60, width = number of primes < 12; call the diagonal the *span*; find the span" (still 13) ([MathForge](https://arxiv.org/abs/2601.20614)). *Search*: layer-wise leaf expansion over a formal query ([WebShaper](https://arxiv.org/abs/2507.15061); [InfoSeek](https://arxiv.org/abs/2509.00375)).
- **Label:** INV. Execute each nested sub-problem and require exact equality with the constant it replaces. For KG or web questions, check only the new sub-question: type consistency, proper replacement, no leak. This keeps cost linear in hops, so synthesized tasks can exceed what the generator itself can solve ([TaskCraft](https://arxiv.org/abs/2506.10055)).
- **Knob:** number of substituted leaves; nesting depth.
- **Evidence:** MathForge's o3 audit found 97–99% answer equivalence ([MathForge](https://arxiv.org/abs/2601.20614)).
- **Fails when:** a constant next to the target is expanded, which creates a shortcut, so expand *leaves*. Injected descriptions add identifying facts that act as shortcuts, and FORT's ablation shows fuzzing and cycles are needed on top ([FORT](https://arxiv.org/abs/2606.12087)).

**A4. Conditional branching (Selection)**
- **Def:** gate sub-requirements behind a condition on the input, possibly nested.
- **E→H:** *Instruction following*: "Describe this product" → "If the listed price is > $100, write a JSON review {pros, cons, verdict}; otherwise a ≤280-character post with exactly 2 hashtags" ([ComplexBench](https://arxiv.org/abs/2407.03978); [LsrIF](https://arxiv.org/abs/2601.06431)). *Math*: conditional fusion ("compare / select between two analogous problems") ([MathFusion](https://arxiv.org/abs/2503.16212)).
- **Label:** REC. Make the condition *computable from the input* so the gold branch is known without an LLM. Score only the active branch; an output on the wrong branch gets 0.
- **Knob:** branching depth; mixing Selection with Chain.
- **Evidence:** GPT-4 on ComplexBench scored And 0.881, Chain 0.766, Selection at depth ≥3 0.694 and Selection+Chain 0.626. On the coherent test for multi-layer Selection it scored 14.9% ([ComplexBench](https://arxiv.org/abs/2407.03978)). Averaging rewards across structure cost about 4 IFEval and 5 CFBench points against structure-aware aggregation ([LsrIF](https://arxiv.org/abs/2601.06431)). **Moderate.**
- **Fails when:** an LLM decides the condition (noise), or rewards are averaged instead of aggregated along the logic.

**A5. Skill / concept mixing (crossover)**
- **Def:** require k skills, concepts or features *jointly*, not one after another.
- **E→H:** *Math*: "divisors of 360?" (24) → "triangle (0,0),(d,0),(0,k) with d = divisors of 360; smallest k making the area a perfect square" (k = 3) ([MATH²](https://arxiv.org/abs/2407.21009)). *Code*: binary search on a sorted array → binary search on the answer + prefix sums + monotone-deque feasibility on a circular array with updates ([EpiCoder](https://arxiv.org/abs/2501.04694)). *Terminal*: "parse a CSV" → 3–5 skills incl. a statistical computation, dependency setup and a test-writing requirement ([Nemotron-Terminal](https://arxiv.org/abs/2602.21193)). *ARC*: "recolour the largest object" + "mirror along the symmetry axis" ([BARC](https://arxiv.org/abs/2411.02272)). *Kernels*: ReLU → conv + bias + norm + activation fusion ([DRTriton](https://arxiv.org/abs/2603.21465)).
- **Label:** REC when the pieces are executable: compose function graphs and execute ([RV-Syn](https://arxiv.org/abs/2504.20426)); for code, the proposer writes program and input and the executor gives the output ([EvoTD](https://arxiv.org/abs/2605.11666)). Otherwise RED: SymPy-equivalence voting ([KPDDS](https://arxiv.org/abs/2403.02333)) or cross-family agreement. A skill audit rejects items where one skill is merely appended ([EvoTD](https://arxiv.org/abs/2605.11666)).
- **Knob:** k; success roughly follows p^k ("the success rate on MATH² is the square on MATH").
- **Evidence:** in DafnyComp, chains of verified functions synthesize at 47% against <8% for DAG compositions, and proving a caller from its callees' specs succeeds at 3.69% ([Re:Form + DafnyComp](https://arxiv.org/abs/2509.23061)). Graph-ordered skill composition beat random multi-skill composition by 3.8 points on TB2.0 ([SkillSynth](https://arxiv.org/abs/2604.25727)).
- **Fails when:** pairs are incoherent (use a co-occurrence prior, [MathScale](https://arxiv.org/abs/2403.02884)), labels come only from a judge ([Genetic-Instruct](https://arxiv.org/abs/2407.21077)), or the algorithm leaks in the prompt.

**A6. Problem / statement fusion**
- **Def:** fuse two seeds into one statement that needs both, by rule or by LLM.
- **E→H:** *Inequalities*: (x+y)/2 ≥ √(xy) → ((a+b)/2)·((c+d)/2) ≥ √(ab)·√(cd) (Type I product) ([Ineq-Comp](https://arxiv.org/abs/2505.12680)). *Code*: EvoEval Combine merges two HumanEval problems ([EvoEval](https://arxiv.org/abs/2403.19114)). *Code prompts*: fuse two random seed prompts ([Instruction Fusion](https://arxiv.org/abs/2312.15692)).
- **Label:** rule-based fusion is CON (seed proofs plus a combination lemma, when side conditions hold). LLM fusion is RED: compile, prove, penalize statement edits, require p > 0 ([GAR](https://arxiv.org/abs/2510.11769)); for code, differential tests across independent references.
- **Knob:** number of fused seeds; fusion type (sequential / parallel / conditional).
- **Evidence:** DeepSeek-Prover-V2-7B dropped from 66.2% to 47.0% (Type I) and 42.1% (Type II) on composed inequalities, even with the parts' proofs in context. SFT on about 8K composed items fixed only in-distribution Type I ([Ineq-Comp](https://arxiv.org/abs/2505.12680)). 5.6% of LLM-fused math problems stayed unreasonable after 5 regenerations ([MathFusion](https://arxiv.org/abs/2503.16212)).
- **Fails when:** LLM fusions are unreasonable; for SFT that tolerance does not carry over to RLVR.

**A7. Multi-bug / multi-fault union**
- **Def:** combine several independently validated faults in one artifact.
- **E→H:** *SWE*: one flipped comparison with an explicit issue → 2–3 interacting bugs across modules with no issue ([SWE-smith](https://arxiv.org/abs/2504.21798); [Active-SWE](https://arxiv.org/abs/2608.04682)). *Terminal*: a mis-pinned transitive dependency + a missing env var + broken permissions ([CLI-Gym](https://arxiv.org/abs/2602.10999)). *Self-play*: the solver's failed repair becomes a higher-order bug ([SSR](https://arxiv.org/abs/2512.18552)).
- **Label:** CON. Validate each bug alone. Gold = union of patches, and the combined F2P set must equal the union of the individual sets, which checks that the bugs don't cancel.
- **Knob:** number of bugs; spread across files or modules.
- **Evidence:** SWE-smith Combine had 96.9% yield at 0¢, a median of 11 changed lines and a median of 15 F2P tests ([SWE-smith](https://arxiv.org/abs/2504.21798)).
- **Fails when:** integration fails. Active-SWE had to use a window of 2 adjacent fixes.

**A8. Long-horizon task chaining (phases, milestones, inherit/merge)**
- **Def:** order verified atomic tasks so each builds on the state or artifacts left by the previous one.
- **E→H:** *SWE*: one isolated milestone (>80% success) → a continuous chain (≤38.03% score, about 13% full resolves) ([SWE-Milestone](https://arxiv.org/abs/2603.13428); [SWE-Flow](https://arxiv.org/abs/2506.09003)). *GUI*: "change the slide background" and "export to PDF" → a length-4, 3-app chain; the best agent completes 31% ([ChainWorld](https://arxiv.org/abs/2606.21654)). *Mobile*: inherit ("in Groceries add three items and pin it"), merge, rewrite ([GSAR](https://arxiv.org/abs/2608.22847)). *Data science*: turn 24 "roll back to the pre-penalty scores from turn 12" ([LongDS-Bench](https://arxiv.org/abs/2605.30434)).
- **Label:** REC. Store each phase's end state and checker, and reward the AND (or partial credit). Later checkers must not depend on artifact identities created by one particular earlier solution ([Qwen-CUA](https://arxiv.org/abs/2608.02352)). Apply hard compatibility rules first (snapshot mismatch, destroyed state, evaluator interference), and use an LLM coherence judge only afterwards ([ChainWorld](https://arxiv.org/abs/2606.21654)).
- **Knob:** chain length; number of apps or modules; data dependency between phases.
- **Evidence:** in AgentSynth, level-6 tasks need 40–60 steps ([AgentSynth](https://arxiv.org/abs/2506.14205)). But **feasibility is not usefulness**: chain-then-summarise generators had the highest executor success (56.4%) and produced the weakest agent (21.6 vs 38.2 AndroidWorld pass@1) ([AutoPlay](https://arxiv.org/abs/2509.25047)). **Strong:** require data dependencies between phases.
- **Fails when:** unrelated operations are concatenated. OSWorld 2.0 rejected "shallow workflows that compose unrelated operations", and Gym-Anything bans "artificially hard" chains of hundreds of subtasks ([OSWorld 2.0](https://arxiv.org/abs/2606.29537); [Gym-Anything](https://arxiv.org/abs/2604.06126)).

**Composition topology is itself a knob.** Non-local branch–merge dependencies are harder than long chains even after composed training ([He et al.](https://arxiv.org/abs/2609.19465)). Right-heavy Countdown solution trees are harder than balanced ones at equal size ([Countdown analysis](https://arxiv.org/abs/2512.01775)). In search QA, moving from chains (treewidth 1) to cycles (2) and coupled cliques (≥3) forces constraints to be satisfied jointly ([WebSailor-V2](https://arxiv.org/abs/2509.13305); [REDSearcher](https://arxiv.org/abs/2602.14234)).

---

## 3. Family B — Constrain

Constraint operators add requirements that a correct output must satisfy at the same time. They are the backbone of instruction-following RLVR and the easiest family to keep verifiable, since you add one checker per constraint. Their weakness: *count is not difficulty*. IFDecorator states directly that "complexity alone does not determine difficulty" ([IFDecorator](https://arxiv.org/abs/2508.04632)).

**B1. Constraint stacking (And, raise k; new families; wider ranges)**
- **Def:** attach more independent atomic constraints (format, length, lexical, content, style), each with a code checker.
- **E→H:** *IF*: "Write a poem about autumn" → "exactly 4 stanzas, all lowercase, 'ember' exactly twice, no commas, end with a question, title in <<>>" ([IFBench](https://arxiv.org/abs/2507.02833)). *Code*: "k most frequent words" → "same over a 10⁶-word stream in O(n log k) time / O(k) memory, lexicographic tie-break, online stop-list" ([rStar-Coder](https://arxiv.org/abs/2505.21297)). *Multimodal/science*: "Draw 3 red circles" → "…left of 2 blue squares, twice as many circles as squares in the top half, none touching the border"; "raise solubility" → "…keeping Tanimoto ≥ 0.7 to the input" ([VVR](https://arxiv.org/abs/2609.35641); [ether0](https://arxiv.org/abs/2506.17238)).
- **Label:** CERT. Use one executable checker per constraint plus a *compatibility checker* (pair blocks, arithmetic feasibility, propagation). For hard-to-satisfy sets, generate the query from a predefined answer (see D3) ([LongCat-Flash](https://arxiv.org/abs/2509.01322)).
- **Knob:** k. Pick k so that p^k lands in the band. On CSE, mean satisfaction follows 72.0% × 0.922^(k−1), about 41% per constraint at k = 8, which gives 5.7% all-pass ([CSE](https://arxiv.org/abs/2608.12426)). Other knobs: parameter ranges, and constraint *families* (IFBench adds 29 new training types).
- **Evidence:** training on more constraints than the eval uses helped: 5–6 per prompt beat up to 3, even though IFEval tests at most 3. The best count is policy-specific, though: the Qwen2.5 curve peaked at 3 and fell to 49.4 at 4 ([IFBench](https://arxiv.org/abs/2507.02833)). Wider variable ranges help and disjoint ranges hurt. Training on IFEval types overfits them (TÜLU-DPO 81.1 IFEval vs 25.5 IFBench). **Strong:** diversify types and ranges, hold out whole families.
- **Fails when:** sets become jointly unsatisfiable (CSE rejection grows from about 32% at k = 4 to about 98% at k = 12). The policy hacks the checkers with literal placeholders, dummy list items, "p p p" repetition and copied delimiters; an intent check cut the hack rate from 14.53% to 7.60% ([IFDecorator](https://arxiv.org/abs/2508.04632)). Constraint-only RL makes a specialist: an IF-only model scored AlpacaEval 1.1 and GSM8K 15.3 ([IFBench](https://arxiv.org/abs/2507.02833)).
- **Variant — convention inversion:** require behavior that contradicts SFT habits ("no comments, one-letter variable names, one deliberate off-by-one bug in a line marked #BUG"), checkable by AST or regex ([Inverse IFEval](https://arxiv.org/abs/2509.04292)).

**B2. Nested level ladder (L1 ⊂ L2 ⊂ … ⊂ Lk)**
- **Def:** nested prefixes of one constraint list, so the item exists at every difficulty level.
- **E→H:** "Recommend three sci-fi novels" → L5: "…published before 1970, as a Markdown table Title|Author|Year, each row ≤ 15 words, without mentioning Asimov" ([FollowBench](https://arxiv.org/abs/2310.20410); [Conifer](https://arxiv.org/abs/2404.02823)). The same structure works for rubric fact groups in web tasks, where each subset of groups is an easier task ([WebGym](https://arxiv.org/abs/2601.02439)).
- **Label:** CERT. V(L_k) = V(L_(k−1)) ∧ v(c_k). Responses at adjacent levels give free positive/negative pairs, and the first failing level is a difficulty label.
- **Knob:** level k.
- **Evidence:** Conifer's easy-to-hard multi-turn packaging beat shuffled or reversed orders by about 1–2 points ([Conifer](https://arxiv.org/abs/2404.02823)). The ladder's main value is supplying intermediate rungs (chapter 05).

**B3. Coupled and scoped constraints**
- **Def:** combine constraints that read the *same output feature* (structure, length, format), or apply constraints to selected scopes nested in the output tree.
- **E→H:** *IF*: "Include 'blue'; mention a city" → "valid JSON only; the 'summary' field 40–50 words; exactly three citations in a list; first letters of the keys spell MAP". *Scoped*: "every paragraph < 60 words" → "in section 2 only, every bullet starts with a verb; bullets under 'Risks' are numbered and each contains a percentage" ([ScopeIF](https://arxiv.org/abs/2609.32189); [IFHierBench](https://arxiv.org/abs/2607.27912)). *OR/science*: an LP whose constraints share variables across resource, time and precedence blocks ([OPT-Zero](https://arxiv.org/abs/2609.34205)).
- **Label:** CERT. Prove satisfiability with a witness response. Parse the output into a tree (headings, lists, sentences) and check each scope. Reward the normalized distance to the bound, exp(−α·d/s), rather than all-or-nothing.
- **Knob:** number of constraints sharing a feature; scope depth.
- **Evidence:** structural and ordering constraints degrade 2.0× faster than lexical ones, and pairwise failures are nearly independent (φ = +0.067) *except* through shared features ([CSE](https://arxiv.org/abs/2608.12426)). "Output JSON" is jointly unsatisfiable with 9 of 24 instructions ([Instruction Stacking Collapse](https://arxiv.org/abs/2608.02639)). On IFHierBench the best model is only marginally above 50%. OPT-Zero's structural-coupling reward beat a solve-rate reward.
- **Fails when:** coupling makes the set infeasible, so a witness is mandatory.

**B4. Constraint intersection for identification (width in search)**
- **Def:** describe the target through k constraints that are jointly, but not individually, sufficient. Some may be ranges or tool-computed.
- **E→H:** *Web*: "Which club did player X join in 2005?" → "Which player born in the early 1990s joined a club founded by someone with initial F and scored in a cup final between 2000 and 2010?" ([InfoSeek](https://arxiv.org/abs/2509.00375); [WebShaper](https://arxiv.org/abs/2507.15061)). *Tool-grounded*: "museum in city X" → "the city about two hours' drive west of [Entity A]" (Maps) ([REDSearcher](https://arxiv.org/abs/2602.14234)). *Puzzles*: enumerate the facts true of a hidden solution and prune to a minimal unique set ([ZebraLogic](https://arxiv.org/abs/2502.01100)).
- **Label:** INV or CON. Build the constraint set first. Enumerate matching entities in the KG, DB or search and require exactly one (no under-determination). Require that no proper subset already determines the answer (no over-determination). For puzzles, re-run SAT/SMT after every edit. Cache tool outputs at synthesis time.
- **Knob:** k; how much each constraint is blurred; treewidth.
- **Evidence:** about 30% of LLM-added puzzle constraints make puzzles unsolvable ([AutoLogi](https://arxiv.org/abs/2502.16906)). Over-determination is the silent failure: one leftover precise clue turns a nominally 6-hop item into a 1-hop lookup (note 07 synthesis).

**B5. Feasibility → optimality; correctness → performance**
- **Def:** replace "any valid X" with "the best X", or keep correctness as a gate and reward a continuous objective (runtime, memory, cost, utility).
- **E→H:** *Puzzles/OR*: any TSP tour → shortest tour; 2-SAT satisfiability → Min-True 2-SAT; MST → degree-constrained spanning tree ([NP-Engine / Forge](https://arxiv.org/abs/2510.16476); [FrontierSmith](https://arxiv.org/abs/2605.14445)). *Code*: a LeetCode problem at 100% pass → "beat 90% of about 107 human solutions" ([Afterburner](https://arxiv.org/abs/2505.23387)). *Kernels/repos*: correct kernel → >5% faster than eager and compile ([CUDA Agent](https://arxiv.org/abs/2602.24286)); correct repo → ≥95% of the expert's speedup ([GSO](https://arxiv.org/abs/2505.23671)). *Agents*: allocation → horizon-optimal allocation with pre-solved u* ([VHD-Play](https://arxiv.org/abs/2609.27321)).
- **Label:** CERT. Use an exact feasibility gate with a penalty, then a ratio to a heuristic, exact or pre-solved optimum (reward clip[0,1]((u − u₀)/(u* − u₀)) in VHD-Play). For performance, use hidden multi-distribution correctness inputs and synchronized, warmed-up, repeated timing with clock locking.
- **Knob:** the bar (Opt_p, Fast@x, baseline strength eager → compile → SOL).
- **Evidence:** GSO Opt_0 was 70% for Claude-4.0; Opt_0.95 was <5% ([GSO](https://arxiv.org/abs/2505.23671)). Afterburner's GRPO kept improving where SFT and DPO saturated, and pass@1 also rose (47% → 62%). CUDA Agent's milestone reward reached 96.8% faster-than-compile against 60.4% with raw speedup. **Strong:** for fully solved seeds, efficiency RL restores gradient without new problems.
- **Fails when:** a flat reward for feasibility gets hacked ([Forge](https://arxiv.org/abs/2605.08905)). Performance rewards are the most-hacked signals documented: stream timing (32.8% of CUDA-L1's early outputs, a fake 18×), lazy tensors, copying the reference, test-value hardcoding ([CUDA-L1](https://arxiv.org/abs/2507.14111); [Kevin](https://arxiv.org/abs/2507.11948)). Raw speedups are heavy-tailed, so use milestones or clipping. Correctness must be reachable before speed is rewarded.

**B6. Resource budgets (steps, turns, tokens, tool calls, submissions)**
- **Def:** keep the task and cap the resources so that only an efficient or target strategy succeeds, without naming the strategy.
- **E→H:** *Research agents*: sequential lookup of 3 entities → wide search under a fixed critical-step budget, which favors parallel sub-agents ([Kimi K2.5](https://arxiv.org/abs/2602.02276)). *Web/GUI*: step caps per rollout ([TTI](https://arxiv.org/abs/2506.07976); [WebGym](https://arxiv.org/abs/2601.02439)). *Black-box replication*: iterate against a public verifier, reward from hidden scenarios with penalties for extra submissions ([Kimi K3](https://arxiv.org/abs/2607.24653)). *Math*: couple the token budget to difficulty ([e3](https://arxiv.org/abs/2506.09026)).
- **Label:** INV. The outcome verifier is unchanged; only the budget changes.
- **Knob:** the cap, in both directions.
- **Evidence (mixed):** TTI's growing horizon (10 → 30, multiplicative 32.25 vs additive 29.50) and MAI-UI's larger budgets (+1.7/+3.8/+6.0) helped, but WebGym's *tighter* budgets raised the peak from 38.2 to 42.9 by discouraging late-recovery success ([MAI-UI](https://arxiv.org/abs/2512.22047)). Tasks whose solutions exceed the context budget give zero reward ([Tongyi DeepResearch](https://arxiv.org/abs/2510.24701)). **Moderate:** tune the budget both ways.

**B7. Answer-space and certificate hardening**
- **Def:** remove guessable or unverifiable answer formats. MCQ → open-ended; canonical exact answers; certificates instead of labels; special judges for multi-answer outputs; precision-penalized set answers.
- **E→H:** *Math*: an MCQ on the circle x² + y² − 22x − 16y + 113 = 0 → "the smallest x-coordinate has the form k − m√n; find k + m + n" (19) ([Big-Math](https://arxiv.org/abs/2502.17387); [DAPO](https://arxiv.org/abs/2503.14476)). *Logic*: SAT/UNSAT label → satisfying assignment or unsatisfiable core ([SATQuest](https://arxiv.org/abs/2509.00930)). *Code*: "print the sorted array" → "print any permutation satisfying the adjacency constraints", checked by a generated special judge ([ScaleBox](https://arxiv.org/abs/2604.27467)). *Theorems*: a single true/false label → consistency across a group of minimally edited variants ([DeepTheorem](https://arxiv.org/abs/2505.23754)).
- **Label:** INV if the mapping from the old answer is deterministic and canonical; CERT for certificates and judges. Drop multi-part and proof items; drop items that need the options to disambiguate.
- **Knob:** answer-space size; which guessing screens are applied.
- **Evidence:** 68.7% of o4-mini's SATBench errors came from a satisfiability bias, so bare labels are gameable ([SATBench](https://arxiv.org/abs/2505.14615)). DeepTheorem's group criterion lowers random accuracy to 11–17%. Labs drop prompts answered correctly without CoT within 8 guesses ([Kimi k1.5](https://arxiv.org/abs/2501.12599)) or by a tool-free model in 1 of 8 ([GLM-5](https://arxiv.org/abs/2602.15763)). 59.01% of correct special-judge solutions fail exact match, so format hardening without a special judge creates fake hardness ([ScaleBox](https://arxiv.org/abs/2604.27467)). **Strong.**
- **Fails when:** the harder format leaves most items at zero. Open-ended conversion gave zero signal for >83% of Golden Goose items, while 9-option MCQ worked ([Golden Goose](https://arxiv.org/abs/2601.22975)). Pick the format that puts most items in the band.

**B8. Verifier and rubric hardening (raise the bar without touching the prompt)**
- **Def:** make "solved" tasks hard again by tightening the check: adversarial, mutant-killing and stress tests; stronger specs; multiple input variants; harder rubric criteria.
- **E→H:** *Code*: about 10 tests that 95% of samples pass → about 35 evolved tests, and EvolveCoder pass@1 fell from 43.80 to 31.22 ([EvolveCoder](https://arxiv.org/abs/2603.12698)). *SWE*: add tests that plausible-but-wrong patches fail; 78.80 → 62.20 ([SWE-ABS](https://arxiv.org/abs/2603.00520)). *Spreadsheets/kernels*: one fixed input → 3–5 generated workbook variants, or up to 5 validated tensor shapes, so only a general solution passes ([SpreadsheetBench](https://arxiv.org/abs/2406.14991); [TritonRL](https://arxiv.org/abs/2510.17891)). *Verified code*: any spec that verifies (including `ensures true`) → a spec that rejects every mutated-output spectest ([SAFE](https://arxiv.org/abs/2410.15756); [SpecRL](https://arxiv.org/abs/2604.05820)). *Rubrics*: add criteria that split the two best responses, regenerated as the policy improves ([RubricHub](https://arxiv.org/abs/2601.08430); [EvoRubrics](https://arxiv.org/abs/2606.23038)).
- **Label:** CERT. Every new test must pass all known-correct references. Track TPR and TNR per task and keep both ≥ 0.9. Drop tests with very low pass rates (likely wrong outputs). For rubrics, check that each new criterion splits current rollouts.
- **Knob:** number and adversariality of tests; spec strength; rubric criteria.
- **Evidence:** on AtCoder 4+, test-suite precision went from 21.67 to 60.00, and TACO's tests had a false-positive rate above 90% on hard problems ([HardTests](https://arxiv.org/abs/2505.24098)). KernelBench's official check misses 16.9% of injected faults, while a kill-matrix-optimized 2-input suite detects 98.0% ([Measuring the Checker](https://arxiv.org/abs/2609.22220)). **Strong:** harden the verifier before synthesizing, because weak verifiers make tasks look easy. See chapter 04.

---

## 4. Family C — Conceal and obfuscate

Concealment keeps the solution fixed and reduces what the prompt reveals. Most of these operators are **INV**, which makes them the cheapest safe way to harden a saturated pool. The central trade-off is **obfuscation vs uniqueness**: every lab that obfuscates also checks that the answer is unique ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556); [GLM-5](https://arxiv.org/abs/2602.15763); [MiniMax-M2](https://arxiv.org/abs/2605.26494); [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725)). **Strong.**

**C1. Entity / attribute fuzzing**
- **Def:** swap exact values for looser descriptions that still identify the answer.
- **E→H:** *Web*: five strategies ([FORT](https://arxiv.org/abs/2606.12087)): category generalization (IMF → "an international financial institution"), range relaxation (1863 → "the second half of the nineteenth century"), meta-attribute description (Hannah → "a given name that is a palindrome"), arithmetic encoding (17,921 → "a five-digit prime whose digits sum to 20"), contrastive exclusion. *Tool use*: "order #W123" → "the order containing the blue shirt and the leather shoes" ([CoVe](https://arxiv.org/abs/2603.01940)). *GUI*: remove an entity's most distinctive attributes ([UI-TARS-2](https://arxiv.org/abs/2509.02544)).
- **Label:** INV, conditional on a uniqueness check after *every* fuzz. Options: KB or DB enumeration (CoVe checks the item combination is unique among the user's orders); treating a no-tool solver's wrong answers as candidate alternatives ([ASearcher](https://arxiv.org/abs/2508.07976)); discarding items where rollouts find alternatives (REDSearcher); repairing over-fuzzed items by adding attributes back ([LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725)).
- **Knob:** number of fuzzed attributes; blur radius.
- **Evidence:** in FORT's cumulative ablation, strong-agent accuracy rose from 29.0 (full controls) to 81.6 once fuzzing was also removed. Fuzzing was the most important single control.
- **Fails when:** answers become non-unique. WebSailor states its answers are not always unique ([WebSailor](https://arxiv.org/abs/2507.02592)). A filter that keeps only items a frontier model fails (4 of 4 in DeepDive) also keeps ambiguous and broken items ([DeepDive](https://arxiv.org/abs/2509.10446)).

**C2. Clue removal (long-to-short)**
- **Def:** keep the answer fixed and delete, or make indirect, the most revealing clues until a reference model's pass rate reaches the target.
- **E→H:** *Web*: drop "set an attendance record" and "died at the age of 44", and replace names with indirect descriptions ([WebExplorer](https://arxiv.org/abs/2509.06501)). *Data/agent tasks*: redact column names or counts, remove format examples, "genericize constraints, obscure the goal" ([Trading Human Curation](https://arxiv.org/abs/2606.03800)).
- **Label:** INV. Fixed answer, with a uniqueness and solvability check after each deletion.
- **Knob:** evolution rounds; number of removed clues.
- **Evidence:** five rounds cut Claude-4-Sonnet accuracy from 86.6% to 67.1% and raised tool turns from 7.9 to 9.9 ([WebExplorer](https://arxiv.org/abs/2509.06501)). Information-removal mutations dropped solve rates by 70–100 points in the gated RLVR study, which is often *too* far ([Trading Human Curation](https://arxiv.org/abs/2606.03800)). **Strong:** harden over-specified seeds by removal, not by lengthening.
- **Fails when:** questions become unanswerable. WebExplorer enforces answer invariance only by prompt.

**C3. Insight hiding (burden of discovery)**
- **Def:** find the step that makes a seed easy (a given hint, an auxiliary object, a stated strategy) and restructure the problem so that step must be discovered.
- **E→H:** *Math*: "Show 1 + … + 5 is divisible by 5" → "How many n ≤ 1000 satisfy n | 1^k + … + n^k for every odd k?" (500) ([Code2Math](https://arxiv.org/abs/2603.03202)). *Geometry*: isosceles base angles, where the proof needs the midpoint D of BC. Move D out of the statement and keep only conclusions provable in X_add but not X_raw ([AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-5); [InternGeometry](https://arxiv.org/abs/2512.10534)). *Vision*: "area of the shaded triangle, base and height labelled" → "…given only the perimeter and one angle marked" ([SynthRL](https://arxiv.org/abs/2506.02096)). *Lean*: keep statements not closed by `aesop`, `exact?` or a fixed tactic portfolio, but provable.
- **Label:** INV for SynthRL-style stem hardening (the answer is hidden from the hardener; accept only if the policy still reaches the answer in ≥ 4/16 rollouts and its pass count drops by ≥ 2). CON for formal hiding (the with-aux proof is the certificate, the failed without-aux run is the hardness certificate). For free-form math, RED: exhaustive or programmatic search plus separate solvability and difficulty critics.
- **Knob:** which ingredient is hidden; reference-solver strength.
- **Evidence:** Code2Math needs 1.56–6.55 failed rollouts per accepted problem, and its difficulty rubric agrees exactly with humans 79.3% of the time. **Moderate.**
- **Fails when:** the reference solver is miscalibrated. In HAGeo, random auxiliary points alone matched AlphaGeometry's 25/30 ([HAGeo](https://arxiv.org/abs/2512.00097)). Insight is the least automatable axis (section 18).

**C4. Implicit parameters and hidden intermediate steps**
- **Def:** hide a value, sub-call or identifier behind a reasoning or lookup hop, so the request is stated at a higher level.
- **E→H:** *IF*: "Use 5 bullets" → "as many bullets as there are primes below the number of letters in the capital of Australia; if that count is even, use the passive voice" ([ImpRIF](https://arxiv.org/abs/2602.21228)). *Tool use*: get_zipcode("Rivermist"), get_zipcode("Stonebrook"), buy_tickets(zips) → "Buy me tickets from Rivermist to Stonebrook" ([HardGen](https://arxiv.org/abs/2601.01498)). *Tool-dependence* (illustrative): "17 × 23?" → "sum of all primes below 2,000,000", with a capped proposer reward for tasks whose solutions require tool calls ([Agent0](https://arxiv.org/abs/2511.16043)). *Anti-tool shortcut*: move Boolean-expression operands into world-knowledge statements, so running Python no longer solves the task ([BBEH](https://arxiv.org/abs/2502.19187)).
- **Label:** REC. Build the dependency DAG from banks of knowledge, math and condition nodes with objective values, execute it, and check the output in code. For tools, the gold trace exists before the query is written; provenance tags show where each argument came from ([SAP](https://arxiv.org/abs/2609.06124)).
- **Knob:** number of hidden hops; node types.
- **Evidence:** ImpRIF gave +7–10 points on external IF sets. HardGen built its tool graph from the 1,204 of 2,095 tools that two small models got wrong. **Moderate.**
- **Fails when:** a hidden hop has several readings, which makes it ambiguous.

**C5. Withhold behind tools, users or partial observation**
- **Def:** keep the solved instance, but make the model *acquire* parameters or specs by querying tools, a simulated user or a hidden state.
- **E→H:** *OR/agents*: written-out knapsack → an 11-period agentic version where parameters are discoverable only through tools ([VHD-Play](https://arxiv.org/abs/2609.27321)). *Puzzles*: a Zebra puzzle with k clues behind fact queries ([ZebraArena](https://arxiv.org/abs/2603.18614)). *Diagnosis*: "which disease matches these findings?" → 12 candidate diseases and 16 tests whose outcomes come from an SAT engine ([KUMO](https://arxiv.org/abs/2504.02810)). *Code*: "compute X for array A" → query array parts through 10–256 tool calls ([CodeGym](https://arxiv.org/abs/2509.17325)); "predict the output" → "synthesize an equivalent function by querying a black box" ([CodeARC](https://arxiv.org/abs/2503.23145)). *Users*: "Book AA123 on 3 May for Alice, seat 12A…" → "Can you sort out my trip for the conference?", details released only when asked ([SynthAgent](https://arxiv.org/abs/2601.22511); [EnvFactory](https://arxiv.org/abs/2605.18703)).
- **Label:** INV. Pre-solve, then grade arithmetically or by end state. Replay reference policies (optimal, greedy, idle) through the generated environment. Fix the simulator's knowledge per task and make the target end state a deterministic function of it, so guessing without asking fails ([Qwen-CUA](https://arxiv.org/abs/2608.02352)).
- **Knob:** fraction of information withheld; query budget.
- **Evidence:** base scores 0.962 written out, 0.231 informed-agentic, 0.204 agentic. Training took agentic to 0.815, with held-out family gains of +0.56 ([VHD-Play](https://arxiv.org/abs/2609.27321)). With the full SWE-bench issue a model scored 43.8%; with Error Info/Reproduction withheld, 23.7%; with learned clarification, 36.8% ([CLARITI](https://arxiv.org/abs/2604.14624)). **Strong:** this is the biggest hidden-difficulty lever for agents.
- **Fails when:** the simulator leaks the hidden spec or is sycophantic. Audit it and evaluate under simulator shift ([SWEET-RL](https://arxiv.org/abs/2503.15478); chapter 04).

**C6. Spec degradation (symptom-only issues, instruction abstraction)**
- **Def:** keep the target state and hidden tests, and make the statement less informative: symptom-only issue, tests-only spec, terse goal, no procedure.
- **E→H:** *SWE*: an issue that names the function and the fix → "export sometimes drops the last row" ([R2E-Gym](https://arxiv.org/abs/2504.07164); [SWE-Hub](https://arxiv.org/abs/2603.00575)). *Spreadsheets*: "In Sheet1 compute BMI = 10000·Wt/(Ht·Ht) into D2:D7 … scatter-plot A2:A7 vs D2:D7" → "Add BMI and chart it against age" ([WTM](https://arxiv.org/abs/2608.07873)). *Enterprise web*: L2 (step list in the ticket) → L3 ("onboard a new employee", procedure in the knowledge base) ([WorkArena++](https://arxiv.org/abs/2407.05291)). *Terminal*: "set PORT=8080" → "the service returns 502; correlate the logs, fix it, write an evidence bundle" ([RST](https://arxiv.org/abs/2608.05466)).
- **Label:** INV. Tests are unchanged. Check that the degraded spec still determines behavior: a reference solver must solve it from the spec alone, and every graded property must be discoverable. RST's contract-validity gate plus an instruction audit cut weakly grounded tasks from 32.8% to 1.2%.
- **Knob:** specificity level (WTM L1 → L3 falls monotonically).
- **Evidence:** GPT-4o fell from 3.0% (L2) to 0% (L3) on WorkArena++. SWE-smith notes that issue text strongly shifts difficulty ([SWE-smith](https://arxiv.org/abs/2504.21798)).
- **Fails when:** a missing contract makes the task *ambiguous*. An under-specified statement gave 0% because of ambiguity, not difficulty ([FrogNano](https://arxiv.org/abs/2609.07925)). Keep the detailed version to tell specification gaps from capability gaps.

**C7. Retrieval obstruction (lexical-overlap removal, indirection, dispersion, source removal)**
- **Def:** make the answer-bearing evidence hard to *find*, not hard to use.
- **E→H:** *Long context*: needle "Yuki has been to Dresden" → "Yuki lives next to the Semper Opera House" for the query "Who has been to Dresden?" ([NoLiMa](https://arxiv.org/abs/2502.05167)); a plain question → UUID pointer chains where only one resolves to the real question ([LoongRL](https://arxiv.org/abs/2510.19363)). *Web*: all clues in one infobox → clues spread across a news archive, a statistics table and a fact derived across two pages (Minimum Source Dispersion) ([REDSearcher](https://arxiv.org/abs/2602.14234)). *Deep research*: delete the source page of a QA from the corpus, or answer only from a fictional SQL-backed index ([LiteResearcher](https://arxiv.org/abs/2604.17931); [Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)).
- **Label:** INV. Link needles through a verified relation (Wikidata IS-A) and filter the haystack for other satisfying spans. Store source attribution per fact. After source masking, re-verify reachability (solver pass 1/8–7/8).
- **Knob:** number of latent hops; number of sources; decoys.
- **Evidence:** Llama 3.3 70B scored 98.5 at 32K with literal overlap, 56.2 with one latent hop and 25.9 with two ([NoLiMa](https://arxiv.org/abs/2502.05167)). Snippets that withhold answers raised page-extraction calls from 2.5 to 4.0, while real-search RL lowered them to 1.5 ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)). **Strong:** treat any generator whose query contains needle keywords as producing easy tasks.

**C8. Equivalence-preserving substitution and renaming (pattern hiding)**
- **Def:** rewrite so the fact is unchanged but its familiar pattern or names are hidden: variable substitutions, library rewrites, opaque function names, namespace remapping.
- **E→H:** *Inequalities*: a + b + c ≥ 3∛(abc) → x/y + y/z + z/x ≥ 3 via a = x/y… ([Alchemy](https://arxiv.org/abs/2410.15748); Ineq-Comp Type II x ↦ x²). *Code composition*: visible function definitions → opaque names such as func_16 ([f(g(x))](https://arxiv.org/abs/2509.25123)). *SWE*: a namespace-remapped repo also serves as a contamination check ([SWE-Hub](https://arxiv.org/abs/2603.00575)). *Rule induction*: an isomorphic, renamed twin; credit only outputs correct on both ([IPT](https://arxiv.org/abs/2604.15149)).
- **Label:** INV or REC (the seed proof composed with the substitution; carry domain conditions into hypotheses).
- **Knob:** depth of rewriting.
- **Evidence:** Ineq-Comp Type II dropped DeepSeek-Prover-V2-7B from 66.2% to 42.1% ([Ineq-Comp](https://arxiv.org/abs/2505.12680)). Extensional-only verifiers let RLVR models enumerate instance labels instead of inducing rules, and shortcuts rose from 40 at levels 1–10 to 458 at levels 11–20. Isomorphic twins remove them ([IPT](https://arxiv.org/abs/2604.15149)). **Moderate:** mostly an anti-memorization and anti-shortcut operator; pair it with a real difficulty operator.

---

## 5. Family D — Invert

Inversion reuses one verified artifact for the harder direction of the same relation: forward evaluation becomes equation solving, abduction, synthesis or proof. It is also the main **correctness** device of the whole taxonomy. Build the answer or certificate first, and the generator never has to solve the hard direction itself.

**D1. Given ↔ unknown swap (backward reasoning)**
- **Def:** mask a given, reveal the original answer, ask for the given; or fix a target outcome and ask for the parameter that achieves it.
- **E→H:** *Math*: "5 packs × 4 lb × $5.50/lb?" ($110) → "…buys x packs, pays $110; find x" ([MetaMath](https://arxiv.org/abs/2309.12284)). *Competition*: "trailing zeros of 100!" (24) → "smallest n such that n! ends in exactly 24 zeros" (100) ([QbQ](https://arxiv.org/abs/2608.01522)). *Physics*: "2 kg block: v at 3 s?" → "what mass gives 5 m/s at 3 s?" ([Sim2Reason](https://arxiv.org/abs/2604.11805)).
- **Label:** CON (the masked value is known). **Check uniqueness**: "n! ends in exactly 24 zeros" has five solutions (100–104), so ask for the smallest or for a count. Discard nonlinear blow-ups, as GSM-Infinite discards quadratic reverse cases ([GSM-Infinite](https://arxiv.org/abs/2502.05252)).
- **Evidence:** adding 20K samples to 80K answer-augmented data gave +2.3 (FOBAR inversion) and +2.6 (self-verification inversion) GSM8K points, against +0.4 for rephrasing ([MetaMath](https://arxiv.org/abs/2309.12284)). Backward rewrites turn an unsolved word problem with 5 quantities into up to 10 inverse items with known answers ([backward rewrites](https://arxiv.org/abs/2605.28388)). **Caveat (Moderate):** in Sim2Reason's 3B ablation, reverse (5.84) and symbolic (7.46) training transferred to IPhO far less than forward numeric questions (13.15). Treat inversion as extra variety, not a replacement.

**D2. Program-direction inversion (deduction → abduction → induction)**
- **Def:** from one verified (function, input, output) triple, pose output prediction, input search, or program synthesis from partial I/O.
- **E→H:** *Code*: "What does `f(x) = sorted(set(x))[-2]` return for [3, 1, 3, 2]?" (2) → "give a length-5 x with f(x) = 7" → "here are 3 of 6 I/O pairs and a hint; write f", graded on the hidden pairs ([AZR](https://arxiv.org/abs/2505.03335); [CodeI/O](https://arxiv.org/abs/2502.07316)). *Logic*: forward state tracking → abductive "which event is missing?" ([Depth × Complexity](https://arxiv.org/abs/2605.26934)). *Calculus*: derivative → antiderivative of a random F′, checked by differentiation ([RLVE](https://arxiv.org/abs/2511.07317)).
- **Label:** CERT. Execute: accept any abduction input that reproduces the output; grade induction on held-out pairs. Require determinism (run twice), sandboxing and time limits.
- **Evidence:** AZR-style self-play helps pass@k at small k, but the base model is better at large k (not significant), and entropy still collapses (note 09). Inversion sharpens more than it expands; pair it with J3/J4.
- **Fails when:** the abduction target has trivial preimages (constant functions), or the verifier checks only outputs, which lets models enumerate instances (see C8).

**D3. Answer-first / planted construction**
- **Def:** sample the answer, certificate or artifact first, then derive a task it provably solves.
- **E→H:** *Puzzles*: a random solved 9×9 grid → a masked puzzle; a planted Hamiltonian path plus random edges ([RLVE](https://arxiv.org/abs/2511.07317); [SATURN](https://arxiv.org/abs/2505.16368)). *OR*: a 2-variable LP → a 40×40 LP built from an interior primal x* and a dual satisfying KKT; solve rate fell from 100% to 8.3% ([A²utoLPBench](https://arxiv.org/abs/2607.02141)). *Inequalities*: random p₁, p₂, p₃ in 4 variables with F = Σpᵢ² + g·q² expanded; "prove F ≥ 0 given g ≥ 0" ([NSPI](https://arxiv.org/abs/2605.15445)). *SQL*: write a 3-table JOIN/GROUP BY, execute, then ask for the NL question ([SQL-Zero](https://arxiv.org/abs/2609.04697); [OmniSQL](https://arxiv.org/abs/2503.02240)). *Tools*: sample an executable 6-tool chain, then write the request that needs it ([LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725)). *IF*: read constraints off a good response ("Explain photosynthesis in 150–200 words across exactly 3 paragraphs…") ([Crab](https://arxiv.org/abs/2410.24175)).
- **Label:** CON, **graded as CERT**. Check the model's answer against the constraints, not against the planted answer, because alternative optima and solutions exist. Reject degenerate artifacts (empty SQL results, NL that leaks the SQL).
- **Evidence:** question-first pipelines often miss their target difficulty, and the error grows with the target step count ([SAGE](https://arxiv.org/abs/2601.18202)). **Strong:** construct the answer first for every RL-bound family.
- **Fails when:** back-translated constraints never exceed the source response's difficulty and can be satisfied by copy-paste ([UNSPECIFIC](https://arxiv.org/abs/2608.09154)). Crab did not help FollowBench L3–L5 relative to Conifer. Use back-translation for volume and warm-up, then switch to policy-adaptive appending (J5).

**D4. Explore or execute first, describe later (reverse task synthesis)**
- **Def:** run exploration or a real tool chain, then write the task the trajectory solves, or ask a question strictly entailed by the trace.
- **E→H:** *GUI/web*: a hand-written "open settings" → a discovered multi-step interaction lifted to "configure application settings for X" ([OS-Genesis](https://arxiv.org/abs/2412.19723); [Go-Browse](https://arxiv.org/abs/2506.03533)). *Tools*: a one-call lookup → a 5-hop question across heterogeneous tools answerable only from the combined trace ([DIVE](https://arxiv.org/abs/2603.11076)).
- **Label:** CON. The recorded end state or trace is gold. Re-execute from reset and unit-test tools for determinism.
- **Evidence and caveat (Moderate):** witnessed solutions guarantee solvability, but tasks are biased toward the explorer's competence (note 17). Treat D4 as a *correctness* operator and layer composition, hint stripping or perturbation on top.

**D5. Checker-writing inversion**
- **Def:** swap roles so the model produces the test, spec or verdict rather than the solution.
- **E→H:** *SWE*: "fix the bug given the failing test" → "write a test that fails before the patch and passes after" ([MiniMax-M2](https://arxiv.org/abs/2605.26494)). *Code*: "write tests that separate your wrong samples from your right ones" ([CURE](https://arxiv.org/abs/2506.03136)). *Verified code*: write the strongest admissible postcondition ([SpecRL](https://arxiv.org/abs/2604.05820)).
- **Label:** CERT. Run the produced test on pre- and post-patch code; reward tests that separate correct from incorrect code and penalize tests that fail correct code. Specs need soundness and completeness checks (B8).

**D6. Prove-or-disprove, negation pairing, answer enumeration**
- **Def:** pose each statement with its negation, add negations of refuted conjectures, or turn "show X works" into "find all X".
- **E→H:** *Formal*: "n² + n is even" → "Decide: n² + n + 41 is prime for all n ≥ 0" (false at n = 40) ([DeepSeek-Prover](https://arxiv.org/abs/2405.14333); [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613)). *Functional equations*: "show f(x) = x works" → "find all f", with hundreds of candidates proposed and formally refuted ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y)). *Proof → answer*: "prove a² + b² + c² ≥ ab + bc + ca" → "find the largest C such that a² + b² + c² ≥ C(ab + bc + ca)" (C = 1) ([IneqMath](https://arxiv.org/abs/2506.07927)).
- **Label:** CERT. The kernel checks both directions. Always run a vacuity check (prove `False` from the hypotheses).
- **Evidence:** AlphaProof trains on about 80M formal statements from about 1M problems, because every well-typed statement is a valid prove-or-disprove task even when it misformalizes the source. Answer-only reformulation invites shortcuts: IneqMath shows up to a 65.5% drop from answer-only to step-checked accuracy.

---

## 6. Family E — Scale

Scaling turns an integer knob on a generator. It gives the smoothest, most controllable difficulty curves, making it the natural substrate for adaptive curricula. GSM-Infinite shows a sigmoid decline with operation count, and TRACE reports r ≈ −0.96 between its difficulty metric and accuracy ([GSM-Infinite](https://arxiv.org/abs/2502.05252); [TRACE](https://arxiv.org/abs/2607.04784)). It also adds tedium most easily. **Strong:** combine scaling with a controller and a length audit.

| Operator | E→H (≥2 domains) | Label / verifier | Knob and controller | Evidence | Fails when |
|---|---|---|---|---|---|
| **E1 Instance size** | Sort 5 numbers → about 3·1.1^d; 4×4 Sudoku → 9×9; TSP 10–20 → 45–55 cities ([Reasoning Gym](https://arxiv.org/abs/2505.24760); [RLVE](https://arxiv.org/abs/2511.07317)). Code: n ≤ 100 (O(n²) passes) → n ≤ 2·10⁵ with worst-case structures ([rStar-Coder](https://arxiv.org/abs/2505.21297)). Kernels: one shape → larger workloads ([KernelBench](https://arxiv.org/abs/2502.10517)) | REC. Programmatic checker, not a stored answer. Partial credit (x/N)^10 for long outputs. Time limits relative to the reference | d. RLVE promotes at τ_acc = 0.9 with window 4. SCALER uses a proportional controller toward a target accuracy ([SCALER](https://arxiv.org/abs/2601.04809)) | RLVE: +3.37 with 400 adaptive environments vs +0.49 from >3× more compute on the original data (saturated 1.5B) | Size only adds length ([Illusion of Thinking](https://arxiv.org/abs/2506.06941)). Output overflows: answer 15-disk Hanoi with a program or a count, not a move list ([Lawsen comment](https://arxiv.org/abs/2506.09250)). Some instances are unsolvable (River Crossing b = 3, N ≥ 6) |
| **E2 Depth / op count / topology** | Math: 1-op story → 12-op DAG with shared intermediates; depth +4 took Claude-3-Opus from about 95% to 41.0% ([DARG](https://arxiv.org/abs/2406.17271); [iGSM](https://arxiv.org/abs/2407.20311)). Logic: depth-2 implication → depth 12 with ∨-elimination and ∀-instantiation ([ScaleLogic](https://arxiv.org/abs/2605.06638)). Vision: one-step triangle sum → four-step angle chase; base pass 72.8% → 41.3% ([TRON](https://arxiv.org/abs/2606.01599)) | REC. Compute from the graph (mod-p arithmetic avoids big numbers). Require a code-writing solver to reproduce the label from the rendered text (DARG) | Depth, width, op count, operator types, tree shape | Depth extrapolates only about 3×; adding operator types (expressiveness) transfers best (ScaleLogic) | Rendering drift; unnatural stories. Hash solution templates to separate train and test |
| **E3 Horizon** | GSM chains h = 1 → 5 ([h1](https://arxiv.org/abs/2510.07312)). Web agents: step cap 10 → 30 ([TTI](https://arxiv.org/abs/2506.07976)). Sudoku: 5 → 40+ empty cells ([Horizon-length study](https://arxiv.org/abs/2605.02572)) | INV or REC. Unchanged outcome verifier | Staged h; multiplicative schedules | Only staged curricula give long-horizon gains (h1) | Horizon-only scaling destabilizes RL. Reduce the horizon first (macro-actions, verifiable subgoals), then lengthen. A spike in max-length responses is the collapse alarm (Horizon-length study). Long horizons amplify checker false positives |
| **E4 Context length** | Math: a 150-token problem → premises dispersed as passages in a 32–64K narrative ([LongReason](https://arxiv.org/abs/2501.15089)). Long context: 5 list operations → 20 relevant operations inside 128K of provably do-nothing padding ([Michelangelo](https://arxiv.org/abs/2409.12640)). QA: 2 gold paragraphs → 3.5M-token stream ([MemAgent](https://arxiv.org/abs/2507.02259)) | INV. Keep the seed gold. Apply no-context filters (MemAgent removed about 50% as closed-book answerable) | Tokens; separately, number of relevant operations | Length generalizes cheaply: training at 16K handled 128K ([LoongRL](https://arxiv.org/abs/2510.19363)). **Strong:** spend budget on hops and distractor quality, not raw length | Random padding is removed easily by retrieval (use F2) |
| **E5 Entities / world / state size** | Multi-hop: "Alice's mother?" in a 50-person universe → "hobby of the second cousin of Alice's friend" in 3K+ people ([PhantomWiki](https://arxiv.org/abs/2502.20377)). Tools: one-table DB → about 18 tables and 35 tools ([AWM](https://arxiv.org/abs/2602.10090)). Terminal: one CSV → multi-workbook, SQLite and JSON reconciliation ([Terminal-Universe](https://arxiv.org/abs/2609.04148)) | REC from a structured world model. Give the verifier privileged access ([Gym-Anything](https://arxiv.org/abs/2604.06126)). Single-source ablations prove every source is needed | Entities, tables, files, sheets | Cross-workspace tasks cut teacher pass@1 from 72.3% to 49.2% and added +2.0 on TB2.1 when mixed in | LLM simulators collapse as simultaneous state changes grow; keep state in code ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)) |
| **E6 Search space / solver effort** | Zebra puzzles with 0 Z3 conflicts → >20 ([ZebraLogic](https://arxiv.org/abs/2502.01100)). CNF chosen by D(n,k,l) ([SATURN](https://arxiv.org/abs/2505.16368)) or by decision/conflict/propagation counts ([SATQuest](https://arxiv.org/abs/2509.00930)). NP-complete generators ([NPPC](https://arxiv.org/abs/2504.11239)) | CERT. Certificates checked in polynomial time | Solver effort, not surface size | SATURN's metric regressed on pass rate with R² ≈ 0.71 | Metric not regressed on *your* policy |

**Controllers matter as much as knobs.** In InternGeometry's controlled ablation (IMO geometry, out of 50), easy-only data scored 29, hard-only 24, unscheduled 38 and controller-scheduled 44 ([InternGeometry](https://arxiv.org/abs/2512.10534)). Even an oracle static range covering all adaptive levels lost to per-environment adaptation (RLVE). Frontier Learning's regret buffer *with mutation* beat PLR with exploration (62.5 vs 33.5) ([Frontier Learning](https://arxiv.org/abs/2609.35426)). **Strong.** Controller details are in chapter 05.

**Self-labels break at scale.** When verifiers fail at large sizes and you fall back on majority vote, filter by ≥ 4/5 agreement across seed-varied models. This raised 5×6 multiplication label accuracy from 31% to 93.3% ([Self-Improving Transformers](https://arxiv.org/abs/2502.01612)).

---

## 7. Family F — Perturb and distract

Perturbation operators add material that must be ignored, resisted or noticed, or make the smallest edit that invalidates a memorized method. Most are INV. The ones that change the answer (F3, F4) need REC via a symbolic program.

**F1. Irrelevant-information injection (NoOp, noise nodes)**
- **E→H:** *Math*: "Oliver picks 44 kiwis on Friday, 58 on Saturday, double Friday's on Sunday" (190) → "…but five of them were a bit smaller than average" (still 190) ([GSM-Symbolic](https://arxiv.org/abs/2410.05229)). *Code*: a target function hidden among 100 irrelevant helpers in 10 files ([CHASE](https://arxiv.org/abs/2502.14678)). *Geometry*: extra premises that are never used in the closure ([TrustGeoGen](https://arxiv.org/abs/2504.15780)).
- **Label:** INV. Attach noise only with edges pointing *away* from the query's ancestor set (GSM-Infinite's spider topology), then re-solve to confirm the answer is unchanged.
- **Evidence:** NoOp caused drops of up to 65%. **Caveat:** a re-analysis found only 8 of 20 open-weight models with significant variant effects under mixed-effects models, and the variants' integers were shifted larger ([GSM-Symbolic re-evaluation](https://arxiv.org/abs/2605.28700)). Control for distribution shift before crediting an operator.
- **Fails when:** a clause creates genuine ambiguity (models *do* subtract the 5, and some readers would too). Screen for it.

**F2. Hard distractors, hard negatives, candidate expansion**
- **E→H:** *Multi-hop*: 2 gold paragraphs + random Wikipedia → 18 *positive* distractors (gold paragraphs of sibling questions) ([MuSiQue](https://arxiv.org/abs/2108.00573)), or pages an agent opened but did not cite ([LongTraceRL](https://arxiv.org/abs/2605.31584)). *MCQ*: 4 → 10 options ([MMLU-Pro](https://arxiv.org/abs/2406.01574)); options from genetically related rules ([VisualSphinx](https://arxiv.org/abs/2505.23977)). *Logic*: binary provable/unprovable → 4+ candidates, each false one a single corrupted axiom ([ScaleLogic](https://arxiv.org/abs/2605.06638)). *Tools*: 2 relevant tools → dozens including look-alikes ([COVERT](https://arxiv.org/abs/2604.09813)). *GUI*: near-duplicate noise expenses the checker requires to stay untouched ([AndroidWorld](https://arxiv.org/abs/2405.14573)). *SQL*: distractor tables from related domains ([Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315)).
- **Label:** INV/CON. Distractors must provably fail (built knowing the true answer, from rule-violating scripts). **Run a false-negative sweep:** MMLU-Pro removed 1,953 correct-but-labelled-wrong options from its MMLU part.
- **Evidence:** distractor quality beats quantity. In LongTraceRL's controlled comparison, DocQA and KeyChain data *lowered* one base model's average (40.6 and 40.1 vs 42.7), while agent-mined distractor data raised it. **Moderate:** validate distractor families per base model.

**F3. Answer-changing minimal edit (hard perturbation)**
- **E→H:** *Math*: "minimize x² − 6x + 13 over reals" (4) → "…over integers not divisible by 3" (5) ([MATH-Perturb](https://arxiv.org/abs/2502.06453)). *Code*: EvoEval Subtle edits ([EvoEval](https://arxiv.org/abs/2403.19114)). *Logic*: replace one Knights-and-Knaves statement; the new solution must differ ([K&K](https://arxiv.org/abs/2410.23123)). *Competition*: Putnam 2011 A1 with 2011 → 4680, answer 10053 → 23398 ([Putnam-AXIOM](https://arxiv.org/abs/2508.08292)).
- **Label:** REC when the seed is a script (functional variation), otherwise RED. Require the new answer to differ from the original, which also detects memorization, and confirm with a CAS, program or independent solvers.
- **Knob:** which assumption is edited; number of variants for worst-case scoring (DynaMath: worst-case accuracy is at most about 50% of average ([DynaMath](https://arxiv.org/abs/2411.00836))).
- **Evidence:** o1-mini −16.49% on MATH-P-Hard; EvoEval Subtle −24.0% at roughly the same difficulty; o1-preview −19.6 points on Putnam variations. Most MATH-P-Hard errors *ignore* the modified assumption rather than copying the old answer. **Moderate:** contrastive seed/variant pairs reward checking whether a method applies.

**F4. Counterfactual rules and world randomization**
- **E→H:** *Arithmetic*: base-10 → base-9 ([Reasoning or Reciting](https://arxiv.org/abs/2307.02477)); altered number and operator semantics ([MatheMagic](https://arxiv.org/abs/2510.05962)). *Logic*: knights lie instead of telling the truth ([K&K](https://arxiv.org/abs/2410.23123)); Sudoku variants with unusual interacting rules, where SOTA solves <15% unaided ([Sudoku-Bench](https://arxiv.org/abs/2505.16135)). *Planning*: standard Blocksworld → a fresh PDDL domain ([Reasoning Core](https://arxiv.org/abs/2603.02208)). *Agents*: displayed intensities inverted, with a legend claiming "0 = Critical" ([AutoEnv](https://arxiv.org/abs/2511.19304)). *Reasoning QA*: add a predicate that revises a stated fact ([RE-IMAGINE](https://arxiv.org/abs/2506.15455)).
- **Label:** REC. Represent the task as a symbolic program, apply the mutation in code, execute. For random worlds, discard those where an external planner times out.
- **Why use it:** priors actively mislead, so this targets the "method change" axis without an LLM judging novelty.

**F5. Environment noise and runtime perturbation**
- **E→H:** *Tools*: tau2-retail with perfect tools → 40% failing calls including silent ±10–25% value corruption, primary path blocked ([BENCH2ROBUST](https://arxiv.org/abs/2608.11977)); `list_tickets` returns 10 per page, one call times out, the batch close partially fails ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)). *Mobile GUI*: "turn on dark mode" + a battery-saver pop-up at step 2 ([AnTrap](https://arxiv.org/abs/2608.24099)). *Data*: the same revenue question on a copy with 3% duplicates, three date formats and amounts in cents ([SandMLE](https://arxiv.org/abs/2604.04872); [RST](https://arxiv.org/abs/2608.05466)). *Users*: a fully specified user → one who discloses only when asked ([LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725)).
- **Label:** INV. Verify the final state against the uncorrupted state. Noise must "not invalidate task solvability": re-verify after injection (AnTrap: 91% passed a 3-criterion audit). Assign each episode a retry/switch/stop solvability scenario, and cap the injection budget.
- **Evidence:** 69 of 70 model–benchmark pairs degraded, by up to 46.7 pp (BENCH2ROBUST). COVERT's oracle-preserving perturbation RL gave +3.4 BFCL and +6.3 ACEBench on Qwen2.5-14B ([COVERT](https://arxiv.org/abs/2604.09813)). LongCat's noise curriculum improved noisy benchmarks with no cost on clean ones. **Moderate.**

**F6. Adversarial injection (prompt injection, privilege conflicts, planted partner errors)**
- **E→H:** *Email agent*: "summarize my emails" → the same inbox with an "important message" injection; GPT-4o utility 69%, targeted ASR 57.7% ([AgentDojo](https://arxiv.org/abs/2406.13352)). *IF*: "answer only in French" + a conflicting lower-privilege message ([IH-Challenge](https://arxiv.org/abs/2603.10521)). *Debate*: a partner confidently argues a known-wrong answer ([MAPoRL + Coral](https://arxiv.org/abs/2502.18439)).
- **Label:** INV. Deterministic utility and security functions on the final state. Reward = completion AND resistance, so "refuse everything" never pays. Restrict the attacker to designated slots.
- **Knob:** attacker strength (a fixed "ignore previous" template at 5.41% ASR on GPT-4o → defense-aware GCG at >50% ASR against all 8 defenses → RL-trained attackers) ([Adaptive Attacks Break Defenses](https://arxiv.org/abs/2503.00061); [GPT-Red](https://arxiv.org/abs/2607.26115)).
- **Evidence:** GPT-5-Mini's instruction-hierarchy robustness went from 84.1% to 94.1% after training. **Fails when:** injections always sit in one position. SecAlign's always-at-end injections taught "ignore the last instruction" and crushed agentic utility to 6.2%; randomizing the position restored +69.0 pp ([Meta-SecAlign](https://arxiv.org/abs/2507.02735)). Capability does not buy robustness, so this family saturates slowly.

**F7. Infeasible, unanswerable and impossible twins**
- **E→H:** *Multi-hop*: pair each item with a twin whose context lacks one required fact; score the pair jointly ([MuSiQue](https://arxiv.org/abs/2108.00573)). *Math*: remove a key premise; gold becomes "insufficient information" ([Hallucination Tax / SUM](https://arxiv.org/abs/2505.13988)). *GUI*: "set an alarm for 7 AM using the Smart-Wake feature", which does not exist ([ZeroGUI](https://arxiv.org/abs/2505.23762)). *Code*: one test contradicts the spec, and the right action is to flag it ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). *Multi-turn*: cut the shards after 3 of 5; gold is "abstain" ([RLAAR](https://arxiv.org/abs/2510.18731)). *Planning*: delete exactly one needed fact; the model must ask the one question that restores solvability ([QuestBench](https://arxiv.org/abs/2503.22674)).
- **Label:** CON. Make infeasibility true by construction (remove the entity, then write the instruction). Certify necessity by solver or ablation. For math, confirm several consistent completions exist.
- **Evidence:** a 10% mix of unanswerable problems restored refusal with minimal accuracy cost (SUM). Removing infeasible tasks dropped ZeroGUI's infeasible-subset success from 41.3 to 22.1. RLAAR raised Qwen3-8B's abstain score from 33.5% to 73.4%. Current models solve QuestBench's one-missing-variable items 40–50% of the time. GPT-5 cheated on 54–76% of impossible SWE variants (ImpossibleBench), so these twins double as hack detectors.
- **Fails when:** the ratio is too high and the policy learns to refuse or ask by default. Cap it (RLAAR m = 0.1; BENCH2ROBUST 25% unsolvable) and add an explicit positive reward, since stock evaluators give none.

---

## 8. Family G — Abstract and generalize

These operators lift a single instance to a family and then either move to a harder region of the family or demand the general object itself. They are the cheapest way to get unbounded, exactly labelled variants from easy seeds. EFAGen found GPT-4o-failing variants even for Level-1 MATH seeds that GPT-4o had solved ([EFAGen](https://arxiv.org/abs/2504.09763)). **Strong:** before writing new problems, lift and resample existing ones.

| Operator | Def | E→H | Label / verifier | Evidence / fails when |
|---|---|---|---|---|
| **G1 Parametric lifting (variabilization)** | Turn the seed into `solve(params)` or `sampler(seed, difficulty)` and sample hard regions | *Math*: a Putnam problem with 2011 → 4680 ([Putnam-AXIOM](https://arxiv.org/abs/2508.08292)); 10 program-generated variants per seed ([DynaMath](https://arxiv.org/abs/2411.00836)). *GUI*: change date ranges, thresholds and sort/filter criteria (L1), then core objects (L2) ([MAI-UI](https://arxiv.org/abs/2512.22047)). *ARC*: 3 demonstrations → per-task generators sampled at diff ∈ [0.8, 1.0] ([RE-ARC](https://arxiv.org/abs/2404.07353)). *Code*: one sample input → controller-chosen scale ([SCALER](https://arxiv.org/abs/2601.04809)) | REC. Unit tests on the lifted program (`matches_original`, `single_valued`, `has_dof`). At least two reference solutions must agree across scales. Exclude seeds whose solution depends on the specific constant | Parameters alone move pass rates: AndroidWorld seeds gave 27.6, 26.3 and 33.2 for the same agent, so evaluate over several seeds ([AndroidWorld](https://arxiv.org/abs/2405.14573)). Score seeds by worst case over variants |
| **G2 Generalize-then-specialize / parametrize-and-sum / change the queried quantity** | Move to a regime where the same lemma applies differently, aggregate over a parameter, or ask for a different unknown in the same configuration | *Math*: "trailing zeros of 100!" (24) → "…in base 12" (48) or "Σ_{n=1}^{50} Z(n!)" (262); triangle 13-14-15: area → inradius ([QbQ](https://arxiv.org/abs/2608.01522)). *Formal*: "3 \| n³ − n" → "p \| n^p − n for all primes p"; two-variable Cauchy-Schwarz → weighted n-variable with p ≥ 1 ([AlphaProof](https://www.nature.com/articles/s41586-025-09833-y); [LEGO-Prover](https://arxiv.org/abs/2310.00656)) | REC for programs; CERT for formal variants (every variant labelled by an actual proof or disproof, never by the generator). QbQ requires the variant's answer to differ from the parent's and disallows number-only edits | QbQ took Qwen2.5-Math-7B from 5.6% to 16.5% AIME pass@1 over 20 rounds; static augmentation capped at 12.5%. AlphaProof's test-time variant RL adds about 15 points but costs 50–500 TPU-days per problem |
| **G3 Demand the general object** | Ask for a formula, program, rule or proof that must hold across instances, and grade it on several | *Spreadsheets*: one checked cell → correct on 3–5 generated workbook variants ([SpreadsheetBench](https://arxiv.org/abs/2406.14991)). *Physics*: "v at 3 s?" → "give v(t)" ([Sim2Reason](https://arxiv.org/abs/2604.11805)). *Puzzles*: a 15-disk Hanoi move list → a generating program or the move count ([Lawsen comment](https://arxiv.org/abs/2506.09250)). *Induction*: a rule checked on its instances → the same rule on a renamed isomorphic twin ([IPT](https://arxiv.org/abs/2604.15149)) | REC per instance or CERT (symbolic equivalence) | Blocks hard-coding. **Caveat:** symbolic questions transferred less than forward numeric ones in Sim2Reason (7.46 vs 13.15) |

---

## 9. Family H — Upgrade the task type

Type upgrades keep the underlying instance and move it up a ladder of task types. Many give very large measured drops. Their evidence is uneven: several ladders are evaluations, not training studies.

**H1. Decision → search → optimization.** Run the same instance up the ladder: SAT decision → satisfying assignment → MaxSAT → minimal correction set → minimal unsatisfiable core on the *same* CNF ([SATQuest](https://arxiv.org/abs/2509.00930)). Closed-ended → open-ended: 2-SAT → Min-True 2-SAT, bipartite MIS → general MIS ([FrontierSmith](https://arxiv.org/abs/2605.14445)). An LP with 3 independent constraints → a MIP with coupled blocks ([OPT-Zero](https://arxiv.org/abs/2609.34205)). *Label:* CERT: reward certificates (assignments, cores) and objective ratios behind a feasibility gate; never reward bare SAT/UNSAT labels. For open-ended problems, use a feasibility checker plus a baseline-normalized [0,1] scorer. See B5 for the reward-shape evidence.

**H2. Compute → prove; code → verified code.** "Prove a² + b² + c² ≥ ab + bc + ca" ↔ "find the optimal C" (D6). *Verified code* (spec lifting): an APPS/TACO function passing 10 tests → a Dafny method with requires/ensures, loop invariants and termination proofs. The lift succeeds on 47% of EASY seeds and about 20% of HARD seeds ([ATLAS](https://arxiv.org/abs/2512.10173)). The *stage and language ladder* keeps the problem fixed: NL→code 92.18% → end-to-end verified 5.29% ([VeriContest](https://arxiv.org/abs/2605.08553)); Dafny 40.3% → Lean 7.8% ([AlgoVeri](https://arxiv.org/abs/2602.09464)). *Label:* CERT (the verifier), plus soundness lemmas (the spec accepts reference I/O), completeness checks by mutation, and blocks on `assume(false)`, `sorry`, axioms and `external_body`. **Fails when:** sound solution checkers move the exploit into the task definition. `assume(false)` spread to every program in AlphaVerus, verification-only rewards produced progressively weaker specs, and about 9% of Vericoding successes had too-weak specs ([AlphaVerus](https://arxiv.org/abs/2412.06176); [Re:Form](https://arxiv.org/abs/2507.16331); [Vericoding](https://arxiv.org/abs/2509.22908)). Spend as much on spec hygiene as on proof search.

**H3. Closed → open-ended; MCQ → free-form.** A USMLE 5-option MCQ → an open-ended diagnosis with an alias-aware judge audited on about 200 samples ([HuatuoGPT-o1](https://arxiv.org/abs/2412.18925)). Math MCQ → canonical integer (B7). Targets with no known ceiling: unsolved Stack Exchange questions (500 kept from about 3 million candidates) ([UQ](https://arxiv.org/abs/2508.17580)). *Label:* filter for answer uniqueness *before* removing options. Nemotron-CrossThink found a unified open-ended format beat mixed formats by 1.21% and short answers beat long ones by 1.20% ([Nemotron-CrossThink](https://arxiv.org/abs/2504.13941)). **Fails when:** the upgrade leaves most items at zero; see B7's Golden Goose result and K2's format ladder.

**H4. Single-turn → multi-turn.**
- *Sharding*: a GSM8K problem in one turn → 5 shards over 5 turns. Performance drops 39% on average across 15 LLMs (aptitude −16%, unreliability +112%) ([Lost in Conversation](https://arxiv.org/abs/2505.06120)).
- *Carry-over*: a system prompt "always French, ≤ 80 words", a turn-3 override "for this answer only, English and a table", then a turn that must revert. o1-preview fell from 0.877 to 0.707 between turns 1 and 3 ([Multi-IF](https://arxiv.org/abs/2410.15553)). Agentic coding reaches 7.04 turns and 91.33 constraints per instance ([MTAC-IFBench](https://arxiv.org/abs/2609.14992)).
- *Follow-ups*: counterfactual ("suppose Ann had 9 instead") or incremental turns derived from the symbolic program ([MathCAMPS](https://arxiv.org/abs/2407.00900)).
- *Self-correction*: turn 2 revises the model's own turn-1 attempt under the same tests ([SCoRe](https://arxiv.org/abs/2409.12917)).
- *Label:* INV/REC. Enforce information preservation and run **CONCAT** (all shards in one turn, about 95% of FULL expected) on every item, dropping items where it fails. Track the active-constraint set per turn in code.
- *Evidence:* **Strong:** RL, not SFT, extrapolates. MEM1's SFT collapsed beyond 6 objectives ([MEM1](https://arxiv.org/abs/2506.15841)). **Caveat:** part of the sharding gap may be structural ambiguity rather than missing capability ([Intent Mismatch](https://arxiv.org/abs/2602.07338)).

**H5. Static → interactive environment.** Q&A posts, problems, PRs or web pages → Dockerized multi-step tasks with tests. Examples: a one-shot code answer → a shell task with input files and pytest checks ([Nemotron-Terminal](https://arxiv.org/abs/2602.21193)); (illustrative) "which customers have overdue invoices?" from a pasted table → a Harbor task with a Postgres container where the agent applies late fees through an API and `tests/test.sh` checks the final DB state, packaged the way [SWE-Factory](https://arxiv.org/abs/2506.10954) packages SWE environments. Other routes: white-box environments whose state is readable by design, such as FSM sites, SQLite mock apps and code-generated surrogates ([AutoWebWorld](https://arxiv.org/abs/2602.14296)); "compute X for array A" → a 10–256-call tool environment ([CodeGym](https://arxiv.org/abs/2509.17325)). *Label:* the oracle passes, a no-op fails, fail-to-pass by exit code (F1 0.99 in SWE-Factory). *Evidence:* 48.6% of raw synthesized web tasks were feasible, 94.8% after environment repair ([Verified Synthetic Web Envs](https://arxiv.org/abs/2608.21898)). Cost runs from about $0.09–0.12 per valid LLM-built SWE environment to about $99–163 per retained curated one (note 18). C5 is the lightweight version of this operator.

**H6. Beyond the context window (memory and compression).** One HotpotQA question → 16 interleaved questions under a fixed memory budget: Qwen2.5-14B falls from about 37% EM per item at 2 objectives to about 3.5% at 16 ([MEM1](https://arxiv.org/abs/2506.15841)). QA over 2 gold paragraphs → the same question in a 3.5M-token stream with a 1,024-token overwrite memory (71.09% at 3.5M after training at 32–60K) ([MemAgent](https://arxiv.org/abs/2507.02259)). BrowseComp at T_max = 32 (15.2%) → T_max = 2048 (42.5%) with a report-as-memory workspace ([IterResearch](https://arxiv.org/abs/2511.07327)). "Total spent" over 2 sessions → 500 sessions ([UMA + Ledger-QA](https://arxiv.org/abs/2602.18493)). *Label:* INV per item. Require answer count = question count, remove closed-book-answerable items, hold state in code. **Moderate:** "train short, test long" works under RL for every horizon knob.

**H7. Single agent → multi-agent or game.** One agent sees the whole puzzle → two agents with complementary halves ([AsymPuzl](https://arxiv.org/abs/2512.03466)). TicTacToe against a random player → Kuhn Poker against the latest self ([SPIRAL](https://arxiv.org/abs/2506.24119)). A vocabulary word → Adversarial Taboo ([SPAG](https://arxiv.org/abs/2404.10642)). "Summarise this report" → a five-player game where one player's input has 20% masked and all vote on the spy ([SpyRL / RLSVR](https://arxiv.org/abs/2607.23802)). *Label:* rule-decided outcomes. For split information, the union of views equals the seed. **Fails when:** strong models solve split-information puzzles in two turns by sharing everything, so difficulty must come from bandwidth or turn limits. Single fixed opponents make training fail, so co-evolve opponents (note 09).

---

## 10. Family I — Break and inject errors

Breaking a working artifact yields a task whose gold solution is the original artifact (CON). It is the dominant SWE operator and a cheap source of verifier and critic training data.

| Operator | E→H | Label / verifier | Evidence | Fails when |
|---|---|---|---|---|
| **I1 Bug injection and feature removal** | *SWE*: flip `<` to `<=` → LM re-implementation of a heavily tested function, reverted real PRs, removal of hunks or files, masking a depth-≥3 call tree ([SWE-smith](https://arxiv.org/abs/2504.21798); [SSR](https://arxiv.org/abs/2512.18552); [SWE-Dev](https://arxiv.org/abs/2505.16975)). *Agentic*: an agent's own feature work that breaks invalidation logic (4 files, 400 lines) ([BugPilot](https://arxiv.org/abs/2510.19898)). *Verified code*: code mutants that still verify expose weak specs ([MutDafny](https://arxiv.org/abs/2511.15403)) | Fail-to-pass: the injected change breaks ≥1 passing test, and the inverse patch is the gold fix. Inverse mutation testing: reverting each changed file alone must fix at least one test. Hide the break by weakening tests only if the weakening is valid | SWE-smith median changed lines: Modify 3, Procedural 5, PR Mirror 14, Rewrite 24; PR Mirror and Rewrite trajectories were the most effective. SSR (removal + history reversion, injector rewarded for low-but-nonzero solve rates) gave +10.4 SWE-bench Verified and +7.8 SWE-Bench Pro over CWM-sft | Vanilla "inject a bug" prompting collapses to one-line edits (SSR). Challengers "win" by writing randomly failing tests or obfuscating code unless their action space is checked. Generated issues leak the fix |
| **I2 State corruption and fault injection** | *Terminal*: a healthy environment → a mis-pinned dependency, a missing env var and broken permissions until 12 tests fail ([CLI-Gym](https://arxiv.org/abs/2602.10999)). *Spreadsheets*: strip 5 derived artifacts (column, SUMIFS summary, pivot, chart, conditional formatting) along a dependency-consistent history ([WTM](https://arxiv.org/abs/2608.07873)). *Long-horizon*: fork a sandbox after the agent's first wrong edit and inject a config corruption ([DSec](https://arxiv.org/abs/2609.22978)). *SFT traces*: inject a failed install at step 4 of an expert trace ([TermiGen](https://arxiv.org/abs/2602.07274)) | The pre-inversion tests or artifacts are the oracle. Respect artifact dependency order. Forbid leftover backups and filter recoveries via cached Git or Conda state. Run the oracle from the forked state | WTM scores collapse at ≥5 transformations, and 88.5% of failures are wrong shape or placement | Recovery through cached state (a shortcut); faults the checker cannot see |
| **I3 Error-localization and critique meta-tasks** | *Vision*: "describe the image" → "in this ~200-word caption, which span is wrong?" (three mugs → two) ([ViCrit](https://arxiv.org/abs/2506.10128)). *Math*: "solve" → "is this solution correct; where is the first wrong step?" ([MR-GSM8K](https://arxiv.org/abs/2312.17080)). *Logic*: corrupt step 7 of a 12-step verified chain and recompute downstream; answer "7" ([Verifiable counterfactual PRM](https://arxiv.org/abs/2605.02395)). *Code*: "write kth_prime()" → "this solution fails 1 of 40 hidden tests: True or False, and why?" ([Critique-Coder](https://arxiv.org/abs/2509.22824)) | CON. The injected location is the label. Prove that the corrupted step is not derivable from its prefix. Balance labels and draw candidates from near-misses | ViCrit gave CharXiv +6.4 at 7B. GPT-4-injected error data was >90% accurate although GPT-4 *detects* such errors only about 40% of the time, so construction outruns solving | **Off-policy injected errors do not teach self-correction**: models repeat the injected mistake ([2512.02389](https://arxiv.org/abs/2512.02389)). Use constructed errors for verifier and PRM training, and on-policy failures (SCoRe, failure prefixes) for self-correction |

---

## 11. Family J — Target the learner

J-operators are *meta-operators*. They decide **where** (which seed), **which** (which operator) and **how far** to push, using the current policy's behavior. The universal ingredient is a pass-rate band that excludes 0 and 1, re-measured every stage (chapter 01, chapter 05). The rest of this section is about what sits on top of that band.

| Operator | Mechanism and E→H | Evidence | Guardrails |
|---|---|---|---|
| **J1 Harden until it fails (solver in the loop)** | Run a solver. If it succeeds, find the shortcut and harden; if it fails, confirm solvability or repair. A 1-day trip with one budget → a multi-day trip with tier-conditional budget and rating rules, with `solve()` and `verify()` co-evolved ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)). Terminal tasks revised until a strong solver passes and a weak one fails ([CalibForge](https://arxiv.org/abs/2608.06352)) | Skill2Env's full-credit rate went from 48.4% to 15.4% in two rounds ([Skill2Env](https://arxiv.org/abs/2609.33772)). 81% of *validated* terminal candidates did not separate strong from weak solvers, and calibrated data beat validate-only data by 8.6 points on TB2.0 (CalibForge). Recursive solution-first escalation took DeepSeek-V4-Pro pass@4 from 90% to 2.5% over 15 rounds at about $0.05 per task ([RST](https://arxiv.org/abs/2608.05466)) | Require a reference solution or pass@N > 0 after every revision (DeepSeek-V3.2 uses pass@100 > 0). Separate "unsolved because ambiguous" from "unsolved because hard". For RL, the weak solver should be your current policy |
| **J2 Failure mining / weakness targeting** | Mine the policy's failures (stagnating prompts, failing tools, error clusters, rare failed prefixes) and synthesize there. Illustrative: combinatorics seeds saturate, but inclusion–exclusion with restrictions stagnates → recombine that concept cluster and keep items at 25–75% policy accuracy ([SwS](https://arxiv.org/abs/2506.08989)). A tool graph built from the 1,204 of 2,095 tools two small models got wrong ([HardGen](https://arxiv.org/abs/2601.01498)). A MATH item solved 127/128 → a rollout started from 40% of its one wrong trajectory ([Failure-prefix conditioning](https://arxiv.org/abs/2601.20829)) | SENTINEL: tau2 Retail 66.4 → 74.9 ([SENTINEL](https://arxiv.org/abs/2606.12908)). TableDreamer's 27K weakness items beat 34K unfiltered ones (60.69 vs 56.28) ([TableDreamer](https://arxiv.org/abs/2506.08646)). Failure-prefix conditioning scored 44.5 against 40.7 for plain RLVR on the same saturated items | Route non-model failures (environment, task, verifier) to repair *before* generating ([Qwen-UI-Agent](https://arxiv.org/abs/2607.28227)). About 35% of SwS problems land in its [25%, 75%] band, so budget rejection |
| **J3 Variational rewrites of near-solved items** | For items solved 12.5–50% of the time, feed a correct policy solution and ask for structurally different problems with the *same* answer. "Divisors of 360?" (24) → "integer-sided garden of area 360 m²: how many ordered (length, width) pairs?" (24) ([SvS](https://arxiv.org/abs/2508.14029)). Seed only from mostly-solved (8–15/16) items ([QbQ](https://arxiv.org/abs/2608.01522)) | SvS: +18.3 / +22.8 pass@32 on AIME24/25 where standard RLVR shows little improvement, with entropy preserved. QbQ: seeding from the harder 1–7/16 band did *worse*. **Strong:** seed from "almost solved", not "never solved" | Answer-preserving, so it fails safe. Reject variants solved 0/k or k/k (leaked or broken). SvS's naive "is it solvable" reward produced variants that leaked hints |
| **J4 Adversarial / regret-based proposers and self-play** | Train a proposer rewarded for valid tasks the solver fails or barely solves: learnability bands (R-Zero), validity × difficulty (VHG), hint-based regret (SPADE), 1 − follower satisfaction ([SEIF](https://arxiv.org/abs/2605.07465)), injector rewards for low-but-nonzero solve rates (SSR), a proposer rewarded only when 1 of K attempts verifies ([ANCORA](https://arxiv.org/abs/2604.27644)) | A validity-gated setter reached 45.4% on competition integrals vs about 31% for R-Zero, and its valid rate rose from 30.6% to 75.5% ([VHG](https://arxiv.org/abs/2605.06660)). Removing SEIF's instructor evolution cost 2.7 IFEval points. Grounded self-play beat ungrounded (SPICE 43.9 vs 40.7) ([SPICE](https://arxiv.org/abs/2510.24684)) | **Validity is the binding constraint.** R-Zero's pseudo-labels fell from 79% to 63% and training peaked after 1–3 iterations ([R-Zero](https://arxiv.org/abs/2508.05004)). Pushing the solve-rate floor from 0.5 to 0.1 cut validity from 70.8% to 42.3% ([OpenSIR](https://arxiv.org/abs/2511.00602)). A −0.1 penalty for invalid questions caused a "death spiral" ([SSP](https://arxiv.org/abs/2510.18821)), so give invalid proposals zero reward. Keep real anchors |
| **J5 Directional rewrite with band acceptance (learned operator choice)** | Choose the rewrite from the pass rate r: harden when r > 0.5, context-shift at matched difficulty when 0 < r ≤ 0.5, simplify when r = 0 ([SETA](https://arxiv.org/abs/2607.10891)). A per-seed MILP over 6 hand-defined actions around a target frontier ([Envs-FORGE](https://arxiv.org/abs/2608.14312)). A planner picks 3 of 5 structural operators per seed ([QbQ](https://arxiv.org/abs/2608.01522)). Append constraints until pass@1 ∈ (0, 0.5] ([IFDecorator](https://arxiv.org/abs/2508.04632)) | Envs-FORGE: tb-core 40.0 → 49.2 on Qwen 3.5 35B. Only 77% of SETA's "decrease" rewrites and 60% of its "increase" rewrites moved the pass rate as declared | Accept a harder rewrite iff α_low < acc(q′) < acc(q), so solvability is witnessed by the policy ([RLAnything](https://arxiv.org/abs/2602.02488)). Re-measure after every rewrite |

**Frontier-prioritized mutation.** Choosing *which* seeds to mutate matters as much as how. Advantage-weighted prompt evolution (60.0–60.1) beat uniform evolution (57.5), while reward-variance selection (54.8) did worse than uniform ([eva](https://arxiv.org/abs/2411.00062)). The choice of signal can flip results. **Moderate.**

---

## 12. Family K — Remove scaffolding (and add it back)

**K1. Hint, harness and affordance removal** (INV: the verifier is unchanged).
- *SWE/terminal*: a task that names the file and output format → the same goal stated abstractly, with no visible failing tests and no stated strategy ([MiniMax-M2](https://arxiv.org/abs/2605.26494); [Qwen3-Coder-Next](https://arxiv.org/abs/2603.00729)).
- *Agents*: H0 (text observations, state, hints, rules) → H4 (raw visuals plus the task description), still sampling earlier levels ([AES + HDC](https://arxiv.org/abs/2608.03571)). "Use Insert → PivotTable; put Promotion in Column Fields" → the bare objective with generic error messages ([Environment Tuning](https://arxiv.org/abs/2510.10197)).
- *Web*: re-issue each page-local task from the site root, which is *start-state regression* ([Go-Browse](https://arxiv.org/abs/2506.03533)).
- *Verified code*: fill one assertion → regenerate all invariants of a 7-invariant program, i.e. hole-punching ([DafnyBench](https://arxiv.org/abs/2406.08467)).
- **Caution:** several popular operators *make tasks easier* (explanatory feedback, hints, cues). For saturated sets, reverse them (note 17). The harness can dominate measured difficulty: a 4B model went from 8.3% to 37.2% just by changing harness, because 96% of runs in the original harness hit the turn limit ([FrogNano](https://arxiv.org/abs/2609.07925)). **Strong:** calibrate difficulty inside the exact RL harness, and randomize harnesses to keep difficulty from being harness-specific ([Kimi K3](https://arxiv.org/abs/2607.24653); [DeepSeek-V4.1](https://arxiv.org/abs/2609.19969)).

**K2. The inverse: add scaffolds to make p = 0 items learnable, then fade them.** Complexification overshoots by default, so every pipeline needs this operator. Options, in order of how on-policy they are (note 20):
1. Answer-preserving format ladder: 4-choice → 10-choice → cloze → open-ended, promoting at accuracy ≥ 0.5. Cloze reached 18.9% against 11.1% for 4-choice after RFT ([Cog-DRIFT](https://arxiv.org/abs/2604.04767)).
2. Inverse rewrite (D1).
3. Teacher candidates in the prompt ([ZPPO](https://arxiv.org/abs/2606.18216)).
4. Expert-prefix anchors, binary-searching the shortest prefix that makes the group non-degenerate ([BREAD](https://arxiv.org/abs/2506.17211)), or partial-solution hints annealed from 50% to 25% ([QuestA](https://arxiv.org/abs/2507.13266)).
5. An off-policy teacher trace in the group ([LUFFY](https://arxiv.org/abs/2504.14945)).

Scaffold fading took AutoOR from 0% at pass@64 to 48.98% ([AutoOR](https://arxiv.org/abs/2604.16804)). Stepping stones (an easier lemma, then a harder "lift") need only be well-posed: 32.8% of SOAR's useful stepping stones were fully correct and 63% well-posed ([SOAR](https://arxiv.org/abs/2601.18778)). This does *not* license unverified final targets. **Strong:** complexify above the band, train inside it, scaffold below it.

---

## 13. Family L — Change representation

Render the same latent instance differently: notation (CNF in math notation → DIMACS → story → negated "DualStory"), language (94 tasks in 14 languages), modality (givens moved from text into the diagram), presentation (constraints listed → woven mid-paragraph next to examples that each violate one constraint) ([SATQuest](https://arxiv.org/abs/2509.00930); [SATBench](https://arxiv.org/abs/2505.14615); [Multilingual Reasoning Gym](https://arxiv.org/abs/2603.10793); [MathVerse](https://arxiv.org/abs/2403.14624); [MulDimIF](https://arxiv.org/abs/2505.07591)). *Label:* INV. For LLM renderings, back-translate and solver-check equivalence (SATBench). For images, render from the same latent state.
- **Mostly diversity, weak difficulty.** Cross-format robustness lags (SATQuest). Incorporation is harder than Listing, which is harder than Example. Models do better when constraints come hard-to-easy ([Order Matters](https://arxiv.org/abs/2502.17204)).
- **Real difficulty only when information moves into a harder channel.** Moving givens from the question text into the figure is the clearest case; geometry data rendered from symbolic states gave +22.21 on MathVerse Vision-Only ([GeoSym127K](https://arxiv.org/abs/2605.16371)).
- Use L operators to fight template overfitting (ether0 paraphrases templated items during RL) and as held-out renderers for evaluation, not as the main hardness lever. **Moderate.**

---

## 14. Anti-operators: fake hardness and how to detect it

An **anti-operator** lowers measured accuracy, or raises apparent complexity, without adding difficulty the policy can learn from. Such items usually hurt, because they either reward the wrong behavior or yield all-zero groups that waste rollouts. The literature converges on one rule: **"complex" is not "hard for your model", and "hard" is not "valuable"** ([IFDecorator](https://arxiv.org/abs/2508.04632); [CompassPlay](https://arxiv.org/abs/2609.32228)). **Strong.**

| Anti-operator | Why it looks like hardening | Evidence that it isn't | Detection test |
|---|---|---|---|
| **Paraphrase / surface rewrite** | Accuracy dips; text looks new | +0.4 GSM8K from rephrasing vs +2.3/+2.6 from inversion ([MetaMath](https://arxiv.org/abs/2309.12284)). Paraphrasing and noising had limited impact in dynamic evaluation ([Benchmark Self-Evolving](https://arxiv.org/abs/2402.11443)). LLM "complex question" augmentation cut BIRD-dev from 64.9 to 62.5 ([Arctic-Text2SQL-R1](https://arxiv.org/abs/2505.20315)). Evol-Instruct rewriting lost to unchanged prompts, 50.24 vs 50.51 ([LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412)). A surface-only rewrite fell below its own SFT start, 67.1 vs 71.0 ([QUBRIC](https://arxiv.org/abs/2606.03968)). Text-only task rewrites did not beat untouched descriptions ([OpenThoughts-Agent](https://arxiv.org/abs/2606.24855)). Evol-Instruct pushed pass@8 below the untuned backbone, 54.2 vs 55.1 ([EvoTD](https://arxiv.org/abs/2605.11666)) | Require the answer to differ from the parent's, or the solution program to differ, or the *solution-signature* novelty (canonical solver code, masked templates) to be real ([QbQ](https://arxiv.org/abs/2608.01522); [R-Diverse](https://arxiv.org/abs/2602.13103)). Track pass@k, not only pass@1 |
| **Verbosity, tedium, padding** | Longer traces, lower accuracy | Code2Math rejects score 1 (unchanged solution path) and score 2 ("computational tedium") ([Code2Math](https://arxiv.org/abs/2603.03202)). STP drops the bottom 20% by proof-length/statement-length ratio to discourage "artificially hard conjectures with complicated goals" ([STP](https://arxiv.org/abs/2502.00212)). Size scaling conflates reasoning with output length ([Illusion of Thinking](https://arxiv.org/abs/2506.06941)). Generators optimized on trace length drift toward verbosity or ambiguity (note 02) | Elegance ratio, method-change rubric, a compact-answer requirement, and a tool-access probe (difficulty that disappears when Python is available is load, not reasoning ([BBEH](https://arxiv.org/abs/2502.19187)); code execution cut MathDuels solver error only from 25.0% to 21.1%, a sign of genuine difficulty ([MathDuels](https://arxiv.org/abs/2604.21916))) |
| **Ambiguity / under-specification injection** | 0% pass looks like "frontier" | Removing a contract detail gave 0% because of *ambiguity*, not difficulty ([FrogNano](https://arxiv.org/abs/2609.07925)). Obfuscated answers are "not always unique" ([WebSailor](https://arxiv.org/abs/2507.02592)). Adversarially filtered HLE had 15.4% expert disagreement, and HLE-Verified kept only 668 of 2,500 items unchanged ([HLE / HLE-Verified](https://arxiv.org/abs/2501.14249)). Self-play questions hacked by non-uniqueness ([SSP](https://arxiv.org/abs/2510.18821)) | Uniqueness enumeration; a reference solver working from the spec alone; CONCAT and oracle-evidence controls; ask a reader to list *all* valid answers |
| **Contradiction / unsolvability** | "Nobody solves it" | 10,772 evolved prompts at pass rate 0 vs 7,324 usable ([IFDecorator](https://arxiv.org/abs/2508.04632)). About 98% of random 12-constraint sets are incompatible ([CSE](https://arxiv.org/abs/2608.12426)). Contradictions are the largest flaw class in synthetic math (13.97% of ValiMath) ([MathQ-Verify](https://arxiv.org/abs/2505.13903)). <3% accuracy on new environments mostly flagged semantic errors ([InternBootcamp](https://arxiv.org/abs/2508.08636)) | Witness solution or pass@N > 0 by a stronger or hinted solver; compatibility checker; vacuity check for formal items |
| **Unrelated concatenation** | Long, multi-step, low pass rate | The most executable generator (chain-then-summarise) made the weakest agent, 21.6 vs 38.2 ([AutoPlay](https://arxiv.org/abs/2509.25047)). Bans on "artificially hard" chains ([Gym-Anything](https://arxiv.org/abs/2604.06126)) | Require a data dependency across every hop; check pass rate against the product of per-link pass rates |
| **Guessable answers and leaked shortcuts** | High difficulty rating, lucky passes | About 28% spurious guessing (correct answers with invalid reasoning) in mid-sized models ([TRACE](https://arxiv.org/abs/2607.04784)). LLM-written NLI is 86–96% hypothesis-only solvable (note 19). One leftover precise clue collapses a multi-hop item: DeepResearch-9K answers were hit at step 3.4 with a 27.2% prior-shortcut rate ([FORT](https://arxiv.org/abs/2606.12087)) | No-CoT, no-tool and partial-input baselines; answer-hit time; prior-knowledge probe |
| **Verifier artifacts** | Correct answers fail, so the item "looks hard" | 59.01% of correct special-judge solutions fail exact match ([ScaleBox](https://arxiv.org/abs/2604.27467)). Rule checkers miss equivalent answers ([rule vs model verifiers](https://arxiv.org/abs/2505.22203)) | Audit TPR/TNR with known-correct alternatives before complexifying (chapter 04) |
| **LLM-regenerated "hard" items** | New items, new wording | Regenerated DROP/CondaQA were "less challenging for LLMs", did not preserve rankings, and were shorter (18.35 vs 27.17 words) ([Gill et al.](https://arxiv.org/abs/2505.22830)) | Compare length, perplexity and embedding overlap with real hard items; measure difficulty on a probe model |

**Difficulty is not training value.** CompassPlay's gradient-alignment proposer reward beat a pure difficulty reward (+1.5 coding, +2.7 math, 40% fewer GPU-hours in Lean) ([CompassPlay](https://arxiv.org/abs/2609.32228)). LLM judges are poor raters of generated-problem quality (o3–human correlation 0.07), while measured difficulty gain correlated up to 0.60 ([AutoCode](https://arxiv.org/abs/2510.12803)). **Strong:** accept an operator's output by measured pass-rate movement on the current policy plus a validity control, never by an LLM's hardness rating.

---

## 15. Operator × domain matrix

Legend: **●** established, with several published pipelines in that domain in the evidence base; **◐** reported in one or two works, or only as evaluation; **○** no direct evidence found, plausible by analogy (**Proposal**); **·** not meaningful. Domain columns: Math (informal), Formal (Lean, geometry, inequalities), Code (algorithmic, verified code, kernels), SWE/Term (repos, terminal, spreadsheets, data), Puzzles/OR, Tools (API/DB agents), Web (search/deep research), GUI, IF (instruction following), LongCtx, MM/Sci/SQL.

| Op | Math | Formal | Code | SWE/Term | Puzzles/OR | Tools | Web | GUI | IF | LongCtx | MM/Sci/SQL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 chain | ● | ◐ | ● | ● | ◐ | ● | ● | ● | ◐ | ◐ | ◐ |
| A2 width | ◐ | · | ○ | ◐ | ○ | ● | ● | ● | ● | ● | ◐ |
| A3 nested substitution | ● | ○ | ○ | · | ○ | ◐ | ● | ◐ | ◐ | ◐ | ○ |
| A4 conditional branch | ◐ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ● | ○ | ○ |
| A5 skill/concept mix | ● | ◐ | ● | ● | ◐ | ◐ | ○ | ◐ | ◐ | ○ | ◐ |
| A6 fusion | ● | ● | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ |
| A7 multi-fault union | · | · | ○ | ● | · | ○ | · | ○ | · | · | · |
| A8 long-horizon chain | ○ | ◐ | ○ | ● | ○ | ◐ | ○ | ● | ◐ | ○ | ○ |
| B1 constraint stacking | ◐ | · | ● | ◐ | ◐ | ● | ◐ | ○ | ● | ○ | ◐ |
| B2 level ladder | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ○ | ● | ○ | ○ |
| B3 coupled/scoped | ○ | · | ○ | ◐ | ◐ | ○ | ○ | ○ | ● | ○ | ○ |
| B4 intersection | ○ | · | · | · | ● | ◐ | ● | ○ | · | ○ | ○ |
| B5 optimality/perf | ○ | ◐ | ● | ● | ● | ◐ | · | · | · | · | ◐ |
| B6 budgets | ◐ | · | ○ | ◐ | ○ | ○ | ● | ● | · | ◐ | ○ |
| B7 answer-space/cert | ● | ◐ | ◐ | · | ● | · | ○ | · | · | ◐ | ● |
| B8 verifier/rubric | ○ | ◐ | ● | ● | ◐ | ○ | ◐ | ◐ | ◐ | ○ | ◐ |
| C1 fuzzing | ○ | · | · | · | ○ | ◐ | ● | ◐ | · | ◐ | ○ |
| C2 clue removal | ○ | · | · | ◐ | ○ | ◐ | ● | ○ | · | ○ | ○ |
| C3 insight hiding | ◐ | ● | ○ | · | ○ | · | · | · | · | · | ◐ |
| C4 implicit params | ◐ | · | ○ | ○ | ◐ | ● | ◐ | ○ | ◐ | ○ | ○ |
| C5 withhold behind tools/users | ○ | · | ◐ | ◐ | ● | ● | ○ | ● | ○ | ◐ | ◐ |
| C6 spec degradation | · | · | ○ | ● | · | ◐ | · | ● | ○ | · | ○ |
| C7 retrieval obstruction | · | · | · | ○ | · | ○ | ● | · | · | ● | ○ |
| C8 renaming/substitution | ◐ | ● | ◐ | ◐ | ◐ | ○ | · | · | · | ○ | ○ |
| D1 given↔unknown | ● | ○ | ○ | · | ○ | ○ | ◐ | · | · | · | ◐ |
| D2 program inversion | · | · | ● | ○ | ◐ | · | · | · | · | · | ◐ |
| D3 answer-first/planted | ◐ | ● | ◐ | ◐ | ● | ● | ● | ● | ● | ◐ | ● |
| D4 explore-then-describe | · | ◐ | · | ◐ | · | ● | ◐ | ● | · | · | ○ |
| D5 checker writing | · | ○ | ● | ◐ | · | · | · | · | · | · | · |
| D6 prove-or-disprove | ◐ | ● | ◐ | · | ◐ | · | · | · | · | · | · |
| E1 size | ◐ | ◐ | ● | ◐ | ● | ○ | · | ◐ | ○ | · | ◐ |
| E2 depth/topology | ● | ● | ◐ | ◐ | ● | ○ | ● | ○ | ◐ | ◐ | ● |
| E3 horizon | ● | · | ○ | ◐ | ◐ | ◐ | ◐ | ● | · | ○ | ○ |
| E4 context length | ◐ | · | ◐ | ○ | · | · | ◐ | · | ○ | ● | ○ |
| E5 entities/state | ○ | · | ○ | ● | ◐ | ● | ◐ | ● | · | ● | ◐ |
| E6 solver effort | ○ | ◐ | ○ | · | ● | · | · | · | · | · | ○ |
| F1 irrelevant info | ● | ◐ | ◐ | ○ | ◐ | ○ | ○ | ○ | ◐ | ● | ◐ |
| F2 hard distractors | ○ | ○ | · | · | ◐ | ● | ◐ | ◐ | ○ | ● | ● |
| F3 minimal edit | ● | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ |
| F4 counterfactual rules | ◐ | · | ○ | ○ | ● | ◐ | · | ○ | ◐ | ○ | ○ |
| F5 environment noise | · | · | · | ◐ | · | ● | ◐ | ● | · | · | ○ |
| F6 adversarial injection | ◐ | · | · | ○ | · | ● | ◐ | ◐ | ● | ○ | ○ |
| F7 infeasible twins | ● | ○ | ○ | ◐ | ◐ | ◐ | ○ | ● | ○ | ● | ○ |
| G1 parametric lifting | ● | ○ | ● | ◐ | ● | ○ | · | ◐ | ○ | ○ | ◐ |
| G2 generalize/specialize | ● | ● | ○ | · | ○ | · | · | · | · | · | ○ |
| G3 demand general object | ◐ | ○ | ◐ | ◐ | ◐ | · | · | · | · | · | ◐ |
| H1 decision→optimization | ○ | · | ● | · | ● | · | · | · | · | · | ◐ |
| H2 compute→prove/verify | ◐ | ● | ● | · | · | · | · | · | · | · | · |
| H3 closed→open / MCQ→free | ● | · | ◐ | · | ○ | · | · | · | · | ◐ | ● |
| H4 single→multi-turn | ◐ | · | ◐ | ◐ | · | ◐ | · | ○ | ● | · | ◐ |
| H5 static→environment | · | · | ◐ | ● | ◐ | ● | ○ | ● | · | · | ○ |
| H6 beyond context window | · | · | · | ◐ | · | ○ | ● | · | · | ● | · |
| H7 multi-agent/game | ◐ | · | · | · | ◐ | · | · | · | · | · | ◐ |
| I1 bug/feature removal | · | · | ◐ | ● | · | · | · | · | · | · | · |
| I2 state/fault injection | · | · | · | ● | · | ○ | · | ○ | · | · | ○ |
| I3 error localization/critique | ● | ◐ | ● | ○ | ◐ | · | · | · | · | · | ● |
| J1 harden-until-fails | ◐ | ◐ | ◐ | ● | ◐ | ● | ● | ◐ | ◐ | ○ | ◐ |
| J2 failure mining | ● | ◐ | ◐ | ◐ | ○ | ● | ◐ | ● | ◐ | ○ | ◐ |
| J3 variational rewrite | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ |
| J4 adversarial proposer/self-play | ● | ● | ● | ● | ◐ | ◐ | ● | ○ | ◐ | ◐ | ◐ |
| J5 directional rewrite + band | ◐ | · | ◐ | ● | ○ | ◐ | ○ | ◐ | ● | ○ | ○ |
| K1 scaffold removal | ○ | ○ | ● | ● | ◐ | ◐ | ○ | ● | · | · | ◐ |
| K2 add scaffolds (inverse) | ● | ◐ | ◐ | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ |
| L representation | ◐ | ○ | ○ | ○ | ● | ○ | · | · | ◐ | ○ | ● |

Three patterns stand out. (1) **Answer-first construction (D3) and failure-targeted meta-operators (J1, J2, J4) are domain-general.** They appear wherever an executable artifact exists. (2) **Fuzzing, clue removal and retrieval obstruction (C1, C2, C7) are mature only in web search and long context.** Porting them to tools, GUI and terminal tasks is a cheap, label-preserving opportunity: fuzz entity IDs and state references (CoVe does this for orders), and remove salient clues from goals. **Proposal.** (3) **Insight-type operators (C3, F3, F4, G2) are concentrated in math, formal and puzzle domains.** Code and agent domains rely on volume, composition, concealment of specs and noise instead. Chapters 07a and 07b give per-domain recipes.

---

## 16. Composing operators: algebra, order, budgets, stopping rules

### 16.1 Label algebra

Treat each operator as a function on the IR and track its label tag (section 1.2):

```
INV ∘ INV        = INV          (e.g. fuzz ∘ distract ∘ shard)
REC ∘ {INV,REC}  = REC          (as long as every step acts on the IR before rendering)
CON / CERT ∘ X   = CON / CERT   (if the final grader checks a certificate or constructed state)
RED ∘ anything   = RED          (one re-derived step makes the whole item need an independent verifier)
```

For RL pools, **keep RED operators out of the chain**, or place them first and verify their output independently before stacking INV/REC operators on top. This preserves the "fails safe" property of answer-preserving items under GRPO. Each step verifies only its *increment* (the new hop, constraint or fault), so verification cost stays linear in depth ([TaskCraft](https://arxiv.org/abs/2506.10055); [WebShaper](https://arxiv.org/abs/2507.15061)). **Moderate.**

### 16.2 Order of application (Proposal, built from evidenced rules)

```
0. LIFT        seed → executable IR + checker; pin the gold answer or end state     (G1, H5)
1. STRUCTURE   compose / scale / invert / upgrade type / break, on the IR (A, E, D, H1–H2, I)
               → recompute label; re-validate verifier (oracle passes, no-op fails)
2. CONSTRAIN   add constraints with a compatibility check + witness                 (B1–B6)
3. TIGHTEN     answer-space and verifier hardening                                   (B7, B8)
4. RENDER      IR → text / image / environment
5. CONCEAL     leaf-first; uniqueness and over-determination check after EACH step  (C1–C8)
6. DISTRACT    noise and distractors computed against the FINAL IR (off-path)       (F1, F2, F5)
7. DELIVER     tools, users, turns, memory, harness; remove scaffolds               (C5, H4–H7, K1)
8. GATE        controls (CONCAT/oracle, shortcut probes, no-op) → policy pass-rate band
```

Why this order: every robust pipeline pins the ground truth before hardening ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556); note 06). Solution-first escalation grows the reference solution, realigns the environment, extends the verifier, and *only then* rewrites the instruction ([RST](https://arxiv.org/abs/2608.05466)). Concealment applied before composition exposes constants near the target and creates shortcuts, which is why leaves are expanded layer-wise ([WebShaper](https://arxiv.org/abs/2507.15061)). Distractors added before scaling can land on the solution path, whereas GSM-Infinite adds noise only with edges pointing away from the query's ancestors. Keep reasoning depth and context length as separate, separately tuned knobs ([Michelangelo](https://arxiv.org/abs/2409.12640); [GSM-Infinite](https://arxiv.org/abs/2502.05252)).

**Well-evidenced compositions:**
- Bug combination + symptom-only issue, with issue sufficiency checked separately ([SWE-smith](https://arxiv.org/abs/2504.21798)).
- Graph composition + fuzzing + cycles + evidence dispersion + shortcut repair ([FORT](https://arxiv.org/abs/2606.12087)).
- Chaining + hindsight summarisation into an implicit goal ([AgentSynth](https://arxiv.org/abs/2506.14205)).
- Structural operators + answer-must-differ + band selection ([QbQ](https://arxiv.org/abs/2608.01522)).
- Sharding on top of any verifiable seed, reusing its verifier ([Lost in Conversation](https://arxiv.org/abs/2505.06120)).
- Stacking *several interaction operators on one seed* (sharded turns + beyond-window horizon + injected tool output + withholding partner) is untested (note 21).

### 16.3 Budgets

- **Rounds.** Free-form LLM evolution peaks at 2–3 rounds. WizardCoder was best after 3; UltraIF's pass rate fell from 91.8% to 79.4% by the third iteration; SEIF saturated after turn 3 ([WizardCoder](https://arxiv.org/abs/2306.08568); [UltraIF](https://arxiv.org/abs/2502.04153); [SEIF](https://arxiv.org/abs/2605.07465)). Recursive escalation sustained 15 rounds only because every child was solution-first and **re-validated from scratch, never inheriting validation**. Yield stayed at 498–572 per 1,000 attempts while hardness came from work (solutions ×5.6), not prose (instructions ×1.4) ([RST](https://arxiv.org/abs/2608.05466)). **Strong.**
- **Knob sizing.** Pick k so that p^k ≈ 0.2–0.5 for conjunctions ([CSE](https://arxiv.org/abs/2608.12426)). Predict chains by the product of link pass rates, then expect *worse* (h1). A bimodal pass-rate histogram (mass at 0 and 1) means the generator's step size is too coarse; add intermediate levels per seed (note 20).
- **Yield per operator.** Log validity and in-band yield for every operator and axis. Examples: 25.5% gate yield overall, 7.7–45.5% per axis, 64% of rejections "too easy" ([Trading Human Curation](https://arxiv.org/abs/2606.03800)); SWE-smith yields of 33.8–96.9% per strategy. Candidates needed per accepted task = 1/(y_valid·y_band). Chapter 03 has the cost model.
- **Diversity caps.** Cap children per parent, category and operator family ([RST](https://arxiv.org/abs/2608.05466)). Spread rewrites over many seeds: about 750 seeds × 9 rewrites beat about 7 seeds × 999 at a fixed budget ([Fidelity–Diversity](https://arxiv.org/abs/2607.04563)). Force structural variety (QbQ's 3 of 5 operators per seed: Vendi 1.59 vs 1.25).

### 16.4 Stopping rules

1. **Stop on measured pass rate, not round count.** Examples: harden until pass@1 ∈ (0, 0.5] ([IFDecorator](https://arxiv.org/abs/2508.04632)), keep 1–7/8 ([Qwen-CUA](https://arxiv.org/abs/2608.02352)), or target about 0.3–0.6 (note 20).
2. **Reject steps that did not move difficulty as intended.** Accept iff α_low < acc(q′) < acc(q) ([RLAnything](https://arxiv.org/abs/2602.02488)). 23–40% of declared rewrites do not move ([SETA](https://arxiv.org/abs/2607.10891)).
3. **Stop concealment one step before uniqueness breaks.** Repair over-fuzzed items rather than discarding them ([FORT](https://arxiv.org/abs/2606.12087)).
4. **At p = 0, back off.** Audit label and verifier, then scaffold (K2) or regenerate. Keep a 0-pass item only with a solvability certificate: a teacher solve, pass@large-k > 0, or a co-built solution ([GLM-4.5](https://arxiv.org/abs/2508.06471); [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220)).
5. **Stop a lineage or family** when yield per 1,000 attempts, domain entropy or nearest-neighbour novelty drops (RST's per-round dashboard). Demand larger gains from repeated mutation families ([BenchEvolver](https://arxiv.org/abs/2606.01286)).
6. **Stop self-play iterations** when gold-slice label accuracy or the valid-proposal rate falls ([R-Zero](https://arxiv.org/abs/2508.05004); [SSP](https://arxiv.org/abs/2510.18821)).

```python
def complexify(seed, policy, ops, band=(0.2, 0.6), k=16, max_steps=6):
    ir = lift(seed)                                    # executable IR + checker, gold pinned
    p = pass_rate(policy, render(ir), k)               # measured on the CURRENT policy
    for _ in range(max_steps):
        if band[0] <= p <= band[1]:
            return ir                                  # in band: stop hardening
        if p < band[0]:
            return scaffold_or_regenerate(ir)          # audit first, then K2 / D1 / stepping stones
        op = choose(ops, ir, history=ir.lineage)       # structural before surface; penalize repeated families
        cand = op.apply(ir)                            # acts on the IR; recompute / inherit / construct label
        if not (cand.oracle_passes() and cand.noop_fails() and cand.unique() and cand.solvable()):
            log_reject(op, "invalid"); continue        # validity is the binding constraint
        p_new = pass_rate(policy, render(cand), k)
        if band[0] <= p_new < p:                       # harder, yet still solved by the policy
            ir, p = cand, p_new
        else:
            log_reject(op, "overshoot" if p_new < band[0] else "no_move")
    return None                                        # per-operator yield feeds the next choose()
```

---

## 17. Selection guide: which operators to try first

**Step 0, for any seed type.** Audit the verifier (TPR/TNR with known-correct alternatives and known-bad solutions), measure the pool's pass-rate histogram on the current policy, and reallocate rollouts and recycle zero-variance items before generating anything. That ordering often buys 1.5–2× at no generation cost (note 18). **Strong.** If the verifier is only a judge, send judge-scored hard tasks to SFT or evaluation rather than to RL reward (note 14), and prefer INV operators whose label does not depend on the judge. **Moderate.**

| Seed type (verifier) | Try first | Then | Guard against |
|---|---|---|---|
| Word problems with code solutions (exact match) | A1 chain via code substitution; E2 DAG depth/width; D1 inversion (with uniqueness) | F1 off-path NoOp; F3 script-based perturbation; G1 lifting | Rephrasing; RED labels from a single teacher |
| Competition math, answer only | J3 same-answer variational rewrites (SvS); G2 structural operators with answer ≠ parent (QbQ); B7 canonical answers | C3 insight hiding with program search; A5 concept pairs + cross-family agreement; K2 for p = 0 | Label decay on hard items; contamination (decontaminate outputs) |
| Formal statements (Lean/geometry/inequalities) | D6 prove-or-disprove + negation; A6 rule-based fusion; C8 substitution | C3 auxiliary hiding with an automation floor; E6/proof depth; subgoal recomposition | Vacuous hypotheses, misformalization, `sorry`/axioms, prover statement edits |
| Algorithmic code with tests | B8 test hardening *first*; E1 input-scale amplification; A5 feature composition | D2 abduction/induction; B5 performance goal; H2 spec lifting | Weak tests (TACO FPR >90%); timer hacks; hard-coding |
| Repositories with tests (SWE) | I1 removal + history reversion; A7 Combine; C6 symptom-only issue | A8 milestone chains; D5 test writing; K1 hint removal | F2P validity; inverse mutation testing; blocked network and git history |
| Procedural puzzles and OR | E1/E6 with an adaptive controller; D3 planted certificates; H1 ladder | F4 counterfactual rules; C5 move parameters behind queries; L formats | Generator bugs; bare labels; size-only tedium |
| Tool/API agents with DB state | C4 hidden intermediate steps; A1/A2 with gold replay; E5 environment enrichment | F5 tool noise; C5 user-held information; F7 infeasible ≤10% | Unit-test tools; final-state checks; stabilize the user simulator |
| Web / deep-search QA over a KB | A3 leaf substitution; B4 intersection; C1 fuzzing | C2 clue removal; C7 dispersion; cycle topologies | Non-unique answers; over-determination; no-tool shortcuts (log Ω, T_hit, p_prior) |
| GUI, terminal, spreadsheet tasks with state checkers | A8 phase chaining with data dependencies; C6 instruction abstraction; E5 cross-workspace breadth | K1 start-state regression; F5 runtime perturbation; I2 state inversion; J1 calibration | checker(initial) = 0 and checker(golden) = 1; information barrier between solver and checker author; trivial-agent baselines |
| Instruction following with code checkers | B1 with p^k and wide ranges; A4 logic structures; B3 coupled/scoped | C4 implicit parameters; H4 multi-turn carry-over; F6 privilege conflicts | Incompatible sets; reward hacking (intent check, trip wires); mix general data |
| Long-context QA | C7 lexical-overlap removal and indirection; F2 hard distractors; A1 with bridge masking | H6 beyond-window stacking; F7 sufficiency twins; E4 length *last* | Closed-book answerability; recall-gameable rewards |
| Multimodal, chart, SQL, science | D3 execute-then-phrase; E2 level scaling from latent state; C3 stem hardening with the answer hidden | L givens → image; I3 error localization; F2 distractor tables and options | Empty or degenerate executions; shortcut ablations; SFT regressions on RL-style data |
| Open-ended, rubric only | B8 rubric hardening (criteria that split rollouts, dropout); H7 latent-variable games; A5 per-skill rubrics | Scenario grounding with key-point rubrics; F7 impossible-rubric probes | Rubric hacking (use a held-out gold judge); keep only INV operators for RL |

**When the policy fails for a different reason than "too easy":**
- *Too easy but brittle* (high pass@1, low pass@k growth): add J3, F3, F4. Method-change operators, not volume.
- *Fails on interaction but not on content*: C5, H4, H6. VHD-Play's written-out vs agentic gap is the diagnostic.
- *Fails on long tasks*: shorten the horizon first (K2 / macro-actions), then E3 with a curriculum.
- *Fails on composition*: install atoms first, then A1/A5 under RL. Do not SFT on composed data and expect transfer ([f(g(x))](https://arxiv.org/abs/2509.25123); [Ineq-Comp](https://arxiv.org/abs/2505.12680)).

---

## 18. Gaps in the operator literature

- **Insight operators.** No known operator reliably produces verifiable tasks that force a *new* strategy rather than a recombination. Transformative generalization is still about zero (OMEGA), and LLM setters mostly recombine known algorithms ([AutoCode](https://arxiv.org/abs/2510.12803)).
- **Operator-level credit assignment.** There is no matched-compute RLVR ablation of inversion, chaining, nesting, distractors, concept injection, lifting, perturbation and insight hiding on identical seeds, and no bandit over (operator × domain) driven by held-out deltas (note 02 and note 01 open problems).
- **Operator-conditioned difficulty prediction** (parent p̂ plus the operator's measured drop) that avoids rollouts, and a validated automatic metric for **tedium vs insight** (Code2Math's rubric agrees with humans only 79.3% of the time).
- **Composing interaction operators on one seed** (sharding × horizon × injection × withholding) with measured knob interactions.

Chapter 10 ranks these as build items.
