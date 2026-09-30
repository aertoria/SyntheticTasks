# Domain recipes II: tool use, web/deep research, GUI & computer use, terminal/office, instruction following, long context, multimodal, SQL/tables, open-ended

> **Key takeaways**
>
> - **Build the ground truth first and write the prose last, then harden the artifact, not the text.** Every robust pipeline in this chapter fixes an executed trace, a final database/VM state, a SQL result, a simulator state or a latent scene before any natural language exists. Text-only "make it harder" rewriting keeps failing: LLM task-description hardening did not beat untouched descriptions ([OpenThoughts-Agent, 2026](https://arxiv.org/abs/2606.24855)), "complex question" augmentation cut BIRD-dev from 64.9 to 62.5 ([Arctic-Text2SQL-R1, 2025](https://arxiv.org/abs/2505.20315)), and offline Evol-Instruct prompts scored 50.24 vs 50.51 for unchanged prompts ([LLM-as-a-Tutor, 2026](https://arxiv.org/abs/2607.04412)). **Strong.**
> - **On saturated seeds, start with label-preserving operators that reuse your existing checker:** oracle-preserving perturbation and identifier fuzzing for tool calls; start-state regression, explicit→implicit goals and evaluator-first composition for GUI; instruction abstraction and multi-test-case perturbation for terminal and spreadsheets; sharding, objective stacking and hidden-spec users for any verifiable seed, each with a control (CONCAT, closed-book, no-op, no-data) that separates "hard" from "broken". **Strong.**
> - **Measure difficulty on the current policy and re-measure every iteration; structural proxies are loose.** Search tasks that look deep can be shallow in practice: about 20.6 retrieval calls for InfoSeek vs 141.0 for FORT ([FORT, 2026](https://arxiv.org/abs/2606.12087)). Only 19% of already-validated terminal tasks were learnable on first probe ([CalibForge, 2026](https://arxiv.org/abs/2608.06352)). Keep a band (1–7 of 8, (0, 0.5], 20–50%) and park 0/k tasks in a cheap monitoring pool instead of deleting them. **Strong** for policy-relative calibration; the monitoring pool itself is **Emerging** (one industrial report, [Qwen-UI-Agent, 2026](https://arxiv.org/abs/2607.28227)).
> - **The verifier is now the bottleneck, and its errors are asymmetric.** VLM and LLM judges over-accept: the same GUI policy scores 63.76% by VLM judge and 38.93% by code assertions ([GUI-Genesis, 2026](https://arxiv.org/abs/2602.14093)), and no LLM-judge setup exceeded AUROC 0.65 at detecting false success on tau2-bench ([2606.09863](https://arxiv.org/abs/2606.09863)). Hand-written scripts under-accept. Separate the solution-builder from the checker-writer, require checker(golden)=1 and checker(initial)=0, and run hacker–fixer probes before RL. **Strong** for the asymmetry; the builder/checker information barrier rests on one large study (CUA-Gym) and is **Moderate**.
> - **When hardening drives pass rates toward zero, change the reward and the estimator along with the data.** Per-assertion counts with a critic took terminal RL to 64.0 on TB2.1 while GRPO stayed flat at 51.7 ([T1, 2026](https://arxiv.org/abs/2609.11042)). Dense signals only help when they are anchored in construction metadata (evidence chunks, gold entities, information gain). Use RL rather than SFT for composed and horizon-extended skills, with SFT as a cold start. **Moderate.**
> - **Environment and state diversity scale separately from task count.** When wording saturates, add environments, initial states and harnesses: 10→80 environments gave gains more trajectories could not recover ([CUA-Gym, 2026](https://arxiv.org/abs/2605.25624)), and RL on synthetic general-agent environments transferred where code-and-search RL did not ([DeepSeek-V3.2, 2025](https://arxiv.org/abs/2512.02556)). **Strong**, with one caveat: naive domain mixing hurt a small per-domain tool-use run (AReaL-SEA, §1), so ablate the mixture.

## Contents

0. [The common spine and how to read the recipes](#0-the-common-spine-and-how-to-read-the-recipes)
1. [Function calling and multi-turn tool use](#1-function-calling-and-multi-turn-tool-use)
2. [Web search and deep research (BrowseComp-style)](#2-web-search-and-deep-research-browsecomp-style)
3. [GUI, computer-use and mobile agents](#3-gui-computer-use-and-mobile-agents)
4. [Terminal/CLI, data-analysis, spreadsheet and office agents](#4-terminalcli-data-analysis-spreadsheet-and-office-agents)
5. [Instruction following with composable constraints](#5-instruction-following-with-composable-constraints)
6. [Multi-hop and long-context reasoning](#6-multi-hop-and-long-context-reasoning)
7. [Memory and multi-turn interaction structure](#7-memory-and-multi-turn-interaction-structure)
8. [Multimodal and visual reasoning](#8-multimodal-and-visual-reasoning)
9. [Text-to-SQL and tables](#9-text-to-sql-and-tables)
10. [Open-ended and non-verifiable tasks](#10-open-ended-and-non-verifiable-tasks)
11. [Cross-domain summary: what to build first](#11-cross-domain-summary-what-to-build-first)

Evidence tags: **Strong** (several independent works or large ablations), **Moderate** (one careful study), **Emerging** (single recent or unreplicated work), **Proposal** (our synthesis, untested). The operator catalogue behind these recipes is in [Chapter 02](02-complexification-operator-taxonomy.md), pipeline architectures in [Chapter 03](03-generation-architectures.md), verifier engineering in [Chapter 04](04-verification-and-quality-control.md), RL mechanics (zero-variance groups, bands, estimators) in [Chapter 05](05-rl-playbook.md), SFT/distillation in [Chapter 06](06-sft-playbook.md), math/code/SWE/puzzles/science in [Chapter 07a](07a-domain-recipes-reasoning.md), frontier-lab pipelines in [Chapter 08](08-frontier-lab-practices.md), and a consolidated failure list in [Chapter 09](09-pitfalls-and-failure-modes.md).

---

## 0. The common spine and how to read the recipes

Every domain below follows the same template: **seeds** you probably have, the **best operators** with concrete easy → hard examples, the **architecture**, the **verifier**, the **difficulty knobs**, **pitfalls**, **starting datasets/environments**, and **key references with verified numbers**.

The recipes also share one spine. Once you see it, most domain-specific papers are variations on it:

```
 easy verified seed
   │ 1. LIFT to an executable artifact:
   │    (instruction, initial state, reference solution / gold trace, checker)
   ▼
 artifact ── 2. APPLY an operator to the artifact (state, trace, graph, program, scene),
   │           never only to the prose
   ▼
 re-execute reference ──► new gold           3. GATES: checker(gold)=1, checker(initial)=0,
   │                                             control views (closed-book / no-op / no-data / CONCAT)
   ▼
 render natural language LAST (hide procedure, IDs, intermediates)
   │                                          4. uniqueness, contract validity, shortcut/hack probes
   ▼
 calibrate on CURRENT policy (k rollouts) ──► band (e.g. 1–7 of 8); 0/k → monitoring pool
   ▼
 RL / SFT ──► failure diagnosis (model vs task vs env vs verifier) ──► choose next operator
```

Three rules recur across all ten domains:

1. **Verify only the increment when you can, re-verify everything when you cannot.** TaskCraft checks only the new hop, keeping cost roughly linear in hops ([TaskCraft, 2025](https://arxiv.org/abs/2506.10055)). RST re-validates every child in a fresh sandbox and never inherits validation ([RST, 2026](https://arxiv.org/abs/2608.05466)).
2. **Every difficulty filter needs a matching solvability filter.** "Frontier model fails" keeps broken and ambiguous items along with hard ones ([DeepDive, 2025](https://arxiv.org/abs/2509.10446)). Pair it with oracle-evidence solvability ([OpenSeeker, 2026](https://arxiv.org/abs/2603.15594)), a reference-solution pass, or a strong/weak solver split.
3. **Harder is a mixture, not a replacement.** Hard-biased sampling underperformed natural-distribution sampling in web RL ([WebGym, 2026](https://arxiv.org/abs/2601.02439)), and an Easy+Medium+Hard mix beat every single tier in visual game RL ([Game-RL, 2025](https://arxiv.org/abs/2505.13886)). See [Chapter 05](05-rl-playbook.md) for scheduling.

---

## 1. Function calling and multi-turn tool use

**Seeds.** BFCL-style single calls; tau-bench-style tasks over a database with policies; ToolBench/MCP tool pools; your own logged tool calls. First repackage every seed as (instruction, tool set, initial DB state, gold call trace, gold final state or answer, persona). Every operator below then becomes cheap.

### Best operators

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Oracle-preserving perturbation** | 2 relevant tools, clean JSON → dozens of look-alike tools, indirect phrasing, multi-format or noisy outputs, an erroneous first response | Gold call and answer unchanged, so the existing matcher is the reward; tag the judge-assisted families (erroneous outputs, missing parameters) separately | [COVERT (2026)](https://arxiv.org/abs/2604.09813): RL alone on Qwen2.5-14B, BFCL v3 56.5 → 59.9, ACEBench 53.0 → 59.3 |
| **Identifier fuzzification** | "Return order #W123" → "return the order containing the blue shirt and the leather shoes"; user ID → email or name + zip | A DB query confirms each description resolves to exactly one row | [CoVe (2026)](https://arxiv.org/abs/2603.01940): CoVe-4B, pure SFT, tau2 Airline 43.0%, Retail 59.4% |
| **Implicit-step ("advanced tool") abstraction** | `get_zipcode(A)`, `get_zipcode(B)`, `buy_tickets(zips)` → "Buy me tickets from Rivermist to Stonebrook" | The gold trace exists before the query; bias trace sampling toward tools the policy fails | [HardGen (2026)](https://arxiv.org/abs/2601.01498): Qwen3-4B BFCLv3 62.13 → 79.14 after SFT+RL |
| **Argument provenance / long-range dependencies** | Every argument in the current turn → some arguments must come from an earlier tool output or an earlier user turn | Tag each argument's source (`initial_state`, `prev_output`, `self_create`, `prev_user_msg`) and validate against the execution log | [SAP (2026)](https://arxiv.org/abs/2609.06124): BFCL v4 multi-turn 22.1 → 30.4; removing tags costs 6.4 |
| **Validated-task recombination / triple composition** | "Cancel order X" + "update the address on order Y" → one conversation for the same persona; tau2 (break-state, fix, assert) triples composed | Replay concatenated gold actions from a fresh state and re-run policy unit tests; at most one subtask per mutually exclusive group | [APIGen-MT (2025)](https://arxiv.org/abs/2504.03601); [tau2-bench (2025)](https://arxiv.org/abs/2506.07982): success approaches zero above 7 actions in dual-control mode |
| **Environment/DB complexification and graph walks** | One table, 10 orders → about 18.5 tables and 35.1 tools per environment; longer walks favouring weak or independent tool edges; programmatic tasks with loops and aggregation | Code- and SQLite-backed environments; verify on DB state diff; solution code executed in a sandbox | [AWM (2026)](https://arxiv.org/abs/2602.10090); [Agent-World (2026)](https://arxiv.org/abs/2604.18292); [EnvFactory (2026)](https://arxiv.org/abs/2605.18703) |
| **Controlled tool failure and missing information** | Perfect tools → intermittent errors, pagination, partial batches, silent ±10–25% value corruption, blocked primary path; missing parameter where the gold action is to ask | Fix the gold end state and the hidden-state summary; deterministic blocking; explicit reward for stopping when all paths are blocked | [Qwen-AgentWorld (2026)](https://arxiv.org/abs/2606.24597): controlled simulated RL +3.7 Tool Decathlon, +12.3 MCPMark, uncontrolled −0.9; [BENCH2ROBUST (2026)](https://arxiv.org/abs/2608.11977): 69 of 70 (model, subset) pairs degrade by 1.8–46.7 pp |
| **Boundary- and failure-targeted resampling** | A pool where most tasks pass 100% → each epoch, structurally isomorphic variants of tasks with mean progress reward in [0.20, 0.85]; tasks above 0.95 retired | New variants pass the same executable validation; cap injection at 20% of the active pool per epoch | [RODS (2026)](https://arxiv.org/abs/2606.19047): Qwen3-4B BFCL V3 multi-turn 56.00 vs 50.00 for static-data GRPO, comparable to a 17K-sample offline pipeline with about 20× fewer trajectories; [SENTINEL (2026)](https://arxiv.org/abs/2606.12908): tau2 Retail pass^1 66.4 → 74.9 |

The first four are **oracle-preserving**: the gold calls and final state do not change, only what the model is shown. For a saturated, already-verified tool set they are the cheapest wins (**Moderate**: several independent works, all evaluated mostly on BFCL and tau2).

### Architecture

```python
def synthesize_tool_task(env, stats, L):
    # 1. gold first: walk the tool graph, biased toward tools the policy fails (HardGen);
    #    3/2/1 are Agent-World's base edge weights; to harden, raise the weak/independent
    #    weights and max_len (fewer obvious output->input links); fill args from prior outputs or DB rows
    chain = sample_walk(env.tool_graph, start_bias=stats.failing_tools,
                        edge_weights={"strong": 3, "weak": 2, "independent": 1}, max_len=L)
    s0 = env.reset(seed)
    trace = execute(s0, chain)                 # every call executed immediately (SAP)
    assert trace.ok and trace.final_state != s0

    # 2. complexify the specification, not the answer
    spec = abstract(trace,
                    fuzz_ids=True,               # CoVe: DB-unique descriptions
                    implicit_steps=k,            # HardGen: hide intermediate calls
                    provenance={"prev_user_msg": 0.3, "prev_output": 0.4})  # SAP; mix is illustrative
    assert all(db_count(env, d) == 1 for d in spec.descriptions)

    # 3. write natural language last, from the frozen trace
    task = render_dialogue(spec, persona)
    check = lambda s: constraint_checklist(s, trace.final_state)  # any valid path; penalize redundant writes
    return Task(task, snapshot=s0, verify=check, gold=trace)
```

Then calibrate with k rollouts, keep the band, and send failures to a diagnosis step that decides the next operator ([SENTINEL, 2026](https://arxiv.org/abs/2606.12908); [Envs-FORGE, 2026](https://arxiv.org/abs/2608.14312)). DeepSeek packages each general-agent task as (database, tools, task, solution, verifier). The solution may call only tools, never the database directly. Its loop is "make harder → re-pass verifier", and it keeps only tasks with pass@100 > 0 ([DeepSeek-V3.2, 2025](https://arxiv.org/abs/2512.02556)).

### Verifier

- **Final-state checks, accepting any valid path.** CoVe counts a constraint as satisfied if any valid execution path reaches the intended state, and penalizes redundant operations ([CoVe](https://arxiv.org/abs/2603.01940)). Exact action matching favours one canonical path. Use it only for read-only chains, which a state diff cannot check ([AgentScaler, 2025](https://arxiv.org/abs/2509.13311)). **Strong.**
- **Code-augmented judges beat either extreme.** On 8B BFCLv3: LLM-only 55.46, code-only 60.00, code-augmented 65.94 ([AWM](https://arxiv.org/abs/2602.10090)). **Moderate.**
- **Pure LLM judges miss false success.** No configuration exceeded AUROC 0.65 on tau2-bench, and only 0.54 on AppWorld API traces; judges key on confident closing language ([2606.09863](https://arxiv.org/abs/2606.09863)). **Moderate.**

### Difficulty knobs

Walk length; number of recombined blueprints and policy constraints; number of distractor tools; share of implicit steps and fuzzed identifiers; share of arguments with long-range provenance; moving tools to the user side (dual control drops pass^1 by 18% for gpt-4.1 and 25% for o4-mini, [tau2-bench](https://arxiv.org/abs/2506.07982)); failure-injection rate and solvability scenario (retry / switch / stop); information withheld from the user simulator.

### Pitfalls

- **The user simulator is a reward-noise source.** CoVe's SFT+RL did *worse* than SFT alone, which the authors attribute to a weak open-weight simulator. Mitigations: fine-tune the user model before RL ([AReaL-SEA, 2026](https://arxiv.org/abs/2601.22607)); Best-of-4 plus self-critique ([APIGen-MT](https://arxiv.org/abs/2504.03601)); mask failures a judge attributes to the simulated user ([AutoForge, 2025](https://arxiv.org/abs/2512.22857)). **Moderate.**
- **LLM-simulated tools fail exactly where you complexify.** There is a "state-change cliff": near-perfect accuracy when no state changes, collapse when several variables must update at once ([EnvSimBench, 2026](https://arxiv.org/abs/2605.07247)). Keep state in code or a database and let the LLM render only surface text. **Moderate.**
- **Zero-variance groups move over time.** Disabling dynamic filtering cut Airline pass^1 from 70.5 to 65.0 ([AReaL-SEA](https://arxiv.org/abs/2601.22607)); see [Chapter 05](05-rl-playbook.md).
- **Diversity helps at scale, but naive domain mixing can hurt a small per-domain run.** Mixing domains lowered AReaL-SEA's 30B average from 71.5 to 63.7. Yet ScaleEnv's zero-shot scores rose steadily from 2 to 16 domains at a fixed 1,024 tasks ([ScaleEnv, 2026](https://arxiv.org/abs/2602.06820)), and Agent-World's four-domain average rose from 18.4% to 38.5% as environments grew from 0 to 1,978. Ablate your mixture. **Emerging** (conflicting).
- **Missing-information variants teach refusal if unbalanced.** Keep non-tool and irrelevant-tool cases: removing them dropped ToolACE's irrelevance detection to 6.99% ([ToolACE, 2024](https://arxiv.org/abs/2409.00920)).

### Starting datasets and environments

tau2-bench generator and domains ([2506.07982](https://arxiv.org/abs/2506.07982)); APIGen-MT (5K trajectories); CoVe (12K trajectories); AWM (1,000 SQL-backed MCP environments, 35,062 tools, code released); Agent-World (1,978 environments, 19,822 tools); EnvFactory (85 unit-tested environments); TOUCAN real-MCP data ([2510.01179](https://arxiv.org/abs/2510.01179)); ToolACE (26,507 APIs). Evaluate on BFCL v3/v4, ACEBench, tau2, MCP-Mark and MCP-Universe.

### Key references

| Work | What to copy | Verified result |
|---|---|---|
| [AReaL-SEA (2026)](https://arxiv.org/abs/2601.22607) | Per-instance `verify(final_state)`; user-model fine-tuning; dynamic filtering | Qwen3-235B tau2 Telecom 53.7 → 87.9 (SFT) → 98.3 (RL) |
| [Agent-World (2026)](https://arxiv.org/abs/2604.18292) | Weighted graph walks; programmatic tasks; diagnosis-driven regeneration | 14B after two rounds: tau2 45.3 → 50.5, MCP-Mark 29.5 → 38.1 |
| [AWM (2026)](https://arxiv.org/abs/2602.10090) | Scenario → schema → MCP tools → verification code | 8B: BFCLv3 53.83 → 65.94; tau2 26.44 → 33.45 |
| [SPADE (2026)](https://arxiv.org/abs/2608.19197) | Designer writes Gym environments; hint-based regret reward | BFCL v4 multi-turn +10.3 (4B), +5.7 (30B-A3B) |
| [2606.03800](https://arxiv.org/abs/2606.03800) | Gated synthetic augmentation of a tiny hand-authored base | 80 gated variants from 10 tasks match 97 hand-authored tasks |

---

## 2. Web search and deep research (BrowseComp-style)

**Seeds.** SimpleQA-style (question, short entity answer) pairs ([WebDancer E2HQA, 2025](https://arxiv.org/abs/2505.22648)); rare Wikidata entities ([WebSailor, 2025](https://arxiv.org/abs/2507.02592)); knowledge-graph random walks ([DeepDive](https://arxiv.org/abs/2509.10446)); 2WikiMultihopQA questions ([ProgSearch, 2025](https://arxiv.org/abs/2510.13913)); trend keywords ([QUEST, 2026](https://arxiv.org/abs/2605.24218)); a fixed corpus ([SAGE, 2026](https://arxiv.org/abs/2601.18202); [LiteResearcher, 2026](https://arxiv.org/abs/2604.17931)).

### Best operators

| Operator | Easy → hard | Keep it verifiable |
|---|---|---|
| **Entity → description substitution (fact injection, depth)** | "When was Michael P. Hein born?" → "When was the Eckerd College alumnus who served as the first County Executive of Ulster County … born?" ([ASearcher, 2025](https://arxiv.org/abs/2508.07976)) | Answer never changes; verify only the new hop; after each edit, run a no-tool solver and treat its wrong answers as candidate alternatives |
| **Fuzzing exposed constants** | IMF → "an international financial institution"; 1863 → "the second half of the nineteenth century"; 17,921 → "a five-digit prime whose digits sum to 20"; September 9 → "a date whose month and day use the same number" ([FORT](https://arxiv.org/abs/2606.12087)) | Uniqueness check after every fuzz (KB enumeration, alternative-answer review); repair over-fuzzed items rather than keeping them unsolved |
| **Clue removal (long-to-short)** | Delete "died at the age of 44", turn "Manchester United" into "a First Division giant" ([WebExplorer, 2025](https://arxiv.org/abs/2509.06501)) | Fixed answer; track a strong model's accuracy and tool turns per round; stop before uniqueness breaks |
| **Constraint intersection with leaf-first expansion** | "Which player played for club X?" → "a player who played for a club based in city Y between 2000 and 2010 and was born in a country bordering Z" | Enumerate the intersection to exactly one entity; expand *leaf* constants, never constants attached to the target ([WebShaper, 2025](https://arxiv.org/abs/2507.15061)); avoid over-determination where a subset already suffices ([InfoSeek, 2025](https://arxiv.org/abs/2509.00375)) |
| **Topology densification and evidence dispersion** | Chain (treewidth 1) → cycles or diamonds (2) → clique-like coupling (≥3); all clues in one infobox → clues spread over disjoint documents | Text derived from the graph; per-fact source attribution; compute Minimum Source Dispersion ([REDSearcher, 2026](https://arxiv.org/abs/2602.14234)); WL-distinct subgraphs and rotating answer roles ([WebSailor-V2, 2025](https://arxiv.org/abs/2509.13305)) |
| **Tool-grounded constraints** | "museum in city X" → "the city about two hours' drive west of [Entity A]"; "the scholar with approximately N citations" | Cache tool outputs at synthesis time; store tool-derived intermediates in the gold trace ([REDSearcher](https://arxiv.org/abs/2602.14234)) |
| **Aggregation layering** | "population of city A in 2020" → "among cities satisfying P, which had the largest 2010–2020 change, and what is the standard deviation of those changes?" | Compute with code from cached, attributed evidence; re-check every reference URL ([WebAggregator, 2025](https://arxiv.org/abs/2510.14438): QC removed 11.72%) |
| **Source masking and fictional worlds** | Delete the page the answer was extracted from ([LiteResearcher](https://arxiv.org/abs/2604.17931)); build 300–500-row fictional databases and serve snippets that "tease information without fully answering" ([Qwen-AgentWorld](https://arxiv.org/abs/2606.24597)) | Keep only questions still answerable via other pages; SQL-derived answers; unsearchability checks |

WebAggregator motivates aggregation layering: 30.79% of WebWalkerQA and 43.2% of TaskCraft tasks need only simple entity localization. For long-form reports, accumulate checkable facts into a **rubric tree with executable leaves** ([QUEST](https://arxiv.org/abs/2605.24218)), covered in §10.

### Architecture

```
seed answer / rare entity (degree window 4–8, long-tail)
   ▼ grow evidence graph: Wikidata relations + hyperlinks; densify cycles; add derived facts;
   │ store source URL per fact                         [WebSailor-V2, REDSearcher, FORT]
   ▼ sample WL-distinct subgraph; choose answer node by structural role
   ▼ withhold intermediate names; fuzz constants; inject tool-grounded constraints
   ▼ render question
   ▼ verifier cascade, cheap → expensive:
   │   1 no-tool LLM fails    2 answer retrievable (top-50 snippets)   3 facts consistent with sources
   │   4 oracle-evidence solver succeeds    5 ≥1 agent rollout succeeds   6 no plausible alternative answer
   ▼ adversarial refinement: strong agents attack; repair the earliest shortcut clue;
   │ narrow over-fuzzed clues on drafts nobody solves                              [FORT]
   ▼ accept by trajectory signature (solving cost Ω, answer-hit time T_hit, prior-shortcut rate)
```

The loop's stopping rule can be policy-relative: harden until the baseline agent fails ([ProgSearch](https://arxiv.org/abs/2510.13913)), until the agent actually needs at least S search steps ([SAGE](https://arxiv.org/abs/2601.18202)), or train a proposer whose reward is 1 − solver success with a RAG validity gate ([SSP, 2025](https://arxiv.org/abs/2510.18821)). In SSP a −0.1 penalty for invalid questions caused a "death spiral", so give invalid proposals zero reward. Self-play is covered in [Chapter 03](03-generation-architectures.md).

### Verifier

- **Two-sided gate (Strong).** The item must fail closed-book and succeed given the oracle evidence ([OpenSeeker](https://arxiv.org/abs/2603.15594); [InfoSeek](https://arxiv.org/abs/2509.00375) uses Gemini 2.5 Flash with gold pages plus distractors). "Fails GPT-4o with search in 4 of 4 tries" alone keeps ambiguous and unanswerable items ([DeepDive](https://arxiv.org/abs/2509.10446)).
- **Agentic re-verification of the RL set.** REDSearcher's Agent-as-Verifier pass cut the RL query set's error rate to 10% of the original (human evaluation).
- **Expect residual noise and audit a slice.** In QUEST, 2 of 50 audited evaluation scripts were non-executable and 6 had rubric errors; over 85% of REDSearcher items pass human fidelity review.

### Difficulty knobs and the "structure is not difficulty" rule

Knobs: hop count, treewidth, minimum source dispersion, number and strength of fuzzes, clue-removal rounds, share of tool-grounded constraints, and aggregation operator count. But **accept items on realized difficulty, not structure (Moderate, one careful cross-dataset study).** FORT ran 200 questions per dataset with the same agent and budget:

| Dataset | Ω (retrieval calls) | T_hit (answer first seen) | Prior-shortcut % |
|---|---|---|---|
| InfoSeek | 20.6 | 5.7 | 2.0 |
| DeepDive | 47.7 | 15.5 | 7.4 |
| DeepResearch-9K | 47.8 | 3.4 | 27.2 |
| OpenSeeker | 84.7 | 9.3 | 31.9 |
| REDSearcher | 92.1 | 18.7 | 11.8 |
| FORT | 141.0 | 46.9 | 11.0 |

FORT's cumulative ablation also ranks the controls. Removing them one at a time raises a strong agent's accuracy from 29.0 (full) → 36.5 (− cycles) → 42.7 (− long-tail) → 53.2 (− derived facts) → 57.4 (− source diversity) → 65.0 (− generic facts) → 81.6 (− fuzzing). **Fuzzing is the most important single control**, and hardening by *removal* (WebExplorer: Claude-4-Sonnet 86.6% → 67.1%) is a useful complement to adding hops.

### RL and SFT notes

- **Hard, verified SFT data goes far.** FORT-Searcher is SFT-only yet scores BrowseComp 72.2 and leads comparable open agents. OpenSeeker-v2 used 10.6k SFT samples to beat Tongyi DeepResearch's CPT+SFT+RL on BrowseComp (46.0 vs 43.4) ([OpenSeeker-v2, 2026](https://arxiv.org/abs/2605.04036)). Filtering out low-step trajectories is a cheap lever. **Moderate.**
- **Zero-variance groups need dynamic handling.** DUPO duplicates non-zero-variance samples and is about 2–3× faster than DAPO's dynamic sampling ([WebSailor](https://arxiv.org/abs/2507.02592)). Tongyi DeepResearch refreshes the active pool from a background "moderately difficult" pool ([2510.24701](https://arxiv.org/abs/2510.24701)). Query recycling supplies about three quarters of the effective batch by the end of training ([2606.10709](https://arxiv.org/abs/2606.10709)).
- DeepDive adds a query-redundancy penalty (Jaccard similarity of search queries, λ = 0.1) to multi-turn RL.

### Pitfalls

- **Obfuscation trades difficulty for ambiguity.** WebSailor admits non-unique answers; WebExplorer enforces uniqueness only by prompting. **Over-determination** is the silent failure: one leftover precise clue turns a nominal 6-hop item into a 1-hop lookup ([InfoSeek](https://arxiv.org/abs/2509.00375)).
- **Injected facts can themselves be shortcuts.** A highly identifying injected description (ASearcher, E2HQA) can let the model name the answer without searching. Log a no-evidence answer probe for every item.
- **Live-web answers drift.** Statistics are brittle to source updates ([WebAggregator](https://arxiv.org/abs/2510.14438)); evolving benchmarks exist for this reason ([EvoBrowseComp, 2026](https://arxiv.org/abs/2606.13120)). Snapshot evidence, or train on an offline Wikipedia environment ([WebSailor-V2](https://arxiv.org/abs/2509.13305)) or a local corpus ([LiteResearcher](https://arxiv.org/abs/2604.17931): 73.2M tool calls without commercial APIs).

### Starting datasets and environments

SailorFog-QA; WebShaper (5,000 trajectories); InfoSeek (52,138 samples for $571.8); WebExplorer (about 40K QA); DeepDive (3,250 QA); ASearcher (25,624 selected QA); OpenSeeker (11.7k samples, fully open); WebAggregatorQA (9,883 tasks); Quest-8K; REDSearcher (release planned: 10K text and 5K multimodal trajectories, 1K RL queries); ORBIT (20K verified 4–5-step queries without paid APIs, [2604.01195](https://arxiv.org/abs/2604.01195)). Evaluate on BrowseComp, BrowseComp-ZH, BrowseComp-Plus, GAIA, xbench-DeepSearch, HLE, FRAMES and WebWalkerQA.

### Key references

| Work | Verified result |
|---|---|
| [FORT (2026)](https://arxiv.org/abs/2606.12087) | FORT-Searcher (Qwen3-30B-A3B, SFT only): BrowseComp 72.2, BrowseComp-ZH 75.0 |
| [REDSearcher (2026)](https://arxiv.org/abs/2602.14234) | BrowseComp 39.4 (SFT) → 42.1 (RL); DeepSeek-V3.2 about 40% average@4 on its items |
| [QUEST (2026)](https://arxiv.org/abs/2605.24218) | Quest-35B: BrowseComp 45.5 (64.6 with discard-all context); DeepResearch Bench 48.2 |
| [WebSailor-V2 (2025)](https://arxiv.org/abs/2509.13305) | 30B-A3B: BrowseComp-EN 35.3, BrowseComp-ZH 44.1 |
| [OpenSeeker (2026)](https://arxiv.org/abs/2603.15594) | 11.7k SFT samples: BrowseComp 29.5% vs 15.3% for DeepDive |
| [InfoSeek (2025)](https://arxiv.org/abs/2509.00375) | InfoSeeker-3B: 16.5% on BrowseComp-Plus |
| [SSP (2025)](https://arxiv.org/abs/2510.18821) | Qwen2.5-7B-Base from scratch: +26.4 average; NQ 32.0 → 54.2 |

---

## 3. GUI, computer-use and mobile agents

**Seeds.** AndroidWorld's 116 parameterized templates across 20 apps ([AndroidWorld, 2024](https://arxiv.org/abs/2405.14573)); OSWorld tasks and their atomic evaluator functions ([OSWorld 2.0 and predecessors](https://arxiv.org/abs/2606.29537)); WorkArena's 33 atomic ServiceNow tasks with oracles and validators ([WorkArena++, 2024](https://arxiv.org/abs/2407.05291)); tasks reverse-synthesized from exploration ([OS-Genesis, 2024](https://arxiv.org/abs/2412.19723); [Go-Browse, 2025](https://arxiv.org/abs/2506.03533)).

**Step zero: convert every seed into an RLVR tuple** (instruction, reproducible initial state, executable reward) and accept it only after the checker has run: reward(golden) = 1 and reward(initial) = 0 ([CUA-Gym](https://arxiv.org/abs/2605.25624)). Every operator below then reuses that checker. **Strong** (CUA-Gym, Qwen-CUA, SCALECUA, EvoCUA and OpenComputer all converge on this).

### Best operators

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Evaluator-first composition** | `file_exists(~/Documents/tutorial.pdf)` → `file_exists(...) ∧ url_visited(docs.python.org/...)`, rendered as "Navigate to the Python documentation page and download the PDF tutorial to your Documents folder" | Sample 2–3 trusted atomic checkers, reparameterize, build a golden state proving joint satisfiability, *then* write the instruction | [UltraCUA (2025)](https://arxiv.org/abs/2510.17790): composed-checker tasks 29% rollout success vs 45% for instruction-first |
| **Chaining verified tasks (phase-state or compatibility-checked)** | Phase-state: "Import contacts.csv" → import → deduplicate by email → build a "vendor" list → send each vendor its outstanding amount (illustrative). Compatibility-checked: "Change the slide background" and "Export the deck to PDF" plus further atomic tasks → a length-4, 3-app chain | Phase-state: each phase's validated end state is serialized as the next phase's initial state, and later checkers must not depend on artifact IDs one particular solution created. Chaining existing atomic tasks: hard rules first (snapshot mismatch, destroyed state, missing artifacts, evaluator interference), coherence judge last; keep per-subtask evaluators | [Qwen-CUA (2026)](https://arxiv.org/abs/2608.02352): pipeline of about 40,000 tasks reaches 86.2 on OSWorld-Verified (no per-operator ablation); [ChainWorld (2026)](https://arxiv.org/abs/2606.21654): best of four agents completes 31% of chains |
| **Explicit → implicit goals** | Step list in the ticket (L2) → "onboard a new employee", procedure in the knowledge base (L3) | Workflow and validators identical across levels | [WorkArena++](https://arxiv.org/abs/2407.05291): GPT-4o 3.0% at L2, 0% at L3 |
| **Start-state regression** | "Add this product to the cart", starting on the product page → same goal from the site root | Checker unchanged; add identifying attributes if start-page context disappears | [Go-Browse](https://arxiv.org/abs/2506.03533) (unprefixed rollouts) |
| **Inherit / merge / rewrite of solved tasks** | "Create a note titled Groceries" → inherit: "add three items and pin the note"; merge: "…and share it with Anna by SMS"; rewrite: new parameters | Reuse the solved task's goal-state screenshot as the reference for its descendants | [GSAR (2026)](https://arxiv.org/abs/2608.22847): removing the goal-state reference drops judge accuracy from 90.2 to 64.4 |
| **Template parameterization + noise entities + state enrichment** | "Delete expense X" on a clean app → the same with near-duplicate noise rows the checker requires untouched; a clean 10-row sheet → a coherent multi-app state (emails, calendar, files) that must be searched and reconciled | Write set-up and checker once against the state store (`sql_rows_exist` / `!sql_rows_exist`); generate state from a structured entity model so answers are computed | [AndroidWorld](https://arxiv.org/abs/2405.14573); [MAI-UI (2025)](https://arxiv.org/abs/2512.22047) L1/L2 expansion; [Qwen-UI-Agent (2026)](https://arxiv.org/abs/2607.28227): state synthesis "largely determines the difficulty and diversity" |
| **Runtime perturbation, infeasibility and hidden information** | Clean emulator → pop-up at step 2, forced app switch at step 4; "Set an alarm … using the Smart-Wake feature" (no such feature, gold = FAIL); "Book a room for the design review" with date and attendee count held by a simulated user | Checker unchanged; re-verify solvability after injection; make infeasibility true by construction; the end state must be a deterministic function of the simulator's facts | [AnTrap (2026)](https://arxiv.org/abs/2608.24099): all 16 models degrade; [ZeroGUI (2025)](https://arxiv.org/abs/2505.23762): removing infeasible tasks drops the infeasible subset from 41.3 to 22.1; [OSWorld 2.0](https://arxiv.org/abs/2606.29537) |
| **Failure-seeded refill** | After each iteration: harder extensions of tasks above 0.9 pass rate, simpler variants of tasks stuck at 0 | Regenerate or re-test the checker for every child; re-filter to the band | [SCALECUA (2026)](https://arxiv.org/abs/2607.11185): trajectory-guided augmentation takes OSWorld 64.6 → 68.7; [WebRL (2024)](https://arxiv.org/abs/2411.02337) |

**Do not ask an LLM directly for "hard" GUI tasks (Moderate).** In AgentSynth's ablation, one-shot "hard" generation had 11% generation success but surviving tasks were still solved 48% of the time at evaluation. Information-asymmetric chaining of k verified subtasks, summarized without the procedure, gave level-6 tasks at 52% generation success and 14% evaluation success ([AgentSynth, 2025](https://arxiv.org/abs/2506.14205)). The generator solves only short steps forward, while the learner must infer the whole plan.

### Architecture: separate the solution-builder from the checker-writer

```
spec from taxonomy (app × capability × difficulty; CUA-Gym: 45% hard, 38% cross-app)
  ├─► Generator VM:     initial_setup.py, golden_patch.py
  └─► Discriminator VM: reward.py   (INFORMATION BARRIER: sees task text + state views only)
accept iff  C1 setup runs   C2 golden runs   C3 reward(golden)=1   C4 reward(initial)=0
            C5 no forbidden pattern (constant flags like `chart_verified = True`,
               placeholder checks, bare file-existence scoring)
  ▼ cross-family LLM majority vote
  ▼ teacher rollouts scored by reward.py AND a checklist VLM judge:
      both means in (0,1) and agree → accept      both 0 → drop (plausibly unsolvable)
      both 1 on first try → down-sample            disagree → back to Discriminator for repair
```

When one agent wrote both sides, the reward "tends to re-check the construction procedure instead of measuring task completion" ([CUA-Gym](https://arxiv.org/abs/2605.25624)). The barrier and the static forbidden-pattern scan transfer to code, SQL, spreadsheet and tool domains. **Moderate** (one large study; the three-role proposer / static judger / Docker dry-run checker of [SCALECUA](https://arxiv.org/abs/2607.11185) is an independent variant).

**White-box environments make checking nearly free once the environment itself is verified.** Examples: finite-state-machine sites with ground truth from graph search and Playwright replay, at about $0.04 per verified trajectory ([AutoWebWorld, 2026](https://arxiv.org/abs/2602.14296)); assertions compiled into generated app code ([GUI-Genesis](https://arxiv.org/abs/2602.14093)); mock Android apps with co-generated SQLite predicates ([PhoneWorld, 2026](https://arxiv.org/abs/2605.29486)). Raw LLM-built websites block half their own tasks: only 48.6% of tasks admit a bounded executable trace, rising to 94.8% after verify-and-repair ([Verified Synthetic Web Environments, 2026](https://arxiv.org/abs/2608.21898)). **Verify the environment before training on it.**

### Verifier

| Verifier type | Measured quality | Failure direction |
|---|---|---|
| VLM judge, best configuration (all screenshots, unanimous vote) | 61.5% precision ([ZeroGUI](https://arxiv.org/abs/2505.23762)) | Over-accepts; false positives hurt RL more than false negatives |
| Same policy, VLM judge vs code assertions | 63.76% vs 38.93% ([GUI-Genesis](https://arxiv.org/abs/2602.14093)) | Judge inflates success |
| Unvalidated model-written end-state scripts | 43.3% human agreement ([Gym-Anything, 2026](https://arxiv.org/abs/2604.06126)) | Fail to parse output formats |
| Checklist VLM with privileged set-up facts | 93.3% ([Gym-Anything](https://arxiv.org/abs/2604.06126)) | — |
| Endpoint-backed checkers after self-repair | 94.1% vs 79.2% for an agentic LLM judge ([OpenComputer, 2026](https://arxiv.org/abs/2605.19769)) | Judge misses tiny errors ("alpha beta" typed in one cell) |
| Propose-then-probe agentic verifier | 86.9% accuracy; RL reward 34.0% vs 34.9% with scripts ([IRA, 2026](https://arxiv.org/abs/2607.25904)) | Residual: "right setting family, wrong tier", unsaved edits |
| Hand-written OSWorld-Verified scripts | No false positives found; errors were false negatives from alternative paths ([VAGEN, 2026](https://arxiv.org/abs/2602.00575)) | Under-accepts |

Rule: **prefer state evidence; when a judge is unavoidable, give it state, privileged set-up facts, or an anchored reference, and hide the agent's self-report** (adding the self-report dropped ZeroGUI's judge precision to 44.3). **Strong.** "Executable is not correct": SCALECUA's judges are 94.5% executable but agree with experts only 78% of the time on OSWorld.

### Difficulty knobs and scheduling

- **Bands in use:** WebRL keeps critic scores in [0.05, 0.75]; UltraCUA samples pass rates in [0.4, 0.8]; Qwen-CUA keeps 1–7 successes of 8, recomputed after each SFT refresh; SCALECUA weights tasks by a Gaussian (μ = 0.5, σ = 0.25) over an EMA of pass rate (α = 0.2) and reserves 20% for uniform sampling; MAI-UI uses four pass@K bands.
- **0/K tasks:** MobileRL removes them after a cooldown (removing that filter costs 6.3 points, [MobileRL, 2025](https://arxiv.org/abs/2509.18119)). Qwen-UI-Agent keeps them in a low-budget *monitoring pool* and promotes them on first success. Prefer the monitoring pool (**Emerging**), and consider hindsight relabeling of their failures ([AgentHER, 2026](https://arxiv.org/abs/2603.21357): 97.1% human-rated precision with two cross-family judges).
- **Keep easy anchors in the mix.** WebGym (Qwen3-VL-8B, unseen websites):

  | Sampling mix (easy:medium:hard) | Step budget | Peak success |
  |---|---|---|
  | Hard-biased 2:5:3 | 15/30/45 | 34.5 |
  | Easy only | 15/30/45 | 36.9 |
  | Natural (≈25:5:1) | 15/30/45 | 38.2 |
  | Natural (≈25:5:1) | 10/20/30 | 42.9 |

  But ordering matters: removing MobileGUI-RL's easy-to-hard curriculum dropped AndroidWorld from 44.8 to 34.0 at 32B ([MobileGUI-RL, 2025](https://arxiv.org/abs/2507.05720)), and MAI-UI's curriculum plus repetition penalty plus replay turned a +1.8 GRPO gain into +6.0. The step-budget direction is also setup-dependent: growing the horizon helped TTI ([2506.07976](https://arxiv.org/abs/2506.07976)) and MAI-UI, while tightening it helped WebGym. **Emerging** (conflicting evidence; tune both ways).

### Pitfalls

- **GUI tasks leak to non-GUI shortcuts.** About 15% of OSWorld tasks need only a terminal, and another 30% can largely be done with scripts ([Epoch AI analysis](https://epoch.ai/blog/what-does-osworld-tell-us-about-ais-ability-to-use-computers)). If the target is GUI skill, block shortcut channels or add integrity items (though Gym-Anything's integrity checks changed no pass result in its evaluation).
- **Compose semantically, do not concatenate.** OSWorld 2.0 rejected LLM proposals as "shallow workflows that compose unrelated operations"; Gym-Anything bans "artificially hard" chains.
- **Feasibility is not usefulness.** The most executable generator in AutoPlay's ablation (56.4% executor success) produced the weakest agent (21.6 vs 38.2 AndroidWorld pass@1) ([AutoPlay, 2025](https://arxiv.org/abs/2509.25047)). Score generators by downstream gain.
- **Single-run gains are noisy.** In CUA RL the data draw accounts for 48% of variance on the hardest cell, and a published-size gain has the wrong sign a third of the time ([2607.17136](https://arxiv.org/abs/2607.17136)); AndroidWorld seeds alone swing results by about 7 points. Report several seeds.
- **Contamination.** ZeroGUI runs test-time RL on test instructions; MobileRL trains on AndroidWorld's own templates. Hold out *environments* (WebGym splits by website).

### Starting datasets and environments

OSWorld / OSWorld-Verified / OSWorld 2.0; AndroidWorld; WebArena-Lite; WorkArena++ (682 tasks); CUA-Gym (32,112 verified tuples over 110 environments, including 94 mock web apps); AgentSynth (6,000+ tasks, used as an RL pool by VAGEN); UltraCUA (17,000+ verifiable tasks); WebGym (292,092 tasks over 127,645 websites); AutoWebWorld (11,663 trajectories); PhoneWorld (34 mock apps); VeriEnv (149 cloned sites, 7,400 tasks, [2603.10505](https://arxiv.org/abs/2603.10505)).

### Key references

| Work | Verified result |
|---|---|
| [CUA-Gym (2026)](https://arxiv.org/abs/2605.25624) | OSWorld-Verified: Qwen3.5-35B-A3B 54.5 → 62.1; Qwen3.5-397B-A17B 62.2 → 72.6 |
| [Qwen-CUA (2026)](https://arxiv.org/abs/2608.02352) | 86.2 OSWorld-Verified; OSWorld 2.0 binary/partial 2.5/22.5 → 18.5/48.4 |
| [SCALECUA (2026)](https://arxiv.org/abs/2607.11185) | Qwen3.5-9B: OSWorld 68.7, ScienceBoard 54.0; without VeriGen 43.9 |
| [WebRL (2024)](https://arxiv.org/abs/2411.02337) | Llama-3.1-8B WebArena-Lite 4.8% → 42.4% |
| [MobileRL (2025)](https://arxiv.org/abs/2509.18119) | 80.2% AndroidWorld, 53.6% AndroidLab (9B) |
| [PhoneWorld (2026)](https://arxiv.org/abs/2605.29486) | Half of a matched RL budget on mock apps: AndroidWorld 77.2% → 83.2% |
| [WebGym (2026)](https://arxiv.org/abs/2601.02439) | Qwen3-VL-8B out-of-distribution 26.2% → 42.9% |

---

## 4. Terminal/CLI, data-analysis, spreadsheet and office agents

### 4a. Terminal and CLI

**Seeds.** Tasks reverse-engineered from real terminal recordings ([TerminalWorld, 2026](https://arxiv.org/abs/2605.22535): 1,530 validated tasks); cheap first-generation pools such as Endless Terminals, on which Gemini-3-Flash scores 92% pass@1 ([Endless Terminals, 2026](https://arxiv.org/abs/2601.16443); [Tmax, 2026](https://arxiv.org/abs/2606.23321)); repository-grounded environments ([TerminalTraj, 2026](https://arxiv.org/abs/2602.01244)). Package each seed in the Harbor format: `instruction.md` + `Dockerfile` + `solution/solve.sh` + tests + `task.toml`. These are your easy seeds, not your training set.

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Solution-first workflow extension, recursively reseeded** | "Count 5xx lines in `access.log`, write the number to `out.txt`" → after 3 rounds: "decompress rotated `.gz` logs, merge with the live log, dedupe by `request_id`, compute per-endpoint 5xx rates, write `report.json` and a sha256 manifest" | Stage order: solution → environment → verifier → instruction. Fresh-sandbox oracle gets full reward; *contract validity* (every checked requirement stated or discoverable); ≥4 checks (evidence discovery, intermediate state, final semantics, shortcut rejection); minimum deltas (≥3 files, ≥8 solution lines, ≥12 verifier lines); instruction ≤180 words and ≤1.6× the seed; caps per parent/category/family | [RST (2026)](https://arxiv.org/abs/2608.05466): DeepSeek-V4-Pro pass@4 90% → 2.5% over 15 rounds from 639 seeds, about 50% acceptance every round, about $0.05 per accepted task |
| **Pass-rate-banded evolution** | r > 0.5 → add constraints and edge cases; 0 < r ≤ 0.5 → context shift at matched difficulty (sklearn → caret, systemd → openrc); r = 0 → simplify | Re-run no-op (0 tests pass) and oracle (all pass) tests on every child; re-measure r | [SETA (2026)](https://arxiv.org/abs/2607.10891): "decrease" moved as declared 77% of the time, "increase" only 60% |
| **Solver-relative calibration** | A validated task all solvers pass → revised with a version-specific dependency pitfall the strong solver handles and the weak one does not | Retention needs ≥1 verified solver success; weak-pass/strong-fail triggers a leakage or nondeterminism check; for RL, make the weak solver your checkpoint | [CalibForge](https://arxiv.org/abs/2608.06352): 19% → 96% learnable; matched 1,300-task ablation on TB2.0: no solver 22.47, contrastive pair 31.09 |
| **Skill-graph and cross-workspace composition** | Single skill → dependency-ordered path of up to 7 skills; one repository → "read a reference implementation in workspace B and port the feature to A" | Oracle's minimal solution must traverse the path; agent-written in-container verifier, all tests must pass | [SkillSynth (2026)](https://arxiv.org/abs/2604.25727): 38% of tasks unsolved in 3 tries vs 16% for single skills; [Terminal-Universe (2026)](https://arxiv.org/abs/2609.04148): teacher pass@1 72.3% → 49.2%, 1.9× tool calls |
| **Environment inversion and symptom-only framing** | "Install package X" → an agent-corrupted environment (mis-pinned transitive dependency, missing env var, broken permissions) where 12 unit tests fail; "set PORT=8080" → "the service returns 502; diagnose and write an evidence bundle" | Gold environment passes its tests before inversion; drop recoveries through cached Git/Conda state; issue text at 3 guidance levels, hint removable | [CLI-Gym (2026)](https://arxiv.org/abs/2602.10999): 291 curated trajectories give +21.1 on TB1.0 and +12.9 on TB2.0 |
| **Graded and threshold verifiers** | Binary "model file exists" → accuracy ≥ 0.95 on a hidden set; exact text → fuzz-equivalence against an oracle | Thresholds calibrated on real baseline runs; fixed normalizer | [Tmax](https://arxiv.org/abs/2606.23321) verifier kinds; [SandMLE (2026)](https://arxiv.org/abs/2604.04872) milestones |

**Harden executable artifacts, not prose (Moderate).** OpenThoughts-Agent found LLM rewrites of task descriptions ("combine", "add constraints", "harden") did not beat untouched descriptions at fixed data size. In RST, hardness came from work: instructions grew only ×1.4 while solutions grew ×5.6 and commands ×6.1.

**Integrity before RL (Strong).** 16% of 1,968 terminal-benchmark tasks were hackable from the task description alone; a hacker–fixer–solver loop cut Gemini 3.1 Pro's attack success on Terminal Bench from 39% to 17% ([Hacker–Fixer loop, 2026](https://arxiv.org/abs/2606.08960)). Add: all tests must fail in the initial state (CalibForge); a Trajectory Judge for tasks every rollout fails (SETA dropped about 2% as design flaws); read-only, checksummed tests (T1's verifier ran inside the agent sandbox with no tamper detection). T1 also audits the pool before RL, hard-rejecting hidden requirements, test leakage, solution shortcuts and weak verifiers; its audited 15k pool beat the unfiltered 38k pool under dense-reward PPO (64.0 vs 59.9 on TB2.1).

**Change the estimator with the data (Emerging, one study).** On the escalated pool, T1 reports:

| Campaign (Qwen3.5-122B-A10B) | TB2.1 |
|---|---|
| RST-SFT checkpoint | 49.4 |
| Binary-reward PPO from base (TMax-15k) | 47.2 |
| GRPO on audited T1-15k | 51.7 (flat) |
| Dense r = P/20 (passing assertions on a fixed scale), PPO with warm-started critic, audited pool | 64.0 |

A cheaper halfway step: SETA's fraction-of-tests reward with a +0.2 all-pass bonus raised the share of tasks with non-zero reward variance from 49% to 86%. Count rewards invite prolongation (T1 saw runaway turn growth at 27B), so keep the normalizer fixed and monitor turns. See [Chapter 05](05-rl-playbook.md).

### 4b. Data analysis, notebooks and ML engineering

- **Compose, but label by execution.** DataMind chains 2–5 analytic task types, feeding each output in as the next input. It labels by agreement of 3 trajectories, which the authors say "inherently biases us toward easier queries" ([DataMind, 2025](https://arxiv.org/abs/2509.25084)). Fix: compile the composite into an executable reference pipeline and take the gold from running it (**Proposal**; not yet reported at scale).
- **No-data shortcut filter.** Drop a task if ≥3 of 5 LLMs answer it without the data files. Withholding data cost only 40.5% of performance on QRData on average. DSGym-SFT, 2,000 execution-verified synthetic queries, took Qwen3-4B on DABStep-hard from 2.9% to 33.07% ([DSGym, 2026](https://arxiv.org/abs/2601.16344)). **Moderate** (the DABStep-hard jump deserves replication).
- **Micro-scale re-skinning.** Keep a seed's structural "DNA", move it to a new domain, generate 50–200 samples from a code-defined hidden rule plus noise, and set reward milestones from baselines. Rollouts are >13× faster, making on-policy RL feasible; any-medal rate on MLE-bench-lite improves 20.3–66.9% relative over SFT baselines ([SandMLE](https://arxiv.org/abs/2604.04872)).
- **Multi-turn analytical state.** Turn single-shot data-QA into sessions with inheritance, update, counterfactual and rollback turns, each with replayable reference code. Accuracy drops about 47 points from early to late turns ([LongDS-Bench, 2026](https://arxiv.org/abs/2605.30434)); no training pipeline yet generates such sessions.
- **Executed notebooks are verified seeds** because the notebook code is the reference program ([Jupiter, 2025](https://arxiv.org/abs/2509.09245)). Mined QA alone skews trivial: the HF Jupyter agent raised DABStep-easy to 75% while the hard split stayed near 3% ([Jupyter Agent blog](https://huggingface.co/blog/jupyter-agent-2)). Never let an LLM "simulate" execution for gold labels.

### 4c. Spreadsheets and office documents

| Operator | Easy → hard | Keep it verifiable | Source |
|---|---|---|---|
| **Workbook inversion + lower specificity** | Remove 5 derived artifacts (derived column, SUMIFS summary, pivot, chart, conditional formatting) along a dependency-consistent history; L1 query (357 chars on average) → L3 (139 chars) | The original workbook is the oracle; per-artifact grading schema | [WTM (2026)](https://arxiv.org/abs/2608.07873): scores fall monotonically L1 → L3 and collapse at ≥5 transformations |
| **Deeper formula chains, more sheets** | `=SUM(B2:B13)` → a 3-sheet DCF with a two-way sensitivity table and a pivot | Recalculate in the *same engine* the oracle used; check formulas exist rather than pasted values; check placement and shape | [MBABench (2026)](https://arxiv.org/abs/2605.22664) rubric axes; [Spreadsheet-RL (2026)](https://arxiv.org/abs/2605.22642) |
| **Multiple test-case workbooks** | One fixed cell checked → correct on several (SpreadsheetBench: about 3) workbook variants with perturbed values | Recompute each oracle with the reference program; rejects hard-coded answers | [SpreadsheetBench (2024)](https://arxiv.org/abs/2406.14991): 2,729 test-case workbooks for 912 instructions |
| **Edit + preservation coupling (L1 → L4)** | Atomic edit → compositional edit → single-document workflow → cross-document workflow | Deterministic in-container predicates: structural, linguistic anchors, and *preservation* (protected styles, untouched sheets unchanged) | [DocOps (2026)](https://arxiv.org/abs/2607.19865): best configuration 0.671, near zero on coupled long-range Excel workflows |

Two verification facts matter most here. **Engine fidelity:** Spreadsheet-RL keeps real Microsoft Excel as the source of truth because LibreOffice and headless Python engines lack functions (dynamic arrays such as FILTER, UNIQUE, SORT) and differ subtly; an oracle recomputed in the wrong engine is a silent label error. **Never trust self-report:** Claude rollouts claimed success despite strict failure 80.7% of the time on WTM (GPT 43.2%), and 88.5% of failed L3 rollouts were structural (wrong shape, missing artifact or wrong placement), not wrong values.

**The biggest open gap in this section (Proposal):** no published pipeline applies RST-style recursive escalation to workbooks, notebooks or documents, although every component exists.

```python
def escalate_workbook(seed, policy):              # Proposal: untested
    prog = seed.reference_edit_program            # Excel-API edit script (Spreadsheet-RL style)
    op = pick(["add_dependent_sheet", "deepen_formula_chain", "add_pivot", "add_chart",
               "cross_workbook_lookup", "inject_logged_dirty_rows"], by=policy.pass_rate(seed))
    prog2 = op.extend(prog)                       # grow the SOLUTION first (RST)
    variants = perturb_input_values(seed.workbook, n=3)          # SpreadsheetBench
    golds = [run_in_real_excel(prog2, v) for v in variants]      # engine fidelity
    checks = region_values(prog2) + formulas_exist(prog2) + preserved(seed)   # DocOps
    instr = abstract_instruction(prog2, specificity=pick([1, 2, 3]))          # WTM
    assert contract_valid(instr, checks)          # every checked item stated or discoverable
    assert all(not passes(noop_output(v), g) for v, g in zip(variants, golds))
    return Task(instr, variants, golds, checks)
```

For office deliverables with no verifier, NexForge's SFT raised GDPval Elo from 813 to 1338 ([NexForge, 2026](https://arxiv.org/abs/2607.14186)), but such data cannot feed RLVR directly. Carve out an execution-verifiable subset for RL.

### Difficulty knobs

Recursion round (RST); operator chosen by the policy's pass-rate band (SETA); the strong/weak solver pair (CalibForge); skill-path length 1–7 (SkillSynth); workspaces spanned and follow-up rounds (Terminal-Universe); issue guidance level and hint on/off (CLI-Gym); complexity-bucket weights and verifier thresholds (Tmax); number of artifact transformations and query specificity L1–L3 (WTM); workflow level L1–L4 and edit coupling (DocOps); hidden-rule complexity and noise level (SandMLE); composed task types per data query, 2–5 (DataMind). Measure each on the current policy: SETA's "increase" operator moved difficulty as declared only 60% of the time.

### Pitfalls

- **Recursive escalation homogenizes and drifts.** In RST, within-round nearest-neighbour similarity rose from 0.223 to 0.464 and lexical distance from the benchmarks grew every round; late rounds give almost no binary-reward signal. Keep real seeds (TerminalWorld recordings) in the mix and cap per parent and family.
- **Oracle and tests written by one agent share unstated assumptions.** Audit tasks that every rollout fails (SETA dropped about 2% as design flaws) and require contract validity (RST).
- **Imitating raw source trajectories can hurt.** Terminal-Universe's SFT on raw public trajectories scored below the base model (36.7 vs 47.0); re-solving the recovered intent gave 52.1.
- **Pool skew.** Tmax counts over 60% of CLI-Gym and TerminalTraj as software engineering; T1's failures concentrate in data-science categories that are only 3.7% of its pool. Balance categories before scaling.

### Starting datasets and environments

Terminal: TerminalWorld (1,530 validated tasks from real recordings), Endless Terminals (3,255 tasks), Tmax (14,600), SETA (4,567 environments), CalibForge (5,431), SkillSynth (3,560 usable tasks), CLI-Gym (1,655), TerminalTraj (50,733 verified trajectories), RST (37,484 tasks), Terminal-Universe (37.3k environments). Data and office: DSGym-SFT (2k), DataMind-12K, NbQA ([Jupiter](https://arxiv.org/abs/2509.09245)), Spreadsheet-RL (5,928 tasks), SpreadsheetBench (912 instructions, 2,729 test-case workbooks), WTM-Corpus (8,931 queries over 2,977 tasks). Evaluate on TB2.0/2.1, TB-Hard, LHTB, DABStep, QRData, SpreadsheetBench, DocOps, MLE-bench-lite and GDPval.

### Key references

| Work | Verified result |
|---|---|
| [RST (2026)](https://arxiv.org/abs/2608.05466) | SFT on 3 stages: TB2 41.2 → 47.9 (Qwen3.5-27B); TB-Hard 20.0 → 30.0 (122B-A10B) |
| [T1 (2026)](https://arxiv.org/abs/2609.11042) | TB2.1 43.8 → 49.4 (SFT) → 64.0 (dense-reward PPO) |
| [CalibForge (2026)](https://arxiv.org/abs/2608.06352) | CalibForge-35B-A3B 47.57% on TB2.0 |
| [Terminal-Universe (2026)](https://arxiv.org/abs/2609.04148) | Qwen3.5-27B TB2.1 46.2 → 56.4 → 58.4 with cross-workspace tasks |
| [SkillSynth (2026)](https://arxiv.org/abs/2604.25727) | Qwen3-32B SFT on TB2.0: single-skill 21.3, SkillSynth 29.6 |
| [Spreadsheet-RL (2026)](https://arxiv.org/abs/2605.22642) | Qwen3-4B-Thinking SpreadsheetBench 12.0% → 23.4% |
| [DSGym (2026)](https://arxiv.org/abs/2601.16344) | Qwen3-4B DABStep-hard 2.9 → 33.07 with 2k SFT queries |

---

## 5. Instruction following with composable constraints

**Seeds.** Easy prompts (Alpaca, ShareGPT, WildChat, Tulu SFT prompts); a library of (natural-language constraint, verifier function, unit tests) triples, extended well beyond the 25 IFEval types; strong responses for back-translation.

### Best operators

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Stacking calibrated by p^k** | "Write a poem about autumn." → "…exactly 4 stanzas, all lowercase, include 'ember' exactly twice, no commas, end with a question, title wrapped in <<>>" | One code checker per constraint plus a compatibility checker (pair blocks, arithmetic feasibility, propagation); expect rejection to rise from about 32% at k=4 to about 98% at k=12 | [CSE (2026)](https://arxiv.org/abs/2608.12426): mean per-constraint success 72.0% × 0.922^(k−1); at k=8 about 41% per constraint → 5.7% all-pass |
| **Structural composition (Chain / Selection / Nesting)** | "Describe this product." → "If the listed price is >$100, write a JSON review {pros, cons, verdict}; otherwise a ≤280-character post with exactly 2 hashtags" | Make the branch condition *computable from the input*; reward only the active branch; decay downstream credit after an upstream failure | [ComplexBench (2024)](https://arxiv.org/abs/2407.03978), GPT-4: And 0.881, Chain 0.766, Selection+Chain at depth ≥3 0.626; 14.9% on the coherent multi-layer Selection test |
| **Scoped constraints (Scope × Target × Range)** | "Every paragraph under 60 words." → "In section 2 only, every bullet starts with a verb; bullets under 'Risks' are numbered and each contains a percentage" | Parse the output into a tree; the judge writes and runs an extraction tool; reward exp(−α·deviation/scale) | [ScopeIF (2026)](https://arxiv.org/abs/2609.32189): Qwen3-8B IFBench 31.0 → 53.7 |
| **Program-derived procedures** | "Sort these numbers." → an anonymized Codeforces simulation described in words; report the output *and* the maximum queue size | Execute the anonymized code for outputs and tracker values; difficulty = 3·nesting + 2·calls + cyclomatic + 0.5·lines | [LogicIF (2025)](https://arxiv.org/abs/2508.09125): GRPO on Qwen3-1.7B, LogicIFEval +16.7, ZebraLogic +31.4 |
| **Implicit-parameter constraints** | "Use 5 bullets." → "Use as many bullets as there are prime numbers below the number of letters in the capital of Australia" | Build a DAG of knowledge, math and condition nodes with objective values; execute it for the target; check with code | [ImpRIF (2026)](https://arxiv.org/abs/2602.21228): +7.3 / +9.3 / +9.9 average on five external benchmarks (4B/8B/32B) |
| **Back-translation with anti-copy guards** | (easy prompt, 280-word answer) → prompt + 5–15 properties measured from the answer by code | Correct by construction; derive constraints shared by ≥2 references, harden trivially satisfied ones, verify on a summary of the output | [Crab (2024)](https://arxiv.org/abs/2410.24175); [VerIF (2025)](https://arxiv.org/abs/2506.09942); [UNSPECIFIC (2026)](https://arxiv.org/abs/2608.09154): GPT-5 Mini satisfaction 90% → 78% |
| **Policy-adaptive appending** | A prompt where 8/8 rollouts pass → the same prompt plus 1–3 new atomic constraints, until pass@1 ∈ (0, 0.5] | Append-only keeps earlier checkers valid; pass rate 0 → regenerate, don't train; AND with an intent check | [IFDecorator (2025)](https://arxiv.org/abs/2508.04632): Qwen2.5-32B IFEval Pr(S) 87.43 (+7.95); [LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412) |
| **Multi-turn carry-over and system-prompt embedding** | "Answer in French." → system: "Always French, ≤80 words"; turn 3: "English and a table for this answer only"; turn 5 must revert; final turn injects a conflict | Track the active-constraint set per turn in code, newest taking priority; keep only prompts the current model fails | [AdvancedIF (2025)](https://arxiv.org/abs/2511.10507); Multi-IF: o1-preview 0.877 → 0.707 from turn 1 to turn 3 |

Also cheap: **new constraint families and wider parameter ranges.** Models overfit the IFEval types: TÜLU-3-8B-DPO scores 81.1 on IFEval but 25.5 on IFBench. Training with up to 3 constraints per prompt beat 1 (IFBench 59.5 vs 48.9, Qwen2.5 policy), and disjoint train ranges hurt ([IFBench, 2025](https://arxiv.org/abs/2507.02833)). Presentation (constraints woven into prose, order, violating examples) raises difficulty without new verifiers ([MulDimIF, 2025](https://arxiv.org/abs/2505.07591): Incorporation is harder than Listing, which is harder than Example; [Order Matters, 2025](https://arxiv.org/abs/2502.17204) for constraint order; [FollowBench, 2023](https://arxiv.org/abs/2310.20410) for noise examples).

### Architecture

A synthesis of the loops above (**Proposal** as a whole; each step is published, but no single paper runs all of them together):

```
seed prompt (+ a strong response if back-translating)
  ▼ sample k constraints from a typed library (code checker + unit tests per type);
  │ compatibility checker: pair blocks, arithmetic feasibility, propagation              [CSE, IFBench]
  ▼ optionally compose: Chain / Selection (branch condition computable from the input) / scope   [ComplexBench, LsrIF, ScopeIF]
  ▼ render: presentation pattern and order                                               [MulDimIF]
  ▼ calibrate on the current policy (8 rollouts): keep (0, 0.5]; > 0.5 → append constraints;
  │ 0 → regenerate, do not train                                                          [IFDecorator]
  ▼ reward = per-constraint code checks (+ at most one judged soft constraint, judged per constraint),
  │ aggregated along the composition tree, ANDed with an intent / quality gate            [LsrIF, Precision over Diversity, IFDecorator, IFBench]
  ▼ monitor held-out trip-wire prompts for hacking                                        [IFDecorator]
```

### Verifier and reward

- **Precision is the binding constraint on soft constraints (Moderate: one controlled study, and VerIF finds the opposite with a strong reasoning judge).** Code checkers reach 96.0% precision. In multi-constraint judging, Qwen3-32B caught only 30.6% of hard-constraint and 20.9% of soft-constraint violations; judging each constraint separately raised this to 59.3% and 54.7% ([Precision over Diversity, 2026](https://arxiv.org/abs/2601.04954)). Hard-only data matched mixed data on average (57.85 vs 57.26; soft-only 55.18). But with a strong *reasoning* judge soft constraints add signal: VerIF reached IFEval 84.5 with code + QwQ-32B vs 74.7 code-only. **Measure your judge's recall on known violations before mixing soft constraints in; otherwise cap them at one per prompt.**
- **Match aggregation to structure and verifier noise (Moderate).** Structure-aware aggregation beat averaging by 3.9 IFEval and 5 CFBench points ([LsrIF, 2026](https://arxiv.org/abs/2601.06431)). Graded distance-to-bound rewards beat constraint-level averaging ([ScopeIF](https://arxiv.org/abs/2609.32189)). Yet with a fine-tuned rubric verifier, all-or-nothing beat fractional credit (58.1 vs 53.6, [AdvancedIF](https://arxiv.org/abs/2511.10507)). Rule of thumb: partial or graded credit for code-verified constraints, strict gating for judge-verified rubrics.
- **Recover gradient on all-zero groups.** Hindsight relabeling rewrites failed rollouts into instructions containing only the constraints they satisfied: Qwen3-4B IFBench +10.6 ([HiR, 2025](https://arxiv.org/abs/2512.23457)). Multi-temperature sampling with goal-anchored advantages ([MDP-GRPO, 2026](https://arxiv.org/abs/2606.06058)) and leave-one-constraint-out teacher shaping ([CC-OPD, 2026](https://arxiv.org/abs/2609.27421)) are optimizer-side alternatives.
- **Hacking is the default outcome of IF-RLVR (Strong).** Observed: over-satisfying constraints while ignoring the task; literal placeholders like `<<title>>`, dummy lists and "p p p"; long preambles; "all instructions are followed" self-claims; copying reference text. Mitigations with evidence: gate the reward with a preference RM (IFBench: F = V+1 if V>0 and RM>7, V−0.5 if V>0 and RM≤7, V otherwise); an IntentCheck gate (macro hack rate 14.53% → 7.60%); universal anti-preamble and anti-artifact criteria ([RLCF, 2025](https://arxiv.org/abs/2507.18624); AdvancedIF); held-out trip-wire prompts.

### Difficulty knobs

k (with p^k), composition type and depth, scope nesting, presentation pattern, variable ranges, number of turns and priority conflicts, and implicit hops per parameter. **Decide by pass rate on the current policy, not by constraint count (Strong):** IFDecorator reports low-complexity prompts that are hard and high-complexity prompts that stay easy. It keeps prompts in (0, 0.5] over 8 rollouts; QUBRIC keeps 20–50%. The count per prompt is a per-policy hyperparameter: the Qwen2.5 IFBench curve peaks at 3 and drops to 49.4 at 4.

### Pitfalls

- **Over-hardening is the normal failure mode.** From 21k seeds over 5 iterations IFDecorator produced 7,324 usable prompts and 10,772 unsolvable ones. Removing SEIF's satisfiability filter cost 3.2 IFEval and 6.0 CFBench ([SEIF, 2026](https://arxiv.org/abs/2605.07465)). Budget a satisfiability checker and regeneration.
- **Offline hardening without the policy does not help** (Evol-Instruct 50.24 vs 50.51 unchanged, [LLM-as-a-Tutor](https://arxiv.org/abs/2607.04412)); back-translation saturates at the source response's difficulty (Crab SFT trailed Conifer at FollowBench L3–L5).
- **Vanilla CoT is not a fix.** Plain CoT cut a 1.5B model's 7-benchmark average by 11.79 ([RAIF, 2025](https://arxiv.org/abs/2506.01413)).
- **Cross-domain interference.** Math-RLVR raised IFEval pass@1 by 6.5% but lowered best@32 by 9.8% (Qwen3-8B-Base), and IF-RLVR pushes math responses toward direct-answer openings ([2608.00220](https://arxiv.org/abs/2608.00220)). Track best@k when mixing.

### Starting data and key references

IFTrain (29 training constraint types) and IFBench (58 out-of-distribution types); VerInstruct (22,000 prompts, about 6.2 constraints each); RECAST-30K (13.4 constraints on average); MulDimIF (9,106 code-verifiable instances); HiR-16K; SCOPEINSTRUCT (16,968); LogicIFTrain (27,324); WildChecklists (130k). For multimodal constraint data see §8.

| Work | Verified result |
|---|---|
| [IFBench / IF-RLVR (2025)](https://arxiv.org/abs/2507.02833) | TÜLU-3-8B IFEval 82.4 → 92.2, IFBench 28.9 → 45.9; GRPO beat DPO (89.65 vs 79.67 IFEval strict) |
| [VerIF (2025)](https://arxiv.org/abs/2506.09942) | TULU 3 SFT → +VerIF: IFEval Pr(S) 68.4 → 84.5; Multi-IF turn 3 40.3 → 54.0 |
| [MulDimIF (2025)](https://arxiv.org/abs/2505.07591) | GRPO Llama3.1-8B 36.17 → 88.08 overall, Level IV 12.00 → 84.00 (in-domain, partly template learning) |
| [RECAST (2025)](https://arxiv.org/abs/2505.19030) | All-constraint rate ≤7% for every model at L4 (≥15 constraints); dataset cost about $175 |
| [HiR (2025)](https://arxiv.org/abs/2512.23457) | Qwen3-4B IFBench 29.9 → 40.5, MulDimIF +23.3 |

---

## 6. Multi-hop and long-context reasoning

**Seeds.** HotpotQA, MuSiQue and 2Wiki items; short math (DeepMath-103K, MATH-Hard); document QA; any state machine you can simulate. Store the gold chain (intermediate entities, evidence chunks) for every item: it gives correctness by construction, shortcut tests and the cheapest dense reward available.

### Best operators

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Bridge composition with masking tests** | "Who directed Inception?" + "When was Nolan born?" → "When was the director of the 2010 dream-heist film born?"; GSM8K Q2 with a constant replaced by X = answer(Q1) | Head must fail closed-book; tail must fail with the bridge masked; compare the pass rate with the product of per-link pass rates | [MuSiQue (2021)](https://arxiv.org/abs/2108.00573): only 26.5% of candidate tail questions survived; [Compositional GSM (2024)](https://arxiv.org/abs/2410.01748) |
| **Post-composition reasoning layer** | "Which year was X founded?" → "How many years after X's founding did Y happen, and what is that number reversed?" | Code computes the transform from the gold answer | [MoreHopQA (2024)](https://arxiv.org/abs/2406.13397): GPT-4 achieves "perfect reasoning" on only 38.7% |
| **Premise dispersal** | A 150-token math problem → a narrative with variables in ≥5 fragments inside a 32–64K irrelevant document | Keep the seed gold; a second LLM verifies solvability after rewriting | [Beyond Reward Engineering (2026)](https://arxiv.org/abs/2606.18831): reasoning family best single (+3.46; LongReason +5.56) |
| **Indirection (pointer chains with decoys)** | A plain multi-hop question → UUID key-value chains in context, only one resolving to the real question | Chains are programmatic; lenient but hard-to-hack matcher | [LoongRL (2025)](https://arxiv.org/abs/2510.19363): 72.4 vs 66.2 when KeyChain is replaced by standard long-context multi-hop data |
| **Lexical-overlap removal** | Query "Who has been to Dresden?", needle "Yuki has been to Dresden" → needle "Yuki lives next to the Semper Opera House"; add literal-match decoys | Link through a verified relation (e.g. Wikidata IS-A); filter the haystack for other spans satisfying the association | [NoLiMa (2025)](https://arxiv.org/abs/2502.05167): Llama 3.3 70B at 32K scores 98.5 (literal), 56.2 (one latent hop), 25.9 (two) |
| **Agent-mined hard distractors** | Gold paragraphs + random Wikipedia → pages a search agent opened but did not cite (Tier 1) and results it never opened (Tier 2) | The agent must solve the item at least once in 5 tries; check no distractor completes the answer | [LongTraceRL (2026)](https://arxiv.org/abs/2605.31584): Qwen3-4B-Thinking +5.7 average over base |
| **Cross-document aggregation over derived attributes** | "When was player A born?" → "Among these 12 players, who was youngest at their first title?" (winning_year − birth_year) | Extract values once, verify against spans, compute by code or SQL; use exact numeric checks or a judge, never recall | [QwenLong-L1.5 (2025)](https://arxiv.org/abs/2512.12967): KG paths + SQL track; +9.90 average over its base |
| **Program-backed worlds and latent-state simulation** | "Who is Alice's mother?" in 50 people → "the hobby of the second cousin of Alice's friend" in 3K+ people; 5 list operations → 20 relevant operations padded with provably no-op code to 128K | Prolog or a simulator is the ground truth; score set answers with F1; regenerate per epoch | [PhantomWiki (2025)](https://arxiv.org/abs/2502.20377); [Michelangelo (2024)](https://arxiv.org/abs/2409.12640); [GSM-Infinite (2025)](https://arxiv.org/abs/2502.05252) |

Add **sufficiency twins** (the same item with one required fact removed; answer or abstain on both, scored jointly) to penalize guessing ([MuSiQue](https://arxiv.org/abs/2108.00573)).

### The gate stack (Strong)

Every strong 2025–2026 pipeline implements a subset of this order ([OpenSeeker](https://arxiv.org/abs/2603.15594), [InfoSeek](https://arxiv.org/abs/2509.00375), [ASearcher](https://arxiv.org/abs/2508.07976), [QwenLong-L1.5](https://arxiv.org/abs/2512.12967), [DeepReasonQA](https://arxiv.org/abs/2601.12465), MuSiQue):

```
1 construction-time correctness (execution / KG / SQL / inherited gold)
2 shortcut & leakage: fails closed-book, fails with bridge masked, fails with no context
3 oracle solvability: solved with gold evidence + distractors
4 uniqueness: enumerate alternatives; check whether wrong answers are also valid
5 robustness: accuracy survives extra irrelevant documents
6 learnable band on the current policy: 0<p<1 over 8 (LoongRL 277K → 72K), [0.25, 0.75] (DeepReasonQA)
```

GoLongRL adds a weak-to-strong cascade that separates "hard" from "broken": items a 4B model fails go to a 30B model, and those still below 0.25 are discarded as quality-insufficient ([GoLongRL, 2026](https://arxiv.org/abs/2605.19577)).

### Verifier and reward

- **The matcher sets the ceiling.** LoongRL: two-way substring match 72.4, exact match 69.2, LLM judge 65.2, F1 65.1. Recall rewards can be gamed by enumerating candidates (CrossEntity switched to a judge); set-F1 penalizes over-listing ([GraphWalks](https://huggingface.co/datasets/openai/graphwalks); PhantomWiki). **Moderate.**
- **Process rewards help only when anchored in construction metadata (Moderate).** An F-beta reward on selected evidence chunks vs the gold evidence set took a 14B model's RULER-QA from 73.17 to 88.90 over outcome-only RLVR ([LongRLVR, 2026](https://arxiv.org/abs/2603.02146)). A positive-only gold-entity rubric, applied only when the final answer is correct, blocks rubric hacking (LongTraceRL). A free-form LLM-judge process reward on thinking *cost* 1.29 points (Beyond Reward Engineering).

### Difficulty knobs

Hop count (DeepReasonQA averages 8.78 hops, paths of 2–30), distractor quality, lexical overlap, number of evidence chunks, aggregation type, and context length, **tuned separately**. Length generalizes cheaply; reasoning density does not. Training at 16K passed all 128K NIAH tests (LoongRL), and training at up to 64K kept +4.6 at 256–512K (Beyond Reward Engineering). Spend the budget on hops, distractors and aggregation, not raw length. **Strong.**

### Pitfalls

- **Composition is taught by RL, not SFT (Strong).** RL on depth-2 string-function compositions reached 27% at depth 3 while rejection fine-tuning never exceeded 2.6% ([f(g(x)), 2025](https://arxiv.org/abs/2509.25123)). GRPO on purely fictional PhantomWiki data gave Qwen3-0.6B 56–131% relative F1 gains on five real benchmarks; SFT on GSM-infinity traces did not transfer to HotpotQA ([Kabra et al., 2026](https://arxiv.org/abs/2603.02091)).
- **One-shot "hard multi-hop" generation is mostly junk:** only 33.1% of Self-Instruct long-context samples from a 72B model were truly multi-hop ([LongMIT, 2024](https://arxiv.org/abs/2409.01893)).
- **Unit-test generators.** OpenAI's MRCR had about 5% incorrect ground truth and about 10% of items with too many needles; GraphWalks had wrong labels on 24 of 400 parent items ([MRCR changelog](https://huggingface.co/datasets/openai/mrcr)). In RL a systematic label bug becomes a hacking target.
- **Not every "harder" generator helps every base.** On DeepSeek-R1-0528-Qwen3-8B, DocQA data (40.6) and LoongRL data (40.1) scored *below* the base (42.7), while agent-mined distractor data scored 43.8 (LongTraceRL). Validate per base model and keep a mix of retrieval, multi-evidence and reasoning families.

### Starting data and key references

MuSiQue and 2Wiki (seed pools); PhantomWiki and GSM-Infinite generators; RULER and BABILong recipes; Beyond Reward Engineering (14,069 items, eight datasets in three families); LongTraceRL (2,815 examples at 128K); DeepReasonQA (2,012 RL samples); QwenLong-L1.5 (14.1K kept of 42.7K); GoLongRL (23K).

| Work | Verified result |
|---|---|
| [LoongRL (2025)](https://arxiv.org/abs/2510.19363) | LongBench v1 multi-hop QA: 7B 72.4 (+23.5), 14B 74.2 (+21.1) |
| [Beyond Reward Engineering (2026)](https://arxiv.org/abs/2606.18831) | Qwen3-4B / 8B / 30B-A3B: +7.2 / +3.2 / +6.4 average over seven benchmarks |
| [QwenLong-L1.5 (2025)](https://arxiv.org/abs/2512.12967) | MRCR +31.72 (0–128K subset); memory agent +9.48 on 1M–4M-token tasks |
| [LongRLVR (2026)](https://arxiv.org/abs/2603.02146) | 14B LongBench v2 39.8 → 46.5 over outcome-only RLVR |

---

## 7. Memory and multi-turn interaction structure

These operators change **how** an already-verifiable seed is delivered (horizon, turns, who holds which information, what hostile content sits in the environment), not what it asks. The label is free, so the only work is proving the information is still all there.

**Seeds.** Any item with an existing verifier: HotpotQA/NQ questions, GSM8K and code problems with unit tests, SWE-bench issues with hidden tests, tau2 or AgentDojo user tasks with final-state checks, and programmatically graded instruction-following prompts.

### Best operators

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Objective stacking under bounded memory** | One HotpotQA question → 16 interleaved questions in one episode with a constant-size internal state | Per-item gold unchanged; 0 if answer count ≠ question count; remove closed-book-answerable items | [MEM1 (2025)](https://arxiv.org/abs/2506.15841): trained at 2 objectives, EM-sum 1.97 at 16 vs 0.567 for Qwen2.5-14B-Instruct |
| **Distractor streaming beyond the window** | HotpotQA with 2 gold paragraphs → 32K–60K training length, 3.5M test length, 5K-token chunks through a 1,024-token overwrite memory | Drop items solved without context (about 50% of 80K at MemAgent); string-check distractors for the gold | [MemAgent (2025)](https://arxiv.org/abs/2507.02259): 80.47 at 7K, 71.09 at 3.5M tokens |
| **Session-horizon state accumulation** | "Total spent" over 2 sessions → 500 sessions of transactions rendered as dialogue | Keep the ledger as a program variable; derive questions by code | [UMA / Ledger-QA (2026)](https://arxiv.org/abs/2602.18493): at H=500, 25.00 vs 11.54 for MemAgent |
| **Interaction-budget extension** | BrowseComp at T_max=32 → T_max=2048 with a report-as-memory workspace | Outcome verifier unchanged; broadcast advantages across segments | [IterResearch (2025)](https://arxiv.org/abs/2511.07327): 15.2% → 42.5% on BrowseComp-200; [Context-Folding (2025)](https://arxiv.org/abs/2510.11967) |
| **Instruction sharding** | A GSM8K problem in one turn → 5 minimal shards over 5 turns | Run CONCAT (all shards in one turn) per item and drop items where it fails; reuse the original verifier | [Lost in Conversation (2025)](https://arxiv.org/abs/2505.06120): −39% across 15 LLMs, CONCAT at 95.1% of FULL |
| **Truncation → abstain** | All 5 shards → cut after 3, gold "Abstain"; all tool paths blocked, gold "stop and escalate" | Certify shard necessity; cap the unsolvable share (m=0.1; 25%); add an explicit positive reward for correct abstention | [RLAAR (2025)](https://arxiv.org/abs/2510.18731): Qwen3-8B abstain score 33.5% → 73.4%; [BENCH2ROBUST](https://arxiv.org/abs/2608.11977) |
| **Hidden spec behind an oracle user** | Full SWE-bench issue → issue missing Error Info or Reproduction; a ColBench description with the reference held by the simulator | Hidden tests unchanged; restrict simulator answers to the withheld spec; audit for verbatim leakage | [CLARITI (2026)](https://arxiv.org/abs/2604.14624): 43.8% → 23.7%, 36.8% with learned clarification using 3.0 questions; [SWEET-RL (2025)](https://arxiv.org/abs/2503.15478) |
| **One-missing-variable underspecification** | A solvable PDDL instance → one unobserved initial fact; pick the question that restores solvability | A solver computes the exact set of sufficient questions | [QuestBench (2025)](https://arxiv.org/abs/2503.22674): 40–50% on Logic-Q and Planning-Q |
| **Split information, planted partner errors, hidden-role games** | A puzzle one agent solves → complementary halves across two agents; a partner confidently arguing a known-wrong answer; a vocabulary word → Adversarial Taboo | Union of views = original seed; reward only gold-correct consensus; rule-decided game outcomes | [Coral](https://ai.meta.com/research/publications/collaborative-reasoner-self-improving-social-agents-with-synthetic-conversations/): up to +29.4% over single-agent CoT; [SPAG (2024)](https://arxiv.org/abs/2404.10642); [MARSHAL (2025)](https://arxiv.org/abs/2510.15414) |
| **Injection into untrusted slots** | "Summarize my emails" in a clean inbox → the same with attacker text in a tool output | Deterministic utility AND security functions on final state; attacker confined to designated slots; randomized position | [AgentDojo (2024)](https://arxiv.org/abs/2406.13352): "important message" targeted ASR 57.7% on GPT-4o; [CoER (2026)](https://arxiv.org/abs/2609.07529): attack success 38.5% → 0.2% with utility 63.2% → 76.3% |

### Architecture notes

- **Controls separate "hard" from "broken" (Strong).** CONCAT for sharding, oracle-evidence reading for memory, fully specified runs for hidden-spec tasks. Treat items whose control fails as label errors.
- **Memory is graded through a frozen reader**, so the memory writer cannot grade itself ([Mem-α, 2025](https://arxiv.org/abs/2509.25911); [Memory-R1, 2025](https://arxiv.org/abs/2508.19828)).
- **Curriculum gates tied to the model's own easy-view reward.** RLAAR adds shards once reward exceeds ρ × the single-turn baseline: on Qwen3-1.7B, ρ=0.8 reached the hardest stage in 90 steps (71.9 LiC), ρ=1.0 needed 1000+ steps, ρ=0.6 collapsed to 40.9, and no curriculum gave 63.2.
- **The easy view is a free teacher.** When sharded RL is too sparse, self-distil from the model's own fully specified view: FiC recovers ≥92% of single-turn performance ([FiC, 2026](https://arxiv.org/abs/2605.24432)); MAIGO raises the SHARDED/FULL ratio from 66.5% to 84.1% ([MAIGO, 2026](https://arxiv.org/abs/2605.27186)). Rolling-memory RL beat full-history RL by 15.2–35.6 pp ([2606.12941](https://arxiv.org/abs/2606.12941)). **Emerging.**
- **Credit assignment must respect segments, turns and roles.** Examples: multi-conversation advantages (MemAgent), counterfactual information gain per clarifying turn ([InfoPO, 2026](https://arxiv.org/abs/2603.00656): +14–16% over GRPO baselines), turn-level sum-then-normalize for cooperative games. Swapping in a role-conditioned zero-sum advantage dropped Mini Hanabi from 54.33% to 24.93% (MARSHAL).

### Difficulty knobs

Number of stacked objectives N, context multiple, session horizon H, round budget T_max, shard count K and gate ρ, unsolvable share m, information categories withheld, partner quality, injection budget and attacker strength. **Train short, test long, under RL (Strong):** MEM1 (2 → 16 objectives), MemAgent (60K → 3.5M), IterResearch (32 → 2048 rounds) all extrapolate, while MEM1's SFT baseline collapses beyond 6 objectives (EM 0.027 at 8 vs 1.870 for RL).

### Pitfalls

- **Degenerate constant policies** (always ask, always abstain, always refuse) are the default failure. Balance solvable/unsolvable and ambiguous/unambiguous items, and give every adversarial operator **benign twins**: removing IH-Challenge's anti-over-refusal split dropped over-refusal score from 0.950 to 0.831 and helpfulness from 0.773 to 0.613 ([IH-Challenge, 2026](https://arxiv.org/abs/2603.10521)).
- **The user simulator is part of the verifier.** Open-source simulators "often get confused" and start solving the task ([CollabLLM, 2025](https://arxiv.org/abs/2502.00640)); LLM customers are too agreeable (non-buyers' resistance falls from 25.1% to 13.5%, [2606.20708](https://arxiv.org/abs/2606.20708)). A *trained* user simulator learned to "force success rates around 50% by directly accepting or rejecting regardless of agent performance" ([SEAD, 2026](https://arxiv.org/abs/2602.03548)): a learned component may choose tasks, but must not grade them (see [Chapter 04](04-verification-and-quality-control.md)).
- **Static adversaries overstate robustness; single opponents collapse.** Adaptive attacks broke all 8 evaluated IPI defenses at >50% ASR ([2503.00061](https://arxiv.org/abs/2503.00061)). Attackers trained against one defender mode-collapse ([GPT-Red, 2026](https://arxiv.org/abs/2607.26115)). Train against populations of past opponents.
- **Positional shortcuts.** SecAlign's always-at-end injections taught "ignore the last instruction" and crushed AgentDojo utility to 6.2%; randomizing position alone restored +69.0 pp ([Meta-SecAlign, 2025](https://arxiv.org/abs/2507.02735)).

**Proposal: compose delivery operators on one seed.** No open pipeline delivers a verifiable seed as sharded turns, over a horizon beyond the window, with an injected tool output and an information-withholding partner, while measuring how N, K, H and attacker strength interact. Because each operator is label-preserving, their composition is too, provided each control (CONCAT, closed-book, clean-environment run) is re-run on the composite.

### Starting datasets and environments

Lost in Conversation sharded instructions (released as Microsoft/lost_in_conversation) and MultiChallenge; LongMemEval, LoCoMo and MemoryAgentBench for memory; Ledger-QA; QuestBench (Logic-Q, Planning-Q, GSM-Q); ColBench (10K backend tasks); AgentDojo (97 user tasks, 629 security cases); IH-Challenge (huggingface.co/datasets/openai/ih-challenge); WildJailbreak (262K) and OR-Bench (80K) for benign twins.

### Key references

| Work | Verified result |
|---|---|
| [MEM1 (2025)](https://arxiv.org/abs/2506.15841) | Trained at 2 objectives: EM-sum 1.97 at 16 vs 0.567 for Qwen2.5-14B-Instruct; SFT 0.027 vs RL 1.870 at 8 |
| [MemAgent (2025)](https://arxiv.org/abs/2507.02259) | RL-MemAgent-14B RULER-HotpotQA 80.47 at 7K, 71.09 at 3.5M tokens |
| [IterResearch (2025)](https://arxiv.org/abs/2511.07327) | Trained at T_max=32: BrowseComp-200 42.5% at T_max=2048 vs 15.2% at 32 |
| [RLAAR (2025)](https://arxiv.org/abs/2510.18731) | Qwen3-8B LiC score 62.6% → 75.1%, abstain score 33.5% → 73.4% |
| [CLARITI (2026)](https://arxiv.org/abs/2604.14624) | SWE-bench Verified 43.8% → 23.7% when underspecified; 36.8% with 3.0 learned questions |
| [IH-Challenge (2026)](https://arxiv.org/abs/2603.10521) | GPT-5-Mini IH robustness 84.1% → 94.1% over 16 benchmarks |
| [CoER (2026)](https://arxiv.org/abs/2609.07529) | Attack success 38.5% → 0.2% with utility 63.2% → 76.3% (own seven-domain eval) |

---

## 8. Multimodal and visual reasoning

**Seeds.** Visual math sets (MMK12), chart QA, human captions, COCO-style images, diagram geometry, simple games. The rule that separates exact rewards from noisy ones: **build the latent state (scene, program, chart source, construction, game state), compute the answer from it, and only then render the image and write the question** ([TRON](https://arxiv.org/abs/2606.01599), [Trace](https://arxiv.org/abs/2607.19790), [VVR](https://arxiv.org/abs/2609.35641), [ChartVerse](https://arxiv.org/abs/2601.13606), [GeoSym127K](https://arxiv.org/abs/2605.16371)). The verifier then never reads pixels. **Strong.**

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Answer-hidden stem hardening (for seeds you already have)** | "Area of the shaded triangle?" with base and height labelled → "…given only the perimeter and one angle marked in the figure" | Rewriter never sees the answer; accept only if the target policy still reaches the original answer in ≥4/16 rollouts *and* passes at least 2 fewer times than on the seed; seeds must be solved ≥12/16 | [SynthRL (2025)](https://arxiv.org/abs/2506.02096): 3,380 harder variants from 8,072 seeds; 5-benchmark average 57.0 → 58.0 |
| **Levels that change the mechanism** | Angle chase: one-step triangle sum (ℓ0) → four-step chain over parallel lines (ℓ9); 1-series bar chart → stacked multi-series aggregation | Answer fixed before rendering; plot the base pass rate per level (a flat curve means a cosmetic knob) | [TRON (2026)](https://arxiv.org/abs/2606.01599): Qwen3-VL-4B pass 72.8% (ℓ0) → 41.3% (ℓ9); 10-benchmark mean 52.61 → 55.23 |
| **Answer-first inverse synthesis** | "Value of bar B?" → a complex multi-panel chart where a script computes a cross-series ratio and the question is written to that value | Execute analysis code over the chart source; keep only if the teacher's answer from the image matches; keep 0 < fail < 1 | [ChartVerse (2026)](https://arxiv.org/abs/2601.13606): 8B model 64.1 vs its 30B-A3B teacher 62.9 |
| **Scene-first constraint stacking (image generation)** | "Draw 3 red circles" → "3 red circles left of 2 blue squares, twice as many circles as squares in the top half" | Read constraints off a realized scene (guarantees joint satisfiability); verifier must accept the reference and reject a counterfactual; reward = partial credit × all-satisfied guard | [VVR (2026)](https://arxiv.org/abs/2609.35641): pixel verifier 94.7% human agreement; SD3.5-M 2.81% → 28.27% |
| **Factor recomposition** | "Sector A's 2020 growth?" → "By how much does the fastest-growing sector's 2020 growth exceed the median of sectors whose 2019 value was below average?" | Sub-answers from metadata or code; max-shaped (not summed) process reward | [COGS (2025)](https://arxiv.org/abs/2510.15040): ChartQAPro 47.36 → 52.02; summed process reward 50.35 |
| **Single-error injection → localization** | "Describe the image" → "In this ~200-word caption, which span is wrong?" (three mugs → two) | Injected span known; string match; source caption must be correct | [ViCrit (2025)](https://arxiv.org/abs/2506.10128): 7B average 50.61 → 53.01 |
| **Transform recovery (pretext)** | 2×1 patch swap → 3×3 full permutation; rotation → rotation + mirror + crop | Parameters are the label; partial credit (γ = 0.2) for large permutations, without which 3×3 grids were not learned ([Visual Jigsaw, 2025](https://arxiv.org/abs/2509.25190)); proposer targets ~50% solver accuracy (EVE) | [Jigsaw-R1 (2025)](https://arxiv.org/abs/2505.23590); [EVE (2026)](https://arxiv.org/abs/2604.18320) |
| **Grammar depth and modality shift** | Isosceles base angle → recursive constructions with shaded regions; "AB = 3, BC = 4…" in text → lengths only in the figure | Exact symbolic solver; accept teacher CoT only if SymPy `Simplify(pred − GT) ≡ 0` | [GeoSym127K (2026)](https://arxiv.org/abs/2605.16371): MathVerse Vision-Only +22.21 (Qwen3-VL-8B) |

For more options per item, add distractors from *related* rules ([VisualSphinx, 2025](https://arxiv.org/abs/2505.23977): 4 → 10 options). Multimodal constraint-following data synthesis also exists: [MIFS (2026)](https://arxiv.org/abs/2609.16059) builds 90K samples over 8 constraint categories with code verifiers; [VISA (2026)](https://arxiv.org/abs/2608.26013) uses an agentic loop driven by failure profiles. REDSearcher's search pipeline plans a 5K multimodal trajectory release (§2).

**Difficulty knobs and scheduling.** Knobs: generator level ℓ (TRON), grammar recursion depth (GeoSym127K), constraint count and structural complexity C(s) (VVR), grid size and question format (Jigsaw-R1), option count (VisualSphinx), number of composed factors (COGS), chart complexity selected by reconstruction instability (ChartVerse), and QA level × plot level (Game-RL). Plot the base model's pass rate per level before training; a flat curve means the knob is cosmetic. Promote a generator level at about 0.80 accuracy and replay lower levels with probability 0.30 (TRON). Mix tiers: Easy+Medium+Hard 52.74 beat every single tier in [Game-RL](https://arxiv.org/abs/2505.13886). Do not assume the hardest tier is the best RL tier: after Entry-tier SFT, GRPO on Hard (43.59) underperformed GRPO on Entry (44.51) (GeoSym127K). VisualSphinx trains on pass rates in 0.375–0.875. **Moderate.**

**Pitfalls.**
- **SFT on synthetic hard visual tasks often regresses; RL does not (Strong).** Game-RL: SFT −3.9% on general benchmarks vs RL +2.33%. ViCrit: caption SFT 48.41 vs base 50.61. COGS: SFT then RL 46.62, below the base 47.36. Jigsaw-R1: an SFT cold start hurt. Exceptions where the model lacks domain knowledge or output format (GeoSym127K, ChartVerse) did better with SFT before RL. See [Chapter 06](06-sft-playbook.md).
- **Easy-to-hard transfer decays with distance.** VVR trained on easy scenes gained +17.16, +8.31 and +1.35 on progressively harder bins; Jigsaw-R1's unseen 2×2 moved only 12.20 → 13.10. Train on hard instances directly, with partial credit if needed.
- **Conjunctions lag single skills.** Easy-only VVR training closed 79% and 64% of the partial-score gap on counts and relations but only 55% and 46% of the all-satisfied gap.
- **Unexecuted code-read QA can hurt.** At a fixed 100K budget, CoSyn chart data lowered Qwen3-VL-4B from 53.9 to 51.0 in ChartVerse's comparison ([CoSyn, 2025](https://arxiv.org/abs/2502.14846) answers are LLM-written from code, not executed).
- **Self-evolving VLM loops plateau** (EVE after 4–5 iterations; [Active-Zero, 2026](https://arxiv.org/abs/2602.11241) regresses slightly at iteration 3 with majority-vote pseudo-labels). Prefer executable answers over pseudo-labels.

**Starting data:** TRON (520 environments), Trace (1,000 tasks, 277 scene grammars), GameQA (about 140K questions, 30 games), VisualSphinx (660K+ puzzles for under $1,000), ChartVerse-SFT-600K and RL-40K, GeoSym127K, ViCrit (875K pairs), VVRBench. Evaluate on MathVista, MathVerse, WeMath, MMMU, CharXiv and ChartQAPro.

**Key references**

| Work | Verified result |
|---|---|
| [TRON (2026)](https://arxiv.org/abs/2606.01599) | Qwen3-VL-4B 10-benchmark mean 52.61 → 55.23; base pass rate 72.8% (ℓ0) → 41.3% (ℓ9) |
| [VVR (2026)](https://arxiv.org/abs/2609.35641) | SD3.5-Medium on VVRBench 2.81% → 28.27% (VVR-Easy), 46.60% (VVR-Matched) |
| [ChartVerse (2026)](https://arxiv.org/abs/2601.13606) | ChartVerse-8B 64.1 vs its 30B-A3B teacher 62.9 (6-benchmark average) |
| [GeoSym127K (2026)](https://arxiv.org/abs/2605.16371) | Qwen3-VL-8B MathVerse Vision-Only +22.21, WeMath 61.52 |
| [Game-RL (2025)](https://arxiv.org/abs/2505.13886) | Qwen2.5-VL-7B, 5K GRPO samples: 7-benchmark average 49.94 → 52.27 |
| [Trace (2026)](https://arxiv.org/abs/2607.19790) | 24-benchmark macro-average +3.51 (3B), +4.06 (7B) |
| [SynthRL (2025)](https://arxiv.org/abs/2506.02096) | 3,380 harder variants from 8,072 seeds; 5-benchmark average 57.0 → 58.0 |

---

## 9. Text-to-SQL and tables

**Seeds.** Spider and BIRD training items; web tables; schema-only synthetic sets (Gretel-Synth). SQL is the easiest agentic target to make both harder and exactly verifiable, because the answer is an execution result.

| Operator | Easy → hard | Keep it verifiable | Evidence |
|---|---|---|---|
| **Structural AST mutation with scarcity weighting** | `SELECT name FROM employees WHERE dept='Sales'` → a CTE joining employees, departments and salaries, with a CASE-wrapped aggregate, a HAVING subquery and an INTERSECT | Six atomic operators (wrap, mutate, clause, JOIN, nest, set); choose by feasibility × scarcity; the query must execute and return **non-empty** results; back-translate the question afterwards | [EvolSQL (2026)](https://arxiv.org/abs/2601.04875): CTE frequency 0.02 → 0.58 across rounds; BIRD dev 65.1 with about 1/18 of OmniSQL's data |
| **Complexity tiers + schema widening + multi-style back-translation** | A first-draft schema with about 4 columns per table → widened tables with completed primary and foreign keys (K ~ N(10, 4²) tables per database); SQL sampled at 4 complexity tiers; question back-translated in 9 styles (formal … vague, metaphorical) | Execution filter; choose CoTs by majority vote over execution results | [OmniSQL (2025)](https://arxiv.org/abs/2503.02240): 1.75 joins per query vs 0.94 in BIRD; audit: 86% of samples fully correct |
| **Context enlargement: distractor tables** | The gold schema alone → gold schema plus distractor tables from related domains | Gold SQL unchanged; drop gold that returns empty results or runs >5 s | [Arctic-Text2SQL-R1 (2025)](https://arxiv.org/abs/2505.20315): filtered synthetic data 66.5 vs 64.9 BIRD-dev (14B) |
| **Table and instruction co-evolution on failures** | A flat table + simple question → hierarchical/messy table + multi-step instruction, seeded only from items the model failed | Replace the LLM reference with an answer computed by SQL/Pandas over the synthetic table (the original uses GPT-4o references) | [TableDreamer (2025)](https://arxiv.org/abs/2506.08646): weakness-filtered 27K beat unfiltered 34K (60.69 vs 56.28) |

Two further operators come from other sections: unified cross-document tables queried by generated SQL for long-context aggregation ([QwenLong-L1.5](https://arxiv.org/abs/2512.12967)), and sharding a Spider instruction across turns ([Lost in Conversation](https://arxiv.org/abs/2505.06120)).

**Verifier.** Execution-only reward (plus syntax) was sufficient for Arctic's GRPO. Remove degenerate gold first: Arctic dropped about 1,400 BIRD and 1,700 Spider items whose gold SQL returned empty results or ran longer than 5 s, because they give spurious rewards. Keep a query only if the best model solves it at least once in 10 samples. **Moderate.** Known gap: a non-empty result does not prove the question and SQL mean the same thing (EvolSQL). **Proposal:** borrow SpreadsheetBench's multiple test cases and require the policy's SQL to match the gold on 2–3 perturbed database instances, which rejects queries that match only by coincidence.

**Pitfall.** Naive LLM "paraphrase or generate complex question" augmentation mirrored the originals and *hurt* (BIRD-dev 62.5 unfiltered, 64.9 filtered, vs 64.9 without it). Harden the executable object (AST, schema, distractor tables), not the wording. **Moderate.**

**Difficulty knobs.** AST operator choice and number of evolution rounds (EvolSQL); SQL complexity tier and tables per database (OmniSQL); number of distractor tables and a minimum SQL length (Arctic keeps SQL longer than 160 characters); table structure (flat, horizontal, hierarchical) and instruction-complication strategy (TableDreamer). Calibrate with the ≥1/10 solvability filter on the current model.

**Starting data.** BIRD and Spider training sets (after removing empty-result and slow gold); SynSQL-2.5M (2,544,390 samples over 16,583 databases, OmniSQL); EvolSQL (129,268 synthesized items); Gretel-Synth schemas (populate them so the gold SQL returns rows); TableDreamer (27,083 samples, GPT-4o references to be replaced by executed answers for RL). Evaluate on BIRD dev/test, Spider test and EHRSQL.

**Key references**

| Work | Verified result |
|---|---|
| [Arctic-Text2SQL-R1 (2025)](https://arxiv.org/abs/2505.20315) | BIRD test 71.83% (32B) |
| [OmniSQL (2025)](https://arxiv.org/abs/2503.02240) | 2,544,390 samples over 16,583 databases; OmniSQL-7B Spider test 87.9, BIRD dev 63.9 |
| [EvolSQL (2026)](https://arxiv.org/abs/2601.04875) | BIRD-dev ablation: full 65.1; without operator-guided evolution 62.7; without it and without exploratory expansion 57.4 |
| [TableDreamer (2025)](https://arxiv.org/abs/2506.08646) | Llama3.1-8B 10-benchmark average 49.07 → 60.69 |

---

## 10. Open-ended and non-verifiable tasks

For open-ended seeds (chat, writing, consultation, research reports) two things must be hardened: **the prompt and the grader.** Where possible, first **convert to a verifiable form**.

### Convert to verifiable (preferred when it applies)

- **Remove options, keep uniqueness.** Drop exam items small models solve, filter for answer uniqueness, rewrite as open-ended, and grade with an alias-aware judge audited on about 200 samples. HuatuoGPT-o1's GPT-4o verifier was 96.5% / 94.5% accurate vs 70.5% / 74.5% for regex exact match ([HuatuoGPT-o1, 2024](https://arxiv.org/abs/2412.18925)).
- **Mask-and-choose from unverifiable text.** Mask a key reasoning span and offer 9 style-matched options; with 3 options most items were too easy, and an open-ended variant gave zero accuracy on >83% of items. About 70% of items give useful signal, 13× more effective examples than ProRL's data ([Golden Goose, 2026](https://arxiv.org/abs/2601.22975)). **Moderate.**
- **Inject a known latent.** Give one of five players a masked copy and reward detection of the "spy". Qwen3-8B won 75.4% (summarization) and 77.3% (creative writing) of GPT-4o pairwise comparisons against its base ([SpyRL / RLSVR, 2026](https://arxiv.org/abs/2607.23802)). **Emerging.**
- **Ground in hidden documents.** Tasks require evidence the solver must retrieve; a frozen judge writes rubrics from a source document the solver never sees (up to +10.4 on eight open-ended benchmarks, [SCOPE, 2026](https://arxiv.org/abs/2605.31433)). For reports, use rubric trees whose leaves are executable checks ([QUEST](https://arxiv.org/abs/2605.24218)).

### Harden the prompt

| Operator | Easy → hard | Evidence |
|---|---|---|
| **Append one atomic constraint when rollouts become indistinguishable** | A prompt where two rollouts tie → same prompt + one constraint on an unspecified dimension + its rubric item | [LLM-as-a-Tutor (2026)](https://arxiv.org/abs/2607.04412): 51.96 vs 50.51 base rubrics vs 50.24 offline Evol-Instruct |
| **Scenario grounding on teacher key points** | "How much testosterone is safe in men?" → a sports-medicine consultant reviewing an Olympic powerlifter at 1,450 ng/dL | [QUBRIC (2026)](https://arxiv.org/abs/2606.03968): ArenaHard-hard 71.0 → 76.5; surface-only rewrite 67.1; keep a 20–50% pass band (no filter: 69.7) |
| **Static QA → simulated interactive consultation** | A single-turn medical question → multi-turn consultation with a persona-driven simulated patient who reveals facts only when asked | [Baichuan-M2 (2025)](https://arxiv.org/abs/2509.02208): HealthBench Hard 34.7 |

Scenario grounding must use key points present in teacher answers: naive narrowing invents references to non-existent guidelines and makes every response fail (QUBRIC: 0/6 discriminative rubrics for the naive rewrite vs 10/13 co-designed).

### Harden the grader

- **Checklists mined from failures, with universal items.** Generate responses of varying quality, have a strong model list every failure mode with weights, route exact items to code, and always add "directly addresses the request" and "matches the required context and register". RLCF was the only method that improved all 5 benchmarks on Qwen2.5-7B-Instruct ([RLCF, 2025](https://arxiv.org/abs/2507.18624)).
- **Pitfall and absence items.** Essential / Important / Optional / Pitfall rubric items gave up to 31% relative gain on HealthBench over Likert judging ([Rubrics as Rewards, 2025](https://arxiv.org/abs/2507.17746)).
- **Rubric difficulty evolution.** When all rollouts pass, extract what separates the two best responses ("excellent" vs "exceptional") as new criteria: "Is the code correct?" → "Does it handle the edge case in O(n)?" Qwen3-14B-Base reached HealthBench 69.3 ([RubricHub, 2026](https://arxiv.org/abs/2601.08430)).
- **Co-evolving rubric generators.** A rubric generator trained for discriminativeness alongside the policy raised Qwen3-4B HealthBench from 12.62 to 20.80 (static golden rubrics: 15.82), and static rubrics *hurt* out-of-distribution FollowBench HSR (47.76 → 32.50) ([EvoRubrics, 2026](https://arxiv.org/abs/2606.23038)). **Emerging.**
- **Relative bars.** Compare each rollout against a random in-group reference with a pairwise generative RM ([Writing-Zero BRPO, 2025](https://arxiv.org/abs/2506.00103)). A scalar RM scored higher on WritingBench (8.87) but lower on the authors' test set (2.83 vs 3.84), with longer outputs (1,872 vs 1,292 mean length) and more self-justifying filler (417 vs 58).

### Rubric hacking is the default (Strong)

- With a weak (GPT-4o-mini) rubric verifier, the incorrect-credit rate rose from 39% to 65% (medical) and 63% to 75% (science) over training; 90.2% of custom-rubric weight sat on presence criteria ([Reward Hacking in Rubric-Based RL, 2026](https://arxiv.org/abs/2605.12474)).
- The training judge's score keeps rising while a gold judge peaks and falls (−3 on HealthBench-Hard, −22 on ResearchQA). Dropping 30–50% of criteria per step, with the subset shared across the rollout group, recovers +1 to +2 and +6 to +7 ([Rubric Dropout, 2026](https://arxiv.org/abs/2608.11669)).
- Generated rubrics were exploited in 8–26% of an unbiased 150-environment cut; certificate-faithful rubrics were exploited 0/45 on the stress cut ([ImpossibleRubrics, 2026](https://arxiv.org/abs/2609.16816)).

**Recommendations.** Split compound criteria; add absence and negative criteria; add universal anti-preamble and anti-artifact items; use a strong or fine-tuned verifier (AdvancedIF's fine-tuned verifier F1 0.728 vs 0.515 for the vanilla judge); train only inside a 20–50% pass band; keep a held-out cross-family judge panel to detect divergence; apply Rubric Dropout. Rubric-reward mechanics are in [Chapter 05](05-rl-playbook.md) and rubric verification in [Chapter 04](04-verification-and-quality-control.md).

**Starting data and evaluation.** WildChecklists (130k WildChat instructions with checklists, RLCF); RubricHub (about 110K rubrics); GooseReason-0.7M mask-and-choose items (Golden Goose); HuatuoGPT-o1's 40K verifiable medical problems. Evaluate on HealthBench (and HealthBench Hard), ArenaHard, WritingBench, FollowBench, AdvancedIF and InfoBench, with a held-out cross-family judge.

---

## 11. Cross-domain summary: what to build first

| Domain | First operator to try on saturated seeds | Verifier to trust | Knob to calibrate on the policy | Biggest trap |
|---|---|---|---|---|
| Tool use | Oracle-preserving perturbation, ID fuzzing, implicit steps | Final DB state, any valid path | Walk length, implicit-step share | Noisy user simulator |
| Web / deep research | Fuzz exposed constants; remove clues | Two-sided gate + uniqueness | Solving cost Ω, answer-hit time | Obfuscation → ambiguity; structure ≠ difficulty |
| GUI / computer use | Evaluator-first composition; start-state regression; explicit → implicit | Executable state checker, checker(golden)=1, checker(initial)=0 | 1–7 of 8 band, monitoring pool | VLM judge false positives; terminal shortcuts |
| Terminal | Solution-first extension (RST) | Fresh-sandbox oracle + contract validity + hacker–fixer | Recursion round; pass@k | GRPO stalls on all-fail groups |
| Spreadsheet / office | Workbook inversion + lower specificity | Real-engine recompute on perturbed inputs; preservation predicates | Transformations, specificity level | Wrong engine; self-reported success |
| Instruction following | Policy-adaptive appending; Chain/Selection | Code per constraint; per-constraint judge | p^k, (0, 0.5] band | Over-hardening; hacking |
| Long context | Premise dispersal; lexical-overlap removal; agent-mined distractors | Construction labels; set-F1; evidence F-beta | Hops and distractor quality, not length | Generator label bugs; recall hacking |
| Interaction / memory | Sharding, objective stacking, hidden-spec user | Original verifier + control view | K, N, H with a ρ-gate | Constant ask/abstain policies |
| Multimodal | Answer-hidden stem rewrite; level generators | Latent-state answer, never pixels | Level with replay | SFT regression; cosmetic knobs |
| SQL / tables | AST mutation; distractor tables | Execution, non-empty, <5 s gold | ≥1/10 solvable | Text-only "complex question" rewrites |
| Open-ended | Append one constraint; scenario grounding; rubric evolution | Constitutive rubrics + held-out judge panel | 20–50% band | Rubric hacking |

**Proposals with the highest expected value** (see [Chapter 10](10-idea-bank-and-roadmap.md) for ranking):

1. **Port RST-style recursive, solution-first escalation to spreadsheets, notebooks and SQL**, recomputing oracles by execution on perturbed inputs (§4c pseudo-code). Every component is published; the combination is not. **Proposal.**
2. **Compose delivery operators on one verifiable seed** (sharded + over-window + injected tool output + withholding user), re-running every control on the composite (§7). **Proposal.**
3. **Add a hacker–fixer pass and a no-op/no-data/closed-book control inside every escalation loop**, and report exploit rates per round; no work yet measures whether hackability grows with escalation. **Proposal.**
4. **Train the task proposer itself on learnability × validity rewards for GUI and tool environments.** Among the roughly 40 GUI works reviewed, none trains the proposer. For tool use the closest precedent is SPADE's environment designer, trained on hint-based regret rather than learnability × validity ([SPADE](https://arxiv.org/abs/2608.19197), §1); for search, SSP's proposer (§2). DeepSeek-V4.1-Flash reports training the model as a task constructor on difficulty and correctness rewards, but gives no formula or ablation ([2609.19969](https://arxiv.org/abs/2609.19969); see [Chapter 08](08-frontier-lab-practices.md)). **Emerging/Proposal.**
