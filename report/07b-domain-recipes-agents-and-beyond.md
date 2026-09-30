# Domain recipes II: tool use, web/deep research, GUI & computer use, terminal/office, instruction following, long context, multimodal, SQL/tables, open-ended

> **Key takeaways**
>
> - **Build the ground truth first and write the prose last, then harden the artifact, not the text.** Every robust pipeline in this chapter fixes an executed trace, a final database/VM state, a SQL result, a simulator state or a latent scene before any natural language exists. Text-only "make it harder" rewriting keeps failing: LLM task-description hardening did not beat untouched descriptions ([OpenThoughts-Agent, 2026](https://arxiv.org/abs/2606.24855)), "complex question" augmentation cut BIRD-dev from 64.9 to 62.5 ([Arctic-Text2SQL-R1, 2025](https://arxiv.org/abs/2505.20315)), and offline Evol-Instruct prompts scored 50.24 vs 50.51 for unchanged prompts ([LLM-as-a-Tutor, 2026](https://arxiv.org/abs/2607.04412)). **Strong.**
> - **On saturated seeds, start with label-preserving operators that reuse your existing checker:** oracle-preserving perturbation and identifier fuzzing for tool calls; start-state regression, explicit→implicit goals and evaluator-first composition for GUI; instruction abstraction and multi-test-case perturbation for terminal and spreadsheets; sharding, objective stacking and hidden-spec users for any verifiable seed, each with a control (CONCAT, closed-book, no-op, no-data) that separates "hard" from "broken". **Strong.**
> - **Measure difficulty on the current policy and re-measure every iteration; structural proxies are loose.** Search tasks that look deep can be shallow in practice: about 20.6 retrieval calls for InfoSeek vs 141.0 for FORT ([FORT, 2026](https://arxiv.org/abs/2606.12087)). Only 19% of already-validated terminal tasks were learnable on first probe ([CalibForge, 2026](https://arxiv.org/abs/2608.06352)). Keep a band (1–7 of 8, (0, 0.5], 20–50%) and park 0/k tasks in a cheap monitoring pool instead of deleting them. **Strong.**
> - **The verifier is now the bottleneck, and its errors are asymmetric.** VLM and LLM judges over-accept: the same GUI policy scores 63.76% by VLM judge and 38.93% by code assertions ([GUI-Genesis, 2026](https://arxiv.org/abs/2602.14093)), and no LLM-judge setup exceeded AUROC 0.65 at detecting false success on tau2-bench ([2606.09863](https://arxiv.org/abs/2606.09863)). Hand-written scripts under-accept. Separate the solution-builder from the checker-writer, require checker(golden)=1 and checker(initial)=0, and run hacker–fixer probes before RL. **Strong.**
> - **When hardening drives pass rates toward zero, change the reward and the estimator along with the data.** Per-assertion counts with a critic took terminal RL to 64.0 on TB2.1 while GRPO stayed flat at 51.7 ([T1, 2026](https://arxiv.org/abs/2609.11042)). Dense signals only help when they are anchored in construction metadata (evidence chunks, gold entities, information gain). Use RL rather than SFT for composed and horizon-extended skills, with SFT as a cold start. **Moderate.**
> - **Environment and state diversity scale separately from task count.** When wording saturates, add environments, initial states and harnesses: 10→80 environments gave gains more trajectories could not recover ([CUA-Gym, 2026](https://arxiv.org/abs/2605.25624)), and RL on synthetic general-agent environments transferred where code-and-search RL did not ([DeepSeek-V3.2, 2025](https://arxiv.org/abs/2512.02556)). **Strong.**

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
| **Boundary- and failure-targeted resampling** | A pool where most tasks pass 100% → each epoch, structurally isomorphic variants of tasks with mean progress reward in [0.20, 0.85]; tasks above 0.95 retired | New variants pass the same executable validation; cap injection at 20% of the active pool per epoch | [RODS (2026)](https://arxiv.org/abs/2606.19047): Qwen3-4B BFCL V3 multi-turn 56.00 vs 50.00 for static-data GRPO, with about 20× fewer trajectories than a 17K offline pipeline; [SENTINEL (2026)](https://arxiv.org/abs/2606.12908): tau2 Retail pass^1 66.4 → 74.9 |

The first four are **oracle-preserving**: the gold calls and final state do not change, only what the model is shown. For a saturated, already-verified tool set they are the cheapest wins (**Moderate**: several independent works, all evaluated mostly on BFCL and tau2).

### Architecture

```python
def synthesize_tool_task(env, stats, L):
    # 1. gold first: walk the tool graph, biased toward tools the policy fails (HardGen)
    #    and toward weak/independent edges (Agent-World); fill args from prior outputs or DB rows
    chain = sample_walk(env.tool_graph, start_bias=stats.failing_tools,
                        edge_weights={"strong": 3, "weak": 2, "independent": 1}, max_len=L)
    s0 = env.reset(seed)
    trace = execute(s0, chain)                 # every call executed immediately (SAP)
    assert trace.ok and trace.final_state != s0

    # 2. complexify the specification, not the answer
    spec = abstract(trace,
                    fuzz_ids=True,               # CoVe: DB-unique descriptions
                    implicit_steps=k,            # HardGen: hide intermediate calls
                    provenance={"prev_user_msg": 0.3, "prev_output": 0.4})  # SAP
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

FORT's cumulative ablation also ranks the controls. Removing them one at a time raises a strong agent's accuracy from 29.0 (full) → 36.5 (− cycles) → 42.7 (− long-tail) → 53.2 (− derived facts) → 57.4 (− source diversity) → 65.0 (− generic facts) → 81.6 (− fuzzing). **Fuzzing is the most important single control**, and hardening by *removal* (WebExplorer: Claude-4-Sonnet 86.6% → 67.1%) is as useful as adding hops.

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
