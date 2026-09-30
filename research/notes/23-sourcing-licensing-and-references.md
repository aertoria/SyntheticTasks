# Build vs source: open hard-task datasets and environment blends, research-mined and expert tasks, licensing and provenance of generator outputs, and the survey/classic reference backbone

*Scope: when to source hard tasks (open RL blends and pools, research-mined tasks, expert-authored tasks) instead of synthesizing them; what licenses and regulations attach to generator outputs; and the surveys and classic automatic-curriculum papers that give the report its reference structure. Compiled 2026-09-30. Verification: 32 entries checked against primary sources (arXiv abs/HTML/PDF, Hugging Face dataset cards and Hub API, license files, official terms pages). 11 corrected, 1 dropped, 4 added. Every Hugging Face license field cited here was read from the Hub API on 2026-09-30.*

Cross-references, not re-covered here: note 12 covers the Nemotron 3, INTELLECT-3, Olmo 3 and Skywork-OR1 training recipes. Note 16 covers OpenThoughts-Agent. Note 14 covers HLE / HLE-Verified. Notes 02, 04, 11 and 12 cover the POLARIS, DeepMath, Big-Math, KodCode, rStar-Coder and HardTests pools, and note 11 covers LLM-judge hacking. Note 19 covers "Idiosyncrasies in LLMs" and subliminal learning, which bear on teacher identifiability.

## TL;DR

- **Before you synthesize, borrow a blend, but re-profile it.** NVIDIA publishes its full RL blends with per-stage mixing ratios:
  - Nano: 93,244 rows.
  - Super: 479,303 rows in 6 stages (RLVR1-3 → SWE1-2 → RLHF).
  - Ultra: 337,721 rows, including an 85,980-row MOPD stage.
  - Lightning: 92,684 rows.

  Every blend is "ordered from higher pass-rate (easier) to lower pass-rate (harder)", but the pass rates come from NVIDIA's checkpoints, and no pass-rate column is shipped. Treat inherited difficulty labels as priors: SYNTHETIC-2 uses Qwen3-4B/32B and R1-0528-Qwen3-8B, and RealMath uses "older, weaker models". Profile every pool with your own policy.
- **Licenses on open RL pools are unreliable, so audit them before training.** On 2026-09-30 the Hub API showed:
  - No license tag: Skywork-OR1-RL-Data, SYNTHETIC-2-RL, SYNTHETIC-2-SFT-verified, OpenThoughts-Agent-v1-RL, II-Thought-RL-v0, INTELLECT-3-RL, Dolci-Think-RL-7B, Nemotron-RL-knowledge-mcqa.
  - Non-commercial (CC-BY-NC-4.0): KodCode-V1, KodCode-Light-RL-10K, facebook/natural_reasoning.
  - Share-alike: Nemotron-RL-coding-competitive_coding (CC-BY-SA-4.0) and ResearchMath-14k (CC BY-SA 4.0 per its paper).

  This matches the audits. DPI found hosting sites omitting licenses 70%+ of the time and miscategorizing them 50%+ of the time. The LLMware study found 35.4% of artifacts undeclared and 52% of supply chains with at least one conflict.
- **Obligations travel through synthetic data, and dataset cards miss some of them.**
  - Llama 3.1 and Llama 4: any distributed model trained on Llama outputs must put "Llama" at the beginning of its name.
  - Gemma 1-3 Terms: a model trained on Gemma synthetic outputs is a "Model Derivative".
  - Qwen2.5-72B: requires "Built with Qwen" or "Improved using Qwen". Qwen2.5-3B is research-only.
  - Llama-Nemotron-Post-Training contains 464,658 Qwen-2.5-72B-Instruct rows, but its card warns only about the Llama 3.1/3.3 licenses.
- **Default to permissively licensed generators and verifiers:** DeepSeek-R1/V3.2 and GLM-4.5/4.7/5 (MIT), and Qwen3/3.5, gpt-oss, Olmo 3 and Gemma 4 (Apache-2.0). Kimi K2/K2.5 and MiniMax-M2 use modified MIT with UI attribution above 100M MAU or a revenue threshold. Anthropic's Commercial Terms (D.4) and the Gemini API terms (effective 2026-03-23) bar using their services to train competing models. Log the generator on every row, as NVIDIA's `license`/`generator` fields do. EU AI Act template Section 2.5 requires naming the GPAI models used to generate synthetic training data, explicitly including "AI feedback through reinforcement learning". AI Office enforcement started 2 Aug 2026.
- **Research-mined tasks are the cheapest self-refreshing source above competition level, but yield is low and difficulty is uneven.**
  - RealMath: 14,747 theorems → 280 usable QA, about 94% valid. o3 scored 97.5% on "easy" items and 27.9% on "hard" ones.
  - LemmaBench: 52.8–64% of extracted lemmas are self-contained. GPT-5-family accuracy rose from 12.3% to 40.8% in nine months.
  - SWE-rebench: about 153,400 candidate PRs → 21,336 executable tasks, with a working install recipe for 31% of repositories.

  Always add a policy pass-rate filter.
- **At the research frontier you cannot filter by correctness, and SFT still helps.** On ResearchMath-14k (14,056 problems mined from open-problem papers, problem sessions and web pages), LLM judges rated only 3.7% and 4.3% of teacher trajectories correct. Training on behaviorally filtered trajectories still gave +2.1 points on graduate- and research-level math, where DASD gave 0.0 and Nemotron-SFT-Math-v4 gave −0.5. Diversity of research-level content is a lever separate from verified difficulty.
- **Expert-authored tasks are still the only reliable way to get tasks that frontier models fail.** Examples:
  - CritPt full challenges: 5.7% (GPT-5 high).
  - FrontierScience-Research: 25%.
  - EXP-Bench complete experiments: 0.5%.
  - PaperBench: 21.0%.

  Known costs: GPQA paid about $95/hour. GDPval gold tasks averaged 404 minutes, or $361, per task plus about 5 reviews. HLE ran a $500K prize pool; 70,000+ LLM attempts produced about 13,000 model-stumping candidates and 2,500 final questions. Spend expert time on adjudication and decomposition, not on bulk authoring.
- **Checkpoint and rubric decomposition turns 0–6%-pass research tasks into dense signal.** CritPt splits 71 challenges into 190 checkpoints. PaperBench splits 20 papers into 8,316 gradable leaves. FrontierScience uses 10-point rubrics with success at 7/10. This is the natural fix for zero-advantage GRPO groups on research-scale tasks.
- **The classic curriculum literature already wrote the acceptance tests.**
  - Goal GAN's intermediate-difficulty band [0.1, 0.9] is "very robust" to any R_min ∈ (0, 0.25) and R_max ∈ (0.75, 1).
  - Setter-solver requires validity, feasibility and coverage; the alchemy environment needed validity and coverage.
  - Asymmetric self-play pays Alice for tasks "just beyond" Bob's ability.
  - Polu et al. found composition depth N_D to be the main difficulty driver, and expert iteration closed N_D = 6 statements that sampling alone never solved.
  - STaR's rationalization rescues zero-pass items.
- **Where a permissive or regenerated equivalent exists, prefer it to an unlicensed pool.** DFM Mimir replaced non-permissible Flan/Platypus/tasksource MCQ tasks with 70 Gemma-4-31B "transplant" datasets (75M tokens) in generative form. Its claim of "comparable or superior performance" appears in the introduction without a matched ablation.

## Methods at a glance

| Method | Year | Link | Domain | Used for | Complexification operator(s) | How correctness is ensured |
|---|---|---|---|---|---|---|
| Nemotron post-training datasets + NeMo Gym RL blends | 2025–26 | [HF Super blends](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) | math, code, STEM, chat, tools, IF, SWE, safety, RLHF | SFT+RL | pass-rate-ordered curriculum; stage-wise re-weighting; complexity/easy-to-guess prompt filtering; placeholder pointers to third-party pools | per-row `verifier_type`/`agent_ref` (NeMo Gym); Math-v2 answers "verified for correctness using GPT-5.2" |
| SYNTHETIC-2 / SYNTHETIC-2-RL | 2025 | [blog](https://www.primeintellect.ai/blog/synthetic-2-release) | math, code, puzzles, formatting | SFT+RL | new verifiable families (code-output prediction, Pydantic, JSON, unscrambling, ASCII trees, Formatask); multi-model pass-rate annotation | programmatic verifiers; SFT-verified keeps reward 1 (binary) or > 0.7 (non-binary) |
| SWE-rebench (+V2) | 2025–26 | [2505.20411](https://arxiv.org/abs/2505.20411) | SWE (Python; V2: 20 languages) | SFT+RL, eval | continuous PR mining; LLM-synthesized install recipes; learned complexity/clarity labels; temporal freshness | fail-to-pass / pass-to-pass execution in containers; V2 adds an LLM-judge ensemble |
| II-Thought-RL-v0 + Multi-subject RLVR | 2025 | [2503.23829](https://arxiv.org/abs/2503.23829) | math, code, science, medicine; 48 exam subjects | RL | PDF mining with sliding-window LLM extraction; MCQ → free-form; RL-suitability filtering | expert reference answers; generative (soft) reward; tests required for code |
| RealMath | 2025 | [2505.12575](https://arxiv.org/abs/2505.12575) | research math (arXiv, Math.SE) | eval (pipeline reusable for RL) | theorem → fixed-answer QA; trivial-question filter; monthly refresh | the paper's stated result; LLM equivalence judge; about 6% bad on manual review |
| *Added:* LemmaBench (+ LiveMathematicianBench) | 2026 | [2602.24173](https://arxiv.org/abs/2602.24173) | research math (arXiv lemmas) | eval; historical runs usable for training | lemma extraction + self-containment rewriting; weekly snapshots; proof-sketch distractors; substitution-resistant MCQ | LLM proof judges with human spot checks (67–83% human confidence) |
| *Added:* ResearchMath-14k | 2026 | [2605.28003](https://arxiv.org/abs/2605.28003) | research / open-problem math | SFT | open-problem harvesting; definition inlining; resolution-status search | none for answers (judges rate 3.7–4.3% of trajectories correct); behavioral filtering; PhD self-containment review |
| MathConstruct | 2025 | [2502.10197](https://arxiv.org/abs/2502.10197) | olympiad constructive proofs | eval | proof → construction; parameter re-instantiation; robust accuracy; brute-force exclusion | executable verifier V_θ checks every constraint |
| SPARK (Spark-234K) | 2026 | [2608.30214](https://arxiv.org/abs/2608.30214) | science, 10 disciplines | SFT | paper → claim-evidence-derivation skeleton; 4 reasoning perspectives | skeleton-consistency check (93.28%); 90.3% expert agreement on 300 samples |
| PaperBench + EXP-Bench | 2025 | [2504.01848](https://arxiv.org/abs/2504.01848) | ML research replication | eval | hierarchical rubric trees; masked files in the real repo | author-co-developed rubrics + validated LLM judge; paper code and outputs as ground truth |
| ORQA | 2026 | [2609.12366](https://arxiv.org/abs/2609.12366) | occupational knowledge (116 SOC occupations) | eval | trusted-source mining → source-traceable QA | authoritative source per answer + human review |
| *Added:* GPQA | 2023 | [2311.12022](https://arxiv.org/abs/2311.12022) | graduate biology, physics, chemistry | eval (authoring protocol) | incentive-aligned expert writing (paid when experts agree and non-experts fail) | 2 expert validators + 3 non-expert validators; revision after the first expert |
| SuperGPQA | 2025 | [2502.14739](https://arxiv.org/abs/2502.14739) | 285 graduate subfields | eval | calculation → MCQ; confusion options (about 9.67 per item); LLM-accuracy-based tailoring of easy items | expert source screening; 7-LLM suspicious-item flagging; expert re-annotation with web |
| GDPval | 2025 | [2510.04374](https://arxiv.org/abs/2510.04374) | 44 occupations, 9 sectors | eval | deliverable-grounded multi-hour tasks with reference files | model screening + an average of 5 expert reviews; blinded pairwise expert grading |
| APEX | 2025 | [2509.25721](https://arxiv.org/abs/2509.25721) | banking, consulting, law, primary care | eval | expert prompts + rubrics; source-license policy | multi-round expert review; LM judge (Gemini 2.5 Flash) |
| CritPt + FrontierScience | 2025–26 | [2509.26574](https://arxiv.org/abs/2509.26574) | frontier physics; olympiad and PhD-level science | eval | composite challenge → checkpoints; guess-resistant answers; 10-point rubrics; rejection against internal models | physics-specific autograder; author and expert review; GPT-5 (high) judge |
| Data Provenance Initiative audits | 2023–24 | [2310.16787](https://arxiv.org/abs/2310.16787) | dataset licensing | analysis | license re-annotation along the derivation chain | manual legal/ML annotation (Data Provenance Explorer) |
| *Added:* Hidden Licensing Risks in LLMware (LiAgent) | 2026 | [2602.10758](https://arxiv.org/abs/2602.10758) | OSS / model / dataset supply chains | analysis | supply-chain license-conflict detection | LLM agent (87% F1); 11 of 60 reported conflicts confirmed by developers |
| Generator license & terms map (+ Who Taught You That?) | 2023–26 | [Gemma ToU](https://ai.google.dev/gemma/terms) | licensing of generator outputs | analysis | generator selection by license; attribution/naming propagation; teacher tracing | N/A (legal terms); PoS-template teacher attribution |
| EU AI Act training-content summary template §2.5 | 2025 | [EC page](https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models) | regulatory provenance | analysis | generator-lineage disclosure | N/A |
| DFM Mimir v1 transplant datasets | 2026 | [2608.13517](https://arxiv.org/abs/2608.13517) | permissible post-training data | SFT | license transplant: regenerate non-permissible tasks with an Apache-2.0 generator (MCQ → generative) | quality audit before inclusion (acceptance from single digits to the high 90s, by category) |
| Agentic Environment Engineering survey | 2026 | [2606.12191](https://arxiv.org/abs/2606.12191) | agent environments | analysis | taxonomy: task-driven / real-world-driven / de novo symbolic synthesis; explicit vs implicit curricula | QC axes: correctness, diversity, complexity, fidelity |
| Scaling LRMs beyond Human Supervision (L0–L4) | 2026 | [2608.31075](https://arxiv.org/abs/2608.31075) | reward and experience autonomy | analysis | ladder; experience-axis failure modes | "recommended minimum scorecard" |
| RL / agentic / self-improvement survey set | 2025–26 | [2509.08827](https://arxiv.org/abs/2509.08827) | RLVR, agentic RL, data-scarce RL, self-evolution | analysis | taxonomies (static corpus vs 5 dynamic-environment types; pruning/synthesis/compression; asymmetric co-evolution) | N/A |
| QDC survey | 2024 | [2412.02980](https://arxiv.org/abs/2412.02980) | synthetic data | analysis | complexity measures (basic, attribute, model-based) | N/A |
| Asymmetric self-play + Goal GAN + AMIGo | 2017–20 | [1703.05407](https://arxiv.org/abs/1703.05407) | goal-conditioned RL | RL | proposer paid for solver effort; GOID band; adaptive step threshold | proposer's own trajectory proves feasibility; simulator checks the goal |
| Setter-solver | 2019 | [1909.12892](https://arxiv.org/abs/1909.12892) | goal-conditioned RL | RL | setter losses: validity, feasibility (uniform target), coverage; desirability; observation conditioning | validity learned from goals the solver actually achieved |
| TSCL + Graves bandit + ACL survey | 2017–20 | [1707.00183](https://arxiv.org/abs/1707.00183) | curricula over subtask sets | RL/SL | learning-progress (\|slope\|) task selection; nonstationary bandit | fixed task set with known evaluators |
| INT + Polu statement curriculum | 2020–22 | [2202.01344](https://arxiv.org/abs/2202.01344) | formal inequalities, Lean | RL (expert iteration) | forward generator with depth (N_D) and size (N_S) knobs; proof-less graded statements | Lean kernel; theorems true by construction (INT) |
| MetaGen | 2020 | [2002.07019](https://arxiv.org/abs/2002.07019) | Metamath | SFT (prover) | learned forward theorem generator (IL / RL) | forward rule application: valid by construction |
| STaR | 2022 | [2203.14465](https://arxiv.org/abs/2203.14465) | arithmetic, GSM8K, CommonsenseQA | SFT | rationalization: hint with the answer, train without it | final answer must match ground truth |

## Method notes

**A. Open pools and environment blends: sourcing instead of synthesizing**

### Nemotron post-training datasets + NeMo Gym RL training blends — Nemotron-Post-Training-Dataset-v1; Nemotron-3-Nano-RL-Training-Blend; Nemotron-RL-Super/Ultra-Training-Blends; Nemotron-RL-Lightning-Training-Blend; Nemotron-RL-Math-v2; Llama-Nemotron-Post-Training-Dataset (NVIDIA et al., 2025–2026)
Links: https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends · https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends · https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend · https://huggingface.co/datasets/nvidia/Nemotron-RL-Lightning-Training-Blend · https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2 · https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1 · Llama-Nemotron report arXiv:2505.00949. Training recipes are in note 12. This entry covers only the released artifacts.

- **Mechanism**
  - *SFT (v1)*: 25,659,642 rows (chat 746,622; code 1,896,395; math 2,044,407; stem 20,662,167; tool calling 310,051) for Llama-3.3-Nemotron-Super-49B-v1.5.
    - Responses: DeepSeek-R1-0528 (24,602,969) and Qwen3-235B-A22B (1,056,673).
    - Prompts were "filtered for quality and complexity, or generated to meet quality and complexity requirements". This included "removing inconsistent prompts, prompts with answers that are easy to guess, and removing prompts with incorrect syntax".
    - Every row has `license` and `generator` fields.
    - Some chat and code prompts ship empty and must be fetched from the original source (e.g., lmsys-chat-1m).
  - *RL blends*: named component datasets with mixing ratios.
    - **Nano** (93,244 rows, ODC-BY): competitive coding 0.25, knowledge-MCQA 0.20, IF 0.17, Skywork-OR1 (excluding OmniMath) 0.12, DAPO-Math-17k 0.10, workplace assistant 0.10, structured outputs 0.05.
    - **Super** (Nemotron-3-Super-120B-A12B; 479,303 rows, CC-BY-4.0): 6 stages, rlvr1 138,712 / rlvr2 156,278 / rlvr3 107,037 / swe1 50,661 / swe2 1,444 / rlhf 25,171. RLVR3 raises conversational tool use to 29.82% and GenRM data to 29.82%. SWE1 is 79.70% R2E-Gym-Subset and 20.30% SWE-Gym (SWE2 81.18/18.82). RLHF is 77% GenRM.
    - **Ultra** (337,721 rows): stages rlvr1 98,424, rlvr2 99,116, ifbench 34,649, rlhf 6,500, reasoning 5,236, swe 7,816, mopd 85,980. It adds SWE-rebench-V2 (97.36% of the swe stage), ARC-AGI, QA-Abstention, InverseIFEval, Indirect-Prompt-Injection and Litmus-Bench. Its "reasoning" stage is 100% virtuoussy/Multi-subject-RLVR (see the II-Thought entry).
    - **Lightning** (Nemotron-3.5-Lightning RLVR stage, created 2026-08-04): 92,684 rows.
  - All blend cards say samples "are ordered from higher pass-rate (easier) to lower pass-rate (harder)".
  - DAPO-Math-17k and Skywork-OR1 rows ship as placeholders that `fill_placeholders.py` fills from the original repos.
  - The `tacos` and `apps` subsets of competitive coding are excluded.
- **How it makes tasks harder**: mostly by selection and ordering, not by construction. Prompts that are easy to guess are removed. Blends are sorted easy → hard. Later stages shift weight toward agentic tool use, SWE and preference/GenRM data.
- **Correctness / verification**
  - Rows reference a NeMo Gym agent and verifier. Nemotron-RL-Math-v2 has `verifier_type`, `agent_ref` and `expected_answer` fields.
  - The Math-v2 card states: "All problems and expected answers are verified for correctness using GPT-5.2 model."
  - Math-v2 (7,732 rows) draws from AoPS, StackExchange math held out from Nemotron-SFT-Math-v4, Skywork-OR1, DAPO and "vendor-purchased data".
- **Difficulty control**: pass-rate ordering against NVIDIA checkpoints, plus stage-wise mixture shifts. The cards I read expose no per-row pass-rate field, so the ordering itself is the only difficulty signal.
- **Reported results**: none on the cards. Downstream results are in the Nemotron 3 technical reports (note 12).
- **Limitations / failure modes**
  - Licensing is layered. Blend cards list CC-BY-4.0 plus "Apache 2.0; MIT; BSD-3", and Ultra and Lightning add CC BY-SA 4.0 and ODC-BY 1.0. Competitive coding is CC-BY-SA-4.0, and knowledge-mcqa has no license tag.
  - Llama-Nemotron-Post-Training warns that trained models "may be subject to redistribution and use requirements in the Llama 3.1 Community License Agreement and Llama 3.3 Community License Agreement". Its generator table also lists 464,658 Qwen-2.5-72B-Instruct rows. The Qwen license requires "Built with Qwen" / "Improved using Qwen" for distributed models trained on its outputs, and the card does not mention this.
  - Answers verified by a closed model (GPT-5.2) are a lineage fact to record.
- **How to reuse with easy seed tasks**
  - Treat the blends as a ready-made multi-domain pool and staging template (verifiable RLVR → SWE → preference).
  - Re-profile every row with your policy (e.g., 8–16 samples), drop p ∈ {0, 1}, and re-sort.
  - Where your own complexified tasks exist, slot them into the same stage structure.
  - Copy the per-row `license` + `generator` schema into your synthetic data.
  - Make a separate license decision for placeholder and externally sourced rows, since the placeholders keep the upstream terms.

### SYNTHETIC-2 / SYNTHETIC-2-RL — SYNTHETIC-2: open reasoning dataset with difficulty annotations (Prime Intellect, 2025)
Links: https://www.primeintellect.ai/blog/synthetic-2-release · https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2 · https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL

- **Mechanism**
  - DeepSeek-R1-0528 generated reasoning traces through decentralized inference; the blog describes "four million verified reasoning traces" and "1,253 GPUs from around the world".
  - Every task was also attempted by Qwen3-32B, Qwen3-4B and DeepSeek-R1-0528-Qwen3-8B.
  - The full dataset has 353,945 prompts, with per-model responses and `_avg_reward` columns for all four models.
  - The RL split has 155,638 tasks with the four average-reward columns and `verification_info`.
  - SFT-verified keeps R1-0528 responses with reward 1 (binary) or above 0.7 (non-binary). SFT-unverified keeps all.
- **How it makes tasks harder**: it adds verifiable task families that are not competition math. Code-output prediction on "complex, library-heavy code". Pydantic adherence to LLM-generated schemas. Complex JSON formatting. Sentence/block unscrambling. ASCII directory-tree formatting. Formatask (exact extraction of described text spans).
- **Correctness / verification**: programmatic verifiers in the prime-rl / verifiers stack (execution, schema validation, exact reorder match). The blog cites TOPLOC checks on the decentralized inference, with a reported false-positive rate of 0.000925%.
- **Difficulty control**: a graded label from three annotator models of increasing strength plus R1-0528 itself. The blog says "difficulty filtering has proven to be crucial for RL training performance".
- **Reported results**: about 4M verified traces. INTELLECT-3 later used 8.6K SYNTHETIC-2 code problems with up to 15 tests (note 12).
- **Limitations / failure modes**
  - The annotators are mid-2025 4B–32B models, so much of the pool will be easy for a 2026 policy.
  - Several families are formatting-heavy.
  - The license varies by split: the main SYNTHETIC-2 card is Apache-2.0, but SYNTHETIC-2-RL and SYNTHETIC-2-SFT-verified carry no license tag (Hub API, 2026-09-30).
- **How to reuse with easy seed tasks**
  - Select rows where all three small annotators have avg_reward = 0 but R1-0528 > 0, then re-profile on your policy.
  - Copy the scheme: annotate each synthetic task with 2–4 reference models of different strength. It gives a graded difficulty label, better than a binary one, for about 3–4× the rollout cost of one model.

### SWE-rebench (+ V2) — SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents (Badertdinov et al., 2025); SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale (Badertdinov et al., 2026)
Links: https://arxiv.org/abs/2505.20411 (NeurIPS 2025) · https://arxiv.org/abs/2602.23866 · https://huggingface.co/datasets/nebius/SWE-rebench-V2

- **Mechanism**
  - Continuous mining of GitHub issue–PR pairs that have test patches. About 153,400 candidates survive the first filters. Repositories are restricted to permissive licenses (MIT, Apache-2.0, BSD variants, ISC, CC0-1.0, and manually verified others).
  - Qwen2.5-72B-Instruct writes up to three install recipes per task and refines a failing recipe from its error logs.
  - Execution confirms that tests fail before and pass after the gold patch.
  - A Qwen2.5-72B-Instruct model fine-tuned on 3,800+ SWE-bench Verified annotations labels task complexity, issue clarity and test-patch correctness.
  - V2 replaces the single-shot recipe with an interactive setup agent and filters unsound instances with an ensemble of LLM judges validated against human SWE-bench annotations.
- **How it makes tasks harder**: tasks are new, not constructed to be harder. Issue creation dates are compared with model release dates, and the leaderboard flags potentially contaminated evaluations. The learned complexity label lets users select harder tasks.
- **Correctness / verification**: fail-to-pass and pass-to-pass execution in containers. V2 ships pre-built images and instance-level flags for confounders "such as overly restrictive tests and underspecified descriptions".
- **Difficulty control**: the complexity label (81% accuracy, weighted F1 0.82), the clarity label (79% accuracy, F1 0.76) and test-patch correctness (67% accuracy, F1 0.65). In V2, a diagnostic study with seven models across five languages.
- **Reported results**
  - V1: 21,336 tasks from 3,468 repositories. A working recipe was produced for at least one task in 31% of repositories.
  - V2: 32,079 tasks, 20 languages, 3,617 repositories, plus "120,000+ tasks with installation instructions, fail-to-pass tests and rich metadata" whose problem statements are generated from PR descriptions.
  - SWE-rebench-V2 is 97.36% of the Nemotron 3 Ultra SWE stage.
- **Limitations / failure modes**: automated labels are noisy (test-patch correctness is only 67% accurate). Real-PR difficulty is not graded against your policy. The 120K extended set uses generated problem statements, which is a synthetic-text provenance step.
- **How to reuse with easy seed tasks**
  - Use SWE-rebench as a large executable seed pool: select by complexity label, re-profile with your agent, then apply the SWE hardening operators in note 04 (bug injection, PR merging, test strengthening, hint removal).
  - Harvest post-cutoff PRs monthly as a clean held-out set.
  - The dataset is CC-BY-4.0, but V2 ships an SPDX `license` field for each instance's repository. Filter on it if you redistribute code.

### II-Thought-RL-v0 + Multi-subject RLVR — II-Thought RL v0 (Intelligent Internet, 2025, HF card); Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains (Su et al., 2025)
Links: https://huggingface.co/datasets/Intelligent-Internet/II-Thought-RL-v0 · https://arxiv.org/abs/2503.23829 · https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR

- **Mechanism**
  - *II-Thought*: IMO / IMO-Shortlist PDFs are parsed with MinerU, and Gemini-2.0-Flash reads the Markdown "in a sliding window fashion" to extract problem/solution pairs from long PDFs. The same extraction runs over "20 years of ICPC/regional" problems, adding test cases.
  - The filters are:
    1. Regex removal of proofs, MCQ and multi-part items.
    2. A Gemini 2.0 Flash quality grade, keeping only "good" and "excellent".
    3. A Qwen-32B RL-suitability filter "following Big-Math".
    4. Domain checks: code must have tests, and science items need their essential figures.

    Near-deduplication and decontamination follow.
  - The II-Thought total is 341,795 samples from 15 sources: about 190k math, 90k code, 39k medical, 11k science and 12k other.
  - *Multi-subject RLVR* takes ExamQA: 638k college-level Chinese exam questions "written by domain experts", covering at least 48 first-level subjects.
    - Distractors are removed to make free-form QA.
    - GPT-4o-mini translates and classifies subjects; 15.8% of test items were unclassified.
    - 6,000 questions are held out.
    - A 7B generative reward model is trained on 160k distilled judgments, with Qwen2.5-72B-Instruct providing the hard labels. Rewards are binary or soft (the first-token probability).
- **How it makes tasks harder**: it recovers expert-written problems that were locked in PDFs, and removing options eliminates guessing.
- **Correctness / verification**: expert reference answers. The paper reports "high consistency across various LLMs" on binary verification when expert references exist.
- **Difficulty control**: quality grades only; no policy pass-rate labels.
- **Reported results**: the paper reports RLVR gains over Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B in free-form cross-domain settings. With RLOO and the RM-7B binary reward, it reports 63.0% average on math and 28.1% on multi-subject.
- **Limitations / failure modes**
  - Soft LLM rewards on this very set are hackable: note 11 records a 67.0% false-positive rate for a Qwen2.5-72B judge given the "Thought process:" master key.
  - II-Thought declares no license, and Multi-subject-RLVR is Apache-2.0 although it was translated with GPT-4o-mini. Both aggregate upstream sets that carry their own terms.
- **How to reuse with easy seed tasks**
  - Mine under-used expert sources (olympiad shortlists, national exams, regional ICPC) with sliding-window extraction.
  - Convert MCQ to free-form and keep the expert answer as the reference.
  - Grade with a hardened verifier rather than a raw judge.
  - Record the extraction and translation models as provenance.

**B. Research-mined tasks: automated harvesting above competition level**

### RealMath — RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics (Zhang et al., 2025)
Link: https://arxiv.org/abs/2505.12575 (NeurIPS 2025)

- **Mechanism**
  - The pipeline pulls papers in a time window from the arXiv API, parses their LaTeX into theorems, and uses an LLM (o3-mini by default) to keep only constructive theorems with a single fixed answer. It excludes inequalities, bounds and existence statements.
  - Each kept theorem becomes a question plus answer, with the preceding paper text as context.
  - A post-processing LLM pass drops "trivial samples that can be easily solved", including those where the answer is guessable or already given in the context.
  - Math.StackExchange answers are handled similarly.
- **How it makes tasks harder**: answers are real research results, often closed forms, such as the number of 3-cliques in the Peisert graph P*(q) for q = p^{2t}, p ≡ 3 (mod 4), which is q(q−1)(q−5)/48.
- **Correctness / verification**
  - The reference is the paper's stated result, and an o3-mini judge checks equivalence.
  - Manual review found about 6% of items below the quality bar.
  - The authors caution that arXiv papers "have not necessarily been peer-reviewed, and errors or ambiguous claims may be present".
- **Difficulty control**: none at construction. The authors stratify difficulty after the fact using the performance of "older, weaker models" and find it "highly nonhomogeneous".
- **Reported results**
  - Funnel: 4,000 papers → 3,922 with LaTeX → 14,747 theorems → 407 constructive fixed-answer → 401 QA → 280 after the trivial filter.
  - Totals: Math.arXiv 633 (May 2022–Mar 2025), CS.arXiv 111, Math.SE 542. The pipeline adds "over 70 new samples each month".
  - o3 scores 49.1 / 44.1 / 70.7% on Math.arXiv / CS.arXiv / Math.SE. Gemini 2.5 Pro scores 32.5 / 25.2 / 60.9, and DeepSeek-R1 30.5 / 31.5 / 62.2.
  - By difficulty stratum, o3 scores 97.5% (easy), 81.4% (medium) and 27.9% (hard).
- **Limitations / failure modes**: about 2% yield from extracted theorems. Only constructive, fixed-answer statements survive. The hard subset is small.
- **How to reuse with easy seed tasks**
  - Run the pipeline on arXiv papers published after your policy's cutoff to get fact-anchored RL questions.
  - Keep the 0 < p < 1 band under your policy.
  - Where possible, cross-check answers with a CAS or a second derivation.
  - Use it for fresh held-out evaluation as well as training.

### LemmaBench (+ LiveMathematicianBench) — LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics (Peyronnet et al., 2026); LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches (He et al., 2026) *(added)*
Links: https://arxiv.org/abs/2602.24173 · https://arxiv.org/abs/2604.01754

- **Mechanism**
  - *LemmaBench*: weekly snapshots of arXiv math.* papers. GPT-5 was chosen as extractor over Gemini 2.5 Pro and GPT-4.
  - Each lemma is rewritten into a self-contained statement "by making all assumptions and required definitions explicit". Two retrieval modes feed this: full-context (the whole preceding article) and vector retrieval (text-embedding-3-small, threshold τ = 0.75). Full-context completes more lemmas but costs about 5–10× more tokens.
  - An LLM judge checks self-containment. Proofs are graded by whole-proof LLM judgment, by step-level LLM judgment (which the authors acknowledge is "likely too strict"), and by expert spot checks.
  - *LiveMathematicianBench*: MCQ questions from post-cutoff arXiv theorems, with a 13-category taxonomy of theorem types.
    - Distractors come from a proof-sketch pipeline: plausible but invalid choices that follow misleading proof directions.
    - A "substitution-resistant" mode separates recognizing an answer from reasoning to it.
- **How it makes tasks harder**: tasks are fresh research results, and proof generation replaces fixed answers. In LiveMathematicianBench, proof-strategy distractors plus substitution resistance defeat surface matching.
- **Correctness / verification**
  - LemmaBench: human mathematicians confirmed that 75.5–96.5% of the lemmas the LLM judged self-contained really were, depending on method and model. Human confidence in the LLM proof verdicts ranged from 67% to 83%.
- **Difficulty control**: none explicit. Freshness is the mechanism. The authors note that "the same pipeline can be run on historical data, allowing the creation of large high-quality training datasets" without contaminating future evaluations.
- **Reported results**
  - LemmaBench batches:

    | Batch | Lemmas extracted | Judged self-contained |
    |---|---|---|
    | Aug 2025 | 376 | about 240 (64%) |
    | Feb 2026 | 677 (from 81 preprints) | 358 (52.8%) |
    | Apr 2026 | 287 | 120 |

  - LemmaBench accuracy: theorem-proving pass@1 rose from 12.3% (GPT-5, Aug 2025) to 29.6% (Feb 2026) to 40.8% (GPT-5.5, Apr 2026).
  - LiveMathematicianBench: the best model, Gemini-3.1-pro-preview, scores 43.5%. Under substitution-resistant evaluation GPT-5.4 is best at 30.6%, and Gemini-3.1-pro-preview falls to 17.6%, below the 20% random baseline.
- **Limitations / failure modes**: proof grading relies on LLM judges of moderate reliability. Self-containment fails on about 40–50% of extracted lemmas. Mined items saturate within months, so the pipeline has to keep running.
- **How to reuse with easy seed tasks**
  - Use self-containment rewriting as a complexification step: take a lemma from a recent paper, inline its definitions, and ask for a proof or a fixed answer.
  - For RL, keep lemmas with a checkable answer or those you can formalize (note 03).
  - Use substitution-resistant MCQ to measure recognition vs reasoning before converting to free-form.

### ResearchMath-14k — ResearchMath-14K: Scaling Research-Level Mathematics via Agents (Son et al., 2026) *(added)*
Link: https://arxiv.org/abs/2605.28003 · data: https://huggingface.co/datasets/amphora/ResearchMath-14k

- **Mechanism**
  - Two agents mine 1,233 documents:
    - 524 arXiv open-problem papers (8,182 problems).
    - 161 open-problem web pages from MathOverflow, Wikipedia and academia.edu (5,331).
    - 548 problem-session sheets, e.g., AIM-style workshops (7,322).
  - An Extractor agent (Codex with GPT-5.5) follows each source to its PDF/HTML and extracts every open problem as a verbatim quote.
  - A Refiner agent (Claude Code with Opus 4.7) re-reads the paper to inline every definition and searches up to ten later citing papers to check whether the problem has been resolved.
  - The result is 20,835 raw problems, curated to 14,056 across 11 domains.
  - 220K teacher trajectories come from GPT-OSS-120B and Qwen3-30B-A3B. A "committed-attempt" prompt cut non-attempts from 22% to 0%. Behavioral filtering removes non-attempts and fabricated citations.
- **How it makes tasks harder**: the problems sit at or beyond the research frontier. In pairwise LLM difficulty judgments on knowledge, novelty and procedural difficulty, the set ranks about 400 Elo above AceMath, AIME, HLE-Verified and NuminaMath.
- **Correctness / verification**
  - Answer correctness is not established: two LLM judges label only 3.7% and 4.3% of sampled training trajectories correct.
  - Self-containment is checked by four math PhD students: of 138 retained reviews, 115 (83.3%) gave the top rating on all five self-containedness dimensions.
- **Difficulty control**: by source type (open problems and problem sessions); no pass-rate filter.
- **Reported results**
  - Full-parameter SFT on ResearchMath gives +2.1 points over the starting checkpoints on graduate- and research-level benchmarks, across three model families. DASD gives 0.0 and Nemotron-SFT-Math-v4 gives −0.5.
  - Mixing DASD (50K) with ResearchMath (77K) beats token-matched DASD by +2.0 (olympiad), +0.8 (short-form), +2.6 (symbolic) and +5.7 (proof).
  - The dataset is released under CC BY-SA 4.0.
- **Limitations / failure modes**
  - These are SFT gains from mostly unverified trajectories, so the gains may come from reasoning diversity rather than correct solutions, as the authors suggest.
  - It is not RLVR-ready.
  - Share-alike license.
  - Extraction used closed-model agents. The extracted text is mostly verbatim source text, but record it anyway.
- **How to reuse with easy seed tasks**
  - Use open-problem and problem-session sources as a diversity supplement when competition-style SFT has saturated.
  - For RL, keep only the subset that has resolved, checkable answers; the Refiner's resolution search surfaces these.
  - Or use the problems as proposer seeds for hint-scaffolded or rubric-graded attempts (notes 10, 20).

### MathConstruct — MathConstruct: Challenging LLM Reasoning with Constructive Proofs (Balunović et al., 2025)
Link: https://arxiv.org/abs/2502.10197

- **Mechanism**
  - Each problem is a tuple (P, θ, V_θ): a symbolic statement P, concrete parameters θ, and a verifier V_θ that maps a proposed object to {0, 1} by checking every constraint.
  - Non-"Any" problems are adapted:
    - *Inf* problems require k constructions that demonstrate the infinite family.
    - *All* problems are demonstrated through multiple constructions.
    - *Max* problems require a construction that achieves the proven bound.
  - A variation generator re-samples θ (e.g., n over ranges like [6, 20]).
- **How it makes tasks harder**
  - Constructing an object replaces stating a number, so guessing does not work.
  - Scaling θ defeats small-case pattern matching.
  - Under "robust accuracy", a problem counts as solved only if all its variants are solved.
- **Correctness / verification**: executable verification; any valid object passes, so no reference answer is needed.
- **Difficulty control**
  - Parameter ranges set difficulty.
  - Pure brute-force agents reached under 10% and "Brute+Infer" agents (extrapolating from small instances, up to 3 refinements) 19.2%.
  - Problems that allow trivial brute force (e.g., n = 2 cases) were excluded in review.
  - Single-integer answers are avoided.
- **Reported results**: 121 problems, 456 instances. The best model, o3-mini, reaches 60.5% average accuracy and 42.2% robust accuracy.
- **Limitations / failure modes**: verifiers are hand-written by people with olympiad training. Incomplete constraints admit degenerate constructions. The pool is small.
- **How to reuse with easy seed tasks**
  - Turn "prove X exists" / "find the maximum" seeds into "output the object" tasks with a Python verifier.
  - Scale θ beyond sizes that can be enumerated.
  - Give reward only when a set of variants is all solved.

### SPARK (Spark-234K) — SPARK: Skeleton-Guided Reasoning Synthesis from Large-Scale Scientific Literature (Li et al., 2026)
Link: https://arxiv.org/abs/2608.30214

- **Mechanism**
  - Source: the Sci-Base corpus (about 3.36M papers, 10 disciplines), filtered to about 370K English seed papers (Jan 2024–Mar 2026) and parsed with MinerU2.5.
  - Each paper is distilled into a reasoning skeleton: "core conclusion and ordered evidence steps with concrete observables".
  - Questions are generated from the skeleton under four perspectives:
    - mechanistic reasoning ("explains why an observed phenomenon occurs");
    - hypothesis falsification ("distinguishes competing explanations");
    - quantitative derivation ("modeling and derivation instead of formula substitution");
    - boundary calibration ("assumptions and validity limits").
  - Then: consistency verification, semantic deduplication (Qwen3-Embedding-8B), and 13-gram decontamination (0 leak hits).
- **How it makes tasks harder**: it asks for mechanism, falsification and validity limits instead of recall. In automated labeling (DeepSeek-V4-Flash on 20K samples), L1 is 0.02%, L2 1.2%, L3 about 4.8%, L4 58.5% and L5 34.6%.
- **Correctness / verification**: independent re-generation plus a semantic-equivalence judge gives a 93.28% consistency rate. Two experts per discipline checked 300 stratified samples: 90.3% agreement on correctness, 97.7% on self-containment and 94.3% on difficulty labels.
- **Difficulty control**: the perspective templates; no explicit difficulty filter.
- **Reported results**
  - Qwen3-8B-Base: 60.83 average on 6 benchmarks with Spark-234K, vs 56.36 with MegaScience (1.2M).
  - Llama3.1-8B-Base: 45.37 vs 42.27. GPQA-Diamond is the one regression (37.37 vs 38.89).
- **Limitations / failure modes**: SFT only, with free-form answers. Correctness depends on the papers. About 10% disagreement with experts.
- **How to reuse with easy seed tasks**
  - Turn a recall question on a passage into a falsification or boundary question over its skeleton.
  - For RL, keep quantitative-derivation items with numeric answers and verify them with the consistency check plus a CAS.

### PaperBench + EXP-Bench — PaperBench: Evaluating AI's Ability to Replicate AI Research (Starace et al., 2025); EXP-Bench: Can AI Conduct AI Research Experiments? (Kon et al., 2025)
Links: https://arxiv.org/abs/2504.01848 · https://arxiv.org/abs/2505.24785 · code: https://github.com/Just-Curieous/Curie/tree/main/benchmark/exp_bench

- **Mechanism**
  - *PaperBench*: replicate 20 ICML 2024 Spotlight/Oral papers from scratch. Rubrics co-developed with each paper's authors decompose the work hierarchically into 8,316 individually gradable leaf tasks. The LLM judge is validated on a separate judge benchmark.
  - *EXP-Bench* is a semi-autonomous pipeline:
    1. Extract the research question, methodology and expected outcome from the paper.
    2. An implementation-extraction agent with PDF, terminal and web tools localizes the script chain in the real repository that realizes the method.
    3. The chain is executed in a clean container, and traces are checked against the paper's outputs; failures loop back for refinement.
    4. AST tracing of the validated scripts yields a natural-language list of implementation requirements, which becomes the implementation ground truth.
    5. A light human cross-check.
    6. Answer-revealing files (README, relevant scripts) are masked through scripted git operations, including submodules.
- **How it makes tasks harder**: tasks are end-to-end research with long horizons. In EXP-Bench, masking the implementation turns a runnable repo into a design-implement-execute-conclude task.
- **Correctness / verification**: author-validated rubrics plus a judge (PaperBench). Ground truth from the paper's own code and results, with execution validation (EXP-Bench).
- **Difficulty control**: rubric depth and partial credit. How much code is masked is a natural difficulty knob (our suggestion; the paper does not vary it systematically).
- **Reported results**
  - PaperBench: the best agent (Claude 3.5 Sonnet (New) with open-source scaffolding) averages a 21.0% replication score, and top ML PhDs still outperform models on a subset.
  - EXP-Bench: 461 tasks from 51 papers. Individual aspects reach 20–35%, and complete executable experiments succeed 0.5% of the time.
- **Limitations / failure modes**: heavy compute, judge noise, and costly author time for rubrics. Masking can leak through pip caches or git history unless it is scripted carefully.
- **How to reuse with easy seed tasks**
  - Build a masking ladder from a real repository: one masked function → a masked module → a full experiment. Verify against the paper's numbers or the repo's tests.
  - Use rubric leaves as dense partial rewards.

### ORQA — ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge (Krishnan et al., 2026)
Link: https://arxiv.org/abs/2609.12366 · site: http://orqabench.org

- **Mechanism**: O*NET occupations are linked to trusted occupation-specific websites (regulators, licensing bodies, professional organizations, government publications). An automated pipeline plus human review turns them into source-traceable QA. The authors position ORQA as complementing expensive expert authoring.
- **How it makes tasks harder**: long-tail professional knowledge. Some occupations (e.g., Sheet Metal Workers, Fish and Game Wardens) are "essentially zero".
- **Correctness / verification**: every answer traces to an authoritative source, and humans review the pairs.
- **Difficulty control**: none explicit; occupation choice drives difficulty.
- **Reported results**
  - 480 questions from 187 websites, covering 116 occupations in all 21 SOC major groups; 15 models tested.
  - Claude Opus 4.6, GPT-5.4 and Claude Sonnet 4.6 score about 58–62%; smaller open-weight models about 33–41%.
  - Healthcare occupations reach 78%, Office and Administrative Support about 40%.
  - Open-ended questions and wage weighting do not significantly change the model ranking.
- **Limitations / failure modes**: small, and knowledge-heavy rather than reasoning-heavy. Terms of use vary across source websites.
- **How to reuse with easy seed tasks**: use it as a cheap probe for long-tail domains where your policy fails, then complexify with the multi-hop and obfuscation operators in note 07. Check each source site's terms before training on mined text.

**C. Expert-authored tasks: the teacher ceiling and what it costs**

### GPQA — GPQA: A Graduate-Level Google-Proof Q&A Benchmark (Rein et al., 2023) *(added)*
Link: https://arxiv.org/abs/2311.12022

- **Mechanism**: domain experts write questions with explanations. Each question goes to two expert validators, with the writer revising after the first, and to three non-expert validators who have unrestricted web access.
- **How it makes tasks harder**: the pay is incentive-aligned for "Google-proof" difficulty:
  - writers get a $10 base per question;
  - $20 bonus for each of two expert validators who answer correctly;
  - $15 for each of three non-expert validators who answer incorrectly;
  - $30 extra if both experts are right and at least 2/3 of non-experts are wrong.
- **Correctness / verification**: expert validator agreement plus post-hoc explanations of mistakes. Expert validators get a $10 bonus for each correct answer.
- **Difficulty control**: non-expert failure with web access is the acceptance criterion. The subsets are Extended 546, Main 448 and Diamond 198.
- **Reported results**
  - Expert accuracy is 65% (74% discounting clear mistakes the experts later identified). Non-experts score 34% (30.4% on main) after an average of 37 minutes (median 30) with the web.
  - Average pay was estimated at about $95/hour, with a maximum of $150/hour.
- **Limitations / failure modes**: expensive, yielding only 448 main questions. The MCQ format invites guessing if the set is used for RL.
- **How to reuse with easy seed tasks**: copy the incentive design for expert-in-the-loop hardening. Pay reviewers when strong models (or your policy) fail and expert validators agree. That rewards hard, correct items instead of hard, ambiguous ones.

### SuperGPQA — SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines (M-A-P Team, Du et al., 2025)
Link: https://arxiv.org/abs/2502.14739

- **Mechanism**
  1. *Source screening*: experts collect credible resources.
  2. *Transcription*: crowd annotators (students at top Chinese universities):
     - translate the questions;
     - convert non-MCQ items to MCQ;
     - standardize "select the correct/incorrect statements" items;
     - add region-specific context;
     - generate "complementary confusion options";
     - estimate difficulty "based on both expert judgments and the accuracy of LLMs' responses".
  3. *Quality inspection* in three stages:
     1. Rule checks.
     2. LLM checks: GPT-4o-2024-08-06, Gemini-2.0-flash, Doubao-1.5-pro, Claude-3.5-Sonnet, DeepSeek-R1, QwQ and Qwen-2.5-72B answer and tag each item for validity, extreme or negative inquiry, multimodality, field relevance, completeness and discrimination.
     3. Experts re-annotate the suspicious items with unrestricted web access (30+ minutes each).

     Finally, "the easy questions are further tailored based on the accuracy of LLMs' responses".
- **How it makes tasks harder**: calculation problems are rewritten into MCQs with many options (9.67 on average), statement items are standardized, and items LLMs find easy are tailored further.
- **Correctness / verification**: expert source screening, LLM-flagged expert adjudication, and real-time plagiarism checks. A useful heuristic from the paper: "Questions where LLMs choose the same incorrect option are highly suspicious". This often means LLMs memorized a wrong answer key from exercise websites.
- **Difficulty control**: LLM accuracy decides which items are tailored.
- **Reported results**: 26,529 questions, 13 disciplines, 72 fields, 285 subfields, 80+ expert annotators. The best model, DeepSeek-R1, scores 61.82%. About 42% of items require calculation or formal reasoning.
- **Limitations / failure modes**
  - Crowd annotators "are not capable of collecting credible resources"; early crowd-sourced items were "too easy or unreliable".
  - Web exercise answers are unreliable and a leakage risk.
  - Even SOTA LLMs generate flawed confounders for statement-selection items.
  - The format is still MCQ.
- **How to reuse with easy seed tasks**
  - Copy the division of labor: experts pick sources, cheap annotators and LLMs transform, a panel of LLMs flags, and experts adjudicate only the flagged items.
  - Apply the same-wrong-option heuristic to audit the answer keys of any hard pool.
  - For RL, strip the options.

### GDPval — GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks (Patwardhan et al., 2025)
Link: https://arxiv.org/abs/2510.04374

- **Mechanism**
  - Sectors each contributing over 5% of US GDP are selected (9 sectors). Within each, the 5 occupations that contribute most to wages and are "predominantly digital" are chosen, where an occupation counts as digital if at least 60% of its tasks are digital.
  - Experts (14 years of experience on average, minimum 4) write tasks from their real work, with reference files.
  - Tasks pass automated model screening (OpenAI models) and an average of five human reviews (minimum three).
  - Grading is by "blinded expert pairwise comparisons" against the expert's own deliverable.
- **How it makes tasks harder**: long-horizon deliverables. Tasks average 7 hours of expert work and "span up to multiple weeks" at the high end.
- **Correctness / verification**: expert review and pairwise preference; there is no answer key. The primary metric is a win rate with "no upper limit".
- **Difficulty control**: occupational and economic-value sampling; the baseline deliverable can be replaced by stronger ones.
- **Reported results**
  - 1,320 tasks, with a 220-task gold subset. Gold tasks average 404 minutes and $361 of expert cost each.
  - Claude Opus 4.1 deliverables were rated better than or as good as the human deliverable 47.6% of the time.
  - The autograder agrees with experts 66% of the time, vs 71% inter-expert agreement.
  - Prompting models to check their outputs raised win rates by 5 pp.
- **Limitations / failure modes**: very expensive, with no verifiable reward. The pairwise-preference signal is noisy (71% inter-rater agreement).
- **How to reuse with easy seed tasks**
  - Use GDPval's per-task cost (about $361 of expert time plus reviews) as the benchmark for comparing synthetic pipelines.
  - For RL, have models generate variants of GDPval-style tasks and spend experts on rubric writing and adjudication only (see APEX).

### APEX — The AI Productivity Index (APEX) (Vidgen et al., 2025)
Link: https://arxiv.org/abs/2509.25721

- **Mechanism**
  - Experts recruited through Mercor (mean 7+ years of industry experience) pass a 30–45-minute interview and a paid 1–2-hour assessment of prompt- and rubric-writing.
  - They write prompts from their daily work, with attached sources.
  - "Sources with restrictive licenses or access controls, such as being behind a paywall, were explicitly prohibited."
  - Reviewers approve, reject or return each prompt. Authors then write rubrics, which are also reviewed. Gemini 2.5 Flash grades responses against the rubric.
- **How it makes tasks harder**: tasks are taken from real work, taking an estimated 2.7 hours for an expert on average (range 0.5–20 hours).
- **Correctness / verification**: multi-round expert review of prompt and rubric, then rubric grading by an LM judge.
- **Difficulty control**: by source (real tasks); results are analyzed by difficulty.
- **Reported results**: the expert pool grew from 76 to 137. There are 400 held-out cases (100 per job) and 100 open cases (25 per job, CC-BY). GPT-5 (Thinking = High) leads at 67.0%.
- **Limitations / failure modes**: per-task cost is not disclosed. It uses a single LM judge and is single-turn.
- **How to reuse with easy seed tasks**
  - Its source-license rule is a provenance template for expert-task vendors.
  - Each prompt comes with its own rubric, which makes a ready seed for rubric-reward RL.
  - Harden by adding documents, cross-document dependencies or conflicting sources, and have experts re-grade the rubric.

### CritPt + FrontierScience — Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark (Zhu et al., 2025); FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks (Wang et al., 2026)
Links: https://arxiv.org/abs/2509.26574 · https://arxiv.org/abs/2601.21165

- **Mechanism**
  - *CritPt*: 50+ active physics researchers wrote unpublished problems from their own research. There are 71 composite challenges that simulate entry-level research projects, decomposed into 190 checkpoints.
    - Every answer is "guess-resistant and machine-verifiable" (e.g., arrays of floats, complex symbolic expressions).
    - An automated grader is "heavily customized for advanced physics-specific output formats".
    - One example challenge is public; the solutions to the other 70 are private.
  - *FrontierScience* has two tracks:
    - The Olympiad track is written by international olympiad medalists and national-team coaches.
    - The Research track is written and verified by PhD scientists, with an expert 10-point rubric per problem; 7/10 counts as success.
    - "Preliminary questions were evaluated against various internal models, where if the model answered correctly, the question was considered invalid and required an update."
    - Research problems received at least two independent reviews plus an expert meta-review.
    - Judge: GPT-5 at high reasoning effort.
- **How it makes tasks harder**: research-scale composition in CritPt, and adversarial rejection against internal models in FrontierScience.
- **Correctness / verification**: author and expert review. CritPt uses a customized autograder. FrontierScience grades against rubric items with a model judge; the authors list rubric reliability as a limitation because it "relies on the model judge's capabilities".
- **Difficulty control**: the challenge-vs-checkpoint level (CritPt) and model-in-the-loop rejection (FrontierScience). The FrontierScience authors warn that selecting against OpenAI models makes the evaluation "biased against these models relative to others".
- **Reported results**
  - CritPt: the best base-model average is 5.7% (GPT-5 high), rising to about 10% with coding tools.
  - FrontierScience: 500+ Olympiad and 200+ Research questions were filtered to a 160-question open gold set (100 Olympiad, 60 Research).
    - GPT-5.2 scores 77% on Olympiad and 25% on Research. Gemini 3 Pro scores 76% on Olympiad, and GPT-5 ties GPT-5.2 on Research at 25%.
    - More test-time tokens lift GPT-5.2 from 67.5% to 77.1% (Olympiad) and from 18% to 25% (Research).
- **Limitations / failure modes**: tiny, with private solutions. Selecting against one model family biases which items survive. Rubrics require a judge.
- **How to reuse with easy seed tasks**
  - Decompose one expert problem into a checkpoint ladder: dense partial credit at checkpoints, with the composite challenge as the hard target.
  - When adversarially filtering, filter against your policy plus a teacher panel, not a single model family.

**D. Licensing and provenance of sources and generator outputs**

### Data Provenance Initiative audits — The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI (Longpre et al., 2023); Bridging the Data Provenance Gap Across Text, Speech and Video (Longpre et al., 2024); Consent in Crisis: The Rapid Decline of the AI Data Commons (Longpre et al., 2024)
Links: https://arxiv.org/abs/2310.16787 · https://arxiv.org/abs/2412.17847 · https://arxiv.org/abs/2407.14933

- **Mechanism**
  - Legal and ML experts traced 1,800+ text fine-tuning datasets (1,858 individual datasets in 44 collections) from source to collection, recording license conditions at each derivation step.
  - The multimodal follow-up covers nearly 4,000 text, speech and video datasets (1990–2024).
  - Consent in Crisis audits robots.txt and terms of service across 14,000 web domains.
- **How it makes tasks harder**: N/A (audit).
- **Correctness / verification**: a manual re-annotation protocol, released as the Data Provenance Explorer.
- **Difficulty control**: N/A.
- **Reported results**
  - Hosting sites omit licenses 70%+ of the time and categorize them wrongly 50%+ of the time.
  - "66% of analyzed Hugging Face licenses were in a different use category, often labeled as more permissive than the author's intended license."
  - "Unspecified" licenses fell from 69–72% on aggregators to 30% after the audit.
  - The most common licenses were CC-BY-SA 4.0 (15.7%), the OpenAI Terms of Use (12.3%) and CC-BY 4.0 (11.6%). The audit treats OpenAI-ToU data as non-commercial and cites "WizardCoder licensed for commercial use, while training on commercially-prohibited OpenAI data".
  - Multimodal audit: fewer than 33% of datasets are restrictively licensed, but over 80% of their source content carries non-commercial restrictions.
  - Consent in Crisis: 5%+ of C4 tokens (28%+ of its most critical sources) became fully restricted in 2023–24, and 45% of C4 is restricted by terms of service.
- **Limitations / failure modes**: legal interpretation (contract vs license) is contested. The audits predate the 2025–26 RL pools.
- **How to reuse with easy seed tasks**
  - Treat a pool's HF license tag as a hint.
  - Trace each seed to its original source and generator.
  - Our 2026-09-30 Hub check shows the pattern persists in RL pools (see TL;DR and Insights).

### Hidden Licensing Risks in the LLMware Ecosystem — Hidden Licensing Risks in the LLMware Ecosystem (Wang et al., 2026) *(added)*
Link: https://arxiv.org/abs/2602.10758

- **Mechanism**: supply chains are built by snowballing through Hugging Face metadata (base models, training datasets) and linked GitHub repos: 12,180 OSS repositories, 3,988 LLMs and 708 datasets. License conflicts along each chain are then analyzed. LiAgent, an LLM-agent framework for ecosystem-level license compatibility, is proposed.
- **How it makes tasks harder**: N/A (analysis).
- **Correctness / verification**: LiAgent reaches 87% F1, 14 pp above prior methods, which reach 58% and 76%. 60 incompatibilities were reported, 11 of them confirmed by developers.
- **Difficulty control**: N/A.
- **Reported results**
  - 35.4% of artifacts "lack any license declaration".
  - Common conflicts are "No License → Apache-2.0" (23.5% of OSS → LLM edges), MIT → Apache-2.0 (15.1%) and Apache-2.0 → CC-BY-4.0 (13.3%).
  - The paper reports that 52% of supply chains show at least one conflict.
  - Two conflicted LLMs have over 107M and 5M downloads.
  - Its recommendation: "downstream components should not adopt licenses more permissive than upstream dependencies".
- **Limitations / failure modes**: it relies on self-reported HF metadata, and synthetic-generation lineage (which model generated the data) is rarely declared, so this class of dependency is undercounted.
- **How to reuse with easy seed tasks**: before releasing a model or dataset trained on sourced or complexified tasks, build the same dependency graph for your seeds, generators and verifiers, and run a compatibility check. Never label derived data more permissively than its most restrictive parent.

### Generator license and terms map — Gemma Terms of Use; Llama 3.1 / Llama 4 Community Licenses; Qwen License / Qwen Research License; DeepSeek-R1 release; Kimi K2 / MiniMax-M2 modified MIT; NVIDIA Nemotron Open Model License; Anthropic Commercial Terms; Gemini API Terms; OpenAI Terms of Use (+ Who Taught You That? Tracing Teachers in Model Distillation, Wadhwa et al., 2025)
Links: https://ai.google.dev/gemma/terms · https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE · https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE · https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE · https://api-docs.deepseek.com/news/news250120 · https://www.anthropic.com/legal/commercial-terms · https://ai.google.dev/gemini-api/terms · https://arxiv.org/abs/2502.06659

- **Mechanism (terms read from primary files, Sep 2026)**
  - **Gemma Terms of Use** (last modified 2026-04-01):
    - "Model Derivatives" include any model created by transfer of patterns of Gemma's "weights, parameters, operations, or Output", including "distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma for training that model".
    - "Google claims no rights in Outputs you generate using Gemma."
    - Distributing a derivative requires passing on the use restrictions, a copy of the terms, and a Notice file.
    - The terms cover Gemma 1, 1.1, 2, 3 and 3n plus variants. "Gemma 4 operates under a separate Apache 2.0 license."
  - **Llama 3.1** (2024-07-23) and **Llama 4** (effective 2025-04-05): "If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the beginning of any such AI model name". Both also require "Built with Llama" and carry the 700M-MAU clause. The Llama 2 license instead forbade using outputs "to improve any other large language model (excluding Llama 2 or derivative works thereof)".
  - **Qwen2.5-72B-Instruct** (Qwen License): distributed models trained on "any outputs or results therefrom" must "prominently display 'Built with Qwen' or 'Improved using Qwen'", and a 100M-MAU commercial clause applies. **Qwen2.5-3B-Instruct** uses the Qwen RESEARCH LICENSE AGREEMENT, "FOR NON-COMMERCIAL PURPOSES ONLY". Qwen2.5-7B/32B, Qwen3-235B-A22B and Qwen3.5-397B-A17B are Apache-2.0 (HF tags).
  - **DeepSeek-R1** (MIT; release 2025-01-20): "Distill & commercialize freely!" and "API outputs can now be used for fine-tuning & distillation". DeepSeek-V3.2 and GLM-4.5/4.7/5 are MIT; gpt-oss-120b, Olmo-3 and gemma-4-31B-it are Apache-2.0 (HF tags).
  - **Kimi K2 / K2.5** (modified MIT): commercial products with >100M MAU or >$20M monthly revenue must "prominently display 'Kimi K2'" (or "Kimi K2.5") in the UI. **MiniMax-M2**: the same with >100M MAU or >$30M ARR, displaying "MiniMax M2".
  - **NVIDIA Nemotron Open Model License** (last modified 2025-12-15): "NVIDIA does not claim ownership to any outputs"; derivative works carry a "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Open Model License" notice.
  - **Anthropic Commercial Terms** (effective 2025-06-17), D.4: the customer may not "access the Services to build a competing product or service, including to train competing AI models". The customer "owns its Outputs".
  - **Gemini API Additional Terms** (effective 2026-03-23): "You may not use the Services to develop models that compete with the Services".
  - **OpenAI Terms of Use** restrict using Output to develop models that compete with OpenAI. The primary page returned HTTP 403 to our fetcher, so this clause is not re-verified here; DPI independently treats OpenAI-ToU data as non-commercial.
- **Correctness / verification**: *Who Taught You That?* shows teacher identity can be inferred from student outputs. "n-gram similarity alone is unreliable for identifying teachers, but part-of-speech (PoS) templates preferred by student models mimic those of their teachers." Distillation from restricted teachers is therefore detectable in principle (see note 19 on idiosyncrasies).
- **Difficulty control**: N/A.
- **Reported results**: obligations reach real pools. The Llama-Nemotron card's warning, and its undeclared Qwen-2.5-72B lineage, are covered in the Nemotron entry.
- **Limitations / failure modes**
  - Not legal advice.
  - Enforceability of output terms against third parties who never accepted them is disputed (DPI).
  - Terms change (Gemma moved to Apache-2.0 with version 4) and differ by size within one family (Qwen2.5-3B vs 7B vs 72B).
  - Share-alike (CC-BY-SA) data raises a separate question: whether a trained model or a regenerated dataset is an "adaptation".
- **How to reuse with easy seed tasks**
  - Default to MIT/Apache generators and verifiers for task and solution synthesis.
  - If you use outputs from Llama, Gemma ≤ 3, Qwen2.5-72B or Qwen2.5-3B, record them per row and plan for naming, attribution or non-commercial constraints.
  - Keep closed-API models for evaluation unless a contract allows otherwise.
  - Log generator, verifier and judge per row.

### EU AI Act public training-content summary — Explanatory Notice and Template for the Public Summary of Training Content for general-purpose AI models (European Commission AI Office, 2025)
Link: https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models (published 24 July 2025; PDF read)

- **Mechanism**
  - Section 2.5 "requires information about synthetic data created by or on behalf of the provider for training the model directly on the outputs of another AI model, in particular through model distillation or model alignment (e.g. AI feedback through reinforcement learning)".
  - It excludes "the use of AI models to clean or enrich data (e.g. AI-generated metadata…)".
  - Providers answer yes or no, give the modality, name the GPAI models used to generate synthetic data "if available on the market" with links to their summaries, and describe other models, including their own unreleased ones, "including a general description of the model training data if known and in so far as this may be needed… to avoid circumvention".
  - Synthetic datasets made by third parties on the provider's behalf are reported here instead of in 2.2.2.
  - Human-generated RL data goes to Section 2.6.
  - A modifier (e.g., a fine-tuner that becomes a provider) reports only the modification's training content and names the modified model.
- **How it makes tasks harder**: N/A.
- **Correctness / verification**: N/A.
- **Difficulty control**: N/A.
- **Reported results (dates)**
  - Obligations apply from 2 Aug 2025.
  - Models placed on the market before then must publish by 2 Aug 2027.
  - "The supervision and enforcement by the AI Office… will start as of 2 Aug 2026", so as of this note it is in force.
- **Limitations / failure modes**
  - Only a summary is required.
  - Whether an LLM answer-checker counts as "enrich/clean" (excluded) or as generating training data (included) is not settled case by case. An LLM-judge reward in RL fits "AI feedback through reinforcement learning" and is in scope.
  - Whether a particular fine-tune makes you a "provider" depends on the Commission's GPAI guidelines (not verified here).
- **How to reuse with easy seed tasks**: if you ship in the EU, your complexification pipeline must record which generator, rewriter, judge and reward models touched each dataset, including RL judges. Build it into row metadata now (the Nemotron-style `generator` field, plus a `judge` field).

### DFM Mimir v1 transplant datasets — DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data (Schneider-Kamp et al., 2026)
Link: https://arxiv.org/abs/2608.13517

- **Mechanism**
  - The model trains on a mixture of 161 datasets (about 70.5B tokens per epoch), all "permissible": openly licensed, made available by agreement, or allowed under the EU text-and-data-mining exception, and excluding personal information and copyright infringement.
  - The reference recipe drew on the Sapient HRM-Text mega-repository. The MCQ tasks it contained from Flan (NIV2, Dialog), Platypus and tasksource did not meet the standard.
  - They were replaced by 70 "Sapient-synth" transplant datasets (75M tokens), regenerated with Gemma 4 31B as open-ended generative tasks and "quality-audited before inclusion, with acceptance rates ranging from single digit percentages to high nineties for different categories".
- **How it makes tasks harder**: incidentally. The rewrite changes format from MCQ to generation, shifting the mix toward generative tasks.
- **Correctness / verification**: audit before inclusion. Version 2 added memorization audits.
- **Difficulty control**: not a goal.
- **Reported results**
  - Mimir 1B averages 69.0 on English benchmarks, vs 66.1 for HRM-Text 1B and 69.3 for Qwen 3.5 4B. It averages 56.8 on Danish (best among compared models) and 64.1 on math and code (+36.7% vs HRM-Text).
  - The claim that transplants "achieve comparable or superior performance" appears in the introduction and is not supported by a matched original-vs-transplant ablation.
- **Limitations / failure modes**: small model. Fidelity of each transplant is not measured. The seeds' upstream permissibility still matters.
- **How to reuse with easy seed tasks**
  - When a hard pool is non-commercial or unlicensed (KodCode, natural_reasoning, pools with no license tag), regenerate equivalent tasks from permissive sources with an MIT/Apache generator.
  - Use the rewrite to also convert MCQ to free-form.
  - Then re-verify and re-profile.
  - Keep the acceptance-rate audit per category.

**E. The survey backbone**

### Agentic Environment Engineering survey — Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application (Li et al., 2026)
Link: https://arxiv.org/abs/2606.12191

- **Mechanism**
  - Covers the environment lifecycle: modeling (8 attributes, 8 domains), synthesis, evaluation, application.
  - *Symbolic synthesis* comes in three kinds:
    - task-driven: "transforming existing static tasks… and data such as tool-calling… or mathematical data into interactive environments by wrapping them with programmatic rules", e.g., SWE-Gym;
    - real-world-driven;
    - de novo, e.g., AutoEnv, Agent World Model.
  - *Neural synthesis* is pixel-, word- or latent-level.
  - Agent evolution follows four pathways (memory-, orchestration-, trajectory-, exploration-centric). Environment evolution has three paradigms (neural-, difficulty-, scaling-driven).
  - Under difficulty-driven evolution, *explicit curriculum signals* control difficulty with "clearly defined signals such as accuracy, regret, reward or curiosity"; the text's examples are RLVE, SCALER, GenEnv and EnvGen. *Implicit mechanisms* are adaptive generation, environment construction or manual curricula.
- **Correctness / verification**: QC is organized as "correctness, diversity, complexity, and fidelity". The authors note that correctness is well studied while "diversity, complexity, and fidelity remains under-researched".
- **Difficulty control**: as above.
- **Limitations / failure modes**: descriptive, with no quantitative comparison. Its taxonomy figure lists RLVE, GenEnv and SCALER under *implicit* mechanisms while the §7.2 text treats them as *explicit*; follow the text.
- **How to reuse with easy seed tasks**: use the task-driven → real-world-driven → de novo axis and the four QC axes as a completeness checklist for the report's environment chapter. Seed-task wrapping, i.e., task-driven synthesis, is the cheapest start.

### Scaling LRMs beyond Human Supervision — Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence (Yang et al., 2026)
Link: https://arxiv.org/abs/2608.31075 · list: https://github.com/visitworld123/Awesome-Scaling-LRM-Beyond-Human-Supervision

- **Mechanism**: a two-axis survey, reward and experience, joined by a five-level ladder:

  | Level | Name | Human role |
  |---|---|---|
  | L0 | per-instance human supervision | a human answer, preference or judgment for each training instance |
  | L1 | human-grounded evaluation | criteria encoded in reusable reward models, rubrics or LM judges |
  | L2 | reward beyond human evaluation | reward from confidence, agreement, references or environment outcomes; tasks still external |
  | L3 | experience beyond human design | tasks, curricula and environments generated or selected during learning |
  | L4 | autonomous co-evolution | humans set intent, safety constraints and independent oversight |

- **How it makes tasks harder**: it catalogs L3 mechanisms and failure modes. *Difficulty miscalibration*: "If an unconstrained proposer drifts toward trivial or unsolvable tasks, group rewards become nearly constant and the GRPO advantages… vanish". Learnability signals, regret objectives and R-Zero's frontier-targeting challenger counter this but depend on noisy early solve-rate estimates. The other listed failure modes are co-adaptive reward hacking, distributional narrowing and mode collapse, non-stationarity, and environment fidelity and cost.
- **Correctness / verification**: reliable verification "may therefore require its own training objective and independent audits".
- **Difficulty control**: "a useful curriculum therefore requires longitudinal measurements of both learnability and coverage". "Archives of unsolved and out-of-distribution tasks… preserve difficult experience for later policies instead of discarding it."
- **Reported results (recommendation)**: the "recommended minimum scorecard" is:
  - frozen capability tests in the training domain plus at least one transfer setting;
  - an audit matched to the feedback source;
  - direct measurement of task or environment quality for methods that generate experience;
  - contamination analysis for heavily reused benchmarks;
  - compute-matched baselines under a shared protocol;
  - for L4-leaning methods, a longitudinal comparison of internal reward vs frozen external evaluation.
- **Limitations / failure modes**: secondary evidence.
- **How to reuse with easy seed tasks**: place each complexification pipeline on the ladder and adopt the scorecard as its acceptance test. Keep an archive of currently unsolvable complexified tasks rather than deleting 0-pass items.

### RL / agentic / self-improvement survey set — A Survey of Reinforcement Learning for Large Reasoning Models (Zhang et al., 2025); The Landscape of Agentic Reinforcement Learning for LLMs: A Survey (Zhang et al., 2025); A Survey of RL for LLMs under Data Scarcity (Yu et al., 2026); Self-Improvement of LLMs: A Technical Overview and Future Outlook (Yang et al., 2026); A Survey of Self-Evolving Agents (Gao et al., 2025)
Links: https://arxiv.org/abs/2509.08827 · https://arxiv.org/abs/2509.02547 · https://arxiv.org/abs/2604.17312 · https://arxiv.org/abs/2603.25681 · https://arxiv.org/abs/2507.21046

- **Mechanism**
  - *2509.08827*: training resources are a static corpus (§5.1) or dynamic environments (§5.2): rule-based, code-based, game-based, model-based and ensemble-based. Sampling strategy (§3.3.1, "Dynamic and Structured Sampling") covers difficulty-aware sampling. GitHub: TsinghuaC3I/Awesome-RL-for-LRMs.
  - *2509.02547*: contrasts "degenerate single-step" LLM-RL MDPs with the POMDPs of agentic RL. It gives a two-fold taxonomy (capabilities: planning, tool use, memory, reasoning, self-improvement, perception; and domains) and a compendium of open environments and frameworks drawn from 500+ works.
  - *2604.17312* has three layers:
    - data-centric: pruning (offline, online, fine-grained); synthesis (static, dynamic, and "hard data synthesis targeting identified weaknesses"); compression (token, step, trajectory, dataset);
    - training-centric: trajectory generation, reward engineering, policy optimization;
    - framework-centric: self-evolving, asymmetric co-evolution, multi-agent evolution.
  - *2603.25681*: a closed-loop lifecycle of data acquisition → data selection → model optimization → inference refinement, with an autonomous evaluation layer.
  - *2507.21046*: what, when and how to evolve (models, memory, tools, architecture; intra- vs inter-test-time; scalar rewards, textual feedback, single- and multi-agent).
- **Correctness / verification / difficulty control**: covered by these taxonomies (dynamic sampling, pruning, hard-data synthesis) but never compared quantitatively.
- **Limitations / failure modes**: heavy overlap; no operator-level comparison.
- **How to reuse with easy seed tasks**: cross-check the report's method list against the taxonomy leaves, especially "hard data synthesis", "asymmetric co-evolution", "data compression" and "ensemble-based environments". Mine the GitHub lists for missing works.

### QDC survey — Surveying the Effects of Quality, Diversity, and Complexity in Synthetic Data From Large Language Models (Havrilla et al., 2024)
Link: https://arxiv.org/abs/2412.02980

- **Mechanism**: evaluates synthetic-data algorithms by the quality, diversity and complexity of their output, and taxonomizes pipelines by the components that promote each property.
- **How it makes tasks harder**: surveys complexity measures. Basic measures: length, human levels such as MATH's, Tree-Instruct node counts. Attribute complexity. Model-based measures.
- **Correctness / verification**: quality measures (ground truth, neural reward models, attribute rubrics).
- **Difficulty control**: complexity as a first-class metric.
- **Reported results (claims)**
  - "Quality to be essential for in-distribution model generalization, diversity to be essential for out-of-distribution generalization, and complexity to be beneficial for both."
  - Methods aim at one of the three "though rarely all three together".
  - "Often complexity is not explicitly considered at all."
  - Quality-diversity trade-offs exist in both data and model outputs.
- **Limitations / failure modes**: written before RLVR, and the evidence is aggregated.
- **How to reuse with easy seed tasks**: track all three metrics per complexified batch, and pair every hardening operator with a diversity check (note 16's OpenThoughts-Agent result).

**F. Classic automatic-curriculum ancestors of proposer-solver RLVR**

### Asymmetric self-play + Goal GAN + AMIGo — Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play (Sukhbaatar et al., 2017); Automatic Goal Generation for Reinforcement Learning Agents (Florensa et al., 2017); Learning with AMIGo: Adversarially Motivated Intrinsic Goals (Campero et al., 2020)
Links: https://arxiv.org/abs/1703.05407 (ICLR 2018) · https://arxiv.org/abs/1705.06366 · https://arxiv.org/abs/2006.12122 (ICLR 2021)

- **Mechanism**
  - *Asymmetric self-play*: Alice proposes a task by acting, and Bob must undo it (reversible environments) or repeat it (resettable ones). The rewards are R_B = −γ t_B and R_A = γ max(0, t_B − t_A), and if Bob fails, t_B = t_Max − t_A. "Alice's optimal behavior is to find simplest tasks that Bob cannot complete… only just beyond his current capabilities."
  - *Goal GAN*: goals are sampled from GOID_i = {g : R_min ≤ R^g(π_i) ≤ R_max}. A GAN generator is trained with labels "in band / not in band".
  - *AMIGo*: the teacher gets r_T = +α if the student reaches the goal in t⁺ ≥ t* steps, and −β if it is faster or fails. t* is raised by 1 whenever the student succeeds in more than t* steps ten times in a row, yielding "increasingly challenging—yet achievable—goals".
- **How it makes tasks harder**: the proposer is paid for solver effort at the frontier.
- **Correctness / verification**: feasibility by construction (Alice's own trajectory certifies the task), and the simulator checks goals.
- **Difficulty control**: Goal GAN uses R_min = 0.1 and R_max = 0.9, "although the algorithm is very robust to these hyperparameters (any value of R_min ∈ (0, 0.25) and R_max ∈ (0.75, 1) would yield basically the same result)". AMIGo uses an adaptive step threshold.
- **Reported results**: Goal GAN Ant-maze coverage went from 0.014 (iteration 1) to 0.53 (10), 0.78 (30) and 0.98 (100).
- **Limitations / failure modes**: simulators only; proposers can drift to exploitable regions; no natural language.
- **How to reuse with easy seed tasks**: use a [0.1, 0.9] pass-rate band, whose robustness to exact cutoffs is documented. Require a feasibility certificate: the proposer must solve, or construct the solution for, its own task. And escalate the difficulty threshold adaptively after consecutive solver successes.

### Setter-solver — Automated curricula through setter-solver interactions (Racanière et al., 2019)
Link: https://arxiv.org/abs/1909.12892 (ICLR 2020)

- **Mechanism**: a setter S(z, f) generates goals conditioned on a desired feasibility f ∈ (0, 1), and a judge J predicts solver success (binary cross-entropy on outcomes). The setter has three losses:
  - *Validity*: "increases the likelihood of the setter generating goals which the solver has achieved", i.e., hindsight from the setter side. A goal is *valid* "if there exists a solver agent policy which has a non-zero probability of achieving" it.
  - *Feasibility*: sample f uniformly and push the judge's rating of the generated goal to f.
  - *Coverage*: goal entropy.

  An optional desirability discriminator aims the curriculum at a known target distribution. Setter and judge can condition on an environment observation.
- **How it makes tasks harder**: sampling f uniformly spreads difficulty over (0, 1) instead of one band.
- **Correctness / verification**: validity is grounded in goals actually achieved; the environment checks attainment.
- **Difficulty control**: continuous feasibility. Compared with Goal GAN's binary partition, the authors note it "allows uniformly sampling feasibility, can be estimated from one run per goal".
- **Reported results**
  - "Complex environments require all three losses". In alchemy, validity and coverage are necessary, and feasibility is not necessary but improves consistency.
  - Observation-conditioned setters beat unconditioned ones in environments that vary, which the authors call the first such result for goal-conditioned RL.
- **Limitations / failure modes**: neural setters in simulators; depends on judge calibration.
- **How to reuse with easy seed tasks**: use validity, feasibility and coverage as the three acceptance tests of any LLM task generator. Validity: a reference solution or verifier exists. Feasibility: predicted pass rates spread across (0, 1). Coverage: embedding entropy. A learned pass-rate predictor (the "judge") saves rollouts (see note 20).

### TSCL + Graves bandit + ACL typology — Teacher-Student Curriculum Learning (Matiisen et al., 2017); Automated Curriculum Learning for Neural Networks (Graves et al., 2017); Automatic Curriculum Learning For Deep RL: A Short Survey (Portelas et al., 2020)
Links: https://arxiv.org/abs/1707.00183 · https://arxiv.org/abs/1704.03003 · https://arxiv.org/abs/2003.04664

- **Mechanism**
  - *TSCL*: the Teacher samples subtasks where the learning-curve slope is highest. To counter forgetting it also picks tasks whose performance is falling, sampling "according to the absolute value of the slope". The slope is estimated by linear regression over recent scores.
  - *Graves et al.*: learning-progress signals (prediction gain, complexity gain) are rewards for a nonstationary multi-armed bandit that sets a stochastic syllabus.
  - *Portelas et al.* ask three questions:
    - *Why* use ACL?
    - *What* does ACL control? Initial states, reward functions, goals, environments, opponents, transition selection and modification.
    - *What* does ACL optimize? Intermediate difficulty, learning progress (LP), diversity, surprise, energy, adversarial reward maximization (ARM).
- **How it makes tasks harder**: it schedules existing task families by learning progress, not by success level.
- **Correctness / verification**: fixed task sets with known evaluators.
- **Difficulty control**: learning progress, the derivative of performance.
- **Reported results**: TSCL's automatic curriculum "enabled to solve a Minecraft maze that could not be solved at all when training directly on solving the maze", and learned "an order of magnitude faster than uniform sampling of subtasks". Graves: "in some cases halving the time required to attain a satisfactory performance level".
- **Limitations / failure modes**: needs a predefined task set; LP estimates are noisy.
- **How to reuse with easy seed tasks**: once your generator emits families (operator × depth), schedule them with an |Δ pass-rate| bandit instead of a static band. The ACL typology is a checklist of what else a curriculum could control.

### INT + Polu statement curriculum — INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving (Wu et al., 2020); Formal Mathematics Statement Curriculum Learning (Polu et al., 2022)
Links: https://arxiv.org/abs/2007.02924 · https://arxiv.org/abs/2202.01344

- **Mechanism**
  - *INT*: a trivial "core logic statement" (an initial condition) is morphed by applying axioms in a chosen *axiom order*. Production rules for each axiom (transformation and extension) find the arguments and premises needed for longer proofs, yielding a theorem together with its proof. The problem distribution can be varied along 6 dimensions: initial conditions, axiom combination, axiom order, number of axioms, proof length, and so on.
  - *Polu et al.*: a Lean inequality generator composes AM-GM, the trivial inequality, Cauchy-Schwarz, Bernoulli, Young and Hölder. N_D sets composition depth and N_S sets input-expression complexity.
    - 5,600 proof-less statements were generated (100 for each 0 ≤ N_S ≤ 7 × 0 ≤ N_D ≤ 6).
    - The seed was 100 formal proofs at N_D = 1, N_S = 5.
    - Expert iteration alternates proof search and training on found proofs.
- **How it makes tasks harder**: explicit depth and size knobs.
- **Correctness / verification**: the Lean kernel checks proofs; INT theorems are true by construction.
- **Difficulty control**: N_D "we found to be the main driver for difficulty".
- **Reported results**
  - Expert iteration closed 6 problems at N_D = 6 with no seed proof at that level. N_D = 6 "remains completely out of reach of simply scaling the number of attempts" (the sample-only loop is stuck at 0).
  - miniF2F-test pass@1/8/64 of 29.6 / 34.5 / 36.6% (θ9 full) vs PACT 24.6 / 29.2%.
- **Limitations / failure modes**: narrow domain; generated statements are out of distribution relative to competitions.
- **How to reuse with easy seed tasks**: the template for knob-controlled complexification. A forward generator is correct by construction, with separate depth and size knobs. Supply a grid of graded statements without answers when a checker exists, and let expert iteration or RL climb it.

### MetaGen — Learning to Prove Theorems by Learning to Generate Theorems (Wang & Deng, 2020)
Link: https://arxiv.org/abs/2002.07019

- **Mechanism**
  - A neural generator applies inference rules to existing theorems and hypotheses to synthesize new theorems with proofs in Metamath (set.mm).
  - Variants: MetaGen-Rand (random), MetaGen-IL (imitation of human-written theorems) and MetaGen-RL (reward from a relevance/adversarial signal).
  - The synthetic proofs augment training of a Holophrasm-style prover.
- **How it makes tasks harder**: forward composition produces theorems whose proofs are longer or newer than the human ones.
- **Correctness / verification**: forward rule application is valid by construction.
- **Difficulty control**: implicit, through the learned generator distribution.
- **Reported results**
  - With 10% of the human proofs, adding 1M MetaGen-IL synthetic proofs proves 472 test theorems, vs 476 when the human proofs are doubled to 20%.
  - With 100% of human proofs, 10M MetaGen-IL proofs raise the count from 557 to 600.
  - Learned generators beat MetaGen-Rand.
- **Limitations / failure modes**: synthetic theorems can be unnatural; pre-LLM.
- **How to reuse with easy seed tasks**: when a symbolic engine exists (CAS, SAT, SQL, Lean), generate tasks forward from known facts, and train or prompt the generator to imitate the human task distribution so the tasks look realistic.

### STaR — STaR: Bootstrapping Reasoning With Reasoning (Zelikman et al., 2022)
Link: https://arxiv.org/abs/2203.14465

- **Mechanism**
  - Generate rationales and keep those that reach the right answer.
  - For failed problems, "rationalization": regenerate the rationale with the correct answer as a hint.
  - Fine-tune on all correct rationales without the hint, and iterate.
- **How it makes tasks harder**: it extends the training signal to problems just beyond current ability.
- **Correctness / verification**: the final answer must match the ground truth; the rationale itself is not verified.
- **Difficulty control**: an implicit curriculum; on n-digit addition, accuracy extends to more digits over iterations.
- **Reported results**
  - CommonsenseQA 72.5%, vs 73.0% for a fine-tuned model 30× larger.
  - Arithmetic reaches 89.5% overall after 16 iterations.
  - Without rationalization the loop "eventually fails to solve any new problems in the training set".
- **Limitations / failure modes**: right answers with wrong rationales get reinforced; needs ground-truth answers.
- **How to reuse with easy seed tasks**: apply rationalization to the zero-pass tail that complexification creates. It is the ancestor of the hint-scaffolding methods in notes 10 and 20.

## Complexification operators from this area

1. **Research-statement mining with self-containment rewriting**
   - *What it does*: extract a statement with a unique, checkable result (theorem, lemma, open problem, forum answer) from post-cutoff research text. Inline all definitions and assumptions from the source. Drop items that are leaked, guessable or not self-contained.
   - *Easy → hard*: "count the 3-cliques in this 7-vertex graph" → "for q = p^{2t}, p ≡ 3 (mod 4), how many 3-cliques does the Peisert graph P*(q) have?" (answer q(q−1)(q−5)/48; RealMath).
   - *Keep it verifiable*:
     - Keep only constructive, fixed-answer statements, or those you can formalize.
     - Run an LLM triviality and leakage filter and an LLM self-containment judge; LemmaBench measured 75.5–96.5% human precision for that judge.
     - Spot-check by hand (RealMath found about 6% bad).
     - Prefer results that are cross-derived or confirmed by later citing papers (the ResearchMath Refiner's resolution search).
   - *Sources*: RealMath, LemmaBench, ResearchMath-14k, SPARK.
2. **Proof → construction reframing**
   - *What it does*: turn "prove existence / infinitely many / find the maximum" into "output the object(s)", graded by an executable V_θ. Inf-problems ask for k constructions.
   - *Easy → hard*: "Show there is a matrix with property P" (judge-graded proof) → "Output a 10×10 real matrix of rank ≤ 3 with zeros on the diagonal and positive entries elsewhere" (Python-checked).
   - *Keep it verifiable*: V_θ must check every constraint. Run brute-force and extrapolating agents, and exclude instances they crack.
   - *Sources*: MathConstruct.
3. **Parameter re-instantiation with robust accuracy**
   - *What it does*: parameterize a statement, sample several θ, and give reward only if all variants are solved.
   - *Easy → hard*: one instance with n = 6 → five sampled n in a range like [6, 20], all of which must pass.
   - *Keep it verifiable*: the verifier is parameterized. Check that a construction exists for each θ, e.g., a reference solver succeeds.
   - *Sources*: MathConstruct (robust accuracy 42.2% vs 60.5% average for o3-mini).
4. **Distractor widening and proof-strategy distractors (MCQ hardening)**
   - *What it does*: widen options to about 10. Standardize statement-selection items. Build distractors from misleading proof strategies. Add a substitution-resistant mode.
   - *Easy → hard*: a 4-option concept MCQ → a 9–10-option item whose distractors follow plausible but invalid proof directions.
   - *Keep it verifiable*: annotators check each distractor, and LLM panels flag items where models converge on the same wrong option.
   - *Caution*: MCQ remains recognition-hackable. Gemini-3.1-pro-preview drops from 43.5% to 17.6% under substitution resistance. For RL, prefer operator 5.
   - *Sources*: SuperGPQA, LiveMathematicianBench.
5. **MCQ → free-form (and MCQ → generative task) conversion**
   - *What it does*: remove the options so the model must produce the answer, and grade against the expert reference.
   - *Easy → hard*: "Which of A–D is the applicable doctrine?" → the same question with no options.
   - *Keep it verifiable*: keep expert answers short and canonical, and use a hardened generative verifier; note 11 records 67.0% false positives for a Qwen2.5-72B judge under a "Thought process:" key on this very set. Also applies to regenerating non-permissive MCQ corpora as generative tasks (Mimir).
   - *Sources*: Crossing the Reward Bridge; DFM Mimir.
6. **Composite challenge with checkpoint or rubric decomposition**
   - *What it does*: author or mine one research-scale problem and split it into graded checkpoints or a rubric tree.
   - *Easy → hard*: one checkpoint ("compute the ground-state energy of H") → the full chained challenge (CritPt: 5.7% for GPT-5 high), or full paper replication through 8,316 rubric leaves (PaperBench).
   - *Keep it verifiable*:
     - Give each checkpoint a guess-resistant, machine-verifiable answer (float arrays, symbolic expressions), or make each rubric item a pass/fail condition.
     - Validate the judge against humans; GDPval's autograder agrees 66% of the time vs 71% between humans.
     - Use a success threshold (7/10 in FrontierScience).
   - *Sources*: CritPt, PaperBench, FrontierScience, APEX.
7. **Masked-implementation replication**
   - *What it does*: in a real repository, mask the files that implement a paper's experiment (README, scripts, submodules). Ask the agent to design, implement, run and conclude.
   - *Easy → hard*: re-implement one masked function against existing tests → re-implement and run a full masked experiment and match the paper's numbers (EXP-Bench: 0.5% complete success).
   - *Keep it verifiable*: validate the unmasked script chain by executing it against the paper's outputs first. Mask with scripted git operations, and check for leaks through history, caches and submodules.
   - *Sources*: EXP-Bench, PaperBench.
8. **Model-in-the-loop adversarial acceptance with incentive-aligned review**
   - *What it does*: accept a candidate only if a panel of reference models fails it, then pay experts according to validation outcomes.
   - *Easy → hard*: a PhD-written question GPT-4o already answers → HLE's funnel (70,000+ attempts → about 13,000 model-stumping candidates → 2,500 final), or FrontierScience's "if the model answered correctly, the question was considered invalid".
   - *Keep it verifiable*:
     - Adversarial selection enriches for wrong or ambiguous keys, so require expert validation of every survivor.
     - GPQA-style bonuses reward expert agreement plus non-expert failure.
     - Filter against your own policy plus a diverse panel, not one model family.
     - Audit afterwards (HLE-Verified, note 14).
   - *Sources*: GPQA, HLE, FrontierScience, SuperGPQA, GDPval.
9. **Temporal freshness mining**
   - *What it does*: continuously harvest tasks created after the policy's cutoff. They are hard because they are new, not because they were built to be.
   - *Easy → hard*: SWE-bench-2023 issues that are likely memorized → this month's PRs with installable environments, or this week's arXiv lemmas.
   - *Keep it verifiable*: execution (fail-to-pass / pass-to-pass), or the paper's stated result. Date-stamp every task and flag evaluations that may be contaminated.
   - *Sources*: SWE-rebench (+V2), RealMath, LemmaBench, LiveMathematicianBench.
10. **Skeleton-guided multi-perspective questioning**
    - *What it does*: distill a source into a claim–evidence–derivation skeleton, then ask mechanistic, falsification, quantitative-derivation or boundary-calibration questions instead of recall.
    - *Easy → hard*: "What value did the paper report?" → "Which explanatory model does the evidence support, and under what regime does the conclusion stop holding?"
    - *Keep it verifiable*: skeleton-consistency re-generation (93.28%) and expert spot checks (90.3%). For RL, keep numeric derivation items.
    - *Sources*: SPARK.
11. **Deliverable-grounded professional task**
    - *What it does*: an expert-authored, multi-hour task with real attached files, graded by rubric or by pairwise comparison with an expert deliverable.
    - *Easy → hard*: "Summarize this memo" → a GDPval-style task (7 hours on average) producing a client-ready deliverable from spreadsheets.
    - *Keep it verifiable*: multiple expert reviews, independently gradable rubric items, a judge validated against experts, and permissively licensed sources only (the APEX rule).
    - *Sources*: GDPval, APEX.
12. **Generator knobs for composition depth and expression size**
    - *What it does*: forward-compose known lemmas or axioms; depth and size knobs set difficulty, and the output is correct by construction.
    - *Easy → hard*: a single AM-GM instance (N_D = 0) → six composed inequalities over complex expressions (N_D = 6, N_S = 7), out of reach of pure sampling.
    - *Keep it verifiable*: a formal checker (Lean) or derivation-by-construction. Proofs are optional when a checker exists.
    - *Sources*: INT, Polu et al., MetaGen.
13. **Frontier-band proposer reward with a feasibility certificate**
    - *What it does*: reward the task proposer only for intermediate solver success, for extra solver effort over the proposer's own, or for high learning progress.
    - *Easy → hard*: random goals → Goal GAN goals with success in [0.1, 0.9]; Alice paid γ·max(0, t_B − t_A); AMIGo's threshold raised after 10 consecutive effortful successes.
    - *Keep it verifiable*: the proposer's own solution, or a validity loss on achieved goals, keeps "hard" from meaning "impossible". Add a coverage/entropy term.
    - *Sources*: Asymmetric self-play, Goal GAN, AMIGo, Setter-solver, TSCL.
14. **Rationalization (hint, then remove the hint)**
    - *What it does*: on failed items, condition on the answer or a hint, keep successful rationales, and train as if unhinted.
    - *Easy → hard*: 2-digit addition (solved) → 5-digit addition that is solved only with the answer shown; accuracy then extends over iterations.
    - *Keep it verifiable*: keep only rationales reaching the ground-truth answer, and audit a sample for post-hoc justification.
    - *Sources*: STaR.
15. **Pass-rate-ordered blend with stage re-weighting**
    - *What it does*: mix environments with explicit ratios, order samples from high to low pass rate, and shift the mix between stages.
    - *Easy → hard*: a uniformly shuffled math pool → Nemotron 3 Super's RLVR1-3 (tool use and GenRM rising to 29.82% each) → SWE1-2 (R2E-Gym about 80%) → RLHF.
    - *Keep it verifiable*: every row names its verifier. Re-profile with your checkpoint, since the published order reflects NVIDIA's.
    - *Sources*: Nemotron RL blends.
16. **Multi-reference difficulty stratification**
    - *What it does*: label each task with pass rates from 2–4 reference models of different strength (or "older, weaker models") to get a graded difficulty label.
    - *Easy → hard*: "solved by the teacher" → "failed by the 4B, 8B and 32B annotators but solved by the teacher", or RealMath's "hard" stratum (o3 27.9%).
    - *Keep it verifiable*: labels need verifiable rewards. Re-profile on your policy before use.
    - *Sources*: SYNTHETIC-2, RealMath.
17. **License transplant (provenance-preserving regeneration)**
    - *What it does*: regenerate a non-permissive or unlicensed seed dataset with a permissively licensed generator, keeping the task distribution (optionally changing the format), then re-verify.
    - *Easy → hard*: a CC-BY-NC or unlicensed pool → the same skills regenerated from permissive sources with an Apache/MIT generator, with per-category audited acceptance.
    - *Keep it verifiable*:
      - Re-run verifiers.
      - Re-derive answers if numbers changed.
      - Log the generator for EU §2.5.
      - Never label the output more permissively than any remaining parent (LLMware).
    - *Sources*: DFM Mimir, Hidden Licensing Risks, EU template.

## Insights & pitfalls

- **Inherited difficulty labels expire.** Examples:
  - Skywork-OR1: per-size R1-Distill difficulty (note 12).
  - POLARIS: R1-Distill-Qwen-7B pass rates.
  - SYNTHETIC-2: Qwen3-4B/32B and R1-0528-Qwen3-8B pass rates.
  - Nemotron blends: NVIDIA checkpoint ordering, with no per-row value shipped.
  - RealMath: difficulty from "older, weaker models".

  Re-profile everything with your policy. RealMath's split (97.5% easy vs 27.9% hard for o3) shows a mined pool is a mixture, not a difficulty level.
- **Card licenses are hints; lineage is what matters.** Hub API on 2026-09-30:
  - *No license tag*: Skywork-OR1-RL-Data, SYNTHETIC-2-RL, SYNTHETIC-2-SFT-verified, OpenThoughts-Agent-v1-RL, II-Thought-RL-v0, INTELLECT-3-RL, Dolci-Think-RL-7B, Nemotron-RL-knowledge-mcqa.
  - *CC-BY-NC-4.0*: KodCode-V1, KodCode-Light-RL-10K, facebook/natural_reasoning.
  - *CC-BY-SA-4.0*: Nemotron-RL-coding-competitive_coding.
  - *MIT*: DeepMath-103K, SWE-smith, SWE-Gym, SynLogic, DeepScaleR-Preview.
  - *Apache-2.0*: POLARIS-53K, DAPO-Math-17k, Big-Math-RL-Verified (gated), OpenR1-Math-220k, NuminaMath-1.5, R2E-Gym-Subset, OpenThoughts3-1.2M, Multi-subject-RLVR, SYNTHETIC-2 (full), OpenThoughts-Agent-v1-SFT.
  - *CC-BY-4.0*: OpenMathReasoning, rStar-Coder, Code-Contests-Plus, OpenCodeReasoning, SWE-rebench, SWE-rebench-V2, and the Nemotron SFT/RL cards.

  DPI's findings still apply (70%+ omitted, 50%+ wrong, 66% in a different use category on HF), as does the LLMware study (35.4% undeclared, about half of supply chains with at least one conflict).
- **Cards can omit generator obligations, so check the generator table yourself.** Llama-Nemotron-Post-Training warns about Llama 3.1/3.3 but also contains 464,658 Qwen-2.5-72B-Instruct rows, which carry "Built with Qwen / Improved using Qwen" terms. Its larger Qwen-generated shares carry no such term: 19.8M Qwen-2.5-Math-7B and 8.9M Qwen-2.5-Coder-32B rows, both Apache-2.0 on the Hub. Obligations differ by size within one family: Qwen2.5-3B is research-only, while 7B and 32B are Apache-2.0.
- **What "provenance" includes has widened.** EU §2.5 explicitly covers "AI feedback through reinforcement learning", so LLM-judge rewards and GenRMs used in RL belong in lineage, not just generators. It excludes pure cleaning or enrichment, and an answer-checker such as GPT-5.2 verifying Nemotron-RL-Math-v2 falls in a gray zone. Log generator, rewriter, verifier and judge per row. AI Office enforcement began 2 Aug 2026.
- **Distillation footprints are detectable.** PoS templates identify teachers where n-grams fail (*Who Taught You That?*), so terms-of-service violations from distilling closed models can be audited in principle. Whether heavy rewriting (transplants, complexification) erases the footprint is untested.
- **Research mining yields little but refreshes itself.**
  - RealMath: 14,747 → 280, about 94% valid.
  - LemmaBench: 52.8–64% of lemmas self-contained.
  - ResearchMath: 20,835 → 14,056.
  - SWE-rebench: about 153K → 21,336, with install success in 31% of repositories.

  Mined sets also saturate fast: LemmaBench pass@1 went from 12.3% to 40.8% in nine months. Mining is a pipeline to run continuously, not a dataset to download.
- **At the research frontier, verification, not generation, is the bottleneck.** ResearchMath's judges accept only 3.7–4.3% of teacher trajectories, and LemmaBench's human confidence in LLM proof verdicts is 67–83%. Yet ResearchMath SFT still beats more competition-style math (+2.1 vs −0.5 for Nemotron-SFT-Math-v4). For RL, keep only the resolved, checkable slice or formalize. For SFT, behavioral filtering (non-attempts, fabricated citations) plus diversity can be enough.
- **Expert authoring: what is known about cost and what it buys.**
  - GPQA: about $95/hour, 448 main-set questions.
  - HLE: $500K prize pool ($5,000 × top 50, $500 × next 500) and a funnel of 70,000+ attempts → about 13,000 candidates → 2,500 questions.
  - GDPval: 404 minutes and $361 per gold task plus about 5 reviews.
  - APEX: 137 experts with 7+ years of experience, and tasks that take an expert 2.7 hours on average.
  - SuperGPQA: crowd sourcing "wasted" early funding on items judged "too easy or unreliable".

  No source reports dollars per unit of downstream RL gain.
- **Expert tasks are the unsaturated tier; synthetic ones mostly are not.** CritPt 5.7%, FrontierScience-Research 25% (vs 77% Olympiad), EXP-Bench 0.5%, PaperBench 21.0%, MathConstruct robust 42.2%, ORQA near zero on some occupations. The scarce resource is long-horizon, open-ended research structure, not harder closed-form questions.
- **Adversarial selection against one family biases the set and enriches for bad keys.** FrontierScience states the bias explicitly. SuperGPQA's heuristic that "questions where LLMs choose the same incorrect option are highly suspicious" is a cheap audit for any hard pool. Budget expert review for survivors.
- **Decompose to escape zero advantage.** Checkpoints (CritPt 190 across 71 challenges), rubric leaves (PaperBench 8,316) and 10-point rubrics with a 7/10 success threshold (FrontierScience) convert near-zero pass rates into partial credit. Validate the judge first: GDPval's autograder agrees 66% of the time vs 71% between humans.
- **MCQ-based "hardness" can be illusory.** LiveMathematicianBench's substitution-resistant mode pushes the best model below random (43.5% → 17.6%). Before RL, convert to free-form or constructions (operators 2 and 5).
- **The classic ACL literature already stated today's acceptance criteria.**
  - Validity, feasibility and coverage (setter-solver).
  - A pass band robust to its cutoffs (Goal GAN).
  - Proposer solutions as feasibility certificates (asymmetric self-play).
  - Adaptive thresholds (AMIGo).
  - Learning progress rather than level (TSCL, Graves).
  - Depth as the main difficulty driver, with expert iteration over proof-less graded statements reaching levels sampling never does (Polu).
  - Rationalization to rescue zero-pass items (STaR).
- **Diversity vs hardening, and source selection.** OpenThoughts-Agent (note 16) ablated 95 task-generation strategies. TB2 ranged from 10.9% to 0.4% across single strategies, top-4 to top-8 mixes were best, and top-16 hurt at 100K. Hardening descriptions at fixed size did not help, while diversifying beyond the unique pool did. The QDC survey's "often complexity is not explicitly considered at all" is the mirror image. Measure quality, diversity and complexity per batch.
- **Blend recipes are free structure.** When your pool saturates, copying NVIDIA's staging (verifiable → agentic/SWE → preference, with MOPD in Ultra) and re-profiling may beat adding new data. Ultra's "reasoning" stage is simply Multi-subject-RLVR, a mined expert-answer pool.
- **Permissive regeneration is an exit, but measure it.** Mimir's transplant claim lacks a matched ablation. If you transplant, run original-vs-transplant evaluations at equal tokens, and keep per-category acceptance rates.

## Open problems & research opportunities

- **A license-aware, policy-relative registry of hard-task pools.** Per row: source license, generator chain, verifier type, judge, decontamination status, and pass rates under several 2026 reference models. Add saturation curves for POLARIS, Skywork, DAPO, DeepMath, SYNTHETIC-2, ResearchMath and the Nemotron blends. Nobody publishes this. A LiAgent-style compatibility check on the lineage graph would make it actionable.
- **Cost-effectiveness of hybrid authoring.** Measured points exist (GPQA $95/hour; GDPval $361 per task; the HLE funnel; RealMath's 6% error; ResearchMath's 3.7–4.3% correct trajectories), but none is tied to downstream RL gain per dollar. A study comparing expert-authored, expert-adjudicated and fully mined tasks at equal budget is missing.
- **Research-mined tasks as RL environments.** RealMath, LemmaBench, ResearchMath, SPARK, EXP-Bench and SWE-rebench could supply post-cutoff RL tasks monthly. Open pieces:
  - verifying unrefereed results (CAS cross-derivation, formalization, citing-paper resolution search);
  - raising self-containment beyond about 50–64%;
  - keeping tasks hard rather than merely new, given LemmaBench's 12% → 41% in nine months.
- **Learning from unverifiable frontier attempts.** ResearchMath shows SFT gains from mostly incorrect trajectories. When does this help or hurt RL initialization? Can partial verification (checkpoints, lemmas, sub-results) turn such problems into dense rewards?
- **Multi-hop lineage and licenses.** A Llama-generated prompt, answered by R1, verified by GPT-5.2, judged by a Qwen GenRM and rewritten by Gemma 4: which obligations attach? No standard "generator chain" schema exists, and EU §2.5's boundary between "enrich/clean" and "generate" is untested for verifiers and judges.
- **Share-alike propagation.** CC-BY-SA pools (ResearchMath-14k, Nemotron competitive coding) raise the question of whether complexified derivatives or trained models are "adaptations". No empirical or legal analysis targets RL task pools.
- **Adversarial acceptance without selecting ambiguity.** Protocols are needed that combine "teacher or expert can solve" certificates with "policy cannot solve" filters and a budget of expert audits per survivor (the GPQA incentive design is a starting point).
- **Checkpoint and rubric decompositions as hack-resistant dense rewards.** CritPt and PaperBench decompositions are evaluation-only. Turning them into RL rewards that resist judge exploitation (note 11) is open.
- **Teacher tracing under complexification.** Do PoS-template footprints survive license transplants, multi-hop regeneration and heavy complexification? This matters for both compliance and contamination analysis.
- **Classic ACL signals for LLM task families.** Schedule operator × depth families by learning progress (TSCL |slope|, bandits) with setter-solver's uniform feasibility targets and AMIGo-style adaptive thresholds, instead of static pass-rate bands. Keep the survey's archive of unsolved tasks for later policies. This is cheap and largely untested at LLM scale.

## References

1. NVIDIA (2025). *Nemotron-Post-Training-Dataset-v1* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1
2. NVIDIA (2025). *Nemotron-3-Nano-RL-Training-Blend* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend
3. NVIDIA (2026). *Nemotron-RL-Super-Training-Blends* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends
4. NVIDIA (2026). *Nemotron-RL-Ultra-Training-Blends* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends
5. NVIDIA (2026). *Nemotron-RL-Lightning-Training-Blend* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-RL-Lightning-Training-Blend
6. NVIDIA (2026). *Nemotron-RL-Math-v2* (dataset card). Hugging Face. https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2
7. Bercovich, A., Levy, I., Golan, I., Dabbah, M., El-Yaniv, R., Puny, O., et al. (2025). *Llama-Nemotron: Efficient Reasoning Models*. arXiv:2505.00949. https://arxiv.org/abs/2505.00949 (dataset card: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
8. Prime Intellect (2025). *SYNTHETIC-2 release* (blog) and SYNTHETIC-2 / SYNTHETIC-2-RL dataset cards. https://www.primeintellect.ai/blog/synthetic-2-release · https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-RL
9. Badertdinov, I., Golubev, A., Nekrashevich, M., Shevtsov, A., Karasik, S., Andriushchenko, A., Trofimova, M., Litvintseva, D., Yangel, B. (2025). *SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents*. NeurIPS 2025; arXiv:2505.20411. https://arxiv.org/abs/2505.20411
10. Badertdinov, I., Nekrashevich, M., Shevtsov, A., Golubev, A. (2026). *SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale*. arXiv:2602.23866. https://arxiv.org/abs/2602.23866
11. Intelligent Internet (2025). *II-Thought-RL-v0* (dataset card). Hugging Face. https://huggingface.co/datasets/Intelligent-Internet/II-Thought-RL-v0
12. Su, Y., Yu, D., Song, L., Li, J., Mi, H., Tu, Z., et al. (2025). *Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains*. arXiv:2503.23829. https://arxiv.org/abs/2503.23829
13. Zhang, J., Petrui, C., Nikolić, K., Tramèr, F. (2025). *RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics*. NeurIPS 2025; arXiv:2505.12575. https://arxiv.org/abs/2505.12575
14. Peyronnet, A., Gloeckle, F., Hayat, A. (2026). *LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics*. arXiv:2602.24173. https://arxiv.org/abs/2602.24173
15. He, L., Yu, Q., Dong, H., Liao, B., Xu, X., Goldblum, M., et al. (2026). *LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches*. arXiv:2604.01754. https://arxiv.org/abs/2604.01754
16. Son, G., Yi, S., Gwak, M., Ko, H., Jang, W., Yu, Y. (2026). *ResearchMath-14K: Scaling Research-Level Mathematics via Agents*. arXiv:2605.28003. https://arxiv.org/abs/2605.28003
17. Balunović, M., Dekoninck, J., Jovanović, N., Petrov, I., Vechev, M. (2025). *MathConstruct: Challenging LLM Reasoning with Constructive Proofs*. arXiv:2502.10197. https://arxiv.org/abs/2502.10197
18. Li, Y., Li, W., Gao, X., Sun, M., Wang, X., Pei, Q., Wu, L. (2026). *SPARK: Skeleton-Guided Reasoning Synthesis from Large-Scale Scientific Literature*. arXiv:2608.30214. https://arxiv.org/abs/2608.30214
19. Starace, G., Jaffe, O., Sherburn, D., Aung, J., Chan, J. S., Maksin, L., et al. (2025). *PaperBench: Evaluating AI's Ability to Replicate AI Research*. arXiv:2504.01848. https://arxiv.org/abs/2504.01848
20. Kon, P. T. J., Liu, J., Zhu, X., Ding, Q., Peng, J., Xing, J., et al. (2025). *EXP-Bench: Can AI Conduct AI Research Experiments?* arXiv:2505.24785. https://arxiv.org/abs/2505.24785
21. Krishnan, S., Chang, S., Nagaraj, A. (2026). *ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge*. arXiv:2609.12366. https://arxiv.org/abs/2609.12366
22. Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*. arXiv:2311.12022. https://arxiv.org/abs/2311.12022
23. M-A-P Team, Du, X., Yao, Y., Ma, K., Wang, B., Zheng, T., et al. (2025). *SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines*. arXiv:2502.14739. https://arxiv.org/abs/2502.14739
24. Patwardhan, T., Dias, R., Proehl, E., Kim, G., Wang, M., Watkins, O., et al. (2025). *GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks*. arXiv:2510.04374. https://arxiv.org/abs/2510.04374
25. Vidgen, B., Fennelly, A., Pinnix, E., Benchek, J., Khan, D., Richards, Z., et al. (2025). *The AI Productivity Index (APEX)*. arXiv:2509.25721. https://arxiv.org/abs/2509.25721
26. Zhu, M., Tian, M., Yang, X., Zhou, T., Yuan, L., Zhu, P., et al. (2025). *Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark*. arXiv:2509.26574. https://arxiv.org/abs/2509.26574
27. Wang, M., Lin, R., Hu, K., Jiao, J., Chowdhury, N., Chang, E., et al. (2026). *FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks*. arXiv:2601.21165. https://arxiv.org/abs/2601.21165
28. Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., et al. (2025). *Humanity's Last Exam*. arXiv:2501.14249. https://arxiv.org/abs/2501.14249
29. Longpre, S., Mahari, R., Chen, A., Obeng-Marnu, N., Sileo, D., Brannon, W., et al. (2023). *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI*. arXiv:2310.16787. https://arxiv.org/abs/2310.16787
30. Longpre, S., Singh, N., Cherep, M., Tiwary, K., Materzynska, J., Brannon, W., et al. (2024). *Bridging the Data Provenance Gap Across Text, Speech and Video*. arXiv:2412.17847. https://arxiv.org/abs/2412.17847
31. Longpre, S., Mahari, R., Lee, A., Lund, C., Oderinwale, H., Brannon, W., et al. (2024). *Consent in Crisis: The Rapid Decline of the AI Data Commons*. arXiv:2407.14933. https://arxiv.org/abs/2407.14933
32. Wang, B., Chen, Y., Shi, J., Li, M., Lyu, Y., Wu, Y., et al. (2026). *Hidden Licensing Risks in the LLMware Ecosystem*. arXiv:2602.10758. https://arxiv.org/abs/2602.10758
33. Wadhwa, S., Shaib, C., Amir, S., Wallace, B. C. (2025). *Who Taught You That? Tracing Teachers in Model Distillation*. arXiv:2502.06659. https://arxiv.org/abs/2502.06659
34. Google (2026). *Gemma Terms of Use* (last modified 2026-04-01). https://ai.google.dev/gemma/terms
35. Meta (2024). *Llama 3.1 Community License Agreement* (release 2024-07-23). https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
36. Meta (2025). *Llama 4 Community License Agreement* (effective 2025-04-05). https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE
37. Meta (2023). *Llama 2 Community License Agreement*. https://github.com/meta-llama/llama/blob/main/LICENSE
38. Alibaba Cloud (2024). *Qwen License Agreement* (Qwen2.5-72B-Instruct) and *Qwen Research License Agreement* (Qwen2.5-3B-Instruct). https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE · https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
39. DeepSeek (2025). *DeepSeek-R1 Release* (2025-01-20). https://api-docs.deepseek.com/news/news250120
40. Moonshot AI (2025). *Kimi K2 License (Modified MIT)*. https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/LICENSE
41. MiniMax (2025). *MiniMax-M2 License (Modified MIT)*. https://github.com/MiniMax-AI/MiniMax-M2/blob/main/LICENSE
42. NVIDIA (2025). *NVIDIA Nemotron Open Model License* (last modified 2025-12-15). https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
43. Anthropic (2025). *Commercial Terms of Service* (effective 2025-06-17). https://www.anthropic.com/legal/commercial-terms
44. Google (2026). *Gemini API Additional Terms of Service* (effective 2026-03-23). https://ai.google.dev/gemini-api/terms
45. OpenAI. *Terms of Use* (not re-fetched: HTTP 403). https://openai.com/policies/terms-of-use/
46. European Commission AI Office (2025). *Explanatory Notice and Template for the Public Summary of Training Content for general-purpose AI models* (24 July 2025). https://digital-strategy.ec.europa.eu/en/library/explanatory-notice-and-template-public-summary-training-content-general-purpose-ai-models
47. Schneider-Kamp, P., Nielsen, J., Barmina, G., Enevoldsen, K., Galke Poech, L. (2026). *DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data*. arXiv:2608.13517. https://arxiv.org/abs/2608.13517
48. Li, J., Jin, Z., Men, T., Hao, Y., Zhu, K., Wang, L., et al. (2026). *Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application*. arXiv:2606.12191. https://arxiv.org/abs/2606.12191
49. Yang, Z., Fu, J., Liu, Y., Liu, H., Zhang, Y., Cao, K., et al. (2026). *Scaling Large Reasoning Models beyond Human Supervision: A Path toward Superintelligence*. arXiv:2608.31075. https://arxiv.org/abs/2608.31075
50. Zhang, K., Zuo, Y., He, B., Sun, Y., Liu, R., Jiang, C., et al. (2025). *A Survey of Reinforcement Learning for Large Reasoning Models*. arXiv:2509.08827. https://arxiv.org/abs/2509.08827
51. Zhang, G., Geng, H., Yu, X., Yin, Z., Zhang, Z., Tan, Z., et al. (2025). *The Landscape of Agentic Reinforcement Learning for LLMs: A Survey*. arXiv:2509.02547. https://arxiv.org/abs/2509.02547
52. Yu, Z., Mou, Y., Yan, J., Luo, J., Chen, C., Wei, X., et al. (2026). *A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions*. arXiv:2604.17312. https://arxiv.org/abs/2604.17312
53. Yang, H., Xerri, M., Park, S., Zhang, H., Feng, Y., Kogilathota, S. A., et al. (2026). *Self-Improvement of Large Language Models: A Technical Overview and Future Outlook*. arXiv:2603.25681. https://arxiv.org/abs/2603.25681
54. Gao, H., Geng, J., Hua, W., Hu, M., Juan, X., Liu, H., et al. (2025). *A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence*. arXiv:2507.21046. https://arxiv.org/abs/2507.21046
55. Havrilla, A., Dai, A., O'Mahony, L., Oostermeijer, K., Zisler, V., Albalak, A., et al. (2024). *Surveying the Effects of Quality, Diversity, and Complexity in Synthetic Data From Large Language Models*. arXiv:2412.02980. https://arxiv.org/abs/2412.02980
56. Sukhbaatar, S., Lin, Z., Kostrikov, I., Synnaeve, G., Szlam, A., Fergus, R. (2017). *Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play*. ICLR 2018; arXiv:1703.05407. https://arxiv.org/abs/1703.05407
57. Florensa, C., Held, D., Geng, X., Abbeel, P. (2017). *Automatic Goal Generation for Reinforcement Learning Agents*. arXiv:1705.06366. https://arxiv.org/abs/1705.06366
58. Campero, A., Raileanu, R., Küttler, H., Tenenbaum, J. B., Rocktäschel, T., Grefenstette, E. (2020). *Learning with AMIGo: Adversarially Motivated Intrinsic Goals*. ICLR 2021; arXiv:2006.12122. https://arxiv.org/abs/2006.12122
59. Racanière, S., Lampinen, A. K., Santoro, A., Reichert, D. P., Firoiu, V., Lillicrap, T. P. (2019). *Automated curricula through setter-solver interactions*. ICLR 2020; arXiv:1909.12892. https://arxiv.org/abs/1909.12892
60. Matiisen, T., Oliver, A., Cohen, T., Schulman, J. (2017). *Teacher-Student Curriculum Learning*. arXiv:1707.00183. https://arxiv.org/abs/1707.00183
61. Graves, A., Bellemare, M. G., Menick, J., Munos, R., Kavukcuoglu, K. (2017). *Automated Curriculum Learning for Neural Networks*. arXiv:1704.03003. https://arxiv.org/abs/1704.03003
62. Portelas, R., Colas, C., Weng, L., Hofmann, K., Oudeyer, P.-Y. (2020). *Automatic Curriculum Learning For Deep RL: A Short Survey*. arXiv:2003.04664. https://arxiv.org/abs/2003.04664
63. Wu, Y., Jiang, A. Q., Ba, J., Grosse, R. (2020). *INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving*. arXiv:2007.02924. https://arxiv.org/abs/2007.02924
64. Polu, S., Han, J. M., Zheng, K., Baksys, M., Babuschkin, I., Sutskever, I. (2022). *Formal Mathematics Statement Curriculum Learning*. arXiv:2202.01344. https://arxiv.org/abs/2202.01344
65. Wang, M., Deng, J. (2020). *Learning to Prove Theorems by Learning to Generate Theorems*. arXiv:2002.07019. https://arxiv.org/abs/2002.07019
66. Zelikman, E., Wu, Y., Mu, J., Goodman, N. D. (2022). *STaR: Bootstrapping Reasoning With Reasoning*. arXiv:2203.14465. https://arxiv.org/abs/2203.14465
67. Raoof, N., Zhuang, R., Nezhurina, M., Guha, E., Tejaswi, A., Marten, R., et al. (2026). *OpenThoughts-Agent: Data Recipes for Agentic Models* (cross-reference; covered in note 16). arXiv:2606.24855. https://arxiv.org/abs/2606.24855
