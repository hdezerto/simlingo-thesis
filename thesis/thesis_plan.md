# Thesis Plan

This file is a planning scaffold, not final thesis prose. The goal is to decide
what each section must argue, what evidence supports it, and which figures or
tables are needed before writing the final LaTeX text.

## Working Approach

- Use this file to organize claims, evidence, figures, tables, and open questions.
- Keep result-dependent sections provisional until the final runs and renders finish.
- Move stable bullets into the LaTeX document only after each section has a clear argument.
- Mark missing information with `TODO` instead of stopping the planning flow.
- Keep detailed experiment logs in `berzelius_training.md`; use this file for thesis structure.

## Research Framing

### Research Question

Can compact temporal adapters over a short front-camera history improve the
closed-loop behavior of single-frame VLA driving models, particularly in dynamic
interaction scenarios, and with what computational cost?

### Hypothesis

Compact temporal adapters over a short front-camera history will improve
behavior in dynamic interaction scenarios for single-frame VLA driving models,
without degrading overall closed-loop performance and with limited additional
computational cost.

### Contribution

This thesis investigates compute-efficient temporal context integration for
single-frame VLA driving through a case study on SimLingo. It implements and
evaluates two compact temporal adapters, a query-based Q-former and a
feature-delta encoder, using closed-loop Bench2Drive experiments to analyze
overall driving performance, behavior in dynamic interaction scenarios, and
computational cost.



## Thesis Outline

0. Abstract
1. Introduction
2. Related Work
3. Baseline: SimLingo and Problem Analysis
4. Temporal Adapter Methods
5. Dataset, Supervision, and Diagnostics
6. Experimental Setup
7. Results
8. Discussion
9. Conclusion and Future Work
10. References
11. Appendix

## Section Plans

### 0. Abstract

**Purpose:** Summarize the whole thesis after the main text is stable.

**Main Points**

- Problem: single-frame VLA driving models can miss motion cues in dynamic
  interaction scenarios.
- Method: compact temporal adapters over short front-camera history.
- Evaluation: closed-loop Bench2Drive, selected interaction routes, diagnostics,
  and computational cost.
- Finding: TODO once final results are known.
- Contribution: a compute-efficient temporal extension study for SimLingo.

**Open Items**

- TODO: write the abstract last, after Results and Discussion are fixed.

### 1. Introduction

**Purpose:** Motivate temporal context for VLA driving and define the thesis
scope, research question, hypothesis, and contribution.

#### 1.1 Field and Motivation

- Define vision-language-action driving and explain why it matters for
  end-to-end decision making, instruction following, and interpretability.
- Explain that single-frame VLA models can miss motion information needed for
  yielding, crossing traffic, turning, and merging.
- Use the VLA survey and the VLA past/present/future paper for terminology,
  taxonomy, and field framing.
- Optional: use industry/truck examples only as motivation, not as core evidence.

#### 1.2 Problem and Case Study

- Explain why adding temporal context is not free: it can increase memory,
  latency, and training cost.
- Introduce SimLingo as the case-study model: a strong single-frame closed-loop
  VLA driving model with explicit language-action alignment.
- Introduce Bench2Drive as the closed-loop benchmark used to evaluate behavior
  in interactive driving scenarios.
- Motivate the thesis from the observed failure pattern: motion-sensitive
  interactions were not reliably solved by the single-frame baseline.

#### 1.3 Research Question and Contribution

- State the research question.
- State the hypothesis.
- State the contribution.
- Clarify the scope: this is a case study on SimLingo, not a claim across many
  base VLA models.

#### 1.4 Thesis Structure

- Briefly explain what each chapter covers.
- Keep this paragraph short and factual.

**Evidence / References**

- A Survey on Vision-Language-Action Models for Autonomous Driving.
- Vision-Language-Action Models for Autonomous Driving: Past, Present, and
  Future.
- SimLingo.
- Bench2Drive.
- Optional context: mbreuss ICLR 2026 blog post.
- Optional context: PlusAI and NVIDIA Alpamayo.

**Figures / Tables**

- Optional high-level problem figure or motivating failure frame.

**Open Items**

- TODO: decide whether to include a short motivating failure-frame figure in the
  introduction or save all qualitative examples for Sections 3 and 7.

### 2. Related Work

**Purpose:** Position the thesis relative to VLA driving, temporal modeling, and
efficient temporal bottlenecks.

#### 2.1 VLA Models for Autonomous Driving

- Explain how VLA driving models combine visual input, language, and action
  prediction.
- Position SimLingo as the direct baseline and architectural anchor.
- Use OpenDriveVLA as a contrast for grounding-heavy multimodal VLA designs.

#### 2.2 Temporal Context in Driving

- Explain why many driving decisions depend on actor motion rather than only
  current-frame appearance.
- Motivate temporal history for dynamic interaction scenarios.
- Explain why naive multi-frame modeling can be expensive.
- Use ORION and HiST-VLA to connect temporal aggregation to driving.

#### 2.3 Compact Temporal Bottlenecks and Query Adapters

- Explain why compact bottlenecks are attractive when many visual tokens are
  available but only a small temporal summary should be passed to the LLM.
- Use BLIP-2 as the main query-bottleneck motivation.
- Connect this to the Q-former temporal adapter.

#### 2.4 Frontier Alternatives

- Briefly discuss methods that move toward grounding-heavy, hierarchical, latent,
  or future-aware reasoning.
- Use LaST-VLA, FutureSightDrive, FLARE, and FastDriveVLA as contrast or future
  work directions.
- Make clear that these are not the main method basis for this thesis.

**Figures / Tables**

- Optional related-work comparison table:
  method, temporal input, adapter type, modality, benchmark, compute strategy.

**Open Items**

- TODO: decide which related-work papers are essential enough for the main text
  and which belong only in discussion or future work.

### 3. Baseline: SimLingo and Problem Analysis

**Purpose:** Explain the starting point and show why temporal context became the
main thesis direction.

#### 3.1 Baseline Architecture

- Summarize SimLingo at the level needed for this thesis.
- Cover front-camera image tokens, language input, waypoint/route/speed heads,
  and commentary supervision.
- Explain what remains unchanged when temporal adapters are added.

#### 3.2 Benchmark and Reproduction Setup

- Summarize Bench2Drive as the closed-loop evaluation setting.
- Explain the baseline reproduction setup.
- Present the aggregate baseline results that motivate further work.

#### 3.3 Failure Analysis

- Show selected failures where static appearance and green-light cues are not
  enough because the ego must account for moving or blocking actors.
- Emphasize dynamic interaction cases: junctions, turning, merging, and actor
  flow.
- Use rendered examples to make the failure concrete.

#### 3.4 Motivation for Temporal Context

- Explain why the failures motivate temporal interaction reasoning.
- Explain why the thesis moved away from a language-alignment-only framing.
- Connect this motivation back to the research question.

**Evidence / References**

- SimLingo paper and code for baseline architecture and recipe.
- Bench2Drive for closed-loop routes, metrics, and scenario context.
- Own baseline results and rendered videos.

**Figures / Tables**

- Baseline architecture diagram.
- Baseline full benchmark table.
- Selected failure-case table.
- One or more rendered examples of dynamic interaction failures.

**Open Items**

- TODO: choose the final baseline checkpoint and metrics to report.
- TODO: choose the clearest qualitative failure frames.

### 4. Temporal Adapter Methods

**Purpose:** Describe the two temporal methods and how they are integrated into
the baseline model.

#### 4.1 Design Goals and Constraints

- Add temporal context while keeping the baseline current-frame path mostly
  intact.
- Keep the method compact enough to be plausible under compute constraints.
- Avoid a full redesign of the VLA model.
- Separate the architectural adapters from later supervision changes.

#### 4.2 Temporal Input Formulation

- Use a short front-camera history.
- Keep the current frame as the main source of lane, traffic light, object, and
  route context.
- Encode history into additional temporal tokens.
- Explain the ordering and role of temporal tokens in the VLA input sequence.

#### 4.3 Query-Based Temporal Adapter

- Learn a fixed number of query tokens.
- Let the query tokens attend over historical visual features.
- Output compact temporal tokens that can be inserted into the LLM input.
- Connect the design to BLIP-2 and temporal query aggregation ideas.

#### 4.4 Feature-Delta Temporal Encoder

- Compute current-vs-past visual feature changes.
- Include motion magnitude and signed feature change information.
- Compress the resulting motion features into a compact set of temporal tokens.
- Explain why this method was introduced after Q-former diagnostics showed weak
  real-history separation.

#### 4.5 Integration into the VLA Input Sequence

- Insert temporal tokens into the LLM input as visual-context tokens.
- Preserve the baseline current-image token path.
- Explain which placeholders are replaced by image, temporal, and waypoint
  embeddings.
- Clarify that the driving heads receive the downstream LLM representation, not
  generated text alone.

#### 4.6 Trainable and Frozen Components

- List what is trained for the main temporal runs.
- Explain why some components are frozen for memory/cost reasons.
- Explain which components are shared between the Q-former and feature-delta
  methods.

**Evidence / References**

- SimLingo: baseline interface and what stays fixed.
- BLIP-2: query-based bottleneck motivation.
- ORION: temporal query aggregation in closed-loop driving.

**Figures / Tables**

- Common integration diagram showing baseline plus temporal tokens.
- Q-former block diagram.
- Feature-delta block diagram.
- Table comparing the two methods:
  input, compression mechanism, number of tokens, trainable components, cost.

**Open Items**

- TODO: decide how much mathematical notation is needed for each adapter.
- TODO: add exact token counts and history length used in final reported runs.

### 5. Dataset, Supervision, and Diagnostics

**Purpose:** Explain what supervision is available, why the language supervision
can be noisy, and how diagnostics guided later changes.

#### 5.1 Training Data and Split

- Explain that training is offline from stored dataset frames and metadata.
- Explain the split and sampling strategy used by the dataloader.
- Distinguish offline training data from closed-loop evaluation rollouts.

#### 5.2 Control, Route, Language, and QA Supervision

- List the available training signals: image/history, target point, command,
  route, speed, waypoints, commentary, QA, and actor boxes.
- Explain the difference between waypoint supervision and commentary
  supervision.
- Emphasize that expert waypoints are the strongest control target.

#### 5.3 Commentary Noise in Dynamic Interactions

- Explain how commentary can contain stale or heuristic claims even when the
  future waypoints are useful.
- Describe the key problematic pattern: green light or route-following text can
  say to accelerate even when the ego should yield to occupied actor flow.
- Include examples such as "junction is clear" or "vehicles are stopped" when
  saved metadata shows moving or blocking actors.

#### 5.4 Temporal Dataloader and Supervision Audits

- Use the temporal dataloader audit to show that history frames and temporal
  inputs are loaded as expected.
- Use supervision audits to check commentary, actor boxes, motion labels, and
  future behavior.
- Explain that these diagnostics are part of the thesis evidence, not just
  debugging notes.

#### 5.5 Interaction-Aware Commentary and Actor-Motion Labels

- Explain the later strict interaction-yield cleanup.
- Explain actor-motion labels and how they support the auxiliary temporal loss.
- Connect this supervision change directly to the selected failure cases.

**Figures / Tables**

- Table of supervision signals:
  image/history, target point, command, route, speed, waypoints, commentary, QA,
  actor boxes.
- Audit summary table.
- Example table:
  original commentary, detected issue, corrected/commentary-enriched target.
- Optional frame examples for strict interaction-yield cases.

**Open Items**

- TODO: decide how much of the commentary-cleanup code should be described in
  the main text versus appendix.
- TODO: update audit numbers after final strict audit results are fixed.

### 6. Experimental Setup

**Purpose:** Make the experiments reproducible and tied directly to the research
question.

#### 6.1 Implementation Details

- Describe the codebase branch and main implementation files at a high level.
- State the final history length, number of temporal tokens, and adapter
  dimensions for reported runs.
- Avoid low-level code walkthroughs unless needed for reproducibility.

#### 6.2 Training Protocol

- Describe dataset, batch size, number of epochs, checkpoint initialization, and
  trainable modules.
- Separate earlier exploratory runs from final runs.
- Explain when training starts from a SimLingo checkpoint versus from the base
  initialization, if both are reported.

#### 6.3 Evaluation Protocol

- Describe Bench2Drive closed-loop evaluation.
- Describe selected routes and seeds.
- Explain rendered diagnostics and saved frame metadata.

#### 6.4 Metrics

- Define driving score, route completion, infraction penalty, collision counts,
  red-light infractions, timeouts, and other reported metrics.
- Explain which metrics answer the main research question and which are
  supporting diagnostics.

#### 6.5 Selected Dynamic-Interaction Routes

- Define the selected route subset.
- Explain why these routes are relevant to dynamic interaction behavior.
- Make clear that selected renders are qualitative/diagnostic, not full
  benchmark replacements.

#### 6.6 Computational Cost Measurement

- Report available training time, GPU memory, inference cost, or cluster energy
  measurements.
- Compare temporal methods against the single-frame baseline where possible.

#### 6.7 Ablations and Diagnostics

- Include history ablations such as real history, repeated-current history, and
  zero temporal tokens.
- Include prompt and commentary diagnostics where relevant.
- Explain which diagnostics are controlled tests rather than main benchmark
  results.

**Figures / Tables**

- Experiment configuration table.
- Metric definition table.
- Selected route/scenario table.
- Hardware and runtime table if enough measurements are available.

**Open Items**

- TODO: finalize which checkpoints are main results versus diagnostic results.
- TODO: decide whether to report both selected renders and full benchmark scores
  for every version or only the strongest versions.

### 7. Results

**Purpose:** Answer the research question using closed-loop performance,
dynamic-interaction behavior, diagnostics, and compute cost.

#### 7.1 Main Closed-Loop Benchmark Results

- Compare baseline, Q-former variants, and feature-delta variants on full
  Bench2Drive metrics.
- Report aggregate driving score, route completion, and infraction penalty.
- Make clear which runs are final and which are exploratory.

#### 7.2 Selected Dynamic-Interaction Route Results

- Compare behavior on the selected motion-sensitive routes.
- Report collisions, timeouts, and route completion.
- Include qualitative examples for successes and failures.

#### 7.3 Temporal Signal and History-Ablation Diagnostics

- Report whether temporal tokens contain motion signal.
- Report whether real history changes the rollout compared with repeated-current
  or zero-history inputs.
- Use this section to explain why temporal signal alone may not imply behavior
  improvement.

#### 7.4 Supervision and Commentary Audit Results

- Report audit results for temporal data correctness.
- Report commentary/supervision audit results.
- Report whether strict interaction-yield cleanup reduces stale commentary
  patterns.

#### 7.5 Computational Cost

- Report training and inference cost relative to the single-frame baseline.
- Compare Q-former and feature-delta cost where possible.

#### 7.6 Summary of What Improved and What Did Not

- Summarize the answer to the research question in one compact table or
  paragraph.
- Separate closed-loop performance, dynamic-interaction behavior, diagnostics,
  and cost.

**Figures / Tables**

- Main benchmark table.
- Selected-route render table.
- Ablation table: real history, repeated-current history, zero temporal tokens.
- Temporal signal diagnostic table or heatmap figure.
- Compute-cost table.
- Qualitative frames for representative successes and failures.

**Open Items**

- TODO: fill in final v3/v6 training and render results.
- TODO: decide whether earlier v1-v5 results go in main text or appendix.

### 8. Discussion

**Purpose:** Interpret the results and explain what they imply for temporal VLA
driving models.

#### 8.1 Interpretation of Main Findings

- Discuss whether compact temporal adapters improved overall closed-loop
  performance.
- Discuss whether they improved dynamic interaction scenarios specifically.
- Discuss whether the compute trade-off was acceptable.

#### 8.2 Why Temporal Context Did or Did Not Help

- Explain why temporal signal can exist in tokens without reliably improving
  closed-loop behavior.
- Discuss Q-former compression as a possible bottleneck when real-history tokens
  remain close to repeat-current tokens.
- Discuss feature-delta tokens as a more explicitly motion-sensitive alternative.

#### 8.3 Role of Supervision and Commentary

- Discuss why stale or incorrect commentary can teach bad interaction priors.
- Explain why temporal inputs alone may not fix behavior if the supervision does
  not reward interaction-aware yielding.
- Connect this back to the strict interaction-yield changes.

#### 8.4 Relation to Prior Work

- Compare the minimal SimLingo temporal extension with larger hierarchical
  designs such as HiST-VLA.
- Contrast with latent/future-aware reasoning approaches such as LaST-VLA,
  FLARE, and FutureSightDrive.
- Contrast with grounding-heavy approaches such as OpenDriveVLA.
- Mention FastDriveVLA if discussing future token-efficiency improvements.

#### 8.5 Limitations

- One base model.
- Front-camera history only.
- Limited selected-route analysis.
- Simulator-specific behavior.
- Incomplete access to all causal interaction labels.

**Figures / Tables**

- Optional summary table:
  problem observed, diagnostic evidence, likely cause, attempted fix.

**Open Items**

- TODO: write this section only after final results are known, otherwise it will
  become too speculative.

### 9. Conclusion and Future Work

**Purpose:** State the final answer to the research question and identify the
most credible next steps.

#### 9.1 Answer to the Research Question

- Summarize whether compact temporal adapters helped.
- Summarize where they helped or failed.
- Summarize the computational cost.

#### 9.2 Key Lessons

- Summarize what was learned about temporal signal.
- Summarize what was learned about LLM/action-head usage of temporal tokens.
- Summarize what was learned about supervision and commentary quality.

#### 9.3 Future Work

- Stronger interaction labels.
- Explicit collision-path prediction.
- Richer multi-camera history.
- Token pruning for efficiency.
- Future-aware latent prediction.
- Larger spatio-temporal model redesigns.

**Evidence / References**

- SimLingo for language-action alignment follow-up.
- FastDriveVLA for efficiency-oriented future work.
- FLARE and FutureSightDrive for future-aware representations.

**Figures / Tables**

- None required.

**Open Items**

- TODO: final conclusion depends on v3/v6 results and final render analysis.

## Appendix Candidates

- Full experiment table for all Q-former and feature-delta versions.
- Full configuration details for main runs.
- Slurm commands, cluster setup, and checkpoint paths.
- Extra supervision audit outputs.
- Extra rendered route frames and commentary examples.
- Additional diagnostic plots and heatmaps.
- Implementation details that are too low-level for the main methods section.
