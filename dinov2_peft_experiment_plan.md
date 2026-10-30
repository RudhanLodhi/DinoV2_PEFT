# Experimental Plan: Parameter-Efficient Fine-Tuning of DINOv2 for Radiology

## 1. Project Overview

**Base paper:** *Evaluating General Purpose Vision Foundation Models for Medical Image Analysis: An Experimental Study of DINOv2 on Radiology Benchmarks* (uploaded version v4, September 2024).

**Working title:** *Layer-Selective Parameter-Efficient Fine-Tuning of DINOv2 for Multi-Label Radiology Classification*

### Research question

Can layer-selective and rank-controlled LoRA achieve competitive multi-label radiology classification performance with fewer trainable parameters than the standard LoRA configuration used in the base paper?

### Hypotheses

- **H1:** The amount of LoRA capacity needed for radiology classification varies with rank and placement of the adaptation modules.
- **H2:** A carefully selected subset of transformer layers can achieve comparable or better AUROC with fewer trainable parameters than applying LoRA broadly.
- **H3:** The best parameter-efficient configuration can be selected using validation data without compromising the held-out test protocol.

These are hypotheses to test, not assumed outcomes.

## 2. What the Base Paper Establishes

The paper evaluates nine public radiology benchmarks spanning X-ray, CT, and MRI. Its dedicated PEFT experiment uses DINOv2 ViT-g/14 (approximately 1.1 billion parameters) on NIH Chest X-ray and CheXpert. It compares LoRA and BitFit with linear probing and full fine-tuning.

The appendix reports the following LoRA configuration:

| Hyperparameter | Original setting |
|---|---:|
| LoRA rank, `r` | 48 |
| LoRA alpha, `alpha` | 16 |
| LoRA dropout | 0.1 |
| LoRA learning rate | `2e-4` |
| Classification-head learning rate | `5e-3` |
| Batch size for larger ViTs | 32 |
| NIH training epochs | 20 |
| CheXpert training epochs | 10 |

The batch size and epoch counts are reported in Appendix B.1; LoRA-specific settings are reported in Appendix B.3. Confirm these against the official implementation before running experiments.

### Reported PEFT results

Table 9 of the base paper reports:

| Method | Trainable parameters | NIH Chest X-ray AUROC | CheXpert AUROC |
|---|---:|---:|---:|
| Linear probing | 1,500 | 0.759 | 0.818 |
| BitFit | 0.8M | 0.768 | 0.817 |
| LoRA | 8M | 0.767 | 0.823 |
| Full fine-tuning | 1,100M | 0.769 | 0.848 |

These values are reference results, not guaranteed results for a new reproduction. Implementation details, preprocessing, checkpoints, random seeds, and hardware can affect outcomes.

## 3. Dataset and Backbone Selection

### Primary dataset: CheXpert

- Multi-label chest X-ray classification.
- Five target labels in the base paper's benchmark.
- The paper uses frontal views only.
- Reuse the authors' preprocessing and data splits wherever possible.

**Rationale:** It provides a direct comparison with the base paper's PEFT results.

### Secondary dataset: NIH Chest X-ray

- Fourteen disease labels in the base paper's benchmark.
- Useful for checking whether findings transfer to another dataset and label distribution.
- Treat this as a secondary evaluation until the primary CheXpert experiments are stable.

### Backbone decision

Choose one of the following tracks before running experiments:

- **Track A — Exact reproduction:** Use DINOv2 ViT-g/14 and reproduce the paper's PEFT setup if available compute permits.
- **Track B — Resource-constrained replication:** Use DINOv2 ViT-B/14, which the paper also evaluates, and reproduce the methodology at smaller scale.

Track B is a scaled replication, not an exact reproduction of the ViT-g/14 PEFT results. Do not directly present ViT-B/14 results as if they used the same backbone as the paper's Table 9.

## 4. Phase I — Reproduce the Baselines

### E0. Linear probing

1. Freeze the entire DINOv2 encoder.
2. Train only the multi-label classification head.
3. Reuse the base paper's preprocessing and data splits.
4. Select hyperparameters using validation data only.
5. Report mean per-label AUROC, consistent with the paper's multi-label evaluation convention.

**Purpose:** Establish the performance of the pretrained representations without encoder adaptation.

### E1. Standard LoRA reproduction

Begin with the configuration reported in the base paper:

- Rank `r = 48`
- Alpha `alpha = 16`
- LoRA dropout `0.1`
- LoRA learning rate `2e-4`
- Classification-head learning rate `5e-3`
- Batch size 32 for larger ViTs, if hardware allows
- 20 epochs for NIH Chest X-ray
- 10 epochs for CheXpert

The original paper selects the best checkpoint from the tested epochs. In this study, select checkpoints using validation performance only; never select a checkpoint based on test AUROC.

### Reproduction acceptance checks

Before introducing new methods, verify that:

1. Dataset and label processing match the paper.
2. The backbone checkpoint and architecture are correct.
3. Trainable parameter counts match the selected configuration.
4. Training and validation curves behave sensibly.
5. AUROC is reasonably close to the reported result, allowing for implementation and hardware differences.

If the results differ materially, investigate the pipeline before proceeding.

## 5. Phase II — Controlled LoRA Experiments

### E2. Rank sweep

Keep the backbone, dataset, target modules, training schedule, and preprocessing fixed. Change only the LoRA rank.

| Run | Rank `r` | Purpose |
|---|---:|---|
| E2a | 4 | Very low adaptation capacity |
| E2b | 8 | Low capacity |
| E2c | 16 | Moderate capacity |
| E2d | 32 | Higher capacity |
| E2e | 48 | Original paper's configuration |

For the initial reproduction, preserve the original `alpha = 16`. In a separate controlled sweep, consider scaling alpha proportionally with rank so that changes in rank do not unintentionally change the effective update scale. Document the convention used.

For every run, record:

- Mean per-label AUROC.
- Per-label AUROC.
- Trainable parameter count and percentage.
- Peak GPU memory.
- Training time per epoch and total training time.
- Best validation AUROC and corresponding epoch.

**Goal:** Determine whether the original rank of 48 is necessary, and characterize the performance–efficiency trade-off.

### E3. LoRA target-module and layer-placement study

Keep rank fixed while changing which transformer projections or blocks receive LoRA.

#### E3a. Attention-only baseline

Apply LoRA to query and value projections if this matches the repository's implementation. If the official implementation uses different target modules, reproduce that configuration first.

#### E3b. All attention projections

Apply LoRA to query, key, value, and output projections where supported by the selected architecture.

#### E3c. Upper-layer adaptation

Apply LoRA only to the upper portion of the transformer, initially the final quarter of its blocks. Test whether later representations are sufficient for task-specific adaptation.

#### E3d. Lower-layer or distributed adaptation

Compare adapting early blocks against distributing adaptation across the network. This tests whether adaptation benefits more from changing lower-level features or higher-level representations.

**Implementation check:** Confirm exact module names and architecture structure in the chosen DINOv2 implementation. Do not assume all implementations expose identical names.

**Fairness control:** Changing target modules changes the number of trainable parameters. Report raw results and, where feasible, parameter-matched comparisons so that gains are not attributed to placement when they may come from additional capacity.

### E4. Layer-selective LoRA (candidate main contribution)

Use findings from E2 and E3 to create a configuration that applies LoRA only to selected layers and projections.

Compare:

1. **Standard LoRA:** Apply LoRA across the chosen attention blocks.
2. **Layer-selective LoRA:** Apply LoRA only to a subset of blocks.
3. **Parameter-matched LoRA:** Where feasible, adjust rank in selected blocks so the total trainable parameter count is comparable to the standard configuration.

Choose the subset using validation data or a predefined layer-selection rule. Do not use test results to choose layers.

**Primary question:** Can layer-selective LoRA preserve AUROC while reducing trainable parameters and/or training cost?

Layer-selective adaptation is an established general idea. A literature review is required before claiming novelty. The potential contribution is a well-motivated allocation strategy and its empirical validation on radiology benchmarks.

## 6. Phase III — Alternative PEFT Method (Optional)

After standard and layer-selective LoRA work, add **one** alternative method initially: IA³ or bottleneck adapters.

| Experiment | Method | Purpose |
|---|---|---|
| E5a | IA³ | Test whether activation scaling can compete with LoRA. |
| E5b | Bottleneck adapters | Test whether small feature-transformation modules are more effective than low-rank weight updates. |

Choose one first; do not implement both unless time and compute allow. Keep the backbone, data splits, classification head, validation protocol, and compute budget as consistent as possible.

This comparison is useful, but it is not by itself a new method. Avoid adding DoRA, AdaLoRA, VeRA, and prompt tuning all at once; each adds implementation and validation work.

## 7. Evaluation Protocol

The base paper uses AUROC for classification and averages per-class results in multi-label evaluation.

| Metric | Purpose |
|---|---|
| Mean per-label AUROC | Primary comparison with the base paper |
| Per-label AUROC | Shows which diseases benefit or regress |
| Macro-F1 | Threshold-dependent classification quality across labels |
| Trainable parameter count | Measures parameter efficiency |
| Peak GPU memory | Measures hardware requirements |
| Training time | Measures adaptation cost |
| Checkpoint size | Measures storage requirements |

If calibration is added as an optional extension, include Brier score and expected calibration error (ECE). Calibration should not delay baseline reproduction.

### Statistical reliability

- Aim to run the final baseline and proposed method with at least three random seeds if compute permits.
- Report mean and standard deviation.
- If broad exploratory sweeps can only use one seed, reserve repeated runs for the final key comparison and disclose this limitation.
- Select hyperparameters and checkpoints using validation data only.
- Evaluate on the test set after selecting the final configuration.

## 8. Prioritized Experiment Matrix

### P0 — Must complete

- E0: Linear probing.
- E1: Standard LoRA reproduction.
- E2: Rank sweep.

These establish a trustworthy baseline and measure the effect of LoRA capacity.

### P1 — Main research contribution

- E3: Target-module and layer-placement study.
- E4: Layer-selective LoRA.
- Parameter-matched comparison against standard LoRA.

This is the core of the proposed paper.

### P2 — Optional extensions

- E5: One alternative PEFT method.
- Cross-dataset evaluation on NIH Chest X-ray.
- Calibration analysis.

Proceed only after the core experiments are stable.

## 9. Compute and Schedule

The paper reports batch size 32 for larger ViTs, 20 epochs for NIH Chest X-ray, and 10 epochs for CheXpert. These settings may not fit every GPU unchanged.

For a resource-constrained study:

- Start with ViT-B/14 if ViT-g/14 is too expensive.
- Use mixed precision and reduce batch size if needed.
- Run the full baseline before the rank sweep.
- Use one seed for exploratory runs and multiple seeds for the final comparison.
- Defer external validation until primary results are stable.

Planning estimates:

| Stage | Estimated duration |
|---|---|
| Environment, dataset, and preprocessing | 2–4 days |
| Baseline reproduction and debugging | 3–7 days |
| Rank and placement experiments | 1–2 weeks |
| Layer-selective method and final comparisons | 1–2 weeks |
| Analysis, tables, and writing | 1 week |

These are planning estimates, not measured runtime guarantees. Actual duration depends on GPU, dataset access, and how much of the repository runs without modification.

## 10. Success Criteria

Define success before running experiments.

- **Performance:** The proposed method matches or improves the baseline's mean per-label AUROC on the held-out test set.
- **Efficiency:** It reduces trainable parameters or training cost by a meaningful amount.
- **Reliability:** The result is consistent across random seeds, or uncertainty is clearly reported.
- **Scientific value:** Ablations establish whether rank, layer placement, or both explain the outcome.

A slightly lower AUROC may still be a useful efficiency result if the reduction in training cost is substantial. Report trade-offs honestly rather than claiming an overall improvement.

## 11. Experiment Log Template

Maintain one row per run in a CSV or spreadsheet.

| Field | Example / instruction |
|---|---|
| Run ID | `lora_r16_upper` |
| Dataset | CheXpert |
| Backbone | DINOv2 ViT-B/14 or ViT-g/14 |
| Method | Layer-selective LoRA |
| Rank and alpha | Record actual values |
| Target modules | Record actual projections |
| Adapted layers | Record selected blocks |
| Trainable parameters | Measure from model |
| Peak GPU memory | Record measured value |
| Training time | Record measured duration |
| Validation AUROC | Record measured score |
| Test AUROC | Record only after configuration selection |
| Random seed | Record actual seed |

Do not fill performance or resource fields with estimates. Collect them from actual runs.

## 12. Recommended Final Scope

The recommended core sequence is:

**Baseline reproduction → LoRA rank sweep → layer-placement analysis → layer-selective LoRA → parameter-matched comparison.**

Use CheXpert as the primary dataset and NIH Chest X-ray as a secondary evaluation if time permits. Start with ViT-g/14 if resources allow; otherwise, document the ViT-B/14 experiment as a scaled replication.

A rank sweep alone is unlikely to provide a strong methodological novelty claim. Layer-selective parameter allocation is a more promising direction, but related work must be reviewed before claiming the method is new.

### Immediate next step

Inspect the official repository and establish the baseline configuration. Once the baseline reproduces successfully, implement the controlled rank and layer-placement experiments.
