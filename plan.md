# Track 1 — Noise Event Detection: Implementation Plan

## 0. Goal

Build a strong, reproducible baseline-to-competitive system for Datathon@IndoML 2026 — Track 1 (Noise Event Detection).

### Task

Input: real-world Indic speech audio.

Output: temporal noise events with onset/offset timestamps (and any required event metadata according to the official Codabench submission schema).

Primary competition metrics:

- Event-based F1
- Segment Dice
- Combined score = Event-based F1 + Segment Dice

The system must ultimately generate the exact `predictions.jsonl` format required by Codabench. Do NOT invent the submission schema: inspect the official sample/template supplied in the competition files.

---

# 1. High-Level Architecture

Start with this architecture:

```text
Raw audio
   ↓
Resampling / normalization
   ↓
Log-mel spectrogram
   ↓
2D CNN acoustic encoder
   ↓
Temporal feature sequence
   ↓
Transformer Encoder
   ↓
Per-time-step temporal classifier
   ↓
Noise probability / class probabilities
   ↓
Temporal post-processing
   ↓
(onset, offset, class/tag if required)
   ↓
predictions.jsonl
```

### Core design principle

Do NOT treat each 0.1-second chunk as an independent classification problem.

Treat short-time frames as a sequence of acoustic tokens:

```text
x_1, x_2, ..., x_T
```

The CNN extracts local acoustic patterns and the Transformer uses self-attention to model temporal context across the sequence.

---

# 2. First Priority: Inspect the Competition Data

Before implementing the model, inspect the actual files.

Tasks:

1. Download/access the Track 1 training data.
2. Inspect directory structure.
3. Inspect annotation files.
4. Inspect train/validation/test splits.
5. Inspect the Codabench sample submission.
6. Determine the exact JSONL schema.
7. Determine how audio IDs map to predictions.
8. Determine whether class/category/tag must be predicted or only timestamps.
9. Determine whether test audio lengths vary.
10. Check sample rates, channels, bit depth, and file formats.
11. Check for corrupted files or missing annotations.
12. Record all findings in `docs/data_spec.md`.

Do not start training until the annotation schema and submission format are understood.

---

# 3. Dataset Structure

Competition training data is described as approximately:

| Tier | Approx. duration | Approx. segments | Supervision |
|---|---:|---:|---|
| Gold | 21.8 h | 11,111 | verified timestamps |
| Silver | 100.32 h | 61,642 | timestamped but less verified |
| Bronze | 32.42 h | 17,884 | noise tags, no timestamps |
| Total | 154.6 h | 90,637 | mixed |

The exact files available to the participant are authoritative.

The noise taxonomy has seven broad categories:

1. Animal
2. Vehicle / Traffic
3. Baby / Child
4. Singing / Music
5. Phone / Signal / Alarm
6. Appliance / Machine
7. Non-speech Human

The underlying annotations may also contain finer-grained tags.

Events may overlap each other and may overlap speech.

---

# 4. Data Exploration

Build an EDA script/notebook before serious model training.

Report:

- number of files
- total duration
- duration distribution
- sample-rate distribution
- channel distribution
- number of events
- events per audio
- event duration distribution
- event onset distribution
- overlap statistics
- category distribution
- tag distribution
- Gold/Silver/Bronze distribution
- number of simultaneous events
- proportion of audio containing no noise
- per-class duration
- per-class event count

Generate plots:

- event-duration histogram
- audio-duration histogram
- class/event-count distribution
- class total-duration distribution
- overlap histogram
- example annotated timelines
- spectrogram examples for each category

Save the EDA outputs so later experiments can be compared against them.

---

# 5. Audio Representation

## Initial representation

Use log-mel spectrograms.

Recommended initial configuration:

- sample rate: 16 kHz
- mono
- STFT window: 25 ms
- hop: 10 ms
- mel bins: 64 or 80
- log compression

Do not permanently hard-code these values until they are confirmed experimentally.

The model can internally downsample the 10 ms acoustic frames to approximately 100 ms temporal tokens.

---

# 6. 0.1-Second Tokenization

The original idea is to operate around 100 ms temporal resolution.

However, do NOT immediately use non-overlapping raw 100 ms audio chunks.

Preferred approach:

```text
audio
 ↓
10 ms spectrogram frames
 ↓
CNN
 ↓
temporal downsampling
 ↓
~100 ms model tokens
```

This preserves richer acoustic information while producing approximately 100 ms output resolution.

Experiment later with:

- ~50 ms token resolution
- ~100 ms token resolution
- ~200 ms token resolution

Use 100 ms as the first baseline.

---

# 7. CNN Acoustic Encoder

Initial model:

```text
Log-mel spectrogram
        ↓
Conv2D
        ↓
BatchNorm / GroupNorm
        ↓
Activation
        ↓
Conv2D
        ↓
Pooling / stride
        ↓
Conv2D
        ↓
Temporal feature sequence
```

The CNN should primarily learn local time-frequency patterns.

Do not over-engineer the CNN initially.

Start with a small/medium CNN and establish a reliable baseline.

Possible later experiments:

- ResNet-style CNN
- EfficientNet-style encoder
- depthwise separable convolutions
- pretrained audio encoder

---

# 8. Transformer Temporal Encoder

After the CNN, reshape the representation into:

```text
[B, T, D]
```

where:

- B = batch size
- T = number of temporal tokens
- D = embedding dimension

Use a Transformer Encoder.

Initial configuration:

- embedding dimension: 256
- layers: 4
- attention heads: 4 or 8
- FFN dimension: 512–1024
- dropout: 0.1
- pre-norm Transformer

Use positional information.

Start with learned positional embeddings or relative positional encoding.

The Transformer should answer:

> Given the acoustic evidence around this time step, how likely is this frame to belong to a noise event?

---

# 9. Classification Head

Start simple.

For each temporal token:

```text
Transformer output
        ↓
LayerNorm
        ↓
Linear
        ↓
class logits
```

Use multi-label classification because multiple noise events can overlap.

If the competition submission requires only event timestamps, the internal model can still predict seven classes and use those predictions during event decoding.

If the official submission requires fine-grained tags, adapt the output space after inspecting the actual annotation/submission specification.

---

# 10. Recommended Initial Output

Use:

```text
[B, T, C]
```

where:

- T = temporal tokens
- C = seven noise categories

Each output is:

```text
P(class c is active at time t)
```

This is preferable to forcing one class per frame because events can overlap.

---

# 11. Loss Function

## Gold/Silver

For timestamped data, use frame-level multi-label binary classification.

Initial loss:

```text
Weighted BCEWithLogitsLoss
```

or focal BCE if class imbalance is severe.

Generate frame targets by converting:

```text
onset → offset
```

into active temporal regions.

Do not use a hard one-class-per-frame target.

---

# 12. Handling Boundary Precision

The competition evaluates temporal localization, so boundary quality matters.

Do not rely only on coarse frame classification.

Potential improvements:

### Option A — frame probability

Predict:

```text
P(noise | t)
```

### Option B — boundary heads

Add separate outputs:

```text
P(event starts at t)
P(event ends at t)
```

Then:

```text
classification + onset + offset
```

This is a strong second-stage experiment.

### Option C — center/extent representation

Predict:

```text
event center
event duration
```

Only investigate this after the simple frame-based model is working.

---

# 13. Gold/Silver/Bronze Training Strategy

This should be a major part of the project.

## Stage 1 — Gold baseline

Train only on Gold.

Purpose:

- establish a clean reference
- validate data pipeline
- validate target generation
- validate evaluation
- establish reproducibility

Do not skip this.

---

## Stage 2 — Gold + Silver

Train using both timestamped tiers.

Treat Silver labels as noisier.

Possible approaches:

### Simple

Same loss, lower Silver loss weight.

Example:

```text
Gold weight   = 1.0
Silver weight = 0.3–0.7
```

Tune this.

### Better

Use different confidence/quality weights per sample/event.

---

## Stage 3 — Bronze weak supervision

Bronze has noise tags but no timestamps.

Do NOT fabricate timestamps as ground truth.

Possible approaches:

### Weak clip-level loss

For a Bronze clip with class c:

```text
clip_probability(c) = max_t frame_probability(t, c)
```

Then train the clip-level prediction against the Bronze tag.

Use a smooth aggregation such as log-sum-exp or attention pooling instead of a hard max if useful.

Example:

```text
frame logits
    ↓
attention / MIL pooling
    ↓
clip-level logits
    ↓
Bronze BCE loss
```

Total objective can become:

```text
L =
  λ_gold   L_gold_temporal
+ λ_silver L_silver_temporal
+ λ_bronze L_bronze_clip
```

Tune the weights.

---

# 14. Multi-Task Model

A potentially strong architecture is:

```text
                    ┌→ frame classification
CNN → Transformer ──┼→ onset probability
                    ├→ offset probability
                    └→ clip-level weak classifier
```

Start with frame classification only.

Add the other heads incrementally.

Every added component must be justified by validation results.

---

# 15. Data Augmentation

Implement configurable augmentation.

Start with:

- random gain
- small time shift
- SpecAugment
- frequency masking
- time masking

Then investigate:

- time stretching
- pitch shifting
- background mixing
- noise mixing
- random cropping

Be careful with augmentation that changes event timestamps.

Any time-domain transformation must transform annotations consistently.

Avoid synthetic augmentation that creates unrealistic acoustic distributions unless experiments show improvement.

---

# 16. Chunking Long Audio

Do not feed arbitrarily long recordings directly into the Transformer.

Use training windows, for example:

```text
window = 5–10 seconds
```

with overlap.

Example:

```text
audio:
0 -------- 10 -------- 20 -------- 30 sec

windows:
0 ------ 8
4 ------ 12
8 ------ 16
12 ----- 20
...
```

Convert annotations into local window coordinates.

At inference:

1. run overlapping windows
2. convert predictions back to global timestamps
3. merge predictions
4. perform event decoding

---

# 17. Important: Avoid Boundary Artifacts

Because events can cross chunk boundaries:

```text
Window A: |-------------|
Window B:       |-------------|
                     ↑
              event crosses boundary
```

overlapping windows are strongly preferred.

At inference, average or otherwise aggregate overlapping frame probabilities.

Do not simply concatenate independent window predictions.

---

# 18. Event Decoding

The neural network produces frame probabilities.

The competition needs events.

Pipeline:

```text
frame probabilities
        ↓
class-specific threshold
        ↓
optional temporal smoothing
        ↓
active/inactive sequence
        ↓
contiguous regions
        ↓
minimum duration filtering
        ↓
onset/offset refinement
        ↓
event list
```

Tune thresholds on the validation set.

Do NOT assume threshold = 0.5 is optimal.

Tune thresholds separately per class if useful.

---

# 19. Post-Processing Experiments

Create configurable post-processing.

Parameters:

- probability threshold
- onset threshold
- offset threshold
- median-filter length
- minimum event duration
- minimum gap for merging
- maximum allowed gap
- class-specific thresholds

Example:

```text
event A: 1.2–2.0
event A: 2.04–2.8
```

If the 40 ms gap is likely to be a model artifact, merge them.

But don't over-smooth, because short human noises are real events.

---

# 20. Evaluation

Implement a local evaluation pipeline as early as possible.

It must reproduce the competition metrics as closely as possible:

### Metric 1

Event-based F1.

### Metric 2

Segment Dice.

### Metric 3

Combined:

```text
combined = event_f1 + segment_dice
```

Track:

```text
event_f1
segment_dice
combined
```

for every experiment.

Also report per-class metrics.

---

# 21. Validation Split

Avoid random leakage.

Because this dataset contains many speakers and geographically diverse recordings, investigate grouping by:

- speaker
- recording/source
- district
- original audio/session

At minimum, make sure the same source/speaker does not accidentally appear across train and validation if the metadata allows identifying it.

A random segment split can make validation look artificially good.

Create:

```text
train
validation
```

with reproducible split generation.

Save the split IDs to disk.

Never regenerate a different split accidentally between experiments.

---

# 22. Experiment Tracking

Every experiment must record:

```text
experiment_id
git_commit
dataset_version
split_version
model
CNN configuration
Transformer configuration
sample rate
mel configuration
token resolution
window length
augmentation
loss
Gold weight
Silver weight
Bronze weight
learning rate
batch size
epochs
thresholds
event post-processing
Event F1
Segment Dice
Combined
```

Use a simple CSV/JSON log initially or an experiment tracking library if already available.

---

# 23. Baseline Ladder

Implement experiments in this order.

## Baseline 0

Majority/random sanity checks.

Purpose: verify evaluation.

## Baseline 1

Log-mel → CNN → frame classifier.

Purpose: simple temporal baseline.

## Baseline 2

Log-mel → CNN → Transformer → frame classifier.

Purpose: test whether attention improves temporal modeling.

## Baseline 3

CNN → Transformer with Gold only.

## Baseline 4

CNN → Transformer with Gold + Silver.

## Baseline 5

Gold + Silver + Bronze weak supervision.

## Baseline 6

Add class-specific thresholds/post-processing.

## Baseline 7

Add onset/offset prediction heads.

## Baseline 8

Investigate stronger pretrained audio representations.

Do not jump directly to the most complicated architecture.

---

# 24. Pretrained Models

After the custom model works, investigate pretrained audio encoders.

Candidates may include:

- wav2vec 2.0
- HuBERT
- WavLM
- BEATs
- AST
- PANNs
- other publicly available audio encoders

The competition allows public models, but verify licenses and document all external components.

A useful comparison:

```text
Custom CNN + Transformer
vs
Pretrained encoder + Transformer
```

Do not assume a larger pretrained model wins.

---

# 25. Class Imbalance

The dataset is highly imbalanced.

In particular, appliance/machine events are much rarer than non-speech human and animal events.

Investigate:

- BCE positive weighting
- focal loss
- class-balanced sampling
- oversampling rare classes
- class-specific thresholds

Do not overfit rare classes at the expense of overall Combined score.

Always compare per-class performance.

---

# 26. Rare and Short Events

Short events such as:

- cough
- lip smacking
- brief signals
- short human sounds

can be destroyed by excessive temporal smoothing/downsampling.

Therefore:

- preserve sufficient temporal resolution
- avoid excessive pooling
- test minimum-event-duration filtering carefully
- measure short-event recall separately

A model with excellent overall F1 but poor short-event recall may lose Segment Dice or event matching performance.

---

# 27. Overlapping Events

The model must support:

```text
P(animal, t)
P(traffic, t)
P(human, t)
...
```

independently.

Do NOT use softmax across the seven classes unless the competition data proves that exactly one class can be active at any instant.

Use sigmoid outputs for multi-label prediction.

---

# 28. Compute Strategy

First establish a small model that fits comfortably on available GPU memory.

Recommended development approach:

```text
small dataset subset
        ↓
pipeline correctness
        ↓
small model
        ↓
full Gold
        ↓
Gold + Silver
        ↓
full training
```

Do not waste hours training a broken pipeline.

Every training run should support:

- checkpoint saving
- resume
- mixed precision
- deterministic/reproducible seed
- validation during training
- best-checkpoint selection

---

# 29. Reproducibility

Project structure should look approximately like:

```text
project/
├── README.md
├── PLAN.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── gold.yaml
│   └── full.yaml
├── data/
│   └── README.md
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   ├── decoding/
│   └── submission/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── make_submission.py
├── notebooks/
├── experiments/
└── docs/
    ├── data_spec.md
    ├── submission_spec.md
    └── experiments.md
```

Keep paths configurable.

Do not hard-code local filesystem paths.

---

# 30. Unit Tests

Before full training, test:

### Audio

- loading
- resampling
- mono conversion
- spectrogram dimensions

### Annotation

- onset/offset parsing
- clipping to windows
- overlapping events
- events crossing window boundaries

### Targets

- frame target generation
- multi-label targets
- Bronze clip-level target generation

### Decoder

- probability → events
- merging
- filtering
- timestamp conversion

### Submission

- exact JSONL schema
- one prediction per required test item
- valid JSON
- valid timestamps
- sorted events if required

Create synthetic examples where the correct output is known exactly.

---

# 31. Submission Validation

Before uploading to Codabench, run:

```text
python scripts/make_submission.py
python scripts/validate_submission.py
```

The validator should check:

- valid JSONL
- required fields
- valid audio IDs
- no NaNs
- onset >= 0
- offset > onset
- offset <= audio duration
- no malformed events
- correct number of prediction records
- correct ordering if required

The validator must be based on the actual Codabench sample format.

---

# 32. Important Experimental Questions

The coding model should explicitly answer these through experiments.

### Q1

Does 100 ms token resolution outperform 50/200 ms?

### Q2

Does Transformer attention improve over CNN-only temporal modeling?

### Q3

How much does Silver improve over Gold-only?

### Q4

Does Bronze weak supervision improve validation performance?

### Q5

What Gold/Silver/Bronze loss weights work best?

### Q6

Do class-specific thresholds improve Combined score?

### Q7

Does onset/offset prediction improve boundary quality?

### Q8

Does pretrained audio representation beat the custom CNN?

### Q9

What context length is optimal?

### Q10

Which post-processing parameters maximize Combined score?

---

# 33. Attention: What It Is Doing

The correct terminology is:

- **attention**
- more specifically **self-attention**
- the Transformer uses **multi-head self-attention**

For temporal tokens:

```text
x1 x2 x3 x4 x5 ... xT
```

self-attention lets the representation at `x_t` use information from other time steps.

This is useful because noise events have temporal structure.

For example, a weak frame in the middle of a bark may be difficult to classify alone, but surrounding frames can provide strong evidence.

---

# 34. Recommended First Model

Start with:

```text
Sample rate:       16 kHz
Mel bins:          80
STFT window:       25 ms
STFT hop:          10 ms

CNN:
  3–4 convolution blocks
  temporal downsampling to ~100 ms tokens
  embedding dimension = 256

Transformer:
  4 encoder layers
  4–8 attention heads
  FFN dimension = 512–1024
  dropout = 0.1

Head:
  Linear(256 → 7)
  sigmoid

Training:
  Gold first
  weighted BCE
  AdamW
  mixed precision
  early stopping/checkpointing
```

These are starting points, not fixed requirements.

---

# 35. Suggested Training Sequence

Follow this exact order initially.

```text
STEP 1
Inspect data + submission schema

STEP 2
Build dataset loader

STEP 3
Build annotation → frame target converter

STEP 4
Build local evaluation

STEP 5
Train CNN-only Gold baseline

STEP 6
Add Transformer

STEP 7
Tune 100 ms tokenization

STEP 8
Add Gold + Silver

STEP 9
Add Bronze weak supervision

STEP 10
Tune class-specific thresholds

STEP 11
Improve event decoding

STEP 12
Add onset/offset heads

STEP 13
Try pretrained audio encoder

STEP 14
Select best model by Combined score

STEP 15
Train final model(s)

STEP 16
Run test inference

STEP 17
Validate `predictions.jsonl`

STEP 18
Submit to Codabench

STEP 19
Record leaderboard result

STEP 20
Iterate based on metric behavior
```

---

# 36. Ensemble — Only Later

If multiple strong models exist, consider ensembling their frame probabilities.

For example:

```text
Model A: CNN + Transformer
Model B: pretrained encoder + Transformer
Model C: different temporal resolution

          ↓
average / weighted-average frame probabilities
          ↓
single decoder
```

Do not ensemble weak models just because they are different.

First establish that each component independently improves validation performance.

---

# 37. Final Competition Strategy

The final system should optimize three things:

```text
80% leaderboard performance
15% novelty
 5% report
```

Therefore maintain two tracks of work:

### Engineering track

Maximize:

- Event F1
- Segment Dice
- Combined score
- robustness
- inference reliability

### Research track

Investigate one genuinely defensible contribution, such as:

- hierarchical supervision across Gold/Silver/Bronze
- weakly supervised temporal localization
- boundary-aware loss
- multi-task onset/offset prediction
- efficient temporal attention
- class-imbalance-aware detection
- novel fusion of pretrained audio representations with weak supervision

The final report should clearly explain:

1. Problem
2. Data
3. Method
4. Training
5. Ablations
6. Results
7. Novelty
8. Limitations
9. Reproducibility

---

# 38. Guardrails for the Coding Model

1. **Never invent the Codabench submission format.**
2. Inspect the actual sample submission first.
3. Never silently change annotation semantics.
4. Preserve overlapping events.
5. Never assume one class per frame.
6. Never leak validation/test data into training.
7. Never use test labels.
8. Never tune thresholds on the hidden test set.
9. Keep every experiment reproducible.
10. Save configurations with checkpoints.
11. Make every major hyperparameter configurable.
12. Prefer a working simple baseline before adding complexity.
13. Measure every architectural change against the same validation split.
14. Report Event F1, Segment Dice, and Combined for every experiment.
15. Keep an experiment log.
16. Do not delete failed experiments/results.
17. Verify external model/data licenses.
18. Validate the final JSONL locally before submission.

---

# 39. Definition of Done

The project is considered ready for competition submission when:

- [ ] Dataset loader works
- [ ] Gold/Silver/Bronze annotations are correctly parsed
- [ ] Windowing works
- [ ] Overlapping events are preserved
- [ ] CNN baseline works
- [ ] CNN + Transformer works
- [ ] Local Event F1 works
- [ ] Local Segment Dice works
- [ ] Combined score is tracked
- [ ] Gold + Silver training works
- [ ] Bronze weak supervision has been evaluated
- [ ] Threshold tuning works
- [ ] Event decoding works
- [ ] Test inference works
- [ ] Exact Codabench JSONL format is implemented
- [ ] Submission validator passes
- [ ] Final model checkpoint is saved
- [ ] Configuration is saved
- [ ] Experiment results are documented
- [ ] Code can reproduce the final result

---

# 40. Immediate First Task for the Coding Model

Do NOT start by implementing the Transformer.

First:

```text
1. Inspect the downloaded Track 1 data.
2. Inspect all annotation files.
3. Inspect the Codabench sample submission.
4. Produce docs/data_spec.md.
5. Produce docs/submission_spec.md.
6. Build a minimal dataset loader.
7. Visualize 10–20 annotated examples.
8. Build and verify frame-target generation.
9. Implement local evaluation.
10. Only then implement CNN → Transformer.
```

The first milestone is **a correct data/evaluation pipeline**, not a high-performing model.
