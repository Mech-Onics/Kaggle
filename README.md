# ENGE707 — Facial Feature Classification (Kaggle)

**Active project:** eight-class cat/dog facial-feature classification using pretrained vision models, with an emphasis on reducing reliance on image-file geometry. This is the README for the **existing `Kaggle` repository**, not a new repository.

**Best public Kaggle score so far: `0.9706` (97.06%)**, from `submission_geometry_adversarial_flip_tta.csv` on **22 October 2026**. The ordinary (no TTA) submission achieved `0.9700` (97.00%). These are *public leaderboard* results, not final hidden-test scores.

> **Source-of-truth note:** This ZIP contains the notebook source versions available in our conversation. Your locally executed notebooks may include later edits (including optional aspect-adversary class weighting). **Do not overwrite the locally executed files or the best `.pt` checkpoint without making a backup first.** The Kaggle 0.9706 result is documented from your reported run and screenshot; it was not reproduced from the bundled notebooks in this container.

> **Report record / evidence log, not report prose.** The competition descriptor requires students to write their own interpretation, discussion, reflection and final report; AI-generated report writing is prohibited. Refer to the notebooks, saved training history, and Kaggle submission records when preparing your own report.

## Project objective and dataset

The task is to classify each input into exactly one of eight provided classes:

`CatEye`, `CatFace`, `CatMouth`, `CatNose`, `DogEye`, `DogFace`, `DogMouth`, `DogNose`.

The official dataset contains **63,660 labelled training images**, with 29,443 test images reported during submission preparation. Only the competition-provided image dataset is used; no additional labelled external datasets are introduced. Publicly available pretrained model weights are allowed under the assessment rules. **Do not upload the private competition image data or labels to a public repository.**

Class distribution in the labelled dataset:

| Class | Images | Training | Validation |
| --- | ---: | ---: | ---: |
| CatEye | 13,884 | 11,108 | 2,776 |
| CatFace | 6,942 | 5,553 | 1,389 |
| CatMouth | 6,942 | 5,554 | 1,388 |
| CatNose | 6,942 | 5,553 | 1,389 |
| DogEye | 11,580 | 9,264 | 2,316 |
| DogFace | 5,790 | 4,632 | 1,158 |
| DogMouth | 5,790 | 4,632 | 1,158 |
| DogNose | 5,790 | 4,632 | 1,158 |
| **Total** | **63,660** | **50,928** | **12,732** |

The original files vary widely in size and aspect ratio. A sample of 1,500 images ranged from **10×10** to **829×783**, with a median of approximately **86×80**. This creates potential unwanted correlations between file geometry and class identity, separate from the geometry of the animal's real features.

## Problem discovered: shortcut learning from metadata-related properties

A separate baseline Random Forest trained using **only original image width, height and width/height aspect ratio** obtained **0.6025 (60.25%)** accuracy on a balanced sampled validation set; the eight-class random baseline is **0.125 (12.5%)**. This establishes that image dimensions provide predictive information in this dataset. It **does not prove** that a particular CNN uses this shortcut, and does not isolate aspect ratio from absolute dimensions.

An image audit found **15 exact-duplicate groups with no conflicting labels**, and 1,747 perceptual-hash candidate pairs (29 cross-label candidates). Candidate similarity is not proof of duplication. No manual near-duplicate review was used as a condition of the active pipeline. A saved 80/20 split and automatic exact-duplicate grouping are retained as validation safeguards; images are **not deleted** simply because they are similarly zoomed, cropped or coloured.

## Active pipeline and design changes

**Working notebooks:**

- `Geometry_Adversarial_Training.ipynb` — exploration, preprocessing, split, training, validation, checkpointing.
- `Test_Submission.ipynb` — checkpoint loading and test inference, ordinary + horizontal-flip TTA submissions.

**Backbone and prediction head**

- **Pretrained torchvision `convnext_tiny`** backbone (`MODEL_NAME='convnext_tiny'`). This is a distinct new trained model, **not** the earlier `convnext_base` experiment and **not** a continuation of the previous variable-blend ensemble weights.
- One **direct eight-class classifier** rather than a sum of separate animal/part logits.
- Two auxiliary geometry adversaries predicting binned **original image-size** and **original aspect ratio** from the pooled backbone representation.
- A **Gradient Reversal Layer (GRL)** reverses/scales geometry gradients flowing into the backbone while geometry heads learn to recover their targets. The raw dimensions are **not** supplied as features to the main classifier.
- The geometry heads are **unused at inference**, where only the eight-class classifier runs.

**Image preparation**

- Convert to **RGB**, not grayscale-only.
- Place the *whole* image on a square canvas using **reflection padding**; vary canvas scale (`1.0–1.28`) and position during training. This preserves anatomy proportions and makes constant image borders a less stable cue.
- Bicubic resize to **224×224**, `ToTensor`, ImageNet-style RGB channel normalisation: mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Label-independent random horizontal flip (`p=0.5`), limited affine transforms (±8°, ±3% translation, ×0.9–1.1 scale with application probability `0.6`), mild brightness/contrast/saturation/hue jitter (application probability `0.6`), and occasional grayscale (`p=0.04`).
- Validation/test preprocessing is **deterministic**: centred reflect-to-square, bicubic 224×224, tensor and normalisation. No training-time randomness at ordinary inference.
- Very small images do **not** regain physically lost texture simply because they are resized.

**Two-view learning and losses**

Each training sample is independently transformed into **two views**. The model is trained with:

1. The average supervised eight-class cross-entropy over both views.
2. A consistency term (`CONSISTENCY_WEIGHT=0.25`) penalising disagreement between their class probability distributions, with a detached average-probability target.
3. Auxiliary size/aspect classification losses from the adversarial heads, with GRL reversing geometry gradients into shared features.

The intended effect is greater prediction stability under changes to presentation rather than pressure to discard genuine anatomical shape. This is a **regularisation strategy**, not proof of geometry independence. A class-weighted aspect-head loss was discussed during development; verify the locally executed notebook before claiming that exact modification was present in the successful run.

**Training configuration (notebook defaults used as the recorded starting configuration)**

| Setting | Value |
| --- | --- |
| Pretrained backbone | `convnext_tiny` |
| Input size | 224×224 RGB |
| Batch size | 4 original images, with 2 views each |
| Gradient accumulation | 4 |
| Optimizer | AdamW |
| Backbone learning rate | `2e-5` |
| Classifier/adversary learning rate | `2e-4` |
| Weight decay | `0.01` |
| Mixed precision | GPU AMP enabled |
| Head-only period | Epoch 1 |
| Backbone fine-tuning | From epoch 2 |
| GRL coefficient | 0 in epoch 1, ramped to `0.12` by epoch 6 |
| Maximum epochs | 16 |
| Early stopping patience | 5 epochs |
| Time budget | 5 hours (training stopped earlier) |
| Random seed | 42 |

**Training run recorded:** 10 epochs completed before early stopping; user-reported duration approximately **129 minutes**. Best validation checkpoint was saved at **epoch 5**.

## Recorded training history

| Epoch | Training CE | Validation accuracy | Validation macro-F1 | GRL coefficient |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4802 | 0.9358 | 0.9294 | 0.000 |
| 2 | 0.0560 | 0.9939 | 0.9943 | 0.024 |
| 3 | 0.0204 | 0.9935 | 0.9945 | 0.048 |
| 4 | 0.0158 | 0.9942 | 0.9946 | 0.072 |
| **5** | **0.0173** | **0.9960** | **0.9963** | **0.096** |
| 6 | 0.0154 | 0.9947 | 0.9951 | 0.120 |
| 7 | 0.0157 | 0.9951 | 0.9953 | 0.120 |
| 8 | 0.0167 | 0.9946 | 0.9951 | 0.120 |
| 9 | 0.0175 | 0.9956 | 0.9956 | 0.120 |
| 10 | 0.0168 | 0.9952 | 0.9954 | 0.120 |

`best_geometry_adversarial.pt` is the checkpoint selected by validation accuracy (with macro-F1 as a tie-breaker); `last_geometry_adversarial.pt` is the end-of-run state. **Use the best checkpoint for submission**.

## Kaggle submission results

| Experiment / saved submission | Public score |
| --- | ---: |
| Earlier original `final_submission.csv` | 0.9231 |
| Earlier augmented `final_submission_augmented.csv` | 0.9444 |
| Earlier augmented TTA `final_submission_augmented_tta.csv` | 0.9463 |
| Earlier wide TTA `final_submission_wide_tta.csv` | 0.9387 |
| Earlier three-model ensemble `final_submission_ensemble_3model.csv` | 0.9410 |
| **Previous best:** `submission_variable_blend_tta.csv` | **0.9490** |
| **Current geometry-adversarial:** `submission_geometry_adversarial.csv` | **0.9700** |
| **Current geometry-adversarial + horizontal-flip TTA:** `submission_geometry_adversarial_flip_tta.csv` | **0.9706** |

**Public leaderboard improvement over previous best:** `0.9706 - 0.9490 = 0.0216`, i.e. **+2.16 percentage points**. It is not possible to attribute all of this improvement to GRL alone because the architecture, training procedure, augmentation and model comparison also differ. The public leaderboard is distinct from the 20% local validation subset, and tuning against it can introduce leaderboard overfitting.

TTA inference averages class **probabilities** from (1) the original image and (2) its horizontal flip before choosing the highest-scoring of the eight classes. The test notebook currently **requires** root-level `sample_submission.csv` for image IDs and column ordering, even though it does not use it for labels. Do **not** delete this CSV unless you first modify the submission notebook.

## Active local file layout (retain existing repository)

```text
Kaggle/
├── README.md
├── .gitignore                   # Original entries preserved; cleanup adds ignore rules
├── gitignore_additions.txt      # Ignore rules merged by cleanup script
├── cleanup_existing_repo.py     # One-time cleanup helper
├── Geometry_Adversarial_Training.ipynb
├── Test_Submission.ipynb
├── train_labels.csv              # Local only; not uploaded to Git
├── sample_submission.csv         # Local only; required by current inference notebook
├── train/train/                 # Kaggle labelled training images; local only
├── test/                        # Kaggle test images; local only
└── outputs/                     # Local only; not uploaded to Git
    ├── preflight_audit/
    │   └── group_assignments.csv
    └── robust_recognition_pipeline/
        ├── best_geometry_adversarial.pt
        ├── last_geometry_adversarial.pt
        ├── geometry_adversarial_history.csv
        ├── train_val_split.csv
        ├── class_to_idx.json
        ├── preprocessing_config.json
        ├── submission_geometry_adversarial.csv
        └── submission_geometry_adversarial_flip_tta.csv
```

The training notebook also caches source image dimensions and sampled statistics in `outputs/robust_recognition_pipeline/`. Keep those if present; they save computation. Only `README.md`, `.gitignore`, the active notebooks and optional cleanup script need to be committed. `gitignore_additions.txt` is an optional migration helper. The model files and Kaggle competition data are intentionally kept out of Git. Back up the **best checkpoint** separately: `.gitignore` does not protect files from local disk failure.

## Reproducibility and next work

1. Use the same approved competition images, `train_labels.csv` and class order as stored in `class_to_idx.json` / the checkpoint.
2. Reuse `outputs/robust_recognition_pipeline/train_val_split.csv` to preserve the validation comparison. Keep `outputs/preflight_audit/group_assignments.csv` for known exact-copy groups.
3. Set the Kaggle folder as the VS Code working directory; run `Geometry_Adversarial_Training.ipynb` with the existing Python environment. **Do not accidentally retrain or overwrite the best checkpoint**.
4. Run `Test_Submission.ipynb` using the best checkpoint, the test images and the Kaggle sample submission template. Ensure all IDs align before uploading.
5. To improve beyond **0.9706**, preserve this checkpoint as a baseline and evaluate changes independently. Possible next experiments include probability-averaged multi-scale TTA, architecture/backbone comparisons and robustness measurements on held-out transformed images. Keep model selection based on validation evidence, not parameter count alone.
6. The competition defines **only eight valid classes**. Human-face / unrecognisable test examples can motivate an internal uncertainty/OOD flag, but an `N/A` submission class is not documented and must not be invented. Do not manually assign hidden-test labels.

## Evidence to retain for your own final report

- `Geometry_Adversarial_Training.ipynb` (executed implementation and exact local modifications).
- `outputs/robust_recognition_pipeline/geometry_adversarial_history.csv` (epoch-level metrics).
- `outputs/robust_recognition_pipeline/train_val_split.csv` (validation reproducibility).
- `outputs/robust_recognition_pipeline/best_geometry_adversarial.pt` (model, class order, stored hyperparameters, epoch).
- `outputs/robust_recognition_pipeline/preprocessing_config.json` and `class_to_idx.json`, if generated.
- Kaggle score screenshots/records and both submitted CSVs.
- Original data quality and geometry-only baseline outputs.

**Limitations to assess yourself, rather than assume resolved:** adversarial suppression may not fully remove geometry cues; augmentation can alter useful features; preprocessing may retain framing signals; very small images contain limited genuine detail; the Kaggle public split may differ from validation/test distributions; class predictions remain mandatory for OOD inputs.
