# Multimodal model, version 1

Frozen ImageNet ResNet-50 -> patch mean per slide -> slide mean per patient ->
clinical fusion -> cosine patient neighbours -> two-layer multihead GAT.

The target is pure reported ductal (0) versus pure reported lobular (1)
histological subtype. Mixed diagnoses, other types, and missing diagnoses are
excluded. This is not molecular subtype or cancer-versus-normal detection.
The fixed 218-patient cohort contains 169 ductal and 49 lobular patients.

Clinical inputs are age (converted from days to years), race, and ethnicity.
Stage and gender are empty in current exports and are excluded. Neither the
diagnosis label nor slide count is an input. Clinical extraction still needs
repair before richer features are used.

An isolated Python 3.9 environment is defined by requirements-model.txt:

```sh
python3 -m venv .venv-model
.venv-model/bin/python -m pip install -r requirements-model.txt
.venv-model/bin/python scripts/legacy_multimodal_gat.py prepare
.venv-model/bin/python scripts/legacy_multimodal_gat.py extract
.venv-model/bin/python scripts/legacy_multimodal_gat.py train
.venv-model/bin/python scripts/legacy_multimodal_gat.py compare
```

Prepare creates a frozen patient split and never overwrites it. Extraction
defaults to a small balanced pilot: up to three available patients per split
and class, four patches per slide. Only slides marked validated in the download
log are used. Pretrained weights are downloaded on the first extraction.
Patch extraction uses level 0 with recorded pixel coordinates and available
microns-per-pixel. This pilot does not harmonize scanner magnification; larger
experiments must choose a physical patch scale and sample more tissue.

The existing adaptive modified Macenko is used with its external demo-slide
reference. It is not standard Macenko. Cache embeddings and per-patient patch
manifests allow subsequent runs to resume. Increasing the patient cap adds
patients; changing patch count or normalization requires a new work directory
and a copy of the original patients.csv to keep splits unchanged.

Clinical encoders and image scaling are fitted on training patients only.
Each normalized modality block has equal weight in cosine graph construction.
Edges point from neighbour to receiving patient; validation/test patients
receive training-neighbour messages and self-loops but cannot send messages
back into the training graph or to other held-out patients. This supports an
inductive experiment. Cosine values are supplied as GAT edge attributes.

Validation loss chooses the checkpoint and early stopping; the test split is
evaluated only after selection. The code saves a GAT state dict, transforms,
training reference features and graph, learning history, predictions, and
metrics. It also evaluates a logistic-regression baseline on the same fused
features. Probabilities are uncalibrated model outputs. Small pilot results
are for checking the pipeline and do not establish clinical performance.

Current scope excludes dashboard, report generation, and attribution outputs.
Patch coordinates are retained for later explanations. Image-only and
clinical-only comparisons, repeated seeded training, and patient cross-validation
are required before claims about multimodal or graph improvements.

The `compare` command runs image-only, clinical-only, and fused logistic
regression baselines with identical training settings and existing splits.
It writes models, transforms, predictions, metrics, the cohort snapshot, and
summary.csv under `comparison/`, preserving the original GAT artifacts.
Saved GAT predictions are included only if held-out IDs and labels match.
It does not tune parameters or use the test results to select a model.

Run `.venv-model/bin/python scripts/tune_patient_gat.py` for a clinical audit
and a small k=3/5/10 graph search. This excludes test patients and their
embeddings entirely, chooses k by minimum validation cross-entropy, and saves
each run under `tuning/` without replacing the original model. Seed 42 is fixed;
this is not a repeated-seed robustness study. The selected configuration is
recorded in `tuning/selection.json`. Existing test results have already been
viewed and should not be described as a fresh confirmatory holdout.

The initial search selected k=10 by validation loss (0.6525), with validation
balanced accuracy 0.5278. k=3 had balanced accuracy 0.5833 but higher loss;
the selection criterion was fixed before the search. No test predictions were
generated for these runs. Clinical audit found 12/68 training race values and
16/68 training ethnicity values missing; ethnicity is nearly constant among
known values. This limits clinical signal but does not establish causation
for the fusion performance gap. Repair extraction before adding richer inputs.

Initial pipeline run completed on 18 patients / 18 slides / 72 patches, with
six patients in each split and no extraction errors. The GAT test balanced
accuracy was 0.50 and ROC-AUC 0.444; it predicted ductal for every test patient
at the fixed 0.5 threshold. The fused logistic baseline reached balanced
accuracy 0.667 on the same six test patients. These results confirm execution,
not useful generalization or a GAT advantage. At this pilot size, k=5 connects
each training patient to all other training patients. Expand the training
cohort before tuning graph sparsity on validation data.

```sh
.venv-model/bin/python -m unittest discover -s scripts -p 'test_multimodal_gat.py'
```

## Active patient graph model

`scripts/patient_graph_model.py` uses bidirectional cross-attention fusion
between 32 image embedding chunks and age/demographic clinical tokens, followed
by two residual GAT layers, per-patient LayerNorm (so held-out batches cannot
alter training representations), and a separate linear classifier. Graph
edges use cosine similarity over an equal-weight concatenation of the
training-fitted, L2-normalized image and clinical feature blocks. This keeps
the topology independent of the randomly initialized attention layer. Nine
fixed configurations are evaluated across seeds 42, 7, and 2026: eight graph
settings (k=3/7, minimum cosine=0/0.25, learning rate=0.0003/0.001) and a
k=0 no-neighbor baseline (learning rate=0.001). Checkpoint/configuration
selection uses validation balanced accuracy, then macro-F1 and lower log loss.
The selected three models form a probability-mean ensemble at a fixed threshold
of 0.5. Test features load only after selection.

```sh
.venv-model/bin/python scripts/patient_graph_model.py
.venv-model/bin/python -m unittest discover -s scripts -p 'test_multimodal_gat*.py'
python3 scripts/report_patient_graph.py
```

The default output is `data/models/patient_graph_cross_attention`; it refuses to overwrite
an existing experiment. Use a fresh `--output` for another experiment. Use
`--validation-only` to omit test data entirely. The plotting command uses the
system environment's Matplotlib/scikit-learn/NumPy; figures are standalone PNGs.
The report is `report.html`, with per-patient predictions, graph edges, metrics,
protocol, checkpoint histories, and the full validation search alongside it.
The selected checkpoint filenames follow the chosen configuration index and
seed. Their architecture configuration is stored in each checkpoint;
clinical/image transforms and reference graph features are stored separately
for reproducible inference.

Each full run also exports `fused_features.csv` with one fused feature row per
patient and `fused_test_vector.csv` with one held-out test patient's vector.
Fusion is bidirectional multihead cross-attention: the 2,048-value ResNet vector
is represented as 32 image tokens and the clinical representation as age and
demographic tokens. The two attention contexts are summed and projected to one
fused vector, which enters the residual GAT. Graph edges come from the fixed,
training-fitted modality features described above; the same initial attention
state initializes every searched model. The graph topology stays fixed while
the learned attention and GAT parameters train.
The `modalities.npz` artifact preserves inputs for independent prediction checks.
Validate the exported attention vectors and recompute the test prediction with:

```sh
.venv-model/bin/python scripts/validate_fused_features.py \
  --run-dir data/models/patient_graph_cross_attention_run2
```

## Fixed-feature graph audit on the expanded slide snapshot

`patient_graph_model.py` now creates graph edges from equal-weight,
training-fitted image and clinical features rather than from the randomly
initialized attention fusion. Its search includes a k=0 no-neighbor baseline.
The 339-slide run is saved under
`data/models/patient_graph_cross_attention_graphfix_run1/`; validation selected
the no-neighbor configuration. Its fixed-split test accuracy was 40.9%,
balanced accuracy 44.1%, and ROC-AUC 0.462.

The five-fold patient-level comparison is saved under
`data/models/patient_graph_cv_5fold_latest_run1/`. With thresholds selected
inside each outer fold, no-graph achieved 46.5% balanced accuracy and 0.452
ROC-AUC; k=7 achieved 47.6% and 0.465. Neither model shows useful
discrimination in this internal analysis. The fixed test set has already been
examined in earlier experiments, so these results are exploratory and do not
count as external confirmation.

Use the matching run directory if you supplied a custom `--output` path.

The cross-attention run is saved under
`data/models/patient_graph_cross_attention_run2/`. In this corrected run,
cross-attention produces the single vector used by the GAT, with no concatenated
image/clinical feature vectors used in fusion or graph construction. It selected
k=3 and learning rate 0.001 by validation balanced accuracy. Validation accuracy
was 58.3% and balanced accuracy 66.7%. The 20-patient test accuracy was 50%,
balanced accuracy 58.2%, and ROC-AUC 0.484. These exploratory results do not
establish useful generalization. The test set has been inspected for this fusion
change; choose future changes using training-only cross-validation and new
held-out patients. The exported attention vectors and one held-out prediction
reproduced successfully.

## GAT depth comparison

Run `.venv-model/bin/python scripts/compare_gat_depth.py` to compare two, three,
and four residual GAT layers. It fixes cross-attention fusion, k=3, minimum
cosine 0, learning rate 0.001, and seeds 42/7/2026, and uses validation only.
Results are saved under `data/models/gat_depth_search/`; use a fresh `--output`
path for another run. All three depths tied on mean validation accuracy (58.3%),
balanced accuracy (66.7%), and macro-F1 (0.571). Three layers had the lowest
mean validation log loss (0.598), compared with 0.609 for two layers and 0.645
for four. By the predeclared tie-break rule, three layers are the validation
candidate, but there was no gain in thresholded classification metrics. Given
the small validation cohort and added complexity, this is not evidence that
deeper GATs generalize better; retain two layers as the conservative default
until training-only cross-validation supports the deeper option.

## Real-patch cross-attention experiment

Run `.venv-model/bin/python scripts/patch_cross_attention_gat.py` to cache
ResNet embeddings for the existing saved patch PNGs and train cross-attention
over those actual patch tokens. The run excludes test patients and is saved at
`data/models/patch_cross_attention_gat_current_slides/`. It used 376 patches
from 68 training and 24 validation patients (4–8 patches per patient). On the
fixed validation split, the three-seed ensemble reached 75.0% accuracy,
66.7% balanced accuracy, 0.667 macro-F1, and 0.648 ROC-AUC. Its accuracy equals
the 75% majority-class rate; it found 3 of 6 lobular cases. Balanced accuracy
matched the patient-vector cross-attention model, so this is a change in the
error trade-off rather than a confirmed improvement. Test data was not used.
The patch embeddings are cached under `data/models/multimodal_gat/patch_embeddings/`.

The saved model validation ensemble accuracy: 83.3%, balanced accuracy: 77.8%. Test accuracy:
65% (13/20), balanced accuracy: 53.3%, ROC-AUC: 0.780. Only one of seven lobular
patients was detected at the fixed threshold; the 70–75% test accuracy target
has not been achieved. The test majority-class accuracy is also 65%. The
accuracy Wilson interval is approximately 43.3–81.9%. Do not select further
changes against these test results; use training-only cross-validation and
collect additional held-out patients for confirmation. Similarity threshold
0.25 did not remove candidate edges in this search, so it is not an established
improvement. Residual/graph contributions require an explicit graph ablation.


Shared clinical preprocessing, feature fusion, cosine-neighbour graph construction, and evaluation metrics are in `scripts/multimodal_features.py`. The previous V1 end-to-end workflow is retained as `scripts/legacy_multimodal_gat.py` for its existing experiments.

## Clinical-only MLP baseline

Run `.venv-model/bin/python scripts/clinical_mlp.py` to train a clinical-only
MLP from the existing frozen patient cohort. It uses age, race, and ethnicity;
preprocessing is fit on training patients only. Three fixed seeds are combined
by averaging validation probabilities, with early stopping based on validation
cross-entropy. The script writes model checkpoints, the fitted preprocessor,
training histories, validation predictions, and `report.json` under
`data/models/clinical_mlp`. It does not use the test split. Choose a fresh
`--output` directory to preserve prior results.
