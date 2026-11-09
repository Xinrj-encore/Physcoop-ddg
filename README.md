# PhysCoop-DDG

Reference implementation of PhysCoop, a knowledge-guided adaptive higher-order
information routing layer for protein--protein binding free-energy prediction.
The model builds physical interface anchors from cross-chain contacts,
per-residue burial, and mutation context, recruits residues dynamically at each
encoder layer, and exchanges information between the resulting hyperedges with
physics-biased cooperative attention.

This repository contains the authors' PhysCoop code and integration files. It
does not redistribute the upstream H3-DDG source tree. A pinned public checkout
is used only as the runnable ProteinMPNN scaffold and for the pre-existing local
quadruplet operator. All H3, PAH, PhysHG, SCHEF, MHCA, and legacy PhysQuad
branches are excluded.

## Method

For every complex and mutation record, PhysCoop performs the following steps.

1. Build overlapping interface patches from cross-chain contacts and
   `DeltaSASA`. Each patch has an eight-dimensional physical attribute vector.
2. Recompute a patch anchor from the current residue states at every encoder
   layer.
3. Score all valid residues against each anchor and select a dynamic Top-K
   member set. Adaptive capacity optionally varies the member budget per anchor.
4. Aggregate selected residues and fuse the result into the physical anchor
   with a learned gate.
5. Apply multi-head cooperative attention between hyperedges. Pair biases use
   membership overlap, spatial proximity, physical-attribute similarity, and
   mutation activity.
6. Write hyperedge messages back to residues and fuse them into the residue
   representation.
7. Decode wild-type and mutant sequence likelihoods, then apply the
   complex--single-chain thermodynamic cycle to obtain `DeltaDeltaG`.

The optional legacy PhysQuad branch is not part of this release.

## Installation

The code targets Python 3.10+ and PyTorch 2.1+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./tools/install_into_base.sh /path/to/PhysCoop-base
```

The installation script clones or verifies the pinned public base commit,
copies the PhysCoop files into that checkout, and leaves the original project
outside this repository. Run commands from the installed base directory.

Run the focused tests after installation:

```bash
python -m unittest discover -s tests/physcoop -v
```

## Data

Expected files under the installed base directory:

```text
SKEMPI_v2/
  skempi_v2.csv
  PDBs/
weights/
  v_48_020.pt
```

`SKEMPI_v2_cache_phys/` is generated on first use. The PDB identifier embedded
in each SKEMPI row determines which structure is parsed. Splits are grouped by
complex before entries are assigned to folds.

`get_skempi_v2.sh` downloads the public SKEMPI v2 mutation table and PDB archive.
It requires `wget` and `tar`; run it from the directory in which `SKEMPI_v2/`
should be created:

```bash
bash /path/to/PhysCoop-DDG/get_skempi_v2.sh
```

The extracted dataset contains `skempi_v2.csv` and `PDBs/`. SKEMPI v2 is
third-party benchmark data and is not redistributed in this repository.

BindingGYM external evaluation additionally expects a structure-resolved
manifest and structure directory:

```text
data/BindingGYM/normalized/bindinggym_manifest.csv
data/BindingGYM/input/structures/
```

BindingGYM is used only for fixed-model external inference. It is not used for
training, early stopping, hyperparameter tuning, or checkpoint selection.

## Training

Each configuration trains one fold while retaining the same three-fold split.

```bash
python train.py \
  --config-path config/physcoop/physcoop_fold0.json \
  --tag physcoop_topk16
```

Repeat with `physcoop_fold1.json` and `physcoop_fold2.json`. The main setting is
Top-K 16, four attention heads, dropout 0.05, and `DeltaSASA`-augmented physical
attributes. Outputs are written to `results/<timestamp>_<tag>/`.

The final full cross-validation state is stored at
`checkpoint/ddg_model.ckpt`; each fold also has a best-state file and prediction
table. Checkpoints are not bundled because of their size.

## External Evaluation

Run a fixed three-fold ensemble on BindingGYM:

```bash
python scripts/physcoop/run_bindinggym_inference.py \
  --run-dir results/<physcoop-run> \
  --input-dir data/BindingGYM/input \
  --manifest data/BindingGYM/normalized/bindinggym_manifest.csv \
  --output-dir results/bindinggym_physcoop \
  --device cuda
```

Compute assay-level metrics:

```bash
python scripts/physcoop/evaluate_bindinggym.py \
  --predictions results/bindinggym_physcoop/bindinggym_predictions.csv \
  --output-dir results/bindinggym_physcoop/metrics
```

Recruitment traces used for mechanism analysis are generated with:

```bash
python scripts/physcoop/analyze_recruitment.py \
  --run-dir results/<physcoop-run> \
  --fold 0 \
  --device cuda
```

## Repository Layout

```text
physcoop/hypergraph.py      physical patch construction and DeltaSASA
physcoop/layers.py          adaptive membership and cooperative attention
physcoop/model.py           integration with the ProteinMPNN scaffold
physcoop/data.py            SKEMPI adapter and thermodynamic-cycle collation
physcoop/bindinggym_data.py BindingGYM adapter for fixed-model inference
ddg_predictor.py            DeltaDeltaG predictor and thermodynamic cycle
train.py                    reproducible fold training entry point
configs/                    primary Top-K 16 fold configurations
scripts/                    external evaluation and recruitment analysis
tests/                      focused unit tests
```

## License

No license has been declared for this repository yet. The upstream checkout
remains governed by its own repository terms.
