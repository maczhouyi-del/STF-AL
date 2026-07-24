# STF-AL: Spatiotemporal-Frequency Feature Representation and Adversarial Learning for Sensor-based Cross-domain Human Activity Recognition

PyTorch implementation of **STF-AL**, a spatio-temporal-frequency adversarial learning framework for sensor-based cross-domain human activity recognition.

The repository supports five model variants:

| Model | Domain adversarial learning | SWDTA | FDFENet | FDSM |
|---|---:|---:|---:|---:|
| M0 | ✗ | ✗ | ✗ | ✗ |
| M1 | ✓ | ✗ | ✗ | ✗ |
| M2 | ✓ | ✓ | ✗ | ✗ |
| M3 | ✓ | ✓ | ✓ | ✗ |
| STF-AL | ✓ | ✓ | ✓ | ✓ |

The spectral-component ablations reported with M3 are available as `m3_mag`,
`m3_phase`, and `m3_mlp` (paper labels: M3-Mag, M3-Phase, and M3-MLP).

## Overview

STF-AL combines:

- spatio-temporal feature extraction;
- domain adversarial learning with gradient reversal;
- Sliding Window Dilated Temporal Attention (SWDTA);
- Frequency-Domain Feature Extraction Network (FDFENet);
- frequency-domain similarity matrix (FDSM).

Supported datasets:

- PAMAP2 / PAMAP
- DSADS
- Opportunity
- TSA

## Environment

```bash
conda create -n stfal python=3.9
conda activate stfal
pip install -r requirements.txt
```

## Dataset Preparation

Place raw datasets under:

```text
data/raw/pamap2/
data/raw/dsads/
data/raw/opportunity/
data/raw/tsa/
```

Dataset-specific paths, labels, sensor columns, window sizes, and subject partitions are configured in:

```text
configs/datasets/pamap2.yaml
configs/datasets/dsads.yaml
configs/datasets/opportunity.yaml
configs/datasets/tsa.yaml
```

Inspect a dataset before training:

```bash
python scripts/prepare_pamap2.py --config configs/datasets/pamap2.yaml
python scripts/prepare_dsads.py --config configs/datasets/dsads.yaml
python scripts/prepare_opportunity.py --config configs/datasets/opportunity.yaml
python scripts/prepare_tsa.py --config configs/datasets/tsa.yaml
```

## Training

The compact training entry uses `configs/default.yaml`.

Activity classification is optimized only with source-domain labels. Unlabeled
target-domain samples participate only in domain alignment during training. The best
checkpoint is selected using source-domain validation performance, and target-domain
labels are read once after training for the final evaluation.

```bash
python -m src.training.train --config configs/default.yaml
```

To change dataset, variant, target partition, or hyperparameters, edit `configs/default.yaml`.

The domain-adversarial variants use a four-layer discriminator with dimensions
`128 -> 384 -> 384 -> 64 -> 1`, ReLU activations, and dropout `0.5`. Training
uses the numerically stable sigmoid-plus-binary-cross-entropy formulation
provided by PyTorch's logits-based loss.

## Evaluation

```bash
python -m src.evaluation.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/pamap2/default/best_checkpoint.pt
```

## Repository Structure

```text
STF/
├── configs/
│   └── datasets/
├── data/
├── scripts/
├── src/
│   ├── datasets/
│   ├── losses/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

Citation information will be updated after publication.

## License

This project is released under the MIT License.
