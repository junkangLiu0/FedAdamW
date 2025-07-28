# FedAdamW
# Federated Learning Framework README

This repository contains two main federated learning scripts: `new_adamw.py` (for CNN-based models) and `new_llm.py` (for transformer-based models). Below is a comprehensive guide for running experiments and understanding all parameters.

---

## Quick Start

### 1. CNN Training (CIFAR-100)
```bash
python new_adamw.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name CIFAR100 \
  --alpha_value 0.1 \
  --alpha 0.01 \
  --epoch 101 \
  --extname FedAvg_adamw_P \
  --lr_decay 2 \
  --gamma 0.5 \
  --CNN swin_tiny \
  --E 5 \
  --batch_size 16 \
  --gpu 2 \
  --p 1 \
  --num_gpus_per 0.2 \
  --normalization BN \
  --selection 0.05 \
  --print 0 \
  --pre 1 \
  --num_workers 100 \
  --preprint 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 224 \
  --lora 0 \
  --K 50
```

### 2. CNN Training (ResNet-18)
```bash
python new_adamw.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name CIFAR100 \
  --alpha_value 0.1 \
  --alpha 0.001 \
  --epoch 301 \
  --extname FedAvg_adamw_P \
  --lr_decay 2 \
  --gamma 0.5 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --gpu 1 \
  --p 1 \
  --num_gpus_per 0.1 \
  --normalization BN \
  --selection 0.1 \
  --print 0 \
  --pre 1 \
  --num_workers 100 \
  --preprint 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 32 \
  --lora 0 \
  --K 50
```

### 3. Vision Transformer Training
```bash
python new_adamw.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name CIFAR100 \
  --alpha_value 0.1 \
  --alpha 0.001 \
  --epoch 301 \
  --extname FedAvg_adamw_P \
  --lr_decay 2 \
  --gamma 0.5 \
  --CNN deit_tiny \
  --E 5 \
  --batch_size 50 \
  --gpu 2 \
  --p 1 \
  --num_gpus_per 0.1 \
  --normalization BN \
  --selection 0.1 \
  --print 0 \
  --pre 1 \
  --num_workers 100 \
  --preprint 10 \
  --beta1 0.9 \
  --beta2 0.999 \
  --rho 0.01 \
  --pix 32 \
  --lora 0 \
  --K 50
```

---

## Parameter Reference

### Core Federated Learning Parameters
| Parameter | Description |
|-----------|-------------|
| `--alg` | Algorithm choice: `FedAvg`, `FedAdamW`, `FedCM`, `SCAFFOLD`, etc. |
| `--lr` | Client learning rate |
| `--lr_decay` | Learning rate decay strategy (1=exponential, 2=cosine annealing) |
| `--gamma` | Momentum parameter for certain algorithms |
| `--alpha` | Weight decay coefficient for AdamW optimizer |

### Data Parameters
| Parameter | Description |
|-----------|-------------|
| `--data_name` | Dataset: `CIFAR10`, `CIFAR100`, `imagenet`, `QQP`, `MNLI`, etc. |
| `--alpha_value` | Dirichlet distribution parameter for non-IID data splitting (0.1=highly non-IID, 1=IID) |
| `--num_workers` | Total number of clients |
| `--selection` | Fraction of clients selected per round (0.1=10%) |

### Model Parameters
| Parameter | Description |
|-----------|-------------|
| `--CNN` | Model architecture: `resnet18`, `swin_tiny`, `deit_tiny`, `roberta_base` |
| `--pre` | Use pretrained weights (1=True, 0=False) |
| `--normalization` | Normalization type: `BN` (BatchNorm) or `GN` (GroupNorm) |
| `--pix` | Input image size (32 for CIFAR, 224 for ImageNet) |

### Training Parameters
| Parameter | Description |
|-----------|-------------|
| `--epoch` | Total communication rounds |
| `--E` | Local epochs per client |
| `--batch_size` | Client batch size |
| `--K` | Maximum local steps per round (overrides E if smaller) |
| `--p` | Parallelism factor for client updates |

### LoRA Parameters
| Parameter | Description |
|-----------|-------------|
| `--lora` | Enable LoRA fine-tuning (1=True, 0=False) |
| `--r` | LoRA rank |
| `--lora_alpha` | LoRA scaling parameter |

### Optimization Parameters
| Parameter | Description |
|-----------|-------------|
| `--beta1` | Adam optimizer β1 parameter |
| `--beta2` | Adam optimizer β2 parameter |
| `--rho` | SAM optimizer perturbation radius |
| `--optimizer` | Base optimizer: `SGD` or `AdamW` |

### System Parameters
| Parameter | Description |
|-----------|-------------|
| `--gpu` | GPU device IDs (e.g., "0,1,2") |
| `--num_gpus_per` | GPU fraction per client (0.2=20% of a GPU) |
| `--print` | Print detailed logs (1=True, 0=False) |
| `--preprint` | Evaluation frequency (in epochs) |

---

## Output Files

- **Logs**: `./log/alg-dataset-lr-workers-batch-epochs-lr_decay.txt`
- **Checkpoints**: `./checkpoint/ckpt-alg-lr-extname-alpha_value-timestamp/`
- **Plots**: `./plot/alg-dataset-...-timestamp.npy` (contains accuracy/loss arrays)
- **Models**: `./model/model-alg-...-timestamp.pth`

---

## Notes

1. **LoRA Usage**: When `--lora 1`, only LoRA parameters are trainable by default
2. **Pretrained Models**: Automatically downloads required pretrained weights
3. **Data Splitting**: Uses Dirichlet distribution for non-IID splits when `--alpha_value < 1`
4. **Memory**: Adjust `--num_gpus_per` based on your GPU memory capacity

For transformer training with GLUE tasks, use `new_llm.py` with appropriate `--data_name` (QQP, MNLI, SST2, etc.).
