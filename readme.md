
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


# 🌌 **联邦学习实验平台 · 中文文档**  
*（支持 CNN & Transformer 双栈训练）*

---

## 📂 一键安装依赖
```bash
# 基础环境
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 联邦学习 & 日志
pip install ray==1.0.0 tensorboardX==2.6.2.2 tqdm==4.67.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Transformer & 数据集
pip install transformers==4.46.3 datasets==3.1.0 peft==0.13.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 科学计算
pip install scikit-learn==1.3.2 scipy==1.9.3 matplotlib==3.7.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 🎯 **CNN 训练示例（CIFAR-100）**
### 1. Swin-Tiny 联邦训练
```bash
python new_adamw.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name CIFAR100 \
  --alpha_value 0.1 \
  --alpha 0.01 \
  --epoch 101 \
  --extname Swin_CIFAR100 \
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
  --pre 1 \
  --num_workers 100 \
  --K 50
```

### 2. ResNet-18 联邦训练
```bash
python new_adamw.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name CIFAR100 \
  --alpha_value 0.1 \
  --alpha 0.001 \
  --epoch 301 \
  --extname ResNet18_CIFAR100 \
  --lr_decay 2 \
  --gamma 0.5 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --gpu 1 \
  --pix 32 \
  --lora 0 \
  --K 50
```

---

## 🤖 **大语言模型训练示例（RoBERTa-base + GLUE-SST2）**
```bash
python new_llm.py \
  --alg FedAdamW \
  --lr 3e-4 \
  --data_name sst2 \
  --alpha_value 0.8 \
  --alpha 0.9 \
  --epoch 101 \
  --extname RoBERTa_SST2 \
  --lr_decay 2 \
  --gamma 0.9 \
  --CNN roberta_base \
  --E 10 \
  --batch_size 32 \
  --gpu 0 \
  --p 1 \
  --num_gpus_per 0.25 \
  --selection 0.2 \
  --pre 1 \
  --num_workers 20 \
  --preprint 5 \
  --K 50 \
  --freeze 1 \
  --beta1 0.9 \
  --beta2 0.999 \
  --r 16 \
  --lora 1 \
  --print 1
```

---

## 🎛️ **参数速查表（中文）**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--alg` | 联邦算法 | `FedAdamW`（Adam优化） / `FedAvg`（SGD优化） |
| `--lr` | 客户端学习率 | `3e-4`（Adam） / `0.1`（SGD） |
| `--data_name` | 数据集 | `CIFAR100` / `sst2` / `qnli` |
| `--alpha_value` | 数据异构度 | `1.0`（IID） / `0.1`（高度非IID） |
| `--CNN` | 模型架构 | `swin_tiny` / `resnet18` / `roberta_base` |
| `--lora` | LoRA微调 | `1`（启用）→显存占用降低90% |
| `--r` | LoRA秩 | `16`（平衡性能与效率） |
| `--gpu` | GPU设备 | `"0"`（单卡） / `"0,1,2"`（多卡） |
| `--epoch` | 通信轮数 | `100`（CNN） / `50`（大模型） |
| `--E` | 本地轮数 | `5`（CNN） / `10`（大模型） |
| `--K` | 本地步数上限 | 覆盖`--E`的步数限制 |
| `--selection` | 每轮参与比例 | `0.1`（10%客户端） |
| `--batch_size` | 本地批次大小 | `16`（GPU显存紧张时） |

---

## 📊 **输出文件结构**
```
实验结果/
├── log/              # 训练日志（txt）
├── plot/             # 训练曲线（npy）
├── model/            # 最终权重（pth）
└── checkpoint/       # 断点续训（ckpt）
```

---

## 💡 **实用技巧**
1. **显存优化**：  
   - CNN训练：`--batch_size 16` + `--num_gpus_per 0.2`  
   - 大模型：`--lora 1` + `--r 8`（显存占用 < 4GB）

2. **数据异构可视化**：  
   修改`--alpha_value`（0.1~1.0）观察精度变化曲线

3. **快速验证**：  
   添加`--print 1`实时查看loss，减少`--epoch`至`20`

---

## 🌈 **支持的完整任务列表**

| 任务类型 | 数据集 | 模型 | 示例命令 |
|----------|--------|------|----------|
| **图像分类** | CIFAR-10/100 | ResNet/Swin/DeiT | 见上方CNN示例 |
| **文本分类** | SST-2（情感分析） | RoBERTa-base | 见上方LLM示例 |
| **自然语言推理** | QNLI/MNLI | RoBERTa-base | 替换`--data_name qnli` |
| **句子对匹配** | MRPC/QQP | RoBERTa-base | 替换`--data_name mrpc` |

---

**🎉 祝实验顺利！有任何问题欢迎提Issue交流~**
