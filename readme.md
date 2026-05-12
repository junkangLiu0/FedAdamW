
# FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models 被AAAI 2026录用！！！

<div align="center">

**Mitigating client drift and stabilizing adaptive optimization for federated Transformer training**

<br>

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)]()
[![FedAdamW](https://img.shields.io/badge/Method-FedAdamW-7b61ff.svg)]()
[![Vision](https://img.shields.io/badge/Vision-ViT%20%7C%20Swin-orange.svg)]()
[![Language](https://img.shields.io/badge/Language-RoBERTa%20%7C%20GLUE-brightgreen.svg)]()

<br>

*Federated Learning · Adaptive Optimization · Non-IID Generalization · Large Model Fine-Tuning*

</div>


* 有代码问题 +vx 15653218567 马上回复！帮忙引用论文一下就行！

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！差分隐私，联邦泛化，联邦大模型，联邦优化，联邦大模型微调lora

* 个人主页：https://junkangliu0.github.io/

---
## 🌟 Overview

**FedAdamW** is a federated optimization framework designed for training and fine-tuning large-scale models under decentralized and non-IID data.

While AdamW is a standard optimizer for modern deep learning, directly applying it to federated learning is non-trivial. In heterogeneous federated environments, local AdamW may suffer from:

1. **High variance in second-moment estimates** across clients.
2. **Local overfitting and client drift** under non-IID data.
3. **Inefficient optimizer-state reinitialization** at every communication round.

FedAdamW addresses these challenges by introducing a principled federated AdamW update with:

* **Global update correction** to align local training with the global optimization direction.
* **Decoupled weight decay** to improve local generalization.
* **Second-moment aggregation** to stabilize adaptive updates while preserving communication efficiency.

The framework supports both **vision models** and **language models**, including ResNet, ViT, Swin Transformer, and RoBERTa with LoRA.

---

## ✨ Highlights

* 🚀 **Federated AdamW for large models**
  Designed for modern architectures where SGD-based federated optimizers often converge slowly.

* 🧠 **Transformer-friendly optimization**
  Supports ViT, Swin Transformer, and RoBERTa-style models.

* 📉 **Communication-efficient adaptive statistics**
  Aggregates compact second-moment information instead of communicating full optimizer states.

* 🔒 **Non-IID federated simulation**
  Uses Dirichlet partitioning to simulate realistic heterogeneous client distributions.

* 🧩 **LoRA fine-tuning support**
  Enables efficient federated fine-tuning of large Transformer models.

* 📊 **Reproducible experiment logging**
  Built-in logging with TensorBoard and text logs.

---

## 🧠 Method at a Glance

FedAdamW improves federated AdamW training through three key mechanisms.

### 1. Global-Local Update Alignment

Each client receives an estimated global update direction and injects it into local AdamW training. This reduces the gap between client-local optimization and the global objective.

### 2. Decoupled Weight Decay

FedAdamW follows the AdamW principle of decoupling weight decay from gradient-based adaptive updates, improving stability and generalization in local training.

### 3. Second-Moment Aggregation

Instead of reinitializing the second-moment estimate from zero in every round, FedAdamW aggregates client-side second-moment statistics and reuses them to initialize future local updates.

This improves convergence and reduces the instability caused by heterogeneous client gradients.

---

## 📁 Repository Structure

```text
.
├── main_FedAdamW.py          # Vision federated training: ResNet / ViT / Swin
├── new_llm.py                # Language federated training: RoBERTa + GLUE
├── dirichlet_data.py         # Dirichlet non-IID data partitioning
├── sam.py                    # SAM-related optimizer utilities
├── dataset.py                # Tiny-ImageNet dataset loader
├── model.py                  # Swin Transformer definitions
├── vit_model.py              # ViT model definitions
├── models/                   # ResNet and other vision backbones
├── data/                     # Dataset directory
├── roberta_base/             # Local RoBERTa-base checkpoint directory
├── log/                      # Training logs
└── checkpoint/               # Model checkpoints
```




## Quick Start
## Requirements

* Python 3.8
* PyTorch
* torchvision
* numpy
* matplotlib
* tensorboardX
* ray==1.0.0
* filelock

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

### 2. CNN Training (ResNet-18)
```bash
python  main_FedAdamW.py --alg FedLADA --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedAdamW.py --alg FedAdamW --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
python  main_FedAdamW.py --alg FedAvg_adamw --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
```

* 这里解释一下 --num_gpus_per 0.1的意思是如果你用的是4090显卡24g显存，那么你每个客户端将分配0.1张显卡，即2.4g显存。
* --lr_decay 2 解释一下，这个是余弦学习率下降
* --gpu 0 是指使用的是第0块gpu（gpu序号）
* --alpha_value 0.1 是迪利克雷非立同分布常数
* --alpha_value 1 这个时候是iid情况
* --lora 0 是否使用lora微调，从头训练的情况下，不用lora微调 选0就行
* --normalization BN resnet的归一化层，我选的是BN层，这个效果更好，选择GN也行，收敛的慢
* --data_name timy imagenet数据集需要自己下载，网址在下面
  
### 3. Vision Transformer Training
```bash
python main_FedAdamW.py --alg FedAdamW --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 101 --extname ViTB_LoRA_CIFAR100_Dir01 --lr_decay 2 --gamma 0.85 --CNN VIT-B --E 5 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --pix 224 --lora 1 --r 16 --K 50

```

---
## 联邦大模型微调 vit


```bash
python  main_FedAdamW.py --alg FedAdamW --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10  --rho 0.01 --pix 224 --lora 1 --K 50
python  main_FedAdamW.py --alg FedAvg_adamw --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   VIT-B --E 5 --batch_size 16   --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10  --rho 0.01 --pix 224 --lora 1 --K 50
```

* --lora 1 使用lora微调
* --batch_size 16 显存限制原因，16效果还可以
* --num_gpus_per 0.2 五个客户端，每个客户端使用0.2张卡
* --lr 1e-3 这个学习率微调lora最好

下载模型权重网址：
下载下来的权重直接放主文件夹下面就行，你也可以自己该目类

vit-base：
https://huggingface.co/Junkang2/vit/tree/main

swin_transformer 
https://huggingface.co/Junkang2/swin_transformer/tree/main

## Dataset

数据集下载网址

Tiny-ImageNet：
https://huggingface.co/datasets/Junkang2/Tiny-ImageNet/upload/main

The code supports multiple datasets:

* **CIFAR-10 / CIFAR-100**
* **Tiny-ImageNet**

## 🤖 **大语言模型训练示例（RoBERTa-base + GLUE-SST2）**
```bash
python new_llm.py --alg FedAdamW --lr 2e-4 --data_name sst2 --alpha_value 0.8 --alpha 0.9 --epoch 101 --extname RoBERTa_SST2_Dir08 --lr_decay 2 --gamma 0.9 --CNN roberta_base --E 10 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 5 --K 50 --freeze 1 --r 16 --lora 1 --print 1
```
数据集和模型权重下载地址：
* RoBERTa_base模型权重下载地址，下载完之后放入 roberta_base 文件夹即可。
https://huggingface.co/FacebookAI/roberta-base/tree/main

* 数据集下载地址在hugging face上
  sst2 https://huggingface.co/datasets/SetFit/sst2/tree/main
 全部数据集下载地址：
https://huggingface.co/datasets/Junkang2/glue/tree/main


---

## 🛠️ Installation

### 1. Create Environment

```bash
conda create -n fedadamw python=3.8 -y
conda activate fedadamw
```

### 2. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets peft
pip install ray==1.0.0 tensorboardX tqdm numpy scipy scikit-learn matplotlib filelock
```

Or install from `requirements.txt` if provided:

```bash
pip install -r requirements.txt
```

---

## 📦 Datasets

### Vision Datasets

The code supports:

* `CIFAR10`
* `CIFAR100`
* `imagenet` for Tiny-ImageNet-style experiments

For CIFAR-10 and CIFAR-100, the dataset can be automatically downloaded by `torchvision`.

Expected directory:

```text
./data/
```

For Tiny-ImageNet, place the dataset under:

```text
./data/tiny-imagenet-200/
```

---

### GLUE / NLP Datasets

The language script supports several GLUE-style tasks:

* `sst2`
* `QQP`
* `MRPC`
* `RTE`
* `MNLI`
* `qnli`
* `cola`
* `STS-B`
* `WNLI`

Expected directory examples:

```text
./data/sst2/
./data/QQP/
./data/MRPC/
./data/RTE/
./data/MNLI/
```

RoBERTa-base should be placed locally as:

```text
./roberta_base/
```

---

## 🚀 Quick Start

### 1. ResNet-18 on CIFAR-100

```bash
python main_FedAdamW.py --alg FedAdamW --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname ResNet18_CIFAR100_Dir01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50
```

---

### 2. ViT-B with LoRA on CIFAR-100

```bash
python main_FedAdamW.py --alg FedAdamW --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 101 --extname ViTB_LoRA_CIFAR100_Dir01 --lr_decay 2 --gamma 0.85 --CNN VIT-B --E 5 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10 --rho 0.01 --pix 224 --lora 1 --r 16 --K 50
```

---

### 3. RoBERTa-base with LoRA on SST-2

```bash
python new_llm.py --alg FedAdamW --lr 2e-4 --data_name sst2 --alpha_value 0.8 --alpha 0.9 --epoch 101 --extname RoBERTa_SST2_Dir08 --lr_decay 2 --gamma 0.9 --CNN roberta_base --E 10 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.25 --selection 0.2 --pre 1 --num_workers 20 --preprint 5 --K 50 --freeze 1 --r 16 --lora 1 --print 1
```

---

## 🔬 Reproducing Baselines

To compare FedAdamW with other federated optimizers, simply replace `--alg FedAdamW`.

### Vision Baselines

```bash
python main_FedAdamW.py --alg FedAvg --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --epoch 301 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --num_workers 100 --selection 0.1 --pix 32 --lora 0 --K 50
```

```bash
python main_FedAdamW.py --alg FedAvg_adamw --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --epoch 301 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --num_workers 100 --selection 0.1 --pix 32 --lora 0 --K 50
```

```bash
python main_FedAdamW.py --alg FedLADA --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --epoch 301 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --num_workers 100 --selection 0.1 --pix 32 --lora 0 --K 50
```

```bash
python main_FedAdamW.py --alg SCAFFOLD --lr 3e-4 --data_name CIFAR100 --alpha_value 0.1 --epoch 301 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --num_workers 100 --selection 0.1 --pix 32 --lora 0 --K 50
```

---

### Language Baselines

```bash
python new_llm.py --alg FedAvg --lr 2e-4 --data_name sst2 --alpha_value 0.8 --epoch 101 --CNN roberta_base --E 10 --batch_size 16 --gpu 0 --num_workers 20 --selection 0.2 --K 50 --r 16 --lora 1
```

```bash
python new_llm.py --alg FedLADA --lr 2e-4 --data_name sst2 --alpha_value 0.8 --epoch 101 --CNN roberta_base --E 10 --batch_size 16 --gpu 0 --num_workers 20 --selection 0.2 --K 50 --r 16 --lora 1
```

```bash
python new_llm.py --alg FedAdamW --lr 2e-4 --data_name sst2 --alpha_value 0.8 --epoch 101 --CNN roberta_base --E 10 --batch_size 16 --gpu 0 --num_workers 20 --selection 0.2 --K 50 --r 16 --lora 1
```

---

## 🧪 Supported Algorithms

The codebase includes the following federated optimization methods.

| Algorithm      | Description                                   |
| -------------- | --------------------------------------------- |
| `FedAvg`       | Standard federated averaging                  |
| `FedAvg_adamw` | FedAvg-style training with local AdamW        |
| `FedAvg_adam`  | FedAvg-style training with local Adam         |
| `FedAdam`      | Federated adaptive baseline                   |
| `FedCM`        | Client-momentum based federated optimization  |
| `SCAFFOLD`     | Control-variate based client drift correction |
| `FedLADA`      | Adaptive federated optimizer baseline         |
| `FedAdamW`     | Proposed federated AdamW optimizer            |

Some additional research variants are implemented in the vision script, including SAM/SWA/proximal or momentum-based variants. For paper reproduction, we recommend focusing on the algorithms listed above.

---

## ⚙️ Key Arguments

### Federated Learning

| Argument        | Meaning                                                       |
| --------------- | ------------------------------------------------------------- |
| `--alg`         | Federated algorithm name                                      |
| `--num_workers` | Total number of clients                                       |
| `--selection`   | Client participation ratio per round                          |
| `--epoch`       | Number of communication rounds                                |
| `--E`           | Local epochs per client                                       |
| `--K`           | Maximum number of local update steps                          |
| `--alpha_value` | Dirichlet non-IID parameter; smaller means more heterogeneous |
| `--p`           | Parallel update group size                                    |

---

### Optimization

| Argument      | Meaning                                                |
| ------------- | ------------------------------------------------------ |
| `--lr`        | Local learning rate                                    |
| `--lr_decay`  | Learning-rate decay strategy                           |
| `--gamma`     | Momentum coefficient used by momentum-style methods    |
| `--alpha`     | Global correction / regularization-related coefficient |
| `--rho`       | Perturbation radius for SAM-style variants             |
| `--optimizer` | Local optimizer type, such as `SGD` or `AdamW`         |

---

### Model

| Argument          | Meaning                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `--CNN`           | Backbone name, e.g., `resnet18`, `VIT-B`, `swin_tiny`, `roberta_base`     |
| `--pix`           | Image resolution, e.g., `32` for CIFAR, `224` for pretrained Transformers |
| `--normalization` | `BN` or `GN` for ResNet-style models                                      |
| `--pre`           | Whether to use pretrained weights                                         |
| `--weights`       | Path to pretrained vision checkpoint                                      |

---

### LoRA

| Argument    | Meaning                                    |
| ----------- | ------------------------------------------ |
| `--lora`    | Enable LoRA fine-tuning; `1` means enabled |
| `--r`       | LoRA rank                                  |
| `--AdaLora` | Enable AdaLoRA in the language script      |

---

### Runtime

| Argument         | Meaning                                  |
| ---------------- | ---------------------------------------- |
| `--gpu`          | GPU id, e.g., `0` or `0,1`               |
| `--num_gpus_per` | GPU fraction assigned to each Ray worker |
| `--batch_size`   | Local batch size                         |
| `--preprint`     | Evaluation / logging interval            |
| `--extname`      | Experiment name suffix                   |
| `--print`        | Whether to print detailed information    |

---

## 📊 Logging and Outputs

The code records training progress in both text logs and TensorBoard.

### Text Logs

Logs are saved to:

```text
./log/{alg}-{data_name}-{lr}-{num_workers}-{batch_size}-{E}-{lr_decay}.txt
```

Each log records key information such as:

* communication round
* test accuracy
* train loss
* test loss
* learning rate
* model backbone
* dataset
* non-IID level

---

### TensorBoard

TensorBoard summaries are written through `SummaryWriter`.

To visualize training curves:

```bash
tensorboard --logdir runs
```

---

### Checkpoints

Checkpoints are saved under:

```text
./checkpoint/
```

The checkpoint name includes algorithm, learning rate, experiment name, non-IID setting, and timestamp.

---

## 📈 Reported Results

The paper reports that FedAdamW consistently improves convergence and test accuracy on both vision and language tasks.

| Setting              | Model               | Dataset                   | Highlight                                                       |
| -------------------- | ------------------- | ------------------------- | --------------------------------------------------------------- |
| Vision training      | ResNet-18           | CIFAR-100                 | Strong performance under non-IID client distributions           |
| Vision fine-tuning   | Swin Transformer    | CIFAR-100 / Tiny-ImageNet | Higher accuracy and lower training loss than adaptive baselines |
| Language fine-tuning | RoBERTa-base + LoRA | GLUE                      | Strong average accuracy across multiple tasks                   |

For exact numbers, please refer to the paper tables and keep your released README synchronized with the final camera-ready version.

---

## 🧩 Recommended Experimental Settings

### Highly Non-IID Vision Setting

```bash
--alpha_value 0.1 --num_workers 100 --selection 0.1 --K 50
```

### IID Setting

```bash
--alpha_value 1
```

### RoBERTa Federated LoRA Setting

```bash
--num_workers 20 --selection 0.2 --K 50 --lora 1 --r 16
```

### Memory-Friendly ViT Setting

```bash
--batch_size 16 --num_gpus_per 0.2 --pix 224 --lora 1
```

---

## 🧠 Practical Notes

* `--alpha_value 0.1` indicates a highly non-IID Dirichlet split.
* `--alpha_value 1` corresponds to the IID setting.
* `--selection 0.1` means 10% of clients participate in each round.
* `--num_gpus_per 0.1` means each Ray worker uses 10% of one GPU resource.
* For LoRA fine-tuning, set `--lora 1`.
* For training ResNet from scratch, set `--lora 0`.
* For large Transformer models, reduce `--batch_size` or `--num_gpus_per` if out-of-memory occurs.
* If changing `num_workers`, `alpha_value`, or dataset, regenerate or remove the cached data split file to avoid loading an old partition.

---

## ❗ Troubleshooting

### CUDA out of memory

Reduce one or more of the following:

```bash
--batch_size
--num_gpus_per
--selection
--K
```

For ViT or Swin Transformer, LoRA fine-tuning is recommended:

```bash
--lora 1 --r 16
```

---

### RoBERTa checkpoint not found

Make sure the local model directory exists:

```text
./roberta_base/
```

The directory should contain the HuggingFace RoBERTa-base checkpoint files.

---

### Tiny-ImageNet path error

Make sure the dataset is placed as:

```text
./data/tiny-imagenet-200/
```

---

### Ray initialization error

If Ray reports an existing runtime error, restart the Python process or clean Ray temporary states:

```bash
ray stop
```

Then rerun the experiment.

---

## 📌 Citation

If you find this repository useful, please cite the paper:

```bibtex
@article{fedadamw,
  title   = {FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models},
  author  = {Anonymous},
  journal = {Manuscript},
  year    = {2026}
}
```

The BibTeX entry can be updated after the official publication information is available.

---

## Acknowledgements

This codebase builds upon the broader federated learning and Transformer fine-tuning ecosystem, including PyTorch, Ray, HuggingFace Transformers, PEFT, and TensorBoardX.

---

## License

Please follow the license of this repository and the licenses of all datasets, pretrained models, and third-party libraries used in the experiments.


---

