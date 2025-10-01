# Mixture-of-Experts_Research
Research MoE application in safety-critical system at Institute of Software Integrated System - Vanderbilt University

# To do list
 **DevOps**
- [ ] Add Unit Test
- [ ] Reactivate Gitlab Runner. Use Golang, because

 **Performance**
- [x] Add GNNV

# User Manual

 **Requirements**

- Python 3.10
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- `pip install tqdm matplotlib netron onnx adversarial-robustness-toolbox`

 **Instruction**

- Start training: 
    `python .\src\Vision_Transformer_Pytorch\train_moe.py`
- Argument:
    `python .\src\Vision_Transformer_Pytorch\train_moe.py --batch_size 256 --epochs 500 --config_overrides "img_size=48,patch_size=8,embed_dim=256,num_class=10"`
    - to see all options run `python .\src\Vision_Transformer_Pytorch\train_moe.py --help`

- Calculate normalization value for the dataset:
    `python .\src\Normalization_Value\gtsrb_normalization_compute.py --dataset PTSD`

# Architecture

| Criteria                                  | Value       | Note    |
|-------------------------------------------|-------------|---------|
| Number of experts:                        | 2 or 3         |         |
| Top K (number of experts active per token)| 1 (sparse) or 3 (dense)         |         |
| Parameters                                | 60,530,371 |         |

Architecture:

Block Diagram architecture:

<img src="utils/doc/block_diagram_moe_architecture.png" alt="Alt Text" width="45%"/>

Netron architecture with 2 Experts (open in new tab to view)

<img src="utils/doc/netron_onnx_architecture.jpg" alt="Alt Text" length="50%"/>


# Performance with GTSRB

Full Name: German Traffic Sign Recognition Benchmark

Purpose: Traffic sign recognition for **autonomous driving** and computer vision research

Content: 50,000 images for 43 dfferent traffc sign classes, vary in size and include real-world distortions.

Reference: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |     |                       |
| **Best testing accuracy** | |                       |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |

<img src="utils/doc/training_metrics_gtsrb_cnn_nat.png" alt="Alt Text" width="70%"/>

Download the dataset from: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

- Training Set: GTSRB-Training_fixed.zip

- Test Images: GTSRB_Final_Test_Images.zip

- Test Annotations: GTSRB_Final_Test_GT.zip


# Performance with CIFAR-10

Full Name: Canadian Institute For Advanced Research (CIFAR-10)

Purpose: Standard dataset for evaluating image classification models

Content: 60,000 32×32 color images across 10 classes. Each class has 6,000 images (5,000 training + 1,000 test).

Reference: https://www.cs.toronto.edu/~kriz/cifar.html

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |    |                       |
| **Best testing accuracy** | |                       |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |

<img src="utils/doc/training_metrics_cifar10_cnn_nat.png" alt="Alt Text" width="70%"/>

# Performance with MNIST

Full Name: 

Content: 

Reference: 

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |    |                       |
| **Best testing accuracy** | |                       |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |

<img src="utils/doc/training_metrics_mnist_cnn_nat.png" alt="Alt Text" width="70%"/>

# Performance with Mixture-of-Experts initial training

Initial training with 2 experts: GTSRB and CIFAR10

<img src="utils/doc/performance_of_initial_training.png" alt="Alt Text" width="70%"/>

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |    |                       |
| **Best testing accuracy** | |                       |

# Performance with Mixture-of-Experts fine-tune training

Fine-tine the initial MoE to integrate MNIST expert

<img src="utils/doc/performance_of_fine_tune_training.png" alt="Alt Text" width="70%"/>

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |    |                       |
| **Best testing accuracy** | |                       |

# GitLab CI/CD DevOps Pipeline
*Why do you a CI/CD pipeline for this? -> Yes👍*

<img src="utils/doc/cicd_pipeline.png" alt="Alt Text" width="75%"/>


