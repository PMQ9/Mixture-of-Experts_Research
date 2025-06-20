# Mixture-of-Experts_Research
Research MoE application in safety-critical system at Institute of Software Integrated System - Vanderbilt University

# To do list
 **DevOps**
- [x] Python automation test
- [x] Jenkins/GitLab pipeline
- [ ] Fine tune training params

 **Performance**
- [x] Add Stiochastic Depth
- [x] Add L2 Regularization
- [x] Add CutMix
- [x] Start testing after N epochs, once every M epochs.
- [x] Add Attention mechanism for Router improvement
- [x] Add RandAugment
- [x] Add Label Smoothing 
- [x] Add Warmup
- [x] Add Args
- [ ] Gradient clipping
- [ ] Add DEBUG mode

 **Inter-Model Performance**
- [x] Add PTSD
- [ ] TBD


# User Manual

 **Requirements**

- Python 3.10
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- `pip install tqdm matplotlib netron onnx`

 **Instruction**

- Start training: 
    `python .\src\Vision_Transformer_Pytorch\train_moe.py`
- Argument:
    `python .\src\Vision_Transformer_Pytorch\train_moe.py --batch_size 256 --epochs 500 --config_overrides "img_size=48,patch_size=8,embed_dim=256,num_class=10"`
- List of most used arguments:
    - `--batch_size`
    - `--epochs`
    - `--learning-rate`
    - `--test_star_epoch`
    - to see all options run `python .\src\Vision_Transformer_Pytorch\train_moe.py --help`

- Calculate normalization value for the dataset:
    `python .\src\Normalization_Value\gtsrb_normalization_compute.py --dataset PTSD`

# Architecture

| Criteria                                  | Value       | Note    |
|-------------------------------------------|-------------|---------|
| Number of experts:                        | 7           |         |
| Top K (number of experts active per token)| 3           |         |
| Number of embedded layers                 | 9           |         |
| Parameters                                | 18.756.846  |         |

Architecture:

<img src="utils/doc/netron_onnx_architecture.jpg" alt="Alt Text" width="25%"/>


# Performance with GTSRB

Full Name: German Traffic Sign Recognition Benchmark

Purpose: Traffic sign recognition for **autonomous driving** and computer vision research

Content: 50,000 images for 43 dfferent traffc sign classes, vary in size and include real-world distortions.

Reference: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    | 0.8408    |                       |
| **Best testing accuracy** |**0.9221** |                       |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |
| Train balance loss        |           |                       |
| Test balance loss         |           |                       |

<img src="utils/doc/training_metrics_gtsrb.png" alt="Alt Text" width="70%"/>

Download the dataset from: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

- Training Set: GTSRB-Training_fixed.zip

- Test Images: GTSRB_Final_Test_Images.zip

- Test Annotations: GTSRB_Final_Test_GT.zip


# Performance with PTSD

Full Name: Persian Traffic Sign Database

Content: 14,000 images for 43 dfferent traffc sign classes, vary in size and include real-world distortions.

Reference: https://www.kaggle.com/datasets/saraparsaseresht/persian-traffic-sign-dataset-ptsd

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    | 0.8546    |                       |
| **Best testing accuracy** |**0.8765** |                       |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |
| Train balance loss        |           |                       |
| Test balance loss         |           |                       |

<img src="utils/doc/training_metrics_ptsd.png" alt="Alt Text" width="70%"/>

## Dataset folder structure: 

### Folder structure
    ./data/
    └── GTSRB/
        ├── Training/
        │   ├── 00000/
        │   │   ├── 00000_00000.ppm
        │   │   ├── 00000_00001.ppm
        │   │   └── GT-00000.csv
        │   ├── 00001/
        │   │   ├── 00001_00000.ppm
        │   │   ├── 00001_00001.ppm
        │   │   └── GT-00001.csv
        │   └── ... (up to 00042)
        ├── Test/
        │   ├── Images/
        │   │   ├── 00000.ppm
        │   │   ├── 00001.ppm
        │   │   └── ...
        │   └── GT-final_test.csv
    └── PTSD/
        ├── Training/
        │   ├── 0_/
        │   │   ├── 00_00001.jpg
        │   │   └── 00_00002.jpg
        │   ├── 1_/
        │   │   ├── 01_00001.jpg
        │   │   └── 01_00002.jpg
        │   └── ... (up to 00042)
        ├── Test/
        │   ├── Images/
        │   │   ├── PTSD_00001.jpg
        │   │   ├── PTSD_00002.jpg
        │   │   └── ...
        │   └── testset_CSV.csv

### Normalization value calculated from the test set:

| Normalization             | GTSRB                 | PTSD                  |
|---------------------------|-----------------------|-----------------------|
| Red mean                  | 0.3432482055626116    | 0.42227414577051153   |
| Green mean                | 0.31312152061376486   | 0.40389899174730964   |
| Blue mean                 | 0.32248030768500435   | 0.42392441068660547   |
| Red standard              | 0.27380229614172485   | 0.2550717671385188    |
| Green standard            | 0.26033050034131744   | 0.2273784047793104    |
| Blue standard             | 0.2660272789537349    | 0.22533597220675006   |

### Modification with PTSD test classes:

Make the following change to the .csv file downloaded for PTSD dataset:

<img src="utils/doc/modify_ptsd_dataset.png" alt="Alt Text" width="15%"/>

Run the following script to modify all PTSD testset files

- `python .\utils\process_ptsd_data\process_ptsd_data.py`

# Performance with CIFAR-10

Full Name: Canadian Institute For Advanced Research (CIFAR-10)

Purpose: Standard dataset for evaluating image classification models

Content: 60,000 32×32 color images across 10 classes. Each class has 6,000 images (5,000 training + 1,000 test).

Reference: https://www.cs.toronto.edu/~kriz/cifar.html

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    | 0.7056    |                       |
| **Best testing accuracy** | **0.7821**|last train: 7e86c261   |
| Best training loss        | 0.7975    |                       |
| Best testing loss         | 0.7461    |                       |
| Train balance loss        | 1.0004    |                       |
| Test balance loss         | 1.0004    |                       |

<img src="utils/doc/training_metrics.png" alt="Alt Text" width="75%"/>

# GitLab CI/CD DevOps Pipeline
*Why do you a CI/CD pipeline for this? -> Yes👍*

<img src="utils/doc/cicd_pipeline.png" alt="Alt Text" width="75%"/>

Benefit of a CI/CD pipeline is freeing up your machine from building/testing/compiling. You can make changes on your slim and light laptop, push changes to be compiled/built/test on your server or more powerful home PC and don't have to worry about lugging around a clunky and power hungry workstation. 

One method to do this is to remotely connect to your PC over the internet. This poses some security risks, since you are exposing your PC to the entire internet to find. Using GitLab Runner or Jenkins Agent is safer (knock on wood).

