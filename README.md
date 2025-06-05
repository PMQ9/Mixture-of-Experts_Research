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
- [ ] Add Attention mechanism for Router improvement
- [ ] Add RandAugment
- [ ] Add Label Smoothing 
- [ ] Add Warmup
- [ ] Gradient clipping
- [ ] Add DEBUG mode

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

# Performance with GTSRB

Full Name: German Traffic Sign Recognition Benchmark

Purpose: Traffic sign recognition for **autonomous driving** and computer vision research

Content: 50,000 images for 43 dfferent traffc sign classes, vary in size and include real-world distortions.

Reference: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data

| Criteria                  | Result    | Note                  |
|---------------------------|-----------|-----------------------|
| Best training accuracy    |           |                       |
| **Best testing accuracy** |           |last train:            |
| Best training loss        |           |                       |
| Best testing loss         |           |                       |
| Train balance loss        |           |                       |
| Test balance loss         |           |                       |


# GitLab CI/CD DevOps Pipeline
*Why do you a CI/CD pipeline for this? -> Yes👍*

<img src="utils/doc/cicd_pipeline.png" alt="Alt Text" width="75%"/>

Note: Pipeline training log file can be heavy, use this command to view quickly: ` get-content C:\your-path-here\training_log.txt -tail 10`

One of the benefit of a CI/CD pipeline is freeing up your machine from building/testing/compiling. You can make changes on your slim and light laptop, push changes to be compiled/built/test on your server or more powerful home PC and don't have to worry about lugging around a clunky and power hungry workstation.
