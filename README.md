# Mixture-of-Experts_Research
Research MoE application in safety-critical system at Institute of Software Integrated System - Vanderbilt University

# To do list
 **DevOps**
- Python automation test - DONE
- Jenkins/GitLab pipeline - DONE
- Fine tune training params - Not started

 **Performance**
- Add Stiochastic Depth - DONE
- Add L2 Regularization - DONE
- Add CutMix
- Add Attention mechanism for Router improvement
- Add RandAugment
- Add Label Smoothing 
- Add Warmup
- Gradient clipping
- Add DEBUG mode

# GitLab CI/CD DevOps Pipeline
*Why do you a CI/CD pipeline for this? -> Yes👍*

![alt text](utils/images/image.png)

One of the benefit of a CI/CD pipeline is freeing up your machine from building/testing/compiling. You can make changes on your slim and light laptop, push changes to be compiled/built/test on your server or more powerful home PC and don't have to worry about lugging around a clunky and power hungry workstation.
