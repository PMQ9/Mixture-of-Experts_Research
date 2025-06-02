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
- [ ] Add CutMix
- [ ] Add Attention mechanism for Router improvement
- [ ] Add RandAugment
- [ ] Add Label Smoothing 
- [ ] Add Warmup
- [ ] Gradient clipping
- [ ] Add DEBUG mode

# Performance

| Creteria                | Result  | Note   |
|-------------------------|---------|--------|
| Best training accuracy  | 0.7056  |        |
| Best testing accuracy   | 0.7543  |        |
| Best training loss      | 0.7975  |        |
| Best testing loss       | 0.7461  |        |
| Train balance loss      | 1.0004  |        |
| Test balance loss       | 1.0004  |        |

<img src="utils/doc/training_metrics.png" alt="Alt Text" width="75%"/>

# GitLab CI/CD DevOps Pipeline
*Why do you a CI/CD pipeline for this? -> Yes👍*

<img src="utils/doc/cicd_pipeline.png" alt="Alt Text" width="75%"/>

One of the benefit of a CI/CD pipeline is freeing up your machine from building/testing/compiling. You can make changes on your slim and light laptop, push changes to be compiled/built/test on your server or more powerful home PC and don't have to worry about lugging around a clunky and power hungry workstation.
