# Derivational Probing
This repository contains the code for the paper "Derivational Probing: Unveiling the Layer-wise Derivation of Syntactic Structures in Neural Language Models".
This code base builds upon the paper "[A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/)" by Hewitt and Manning (2019) and their [official codebase](https://github.com/john-hewitt/structural-probes).
A part of our evaluation dataset is based on the paper "[Targeted Syntactic Evaluation of Language Models](https://aclanthology.org/D18-1151/)" by Marvin and Linzen (2018) and their [codebase](https://github.com/BeckyMarvin/LM_syneval).

## Setup
We use uv to manage the environment.
To set up the environment:
```bash
uv sync
```

## Experiments
See the `experiments` directory for `.sh` scripts to reimplement our experiments.


## Citation
```
@inproceedings{
someya2025derivational,
title={Derivational Probing: Unveiling the Layer-wise Derivation of Syntactic Structures in Neural Language Models},
author={Taiga Someya and Ryo Yoshida and Hitomi Yanaka and Yohei Oseki},
booktitle={The SIGNLL Conference on Computational Natural Language Learning},
year={2025},
url={https://openreview.net/forum?id=y0RTGA1j5D}
}
```
