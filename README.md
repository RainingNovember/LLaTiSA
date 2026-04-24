# LLaTiSA

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E.svg)](https://huggingface.co/datasets/November-Rain/HiTSR)
[![arXiv](https://img.shields.io/badge/arXiv-2604.17295-FF6B6B.svg)](https://arxiv.org/abs/2604.17295)


This is the official repository of the ACL 2026 Findings paper: "LLaTiSA: Towards Difficulty-Stratified Time Series Reasoning from Visual Perception to Semantics".

## Overview

<p align="center">
<img src="./Figs/fig_1.png" alt="" align=center />
</p>


## Key Contributions

1. **Difficulty-Stratifed Dataset**: A comprehensive multimodal time series understanding dataset with three levels of complexity;
2. **Multi-modal Reasoning**: Combines visual perception (plots, numeric grids) with natural language instructions for advanced time series reasoning;
3. **Comprehensive Evaluation**: Benchmarks across multiple reasoning tasks and different time series encoding strategies.

<p align="center">
<img src="./Figs/fig_2.png" alt="" align=center />
</p>

## Dataset

- **Hugging Face**: [HiTSR Dataset]
- **Statistics**:
  - Level 1 (Basic): 54,000 training samples
  - Level 2 (Intermediate): 45,632 training samples  
  - Level 3 (Advanced): 3,515 training samples


## Citation

```bibtex
@article{llatisa2026,
  title={LLaTiSA: Towards Difficulty-Stratified Time Series Reasoning from Visual Perception to Semantics},
  author={Yueyang Ding, HaoPeng Zhang, Rui Dai, Yi Wang, Tianyu Zong, Kaikui Liu, Xiangxiang Chu},
  journal={arxiv preprint arxiv: 2604.17295},
  year={2026}
}
