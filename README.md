# SPECTRA: Sparse Structured Text Rationalization
Official implementation of the **EMNLP 2021** paper **SPECTRA: Sparse Structured Text Rationalization**.

*Nuno M Guerreiro* and *André Martins*

**Abstract**: *Selective  rationalization  aims  to  produce  decisions  along  with  rationales  (e.g.,  text  highlights  or  word  alignments  between  two  sentences). Commonly, rationales are modeled as stochastic  binary  masks,  requiring  sampling-based gradient estimators, which complicates training  and  requires  careful  hyperparameter tuning.  Sparse attention mechanisms are a deterministic alternative, but they lack a way to regularize the rationale extraction (e.g., to control the sparsity of a text highlight or the number of alignments).  In this paper, we present a  unified  framework  for  deterministic  extraction of structured explanations via constrained inference on a factor graph, forming a differentiable layer.  Our approach greatly eases training and rationale regularization, generally outperforming  previous  work  on  what  comes  to performance and plausibility of the extracted rationales.  We further provide a comparative study of stochastic and deterministic methods for  rationale  extraction  for  classification  and natural  language  inference  tasks,  jointly  assessing  their  predictive  power,  quality  of  the explanations, and model variability.*

----------

**If you use this code in your work, please cite our paper.**

----------

## Resources

- [Paper](https://arxiv.org/abs/2109.04552) (arXiv)

All material is made available under the MIT license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.


## Python requirements and installation

This code was tested on `Python 3.8.2`. To install, follow these steps:

1. In a virtual environment, first install Cython: `pip install cython`
2. Clone the [Eigen](https://gitlab.com/libeigen/eigen.git) repository to the main folder: `git clone git@gitlab.com:libeigen/eigen.git`
3. Clone the [LP-SparseMAP](https://github.com/nunonmg/lp-sparsemap) fork repository to main folder, and follow the installation instructions found there
   - Follow this fix in case of compilation errors: https://github.com/deep-spin/lp-sparsemap/issues/9
4. Install PyTorch: `pip install torch` (we used version 1.6.0)
5. Install the requirements: `pip install -r requirements.txt`
6. Install the `spectra-rationalization` package: `pip install .` (or in editable mode if you want to make changes: `pip install -e .`)

---
## Datasets

We have used [Hugging Face Datasets](https://github.com/huggingface/datasets) to get the data for our models. All data modules can be found in `rationalizers/data_modules`. If you wish to use a dataset that you cannot find on Datasets, please refer to `rationalizers/custom_hf_datasets` and follow the examples there. The data is downloaded automatically when you start training and will remain cached.

---
## Running

### Training

To train a model you need to define a `.yaml` config. We have made available several of them in `/configs/`.<sup>[1](#myfootnote1)</sup> This config will include all relevant hyperparameters for that run. Below, we will show some examples:

Train **SPECTRA** on *AgNews*:
```bash
python3 -W ignore rationalizers train --config configs/agnews/agnews_spectra
```

Train **HardKuma** on *IMDB*:
```bash
python3 -W ignore rationalizers train --config configs/imdb/imdb_hardkuma
```
---

### Testing

To test a model, you can use the same `.yaml` config you used for training. After training, the test set is ran automatically. However, if you want to run the test loop afterwards, you may run:

Test **SPECTRA** on *AgNews*:
```bash
python3 -W ignore rationalizers predict --config configs/agnews/agnews_spectra --ckpt {ckpt_path}
```
---
### Resume Training

If for some reason you want to resume training from a given checkpoint, you will need to change your `.yaml` config so as to include `resume` args (you may copy the `predict` ones). Then, you may run:

Resume **SPECTRA** on *AgNews*:
```bash
python3 -W ignore rationalizers predict --config configs/agnews/agnews_spectra --ckpt {ckpt_path}
```
---

## Acknowledgments

We want to thank Marcos Treviso for helping starting this codebase. We are also very grateful to Vlad Niculae for all the help he provided regarding the use of the LP-SparseMAP library. We also thank Jasmijn Bastings because the code in this repository was inspired by the structure and implementations in [Interpretable Predictions](https://github.com/bastings/interpretable_predictions).

---

<a name="myfootnote1">1</a>: Soon, we will add the fusedmax strategy to the repo.
