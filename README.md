# scikit-obliquetree

<div align="center">

[![Build status](https://github.com/zhenlingcn/scikit-obliquetree/workflows/build/badge.svg?branch=master&event=push)](https://github.com/zhenlingcn/scikit-obliquetree/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/scikit-obliquetree.svg)](https://pypi.org/project/scikit-obliquetree/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/zhenlingcn/scikit-obliquetree/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/zhenlingcn/scikit-obliquetree/releases)
[![License](https://img.shields.io/github/license/zhenlingcn/scikit-obliquetree)](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/LICENSE)

Oblique Decision Tree in Python

</div>

## Introduction

The oblique decision tree is a popular choice in the machine learning domain for improving the performance of traditional decision tree algorithms. In contrast to the traditional decision tree, which uses an axis-parallel split point to determine whether a data point should be assigned to the left or right branch of a decision tree, the oblique decision tree uses a hyper-plane based on all data point features. 

Numerous works in the machine learning domain have shown that oblique decision trees can achieve exceptional performance in a wide range of domains. However, there is still a lack of a package that has implemented oblique decision tree algorithms, which stymies the development of this domain. As a result, the goal of this project is to solve this problem by implementing some well-known algorithms in this domain. We hope that by doing so, these algorithms will serve as a baseline for machine learning practitioners to compare newly designed algorithms to existing algorithms.


## 🚀 Features
* A simple scikit-learn interface for oblique decision tree algorithms
* A general gradient boosting estimator that can be used to improve arbitrary base estimators

## Installation

```bash
pip install -U scikit-obliquetree
```

or install with `Poetry`

```bash
poetry add scikit-obliquetree
```

Then you can run

```bash
scikit-obliquetree --help
```

```bash
scikit-obliquetree --name Roman
```

or if installed with `Poetry`:

```bash
poetry run scikit-obliquetree --help
```

```bash
poetry run scikit-obliquetree --name Roman
```

## Example
Example of usage:
```python
from sklearn.datasets import load_boston
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

from scikit_obliquetree.HHCART import HouseHolderCART
from scikit_obliquetree.segmentor import MSE, MeanSegmentor

X, y = load_boston(return_X_y=True)
reg = BaggingRegressor(
    HouseHolderCART(MSE(), MeanSegmentor(), max_depth=3),
    n_estimators=100,
    n_jobs=-1,
)
print('CV Score', cross_val_score(reg, X, y))
```

## 🛡 License

[![License](https://img.shields.io/github/license/zhenlingcn/scikit-obliquetree)](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/LICENSE)

This project is licensed under the terms of the `Apache Software License 2.0` license. See [LICENSE](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/LICENSE) for more details.

## 📃 Citation

```
@misc{scikit-obliquetree,
  author = {ECNU},
  title = {Oblique Decision Tree in Python},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhenlingcn/scikit-obliquetree}}
}
```

## Credits

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
