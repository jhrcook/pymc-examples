---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(tumoroscope)=
# Tumoroscope – a generative model for inferring cancer cell clonality in spatial transcriptomics

:::{post} Oct 26, 2022
:tags: generative model, case study 
:category: intermediate, tutorial
:author:  Joshua Cook
:::

```{code-cell} ipython3
import os

from dataclasses import dataclass

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

%matplotlib inline
print(f"Running on PyMC v{pm.__version__}")
```

```{code-cell} ipython3
RANDOM_SEED = 6
az.style.use("arviz-darkgrid")
```

## Introduction

Brief statement about goal of model and what is shown here.
Make sure to cite the original paper using the following – {cite:p}`shafighi2022tumoroscope`

### Motivation

Research question.

### The Tumoroscope model

Describe the model

+++ {"tags": []}

## Tumoroscope Model

### Model definition

Math description of the model.

+++

### PyMC Implementation

```{code-cell} ipython3
@dataclass
class TumoroscopeData:
    """Tumoroscope model input data."""

    K: int  # number of clones
    S: int  # number of spots
    M: int  # number of mutation positions
    F: np.ndarray  # Prevalence of clones from bulk-DNA seq.
    cell_counts: np.ndarray  # Number of cell counted per spot
    C: np.ndarray  # Zygosity per position and clone
    D_obs: np.ndarray | None  # Read count per position per spot
    A_obs: np.ndarray | None  # Alternated reads per position per spot
    zeta_s: float = 1.0  # Pi hyper-parameter
    F_0: float = 1.0  # "pseudo-frequency" for lower bound on clone proportion
    l: float = 100  # Scaling factor to discretize F
    r: float = 0.1  # shape parameter for Gamma over Phi
    p: float = 1.0  # rate parameter for Gamma over Phi


def _prefixed_index(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in np.arange(n)]


def _make_tumoroscope_model_coords(data: TumoroscopeData) -> dict[str, list[str]]:
    coords = {
        "clone": _prefixed_index(data.K, "c"),
        "spot": _prefixed_index(data.S, "s"),
        "position": _prefixed_index(data.M, "p"),
    }
    return coords


def build_tumoroscope_model(data: TumoroscopeData, fixed: bool = False) -> pm.Model:
    """Build the 'Tumoroscope' model.
    Args:
        data (TumoroscopeData): Input data.
        fixed (bool, optional): Whether to use the "fixed" version of the model
        where the number of cells is assumed accurate. If `False` (default), the
        provided number of cells per spot is used for the prior over of random
        variable.
    Returns:
        pm.Model: PyMC model.
    """
    coords = _make_tumoroscope_model_coords(data)
    with pm.Model(coords=coords) as model:
        zeta_s = pm.ConstantData("zeta_s", data.zeta_s)
        ell = pm.ConstantData("ell", data.l)
        F_0 = pm.ConstantData("F0", data.F_0)
        F = pm.ConstantData("F", data.F, dims="clone")
        if not fixed:
            Lambda = pm.ConstantData("Lambda", data.cell_counts, dims="spot")
        r = pm.ConstantData("r", data.r)
        p = pm.ConstantData("p", data.p)
        C = pm.ConstantData("C", data.C, dims=("position", "clone"))

        F_prime = pm.Deterministic("F_prime", ell * at.ceil(20 * F) / 20, dims="clone")

        Pi = pm.Beta("Pi", alpha=zeta_s / data.K, beta=1, dims=("spot", "clone"))
        Z = pm.Bernoulli("Z", p=Pi, dims=("spot", "clone"))
        G = pm.Gamma("G", (F_prime[None, :] ** Z) * (F_0 ** (1 - Z)), 1, dims=("spot", "clone"))
        H = pm.Deterministic("H", G / G.sum(axis=1)[:, None], dims=("spot", "clone"))

        if fixed:
            N = pm.ConstantData("N", data.cell_counts, dims="spot")
        else:
            N = pm.Poisson("N", Lambda, dims="spot")
        Phi = pm.Gamma("Phi", r, p, dims=("position", "clone"))

        D = pm.Poisson("D", N * H.dot(Phi.T).T, dims=("position", "spot"), observed=data.D_obs)
        _A_num = H.dot((Phi * C).T).T
        _A_denom = H.dot(Phi.T).T
        A_prob = pm.Deterministic("A_prob", _A_num / _A_denom, dims=("position", "spot"))
        pm.Binomial("A", D, A_prob, dims=("position", "spot"), observed=data.A_obs)
    return model
```

Build the model with random data to show structure of DAG

```{code-cell} ipython3
:tags: []

np.random.seed(RANDOM_SEED)
mock_tumor_data = TumoroscopeData(
    K=5,
    S=10,
    M=40,
    F=np.ones(5) / 5.0,
    cell_counts=np.random.randint(1, 20, size=10),
    C=np.random.beta(2, 2, size=(40, 5)),
    D_obs=np.random.randint(2, 20, size=(40, 10)),
    A_obs=np.random.randint(2, 20, size=(40, 10)),
)
pm.model_to_graphviz(tumoroscope(mock_tumor_data))
```

## Simultation

### Data generation

Describe the general data simulation process.

+++

## Sampling and convergence

```{code-cell} ipython3
# with model:
#     trace = pm.sample(1000, tune=1500, random_seed=RANDOM_SEED)
```

```{code-cell} ipython3
# az.plot_energy(trace);
```

```{code-cell} ipython3
# Also show R_hat and ESS values
```

## Posterior analysis

+++

## Potential model changes

Suggest a few changes such as adding priors to some hyperparameters and directly using the Dirichlet distribution.

```{code-cell} ipython3

```

+++ {"tags": []}

## Authors
 
* Based on the description of Tumoroscope from the bioR$\chi$iv pre-print ["Tumoroscope: a probabilistic model for mapping cancer clones in tumor tissues"](https://www.biorxiv.org/content/10.1101/2022.09.22.508914v1) by [Shafighi](http://orcid.org/0000-0003-0367-8864) *et al.* published in 2022.
* Adapted from the [notebook on the PyMC implementation of Tumoroscope](https://github.com/jhrcook/pymc-tumoroscope) by [Joshua Cook](https://joshuacook.netlify.app/) ([GitHub profile](https://github.com/jhrcook)).

+++

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
:tags: []

%load_ext watermark
%watermark -n -u -v -iv -w -p aesara,aeppl,xarray
```

:::{include} ../page_footer.md
:::
