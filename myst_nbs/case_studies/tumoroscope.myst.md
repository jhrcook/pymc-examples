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

Many (if not all) cancers progress via an evolutionary process whereby multiple clones develop with new mutations over time, resulting in a heterogenous, multi-clonal population of cells comprising a tumor.
Understanding this evolutionary process and the physical distribution of clones is essential to optimal therapeutic development and treatment.
Tumoroscope {cite:p}`shafighi2022tumoroscope` was designed as a generative, probabilistic model to estimate the proportion of each known clone of a tumor in each spot in a spatial transcriptomics dataset.
Here, I have built the Tumoroscope model in PyMC from the description in the manuscript and then present a demonstration with simulated data.

In addition, this implementation has been converted into a `pip`-installable package: [`tumoroscope-pymc`](https://github.com/jhrcook/tumoroscope-pymc).
Finally, more simulations and tweaks of the model have been explored in the original notebook from which this tutorial was adapted - a link to this notebook is provided at the top of the page.

+++

### Motivation

> This case study is biology-heavy and can be rather niche.
> I have done my best to simplify the problem and only present the level of complexity necessary to understand the model.
> Changes to improve the readability or clarity of this notebook are welcome.

Tumor progression can be modeled as an evolutionary process whereby cells develop mutations that are either neutral, beneficial, or determental to their growth.
This process results in a heterogenous mixture of subpopulations, or *clones*, present in different proportions.
Understanding this process and the variaty of clonality of a tumor are important factors in how a patient's tumor responds to therapy.

Often, the evolutionary tree of a tumor can be built through single cell or bulk DNA sequencing, that is sequencing the genomes of individual cells or a large sample of the tumor.
While the former provides greater detail (by inspecting a single cell at a time), single cell DNA sequencing methods are still rather error-prone and suffer from far lower coverage than more traditional bulk sequencing methods.
The term "coverage" refers to the level of evidence across the entire genome.
Since each cell only has one copy of each chromosome, coverage can be increased by amplifying the DNA, a process that can introduce further errors and bias.
Bulk DNA sequencing, reading the genome of many (hundreds of thousands) of cells at a time, has greater coverage (because there is more DNA), but lower resolution because the cells' genomes are all pooled together.
Yet, there exist methods to use the proportions of detected mutations to build the evolutionary tree by which the clones developed.
Tumoroscope, uses such methods and bulk DNA sequencing to determine the clones present in a tumor and the identifying mutations of each clone.

Transcriptomics is the study of reading the RNA in a biological sample.
A common use of these methods is to study the levels of messenger RNA (mRNA), the intermediate molecule between the information stored in genes in DNA and functional elements of a cell, proteins.
These measurements, often referred to as the *expression* of a gene, can tell a researcher a lot about the behavior and state of a cell.
Spatial transcriptomics is a relatively nascent field of study whereby the RNA of a small region on a tissue, often down to just a few cells and called a *spot*, can be read.
Repeating this process over a grid of a tissue results in spatially-resolved transcriptomics.

Tumoroscope integrates bulk DNA sequencing and spatial transcriptomics to estimate the proportion of each cancer clone in a spot.
Bulk DNA sequencing is used to discover the clones in a tumor and their identifying mutations.
The RNA transcript data of each spot is used to estimate the proportion of each clone in the given spot.
The main difficulty is in the relatively low coverage of the genome by the RNA, information for some of the identifying mutations will be captured, but many will be uninformed.
Tumoroscope models the data-generating process to provide a probability for these proportions.

+++

### The Tumoroscope model

Below is a diagram of the (**a-f**) data generating process and (**g**) Tumoroscope model.
(Panel **h** shows an additional model presented in the paper for studying the transcriptomic profile of each tumor clone after determining the clone proportions in each spot of the spatial transcriptomics; it is an optimization problem and thus not included in this tutorial.)
First, I shall briefly describe the data-generating process as described by the figure, followed by how this is modeled in Tumoroscope.

![Tumoroscope diagram](https://raw.githubusercontent.com/jhrcook/tumoroscope-pymc/master/tumoroscope-diagram.jpeg)

+++

#### Data-generating process

Here is a somewhat hypothetical description of the system that is being measured.
The tumor has grown, and along the way multiple subpopulations have formed as various cells have acquired mutations and continued to divide.
These sets of mutations at the identified positions of the genome act as a fingerprint for each clone.
The clonal structure can be inferred from sequencing the DNA from a sample of the tumor (**b** and **e**).
Humans have two copies of each chromosome, and usually a unique mutation is only acquired on one of the copies.
It is possible to gain or lose sections of DNA, though, resulting in changes to the *zygosity* of the clone at those locations of the genome.
If the *copy number* of the region in which a mutation exists changes, the number of instances of the mutation in the genome of a single cell will change, too.
For instance, a duplication of the region on the chromsome with the mutation reults in two copies of the mutation and copy of the original (called *wild type*) DNA sequence.

A slice of the tumor has been taken and prepared for microscopy (**a**) and spatial transcriptomics (**c**).
Each spot of the spatial transcriptomics potentially contains multiple cells (in this case usually between 3-6 cells).
Each of these cells belongs to one of the known clones and carries that specific pattern of mutations and the corresponding zygosity in each region.
These cells produce mRNA by transcribing their genes and these transcripts contain the mutations for each clone, too.
The zygosity (ratio of mutated to not-mutated) at the mutation position likely corresponds to the ratio of transcripts, too, that is if there are multiple copies of the mutated region, there will be more transcripts with the mutation.
Due to limitations of the technology, the actual mRNA transcripts captured during the spatial transcriptomics process will only cover part of the genome in each spot.
Depending on the clonal composition of the cells in the spot, only a fraction of the total captured mRNA transcripts, called *reads* will contain a mutation in a position – these trancripts are referred to as *alternative reads*.

+++

#### Model structure

The authors of the original publication provide a great description of the model in the Methods section, so if you need clarification on any parts of the model, I would recommend consulting the original text first.

Below is a list of the indices used in this model:

- mutation position $i \in \{1, \dots, M\}$
- clone $k \in \{1, \dots, K\}$
- spot $s \in \{1, \dots, S\}$

Below is a list of the observed data in the model:

- $C^{[M\times K]} \in [0, 1]$: the zygosity at position $i$ in clone $k$
- $D^{[M\times S]} \in \mathbb{Z}^*$: the total number of reads that cover position $i$ in spot $s$ where $\mathbb{Z}^+$ is the set of all non-negative integers
- $A^{[M\times S]} \in \mathbb{Z}^*$: the total number of *alternative* (i.e. mutated) reads that cover position $i$ in spot $s$
- $N^{[S]} \in \mathbb Z^+$: number of cells in spot $s$ where $\mathbb{Z}^+$ is the set of all positive integers
- $F^{[K]} \in [0, 1]$: the proportion of clones $k$ in the tumor sample; $\sum_k^K F = 1$

With all of this infrastructure, we can get to the description of the model.
Variable $Z^{[S \times K]}$ is an indicator of whether clone $k$ is present in spot $s$ and is defined as a Bernoulli distribution with parameter $\Pi$ which in turn has a Beta prior distribution with hyperparameter $\zeta_s$

\begin{aligned}
\mathbb{P}(Z | \Pi) &\sim \text{Bernoulli}(\Pi) \\
\mathbb{P}(\Pi | \zeta_s, K) &\sim \text{Beta}(\frac{\zeta_s}{K}, 1)
\end{aligned}

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
