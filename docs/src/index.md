![Logo](assets/logo-full.svg)

MadNLP is an open-source nonlinear programming solver,
purely implemented in Julia. MadNLP implements a filter line-search
interior-point algorithm, as used in [Ipopt](https://github.com/coin-or/Ipopt). MadNLP
seeks to streamline the development of modeling and algorithmic paradigms in
order to exploit structures and to make efficient use of high-performance computers.
Notably, MadNLP is part of [MadSuite](https://madsuite.org/), a suite of GPU-accelerated optimization solvers.


## Design

### MadNLP's problem structure
MadNLP targets the solution of constrained nonlinear problems, formulating as
```math
  \begin{aligned}
    \min_{x} \; & f(x) \\
    \text{subject to} \quad & g_\ell \leq g(x) \leq g_u \\
                            & x_\ell \leq x \leq x_u
  \end{aligned}
```
where $$x \in \mathbb{R}^n$$ is the decision variable, $$f: \mathbb{R}^n \to \mathbb{R}$$
and $$g: \mathbb{R}^n \to \mathbb{R}^m$$ two smooth nonlinear functions.
MadNLP makes the distinction between the **bound constraints** $$x_\ell \leq x \leq x_u$$
and the **generic constraints** $$g_\ell \leq g(x) \leq g_u$$.
No other structure is assumed _a priori_.

MadNLP is built on top of [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/),
a generic package to represent optimization models in Julia. In addition,
MadNLP is interfaced with [ExaModels](https://github.com/exanauts/ExaModels.jl),
[JuMP](https://github.com/jump-dev/JuMP.jl) and [Plasmo](https://github.com/zavalab/Plasmo.jl).


### MadNLP's performance
In any interior-point algorithm, the two computational bottlenecks are
1. The evaluation of the first- and second-order derivatives.
2. The factorization of the primal-dual KKT system.

The first point is problem-dependent and is often related to the automatic differentiation backend being employed.
The second point is more problematic, as the primal-dual KKT system is symmetric indefinite and ill-conditioned.

### Linear solvers
By default, MadNLP is using [MUMPS](https://mumps-solver.org/) to solve the primal-dual KKT system.
Other efficient linear solvers are interfaced, all listed in the table below.


Linear solver | Type | Hardware
:--- | :--- | :---
||
__MadNLP (base)__||
||
MumpsSolver (default)   | sparse | CPU
UmfpackSolver           | sparse | CPU
LapackCPUSolver         | dense  | CPU
LDLSolver               | sparse | CPU
CHOLMODSolver           | sparse | CPU
||
__MadNLPGPU__||
||
CUDSSSolver             | sparse | CUDA
LapackCUDASolver         | dense  | CUDA
LapackROCmSolver         | dense  | AMD GPU
||
__MadNLPHSL__||
||
Ma27Solver              | sparse | CPU
Ma57Solver              | sparse | CPU
Ma86Solver              | sparse | CPU
Ma97Solver              | sparse | CPU
||
__MadNLPPardiso__||
||
PardisoSolver           | sparse | CPU
PardisoMKLSolver        | sparse | CPU

!!! warning
    In general, we recommend using a sparse linear solver as soon as the number of variables
    in the problem is greater than 1,000.


## Citing MadNLP.jl
If you use MadNLP.jl in your research, we would greatly appreciate your citing it.

To cite the software itself, use the metadata in [`CITATION.cff`](https://github.com/MadNLP/MadNLP.jl/blob/master/CITATION.cff), which GitHub also renders under *Cite this repository* in the sidebar:

```bibtex
@software{madnlp,
  author = {Shin, Sungho and Pacaud, Fran{\c{c}}ois and Montoison, Alexis and {The MadNLP.jl contributors}},
  title  = {{MadNLP.jl}: A nonlinear programming solver in {J}ulia},
  url    = {https://github.com/MadNLP/MadNLP.jl},
  doi    = {10.5281/zenodo.5825775}
}
```

Depending on which parts of MadNLP.jl you use, please also consider citing the following references:

```bibtex
@article{shin2024accelerating,
  title     = {Accelerating optimal power flow with {GPU}s: {SIMD} abstraction of nonlinear programs and condensed-space interior-point methods},
  author    = {Shin, Sungho and Anitescu, Mihai and Pacaud, Fran{\c{c}}ois},
  journal   = {Electric Power Systems Research},
  volume    = {236},
  pages     = {110651},
  year      = {2024},
  publisher = {Elsevier}
}

@article{shin2021graph,
  title     = {Graph-based modeling and decomposition of energy infrastructures},
  author    = {Shin, Sungho and Coffrin, Carleton and Sundar, Kaarthik and Zavala, Victor M},
  journal   = {IFAC-PapersOnLine},
  volume    = {54},
  number    = {3},
  pages     = {693--698},
  year      = {2021},
  publisher = {Elsevier}
}
```
