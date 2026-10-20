![logo](https://github.com/madsuite-org/MadNLP.jl/blob/master/logo-full.svg)

*A [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) solver based on the filter line-search [interior point method](https://en.wikipedia.org/wiki/Interior-point_method) (as in [Ipopt](https://github.com/coin-or/Ipopt)) that can handle/exploit diverse classes of data structures, either on [host](https://en.wikipedia.org/wiki/Central_processing_unit) or [device](https://en.wikipedia.org/wiki/Graphics_processing_unit) memories.*

---

| **License** | **Documentation** | **Build Status** | **Coverage** | **DOI** |
|:-----------:|:-----------------:|:----------------:|:------------:|:-------:|
| [![License: MIT][license-img]][license-url] | [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[license-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/madsuite-org/MadNLP.jl/blob/master/LICENSE
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://madsuite-org.github.io/MadNLP.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://madsuite-org.github.io/MadNLP.jl/dev
[build-gh-img]: https://github.com/madsuite-org/MadNLP.jl/actions/workflows/test.yml/badge.svg
[build-gh-url]: https://github.com/madsuite-org/MadNLP.jl/actions/workflows/test.yml
[codecov-img]: https://codecov.io/gh/madsuite-org/MadNLP.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/madsuite-org/MadNLP.jl
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5825776.svg
[doi-url]: https://doi.org/10.5281/zenodo.5825776


## Quickstart

The following example shows how to solve the HS15 problem with JuMP and MadNLP:
```julia
using JuMP, MadNLP
model = Model()
@variable(model, x1 <= 0.5)
@variable(model, x2)
@objective(model, Min, 100.0 * (x2 - x1^2)^2 + (1.0 - x1)^2)
@constraint(model, x1 * x2 >= 1.0)
@constraint(model, x1 + x2^2 >= 0.0)
JuMP.set_optimizer(model, MadNLP.Optimizer)
JuMP.set_optimizer_attribute(model, "max_iter", 100)
JuMP.set_optimizer_attribute(model, "print_level", MadNLP.INFO)
optimize!(model)
```

## Installation

MadNLP can be installed directly via the Julia package manager:
```julia
pkg> add MadNLP
```
Furthermore, MadNLP comes with several extensions:

- `MadNLPGPU`: import GPU-accelerated linear solvers in MadNLP
- `MadNLPHSL`: import the HSL linear solvers in MadNLP
- `MadNLPPardiso`: import the Pardiso linear solver in MadNLP

## Citing MadNLP.jl

If you use MadNLP.jl in your research, we would greatly appreciate your citing it.

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

## Supporting MadNLP.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/madsuite-org/MadNLP.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/madsuite-org/MadNLP.jl/discussions).
