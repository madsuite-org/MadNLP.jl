# Running MadNLP in arbitrary precision

```@meta
CurrentModule = MadNLP
```
```@setup multiprecision
using NLPModels
using MadNLP

```

MadNLP supports solving optimization problems in arbitrary precision.
By default, MadNLP adapts its precision following the `NLPModel`
passed in input. Most models are using `Float64` (in fact, almost
all optimization modelers are implemented using double
precision), but for certain applications it can be useful to use
arbitrary precision to get more accurate solution.

!!! info
    There exists different packages to instantiate a optimization
    model in arbitrary precision in Julia. Most of them
    leverage the flexibility offered by [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
    In particular, we recommend:
    - [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl/): supports `Float32`, `Float64` and `Float128`.
    - [ExaModels](https://github.com/exanauts/ExaModels.jl): supports `AbstractFloat`.

## Defining a problem in arbitrary precision

As a demonstration, we implement the model [airport](https://vanderbei.princeton.edu/ampl/nlmodels/cute/airport.mod)
from CUTEst using ExaModels. The code writes:
```@example multiprecision
using ExaModels

function airport_model(T)
    N = 42
    # Data
    r = T[0.09 , 0.3, 0.09, 0.45, 0.5, 0.04, 0.1, 0.02, 0.02, 0.07, 0.4, 0.045, 0.05, 0.056, 0.36, 0.08, 0.07, 0.36, 0.67, 0.38, 0.37, 0.05, 0.4, 0.66, 0.05, 0.07, 0.08, 0.3, 0.31, 0.49, 0.09, 0.46, 0.12, 0.07, 0.07, 0.09, 0.05, 0.13, 0.16, 0.46, 0.25, 0.1]
    cx = T[-6.3, -7.8, -9.0, -7.2, -5.7, -1.9, -3.5, -0.5, 1.4, 4.0, 2.1, 5.5, 5.7, 5.7, 3.8, 5.3, 4.7, 3.3, 0.0, -1.0, -0.4, 4.2, 3.2, 1.7, 3.3, 2.0, 0.7, 0.1, -0.1, -3.5, -4.0, -2.7, -0.5, -2.9, -1.2, -0.4, -0.1, -1.0, -1.7, -2.1, -1.8, 0.0]
    cy = T[8.0, 5.1, 2.0, 2.6, 5.5, 7.1, 5.9, 6.6, 6.1, 5.6, 4.9, 4.7, 4.3, 3.6, 4.1, 3.0, 2.4, 3.0, 4.7, 3.4, 2.3, 1.5, 0.5, -1.7, -2.0, -3.1, -3.5, -2.4, -1.3, 0.0, -1.7, -2.1, -0.4, -2.9, -3.4, -4.3, -5.2, -6.5, -7.5, -6.4, -5.1, 0.0]
    # Wrap all data in a single iterator for ExaModels
    data = [(i, cx[i], cy[i], r[i]) for i in 1:N]
    IJ = [(i, j) for i in 1:N-1 for j in i+1:N]
    # Write model using ExaModels
    core = ExaModels.ExaCore(T)
    ExaModels.@add_var(core, x, 1:N; lvar = -10.0, uvar = 10.0)
    ExaModels.@add_var(core, y, 1:N; lvar = -10.0, uvar = 10.0)
    ExaModels.@add_obj(
        core,
        ((x[i] - x[j])^2 + (y[i] - y[j])^2) for (i, j) in IJ
    )
    ExaModels.@add_con(core, (x[i]-dcx)^2 + (y[i] - dcy)^2 - dr for (i, dcx, dcy, dr) in data; lcon=-Inf)
    return ExaModels.ExaModel(core)
end
```

The function `airport_model` takes as input the type used to define the model in ExaModels.
For example, `ExaCore(Float64)` instantiates a model with `Float64`, whereas `ExaCore(Float32)`
instantiates a model using `Float32`. Thus, instantiating the instance `airport` using `Float32`
simply amounts to
```@example multiprecision
nlp = airport_model(Float32)

```
We verify that the model is correctly instantiated using `Float32`:
```@example multiprecision
x0 = NLPModels.get_x0(nlp)
println(typeof(x0))
```

## Solving a problem in Float32
Now that we have instantiated our model in `Float32`, we solve
it using MadNLP. As `nlp` is using `Float32`, MadNLP will automatically adjust
its internal types to `Float32` during the instantiation.
By default, the convergence tolerance is also adjusted to the input type, such that `tol = sqrt(eps(T))`.
Hence, in our case the tolerance is set automatically to
```@example multiprecision
tol = sqrt(eps(Float32))
```
We solve the problem using Lapack as linear solver:
```@example multiprecision
results = madnlp(nlp; linear_solver=LapackCPUSolver)
nothing
```

!!! note
    Note that the distribution of Lapack shipped with Julia supports
    `Float32`, so here we do not have to worry whether the
    type is supported by the linear solver. Almost all linear solvers shipped
    with MadNLP supports `Float32`.

The final solution is
```@example multiprecision
results.solution

```
and the objective is
```@example multiprecision
results.objective

```

For completeness, we compare with the solution returned when we solve the
same problem using `Float64`:
```@example multiprecision
nlp_64 = airport_model(Float64)
results_64 = madnlp(nlp_64; linear_solver=LapackCPUSolver)
nothing
```
The final objective is now
```@example multiprecision
results_64.objective

```
As expected when solving an optimization problem with `Float32`,
the relative difference between the two solutions is far from being negligible:
```@example multiprecision
rel_diff = abs(results.objective - results_64.objective) / results_64.objective
```

## Solving a problem in Float128
Now, we go in the opposite direction and solve a problem using `Float128`
to get a better accuracy. We start by loading the library `Quadmath` to
work with quadruple precision:
```@example multiprecision
using Quadmath
```
We can instantiate our problem using `Float128` directly as:
```@example multiprecision
nlp_128 = airport_model(Float128)
```


!!! warning
    On the contrary to `Float32`, a few linear solvers support `Float128` out of the box.
    Currently, the only solvers supporting quadruple in MadNLP are `LDLSolver`
    and the HSL solvers (require MadNLPHSL).
    `LDLSolvers` uses [an LDL factorization implemented in pure Julia](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).
    The solver `LDLSolver` is not adapted to solve large-scale nonconvex nonlinear programs,
    but works if the problem is small enough (as it is the case here).

Once we have replaced the solver by `LDLSolver`, solving the problem with MadNLP in `Float128` just amounts to
```@example multiprecision
results_128 = madnlp(nlp_128; linear_solver=LDLSolver)
nothing

```
Note that the final tolerance is much lower than before. We get the solution in quadruple precision
```@example multiprecision
results_128.solution
```
as well as the final objective:
```@example multiprecision
results_128.objective
```
