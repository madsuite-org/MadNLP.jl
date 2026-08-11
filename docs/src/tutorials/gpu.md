# Running MadNLP on the GPU

```@meta
CurrentModule = MadNLP
```
```@setup gpu
using NLPModels
using MadNLP
```

MadNLP supports the solution of large-scale optimization problems
on the GPU, [with significant speedups reported on some instances](https://arxiv.org/html/2405.14236v2).
In this tutorial, we show how to solve a nonlinear program on a NVIDIA GPU. We start
by importing MadNLPGPU:
```@example gpu
using MadNLPGPU
using CUDA
using CUDSS

```


## How to solve a problem on the GPU with ExaModels?

MadNLP has been designed to run entirely on the GPU, without data exchange between
the host and the device. We recommend implementing the model
with the modeler [ExaModels](https://github.com/exanauts/ExaModels.jl) whenever possible.

We import ExaModels as:
```@example gpu
using ExaModels
```

!!! info
    MadNLP does not yet support solving sparse optimization problems on AMD GPUs.

### Evaluating the model on the GPU with ExaModels
The first step requires implementing the model with ExaModels.

As a demonstration, we implement the model [airport](https://vanderbei.princeton.edu/ampl/nlmodels/cute/airport.mod)
from the CUTEst benchmark using ExaModels. The code writes:
```@example gpu
function airport_model(T, backend)
    N = 42
    # Data
    r = T[0.09 , 0.3, 0.09, 0.45, 0.5, 0.04, 0.1, 0.02, 0.02, 0.07, 0.4, 0.045, 0.05, 0.056, 0.36, 0.08, 0.07, 0.36, 0.67, 0.38, 0.37, 0.05, 0.4, 0.66, 0.05, 0.07, 0.08, 0.3, 0.31, 0.49, 0.09, 0.46, 0.12, 0.07, 0.07, 0.09, 0.05, 0.13, 0.16, 0.46, 0.25, 0.1]
    cx = T[-6.3, -7.8, -9.0, -7.2, -5.7, -1.9, -3.5, -0.5, 1.4, 4.0, 2.1, 5.5, 5.7, 5.7, 3.8, 5.3, 4.7, 3.3, 0.0, -1.0, -0.4, 4.2, 3.2, 1.7, 3.3, 2.0, 0.7, 0.1, -0.1, -3.5, -4.0, -2.7, -0.5, -2.9, -1.2, -0.4, -0.1, -1.0, -1.7, -2.1, -1.8, 0.0]
    cy = T[8.0, 5.1, 2.0, 2.6, 5.5, 7.1, 5.9, 6.6, 6.1, 5.6, 4.9, 4.7, 4.3, 3.6, 4.1, 3.0, 2.4, 3.0, 4.7, 3.4, 2.3, 1.5, 0.5, -1.7, -2.0, -3.1, -3.5, -2.4, -1.3, 0.0, -1.7, -2.1, -0.4, -2.9, -3.4, -4.3, -5.2, -6.5, -7.5, -6.4, -5.1, 0.0]
    # Wrap all data in a single iterator for ExaModels
    data = [(i, cx[i], cy[i], r[i]) for i in 1:N]
    IJ = [(i, j) for i in 1:N-1 for j in i+1:N]
    # Write model using ExaModels
    core = ExaModels.ExaCore(T; backend=backend)
    x = ExaModels.variable(core, 1:N, lvar = -10.0, uvar=10.0)
    y = ExaModels.variable(core, 1:N, lvar = -10.0, uvar=10.0)
    ExaModels.objective(
        core,
        ((x[i] - x[j])^2 + (y[i] - y[j])^2) for (i, j) in IJ
    )
    ExaModels.constraint(core, (x[i]-dcx)^2 + (y[i] - dcy)^2 - dr for (i, dcx, dcy, dr) in data; lcon=-Inf)
    return ExaModels.ExaModel(core)
end
```

The first argument `T` specifies the numerical precision (`Float32`, `Float64` or any `AbstractFloat`) whereas the second
argument `backend` sets the device used to evaluate the model.
We instantiate the model on the GPU with:
```@example gpu
nlp = airport_model(Float64, CUDABackend())
```
By passing a `CUDABackend()`, we make sure that all the attributes in `nlp` are instantiated on the GPU. E.g., the initial point becomes a `CuVector`:
```@example gpu
NLPModels.get_x0(nlp)
```

!!! info
    If your problem is implemented with JuMP in a given `model`, ExaModel can load it for you on the GPU just
    by using `nlp = ExaModel(model; backend=CUDABackend())`.


### Solving the problem on the GPU with MadNLP
Once the model `nlp` loaded on the GPU, you can solve it using the function `madnlp`:
```@example gpu
results = madnlp(nlp; linear_solver=CUDSSSolver)
nothing
```
Note that we are using the sparse linear solver [NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/getting_started.html)
to solve the primal-dual KKT system entirely on the GPU.

When solving an optimization problem on the GPU, MadNLP proceeds to some automatic modifications. In order:
1. It increases the parameter `bound_relax_factor` to `1e-4`.
2. It relaxes all the equality constraints ``g(x) = 0`` as a pair of inequality constraints ``-\tau \leq g(x) \leq \tau``, with ``\tau`` being equal to `bound_relax_factor`;
3. It reduces the KKT system down to sparse condensed system, exploiting the fact that the relaxed problem has only (potentially tight) inequality constraints. Up to a given primal regularization, the resulting KKT system is positive definite and can be factorized using a pivoting-free factorization routines (e.g. a Cholesky or an LDL' decompositions). This is the so-called *Lifted-KKT* formulation documented in [this article](https://arxiv.org/html/2405.14236v2).
4. The new condensed KKT system increases the ill-conditioning inherent to the interior-point method, amplifying the loss of accuracy when the iterates get close to a local solution. As a result, the termination tolerance `tol` is also increased to `1e-4` to recover convergence.

As a result, the convergence observed can be significantly different than what we
observe on the CPU. In particular, relaxing the parameter `bound_relax_factor` has a non-marginal impact
on the feasible set's geometry. You can limit the loss of accuracy by
specifying explicitly the relaxation and tolerance parameters to MadNLP:
```@example gpu
results = madnlp(
    nlp;
    tol=1e-7,
    bound_relax_factor=1e-7,
    linear_solver=CUDSSSolver,
)
nothing
```
Decreasing the tolerance `tol` too much is likely to cause numerical issues inside the algorithm.
In general, we recommend keeping the value of `bound_relax_factor` below `tol`.


### Solving the problem on the GPU with HyKKT

Some applications require accurate solutions. In that case, we recommend using the
extension [HybridKKT.jl](https://github.com/madsuite-org/HybridKKT.jl), which implements
the Golub & Greif augmented Lagrangian formulation detailed [in this article](https://www.tandfonline.com/doi/abs/10.1080/10556788.2022.2124990). Compared to Lifted-KKT, the Hybrid-KKT
strategy is more accurate (it doesn't relax the equality constraints in the problem) but
slightly slower (it computes the descent direction using a conjugate gradient at every IPM iterations).

Once the package `HybridKKT` installed, the solution proceeds as
```
using HybridKKT

results = madnlp(
    nlp;
    linear_solver=MadNLPGPU.CUDSSSolver,
    kkt_system=HybridKKT.HybridCondensedKKTSystem,
    equality_treatment=MadNLP.EnforceEquality,
    fixed_variable_treatment=MadNLP.MakeParameter,
)
```


## How to solve on the GPU a legacy model running on the CPU?

If your optimization problem is not instantiated on the GPU, you can still
solve it on the GPU by wrapping your model in a `SparseWrapperModel`.
Oftentimes, this is the most convenient solution if your problem does not formulate easilly with ExaModels.
In that case, the evaluation of the model runs on the CPU, but all MadNLP's internals
are instantiated on the GPU (including the sparse linear solver).

To give an example, we suppose we have instantiate our `airport` model on the CPU:
```@example gpu
nlp_cpu = airport_model(Float64, nothing)
```
You can wrap it on the GPU using:
```@example gpu
nlp_wrapped = MadNLP.SparseWrapperModel(CuArray, nlp_cpu)
```
Then, we proceed as above to call MadNLP:
```@example gpu
results = madnlp(nlp_wrapped; linear_solver=CUDSSSolver)
nothing
```

## How to solve on the GPU a JuMP model?

If your model is implemented with JuMP, you can solve it on the GPU using the
same trick as in the previous paragraph just by passing a new option `array_type`
specifying which array type to use to instantiate the `SparseWrapperModel`.

This is demonstrated in the following example:
```julia
using CUDA
using JuMP
using MadNLP, MadNLPGPU

model = Model(MadNLP.Optimizer)
@variable(model, x1 <= 0.5)
@variable(model, x2)

@objective(model, Min, 100.0 * (x2 - x1^2)^2 + (1.0 - x1)^2)
@constraint(model, x1 * x2 >= 1.0)
@constraint(model, x1 + x2^2 >= 0.0)

JuMP.set_optimizer_attribute(model, "array_type", CuArray)
JuMP.set_optimizer_attribute(model, "linear_solver", CUDSSSolver)
JuMP.optimize!(model) # Solve the model on the GPU!
```
