```@meta
CurrentModule = MadNLP
```

# Linear solvers

## Direct linear solvers
Each linear solver employed in MadNLP
implements the following interface.

```@docs
AbstractLinearSolver
introduce
factorize!
solve_linear_system!
is_factorized
is_inertia
inertia
```

A new linear solver also implements [`create_linear_solver`](@ref), `default_options`,
`input_type` and `is_supported`, which are called with its type.

## Selecting a linear solver
The sparse linear solvers are implemented in package extensions. Each is selected
with a marker type, passed as the option `linear_solver`; the extension is loaded
together with the package it wraps (MadNLP loads MUMPS, SuiteSparse, LDLFactorizations and HSL).

```@docs
create_linear_solver
backend_package
default_sparse_solver
DummySolver
MumpsSolver
LDLSolver
UmfpackSolver
CHOLMODSolver
Ma27Solver
Ma57Solver
Ma77Solver
Ma86Solver
Ma97Solver
PardisoMKLSolver
PanuaPardisoSolver
```

## Iterative refinement
MadNLP uses iterative refinement to improve the
accuracy of the solution returned by the linear solver.

```@docs
solve_refine!
KrylovIterator

```

## Scaling and shifts of the factorized matrix
With the options `kkt_scaler`, `primal_shift` and `dual_shift`, a sparse KKT system hands its
linear solver a scaled and shifted copy of its matrix; iterative refinement works with the
unscaled, unshifted system.

```@docs
create_kkt_linear_solver
ScaledShiftedSolver
AbstractScaler
NoScaler
JacobiScaler
MaxNormScaler
RuizScaler
```
