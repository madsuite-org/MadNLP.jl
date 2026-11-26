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
AbstractSolverMarker
solver_type
backend_package
default_sparse_solver
default_dense_solver
DummySolver
MumpsSolver
LDLSolver
UmfpackSolver
CHOLMODSolver
AccelerateSolver
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

## Fill-reducing orderings
The option `kkt_ordering` replaces the fill-reducing ordering of the sparse linear solver by
one computed once on the host, from the pattern of the KKT matrix: `DefaultOrdering()` (the
default) leaves the ordering to the linear solver, `MetisOrdering()` (with Metis loaded) and
`AMDOrdering()` (with AMD loaded) hand their permutation to the solver as its pivot order,
through its user-permutation interface:

```julia
using MadNLP, Metis
madnlp(nlp; kkt_ordering = MetisOrdering())
```

The solvers that accept a permutation are `MumpsSolver` (`ICNTL(7) = 1`), the HSL solvers
(the user pivot order of MA27, MA57, MA77, MA86 and MA97), `LDLSolver`, `UmfpackSolver`
(column pre-ordering), `CHOLMODSolver`, `PardisoMKLSolver` and `PanuaPardisoSolver`
(`iparm[5] = 1`), `AccelerateSolver` (`SparseOrderUser`), `CUDSSSolver` and `SDSSolver`
(`user_perm`). Each stores it in its option `*_user_perm`. Pivoting and matching during the
factorization still apply. Any other linear solver throws an `ArgumentError` with an ordering
other than `DefaultOrdering()`.

```@docs
set_permutation!
AbstractOrdering
DefaultOrdering
MetisOrdering
AMDOrdering
ordering_permutation
```
