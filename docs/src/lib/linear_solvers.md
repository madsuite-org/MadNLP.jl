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
With the option `kkt_ordering`, a sparse KKT system hands its linear solver the matrix
`P A Pᵀ` permuted by a fill-reducing ordering, for every sparse linear solver on the host and on
a GPU. The ordering is computed once on the host, from the pattern of the KKT matrix; the
scaling and the shifts above apply first. Load Metis to use `MetisOrdering`, and AMD to use
`AMDOrdering`, `SymAMDOrdering` and `ColAMDOrdering`:

```julia
using MadNLP, Metis
madnlp(nlp; kkt_ordering = MetisOrdering())
```

A linear solver that computes its own ordering reorders the permuted matrix again. Solvers
whose factorization depends on the primal and dual blocks of the KKT matrix (`SDSSolver`) do
not support an ordering yet.

```@docs
PermutedSolver
AbstractOrdering
DefaultOrdering
MetisOrdering
AMDOrdering
SymAMDOrdering
ColAMDOrdering
UserOrdering
ordering_permutation
```
