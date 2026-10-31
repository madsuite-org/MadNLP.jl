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
is_inertia
inertia
```

## Selecting a linear solver
The sparse linear solvers are implemented in package extensions. Each is selected
with a marker type, passed as the option `linear_solver`; the extension is loaded
together with the package it wraps (MadNLP loads MUMPS, SuiteSparse and LDLFactorizations).

```@docs
create_linear_solver
backend_package
default_sparse_solver
DummySolver
MumpsSolver
LDLSolver
UmfpackSolver
CHOLMODSolver
```

## Iterative refinement
MadNLP uses iterative refinement to improve the
accuracy of the solution returned by the linear solver.

```@docs
solve_refine!

```
