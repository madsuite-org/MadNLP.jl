```@meta
CurrentModule = MadNLP
```

# KKT systems

MadNLP manipulates KKT systems using two abstractions:
an `AbstractKKTSystem` storing the KKT system' matrix
and an `AbstractKKTVector` storing the KKT system's right-hand-side.


## AbstractKKTSystem

```@docs
AbstractKKTSystem

```

MadNLP implements three different types of `AbstractKKTSystem`,
depending how far we reduce the KKT system.

```@docs
AbstractUnreducedKKTSystem
AbstractReducedKKTSystem
AbstractCondensedKKTSystem
```

Each `AbstractKKTSystem` follows the interface described below:
```@docs

create_kkt_system

num_variables
get_kkt
get_jacobian
get_hessian

initialize!
build_kkt!
factorize_kkt!
solve_kkt!
compress_hessian!
compress_jacobian!
jtprod!
regularize_diagonal!
is_inertia_correct
should_regularize_dual
nnz_jacobian
```

## Sparse KKT systems

By default, MadNLP stores a `AbstractReducedKKTSystem` in sparse format,
as implemented by `SparseKKTSystem`:
```@docs
SparseKKTSystem

```
The user has the alternative choices:
```@docs
SparseUnreducedKKTSystem
SparseCondensedKKTSystem
ScaledSparseKKTSystem
```

## Dense KKT systems

MadNLP provides also two structures to store the KKT system
in a dense matrix. Although less efficient than their sparse counterparts,
these two structures allow to store the KKT system efficiently when the
problem is instantiated on the GPU.
```@docs
DenseKKTSystem
DenseCondensedKKTSystem

```

## Schur complement KKT system

For two-stage stochastic programs with a block-arrowhead structure,
MadNLP provides a specialized KKT system that eliminates the per-scenario
blocks and reduces the linear system to a dense `nd × nd` Schur complement
on the design variables.
```@docs
SchurComplementCondensedKKTSystem

```

## AbstractKKTVector
Each instance of `AbstractKKTVector` implements the following interface.

```@docs
AbstractKKTVector

number_primal
number_dual
full
primal
dual
primal_dual
dual_lb
dual_ub

```

By default, MadNLP provides one implementation of an `AbstractKKTVector`.

```@docs
UnreducedKKTVector
```

