# KKT systems

```@meta
CurrentModule = MadNLP
```
```@setup kkt_example
using LinearAlgebra
using SparseArrays
using NLPModels
using MadNLP
using MadNLPTests
nlp = MadNLPTests.HS15Model()

```

KKT systems are linear system with a special KKT structure.
MadNLP uses a special structure [`AbstractKKTSystem`](@ref) to represent internally
the KKT system. The [`AbstractKKTSystem`](@ref) fulfills two goals:

1. Store the values of the Hessian of the Lagrangian and of the Jacobian.
2. Assemble the corresponding KKT matrix $$K$$.

## A brief look at the math

We have seen [previously](../algorithm.md) that MadNLP reformulates the
inequality constraints as equality constraints, by introducing slack variables.
In what follows, we denote by ``w = (x, s)`` the primal iterate concatenating
the original decision variable ``x`` and the slack variable ``s``.
Once reformulated, the problem has the following structure:
```math
  \begin{aligned}
    \min_{w} \; & f(w) \; , \\
    \text{subject to} \quad & c(w) = 0 \; , \quad w \geq 0 \; .
  \end{aligned}
```

At each iteration, MadNLP aims at solving the following system of nonlinear
equations, parameterized by a positive barrier parameter ``\mu``:
```math
\begin{aligned}
& \nabla f(w) + \nabla c(w)^\top y - z = 0 \; , \\
& c(w) = 0 \; , \\
& WZe = \mu e \;, \; (w, z) > 0 \; .
\end{aligned}
```

MadNLP solves this system using the Newton method, and computes a
descent direction ``(\Delta w, \Delta y, \Delta z)`` as solution of
```math
\begin{bmatrix}
H_k +\delta_x I & J_k^\top & -I \\
J_k & -\delta_y & 0 \\
Z_k & 0 & W_k
\end{bmatrix}
\begin{bmatrix}
\Delta w \\ \Delta y \\ \Delta z
\end{bmatrix}
= -
\begin{bmatrix}
 \nabla f(w_k) + \nabla c(w_k)^\top y_k - z_k \\
 c(w_k)  \\
 W_k Z_k e - \mu e
\end{bmatrix} \; ,
```
with $$\delta_x$$ and $$\delta_y$$ appropriate primal-dual regularization terms
that guarantee ``(\Delta w, \Delta y, \Delta z)`` is a descent direction for the current filter.
The system needs evaluating the Jacobian of the constraints ``J_k = \nabla c(w_k)^\top``
and the Hessian of the Lagrangian ``H_k = \nabla_{xx}^2 L(w_k, y_k, z_k)``.

We note by ``(r_1, r_2, r_3)`` the right-hand-side. The primal-dual KKT system
can be symmetrized as
```math
\begin{bmatrix}
H_k +\delta_x I & J_k^\top & Z_k^{1/2} \\
J_k & -\delta_y & 0 \\
Z_k^{1/2} & 0 & -W_k
\end{bmatrix}
\begin{bmatrix}
\Delta w \\ \Delta y \\ -Z_k^{-1/2} \Delta z
\end{bmatrix}
=
\begin{bmatrix}
r_1 \\ r_2 \\ Z_k^{-1/2} r_3
\end{bmatrix}
```
This system is implemented as an [`AbstractUnreducedKKTSystem`](@ref) in MadNLP.

We can obtain a smaller augmented KKT system by eliminating the last block
of rows associated to ``\Delta z = W_k^{-1} (r_3 - Z_k \Delta w)``:
```math
\begin{bmatrix}
H_k + \Sigma_k + \delta_x I & J_k^\top \\
J_k & -\delta_y
\end{bmatrix}
\begin{bmatrix}
\Delta w \\ \Delta y
\end{bmatrix}
=
\begin{bmatrix}
r_1 + W_k^{-1} r_3 \\ r_2
\end{bmatrix}
```
with ``\Sigma_k = W_k^{-1} Z_k``.
This system is implemented as an [`AbstractReducedKKTSystem`](@ref) in MadNLP.


## Assembling a KKT system, step by step

The primal-dual KKT systems depend on the Hessian of the Lagrangian ``H_k``
and the Jacobian ``J_k``.
Hence, we have to update the values in the KKT system at each iteration
of the interior-point algorithm.

By default, MadNLP stores the KKT system as a [`SparseKKTSystem`](@ref).
The KKT system takes as input a [`SparseCallback`](@ref) wrapping
a given `NLPModel` `nlp`. We instantiate the callback `cb` with
the function [`create_callback`](@ref):
```@example kkt_example
cb = MadNLP.create_callback(
    MadNLP.SparseCallback,
    nlp,
)

```

### Initializing a KKT system

The size of the KKT system depends directly on the problem's characteristics
(number of variables, number of of equality and inequality constraints).
A [`SparseKKTSystem`](@ref) stores the Hessian and the Jacobian in sparse
(COO) format. The KKT matrix can be factorized using either a
dense or a sparse linear solvers. Here we use the solver provided
in LAPACK:
```@example kkt_example
linear_solver = LapackCPUSolver
```

We can instantiate a `SparseKKTSystem` using
the function [`create_kkt_system`](@ref):
```@example kkt_example
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    linear_solver,
)

```

Once the KKT system built, one has to initialize it
to use it inside the interior-point algorithm:
```@example kkt_example
MadNLP.initialize!(kkt);

```

The user can query the KKT matrix inside `kkt`, simply as
```@example kkt_example
kkt_matrix = MadNLP.get_kkt(kkt)
```
This returns a reference to the KKT matrix stored internally
inside `kkt`. Each time the matrix is assembled inside `kkt`,
`kkt_matrix` is updated automatically.


### Updating a KKT system
We suppose now we want to refresh the values stored in the KKT system.


#### Updating the values of the Hessian
The Hessian part of the KKT system can be queried as
```@example kkt_example
hess_values = MadNLP.get_hessian(kkt)

```
For a `SparseKKTSystem`, `hess_values` is a `Vector{Float64}` storing
the nonzero values of the Hessian.
Then, one can update the vector `hess_values` by using NLPModels.jl:
```@example kkt_example
n = NLPModels.get_nvar(nlp)
m = NLPModels.get_ncon(nlp)
x = NLPModels.get_x0(nlp) # primal variables
y = zeros(m) # dual variables

NLPModels.hess_coord!(nlp, x, y, hess_values)

```
Eventually, a post-processing step is applied to refresh all the values internally:
```@example kkt_example
MadNLP.compress_hessian!(kkt)

```
!!! note
    By default, the function [`compress_hessian!`](@ref) does nothing.
    But it can be required for very specific use-case, for instance building internally a
    Schur complement matrix.


#### Updating the values of the Jacobian
We proceed exactly the same way to update the values in the Jacobian.
One queries the Jacobian values in the KKT system as
```@example kkt_example
jac_values = MadNLP.get_jacobian(kkt)

```
We can refresh the values with NLPModels as
```@example kkt_example
NLPModels.jac_coord!(nlp, x, jac_values)

```
And then applies a post-processing step as
```@example kkt_example
MadNLP.compress_jacobian!(kkt)

```

#### Updating the values of the diagonal matrices
Once the Hessian and the Jacobian updated, the algorithm
can apply primal and dual regularization terms on the diagonal
of the KKT system, to improve the numerical behavior in the linear solver.
This operation is implemented inside the [`regularize_diagonal!`](@ref) function:
```@example kkt_example
pr_value = 1.0
du_value = 0.0

MadNLP.regularize_diagonal!(kkt, pr_value, du_value)

```

### Assembling the KKT matrix
Once the values updated, one can assemble the resulting KKT matrix.
This translates to
```@example kkt_example
MadNLP.build_kkt!(kkt)
```
By doing so, the values stored inside `kkt` will be transferred
to the KKT matrix `kkt_matrix` (as returned by the function [`get_kkt`](@ref)):
```@example kkt_example
kkt_matrix
```

Internally, a [`SparseKKTSystem`](@ref) stores the KKT system in
a sparse COO format. When [`build_kkt!`](@ref) is called, the sparse COO matrix
is transferred to `SparseMatrixCSC` if the linear solver is sparse,
or alternatively to a `Matrix` if the linear solver is dense.

!!! note
    The KKT system stores only the lower-triangular part of the KKT system,
    as it is symmetric.


## Solution of the KKT system
Now the KKT system is assembled in a matrix ``K`` (here stored in `kkt_matrix`), we want
to solve a linear system ``K x = b``, for instance to evaluate the
next descent direction. To do so, we use the linear solver stored
internally inside `kkt` (here an instance of `LapackCPUSolver`).

We start by factorizing the KKT matrix ``K``:
```@example kkt_example
MadNLP.factorize!(kkt.linear_solver)

```
By default, MadNLP uses an ``LBL^\top`` factorization (with ``L`` lower triangular
and ``B`` block-diagonal) to decompose the symmetric indefinite KKT matrix.

Once the KKT matrix has been factorized, we can compute the solution of the linear
system with a backsolve. The function takes as input a [`AbstractKKTVector`](@ref),
an object used to do algebraic manipulation with a [`AbstractKKTSystem`](@ref).
We start by instantiating two [`UnreducedKKTVector`](@ref) (encoding respectively
the right-hand-side and the solution):
```@example kkt_example
b = MadNLP.UnreducedKKTVector(kkt)
fill!(MadNLP.full(b), 1.0)
x = copy(b)

```
The right-hand-side encodes a vector of 1:
```@example kkt_example
MadNLP.full(b)
```
We solve the system ``K x = b`` using the [`solve_kkt!`](@ref) function:
```@example kkt_example
MadNLP.solve_kkt!(kkt, x)
MadNLP.full(x)
```
We verify that the solution is correct by multiplying it on the left
with the KKT system, using `mul!`:
```@example kkt_example
mul!(b, kkt, x) # overwrite b!
MadNLP.full(b)
```
We recover a vector filled with `1`, which was the initial right-hand-side.

