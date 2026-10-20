# Implementing a custom KKT system

```@meta
CurrentModule = MadNLP
```
```@setup kktsystem
using NLPModels
using MadNLP
using Random

include("hs15.jl")
include("diag_kkt.jl")

```

This tutorial explains how to implement a custom [`AbstractKKTSystem`](@ref) in MadNLP.

## Structure exploiting methods

MadNLP gives the user the possibility to exploit the problem's structure
at the linear algebra level, when solving the KKT system at every Newton iteration.
By default, the KKT system is factorized using a sparse linear solver (like MUMPS, PARDISO or HSL MA57).
A sparse linear solver analyses the problem's algebraic structure when computing the
symbolic factorization, with a heuristic that determines an optimal elimination tree.
As an alternative, the problem's structure can be exploited directly, by specifying
the order of the pivots to perform (e.g. using a block elimination algorithm).
Doing so usually leads to significant speed-up in the algorithm.

We recall that the KKT system solved at each Newton iteration has the structure:
```math
\overline{
\begin{pmatrix}
 H       & J^\top & - I    & I \\
 J       & 0      & 0      & 0 \\
 -Z_\ell & 0      & X_\ell & 0 \\
 Z_u     & 0      & 0      & X_u
\end{pmatrix}}^{K}
\begin{pmatrix}
    \Delta x \\
    \Delta y \\
    \Delta z_\ell \\
    \Delta z_u
\end{pmatrix}
=
\begin{pmatrix}
    r_1 \\ r_2 \\ r_3 \\ r_4
\end{pmatrix}
```
with ``H`` a sparse matrix storing the (approximated) Hessian of the Lagrangian,
and ``J`` a sparse matrix storing the Jacobian of the constraints.
We note the diagonal matrices
``Z_\ell = diag(z_\ell)``,
``Z_u = diag(z_u)``,
``X_\ell = diag(x_\ell - x)``,
``X_u = diag(x - x_u)``.

In MadNLP, every linear system with the structure ``K`` is implemented as an [`AbstractKKTSystem`](@ref).
By default, MadNLP represents a KKT system as a [`SparseKKTSystem`](@ref).

```@example kktsystem
nlp = HS15Model()
results = madnlp(nlp; kkt_system=MadNLP.SparseKKTSystem, linear_solver=LapackCPUSolver)
nothing

```


## Solving an AbstractKKTSystem in MadNLP

The `AbstractKKTSystem` object is an abstraction to solve the generic
system ``K x = b``. Depending on the implementation, the structure of the linear system
is exploited in different fashions. Solving a KKT system amounts to the four
following operations:
1. Querying the current sensitivities to assemble the different blocks constituting the matrix ``K``.
2. Assembling a reduced sparse matrix condensing the sparse matrix ``K`` to an equivalent smaller symmetric system.
3. Calling a linear solver to solve the condensed system.
4. Calling a routine to unpack the condensed solution to get the original descent direction ``(\Delta x, \Delta y, \Delta z_\ell, \Delta z_u)``.

Exploiting the problem's structure usually happens in steps (2), (3) and (4).
We skim through the four successive steps in more details.

### Getting the sensitivities

The KKT system requires the following information:

- the (approximated) Hessian of the Lagrangian ``H`` ;
- the constraints' Jacobian ``J`` ;
- the diagonal matrices ``Z_\ell``, ``Z_u`` and ``X_\ell``, ``X_u``.

The Hessian and the Jacobian are assumed sparse by default.

At every IPM iteration, MadNLP updates automatically the values in
``H``, ``J`` and in the diagonal matrices ``Z_\ell, Z_u, X_\ell, X_u``.
By default, we expect the following attributes available in every instance `kkt` of an `AbstractKKTSystem`:

- `kkt.hess`: stores the nonzeros of the Hessian ``H``;
- `kkt.jac`: stores the nonzeros of the Jacobian ``J``;
- `kkt.l_diag`: stores the diagonal entries in ``X_\ell``;
- `kkt.u_diag`: stores the diagonal entries in ``X_u``;
- `kkt.l_lower`: stores the diagonal entries in ``Z_\ell``;
- `kkt.u_lower`: stores the diagonal entries in ``Z_u``.

The attributes `kkt.hess` and `kkt.jac` are accessed respectively using
the getters [`get_hessian`](@ref) and [`get_jacobian`](@ref).

Every time MadNLP queries the Hessian and the Jacobian, it updates the
nonzero values in `kkt.hess` and `kkt.jac`.
Right after, MadNLP calls
the functions [`compress_hessian!`](@ref) and [`compress_jacobian!`](@ref) respectively, to propagate these updates to all internal structures of the KKT system `kkt` associated with the Hessian and the Jacobian.

To recap, every time we evaluate the Hessian and the Jacobian, MadNLP
calls automatically the functions:
```julia
hess = MadNLP.get_hessian(kkt)
MadNLP.compress_hessian!(kkt)
```
to update the values in the Hessian, and for the Jacobian:
```julia
jac = MadNLP.get_jacobian(kkt)
MadNLP.compress_jacobian!(kkt)
```

### Assembling the KKT system

Once the sensitivities have been updated, we can assemble the KKT matrix ``K``
and condense it to an equivalent system ``K_{c}`` before factorizing it with a linear solver.
The assembling of the KKT system is done in the function [`build_kkt!`](@ref).

The system is usually stored in the attribute `kkt.aug_com`. Its dimension depends on the condensation used.
The matrix `kkt.aug_com` can be dense or sparse, depending on the condensation used.
MadNLP uses the getter [`get_kkt`](@ref) to query the matrix `kkt.aug_com` stored in the KKT system `kkt`.


### Solving the system

Once the matrix ``K_c`` is assembled, we pass it to the linear solver
for factorization.
The linear solver is stored internally in `kkt`.
By default, it is stored in the attribute `kkt.linear_solver`.
The factorization is handled internally in MadNLP.

Once factorized, it remains to solve the linear system using a backsolve.
The backsolve has to be implemented by the user in the function [`solve_kkt!`](@ref).
It reduces the right-hand-side (RHS) down to a form adapted to the condensed matrix
``K_c`` and calls the linear solver to perform the backsolve.
Then the condensed solution
is unpacked to recover the full solution ``(\Delta x, \Delta y, \Delta z_\ell, \Delta z_u)``.

To recap, MadNLP assembles and solves the KKT linear system using the
following operations:
```julia
# Assemble the KKT system
MadNLP.build_kkt!(kkt)

# Factorize the KKT system
MadNLP.factorize_kkt!(kkt)

# Backsolve
MadNLP.solve_kkt!(kkt, w)
```

## Example: Implementing a new KKT system

As an example, we detail how to implement a custom KKT system in MadNLP.
Note that we consider this usage as an advanced use of MadNLP.
After this word of caution, let's dive into the details!

In this example, we want to approximate the Hessian of the Lagrangian ``H`` as a diagonal matrix ``D_H``
and solve the following KKT system at each IPM iteration:
```math
\begin{pmatrix}
 D_H & J^\top & - I & I \\
 J & 0 &  0 & 0 \\
 -Z_\ell &  0 & X_\ell & 0 \\
 Z_u & 0 &  0 & X_u
\end{pmatrix}
\begin{pmatrix}
    \Delta x \\
    \Delta y \\
    \Delta z_\ell \\
    \Delta z_u
\end{pmatrix}
=
\begin{pmatrix}
    r_1 \\ r_2 \\ r_3 \\ r_4
\end{pmatrix}
```
This new system is not equivalent to the original system ``K``, but it's much easier to solve
as it does not involve the full Hessian ``H``.
If the diagonal values of ``D_H`` are constant and are equal to ``\alpha``, the algorithm becomes equivalent to a gradient descent with step ``\alpha^{-1}``.

Using the relations
``\Delta z_\ell = X_\ell^{-1} (r_3 + Z_\ell \Delta x)`` and ``\Delta z_u = X_u^{-1} (r_4 - Z_u \Delta x)``,
we condense the matrix down to the reduced form:
```math
\begin{pmatrix}
 D_H + \Sigma_x & J^\top \\
 J & 0  \\
\end{pmatrix}
\begin{pmatrix}
    \Delta x  \\
    \Delta y
\end{pmatrix}
=
\begin{pmatrix}
    r_1 + X_\ell^{-1} r_3 - X_u^{-1} r_4\\ r_2
\end{pmatrix}
```
with the diagonal matrix ``\Sigma_x = -X_\ell^{-1} Z_\ell - X_u^{-1} Z_u``.
The new system is symmetric indefinite, but much easier to solve than the original one.

The previous reduction is standard in NLP solvers: MadNLP implements the reduced KKT
system operating in the space ``(\Delta x, \Delta y)`` using the abstraction
[`AbstractReducedKKTSystem`](@ref). If ``D_H`` is replaced by the original
Hessian matrix ``H``, we recover exactly the [`SparseKKTSystem`](@ref) used by default in MadNLP.

### Creating the KKT system

We create a new KKT system `DiagonalHessianKKTSystem`, inheriting from [`AbstractReducedKKTSystem`](@ref).
Using generic types, the structure `DiagonalHessianKKTSystem` is defined as:

```julia
struct DiagonalHessianKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, QN}
    # Nonzeroes values for Hessian and Jacobian
    hess::VT
    jac_callback::VT
    jac::VT
    # Diagonal matrices
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    # Augmented system K
    aug_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Diagonal of the Hessian
    diag_hess::VT
    # Jacobian
    jac_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    n_var::Int
    n_ineq::Int
    n_tot::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    # Quasi-Newton approximation
    quasi_newton::QN
end

```

!!! info
    Remind that the primal vector manipulated by MadNLP concatenates the
    original primal variable together with a slack variable: ``w = (x, s)``.
    The KKT system requires the following information from MadNLP:
    - `n_var` is the dimension of the original primal variable ``x``;
    - `n_tot` is the dimension of the extended primal variable ``w``;
    - `n_ineq` is the number of inequality constraints, equal to the dimension of ``s``;
    - `ind_lb` is the index of the variables in ``w`` which are lower-bounded ;
    - `ind_ub` is the index of the variables in ``w`` which are upper-bounded.

!!! info
    Here, we define a DiagonalHessianKKTSystem as a subtype
    of a [`AbstractReducedKKTSystem`](@ref). Depending on the condensation,
    the following alternatives are available:
    - [`AbstractUnreducedKKTSystem`](@ref): no condensation is applied.
    - [`AbstractCondensedKKTSystem`](@ref): the reduced KKT system is condensed further by removing the blocks associated to the slack variables and inequality constraints.


!!! info
    The attributes `pr_diag` and `du_diag` store the primal and dual regularization terms, respectively.
    The primal regularization corresponds to the entries added to the diagonal of the (1,1) block, while the dual regularization corresponds to those added to the diagonal of the (2,2) block.
    By default, the dual regularization is set to zero, whereas the primal regularization is initialized to ``\Sigma_x``.

MadNLP instantiates a new KKT system with the function [`create_kkt_system`](@ref), with the following signature:
```julia
function MadNLP.create_kkt_system(
    ::Type{DiagonalHessianKKTSystem},
    cb::MadNLP.SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T,VT}
```
We pass as arguments:
1. the type of the KKT system to build (here, `DiagonalHessianKKTSystem`),
2. the structure used to evaluate the callbacks `cb`,
3. a generic linear solver `linear_solver`.

This function instantiates all the data structures needed in `DiagonalHessianKKTSystem`.
The most difficult part is to assemble the sparse matrices `aug_raw` and `jac_raw`,
here stored in COO format.
This is done in four steps:

**Step 1.** We import the sparsity pattern of the Jacobian :
```julia
jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)
```
**Step 2.** We build the resulting KKT matrix `aug_raw` in COO format, knowing that ``D_H`` is diagonal:
```julia
# System's dimension
n_hess = n_tot # Diagonal Hessian!
n_jac = length(jac_sparsity_I)
aug_vec_length = n_tot+m
aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

# Build vectors to store COO coordinates
I = MadNLP.create_array(cb, Int32, aug_mat_length)
J = MadNLP.create_array(cb, Int32, aug_mat_length)
V = VT(undef, aug_mat_length)
fill!(V, 0.0)  # Need to initiate V to avoid NaN

offset = n_tot+n_jac+n_slack+n_hess+m

# Primal regularization block
I[1:n_tot] .= 1:n_tot
J[1:n_tot] .= 1:n_tot

# Hessian block
I[n_tot+1:n_tot+n_hess] .= 1:n_tot # diagonal Hessian!
J[n_tot+1:n_tot+n_hess] .= 1:n_tot # diagonal Hessian!

# Jacobian block
I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J

# Slack block
I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)

# Dual regularization block
I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)
J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, I, J, V)
```
**Step 3.** We convert `aug_raw` from COO to CSC using the following utilities:

```julia
aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)
```

**Step 4.** We pass the matrix in CSC format to the linear solver:

```julia
_linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
```

!!! info
    Storing the Hessian and Jacobian, even in sparse format, is expensive in
    term of memory. For that reason, MadNLP stores the Hessian and Jacobian
    only once in the KKT system.


### Getting the sensitivities

MadNLP requires the following getters to update the sensitivities. As much as we can,
we try to update the values inplace in the matrix `aug_raw`.
We use the default implementation of [`compress_jacobian!`](@ref) in MadNLP:
```julia
function MadNLP.compress_jacobian!(kkt::DiagonalHessianKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    MadNLP.transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
    return
end
```
The term `-1.0` accounts for the slack variables used to reformulate
the inequality constraints as equality constraints.

For [`compress_hessian!`](@ref), we take into account that the diagonal matrix
``D_H`` is the diagonal of the Hessian:

```julia
function MadNLP.compress_hessian!(kkt::DiagonalHessianKKTSystem)
    kkt.diag_hess .= 1.0
    return
end
```

MadNLP also needs the following basic functions to get the different matrices and
the dimension of the linear system:
```
MadNLP.num_variables(kkt::DiagonalHessianKKTSystem) = length(kkt.diag_hess)
MadNLP.get_kkt(kkt::DiagonalHessianKKTSystem) = kkt.aug_com
MadNLP.get_jacobian(kkt::DiagonalHessianKKTSystem) = kkt.jac_callback
function MadNLP.jtprod!(y::AbstractVector, kkt::DiagonalHessianKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end
```

### Assembling the KKT system
Once the sensitivities are updated, we assemble the new matrix ``K_c`` first
in COO format in `kkt.aug_raw`, before converting the matrix to CSC format in `kkt.aug_com`
using the utility `MadNLP.transfer!`:

```julia
function MadNLP.build_kkt!(kkt::DiagonalHessianKKTSystem)
    MadNLP.transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end
```

### Solving the system

It remains to implement the backsolve. For the reduced KKT formulation, the RHS
``r_1 + X_\ell^{-1} r_3 - X_u^{-1} r_4`` is built automatically using the
function `MadNLP.reduce_rhs!`.
The backsolve solves for ``(\Delta x, \Delta y)``. The dual's descent direction
``\Delta z_\ell`` and ``\Delta z_u`` are recovered afterwards using
the function `MadNLP.finish_aug_solve!`:
```julia
function MadNLP.solve_kkt!(kkt::DiagonalHessianKKTSystem, w::MadNLP.AbstractKKTVector)
    MadNLP.reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)
    MadNLP.solve_linear_system!(kkt.linear_solver, primal_dual(w))
    MadNLP.finish_aug_solve!(kkt, w)
    return w
end
```

!!! note
    The function `solve_kkt!` takes as second argument a vector `w` being an
    [`AbstractKKTVector`](@ref). An `AbstractKKTVector` is a convenient data
    structure used in MadNLP to store and access the elements in the primal-dual vector
    ``(\Delta x, \Delta y, \Delta z_\ell, \Delta z_u)``.

!!! warning
    When calling `solve_kkt!`, the values in the vector `w` are updated inplace.
    The vector `w` should be initialized with the RHS ``(r_1, r_2, r_3, r_4)`` before calling
    the function `solve_kkt!`. The function modifies the values directly in the vector `w`
    to return the solution ``(\Delta x, \Delta y, \Delta z_\ell, \Delta z_u)``.

Last, MadNLP implements an iterative refinement method to get accurate descent
directions in the final iterations. The iterative refinement algorithm implements
Richardson's method, which requires multiplying the KKT matrix ``K`` on the right
by any vector ``w = (w_x, w_y, w_{z_l}, w_{z_u})``. This is provided in MadNLP
by overloading the function `LinearAlgebra.mul!`:

```julia
function LinearAlgebra.mul!(
    w::MadNLP.AbstractKKTVector{T},
    kkt::DiagonalHessianKKTSystem,
    x::MadNLP.AbstractKKTVector{T},
    alpha = one(T),
    beta = zero(T),
) where {T}

    mul!(primal(w), Diagonal(kkt.diag_hess), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)

    # Reduce KKT vector
    MadNLP._kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end
```


### Demonstration

We now have all the elements needed to solve the problem with the new KKT linear
system `DiagonalHessianKKTSystem`. We just have to pass the KKT system to MadNLP
using the option `kkt_system`:

```@example kktsystem
nlp = HS15Model()
results = madnlp(nlp; kkt_system=DiagonalHessianKKTSystem, linear_solver=LapackCPUSolver)
nothing
```
