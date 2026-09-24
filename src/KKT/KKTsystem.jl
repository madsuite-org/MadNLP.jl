
"""
    AbstractKKTSystem{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, QN<:AbstractHessian{T}}

Abstract type for KKT system.
"""
abstract type AbstractKKTSystem{T, VT, MT<:AbstractMatrix{T}, QN<:AbstractHessian{T, VT}} end

"""
    AbstractUnreducedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN}

Augmented KKT system associated to the linearization of the KKT
conditions at the current primal-dual iterate ``(x, s, y, z, ν, w)``.

The associated matrix is
```
[Wₓₓ  0   Aₑ'  Aᵢ' V½  0  ]  [Δx]
[0    0   0   -I   0   W½ ]  [Δs]
[Aₑ   0   0    0   0   0  ]  [Δy]
[Aᵢ  -I   0    0   0   0  ]  [Δz]
[V½   0   0    0  -X   0  ]  [Δτ]
[0    W½  0    0   0  -S  ]  [Δρ]
```
with
* ``Wₓₓ``: Hessian of the Lagrangian.
* ``Aₑ``: Jacobian of the equality constraints
* ``Aᵢ``: Jacobian of the inequality constraints
* ``X = diag(x)``
* ``S = diag(s)``
* ``V = diag(ν)``
* ``W = diag(w)``
* ``Δτ = -W^{-½}Δν``
* ``Δρ = -W^{-½}Δw``
"""
abstract type AbstractUnreducedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN} end

"""
    AbstractReducedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN}

The reduced KKT system is a simplification of the original Augmented KKT system.
Comparing to [`AbstractUnreducedKKTSystem`](@ref)), `AbstractReducedKKTSystem` removes
the two last rows associated to the bounds' duals ``(ν, w)``.

At a primal-dual iterate ``(x, s, y, z)``, the matrix writes
```
[Wₓₓ + Σₓ   0    Aₑ'   Aᵢ']  [Δx]
[0          Σₛ    0    -I ]  [Δs]
[Aₑ         0     0     0 ]  [Δy]
[Aᵢ        -I     0     0 ]  [Δz]
```
with
* ``Wₓₓ``: Hessian of the Lagrangian.
* ``Aₑ``: Jacobian of the equality constraints
* ``Aᵢ``: Jacobian of the inequality constraints
* ``Σₓ = X⁻¹ V``
* ``Σₛ = S⁻¹ W``

"""
abstract type AbstractReducedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN} end

"""
    AbstractCondensedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN}

The condensed KKT system simplifies further the [`AbstractReducedKKTSystem`](@ref)
by removing the rows associated to the slack variables ``s`` and the inequalities.

At the primal-dual iterate ``(x, y)``, the matrix writes
```
[Wₓₓ + Σₓ + Aᵢ' Σₛ Aᵢ    Aₑ']  [Δx]
[         Aₑ              0 ]  [Δy]
```
with
* ``Wₓₓ``: Hessian of the Lagrangian.
* ``Aₑ``: Jacobian of the equality constraints
* ``Aᵢ``: Jacobian of the inequality constraints
* ``Σₓ = X⁻¹ V``
* ``Σₛ = S⁻¹ W``
"""
abstract type AbstractCondensedKKTSystem{T, VT, MT, QN} <: AbstractKKTSystem{T, VT, MT, QN} end


#=
    Templates
=#

"""
    create_kkt_system(
        ::Type{KKT},
        cb::AbstractCallback,
        ind_cons::NamedTuple,
        linear_solver::Type{LinSol};
        opt_linear_solver=default_options(linear_solver),
        hessian_approximation=ExactHessian,
    ) where {KKT<:AbstractKKTSystem, LinSol<:AbstractLinearSolver}

Instantiate a new KKT system with type `KKT`, associated to the
the nonlinear program encoded inside the callback `cb`. The
`NamedTuple` `ind_cons` stores the indexes of all the variables and
constraints in the callback `cb`. In addition, the user should pass
the linear solver `linear_solver` that will be used to solve the KKT system
after it has been assembled.

"""
function create_kkt_system end

"Number of primal variables (including slacks) associated to the KKT system."
function num_variables end

"""
    get_kkt(kkt::AbstractKKTSystem)::AbstractMatrix

Return a pointer to the KKT matrix implemented in `kkt`.
The pointer is passed afterward to a linear solver.
"""
function get_kkt end

"Get Jacobian matrix"
function get_jacobian end

"Get Hessian matrix"
function get_hessian end

"""
    initialize!(kkt::AbstractKKTSystem)

Initialize KKT system with default values.
Called when we initialize the `MadNLPSolver` storing the current KKT system `kkt`.
"""
function initialize! end

"""
    build_kkt!(kkt::AbstractKKTSystem)

Assemble the KKT matrix before calling the factorization routine.

"""
function build_kkt! end

"""
    factorize_kkt!(kkt::AbstractKKTSystem)

Factorize the KKT matrix using the linear solver stored in `kkt`.

"""
function factorize_kkt! end

"""
    compress_hessian!(kkt::AbstractKKTSystem)

Compress the Hessian inside `kkt`'s internals.
This function is called every time a new Hessian is evaluated.

Default implementation do nothing.

"""
function compress_hessian! end

"""
    compress_jacobian!(kkt::AbstractKKTSystem)

Compress the Jacobian inside `kkt`'s internals.
This function is called every time a new Jacobian is evaluated.

By default, the function updates in the Jacobian the coefficients
associated to the slack variables.

"""
function compress_jacobian! end

"""
    jtprod!(y::AbstractVector, kkt::AbstractKKTSystem, x::AbstractVector)

Multiply with transpose of Jacobian and store the result
in `y`, such that ``y = A' x`` (with ``A`` current Jacobian).
"""
function jtprod! end

"""
    solve_kkt!(kkt::AbstractKKTSystem, w::AbstractKKTVector)

Solve the KKT system ``K x = w`` with the linear solver stored
inside `kkt` and stores the result inplace inside the `AbstractKKTVector` `w`.

"""
function solve_kkt! end

"""
    regularize_diagonal!(kkt::AbstractKKTSystem, primal_values::Number, dual_values::Number)

Regularize the values in the diagonal of the KKT system in an incremental fashion.
Called internally inside the interior-point routine.
"""
function regularize_diagonal! end

"""
    is_inertia_correct(kkt::AbstractKKTSystem, n::Int, m::Int, p::Int)

Check if the inertia ``(n, m, p)`` returned by the linear solver is adapted
to the KKT system implemented in `kkt`.

"""
function is_inertia_correct end

"Nonzero in Jacobian"
function nnz_jacobian end

#=
    Generic functions
=#
function initialize!(kkt::AbstractKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    return
end

function factorize_kkt!(kkt::AbstractKKTSystem)
    return factorize!(kkt.linear_solver)
end

function regularize_diagonal!(kkt::AbstractKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .-= dual
end

Base.size(kkt::AbstractKKTSystem) = size(kkt.aug_com)
Base.size(kkt::AbstractKKTSystem, dim::Int) = size(kkt.aug_com, dim)

# Getters
get_kkt(kkt::AbstractKKTSystem) = kkt.aug_com
get_jacobian(kkt::AbstractKKTSystem) = kkt.jac
get_hessian(kkt::AbstractKKTSystem) = kkt.hess

"""
    is_inertia(kkt::AbstractKKTSystem)

Whether `kkt` can report the inertia of the full KKT matrix, and hence whether
`InertiaAuto` may select the inertia-based regularization. Defaults to the capability of
the KKT system's linear solver; KKT systems whose matrix is factorized by more than one
solver (e.g. the Schur path) must override this to require all of them.
"""
is_inertia(kkt::AbstractKKTSystem) = is_inertia(kkt.linear_solver)

"""
        is inertia_correct(kkt::AbstractKKTSystem, num_pos, num_zero, num_neg)

Check if the inertia ``(num_pos, num_zero, num_neg)`` returned by the linear solver
ensures the desired conditions
"""
function is_inertia_correct(kkt::AbstractKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == num_variables(kkt))
end

"""
        should_regularize_dual(kkt::AbstractKKTSystem, num_pos, num_zero, num_neg)

Check if the inertia ``(num_pos, num_zero, num_neg)`` returned by the linear solver
necessitates the dual regularization of the KKT system.
"""
function should_regularize_dual(kkt::AbstractKKTSystem, num_pos, num_zero, num_neg)
    return num_zero != 0
end

compress_hessian!(kkt::AbstractKKTSystem) = nothing

include("rhs.jl")
include("Dense/augmented.jl")
include("Dense/condensed.jl")
include("Dense/utils.jl")
include("Sparse/unreduced.jl")
include("Sparse/augmented.jl")
include("Sparse/condensed.jl")
include("Sparse/scaled_augmented.jl")
include("Sparse/utils.jl")
include("Schur/schur.jl")

