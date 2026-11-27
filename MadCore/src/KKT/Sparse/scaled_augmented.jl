"""
    ScaledSparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Scaled version of the [`AbstractReducedKKTSystem`](@ref) (using the K2.5 formulation introduced in [GOS]).

The K2.5 formulation of the augmented KKT system has a better conditioning
than the original (K2) formulation. It is recommended to switch to a `ScaledSparseKKTSystem`
if you encounter numerical difficulties in MadNLP.

At a primal-dual iterate ``(x, s, y, z)``, the matrix writes
```
[√Ξₓ Wₓₓ √Ξₓ + Δₓ   0      √Ξₓ Aₑ'   √Ξₓ Aᵢ']  [√Ξₓ⁻¹ Δx]
[0                  Δₛ     0           -√Ξₛ ]  [√Ξₛ⁻¹ Δs]
[Aₑ√Ξₓ              0      0              0 ]  [Δy      ]
[Aᵢ√Ξₓ             -√Ξₛ    0              0 ]  [Δz      ]
```
with
* ``Wₓₓ``: Hessian of the Lagrangian.
* ``Aₑ``: Jacobian of the equality constraints
* ``Aᵢ``: Jacobian of the inequality constraints
* ``Δₓ = Xᵤ Zₗˣ + Xₗ Zᵤˣ``
* ``Δₛ = Sᵤ Zₗˢ + Sₗ Zᵤˢ``
* ``Ξₓ = Xₗ Xᵤ``
* ``Ξₛ = Sₗ Sᵤ``

# References
[GOS] Ghannad, Alexandre, Dominique Orban, and Michael A. Saunders.
"Linear systems arising in interior methods for convex optimization: a symmetric formulation with bounded condition number."
Optimization Methods and Software 37, no. 4 (2022): 1344-1369.

"""
struct ScaledSparseKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    scaling_factor::VT
    buffer1::VT
    buffer2::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    scaled_aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

# Build KKT system directly from SparseCallback
function create_kkt_system(::Type{ScaledSparseKKTSystem}, cb::SparseCallback{T, VT}, linear_solver::Type; kwargs...) where {T, VT}
    d = _sparse_augmented_data(cb, linear_solver; kwargs...)
    aug = d.aug_raw
    scaled_aug_raw = SparseMatrixCOO(aug.m, aug.n, aug.I, aug.J, copy(aug.V))
    return ScaledSparseKKTSystem(
        d.hess, d.jac_callback, d.jac, d.quasi_newton, d.reg, d.pr_diag, d.du_diag,
        d.l_diag, d.u_diag, d.l_lower, d.u_lower,
        VT(undef, d.n_tot), VT(undef, d.n_tot), VT(undef, d.n_tot),  # scaling_factor, buffer1, buffer2
        d.aug_raw, d.aug_com, d.aug_csc_map, scaled_aug_raw,
        d.hess_raw, d.hess_com, d.hess_csc_map,
        d.jac_raw, d.jac_com, d.jac_csc_map,
        d.linear_solver,
        d.ind_ineq, cb.ind_lb, cb.ind_ub,
    )
end

num_variables(kkt::ScaledSparseKKTSystem) = length(kkt.pr_diag)
get_jacobian(kkt::ScaledSparseKKTSystem) = kkt.jac_callback

function initialize!(kkt::ScaledSparseKKTSystem{T}) where T
    _initialize_sparse!(kkt)
    fill!(kkt.scaling_factor, one(T))
    return
end

function jtprod!(y::AbstractVector, kkt::ScaledSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

function compress_jacobian!(kkt::ScaledSparseKKTSystem{T}) where T
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -one(T)
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_hessian!(kkt::ScaledSparseKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

# N.B. Matrices are assumed to have an augmented KKT structure and be lower-triangular.
function _build_scale_augmented_system_coo!(dest, src, scaling, n, m)
    for (k, i, j) in zip(1:nnz(src), src.I, src.J)
        # Primal regularization pr_diag
        if k <= n
            dest.V[k] = src.V[k]
        # Hessian block
        elseif i <= n && j <= n
            dest.V[k] = src.V[k] * scaling[i] * scaling[j]
        # Jacobian block
        elseif n + 1 <= i <= n + m && j <= n
            dest.V[k] = src.V[k] * scaling[j]
        # Dual regularization du_diag
        elseif n + 1 <= i <= n + m && n + 1 <= j <= n + m
            dest.V[k] = src.V[k]
        end
    end
end

function build_kkt!(kkt::ScaledSparseKKTSystem)
    m, n = size(kkt.jac_raw)
    _build_scale_augmented_system_coo!(
        kkt.scaled_aug_raw,
        kkt.aug_raw,
        kkt.scaling_factor,
        n, m,
    )
    transfer!(kkt.aug_com, kkt.scaled_aug_raw, kkt.aug_csc_map)
end

function regularize_diagonal!(kkt::ScaledSparseKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal .* kkt.scaling_factor.^2
    kkt.du_diag .-= dual
end

