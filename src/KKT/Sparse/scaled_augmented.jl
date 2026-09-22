"""
    ScaledSparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Scaled version of the [`AbstractReducedKKTSystem`](@ref) (using the K2.5 formulation introduced in [GOS]).

The K2.5 formulation of the augmented KKT system has a better conditioning
than the original (K2) formulation. It is recommend switching to a `ScaledSparseKKTSystem`
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
function create_kkt_system(
    ::Type{ScaledSparseKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T,VT}

    n_slack = length(cb.ind_ineq)
    # Deduce KKT size.

    n = cb.nvar
    m = cb.ncon
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)

    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    ind_ineq = cb.ind_ineq

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack


    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = create_array(cb, Int32, aug_mat_length)
    J = create_array(cb, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    pr_diag = _madnlp_unsafe_wrap(V, n_tot)
    du_diag = _madnlp_unsafe_wrap(V, m, n_jac+n_slack+n_hess+n_tot+1)

    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    scaling_factor = VT(undef, n_tot)
    buffer1 = VT(undef, n_tot)
    buffer2 = VT(undef, n_tot)

    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )
    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )
    scaled_aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,copy(V))

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver, pos = n_tot
    )

    return ScaledSparseKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        scaling_factor, buffer1, buffer2,
        aug_raw, aug_com, aug_csc_map, scaled_aug_raw,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, cb.ind_lb, cb.ind_ub,
    )
end

num_variables(kkt::ScaledSparseKKTSystem) = length(kkt.pr_diag)
get_jacobian(kkt::ScaledSparseKKTSystem) = kkt.jac_callback

function initialize!(kkt::ScaledSparseKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(kkt.scaling_factor, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

function jtprod!(y::AbstractVector, kkt::ScaledSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

function compress_jacobian!(kkt::ScaledSparseKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
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

