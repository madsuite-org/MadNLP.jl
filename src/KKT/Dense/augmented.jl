
"""
    DenseKKTSystem{T, VT, MT, QN, VI} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement [`AbstractReducedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to be factorized (otherwise an error is returned).

"""
struct DenseKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int},
    } <: AbstractReducedKKTSystem{T, VT, MT, QN}

    hess::MT
    jac::MT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    diag_hess::VT
    # KKT system
    aug_com::MT
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    # Linear Solver
    linear_solver::LS
    # Buffers
    etc::Dict{Symbol, Any}
end

function create_kkt_system(
    ::Type{DenseKKTSystem},
    cb::AbstractCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T, VT}

    ind_ineq = cb.ind_ineq
    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    n = cb.nvar
    m = cb.ncon
    ns = length(ind_ineq)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    hess = create_array(cb, n, n)
    jac = create_array(cb, m, n)
    aug_com = create_array(cb, n+ns+m, n+ns+m)
    reg = create_array(cb, n+ns)
    pr_diag = create_array(cb, n+ns)
    du_diag = create_array(cb, m)
    diag_hess = create_array(cb, n)

    l_diag = fill!(VT(undef, nlb), one(T))
    u_diag = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(reg,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(diag_hess, zero(T))

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver, pos = n + ns)

    return DenseKKTSystem(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        diag_hess, aug_com,
        ind_ineq, cb.ind_lb, cb.ind_ub,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end

num_variables(kkt::DenseKKTSystem) = length(kkt.pr_diag)

function mul!(y::AbstractVector, kkt::DenseKKTSystem{T}, x::AbstractVector) where T
    _symv!('L', one(T), kkt.aug_com, x, zero(T), y)
end

# Special getters for Jacobian
function get_jacobian(kkt::DenseKKTSystem)
    n = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    return view(kkt.jac, :, 1:n)
end

function diag_add!(dest::AbstractMatrix, d1::AbstractVector, d2::AbstractVector)
    n = length(d1)
    @inbounds for i in 1:n
        dest[i, i] = d1[i] + d2[i]
    end
end

function _build_dense_kkt_system!(dest::VT, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq, n, m, ns) where {T, VT <: AbstractMatrix{T}}
    # Transfer Hessian
    for i in 1:n, j in 1:i
        if i == j
            dest[i, i] = pr_diag[i] + diag_hess[i]
        else
            dest[i, j] = hess[i, j]
            dest[j, i] = hess[j, i]
        end
    end
    # Transfer slack diagonal
    for i in 1:ns
        dest[i+n, i+n] = pr_diag[i+n]
    end
    # Transfer Jacobian / variables
    for i in 1:m, j in 1:n
        dest[i + n + ns, j] = jac[i, j]
        dest[j, i + n + ns] = jac[i, j]
    end
    # Transfer Jacobian / slacks
    for j in 1:ns
        is = ind_ineq[j]
        dest[is + n + ns, j + n] = - one(T)
        dest[j + n, is + n + ns] = - one(T)
    end
    # Transfer dual regularization
    for i in 1:m
        dest[i + n + ns, i + n + ns] = du_diag[i]
    end
end

function build_kkt!(kkt::DenseKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    m = size(kkt.jac, 1)
    ns = length(kkt.ind_ineq)

    _build_dense_kkt_system!(kkt.aug_com, kkt.hess, kkt.jac,
                                kkt.pr_diag, kkt.du_diag, kkt.diag_hess,
                                kkt.ind_ineq,
                                n, m, ns)
end

function compress_hessian!(kkt::DenseKKTSystem)
    # Transfer diagonal term for future regularization
    diag!(kkt.diag_hess, kkt.hess)
end

