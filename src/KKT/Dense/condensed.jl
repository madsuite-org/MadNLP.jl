
"""
    DenseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement [`AbstractCondensedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to factorize the associated KKT system (otherwise an error is returned).

"""
struct DenseCondensedKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int}
    } <: AbstractCondensedKKTSystem{T, VT, MT, QN}

    hess::MT
    jac::MT
    quasi_newton::QN
    jac_ineq::MT

    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    pd_buffer::VT
    diag_buffer::VT
    buffer::VT
    # KKT system
    aug_com::MT
    # Info
    n_eq::Int
    ind_eq::VI
    ind_eq_shifted::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    ind_ineq_shifted::VI
    # Linear Solver
    linear_solver::LS
    # Buffers
    etc::Dict{Symbol, Any}
end

function create_kkt_system(
    ::Type{DenseCondensedKKTSystem},
    cb::AbstractCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ns = length(cb.ind_ineq)
    n_eq = m - ns
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    aug_com  = create_array(cb, n+m-ns, n+m-ns)
    hess     = create_array(cb, n, n)
    jac      = create_array(cb, m, n)
    jac_ineq = create_array(cb, ns, n)

    reg  = VT(undef, n+ns)
    pr_diag  = VT(undef, n+ns)
    du_diag  = VT(undef, m)
    l_diag = fill!(VT(undef, nlb), one(T))
    u_diag = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    pd_buffer = VT(undef, n + n_eq)
    diag_buffer = VT(undef, ns)
    buffer = VT(undef, m)

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # Shift indexes to avoid additional allocation in views
    ind_eq_shifted = cb.ind_eq .+ n .+ ns
    ind_ineq_shifted = cb.ind_ineq .+ n .+ ns

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver, pos = n + m - ns)

    return DenseCondensedKKTSystem(
        hess, jac, quasi_newton, jac_ineq,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        pd_buffer, diag_buffer, buffer,
        aug_com,
        n_eq, cb.ind_eq, ind_eq_shifted,
        ns,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        ind_ineq_shifted,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end

num_variables(kkt::DenseCondensedKKTSystem) = size(kkt.hess, 1)

function get_slack_regularization(kkt::DenseCondensedKKTSystem)
    n, ns = num_variables(kkt), kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns)
end

function _build_condensed_kkt_system!(
    dest::AbstractMatrix, hess::AbstractMatrix, jac::AbstractMatrix,
    pr_diag::AbstractVector, du_diag::AbstractVector, ind_eq::AbstractVector, n, m_eq,
)
    # Transfer Hessian
    @inbounds for i in 1:n, j in 1:i
        if i == j
            dest[i, i] += pr_diag[i] + hess[i, i]
        else
            dest[i, j] += hess[i, j]
            dest[j, i] += hess[j, i]
        end
    end
    # Transfer Jacobian / variables
    @inbounds for i in 1:m_eq, j in 1:n
        is = ind_eq[i]
        dest[i + n, j] = jac[is, j]
        dest[j, i + n] = jac[is, j]
    end
    # Transfer dual regularization
    @inbounds for i in 1:m_eq
        is = ind_eq[i]
        dest[i + n, i + n] = du_diag[is]
    end
end

function _build_ineq_jac!(
    dest::AbstractMatrix, jac::AbstractMatrix, diag_buffer::AbstractVector,
    ind_ineq::AbstractVector,
    n, m_ineq,
)
    @inbounds for i in 1:m_ineq, j in 1:n
        is = ind_ineq[i]
        dest[i, j] = jac[is, j] * sqrt(diag_buffer[i])
    end
end

function build_kkt!(kkt::DenseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    ns = kkt.n_ineq
    n_eq = length(kkt.ind_eq)
    m = size(kkt.jac, 1)

    fill!(kkt.aug_com, zero(T))

    # Build √Σₛ * J
    Σs = view(kkt.pr_diag, n+1:n+ns)
    Σd = @view(kkt.du_diag[kkt.ind_ineq])
    kkt.diag_buffer .= Σs ./ ( 1 .- Σd .* Σs)
    _build_ineq_jac!(kkt.jac_ineq, kkt.jac, kkt.diag_buffer, kkt.ind_ineq, n, ns)

    # Select upper-left block
    W = if n_eq > 0
        view(kkt.aug_com, 1:n, 1:n) # TODO: does not work on GPU
    else
        kkt.aug_com
    end
    # Build J' * Σₛ * J
    mul!(W, kkt.jac_ineq', kkt.jac_ineq)


    _build_condensed_kkt_system!(
        kkt.aug_com, kkt.hess, kkt.jac,
        kkt.pr_diag, kkt.du_diag,
        kkt.ind_eq, n, kkt.n_eq,
    )
end

# TODO: check how to handle inertia with the condensed form
function is_inertia_correct(kkt::DenseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0 && num_neg == kkt.n_eq)
end

# For inertia-free regularization
function _mul_expanded!(y::AbstractVector, kkt::DenseCondensedKKTSystem{T}, x::AbstractVector) where T
    n = size(kkt.hess, 1)
    ns = kkt.n_ineq
    m = size(kkt.jac, 1)

    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, 1+n:n+ns)
    Σd = kkt.du_diag

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, 1+n:n+ns)
    xy = view(x, 1+n+ns:n+ns+m)

    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, 1+n:n+ns)
    yy = view(y, 1+n+ns:n+ns+m)

    # / x (variable)
    yx .= Σx .* xx
    _symv!('L', one(T), kkt.hess, xx, zero(T), yx)
    mul!(yx, kkt.jac', xy, one(T), one(T))

    # / s (slack)
    ys .= Σs .* xs
    ys .-= xy[kkt.ind_ineq]

    # / y (multiplier)
    yy .= Σd .* xy
    mul!(yy, kkt.jac, xx, one(T), one(T))
    yy[kkt.ind_ineq] .-= xs
    return
end

function mul!(y::AbstractVector, kkt::DenseCondensedKKTSystem{T}, x::AbstractVector) where T
    # TODO: implement properly with AbstractKKTRHS
    if length(y) == length(x) == size(kkt.aug_com, 1)
        _symv!('L', one(T), kkt.aug_com, x, zero(T), y)
    else
        _mul_expanded!(y, kkt, x)
    end
end

function jprod_ineq!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_ineq, x)
end

