
"""
    SparseUnreducedKKTSystem{T, VT, MT, QN} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractUnreducedKKTSystem`](@ref) in sparse COO format.

"""
struct SparseUnreducedKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}
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
    l_lower_aug::VT
    u_lower_aug::VT

    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

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

function create_kkt_system(
    ::Type{SparseUnreducedKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T, VT}
    ind_ineq = cb.ind_ineq
    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    n_slack = length(ind_ineq)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)
    # Deduce KKT size.
    n = cb.nvar
    m = cb.ncon

    # Quasi-newton
    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)

    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I = create_array(cb, Int32, cb.nnzh)
    hess_sparsity_J = create_array(cb, Int32, cb.nnzh)
    _hess_sparsity_wrapper!(cb,hess_sparsity_I,hess_sparsity_J)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack + 2*nlb + 2*nub
    aug_vec_length = n_tot + m + nlb + nub

    I = create_array(cb, Int32, aug_mat_length)
    J = create_array(cb, Int32, aug_mat_length)
    V = zeros(aug_mat_length)

    offset = n_tot + n_jac + n_slack + n_hess + m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac].=(jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack].=(ind_ineq.+n_tot)
    I[n_tot+n_hess+n_jac+n_slack+1:offset].=(n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset].=(n_tot+1:n_tot+m)

    I[offset+1:offset+nlb] .= (1:nlb).+(n_tot+m)
    I[offset+nlb+1:offset+2nlb] .= (1:nlb).+(n_tot+m)
    I[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n_tot+m+nlb)
    I[offset+2nlb+nub+1:offset+2nlb+2nub] .= (1:nub).+(n_tot+m+nlb)
    J[offset+1:offset+nlb] .= (1:nlb).+(n_tot+m)
    J[offset+nlb+1:offset+2nlb] .= ind_lb
    J[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n_tot+m+nlb)
    J[offset+2nlb+nub+1:offset+2nlb+2nub] .= ind_ub

    pr_diag = _madnlp_unsafe_wrap(V,n_tot)
    du_diag = _madnlp_unsafe_wrap(V,m, n_jac + n_slack+n_hess+n_tot+1)

    l_diag = _madnlp_unsafe_wrap(V, nlb, offset+1)
    u_diag = _madnlp_unsafe_wrap(V, nub, offset+2nlb+1)
    l_lower_aug = _madnlp_unsafe_wrap(V, nlb, offset+nlb+1)
    u_lower_aug = _madnlp_unsafe_wrap(V, nub, offset+2nlb+nub+1)
    reg = VT(undef, n_tot)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac + n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )
    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver, pos = n_tot)
    return SparseUnreducedKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower, l_lower_aug, u_lower_aug,
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, ind_lb, ind_ub,
    )
end

function initialize!(kkt::SparseUnreducedKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, -one(T))
    fill!(kkt.u_diag, -one(T))
    fill!(kkt.l_lower_aug, zero(T))
    fill!(kkt.u_lower_aug, zero(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

num_variables(kkt::SparseUnreducedKKTSystem) = length(kkt.pr_diag)

function build_kkt!(kkt::SparseUnreducedKKTSystem)
    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end

