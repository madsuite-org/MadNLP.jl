
using MadNLP
using LinearAlgebra
import MadNLP: primal_dual, primal, dual, dual_lb, dual_ub

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

function MadNLP.create_kkt_system(
    ::Type{DiagonalHessianKKTSystem},
    cb::MadNLP.SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T,VT}
    n = cb.nvar
    m = cb.ncon

    # ind_cons stores the indexes of all constraints
    n_slack = length(cb.ind_ineq)
    n_tot = n + n_slack

    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)
    ind_ineq = cb.ind_ineq

    #=
        Import Jacobian's sparsity pattern
    =#
    jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    #=
        Determine size of condensed KKT system
    =#
    n_hess = n_tot # Diagonal Hessian!
    n_jac = length(jac_sparsity_I)
    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    #=
        Build augmented KKT system
    =#
    I = MadNLP.create_array(cb, Int32, aug_mat_length)
    J = MadNLP.create_array(cb, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] .= 1:n_tot # diagonal Hessian!
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] .= 1:n_tot # diagonal Hessian!
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    #=
        Diagonal matrices.
    =#
    pr_diag = MadNLP._madnlp_unsafe_wrap(V, n_tot)
    du_diag = VT(undef, m)
    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    #=
        Nonzeroes values for Hessian and Jacobian
    =#
    hess = VT(undef, cb.nnzh)
    jac = MadNLP._madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = MadNLP._madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    diag_hess = MadNLP._madnlp_unsafe_wrap(V, n_hess, n_tot+1)

    #=
        Build condensed matrix and Jacobian in COO format
    =#
    aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = MadNLP.SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    #=
        Build condensed matrix and Jacobian in CSC format for linear solver
    =#
    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)
    jac_com, jac_csc_map = MadNLP.coo_to_csc(jac_raw)

    #=
        Initialize linear solver
    =#
    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )
    quasi_newton = MadNLP.ExactHessian{Float64, Vector{Float64}}()

    return DiagonalHessianKKTSystem(
        hess, jac_callback, jac, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        aug_raw, aug_com, aug_csc_map,
        diag_hess,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        n, n_slack, n_tot,
        ind_ineq, cb.ind_lb, cb.ind_ub,
        quasi_newton,
    )
end

# Getters
MadNLP.num_variables(kkt::DiagonalHessianKKTSystem) = length(kkt.diag_hess)
MadNLP.get_kkt(kkt::DiagonalHessianKKTSystem) = kkt.aug_com
MadNLP.get_jacobian(kkt::DiagonalHessianKKTSystem) = kkt.jac_callback
function MadNLP.jtprod!(y::AbstractVector, kkt::DiagonalHessianKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

function MadNLP.initialize!(kkt::DiagonalHessianKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(kkt.diag_hess, 0.)
end


function MadNLP.compress_jacobian!(kkt::DiagonalHessianKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    MadNLP.transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
    return
end

function MadNLP.compress_hessian!(kkt::DiagonalHessianKKTSystem)
    kkt.diag_hess .= 1.0
    return
end

function MadNLP.build_kkt!(kkt::DiagonalHessianKKTSystem)
    MadNLP.transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end

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

function MadNLP.solve_kkt!(kkt::DiagonalHessianKKTSystem, w::MadNLP.AbstractKKTVector)
    MadNLP.reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)
    MadNLP.solve_linear_system!(kkt.linear_solver, primal_dual(w))
    MadNLP.finish_aug_solve!(kkt, w)
    return w
end
