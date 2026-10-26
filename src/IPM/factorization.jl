function solve_refine_wrapper!(d, solver, p, w)
    result = false

    get_cnt(solver).linear_solver_time += @elapsed begin
        if solve_refine!(d, get_iterator(solver), p, w)
            result = true
        else
            if improve!(get_kkt(solver).linear_solver)
                if solve_refine!(d, get_iterator(solver), p, w)
                    result = true
                end
            end
        end
    end
    # Get number of iterations in Richardson's iterative refinement
    iter_richardson = get_cnt(solver).ir
    get_cnt(solver).backsolve_cnt += iter_richardson
    return result
end

function factorize_wrapper!(solver::AbstractMadNLPSolver)
    @trace(get_logger(solver),"Factorization started.")
    build_kkt!(get_kkt(solver))
    get_cnt(solver).linear_solver_time += @elapsed factorize!(get_kkt(solver).linear_solver)
    get_cnt(solver).factorization_cnt += 1
    return
end

function solve_kkt!(kkt::SparseUnreducedKKTSystem, w::AbstractKKTVector)
    wzl = dual_lb(w)
    wzu = dual_ub(w)
    f(x,y) = iszero(y) ? x : x/y
    wzl .= f.(wzl, kkt.l_lower_aug)
    wzu .= f.(wzu, kkt.u_lower_aug)
    solve_linear_system!(kkt.linear_solver, full(w))
    wzl .*= .-kkt.l_lower_aug
    wzu .*= kkt.u_lower_aug
    return w
end

function solve_kkt!(kkt::AbstractReducedKKTSystem, w::AbstractKKTVector)
    reduce_rhs!(kkt, w)
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    finish_aug_solve!(kkt, w)
    return w
end

function solve_kkt!(kkt::ScaledSparseKKTSystem, w::AbstractKKTVector)
    r3 = kkt.buffer1
    r4 = kkt.buffer2
    fill!(r3, 0.0)
    fill!(r4, 0.0)

    wzl = dual_lb(w)  # size nlb
    wzu = dual_ub(w)  # size nub

    r3[kkt.ind_lb] .= wzl
    r3[kkt.ind_ub] .*= sqrt.(kkt.u_diag)
    r3[kkt.ind_lb] ./= sqrt.(kkt.l_diag)
    r4[kkt.ind_ub] .= wzu
    r4[kkt.ind_lb] .*= sqrt.(kkt.l_diag)
    r4[kkt.ind_ub] ./= sqrt.(kkt.u_diag)
    # Build RHS
    w.xp .*= kkt.scaling_factor
    w.xp .+= (r3 .+ r4)
    # Backsolve
    solve_linear_system!(kkt.linear_solver, primal_dual(w))
    # Unpack solution
    w.xp .*= kkt.scaling_factor

    wzl .= (wzl .- kkt.l_lower .* w.xp_lr) ./ kkt.l_diag
    wzu .= (.-wzu .+ kkt.u_lower .* w.xp_ur) ./ kkt.u_diag
    return w
end

function solve_kkt!(
    kkt::SparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}

    qn = kkt.quasi_newton
    n, p = size(qn)
    # Load buffers
    xr = qn._w2
    Tk = qn.Tk
    w_ = primal_dual(w)
    nn = length(w_)

    fill!(Tk, zero(T))
    reduce_rhs!(kkt, w)

    # Resize arrays with correct dimension
    if size(qn.E) != (nn, 2*p)
        qn.E = zeros(T, nn, 2*p)
        qn.H = zeros(T, nn, 2*p)
    else
        fill!(qn.E, zero(T))
        fill!(qn.H, zero(T))
    end

    # Solve LBFGS system with Sherman-Morrison-Woodbury formula
    # (C + E P Eᵀ)⁻¹ = C⁻¹ - C⁻¹ E (P + Eᵀ C⁻¹ E) Eᵀ C⁻¹
    #
    # P = [ -Iₚ  0  ] (size 2p × 2p) and E = [ U  V ] (size (n+m) × 2p)
    #     [  0   Iₚ ]                        [ 0  0 ]

    # Solve linear system without low-rank part
    solve_linear_system!(kkt.linear_solver, w_)  # w_ stores the solution of Cx = b

    # Add low-rank correction
    if p > 0
        @inbounds for i in 1:n, j in 1:p
            qn.E[i, j] = qn.U[i, j]
            qn.E[i, j+p] = qn.V[i, j]
        end
        copyto!(qn.H, qn.E)

        solve_linear_system!(kkt.linear_solver, qn.H)  # H = C⁻¹ E

        for i = 1:p
            Tk[i,i] = -one(T)                  # Tₖ = P
            Tk[i+p,i+p] = one(T)
        end
        mul!(Tk, qn.E', qn.H, one(T), one(T))  # Tₖ = (P + Eᵀ C⁻¹ E)
        mul!(xr, qn.E', w_)                    # xᵣ = Eᵀ C⁻¹ b

        if T <: BlasReal
            F, ipiv, info = LAPACK.sytrf!('L', Tk)  # Tₖ⁻¹
            LAPACK.sytrs!('L', F, ipiv, xr)         # xᵣ = (P + Eᵀ C⁻¹ E)⁻¹ Eᵀ C⁻¹ b
        else
            F = bunchkaufman!(Symmetric(Tk, :L))    # Tₖ⁻¹
            ldiv!(F, xr)                            # xᵣ = (P + Eᵀ C⁻¹ E)⁻¹ Eᵀ C⁻¹ b
        end

        mul!(w_, qn.H, xr, -one(T), one(T))    # x = x - C⁻¹ E xᵣ
    end

    finish_aug_solve!(kkt, w)
    return w
end


function solve_kkt!(kkt::SparseCondensedKKTSystem{T}, w::AbstractKKTVector) where {T}

    (n, m) = size(kkt.jt_csc)
    b = kkt.border
    mD = num_border_rows(b)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), (n + 1):(n + m))
    wz = view(full(w), (n + m + 1):(n + 2 * m))
    Σs = view(kkt.pr_diag, (n + 1):(n + m))

    reduce_rhs!(kkt, w)

    if mD == 0
        kkt.buffer .= kkt.diag_buffer .* (wz .+ ws ./ Σs)

        mul!(wx, kkt.jt_csc, kkt.buffer, one(T), one(T))
        solve_linear_system!(kkt.linear_solver, wx)

        mul!(kkt.buffer2, kkt.jt_csc', wx) # TODO: investigate why directly using wz here is causing an error

        wz .= .- kkt.buffer .+ kkt.diag_buffer .* kkt.buffer2
        ws .= (ws .+ wz) ./ Σs

        finish_aug_solve!(kkt, w)
        return w
    end

    # folded right-hand side r = wz + ws / Σs of every row; the condensed rows add
    # J_Sᵀ D_S r_S to the primal block, the kept rows enter the border
    kkt.buffer .= wz .+ ws ./ Σs
    kkt.buffer3 .= kkt.diag_buffer .* kkt.buffer
    view(kkt.buffer3, b.ind_D) .= zero(T)
    pdx = view(b.pd_buffer, 1:n)
    pdy = view(b.pd_buffer, (n + 1):(n + mD))
    pdx .= wx
    mul!(pdx, kkt.jt_csc, kkt.buffer3, one(T), one(T))
    pdy .= view(kkt.buffer, b.ind_D)

    solve_linear_system!(kkt.linear_solver, b.pd_buffer)

    wx .= pdx
    # multipliers: recovered for the condensed rows, solved for the kept rows
    mul!(kkt.buffer2, kkt.jt_csc', wx)
    view(wz, b.ind_S) .= view(kkt.diag_buffer, b.ind_S) .* (view(kkt.buffer2, b.ind_S) .- view(kkt.buffer, b.ind_S))
    view(wz, b.ind_D) .= pdy
    ws .= (ws .+ wz) ./ Σs

    finish_aug_solve!(kkt, w)
    return w
end

function solve_kkt!(
    kkt::SparseUnreducedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseUnreducedKKTSystem. Please use SparseKKTSystem instead.")
end

function solve_kkt!(
    kkt::SparseCondensedKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation SparseCondensedKKTSystem. Please use SparseKKTSystem instead.")
end

function solve_kkt!(
    kkt::ScaledSparseKKTSystem{T, VT, MT, QN},
    w::AbstractKKTVector
    ) where {T, VT, MT, QN<:CompactLBFGS}
    error("Quasi-Newton approximation of the Hessian is not supported by the KKT formulation ScaledSparseKKTSystem. Please use SparseKKTSystem instead.")
end

function solve_kkt!(
    kkt::DenseCondensedKKTSystem,
    w::AbstractKKTVector{T},
    ) where T

    n = num_variables(kkt)
    n_eq, ns = kkt.n_eq, kkt.n_ineq
    n_condensed = n + n_eq

    # Decompose rhs
    wx = view(full(w), 1:n)
    ws = view(full(w), n+1:n+ns)
    wy = view(full(w), kkt.ind_eq_shifted)
    wz = view(full(w), kkt.ind_ineq_shifted)

    x = kkt.pd_buffer
    xx = view(x, 1:n)
    xy = view(x, n+1:n+n_eq)

    Σs = get_slack_regularization(kkt)

    reduce_rhs!(kkt, w)

    fill!(kkt.buffer, zero(T))
    kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wz .+ ws ./ Σs)
    mul!(xx, kkt.jac', kkt.buffer)
    xx .+= wx
    xy .= wy
    solve_linear_system!(kkt.linear_solver, x)

    wx .= xx
    mul!(dual(w), kkt.jac, wx)
    wy .= xy
    wz .*= kkt.diag_buffer
    dual(w) .-= kkt.buffer
    ws .= (ws .+ wz) ./ Σs

    finish_aug_solve!(kkt, w)
    return w
end

function mul!(w::AbstractKKTVector{T}, kkt::Union{SparseKKTSystem{T,VT,MT,QN},SparseUnreducedKKTSystem{T,VT,MT,QN}}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T, VT, MT, QN<:ExactHessian}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

function mul!(w::AbstractKKTVector{T}, kkt::ScaledSparseKKTSystem{T,VT,MT,QN}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T, VT, MT, QN<:ExactHessian}
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    # Custom reduction
    primal(w) .+= alpha .* kkt.reg .* primal(x)
    dual(w) .+= alpha .* kkt.du_diag .* dual(x)
    w.xp_lr .-= alpha .* dual_lb(x)
    w.xp_ur .+= alpha .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (x.xp_lr .* kkt.l_lower .+ dual_lb(x) .* kkt.l_diag)
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* (x.xp_ur .* kkt.u_lower .- dual_ub(x) .* kkt.u_diag)
    return w
end

function mul!(w::AbstractKKTVector{T}, kkt::Union{SparseKKTSystem{T,VT,MT,QN},SparseUnreducedKKTSystem{T,VT,MT,QN}}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T, VT, MT, QN<:CompactLBFGS}
    qn = kkt.quasi_newton
    n, p = size(qn)
    nn = length(primal_dual(w))
    # Load buffers (size: 2p)
    vx = qn._w2
    # Reset E
    fill!(qn.E, zero(T))
    @inbounds for i in 1:n, j in 1:p
        qn.E[i, j] = qn.U[i, j]
        qn.E[i, j+p] = qn.V[i, j]
    end
    # Upper-left block is B = ξI - UUᵀ + VVᵀ
    mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    mul!(vx, qn.E', primal_dual(x))
    @inbounds for k in 1:p
        vx[k] = -vx[k]
    end
    mul!(primal_dual(w), qn.E, vx, alpha, one(T))

    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
end

function mul!(w::AbstractKKTVector{T}, kkt::SparseCondensedKKTSystem, x::AbstractKKTVector, alpha, beta) where T
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), n+1:n+m)
    xz = view(full(x), n+m+1:n+2*m)

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+m)
    wz = view(full(w), n+m+1:n+2*m)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta) # TODO: make this symmetric

    mul!(wx, kkt.jt_csc,  xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    axpy!(-alpha, xs, wz)
    ws .= beta.*ws .- alpha.* xz

    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

function mul!(w::AbstractKKTVector{T}, kkt::AbstractDenseKKTSystem, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    (m, n) = size(kkt.jac)
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)
    wz = @view(dual(w)[kkt.ind_ineq])

    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])

    _symv!('L', alpha, kkt.hess, xx, beta, wx)
    if m > 0  # otherwise, CUDA causes an error
        mul!(wx, kkt.jac', dual(x), alpha, one(T))
        mul!(wy, kkt.jac,  xx, alpha, beta)
    end
    ws .= beta.*ws .- alpha.* xz
    wz .-= alpha.* xs
    _kktmul!(w,x,kkt.reg,kkt.du_diag,kkt.l_lower,kkt.u_lower,kkt.l_diag,kkt.u_diag, alpha, beta)
    return w
end

function mul_hess_blk!(wx, kkt::Union{DenseKKTSystem,DenseCondensedKKTSystem}, t)
    n = size(kkt.hess, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end

function mul_hess_blk!(wx, kkt::Union{SparseKKTSystem,SparseCondensedKKTSystem}, t)
    n = size(kkt.hess_com, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_com, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end

function mul_hess_blk!(wx, kkt::SparseUnreducedKKTSystem, t)
    ind_lb = kkt.ind_lb
    ind_ub = kkt.ind_ub

    n = size(kkt.hess_com, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess_com, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
    wx[ind_lb] .-= @view(t[ind_lb]) .* (kkt.l_lower ./ kkt.l_diag)
    wx[ind_ub] .-= @view(t[ind_ub]) .* (kkt.u_lower ./ kkt.u_diag)
end
