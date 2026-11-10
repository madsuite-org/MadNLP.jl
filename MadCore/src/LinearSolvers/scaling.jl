# Scaling and diagonal shifts of the matrix a sparse KKT system hands to its linear solver.

"""
    create_kkt_linear_solver(linear_solver, A, n_primal; opt, logger, kkt_scaler, primal_shift, dual_shift)

Create the linear solver of a sparse KKT system for its matrix `A` (lower triangle, CSC), whose
first `n_primal` rows and columns are the primal block. With a scaler or a nonzero shift, the
solver factorizes

```math
W A W + \\begin{bmatrix} σ_p I & 0 \\\\ 0 & -σ_d I \\end{bmatrix}
```

(`σ_p = primal_shift`, `σ_d = dual_shift`) instead of `A`; without, it factorizes `A` itself.
"""
function create_kkt_linear_solver(
        linear_solver, A, n_primal;
        opt = default_options(linear_solver), logger = MadNLPLogger(),
        kkt_scaler::AbstractScaler = NoScaler(), primal_shift = 0.0, dual_shift = 0.0,
    )
    set_block_structure!(opt, n_primal, size(A, 1) - n_primal)
    if kkt_scaler isa NoScaler && iszero(primal_shift) && iszero(dual_shift)
        return create_linear_solver(linear_solver, A; opt, logger)
    end
    return ScaledShiftedSolver(linear_solver, A, n_primal; opt, logger, kkt_scaler, primal_shift, dual_shift)
end

"""
    ScaledShiftedSolver

The linear solver `create_kkt_linear_solver` returns for a scaled or shifted
factorization: it keeps the scaled and shifted copy of the KKT matrix, factorizes it with the
selected solver, and reads the inertia from that factorization.
"""
mutable struct ScaledShiftedSolver{T, MT, VT, VI, LS, SC <: AbstractScaler} <: AbstractLinearSolver{T}
    A::MT                    # the KKT matrix, read only
    fac::MT                  # its scaled and shifted copy, factorized by `inner`
    inner::LS
    scaler::SC
    primal_shift::T
    dual_shift::T
    n_primal::Int
    d::VT                    # the scaling W
    dnew::VT
    absnz::VT                # |A| on the pattern with both triangles
    fmap::VI                 # nonzero of that pattern -> nonzero of A
    fcolptr::VI
    frow::VI
    long_cols::VI            # columns of that pattern with more than SCALING_LONG_COLUMN nonzeros
    arow::VI                 # row and column of each nonzero of A
    acol::VI
    adiag::VI                # nonzero of each diagonal entry of A
    logger::MadNLPLogger
end

# The sparse matrix types a ScaledShiftedSolver handles; a GPU package adds its own.
scaling_nzval(A::SparseMatrixCSC) = nonzeros(A)
scaling_index_type(::SparseMatrixCSC{T, Ti}) where {T, Ti} = Ti
scaling_host_pattern(A::SparseMatrixCSC) = (Vector{Int}(getcolptr(A)), Vector{Int}(rowvals(A)))
scaling_device(::SparseMatrixCSC, v::Vector) = v

function ScaledShiftedSolver(
        linear_solver, A, n_primal;
        opt, logger, kkt_scaler, primal_shift, dual_shift,
    )
    T = eltype(scaling_nzval(A))
    n = size(A, 1)
    colptr, rows = scaling_host_pattern(A)
    cols = similar(rows)
    adiag = zeros(Int, n)
    for j in 1:n, k in colptr[j]:(colptr[j + 1] - 1)
        cols[k] = j
        rows[k] == j && (adiag[j] = k)
    end
    any(iszero, adiag) && error(logger, "a scaled or shifted factorization needs every diagonal entry of the KKT matrix in its pattern")
    # the pattern with both triangles, each nonzero mapped to its nonzero of A
    off = rows .!= cols
    S = sparse(vcat(rows, cols[off]), vcat(cols, rows[off]), vcat(1:length(rows), findall(off)), n, n)
    Ti = scaling_index_type(A)
    dv(v) = scaling_device(A, v)
    fac = copy(A)
    inner = create_linear_solver(linear_solver, fac; opt, logger)
    return ScaledShiftedSolver(
        A, fac, inner, kkt_scaler, T(primal_shift), T(dual_shift), n_primal,
        dv(ones(T, n)), dv(ones(T, n)), dv(zeros(T, nnz(S))), dv(Ti.(nonzeros(S))), dv(Ti.(S.colptr)),
        dv(Ti.(S.rowval)), dv(Ti.(findall(>(SCALING_LONG_COLUMN), diff(S.colptr)))), dv(Ti.(rows)), dv(Ti.(cols)), dv(Ti.(adiag)), logger,
    )
end

# A GPU sweep gives each column with more nonzeros than this a work group of its own.
const SCALING_LONG_COLUMN = 1024

# One sweep of symmetric infinity-norm scaling over the pattern with both triangles, stored by
# columns: dnew_j = d_j / sqrt(max_i |A_ij| d_i d_j). On the host every column is handled
# alike and `long_cols` is not used.
function scaling_sweep!(dnew::Vector, d, colptr, rowval, absnz, long_cols)
    for j in eachindex(dnew)
        r = zero(eltype(d))
        for k in colptr[j]:(colptr[j + 1] - 1)
            r = max(r, absnz[k] * d[rowval[k]])
        end
        r *= d[j]
        dnew[j] = r > 0 ? d[j] / sqrt(r) : d[j]
    end
    return dnew
end

function update_scaling!(M::ScaledShiftedSolver{T}, ::JacobiScaler) where {T}
    a = view(scaling_nzval(M.A), M.adiag)
    M.d .= ifelse.(a .== 0, one(T), one(T) ./ sqrt.(abs.(a)))
    return
end

function update_scaling!(M::ScaledShiftedSolver{T}, scaler::AbstractScaler) where {T}
    fill!(M.d, one(T))
    sweeps = scaler_sweeps(scaler)
    sweeps > 0 || return
    M.absnz .= abs.(view(scaling_nzval(M.A), M.fmap))
    for _ in 1:sweeps
        scaling_sweep!(M.dnew, M.d, M.fcolptr, M.frow, M.absnz, M.long_cols)
        M.d, M.dnew = M.dnew, M.d
    end
    return
end

# AugmentedRuizScaler: the scaling is set by the KKT system before factorize! (update_kkt_scaling!)
update_scaling!(M::ScaledShiftedSolver, ::AugmentedRuizScaler) = nothing

update_kkt_scaling!(kkt, ls::LinearSolverBarrier) = update_kkt_scaling!(kkt, getfield(ls, :wrapped))
update_kkt_scaling!(kkt::SparseCondensedKKTSystem, M::ScaledShiftedSolver) =
    M.scaler isa AugmentedRuizScaler ? augmented_ruiz_scaling!(M, kkt, M.scaler.sweeps, M.scaler.bounds) : nothing

# Ruiz sweeps on the augmented system [H + Σx  Jᵀ; J  a] of a condensed KKT system, a = Σd - Σs⁻¹ on
# the inequality rows and Σd on the others; Wx scales the condensed block and Wy the border rows.
function augmented_ruiz_scaling!(M::ScaledShiftedSolver{T}, kkt::SparseCondensedKKTSystem, sweeps, bounds = :folded) where {T}
    M.d isa Vector || error("AugmentedRuizScaler runs on the host only")
    H = kkt.hess_com
    Jt = kkt.jt_csc
    n, m = size(Jt)
    ns = length(kkt.ind_ineq)
    a = copy(Array(kkt.du_diag))
    Σs = Array(view(kkt.pr_diag, (n + 1):(n + ns)))
    view(a, Array(kkt.ind_ineq)) .-= one(T) ./ Σs
    sym = bounds === :symmetrized
    bounds in (:folded, :symmetrized) || error("AugmentedRuizScaler: bounds is :folded or :symmetrized")
    # symmetrized: the primal diagonal keeps only the regularization; each bound of a variable (not of
    # a slack) is a row with coupling √z and diagonal |x - xˡ| (|xᵘ - x|)
    p = sym ? Array(view(kkt.reg, 1:n)) : Array(view(kkt.pr_diag, 1:n))
    lb = sym ? Array(kkt.ind_lb) : Int[]
    ub = sym ? Array(kkt.ind_ub) : Int[]
    zl = sym ? sqrt.(Array(kkt.l_lower)) : T[]
    zu = sym ? sqrt.(Array(kkt.u_lower)) : T[]
    gl = sym ? abs.(Array(kkt.l_diag)) : T[]
    gu = sym ? abs.(Array(kkt.u_diag)) : T[]
    dl = ones(T, length(lb)); rl = similar(dl)
    du = ones(T, length(ub)); ru = similar(du)
    dx = ones(T, n)
    dy = ones(T, m)
    rx = zeros(T, n)
    ry = zeros(T, m)
    for _ in 1:sweeps
        rx .= abs.(p) .* dx .^ 2
        ry .= abs.(a) .* dy .^ 2
        rl .= gl .* dl .^ 2
        ru .= gu .* du .^ 2
        for (k, i) in enumerate(lb)
            i <= n || continue
            v = zl[k] * dx[i] * dl[k]
            rx[i] = max(rx[i], v); rl[k] = max(rl[k], v)
        end
        for (k, i) in enumerate(ub)
            i <= n || continue
            v = zu[k] * dx[i] * du[k]
            rx[i] = max(rx[i], v); ru[k] = max(ru[k], v)
        end
        dl .= ifelse.(rl .> 0, dl ./ sqrt.(rl), dl)
        du .= ifelse.(ru .> 0, du ./ sqrt.(ru), du)
        for j in 1:n, k in H.colptr[j]:(H.colptr[j + 1] - 1)   # H holds one triangle
            i = H.rowval[k]
            v = abs(H.nzval[k]) * dx[i] * dx[j]
            i == j && (v = abs(H.nzval[k] + p[i]) * dx[i]^2)
            rx[i] = max(rx[i], v)
            rx[j] = max(rx[j], v)
        end
        for c in 1:m, k in Jt.colptr[c]:(Jt.colptr[c + 1] - 1)
            i = Jt.rowval[k]
            v = abs(Jt.nzval[k]) * dx[i] * dy[c]
            rx[i] = max(rx[i], v)
            ry[c] = max(ry[c], v)
        end
        dx .= ifelse.(rx .> 0, dx ./ sqrt.(rx), dx)
        dy .= ifelse.(ry .> 0, dy ./ sqrt.(ry), dy)
    end
    view(M.d, 1:n) .= dx
    ind_D = kkt.border.ind_D
    for (t, c) in enumerate(Array(ind_D))
        M.d[n + t] = dy[c]
    end
    return
end

function factorize!(M::ScaledShiftedSolver{T}) where {T}
    update_scaling!(M, M.scaler)
    scaling_nzval(M.fac) .= scaling_nzval(M.A) .* view(M.d, M.arow) .* view(M.d, M.acol)
    np = M.n_primal
    iszero(M.primal_shift) || (view(scaling_nzval(M.fac), view(M.adiag, 1:np)) .+= M.primal_shift)
    iszero(M.dual_shift) || (view(scaling_nzval(M.fac), view(M.adiag, (np + 1):length(M.adiag))) .-= M.dual_shift)
    factorize!(M.inner)
    return M
end

# x = W (W A W + shifts)⁻¹ W b
function solve_linear_system!(M::ScaledShiftedSolver, b::AbstractVector)
    b .*= M.d
    solve_linear_system!(M.inner, b)
    b .*= M.d
    return b
end

is_inertia(M::ScaledShiftedSolver) = is_inertia(M.inner)
inertia(M::ScaledShiftedSolver) = inertia(M.inner)
improve!(M::ScaledShiftedSolver) = improve!(M.inner)
is_factorized(M::ScaledShiftedSolver) = is_factorized(M.inner)
is_async(M::ScaledShiftedSolver) = is_async(M.inner)
introduce(M::ScaledShiftedSolver) = "$(introduce(M.inner)) (scaled and shifted)"
