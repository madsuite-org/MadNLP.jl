# Prototype: scaling of the condensed matrix from a larger system. The condensed matrix of a
# symmetrically scaled augmented (or unreduced) system is W_x K W_x, with W_x the primal part of that
# scaling, so these scalers scale K by the W_x of a Ruiz scaling computed on the augmented (x, s, y)
# matrix, with the equality shift on the equality rows, or on the symmetrized unreduced
# (x, s, y, zl, zu) matrix, whose bound rows couple through z^(1/2) and carry -(x - xˡ), -(xᵘ - x).
# SparseCondensedKKTSystem only; the KKT system sets the scaling when it builds its matrix.

"""
    AugmentedRuizScaler(sweeps = 4)

Prototype, `SparseCondensedKKTSystem` only: scales the condensed matrix by the primal part of a Ruiz
scaling of the augmented (x, s, y) KKT matrix.
"""
@kwdef struct AugmentedRuizScaler <: AbstractScaler
    sweeps::Int = 4
end

"""
    UnreducedRuizScaler(sweeps = 4)

Prototype, `SparseCondensedKKTSystem` only: scales the condensed matrix by the primal part of a Ruiz
scaling of the symmetrized unreduced (x, s, y, zl, zu) KKT matrix.
"""
@kwdef struct UnreducedRuizScaler <: AbstractScaler
    sweeps::Int = 4
end

const OuterScaler = Union{AugmentedRuizScaler, UnreducedRuizScaler}
scaler_sweeps(s::OuterScaler) = s.sweeps
update_scaling!(::ScaledShiftedSolver, ::OuterScaler) = nothing   # set by the KKT system (_outer_scaling!)

# The larger matrix: its pattern with both triangles by columns, and for each kind of nonzero its
# positions in that pattern and the index of its value in the KKT system's arrays.
struct OuterScaling{VT, VI}
    colptr::VI
    rowval::VI
    long_cols::VI
    absnz::VT
    d::VT
    dnew::VT
    dx::VT      # x and s diagonal values
    dy::VT      # y diagonal values
    hd::VT      # Hessian diagonal (n)
    hd_row::VI; hd_nz::VI
    pH::VI; sH::VI
    pJ::VI; sJ::VI
    pD::VI; sD::VI
    pC::VI
    pY::VI; sY::VI
    pLc::VI; sLc::VI
    pUc::VI; sUc::VI
    pLd::VI; sLd::VI
    pUd::VI; sUd::VI
end

const OUTER_SCALING = WeakKeyDict{Any, Any}()

_scaled_solver(s::LinearSolverBarrier) = _scaled_solver(s.wrapped)
_scaled_solver(s::ScaledShiftedSolver) = s
_scaled_solver(s) = nothing

function OuterScaling(kkt::SparseCondensedKKTSystem{T}, unreduced::Bool) where {T}
    H, Jt = kkt.hess_com, kkt.jt_csc
    n = size(H, 1); m = size(Jt, 2); ns = length(kkt.ind_ineq)
    ind_ineq = Array(kkt.ind_ineq); ind_eq = Array(kkt.ind_eq)
    ind_lb = Array(kkt.ind_lb); ind_ub = Array(kkt.ind_ub)
    nlb = unreduced ? length(ind_lb) : 0; nub = unreduced ? length(ind_ub) : 0
    N3 = n + ns + m; N = N3 + nlb + nub
    I = Int[]; J = Int[]; K = Int[]; S = Int[]   # entry: row, column, kind, source
    add!(i, j, k, s) = (push!(I, i); push!(J, j); push!(K, k); push!(S, s))
    hcol, hrow = scaling_host_pattern(H)
    hd_row = Int[]; hd_nz = Int[]
    for j in 1:n, p in hcol[j]:(hcol[j + 1] - 1)
        i = hrow[p]
        if i == j
            push!(hd_row, i); push!(hd_nz, p)
        else
            add!(i, j, 1, p); add!(j, i, 1, p)
        end
    end
    jcol, jrow = scaling_host_pattern(Jt)
    for k in 1:m, p in jcol[k]:(jcol[k + 1] - 1)
        i = jrow[p]
        add!(i, n + ns + k, 2, p); add!(n + ns + k, i, 2, p)
    end
    for i in 1:(n + ns)
        add!(i, i, 3, i)
    end
    for (j, k) in enumerate(ind_ineq)
        add!(n + j, n + ns + k, 4, 0); add!(n + ns + k, n + j, 4, 0)
    end
    for k in 1:m
        add!(n + ns + k, n + ns + k, 5, k)
    end
    if unreduced
        for (r, i) in enumerate(ind_lb)
            add!(N3 + r, i, 6, r); add!(i, N3 + r, 6, r); add!(N3 + r, N3 + r, 8, r)
        end
        for (r, i) in enumerate(ind_ub)
            add!(N3 + nlb + r, i, 7, r); add!(i, N3 + nlb + r, 7, r); add!(N3 + nlb + r, N3 + nlb + r, 9, r)
        end
    end
    A = sparse(I, J, collect(1:length(I)), N, N)
    ids = nonzeros(A)
    length(ids) == length(I) || error("outer scaling: duplicate entries in the pattern")
    pos = Dict{Int, Vector{Int}}(); src = Dict{Int, Vector{Int}}()
    for (p, id) in enumerate(ids)
        push!(get!(pos, K[id], Int[]), p); push!(get!(src, K[id], Int[]), S[id])
    end
    Ti = scaling_index_type(H)
    dv(v::Vector{<:Integer}) = scaling_device(H, Ti.(v))
    dv(v::Vector{<:AbstractFloat}) = scaling_device(H, T.(v))
    P(k) = dv(get(pos, k, Int[])); Q(k) = dv(get(src, k, Int[]))
    return OuterScaling(
        dv(A.colptr), dv(rowvals(A)), dv(findall(>(SCALING_LONG_COLUMN), diff(A.colptr))),
        dv(zeros(T, nnz(A))), dv(ones(T, N)), dv(ones(T, N)), dv(zeros(T, n + ns)), dv(zeros(T, m)), dv(zeros(T, n)),
        dv(hd_row), dv(hd_nz), P(1), Q(1), P(2), Q(2), P(3), Q(3), P(4), P(5), Q(5),
        P(6), Q(6), P(7), Q(7), P(8), Q(8), P(9), Q(9),
    )
end

# sets the scaling of the condensed factorization from the larger matrix (after build_kkt!)
function _outer_scaling!(kkt::SparseCondensedKKTSystem{T}) where {T}
    M = _scaled_solver(kkt.linear_solver)
    (M === nothing || !(M.scaler isa OuterScaler)) && return
    unreduced = M.scaler isa UnreducedRuizScaler
    O = get!(() -> OuterScaling(kkt, unreduced), OUTER_SCALING, M)::OuterScaling
    n = size(kkt.hess_com, 1); ns = length(kkt.ind_ineq)
    nzH = scaling_nzval(kkt.hess_com); nzJ = scaling_nzval(kkt.jt_csc)
    view(O.absnz, O.pH) .= abs.(view(nzH, O.sH))
    view(O.absnz, O.pJ) .= abs.(view(nzJ, O.sJ))
    fill!(O.hd, zero(T))
    view(O.hd, O.hd_row) .= view(nzH, O.hd_nz)
    # x and s diagonals: H + pr_diag (augmented: the bound terms Z/X included) or H + reg (unreduced:
    # the bounds are rows of their own)
    O.dx .= unreduced ? kkt.reg : kkt.pr_diag
    view(O.dx, 1:n) .+= O.hd
    view(O.absnz, O.pD) .= abs.(view(O.dx, O.sD))
    fill!(view(O.absnz, O.pC), one(T))
    O.dy .= kkt.du_diag
    view(O.dy, kkt.ind_eq) .-= kkt.eq_shift.current
    view(O.absnz, O.pY) .= abs.(view(O.dy, O.sY))
    if unreduced
        view(O.absnz, O.pLc) .= sqrt.(view(kkt.l_lower, O.sLc))
        view(O.absnz, O.pUc) .= sqrt.(view(kkt.u_lower, O.sUc))
        view(O.absnz, O.pLd) .= abs.(view(kkt.l_diag, O.sLd))
        view(O.absnz, O.pUd) .= abs.(view(kkt.u_diag, O.sUd))
    end
    fill!(O.d, one(T))
    d, dnew = O.d, O.dnew
    for _ in 1:scaler_sweeps(M.scaler)
        scaling_sweep!(dnew, d, O.colptr, O.rowval, O.absnz, O.long_cols)
        d, dnew = dnew, d
    end
    M.d .= view(d, 1:n)
    return
end
