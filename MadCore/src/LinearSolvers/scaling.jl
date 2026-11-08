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
        dv(Ti.(S.rowval)), dv(Ti.(rows)), dv(Ti.(cols)), dv(Ti.(adiag)), logger,
    )
end

# One sweep of symmetric infinity-norm scaling over the pattern with both triangles, stored by
# columns: dnew_j = d_j / sqrt(max_i |A_ij| d_i d_j).
function scaling_sweep!(dnew::Vector, d, colptr, rowval, absnz)
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
        scaling_sweep!(M.dnew, M.d, M.fcolptr, M.frow, M.absnz)
        M.d, M.dnew = M.dnew, M.d
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
introduce(M::ScaledShiftedSolver) = "$(introduce(M.inner)) (scaled and shifted)"
