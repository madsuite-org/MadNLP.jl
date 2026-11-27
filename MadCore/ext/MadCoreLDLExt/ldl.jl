@kwdef mutable struct LDLFactorizationsOptions <: AbstractOptions
    # the pivot order, the permutation P of LDLFactorizations instead of its AMD ordering:
    # row and column i of the factorized matrix are row and column ldl_user_perm[i]
    ldl_user_perm::Union{Nothing, Vector{Int}} = nothing
    # LDLFactorizations' dynamic regularization: a pivot smaller than `ldl_tol` in magnitude is
    # replaced by `ldl_r1` in the first `ldl_n_d` rows and columns (in the order of the matrix,
    # before any permutation) and by `ldl_r2` in the others; off with both `ldl_r1` and `ldl_r2` zero
    ldl_r1::Float64 = 0.0
    ldl_r2::Float64 = 0.0
    ldl_tol::Float64 = 0.0
    ldl_n_d::Int = 0
end
const LDLF = LDLFactorizations

mutable struct LDLState{T} <: AbstractLinearSolver{T}
    inner::LDLF.LDLFactorization{T, Int32}
    tril::SparseMatrixCSC{T,Int32}
    # the upper triangle, the transpose of `tril`, which LDLFactorizations factorizes: its
    # factorization of a full matrix ignores the dynamic regularization
    triu::SparseMatrixCSC{T,Int32}
    tril_to_triu::Vector{Int}   # triu.nzval .= tril.nzval[tril_to_triu]
    opt::LDLFactorizationsOptions
    logger::MadNLPLogger
end

function LDLState(
    tril::SparseMatrixCSC{T};
    opt=LDLFactorizationsOptions(), logger=MadNLPLogger(),
) where T
    index = SparseMatrixCSC(tril.m, tril.n, tril.colptr, tril.rowval, collect(1:nnz(tril)))
    pattern = SparseMatrixCSC(transpose(index))
    tril_to_triu = pattern.nzval
    triu = SparseMatrixCSC{T,Int32}(tril.m, tril.n, pattern.colptr, pattern.rowval, tril.nzval[tril_to_triu])
    upper = Symmetric(triu, :U)
    inner = opt.ldl_user_perm === nothing ? LDLF.ldl_analyze(upper, T) : LDLF.ldl_analyze(upper, T; P = opt.ldl_user_perm)
    M = LDLState(inner, tril, triu, tril_to_triu, opt, logger)
    return factorize!(M)
end

function factorize!(M::LDLState{T}) where T
    M.triu.nzval .= view(M.tril.nzval, M.tril_to_triu)
    M.inner.r1, M.inner.r2, M.inner.tol = T(M.opt.ldl_r1), T(M.opt.ldl_r2), T(M.opt.ldl_tol)
    M.inner.n_d = M.opt.ldl_n_d
    LDLF.ldl_factorize!(Symmetric(M.triu, :U), M.inner)
    return M
end

function solve_linear_system!(M::LDLState{T},rhs::Vector{T}) where T
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    LDLF.factorized(M.inner) && ldiv!(M.inner, rhs)
    return rhs
end

is_inertia(::LDLState) = true
function inertia(M::LDLState)
    (m, n) = size(M.tril)
    LDLF.factorized(M.inner) || return (0, n, 0)
    (pos, zero, neg) = (0, 0, 0)
    D = M.inner.D
    for i=1:n
        d = D[i,i]
        if d > 0
            pos += 1
        elseif d == 0
            zero += 1
        else
            neg += 1
        end
    end
    return pos, zero, neg
end
input_type(::Type{LDLState}) = :csc
default_options(::Type{LDLState}) = LDLFactorizationsOptions()
MadCore.set_permutation!(opt::LDLFactorizationsOptions, perm) = (opt.ldl_user_perm = perm; opt)

function improve!(M::LDLState)
    return false
end
introduce(::LDLState) = "LDLFactorizations v$(pkgversion(LDLF))"
is_supported(::Type{LDLState},::Type{T}) where T <: AbstractFloat = true
is_factorized(M::LDLState) = LDLF.factorized(M.inner)
