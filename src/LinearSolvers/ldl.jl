@kwdef mutable struct LDLFactorizationsOptions <: AbstractOptions
end
const LDLF = LDLFactorizations

mutable struct LDLSolver{T} <: AbstractLinearSolver{T}
    inner::LDLF.LDLFactorization{T, Int32}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{T,Int32}
    tril_to_full_view::SubVector{T}
    opt::LDLFactorizationsOptions
    logger::MadNLPLogger
end

function LDLSolver(
    tril::SparseMatrixCSC{T};
    opt=LDLFactorizationsOptions(), logger=MadNLPLogger(), pos=nothing,
) where T
    # TODO: convert tril to triu, not full
    full, tril_to_full_view = get_tril_to_full(T,tril)

    return LDLSolver(
        LDLF.ldl(
            full
        ),
        tril, full, tril_to_full_view, opt, logger
    )
end

function factorize!(M::LDLSolver)
    M.full.nzval .= M.tril_to_full_view
    LDLF.ldl_factorize!(M.full, M.inner)
    return M
end

function solve_linear_system!(M::LDLSolver{T},rhs::Vector{T}) where T
    ldiv!(M.inner, rhs)
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

is_inertia(::LDLSolver) = true
function inertia(M::LDLSolver)
    (m, n) = size(M.tril)
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
input_type(::Type{LDLSolver}) = :csc
default_options(::Type{LDLSolver}) = LDLFactorizationsOptions()

function improve!(M::LDLSolver)
    return false
end
introduce(::LDLSolver) = "LDLFactorizations v$(pkgversion(LDLF))"
is_supported(::Type{LDLSolver},::Type{T}) where T <: AbstractFloat = true
