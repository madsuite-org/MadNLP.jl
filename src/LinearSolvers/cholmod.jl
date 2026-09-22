@kwdef mutable struct CHOLMODOptions <: AbstractOptions
    cholmod_algorithm::LinearFactorization = CHOLESKY
end

mutable struct CHOLMODSolver{T} <: AbstractLinearSolver{T}
    inner::CHOLMOD.Factor{Float64}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{Float64,Int}
    tril_to_full_view::SubVector{T}

    p::Vector{Float64}
    d::Vector{Float64}

    opt::CHOLMODOptions
    logger::MadNLPLogger
end

function CHOLMODSolver(
    csc::SparseMatrixCSC{T};
    opt=CHOLMODOptions(), logger=MadNLPLogger(), pos=nothing,
) where T
    p = Vector{Float64}(undef,csc.n)
    d = Vector{Float64}(undef,csc.n)
    full, tril_to_full_view = get_tril_to_full(Float64, csc)

    if opt.cholmod_algorithm == LDL && VERSION <= v"1.9"
        error("[CHOLMOD] Option `cholmod_algorithm=LDL` is not supported for Julia version <= 1.9")
    end

    full = SparseMatrixCSC{Float64,Int}(
        full.m,
        full.n,
        Vector{Int64}(full.colptr),
        Vector{Int64}(full.rowval),
        full.nzval
    )
    fill!(full.nzval, one(T))

    A = CHOLMOD.Sparse(full)
    inner = CHOLMOD.symbolic(A)

    return CHOLMODSolver(inner, csc, full, tril_to_full_view, p, d, opt, logger)
end

function factorize!(M::CHOLMODSolver)
    M.full.nzval .= M.tril_to_full_view
    # We check the factorization succeeded later in the backsolve
    if M.opt.cholmod_algorithm == LDL
        CHOLMOD.ldlt!(M.inner, M.full; check=false)
    elseif M.opt.cholmod_algorithm == CHOLESKY
        CHOLMOD.cholesky!(M.inner, M.full; check=false)
    end
    return M
end

function solve_linear_system!(M::CHOLMODSolver{T}, rhs::Vector{T}) where T
    if issuccess(M.inner)
        B = CHOLMOD.Dense(rhs)
        X = CHOLMOD.solve(CHOLMOD.CHOLMOD_A, M.inner, B)
        copyto!(rhs, X)
    end
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

# Utils function to copy the diagonal entries directly from CHOLMOD's factor.
function _get_diagonal!(F::CHOLMOD.Factor{T}, d::Vector{T}) where T
    s = unsafe_load(CHOLMOD.typedpointer(F))
    # Wrap in memory the factor LD stored in CHOLMOD.
    colptr = unsafe_wrap(Array, s.p, s.n+1, own=false)
    nz = unsafe_wrap(Array, s.nz, s.n, own=false)
    rowval = unsafe_wrap(Array, s.i, s.nzmax, own=false)
    nzvals = unsafe_wrap(Array, Ptr{T}(s.x), s.nzmax, own=false)
    # Read LD factor and load diagonal entries
    for j in 1:s.n
        for c in colptr[j]:colptr[j]+nz[j]-1
            i = rowval[c+1] + 1  # convert to 1-based indexing
            if i == j
                d[i] = nzvals[c+1]
            end
        end
    end
    return d
end

is_inertia(::CHOLMODSolver) = true
function _inertia_cholesky(M::CHOLMODSolver)
    n = size(M.full, 1)
    if issuccess(M.inner)
        return (n, 0, 0)
    else
        return (0, n, 0)
    end
end
function _inertia_ldlt(M::CHOLMODSolver{T}) where T
    n = size(M.full, 1)
    if !issuccess(M.inner)
        return (0, n, 0)
    end
    D = M.d
    # Extract diagonal elements
    _get_diagonal!(M.inner, D)
    (pos, zero, neg) = (0, 0, 0)
    @inbounds for i in 1:n
        d = D[i]
        if d > 0
            pos += 1
        elseif d == 0
            zero += 1
        else
            neg += 1
        end
    end
    @assert pos + zero + neg == n
    return pos, zero, neg
end
function inertia(M::CHOLMODSolver)
    if M.opt.cholmod_algorithm == CHOLESKY
        return _inertia_cholesky(M)
    elseif M.opt.cholmod_algorithm == LDL
        return _inertia_ldlt(M)
    end
end
input_type(::Type{CHOLMODSolver}) = :csc
default_options(::Type{CHOLMODSolver}) = CHOLMODOptions()

improve!(M::CHOLMODSolver) = false
introduce(::CHOLMODSolver) = "cholmod v$(CHOLMOD.BUILD_VERSION)"
is_supported(::Type{CHOLMODSolver},::Type{Float32}) = true
is_supported(::Type{CHOLMODSolver},::Type{Float64}) = true
