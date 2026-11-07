using SparseArrays.LibSuiteSparse: cholmod_dense, cholmod_solve2, cholmod_l_solve2,
    cholmod_free_dense, cholmod_l_free_dense

@kwdef mutable struct CHOLMODOptions <: AbstractOptions
    cholmod_algorithm::LinearFactorization = CHOLESKY
end

# `cholmod_analyze` reports the entries of L in `lnz`, and the first numeric factorization
# allocates about 2.5 times that before packing. Int32 indices must address that allocation.
const CHOLMOD_INT32_MAX_LNZ = typemax(Int32) / 3

mutable struct CHOLMODState{T, Ti} <: AbstractLinearSolver{T}
    inner::CHOLMOD.Factor{Float64, Ti}
    tril::SparseMatrixCSC{T,Int32}
    tril_to_full_view::SubVector{T,Vector{T},Vector{Int}}
    # The matrix as CHOLMOD holds it, and its values. Updating them in place avoids a
    # conversion and a Hermitian check at every factorization.
    A::CHOLMOD.Sparse{Float64, Ti}
    Ax::Vector{Float64}
    # The right-hand side and the buffers of `cholmod_solve2`, which CHOLMOD allocates on
    # the first solve and reuses afterwards.
    B::CHOLMOD.Dense{Float64}
    Bx::Vector{Float64}
    X::Base.RefValue{Ptr{cholmod_dense}}
    Y::Base.RefValue{Ptr{cholmod_dense}}
    E::Base.RefValue{Ptr{cholmod_dense}}

    d::Vector{Float64}

    opt::CHOLMODOptions
    logger::MadNLPLogger
end

_cholmod_solve2(::Type{Int32}, args...) = cholmod_solve2(args...)
_cholmod_solve2(::Type{Int64}, args...) = cholmod_l_solve2(args...)
_cholmod_free_dense(::Type{Int32}, args...) = cholmod_free_dense(args...)
_cholmod_free_dense(::Type{Int64}, args...) = cholmod_l_free_dense(args...)

function _cholmod_analyze(::Type{Ti}, full::SparseMatrixCSC{Float64}) where {Ti}
    A = CHOLMOD.Sparse(SparseMatrixCSC{Float64,Ti}(
        full.m,
        full.n,
        Vector{Ti}(full.colptr),
        Vector{Ti}(full.rowval),
        full.nzval,
    ))
    inner = CHOLMOD.symbolic(A)
    return A, inner, CHOLMOD.getcommon(Ti)[].lnz
end

function CHOLMODState(
    A::CHOLMOD.Sparse{Float64, Ti},
    inner::CHOLMOD.Factor{Float64, Ti},
    csc::SparseMatrixCSC{T},
    tril_to_full_view,
    opt,
    logger,
) where {T, Ti}
    s = unsafe_load(CHOLMOD.typedpointer(A))
    Ax = unsafe_wrap(Array, Ptr{Float64}(s.x), length(tril_to_full_view); own=false)
    B = CHOLMOD.Dense(zeros(Float64, csc.n))
    Bx = unsafe_wrap(Array, Ptr{Float64}(unsafe_load(pointer(B)).x), csc.n; own=false)
    null() = Ref(Ptr{cholmod_dense}(C_NULL))
    d = Vector{Float64}(undef, csc.n)
    M = CHOLMODState{T, Ti}(
        inner, csc, tril_to_full_view, A, Ax, B, Bx, null(), null(), null(), d, opt, logger,
    )
    return finalizer(_free_solve_buffers!, M)
end

function CHOLMODState(
    csc::SparseMatrixCSC{T};
    opt=CHOLMODOptions(), logger=MadNLPLogger(), max_lnz_int32=CHOLMOD_INT32_MAX_LNZ,
) where T
    full, tril_to_full_view = get_tril_to_full(Float64, csc)

    if opt.cholmod_algorithm == LDL && VERSION <= v"1.9"
        error("[CHOLMOD] Option `cholmod_algorithm=LDL` is not supported for Julia version <= 1.9")
    end

    fill!(full.nzval, one(T))

    A, inner, lnz = _cholmod_analyze(Int32, full)
    if lnz > max_lnz_int32
        A, inner, _ = _cholmod_analyze(Int64, full)
    end
    return CHOLMODState(A, inner, csc, tril_to_full_view, opt, logger)
end

function _free_solve_buffers!(M::CHOLMODState{T, Ti}) where {T, Ti}
    for buffer in (M.X, M.Y, M.E)
        buffer[] == C_NULL || _cholmod_free_dense(Ti, buffer, CHOLMOD.getcommon(Ti))
    end
    return
end

function factorize!(M::CHOLMODState)
    M.Ax .= M.tril_to_full_view
    # We check the factorization succeeded later in the backsolve
    if M.opt.cholmod_algorithm == LDL
        CHOLMOD.ldlt!(M.inner, M.A; check=false)
    elseif M.opt.cholmod_algorithm == CHOLESKY
        CHOLMOD.cholesky!(M.inner, M.A; check=false)
    end
    return M
end

function solve_linear_system!(M::CHOLMODState{T, Ti}, rhs::Vector{T}) where {T, Ti}
    if issuccess(M.inner)
        M.Bx .= rhs
        status = _cholmod_solve2(
            Ti, CHOLMOD.CHOLMOD_A, CHOLMOD.typedpointer(M.inner), pointer(M.B), C_NULL,
            M.X, C_NULL, M.Y, M.E, CHOLMOD.getcommon(Ti),
        )
        status == 1 || error("[CHOLMOD] cholmod_solve2 failed")
        x = Ptr{Float64}(unsafe_load(M.X[]).x)
        @inbounds for i in eachindex(rhs)
            rhs[i] = unsafe_load(x, i)
        end
    end
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

# Copy the diagonal of D from CHOLMOD's simplicial LDL factor, where the diagonal is the
# first entry of each column.
function _get_diagonal!(F::CHOLMOD.Factor{T, Ti}, d::Vector{T}) where {T, Ti}
    s = unsafe_load(CHOLMOD.typedpointer(F))
    colptr = unsafe_wrap(Array, Ptr{Ti}(s.p), s.n+1, own=false)
    nzvals = unsafe_wrap(Array, Ptr{T}(s.x), s.nzmax, own=false)
    for j in 1:s.n
        d[j] = nzvals[colptr[j]+1]
    end
    return d
end

is_inertia(::CHOLMODState) = true
function _inertia_cholesky(M::CHOLMODState)
    n = size(M.tril, 1)
    if issuccess(M.inner)
        return (n, 0, 0)
    else
        return (0, n, 0)
    end
end
function _inertia_ldlt(M::CHOLMODState)
    n = size(M.tril, 1)
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
function inertia(M::CHOLMODState)
    if M.opt.cholmod_algorithm == CHOLESKY
        return _inertia_cholesky(M)
    elseif M.opt.cholmod_algorithm == LDL
        return _inertia_ldlt(M)
    end
end
input_type(::Type{CHOLMODState}) = :csc
default_options(::Type{CHOLMODState}) = CHOLMODOptions()

improve!(M::CHOLMODState) = false
introduce(::CHOLMODState) = "cholmod v$(CHOLMOD.BUILD_VERSION)"
is_supported(::Type{CHOLMODState},::Type{Float32}) = true
is_supported(::Type{CHOLMODState},::Type{Float64}) = true
is_factorized(M::CHOLMODState) = issuccess(M.inner)
