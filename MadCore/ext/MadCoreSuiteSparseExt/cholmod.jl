using SparseArrays: nnz, sparse
using SparseArrays.LibSuiteSparse: cholmod_dense, cholmod_factor_struct, cholmod_solve2,
    cholmod_l_solve2, cholmod_free_dense, cholmod_l_free_dense, CHOLMOD_NATURAL

@kwdef mutable struct CHOLMODOptions <: AbstractOptions
    cholmod_algorithm::LinearFactorization = CHOLESKY
end

# `cholmod_analyze` reports the entries of L in `lnz`. Int32 indices must address L, and the
# limit leaves headroom below `typemax(Int32)`.
const CHOLMOD_INT32_MAX_LNZ = typemax(Int32) / 2

mutable struct CHOLMODState{T, Ti} <: AbstractLinearSolver{T}
    inner::CHOLMOD.Factor{Float64, Ti}
    tril::SparseMatrixCSC{T, Int32}
    # The fill-reducing permutation, applied once here so that `cholmod_factorize_p` works
    # on A(perm, perm) directly instead of building a permuted copy at every factorization.
    perm::Vector{Int}
    # The triangle of A(perm, perm) that CHOLMOD factorizes, its values, and where each
    # value comes from in `tril`.
    A::CHOLMOD.Sparse{Float64, Ti}
    Ax::Vector{Float64}
    tril_to_permuted_view::SubVector{T,Vector{T},Vector{Int}}
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

# Return the fill-reducing permutation of `csc` and the entries of L it gives.
function _cholmod_ordering(::Type{Ti}, csc::SparseMatrixCSC) where {Ti}
    pattern = SparseMatrixCSC{Float64, Ti}(
        csc.m, csc.n, Vector{Ti}(csc.colptr), Vector{Ti}(csc.rowval), ones(nnz(csc)),
    )
    F = CHOLMOD.symbolic(CHOLMOD.Sparse(pattern, -1))
    lnz = CHOLMOD.getcommon(Ti)[].lnz
    s = unsafe_load(CHOLMOD.typedpointer(F))
    perm = Int.(unsafe_wrap(Array, Ptr{Ti}(s.Perm), csc.n; own = false)) .+ 1
    finalize(F)
    return perm, lnz
end

# Return the upper (`stype = 1`) or lower (`stype = -1`) triangle of csc(perm, perm) and the
# position in `csc` of each of its entries.
function _cholmod_permuted(::Type{Ti}, csc::SparseMatrixCSC, perm, stype) where {Ti}
    iperm = invperm(perm)
    rows, cols = Vector{Ti}(undef, nnz(csc)), Vector{Ti}(undef, nnz(csc))
    for j in 1:csc.n, c in csc.colptr[j]:(csc.colptr[j + 1] - 1)
        pi, pj = iperm[csc.rowval[c]], iperm[j]
        rows[c], cols[c] = stype > 0 ? minmax(pi, pj) : reverse(minmax(pi, pj))
    end
    map = sparse(rows, cols, collect(1:nnz(csc)), csc.n, csc.n)
    A = SparseMatrixCSC{Float64, Ti}(csc.n, csc.n, map.colptr, map.rowval, ones(nnz(csc)))
    return CHOLMOD.Sparse(A, stype), map.nzval
end

function CHOLMODState(
        ::Type{Ti},
        csc::SparseMatrixCSC{T},
        perm,
        opt,
        logger,
    ) where {T, Ti}
    A, tril_to_permuted = _cholmod_permuted(Ti, csc, perm, 1)
    # The Julia wrappers analyze with a given ordering. Marking the factor as naturally
    # ordered makes `cholmod_factorize_p` skip the copy, and `cholmod_solve2` the
    # permutation, which we apply ourselves.
    supernodal = opt.cholmod_algorithm == LDL ? CHOLMOD.CHOLMOD_SIMPLICIAL : CHOLMOD.CHOLMOD_AUTO
    inner = CHOLMOD.@cholmod_param supernodal = supernodal begin
        CHOLMOD.symbolic(A; perm = 1:csc.n, postorder = false)
    end
    L = CHOLMOD.typedpointer(inner)
    ordering = Base.fieldindex(cholmod_factor_struct, :ordering)
    unsafe_store!(
        Ptr{fieldtype(cholmod_factor_struct, ordering)}(L + fieldoffset(cholmod_factor_struct, ordering)),
        CHOLMOD_NATURAL,
    )
    # `cholmod_factorize_p` copies unless it receives triu(A) for the simplicial
    # factorization and tril(A) for the supernodal one.
    if unsafe_load(L).is_super != 0
        A, tril_to_permuted = _cholmod_permuted(Ti, csc, perm, -1)
    end
    s = unsafe_load(CHOLMOD.typedpointer(A))
    Ax = unsafe_wrap(Array, Ptr{Float64}(s.x), nnz(csc); own = false)
    B = CHOLMOD.Dense(zeros(Float64, csc.n))
    Bx = unsafe_wrap(Array, Ptr{Float64}(unsafe_load(pointer(B)).x), csc.n; own = false)
    null() = Ref(Ptr{cholmod_dense}(C_NULL))
    d = Vector{Float64}(undef, csc.n)
    M = CHOLMODState{T, Ti}(
        inner, csc, perm, A, Ax, view(csc.nzval, tril_to_permuted),
        B, Bx, null(), null(), null(), d, opt, logger,
    )
    return finalizer(_free_solve_buffers!, M)
end

function CHOLMODState(
        csc::SparseMatrixCSC{T};
        opt = CHOLMODOptions(), logger = MadNLPLogger(), max_lnz_int32 = CHOLMOD_INT32_MAX_LNZ,
    ) where {T}
    if opt.cholmod_algorithm == LDL && VERSION <= v"1.9"
        error("[CHOLMOD] Option `cholmod_algorithm=LDL` is not supported for Julia version <= 1.9")
    end

    perm, lnz = _cholmod_ordering(Int32, csc)
    Ti = lnz > max_lnz_int32 ? Int64 : Int32
    return CHOLMODState(Ti, csc, perm, opt, logger)
end

function _free_solve_buffers!(M::CHOLMODState{T, Ti}) where {T, Ti}
    for buffer in (M.X, M.Y, M.E)
        buffer[] == C_NULL || _cholmod_free_dense(Ti, buffer, CHOLMOD.getcommon(Ti))
    end
    return
end

function factorize!(M::CHOLMODState)
    M.Ax .= M.tril_to_permuted_view
    # We check the factorization succeeded later in the backsolve
    if M.opt.cholmod_algorithm == LDL
        # `CHOLMOD.ldlt!` converts the symbolic factor first, which makes CHOLMOD allocate
        # about 2.5 times the size of L at the first factorization.
        CHOLMOD.factorize_p!(M.A, 0.0, M.inner)
    elseif M.opt.cholmod_algorithm == CHOLESKY
        CHOLMOD.cholesky!(M.inner, M.A; check = false)
    end
    return M
end

function solve_linear_system!(M::CHOLMODState{T, Ti}, rhs::Vector{T}) where {T, Ti}
    if issuccess(M.inner)
        perm, Bx = M.perm, M.Bx
        @inbounds for i in eachindex(perm)
            Bx[i] = rhs[perm[i]]
        end
        status = _cholmod_solve2(
            Ti, CHOLMOD.CHOLMOD_A, CHOLMOD.typedpointer(M.inner), pointer(M.B), C_NULL,
            M.X, C_NULL, M.Y, M.E, CHOLMOD.getcommon(Ti),
        )
        status == 1 || error("[CHOLMOD] cholmod_solve2 failed")
        x = Ptr{Float64}(unsafe_load(M.X[]).x)
        @inbounds for i in eachindex(perm)
            rhs[perm[i]] = unsafe_load(x, i)
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
    colptr = unsafe_wrap(Array, Ptr{Ti}(s.p), s.n + 1, own = false)
    nzvals = unsafe_wrap(Array, Ptr{T}(s.x), s.nzmax, own = false)
    for j in 1:s.n
        d[j] = nzvals[colptr[j] + 1]
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
