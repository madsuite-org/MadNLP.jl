@kwdef mutable struct UmfpackOptions <: AbstractOptions
    umfpack_pivtol::Float64 = 1e-4
    umfpack_pivtolmax::Float64 = 1e-1
    umfpack_sym_pivtol::Float64 = 1e-3
    umfpack_block_size::Float64 = 16
    umfpack_strategy::Float64 = 2.
end

mutable struct UmfpackState{T} <: AbstractLinearSolver{T}
    inner::UMFPACK.UmfpackLU{Float64, Int32}
    tril::SparseMatrixCSC{T,Int32}
    full::SparseMatrixCSC{Float64,Int32}
    tril_to_full_view::SubVector{T,Vector{T},Vector{Int}}

    p::Vector{Float64}
    d::Vector{Float64}

    opt::UmfpackOptions
    logger::MadNLPLogger
end

function UmfpackState(
    csc::SparseMatrixCSC{T};
    opt=UmfpackOptions(), logger=MadNLPLogger(),
) where T
    p = Vector{Float64}(undef,csc.n)
    d = Vector{Float64}(undef,csc.n)
    full, tril_to_full_view = get_tril_to_full(Float64,csc)
    controls = UMFPACK.get_umfpack_control(Float64, Int)
    # Override default controls with custom setting
    controls[4] = opt.umfpack_pivtol
    controls[5] = opt.umfpack_block_size
    controls[6] = opt.umfpack_strategy
    controls[12] = opt.umfpack_sym_pivtol
    inner = UMFPACK.UmfpackLU(full; control=controls)
    return UmfpackState(inner, csc, full, tril_to_full_view, p, d, opt, logger)
end

function factorize!(M::UmfpackState)
    M.full.nzval .= M.tril_to_full_view
    # We check the factorization succeeded later in the backsolve
    UMFPACK.lu!(M.inner, M.full; check=false)
    return M
end

function solve_linear_system!(M::UmfpackState{T},rhs::Vector{T}) where T
    if UMFPACK.issuccess(M.inner)
        M.p .= rhs
        UMFPACK.ldiv!(M.d, M.inner, M.p)
        rhs .= M.d
    end
    # If the factorization failed, we return the same
    # rhs to enter into a primal-dual regularization phase.
    return rhs
end

is_inertia(::UmfpackState) = false
inertia(M::UmfpackState) = throw(InertiaException())
input_type(::Type{UmfpackState}) = :csc
default_options(::Type{UmfpackState}) = UmfpackOptions()

function improve!(M::UmfpackState)
    if M.inner.control[4] == M.opt.umfpack_pivtolmax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.inner.control[4] = min(M.opt.umfpack_pivtolmax, M.inner.control[4]^.75)
    @debug(M.logger, "improved quality: pivtol = $(M.inner.control[4])")
    return true
end
introduce(::UmfpackState) = "umfpack"
is_supported(::Type{UmfpackState},::Type{Float32}) = true
is_supported(::Type{UmfpackState},::Type{Float64}) = true
