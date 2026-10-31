@enum(MatchingStrategy::Int, COMPLETE = 1, COMPLETE2x2 = 2, CONSTRAINTS = 3)

@kwdef mutable struct PardisoMKLOptions <: AbstractOptions
    pardisomkl_num_threads::Int = 1
    pardiso_matching_strategy::MatchingStrategy = COMPLETE2x2
    pardisomkl_max_iterative_refinement_steps::Int = 1
    pardisomkl_msglvl::Int = 0
    pardisomkl_order::Int = 2
end

@kwdef mutable struct PanuaPardisoOptions <: AbstractOptions
    pardiso_matching_strategy::MatchingStrategy = COMPLETE2x2
    pardiso_max_inner_refinement_steps::Int = 1
    pardiso_msglvl::Int = 0
    pardiso_order::Int = 2
    pardiso_algorithm::LinearFactorization = BUNCHKAUFMAN
end

# A Pardiso solver of Pardiso.jl and the lower triangle of the KKT matrix in CSC,
# which Pardiso reads as the upper triangle in CSR.
mutable struct PardisoState{T, PS <: Pardiso.AbstractPardisoSolver, Opt} <: AbstractLinearSolver{T}
    ps::PS
    csc::SparseMatrixCSC{T, Int32}
    w::Vector{T}
    opt::Opt
    logger::MadNLPLogger
end
const PardisoMKLState{T} = PardisoState{T, Pardiso.MKLPardisoSolver, PardisoMKLOptions}
const PanuaPardisoState{T} = PardisoState{T, Pardiso.PardisoSolver, PanuaPardisoOptions}

# Run one Pardiso phase. Pardiso.jl throws a `PardisoException` when Pardiso
# returns an error, or a `PardisoPosDefException` for a zero pivot (error -4); both
# are reported as the MadCore exception `exc`.
function _pardiso!(M::PardisoState{T}, phase, x, b, exc) where {T}
    Pardiso.set_phase!(M.ps, phase)
    try
        Pardiso.pardiso(M.ps, x, M.csc, b)
    catch err
        err isa Union{Pardiso.PardisoException, Pardiso.PardisoPosDefException} || rethrow()
        throw(exc)
    end
    return M
end

_with_threads(f, ::PardisoState) = f()
function _with_threads(f, M::PardisoMKLState)
    Pardiso.set_nprocs!(M.ps, M.opt.pardisomkl_num_threads)
    try
        return f()
    finally
        Pardiso.set_nprocs!(M.ps, blas_num_threads[])
    end
end

function _analyze!(M::PardisoState{T}) where {T}
    _with_threads(M) do
        _pardiso!(M, Pardiso.ANALYSIS, T[], M.w, SymbolicException())
    end
    return M
end

function PardisoMKLState(
    csc::SparseMatrixCSC{T};
    opt = PardisoMKLOptions(),
    logger = MadNLPLogger(),
) where {T}
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    Pardiso.set_iparm!(ps, 2, opt.pardisomkl_order)
    Pardiso.set_iparm!(ps, 3, 0)
    Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_iparm!(ps, 8, opt.pardisomkl_max_iterative_refinement_steps)
    Pardiso.set_iparm!(ps, 10, 12)    # pivot perturbation
    Pardiso.set_iparm!(ps, 11, 2)
    Pardiso.set_iparm!(ps, 13, 1)     # matching strategy
    Pardiso.set_iparm!(ps, 21, 3)     # Bunch-Kaufman pivoting
    Pardiso.set_iparm!(ps, 24, 1)     # parallel factorization
    Pardiso.set_iparm!(ps, 25, 0)     # sequential solve
    Pardiso.set_iparm!(ps, 28, T == Float64 ? 0 : 1)
    Pardiso.set_msglvl!(ps, opt.pardisomkl_msglvl == 0 ? Pardiso.MESSAGE_LEVEL_OFF : Pardiso.MESSAGE_LEVEL_ON)
    csc = SparseMatrixCSC{T, Int32}(csc)
    M = PardisoState(ps, csc, Vector{T}(undef, csc.n), opt, logger)
    return _analyze!(M)
end

function PanuaPardisoState(
    csc::SparseMatrixCSC{T};
    opt = PanuaPardisoOptions(),
    logger = MadNLPLogger(),
) where {T}
    mtype = if opt.pardiso_algorithm in (BUNCHKAUFMAN, LDL)
        Pardiso.REAL_SYM_INDEF
    elseif opt.pardiso_algorithm == CHOLESKY
        Pardiso.REAL_SYM_POSDEF
    else
        error(
            "Only the factorizations CHOLESKY, LDL or BUNCHKAUFMAN are supported by " *
            "PanuaPardisoSolver. Please change the option `pardiso_algorithm` accordingly.",
        )
    end
    ps = Pardiso.PardisoSolver()
    Pardiso.set_solver!(ps, Pardiso.DIRECT_SOLVER)
    Pardiso.set_matrixtype!(ps, mtype)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    Pardiso.set_iparm!(ps, 2, opt.pardiso_order)
    Pardiso.set_iparm!(ps, 3, haskey(ENV, "OMP_NUM_THREADS") ? parse(Int32, ENV["OMP_NUM_THREADS"]) : 1)
    Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_iparm!(ps, 8, opt.pardiso_max_inner_refinement_steps)
    Pardiso.set_iparm!(ps, 10, 12)    # pivot perturbation
    Pardiso.set_iparm!(ps, 11, 0)     # no scaling
    Pardiso.set_iparm!(ps, 13, 1)     # matching strategy
    Pardiso.set_iparm!(ps, 21, opt.pardiso_algorithm == BUNCHKAUFMAN ? 1 : 0)
    Pardiso.set_iparm!(ps, 24, 1)     # parallel factorization
    Pardiso.set_iparm!(ps, 25, 1)     # parallel solve
    Pardiso.set_msglvl!(ps, opt.pardiso_msglvl == 0 ? Pardiso.MESSAGE_LEVEL_OFF : Pardiso.MESSAGE_LEVEL_ON)
    csc = SparseMatrixCSC{T, Int32}(csc)
    M = PardisoState(ps, csc, Vector{T}(undef, csc.n), opt, logger)
    return _analyze!(M)
end

function factorize!(M::PardisoState{T}) where {T}
    _with_threads(M) do
        _pardiso!(M, Pardiso.NUM_FACT, T[], M.w, FactorizationException())
        # With perturbed pivots, compute the symbolic factorization again.
        if Pardiso.get_iparm(M.ps, 14) != 0
            _pardiso!(M, Pardiso.ANALYSIS_NUM_FACT, T[], M.w, FactorizationException())
        end
    end
    return M
end

# With iparm[6] = 1, Pardiso writes the solution into the right-hand side and uses
# `M.w` as workspace.
function solve_linear_system!(M::PardisoState{T}, rhs::Vector{T}) where {T}
    _with_threads(M) do
        _pardiso!(M, Pardiso.SOLVE_ITERATIVE_REFINE, M.w, rhs, SolveException())
    end
    return rhs
end

is_inertia(::PardisoState) = true
function inertia(M::PardisoState)
    # Pardiso reports the inertia only for symmetric indefinite matrices.
    M isa PanuaPardisoState && M.opt.pardiso_algorithm == CHOLESKY && return (M.csc.n, 0, 0)
    pos = Int(Pardiso.get_iparm(M.ps, 22))
    neg = Int(Pardiso.get_iparm(M.ps, 23))
    return (pos, M.csc.n - pos - neg, neg)
end

function improve!(M::PardisoState)
    @debug(M.logger, "improve quality failed.")
    return false
end

introduce(::PardisoMKLState) = "pardiso-mkl"
introduce(::PanuaPardisoState) = "pardiso"

input_type(::Type{<:PardisoState}) = :csc
default_options(::Type{PardisoMKLState}) = PardisoMKLOptions()
default_options(::Type{PanuaPardisoState}) = PanuaPardisoOptions()
is_supported(::Type{PardisoMKLState}, ::Type{T}) where {T <: Union{Float32, Float64}} = true
# Pardiso.jl supports single precision only with MKL.
is_supported(::Type{PanuaPardisoState}, ::Type{Float64}) = true
