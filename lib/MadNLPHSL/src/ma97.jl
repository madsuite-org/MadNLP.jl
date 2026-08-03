@kwdef mutable struct Ma97Options <: AbstractOptions
    ma97_num_threads::Int = 1
    ma97_print_level::Int = -1
    ma97_nemin::Int = 8
    ma97_small::Float64 = 1e-20
    ma97_order::Ordering = METIS
    ma97_scaling::Scaling = SCALING_NONE
    ma97_u::Float64 = 1e-8
    ma97_umax::Float64 = 1e-4
end

mutable struct Ma97Solver{T,INT} <: AbstractLinearSolver{T}
    n::INT

    csc::SparseMatrixCSC{T,INT}

    control::Ma97Control{T,INT}
    info::Ma97Info{T,INT}

    akeep::Vector{Ptr{Cvoid}}
    fkeep::Vector{Ptr{Cvoid}}

    opt::Ma97Options
    logger::MadNLPLogger
end

ma97_set_num_threads(n) = HSL.omp_set_num_threads(n)

function Ma97Solver(
    csc::SparseMatrixCSC{T,INT};
    opt = Ma97Options(),
    logger = MadNLPLogger(),
) where {T,INT}

    ma97_set_num_threads(opt.ma97_num_threads)

    n = INT(csc.n)

    info = Ma97Info{T,INT}()
    control = Ma97Control{T,INT}()
    HSL.ma97_default_control(T, INT, control)

    control.print_level = opt.ma97_print_level
    control.f_arrays = 1
    control.nemin = opt.ma97_nemin
    control.ordering = INT(opt.ma97_order)
    control.small = opt.ma97_small
    control.u = opt.ma97_u
    control.scaling = INT(opt.ma97_scaling)

    akeep = [C_NULL]
    fkeep = [C_NULL]

    HSL.ma97_analyse(
        T,
        INT,
        INT(1),
        n,
        csc.colptr,
        csc.rowval,
        C_NULL,
        akeep,
        control,
        info,
        C_NULL,
    )
    info.flag < 0 && throw(SymbolicException())
    M = Ma97Solver{T,INT}(n, csc, control, info, akeep, fkeep, opt, logger)
    finalizer(finalize, M)
    return M
end
function factorize!(M::Ma97Solver{T,INT}) where {T,INT}
    HSL.ma97_factor(
        T,
        INT,
        INT(4),
        C_NULL,
        C_NULL,
        M.csc.nzval,
        M.akeep,
        M.fkeep,
        M.control,
        M.info,
        C_NULL,
    )
    M.info.flag < 0 && throw(FactorizationException())
    return M
end
function solve_linear_system!(M::Ma97Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    HSL.ma97_solve(T, INT, INT(0), INT(1), rhs, M.n, M.akeep, M.fkeep, M.control, M.info)
    M.info.flag < 0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma97Solver) = true
function inertia(M::Ma97Solver)
    return (M.info.matrix_rank - M.info.num_neg, M.n - M.info.matrix_rank, M.info.num_neg)
end

function finalize(M::Ma97Solver{T,INT}) where {T,INT}
    HSL.ma97_finalise(T, INT, M.akeep, M.fkeep)
end

function improve!(M::Ma97Solver)
    if M.control.u == M.opt.ma97_umax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma97_umax, M.control.u^0.75)
    @debug(M.logger, "improved quality: pivtol = $(M.control.u)")
    return true
end
introduce(::Ma97Solver) = "ma97 v$(HSL.HSL_MA97_version())"
input_type(::Type{Ma97Solver}) = :csc
default_options(::Type{Ma97Solver}) = Ma97Options()
MadNLP._default_ls_options(ls::Val{:Ma97Solver}) = Ma97Options()
is_supported(::Type{Ma97Solver}, ::Type{T}) where T <: AbstractFloat = HSL.is_supported(Val(:hsl_ma97), T)
