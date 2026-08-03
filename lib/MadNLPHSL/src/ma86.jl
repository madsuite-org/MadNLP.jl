@kwdef mutable struct Ma86Options <: AbstractOptions
    ma86_num_threads::Int = 1
    ma86_print_level::Float64 = -1
    ma86_nemin::Int = 32
    ma86_order::Ordering = METIS
    ma86_scaling::Scaling = SCALING_NONE
    ma86_small::Float64 = 1e-20
    ma86_static::Float64 = 0.0
    ma86_u::Float64 = 1e-8
    ma86_umax::Float64 = 1e-4
end

mutable struct Ma86Solver{T,INT} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,INT}

    control::Ma86Control{T,INT}
    info::Ma86Info{T,INT}

    mc68_control::Mc68Control{INT}
    mc68_info::Mc68Info{INT}

    order::Vector{INT}
    keep::Vector{Ptr{Cvoid}}

    opt::Ma86Options
    logger::MadNLPLogger
end

ma86_set_num_threads(n) = HSL.omp_set_num_threads(n)

function Ma86Solver(
    csc::SparseMatrixCSC{T,INT};
    opt = Ma86Options(),
    logger = MadNLPLogger(),
) where {T,INT}

    ma86_set_num_threads(opt.ma86_num_threads)

    order = Vector{INT}(undef, csc.n)

    mc68info = Mc68Info{INT}()
    mc68control = Mc68Control{INT}()
    HSL.mc68_default_control(INT, mc68control)

    keep = [C_NULL]

    mc68control.f_array_in = 1
    mc68control.f_array_out = 1
    HSL.mc68_order(
        INT,
        INT(opt.ma86_order),
        INT(csc.n),
        csc.colptr,
        csc.rowval,
        order,
        mc68control,
        mc68info,
    )

    info = Ma86Info{T,INT}()
    control = Ma86Control{T,INT}()
    HSL.ma86_default_control(T, INT, control)
    control.diagnostics_level = INT(opt.ma86_print_level)
    control.f_arrays = 1
    control.nemin = opt.ma86_nemin
    control.small_ = opt.ma86_small
    control.u = opt.ma86_u
    control.scaling = INT(opt.ma86_scaling)

    HSL.ma86_analyse(T, INT, INT(csc.n), csc.colptr, csc.rowval, order, keep, control, info)
    info.flag < 0 && throw(SymbolicException())

    M = Ma86Solver{T,INT}(
        csc,
        control,
        info,
        mc68control,
        mc68info,
        order,
        keep,
        opt,
        logger,
    )
    finalizer(finalize, M)

    return M
end
function factorize!(M::Ma86Solver{T,INT}) where {T,INT}
    HSL.ma86_factor(
        T,
        INT,
        INT(M.csc.n),
        M.csc.colptr,
        M.csc.rowval,
        M.csc.nzval,
        M.order,
        M.keep,
        M.control,
        M.info,
        C_NULL,
    )
    M.info.flag < 0 && throw(FactorizationException())
    return M
end
function solve_linear_system!(M::Ma86Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    HSL.ma86_solve(
        T,
        INT,
        INT(0),
        INT(1),
        INT(M.csc.n),
        rhs,
        M.order,
        M.keep,
        M.control,
        M.info,
        C_NULL,
    )
    M.info.flag < 0 && throw(SolveException())
    return rhs
end
is_inertia(::Ma86Solver) = true
function inertia(M::Ma86Solver)
    return (
        M.info.matrix_rank - M.info.num_neg,
        M.csc.n - M.info.matrix_rank,
        M.info.num_neg,
    )
end

function finalize(M::Ma86Solver{T,INT}) where {T,INT}
    HSL.ma86_finalise(T, INT, M.keep, M.control)
end

function improve!(M::Ma86Solver)
    if M.control.u == M.opt.ma86_umax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma86_umax, M.control.u^0.75)
    @debug(M.logger, "improved quality: pivtol = $(M.control.u)")
    return true
end
introduce(::Ma86Solver) = "ma86 v$(HSL.HSL_MA86_version())"
input_type(::Type{Ma86Solver}) = :csc
default_options(::Type{Ma86Solver}) = Ma86Options()
MadNLP._default_ls_options(ls::Val{:Ma86Solver}) = Ma86Options()
is_supported(::Type{Ma86Solver}, ::Type{T}) where T <: AbstractFloat = HSL.is_supported(Val(:hsl_ma86), T)
