@kwdef mutable struct Ma77Options <: AbstractOptions
    ma77_buffer_lpage::Int = 4096
    ma77_buffer_npage::Int = 1600
    ma77_file_size::Int = 2097152
    ma77_maxstore::Int = 0
    ma77_nemin::Int = 8
    ma77_order::Ordering = METIS
    ma77_print_level::Int = -1
    ma77_small::Float64 = 1e-20
    ma77_static::Float64 = 0.0
    ma77_u::Float64 = 1e-8
    ma77_umax::Float64 = 1e-4
end

mutable struct Ma77Solver{T,INT} <: AbstractLinearSolver{T}
    tril::SparseMatrixCSC{T,INT}
    full::SparseMatrixCSC{T,INT}
    tril_to_full_view::SubVector{T}

    control::Ma77Control{T,INT}
    info::Ma77Info{T,INT}

    mc68_control::Mc68Control{INT}
    mc68_info::Mc68Info{INT}

    order::Vector{INT}
    keep::Vector{Ptr{Cvoid}}

    opt::Ma77Options
    logger::MadNLPLogger
end

function Ma77Solver(
    csc::SparseMatrixCSC{T,INT};
    opt = Ma77Options(),
    logger = MadNLPLogger(),
) where {T,INT}
    full, tril_to_full_view = get_tril_to_full(csc)
    order = Vector{INT}(undef, csc.n)

    mc68info = Mc68Info{INT}()
    mc68control = Mc68Control{INT}()
    HSL.mc68_default_control(INT, mc68control)

    keep = [C_NULL]

    mc68control.f_array_in = 1
    mc68control.f_array_out = 1
    HSL.mc68_order(
        INT,
        INT(opt.ma77_order),
        INT(csc.n),
        csc.colptr,
        csc.rowval,
        order,
        mc68control,
        mc68info,
    )

    info = Ma77Info{T,INT}()
    control = Ma77Control{T,INT}()
    HSL.ma77_default_control(T, INT, control)
    control.f_arrays = 1
    control.bits = 32
    control.file_size = opt.ma77_file_size
    control.maxstore = opt.ma77_maxstore
    control.print_level = -1

    control.buffer_lpage = (opt.ma77_buffer_lpage, opt.ma77_buffer_lpage)
    control.buffer_npage = (opt.ma77_buffer_npage, opt.ma77_buffer_npage)

    control.nemin = opt.ma77_nemin
    control.small = opt.ma77_small
    control.static_ = opt.ma77_static
    control.u = opt.ma77_u

    HSL.ma77_open(
        T,
        INT,
        INT(full.n),
        tempname(cleanup = false),
        tempname(cleanup = false),
        tempname(cleanup = false),
        tempname(cleanup = false),
        keep,
        control,
        info,
    )

    info.flag < 0 && throw(SymbolicException())

    for i = 1:full.n
        HSL.ma77_input_vars(
            T,
            INT,
            INT(i),
            full.colptr[i+1] - full.colptr[i],
            _madnlp_unsafe_wrap(
                full.rowval,
                full.colptr[i+1] - full.colptr[i],
                full.colptr[i],
            ),
            keep,
            control,
            info,
        )
        info.flag < 0 && throw(LinearSymbolicException())
    end

    HSL.ma77_analyse(T, INT, order, keep, control, info)
    info.flag < 0 && throw(SymbolicException())

    M = Ma77Solver{T,INT}(
        csc,
        full,
        tril_to_full_view,
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

function factorize!(M::Ma77Solver{T,INT}) where {T,INT}
    M.full.nzval .= M.tril_to_full_view
    for i = 1:M.full.n
        HSL.ma77_input_reals(
            T,
            INT,
            INT(i),
            M.full.colptr[i+1] - M.full.colptr[i],
            _madnlp_unsafe_wrap(
                M.full.nzval,
                M.full.colptr[i+1] - M.full.colptr[i],
                M.full.colptr[i],
            ),
            M.keep,
            M.control,
            M.info,
        )
        M.info.flag < 0 && throw(FactorizationException())
    end
    HSL.ma77_factor(T, INT, INT(0), M.keep, M.control, M.info, C_NULL)
    M.info.flag < 0 && throw(FactorizationException())
    return M
end
function solve_linear_system!(M::Ma77Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    HSL.ma77_solve(
        T,
        INT,
        INT(0),
        INT(1),
        INT(M.full.n),
        rhs,
        M.keep,
        M.control,
        M.info,
        C_NULL,
    )
    M.info.flag < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma77Solver) = true
function inertia(M::Ma77Solver)
    return (
        M.info.matrix_rank - M.info.num_neg,
        M.full.n - M.info.matrix_rank,
        M.info.num_neg,
    )
end

function finalize(M::Ma77Solver{T,INT}) where {T,INT}
    HSL.ma77_finalise(T, INT, M.keep, M.control, M.info)
end

function improve!(M::Ma77Solver)
    if M.control.u == M.opt.ma77_umax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.control.u = min(M.opt.ma77_umax, M.control.u^0.75)
    @debug(M.logger, "improved quality: pivtol = $(M.control.u)")
    return true
end

introduce(::Ma77Solver) = "ma77 v$(HSL.HSL_MA77_version())"
input_type(::Type{Ma77Solver}) = :csc
default_options(::Type{Ma77Solver}) = Ma77Options()
MadNLP._default_ls_options(ls::Val{:Ma77Solver}) = Ma77Options()
is_supported(::Type{Ma77Solver}, ::Type{T}) where T <: AbstractFloat = HSL.is_supported(Val(:hsl_ma77), T)
