ma57_default_icntl(INT) =
    INT[0, 0, 6, 1, 0, 5, 1, 0, 10, 0, 16, 16, 10, 100, 0, 0, 0, 0, 0, 0]
ma57_default_cntl(T) = T[1e-8, 1.0e-20, 0.5, 0.0, 0.0]

@kwdef mutable struct Ma57Options <: AbstractOptions
    ma57_pivtol::Float64 = 1e-8
    ma57_pivtolmax::Float64 = 1e-4
    ma57_pre_alloc::Float64 = 1.05
    ma57_pivot_order::Int = 5
    ma57_automatic_scaling::Bool = false

    ma57_block_size::Int = 16
    ma57_node_amalgamation::Int = 16
    ma57_small_pivot_flag::Int = 0
end

mutable struct Ma57Solver{T,INT} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,INT}
    I::Vector{INT}
    J::Vector{INT}

    icntl::Vector{INT}
    cntl::Vector{T}

    info::Vector{INT}
    rinfo::Vector{T}

    lkeep::INT
    keep::Vector{INT}

    lfact::INT
    fact::Vector{T}

    lifact::INT
    ifact::Vector{INT}

    iwork::Vector{INT}
    lwork::INT
    work::Vector{T}

    opt::Ma57Options
    logger::MadNLPLogger
end


function Ma57Solver(
    csc::SparseMatrixCSC{T,INT};
    opt = Ma57Options(),
    logger = MadNLPLogger(),
) where {T,INT}
    I, J = findIJ(csc)

    icntl = ma57_default_icntl(INT)
    cntl = ma57_default_cntl(T)

    cntl[1] = opt.ma57_pivtol
    icntl[1] = -1
    icntl[2] = -1
    icntl[3] = -1
    icntl[6] = opt.ma57_pivot_order
    icntl[15] = opt.ma57_automatic_scaling ? 1 : 0
    icntl[11] = opt.ma57_block_size
    icntl[12] = opt.ma57_node_amalgamation

    info = Vector{INT}(undef, 40)
    rinfo = Vector{T}(undef, 20)

    lkeep = INT(5 * csc.n + nnz(csc) + max(csc.n, nnz(csc)) + 42)
    keep = Vector{INT}(undef, lkeep)

    HSL.ma57ar(
        T,
        INT,
        INT(csc.n),
        INT(nnz(csc)),
        I,
        J,
        lkeep,
        keep,
        Vector{INT}(undef, 5 * csc.n),
        icntl,
        info,
        rinfo,
    )

    info[1] < 0 && throw(SymbolicException())

    lfact = ceil(INT, opt.ma57_pre_alloc * info[9])
    lifact = ceil(INT, opt.ma57_pre_alloc * info[10])

    fact = Vector{T}(undef, lfact)
    ifact = Vector{INT}(undef, lifact)
    iwork = Vector{INT}(undef, csc.n)
    lwork = INT(csc.n)
    work = Vector{T}(undef, lwork)

    return Ma57Solver{T,INT}(
        csc,
        I,
        J,
        icntl,
        cntl,
        info,
        rinfo,
        lkeep,
        keep,
        lfact,
        fact,
        lifact,
        ifact,
        iwork,
        lwork,
        work,
        opt,
        logger,
    )
end

function factorize!(M::Ma57Solver{T,INT}) where {T,INT}
    while true
        HSL.ma57br(
            T,
            INT,
            M.csc.n,
            nnz(M.csc),
            M.csc.nzval,
            M.fact,
            M.lfact,
            M.ifact,
            M.lifact,
            M.lkeep,
            M.keep,
            M.iwork,
            M.icntl,
            M.cntl,
            M.info,
            M.rinfo,
        )
        if M.info[1] == -3 || M.info[1] == 10
            M.lfact = ceil(INT, M.opt.ma57_pre_alloc * M.info[17])
            resize!(M.fact, M.lfact)
            @debug(M.logger, "Reallocating memory: lfact ($(M.lfact))")
        elseif M.info[1] == -4 || M.info[1] == 11
            M.lifact = ceil(INT, M.opt.ma57_pre_alloc * M.info[18])
            resize!(M.ifact, M.lifact)
            @debug(M.logger, "Reallocating memory: lifact ($(M.lifact))")
        elseif M.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve_linear_system!(M::Ma57Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    HSL.ma57cr(
        T,
        INT,
        one(INT),
        INT(M.csc.n),
        M.fact,
        M.lfact,
        M.ifact,
        M.lifact,
        one(INT),
        rhs,
        INT(M.csc.n),
        M.work,
        M.lwork,
        M.iwork,
        M.icntl,
        M.info,
    )
    M.info[1] < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma57Solver) = true
function inertia(M::Ma57Solver)
    return (M.info[25] - M.info[24], Int64(M.csc.n) - M.info[25], M.info[24])
end
function improve!(M::Ma57Solver)
    if M.cntl[1] == M.opt.ma57_pivtolmax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.cntl[1] = min(M.opt.ma57_pivtolmax, M.cntl[1]^0.75)
    @debug(M.logger, "improved quality: pivtol = $(M.cntl[1])")
    return true
end

introduce(::Ma57Solver) = "ma57 v$(HSL.MA57_version())"
input_type(::Type{Ma57Solver}) = :csc
default_options(::Type{Ma57Solver}) = Ma57Options()
MadNLP._default_ls_options(ls::Val{:Ma57Solver}) = Ma57Options()
is_supported(::Type{Ma57Solver}, ::Type{T}) where T <: AbstractFloat = HSL.is_supported(Val(:ma57), T)
