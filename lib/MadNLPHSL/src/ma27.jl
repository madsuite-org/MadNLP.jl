ma27_default_icntl(INT) = INT[
    6,
    6,
    0,
    2139062143,
    1,
    32639,
    32639,
    32639,
    32639,
    14,
    9,
    8,
    8,
    9,
    10,
    32639,
    32639,
    32639,
    32689,
    24,
    11,
    9,
    8,
    9,
    10,
    0,
    0,
    0,
    0,
    0,
]
ma27_default_cntl(T) = T[0.1, 1.0, 0.0, 0.0, 0.0]

@kwdef mutable struct Ma27Options <: AbstractOptions
    ma27_pivtol::Float64 = 1e-8
    ma27_pivtolmax::Float64 = 1e-4
    ma27_liw_init_factor::Float64 = 5.0
    ma27_la_init_factor::Float64 = 5.0
    ma27_meminc_factor::Float64 = 2.0
end

mutable struct Ma27Solver{T,INT} <: AbstractLinearSolver{T}
    csc::SparseMatrixCSC{T,INT}
    I::Vector{INT}
    J::Vector{INT}

    icntl::Vector{INT}
    cntl::Vector{T}

    info::Vector{INT}

    a::Vector{T}
    a_view::SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
    la::INT
    ikeep::Vector{INT}

    iw::Vector{INT}
    liw::INT
    iw1::Vector{INT}
    nsteps::Vector{INT}
    w::Vector{T}
    maxfrt::Vector{INT}

    opt::Ma27Options
    logger::MadNLPLogger
end

function Ma27Solver(
    csc::SparseMatrixCSC{T,INT};
    opt = Ma27Options(),
    logger = MadNLPLogger(),
) where {T,INT}
    I, J = findIJ(csc)
    nz = nnz(csc) |> INT

    liw = INT(2 * (2 * nz + 3 * csc.n + 1))
    iw = Vector{INT}(undef, liw)
    ikeep = Vector{INT}(undef, 3 * csc.n)
    iw1 = Vector{INT}(undef, 2 * csc.n)
    nsteps = INT[1]
    iflag = INT(0)

    icntl = ma27_default_icntl(INT)
    cntl = ma27_default_cntl(T)
    icntl[1:2] .= 0
    cntl[1] = opt.ma27_pivtol

    info = Vector{INT}(undef, 20)
    HSL.ma27ar(
        T,
        INT,
        csc.n,
        nz,
        I,
        J,
        iw,
        liw,
        ikeep,
        iw1,
        nsteps,
        iflag,
        icntl,
        cntl,
        info,
        zero(T),
    )
    info[1] < 0 && throw(SymbolicException())

    la = ceil(INT, max(nz, opt.ma27_la_init_factor * info[5]))
    a = Vector{T}(undef, la)
    a_view = view(a, 1:nnz(csc)) # _madnlp_unsafe_wrap is not used because we may resize a
    liw = ceil(INT, opt.ma27_liw_init_factor * info[6])
    resize!(iw, liw)
    maxfrt = INT[1]

    return Ma27Solver{T,INT}(
        csc,
        I,
        J,
        icntl,
        cntl,
        info,
        a,
        a_view,
        la,
        ikeep,
        iw,
        liw,
        iw1,
        nsteps,
        Vector{T}(),
        maxfrt,
        opt,
        logger,
    )
end


function factorize!(M::Ma27Solver{T,INT}) where {T,INT}
    M.a_view .= M.csc.nzval
    while true
        HSL.ma27br(
            T,
            INT,
            M.csc.n,
            nnz(M.csc),
            M.I,
            M.J,
            M.a,
            M.la,
            M.iw,
            M.liw,
            M.ikeep,
            M.nsteps,
            M.maxfrt,
            M.iw1,
            M.icntl,
            M.cntl,
            M.info,
        )
        if M.info[1] == -3
            M.liw = ceil(INT, M.opt.ma27_meminc_factor * M.liw)
            resize!(M.iw, M.liw)
            @debug(M.logger, "Reallocating memory: liw ($(M.liw))")
        elseif M.info[1] == -4
            M.la = ceil(INT, M.opt.ma27_meminc_factor * M.la)
            resize!(M.a, M.la)
            @debug(M.logger, "Reallocating memory: la ($(M.la))")
        elseif M.info[1] < 0
            throw(FactorizationException())
        else
            break
        end
    end
    return M
end

function solve_linear_system!(M::Ma27Solver{T,INT}, rhs::Vector{T}) where {T,INT}
    length(M.w) < M.maxfrt[1] && resize!(M.w, M.maxfrt[1])
    length(M.iw1) < M.nsteps[1] && resize!(M.iw1, M.nsteps[1])
    HSL.ma27cr(
        T,
        INT,
        M.csc.n,
        M.a,
        M.la,
        M.iw,
        M.liw,
        M.w,
        M.maxfrt,
        rhs,
        M.iw1,
        M.nsteps,
        M.icntl,
        M.info,
    )
    M.info[1] < 0 && throw(SolveException())
    return rhs
end

is_inertia(::Ma27Solver) = true
function inertia(M::Ma27Solver)
    dim = M.csc.n
    rank = (Int(M.info[1]) == 3) ? Int(M.info[2]) : dim
    neg = Int(M.info[15])

    return (rank - neg, dim - rank, neg)
end

function improve!(M::Ma27Solver)
    if M.cntl[1] == M.opt.ma27_pivtolmax
        @debug(M.logger, "improve quality failed.")
        return false
    end
    M.cntl[1] = min(M.opt.ma27_pivtolmax, M.cntl[1]^0.75)
    @debug(M.logger, "improved quality: pivtol = $(M.cntl[1])")
    return true
end

introduce(::Ma27Solver) = "ma27 v$(HSL.MA27_version())"
input_type(::Type{Ma27Solver}) = :csc
default_options(::Type{Ma27Solver}) = Ma27Options()
MadNLP._default_ls_options(ls::Val{:Ma27Solver}) = Ma27Options()
is_supported(::Type{Ma27Solver}, ::Type{T}) where T <: AbstractFloat = HSL.is_supported(Val(:ma27), T)
