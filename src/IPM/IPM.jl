# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractMadNLPSolver{T} end

include("types.jl")
include("options.jl")

mutable struct MadNLPSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: AbstractKKTSystem{T},
    Model <: AbstractNLPModel{T,VT},
    CB <: AbstractCallback{T},
    Iterator <: AbstractIterator{T},
    IC <: AbstractInertiaCorrector,
    KKTVec <: AbstractKKTVector{T, VT},
    ICB
    } <: AbstractMadNLPSolver{T}

    nlp::Model
    cb::CB
    kkt::KKTSystem

    opt::MadNLPOptions{T}
    cnt::MadNLPCounters
    logger::MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::PrimalVector{T, VT, VI} # primal (after reformulation)
    y::VT # dual
    zl::PrimalVector{T, VT, VI} # dual (after reformulation)
    zu::PrimalVector{T, VT, VI} # dual (after reformulation)
    xl::PrimalVector{T, VT, VI} # primal lower bound (after reformulation)
    xu::PrimalVector{T, VT, VI} # primal upper bound (after reformulation)

    obj_val::T
    f::PrimalVector{T, VT, VI}
    c::VT

    jacl::VT

    d::KKTVec
    p::KKTVec

    _w1::KKTVec
    _w2::KKTVec
    _w3::KKTVec
    _w4::KKTVec

    x_trial::PrimalVector{T, VT, VI}
    c_trial::VT
    obj_val_trial::T

    c_slk::SubVector{T,VT,VI}
    rhs::VT

    ind_ineq::VI
    ind_fixed::VI
    ind_llb::VI
    ind_uub::VI

    x_lr::SubVector{T,VT,VI}
    x_ur::SubVector{T,VT,VI}
    xl_r::SubVector{T,VT,VI}
    xu_r::SubVector{T,VT,VI}
    zl_r::SubVector{T,VT,VI}
    zu_r::SubVector{T,VT,VI}
    dx_lr::SubVector{T,VT,VI}
    dx_ur::SubVector{T,VT,VI}
    x_trial_lr::SubVector{T,VT,VI}
    x_trial_ur::SubVector{T,VT,VI}

    iterator::Iterator

    inf_pr::T
    inf_du::T
    inf_compl::T
    inf_compl_mu::T

    theta_min::T
    theta_max::T
    mu::T
    tau::T

    alpha::T
    alpha_z::T
    ftype::String

    del_w::T
    del_w_last::T
    del_c::T

    filter::Vector{Tuple{T,T}}

    inertia_corrector::IC
    RR::Union{Nothing,RobustRestorer{T,VT}}
    intermediate_callback::ICB
    status::Status
    output::Dict
end

include("api.jl")
include("restoration.jl")
include("inertiacorrector.jl")
include("barrier.jl")


"""
    MadNLPSolver(nlp::AbstractNLPModel{T, VT}; options...) where {T, VT}

Instantiate a new `MadNLPSolver` associated to the nonlinear program
`nlp::AbstractNLPModel`. The options are passed as optional arguments.

The constructor allocates all the memory required in the interior-point
algorithm, so the main algorithm remains allocation free.

"""
function MadNLPSolver(nlp::AbstractNLPModel{T,VT}; kwargs...) where {T, VT}

    options = load_options(nlp; kwargs...)

    ipm_opt = options.interior_point
    logger = options.logger
    @assert is_supported(ipm_opt.linear_solver, T)

    cnt = MadNLPCounters(start_time=time())
    cb = create_callback(
        ipm_opt.callback,
        nlp;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
    )

    # generic options
    ipm_opt.disable_garbage_collector &&
        (GC.enable(false); @warn(logger,"Julia garbage collector is temporarily disabled"))
    set_blas_num_threads(ipm_opt.blas_num_threads; permanent=true)
    @trace(logger,"Initializing variables.")

    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    ns = length(cb.ind_ineq)
    nx = n_variables(cb)
    n = nx+ns
    m = n_constraints(cb)
    nlb = length(ind_lb)
    nub = length(ind_ub)

    @trace(logger,"Initializing KKT system.")
    kkt = create_kkt_system(
        ipm_opt.kkt_system,
        cb,
        ipm_opt.linear_solver;
        hessian_approximation=ipm_opt.hessian_approximation,
        opt_linear_solver=options.linear_solver,
        qn_options=ipm_opt.quasi_newton_options,
        ipm_opt.kkt_options...,
    )

    @trace(logger,"Initializing iterative solver.")
    iterator = ipm_opt.iterator(kkt; cnt = cnt, logger = logger, opt = options.iterative_refinement)

    x = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xl = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xu = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zl = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zu = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    f = PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    x_trial = PrimalVector(VT, nx, ns, ind_lb, ind_ub)

    d = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    p = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w1 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w2 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w3 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w4 = UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)

    jacl = VT(undef,n)
    c_trial = VT(undef, m)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    c_slk = view(c,cb.ind_ineq)
    x_lr = view(full(x), cb.ind_lb)
    x_ur = view(full(x), cb.ind_ub)
    xl_r = view(full(xl), cb.ind_lb)
    xu_r = view(full(xu), cb.ind_ub)
    zl_r = view(full(zl), cb.ind_lb)
    zu_r = view(full(zu), cb.ind_ub)
    x_trial_lr = view(full(x_trial), cb.ind_lb)
    x_trial_ur = view(full(x_trial), cb.ind_ub)
    dx_lr = view(d.xp, cb.ind_lb) # TODO
    dx_ur = view(d.xp, cb.ind_ub) # TODO

    inertia_correction_method = if ipm_opt.inertia_correction_method == InertiaAuto
        is_inertia(kkt)::Bool ? InertiaBased : InertiaFree
    else
        ipm_opt.inertia_correction_method
    end

    inertia_corrector = build_inertia_corrector(
        inertia_correction_method,
        VT,
        n, m, nlb, nub, ind_lb, ind_ub
    )

    cnt.init_time = time() - cnt.start_time

    return MadNLPSolver(
        nlp, cb, kkt,
        ipm_opt, cnt, options.logger,
        n, m, nlb, nub,
        x, y, zl, zu, xl, xu,
        zero(T), f, c,
        jacl,
        d, p,
        _w1, _w2, _w3, _w4,
        x_trial, c_trial, zero(T), c_slk, rhs,
        cb.ind_ineq, cb.ind_fixed, cb.ind_llb, cb.ind_uub,
        x_lr, x_ur, xl_r, xu_r, zl_r, zu_r, dx_lr, dx_ur, x_trial_lr, x_trial_ur,
        iterator,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
        " ",
        zero(T), zero(T), zero(T),
        Tuple{T, T}[],
        inertia_corrector, nothing,
        options.intermediate_callback,
        INITIAL, Dict(),
    )

end

include("utils.jl")
include("kernels.jl")
include("callbacks.jl")
include("factorization.jl")
include("line_search.jl")
include("solver.jl")
