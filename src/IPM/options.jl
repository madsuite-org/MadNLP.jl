# Options

parse_option(::Type{Module},str::String) = eval(Symbol(str))
parse_option(::Type{<:AbstractUserCallback}, f::Any) = f
parse_option(type::Type{T},i::Int64) where {T<:Enum} = type(i)

const CondensedKKTSystems = Union{SparseCondensedKKTSystem, SchurComplementCondensedKKTSystem}

function set_options!(opt::AbstractOptions, options)
    other_options = Dict{Symbol, Any}()
    for (key, val) in options
        if hasproperty(opt, key)
            T = fieldtype(typeof(opt), key)
            val isa T ? setproperty!(opt,key,val) :
                setproperty!(opt,key,parse_option(T,val))
        else
            other_options[key] = val
        end
    end
    return other_options
end

"""
Option | Default Value | Description
:--- | :--- | :---
||
__Primary options__||
||
tol                            | 1e-8                 | termination tolerance on KKT residual
callback                       | [`SparseCallback`](@ref) | type of callback (`SparseCallback` or `DenseCallback`)
kkt\\_system                   | [`SparseKKTSystem`](@ref) | type of primal-dual KKT system
linear\\_solver                | MumpsSolver          | linear solver used for solving primal-dual KKT system
||
__General options__||
||
rethrow\\_error                | true                 | rethrow any error encountered during the algorithm
disable\\_garbage\\_collector  | false                | disable garbage collector in MadNLP
blas\\_num\\_thread            | 1                    | number of threads to use in the BLAS backend
intermediate_callback          | AbstractUserCallback | Intermediate callback called at each IPM iteration
__Output options__||
||
output\\_file                  | `""`                 | if not `""`, the output log is teed to the file at this path
print\\_level                  | INFO                 | verbosity level in MadNLP
file\\_print\\_level           | INFO                 | verbosity level in file output
||
__Termination options__||
||
max\\_iter                     | 3000                 | maximum number of interior-point iterations
max\\_wall_time                | 1e6                  | maximum wall time in seconds
acceptable\\_tol               | 1e-6                 | acceptable tolerance on KKT residual
acceptable\\_iter              | 15                   | number of acceptable iterates before stopping algorithm
diverging\\_iter               | 1e20                 | threshold on KKT residual to declare algorithm as diverging
s\\_max                        | 100.0                | scaling threshold for KKT residual
||
__NLP options__||
||
kappa\\_d                      | 1e-5                 | weight for linear damping term
fixed\\_variable\\_treatment   | [`MakeParameter`](@ref) | treatment for the fixed variables (`MakeParameter` or `RelaxBound`)
equality\\_treatment           | [`EnforceEquality`](@ref) | treatment for the equality constraints (`EnforceEquality` or `RelaxEquality`)
bound\\_relax\\_factor         | 1e-8                 | factor for initial relaxation of bounds
jacobian\\_constant            | false                | set to true if the constraints are linear.
hessian\\_constant             | false                | set to true if the problem is linear or quadratic
hessian\\_approximation        | `ExactHessian` | method used to approximate the Hessian
quasi\\_newton\\_options       | QuasiNewtonOptions() | options for quasi-Newton algorithm
inertia\\_correction\\_method  | InertiaAuto          | inertia correction mechanism
inertia\\_free\\_tol           | 0.0                  | tolerance for inertia free method
default\\_primal\\_regularization | 0.0 | default regularization for primal variable blocks in the KKT system
default\\_dual\\_regularization   | 0.0 | default regularization for dual variable blocks in the KKT system
||
__Initialization__||
||
dual\\_initialized             | false                | specify if dual initial point is available
dual\\_initialization\\_method | DualInitializeLeastSquares | method to compute the initial dual multipliers
constr\\_mult\\_init\\_max     | 1e3                  | maximum allowable value in initial dual multipliers
bound\\_push                   | 1e-2                 | minimum absolute distance from the initial point to bound
bound\\_fac                    | 1e-2                 | minimum relative distance from the initial point to bound
nlp\\_scaling                  | true                 | scale nonlinear program
nlp\\_scaling\\_max\\_gradient | 100.0                | maximum gradient after NLP scaling
||
__Hessian perturbation__||
||
min\\_hessian\\_perturbation       | 1e-20                | smallest perturbation of Hessian block in inertia correction
first\\_hessian\\_perturbation     | 1e-4                 | first value tried in inertia correction
max\\_hessian\\_perturbation       | 1e20                 | largest perturbation of Hessian block in inertia correction
perturb\\_inc\\_fact\\_first       | 1e2                  | increase factor for primal perturbation for very first perturbation
perturb\\_inc\\_fact               | 8.                   | increase factor for primal perturbation
perturb\\_dec\\_fact               | 1/3                  | decrease factor for primal perturbation
jacobian\\_regularization\\_exponent | 1/4                | exponent for mu in the regularization of rank-defficient Jacobian
jacobian\\_regularization\\_value  | 1e-8                 | size of regularization for rank-defficient Jacobian
||
__Feasible restoration__||
||
soft\\_resto\\_pderror\\_reduction\\_factor | 0.9999          | required reduction in primal-dual error in the soft restoration phase
required\\_infeasibility\\_reduction        | 0.9             | required reduction of infeasibility before leaving restoration phase
||
__Line search__||
||
obj\\_max\\_inc                | 5.                   | upper bound on the acceptable increase of barrier objective function
kappha\\_soc                   | 0.99                 | factor in the sufficient reduction rule for second order correction
max\\_soc                      | 4                    | maximum number of second order correction trial steps at each iteration
alpha\\_min\\_frac             | 0.05                 | safety factor for the minimal step size
s\\_theta                      | 1.1                  | exponent for current constraint violation in the switching rule
s\\_phi                        | 2.3                  | exponent for linear barrier function model in the switching rule
eta\\_phi                      | 1e-4                 | relaxation factor in the Armijo condition
kappa\\_soc                    | 0.99                 | factor in the sufficient reduction rule for second order correction
gamma\\_theta                  | 1e-5                 | relaxation factor in the filter margin for the constraint violation
gamma\\_phi                    | 1e-5                 | relaxation factor in the filter margin for the barrier function
delta                          | 1.0                  | multiplier for constraint violation in the switching rule
kappa\\_sigma                  | 1e10                 | factor limiting the deviation of dual variables from primal estimates
barrier\\_tol\\_factor         | 10.0                 | factor for mu in barrier stop test
rho                            | 1000.0               | value in penalty parameter update formula
||
__Barrier__||
||
barrier                        | [`MonotoneUpdate`](@ref) | algorithm to update barrier parameter
tau\\_min                      | 0.99                 | lower bound on fraction-to-the-boundary parameter tau
||
"""
@kwdef mutable struct MadNLPOptions{T} <: AbstractOptions
    # Primary options
    tol::T
    callback::Type
    kkt_system::Type
    linear_solver::Type

    # General options
    rethrow_error::Bool = true
    disable_garbage_collector::Bool = false
    blas_num_threads::Int = 1
    iterator::Type = RichardsonIterator
    intermediate_callback::AbstractUserCallback = NoUserCallback()

    # Output options
    output_file::String = ""
    print_level::LogLevels = INFO
    file_print_level::LogLevels = INFO

    # Termination options
    acceptable_tol::T = 1e-6
    acceptable_iter::Int = 15
    diverging_iterates_tol::T = 1e20
    max_iter::Int = 3000
    max_wall_time::T = 1e6
    s_max::T = 100.

    # NLP options
    kappa_d::T = 1e-5
    fixed_variable_treatment::Type = kkt_system <: CondensedKKTSystems ? MadNLP.RelaxBound : MadNLP.MakeParameter
    equality_treatment::Type = kkt_system <: CondensedKKTSystems ? MadNLP.RelaxEquality : MadNLP.EnforceEquality
    bound_relax_factor::T = 1e-8
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    hessian_approximation::Type = ExactHessian
    quasi_newton_options::QuasiNewtonOptions{T} = QuasiNewtonOptions{T}()
    inertia_correction_method::Type = InertiaAuto
    inertia_free_tol::T = 0.
    default_primal_regularization::T = 0.
    default_dual_regularization::T = 0.

    # initialization options
    dual_initialized::Bool = false
    dual_initialization_method::Type = kkt_system <: CondensedKKTSystems ? DualInitializeSetZero : DualInitializeLeastSquares
    constr_mult_init_max::T = 1e3
    bound_push::T = 1e-2
    bound_fac::T = 1e-2
    nlp_scaling::Bool = true
    nlp_scaling_max_gradient::T = 100.

    # Hessian Perturbation
    min_hessian_perturbation::T = 1e-20
    first_hessian_perturbation::T = 1e-4
    max_hessian_perturbation::T = 1e20
    perturb_inc_fact_first::T = 1e2
    perturb_inc_fact::T = 8.
    perturb_dec_fact::T = 1/3
    jacobian_regularization_exponent::T = 1/4
    jacobian_regularization_value::T = 1e-8

    # restoration options
    soft_resto_pderror_reduction_factor::T = 0.9999
    required_infeasibility_reduction::T = 0.9

    # Line search
    obj_max_inc::T = 5.
    kappha_soc::T = 0.99
    max_soc::Int = 4
    alpha_min_frac::T = 0.05
    s_theta::T = 1.1
    s_phi::T = 2.3
    eta_phi::T = 1e-4
    kappa_soc::T = 0.99
    gamma_theta::T = 1e-5
    gamma_phi::T = 1e-5
    delta::T = 1
    kappa_sigma::T = 1e10
    barrier_tol_factor::T = 10.
    rho::T = 1000.

    # Barrier
    # mu_min by courtesy of Ipopt
    barrier::AbstractBarrierUpdate{T} = MonotoneUpdate(T(tol), barrier_tol_factor)
    tau_min::T = 0.99

    # KKT system options (passed through to create_kkt_system)
    kkt_options::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

is_dense_callback(nlp) = !nlp.meta.sparse_jacobian && !nlp.meta.sparse_hessian

# smart option presets
function MadNLPOptions{T}(
    nlp::AbstractNLPModel{T};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? DenseCallback : SparseCallback,
    kkt_system = dense_callback ? DenseCondensedKKTSystem : SparseKKTSystem,
    linear_solver = dense_callback ? LapackCPUSolver : default_sparse_solver(nlp),
    tol = get_tolerance(T,kkt_system),
    # Condensed KKT systems (Sparse + Schur) relax equalities into the barrier — the SAME
    # systems selected for RelaxEquality below. Flooring those slacks at a tiny 1e-8 box blows up
    # the condensation weights z/s and ruins the conditioning of the (SPD) factorization, so relax
    # by `tol` there (feasible to ~tol, but well-conditioned). Non-condensed systems keep equalities
    # exact and use the tight 1e-8. Must match the CUDA / ROCm extension constructors so a problem
    # behaves identically on CPU and GPU.
    bound_relax_factor = (kkt_system <: CondensedKKTSystems) ? tol : T(1.0e-8),
) where {T}
    return MadNLPOptions{T}(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
    )
end

get_tolerance(::Type{T},::Type{KKT}) where {T, KKT} = 10^round(log10(eps(T))/2)
get_tolerance(::Type{T},::Type{SparseCondensedKKTSystem}) where T = 10^(round(log10(eps(T))/4))
get_tolerance(::Type{T},::Type{SchurComplementCondensedKKTSystem}) where T = 10^(round(log10(eps(T))/4))

default_sparse_solver(nlp::AbstractNLPModel) = MumpsSolver

function check_option_sanity(options)
    is_kkt_dense = options.kkt_system <: AbstractDenseKKTSystem
    is_hess_approx_dense = options.hessian_approximation <: Union{BFGS, DampedBFGS}
    # `SchurComplementCondensedKKTSystem` is a *sparse-callback* KKT system: on both CPU and
    # GPU it factorizes a sparse lower-triangular first-stage Schur complement (MumpsSolver /
    # any `:csc` solver on CPU, cuDSS on GPU), so a `:csc` linear solver is valid for it — only
    # the genuinely dense KKT systems require a dense linear solver here. It also has no dense
    # quasi-Newton wiring, so it is NOT classified as dense above: a dense BFGS/DampedBFGS
    # approximation is rejected below with a clear options error instead of failing later with
    # an obscure constructor/dispatch error.
    if input_type(options.linear_solver) == :csc && options.kkt_system <: AbstractDenseKKTSystem
        error("[options] Sparse Linear solver is not supported in dense mode.\n"*
              "Please use a dense linear solver or change `kkt_system` ")
    end
    if is_hess_approx_dense && !is_kkt_dense
        error("[options] DENSE_BFGS and DENSE_DAMPED_BFGS quasi-Newton approximations\n"*
              "require a dense KKT system (DENSE_KKT_SYSTEM or DENSE_CONDENSED_KKT_SYSTEM).")
    end
end

function print_ignored_options(logger,option_dict)
    @warn(logger,"The following options are ignored: ")
    for (key,val) in option_dict
        @warn(logger," - "*string(key))
    end
end

function _get_primary_options(options)
    primary_opt = Dict{Symbol,Any}()
    remaining_opt = Dict{Symbol,Any}()
    for (k,v) in options
        if k in [:tol, :linear_solver, :callback, :kkt_system]
            primary_opt[k] = v
        else
            remaining_opt[k] = v
        end
    end

    return primary_opt, remaining_opt
end

function load_options(nlp::AbstractNLPModel{T,VT}; options...) where {T, VT}

    primary_opt, options = _get_primary_options(options)

    # Initiate interior-point options
    opt_ipm = MadNLPOptions{T}(nlp; primary_opt...)
    linear_solver_options = set_options!(opt_ipm, options)

    check_option_sanity(opt_ipm)
    # Initiate linear-solver options
    opt_linear_solver = default_options(nlp, opt_ipm.kkt_system, opt_ipm.linear_solver)
    iterator_options = set_options!(opt_linear_solver, linear_solver_options)
    # Initiate iterator options
    opt_iterator = default_options(opt_ipm.iterator, opt_ipm.tol)
    remaining_options = set_options!(opt_iterator, iterator_options)

    # Initiate logger
    logger = MadNLPLogger(
        print_level=opt_ipm.print_level,
        file_print_level=opt_ipm.file_print_level,
        file = opt_ipm.output_file == "" ? nothing : open(opt_ipm.output_file,"w+"),
    )
    @trace(logger,"Logger is initialized.")

    # Print remaning options (unsupported)
    if !isempty(remaining_options)
        print_ignored_options(logger, remaining_options)
    end
    return (
        interior_point=opt_ipm,
        linear_solver=opt_linear_solver,
        iterative_refinement=opt_iterator,
        logger=logger,
        intermediate_callback=opt_ipm.intermediate_callback,
    )
end

"""
        default_options(
                nlp::AbstractNLPModel,
                kkt::Type,
                linear_solver::Type;
                iterator_options=Dict{Symbol,Any}(),
        )

default options for `linear_solver` associated to the KKT system `kkt_system` and `nlp`.
"""
default_options(nlp::AbstractNLPModel, kkt, linear_solver) = default_options(linear_solver)
