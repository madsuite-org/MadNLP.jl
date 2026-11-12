# Options of the filter line-search interior-point method

# Convert a barrier update built with another float type (e.g. `MonotoneUpdate()`,
# which defaults to Float64) to the precision of the solver.
function parse_option(::Type{AbstractBarrierUpdate{T}}, barrier::AbstractBarrierUpdate) where T
    B = Base.typename(typeof(barrier)).wrapper
    fields = (getfield(barrier, k) for k in fieldnames(typeof(barrier)))
    return try
        B{T}(fields...)
    catch
        error("[options] The barrier update $(typeof(barrier)) cannot be converted to precision $T.\n"*
              "Please instantiate it with the precision of the model, e.g. $(nameof(B)){$T}().")
    end
end

"""
    AbstractPreset

A set of options chosen together for a sparse model, given as a type: `madnlp(nlp; preset =
ShiftedPreset)`. The preset also chooses the options of the linear solver, through
`default_options(nlp, preset, kkt_system, linear_solver)`. Options given explicitly take
precedence over those of the preset.
"""
abstract type AbstractPreset end

"""
    ShiftedPreset

The default preset on a GPU: the augmented KKT system factorized LDLᵀ with small
pivots bounded, after Ruiz scaling and with a dual shift (both 1e-10 in `Float64`, see
[`preset_perturbation`](@ref MadNLPCore.preset_perturbation)), each solve refined by GMRES on the
unshifted system, with the quality-function barrier rule (`QualityFunctionUpdate(sigma_max = 2,
mu_decrease_max = 10, mu_min = 1e-3 tol)`) and slack reset. With `kkt_system = SparseCondensedKKTSystem` the
same options apply to the condensed system: the equalities are enforced and kept uncondensed, with
the inequality rows of 20 or more nonzeros, as the border of the condensed matrix, and the other
inequality rows are condensed (`kkt_options = Dict(:condense_max_nnz => N)` sets the threshold;
the KKT options given are merged over the preset's).
On the host the factorization is CHOLMOD's (MUMPS without
Julia's GPL libraries), on an NVIDIA GPU cuDSS's; GMRES needs the Krylov package, without which a
solve is refined by Richardson iterations. The options beyond the linear solver apply to
`SparseKKTSystem` and `SparseCondensedKKTSystem`. On AMD, Intel and Apple GPUs the factorization is SparseDirectSolver.jl's
static LDLᵀ with the same pivot bound.
"""
struct ShiftedPreset <: AbstractPreset end

"""
    AugmentedPreset

The default preset on the host: the augmented KKT system with the default sparse linear
solver, its default options and the monotone barrier rule.
"""
struct AugmentedPreset <: AbstractPreset end

"""
    CondensedPreset

The condensed KKT system (`SparseCondensedKKTSystem`): the equality constraints relaxed as pairs
of inequalities and the bounds relaxed, with zero initial multipliers. The inequality rows of 20 or
more nonzeros are kept uncondensed, as the border of the condensed matrix, which keeps it
indefinite; it is factorized LDLᵀ with small pivots bounded on the host (CHOLMOD, as `ShiftedPreset`)
and without a pivot bound on an NVIDIA GPU (cuDSS). The condensed matrix is scaled by Ruiz's
equilibration of the augmented system before condensation (`AugmentedRuizScaler`), each solve is
refined by GMRES where the Krylov package is loaded, and the barrier rule and slack reset are those
of `ShiftedPreset` (`QualityFunctionUpdate(sigma_max = 2, mu_decrease_max = 10, mu_min = 1e-3 tol)`).
"""
struct CondensedPreset <: AbstractPreset end

"""
    EqCondensedPreset

The condensed KKT system (`SparseCondensedKKTSystem`) with the equality constraints enforced and
condensed through a dual shift δ (`kkt_options` `:condense_equalities => true`, `:equality_shift`,
1e-8 in `Float64`): the multiplier of an equality row is eliminated with the weight 1 / (δ - Σd), so
the condensed matrix is positive definite and is factorized by Cholesky (cuDSS on an NVIDIA GPU,
CHOLMOD on the host) without pivoting, with the monotone barrier rule and Richardson refinement on
the unshifted system. A model with dense Jacobian rows (the rule of `SparseCondensedKKTSystem`'s
`dense_row_abs`, `dense_row_rel` and `dense_row_fill_min`, at their defaults) is solved with
[`ShiftedPreset`](@ref) instead; `dense_row_fallback = false` keeps this preset.
"""
struct EqCondensedPreset <: AbstractPreset end

# The KKT system of a sparse model under a preset, when `kkt_system` is not given.
preset_kkt_system(::Type{<:AbstractPreset}) = SparseKKTSystem
preset_kkt_system(::Type{CondensedPreset}) = SparseCondensedKKTSystem
preset_kkt_system(::Type{EqCondensedPreset}) = SparseCondensedKKTSystem

# The presets whose sparse KKT systems are factorized LDLᵀ with small pivots bounded: the condensed
# system can keep dense rows uncondensed, which leaves it indefinite.
const PivotBoundedPreset = Union{ShiftedPreset, CondensedPreset}
const SparsePresetKKTSystem = Union{SparseKKTSystem, SparseCondensedKKTSystem}

# The host default; the GPU option preset (MadNLPCoreGPUExt) makes it ShiftedPreset on GPU models.
default_preset(nlp) = AugmentedPreset

# The linear solver of the shifted preset on the host: the default sparse one, unless a package
# with an LDLᵀ factorization that bounds small pivots provides one (MadNLP: CHOLMOD).
shifted_sparse_solver(nlp) = default_sparse_solver(nlp)

const SHIFTED_SIGMA_MAX = 2.0
# under ShiftedPreset, SparseCondensedKKTSystem condenses the inequality rows with fewer nonzeros
const SHIFTED_CONDENSE_MAX_NNZ = 20
# under ShiftedPreset the barrier parameter decreases at most tenfold per iteration
const SHIFTED_MU_DECREASE_MAX = 10.0
const SHIFTED_MU_MIN_RATIO = 1.0e-3

"""
    preset_perturbation(T)

The dual shift of [`ShiftedPreset`](@ref) and the pivot bound of its LDLᵀ factorization, and of
[`CondensedPreset`](@ref)'s, in precision `T`: 1e-10 in `Float64`. In another precision it is
scaled by `(eps(T) / eps(Float64))^(5/8)`, which keeps it at about the same place, on a logarithmic
scale, between the unit roundoff and 1 (about 3e-5 in `Float32`).
"""
preset_perturbation(::Type{T}) where {T} =
    T(1.0e-10 * exp2(5 * (exponent(eps(T)) - exponent(eps(Float64))) / 8))

# EqCondensedPreset falls back to ShiftedPreset on a model with dense Jacobian rows, which would
# either be kept as an indefinite border (not factorizable by Cholesky) or fill the condensed matrix.
resolve_preset(nlp, preset) = preset
resolve_preset(nlp, ::Type{EqCondensedPreset}) = has_dense_rows(nlp) ? ShiftedPreset : EqCondensedPreset

function has_dense_rows(nlp)
    m = MadCore.get_ncon(nlp)
    m == 0 && return false
    I = similar(MadCore.get_x0(nlp), Int, MadCore.get_nnzj(nlp))
    J = similar(I)
    MadCore.jac_structure!(nlp, I, J)
    return !isempty(MadCore._select_dense_rows(Array(I), MadCore.get_nvar(nlp), m, MadCore.get_nnzh(nlp), 512, 0.10, 1e7))
end

# The options of a preset beyond the KKT system and the linear solver.
preset_options(::Type{<:AbstractPreset}, ::Type{T}, kkt_system, tol) where {T} = (;)
# CondensedPreset keeps the inequality rows of CONDENSED_CONDENSE_MAX_NNZ or more nonzeros out of the
# condensation, as the border of the condensed matrix
const CONDENSED_CONDENSE_MAX_NNZ = 20
function preset_options(::Type{CondensedPreset}, ::Type{T}, kkt_system, tol) where {T}
    kkt_system == SparseCondensedKKTSystem || return (;)
    condensed = (;
        kkt_options = Dict{Symbol, Any}(:condense_max_nnz => CONDENSED_CONDENSE_MAX_NNZ),
        barrier = QualityFunctionUpdate{T}(sigma_max = T(SHIFTED_SIGMA_MAX), mu_decrease_max = T(SHIFTED_MU_DECREASE_MAX), mu_min = T(SHIFTED_MU_MIN_RATIO) * tol),
        slack_reset = true, kkt_scaler = AugmentedRuizScaler(),
    )
    # GMRES refinement where the Krylov package is loaded (MadNLP loads it)
    return Base.get_extension(MadCore, :MadCoreKrylovExt) === nothing ? condensed : merge(condensed, (; iterator = KrylovIterator))
end
function preset_options(::Type{ShiftedPreset}, ::Type{T}, kkt_system, tol) where {T}
    kkt_system in (SparseKKTSystem, SparseCondensedKKTSystem) || return (;)
    shifted = (;
        barrier = QualityFunctionUpdate{T}(sigma_max = T(SHIFTED_SIGMA_MAX), mu_decrease_max = T(SHIFTED_MU_DECREASE_MAX), mu_min = T(SHIFTED_MU_MIN_RATIO) * tol),
        slack_reset = true, kkt_scaler = RuizScaler(), dual_shift = preset_perturbation(T),
    )
    if kkt_system == SparseCondensedKKTSystem
        # the shifted method on the condensed system: the equalities are enforced and kept, with
        # the inequality rows of SHIFTED_CONDENSE_MAX_NNZ or more nonzeros, as the border of the
        # condensed matrix; the dual shift takes the place of the border perturbation
        shifted = merge(
            shifted, (;
                equality_treatment = EnforceEquality, fixed_variable_treatment = MakeParameter,
                dual_initialization_method = DualInitializeLeastSquares,
                kkt_options = Dict{Symbol, Any}(:condense_max_nnz => SHIFTED_CONDENSE_MAX_NNZ, :border_delta => 0.0),
            )
        )
    end
    # GMRES refinement where the Krylov package is loaded (MadNLP loads it)
    return Base.get_extension(MadCore, :MadCoreKrylovExt) === nothing ? shifted : merge(shifted, (; iterator = KrylovIterator))
end
function preset_options(::Type{EqCondensedPreset}, ::Type{T}, kkt_system, tol) where {T}
    kkt_system == SparseCondensedKKTSystem || return (;)
    return (;
        equality_treatment = EnforceEquality, fixed_variable_treatment = MakeParameter,
        dual_initialization_method = DualInitializeLeastSquares,
        kkt_options = Dict{Symbol, Any}(:condense_equalities => true, :equality_shift => Float64(100 * preset_perturbation(T)), :equality_shift_inertia_retry => 1),
    )
end

"""
    default_options(nlp, preset, kkt_system, linear_solver)

The options of `linear_solver` for the KKT system `kkt_system` of `nlp` under `preset`, which
`load_options` reads. Falls back to `default_options(nlp, kkt_system, linear_solver)`. Packages
specialize it per preset; the pivot-bounded LDLᵀ of `ShiftedPreset` and `CondensedPreset`
dispatches on `PivotBoundedPreset` and `SparsePresetKKTSystem` (MadNLP for CHOLMOD, cuMadNLP for
cuDSS, the GPU extension of MadNLPCore for SparseDirectSolver.jl).
"""
default_options(nlp::AbstractNLPModel, ::Type{<:AbstractPreset}, kkt_system, linear_solver) =
    default_options(nlp, kkt_system, linear_solver)


"""
Option | Default Value | Description
:--- | :--- | :---
||
__Primary options__||
||
tol                            | 1e-8                 | termination tolerance on KKT residual
callback                       | [`SparseCallback`](@ref) | type of callback (`SparseCallback` or `DenseCallback`)
kkt\\_system                   | [`SparseKKTSystem`](@ref) | type of primal-dual KKT system
linear\\_solver                | `DummySolver` (`MumpsSolver` with MadNLP) | linear solver used for solving primal-dual KKT system; `CHOLMODSolver` under `ShiftedPreset` with MadNLP
preset                         | [`AugmentedPreset`](@ref) | set of options chosen together for a sparse model (an `AbstractPreset`): `AugmentedPreset`, `ShiftedPreset` (the default on a GPU) or `CondensedPreset`; options given explicitly take precedence
||
__General options__||
||
rethrow\\_error                | true                 | rethrow any error encountered during the algorithm
disable\\_garbage\\_collector  | false                | disable garbage collector in MadNLP
blas\\_num\\_threads           | 1                    | number of threads to use in the BLAS backend
iterator                       | `RichardsonIterator` | iterative refinement method used to solve the KKT system; `KrylovIterator` under `ShiftedPreset` where Krylov is loaded
intermediate\\_callback        | `NoUserCallback()`   | intermediate callback called at each IPM iteration
kkt\\_options                  | `Dict{Symbol,Any}()` | additional options passed to `create_kkt_system`
kkt\\_scaler                   | `NoScaler()`         | scaling of the matrix factorized by a sparse KKT system (an `AbstractScaler`); `RuizScaler()` under `ShiftedPreset`
primal\\_shift                 | 0.0                  | shift added to the primal diagonal of the matrix factorized by a sparse KKT system
dual\\_shift                   | 0.0                  | shift subtracted from the dual diagonal of the matrix factorized by a sparse KKT system; `preset_perturbation(T)` (1e-10 in `Float64`) under `ShiftedPreset`
equality\\_shift\\_adaptive      | `false`              | move `equality_shift` tenfold from Richardson refinement feedback (slow steady contraction: smaller; stagnation: larger), within [1e-10, 1e-4]
equality\\_shift\\_scaled        | `false`              | scale the shift of each condensed equality row by ‖J_i‖²
equality\\_shift               | 0.0                  | shift δ of the equality rows condensed by `SparseCondensedKKTSystem` (weight 1 / (δ - Σd)); 0 keeps the KKT option `equality_shift` (1e-8 in `Float64` under `EqCondensedPreset`)
__Output options__||
||
output\\_file                  | `""`                 | if not `""`, the output log is teed to the file at this path
print\\_level                  | INFO                 | verbosity level in MadNLP
file\\_print\\_level           | INFO                 | verbosity level in file output
||
__Termination options__||
||
max\\_iter                     | 3000                 | maximum number of interior-point iterations
max\\_wall\\_time               | 1e6                  | maximum wall time in seconds
acceptable\\_tol               | 1e-6                 | acceptable tolerance on KKT residual
acceptable\\_iter              | 15                   | number of acceptable iterates before stopping algorithm
diverging\\_iterates\\_tol     | 1e20                | threshold on KKT residual to declare algorithm as diverging
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
jacobian\\_regularization\\_exponent | 1/4                | exponent for mu in the regularization of rank-deficient Jacobian
jacobian\\_regularization\\_value  | 1e-8                 | size of regularization for rank-deficient Jacobian
||
slack\\_reset                   | `false`              | in the regular phase, move the slacks onto the constraints after an accepted step, if every original row holds to `slack_reset_tol`; `true` under `ShiftedPreset`
relaxed\\_equality\\_complementarity | `true`     | with `RelaxEquality`, count the complementarity of the two bounds of a relaxed equality in the termination test and the barrier update; `false` leaves them out (they are an artifact of the relaxation: the equality needs only `|g - b| ≤ tol` and its multiplier `zˡ - zᵘ`)
relaxed\\_equality\\_active\\_side | `false`    | with `RelaxEquality`, count only the active side (the larger dual) of the two bounds of each relaxed equality in the termination test and the barrier update
relaxed\\_equality\\_step\\_active\\_side | `false` | with `RelaxEquality`, only the active side (the larger dual) of each relaxed equality limits the primal step (fraction to the boundary); the slack of the trial point is projected back inside its interval
relaxed\\_equality\\_termination\\_only | `false` | apply `relaxed_equality_complementarity` and `relaxed_equality_active_side` to the termination test only; the barrier update counts every bound
relaxed\\_equality\\_continuation | `0`                  | with `RelaxEquality`, a continuation on the relaxation of each equality `g(x) = b`: its slack is bounded by `b ± τ` with `τ = max(τ_min, κ μ max(1, |b|))`, `κ` this value, `τ_min` the relaxation of `bound_relax_factor`; `τ` is tightened after every barrier update, never widened, and is `τ_min` at convergence. `0` keeps `τ_min` throughout
separate\\_slack\\_step         | `false`              | in the filter line search, the fraction to the boundary of the variables limits their step α and that of the slacks a separate step α_s; the slacks move by min(α, α_s), so a slack at its bound does not hold back the variables
slack\\_reset\\_tol             | 1e-6                 | violation of the original rows allowed for the slack reset
watchdog                      | `true`               | watchdog line search: after `watchdog_shortened_iter_trigger` consecutive shortened steps, take up to `watchdog_trial_iter_max` full steps and roll back if they make no progress
watchdog\\_shortened\\_iter\\_trigger | 10            | consecutive shortened steps that start the watchdog
watchdog\\_trial\\_iter\\_max    | 3                    | full steps the watchdog tries before rolling back
||
__Feasible restoration__||
||
soft\\_resto\\_pderror\\_reduction\\_factor | 0.9999          | required reduction in primal-dual error in the soft restoration phase
required\\_infeasibility\\_reduction        | 0.9             | required reduction of infeasibility before leaving restoration phase
feasibility\\_fallback       | `true`               | after a failure exit (a failed restoration phase or line search, a detected local infeasibility, diverging iterates), retry by projected gradient descent on the constraint violation, from where restoration started or from the current point; at most 3 times per solve
feasibility\\_fallback\\_max\\_iter | 10000     | iteration limit of the fallback
feasibility\\_fallback\\_reduction | 1e-2         | the fallback stops once the constraint violation falls by this factor
feasibility\\_fallback\\_method | `:gradient`         | `:gradient` for projected gradient descent, `:gauss_newton` for Gauss-Newton steps solved by CGLS, with Jacobian products by finite differences of the constraints and at most 500 iterations
||
__Line search__||
||
obj\\_max\\_inc                | 5.                   | upper bound on the acceptable increase of barrier objective function
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
barrier                        | [`MonotoneUpdate`](@ref) | algorithm to update barrier parameter; `QualityFunctionUpdate(sigma_max = 2, mu_decrease_max = 10, mu_min = 1e-3 * tol)` under `ShiftedPreset`
tau\\_min                      | 0.99                 | lower bound on fraction-to-the-boundary parameter tau
||
"""
@kwdef mutable struct MadNLPOptions{T} <: AbstractOptions
    # Primary options
    tol::T
    callback::Type
    kkt_system::Type
    linear_solver::Type # any type implementing `create_linear_solver`; see MadCore

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
    fixed_variable_treatment::Type = kkt_system <: SparseCondensedKKTSystem ? RelaxBound : MakeParameter
    equality_treatment::Type = kkt_system <: SparseCondensedKKTSystem ? RelaxEquality : EnforceEquality
    bound_relax_factor::T = 1e-8
    jacobian_constant::Bool = false
    hessian_constant::Bool = false
    hessian_approximation::Type = ExactHessian
    quasi_newton_options::QuasiNewtonOptions{T} = QuasiNewtonOptions{T}()
    inertia_correction_method::Type = InertiaAuto
    inertia_free_tol::T = 0.
    default_primal_regularization::T = 0.
    default_dual_regularization::T = 0.
    # scaling and diagonal shifts of the matrix factorized by a sparse KKT system; the
    # iterative refinement works with the unscaled, unshifted system
    kkt_scaler::AbstractScaler = NoScaler()
    primal_shift::T = 0.
    dual_shift::T = 0.
    # the shift δ of the equality rows that SparseCondensedKKTSystem condenses (condense_equalities);
    # 0 keeps the KKT system's value (EqCondensedPreset: 100 preset_perturbation(T))
    equality_shift::T = 0.
    # δ from Richardson refinement feedback, and δ_i = δ ‖J_i‖² (SparseCondensedKKTSystem, condense_equalities)
    equality_shift_adaptive::Bool = false
    equality_shift_scaled::Bool = false
    # δ = clamp(1 / max D_I, equality_shift, 1e-6) from the condensed inequality weights, with the retry before restoration
    equality_shift_from_weights::Bool = false
    # the largest equality shift the adaptive rules and the retry before restoration may reach
    equality_shift_max::T = 1e-4
    # primal shift of the condensed factorization = primal_shift_ratio * the equality shift (0: off)
    primal_shift_ratio::T = 0.
    # equality_shift_from_weights: delta = clamp(factor / max D_I, equality_shift, 1e-6)
    equality_shift_weight_factor::T = 1.
    # pivot feedback: keep the smallest/largest pivot ratio of the condensed factorization at this target (0: off)
    equality_shift_pivot_target::T = 0.
    # wrong inertia: retry the condensed factorization with a larger equality shift before the Hessian is
    # perturbed (-1 keeps the KKT system's value: 1 under EqCondensedPreset; 0: off; 1: for that step;
    # 2: kept as the base; 3: kept, decaying tenfold per factorization; 4: kept, lowered after a slowly
    # contracting refinement; 5: as 4 with a backoff after failed retries; 6: as 1 with that backoff)
    equality_shift_inertia_retry::Int = -1
    # the retry only once the barrier parameter is at most this (the stall it addresses is near the solution)
    equality_shift_retry_mu::T = Inf
    # the factor by which each retry raises the equality shift
    equality_shift_retry_factor::T = 100.
    # multiplier on the refinement's step budget for a retried factorization
    equality_shift_retry_refinement::Int = 3
    # fallback: :none, :partial (GMRES + quality-function barrier), :shifted (ShiftedPreset) or
    # :partial_then_shifted, after a failed solve or (fallback_on_solve_failure) a refined KKT solve that
    # failed after improve! ran out; warm start from where it stopped
    fallback::Symbol = :none
    fallback_on_solve_failure::Bool = true
    fallback_warm_start::Bool = true

    # initialization options
    dual_initialized::Bool = false
    dual_initialization_method::Type = kkt_system <: SparseCondensedKKTSystem ? DualInitializeSetZero : DualInitializeLeastSquares
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
    feasibility_fallback::Bool = true
    feasibility_fallback_max_iter::Int = 10_000
    feasibility_fallback_reduction::T = 1.0e-2
    feasibility_fallback_method::Symbol = :gradient

    # Line search
    obj_max_inc::T = 5.
    max_soc::Int = 4
    alpha_min_frac::T = 0.05
    s_theta::T = 1.1
    s_phi::T = 2.3
    eta_phi::T = 1e-4
    kappa_soc::T = 0.99
    slack_reset::Bool = false
    slack_reset_tol::T = 1.0e-6
    relaxed_equality_complementarity::Bool = true
    relaxed_equality_active_side::Bool = false
    relaxed_equality_step_active_side::Bool = false
    relaxed_equality_termination_only::Bool = false
    separate_slack_step::Bool = false
    relaxed_equality_continuation::T = 0.0
    watchdog::Bool = true
    watchdog_shortened_iter_trigger::Int = 10
    watchdog_trial_iter_max::Int = 3
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
    preset::Type{<:AbstractPreset} = AugmentedPreset
end


# smart option presets
function MadNLPOptions{T}(
    nlp::AbstractNLPModel{T};
    preset::Type{<:AbstractPreset} = default_preset(nlp),
    dense_row_fallback::Bool = true,
    kwargs...,
) where {T}
    return _preset_madnlp_options(T, nlp; preset = dense_row_fallback ? resolve_preset(nlp, preset) : preset, kwargs...)
end
function _preset_madnlp_options(
    ::Type{T}, nlp::AbstractNLPModel{T};
    preset::Type{<:AbstractPreset},
    dense_callback = is_dense_callback(nlp),
    callback = dense_callback ? DenseCallback : SparseCallback,
    kkt_system = dense_callback ? DenseCondensedKKTSystem : preset_kkt_system(preset),
    linear_solver = dense_callback ? default_dense_solver(nlp) :
        preset <: Union{PivotBoundedPreset, EqCondensedPreset} && kkt_system <: SparsePresetKKTSystem ? shifted_sparse_solver(nlp) : default_sparse_solver(nlp),
    tol = get_tolerance(T,kkt_system)
) where {T}
    return MadNLPOptions{T}(;
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        preset = preset,
        preset_options(preset, T, kkt_system, tol)...,
    )
end

function check_option_sanity(options)
    is_kkt_dense = options.kkt_system <: AbstractDenseKKTSystem || options.kkt_system <: SchurComplementKKTSystem
    is_hess_approx_dense = options.hessian_approximation <: Union{BFGS, DampedBFGS}
    if !(options.feasibility_fallback_method in (:gradient, :gauss_newton))
        error("[options] feasibility_fallback_method must be :gradient or :gauss_newton.")
    end
    if input_type(options.linear_solver) == :csc && is_kkt_dense
        error("[options] Sparse Linear solver is not supported in dense mode.\n"*
              "Please use a dense linear solver or change `kkt_system` ")
    end
    if is_hess_approx_dense && !is_kkt_dense
        error("[options] BFGS and DampedBFGS quasi-Newton approximations\n"*
              "require a dense KKT system (DenseKKTSystem or DenseCondensedKKTSystem).")
    end
end

function load_options(nlp::AbstractNLPModel{T,VT}; options...) where {T, VT}

    primary_opt, options = _get_primary_options(options)

    # Initiate interior-point options
    opt_ipm = MadNLPOptions{T}(nlp; primary_opt...)
    preset_kkt_options = copy(opt_ipm.kkt_options)
    linear_solver_options = set_options!(opt_ipm, options)
    # the KKT options given take precedence over the preset's, key by key
    opt_ipm.kkt_options = merge(preset_kkt_options, opt_ipm.kkt_options)

    check_option_sanity(opt_ipm)
    # Initiate linear-solver options
    opt_linear_solver = default_options(nlp, opt_ipm.preset, opt_ipm.kkt_system, opt_ipm.linear_solver)
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

    # Print remaining options (unsupported)
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


# The options `kkt_scaler`, `primal_shift` and `dual_shift` that differ from their defaults, as
# keyword arguments of `create_kkt_system`; a KKT system that does not take them is unaffected
# as long as they are left at their defaults.
function factorization_options(opt)
    kw = (;)
    opt.kkt_scaler isa NoScaler || (kw = merge(kw, (; kkt_scaler = opt.kkt_scaler)))
    iszero(opt.primal_shift) || (kw = merge(kw, (; primal_shift = opt.primal_shift)))
    iszero(opt.dual_shift) || (kw = merge(kw, (; dual_shift = opt.dual_shift)))
    # the equality-shift options belong to SparseCondensedKKTSystem; another KKT system (the dense-row
    # fallback to ShiftedPreset) refuses them
    opt.kkt_system <: SparseCondensedKKTSystem || return kw
    iszero(opt.equality_shift) || (kw = merge(kw, (; equality_shift = Float64(opt.equality_shift))))
    opt.equality_shift_adaptive && (kw = merge(kw, (; equality_shift_adaptive = true)))
    opt.equality_shift_scaled && (kw = merge(kw, (; equality_shift_scaled = true)))
    opt.equality_shift_from_weights && (kw = merge(kw, (; equality_shift_from_weights = true)))
    opt.equality_shift_max == 1e-4 || (kw = merge(kw, (; equality_shift_max = Float64(opt.equality_shift_max))))
    iszero(opt.primal_shift_ratio) || (kw = merge(kw, (; primal_shift_ratio = Float64(opt.primal_shift_ratio))))
    opt.equality_shift_weight_factor == 1 || (kw = merge(kw, (; equality_shift_weight_factor = Float64(opt.equality_shift_weight_factor))))
    iszero(opt.equality_shift_pivot_target) || (kw = merge(kw, (; equality_shift_pivot_target = Float64(opt.equality_shift_pivot_target))))
    opt.equality_shift_inertia_retry < 0 || (kw = merge(kw, (; equality_shift_inertia_retry = opt.equality_shift_inertia_retry)))
    opt.equality_shift_retry_factor == 100 || (kw = merge(kw, (; equality_shift_retry_factor = Float64(opt.equality_shift_retry_factor))))
    opt.equality_shift_retry_refinement == 3 || (kw = merge(kw, (; equality_shift_retry_refinement = opt.equality_shift_retry_refinement)))
    return kw
end
