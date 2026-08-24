mutable struct RobustRestorer{T, VT}
    obj_val_R::T
    f_R::VT
    x_ref::VT

    theta_ref::T
    D_R::VT
    obj_val_R_trial::T

    pp::VT
    nn::VT
    zp::VT
    zn::VT

    dpp::VT
    dnn::VT
    dzp::VT
    dzn::VT

    pp_trial::VT
    nn_trial::VT

    inf_pr_R::T
    inf_du_R::T
    inf_compl_R::T

    mu_R::T
    tau_R::T
    zeta::T

    filter::Vector{Tuple{T,T}}
end

abstract type AbstractInertiaCorrector end
struct InertiaAuto <: AbstractInertiaCorrector end
struct InertiaBased <: AbstractInertiaCorrector end
struct InertiaIgnore <: AbstractInertiaCorrector end
struct InertiaFree{
    T,
    VT <: AbstractVector{T},
    KKTVec <: AbstractKKTVector{T, VT}
} <: AbstractInertiaCorrector
    p0::KKTVec
    d0::KKTVec
    t::VT
    wx::VT
    g::VT
end


"""
    AbstractBarrierUpdate{T}

Abstraction used to implement the different rules to update the barrier parameter.
The barrier is updated using either a monotone rule or an adaptive rule.

"""
abstract type AbstractBarrierUpdate{T} end

"""
    MonotoneUpdate{T} <: AbstractBarrierUpdate{T}

Update the barrier parameter using the classical Fiacco-McCormick monotone rule.

"""
@kwdef mutable struct MonotoneUpdate{T} <: AbstractBarrierUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
end
function MonotoneUpdate(tol::T, barrier_tol_factor) where T
    return MonotoneUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

"""
    AbstractAdaptiveUpdate{T} <: AbstractBarrierUpdate{T}

Abstraction used to implement the adaptive barrier updates described in [Nocedal2009].

## References
[Nocedal2009] Nocedal, J., Wächter, A., & Waltz, R. A. (2009).
Adaptive barrier update strategies for nonlinear interior methods.
SIAM Journal on Optimization, 19(4), 1674-1693.

"""
abstract type AbstractAdaptiveUpdate{T} <: AbstractBarrierUpdate{T} end

"""
    QualityFunctionUpdate{T} <: AbstractAdaptiveUpdate{T}

Find the barrier parameter using a quality function encoding the ℓ1-norm of the KKT violations.
At each IPM iteration, the minimum of the quality function is found using a Golden search algorithm.

If no sufficient progress is made, the barrier fallbacks to a monotone rule.

## References
The algorithm is described in [Nocedal2009, Section 4].

"""
@kwdef mutable struct QualityFunctionUpdate{T} <: AbstractAdaptiveUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_max::T = 1e5
    sigma_min::T = 1e-6
    sigma_max::T = 1e2
    sigma_tol::T = 1e-2
    gamma::T = 1.0
    max_gs_iter::Int = 8
    # For non-free mode
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
    free_mode::Bool = true
    globalization::Bool = true
    n_update::Int = 0
end
function QualityFunctionUpdate(tol::T, barrier_tol_factor) where T
    return QualityFunctionUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end

"""
    LOQOUpdate{T} <: AbstractAdaptiveUpdate{T}

Find the barrier parameter using the rule used in the LOQO solver.
The rule is explicit in [Nocedal2009, Eq (3.6)].

If no sufficient progress is made, the barrier fallbacks to a monotone rule.

## References
The algorithm is described in [Nocedal2009, Section 3].

"""
@kwdef mutable struct LOQOUpdate{T} <: AbstractAdaptiveUpdate{T}
    mu_init::T = 1e-1
    mu_min::T = 1e-11
    mu_max::T = 1e5
    gamma::T = 0.1 # scale factor
    r::T = .95 # Steplength param
    mu_superlinear_decrease_power::T = 1.5
    mu_linear_decrease_factor::T = .2
    free_mode::Bool = true
    globalization::Bool = true
end
function LOQOUpdate(tol::T, barrier_tol_factor) where T
    return LOQOUpdate{T}(; mu_min=min(1e-4, tol ) / (barrier_tol_factor + 1))
end
