module MadNLPCore

import MadCore
for name in MadCore.public_names()
    @eval import MadCore: $name
end
for field in MadCore.SOLVER_FIELDS
    getter, setter = Symbol("get_", field), Symbol("set_", field, "!")
    @eval import MadCore: $getter, $setter
end
import NLPModels
import Base: @kwdef, copyto!, finalize, string
import LinearAlgebra: axpy!, dot, mul!, norm
import MadCore: @phase_region, @timed_region, AbstractDenseKKTSystem, AbstractIterator,
    AbstractUserCallback, BFGS, DampedBFGS, DenseCallback, DenseCondensedKKTSystem,
    DenseKKTSystem, ModelBarrier, NoUserCallback, RichardsonIterator, UserCallbackRestore,
    UserCallbackRobust, blas_num_threads, create_timer, default_dense_solver,
    has_constraints, is_dense_callback, is_valid, mul_hess_blk!, parse_option,
    set_blas_num_threads, timer_flush!, _now, _reset_regions!, attach_counters!,
    inertia_retry!, get_kkt, input_type, is_inertia, should_regularize_dual, get_iterator
import NLPModels: AbstractNLPModel, get_x0, obj
import Printf: @sprintf

export MadNLPSolver, MadNLPOptions, madnlp, AbstractPreset, ShiftedPreset, AugmentedPreset, CondensedPreset
# MadNLPCore also exports what MadCore exports.
for name in names(MadCore)
    (name === :MadCore || !Base.isexported(MadCore, name)) && continue
    @eval import MadCore: $name
    @eval export $name
end

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include(joinpath("IPM", "IPM.jl"))

# The public names of MadNLPCore besides its exports: the option values a user passes.
const PUBLIC_NAMES = (
    :AbstractBarrierUpdate, :AbstractAdaptiveUpdate, :MonotoneUpdate, :LOQOUpdate,
    :QualityFunctionUpdate, :AbstractInertiaCorrector, :InertiaAuto, :InertiaBased,
    :InertiaFree, :InertiaIgnore, :DualInitializeOptions, :DualInitializeLeastSquares,
    :DualInitializeSetZero, :preset_perturbation,
)
# Declared public, with the public names of MadCore it imports, so that `MadNLPCore.name`
# raises no warning in the REPL. The keyword `public` does not parse on Julia 1.10.
@static if VERSION >= v"1.11"
    eval(Expr(:public, filter(n -> !Base.isexported(@__MODULE__, n), [PUBLIC_NAMES..., MadCore.public_names()...])...))
end
include("precompile_models.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

end # module
