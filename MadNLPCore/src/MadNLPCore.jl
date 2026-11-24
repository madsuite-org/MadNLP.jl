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
include("precompile_models.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

end # module
