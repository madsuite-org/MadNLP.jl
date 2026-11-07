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
import LinearAlgebra: axpy!, dot, norm
import MadCore: @warn, AbstractDenseKKTSystem,
    AbstractIterator, AbstractUserCallback, BFGS, DampedBFGS, DenseCallback,
    DenseCondensedKKTSystem, DenseKKTSystem, LINESEARCH_SUCCEEDED, LapackCPUSolver,
    ModelBarrier, NoUserCallback, RESTORE, ROBUST, RichardsonIterator,
    SchurComplementKKTSystem, UserCallbackRegular, UserCallbackRestore, UserCallbackRobust,
    _get_primary_options, _set_aug_diagonal!, blas_num_threads, create_timer,
    dual_inf_perturbation!, get_average_complementarity, get_inf_barrier, get_inf_total,
    get_min_complementarity, get_sc, get_sd, get_tolerance, has_constraints,
    is_dense_callback, mul_hess_blk!, parse_option, print_init, reset_bound_dual!,
    set_aug_rhs!, set_blas_num_threads, set_initial_rhs!, timer_flush!
import NLPModels: AbstractNLPModel, get_x0, obj
import Printf: @sprintf
import SolverCore: solve!

export MadNLPSolver, MadNLPOptions, madnlp
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
