module MadCoreHSLExt

import MadCore
import MadCore:
    @kwdef,
    MadNLPLogger,
    @debug,
    @warn,
    @error,
    AbstractOptions,
    AbstractLinearSolver,
    set_options!,
    SparseMatrixCSC,
    SubVector,
    SymbolicException,
    FactorizationException,
    SolveException,
    InertiaException,
    introduce,
    factorize!,
    solve_linear_system!,
    improve!,
    is_inertia,
    inertia,
    findIJ,
    nnz,
    get_tril_to_full,
    transfer!,
    input_type,
    madnlp_unsafe_wrap,
    is_supported,
    default_options

import HSL
import HSL:
    Mc68Control,
    Mc68Info,
    Ma77Control,
    Ma77Info,
    Ma86Control,
    Ma86Info,
    Ma97Control,
    Ma97Info

import LinearAlgebra

import MadCore: HSLOrdering, HSLScaling, AMD, METIS, SCALING_NONE, MC64, MC77, MC30

include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

MadCore.solver_type(::Type{MadCore.Ma27Solver}) = Ma27State
MadCore.solver_type(::Type{MadCore.Ma57Solver}) = Ma57State
MadCore.solver_type(::Type{MadCore.Ma77Solver}) = Ma77State
MadCore.solver_type(::Type{MadCore.Ma86Solver}) = Ma86State
MadCore.solver_type(::Type{MadCore.Ma97Solver}) = Ma97State

end # module
