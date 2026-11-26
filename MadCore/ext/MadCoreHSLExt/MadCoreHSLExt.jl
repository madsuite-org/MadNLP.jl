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

import MadCore: HSLOrdering, HSLScaling, AMD, METIS, SCALING_NONE, MC64, MC77, MC30, set_permutation!

include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

# The HSL solvers read the pivot order as the position of each variable, `invperm(perm)`.
set_permutation!(opt::Ma27Options, perm) = (opt.ma27_user_perm = perm; opt)
set_permutation!(opt::Ma57Options, perm) = (opt.ma57_user_perm = perm; opt)
set_permutation!(opt::Ma77Options, perm) = (opt.ma77_user_perm = perm; opt)
set_permutation!(opt::Ma86Options, perm) = (opt.ma86_user_perm = perm; opt)
set_permutation!(opt::Ma97Options, perm) = (opt.ma97_user_perm = perm; opt)

# The pivot order of MA77 and MA86 (position of each variable): MC68's ordering `ordering`,
# or the given `perm`.
function hsl_pivot_order!(order::Vector{INT}, ::Nothing, ordering, csc, control, info) where {INT}
    control.f_array_in = 1
    control.f_array_out = 1
    HSL.mc68_order(INT, INT(ordering), INT(csc.n), csc.colptr, csc.rowval, order, control, info)
    return order
end
hsl_pivot_order!(order, perm::Vector{Int}, ordering, csc, control, info) = copyto!(order, invperm(perm))

MadCore.solver_type(::Type{MadCore.Ma27Solver}) = Ma27State
MadCore.solver_type(::Type{MadCore.Ma57Solver}) = Ma57State
MadCore.solver_type(::Type{MadCore.Ma77Solver}) = Ma77State
MadCore.solver_type(::Type{MadCore.Ma86Solver}) = Ma86State
MadCore.solver_type(::Type{MadCore.Ma97Solver}) = Ma97State

end # module
