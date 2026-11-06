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
    _madnlp_unsafe_wrap,
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

MadCore.create_linear_solver(::Type{MadCore.Ma27Solver}, A; kwargs...) = Ma27State(A; kwargs...)
MadCore.default_options(::Type{MadCore.Ma27Solver}) = MadCore.default_options(Ma27State)
MadCore.input_type(::Type{MadCore.Ma27Solver}) = MadCore.input_type(Ma27State)
MadCore.is_supported(::Type{MadCore.Ma27Solver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(Ma27State, T)
MadCore.create_linear_solver(::Type{MadCore.Ma57Solver}, A; kwargs...) = Ma57State(A; kwargs...)
MadCore.default_options(::Type{MadCore.Ma57Solver}) = MadCore.default_options(Ma57State)
MadCore.input_type(::Type{MadCore.Ma57Solver}) = MadCore.input_type(Ma57State)
MadCore.is_supported(::Type{MadCore.Ma57Solver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(Ma57State, T)
MadCore.create_linear_solver(::Type{MadCore.Ma77Solver}, A; kwargs...) = Ma77State(A; kwargs...)
MadCore.default_options(::Type{MadCore.Ma77Solver}) = MadCore.default_options(Ma77State)
MadCore.input_type(::Type{MadCore.Ma77Solver}) = MadCore.input_type(Ma77State)
MadCore.is_supported(::Type{MadCore.Ma77Solver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(Ma77State, T)
MadCore.create_linear_solver(::Type{MadCore.Ma86Solver}, A; kwargs...) = Ma86State(A; kwargs...)
MadCore.default_options(::Type{MadCore.Ma86Solver}) = MadCore.default_options(Ma86State)
MadCore.input_type(::Type{MadCore.Ma86Solver}) = MadCore.input_type(Ma86State)
MadCore.is_supported(::Type{MadCore.Ma86Solver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(Ma86State, T)
MadCore.create_linear_solver(::Type{MadCore.Ma97Solver}, A; kwargs...) = Ma97State(A; kwargs...)
MadCore.default_options(::Type{MadCore.Ma97Solver}) = MadCore.default_options(Ma97State)
MadCore.input_type(::Type{MadCore.Ma97Solver}) = MadCore.input_type(Ma97State)
MadCore.is_supported(::Type{MadCore.Ma97Solver}, ::Type{T}) where {T <: AbstractFloat} = MadCore.is_supported(Ma97State, T)

end # module
