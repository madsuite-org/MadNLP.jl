# `linear_solver` is intentionally left broad: any type is accepted, not only subtypes
# of `AbstractLinearSolver`. A package that keeps
# MadCore as a weak dependency cannot subtype them in its main module, so it exports a
# plain type and implements these methods in its extension, without `setglobal!`.
"""
    create_linear_solver(linear_solver::Type, A; opt = default_options(linear_solver), logger = MadNLPLogger())

Build the linear solver selected by `linear_solver` for the matrix `A`, a sparse or
dense matrix as given by `input_type`. Every linear solver implements it,
together with `default_options`, `input_type` and `is_supported`, which are called
with the type. A package that keeps MadCore as a weak dependency can export a plain
`struct FooSolver end` and define these methods in its extension.

A marker such as [`MumpsSolver`](@ref) is an empty type that gets these methods from
its package extension; until the extension is loaded, the `MethodError` names the
package to load.
"""
function create_linear_solver end

"""
    backend_package(marker::Type)

Return the name of the package that loads the extension implementing `marker`.
"""
function backend_package end

# A marker whose extension is not loaded has no methods; this hint names the package
# to load in the resulting `MethodError`.
function _missing_extension_hint(io, exc, argtypes, kwargs)
    exc.f in (create_linear_solver, default_options, input_type, is_supported) || return
    isempty(exc.args) && return
    M = exc.args[1]
    (M isa Type && hasmethod(backend_package, Tuple{Type{M}})) || return
    print(io, "\n$(nameof(M)) is implemented in a package extension that is not loaded. ")
    print(io, "Load $(backend_package(M)) to use $(nameof(M)).")
end

"""
    MumpsSolver

Select the sparse symmetric indefinite solver MUMPS. Implemented in the extension
loaded by `MUMPS_seq_jll` and `OpenBLAS32_jll`.
"""
struct MumpsSolver end
backend_package(::Type{MumpsSolver}) = "MUMPS_seq_jll and OpenBLAS32_jll"

"""
    LDLSolver

Select the sparse LDLᵀ factorization of LDLFactorizations.jl. Implemented in the
extension loaded by `LDLFactorizations`.
"""
struct LDLSolver end
backend_package(::Type{LDLSolver}) = "LDLFactorizations"

"""
    UmfpackSolver

Select the sparse LU factorization of UMFPACK. Implemented in the extension loaded
by `SuiteSparse`.
"""
struct UmfpackSolver end
backend_package(::Type{UmfpackSolver}) = "SuiteSparse"

"""
    CHOLMODSolver

Select the sparse Cholesky or LDLᵀ factorization of CHOLMOD. Implemented in the
extension loaded by `SuiteSparse`.
"""
struct CHOLMODSolver end
backend_package(::Type{CHOLMODSolver}) = "SuiteSparse"

"""
    DummySolver

Default sparse linear solver of MadCore and MadNLPCore, which ship without one.
Options and solver setup accept it, but building it throws an error asking for the
option `linear_solver`. MadNLP replaces it with [`MumpsSolver`](@ref).
"""
struct DummySolver{T} <: AbstractLinearSolver{T} end

@kwdef mutable struct DummySolverOptions <: AbstractOptions end

function DummySolver(args...; kwargs...)
    error(
        "No sparse linear solver is selected. Pass the option `linear_solver` " *
        "(for example `linear_solver = LDLSolver` with LDLFactorizations loaded), " *
        "or load MadNLP, which uses MUMPS by default.",
    )
end
default_options(::Type{DummySolver}) = DummySolverOptions()
input_type(::Type{DummySolver}) = :csc
is_supported(::Type{DummySolver}, ::Type{T}) where {T <: AbstractFloat} = true
create_linear_solver(::Type{DummySolver}, A; kwargs...) = DummySolver(A; kwargs...)

"""
    KrylovIterator

Select GMRES iterative refinement of the KKT system, preconditioned by the
factorized KKT system, with the option `iterator = KrylovIterator`. The extension
loaded by `Krylov` makes `KrylovIterator(kkt; opt, cnt, logger)` build the iterator
and defines its `default_options`.
"""
struct KrylovIterator end
backend_package(::Type{KrylovIterator}) = "Krylov"

# HSL solvers, implemented in the extension loaded by `HSL`.

@enum(HSLOrdering::Int, AMD = 1, METIS = 3,)
@enum(HSLScaling::Int, SCALING_NONE = 0, MC64 = 1, MC77 = 2, MC30 = 4,)

for (name, number) in ((:Ma27Solver, 27), (:Ma57Solver, 57), (:Ma77Solver, 77), (:Ma86Solver, 86), (:Ma97Solver, 97))
    doc = """
        $name

    Select the sparse symmetric indefinite solver MA$number of the HSL library.
    Implemented in the extension loaded by `HSL`.
    """
    @eval begin
        @doc $doc struct $name end
        backend_package(::Type{$name}) = "HSL"
    end
end

"""
    PardisoMKLSolver

Select the sparse symmetric indefinite solver Pardiso of Intel MKL. Implemented in
the extension loaded by `Pardiso`.
"""
struct PardisoMKLSolver end
backend_package(::Type{PardisoMKLSolver}) = "Pardiso"

"""
    PanuaPardisoSolver

Select the sparse symmetric solver of Panua Pardiso, which needs a license.
Implemented in the extension loaded by `Pardiso`. It was called `PardisoSolver`
in MadNLPPardiso; that name is now exported by Pardiso.jl.
"""
struct PanuaPardisoSolver end
backend_package(::Type{PanuaPardisoSolver}) = "Pardiso"
