#=
    LinearSolver's interface
=#
"""
    AbstractLinearSolver

Abstract type for linear solver targeting
the resolution of the linear system ``Ax=b``.
"""
abstract type AbstractLinearSolver{T} end
abstract type AbstractLapackSolver{T, Alg} <: AbstractLinearSolver{T} end

"""
    introduce(::AbstractLinearSolver)

Print the name of the linear solver.
"""
function introduce end

"""
    factorize!(::AbstractLinearSolver)

Factorize the matrix ``A`` and updates the factors
inside the `AbstractLinearSolver` instance.
"""
function factorize! end

"""
    solve_linear_system!(::AbstractLinearSolver, x::AbstractVector)

Solve the linear system ``Ax = b``.

This function assumes the linear system has been
factorized previously with [`factorize!`](@ref).
"""
function solve_linear_system! end

"""
    is_supported(solver,T)

Return `true` if `solver` supports the floating point
number type `T`.

# Examples
```julia-repl
julia> is_supported(UmfpackSolver,Float64)
true

julia> is_supported(LDLSolver,Float32)
true
```
"""
function is_supported(::Type{LS},::Type{T}) where {LS <: AbstractLinearSolver, T <: AbstractFloat}
    return false
end

"""
    is_async(::AbstractLinearSolver)

Return `true` if the linear solver runs asynchronously
i.e., the `factorize!` and `solve_linear_system!` functions return before
the factorization and solve operations are completed.
"""
function is_async(::AbstractLinearSolver)
    return false
end

"""
    is_factorized(::AbstractLinearSolver)

Return `true` if the last call to [`factorize!`](@ref) succeeded. The default
returns `true`; solvers that can detect a failed factorization overload it.
MadNLPCore does not call it; MadIPM uses it to detect a failed factorization.
"""
is_factorized(::AbstractLinearSolver) = true

"""
    is_inertia(::AbstractLinearSolver)

Return `true` if the linear solver supports the
computation of the inertia of the linear system.
"""
function is_inertia end

"""
    inertia(::AbstractLinearSolver)

Return the inertia `(num_pos, num_zero, num_neg)` of the linear system
as a tuple.

### Note
The inertia is defined as a tuple ``(num_pos, num_zero, num_neg)``,
with
- ``num_pos``: number of positive eigenvalues
- ``num_zero``: number of zero eigenvalues
- ``num_neg``: number of negative eigenvalues
"""
function inertia end


function improve! end

"""
    set_block_structure!(opt, n_primal::Integer, n_dual::Integer) -> opt

Declare to a linear solver's option object the saddle-point partition
`(n_primal, n_dual)` of the matrix it is about to factorize (see
[`block_structure`](@ref)). The KKT system calls this *before* constructing the
solver, so the partition is available to the symbolic analysis and not only to
the numeric phase.

A solver that can exploit the partition declares mutable `n_primal` and/or
`n_dual` option fields, which are filled in here; for every other solver this is
a no-op, so no existing option type has to change.
"""
function set_block_structure!(opt, n_primal::Integer, n_dual::Integer)
    ismutable(opt) || return opt
    hasproperty(opt, :n_primal) && setproperty!(opt, :n_primal, Int(n_primal))
    hasproperty(opt, :n_dual)   && setproperty!(opt, :n_dual,   Int(n_dual))
    return opt
end

# Default function for AbstractKKTVector
function solve_linear_system!(s::AbstractLinearSolver, x::AbstractKKTVector)
    solve_linear_system!(s, full(x))
end

function solve_linear_system!(s::AbstractLinearSolver{T}, X::AbstractMatrix) where T
    n, nrhs = size(X)
    x = zeros(T, n)
    for i in 1:nrhs
        copyto!(x, 1, X, (i-1)*n + 1, n)
        solve_linear_system!(s, x)
        copyto!(X, (i-1)*n + 1, x, 1, n)
    end
end

"""
    LinearSolverBarrier{T} <: AbstractLinearSolver{T}

Wraps a linear solver behind an abstractly typed field, so a KKT system that stores it
has the same type for every linear solver. The interior-point code, which is specialized
on the KKT system type, is then compiled once and shared by all linear solvers; only the
calls forwarded below dispatch dynamically, a few times per iteration. They forward
through `Base.invokelatest`, so linear solver methods added by packages loaded later do
not invalidate the compiled interior-point code. Fields of the
wrapped solver are reachable as fields of the barrier; the solver itself is `wrapped`.
"""
struct LinearSolverBarrier{T} <: AbstractLinearSolver{T}
    wrapped::AbstractLinearSolver{T}
end
LinearSolverBarrier(s::LinearSolverBarrier) = s

Base.getproperty(s::LinearSolverBarrier, name::Symbol) =
    name === :wrapped ? getfield(s, :wrapped) : getproperty(getfield(s, :wrapped), name)
Base.hasproperty(s::LinearSolverBarrier, name::Symbol) =
    name === :wrapped || hasproperty(getfield(s, :wrapped), name)

introduce(s::LinearSolverBarrier) = Base.invokelatest(introduce, s.wrapped)
factorize!(s::LinearSolverBarrier) = (Base.invokelatest(factorize!, s.wrapped); s)
solve_linear_system!(s::LinearSolverBarrier, x::AbstractVector) =
    Base.invokelatest(solve_linear_system!, s.wrapped, x)
solve_linear_system!(s::LinearSolverBarrier, x::AbstractKKTVector) =
    Base.invokelatest(solve_linear_system!, s.wrapped, x)
solve_linear_system!(s::LinearSolverBarrier, X::AbstractMatrix) =
    Base.invokelatest(solve_linear_system!, s.wrapped, X)
is_inertia(s::LinearSolverBarrier) = Base.invokelatest(is_inertia, s.wrapped)::Bool
inertia(s::LinearSolverBarrier) = Base.invokelatest(inertia, s.wrapped)
improve!(s::LinearSolverBarrier) = Base.invokelatest(improve!, s.wrapped)::Bool
is_async(s::LinearSolverBarrier) = Base.invokelatest(is_async, s.wrapped)::Bool

#=
    Iterator's interface
=#
abstract type AbstractIterator{T} end

"""
    solve_refine!(x::VT, ::AbstractIterator, b::VT, w::VT) where {VT <: AbstractKKTVector}

Solve the linear system ``Ax = b`` using iterative
refinement. The object `AbstractIterator` stores an instance
of a [`AbstractLinearSolver`](@ref) for the backsolve
operations.

### Notes
This function assumes the matrix stored in the linear solver
has been factorized previously.

"""
function solve_refine! end

# LinearSolverExceptions
struct SymbolicException <: Exception end
struct FactorizationException <: Exception end
struct SolveException <: Exception end
struct InertiaException <: Exception end
LinearSolverException=Union{SymbolicException,FactorizationException,SolveException,InertiaException}

@enum(
    LinearFactorization::Int,
    BUNCHKAUFMAN = 1,
    LU = 2,
    QR = 3,
    CHOLESKY = 4,
    LDL = 5,
    EVD = 6,
)

# iterative solvers
include("backsolve.jl")

#=
    DEFAULT SOLVERS
=#

# dense solvers
include("lapack_common.jl")
include("lapack.jl")

# sparse solvers, implemented in package extensions
include("markers.jl")
