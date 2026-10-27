# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

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
include("ldl.jl")
include("mumps.jl")

# These solvers are only available if Julia was built with SuiteSparse,
# which is GPL.  If not, then the `using`s here will fail.
if Base.USE_GPL_LIBS
    include("umfpack.jl")
    include("cholmod.jl")
end
