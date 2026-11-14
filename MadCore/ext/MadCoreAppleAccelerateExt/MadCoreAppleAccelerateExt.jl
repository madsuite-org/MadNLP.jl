module MadCoreAppleAccelerateExt

# The sparse solvers of Accelerate exist on macOS only. AppleAccelerate also forwards BLAS and
# LAPACK to Accelerate (macOS 13.4+), so LapackCPUSolver runs on Accelerate too.
@static if Sys.isapple()

import MadCore
import AppleAccelerate as AA
using SparseArrays: SparseMatrixCSC
for name in MadCore.internal_names()
    @eval import MadCore: $name
end

"""
    AccelerateOptions

Options of [`AccelerateSolver`](@ref), with the enumerations of AppleAccelerate:

- `accelerate_factorization`: `SparseFactorizationLDLTTPP` (threshold partial pivoting, the
  only LDLᵀ with an inertia), `SparseFactorizationLDLTSBK`, `SparseFactorizationLDLTUnpivoted`
  or `SparseFactorizationCholesky`.
- `accelerate_order`: `SparseOrderDefault`, `SparseOrderAMD`, `SparseOrderMetis` or `SparseOrderCOLAMD`.
- `accelerate_scaling`: `SparseScalingUser` (no scaling, the fastest on PGLib),
  `SpraseScalingDefault` or `SparseScalingEquilibriationInf`.
- `accelerate_pivtol`, `accelerate_pivtolmax`: initial and largest relative pivot threshold.
- `accelerate_zero_tolerance`: pivots smaller in absolute value are zero.
"""
@kwdef mutable struct AccelerateOptions <: AbstractOptions
    accelerate_factorization::AA.SparseFactorization_t = AA.SparseFactorizationLDLTTPP
    accelerate_order::AA.SparseOrder_t = AA.SparseOrderDefault
    accelerate_scaling::AA.SparseScaling_t = AA.SparseScalingUser
    accelerate_pivtol::Float64 = 1e-4
    accelerate_pivtolmax::Float64 = 1e-1
    accelerate_zero_tolerance::Float64 = 1e-20
end

mutable struct AccelerateState{T, Ti} <: AbstractLinearSolver{T}
    tril::SparseMatrixCSC{T, Ti}
    colptr::Vector{Clong} # 0-based indices; the values are tril.nzval
    rowval::Vector{Cint}
    matrix::AA.SparseMatrix{T}
    fact::Base.RefValue{AA.SparseOpaqueFactorization{T}}
    nfoptions::AA.SparseNumericFactorOptions
    sfoptions::AA.SparseSymbolicFactorOptions
    workspace::Vector{UInt8}
    opt::AccelerateOptions
    logger::MadNLPLogger
end

# With SparseScalingUser, a NULL scaling vector means no scaling.
_numeric_options(T, opt, pivtol = opt.accelerate_pivtol) = AA.SparseNumericFactorOptions(
    AA.SparseNumericFactorOptions(T).control, opt.accelerate_scaling, C_NULL, pivtol, opt.accelerate_zero_tolerance,
)

# AppleAccelerate's error callback throws through the C++ frames of libSparse; the status
# of the factorization reports the failure instead.
_ignore_error(::Cstring) = nothing

function AccelerateState(tril::SparseMatrixCSC{T, Ti}; opt = AccelerateOptions(), logger = MadNLPLogger()) where {T, Ti}
    n = size(tril, 1)
    colptr, rowval = Clong.(tril.colptr .- 1), Cint.(tril.rowval .- 1)
    structure = AA.SparseMatrixStructure(
        Cint(n), Cint(n), pointer(colptr), pointer(rowval), AA.ATT_SYMMETRIC | AA.ATT_LOWER_TRIANGLE, UInt8(1),
    )
    d = AA.SparseSymbolicFactorOptions()
    sfoptions = AA.SparseSymbolicFactorOptions(
        d.control, opt.accelerate_order, C_NULL, C_NULL, d.malloc, d.free, @cfunction(_ignore_error, Cvoid, (Cstring,)),
    )
    M = AccelerateState{T, Ti}(
        tril, colptr, rowval, AA.SparseMatrix(structure, pointer(tril.nzval)), Ref(AA.SparseOpaqueFactorization(T)),
        _numeric_options(T, opt), sfoptions, UInt8[], opt, logger,
    )
    return finalizer(M) do M
        M.fact[].status in (AA.SparseYetToBeFactored, AA.SparseStatusReleased) || AA.SparseCleanup(M.fact[])
    end
end

for (T, refactor, getinertia) in (
        (Float64, :_SparseRefactorSymmetric_Double, :_Z16SparseGetInertia32SparseOpaqueFactorization_DoublePiS0_S0_),
        (Float32, :_SparseRefactorSymmetric_Float, :_Z16SparseGetInertia31SparseOpaqueFactorization_FloatPiS0_S0_),
    )
    @eval begin
        _refactor!(M::AccelerateState{$T}) = GC.@preserve M ccall(
            ($(QuoteNode(refactor)), AA.LIBSPARSE), Cvoid,
            (Ref{AA.SparseMatrix{$T}}, Ptr{AA.SparseOpaqueFactorization{$T}}, Ref{AA.SparseNumericFactorOptions}, Ptr{Cvoid}),
            M.matrix, M.fact, M.nfoptions, M.workspace,
        )
        function _get_inertia(M::AccelerateState{$T})
            pos, zero, neg = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
            ccall(
                ($(QuoteNode(getinertia)), AA.LIBSPARSE), Cint,
                (AA.SparseOpaqueFactorization{$T}, Ref{Cint}, Ref{Cint}, Ref{Cint}), M.fact[], pos, zero, neg,
            ) == 0 || return (0, Int(size(M.tril, 1)), 0)
            return (Int(pos[]), Int(zero[]), Int(neg[]))
        end
    end
end

# The first factorization is symbolic and numeric; the next ones, also after a failure,
# refactorize the values in place.
function factorize!(M::AccelerateState{T}) where {T}
    sym = M.fact[].symbolicFactorization
    if sym.status == AA.SparseStatusOk && M.fact[].numericFactorization != C_NULL
        _refactor!(M)
    else
        M.fact[] = GC.@preserve M AA.SparseFactor(M.opt.accelerate_factorization, M.matrix, M.sfoptions, M.nfoptions)
        sym = M.fact[].symbolicFactorization
        sym.status == AA.SparseStatusOk &&
            resize!(M.workspace, Int(T == Float64 ? sym.workspaceSize_Double : sym.workspaceSize_Float))
    end
    is_factorized(M) || @debug(M.logger, "Accelerate factorization failed: $(M.fact[].status)")
    return M
end

function solve_linear_system!(M::AccelerateState{T}, rhs::Vector{T}) where {T}
    # After a failure, the same rhs enters the primal-dual regularization.
    is_factorized(M) && AA.SparseSolve(M.fact[], rhs)
    return rhs
end

is_factorized(M::AccelerateState) = M.fact[].status == AA.SparseStatusOk
const _TPP = (AA.SparseFactorizationLDLT, AA.SparseFactorizationLDLTTPP) # LDLT is TPP
is_inertia(M::AccelerateState) = M.opt.accelerate_factorization in (AA.SparseFactorizationCholesky, _TPP...)
function inertia(M::AccelerateState)
    n = size(M.tril, 1)
    is_factorized(M) || return (0, n, 0)
    # A Cholesky factorization succeeds only on positive definite matrices.
    M.opt.accelerate_factorization == AA.SparseFactorizationCholesky && return (n, 0, 0)
    M.opt.accelerate_factorization in _TPP || throw(InertiaException())
    return _get_inertia(M)
end

function improve!(M::AccelerateState)
    pivtol = M.nfoptions.pivotTolerance
    pivtol == M.opt.accelerate_pivtolmax && return false
    pivtol = min(M.opt.accelerate_pivtolmax, pivtol^0.75)
    M.nfoptions = _numeric_options(eltype(M.tril), M.opt, pivtol)
    @debug(M.logger, "improved quality: pivtol = $pivtol")
    return true
end

introduce(M::AccelerateState) = "Apple Accelerate ($(M.opt.accelerate_factorization))"

MadCore.create_linear_solver(::Type{MadCore.AccelerateSolver}, A; kwargs...) = AccelerateState(A; kwargs...)
MadCore.default_options(::Type{MadCore.AccelerateSolver}) = AccelerateOptions()
MadCore.input_type(::Type{MadCore.AccelerateSolver}) = :csc
MadCore.is_supported(::Type{MadCore.AccelerateSolver}, ::Type{T}) where {T <: AbstractFloat} = T in (Float32, Float64)

end # @static if

end # module
