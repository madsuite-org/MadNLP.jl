# Option presets for models whose data live on a GPU. The GPU entry points choose the linear
# solvers through `default_sparse_solver` and `default_dense_solver`, and the options of a sparse
# solver through `default_options(nlp, preset, kkt_system, linear_solver)`.
# The preset sets the rest; on a GPU it is `ShiftedPreset` by default.

MadNLPCore.default_preset(::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}) where {T} = MadNLPCore.ShiftedPreset

# The sparse solver of a shifted preset is the GPU entry point's default, not the bundle's
# CPU choice (CHOLMOD in MadNLP).
MadNLPCore.shifted_sparse_solver(nlp::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}) where {T} =
    MadNLPCore.default_sparse_solver(nlp)

# SparseDirectSolver.jl (AMD, Intel and Apple GPUs) under ShiftedPreset: LDLᵀ with diagonal pivoting
# inside each front, on the fused pipeline where SparseDirectSolver has it (cuDSS pivots on the diagonal under this preset too), after
# matching with scaling, with pivots below the bound perturbed to it, as cuDSS's pivot epsilon and
# CHOLMOD's dbound do, and two steps of iterative refinement in the solver. Without matching the inertia
# of some KKT matrices comes out wrong. The matching depends on the values, so the analysis waits for
# the first matrix; later factorizations keep it, and the refinement recovers the accuracy lost as it
# goes stale.
function MadNLPCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}, ::Type{MadNLPCore.ShiftedPreset},
        ::Type{MadNLPCore.SparseKKTSystem}, ::Type{SDSSolver},
    ) where {T}
    opt = MadNLPCore.default_options(SDSSolver)
    opt.sds_factorization = :ldl
    opt.sds_pivoting = :diagonal
    opt.sds_matching = "algo6"
    opt.sds_deferred_analysis = true
    opt.sds_pivot_epsilon = Float64(MadNLPCore.preset_perturbation(T))
    opt.sds_ir = 2
    return opt
end

# SparseDirectSolver.jl on the condensed system, under any preset: Cholesky of the positive definite
# matrix, without pivoting or matching, on the pipeline SparseDirectSolver chooses (the fused one on
# CUDA, AMDGPU and oneAPI), as cuDSS. A failed factorization reports a wrong inertia, on which the
# solver retries with a larger equality shift.
function MadNLPCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}, ::Type{MadNLPCore.SparseCondensedKKTSystem}, ::Type{SDSSolver},
    ) where {T}
    opt = MadNLPCore.default_options(SDSSolver)
    opt.sds_factorization = :cholesky
    opt.sds_deferred_analysis = true
    return opt
end

# SparseDirectSolver.jl on the augmented system under AugmentedPreset: LDLᵀ with Bunch-Kaufman pivoting
# and matching, as cuDSS under the same preset, with the analysis redone when MadNLP's refinement
# rejects a solve (as `cudss_reanalyze_on_improve`).
function MadNLPCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}, ::Type{MadNLPCore.AugmentedPreset},
        ::Type{MadNLPCore.SparseKKTSystem}, ::Type{SDSSolver},
    ) where {T}
    opt = MadNLPCore.default_options(SDSSolver)
    opt.sds_factorization = :ldl
    opt.sds_pivoting = :bk
    opt.sds_matching = "algo6"
    opt.sds_deferred_analysis = true
    opt.sds_reanalyze_on_improve = true
    return opt
end
