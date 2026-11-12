# Option presets for models on NVIDIA GPUs: the linear solvers of the GPU option presets
# (MadNLPCoreGPUExt), and cuDSS's options under each preset.

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = CUDSSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = LapackCUDASolver

# cuDSS under ShiftedPreset and CondensedPreset: LDLᵀ with small pivots bounded and with matching,
# without which the solve of the indefinite augmented matrix has a backward error near 1e-5.
# Matching depends on the values, so the analysis waits for the first matrix.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{<:MadNLPCore.PivotBoundedPreset}, ::Type{<:MadNLPCore.SparsePresetKKTSystem},
        ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = Float64(MadNLPCore.preset_perturbation(T))
    opt.cudss_matching = true
    opt.cudss_deferred_analysis = true
    return opt
end

# cuDSS on the condensed system under CondensedPreset: LDLᵀ without a pivot bound and without
# matching (so without the deferred analysis matching needs). With the matrix equilibrated before condensation (AugmentedRuizScaler), matching makes
# the factorization useless as a preconditioner: on case24464_goc_polar one factorized solve leaves
# a residual 3e5 times the right-hand side, GMRES runs to its cap and the solver refactorizes with a
# larger perturbation at nearly every iteration; without matching the solve is as accurate as with
# no scaling (residual 1, 4 backsolves per iteration against 31). The pivot bound has the same effect
# (residual 5e6).
function MadCore.default_options(
        nlp::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.CondensedPreset}, ::Type{MadNLPCore.SparseCondensedKKTSystem},
        ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = 0.0
    opt.cudss_matching = false
    return opt
end

# cuDSS under EqCondensedPreset: Cholesky of the positive definite condensed matrix, without
# pivoting or matching.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.EqCondensedPreset}, ::Type{MadCore.SparseCondensedKKTSystem},
        ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.CHOLESKY
    opt.cudss_pivot_epsilon = 0.0
    opt.cudss_pivoting = false
    opt.cudss_matching = false
    opt.cudss_deferred_analysis = true
    # conservative inertia: a pivot whose relative rounding error (eps A_ii / d_i) exceeds 10% counts
    # as incorrect inertia; benchmark job 179 (GPU, 1579 of 1582 instances): 1429 solved against 1423
    # without it, ACOPF 128 and CUTEst 1206 (0.03 also gave ACOPF 128 but lost CUTEst instances)
    opt.cudss_pivot_rel_error = 0.1
    return opt
end

# cuDSS on the augmented system under AugmentedPreset: LDLᵀ with Bunch-Kaufman pivoting and
# matching, redone when iterative refinement rejects a solve (the matching can go stale as the
# values change). Refactorization would keep the first factorization's pivot sequence, so it is
# off and every factorization re-pivots on the current values.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.AugmentedPreset}, ::Type{MadNLPCore.SparseKKTSystem},
        ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivoting = true
    opt.cudss_pivot_type = 'B'
    opt.cudss_matching = true
    opt.cudss_matching_alg = "algo6"
    opt.cudss_deferred_analysis = true
    opt.cudss_refactorization = false
    opt.cudss_reanalyze_on_improve = true
    return opt
end
