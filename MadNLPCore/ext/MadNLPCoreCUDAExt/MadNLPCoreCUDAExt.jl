module MadNLPCoreCUDAExt

# cuDSS's options on the augmented system under each option preset of MadNLPCore, for models on
# NVIDIA GPUs (on the condensed system they follow the matrix: MadCore's CUDSS extension).

import MadCore
import MadNLPCore
using CUDACore: CuVector

# cuDSS under ShiftedPreset: LDLᵀ with small pivots bounded and with matching,
# without which the solve of the indefinite augmented matrix has a backward error near 1e-5.
# Matching depends on the values, so the analysis waits for the first matrix.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadNLPCore.SparseKKTSystem},
        ::Type{MadCore.CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(MadCore.CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = Float64(MadNLPCore.preset_perturbation(T))
    opt.cudss_matching = true
    opt.cudss_deferred_analysis = true
    return opt
end

# cuDSS on the augmented system under AugmentedPreset: LDLᵀ with Bunch-Kaufman pivoting and
# matching, redone when iterative refinement rejects a solve (the matching can go stale as the
# values change). Refactorization would keep the first factorization's pivot sequence, so it is
# off and every factorization re-pivots on the current values.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.AugmentedPreset}, ::Type{MadNLPCore.SparseKKTSystem},
        ::Type{MadCore.CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(MadCore.CUDSSSolver)
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

end # module
