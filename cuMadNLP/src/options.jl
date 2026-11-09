# Option presets for models on NVIDIA GPUs: the linear solvers of the GPU option presets
# (MadNLPCoreGPUExt), and cuDSS's options under ShiftedPreset and CondensedPreset.

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = CUDSSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = LapackCUDASolver

const GPU_PIVOT_EPSILON = 1.0e-10

# cuDSS under ShiftedPreset and CondensedPreset: LDLᵀ with small pivots bounded and with matching,
# without which the solve of the indefinite augmented matrix has a backward error near 1e-5.
# Matching depends on the values, so the analysis waits for the first matrix.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{<:MadNLPCore.PivotBoundedPreset}, ::Type{<:MadNLPCore.SparsePresetKKTSystem},
        ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = GPU_PIVOT_EPSILON
    opt.cudss_matching = true
    opt.cudss_deferred_analysis = true
    return opt
end
