# Option presets for models on NVIDIA GPUs: the linear solvers of the GPU option presets
# (MadNLPCoreGPUExt), and cuDSS's options on the augmented system under ShiftedPreset.

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = CUDSSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:CuVector{T}}) where {T} = LapackCUDASolver

const GPU_PIVOT_EPSILON = 1.0e-10

# cuDSS on the augmented KKT system under ShiftedPreset: LDLᵀ with small pivots bounded and with
# matching, without which the solve of this indefinite matrix has a backward error near 1e-5.
# Matching depends on the values, so the analysis waits for the first matrix.
function MadCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, VT}, ::Type{MadNLPCore.ShiftedPreset}, ::Type{MadNLPCore.SparseKKTSystem}, ::Type{CUDSSSolver},
    ) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(CUDSSSolver)
    opt.cudss_algorithm = MadCore.LDL
    opt.cudss_pivot_epsilon = GPU_PIVOT_EPSILON
    opt.cudss_matching = true
    opt.cudss_deferred_analysis = true
    return opt
end
