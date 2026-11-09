# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on AMD GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = SDSSolver
# SDSSolver does not yet solve the augmented system with the scaling, shift and adaptive barrier
# of the default preset, so sparse models on AMD GPUs keep the condensed system by default.
MadCore.default_sparse_kkt_system(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = MadCore.SparseCondensedKKTSystem
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = LapackROCmSolver
