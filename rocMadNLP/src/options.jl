# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on AMD GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = SDSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:ROCVector{T}}) where {T} = LapackROCmSolver
