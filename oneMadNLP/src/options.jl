# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on Intel GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:oneVector{T}}) where {T} = SDSSolver
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:oneVector{T}}) where {T} = LapackOneMKLSolver
