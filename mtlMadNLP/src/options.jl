# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on Apple GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = SDSSolver
# Metal has no dense factorizations: LapackCPUSolver factorizes the KKT matrix on the host,
# with the LAPACK that libblastrampoline forwards to (Accelerate after `using AppleAccelerate`).
MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = MadCore.LapackCPUSolver
