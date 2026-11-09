# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on Apple GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = SDSSolver
function MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T}
    return error("mtlMadNLP has no dense linear solver; use a model with sparse callbacks")
end
