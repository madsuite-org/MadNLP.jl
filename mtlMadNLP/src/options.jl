# Linear solvers of the GPU option presets (MadNLPCoreGPUExt) for models on Apple GPUs

MadCore.default_sparse_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = SDSSolver
# SDSSolver does not yet solve the augmented system with the scaling, shift and adaptive barrier
# of the default preset, so sparse models on Apple GPUs keep the condensed system by default.
MadCore.default_sparse_kkt_system(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T} = MadCore.SparseCondensedKKTSystem
function MadCore.default_dense_solver(::MadCore.AbstractNLPModel{T, <:MtlVector{T}}) where {T}
    return error("mtlMadNLP has no dense linear solver; use a model with sparse callbacks")
end
