# Option presets for models whose data live on a GPU. The GPU entry points choose the linear
# solvers through `default_sparse_solver` and `default_dense_solver`, and the options of a sparse
# solver through `default_options(nlp, preset, kkt_system, linear_solver)`.
# The preset sets the rest; on a GPU it is `ShiftedPreset` by default.

MadNLPCore.default_preset(::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}) where {T} = MadNLPCore.ShiftedPreset

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        preset::Type{<:MadNLPCore.AbstractPreset} = MadNLPCore.default_preset(nlp),
        dense_row_fallback::Bool = true,
        kwargs...,
    ) where {T, VT <: AbstractGPUVector{T}}
    kkt_options = get(kwargs, :kkt_options, Dict{Symbol, Any}())
    preset = dense_row_fallback ? MadNLPCore.resolve_preset(nlp, preset, kkt_options) : preset
    return _gpu_preset_madnlp_options(T, nlp; preset, kwargs...)
end
function _gpu_preset_madnlp_options(
        ::Type{T}, nlp::MadNLPCore.AbstractNLPModel{T, VT};
        preset::Type{<:MadNLPCore.AbstractPreset},
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.preset_kkt_system(preset),
        linear_solver = dense_callback ? MadNLPCore.default_dense_solver(nlp) : MadNLPCore.default_sparse_solver(nlp),
        tol = MadNLPCore.get_tolerance(T, kkt_system),
    ) where {T, VT <: AbstractGPUVector{T}}
    return MadNLPCore.MadNLPOptions{T}(;
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        preset = preset,
        MadNLPCore.preset_options(preset, T, kkt_system, tol)...,
    )
end

# SparseDirectSolver.jl (AMD, Intel and Apple GPUs) under ShiftedPreset:
# a static LDLᵀ factorization in the order of the reordering, with pivots below the bound perturbed
# to it, as cuDSS's pivot epsilon and CHOLMOD's dbound do.
function MadNLPCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}, ::Type{<:MadNLPCore.PivotBoundedPreset},
        ::Type{<:MadNLPCore.SparsePresetKKTSystem}, ::Type{SDSSolver},
    ) where {T}
    opt = MadNLPCore.default_options(SDSSolver)
    opt.sds_factorization = :ldl
    opt.sds_pivoting = :none
    opt.sds_pivot_epsilon = Float64(MadNLPCore.preset_perturbation(T))
    return opt
end

# SparseDirectSolver.jl on the augmented system under AugmentedPreset: LDLᵀ with Bunch-Kaufman pivoting.
function MadNLPCore.default_options(
        ::MadNLPCore.AbstractNLPModel{T, <:AbstractGPUVector{T}}, ::Type{MadNLPCore.AugmentedPreset},
        ::Type{MadNLPCore.SparseKKTSystem}, ::Type{SDSSolver},
    ) where {T}
    opt = MadNLPCore.default_options(SDSSolver)
    opt.sds_factorization = :ldl
    opt.sds_pivoting = :bk
    return opt
end
