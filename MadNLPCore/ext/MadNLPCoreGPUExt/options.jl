# Option presets for models whose data live on a GPU. The GPU entry points choose the linear
# solvers through `default_sparse_solver` and `default_dense_solver`, and the options of a sparse
# solver through `default_options(nlp, preset, kkt_system, linear_solver)`.
# The preset (`ShiftedPreset` by default) sets the rest, as on the host.

function MadNLPCore.MadNLPOptions{T}(
        nlp::MadNLPCore.AbstractNLPModel{T, VT};
        preset::Type{<:MadNLPCore.AbstractPreset} = MadNLPCore.default_preset(nlp),
        dense_callback = MadNLPCore.is_dense_callback(nlp),
        callback = dense_callback ? MadNLPCore.DenseCallback : MadNLPCore.SparseCallback,
        kkt_system = dense_callback ? MadNLPCore.DenseCondensedKKTSystem : MadNLPCore.default_sparse_kkt_system(nlp),
        linear_solver = dense_callback ? MadNLPCore.default_dense_solver(nlp) : MadNLPCore.default_sparse_solver(nlp),
        tol = MadNLPCore.get_tolerance(T, kkt_system),
        bound_relax_factor = (kkt_system == MadNLPCore.SparseCondensedKKTSystem) ? tol : T(1.0e-8),
    ) where {T, VT <: AbstractGPUVector{T}}
    return MadNLPCore.MadNLPOptions{T}(;
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
        bound_relax_factor = bound_relax_factor,
        preset = preset,
        MadNLPCore.preset_options(preset, T, kkt_system, tol)...,
    )
end
