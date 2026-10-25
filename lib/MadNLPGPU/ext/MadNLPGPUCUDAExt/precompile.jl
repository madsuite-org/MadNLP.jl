# Solve small problems on the GPU during precompilation, so the solver's GPU kernels
# (KKT assembly, cuDSS calls, vector broadcasts and reductions) are compiled and cached
# in this extension's package image. Needs a functional GPU at precompile time; the
# cached cubins target that GPU's architecture.
PrecompileTools.@setup_workload begin
    if CUDACore.functional()
        __init__()
        PrecompileTools.@compile_workload begin
            # Build models in here, not in the setup block: constructing them compiles
            # GPU kernels (the bounds analysis runs `findall`), and only code compiled
            # inside `@compile_workload` is kept in the package image.
            wrapped = MadNLP.SparseWrapperModel(CuVector, MadNLP.HS15Model())
            MadNLP.madnlp(wrapped; print_level = MadNLP.ERROR)
            MadNLP._precompile_solves(CuVector{Float64}(undef, 0))
            MadNLP._precompile_linear_solver(CuVector{Float64}(undef, 0), LapackCUDASolver)
        end
    end
end
