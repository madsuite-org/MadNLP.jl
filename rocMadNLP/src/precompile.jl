# Solve small problems on the GPU during precompilation, so the solver's GPU kernels are
# compiled and cached in this package image. Needs a functional GPU at
# precompile time; the cached code objects target that GPU's architecture.
PrecompileTools.@setup_workload begin
    if AMDGPU.functional()
        # A failure here (device busy, out of memory, driver problem) must not make the
        # package unloadable: skip the workload and let GPU code compile on first use.
        try
            PrecompileTools.@compile_workload begin
                # Build models in here, not in the setup block: constructing them compiles
                # GPU kernels, and only code compiled inside `@compile_workload` is kept.
                wrapped = MadNLP.SparseWrapperModel(ROCVector, MadNLP.HS15Model())
                MadNLP.madnlp(wrapped; print_level = MadNLP.ERROR)
                # The default linear solver on AMD GPUs factorizes the condensed KKT system
                # densely, so the large size stays moderate (above one thread block).
                MadNLP._precompile_solves(ROCVector{Float64}(undef, 0); sizes = (100, 4096))
            end
        catch err
            @warn "rocMadNLP: the AMDGPU precompile workload failed; GPU solves compile on first use" exception = (err, catch_backtrace())
        end
    end
end
