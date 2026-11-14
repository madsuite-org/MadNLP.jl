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
                wrapped = MadCore.SparseWrapperModel(ROCVector, MadNLPCore.HS15Model())
                MadNLPCore.madnlp(wrapped; print_level = MadNLPCore.ERROR)
                MadNLPCore._precompile_solves(ROCVector{Float64}(undef, 0); sizes = (100, 4096))
            end
        catch err
            @warn "rocMadNLP: the AMDGPU precompile workload failed; GPU solves compile on first use" exception = (err, catch_backtrace())
        end
    end
end
