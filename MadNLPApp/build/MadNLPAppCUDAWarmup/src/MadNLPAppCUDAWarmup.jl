"""
    MadNLPAppCUDAWarmup

The device warm-up of libmadnlp: its precompile workload runs MadNLP's
standard GPU precompile routine and solves small synthetic problems (with and without a
fixed variable) through `cnlp_solve_device`, so that the image carries the compiled device
path. No model library is needed. It is a package of its own, depending on MadNLPApp,
CNLPLib, CUDACore and cuMadNLP, so that MadNLPApp's own precompilation needs no GPU (a
machine without one, CI, can build and test it); the library build loads it. It then compiles those
kernels for other GPU architectures as well (multiarch.jl), so that a GPU other than the
build machine's finds its kernels in the image.
"""
module MadNLPAppCUDAWarmup

import CNLPLib, CUDACore, cuMadNLP, MadNLPApp, MadNLPCore
using PrecompileTools: @setup_workload, @compile_workload
const GPUCompiler = CUDACore.GPUCompiler

include("multiarch.jl")

@setup_workload begin
    @compile_workload begin
        # the library ships this compiled device path: a build without a GPU would ship it
        # uncompiled, so refuse it (MADNLPAPP_ALLOW_NO_GPU=1 builds one anyway, for tests)
        CUDACore.functional() || get(ENV, "MADNLPAPP_ALLOW_NO_GPU", "0") == "1" ||
            error("building libmadnlp needs a working CUDA GPU (the device warm-up solves on it)")
        if CUDACore.functional()
            ext = MadNLPApp
            Base.get_extension(CNLPLib, :CNLPLibCUDAExt) === nothing && error("CNLPLib's CUDA extension is not loaded")
            # cuMadNLP's own precompile workload, again here, where everything the build loads
            # is loaded, so that nothing loaded after it can have invalidated it
            like = CUDACore.CuVector{Float64}(undef, 0)
            MadNLPCore._precompile_solves(like)
            MadNLPCore._precompile_linear_solver(like, cuMadNLP.LapackCUDASolver)
            # the device entry point itself (ProblemModel{CuVector}, cnlp_solve_device), with
            # and without a fixed variable: the two solver types a default solve can reach
            ext.solve_toy_device(false)
            ext.solve_toy_device(true)
            # the kernels just compiled, for the other GPUs the library runs on as well
            # (compute capabilities; this cannot be done while juliac writes the image)
            compile_for(VersionNumber.(split(get(ENV, "MADNLPAPP_CUDA_CAPS", "7.0,7.5,8.0,8.6,8.9,9.0"), ',')))
        end
    end
end

end # module
