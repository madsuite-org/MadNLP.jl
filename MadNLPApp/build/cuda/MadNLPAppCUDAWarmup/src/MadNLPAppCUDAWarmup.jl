"""
    MadNLPAppCUDAWarmup

The device warm-up of the CUDA build of libmadnlp: its precompile workload runs MadNLP's
standard GPU precompile routine and solves small synthetic problems (with and without a
fixed variable) through `cnlp_solve_device`, so that the image carries the compiled device
path. No model library is needed. It is a package of
its own, depending on MadNLPApp, CNLPLib, CUDACore and cuMadNLP, because its dependencies'
extensions (MadNLPAppCUDAExt, CNLPLib's CUDA extension) are loaded while it precompiles,
which is not the case inside MadNLPAppCUDAExt's own precompile. It then compiles those
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
        if CUDACore.functional()
            ext = Base.get_extension(MadNLPApp, :MadNLPAppCUDAExt)
            ext === nothing && error("MadNLPAppCUDAExt is not loaded")
            Base.get_extension(CNLPLib, :CNLPLibCUDAExt) === nothing && error("CNLPLib's CUDA extension is not loaded")
            # cuMadNLP's own precompile workload, again here, where everything the CUDA build
            # loads is loaded: loading those packages invalidates much of what cuMadNLP
            # compiled, and a solve recompiled it at run time
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
