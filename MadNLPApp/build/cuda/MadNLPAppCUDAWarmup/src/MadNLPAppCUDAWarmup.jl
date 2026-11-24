"""
    MadNLPAppCUDAWarmup

The device warm-up of the CUDA build of libmadnlp: its precompile workload solves each
`LIB MODEL ARGS...` of MADNLPAPP_WARMUP_RUNS whose library has a CUDA tier through
`cnlp_solve_device`, so that the image carries the compiled device path. It is a package of
its own, depending on MadNLPApp, CNLPLib, CUDACore and cuMadNLP, because its dependencies'
extensions (MadNLPAppCUDAExt, CNLPLib's CUDA extension) are loaded while it precompiles,
which is not the case inside MadNLPAppCUDAExt's own precompile. It then compiles those
kernels for other GPU architectures as well (multiarch.jl), so that a GPU other than the
build machine's finds its kernels in the image.
"""
module MadNLPAppCUDAWarmup

import CNLPLib, CUDACore, cuMadNLP, MadNLPApp
using PrecompileTools: @setup_workload, @compile_workload
const GPUCompiler = CUDACore.GPUCompiler

include("multiarch.jl")

@setup_workload begin
    runs = get(ENV, "MADNLPAPP_WARMUP_RUNS", "")
    @compile_workload begin
        if !isempty(runs) && CUDACore.functional()
            ext = Base.get_extension(MadNLPApp, :MadNLPAppCUDAExt)
            ext === nothing && error("MadNLPAppCUDAExt is not loaded")
            Base.get_extension(CNLPLib, :CNLPLibCUDAExt) === nothing && error("CNLPLib's CUDA extension is not loaded")
            for line in eachline(runs)
                w = split(line)
                (isempty(w) || startswith(w[1], "#")) && continue
                l = CNLPLib.load(String(w[1]))
                CNLPLib.Libdl.dlsym(l.handle, Symbol(w[2], "_device"); throw_error = false) === nothing && continue
                ext.solve_lib_device(String(w[1]), String(w[2]), String.(w[3:end]))
            end
            # the kernels just compiled, for the other GPUs the library runs on as well
            # (compute capabilities; this cannot be done while juliac writes the image)
            compile_for(VersionNumber.(split(get(ENV, "MADNLPAPP_CUDA_CAPS", "7.0,7.5,8.0,8.6,8.9,9.0"), ',')))
        end
    end
end

end # module
