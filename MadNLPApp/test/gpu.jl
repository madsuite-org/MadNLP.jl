# GPU check of libmadnlp's device entry point, run by hand on a machine with CUDA (CI has none):
#
#   julia --project=MadNLPApp/build MadNLPApp/test/gpu.jl LIB MODEL ARG...
#
# LIB has a CUDA tier. The model is solved through cnlp_solve_device (MadNLP on the GPU).
using Test, MadNLPApp, CNLPLib, CUDACore, cuMadNLP

lib, model, args = ARGS[1], ARGS[2], ARGS[3:end]
@testset "cnlp_solve_device: $model $(join(args, ' '))" begin
    @test cuMadNLP in MadNLPApp.OPTION_MODULES
    @test MadNLPApp.solve_lib_device(lib, model, args) == CNLPLib.CNLP_SOLVED
    @test MadNLPApp.solve_toy_device(false) == CNLPLib.CNLP_SOLVED
end
