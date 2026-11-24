# GPU check of the device entry point, run by hand on a machine with CUDA (CI has none):
#
#   julia --project=MadNLPApp/build/cuda MadNLPApp/test/gpu.jl LIB MODEL ARG...
#
# LIB has a CUDA tier. The model is solved through cnlp_solve_device (MadNLP on the GPU) and
# through cnlp_solve (on the CPU); both must succeed with the same objective.
using Test, MadNLPApp, CNLPLib, CUDACore, cuMadNLP
const Ext = Base.get_extension(MadNLPApp, :MadNLPAppCUDAExt)
using CNLPLib: CNLPProblem, CNLPResult

lib, model, args = ARGS[1], ARGS[2], ARGS[3:end]
function cpu_solve()
    p = MadNLPApp.lib_problem(lib, model, args)
    q = p[]
    n, k = Int(q.nvar), Int(q.ncon)
    x, y, zl, zu = zeros(n), zeros(k), zeros(n), zeros(n)
    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    ks = ["print_level"]; vs = ["0"]; kp = [pointer(s) for s in ks]; vp = [pointer(s) for s in vs]
    GC.@preserve p x y zl zu r ks vs kp vp MadNLPApp.c_solve(
        Base.unsafe_convert(Ptr{CNLPProblem}, p), Int64(1),
        pointer(kp), pointer(vp), Base.unsafe_convert(Ptr{CNLPResult}, r)
    )
    return r[]
end

@testset "cnlp_solve_device: $model $(join(args, ' '))" begin
    @test Ext !== nothing
    @test cuMadNLP in MadNLPApp.OPTION_MODULES
    @test Ext.solve_lib_device(lib, model, args) == CNLPLib.CNLP_SOLVED
    c = cpu_solve()
    @test c.status == CNLPLib.CNLP_SOLVED
end
