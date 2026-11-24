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

# A 3-variable problem (one variable fixed when `fixed`) given as host and device callbacks,
# the way cnlp_solve_device receives a model:
# min (x1-1)^2 + (x2-1)^2 + x3^2  s.t.  x1^2 + x2 + x3 = 1
const N3 = 3
_h(x) = unsafe_wrap(Array, x, N3)
_d(p::Ptr{Float64}, n) = unsafe_wrap(CUDACore.CuArray, CUDACore.CuPtr{Float64}(UInt(p)), n)
fval(v) = (v[1] - 1)^2 + (v[2] - 1)^2 + v[3]^2
f3(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) = (unsafe_store!(f, fval(_h(x))); Int64(0))
g3(c::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64}) = (v = _h(x); unsafe_wrap(Array, g, 3) .= (2(v[1] - 1), 2(v[2] - 1), 2v[3]); Int64(0))
c3(c::Ptr{Cvoid}, x::Ptr{Float64}, o::Ptr{Float64}) = (v = _h(x); unsafe_store!(o, v[1]^2 + v[2] + v[3]); Int64(0))
js3(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (unsafe_wrap(Array, r, 3) .= 1; unsafe_wrap(Array, k, 3) .= (1, 2, 3); Int64(0))
j3(c::Ptr{Cvoid}, x::Ptr{Float64}, o::Ptr{Float64}) = (v = _h(x); unsafe_wrap(Array, o, 3) .= (2v[1], 1.0, 1.0); Int64(0))
hs3(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (unsafe_wrap(Array, r, 3) .= (1, 2, 3); unsafe_wrap(Array, k, 3) .= (1, 2, 3); Int64(0))
h3(c::Ptr{Cvoid}, x::Ptr{Float64}, y::Ptr{Float64}, w::Float64, o::Ptr{Float64}) = (unsafe_wrap(Array, o, 3) .= (2w + 2unsafe_load(y), 2w, 2w); Int64(0))
# device versions: copy to the host, evaluate, copy back
f3d(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) = (unsafe_store!(f, fval(Array(_d(x, 3)))); Int64(0))
g3d(c::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64}) = (v = Array(_d(x, 3)); copyto!(_d(g, 3), [2(v[1] - 1), 2(v[2] - 1), 2v[3]]); Int64(0))
c3d(c::Ptr{Cvoid}, x::Ptr{Float64}, o::Ptr{Float64}) = (v = Array(_d(x, 3)); copyto!(_d(o, 1), [v[1]^2 + v[2] + v[3]]); Int64(0))
j3d(c::Ptr{Cvoid}, x::Ptr{Float64}, o::Ptr{Float64}) = (v = Array(_d(x, 3)); copyto!(_d(o, 3), [2v[1], 1.0, 1.0]); Int64(0))
h3d(c::Ptr{Cvoid}, x::Ptr{Float64}, y::Ptr{Float64}, w::Float64, o::Ptr{Float64}) = (yy = Array(_d(y, 1)); copyto!(_d(o, 3), [2w + 2yy[1], 2w, 2w]); Int64(0))

const KEEP = Any[]
function toy_problem(fixed::Bool)
    x0 = [0.5, 0.5, 0.5]; lv = fixed ? [-2.0, -2.0, 0.5] : fill(-2.0, 3); uv = fixed ? [2.0, 2.0, 0.5] : fill(2.0, 3)
    lc = [1.0]; uc = [1.0]
    push!(KEEP, (x0, lv, uv, lc, uc))
    return CNLPLib.CNLPProblem(
        C_NULL, 3, 1, 3, 3, 1, pointer(x0), pointer(lv), pointer(uv), pointer(lc), pointer(uc),
        C_NULL, C_NULL, C_NULL,
        @cfunction(f3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(g3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(c3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(js3, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(j3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(hs3, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(h3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}))
    )
end
toy_device_problem() = CNLPLib.CNLPDeviceProblem(
    1,
    @cfunction(f3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(g3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(c3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(j3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(h3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}))
)
toy_device_model(fixed::Bool) = CNLPLib.ProblemModel(toy_problem(fixed), toy_device_problem())

# Solve the toy through cnlp_solve_device, as a C caller does.
function solve_toy_device(ext, fixed::Bool)
    p = Ref(toy_problem(fixed)); d = Ref(toy_device_problem())
    x, y, zl, zu = zeros(3), zeros(1), zeros(3), zeros(3)
    r = Ref(CNLPLib.CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    ks = ["print_level"]; vs = ["0"]; kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
    GC.@preserve p d x y zl zu r ks vs kp vp begin
        ext.cnlp_solve_device(
            Base.unsafe_convert(Ptr{CNLPLib.CNLPProblem}, p),
            Base.unsafe_convert(Ptr{CNLPLib.CNLPDeviceProblem}, d), Int64(1), pointer(kp), pointer(vp),
            Base.unsafe_convert(Ptr{CNLPLib.CNLPResult}, r)
        )
    end
    return r[].status
end

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
            solve_toy_device(ext, false)
            solve_toy_device(ext, true)
            # the kernels just compiled, for the other GPUs the library runs on as well
            # (compute capabilities; this cannot be done while juliac writes the image)
            compile_for(VersionNumber.(split(get(ENV, "MADNLPAPP_CUDA_CAPS", "7.0,7.5,8.0,8.6,8.9,9.0"), ',')))
        end
    end
end

end # module
