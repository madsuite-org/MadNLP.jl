# The device entry point, `cnlp_solve_device`: a problem evaluated through its device (CUDA)
# callbacks, solved with cuMadNLP with the solver's vectors on the GPU. The GPU's context and
# the cuBLAS/cuSPARSE/cuDSS handles are made by the first device solve; the CUDA packages'
# own `__init__` still finds the driver when the library loads, on a machine that has one.

function c_solve_device(
        p::Ptr{CNLPProblem}, d::Ptr{CNLPDeviceProblem}, nopt::Int64, keys::Ptr{Ptr{UInt8}},
        vals::Ptr{Ptr{UInt8}}, result::Ptr{CNLPResult}
    )
    CNLPLib.tracing(TRACE) || return c_solve(() -> ProblemModel(p, d), nopt, keys, vals, result; device = true)
    # MADNLPAPP_TRACE=1: evaluate the model once on the device before solving, and report how
    # long its first evaluations took: loading the model library's GPU code, and the driver
    # compiling its PTX when the driver's cache (~/.nv/ComputeCache) does not hold it yet
    m, t_first, t_again = try
        m = ProblemModel(p, d)
        m, first_evaluations(m), first_evaluations(m)
    catch
        # building or evaluating the model failed: let the solve report it (rc 2)
        return c_solve(() -> ProblemModel(p, d), nopt, keys, vals, result; device = true)
    end
    println(
        stderr, string(
            "madnlp: model library, first GPU evaluations ", round(Int, 1000t_first), " ms (then ",
            round(Int, 1000t_again), " ms): loading its GPU code, and compiling it in the driver if not cached"
        )
    )
    solve() = c_solve(() -> m, nopt, keys, vals, result; device = true)
    # MADNLPAPP_TRACE=1: also report the GPU kernels compiled at run time
    empty!(COMPILED)
    rc = CUDACore.GPUCompiler.with_compile_hook(solve, kernel_hook)
    println(
        stderr, string(
            "madnlp: GPU kernels compiled at run time: ", length(COMPILED),
            isempty(COMPILED) ? "" : " (" * join(COMPILED, ", ") * ")"
        )
    )
    flush(stderr)
    return rc
end

function first_evaluations(m)
    x = copy(m.meta.x0); y = CUDACore.zeros(Float64, m.meta.ncon)
    g = similar(x); c = similar(y)
    jv = CUDACore.zeros(Float64, m.meta.nnzj); hv = CUDACore.zeros(Float64, m.meta.nnzh)
    return @elapsed begin
        NLPModels.obj(m, x); NLPModels.grad!(m, x, g); NLPModels.cons!(m, x, c)
        NLPModels.jac_coord!(m, x, jv); NLPModels.hess_coord!(m, x, y, hv)
        CUDACore.synchronize()
    end
end

# The warm-up problem: 3 variables (one fixed when `fixed`, the other solver type a default
# solve reaches), given as host and device callbacks, the way cnlp_solve_device receives a
# model. Used at build time (build/MadNLPAppCUDAWarmup).
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

# the toy's bounds and starting point; the caller keeps them alive while the problem is used
toy_arrays(fixed::Bool) = ([0.5, 0.5, 0.5], fixed ? [-2.0, -2.0, 0.5] : fill(-2.0, 3), fixed ? [2.0, 2.0, 0.5] : fill(2.0, 3), [1.0], [1.0])
function toy_problem((x0, lv, uv, lc, uc))
    return CNLPProblem(
        C_NULL, 3, 1, 3, 3, 1, pointer(x0), pointer(lv), pointer(uv), pointer(lc), pointer(uc),
        C_NULL, C_NULL, C_NULL,
        @cfunction(f3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(g3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
        @cfunction(c3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(js3, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(j3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(hs3, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
        @cfunction(h3, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}))
    )
end
toy_device_problem() = CNLPDeviceProblem(
    1,
    @cfunction(f3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(g3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(c3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})), @cfunction(j3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(h3d, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}))
)

# Solve the toy through cnlp_solve_device, as a C caller does.
function solve_toy_device(fixed::Bool)
    a = toy_arrays(fixed)
    p = Ref(toy_problem(a)); d = Ref(toy_device_problem())
    x, y, zl, zu = zeros(3), zeros(1), zeros(3), zeros(3)
    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    ks = ["print_level"]; vs = ["0"]; kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
    GC.@preserve a p d x y zl zu r ks vs kp vp begin
        cnlp_solve_device(
            Base.unsafe_convert(Ptr{CNLPProblem}, p),
            Base.unsafe_convert(Ptr{CNLPDeviceProblem}, d), Int64(1), pointer(kp), pointer(vp),
            Base.unsafe_convert(Ptr{CNLPResult}, r)
        )
    end
    return r[].status
end

# GPUCompiler reports every kernel lookup to the hook; a lookup with no compiled image
# cached is a kernel about to be compiled at run time.
const COMPILED = String[]
const COMPILED_LOCK = ReentrantLock()
function kernel_hook(job)
    res = CUDACore.GPUCompiler.cached_results(CUDACore.CUDACompilerResults, job)
    if res === nothing || res.image === nothing
        name = string(job.source.def.name)
        lock(COMPILED_LOCK) do
            name in COMPILED || push!(COMPILED, name)
        end
    end
    return nothing
end

# A library model solved through cnlp_solve_device: what the build runs to compile the
# device path (build/MadNLPAppCUDAWarmup), and a test helper.
function solve_lib_device(path::String, model::String, args::Vector{String})
    lib = CNLPLib.load(path)
    vals = Any[something(tryparse(Int, a), tryparse(Float64, a), a) for a in args]
    m = CNLPLib.CNLPModel(lib, Symbol(model), vals...)
    md = CNLPLib.CNLPModel(lib, Symbol(model), vals...; backend = CUDACore.CUDABackend())
    h = CNLPLib.problem(m)
    d = CNLPLib.device_problem(h, md)
    n, k = m.meta.nvar, m.meta.ncon
    x, y, zl, zu = zeros(n), zeros(k), zeros(n), zeros(n)
    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    ks = ["print_level"]; vs = ["0"]
    kp = [pointer(s) for s in ks]; vp = [pointer(s) for s in vs]
    GC.@preserve h d x y zl zu r ks vs kp vp begin
        cnlp_solve_device(
            Base.unsafe_convert(Ptr{CNLPProblem}, h), Base.unsafe_convert(Ptr{CNLPDeviceProblem}, d),
            Int64(1), pointer(kp), pointer(vp), Base.unsafe_convert(Ptr{CNLPResult}, r)
        )
    end
    return r[].status
end
