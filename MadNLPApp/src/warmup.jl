# Build-time warm-up (a PrecompileTools workload): solve problems through the C entry
# point the way `cnlp solve` calls it, and through the `.nl` main, so that the compiled
# solve path is in the image and a run compiles nothing.
#
#   MADNLPAPP_WARMUP_RUNS  a file of lines `LIB MODEL ARGS...`: a cnlp 0.2 library, one of
#                          its models and its instance arguments (integers or a file path);
#                          each is solved once through cnlp_solve. Models that reach
#                          different solver branches (restoration, iteration limit, ...)
#                          compile those branches.
#   MADNLPAPP_WARMUP_NL    .nl files (separated by spaces), each solved through `madnlp`.
#
# Without either, the workload still solves the small problems below.

using Libdl: Libdl
using LinearAlgebra: BLAS
using PrecompileTools: @setup_workload, @compile_workload

# A model instance of a cnlp library: its id and evaluation functions, P_obj(id, ...) etc.
mutable struct LibInstance
    id::Int64
    f::NTuple{7, Ptr{Cvoid}}   # obj grad cons jac_structure jac hess_structure hess
end
_li(c) = unsafe_pointer_to_objref(c)::LibInstance
li_obj(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) = (m = _li(c); ccall(m.f[1], Int64, (Int64, Ptr{Float64}, Ptr{Float64}), m.id, x, f))
li_grad(c::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64}) = (m = _li(c); ccall(m.f[2], Int64, (Int64, Ptr{Float64}, Ptr{Float64}), m.id, x, g))
li_cons(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) = (m = _li(c); ccall(m.f[3], Int64, (Int64, Ptr{Float64}, Ptr{Float64}), m.id, x, v))
li_js(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (m = _li(c); ccall(m.f[4], Int64, (Int64, Ptr{Int64}, Ptr{Int64}), m.id, r, k))
li_jac(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) = (m = _li(c); ccall(m.f[5], Int64, (Int64, Ptr{Float64}, Ptr{Float64}), m.id, x, v))
li_hs(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (m = _li(c); ccall(m.f[6], Int64, (Int64, Ptr{Int64}, Ptr{Int64}), m.id, r, k))
li_hess(c::Ptr{Cvoid}, x::Ptr{Float64}, y::Ptr{Float64}, w::Float64, v::Ptr{Float64}) =
    (m = _li(c); ccall(m.f[7], Int64, (Int64, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64}), m.id, x, y, w, v))

function _copyout(f::Ptr{Cvoid})
    buf = zeros(UInt8, 4096)
    n = ccall(f, Int64, (Ptr{UInt8}, Int64), buf, length(buf) - 1)
    return String(buf[1:clamp(n, 0, length(buf) - 1)])
end

# Instantiate MODEL of the library at PATH from command-line style arguments: one integer
# (P_new), a file (P_new_str), or several scalars through the builder (P_schema names).
function lib_instance(path::String, model::String, args::Vector{String})
    # A library bundle's runtime points libblastrampoline at its own BLAS when it starts
    # (on the first call); point it back at this process's afterwards (as CNLPLib does), or
    # the solver has no BLAS.
    blas = [l.libname for l in BLAS.get_config().loaded_libs]
    h = Libdl.dlopen(path, Libdl.RTLD_LOCAL | Libdl.RTLD_NOW)
    s(n) = Libdl.dlsym(h, Symbol(model, "_", n))
    types = [first(split(t, '|')) for t in split(_copyout(s(:argtype)), ',') if !isempty(t)]
    id = if isempty(types)
        ccall(s(:new), Int64, (Int64,), 0)
    elseif length(types) == 1 && types[1] == "int"
        ccall(s(:new), Int64, (Int64,), parse(Int64, args[1]))
    elseif length(types) == 1 && types[1] == "string"
        ccall(s(:new_str), Int64, (Cstring,), args[1])
    else
        fields = [m.captures[1] for m in eachmatch(r"\"name\"\s*:\s*\"([^\"]*)\"", _copyout(s(:schema)))]
        b = ccall(s(:data_begin), Int64, ())
        for (t, f, a) in zip(types, fields, args)
            t == "int" ? ccall(s(:set_scalar_i64), Int64, (Int64, Cstring, Int64), b, f, parse(Int64, a)) :
                         ccall(s(:set_scalar_f64), Int64, (Int64, Cstring, Float64), b, f, parse(Float64, a))
        end
        ccall(s(:data_ready), Int64, (Int64,), b) == 1 || error("$model: incomplete instance data")
        ccall(s(:new_from_data), Int64, (Int64,), b)
    end
    id > 0 || error("$model: could not instantiate from $args")
    li = LibInstance(id, (s(:obj), s(:grad), s(:cons), s(:jac_structure), s(:jac), s(:hess_structure), s(:hess)))
    q(n) = ccall(s(n), Int64, (Int64,), id)
    nv, nc = q(:nvar), q(:ncon)
    x0, lv, uv, lc, uc = zeros(nv), zeros(nv), zeros(nv), zeros(nc), zeros(nc)
    ccall(s(:meta), Int64, (Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          id, x0, lv, uv, lc, uc)
    mini = Libdl.dlsym(h, Symbol(model, "_minimize"); throw_error = false)
    minimize = mini === nothing ? 1 : (ccall(mini, Int64, (Int64,), id) != 0 ? 1 : 0)
    for (i, l) in enumerate(blas)
        BLAS.lbt_forward(l; clear = i == 1)
    end
    return li, (nv, nc, q(:nnzj), q(:nnzh), minimize), (x0, lv, uv, lc, uc)
end

# Solve through c_solve, as a C caller does; returns the solver-neutral status.
function solve_via_c(ctx, dims, arrays, fs, opts::Vector{Pair{String, String}})
    nv, nc, nj, nh, minimize = dims
    x0, lv, uv, lc, uc = arrays
    x, y, zl, zu = zeros(nv), zeros(nc), zeros(nv), zeros(nv)
    ks = [k for (k, _) in opts]; vs = [v for (_, v) in opts]
    GC.@preserve ctx x0 lv uv lc uc x y zl zu ks vs begin
        c = ctx isa Ptr ? ctx : pointer_from_objref(ctx)
        p = Ref(CProblem(c, nv, nc, nj, nh, minimize, pointer(x0), pointer(lv), pointer(uv),
                         pointer(lc), pointer(uc), C_NULL, C_NULL, C_NULL, fs...))
        r = Ref(CResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
        kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
        GC.@preserve p r kp vp c_solve(Base.unsafe_convert(Ptr{CProblem}, p), Int64(length(ks)),
                                       pointer(kp), pointer(vp), Base.unsafe_convert(Ptr{CResult}, r))
        return r[].status
    end
end

lib_callbacks() = (
    @cfunction(li_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(li_grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(li_cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(li_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    @cfunction(li_jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(li_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    @cfunction(li_hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64})),
)

# A small problem given in Julia: min s * sum (x - t)^2 s.t. x1^2 + x2 = b, -1 <= x <= 2,
# with b out of reach for `infeasible` (restoration, then infeasibility detection).
mutable struct Toy
    s::Float64
    b::Float64
end
_t(c) = unsafe_pointer_to_objref(c)::Toy
toy_obj(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) =
    (t = _t(c); unsafe_store!(f, t.s * ((unsafe_load(x, 1) - 1)^2 + (unsafe_load(x, 2) - 1)^2)); Int64(0))
function toy_grad(c::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64})
    t = _t(c)
    unsafe_store!(g, 2t.s * (unsafe_load(x, 1) - 1), 1); unsafe_store!(g, 2t.s * (unsafe_load(x, 2) - 1), 2)
    return Int64(0)
end
toy_cons(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) = (unsafe_store!(v, unsafe_load(x, 1)^2 + unsafe_load(x, 2)); Int64(0))
toy_js(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) =
    (unsafe_store!(r, 1, 1); unsafe_store!(r, 1, 2); unsafe_store!(k, 1, 1); unsafe_store!(k, 2, 2); Int64(0))
toy_jac(c::Ptr{Cvoid}, x::Ptr{Float64}, v::Ptr{Float64}) = (unsafe_store!(v, 2unsafe_load(x, 1), 1); unsafe_store!(v, 1.0, 2); Int64(0))
toy_hs(c::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) =
    (unsafe_store!(r, 1, 1); unsafe_store!(r, 2, 2); unsafe_store!(k, 1, 1); unsafe_store!(k, 2, 2); Int64(0))
function toy_hess(c::Ptr{Cvoid}, x::Ptr{Float64}, y::Ptr{Float64}, w::Float64, v::Ptr{Float64})
    t = _t(c)
    unsafe_store!(v, 2w * t.s + 2unsafe_load(y, 1), 1); unsafe_store!(v, 2w * t.s, 2)
    return Int64(0)
end
toy_callbacks() = (
    @cfunction(toy_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(toy_grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(toy_cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(toy_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    @cfunction(toy_jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    @cfunction(toy_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    @cfunction(toy_hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64})),
)
solve_toy(; s = 1.0, b = 1.0, opts = Pair{String, String}[]) =
    solve_via_c(Toy(s, b), (Int64(2), Int64(1), Int64(2), Int64(2), Int64(s > 0 ? 1 : 0)),
                ([0.5, 0.5], [-1.0, -1.0], [2.0, 2.0], [b], [b]), toy_callbacks(),
                ["print_level" => "0"; opts])

@setup_workload begin
    runs = get(ENV, "MADNLPAPP_WARMUP_RUNS", "")
    nls = split(get(ENV, "MADNLPAPP_WARMUP_NL", ""))
    @compile_workload begin
        solve_toy()                                             # minimize
        solve_toy(s = -1.0)                                     # maximize
        solve_toy(b = 10.0)                                     # infeasible: restoration
        solve_toy(opts = ["max_iter" => "1"])                   # iteration limit
        solve_toy(opts = ["tol" => "1e-10", "print_level" => "ERROR", "sb" => "yes"])
        if !isempty(runs)
            for line in eachline(runs)
                w = split(line)
                (isempty(w) || startswith(w[1], "#")) && continue
                li, dims, arrays = lib_instance(String(w[1]), String(w[2]), String.(w[3:end]))
                solve_via_c(li, dims, arrays, lib_callbacks(), ["print_level" => "0"])
            end
        end
        for nl in nls
            dir = mktempdir()
            stub = joinpath(dir, first(splitext(basename(nl))))
            cp(nl, stub * ".nl")
            redirect_stdout(devnull) do
                main([stub, "-AMPL", "print_level=ERROR"])
            end
        end
    end
end
