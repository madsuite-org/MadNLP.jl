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

using PrecompileTools: @setup_workload, @compile_workload

# A model of a cnlp library, instantiated from command-line style arguments (integers,
# numbers or a file path), as a cnlp_problem (CNLPLib, which also keeps BLAS working
# after the library's runtime starts).
function lib_problem(path::String, model::String, args::Vector{String})
    lib = CNLPLib.load(path)
    vals = Any[something(tryparse(Int, a), tryparse(Float64, a), a) for a in args]
    return CNLPLib.problem(CNLPLib.CNLPModel(lib, Symbol(model), vals...))
end

# Solve a cnlp_problem through c_solve, as a C caller does; returns the solver-neutral status.
function solve_via_c(p, opts::Vector{Pair{String, String}})
    q = p[]
    nv, nc = Int(q.nvar), Int(q.ncon)
    x, y, zl, zu = zeros(nv), zeros(nc), zeros(nv), zeros(nv)
    ks = [k for (k, _) in opts]; vs = [v for (_, v) in opts]
    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
    kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
    GC.@preserve p x y zl zu ks vs r kp vp begin
        c_solve(
            Base.unsafe_convert(Ptr{CNLPProblem}, p), Int64(length(ks)), pointer(kp), pointer(vp),
            Base.unsafe_convert(Ptr{CNLPResult}, r)
        )
    end
    return r[].status
end

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
function solve_toy(; s = 1.0, b = 1.0, opts = Pair{String, String}[])
    t = Toy(s, b)
    x0, lv, uv, lc, uc = [0.5, 0.5], [-1.0, -1.0], [2.0, 2.0], [b], [b]
    GC.@preserve t x0 lv uv lc uc begin
        p = Ref(
            CNLPProblem(
                pointer_from_objref(t), 2, 1, 2, 2, s > 0 ? 1 : 0, pointer(x0), pointer(lv),
                pointer(uv), pointer(lc), pointer(uc), C_NULL, C_NULL, C_NULL, toy_callbacks()...
            )
        )
        return solve_via_c(p, ["print_level" => "0"; opts])
    end
end

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
                p = lib_problem(String(w[1]), String(w[2]), String.(w[3:end]))
                solve_via_c(p, ["print_level" => "0"])
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
