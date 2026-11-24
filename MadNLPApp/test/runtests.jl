using Test, MadNLPApp
using MadNLPApp: c_solve
using CNLPLib: CNLPProblem as CProblem, CNLPResult as CResult

# min s * sum_i (x_i - 1)^2  s.t.  x_1 + x_2 = 1 (n = 4), with s = 1 (min) or the
# maximization of -sum_i (x_i - 1)^2: the same solution x = (0.5, 0.5, 1, 1), and in the
# convention of madnlp.h the same y = 1 in both senses.
const SIGN = Ref(1.0)
qp_obj(ctx, x, f) = (v = unsafe_wrap(Array, x, 4); unsafe_store!(f, SIGN[] * sum(abs2, v .- 1)); Int64(0))
qp_grad(ctx, x, g) = (v = unsafe_wrap(Array, x, 4); unsafe_wrap(Array, g, 4) .= SIGN[] .* 2 .* (v .- 1); Int64(0))
qp_cons(ctx, x, c) = (v = unsafe_wrap(Array, x, 4); unsafe_store!(c, v[1] + v[2]); Int64(0))
qp_js(ctx, r, c) = (unsafe_wrap(Array, r, 2) .= 1; unsafe_wrap(Array, c, 2) .= (1, 2); Int64(0))
qp_jac(ctx, x, v) = (unsafe_wrap(Array, v, 2) .= 1.0; Int64(0))
qp_hs(ctx, r, c) = (unsafe_wrap(Array, r, 4) .= 1:4; unsafe_wrap(Array, c, 4) .= 1:4; Int64(0))
qp_hess(ctx, x, y, w, v) = (unsafe_wrap(Array, v, 4) .= SIGN[] * 2w; Int64(0))
qp_fail(ctx, x, f) = Int64(7)

const F = (
    obj = @cfunction(qp_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    grad = @cfunction(qp_grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    cons = @cfunction(qp_cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    js = @cfunction(qp_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    jac = @cfunction(qp_jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
    hs = @cfunction(qp_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
    hess = @cfunction(qp_hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64})),
    fail = @cfunction(qp_fail, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
)

function run(; minimize = true, opts = ("print_level" => "0",), obj = F.obj)
    SIGN[] = minimize ? 1.0 : -1.0
    x0, lv, uv = zeros(4), fill(-Inf, 4), fill(Inf, 4)
    lc, uc = [1.0], [1.0]
    x, y, zl, zu = zeros(4), zeros(1), zeros(4), zeros(4)
    GC.@preserve x0 lv uv lc uc x y zl zu begin
        p = Ref(CProblem(C_NULL, 4, 1, 2, 4, minimize ? 1 : 0, pointer(x0), pointer(lv), pointer(uv),
                         pointer(lc), pointer(uc), C_NULL, C_NULL, C_NULL,
                         obj, F.grad, F.cons, F.js, F.jac, F.hs, F.hess))
        r = Ref(CResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
        ks = [String(k) for (k, _) in opts]; vs = [String(v) for (_, v) in opts]
        GC.@preserve ks vs begin
            kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
            rc = GC.@preserve kp vp c_solve(Base.unsafe_convert(Ptr{CProblem}, p), Int64(length(ks)),
                                             pointer(kp), pointer(vp), Base.unsafe_convert(Ptr{CResult}, r))
        end
        msg = String(filter(!iszero, collect(r[].solver_status)))
        return (; rc, x = copy(x), y = copy(y), zl = copy(zl), zu = copy(zu), r = r[], msg)
    end
end

@testset "MadNLPApp C entry point" begin
    @testset "minimize" begin
        s = run()
        @test s.rc == 0 && s.r.status == 0
        @test s.x ≈ [0.5, 0.5, 1, 1] atol = 1e-7
        @test s.y ≈ [1.0] atol = 1e-7
        @test s.r.objective ≈ 0.5 atol = 1e-8
        @test s.r.iterations >= 1
        @test startswith(s.msg, "MadNLP: SOLVE_SUCCEEDED")
    end
    @testset "maximize: same x, same y in the convention" begin
        s = run(minimize = false)
        @test s.rc == 0 && s.r.status == 0
        @test s.x ≈ [0.5, 0.5, 1, 1] atol = 1e-7
        @test s.y ≈ [1.0] atol = 1e-7
        @test s.r.objective ≈ -0.5 atol = 1e-8
    end
    @testset "options" begin
        s = run(opts = ("print_level" => "ERROR", "tol" => "1e-10", "max_iter" => "50", "sb" => "yes"))
        @test s.rc == 0 && s.r.status == 0
        s = run(opts = ("print_level" => "0", "max_iter" => "0"))
        @test s.rc == 0 && s.r.status == 4            # iteration limit
        s = run(opts = ("print_level" => "NOSUCHLEVEL",))
        @test s.rc == 1 && occursin("invalid option", s.msg)
    end
    @testset "a failing callback is reported, not thrown" begin
        s = run(obj = F.fail)
        @test s.rc == 2 && s.r.status == 6
        @test occursin("obj callback returned 7", s.msg)
    end
end

@testset "quasi-Newton (Jacobian products)" begin
    s = run(opts = ("print_level" => "0", "hessian_approximation" => "CompactLBFGS"))
    @test s.rc == 0 && s.r.status == 0
    @test s.x ≈ [0.5, 0.5, 1, 1] atol = 1e-6
end
