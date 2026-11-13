testset = [
    [
        "SparseKKTSystem + Mumps",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.MumpsSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + Umfpack",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.UmfpackSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseKKTSystem + RelaxBound",
        ()->MadNLP.Optimizer(
            fixed_variable_treatment=MadNLP.RelaxBound,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "ScaledSparseKKTSystem + LapackCPU",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLP.LapackCPUSolver,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-LU",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-QR",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "DenseKKTSystem + LapackCPU-EVD",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.EVD,
            print_level=MadNLP.ERROR),
        [
            "eigmina" # fails; regularization does not correct the inertia; inertia calculation based on EVD does not seem reliable
         ]
    ],
    [
        "DenseKKTSystem + LapackCPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.DenseKKTSystem,
            linear_solver=MadNLP.LapackCPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR),
        ["infeasible", "lootsma", "eigmina", "lp_examodels_issue75"]
    ],
    [
        "SparseUnreducedKKTSystem",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            linear_solver=UmfpackSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseUnreducedKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            linear_solver=UmfpackSolver,
            kkt_system=MadNLP.SparseUnreducedKKTSystem,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseCondensedKKTSystem + CHOLMOD-CHOLESKY",
        ()->MadNLP.Optimizer(
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            linear_solver=MadNLP.CHOLMODSolver,
            print_level=MadNLP.ERROR),
        []
    ],
    [
        "SparseCondensedKKTSystem + InertiaFree",
        ()->MadNLP.Optimizer(
            inertia_correction_method=MadNLP.InertiaFree,
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            print_level=MadNLP.ERROR),
        []
    ],
]

# N.B. Current CHOLMOD interface is supported only starting from Julia v1.10.
if VERSION >= v"1.10"
    push!(
        testset,
        [
            "SparseCondensedKKTSystem + CHOLMOD-LDL",
            ()->MadNLP.Optimizer(
                kkt_system=MadNLP.SparseCondensedKKTSystem,
                equality_treatment = MadNLP.RelaxEquality,
                fixed_variable_treatment = MadNLP.RelaxBound,
                linear_solver=MadNLP.CHOLMODSolver,
                cholmod_algorithm=MadNLP.LDL,
                print_level=MadNLP.ERROR),
            []
        ]
    )
end

for (name,optimizer_constructor,exclude) in testset
    test_madnlp(name,optimizer_constructor,exclude)
end

# min x1 + x2  s.t.  x1 + 2 x2 = 3,  0 <= x <= 10.
# The equality multiplier is nonzero, so a relaxed equality ends at its relaxed bound.
struct ActiveEqualityLP <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
end
ActiveEqualityLP() = ActiveEqualityLP(
    NLPModels.NLPModelMeta(
        2; ncon = 1, nnzj = 2, nnzh = 0,
        x0 = [1.0, 1.0], lvar = [0.0, 0.0], uvar = [10.0, 10.0],
        lcon = [3.0], ucon = [3.0],
    ),
    NLPModels.Counters(),
)
NLPModels.obj(::ActiveEqualityLP, x::AbstractVector) = x[1] + x[2]
NLPModels.grad!(::ActiveEqualityLP, x::AbstractVector, g::AbstractVector) = (g .= 1.0; g)
NLPModels.cons!(::ActiveEqualityLP, x::AbstractVector, c::AbstractVector) = (c[1] = x[1] + 2x[2]; c)
NLPModels.jac_structure!(::ActiveEqualityLP, I::AbstractVector, J::AbstractVector) = (I .= 1; J .= [1, 2]; (I, J))
NLPModels.jac_coord!(::ActiveEqualityLP, x::AbstractVector, J::AbstractVector) = (J .= [1.0, 2.0]; J)
NLPModels.hess_structure!(::ActiveEqualityLP, I::AbstractVector, J::AbstractVector) = (I, J)
NLPModels.hess_coord!(::ActiveEqualityLP, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0) = H

@testset "RelaxEquality relaxes equalities once" begin
    brf = 1e-4
    width = brf * 3.0   # bound_relax_factor * max(1, |b|)
    for kkt in (MadNLP.SparseKKTSystem, MadNLP.SparseCondensedKKTSystem)
        res = madnlp(
            ActiveEqualityLP();
            kkt_system = kkt,
            equality_treatment = MadNLP.RelaxEquality,
            fixed_variable_treatment = MadNLP.RelaxBound,
            bound_relax_factor = brf,
            tol = 1e-10,
            print_level = MadNLP.ERROR,
        )
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        viol = abs(res.constraints[1] - 3.0)
        @test viol > 0.5 * width
        @test viol <= width * (1 + 1e-4)
    end
end

@testset "HS15 problem" begin
    nlp = MadNLPTests.HS15Model()
    n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
    solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "ModelBarrier" begin
    nlp = MadNLPTests.HS15Model()
    r1 = madnlp(nlp; print_level = MadNLP.ERROR)
    r2 = madnlp(nlp; print_level = MadNLP.ERROR, specialize_model = true)
    @test r1.status == r2.status == MadNLP.SOLVE_SUCCEEDED
    @test r1.iter == r2.iter
    @test r1.objective ≈ r2.objective
    @test r1.solution ≈ r2.solution
    @test r1.multipliers ≈ r2.multipliers
    # the solver type does not depend on the model type (two sparse models of different types)
    a = MadNLP.SparseWrapperModel(Array, nlp)
    b = MadNLP.SparseWrapperModel(Array, a)
    @test typeof(a) != typeof(b)
    @test typeof(MadNLPSolver(MadNLP.ModelBarrier(a))) ==
          typeof(MadNLPSolver(MadNLP.ModelBarrier(b)))
    @test MadNLP.ModelBarrier(MadNLP.ModelBarrier(nlp)) isa MadNLP.ModelBarrier{Float64, Vector{Float64}}
end

@testset "LinearSolverBarrier" begin
    nlp = MadNLP.SparseWrapperModel(Array, MadNLPTests.HS15Model())
    a = MadNLPSolver(nlp; print_level = MadNLP.ERROR, linear_solver = MumpsSolver)
    b = MadNLPSolver(nlp; print_level = MadNLP.ERROR, linear_solver = LDLSolver)
    # the KKT system type does not depend on the linear solver
    @test typeof(a.kkt) == typeof(b.kkt)
    @test a.kkt.linear_solver isa MadNLP.LinearSolverBarrier{Float64}
    @test startswith(MadNLP.introduce(a.kkt.linear_solver.wrapped), "MUMPS")
    # fields of the wrapped solver are forwarded, including one named `inner`
    @test a.kkt.linear_solver.csc === a.kkt.linear_solver.wrapped.csc
    @test b.kkt.linear_solver.inner === getfield(b.kkt.linear_solver.wrapped, :inner)
    @test hasproperty(a.kkt.linear_solver, :csc) && !hasproperty(a.kkt.linear_solver, :tril)
    @test MadNLP.LinearSolverBarrier(a.kkt.linear_solver) === a.kkt.linear_solver
    @test MadNLP.solve!(a).status == MadNLP.solve!(b).status == MadNLP.SOLVE_SUCCEEDED
    @test a.obj_val ≈ b.obj_val
end

@testset "MadNLP scaling" begin
    MadNLPTests.test_scaling()
end

@testset "Max optimization problem" begin
    MadNLPTests.test_max_problem()
end

@testset "Fixed variables" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test isa(solver.cb.fixed_handler, MadNLP.NoFixedVariables)

    # Fix first variable:
    nlp.meta.lvar[1] = 0.5
    solver_sparse = MadNLP.MadNLPSolver(nlp; callback=MadNLP.SparseCallback, print_level=MadNLP.ERROR)
    sol_sparse = MadNLP.solve!(solver_sparse)
    @test length(solver_sparse.ind_fixed) == 1
    @test isa(solver_sparse.cb.fixed_handler, MadNLP.MakeParameter)
    @test solver_sparse.n == 3 # fixed variables are removed

    solver_dense = MadNLP.MadNLPSolver(nlp; callback=MadNLP.DenseCallback, print_level=MadNLP.ERROR)
    sol_dense = MadNLP.solve!(solver_dense)
    @test length(solver_dense.ind_fixed) == 1
    @test isa(solver_dense.cb.fixed_handler, MadNLP.MakeParameter)
    @test solver_dense.n == 4 # fixed variables are frozen

    @test sol_dense.iter == sol_sparse.iter
    @test sol_dense.objective == sol_sparse.objective
    @test sol_dense.solution == sol_sparse.solution
    @test sol_dense.multipliers == sol_sparse.multipliers
    @test sol_dense.multipliers_L == sol_sparse.multipliers_L
    @test sol_dense.multipliers_U == sol_sparse.multipliers_U
end

@testset "MadNLP warmstart" begin
    nlp = MadNLPTests.HS15Model()
    x0 = NLPModels.get_x0(nlp)
    y0 = NLPModels.get_y0(nlp)

    solver = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    @test solver.status == MadNLP.INITIAL
    MadNLP.solve!(solver; x=x0, y=y0)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
    # Update barrier term and solve again
    MadNLP.solve!(solver; x=x0, y=y0, mu_init=1e-5)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end


@testset "NLS problem" begin
    nlp = MadNLPTests.NLSModel()
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    @test solver.status == MadNLP.SOLVE_SUCCEEDED
end

@testset "MadNLP callback allocations" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp)
    MadNLP.initialize!(solver)
    kkt = solver.kkt
    x, f, c = solver.x, solver.f, solver.c
    # Precompile
    MadNLP.eval_f_wrapper(solver, x)
    MadNLP.eval_grad_f_wrapper!(solver, f, x)
    MadNLP.eval_cons_wrapper!(solver, c, x)
    MadNLP.eval_jac_wrapper!(solver, kkt, x)
    MadNLP.eval_lag_hess_wrapper!(solver, kkt, x, solver.y)

    n_allocs = @allocated MadNLP.eval_f_wrapper(solver, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_grad_f_wrapper!(solver, f, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_cons_wrapper!(solver, c, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_jac_wrapper!(solver, kkt, x)
    @test n_allocs == 0
    n_allocs = @allocated MadNLP.eval_lag_hess_wrapper!(solver, kkt, x, solver.y)
    @test n_allocs == 0
end

@testset "MadNLP timings" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp)
    MadNLP.initialize!(solver)
    time_callbacks = MadNLP.timing_callbacks(solver)
    @test isa(time_callbacks, NamedTuple)
    time_linear_solver = MadNLP.timing_linear_solver(solver)
    @test isa(time_linear_solver, NamedTuple)
    time_madnlp = MadNLP.timing_madnlp(solver)
    @test isa(time_madnlp.time_linear_solver, NamedTuple)
    @test isa(time_madnlp.time_callbacks, NamedTuple)
end

@testset "MadNLP profiling counters" begin
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    cnt = solver.cnt
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
    @test isempty(cnt.timer.starts)
    eval_times = (cnt.eval_obj_time, cnt.eval_grad_time, cnt.eval_cons_time, cnt.eval_jac_time, cnt.eval_hess_time)
    @test all(>(0), eval_times)
    @test cnt.factorization_time > 0 && cnt.solve_time > 0
    # the per-category regions are nested in the totals
    @test sum(eval_times) <= cnt.eval_function_time
    @test sum(eval_times) >= 0.5 * cnt.eval_function_time
    @test cnt.factorization_time + cnt.solve_time <= cnt.linear_solver_time
    @test cnt.factorization_time + cnt.solve_time >= 0.5 * cnt.linear_solver_time
    @test cnt.backsolve_cnt >= cnt.k
    @test cnt.factorization_cnt >= cnt.k
end

@testset "Adaptive barrier" begin
    nlp = MadNLPTests.HS15Model(; x0=[1.0, 1.0])
    ref = madnlp(nlp; print_level = MadNLP.ERROR)
    for barrier in [
        MadNLP.LOQOUpdate(),
        MadNLP.QualityFunctionUpdate(),
        MadNLP.QualityFunctionUpdate(; globalization=false),
    ]
        results = madnlp(nlp; print_level = MadNLP.ERROR, barrier=barrier)
        @test results.status == MadNLP.SOLVE_SUCCEEDED
        @test results.objective ≈ ref.objective
        @test results.solution ≈ ref.solution
        @test results.multipliers ≈ ref.multipliers
    end
end

@testset "Adaptive barrier ($T)" for T in (Float32, Float128)
    nlp = MadNLPTests.HS15Model(; T=T, x0=T[1, 1])
    tol = T == Float32 ? T(1e-4) : T(1e-8)
    for barrier in [
        MadNLP.MonotoneUpdate(),
        MadNLP.LOQOUpdate{T}(),
        MadNLP.QualityFunctionUpdate{T}(),
    ]
        solver = MadNLPSolver(
            nlp;
            tol=tol,
            callback=MadNLP.SparseCallback,
            kkt_system=MadNLP.SparseKKTSystem,
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            barrier=barrier,
        )
        @test solver.opt.barrier isa MadNLP.AbstractBarrierUpdate{T}
        @test solver.opt.barrier.mu_min == T(barrier.mu_min)

        results = MadNLP.solve!(solver)
        @test results.status == MadNLP.SOLVE_SUCCEEDED
        @test eltype(results.solution) == T
    end
end

@testset "SparseUnreducedKKTSystem ($T)" for T in (Float32, Float128)
    nlp = MadNLPTests.HS15Model(; T=T, x0=T[1, 1])
    solver = MadNLPSolver(
        nlp;
        tol=T == Float32 ? T(1e-4) : T(1e-8),
        callback=MadNLP.SparseCallback,
        kkt_system=MadNLP.SparseUnreducedKKTSystem,
        linear_solver=LDLSolver,
        print_level=MadNLP.ERROR,
    )
    @test eltype(solver.kkt.pr_diag) == T
    @test eltype(solver.kkt.hess) == T
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test eltype(results.solution) == T
end

# Maximize -f(x), where f is the objective of the wrapped minimization problem.
struct NegatedModel{T, VT, M <: NLPModels.AbstractNLPModel{T, VT}} <: NLPModels.AbstractNLPModel{T, VT}
    inner::M
    meta::NLPModels.NLPModelMeta{T, VT}
    counters::NLPModels.Counters
end
NegatedModel(inner; kwargs...) = NegatedModel(
    inner,
    NLPModels.NLPModelMeta(inner.meta; minimize = false, kwargs...),
    NLPModels.Counters(),
)
NLPModels.obj(nlp::NegatedModel, x::AbstractVector) = -NLPModels.obj(nlp.inner, x)
NLPModels.grad!(nlp::NegatedModel, x::AbstractVector, g::AbstractVector) = (NLPModels.grad!(nlp.inner, x, g); g .*= -1)
NLPModels.cons!(nlp::NegatedModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(nlp.inner, x, c)
NLPModels.jac_structure!(nlp::NegatedModel, I::AbstractVector, J::AbstractVector) = NLPModels.jac_structure!(nlp.inner, I, J)
NLPModels.jac_coord!(nlp::NegatedModel, x::AbstractVector, J::AbstractVector) = NLPModels.jac_coord!(nlp.inner, x, J)
NLPModels.jtprod!(nlp::NegatedModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = NLPModels.jtprod!(nlp.inner, x, v, Jtv)
NLPModels.hess_structure!(nlp::NegatedModel, I::AbstractVector, J::AbstractVector) = NLPModels.hess_structure!(nlp.inner, I, J)
function NLPModels.hess_coord!(nlp::NegatedModel, x::AbstractVector, y::AbstractVector, H::AbstractVector; obj_weight = 1.0)
    return NLPModels.hess_coord!(nlp.inner, x, y, H; obj_weight = -obj_weight)
end
NLPModels.jac_dense!(nlp::NegatedModel, x, J) = NLPModels.jac_dense!(nlp.inner, x, J)
NLPModels.hess_dense!(nlp::NegatedModel, x, y, H; obj_weight = 1.0) = NLPModels.hess_dense!(nlp.inner, x, y, H; obj_weight = -obj_weight)

@testset "Issue #660: sign of the multipliers for a maximization problem" begin
    # x1 <= 0.5 is active at the solution of HS15: fixing x1 = 0.5 does not change it.
    for (lvar, uvar) in (([-Inf, -Inf], [0.5, Inf]), ([0.5, -Inf], [0.5, Inf]))
        nlp_min = MadNLPTests.HS15Model()
        nlp_min.meta.lvar .= lvar
        nlp_min.meta.uvar .= uvar
        nlp_max = NegatedModel(nlp_min)
        sol_min = madnlp(nlp_min; print_level=MadNLP.ERROR)
        sol_max = madnlp(nlp_max; print_level=MadNLP.ERROR)
        @test sol_min.status == sol_max.status == MadNLP.SOLVE_SUCCEEDED
        @test sol_max.objective ≈ -sol_min.objective
        @test sol_max.solution ≈ sol_min.solution
        @test sol_max.multipliers ≈ -sol_min.multipliers
        @test sol_max.multipliers_L ≈ -sol_min.multipliers_L
        @test sol_max.multipliers_U ≈ -sol_min.multipliers_U
        # Stationarity of the problem as written: ∇f + Jᵀy - zl + zu = 0
        for (nlp, sol) in ((nlp_min, sol_min), (nlp_max, sol_max))
            x = sol.solution
            res = NLPModels.grad(nlp, x) + NLPModels.jac(nlp, x)' * sol.multipliers -
                sol.multipliers_L + sol.multipliers_U
            @test maximum(abs, res) <= 1e-6
        end
    end
end

@testset "Issue #430" begin
    # Test MadNLP is working with bound_relax_factor=0
    nlp = MadNLPTests.HS15Model()
    solver = MadNLPSolver(nlp; bound_relax_factor=0.0, print_level=MadNLP.ERROR)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.SOLVE_SUCCEEDED
end

@kwdef struct UserCallback <: MadNLP.AbstractUserCallback
    counter::Base.RefValue{Int} = Ref(0)
end

function (cb::UserCallback)(solver::MadNLP.AbstractMadNLPSolver, mode)
    cb.counter[] += 1
    return solver.cnt.k <= 3
end

@testset "User-defined callback termination" begin
    nlp = MadNLPTests.HS15Model()
    cb = UserCallback()
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, intermediate_callback=cb)
    stats = MadNLP.solve!(solver)
    @test stats.status == MadNLP.USER_REQUESTED_STOP
    @test solver.cnt.k == 4
    @test cb.counter[] == 5
end

@testset "Warn on option ignore" begin
    pipe = Pipe()
    redirect_stdout(pipe) do
        MadNLPSolver(MadNLPTests.HS15Model(); fake_option = true)
    end
    @test readline(pipe) == "The following options are ignored: "
    @test readline(pipe) == " - fake_option"
end

@testset "Restoration converged at a feasible point" begin
    # robust! entered at the solution of a feasible problem converges at a feasible point.
    function solved()
        solver = MadNLPSolver(MadNLPTests.HS15Model(; x0 = [1.0, 1.0]); print_level = MadNLP.ERROR)
        @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
        solver.opt.max_iter = 2000
        return solver
    end

    solver = solved()
    @test MadNLP.robust!(solver) == MadNLP.REGULAR
    @test solver.RR.n_feasible_returns == 1

    # the limit applies to one solve
    MadNLP.reinitialize!(solver)
    @test solver.RR.n_feasible_returns == 0

    # past the limit, such a phase ends the solve
    solver = solved()
    MadNLP.initialize_robust_restorer!(solver)
    solver.RR.n_feasible_returns = MadNLP.MAX_FEASIBLE_RETURNS
    @test MadNLP.robust!(solver) == MadNLP.INFEASIBLE_PROBLEM_DETECTED
end

@testset "Slack reset" begin
    nlp = MadNLPTests.HS15Model(; x0 = [1.0, 1.0])
    results = madnlp(nlp; print_level = MadNLP.ERROR, slack_reset = true)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test results.objective ≈ madnlp(nlp; print_level = MadNLP.ERROR).objective

    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, slack_reset = true)
    MadNLP.solve!(solver)
    xs = MadNLP.slack(MadNLP.get_x_trial(solver))
    c = MadNLP.get_c_trial(solver)
    cs = view(c, MadNLP.get_ind_ineq(solver))
    s0 = copy(xs)

    # a slack below g(x) moves up to it, when the rows hold
    cs[1] = 1.0e-3
    MadNLP.reset_slacks!(solver)
    @test xs[1] ≈ s0[1] + 1.0e-3
    @test cs[1] == 0

    # nothing moves when a row is violated
    copyto!(xs, s0)
    cs .= 0
    cs[1] = -1.0 - s0[1] - 1.0e-3
    MadNLP.reset_slacks!(solver)
    @test xs == s0
end

@testset "Barrier decrease after a tiny step" begin
    function mu_after(tiny)
        solver = MadNLPSolver(MadNLPTests.HS15Model(; x0 = [1.0, 1.0]); print_level = MadNLP.ERROR)
        MadNLP.initialize!(solver)
        mu = MadNLP.get_mu(solver)
        # a barrier sub-problem far from solved
        MadNLP.set_inf_pr!(solver, 1.0e3)
        solver.cnt.tiny_step = tiny
        MadNLP._update_monotone!(solver.opt.barrier, solver, 1.0)
        return MadNLP.get_mu(solver) / mu, solver.cnt.tiny_step
    end
    @test mu_after(false) == (1.0, false)
    ratio, flag = mu_after(true)
    @test ratio < 1
    @test !flag

    # the flag does not outlive the iteration under an adaptive barrier, nor a resolve
    nlp = MadNLPTests.HS15Model(; x0 = [1.0, 1.0])
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, barrier = MadNLP.QualityFunctionUpdate(1.0e-8, 10.0))
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    @test !solver.cnt.tiny_step
    solver.cnt.tiny_step = true
    MadNLP.reinitialize!(solver)
    @test !solver.cnt.tiny_step
end      
      
# min (x1 - 3)^4 + x2^2 subject to x1 + x2 = 1 written twice: the Jacobian is rank deficient
struct RedundantEqualityModel <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
end
RedundantEqualityModel() = RedundantEqualityModel(
    NLPModels.NLPModelMeta(2; ncon = 2, nnzj = 4, nnzh = 2, x0 = [-5.0, 9.0], lcon = [1.0, 1.0], ucon = [1.0, 1.0]),
    NLPModels.Counters(),
)
NLPModels.obj(::RedundantEqualityModel, x::AbstractVector) = (x[1] - 3)^4 + x[2]^2
function NLPModels.grad!(::RedundantEqualityModel, x::AbstractVector, g::AbstractVector)
    g[1] = 4 * (x[1] - 3)^3
    g[2] = 2 * x[2]
    return g
end
NLPModels.cons!(::RedundantEqualityModel, x::AbstractVector, c::AbstractVector) = (c .= x[1] + x[2]; c)
function NLPModels.jac_structure!(::RedundantEqualityModel, I::AbstractVector, J::AbstractVector)
    I .= [1, 1, 2, 2]
    J .= [1, 2, 1, 2]
    return I, J
end
NLPModels.jac_coord!(::RedundantEqualityModel, x::AbstractVector, vals::AbstractVector) = (vals .= 1.0; vals)
function NLPModels.hess_structure!(::RedundantEqualityModel, I::AbstractVector, J::AbstractVector)
    I .= [1, 2]
    J .= [1, 2]
    return I, J
end
function NLPModels.hess_coord!(::RedundantEqualityModel, x, y, vals::AbstractVector; obj_weight = 1.0)
    vals[1] = 12 * (x[1] - 3)^2 * obj_weight
    vals[2] = 2 * obj_weight
    return vals
end

@testset "Dual regularization without Hessian perturbation" begin
    solver = MadNLPSolver(RedundantEqualityModel(); print_level = MadNLP.ERROR)
    results = MadNLP.solve!(solver)
    @test results.status == MadNLP.SOLVE_SUCCEEDED
    @test sum(results.solution) ≈ 1 atol = 1.0e-6
    # the dual block needed regularization at every iteration, so the Jacobian is taken as degenerate
    @test solver.inertia_corrector.jac_degenerate
    # and the Hessian was never perturbed
    @test solver.del_w_last == 0

    # a full-rank Jacobian is not taken as degenerate, on the augmented and the condensed system
    for kkt_system in (MadNLP.SparseKKTSystem, MadNLP.SparseCondensedKKTSystem)
        nlp = MadNLP.SparseWrapperModel(Array, MadNLPTests.HS15Model(; x0 = [1.0, 1.0]))
        solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, kkt_system)
        @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
        @test !solver.inertia_corrector.jac_degenerate
    end

    # a failed Cholesky factorization of the condensed system is not a singular dual block
    if Base.USE_GPL_LIBS
        nlp = MadNLP.SparseWrapperModel(Array, MadNLPTests.HS15Model(; x0 = [1.0, 1.0]))
        reference = madnlp(nlp; print_level = MadNLP.ERROR)
        solver = MadNLPSolver(
            nlp; print_level = MadNLP.ERROR,
            kkt_system = MadNLP.SparseCondensedKKTSystem, linear_solver = MadNLP.CHOLMODSolver,
        )
        results = MadNLP.solve!(solver)
        @test results.status == MadNLP.SOLVE_SUCCEEDED
        @test !solver.inertia_corrector.jac_degenerate
        @test results.solution ≈ reference.solution atol = 1.0e-6
    end
end


# counts the tentative watchdog steps of a solve
struct WatchdogSteps <: MadNLP.AbstractUserCallback
    count::Base.RefValue{Int}
end
(cb::WatchdogSteps)(solver, mode) = (MadNLP.get_ftype(solver) == "w" && (cb.count[] += 1); true)

@testset "Watchdog line search" begin
    nlp = MadNLPTests.HS15Model(; x0 = [1.0, 1.0])
    on = madnlp(nlp; print_level = MadNLP.ERROR)
    off = madnlp(nlp; print_level = MadNLP.ERROR, watchdog = false)
    @test on.status == off.status == MadNLP.SOLVE_SUCCEEDED
    @test on.objective ≈ off.objective

    # a watchdog started at every iteration still reaches the same solution
    for trigger in (0, 1)
        r = madnlp(nlp; print_level = MadNLP.ERROR, watchdog_shortened_iter_trigger = trigger)
        @test r.status == MadNLP.SOLVE_SUCCEEDED
        @test r.objective ≈ off.objective atol = 1.0e-6
    end

    # tentative watchdog steps are taken and the solve still reaches the solution
    steps = WatchdogSteps(Ref(0))
    nlp_far = MadNLPTests.HS15Model(; x0 = [-2.0, 1.0])
    r = madnlp(nlp_far; print_level = MadNLP.ERROR, watchdog_shortened_iter_trigger = 0, intermediate_callback = steps)
    @test r.status == MadNLP.SOLVE_SUCCEEDED
    @test steps.count[] > 0
    @test r.objective ≈ madnlp(nlp_far; print_level = MadNLP.ERROR, watchdog = false).objective atol = 1.0e-6

    # the reference is restored exactly
    solver = MadNLPSolver(nlp; print_level = MadNLP.ERROR, watchdog = true, watchdog_shortened_iter_trigger = 0)
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    wd = MadNLP.WatchdogState(solver)
    MadNLP.save_watchdog_iterate!(solver, wd)
    # the reference values follow μ: at the saved μ they are unchanged
    varphi_ref, varphi_d_ref = wd.varphi_ref, wd.varphi_d_ref
    MadNLP.update_watchdog_reference!(wd, MadNLP.get_mu(solver))
    @test wd.varphi_ref ≈ varphi_ref
    @test wd.varphi_d_ref ≈ varphi_d_ref
    x = copy(MadNLP.full(MadNLP.get_x(solver)))
    y = copy(MadNLP.get_y(solver))
    MadNLP.full(MadNLP.get_x(solver)) .+= 1.0
    MadNLP.get_y(solver) .+= 1.0
    MadNLP.restore_watchdog_iterate!(solver, wd)
    @test MadNLP.full(MadNLP.get_x(solver)) == x
    @test MadNLP.get_y(solver) == y

    # a filter reset after the save (as by a change of μ) survives the rollback
    MadNLP.save_watchdog_iterate!(solver, wd)
    push!(MadNLP.get_filter(solver), (1.0, 1.0))
    empty!(MadNLP.get_filter(solver))
    push!(MadNLP.get_filter(solver), (MadNLP.get_theta_max(solver), -Inf))
    MadNLP.restore_watchdog_iterate!(solver, wd)
    @test MadNLP.get_filter(solver) == [(MadNLP.get_theta_max(solver), -Inf)]
end


@testset "Barrier parameter floor" begin
    # μ stops at tol / (barrier_tol_factor + 1) when mu_min is smaller
    @test MadNLP.get_mu(1.0e-9, 1.0e-11, 0.2, 1.5, 1.0e-8, 10.0) ≈ 1.0e-8 / 11
    @test MadNLP.get_mu(1.0e-9, 1.0e-6, 0.2, 1.5, 1.0e-8, 10.0) == 1.0e-6
    @test MadNLP.get_mu(1.0, 1.0e-11, 0.2, 1.5, 1.0e-8, 10.0) == 0.2
    # with a loose tol, the floor stays at the default mu_min
    @test MadNLP.get_mu(1.0e-6, 1.0e-11, 0.2, 1.5, 1.0e-3, 10.0) ≈ 1.0e-4 / 11
end
      
@testset "Divergence test on the primal variables" begin
    # From x0 = 0, the termination measure of HS15 reaches about 3e4 while |x| stays below 1.3:
    # the divergence test looks at the variables only, as Ipopt does.
    solver = MadNLPSolver(MadNLPTests.HS15Model(); print_level = MadNLP.ERROR, diverging_iterates_tol = 1e3)
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    solver = MadNLPSolver(MadNLPTests.HS15Model(); print_level = MadNLP.ERROR, diverging_iterates_tol = 1.0)
    @test MadNLP.solve!(solver).status == MadNLP.DIVERGING_ITERATES
end

@testset "Original constraint violation" begin
    solver = MadNLPSolver(MadNLPTests.HS15Model(; x0 = [1.0, 1.0]); print_level = MadNLP.ERROR)
    @test MadNLP.solve!(solver).status == MadNLP.SOLVE_SUCCEEDED
    c = MadNLP.get_c(solver)
    cs = view(c, MadNLP.get_ind_ineq(solver))
    s = MadNLP.slack(MadNLP.get_x(solver))

    # a slack moved away from g(x) inside the bounds on a row with no multiplier: a slack
    # residual, no violation
    y = view(MadNLP.get_y(solver), MadNLP.get_ind_ineq(solver))
    y .= 0
    s .+= 1.0e-3
    cs .-= 1.0e-3
    @test MadNLP.get_inf_pr(c) >= 1.0e-3
    @test MadNLP.get_inf_pr_orig(solver) < 1.0e-6

    # with a multiplier the residual counts, weighted by it
    y .= 2.0
    @test MadNLP.get_inf_pr_orig(solver) ≈ 2.0e-3
    y .= 0

    # g(x) below its lower bound: a violation
    cs .-= 2.0
    @test MadNLP.get_inf_pr_orig(solver) > 1.0
end
      
@testset "Filter margins" begin
    filter = Tuple{Float64, Float64}[]
    MadNLP.augment_filter!(filter, 2.0, 3.0, 0.1, 0.25)
    @test filter[1] == (1.8, 2.5)
end
