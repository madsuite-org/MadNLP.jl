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
