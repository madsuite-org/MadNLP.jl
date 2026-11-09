@testset "Presets" begin
    MC = MadNLP.MadCore
    nlp = MadNLP.MadNLPCore.PrecompileChainModel(100, Float64[])

    # The default, ShiftedPreset, on a sparse model
    opt = MadNLP.MadNLPOptions{Float64}(nlp)
    @test opt.kkt_system === MadNLP.SparseKKTSystem
    @test opt.linear_solver === (Base.USE_GPL_LIBS ? MadNLP.CHOLMODSolver : MadNLP.MumpsSolver)
    @test opt.kkt_scaler isa MadNLP.RuizScaler
    @test opt.dual_shift == 1.0e-10
    @test opt.iterator === MadNLP.KrylovIterator
    @test opt.barrier isa MadNLP.QualityFunctionUpdate{Float64}
    @test opt.barrier.sigma_max == 2.0 && opt.barrier.mu_min ≈ 1.0e-3 * opt.tol
    @test opt.slack_reset
    if Base.USE_GPL_LIBS
        lopt = MC.default_options(nlp, opt.kkt_system, opt.linear_solver)
        @test lopt.cholmod_algorithm == MadNLP.LDL && lopt.cholmod_dbound == 1.0e-10
    end

    # AugmentedPreset: the augmented system with the default sparse solver and the monotone rule
    aug = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.AugmentedPreset())
    @test aug.kkt_system === MadNLP.SparseKKTSystem
    @test aug.linear_solver === MadNLP.MumpsSolver
    @test aug.kkt_scaler isa MadNLP.NoScaler && iszero(aug.dual_shift)
    @test aug.iterator === MadNLP.RichardsonIterator
    @test aug.barrier isa MadNLP.MonotoneUpdate && !aug.slack_reset

    # The shifted options apply to the augmented system only
    cond = MadNLP.MadNLPOptions{Float64}(nlp; kkt_system = MadNLP.SparseCondensedKKTSystem)
    @test cond.kkt_scaler isa MadNLP.NoScaler && iszero(cond.dual_shift)
    @test cond.barrier isa MadNLP.MonotoneUpdate && !cond.slack_reset

    # Both presets solve, and an option given explicitly overrides the preset
    for preset in (MadNLP.ShiftedPreset(), MadNLP.AugmentedPreset())
        @test madnlp(nlp; preset, print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
    end
    solver = MadNLP.MadNLPSolver(nlp; barrier = MadNLP.MonotoneUpdate(1.0e-8, 10.0), print_level = MadNLP.ERROR)
    @test MadNLP.get_opt(solver).barrier isa MadNLP.MonotoneUpdate
    @test MadNLP.get_opt(solver).kkt_scaler isa MadNLP.RuizScaler
end
