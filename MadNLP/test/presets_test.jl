@testset "Presets" begin
    MC = MadNLP.MadCore
    nlp = MadNLP.MadNLPCore.PrecompileChainModel(100, Float64[])

    # The host default, AugmentedPreset: the augmented system with the default sparse solver,
    # its default options and the monotone rule
    opt = MadNLP.MadNLPOptions{Float64}(nlp)
    @test opt.preset === MadNLP.AugmentedPreset
    @test opt.kkt_system === MadNLP.SparseKKTSystem
    @test opt.linear_solver === MadNLP.MumpsSolver
    @test opt.kkt_scaler isa MadNLP.NoScaler && iszero(opt.dual_shift)
    @test opt.iterator === MadNLP.RichardsonIterator
    @test opt.barrier isa MadNLP.MonotoneUpdate && !opt.slack_reset
    if Base.USE_GPL_LIBS
        lopt = MC.default_options(nlp, MadNLP.AugmentedPreset, MadNLP.SparseKKTSystem, MadNLP.CHOLMODSolver)
        @test lopt.cholmod_algorithm == MC.default_options(MadNLP.CHOLMODSolver).cholmod_algorithm
        @test lopt.cholmod_dbound == 0.0
    end

    # ShiftedPreset
    sh = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.ShiftedPreset)
    @test sh.preset === MadNLP.ShiftedPreset
    @test sh.kkt_system === MadNLP.SparseKKTSystem
    @test sh.linear_solver === (Base.USE_GPL_LIBS ? MadNLP.CHOLMODSolver : MadNLP.MumpsSolver)
    @test sh.kkt_scaler isa MadNLP.RuizScaler
    @test sh.dual_shift == 1.0e-10
    @test sh.iterator === MadNLP.KrylovIterator
    @test sh.barrier isa MadNLP.QualityFunctionUpdate{Float64}
    @test sh.barrier.sigma_max == 2.0 && sh.barrier.mu_min ≈ 1.0e-3 * sh.tol
    @test sh.slack_reset
    if Base.USE_GPL_LIBS
        lopt = MC.default_options(nlp, sh.preset, sh.kkt_system, sh.linear_solver)
        @test lopt.cholmod_algorithm == MadNLP.LDL && lopt.cholmod_dbound == 1.0e-10
    end

    # The dual shift and the pivot bound: 1e-10 in Float64, grown with the unit roundoff in Float32
    @test MadNLP.MadNLPCore.preset_perturbation(Float64) === 1.0e-10
    @test MadNLP.MadNLPCore.preset_perturbation(Float32) ≈ 1.0f-10 * (eps(Float32) / eps(Float64))^(5 / 8)
    nlp32 = MadNLP.MadNLPCore.PrecompileChainModel(100, Float32[])
    sh32 = MadNLP.MadNLPOptions{Float32}(nlp32; preset = MadNLP.ShiftedPreset)
    @test sh32.dual_shift === MadNLP.MadNLPCore.preset_perturbation(Float32)

    # The shifted options apply to the condensed system too, with the equalities enforced
    cond = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.ShiftedPreset, kkt_system = MadNLP.SparseCondensedKKTSystem)
    @test cond.kkt_scaler isa MadNLP.RuizScaler && cond.dual_shift == MadNLP.MadNLPCore.preset_perturbation(Float64)
    @test cond.barrier isa MadNLP.QualityFunctionUpdate && cond.slack_reset
    @test cond.equality_treatment == MadNLP.EnforceEquality && cond.kkt_options[:condense_max_nnz] == 20

    # CondensedPreset: the condensed system, factorized by the same pivot-bounded LDLᵀ
    co = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.CondensedPreset)
    @test co.preset === MadNLP.CondensedPreset
    @test co.kkt_system === MadNLP.SparseCondensedKKTSystem
    @test co.equality_treatment === MadNLP.RelaxEquality
    @test co.linear_solver === (Base.USE_GPL_LIBS ? MadNLP.CHOLMODSolver : MadNLP.MumpsSolver)
    @test co.kkt_scaler isa MadNLP.NoScaler && co.barrier isa MadNLP.MonotoneUpdate
    if Base.USE_GPL_LIBS
        lopt = MC.default_options(nlp, co.preset, co.kkt_system, co.linear_solver)
        @test lopt.cholmod_algorithm == MadNLP.LDL && lopt.cholmod_dbound == 1.0e-10
    end

    # Every preset solves, and an option given explicitly overrides the preset
    for preset in (MadNLP.ShiftedPreset, MadNLP.AugmentedPreset, MadNLP.CondensedPreset)
        @test madnlp(nlp; preset, print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED
    end
    # With zero initial duals nothing is factorized before the quality-function barrier's first
    # solves, so the barrier update factorizes first (CHOLMOD cannot solve without a factor)
    if Base.USE_GPL_LIBS
        stats = madnlp(
            nlp; preset = MadNLP.ShiftedPreset, dual_initialization_method = MadNLP.DualInitializeSetZero, print_level = MadNLP.ERROR,
        )
        @test stats.status == MadNLP.SOLVE_SUCCEEDED
    end
    solver = MadNLP.MadNLPSolver(
        nlp; preset = MadNLP.ShiftedPreset, barrier = MadNLP.MonotoneUpdate(1.0e-8, 10.0), print_level = MadNLP.ERROR,
    )
    @test MadNLP.get_opt(solver).barrier isa MadNLP.MonotoneUpdate
    @test MadNLP.get_opt(solver).kkt_scaler isa MadNLP.RuizScaler
end
