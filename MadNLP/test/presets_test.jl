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
    @test sh.barrier.mu_decrease_max == 10.0 && sh.barrier.sigma_min == 1.0e-6

    # the other LDLᵀ solvers under ShiftedPreset bound small pivots as CHOLMOD does: LDLFactorizations
    # by block, positive in the primal block and negative in the dual one, the blocks the KKT system declares
    δ = MadNLP.MadNLPCore.preset_perturbation(Float64)
    lopt = MC.default_options(nlp, MadNLP.ShiftedPreset, MadNLP.SparseKKTSystem, MadNLP.LDLSolver)
    @test (lopt.ldl_r1, lopt.ldl_r2, lopt.ldl_tol) == (δ, -δ, δ)
    mopt = MC.default_options(nlp, MadNLP.ShiftedPreset, MadNLP.SparseKKTSystem, MadNLP.Ma86Solver)
    @test mopt.ma86_static == δ && mopt.ma86_u == 0.0
    mopt = MC.default_options(nlp, MadNLP.ShiftedPreset, MadNLP.SparseKKTSystem, MadNLP.Ma57Solver)
    @test mopt.ma57_pivot_bound == δ
    # [1 0 0; 0 0 1; 0 1 0] in the order 1, 2, 3, primal rows 1 and 2: the second pivot is zero, so
    # the factorization fails without the bound; with it, that pivot becomes δ and the third -1/δ
    tril = sparse(Int32[1, 2, 3, 3], Int32[1, 2, 2, 3], [1.0, 0.0, 1.0, 0.0])
    off = MC.default_options(MadNLP.LDLSolver)
    off.ldl_user_perm = [1, 2, 3]
    @test !MC.is_factorized(MC.solver_type(MadNLP.LDLSolver)(tril; opt = off))
    on = MC.default_options(MadNLP.LDLSolver)
    on.ldl_user_perm = [1, 2, 3]
    on.ldl_r1, on.ldl_r2, on.ldl_tol = δ, -δ, δ
    MC.set_block_structure!(on, 2, 1)
    M = MC.solver_type(MadNLP.LDLSolver)(tril; opt = on)
    @test MC.is_factorized(M) && MC.inertia(M) == (2, 0, 1)
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
    # CondensedPreset in Float32: the cap of the retried equality shift scales with the shift, so that
    # a retry stays possible (the shift, 100 preset_perturbation, is about 3e-3 there)
    co32 = MadNLP.MadNLPOptions{Float32}(nlp32; preset = MadNLP.CondensedPreset)
    @test co32.equality_shift_max ≈ 1.0f6 * MadNLP.MadNLPCore.preset_perturbation(Float32)
    @test co32.equality_shift * co32.equality_shift_retry_factor <= co32.equality_shift_max
    @test MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.CondensedPreset).equality_shift_max ≈ 1.0e-4
    # the same defaults without the preset (the condensed KKT system under another preset)
    o32 = MadNLP.MadNLPOptions{Float32}(nlp32; kkt_system = MadNLP.SparseCondensedKKTSystem)
    @test o32.equality_shift ≈ 100 * MadNLP.MadNLPCore.preset_perturbation(Float32) && o32.equality_shift_max ≈ co32.equality_shift_max

    # The shifted options apply to the augmented system only
    cond = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.ShiftedPreset, kkt_system = MadNLP.SparseCondensedKKTSystem)
    @test cond.kkt_scaler isa MadNLP.NoScaler && iszero(cond.dual_shift)
    @test cond.barrier isa MadNLP.MonotoneUpdate && !cond.slack_reset

    # CondensedPreset: the condensed system with the equalities enforced and condensed through the
    # shift, factorized by Cholesky after Ruiz equilibration, with ShiftedPreset's quality-function
    # barrier and Richardson refinement
    co = MadNLP.MadNLPOptions{Float64}(nlp; preset = MadNLP.CondensedPreset)
    @test co.preset === MadNLP.CondensedPreset
    @test co.kkt_system === MadNLP.SparseCondensedKKTSystem
    @test co.equality_treatment === MadNLP.EnforceEquality && co.fixed_variable_treatment === MadNLP.MakeParameter
    @test co.dual_initialization_method === MadNLP.DualInitializeLeastSquares
    @test co.linear_solver === (Base.USE_GPL_LIBS ? MadNLP.CHOLMODSolver : MadNLP.MumpsSolver)
    @test co.kkt_scaler isa MadNLP.RuizScaler && !co.slack_reset
    @test co.barrier isa MadNLP.QualityFunctionUpdate{Float64}
    @test co.barrier.sigma_max == 2.0 && co.barrier.mu_decrease_max == 10.0 && co.barrier.mu_min ≈ 1.0e-3 * co.tol
    @test co.iterator === MadNLP.RichardsonIterator && iszero(co.dual_shift)
    @test co.equality_shift == 1.0e-8
    @test co.equality_shift_retry && !co.equality_shift_retry_on_refinement
    @test co.equality_shift_retry
    if Base.USE_GPL_LIBS
        lopt = MC.default_options(nlp, co.preset, co.kkt_system, co.linear_solver)
        @test lopt.cholmod_algorithm == MadNLP.CHOLESKY && lopt.cholmod_dbound == 0.0
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
