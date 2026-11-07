# The problem sets. `SETS[name]()` returns the instances of a set as `name => build`, where
# `build(backend)` returns the model on that backend (`nothing` for the host). Included by
# session.jl after the problem packages are loaded.

# The 66 typical-operating-point cases of pglib-opf, each in the polar and in the rectangular form.
const ACOPF_CASES = [
    "case3_lmbd", "case5_pjm", "case14_ieee", "case24_ieee_rts", "case30_as", "case30_ieee",
    "case39_epri", "case57_ieee", "case60_c", "case73_ieee_rts", "case89_pegase", "case118_ieee",
    "case162_ieee_dtc", "case179_goc", "case197_snem", "case200_activ", "case240_pserc",
    "case300_ieee", "case500_goc", "case588_sdet", "case793_goc", "case1354_pegase",
    "case1803_snem", "case1888_rte", "case1951_rte", "case2000_goc", "case2312_goc",
    "case2383wp_k", "case2736sp_k", "case2737sop_k", "case2742_goc", "case2746wop_k",
    "case2746wp_k", "case2848_rte", "case2853_sdet", "case2868_rte", "case2869_pegase",
    "case3012wp_k", "case3022_goc", "case3120sp_k", "case3375wp_k", "case3970_goc",
    "case4020_goc", "case4601_goc", "case4619_goc", "case4661_sdet", "case4837_goc",
    "case4917_goc", "case5658_epigrids", "case6468_rte", "case6470_rte", "case6495_rte",
    "case6515_rte", "case7336_epigrids", "case8387_pegase", "case9241_pegase", "case9591_goc",
    "case10000_goc", "case10192_epigrids", "case10480_goc", "case13659_pegase", "case19402_goc",
    "case20758_epigrids", "case24464_goc", "case30000_goc", "case78484_epigrids",
]
const ACOPF_FORMS = ("polar" => ExaModelsPower.Polar(), "rect" => ExaModelsPower.Rect())
acopf_case(name, (label, form)) =
    "$(name)_$(label)" => (backend -> ExaModelsPower.opf_model("pglib_opf_$name.m"; backend, form)[1])
acopf_cases(names) = [acopf_case(name, form) for name in names for form in ACOPF_FORMS]

# COPS: the nine scaling models of the ExaModels GPU study at five sizes each, and the other
# models of COPSBenchmark at their canonical sizes.
const COPS_CASES = [
    (:bearing, (200, 200)), (:bearing, (300, 300)), (:bearing, (400, 400)), (:bearing, (600, 600)), (:bearing, (800, 800)),
    (:camshape, (1600,)), (:camshape, (3200,)), (:camshape, (6400,)), (:camshape, (12800,)), (:camshape, (25600,)),
    (:elec, (100,)), (:elec, (200,)), (:elec, (400,)), (:elec, (800,)), (:elec, (1600,)),
    (:gasoil, (800,)), (:gasoil, (1600,)), (:gasoil, (3200,)), (:gasoil, (6400,)), (:gasoil, (12800,)),
    (:marine, (400,)), (:marine, (800,)), (:marine, (1600,)), (:marine, (3200,)), (:marine, (6400,)),
    (:pinene, (800,)), (:pinene, (1600,)), (:pinene, (3200,)), (:pinene, (6400,)), (:pinene, (12800,)),
    (:robot, (400,)), (:robot, (800,)), (:robot, (1600,)), (:robot, (3200,)), (:robot, (6400,)),
    (:rocket, (3200,)), (:rocket, (6400,)), (:rocket, (12800,)), (:rocket, (25600,)), (:rocket, (51200,)),
    (:steering, (3200,)), (:steering, (6400,)), (:steering, (12800,)), (:steering, (25600,)), (:steering, (51200,)),
    (:catmix, (100,)), (:chain, (800,)), (:channel, (100,)), (:glider, (100,)), (:methanol, (100,)),
    (:minsurf, (50, 50)), (:polygon, (100,)), (:torsion, (50, 50)),
    (:dirichlet, (20,)), (:henon, (10,)), (:lane_emden, (20,)),
    (:tetra_duct12, ()), (:tetra_duct15, ()), (:tetra_duct20, ()), (:tetra_foam5, ()), (:tetra_gear, ()), (:tetra_hook, ()),
    (:triangle_deer, ()), (:triangle_pacman, ()), (:triangle_turtle, ()),
]
function cops_case((model, args))
    name = string(model) * (isempty(args) ? "" : "_" * join(args, "x"))
    f = getproperty(COPSBenchmark, Symbol(model, :_model))
    # prod = true: the tetra and triangle models need Jacobian-transpose products on a device
    return name => (backend -> f(COPSBenchmark.ExaModelsBackend(), args...; backend, prod = true))
end

# CUTEst.jl decodes and compiles a problem in one directory of its installation, which two
# processes cannot do at once, so one process of run.jl decodes them and workers only load them.
cutest_host_model(name) = CUTEst.CUTEstModel{Float64}(name; decode = false)
cutest_decoded(name) = all(isfile, joinpath.(CUTEst.libsif_path, ["lib$(name)_double.so", "OUTSDIF_$name.d"]))
function decode_cutest(names, progress = nothing)
    for name in names
        cutest_decoded(name) || try
            finalize(CUTEst.CUTEstModel{Float64}(name))
        catch err
            @warn "$name: could not be decoded" exception = err
        end
        progress === nothing || put!(progress, name)
    end
    progress === nothing || close(progress)
    return
end

# Instances of one COPS model share a compiled model type, and so do the meshes of one kind.
instance_group(set, case) = set == "cops" ? replace(case, r"_[0-9x]+$|(?<=tetra|triangle)_.*$" => "") : case

const SETS = Dict{String, Function}(
    "acopf" => () -> acopf_cases(ACOPF_CASES),
    "cops" => () -> cops_case.(COPS_CASES),
    # the instances scored in Hans Mittelmann's AMPL-NLP benchmark table
    "mittelmann" => () -> [
        name => (backend -> ExaMittelmann.mittelmann(name; backend))
            for name in ExaMittelmann.mittelmann_instances(; benchmark = true)
    ],
    # the CUTEst problems with at least as many free variables as equality constraints: on the
    # host through CUTEst.jl, on a device as ExaCUTEst models
    "cutest" => () -> [
        name => (backend -> backend === nothing ? cutest_host_model(name) : ExaCUTEst.ExaCUTEstModel(name; backend))
            for name in readlines(joinpath(@__DIR__, "sets", "cutest.txt"))
    ],
    # a few small instances, for a functional check of a configuration and for CI
    "smoke" => () -> vcat(
        acopf_cases(["case14_ieee", "case118_ieee", "case1354_pegase"]),
        cops_case.([(:catmix, (100,)), (:robot, (400,)), (:rocket, (3200,)), (:marine, (400,)), (:gasoil, (800,)), (:steering, (3200,))]),
    ),
)
