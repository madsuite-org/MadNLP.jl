module MadNLPPardiso

include(joinpath("..", "deps", "deps.jl"))
const libpardiso = PARDISO_PATH

import Libdl: dlopen, RTLD_DEEPBIND
import MadNLP:
    MadNLP,
    @kwdef,
    MadNLPLogger,
    @debug,
    @warn,
    @error,
    SubVector,
    SparseMatrixCSC,
    SymbolicException,
    FactorizationException,
    SolveException,
    InertiaException,
    AbstractOptions,
    AbstractLinearSolver,
    set_options!,
    introduce,
    factorize!,
    solve_linear_system!,
    improve!,
    is_inertia,
    inertia,
    input_type,
    blas_num_threads,
    is_supported,
    default_options
import MKL_jll: libmkl_rt
import PrecompileTools

if @isdefined(libpardiso)
    include("libpardiso.jl")
    include("pardiso.jl")
end
include("pardisomkl.jl")

function __init__()
    # check_deps()
    return @isdefined(libpardiso) && dlopen(libpardiso, RTLD_DEEPBIND)
end

export PardisoSolver, PardisoMKLSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all = true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

include("precompile.jl")

end # module
