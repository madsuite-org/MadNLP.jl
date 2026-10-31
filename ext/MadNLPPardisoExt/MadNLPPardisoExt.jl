module MadNLPPardisoExt

import MadNLP
import Pardiso
import PrecompileTools

# Solve small problems with MKL Pardiso during precompilation, so its first use does not
# compile the solver. Panua Pardiso is left out: it needs a license at run time.
PrecompileTools.@compile_workload begin
    if Pardiso.mkl_is_available()
        MadNLP._precompile_linear_solver(Float64[], MadNLP.PardisoMKLSolver)
    end
end

end # module
