# Solve small problems with MKL Pardiso during precompilation, so its first use does not
# compile the solver. Panua Pardiso is left out: it needs a license at run time.
PrecompileTools.@compile_workload begin
    MadNLP._precompile_linear_solver(Float64[], PardisoMKLSolver)
end
