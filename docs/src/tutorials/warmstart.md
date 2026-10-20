# Warmstarting MadNLP

```@meta
CurrentModule = MadNLP
```
```@setup warmstart
using NLPModels
using MadNLP
include("hs15.jl")

```

We use a parameterized version of the instance HS15
used in the [introduction](../quickstart.md). This updated
version of `HS15Model` stores the parameters of the model in an
attribute `nlp.params`:
```@example warmstart

nlp = HS15Model()
println(nlp.params)
```
By default the parameters are set to `[100.0, 1.0]`.
First, we find a solution associated to these parameters.
Then, we warmstart MadNLP from the solution found in the first solve,
after a small update in the problem's parameters.

!!! info
    It is known that the interior-point method has a poor
    support of warmstarting, on the contrary to active-set methods.
    However, if the new parameters remain close enough and do not lead
    to significant changes in the active set, warmstarting the
    interior-point algorithm can significantly reduce the total number of barrier iterations
    in the second solve.

!!! warning
    The warm-start described in this tutorial remains basic.
    Its main application is updating the solution of a parametric
    problem after a small update in the parameters. **The warm-start
    always assumes that the structure of the problem remains the same between
    two consecutive solves**.
    MadNLP cannot be warm-started if variables or constraints
    are added to the problem.

## Naive solution: starting from the previous solution
By default, MadNLP starts its interior-point algorithm
at the primal variable stored in `nlp.meta.x0`. We can
access this attribute using the function `get_x0`:
```@example warmstart
x0 = NLPModels.get_x0(nlp)

```
Here, we observe that the initial solution is `[0, 0]`.
We solve the problem starting from this point using the function [`madnlp`](@ref):
```@example warmstart
results = madnlp(nlp)
nothing
```
MadNLP converges in a bit less than 20 barrier iterations.  The solution is:
```@example warmstart
println("Objective: ", results.objective)
println("Solution:  ", results.solution)
```

### Updating the initial guess
We have found a solution to the problem. Now, what happens if we update
the parameters inside `nlp`?
```@example warmstart
nlp.params .= [101.0, 1.1]
```
As MadNLP starts the algorithm at `nlp.meta.x0`, we pass
the solution found previously to the initial vector:
```@example warmstart
copyto!(NLPModels.get_x0(nlp), results.solution)
```
Solving the problem again with MadNLP, we observe that MadNLP converges
in only 6 iterations:
```@example warmstart
results_new = madnlp(nlp)
nothing

```
By decreasing the initial barrier parameter, we can reduce the total number
of iterations to 5:
```@example warmstart
results_new = madnlp(nlp; barrier=MadNLP.MonotoneUpdate(; mu_init=1e-7))
nothing

```
The struct [`MonotoneUpdate`](@ref) implements the barrier strategy used by default in MadNLP,
and has an explicit attribute `mu_init` to define the initial barrier parameter.

The solution with the new parameters is slightly different from the former one:
```@example warmstart
results_new.solution

```

!!! info
    Similarly as with the primal solution, we can pass the initial dual solution to MadNLP
    using the function `get_y0`. We can overwrite the value of `y0` in `nlp` using:
    ```
    copyto!(NLPModels.get_y0(nlp), results.multipliers)
    ```
    and we specify to MadNLP to not recompute the dual multipliers using
    the option `dual_initialized`:
    ```
    madnlp(nlp; dual_initialized=true)
    ```
    In our particular example, setting the dual multipliers has only a minor influence
    on the convergence of the algorithm.

!!! info
    MadNLP does not support passing the initial values for the bounds' multipliers ``z_l`` and ``z_u``.


## Advanced solution: keeping the solver in memory

The previous solution works but is far from being efficient: each time we call
the function [`madnlp`](@ref) we create a new instance of [`MadNLPSolver`](@ref),
leading to a significant number of memory allocations. A workaround is to keep
the solver in memory to have more fine-grained control on the warm-start.

We start by creating a new model `nlp` and we instantiate a new instance [`MadNLPSolver`](@ref) attached to this model:
```@example warmstart
nlp = HS15Model()
solver = MadNLP.MadNLPSolver(nlp)
```
Note that
```@example warmstart
nlp === solver.nlp
```
Hence, updating the parameter values in `nlp` will automatically update the
parameters in the solver.

We first solve the problem using the function [`solve!`](@ref):
```@example warmstart
results = MadNLP.solve!(solver)
```
Before warmstarting MadNLP, we update the parameters and the primal solution in `nlp`:
```@example warmstart
nlp.params .= [101.0, 1.1]
copyto!(NLPModels.get_x0(nlp), results.solution)
```
MadNLP stores in memory the dual solutions computed during the first solve.
One can access to the (scaled) multipliers as
```@example warmstart
solver.y
```
and to the multipliers of the bound constraints with
```@example warmstart
[solver.zl.values solver.zu.values]
```

As before, it is advised to decrease the initial barrier parameter:
if the initial point is close enough to the solution, this reduces drastically
the total number of iterations. We solve the problem again using:
```@example warmstart
MadNLP.solve!(solver; barrier=MadNLP.MonotoneUpdate(; mu_init=1e-7))
nothing
```
Four observations are in order:
- The iteration count starts directly from the previous count (as stored in `solver.cnt.k`).
- MadNLP converges in only 4 iterations.
- The symbolic factorization of the KKT system stored in `solver` is directly re-used, leading to significant savings.
- The warm-start does not work if the structure of the problem changes between
  two consecutive solves (e.g, if variables or constraints are added to the problem).

!!! warning
    If we call the function [`solve!`](@ref) a second-time,
    MadNLP will use the following rule:
    - The initial primal solution is copied from `NLPModels.get_x0(nlp)`
    - The initial dual solution is directly taken from the values specified
      in `solver.y`, `solver.zl` and `solver.zu`.
      (MadNLP is not using the values stored in `nlp.meta.y0` in the second solve).


