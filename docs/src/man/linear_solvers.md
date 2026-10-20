# Linear solvers

```@meta
CurrentModule = MadNLP
```
```@setup linear_solver_example
using SparseArrays
using NLPModels
using MadNLP
using MadNLPTests
# Build nonlinear model
nlp = MadNLPTests.HS15Model()
# Build KKT
cb = MadNLP.create_callback(
    MadNLP.SparseCallback,
    nlp,
)
linear_solver = LapackCPUSolver
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    linear_solver,
)

n = NLPModels.get_nvar(nlp)
m = NLPModels.get_ncon(nlp)
x = NLPModels.get_x0(nlp)
y = zeros(m)
hess_values = MadNLP.get_hessian(kkt)
NLPModels.hess_coord!(nlp, x, y, hess_values)

jac_values = MadNLP.get_jacobian(kkt)
NLPModels.jac_coord!(nlp, x, jac_values)
MadNLP.compress_jacobian!(kkt)

MadNLP.build_kkt!(kkt)

```

We suppose that the KKT system has been assembled previously
into a given [`AbstractKKTSystem`](@ref). Then, it remains to compute
the Newton step by solving the KKT system for a given
right-hand-side (given as a [`AbstractKKTVector`](@ref)).
That's exactly the role of the linear solver.

If we do not assume any structure, the KKT system writes in generic form
```math
K x = b
```
with $$K$$ the KKT matrix and $$b$$ the current right-hand-side.
MadNLP provides a suite of specialized linear solvers to solve
the linear system.

## Inertia detection
If the matrix $$K$$ has negative eigenvalues, we have no guarantee
that the solution of the KKT system is a descent direction with regards
to the original nonlinear problem. That's the reason why most of the linear
solvers compute the inertia
of the linear system when factorizing the matrix $$K$$.
The inertia counts the number of positive,
negative and zero eigenvalues in the matrix. If the inertia does not
meet a given criteria, then the matrix $$K$$ is regularized by adding
a multiple of the identity to it: $$K_r = K + \alpha I$$.

!!! note
    We recall that the inertia of a matrix $$K$$ is given as
    a triplet ``(n_+, n_-, n_0)``, with ``n_+`` the number of positive eigenvalues,
    ``n_-`` the number of negative eigenvalues and ``n_0`` the number of
    zero eigenvalues.


## Factorization algorithm
In nonlinear programming, it is common
to employ an ``LBL^\top`` factorization (with ``L`` lower triangular
and ``B`` block-diagonal) to decompose the symmetric indefinite
matrix $$K$$, as this algorithm returns the inertia
of the matrix directly as a result of the factorization.

!!! note
    When MadNLP runs in inertia-free mode, the algorithm
    does not require to compute the inertia when factorizing
    the matrix $$K$$. In that case, MadNLP can use a classical
    LU or QR factorization to solve the linear system $$Kx = b$$.


## Solving a KKT system with MadNLP

We suppose available a [`AbstractKKTSystem`](@ref) `kkt`, properly assembled
following the procedure presented [previously](kkt.md).
We can query the assembled matrix $$K$$ as
```@example linear_solver_example
K = MadNLP.get_kkt(kkt)

```
Then, if we want to pass the KKT matrix `K` to Lapack, this
translates to
```@example linear_solver_example
linear_solver = LapackCPUSolver(K)

```
The instance `linear_solver` does not copy the matrix $$K$$ and
instead keep a reference to it.
```@example linear_solver_example
linear_solver.A === K
```
That way every time we re-assemble the matrix $$K$$ in `kkt`,
the values are directly updated inside `linear_solver`.

To compute the factorization inside `linear_solver`,
one simply has to call:
```@example linear_solver_example
MadNLP.factorize!(linear_solver)

```
Once the factorization computed, computing the backsolve
for a right-hand-side `b` amounts to
```@example linear_solver_example
nk = size(kkt, 1)
b = rand(nk)
MadNLP.solve_linear_system!(linear_solver, b)
```
The values of `b` being modified inplace to store the solution $$x$$ of the linear
system $$Kx =b$$.

