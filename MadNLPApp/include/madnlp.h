/* madnlp.h: the C interface of libmadnlp, MadNLP compiled ahead of time.
 *
 *   min (or max) f(x)  s.t.  lcon <= c(x) <= ucon,  lvar <= x <= uvar
 *
 * The caller describes one problem with sizes, bounds, a starting point and evaluation
 * callbacks, and libmadnlp solves it with MadNLP's interior point method on the CPU:
 *
 *   madnlp_problem p = { ctx, n, m, nnzj, nnzh, 1, x0, lvar, uvar, lcon, ucon, NULL, NULL, NULL,
 *                        obj, grad, cons, jac_structure, jac, hess_structure, hess };
 *   const char* keys[] = { "tol" }, *vals[] = { "1e-8" };
 *   madnlp_result r = { x, y, zl, zu };
 *   int64_t rc = madnlp_solve(&p, 1, keys, vals, &r);   // 0: it ran; r.status says how it ended
 *
 *   cc main.c -Imadnlp/include -Lmadnlp/lib -lmadnlp -Wl,-rpath,madnlp/lib
 *
 * Conventions. Integers are int64_t. Sparse structures are 1-based coordinate lists (each
 * nonzero once). The Hessian is the lower triangle of w * Hess f + sum_i y_i Hess c_i, w the
 * objective weight passed to `hess`. ±INFINITY marks a missing bound. Callbacks return 0 on
 * success; anything else stops the solve (madnlp_solve then returns 2). The returned
 * multipliers make L = s f + y'c - zL'(x - lvar) + zU'(x - uvar) stationary, s = 1 when
 * minimizing and -1 when maximizing, with zL, zU >= 0.
 *
 * These structs have the layout of cnlp_solver.h's cnlp_problem and cnlp_result, and
 * libmadnlp also exports cnlp_solve, so it is a cnlp solver as well (`cnlp solve
 * --solver madnlp`, https://cnlplib.madsuite.org).
 */
#ifndef MADNLP_H
#define MADNLP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct madnlp_problem {
    void* ctx;                            /* passed back to every callback */
    int64_t nvar, ncon, nnzj, nnzh;
    int64_t minimize;                     /* 1 minimize, 0 maximize */
    const double *x0, *lvar, *uvar;       /* nvar */
    const double *lcon, *ucon;            /* ncon */
    const double *y0, *zl0, *zu0;         /* starting multipliers, or NULL */
    int64_t (*obj)(void* ctx, const double* x, double* f);
    int64_t (*grad)(void* ctx, const double* x, double* g);
    int64_t (*cons)(void* ctx, const double* x, double* c);
    int64_t (*jac_structure)(void* ctx, int64_t* rows, int64_t* cols);
    int64_t (*jac)(void* ctx, const double* x, double* vals);
    int64_t (*hess_structure)(void* ctx, int64_t* rows, int64_t* cols);
    int64_t (*hess)(void* ctx, const double* x, const double* y, double obj_weight, double* vals);
} madnlp_problem;

/* status: 0 solved, 1 acceptable, 2 infeasible, 3 unbounded, 4 iteration limit,
 * 5 time limit, 6 error */
typedef struct madnlp_result {
    double *x, *y, *zl, *zu;              /* the caller's arrays: nvar, ncon, nvar, nvar */
    double objective;                     /* f(x), in the problem's sense */
    int64_t status;
    int64_t iterations;
    char solver_status[128];              /* MadNLP's own status, NUL-terminated */
} madnlp_result;

/* Solve. Options are key=value pairs, as MadNLP's keyword options: tol=1e-8,
 * max_iter=500, print_level=INFO (or 1-6, 0 for quiet), linear_solver=LapackCPUSolver, ...
 * Returns 0 when the solver ran (result->status says how it ended), 1 for an invalid
 * option, 2 for a failure (result->solver_status says why). */
int64_t madnlp_solve(const madnlp_problem* problem, int64_t noptions,
                     const char* const* keys, const char* const* values, madnlp_result* result);

/* The version, copied out (returns its length). */
int64_t madnlp_version(uint8_t* buf, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
