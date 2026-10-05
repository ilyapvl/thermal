#include "solver.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

double dot(const double* a, const double* b, int n)
{
#ifdef __APPLE__
    double result = 0.0;
    vDSP_dotprD(a, 1, b, 1, &result, static_cast<vDSP_Length>(n));
    return result;
#else
    double s = 0.0;
    for (int i = 0; i < n; i++)
    {
        s += a[i] * b[i];
    }
    return s;
#endif
}

SolverResult cg_solve_generic(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_A,
    double bnorm_sq,
    const std::function<void(std::vector<double>&)>& fill_rhs,
    std::vector<double>& x,
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_M,
    int max_iter,
    double tol)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const int n = static_cast<int>(x.size());
    if (n <= 0) throw std::runtime_error("cg_solve_generic: x must be pre-sized");

    std::vector<double> r(n, 0.0), p(n), Ap(n);

    fill_rhs(r);

    const double bnorm = std::sqrt(std::max(bnorm_sq, 1e-300));

    if (apply_M) apply_M(r, Ap);
    else std::copy(r.begin(), r.end(), Ap.begin());

    std::copy(Ap.begin(), Ap.end(), p.begin());

    double rz = dot(r, Ap);

    for (int it = 1; it <= max_iter; it++)
    {
        apply_A(p, Ap);
        const double pAp = dot(p, Ap);

        if (pAp <= 0.0)
        {
            SolverResult res;
            res.iterations = it - 1;
            res.rel_residual = std::sqrt(dot(r, r)) / bnorm;
            res.converged = false;
            res.seconds = std::chrono::duration<double>(clock::now() - t0).count();
            return res;
        }

        const double alpha = rz / pAp;

#ifdef __APPLE__
        {
            const double pos_a = alpha;
            const double neg_a = -alpha;
            vDSP_vsmaD(p.data(), 1, &pos_a, x.data(), 1, x.data(), 1, static_cast<vDSP_Length>(n));
            vDSP_vsmaD(Ap.data(), 1, &neg_a, r.data(), 1, r.data(), 1, static_cast<vDSP_Length>(n));
        }
#else
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
#endif

        const double rnorm = std::sqrt(dot(r, r));

        if (rnorm / bnorm < tol)
        {
            SolverResult res;
            res.iterations = it;
            res.rel_residual = rnorm / bnorm;
            res.converged = true;
            res.seconds = std::chrono::duration<double>(clock::now() - t0).count();
            return res;
        }

        if (apply_M) apply_M(r, Ap);
        else std::copy(r.begin(), r.end(), Ap.begin());

        const double rz_new = dot(r, Ap);
        const double beta = rz_new / rz;

#ifdef __APPLE__
        {
            const double bb = beta;
            vDSP_vsmaD(p.data(), 1, &bb, Ap.data(), 1, p.data(), 1, static_cast<vDSP_Length>(n));
        }
#else
        for (int i = 0; i < n; i++) p[i] = Ap[i] + beta * p[i];
#endif

        rz = rz_new;
    }

    SolverResult res;
    res.iterations = max_iter;
    res.rel_residual = std::sqrt(dot(r, r)) / bnorm;
    res.converged = false;
    res.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    return res;
}
