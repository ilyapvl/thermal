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
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

SolverResult cg_solve(
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
    if (n <= 0) throw std::runtime_error("cg_solve: x must be pre-sized");

    std::vector<double> r(n), p(n), Ap(n);

    fill_rhs(r);

    const double bnorm = std::sqrt(std::max(bnorm_sq, 1e-300));

    double rz;
    
    {
        std::vector<double> z(n);
        if (apply_M) apply_M(r, z);
        else std::copy(r.begin(), r.end(), z.begin());
        std::copy(z.begin(), z.end(), p.begin());
        rz = dot(r, z);
    }

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
            const double a = alpha;
            vDSP_vsmaD(p.data(), 1, &a, x.data(), 1, x.data(), 1,
                       static_cast<vDSP_Length>(n));
        }
#else
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
#endif

        fill_rhs(r);
        apply_A(x, Ap);

#ifdef __APPLE__
        {
            const double neg_one = -1.0;
            vDSP_vsmaD(Ap.data(), 1, &neg_one, r.data(), 1, r.data(), 1,
                       static_cast<vDSP_Length>(n));
        }
#else
        for (int i = 0; i < n; i++) r[i] -= Ap[i];
#endif

        const double rnorm = std::sqrt(dot(r, r));
        const double rel_res = rnorm / bnorm;

        if (rel_res < tol)
        {
            SolverResult res;
            res.iterations = it;
            res.rel_residual = rel_res;
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
            const double b = beta;
            vDSP_vsmaD(p.data(), 1, &b, Ap.data(), 1, p.data(), 1,
                       static_cast<vDSP_Length>(n));
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
