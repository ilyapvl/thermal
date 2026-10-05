#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <functional>

struct SolverResult
{
    int    iterations   = 0;
    double rel_residual = 0.0;
    bool   converged    = false;
    double seconds      = 0.0;
};

double dot(const double* a, const double* b, int n);
inline double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    return dot(a.data(), b.data(), static_cast<int>(a.size()));
}

SolverResult cg_solve_generic(
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_A,
    double bnorm_sq,
    const std::function<void(std::vector<double>&)>& fill_rhs,
    std::vector<double>& x,
    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_M,
    int max_iter,
    double tol);



#endif // SOLVER_HPP
