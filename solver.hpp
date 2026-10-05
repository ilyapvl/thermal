#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <vector>
#include <functional>
#include <cstddef>

struct SolverResult
{
    int iterations = 0;
    double rel_residual = 0.0;
    bool converged = false;
    double seconds = 0.0;
};

double dot(const float* a, const float* b, int n);

inline double dot(const std::vector<float>& a, const std::vector<float>& b)
{
    return dot(a.data(), b.data(), static_cast<int>(a.size()));
}

SolverResult cg_solve_generic(
    const std::function<void(const std::vector<float>&, std::vector<float>&)>& apply_A,
    double bnorm_sq,
    const std::function<void(std::vector<float>&)>& fill_rhs,
    std::vector<float>& x,
    const std::function<void(const std::vector<float>&, std::vector<float>&)>& apply_M,
    int max_iter,
    double tol);



#endif // SOLVER_HPP
