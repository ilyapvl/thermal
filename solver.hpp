#ifndef SOLVER_HPP
#define SOLVER_HPP


#include <vector>
#include <functional>
#include <cstddef>
#include <cstdint>


class CSR
{
public:
    CSR() = default;
    explicit CSR(int n)
    {
        resize(n);
    }

    CSR(int n,
        const std::vector<int>&    rows,
        const std::vector<int>&    cols,
        const std::vector<double>& vals);

    int rows() const noexcept { return n_; }
    int cols() const noexcept { return n_; }
    std::size_t nnz() const noexcept { return values_.size(); }

    void clear();
    void resize(int n);
    void reserve(std::size_t nnz_hint);
    void sort_rows();

    // y = A*x
    void spmv(const double* x, double* y) const;
    void spmv(const std::vector<double>& x, std::vector<double>& y) const;


private:
    int n_ = 0;
    std::vector<int>    row_ptr_;
    std::vector<int>    col_idx_;
    std::vector<double> values_;
};


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

SolverResult cg_solve(const CSR& A,
                    const std::vector<double>& b,
                    std::vector<double>& x,
                    int max_iter,
                    double tol,
                    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_M);
#endif // SOLVER_HPP
