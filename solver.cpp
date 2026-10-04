#ifndef SOLVER_CPP
#define SOLVER_CPP

#include "solver.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif


CSR::CSR(int n,
         const std::vector<int>&    rows,
         const std::vector<int>&    cols,
         const std::vector<double>& vals)
{
    if (rows.size() != cols.size() || rows.size() != vals.size())
    {
        throw std::invalid_argument("CSR: rows/cols/vals different sizes");
    }

    n_ = n;
    row_ptr_.assign(n + 1, 0);

    for (int r : rows)
    {
        if (r < 0 || r >= n) throw std::out_of_range("CSR: row index out of range");

        ++row_ptr_[r + 1];
    }

    for (int i = 0; i < n; ++i)
    {
        row_ptr_[i + 1] += row_ptr_[i];
    }

    col_idx_.assign(vals.size(), 0);
    values_.assign(vals.size(), 0.0);

    std::vector<int> pos = row_ptr_;

    for (std::size_t k = 0; k < vals.size(); k++)
    {
        const int r = rows[k];
        if (cols[k] < 0 || cols[k] >= n)
        {
            throw std::out_of_range("CSR: col index out of range");
        }

        const int p = pos[r]++;

        col_idx_[p] = cols[k];
        values_ [p] = vals[k];
    }

    sort_rows();

    std::vector<int>    new_col;
    std::vector<double> new_val;
    new_col.reserve(col_idx_.size());
    new_val.reserve(values_.size());

    int write = 0;
    for (int i = 0; i < n; i++)
    {
        const int lo = row_ptr_[i], hi = row_ptr_[i + 1];

        int p = lo;

        while (p < hi)
        {
            const int c = col_idx_[p];
            double v = values_[p];
            int q = p + 1;
            while (q < hi && col_idx_[q] == c)
            {
                v += values_[q];
                ++q;
            }

            col_idx_[write] = c;
            values_ [write] = v;

            write++;
            p = q;
        }

        row_ptr_[i + 1] = write;
    }

    col_idx_.resize(write);
    values_ .resize(write);
}


void CSR::clear()
{
    n_ = 0;
    row_ptr_.clear();
    col_idx_.clear();
    values_.clear();
}

void CSR::resize(int n)
{
    n_ = n;
    row_ptr_.assign(n + 1, 0);
    col_idx_.clear();
    values_.clear();
}

void CSR::reserve(std::size_t nnz_hint)
{
    col_idx_.reserve(nnz_hint);
    values_ .reserve(nnz_hint);
}

void CSR::sort_rows()
{
    for (int i = 0; i < n_; ++i)
    {
        const int lo = row_ptr_[i];
        const int hi = row_ptr_[i + 1];

        if (hi - lo <= 1) continue;

        std::vector<int> idx(hi - lo);
        std::iota(idx.begin(), idx.end(), lo);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return col_idx_[a] < col_idx_[b]; });

        std::vector<int> new_col(hi - lo);
        std::vector<double> new_val(hi - lo);

        for (int k = 0; k < hi - lo; k++)
        {
            new_col[k] = col_idx_[idx[k]];
            new_val[k] = values_ [idx[k]];
        }

        for (int k = 0; k < hi - lo; k++)
        {
            col_idx_[lo + k] = new_col[k];
            values_ [lo + k] = new_val[k];
        }
    }
}


void CSR::build_from_sorted(int n,
                            std::vector<int>    row_ptr,
                            std::vector<int>    col_idx,
                            std::vector<double> values)
{
    n_ = n;
    row_ptr_ = std::move(row_ptr);
    col_idx_ = std::move(col_idx);
    values_  = std::move(values);
}





void CSR::spmv(const double* x, double* y) const
{
    for (int i = 0; i < n_; ++i)
    {
        double s = 0.0;

        for (int k = row_ptr_[i]; k < row_ptr_[i + 1]; k++)
        {
            s += values_[k] * x[col_idx_[k]];
        }

        y[i] = s;
    }
}

void CSR::spmv(const std::vector<double>& x, std::vector<double>& y) const
{
    if (y.size() != n_) y.resize(n_);

    spmv(x.data(), y.data());
}

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



SolverResult cg_solve(const CSR& A,
                    const std::vector<double>& b,
                    std::vector<double>& x,
                    int max_iter,
                    double tol,
                    const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_M)
{
    auto apply_A = [&A](const std::vector<double>& xx, std::vector<double>& yy) {
        A.spmv(xx, yy);
    };
    return cg_solve_generic(apply_A, b, x, apply_M, max_iter, tol);
}



static inline int A_rows_from_b(const std::vector<double>& b) {
    return static_cast<int>(b.size());
}











SolverResult cg_solve_generic(const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_A,
                            const std::vector<double>& b,
                            std::vector<double>& x,
                            const std::function<void(const std::vector<double>&, std::vector<double>&)>& apply_M,
                            int max_iter,
                            double tol)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const int n = A_rows_from_b(b);
    if ((int)x.size() != n) x.assign(n, 0.0);

    std::vector<double> r(n), z(n), p(n), Ap(n);

    apply_A(x, r);
    #ifdef __APPLE__
    {
        const double neg_one = -1.0;
        vDSP_vsmaD(r.data(), 1, &neg_one, b.data(), 1, r.data(), 1, static_cast<vDSP_Length>(n));
    }

    #else
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    #endif

    const double bnorm = std::sqrt(std::max(dot(b, b), 1e-300));

    if (apply_M) apply_M(r, z);
    else z = r;

    p = z;
    double rz = dot(r, z);

    for (int it = 1; it <= max_iter; it++)
    {
        apply_A(p, Ap);
        const double pAp = dot(p, Ap);

        if (pAp <= 0.0)
        {
            SolverResult res;
            res.iterations   = it - 1;
            res.rel_residual = std::sqrt(dot(r, r)) / bnorm;
            res.converged    = false;
            res.seconds      = std::chrono::duration<double>(clock::now() - t0).count();
            return res;
        }

        const double alpha = rz / pAp;

        #ifdef __APPLE__

        {
            const double pos_a =  alpha;
            const double neg_a = -alpha;
            vDSP_vsmaD(p.data(),  1, &pos_a, x.data(), 1, x.data(), 1, static_cast<vDSP_Length>(n));
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
            res.iterations   = it;
            res.rel_residual = rnorm / bnorm;
            res.converged    = true;
            res.seconds      = std::chrono::duration<double>(clock::now() - t0).count();

            return res;
        }

        if (apply_M) apply_M(r, z);
        else z = r;

        const double rz_new = dot(r, z);
        const double beta = rz_new / rz;

        #ifdef __APPLE__

        {
            const double bb = beta;
            vDSP_vsmaD(p.data(), 1, &bb, z.data(), 1, p.data(), 1, static_cast<vDSP_Length>(n));
        }

        #else
        for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        #endif

        rz = rz_new;
    }

    SolverResult res;
    res.iterations   = max_iter;
    res.rel_residual = std::sqrt(dot(r, r)) / bnorm;
    res.converged    = false;
    res.seconds      = std::chrono::duration<double>(clock::now() - t0).count();

    return res;
}


#endif // SOLVER_CPP
