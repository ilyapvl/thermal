#include "solver.hpp"
#include "multigrid.hpp"
#include "metal_backend.hpp"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <chrono>
#include <vector>
#include <functional>

struct Params
{
    double Lx = 2.0;
    double Ly = 1.0;
    int    Nx = 1025;
    int    Ny = 1025;

    double T_bottom = 0.0;
    double T_right  = 0.0;
    double T_top    = 0.0;
    double T_left   = 0.0;

    int    max_iter = 1000000;
    double tol      = 1e-9;

    std::string method = "mg";
};

struct Grid
{
    int Nx, Ny;
    double Lx, Ly;
    double hx, hy;

    int idx(int i, int j) const { return j * Nx + i; }

    int inner(int i, int j) const
    {
        if (i <= 0 || i >= Nx - 1 || j <= 0 || j >= Ny - 1) return -1;

        return (j - 1) * (Nx - 2) + (i - 1);
    }

    int num_inner() const { return (Nx - 2) * (Ny - 2); }
};

static void assemble(const Grid& g, const Params& p, CSR& A, std::vector<double>& b)
{
    const int Mx = g.Nx - 2;
    const int My = g.Ny - 2;
    const int M  = Mx * My;

    b.assign(M, 0.0);

    const double cx   = 1.0 / (g.hx * g.hx);
    const double cy   = 1.0 / (g.hy * g.hy);
    const double diag = 2.0 * (cx + cy);

    std::vector<int> rp(M + 1, 0);

    #pragma omp parallel for schedule(static) if(M > 100000)
    for (int j = 0; j < My; j++)
    {
        for (int i = 0; i < Mx; i++)
        {
            const int k = j * Mx + i;
            int cnt = 1;
            if (i > 0)      cnt++;
            if (i + 1 < Mx) cnt++;
            if (j > 0)      cnt++;
            if (j + 1 < My) cnt++;
            rp[k + 1] = cnt;
        }
    }
    for (int k = 0; k < M; k++) rp[k + 1] += rp[k];

    const int nnz = rp[M];
    std::vector<int>    ci(nnz);
    std::vector<double> v(nnz);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < My; j++)
    {
        for (int i = 0; i < Mx; i++)
        {
            const int k = j * Mx + i;
            int pos = rp[k];

            if (j > 0)      { ci[pos] = k - Mx; v[pos] = -cy;  pos++; }
            if (i > 0)      { ci[pos] = k - 1;  v[pos] = -cx;  pos++; }
                              ci[pos] = k;      v[pos] = diag; pos++;
            if (i + 1 < Mx) { ci[pos] = k + 1;  v[pos] = -cx;  pos++; }
            if (j + 1 < My) { ci[pos] = k + Mx; v[pos] = -cy;  pos++; }

            double bk = 0.0;
            if (i == 0)      bk += cx * p.T_left;
            if (i + 1 == Mx) bk += cx * p.T_right;
            if (j == 0)      bk += cy * p.T_bottom;
            if (j + 1 == My) bk += cy * p.T_top;
            b[k] = bk;
        }
    }

    A.build_from_sorted(M, std::move(rp), std::move(ci), std::move(v));
}

void write_file(const std::string& path, const Grid& g, const std::vector<double>& T_inner, const Params& p)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    const uint32_t Nx = static_cast<uint32_t>(g.Nx);
    const uint32_t Ny = static_cast<uint32_t>(g.Ny);

    f.write(reinterpret_cast<const char*>(&Nx), 4);
    f.write(reinterpret_cast<const char*>(&Ny), 4);
    f.write(reinterpret_cast<const char*>(&g.Lx), 8);
    f.write(reinterpret_cast<const char*>(&g.Ly), 8);
    f.write(reinterpret_cast<const char*>(&p.T_bottom), 8);
    f.write(reinterpret_cast<const char*>(&p.T_right), 8);
    f.write(reinterpret_cast<const char*>(&p.T_top), 8);
    f.write(reinterpret_cast<const char*>(&p.T_left), 8);

    std::vector<double> Tfull(static_cast<std::size_t>(g.Nx) * g.Ny, 0.0);
    for (int j = 0; j < g.Ny; j++)
    {
        for (int i = 0; i < g.Nx; i++)
        {
            const int k = g.inner(i, j);
            double val;

            if (k >= 0) val = T_inner[k];

            else if (j == 0)
            {
                val = p.T_bottom;
            }

            else if (j == g.Ny - 1)
            {
                val = p.T_top;
            }

            else if (i == 0)
            {
                val = p.T_left;
            }

            else
            {
                val = p.T_right;
            }

            Tfull[g.idx(i, j)] = val;
        }
    }
    f.write(reinterpret_cast<const char*>(Tfull.data()), static_cast<std::streamsize>(Tfull.size() * sizeof(double)));
}




static void laplacian_spmv_stencil(const double* x, double* y, int Mx, int My, double cx, double cy, double diag)
{
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < My; j++)
    {
        for (int i = 0; i < Mx; i++)
        {
            const int k = j * Mx + i;

            double s = diag * x[k];
            if (i > 0) s -= cx * x[k - 1];
            if (i + 1 < Mx) s -= cx * x[k + 1];
            if (j > 0) s -= cy * x[k - Mx];
            if (j + 1 < My) s -= cy * x[k + Mx];

            y[k] = s;
        }
    }
}

int main(int argc, char** argv)
{
    Params p;

    if (argc > 1) p.Nx       = std::max(3, std::atoi(argv[1]));
    if (argc > 2) p.Ny       = std::max(3, std::atoi(argv[2]));
    if (argc > 3) p.T_bottom = std::atof(argv[3]);
    if (argc > 4) p.T_right  = std::atof(argv[4]);
    if (argc > 5) p.T_top    = std::atof(argv[5]);
    if (argc > 6) p.T_left   = std::atof(argv[6]);
    if (argc > 7) p.method   = argv[7];
    if (argc > 8) p.max_iter = std::atoi(argv[8]);
    if (argc > 9) p.tol      = std::atof(argv[9]);

    Grid g;
    g.Nx = p.Nx;
    g.Ny = p.Ny;
    g.Lx = p.Lx;
    g.Ly = p.Ly;
    g.hx = p.Lx / (p.Nx - 1);
    g.hy = p.Ly / (p.Ny - 1);

    CSR A;
    std::vector<double> b;

    assemble(g, p, A, b);

    std::cout << "Grid: "        << g.Nx << " x " << g.Ny << std::endl;
    std::cout << "Inner nodes: " << A.rows() << "\n";
    std::cout << "NNZ: "         << A.nnz()  << "\n";

    metal_backend::Context* metal_ctx = nullptr;

    if (p.method == "mg_gpu")
    {
        try
        {
            metal_ctx = new metal_backend::Context();
            std::cout << "Metal device: " << metal_ctx->device_name() << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "Metal init failed: " << e.what() << "\n";
            delete metal_ctx;
            return 1;
        }
    }

    std::function<void(const std::vector<double>&, std::vector<double>&)> apply_A;


    const int Mx = g.Nx - 2;
    const int My = g.Ny - 2;
    const double cx_d = 1.0 / (g.hx * g.hx);
    const double cy_d = 1.0 / (g.hy * g.hy);
    const double diag_d = 2.0 * (cx_d + cy_d);

    apply_A = [Mx, My, cx_d, cy_d, diag_d](const std::vector<double>& x, std::vector<double>& y)
    {
        if (y.size() != x.size()) y.resize(x.size());
        laplacian_spmv_stencil(x.data(), y.data(), Mx, My, cx_d, cy_d, diag_d);
    };


    MG* mg = nullptr;
    std::function<void(const std::vector<double>&, std::vector<double>&)> apply_M;

    if (p.method == "mg" || p.method == "mg_gpu")
    {
        try
        {
            mg = new MG(metal_ctx, g.Nx, g.Ny, g.Lx, g.Ly, 3, 3, 0.667);
        }
        catch (const std::exception& e)
        {
            std::cerr << "MG init failed: " << e.what() << "\n";
            delete metal_ctx;
            return 1;
        }

        apply_M = [mg](const std::vector<double>& r, std::vector<double>& z)
        {
            mg->apply(r, z);
        };
    }

    std::vector<double> x(A.rows(), 0.0);

    const SolverResult res = cg_solve_generic(apply_A, b, x, apply_M, p.max_iter, p.tol);


    std::cout << "Method:        " << p.method << "\n";
    std::cout << "Iterations:    " << res.iterations    << "\n";
    std::cout << "Rel residual:  " << res.rel_residual  << "\n";
    std::cout << "Converged:     " << (res.converged ? "yes" : "no") << "\n";
    std::cout << "Time:          " << res.seconds       << " s\n";

    write_file("field.bin", g, x, p);
    std::cout << "Written: field.bin\n";

    delete mg;
    delete metal_ctx;

    return 0;
}
