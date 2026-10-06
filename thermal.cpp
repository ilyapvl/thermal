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
    double Lx = 6.0, Ly = 6.0, Lz = 3.0;
    int Nx = 65, Ny = 65, Nz = 65;

    double T_xmin = 0.0, T_xmax = 0.0;
    double T_ymin = 0.0, T_ymax = 0.0;
    double T_zmin = 0.0, T_zmax = 0.0;

    int max_iter = 1000000;
    double tol = 1e-6;
    std::string method = "mg";
};

struct Grid
{
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    double hx, hy, hz;

    int idx(int i, int j, int k) const { return (k * Ny + j) * Nx + i; }

    int inner(int i, int j, int k) const
    {
        if (i <= 0 || i >= Nx - 1 ||
            j <= 0 || j >= Ny - 1 ||
            k <= 0 || k >= Nz - 1) return -1;
        return (k - 1) * (Ny - 2) * (Nx - 2) + (j - 1) * (Nx - 2) + (i - 1);
    }

    int num_inner() const { return (Nx - 2) * (Ny - 2) * (Nz - 2); }
};

void write_file(const std::string& path, const Grid& g, const std::vector<double>& T_inner, const Params& p)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    const uint32_t Nx = static_cast<uint32_t>(g.Nx);
    const uint32_t Ny = static_cast<uint32_t>(g.Ny);
    const uint32_t Nz = static_cast<uint32_t>(g.Nz);

    f.write(reinterpret_cast<const char*>(&Nx), 4);
    f.write(reinterpret_cast<const char*>(&Ny), 4);
    f.write(reinterpret_cast<const char*>(&Nz), 4);
    f.write(reinterpret_cast<const char*>(&g.Lx), 8);
    f.write(reinterpret_cast<const char*>(&g.Ly), 8);
    f.write(reinterpret_cast<const char*>(&g.Lz), 8);
    f.write(reinterpret_cast<const char*>(&p.T_xmin), 8);
    f.write(reinterpret_cast<const char*>(&p.T_xmax), 8);
    f.write(reinterpret_cast<const char*>(&p.T_ymin), 8);
    f.write(reinterpret_cast<const char*>(&p.T_ymax), 8);
    f.write(reinterpret_cast<const char*>(&p.T_zmin), 8);
    f.write(reinterpret_cast<const char*>(&p.T_zmax), 8);

    std::vector<double> layer(static_cast<std::size_t>(g.Nx) * g.Ny);

    for (int k = 0; k < g.Nz; k++)
    {
        for (int j = 0; j < g.Ny; j++)
        {
            for (int i = 0; i < g.Nx; i++)
            {
                const int ki = g.inner(i, j, k);
                double val;

                if (ki >= 0) val = T_inner[ki];
                else if (i == 0) val = p.T_xmin;
                else if (i == g.Nx - 1) val = p.T_xmax;
                else if (j == 0) val = p.T_ymin;
                else if (j == g.Ny - 1) val = p.T_ymax;
                else if (k == 0) val = p.T_zmin;
                else val = p.T_zmax;

                layer[j * g.Nx + i] = val;
            }
        }

        f.write(reinterpret_cast<const char*>(layer.data()),
                static_cast<std::streamsize>(layer.size() * sizeof(double)));
    }
}



static void spmv(const double* x, double* y,
                                      int Mx, int My, int Mz,
                                      double cx, double cy, double cz, double diag)
{
    const long long Mxy = static_cast<long long>(Mx) * My;

    #pragma omp parallel for schedule(static) if(static_cast<long long>(Mx) * My * Mz > 100000)
    for (int l = 0; l < Mz; l++)
    {
        for (int j = 0; j < My; j++)
        {
            for (int i = 0; i < Mx; i++)
            {
                const long long k = (static_cast<long long>(l) * My + j) * Mx + i;

                double s = diag * x[k];
                if (i > 0) s -= cx * x[k - 1];
                if (i + 1 < Mx) s -= cx * x[k + 1];
                if (j > 0) s -= cy * x[k - Mx];
                if (j + 1 < My) s -= cy * x[k + Mx];
                if (l > 0) s -= cz * x[k - Mxy];
                if (l + 1 < Mz) s -= cz * x[k + Mxy];

                y[k] = s;
            }
        }
    }
}


static void fill_rhs_boundary(std::vector<double>& r,
                                 int Mx, int My, int Mz,
                                 double cx, double cy, double cz,
                                 const Params& p)
{
    for (int l = 0; l < Mz; l++)
    {
        for (int j = 0; j < My; j++)
        {
            for (int i = 0; i < Mx; i++)
            {
                double v = 0.0;

                if (i == 0) v += cx * p.T_xmin;
                if (i + 1 == Mx) v += cx * p.T_xmax;
                if (j == 0) v += cy * p.T_ymin;
                if (j + 1 == My) v += cy * p.T_ymax;
                if (l == 0) v += cz * p.T_zmin;
                if (l + 1 == Mz) v += cz * p.T_zmax;

                if (v != 0.0)
                {
                    const long long k = (static_cast<long long>(l) * My + j) * Mx + i;
                    r[k] = v;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    Params p;

    if (argc > 1) p.Nx = std::max(3, std::atoi(argv[1]));
    if (argc > 2) p.Ny = std::max(3, std::atoi(argv[2]));
    if (argc > 3) p.Nz = std::max(3, std::atoi(argv[3]));
    if (argc > 4) p.T_xmin = std::atof(argv[4]);
    if (argc > 5) p.T_xmax = std::atof(argv[5]);
    if (argc > 6) p.T_ymin = std::atof(argv[6]);
    if (argc > 7) p.T_ymax = std::atof(argv[7]);
    if (argc > 8) p.T_zmin = std::atof(argv[8]);
    if (argc > 9) p.T_zmax = std::atof(argv[9]);
    if (argc > 10) p.method = argv[10];
    if (argc > 11) p.max_iter = std::atoi(argv[11]);
    if (argc > 12) p.tol = std::atof(argv[12]);
    if (argc > 13) p.Lx = std::atof(argv[13]);
    if (argc > 14) p.Ly = std::atof(argv[14]);
    if (argc > 15) p.Lz = std::atof(argv[15]);

    Grid g;
    g.Nx = p.Nx;
    g.Ny = p.Ny;
    g.Nz = p.Nz;
    g.Lx = p.Lx;
    g.Ly = p.Ly;
    g.Lz = p.Lz;
    g.hx = p.Lx / (p.Nx - 1);
    g.hy = p.Ly / (p.Ny - 1);
    g.hz = p.Lz / (p.Nz - 1);

    const int Mx = g.Nx - 2, My = g.Ny - 2, Mz = g.Nz - 2;
    const long long M = static_cast<long long>(Mx) * My * Mz;

    const double cx_d = 1.0 / (g.hx * g.hx);
    const double cy_d = 1.0 / (g.hy * g.hy);
    const double cz_d = 1.0 / (g.hz * g.hz);
    const double diag_d = 2.0 * (cx_d + cy_d + cz_d);

    std::function<void(std::vector<double>&)> fill_rhs =
        [Mx, My, Mz, cx_d, cy_d, cz_d, &p](std::vector<double>& r)
    {
        std::fill(r.begin(), r.end(), 0.0);
        fill_rhs_boundary(r, Mx, My, Mz, cx_d, cy_d, cz_d, p);
    };

    double bnorm_sq = 0.0;
    {
        std::vector<double> btmp(M, 0.0);
        fill_rhs(btmp);
        for (long long i = 0; i < M; i++) bnorm_sq += btmp[i] * btmp[i];
    }

    std::cout << "Grid: " << g.Nx << " x " << g.Ny << " x " << g.Nz << std::endl;
    std::cout << "Inner nodes: " << M << "\n";
    std::cout << "NNZ: " << (7LL * M - 2LL * Mx * My - 2LL * Mx * Mz - 2LL * My * Mz) << "\n";

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

    std::function<void(const std::vector<double>&, std::vector<double>&)> apply_A =
        [Mx, My, Mz, cx_d, cy_d, cz_d, diag_d](const std::vector<double>& x, std::vector<double>& y)
    {
        if (y.size() != x.size()) y.resize(x.size());
        spmv(x.data(), y.data(), Mx, My, Mz, cx_d, cy_d, cz_d, diag_d);
    };

    MG* mg = nullptr;
    std::function<void(const std::vector<double>&, std::vector<double>&)> apply_M;

    if (p.method == "mg" || p.method == "mg_gpu")
    {
        try
        {
            mg = new MG(metal_ctx, g.Nx, g.Ny, g.Nz, g.Lx, g.Ly, g.Lz, 3, 3, 0.667);
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

    std::vector<double> x(M, 0.0);

    const SolverResult res = cg_solve(apply_A, bnorm_sq, fill_rhs, x, apply_M, p.max_iter, p.tol);

    std::cout << "Method:        " << p.method << "\n";
    std::cout << "Iterations:    " << res.iterations << "\n";
    std::cout << "Rel residual:  " << res.rel_residual << "\n";
    std::cout << "Converged:     " << (res.converged ? "yes" : "no") << "\n";
    std::cout << "Time:          " << res.seconds << " s\n";

    write_file("field.bin", g, x, p);
    std::cout << "Written: field.bin\n";

    delete mg;
    delete metal_ctx;

    return 0;
}
