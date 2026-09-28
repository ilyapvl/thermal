#include "solver.hpp"


#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <chrono>
#include <vector>

struct Params
{
    double Lx = 1.0;
    double Ly = 1.0;
    int    Nx = 1025;
    int    Ny = 1025;

    double T_bottom = 0.0;
    double T_right  = 0.0;
    double T_top    = 0.0;
    double T_left   = 0.0;

    int max_iter = 1000000;
    double tol      = 1e-9;
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
    const int M = g.num_inner();
    b.assign(M, 0.0);

    const double cx   = 1.0 / (g.hx * g.hx);
    const double cy   = 1.0 / (g.hy * g.hy);
    const double diag = 2.0 * (cx + cy);

    auto wall_T = [&](int i, int j) -> double
    {
        if (j == 0)            return p.T_bottom;
        if (j == g.Ny - 1)     return p.T_top;
        if (i == 0)            return p.T_left;
        if (i == g.Nx - 1)     return p.T_right;

        throw std::runtime_error("invalid wall node");
    };

    std::vector<int> rows, cols;
    std::vector<double> vals;

    rows.reserve(M * 5);
    cols.reserve(M * 5);
    vals.reserve(M * 5);

    for (int j = 1; j < g.Ny - 1; ++j)
    {
        for (int i = 1; i < g.Nx - 1; ++i)
        {
            const int k = g.inner(i, j);

            rows.push_back(k); cols.push_back(k); vals.push_back(diag);

            const int di[4] = {-1, +1,  0,  0};
            const int dj[4] = { 0,  0, -1, +1};
            const double c[4] = {cx, cx, cy, cy};

            for (int s = 0; s < 4; ++s)
            {
                const int ni = i + di[s];
                const int nj = j + dj[s];
                const int nk = g.inner(ni, nj);

                if (nk >= 0)
                {
                    rows.push_back(k);
                    cols.push_back(nk);
                    vals.push_back(-c[s]);
                }
                
                else
                {
                    b[k] += c[s] * wall_T(ni, nj);
                }
            }
        }
    }

    A = CSR(M, rows, cols, vals);
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

int main(int argc, char** argv)
{
    Params p;

    if (argc > 1) p.Nx       = std::max(3, std::atoi(argv[1]));
    if (argc > 2) p.Ny       = std::max(3, std::atoi(argv[2]));
    if (argc > 3) p.T_bottom = std::atof(argv[3]);
    if (argc > 4) p.T_right  = std::atof(argv[4]);
    if (argc > 5) p.T_top    = std::atof(argv[5]);
    if (argc > 6) p.T_left   = std::atof(argv[6]);
    if (argc > 7) p.max_iter = std::atoi(argv[7]);
    if (argc > 8) p.tol      = std::atof(argv[8]);

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


    std::vector<double> x(A.rows(), 0.0);
    const SolverResult res = cg_solve(A, b, x, p.max_iter, p.tol);

    std::cout << "Iterations:    " << res.iterations << "\n";
    std::cout << "Rel residual:  " << res.rel_residual << "\n";
    std::cout << "Converged:     " << (res.converged ? "yes" : "no") << "\n";
    std::cout << "Time:          " << res.seconds << " s\n";

    write_file("field.bin", g, x, p);

    return 0;
}
