#include "multigrid.hpp"
#include <cmath>
#include <algorithm>

MG::MG(int Nx, int Ny, double Lx, double Ly, int nu1, int nu2, double omega) : nu1_(nu1), nu2_(nu2), omega_(omega)
{
    auto is_pow2 = [](int n){ return n > 0 && (n & (n-1)) == 0; };
    if (!is_pow2(Nx - 1) || !is_pow2(Ny - 1))
        throw std::runtime_error("MG: Nx-1 and Ny-1 != n^2 + 1");

    int Nxi = Nx, Nyi = Ny;
    double hx = Lx / (Nx - 1);
    double hy = Ly / (Ny - 1);

    while (true)
    {
        Level L;

        L.Nx = Nxi;
        L.Ny = Nyi;
        L.hx = hx;
        L.hy = hy;

        L.cx = 1.0 / (hx * hx);
        L.cy = 1.0 / (hy * hy);

        L.diag = 2.0 * (L.cx + L.cy);

        const std::size_t sz = (std::size_t)L.Nx * L.Ny;

        L.u.assign(sz, 0.0);
        L.f.assign(sz, 0.0);
        L.r.assign(sz, 0.0);

        levels_.push_back(std::move(L));

        if (Nxi <= 3 || Nyi <= 3) break;

        Nxi = (Nxi + 1) / 2;
        Nyi = (Nyi + 1) / 2;

        hx *= 2.0;
        hy *= 2.0;
    }
}


void MG::smooth(int lvl, int nu, bool forward)
{
    Level& L = levels_[lvl];

    const int Nx = L.Nx, Ny = L.Ny;
    const double cx = L.cx, cy = L.cy, diag = L.diag;
    const double inv_diag = 1.0 / diag;
    const double w = omega_;

    for (int s = 0; s < nu; s++)
    {
        if (forward)
        {
            for (int j = 1; j < Ny - 1; j++)
            {
                for (int i = 1; i < Nx - 1; i++)
                {
                    const int k = idx(i, j, Nx);
                    const double Au = diag * L.u[k]
                        - cx * (L.u[k - 1] + L.u[k + 1])
                        - cy * (L.u[k - Nx] + L.u[k + Nx]);

                    L.u[k] += w * (L.f[k] - Au) * inv_diag;
                }
            }
        }
        
        else // backward
        {
            for (int j = Ny - 2; j >= 1; j--)
            {
                for (int i = Nx - 2; i >= 1; i--)
                {
                    const int k = idx(i, j, Nx);
                    const double Au = diag * L.u[k]
                        - cx * (L.u[k - 1] + L.u[k + 1])
                        - cy * (L.u[k - Nx] + L.u[k + Nx]);

                    L.u[k] += w * (L.f[k] - Au) * inv_diag;
                }
            }
        }
    }
}

void MG::compute_residual(int lvl)
{
    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny;
    const double cx = L.cx, cy = L.cy, diag = L.diag;

    std::fill(L.r.begin(), L.r.end(), 0.0);

    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; ++i)
        {
            const int k = idx(i, j, Nx);
            const double Au = diag * L.u[k] - cx * (L.u[k - 1] + L.u[k + 1]) - cy * (L.u[k - Nx] + L.u[k + Nx]);
            L.r[k] = L.f[k] - Au;
        }
    }
}


void MG::conv2d_restrict(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    const int Nxf = F.Nx, Nxc = C.Nx;
    const int Nyf = F.Ny, Nyc = C.Ny;

    std::fill(C.f.begin(), C.f.end(), 0.0);
    for (int jc = 1; jc < Nyc - 1; jc++)
    {
        for (int ic = 1; ic < Nxc - 1; ic++)
        {
            const int fi = 2 * ic, fj = 2 * jc;

            const double v =
                1.0 * F.r[idx(fi-1, fj-1, Nxf)] +
                2.0 * F.r[idx(fi,   fj-1, Nxf)] +
                1.0 * F.r[idx(fi+1, fj-1, Nxf)] +
                2.0 * F.r[idx(fi-1, fj,   Nxf)] +
                4.0 * F.r[idx(fi,   fj,   Nxf)] +
                2.0 * F.r[idx(fi+1, fj,   Nxf)] +
                1.0 * F.r[idx(fi-1, fj+1, Nxf)] +
                2.0 * F.r[idx(fi,   fj+1, Nxf)] +
                1.0 * F.r[idx(fi+1, fj+1, Nxf)];

            C.f[idx(ic, jc, Nxc)] = v / 16.0;
        }
    }
}


void MG::interpolate_reverse(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    const int Nxf = F.Nx, Nxc = C.Nx;
    const int Nyf = F.Ny, Nyc = C.Ny;

    // even even
    for (int jc = 0; jc < Nyc; jc++)
        for (int ic = 0; ic < Nxc; ic++)
            F.u[idx(2 * ic, 2 * jc, Nxf)] += C.u[idx(ic, jc, Nxc)];

    // odd even
    for (int jc = 0; jc < Nyc; jc++)
        for (int ic = 0; ic + 1 < Nxc; ic++)
        {
            const double v = 0.5 * (C.u[idx(ic, jc, Nxc)] + C.u[idx(ic + 1, jc, Nxc)]);
            F.u[idx(2 * ic + 1, 2 * jc, Nxf)] += v;
        }

    // even odd
    for (int jc = 0; jc + 1 < Nyc; jc++)
        for (int ic = 0; ic < Nxc; ic++)
        {
            const double v = 0.5 * (C.u[idx(ic, jc, Nxc)] + C.u[idx(ic, jc + 1, Nxc)]);
            F.u[idx(2 * ic, 2 * jc + 1, Nxf)] += v;
        }

    // odd odd
    for (int jc = 0; jc + 1 < Nyc; jc++)
        for (int ic = 0; ic + 1 < Nxc; ic++)
        {
            const double v = 0.25 * (
                C.u[idx(ic, jc, Nxc)] + C.u[idx(ic + 1, jc, Nxc)] +
                C.u[idx(ic, jc + 1, Nxc)] + C.u[idx(ic + 1, jc + 1, Nxc)]);
            F.u[idx(2 * ic + 1, 2 * jc + 1, Nxf)] += v;
        }
}


void MG::coarse_solve(int lvl)
{
    static constexpr int num_iter = 100;

    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny;
    const double cx = L.cx, cy = L.cy, diag = L.diag;

    for (int s = 0; s < num_iter; s++)
    {
        for (int j = 1; j < Ny - 1; j++)
        {
            for (int i = 1; i < Nx - 1; i++)
            {
                const int k = idx(i, j, Nx);
                const double rhs = L.f[k]
                        + cx * (L.u[k-1] + L.u[k+1])
                        + cy * (L.u[k-Nx] + L.u[k+Nx]);

                L.u[k] = rhs / diag;
            }
        }
    }
}


void MG::v_cycle(int lvl)
{
    if (lvl == (int)levels_.size() - 1)
    {
        coarse_solve(lvl);
        return;
    }

    smooth(lvl, nu1_, true);

    compute_residual(lvl);

    conv2d_restrict(lvl);

    std::fill(levels_[lvl + 1].u.begin(), levels_[lvl + 1].u.end(), 0.0);

    v_cycle(lvl + 1);

    interpolate_reverse(lvl);

    smooth(lvl, nu2_, false);
}

// r -> z
void MG::apply(const std::vector<double>& r_fine,
                std::vector<double>& z_fine)
{
    Level& L0 = levels_.front();
    const int Nx = L0.Nx, Ny = L0.Ny;

    for (auto& L : levels_)
    {
        std::fill(L.u.begin(), L.u.end(), 0.0);
        std::fill(L.f.begin(), L.f.end(), 0.0);
    }

    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; i++)
        {
            const int kf = (j - 1) * (Nx - 2) + (i - 1);

            L0.f[idx(i, j, Nx)] = r_fine[kf];
        }
    }

    v_cycle(0);

    if (z_fine.size() != (Nx - 2) * (Ny - 2)) z_fine.resize((Nx - 2) * (Ny - 2));

    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; i++)
        {
            const int kf = (j - 1) * (Nx - 2) + (i - 1);

            z_fine[kf] = L0.u[idx(i, j, Nx)];
        }
    }
}
