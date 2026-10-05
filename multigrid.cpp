#include "multigrid.hpp"
#include <cmath>
#include <algorithm>

MG::MG(metal_backend::Context* ctx,
       int Nx, int Ny, int Nz,
       float Lx, float Ly, float Lz,
       int nu1, int nu2, float omega)
    : nu1_(nu1), nu2_(nu2), omega_(omega), ctx_(ctx)
{
    auto is_pow2 = [](int n){ return n > 0 && (n & (n-1)) == 0; };
    if (!is_pow2(Nx - 1) || !is_pow2(Ny - 1) || !is_pow2(Nz - 1))
        throw std::runtime_error("MG: Nx-1, Ny-1, Nz-1 must be powers of 2");

    enable_omp_ = (static_cast<long long>(Nx) * Ny * Nz) > 200000;
    enable_gpu_ = (ctx_ != nullptr) && ctx_->available();

    int Nxi = Nx, Nyi = Ny, Nzi = Nz;
    float hx = Lx / (Nx - 1);
    float hy = Ly / (Ny - 1);
    float hz = Lz / (Nz - 1);

    while (true)
    {
        Level L;
        L.Nx = Nxi; L.Ny = Nyi; L.Nz = Nzi;
        L.hx = hx; L.hy = hy; L.hz = hz;
        L.cx = 1.0 / (hx * hx);
        L.cy = 1.0 / (hy * hy);
        L.cz = 1.0 / (hz * hz);
        L.diag = 2.0 * (L.cx + L.cy + L.cz);

        const std::size_t sz = static_cast<std::size_t>(L.Nx) * L.Ny * L.Nz;
        L.u.resize(sz);
        L.u_new.resize(sz);
        L.f.resize(sz);
        L.u.fill(0.0f);
        L.u_new.fill(0.0f);
        L.f.fill(0.0f);

        levels_.push_back(std::move(L));

        if (Nxi <= 3 || Nyi <= 3 || Nzi <= 3) break;

        Nxi = (Nxi + 1) / 2;
        Nyi = (Nyi + 1) / 2;
        Nzi = (Nzi + 1) / 2;
        hx *= 2.0; hy *= 2.0; hz *= 2.0;
    }
}

void MG::smooth(int lvl, int nu)
{
    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz;

    if (enable_gpu_ && (Nx * Ny * Nz) > gpu_threshold_)
    {
        metal_backend::Context::JacobiParams jp { Nx, Ny, Nz, L.hx, L.hy, L.hz, omega_ };
        ctx_->jacobi_smooth_device(L.u, L.f, L.u_new, jp, nu);
        if (nu % 2 != 0) std::swap(L.u, L.u_new);
        return;
    }

    const float cx = L.cx, cy = L.cy, cz = L.cz, diag = L.diag;
    const float inv_diag = 1.0 / diag;
    const float w = omega_;
    const bool use_omp = enable_omp_ && (Nx * Ny * Nz) > 100000;

    const float* f_ptr = L.f.data();

    bool u_is_current = true;
    float* u_cur = L.u.data();
    float* u_alt = L.u_new.data();

    #pragma omp parallel if(use_omp)
    {
        for (int s = 0; s < nu; s++)
        {
            const float* src = u_is_current ? u_cur : u_alt;
            float* dst = u_is_current ? u_alt : u_cur;

            #pragma omp for schedule(static) nowait
            for (int l = 1; l < Nz - 1; l++)
            {
                for (int j = 1; j < Ny - 1; j++)
                {
                    for (int i = 1; i < Nx - 1; i++)
                    {
                        const int k = idx(i, j, l, Nx, Ny);
                        const float Au = diag * src[k]
                            - cx * (src[k - 1] + src[k + 1])
                            - cy * (src[k - Nx] + src[k + Nx])
                            - cz * (src[k - Nx * Ny] + src[k + Nx * Ny]);
                        dst[k] = src[k] + w * (f_ptr[k] - Au) * inv_diag;
                    }
                }
            }

            #pragma omp barrier
            #pragma omp single
            { u_is_current = !u_is_current; }
        }
    }
    if (!u_is_current) std::swap(L.u, L.u_new);
}


void MG::interpolate_reverse(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    const int Nxf = F.Nx, Nyf = F.Ny;
    const int Nxc = C.Nx, Nyc = C.Ny, Nzc = C.Nz;

    const bool use_omp = enable_omp_ && (Nxc * Nyc * Nzc) > 50000;

    const float* cu_ptr = C.u.data();
    float* fu_ptr = F.u.data();

    auto C_at = [&](int cx, int cy, int cz) -> float {
        return cu_ptr[(cz * Nyc + cy) * Nxc + cx];
    };

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc; lc++)
    {
        for (int jc = 0; jc < Nyc; jc++)
        {
            for (int ic = 0; ic < Nxc; ic++)
            {
                fu_ptr[idx(2 * ic, 2 * jc, 2 * lc, Nxf, Nyf)] += C_at(ic, jc, lc);
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc; lc++)
    {
        for (int jc = 0; jc < Nyc; jc++)
        {
            for (int ic = 0; ic < Nxc - 1; ic++)
            {
                const float v = 0.5f * (C_at(ic, jc, lc) + C_at(ic + 1, jc, lc));
                fu_ptr[idx(2 * ic + 1, 2 * jc, 2 * lc, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc; lc++)
    {
        for (int jc = 0; jc < Nyc - 1; jc++)
        {
            for (int ic = 0; ic < Nxc; ic++)
            {
                const float v = 0.5f * (C_at(ic, jc, lc) + C_at(ic, jc + 1, lc));
                fu_ptr[idx(2 * ic, 2 * jc + 1, 2 * lc, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc; lc++)
    {
        for (int jc = 0; jc < Nyc - 1; jc++)
        {
            for (int ic = 0; ic < Nxc - 1; ic++)
            {
                const float v = 0.25f * (C_at(ic, jc, lc) + C_at(ic + 1, jc, lc)
                                       + C_at(ic, jc + 1, lc) + C_at(ic + 1, jc + 1, lc));
                fu_ptr[idx(2 * ic + 1, 2 * jc + 1, 2 * lc, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc - 1; lc++)
    {
        for (int jc = 0; jc < Nyc; jc++)
        {
            for (int ic = 0; ic < Nxc; ic++)
            {
                const float v = 0.5f * (C_at(ic, jc, lc) + C_at(ic, jc, lc + 1));
                fu_ptr[idx(2 * ic, 2 * jc, 2 * lc + 1, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc - 1; lc++)
    {
        for (int jc = 0; jc < Nyc; jc++)
        {
            for (int ic = 0; ic < Nxc - 1; ic++)
            {
                const float v = 0.25f * (C_at(ic, jc, lc) + C_at(ic + 1, jc, lc)
                                       + C_at(ic, jc, lc + 1) + C_at(ic + 1, jc, lc + 1));
                fu_ptr[idx(2 * ic + 1, 2 * jc, 2 * lc + 1, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc - 1; lc++)
    {
        for (int jc = 0; jc < Nyc - 1; jc++)
        {
            for (int ic = 0; ic < Nxc; ic++)
            {
                const float v = 0.25f * (C_at(ic, jc, lc) + C_at(ic, jc + 1, lc)
                                       + C_at(ic, jc, lc + 1) + C_at(ic, jc + 1, lc + 1));
                fu_ptr[idx(2 * ic, 2 * jc + 1, 2 * lc + 1, Nxf, Nyf)] += v;
            }
        }
    }

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 0; lc < Nzc - 1; lc++)
    {
        for (int jc = 0; jc < Nyc - 1; jc++)
        {
            for (int ic = 0; ic < Nxc - 1; ic++)
            {
                const float v = 0.125f * (C_at(ic, jc, lc) + C_at(ic + 1, jc, lc)
                                        + C_at(ic, jc + 1, lc) + C_at(ic + 1, jc + 1, lc)
                                        + C_at(ic, jc, lc + 1) + C_at(ic + 1, jc, lc + 1)
                                        + C_at(ic, jc + 1, lc + 1) + C_at(ic + 1, jc + 1, lc + 1));
                fu_ptr[idx(2 * ic + 1, 2 * jc + 1, 2 * lc + 1, Nxf, Nyf)] += v;
            }
        }
    }
}

void MG::conv3d_restrict(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    if (enable_gpu_ && (F.Nx * F.Ny * F.Nz) > gpu_threshold_)
    {
        metal_backend::Context::Conv3dRestrictParams p {
            F.Nx, F.Ny, F.Nz, C.Nx, C.Ny, C.Nz, F.hx, F.hy, F.hz
        };
        ctx_->conv3d_restrict_device(F.u, F.f, C.f, p);
        return;
    }

    const int Nxf = F.Nx, Nyf = F.Ny, Nzf = F.Nz;
    const int Nxc = C.Nx, Nyc = C.Ny, Nzc = C.Nz;

    const float cx_f = static_cast<float>(F.cx);
    const float cy_f = static_cast<float>(F.cy);
    const float cz_f = static_cast<float>(F.cz);
    const float dg_f = static_cast<float>(F.diag);

    const float* u_ptr = F.u.data();
    const float* f_ptr = F.f.data();
    float* cf_ptr = C.f.data();

    const long long Mxy_f = static_cast<long long>(Nxf) * Nyf;

    auto r_at = [&](int i, int j, int k) -> float {
        const long long kk = (static_cast<long long>(k) * Nyf + j) * Nxf + i;
        const float Au = dg_f * u_ptr[kk]
            - cx_f * (u_ptr[kk - 1] + u_ptr[kk + 1])
            - cy_f * (u_ptr[kk - Nxf] + u_ptr[kk + Nxf])
            - cz_f * (u_ptr[kk - Mxy_f] + u_ptr[kk + Mxy_f]);
        return f_ptr[kk] - Au;
    };

    const bool use_omp = enable_omp_ && (static_cast<long long>(Nxc) * Nyc * Nzc) > 50000;

    C.f.fill(0.0f);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int lc = 1; lc < Nzc - 1; lc++)
    {
        for (int jc = 1; jc < Nyc - 1; jc++)
        {
            for (int ic = 1; ic < Nxc - 1; ic++)
            {
                const int fi = 2 * ic;
                const int fj = 2 * jc;
                const int fl = 2 * lc;

                float sum = 0.0f;

                for (int dl = -1; dl <= 1; dl++)
                {
                    const float wl = (dl == 0) ? 2.0f : 1.0f;
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        const float wj = (dj == 0) ? 2.0f : 1.0f;
                        for (int di = -1; di <= 1; di++)
                        {
                            const float wi = (di == 0) ? 2.0f : 1.0f;
                            sum += wi * wj * wl * r_at(fi + di, fj + dj, fl + dl);
                        }
                    }
                }

                cf_ptr[idx(ic, jc, lc, Nxc, Nyc)] = sum * 0.015625f;
            }
        }
    }
}


void MG::prolong(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    if (enable_gpu_ && (F.Nx * F.Ny * F.Nz) > gpu_threshold_)
    {
        metal_backend::Context::ProlongParams p {
            F.Nx, F.Ny, F.Nz, C.Nx, C.Ny, C.Nz
        };
        ctx_->prolong_add_device(F.u, C.u, p);
        return;
    }

    interpolate_reverse(lvl);
}

void MG::coarse_solve(int lvl)
{
    static constexpr int num_iter = 100;

    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny, Nz = L.Nz;
    const float cx = L.cx, cy = L.cy, cz = L.cz, diag = L.diag;

    const float dg_f = static_cast<float>(diag);
    const float cx_f = static_cast<float>(cx);
    const float cy_f = static_cast<float>(cy);
    const float cz_f = static_cast<float>(cz);

    const float* f_ptr = L.f.data();
    float* u_ptr = L.u.data();

    for (int s = 0; s < num_iter; s++)
    {
        for (int l = 1; l < Nz - 1; l++)
        {
            for (int j = 1; j < Ny - 1; j++)
            {
                for (int i = 1; i < Nx - 1; i++)
                {
                    const int k = idx(i, j, l, Nx, Ny);

                    const float rhs = f_ptr[k]
                        + cx_f * (u_ptr[k - 1] + u_ptr[k + 1])
                        + cy_f * (u_ptr[k - Nx] + u_ptr[k + Nx])
                        + cz_f * (u_ptr[k - Nx * Ny] + u_ptr[k + Nx * Ny]);

                    u_ptr[k] = rhs / dg_f;
                }
            }
        }
    }
}

void MG::v_cycle(int lvl)
{
    const int last = static_cast<int>(levels_.size()) - 1;

    if (lvl == last)
    {
        if (gpu_batch_open_) { ctx_->end_batch_and_wait(); gpu_batch_open_ = false; }
        coarse_solve(lvl);
        return;
    }

    const bool this_gpu = enable_gpu_ && (levels_[lvl].Nx * levels_[lvl].Ny * levels_[lvl].Nz > gpu_threshold_);

    if (this_gpu && !gpu_batch_open_) { ctx_->begin_batch(); gpu_batch_open_ = true; }
    else if (!this_gpu && gpu_batch_open_) { ctx_->end_batch_and_wait(); gpu_batch_open_ = false; }

    smooth(lvl, nu1_);
    conv3d_restrict(lvl);

    levels_[lvl + 1].u.fill(0.0f);

    v_cycle(lvl + 1);

    if (this_gpu && !gpu_batch_open_) { ctx_->begin_batch(); gpu_batch_open_ = true; }
    else if (!this_gpu && gpu_batch_open_) { ctx_->end_batch_and_wait(); gpu_batch_open_ = false; }

    prolong(lvl);
    smooth(lvl, nu2_);
}

void MG::apply(const std::vector<float>& r_fine, std::vector<float>& z_fine)
{
    Level& L0 = levels_.front();
    const int Nx = L0.Nx, Ny = L0.Ny, Nz = L0.Nz;
    const int Mx = Nx - 2, My = Ny - 2, Mz = Nz - 2;

    for (auto& L : levels_)
    {
        L.u.fill(0.0f);
        L.u_new.fill(0.0f);
        L.f.fill(0.0f);
    }

    float* L0f = L0.f.data();

    for (int l = 0; l < Mz; l++)
    {
        for (int j = 0; j < My; j++)
        {
            for (int i = 0; i < Mx; i++)
            {
                const int kf = (l * My + j) * Mx + i;
                L0f[idx(i + 1, j + 1, l + 1, Nx, Ny)] = r_fine[kf];
            }
        }
    }

    gpu_batch_open_ = false;
    v_cycle(0);
    if (gpu_batch_open_) { ctx_->end_batch_and_wait(); gpu_batch_open_ = false; }

    const std::size_t Mfull = static_cast<std::size_t>(Mx) * My * Mz;
    if (z_fine.size() != Mfull) z_fine.resize(Mfull);

    const float* L0u = L0.u.data();

    for (int l = 0; l < Mz; l++)
    {
        for (int j = 0; j < My; j++)
        {
            for (int i = 0; i < Mx; i++)
            {
                const int kf = (l * My + j) * Mx + i;
                z_fine[kf] = L0u[idx(i + 1, j + 1, l + 1, Nx, Ny)];
            }
        }
    }
}
