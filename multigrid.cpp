#include "multigrid.hpp"
#include <cmath>
#include <algorithm>

MG::MG(metal_backend::Context* ctx,
       int Nx, int Ny, double Lx, double Ly,
       int nu1, int nu2, double omega)
    : nu1_(nu1), nu2_(nu2), omega_(omega), ctx_(ctx)
{
    auto is_pow2 = [](int n){ return n > 0 && (n & (n-1)) == 0; };
    if (!is_pow2(Nx - 1) || !is_pow2(Ny - 1))
        throw std::runtime_error("MG: Nx-1 and Ny-1 must be powers of 2");

    enable_omp_ = ((long long)Nx * Ny) > 200000;
    enable_gpu_ = (ctx_ != nullptr) && ctx_->available();

    int Nxi = Nx, Nyi = Ny;
    double hx = Lx / (Nx - 1);
    double hy = Ly / (Ny - 1);

    while (true)
    {
        Level L;
        L.Nx = Nxi; L.Ny = Nyi;
        L.hx = hx;  L.hy = hy;
        L.cx = 1.0 / (hx * hx);
        L.cy = 1.0 / (hy * hy);
        L.diag = 2.0 * (L.cx + L.cy);

        const std::size_t sz = (std::size_t)L.Nx * L.Ny;
        L.u.resize(sz);
        L.u_new.resize(sz);
        L.f.resize(sz);
        L.r.resize(sz);
        L.u.fill(0.0f);
        L.u_new.fill(0.0f);
        L.f.fill(0.0f);
        L.r.fill(0.0f);

        levels_.push_back(std::move(L));

        if (Nxi <= 3 || Nyi <= 3) break;

        Nxi = (Nxi + 1) / 2;
        Nyi = (Nyi + 1) / 2;
        hx *= 2.0;
        hy *= 2.0;
    }
}


void MG::smooth(int lvl, int nu)
{
    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny;

    if (enable_gpu_ && (Nx * Ny) > gpu_threshold_)
    {
        metal_backend::Context::JacobiParams jp { Nx, Ny, L.hx, L.hy, omega_ };
        ctx_->jacobi_smooth_device(L.u, L.f, L.u_new, jp, nu);
        if (nu % 2 != 0) std::swap(L.u, L.u_new);
        return;
    }

    const double cx = L.cx, cy = L.cy, diag = L.diag;
    const double inv_diag = 1.0 / diag;
    const double w = omega_;
    const bool use_omp = enable_omp_ && (Nx * Ny) > 50000;

    const float cx_f   = static_cast<float>(cx);
    const float cy_f   = static_cast<float>(cy);
    const float dg_f   = static_cast<float>(diag);
    const float inv_f  = static_cast<float>(inv_diag);
    const float w_f    = static_cast<float>(w);

    const float* f_ptr = L.f.data();

    bool u_is_current = true;
    float* u_cur  = L.u.data();
    float* u_alt  = L.u_new.data();

    #pragma omp parallel if(use_omp)
    {
        for (int s = 0; s < nu; s++)
        {
            const float* src = u_is_current ? u_cur : u_alt;
            float*       dst = u_is_current ? u_alt : u_cur;

            #pragma omp for schedule(static) nowait
            for (int j = 1; j < Ny - 1; j++)
            {
                for (int i = 1; i < Nx - 1; i++)
                {
                    const int k = idx(i, j, Nx);
                    const float Au = dg_f * src[k]
                        - cx_f * (src[k - 1] + src[k + 1])
                        - cy_f * (src[k - Nx] + src[k + Nx]);
                    dst[k] = src[k] + w_f * (f_ptr[k] - Au) * inv_f;
                }
            }

            #pragma omp barrier
            #pragma omp single
            { u_is_current = !u_is_current; }
        }
    }
    if (!u_is_current) std::swap(L.u, L.u_new);
}

void MG::compute_residual(int lvl)
{
    Level& L = levels_[lvl];
    const int Nx = L.Nx, Ny = L.Ny;
    const float cx_f = static_cast<float>(L.cx);
    const float cy_f = static_cast<float>(L.cy);
    const float dg_f = static_cast<float>(L.diag);
    const bool use_omp = enable_omp_ && (Nx * Ny) > 8192;

    L.r.fill(0.0f);

    const float* u_ptr = L.u.data();
    const float* f_ptr = L.f.data();
    float*       r_ptr = L.r.data();

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; i++)
        {
            const int k = idx(i, j, Nx);
            const float Au = dg_f * u_ptr[k]
                - cx_f * (u_ptr[k - 1] + u_ptr[k + 1])
                - cy_f * (u_ptr[k - Nx] + u_ptr[k + Nx]);
            r_ptr[k] = f_ptr[k] - Au;
        }
    }
}

void MG::conv2d_restrict(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    const int Nxf = F.Nx, Nxc = C.Nx;
    const int Nyf = F.Ny, Nyc = C.Ny;

    C.f.fill(0.0);

    const float* r_ptr  = F.r.data();
    float*       cf_ptr = C.f.data();

    const bool use_omp = enable_omp_ && (Nxc * Nyc) > 8192;

    #pragma omp parallel for schedule(static) if(use_omp && enable_omp_)
    for (int jc = 1; jc < Nyc - 1; jc++)
    {
        for (int ic = 1; ic < Nxc - 1; ic++)
        {
            const int fi = 2 * ic, fj = 2 * jc;

            const double v =
                1.0 * r_ptr[idx(fi-1, fj-1, Nxf)] +
                2.0 * r_ptr[idx(fi,   fj-1, Nxf)] +
                1.0 * r_ptr[idx(fi+1, fj-1, Nxf)] +
                2.0 * r_ptr[idx(fi-1, fj,   Nxf)] +
                4.0 * r_ptr[idx(fi,   fj,   Nxf)] +
                2.0 * r_ptr[idx(fi+1, fj,   Nxf)] +
                1.0 * r_ptr[idx(fi-1, fj+1, Nxf)] +
                2.0 * r_ptr[idx(fi,   fj+1, Nxf)] +
                1.0 * r_ptr[idx(fi+1, fj+1, Nxf)];

            cf_ptr[idx(ic, jc, Nxc)] = v * 0.0625f;
        }
    }
}

void MG::interpolate_reverse(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    const int Nxf = F.Nx, Nxc = C.Nx;
    const int Nyf = F.Ny, Nyc = C.Ny;

    const bool use_omp = (Nxc * Nyc) > 4096;

    const float* cu_ptr = C.u.data();
    float*       fu_ptr = F.u.data();

    // even even
    #pragma omp parallel for schedule(static) if(use_omp && enable_omp_)
    for (int jc = 0; jc < Nyc; jc++)
        for (int ic = 0; ic < Nxc; ic++)
            fu_ptr[idx(2 * ic, 2 * jc, Nxf)] += cu_ptr[idx(ic, jc, Nxc)];

    // odd even
    #pragma omp parallel for schedule(static) if(use_omp && enable_omp_)
    for (int jc = 0; jc < Nyc; jc++)
        for (int ic = 0; ic < Nxc - 1; ic++)
        {
            const double v = 0.5 * (cu_ptr[idx(ic, jc, Nxc)] + cu_ptr[idx(ic + 1, jc, Nxc)]);
            fu_ptr[idx(2 * ic + 1, 2 * jc, Nxf)] += v;
        }

    // even odd
    #pragma omp parallel for schedule(static) if(use_omp && enable_omp_)
    for (int jc = 0; jc < Nyc - 1; jc++)
        for (int ic = 0; ic < Nxc; ic++)
        {
            const double v = 0.5 * (cu_ptr[idx(ic, jc, Nxc)] + cu_ptr[idx(ic, jc + 1, Nxc)]);
            fu_ptr[idx(2 * ic, 2 * jc + 1, Nxf)] += v;
        }

    // odd odd
    #pragma omp parallel for schedule(static) if(use_omp && enable_omp_)
    for (int jc = 0; jc < Nyc - 1; jc++)
        for (int ic = 0; ic < Nxc - 1; ic++)
        {
            const double v = 0.25 * (
                cu_ptr[idx(ic, jc, Nxc)] + cu_ptr[idx(ic + 1, jc, Nxc)] +
                cu_ptr[idx(ic, jc + 1, Nxc)] + cu_ptr[idx(ic + 1, jc + 1, Nxc)]);
            fu_ptr[idx(2 * ic + 1, 2 * jc + 1, Nxf)] += v;
        }
}

void MG::residual_restrict(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    if (enable_gpu_ && (F.Nx * F.Ny) > gpu_threshold_)
    {
        metal_backend::Context::ResidualRestrictParams p {
            F.Nx, F.Ny, C.Nx, C.Ny, F.hx, F.hy
        };
        ctx_->residual_restrict_device(F.u, F.f, C.f, p);
        return;
    }

    compute_residual(lvl);
    conv2d_restrict(lvl);
}

void MG::prolong(int lvl)
{
    Level& F = levels_[lvl];
    Level& C = levels_[lvl + 1];

    if (enable_gpu_ && (F.Nx * F.Ny) > gpu_threshold_)
    {
        metal_backend::Context::ProlongParams p {
            F.Nx, F.Ny, C.Nx, C.Ny
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
    const int Nx = L.Nx, Ny = L.Ny;
    const double cx = L.cx, cy = L.cy, diag = L.diag;

    const float dg_f = static_cast<float>(diag);
    const float cx_f = static_cast<float>(cx);
    const float cy_f = static_cast<float>(cy);

    const float* f_ptr = L.f.data();
    float*       u_ptr = L.u.data();

    for (int s = 0; s < num_iter; s++)
    {
        for (int j = 1; j < Ny - 1; j++)
        {
            for (int i = 1; i < Nx - 1; i++)
            {
                const int k = idx(i, j, Nx);

                const float rhs = f_ptr[k]
                    + cx_f * (u_ptr[k - 1] + u_ptr[k + 1])
                    + cy_f * (u_ptr[k - Nx] + u_ptr[k + Nx]);

                u_ptr[k] = rhs / dg_f;
            }
        }
    }
}

void MG::v_cycle(int lvl)
{
    const int last = static_cast<int>(levels_.size()) - 1;

    if (lvl == last)
    {
        if (gpu_batch_open_)
        {
            ctx_->end_batch_and_wait();
            gpu_batch_open_ = false;
        }
        coarse_solve(lvl);
        return;
    }

    const bool this_gpu =
        enable_gpu_ &&
        (levels_[lvl].Nx * levels_[lvl].Ny > gpu_threshold_);

    if (this_gpu && !gpu_batch_open_)
    {
        ctx_->begin_batch();
        gpu_batch_open_ = true;
    }
    else if (!this_gpu && gpu_batch_open_)
    {
        ctx_->end_batch_and_wait();
        gpu_batch_open_ = false;
    }

    smooth(lvl, nu1_);
    residual_restrict(lvl);

    levels_[lvl + 1].u.fill(0.0);

    v_cycle(lvl + 1);

    if (this_gpu && !gpu_batch_open_)
    {
        ctx_->begin_batch();
        gpu_batch_open_ = true;
    }
    else if (!this_gpu && gpu_batch_open_)
    {
        ctx_->end_batch_and_wait();
        gpu_batch_open_ = false;
    }

    prolong(lvl);
    smooth(lvl, nu2_);
}

void MG::apply(const std::vector<double>& r_fine, std::vector<double>& z_fine)
{
    Level& L0 = levels_.front();
    const int Nx = L0.Nx, Ny = L0.Ny;

    for (auto& L : levels_)
    {
        L.u.fill(0.0f);
        L.u_new.fill(0.0f);
        L.f.fill(0.0f);
    }

    float* L0f = L0.f.data();
    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; i++)
        {
            const int kf = (j - 1) * (Nx - 2) + (i - 1);
            L0f[idx(i, j, Nx)] = static_cast<float>(r_fine[kf]);
        }
    }

    gpu_batch_open_ = false;
    v_cycle(0);
    if (gpu_batch_open_)
    {
        ctx_->end_batch_and_wait();
        gpu_batch_open_ = false;
    }

    if (z_fine.size() != (std::size_t)(Nx - 2) * (Ny - 2))
        z_fine.resize((Nx - 2) * (Ny - 2));

    const float* L0u = L0.u.data();
    for (int j = 1; j < Ny - 1; j++)
    {
        for (int i = 1; i < Nx - 1; i++)
        {
            const int kf = (j - 1) * (Nx - 2) + (i - 1);
            z_fine[kf] = static_cast<double>(L0u[idx(i, j, Nx)]);
        }
    }
}
