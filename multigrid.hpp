#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP



#include <vector>
#include <stdexcept>
#include "metal_backend.hpp"

class MG
{
public:
    MG(metal_backend::Context* ctx,
       int Nx, int Ny, double Lx, double Ly,
       int nu1 = 3, int nu2 = 3, double omega = 2.0 / 3.0);

    // z = M^-1 * r
    // r and z have size (Nx-2)*(Ny-2)
    void apply(const std::vector<double>& r, std::vector<double>& z);

    int finest_inner() const
    {
        return (levels_.front().Nx - 2) * (levels_.front().Ny - 2);
    }

    mutable int gpu_smooth_calls_ = 0;
    mutable int gpu_rr_calls_     = 0;
    mutable int gpu_prol_calls_   = 0;
    

private:
    struct Level
    {
        int Nx, Ny;
        double hx, hy;
        double cx, cy, diag;

        metal_backend::DeviceBuffer u;
        metal_backend::DeviceBuffer u_new;
        metal_backend::DeviceBuffer f;
        metal_backend::DeviceBuffer r;
    };

    std::vector<Level> levels_;
    int nu1_, nu2_;
    double omega_;
    metal_backend::Context* ctx_ = nullptr;
    bool enable_omp_ = false;
    bool enable_gpu_ = true;

    bool gpu_batch_open_ = false;
    int  gpu_threshold_ = 10000000;

    void v_cycle(int lvl);
    void smooth(int lvl, int nu);
    void compute_residual(int lvl);
    void conv2d_restrict(int lvl);
    void interpolate_reverse(int lvl);
    void coarse_solve(int lvl);

    void residual_restrict(int lvl);
    void prolong(int lvl);

    static int idx(int i, int j, int Nx) { return j * Nx + i; }
};

#endif
