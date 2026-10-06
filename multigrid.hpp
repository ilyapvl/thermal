#pragma once
#include <vector>
#include <stdexcept>
#include "metal_backend.hpp"

class MG
{
public:
    MG(metal_backend::Context* ctx,
       int Nx, int Ny, int Nz,
       float Lx, float Ly, float Lz,
       int nu1 = 3, int nu2 = 3, float omega = 2.0 / 3.0);

    void apply(const std::vector<double>& r, std::vector<double>& z);

    int finest_inner() const
    {
        return (levels_.front().Nx - 2) * (levels_.front().Ny - 2) * (levels_.front().Nz - 2);
    }

private:
    struct Level
    {
        int Nx, Ny, Nz;
        float hx, hy, hz;
        float cx, cy, cz, diag;

        metal_backend::DeviceBuffer u;
        metal_backend::DeviceBuffer u_new;
        metal_backend::DeviceBuffer f;
    };

    std::vector<Level> levels_;
    int nu1_, nu2_;
    float omega_;
    metal_backend::Context* ctx_ = nullptr;
    bool enable_omp_ = false;
    bool enable_gpu_ = false;
    bool gpu_batch_open_ = false;
    int gpu_threshold_ = 2000000;

    void v_cycle(int lvl);
    void smooth(int lvl, int nu);
    void interpolate_reverse(int lvl);
    void solve_final(int lvl);

    void conv3d(int lvl);
    void deconv3d(int lvl);

    static int idx(int i, int j, int k, int Nx, int Ny) { return (k * Ny + j) * Nx + i; }
};
