#include <metal_stdlib>
using namespace metal;

kernel void jacobi_smooth_3d(device const float* u [[buffer(0)]],
                            device const float* f [[buffer(1)]],
                            device float* u_new [[buffer(2)]],
                            constant uint& Nx [[buffer(3)]],
                            constant uint& Ny [[buffer(4)]],
                            constant uint& Nz [[buffer(5)]],
                            constant float& cx [[buffer(6)]],
                            constant float& cy [[buffer(7)]],
                            constant float& cz [[buffer(8)]],
                            constant float& diag [[buffer(9)]],
                            constant float& omega [[buffer(10)]],
                            uint k [[thread_position_in_grid]])
{
    const uint Mx = Nx - 2;
    const uint My = Ny - 2;
    const uint Mz = Nz - 2;
    const uint M = Mx * My * Mz;
    if (k >= M) return;

    const uint i = k % Mx;
    const uint j = (k / Mx) % My;
    const uint l = k / (Mx * My);

    const uint idx = ((l + 1) * Ny + (j + 1)) * Nx + (i + 1);

    const float Au = diag * u[idx]
        - cx * (u[idx - 1] + u[idx + 1])
        - cy * (u[idx - Nx] + u[idx + Nx])
        - cz * (u[idx - Nx * Ny] + u[idx + Nx * Ny]);

    u_new[idx] = u[idx] + omega * (f[idx] - Au) / diag;
}

kernel void conv3d(device const float* u_fine [[buffer(0)]],
                    device const float* f_fine [[buffer(1)]],
                    device float* f_coarse [[buffer(2)]],
                    constant uint& Nxf [[buffer(3)]],
                    constant uint& Nyf [[buffer(4)]],
                    constant uint& Nzf [[buffer(5)]],
                    constant uint& Nxc [[buffer(6)]],
                    constant uint& Nyc [[buffer(7)]],
                    constant uint& Nzc [[buffer(8)]],
                    constant float& cx [[buffer(9)]],
                    constant float& cy [[buffer(10)]],
                    constant float& cz [[buffer(11)]],
                    constant float& diag [[buffer(12)]],
                    uint kc [[thread_position_in_grid]])
{
    const uint Mcx = Nxc - 2;
    const uint Mcy = Nyc - 2;
    const uint Mcz = Nzc - 2;
    const uint Mc = Mcx * Mcy * Mcz;
    if (kc >= Mc) return;

    const uint ic = kc % Mcx;
    const uint jc = (kc / Mcx) % Mcy;
    const uint lc = kc / (Mcx * Mcy);

    const uint fi = 2 * (ic + 1);
    const uint fj = 2 * (jc + 1);
    const uint fl = 2 * (lc + 1);

    auto r_at = [&](uint i, uint j, uint l) -> float {
        const uint k = (l * Nyf + j) * Nxf + i;
        const float Au = diag * u_fine[k]
            - cx * (u_fine[k - 1] + u_fine[k + 1])
            - cy * (u_fine[k - Nxf] + u_fine[k + Nxf])
            - cz * (u_fine[k - Nxf * Nyf] + u_fine[k + Nxf * Nyf]);
        return f_fine[k] - Au;
    };

    const float w[3] = {1.0f, 2.0f, 1.0f};

    float sum = 0.0f;

    for (int dl = -1; dl <= 1; dl++)
    {
        for (int dj = -1; dj <= 1; dj++)
        {
            for (int di = -1; di <= 1; di++)
            {
                const float ww = w[di + 1] * w[dj + 1] * w[dl + 1];
                sum += ww * r_at(fi + di, fj + dj, fl + dl);
            }
        }
    }

    f_coarse[((lc + 1) * Nyc + (jc + 1)) * Nxc + (ic + 1)] = sum * 0.015625f;
}

kernel void prolong_add_3d(device float* u_fine [[buffer(0)]],
                        device const float* u_coarse [[buffer(1)]],
                        constant uint& Nxf [[buffer(2)]],
                        constant uint& Nyf [[buffer(3)]],
                        constant uint& Nzf [[buffer(4)]],
                        constant uint& Nxc [[buffer(5)]],
                        constant uint& Nyc [[buffer(6)]],
                        constant uint& Nzc [[buffer(7)]],
                        uint kf [[thread_position_in_grid]])
{
    const uint Mxf = Nxf - 2;
    const uint Myf = Nyf - 2;
    const uint Mzf = Nzf - 2;
    const uint Mf = Mxf * Myf * Mzf;
    if (kf >= Mf) return;

    const uint ii = kf % Mxf;
    const uint jj = (kf / Mxf) % Myf;
    const uint ll = kf / (Mxf * Myf);
    const uint i = ii + 1;
    const uint j = jj + 1;
    const uint l = ll + 1;

    const uint kx = i >> 1;
    const uint ky = j >> 1;
    const uint kz = l >> 1;
    const bool ox = (i & 1u) != 0;
    const bool oy = (j & 1u) != 0;
    const bool oz = (l & 1u) != 0;

    auto C = [&](uint cx, uint cy, uint cz) -> float {
        return u_coarse[(cz * Nyc + cy) * Nxc + cx];
    };

    const float c000 = C(kx, ky, kz);

    float val = 0.0f;

    if (!ox && !oy && !oz)
    {
        val = c000;
    }
    else if (ox && !oy && !oz)
    {
        val = 0.5f * (c000 + C(kx + 1, ky, kz));
    }
    else if (!ox && oy && !oz)
    {
        val = 0.5f * (c000 + C(kx, ky + 1, kz));
    }
    else if (!ox && !oy && oz)
    {
        val = 0.5f * (c000 + C(kx, ky, kz + 1));
    }
    else if (ox && oy && !oz)
    {
        val = 0.25f * (c000 + C(kx + 1, ky, kz)
                            + C(kx, ky + 1, kz)
                            + C(kx + 1, ky + 1, kz));
    }
    else if (ox && !oy && oz)
    {
        val = 0.25f * (c000 + C(kx + 1, ky, kz)
                            + C(kx, ky, kz + 1)
                            + C(kx + 1, ky, kz + 1));
    }
    else if (!ox && oy && oz)
    {
        val = 0.25f * (c000 + C(kx, ky + 1, kz)
                            + C(kx, ky, kz + 1)
                            + C(kx, ky + 1, kz + 1));
    }
    else
    {
        val = 0.125f * (c000
                      + C(kx + 1, ky, kz)
                      + C(kx, ky + 1, kz)
                      + C(kx, ky, kz + 1)
                      + C(kx + 1, ky + 1, kz)
                      + C(kx + 1, ky, kz + 1)
                      + C(kx, ky + 1, kz + 1)
                      + C(kx + 1, ky + 1, kz + 1));
    }

    u_fine[(l * Nyf + j) * Nxf + i] += val;
}
