#include <metal_stdlib>
using namespace metal;

kernel void jacobi_smooth(device const float* u     [[buffer(0)]],
                        device const float* f     [[buffer(1)]],
                        device float*       u_new [[buffer(2)]],
                        constant uint&      Nx    [[buffer(3)]],
                        constant uint&      Ny    [[buffer(4)]],
                        constant float&     cx    [[buffer(5)]],
                        constant float&     cy    [[buffer(6)]],
                        constant float&     diag  [[buffer(7)]],
                        constant float&     omega [[buffer(8)]],
                        uint k [[thread_position_in_grid]])
{
    const uint Mx = Nx - 2;
    const uint My = Ny - 2;
    const uint M  = Mx * My;
    if (k >= M) return;

    const uint ii = k % Mx;
    const uint jj = k / Mx;
    const uint i  = ii + 1;
    const uint j  = jj + 1;
    const uint idx = j * Nx + i;

    const float Au = diag * u[idx]
        - cx * (u[idx - 1] + u[idx + 1])
        - cy * (u[idx - Nx] + u[idx + Nx]);

    u_new[idx] = u[idx] + omega * (f[idx] - Au) / diag;
}

kernel void residual_restrict(device const float* u_fine  [[buffer(0)]],
                            device const float* f_fine  [[buffer(1)]],
                            device float*       f_coarse[[buffer(2)]],
                            constant uint&      Nxf     [[buffer(3)]],
                            constant uint&      Nxc     [[buffer(4)]],
                            constant uint&      Nyc     [[buffer(5)]],
                            constant float&     cx      [[buffer(6)]],
                            constant float&     cy      [[buffer(7)]],
                            constant float&     diag    [[buffer(8)]],
                            uint                kc      [[thread_position_in_grid]])
{
    const uint Mcx = Nxc - 2;
    const uint Mcy = Nyc - 2;
    const uint Mc  = Mcx * Mcy;
    if (kc >= Mc) return;

    const uint ic = kc % Mcx;
    const uint jc = kc / Mcx;

    const uint fi = 2 * (ic + 1);
    const uint fj = 2 * (jc + 1);

    auto r_at = [&](uint i, uint j) -> float {
        const uint k = j * Nxf + i;
        const float Au = diag * u_fine[k]
            - cx * (u_fine[k - 1] + u_fine[k + 1])
            - cy * (u_fine[k - Nxf] + u_fine[k + Nxf]);
        return f_fine[k] - Au;
    };

    const float v =
          1.0f * r_at(fi - 1, fj - 1)
        + 2.0f * r_at(fi,     fj - 1)
        + 1.0f * r_at(fi + 1, fj - 1)
        + 2.0f * r_at(fi - 1, fj)
        + 4.0f * r_at(fi,     fj)
        + 2.0f * r_at(fi + 1, fj)
        + 1.0f * r_at(fi - 1, fj + 1)
        + 2.0f * r_at(fi,     fj + 1)
        + 1.0f * r_at(fi + 1, fj + 1);

    f_coarse[(jc + 1) * Nxc + (ic + 1)] = v * 0.0625f;
}

kernel void prolong_add(device float*       u_fine  [[buffer(0)]],
                        device const float* u_coarse[[buffer(1)]],
                        constant uint&      Nxf     [[buffer(2)]],
                        constant uint&      Nyf     [[buffer(3)]],
                        constant uint&      Nxc     [[buffer(4)]],
                        constant uint&      Nyc     [[buffer(5)]],
                        uint                kf      [[thread_position_in_grid]])
{
    const uint Mxf = Nxf - 2;
    const uint Myf = Nyf - 2;
    const uint Mf  = Mxf * Myf;
    if (kf >= Mf) return;

    const uint ii = kf % Mxf;
    const uint jj = kf / Mxf;
    const uint i  = ii + 1;
    const uint j  = jj + 1;

    const uint kx = i >> 1;
    const uint ky = j >> 1;
    const bool odd_x = (i & 1u) != 0;
    const bool odd_y = (j & 1u) != 0;

    auto C = [&](uint cx, uint cy) -> float {
        return u_coarse[cy * Nxc + cx];
    };

    float val = 0.0f;
    if (!odd_x && !odd_y)
    {
        val = C(kx, ky);
    }
    else if (odd_x && !odd_y)
    {
        val = 0.5f * (C(kx, ky) + C(kx + 1, ky));
    }
    else if (!odd_x && odd_y)
    {
        val = 0.5f * (C(kx, ky) + C(kx, ky + 1));
    }
    else
    {
        val = 0.25f * (C(kx, ky) + C(kx + 1, ky) +
                       C(kx, ky + 1) + C(kx + 1, ky + 1));
    }

    u_fine[j * Nxf + i] += val;
}
