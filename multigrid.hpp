#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP



#include <vector>
#include <stdexcept>

class MG
{
public:
    // Nx, Ny = 2^n+1
    MG(int Nx, int Ny, double Lx, double Ly, int nu1 = 3, int nu2 = 3, double omega = 2.0 / 3.0);

    // z = M^-1 * r
    // r and z have size (Nx-2)*(Ny-2)
    void apply(const std::vector<double>& r, std::vector<double>& z);

    int finest_inner() const
    {
        return (levels_.front().Nx - 2) * (levels_.front().Ny - 2);
    }

private:
    struct Level
    {
        int Nx, Ny;
        double hx, hy; // = Lx / (Nx - 1);
        double cx, cy, diag; // 1 / hx^2, 1 / hy^2, 2(cx + cy)

        std::vector<double> u; // solution
        std::vector<double> u_new; // solution temp
        std::vector<double> f; // right part

        std::vector<double> r;
    };

    std::vector<Level> levels_;
    int nu1_, nu2_;
    double omega_;

    bool enable_omp_ = false;

    void v_cycle(int lvl);
    void smooth(int lvl, int nu); 
    void compute_residual(int lvl);
    void conv2d_restrict(int lvl);
    void interpolate_reverse(int lvl);
    void coarse_solve(int lvl);

    static int idx(int i, int j, int Nx) { return j * Nx + i; }
};



#endif
