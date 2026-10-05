#ifndef METAL_BACKEND_HPP
#define METAL_BACKEND_HPP

#include <vector>
#include <cstddef>

namespace metal_backend
{

class DeviceBuffer
{
public:
    DeviceBuffer();
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void        resize(std::size_t n);
    std::size_t size()   const;
    bool        empty()  const { return size() == 0; }

    float*       data();
    const float* data() const;

    float&       operator[](std::size_t i)       { return data()[i]; }
    const float& operator[](std::size_t i) const { return data()[i]; }

    void fill(float v);

    void* native() const;

private:
    struct Impl;
    Impl* impl_;
};

class Context
{
public:
    Context();
    ~Context();

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    bool        available()   const;
    const char* device_name() const;


    struct JacobiParams
    {
        int    Nx, Ny;
        double hx, hy;
        double omega;
    };




    void jacobi_smooth_device(DeviceBuffer&       u,
                              const DeviceBuffer& f,
                              DeviceBuffer&       u_new,
                              const JacobiParams& p,
                              int                 nu);

    struct ResidualRestrictParams
    {
        int    Nxf, Nyf;
        int    Nxc, Nyc;
        double hx, hy;
    };

    void residual_restrict_device(const DeviceBuffer& u_fine,
                                  const DeviceBuffer& f_fine,
                                  DeviceBuffer&       f_coarse,
                                  const ResidualRestrictParams& p);

    struct ProlongParams
    {
        int Nxf, Nyf;
        int Nxc, Nyc;
    };

    void prolong_add_device(DeviceBuffer&       u_fine,
                            const DeviceBuffer& u_coarse,
                            const ProlongParams& p);

    void begin_batch();
    void end_batch_and_wait();


    struct Impl;
    Impl* impl_;
};

}



#endif
