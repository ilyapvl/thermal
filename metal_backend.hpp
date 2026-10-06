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
        int    Nx, Ny, Nz;
        double hx, hy, hz;
        double omega;
    };




    void jacobi_smooth_device(DeviceBuffer&       u,
                            const DeviceBuffer& f,
                            DeviceBuffer&       u_new,
                            const JacobiParams& p,
                            int                 nu);

    struct Conv3dParams
    {
        int    Nxf, Nyf, Nzf;
        int    Nxc, Nyc, Nzc;
        double hx, hy, hz;
    };

    void conv3d_device(const DeviceBuffer& u_fine,
                                const DeviceBuffer& f_fine,
                                DeviceBuffer&       f_coarse,
                                const Conv3dParams& p);

    struct Deconv3dParams
    {
        int Nxf, Nyf, Nzf;
        int Nxc, Nyc, Nzc;
    };

    void deconv3d_device(DeviceBuffer&       u_fine,
                            const DeviceBuffer& u_coarse,
                            const Deconv3dParams& p);

    void begin_batch();
    void end_batch_and_wait();


    struct Impl;
    Impl* impl_;
};

}



#endif
