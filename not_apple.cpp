#include "metal_backend.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace metal_backend
{

struct DeviceBuffer::Impl
{
    std::vector<float> v;
};

DeviceBuffer::DeviceBuffer() : impl_(new Impl) {}

DeviceBuffer::~DeviceBuffer()
{
    delete impl_;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : impl_(other.impl_)
{
    other.impl_ = new Impl;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept
{
    if (this != &other)
    {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = new Impl;
    }
    return *this;
}

void DeviceBuffer::resize(std::size_t n)
{
    impl_->v.assign(n, 0.0f);
}

std::size_t DeviceBuffer::size() const
{
    return impl_->v.size();
}

float* DeviceBuffer::data()
{
    return impl_->v.data();
}

const float* DeviceBuffer::data() const
{
    return impl_->v.data();
}

void DeviceBuffer::fill(float v)
{
    std::fill(impl_->v.begin(), impl_->v.end(), v);
}

void* DeviceBuffer::native() const
{
    return nullptr;
}

struct Context::Impl
{
};

Context::Context() : impl_(new Impl)
{
    throw std::runtime_error("Metal backend is not available on this platform");
}

Context::~Context()
{
    delete impl_;
}

bool Context::available() const
{
    return false;
}

const char* Context::device_name() const
{
    return "stub";
}

void Context::jacobi_smooth_device(DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&, const JacobiParams&, int)
{
    throw std::runtime_error("Metal backend not available");
}

void Context::conv3d_device(const DeviceBuffer&, const DeviceBuffer&, DeviceBuffer&, const Conv3dParams&)
{
    throw std::runtime_error("Metal backend not available");
}

void Context::deconv3d_device(DeviceBuffer&, const DeviceBuffer&, const Deconv3dParams&)
{
    throw std::runtime_error("Metal backend not available");
}

void Context::begin_batch()
{
    throw std::runtime_error("Metal backend not available");
}

void Context::end_batch_and_wait()
{
    throw std::runtime_error("Metal backend not available");
}

}
