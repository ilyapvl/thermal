#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace metal_backend
{

struct DeviceBuffer::Impl
{
    id<MTLBuffer> buffer = nil;
    std::size_t n = 0;
};

static id<MTLDevice> get_device()
{
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

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
    if (n == impl_->n && impl_->buffer) return;

    impl_->buffer = nil;
    impl_->n = 0;

    if (n == 0) return;

    id<MTLDevice> dev = get_device();
    if (!dev) throw std::runtime_error("DeviceBuffer::resize: no Metal device");

    const std::size_t bytes = n * sizeof(float);
    impl_->buffer = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!impl_->buffer) throw std::runtime_error("DeviceBuffer::resize: cannot allocate MTLBuffer");

    impl_->n = n;
}

std::size_t DeviceBuffer::size() const
{
    return impl_->n;
}

float* DeviceBuffer::data()
{
    if (!impl_->buffer) return nullptr;
    return static_cast<float*>([impl_->buffer contents]);
}

const float* DeviceBuffer::data() const
{
    if (!impl_->buffer) return nullptr;
    return static_cast<const float*>([impl_->buffer contents]);
}

void DeviceBuffer::fill(float v)
{
    if (!impl_->buffer) return;

    float* p = static_cast<float*>([impl_->buffer contents]);

    if (v == 0.0f)
    {
        std::memset(p, 0, impl_->n * sizeof(float));
    }
    else
    {
        for (std::size_t i = 0; i < impl_->n; i++) p[i] = v;
    }
}

void* DeviceBuffer::native() const
{
    return impl_->buffer ? (__bridge void*)impl_->buffer : nullptr;
}

struct Context::Impl
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> pso_jacobi = nil;
    id<MTLComputePipelineState> pso_conv3d = nil;
    id<MTLComputePipelineState> pso_deconv3d = nil;

    id<MTLCommandBuffer> batch_cb = nil;
    id<MTLComputeCommandEncoder> batch_enc = nil;

    std::string device_name;
};

static void check_ns_error(NSError* err, const char* what)
{
    if (err)
    {
        NSString* msg = [err localizedDescription];
        throw std::runtime_error(std::string(what) + ": " + ([msg UTF8String] ? [msg UTF8String] : "unknown"));
    }
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, const char* name)
{
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) throw std::runtime_error(std::string("Metal: kernel '") + name + "' not found");

    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];

    check_ns_error(err, ("Metal: cannot create pipeline for " + std::string(name)).c_str());

    if (!pso) throw std::runtime_error(std::string("Metal: pipeline is nil for ") + name);

    return pso;
}

static NSUInteger pick_threadgroup_size(id<MTLComputePipelineState> pso)
{
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    if (tg < 1) tg = 1;
    return tg;
}

static void acquire_enc(Context::Impl* impl, id<MTLCommandBuffer>& cb_out, id<MTLComputeCommandEncoder>& enc_out, bool& owned_out)
{
    if (impl->batch_enc)
    {
        cb_out = nil;
        enc_out = impl->batch_enc;
        owned_out = false;
    }
    else
    {
        cb_out = [impl->queue commandBuffer];
        enc_out = [cb_out computeCommandEncoder];
        owned_out = true;
    }
}

static void release_enc(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc, bool owned, const char* name)
{
    if (!owned) return;

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.error)
    {
        NSString* msg = [cb.error localizedDescription];
        throw std::runtime_error(std::string(name) + ([msg UTF8String] ? [msg UTF8String] : "unknown"));
    }
}

Context::Context() : impl_(new Impl)
{
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) throw std::runtime_error("Metal: no default device");

    impl_->queue = [impl_->device newCommandQueue];
    if (!impl_->queue) throw std::runtime_error("Metal: cannot create command queue");

    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:@"default.metallib"];
    impl_->library = [impl_->device newLibraryWithURL:url error:&err];
    check_ns_error(err, "Metal: cannot load default.metallib");
    if (!impl_->library) throw std::runtime_error("Metal: library is nil");

    impl_->pso_jacobi = make_pipeline(impl_->device, impl_->library, "jacobi_smooth_3d");
    impl_->pso_conv3d = make_pipeline(impl_->device, impl_->library, "conv3d");
    impl_->pso_deconv3d = make_pipeline(impl_->device, impl_->library, "prolong_add_3d");

    NSString* name = [impl_->device name];
    impl_->device_name = name ? [name UTF8String] : "unknown";
}

Context::~Context()
{
    delete impl_;
}

bool Context::available() const
{
    return impl_ && impl_->device != nil;
}

const char* Context::device_name() const
{
    return impl_->device_name.c_str();
}


void Context::jacobi_smooth_device(DeviceBuffer& u, const DeviceBuffer& f, DeviceBuffer& u_new,
                                   const JacobiParams& p, int nu)
{
    if (nu <= 0) return;

    const std::size_t N = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (u.size() != N || f.size() != N || u_new.size() != N) throw std::runtime_error("jacobi_smooth_device: buffer size mismatch");

    id<MTLBuffer> buf_u = (__bridge id<MTLBuffer>)u.native();
    id<MTLBuffer> buf_f = (__bridge id<MTLBuffer>)f.native();
    id<MTLBuffer> buf_unew = (__bridge id<MTLBuffer>)u_new.native();
    if (!buf_u || !buf_f || !buf_unew) throw std::runtime_error("jacobi_smooth_device: null native buffer");

    const double cx_d = 1.0 / (p.hx * p.hx);
    const double cy_d = 1.0 / (p.hy * p.hy);
    const double cz_d = 1.0 / (p.hz * p.hz);
    const double diag_d = 2.0 * (cx_d + cy_d + cz_d);

    const uint32_t Nx32 = static_cast<uint32_t>(p.Nx);
    const uint32_t Ny32 = static_cast<uint32_t>(p.Ny);
    const uint32_t Nz32 = static_cast<uint32_t>(p.Nz);
    const float cx_f = static_cast<float>(cx_d);
    const float cy_f = static_cast<float>(cy_d);
    const float cz_f = static_cast<float>(cz_d);
    const float diag_f = static_cast<float>(diag_d);
    const float om_f = static_cast<float>(p.omega);

    const uint32_t M = (p.Nx - 2) * (p.Ny - 2) * (p.Nz - 2);
    const NSUInteger tg = pick_threadgroup_size(impl_->pso_jacobi);
    MTLSize grid = MTLSizeMake((M + tg - 1) / tg, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);

    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    bool owned;
    acquire_enc(impl_, cb, enc, owned);

    [enc setComputePipelineState:impl_->pso_jacobi];

    bool in_u = true;
    for (int s = 0; s < nu; s++)
    {
        id<MTLBuffer> src = in_u ? buf_u : buf_unew;
        id<MTLBuffer> dst = in_u ? buf_unew : buf_u;

        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:buf_f offset:0 atIndex:1];
        [enc setBuffer:dst offset:0 atIndex:2];
        [enc setBytes:&Nx32 length:sizeof(Nx32) atIndex:3];
        [enc setBytes:&Ny32 length:sizeof(Ny32) atIndex:4];
        [enc setBytes:&Nz32 length:sizeof(Nz32) atIndex:5];
        [enc setBytes:&cx_f length:sizeof(cx_f) atIndex:6];
        [enc setBytes:&cy_f length:sizeof(cy_f) atIndex:7];
        [enc setBytes:&cz_f length:sizeof(cz_f) atIndex:8];
        [enc setBytes:&diag_f length:sizeof(diag_f) atIndex:9];
        [enc setBytes:&om_f length:sizeof(om_f) atIndex:10];

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

        if (s + 1 < nu) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        in_u = !in_u;
    }

    release_enc(cb, enc, owned, "Metal jacobi_smooth_device: ");
}

void Context::conv3d_device(const DeviceBuffer& u_fine, const DeviceBuffer& f_fine,
                                    DeviceBuffer& f_coarse, const Conv3dParams& p)
{
    const uint32_t Nxf32 = static_cast<uint32_t>(p.Nxf);
    const uint32_t Nyf32 = static_cast<uint32_t>(p.Nyf);
    const uint32_t Nzf32 = static_cast<uint32_t>(p.Nzf);
    const uint32_t Nxc32 = static_cast<uint32_t>(p.Nxc);
    const uint32_t Nyc32 = static_cast<uint32_t>(p.Nyc);
    const uint32_t Nzc32 = static_cast<uint32_t>(p.Nzc);

    const float cx_f = static_cast<float>(1.0 / (p.hx * p.hx));
    const float cy_f = static_cast<float>(1.0 / (p.hy * p.hy));
    const float cz_f = static_cast<float>(1.0 / (p.hz * p.hz));
    const float dg_f = 2.0f * (cx_f + cy_f + cz_f);

    const uint32_t Mc = (p.Nxc - 2) * (p.Nyc - 2) * (p.Nzc - 2);
    const NSUInteger tg = pick_threadgroup_size(impl_->pso_conv3d);
    MTLSize grid = MTLSizeMake((Mc + tg - 1) / tg, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);

    id<MTLBuffer> buf_u = (__bridge id<MTLBuffer>)u_fine.native();
    id<MTLBuffer> buf_f = (__bridge id<MTLBuffer>)f_fine.native();
    id<MTLBuffer> buf_fc = (__bridge id<MTLBuffer>)f_coarse.native();

    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    bool owned;
    acquire_enc(impl_, cb, enc, owned);

    [enc setComputePipelineState:impl_->pso_conv3d];
    [enc setBuffer:buf_u offset:0 atIndex:0];
    [enc setBuffer:buf_f offset:0 atIndex:1];
    [enc setBuffer:buf_fc offset:0 atIndex:2];
    [enc setBytes:&Nxf32 length:sizeof(Nxf32) atIndex:3];
    [enc setBytes:&Nyf32 length:sizeof(Nyf32) atIndex:4];
    [enc setBytes:&Nzf32 length:sizeof(Nzf32) atIndex:5];
    [enc setBytes:&Nxc32 length:sizeof(Nxc32) atIndex:6];
    [enc setBytes:&Nyc32 length:sizeof(Nyc32) atIndex:7];
    [enc setBytes:&Nzc32 length:sizeof(Nzc32) atIndex:8];
    [enc setBytes:&cx_f length:sizeof(cx_f) atIndex:9];
    [enc setBytes:&cy_f length:sizeof(cy_f) atIndex:10];
    [enc setBytes:&cz_f length:sizeof(cz_f) atIndex:11];
    [enc setBytes:&dg_f length:sizeof(dg_f) atIndex:12];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

    release_enc(cb, enc, owned, "Metal conv3d: ");
}

void Context::deconv3d_device(DeviceBuffer& u_fine, const DeviceBuffer& u_coarse, const Deconv3dParams& p)
{
    const uint32_t Nxf32 = static_cast<uint32_t>(p.Nxf);
    const uint32_t Nyf32 = static_cast<uint32_t>(p.Nyf);
    const uint32_t Nzf32 = static_cast<uint32_t>(p.Nzf);
    const uint32_t Nxc32 = static_cast<uint32_t>(p.Nxc);
    const uint32_t Nyc32 = static_cast<uint32_t>(p.Nyc);
    const uint32_t Nzc32 = static_cast<uint32_t>(p.Nzc);

    const uint32_t Mf = (p.Nxf - 2) * (p.Nyf - 2) * (p.Nzf - 2);
    const NSUInteger tg = pick_threadgroup_size(impl_->pso_deconv3d);
    MTLSize grid = MTLSizeMake((Mf + tg - 1) / tg, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);

    id<MTLBuffer> buf_uf = (__bridge id<MTLBuffer>)u_fine.native();
    id<MTLBuffer> buf_uc = (__bridge id<MTLBuffer>)u_coarse.native();

    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    bool owned;
    acquire_enc(impl_, cb, enc, owned);

    [enc setComputePipelineState:impl_->pso_deconv3d];
    [enc setBuffer:buf_uf offset:0 atIndex:0];
    [enc setBuffer:buf_uc offset:0 atIndex:1];
    [enc setBytes:&Nxf32 length:sizeof(Nxf32) atIndex:2];
    [enc setBytes:&Nyf32 length:sizeof(Nyf32) atIndex:3];
    [enc setBytes:&Nzf32 length:sizeof(Nzf32) atIndex:4];
    [enc setBytes:&Nxc32 length:sizeof(Nxc32) atIndex:5];
    [enc setBytes:&Nyc32 length:sizeof(Nyc32) atIndex:6];
    [enc setBytes:&Nzc32 length:sizeof(Nzc32) atIndex:7];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

    release_enc(cb, enc, owned, "Metal prolong_add: ");
}

void Context::begin_batch()
{
    if (impl_->batch_enc) return;

    impl_->batch_cb = [impl_->queue commandBuffer];
    if (!impl_->batch_cb) throw std::runtime_error("Metal begin_batch: no command buffer");

    impl_->batch_enc = [impl_->batch_cb computeCommandEncoder];
    if (!impl_->batch_enc)
    {
        impl_->batch_cb = nil;
        throw std::runtime_error("Metal begin_batch: no encoder");
    }
}

void Context::end_batch_and_wait()
{
    if (!impl_->batch_enc) return;

    id<MTLCommandBuffer> cb = impl_->batch_cb;
    id<MTLComputeCommandEncoder> enc = impl_->batch_enc;
    impl_->batch_cb = nil;
    impl_->batch_enc = nil;

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.error)
    {
        NSString* msg = [cb.error localizedDescription];
        throw std::runtime_error(std::string("Metal end_batch: ") + ([msg UTF8String] ? [msg UTF8String] : "unknown"));
    }
}

}
