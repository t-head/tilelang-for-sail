/*
 * PPU Device API — uses native HGGC runtime APIs (hggcMalloc, hggcFree, etc.)
 * Registered as "device_api.ppu" for kDLPPU.
 */
#include <hggc.h>
#include <hggc_runtime_api.h>

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <cstring>
#include <mutex>
#include <vector>

namespace tvm {
namespace runtime {

#define HGGC_RT_CALL(func)                                                     \
  {                                                                            \
    hggcError_t e = (func);                                                    \
    TVM_FFI_ICHECK(e == hggcSuccess || e == hggcErrorHggcrtUnloading)          \
        << "HGGC: " << hggcGetErrorString(e);                                  \
  }

class PPUThreadEntry {
public:
  hggcStream_t stream{nullptr};
  static PPUThreadEntry *ThreadLocal();
};

PPUThreadEntry *PPUThreadEntry::ThreadLocal() {
  static thread_local PPUThreadEntry inst;
  return &inst;
}

class PPUDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(Device dev) final {
    HGGC_RT_CALL(hggcSetDevice(dev.device_id));
  }

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any *rv) final {
    int value = 0;
    switch (kind) {
    case kExist: {
      int count;
      auto err = hggcGetDeviceCount(&count);
      value = (err == hggcSuccess && dev.device_id < count);
      break;
    }
    case kMaxThreadsPerBlock: {
      HGGC_RT_CALL(hggcDeviceGetAttribute(&value, hggcDevAttrMaxThreadsPerBlock,
                                          dev.device_id));
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      struct hggcDeviceProp prop;
      HGGC_RT_CALL(hggcGetDeviceProperties(&prop, dev.device_id));
      value = static_cast<int>(prop.sharedMemPerBlock);
      break;
    }
    case kComputeVersion: {
      struct hggcDeviceProp prop;
      HGGC_RT_CALL(hggcGetDeviceProperties(&prop, dev.device_id));
      std::ostringstream os;
      os << prop.major << "." << prop.minor;
      *rv = os.str();
      return;
    }
    case kMaxRegistersPerBlock: {
      struct hggcDeviceProp prop;
      HGGC_RT_CALL(hggcGetDeviceProperties(&prop, dev.device_id));
      value = prop.regsPerBlock;
      break;
    }
    default:
      LOG(FATAL) << "Unknown attribute kind " << static_cast<int>(kind);
    }
    *rv = value;
  }

  void *AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    HGGC_RT_CALL(hggcSetDevice(dev.device_id));
    void *ptr = nullptr;
    HGGC_RT_CALL(hggcMalloc(&ptr, nbytes));
    return ptr;
  }

  void *AllocDataSpace(Device dev, int ndim, const int64_t *shape,
                       DLDataType dtype,
                       ffi::Optional<ffi::String> mem_scope) final {
    size_t nbytes = 1;
    for (int i = 0; i < ndim; ++i) {
      nbytes *= shape[i];
    }
    nbytes *= (dtype.bits * dtype.lanes + 7) / 8;
    return AllocDataSpace(dev, nbytes, 0, dtype);
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    HGGC_RT_CALL(hggcFree(ptr));
  }

  void CopyDataFromTo(DLTensor *from, DLTensor *to,
                      TVMStreamHandle stream) final {
    size_t nbytes = GetDataSize(*from);
    int from_dev = static_cast<int>(from->device.device_type);
    int to_dev = static_cast<int>(to->device.device_type);

    hggcMemcpyKind kind;
    if (from_dev == kDLCPU && to_dev == kDLCPU) {
      kind = hggcMemcpyHostToHost;
    } else if (from_dev == kDLCPU && to_dev == kDLPPU) {
      kind = hggcMemcpyHostToDevice;
    } else if (from_dev == kDLPPU && to_dev == kDLCPU) {
      kind = hggcMemcpyDeviceToHost;
    } else if (from_dev == kDLPPU && to_dev == kDLPPU) {
      kind = hggcMemcpyDeviceToDevice;
    } else {
      if (from_dev == kDLCPU && to_dev == kDLPPU) {
        kind = hggcMemcpyHostToDevice;
      } else if (from_dev == kDLPPU && to_dev == kDLCPU) {
        kind = hggcMemcpyDeviceToHost;
      } else if (from_dev == kDLPPU && to_dev == kDLPPU) {
        kind = hggcMemcpyDeviceToDevice;
      } else {
        LOG(FATAL) << "Unsupported copy from " << from_dev << " to " << to_dev;
      }
    }

    hggcStream_t hstream = nullptr;
    if (stream != nullptr) {
      hstream = *static_cast<hggcStream_t *>(stream);
    } else {
      hstream = PPUThreadEntry::ThreadLocal()->stream;
    }

    if (hstream != nullptr) {
      HGGC_RT_CALL(
          hggcMemcpyAsync(to->data, from->data, nbytes, kind, hstream));
    } else {
      HGGC_RT_CALL(hggcMemcpy(to->data, from->data, nbytes, kind));
    }
  }

  // Also support the raw pointer overload
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t size, Device dev_from,
                      Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    int from_dev = static_cast<int>(dev_from.device_type);
    int to_dev = static_cast<int>(dev_to.device_type);

    hggcMemcpyKind kind;
    if (from_dev == kDLCPU && to_dev == kDLCPU) {
      kind = hggcMemcpyHostToHost;
    } else if (from_dev == kDLCPU && to_dev == kDLPPU) {
      kind = hggcMemcpyHostToDevice;
    } else if (from_dev == kDLPPU && to_dev == kDLCPU) {
      kind = hggcMemcpyDeviceToHost;
    } else if (from_dev == kDLPPU && to_dev == kDLPPU) {
      kind = hggcMemcpyDeviceToDevice;
    } else {
      LOG(FATAL) << "Unsupported copy from " << from_dev << " to " << to_dev;
    }

    hggcStream_t hstream = nullptr;
    if (stream != nullptr) {
      hstream = *static_cast<hggcStream_t *>(stream);
    } else {
      hstream = PPUThreadEntry::ThreadLocal()->stream;
    }

    const char *from_ptr = static_cast<const char *>(from) + from_offset;
    char *to_ptr = static_cast<char *>(to) + to_offset;

    if (hstream != nullptr) {
      HGGC_RT_CALL(hggcMemcpyAsync(to_ptr, from_ptr, size, kind, hstream));
    } else {
      HGGC_RT_CALL(hggcMemcpy(to_ptr, from_ptr, size, kind));
    }
  }

  TVMStreamHandle CreateStream(Device dev) final {
    HGGC_RT_CALL(hggcSetDevice(dev.device_id));
    hggcStream_t *stream = new hggcStream_t();
    HGGC_RT_CALL(hggcStreamCreate(stream));
    return static_cast<TVMStreamHandle>(stream);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) final {
    hggcStream_t *hstream = static_cast<hggcStream_t *>(stream);
    if (*hstream != nullptr) {
      HGGC_RT_CALL(hggcStreamDestroy(*hstream));
    }
    delete hstream;
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    hggcStream_t hstream = nullptr;
    if (stream != nullptr) {
      hstream = *static_cast<hggcStream_t *>(stream);
    } else {
      hstream = PPUThreadEntry::ThreadLocal()->stream;
    }
    if (hstream != nullptr) {
      HGGC_RT_CALL(hggcStreamSynchronize(hstream));
    } else {
      HGGC_RT_CALL(hggcDeviceSynchronize());
    }
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    PPUThreadEntry *entry = PPUThreadEntry::ThreadLocal();
    if (stream != nullptr) {
      entry->stream = *static_cast<hggcStream_t *>(stream);
    } else {
      entry->stream = nullptr;
    }
  }

  void *AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return AllocDataSpace(dev, size, 0, type_hint);
  }

  void FreeWorkspace(Device dev, void *ptr) final { FreeDataSpace(dev, ptr); }

  static PPUDeviceAPI *Global() {
    static PPUDeviceAPI *inst = new PPUDeviceAPI();
    return inst;
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.ppu",
                               [](ffi::PackedArgs args, ffi::Any *rv) {
                                 DeviceAPI *ptr = PPUDeviceAPI::Global();
                                 *rv = static_cast<void *>(ptr);
                               });
}

} // namespace runtime
} // namespace tvm
