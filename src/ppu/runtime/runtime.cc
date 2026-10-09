/*!
 * \file tl/ppu/runtime/runtime.cc
 * \brief PPU runtime functions.
 *
 * Registers HGGC driver API packed functions for L2 persistent cache
 * access policy window management
 */

#include <hggc.h>
#include <tvm/runtime/logging.h>

#include "ppu/runtime/runtime.h"
#include "support/check.h"

#include <cstdint>
#include <cstring>

namespace tvm {
namespace tl {

using namespace ffi;

// Thread-local storage for restoring the L2 persisting cache limit
static thread_local size_t __tl_prev_persisting_l2_cache_size = 0;
static thread_local bool __tl_prev_persisting_l2_cache_saved = false;

//
// HGGC L2 Persisting Cache Access Policy Window helpers.
// Exposed as TVM FFI packed functions, following the same pattern as other
// backends.
//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;

  // Set stream access policy window and adjust persisting L2 cache size
  // Args:
  //  [0]: void* base_ptr (required)
  //  [1]: int64 num_bytes (required)
  //  [2]: float hit_ratio (optional, default 0.8)
  //  [3]: void* stream (optional, default 0 => default stream)
  //  [4]: int64 l2_limit_bytes (optional, default = num_bytes)
  refl::GlobalDef().def_packed(
      tl::tvm_ppu_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        HGstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<HGstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        // Clamp requested limit to device capability
        HGdevice device;
        HGresult result = hgCtxGetDevice(&device);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to get current HGGC device: " << result;
        }
        int max_persisting = 0;
        result = hgDeviceGetAttribute(
            &max_persisting, HG_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        // Save current limit to restore later
        size_t init_persisting_l2_cache_size = 0;
        result = hgCtxGetLimit(&init_persisting_l2_cache_size,
                               HG_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        // Set new limit
        result =
            hgCtxSetLimit(HG_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        // Apply access policy window to stream
        HGstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            HG_ACCESS_PROPERTY_L2_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            HG_ACCESS_PROPERTY_STREAMING;

        result = hgStreamSetAttribute(stream,
                                      HG_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                                      &stream_attribute);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  // Reset stream access policy window and restore the previous L2 cache size
  // Args:
  //  [0]: void* stream (optional, default 0)
  refl::GlobalDef().def_packed(
      tl::tvm_ppu_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        HGstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<HGstream>(args[0].cast<void *>());
        }

        HGstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        // num_bytes = 0 disables the access policy window on the stream
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        HGresult result = hgStreamSetAttribute(
            stream, HG_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &stream_attribute);
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = hgCtxResetPersistingL2Cache();
        if (result != HGGC_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = hgCtxSetLimit(HG_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != HGGC_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}

} // namespace tl
} // namespace tvm
