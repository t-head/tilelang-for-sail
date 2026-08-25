/*!
 * \file tl/ppu/target/target_ppu.cc
 * \brief PPU: register the standalone "ppu" TVM target kind.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

namespace refl = ffi::reflection;

// PPU: standalone target kind, device type kDLPPU
TVM_REGISTER_TARGET_KIND("ppu", kDLPPU)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("arch")
    .add_attr_option<int64_t>("max_shared_memory_per_block")
    .add_attr_option<int64_t>("max_threads_per_block")
    .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
    .add_attr_option<int64_t>("registers_per_block")
    .add_attr_option<int64_t>("l2_cache_size_bytes")
    .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(1024))
    .set_default_keys({"ppu", "gpu"});

} // namespace tvm
