// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>

namespace {

void packed_w2(torch::Tensor x, torch::Tensor codes, torch::Tensor scales,
               torch::Tensor gscales, torch::Tensor perm, torch::Tensor ids,
               torch::Tensor offsets, int64_t topk, torch::Tensor y,
               int64_t tokens, int64_t prefill_rb) {
  constexpr int kWidth = 576;
  constexpr int kHidden = 5120;
  constexpr int kTile = 2;
  constexpr int kSplits = 8;
  TORCH_CHECK(x.is_cuda(), "packed576 W2 needs CUDA");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto* props = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(props->major == 7 && props->minor == 0, "packed576 W2 needs SM70");
  TORCH_CHECK(prefill_rb == 2 || prefill_rb == 4, "packed576 W2 needs RB2 or RB4");
  TORCH_CHECK(tokens > 0 && topk == 6 && tokens <= 65535 / topk,
              "packed576 W2 needs top-6 and at most 65535 slots");
  const int64_t slots = tokens * topk;
  const int64_t experts = gscales.numel();
  TORCH_CHECK(gscales.dim() == 1 && experts > 0 && experts <= 384,
              "packed576 W2 needs 1..384 expert scales");
  const auto check = [&x](const torch::Tensor& t, at::ScalarType type) {
    TORCH_CHECK(t.device() == x.device() && t.dtype() == type && t.is_contiguous(),
                "packed576 W2 requires colocated, contiguous tensors of the declared type");
  };
  check(x, torch::kHalf);
  check(y, torch::kHalf);
  check(codes, torch::kUInt8);
  check(scales, torch::kUInt8);
  check(gscales, torch::kFloat);
  for (const auto& t : {perm, ids, offsets}) check(t, torch::kInt);
  TORCH_CHECK(x.sizes() == at::IntArrayRef({slots, kWidth}) &&
              y.sizes() == at::IntArrayRef({slots, kHidden}),
              "packed576 W2 activation or output shape mismatch");
  TORCH_CHECK(codes.numel() == experts * kHidden * (kWidth / 2) &&
              scales.numel() == experts * kHidden * (kWidth / 32),
              "packed576 W2 weight geometry mismatch");
  TORCH_CHECK(perm.sizes() == at::IntArrayRef({slots}) &&
              ids.sizes() == at::IntArrayRef({slots}) &&
              offsets.sizes() == at::IntArrayRef({slots + 1}),
              "packed576 W2 route geometry mismatch");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0,
              "packed576 W2 requires 16-byte alignment");

  const dim3 grid(kHidden / (32 * kTile), (unsigned)std::min(slots, experts));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool prefill = (slots + experts - 1) / experts > 8;
#define C7_W2(RB)                                                             \
  skinny_nvfp4_moe_qpn<kSplits, SCALE_MXFP4_E8M0_32, RB, kTile, true>           \
      <<<grid, dim3(32 * kSplits), 0, stream>>>(                               \
          codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),               \
          gscales.data_ptr<float>(),                                           \
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),              \
          reinterpret_cast<half*>(y.data_ptr<at::Half>()),                     \
          perm.data_ptr<int>(), ids.data_ptr<int>(), offsets.data_ptr<int>(),  \
          kHidden, kWidth, static_cast<int>(topk), 1)
  if (!prefill)
    C7_W2(1);
  else if (prefill_rb == 2)
    C7_W2(2);
  else
    C7_W2(4);
#undef C7_W2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(v41qpn576, m) {
  m.def("w2(Tensor x, Tensor codes, Tensor scales, Tensor gscales, Tensor perm, "
        "Tensor ids, Tensor offsets, int topk, Tensor(a!) y, int tokens, int prefill_rb) -> ()");
}
TORCH_LIBRARY_IMPL(v41qpn576, CUDA, m) {
  m.impl("w2", &packed_w2);
}
