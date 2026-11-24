// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Register-resident expansion and collapse adapted from 1CatAI/1Cat-vLLM PR 1588.
// The coefficient projection preserves the 9Z Triton reduction graph.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace v41mhc {

constexpr int H = 5120;
constexpr int Streams = 4;
constexpr int Mixes = 24;
constexpr int ExpandThreads = 256;
constexpr int VectorsPerThread = H / (4 * ExpandThreads);
constexpr int TileColumns = 256;
constexpr int CollapseTiles = H / TileColumns;
constexpr int LegacyWarpsPerTile = 4;
static_assert(H == 4 * ExpandThreads * VectorsPerThread);
static_assert(CollapseTiles <= 32);
static_assert(2 * (ExpandThreads / 32) * VectorsPerThread ==
              CollapseTiles * LegacyWarpsPerTile);

struct alignas(8) Half4 {
  half2 low;
  half2 high;
};

__device__ __forceinline__ float4 load4(const float* ptr) {
  return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ float4 load4(const half* ptr) {
  const Half4 values = *reinterpret_cast<const Half4*>(ptr);
  const float2 lo = __half22float2(values.low);
  const float2 hi = __half22float2(values.high);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ float4 weighted_sum(
    float a, float4 x, float b, float4 y, float c, float4 z, float d, float4 w) {
  return make_float4(a * x.x + b * y.x + c * z.x + d * w.x,
                     a * x.y + b * y.y + c * z.y + d * w.y,
                     a * x.z + b * y.z + c * z.z + d * w.z,
                     a * x.w + b * y.w + c * z.w + d * w.w);
}

// A physical warp's 128 consecutive columns cover two 9Z virtual warps.
// Each virtual lane owns an adjacent pair. Physical lanes carry two such
// pairs, so virtual XORs 16, 8, 4, 2 become physical XORs 8, 4, 2, 1.
// Adding the two tracks implements the virtual XOR 1.
__device__ __forceinline__ float legacy_collapse_warp(float4 c) {
  float left = c.x * c.x + c.y * c.y;
  float right = c.z * c.z + c.w * c.w;
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    left += __shfl_xor_sync(0xffffffffu, left, offset);
    right += __shfl_xor_sync(0xffffffffu, right, offset);
  }
  return left + right;
}

__device__ __forceinline__ float legacy_collapse_inverse(float sum, float eps) {
  // Match the 9Z Triton SM70 lowering, including division and FTZ semantics.
  float mean;
  float inverse;
  asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(float(H)));
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean + eps));
  return inverse;
}

// A CTA owns the entire row. Each thread retains five float4 collapse
// fragments until the CTA has reduced the collapse squared norm.
template <typename Y, typename W>
__global__ void __launch_bounds__(ExpandThreads) expand_norm_kernel(
    half* __restrict__ out, float* __restrict__ residual,
    const Y* __restrict__ y, const float* __restrict__ post,
    const float* __restrict__ comb, const float* __restrict__ collapse_pre,
    const W* __restrict__ weight, float eps) {
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  __shared__ float coefficients[Mixes];
  __shared__ float collapse_warp_sq[CollapseTiles * LegacyWarpsPerTile];
  __shared__ float inv_rms;

  if (tid < 16) {
    coefficients[tid] = comb[row * 16 + tid];
  } else if (tid < 20) {
    coefficients[tid] = post[row * 4 + tid - 16];
  } else if (tid < 24) {
    coefficients[tid] = collapse_pre[row * 4 + tid - 20];
  }
  __syncthreads();

  float* r = residual + row * Streams * H;
  float4 collapse[VectorsPerThread];
#pragma unroll
  for (int v = 0; v < VectorsPerThread; ++v) {
    const int col = 4 * (tid + v * ExpandThreads);
    const float4 update = load4(y + row * H + col);
    const float4 r0 = load4(r + col);
    const float4 r1 = load4(r + H + col);
    const float4 r2 = load4(r + 2 * H + col);
    const float4 r3 = load4(r + 3 * H + col);
    float4 next[Streams];
#pragma unroll
    for (int dst = 0; dst < Streams; ++dst) {
      const float4 sum = weighted_sum(
          coefficients[dst], r0, coefficients[4 + dst], r1,
          coefficients[8 + dst], r2, coefficients[12 + dst], r3);
      const float p = coefficients[16 + dst];
      const float4 n = make_float4(sum.x + p * update.x, sum.y + p * update.y,
                                   sum.z + p * update.z, sum.w + p * update.w);
      next[dst] = n;
      *reinterpret_cast<float4*>(r + dst * H + col) = n;
    }
    const float4 c = weighted_sum(coefficients[20], next[0], coefficients[21], next[1],
                                  coefficients[22], next[2], coefficients[23], next[3]);
    collapse[v] = c;
    const float partial = legacy_collapse_warp(c);
    if ((lane & 15) == 0) {
      const int first_virtual_warp = 2 * (v * (ExpandThreads / 32) + warp);
      collapse_warp_sq[first_virtual_warp + lane / 16] = partial;
    }
  }

  __syncthreads();
  if (warp == 0) {
    float partial = 0.0f;
    if (lane < CollapseTiles) {
      const float* p = collapse_warp_sq + lane * LegacyWarpsPerTile;
      // Descending XOR 2, then 1 within each 9Z tile.
      partial = (p[0] + p[2]) + (p[1] + p[3]);
    }
    float total = 0.0f;
#pragma unroll
    for (int tile = 0; tile < CollapseTiles; ++tile) {
      total += __shfl_sync(0xffffffffu, partial, tile);
    }
    if (lane == 0) {
      inv_rms = legacy_collapse_inverse(total, eps);
    }
  }
  __syncthreads();

#pragma unroll
  for (int v = 0; v < VectorsPerThread; ++v) {
    const int col = 4 * (tid + v * ExpandThreads);
    const float4 w = load4(weight + col);
    const float4 c = collapse[v];
    const Half4 output{
        __floats2half2_rn(c.x * inv_rms * w.x, c.y * inv_rms * w.y),
        __floats2half2_rn(c.z * inv_rms * w.z, c.w * inv_rms * w.w)};
    *reinterpret_cast<Half4*>(out + row * H + col) = output;
  }
}

constexpr int ProjectionRows = 8;
constexpr int ProjectionThreads = TileColumns / 2;
constexpr int Partials = Mixes + 1;
static_assert(ProjectionThreads == LegacyWarpsPerTile * 32);

__device__ __forceinline__ float2 column_pair(const float* ptr) {
  return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ float pair_dot(const float2 (&a)[Streams],
                                         const float2 (&b)[Streams]) {
  const float even = a[0].x * b[0].x + a[1].x * b[1].x +
                     a[2].x * b[2].x + a[3].x * b[3].x;
  const float odd = a[0].y * b[0].y + a[1].y * b[1].y +
                    a[2].y * b[2].y + a[3].y * b[3].y;
  return even + odd;
}

__device__ __forceinline__ float projection_warp_sum(float pair) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    pair += __shfl_xor_sync(0xffffffffu, pair, offset);
  }
  return pair;
}

// Match 9Z's FP32 reduction for each 256-column partial. Each thread owns
// the same two adjacent columns as Triton, reusing weights across rows.
template <bool AlignedWeights>
__global__ void __launch_bounds__(ProjectionThreads) coefficient_partials_kernel(
    const float* __restrict__ residual, const float* __restrict__ fn,
    float* __restrict__ dots, float* __restrict__ squares, int rows) {
  const int tile = blockIdx.x % CollapseTiles;
  const int first_row = blockIdx.x / CollapseTiles * ProjectionRows;
  const int tid = threadIdx.x;
  const int col = tile * TileColumns + 2 * tid;
  const int warp = tid / 32;
  const int lane = tid % 32;
  __shared__ float warp_partials[LegacyWarpsPerTile][ProjectionRows * Partials];

  float2 n[ProjectionRows][Streams];
#pragma unroll
  for (int r = 0; r < ProjectionRows; ++r) {
    const int row = first_row + r;
    const size_t row_column = size_t(row) * Streams * H + col;
#pragma unroll
    for (int s = 0; s < Streams; ++s) {
      n[r][s] = row < rows ? column_pair(residual + row_column + s * H)
                           : make_float2(0.0f, 0.0f);
    }
    const float sum = projection_warp_sum(pair_dot(n[r], n[r]));
    if (lane == 0) {
      warp_partials[warp][r * Partials + Mixes] = sum;
    }
  }

#pragma unroll 2
  for (int m = 0; m < Mixes; ++m) {
    const float* wcol = fn + m * Streams * H + col;
    float2 w[Streams];
#pragma unroll
    for (int s = 0; s < Streams; ++s) {
      const float* pair = wcol + s * H;
      if constexpr (AlignedWeights) {
        w[s] = column_pair(pair);
      } else {
        w[s] = make_float2(pair[0], pair[1]);
      }
    }
#pragma unroll
    for (int r = 0; r < ProjectionRows; ++r) {
      const float sum = projection_warp_sum(pair_dot(w, n[r]));
      if (lane == 0) {
        warp_partials[warp][r * Partials + m] = sum;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < ProjectionRows * Partials; idx += ProjectionThreads) {
    const int row = first_row + idx / Partials;
    const int mix = idx % Partials;
    if (row < rows) {
      // Virtual warp reduction: descending XOR 2 then 1.
      const float sum = (warp_partials[0][idx] + warp_partials[2][idx]) +
                        (warp_partials[1][idx] + warp_partials[3][idx]);
      const size_t partial = size_t(row) * CollapseTiles + tile;
      if (mix == Mixes) {
        squares[partial] = sum;
      } else {
        dots[partial * Mixes + mix] = sum;
      }
    }
  }
}

}  // namespace v41mhc

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <limits>

namespace v41mhc {
namespace {

void check_tensor(const at::Tensor& x, const at::Tensor& owner,
                  at::IntArrayRef shape, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(x.device() == owner.device() && x.is_cuda() && x.sizes() == shape &&
                  x.scalar_type() == dtype && x.is_contiguous(),
              "v41mhc: invalid ", name, " shape, dtype, layout or device");
}

void check_vector_input(const at::Tensor& x, const at::Tensor& owner,
                        at::IntArrayRef shape, const char* name) {
  const auto dtype = x.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "v41mhc: ", name, " must use FP16 or FP32");
  check_tensor(x, owner, shape, dtype, name);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % (4 * x.element_size()) == 0,
              "v41mhc: ", name, " must be aligned to four elements");
}

template <typename Y, typename W>
void launch_expand(const at::Tensor& out, const at::Tensor& residual,
                   const at::Tensor& y, const at::Tensor& post,
                   const at::Tensor& comb, const at::Tensor& collapse_pre,
                   const at::Tensor& weight, float eps,
                   cudaStream_t stream) {
  expand_norm_kernel<Y, W><<<residual.size(0), ExpandThreads, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      residual.data_ptr<float>(), static_cast<const Y*>(y.const_data_ptr()),
      post.const_data_ptr<float>(), comb.const_data_ptr<float>(),
      collapse_pre.const_data_ptr<float>(), static_cast<const W*>(weight.const_data_ptr()),
      eps);
}

void expand_norm(at::Tensor out, at::Tensor residual, const at::Tensor& y,
                 const at::Tensor& post, const at::Tensor& comb,
                 const at::Tensor& collapse_pre, const at::Tensor& weight,
                 double eps) {
  const auto rows = residual.size(0);
  TORCH_CHECK(0 < rows && rows <= std::numeric_limits<int>::max(),
              "v41mhc: invalid expand row count");
  check_vector_input(residual, residual, {rows, Streams, H}, "residual");
  check_tensor(residual, residual, {rows, Streams, H}, at::kFloat, "residual");
  check_vector_input(out, residual, {rows, H}, "out");
  check_tensor(out, residual, {rows, H}, at::kHalf, "out");
  check_vector_input(y, residual, {rows, H}, "y");
  check_vector_input(weight, residual, {H}, "weight");
  check_tensor(post, residual, {rows, Streams}, at::kFloat, "post");
  check_tensor(comb, residual, {rows, Streams, Streams}, at::kFloat, "comb");
  check_tensor(collapse_pre, residual, {rows, Streams}, at::kFloat, "collapse_pre");
  const c10::cuda::CUDAGuard guard(residual.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(residual.get_device());
  if (y.scalar_type() == at::kHalf) {
    if (weight.scalar_type() == at::kHalf) {
      launch_expand<half, half>(out, residual, y, post, comb, collapse_pre, weight,
                                eps, stream);
    } else {
      launch_expand<half, float>(out, residual, y, post, comb, collapse_pre, weight,
                                 eps, stream);
    }
  } else if (weight.scalar_type() == at::kHalf) {
    launch_expand<float, half>(out, residual, y, post, comb, collapse_pre, weight,
                               eps, stream);
  } else {
    launch_expand<float, float>(out, residual, y, post, comb, collapse_pre, weight,
                                eps, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void coefficient_partials(const at::Tensor& residual, const at::Tensor& fn,
                          at::Tensor dots, at::Tensor squares) {
  const auto rows = residual.size(0);
  TORCH_CHECK(0 < rows &&
                  rows <= (std::numeric_limits<int>::max() / CollapseTiles) * ProjectionRows,
              "v41mhc: invalid projection row count");
  check_vector_input(residual, residual, {rows, Streams, H}, "residual");
  check_tensor(residual, residual, {rows, Streams, H}, at::kFloat, "residual");
  check_tensor(fn, residual, {Mixes, Streams * H}, at::kFloat, "fn");
  check_tensor(dots, residual, {rows, CollapseTiles, Mixes}, at::kFloat, "dots");
  check_tensor(squares, residual, {rows, CollapseTiles}, at::kFloat, "squares");
  const c10::cuda::CUDAGuard guard(residual.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(residual.get_device());
  const int blocks = ((rows + ProjectionRows - 1) / ProjectionRows) * CollapseTiles;
  const float* weights = fn.const_data_ptr<float>();
  if (reinterpret_cast<uintptr_t>(weights) % alignof(float2) == 0) {
    coefficient_partials_kernel<true><<<blocks, ProjectionThreads, 0, stream>>>(
        residual.const_data_ptr<float>(), weights, dots.data_ptr<float>(),
        squares.data_ptr<float>(), rows);
  } else {
    coefficient_partials_kernel<false><<<blocks, ProjectionThreads, 0, stream>>>(
        residual.const_data_ptr<float>(), weights, dots.data_ptr<float>(),
        squares.data_ptr<float>(), rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(v41mhc, m) {
  m.def("expand_norm(Tensor(a!) out, Tensor(b!) residual, Tensor y, "
        "Tensor post, Tensor comb, Tensor collapse_pre, Tensor weight, "
        "float eps) -> ()");
  m.def("coefficient_partials(Tensor residual, Tensor fn, Tensor(a!) dots, "
        "Tensor(b!) squares) -> ()");
}

TORCH_LIBRARY_IMPL(v41mhc, CUDA, m) {
  m.impl("expand_norm", &expand_norm);
  m.impl("coefficient_partials", &coefficient_partials);
}
}  // namespace v41mhc
