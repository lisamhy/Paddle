// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"

#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        const DenseTensor& value,
                        int axis,
                        const std::string& reduce,
                        DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      errors::PreconditionNotMet("PutAlongAxisOpKernel only runs on CPU."));

  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  const auto& index_type = index.dtype();
  if (reduce == "add") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_add_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_add_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "multiply" || reduce == "mul") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_mul_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_mul_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "assign") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_assign_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_assign_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "min") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_min_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_min_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "max") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_max_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_max_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "mean") {
    auto self_dims = out->dims();
    auto zeros = Full<T, Context>(dev_ctx, vectorize(self_dims), 0);
    auto ones = Full<T, Context>(dev_ctx, vectorize(self_dims), 1);

    bool include_self = false;
    auto counts = include_self ? ones : zeros;

    auto src_ones = Full<T, Context>(dev_ctx, vectorize(value.dims()), 1);
    IndexAddKernel<T, Context>(dev_ctx, counts, index, src_ones, axis, &counts);

    phi::DenseTensor mask;
    EqualKernel<T, Context>(dev_ctx, counts, zeros, &mask);

    phi::DenseTensor cnt;
    WhereKernel<T, Context>(dev_ctx, mask, ones, counts, &cnt);

    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_mean_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_mean_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
    *out = phi::Divide<T>(dev_ctx, *out, cnt);
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "can not support reduce: '%s' for scatter kernel, only "
        "support reduce op: 'add', 'assign', 'mul' or 'multiply', 'amax', "
        "'amin',and 'mean', the "
        "default reduce "
        "op is 'assign' ",
        reduce));
    return;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
