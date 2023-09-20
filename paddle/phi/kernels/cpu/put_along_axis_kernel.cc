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
// #include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"

#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T>
std::ostream& print_tensor(std::ostream& os,
                           const phi::DenseTensor& tensor,
                           const char* tag) {
  os << tag << " ";
  auto inspect = tensor.data<T>();
  auto element_num = tensor.numel();

  os << "  - data: [";
  // Note: int8_t && uint8_t is typedf of char, ostream unable to print properly
  if (typeid(int8_t) == typeid(T) || typeid(uint8_t) == typeid(T)) {
    if (element_num > 0) {
      os << signed(inspect[0]);
      for (int j = 1; j < element_num; ++j) {
        os << " " << signed(inspect[j]);
      }
    }
  } else {
    if (element_num > 0) {
      os << inspect[0];
      for (int j = 1; j < element_num; ++j) {
        os << " " << inspect[j];
      }
    }
  }
  os << "]";
  os << std::endl;
  return os;
}

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        const DenseTensor& value,
                        int axis,
                        const std::string& reduce,
                        bool include_self,
                        DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      errors::PreconditionNotMet("PutAlongAxisOpKernel only runs on CPU."));

  auto self_dims = x.dims();
  auto zeros = Full<T, Context>(dev_ctx, vectorize(self_dims), 0);
  auto ones = Full<T, Context>(dev_ctx, vectorize(self_dims), 1);

  auto counts = include_self ? ones : zeros;
  auto src_ones = Full<T, Context>(dev_ctx, vectorize(value.dims()), 1);
  auto src_cnts = IndexAdd<T, Context>(dev_ctx, counts, index, src_ones, axis);

  auto mask = Equal<T, Context>(dev_ctx, src_cnts, zeros);

  if (!include_self) {
    T init_val;
    if (reduce == "mul" || reduce == "multiply") {
      init_val = static_cast<T>(1);
    } else if (reduce == "amin") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::max();
    } else if (reduce == "amax") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? -std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::lowest();
    } else {
      init_val = static_cast<T>(0);
    }

    auto init = Full<T, Context>(dev_ctx, vectorize(self_dims), init_val);
    dev_ctx.template Alloc<T>(out);
    *out = Where<T, Context>(dev_ctx, mask, x, init);
  } else {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

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
  } else if (reduce == "amin") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_min_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_min_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "amax") {
    if (index_type == DataType::INT32) {
      phi::funcs::cpu_scatter_max_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::cpu_scatter_max_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "mean") {
    auto cnt = Where<T, Context>(dev_ctx, mask, ones, src_cnts);

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
