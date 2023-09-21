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

#include "paddle/phi/kernels/put_along_axis_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/gpu/unique_consecutive_functor.h"

#include "paddle/phi/kernels/put_along_axis_kernel.h"

namespace phi {

template <typename T, typename Context>
void PutAlongAxisGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& index,
                            const DenseTensor& source,
                            const DenseTensor& out,
                            const DenseTensor& out_grad,
                            int axis,
                            const std::string& reduce,
                            bool include_self,
                            DenseTensor* x_grad,
                            DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      errors::PreconditionNotMet("PutAlongAxisGradOpKernel only runs on CPU."));

  const auto& index_type = index.dtype();

  x_grad->Resize(x.dims());
  dev_ctx.template Alloc<T>(x_grad);
  value_grad->Resize(index.dims());
  dev_ctx.template Alloc<T>(value_grad);

  if (reduce == "mul" || reduce == "multiply") {
    if (x_grad) {
      // Tensor masked_self = x.masked_fill(x == 0, 1);
      auto zeros = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 0);
      auto ones = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 1);
      auto mask = Equal<T, Context>(dev_ctx, x, zeros);
      auto masked_self = Where<T, Context>(dev_ctx, mask, ones, x);

      // Tensor masked_self_result = masked_self.index_reduce(dim, index,
      // source, reduce, include_self);
      auto masked_self_result =
          PutAlongAxis(dev_ctx, x, index, source, axis, reduce, include_self);

      // grad_self = grad * masked_self_result / masked_self;
      grad_mul_masked_self_result =
          Multiply<T, Context>(dev_ctx, out_grad, masked_self_result);
      *x_grad =
          Divide<T, Context>(dev_ctx, grad_mul_masked_self_result, masked_self);
    }

    if (value_grad) {
      // Tensor src_zero = source == 0;
      // Tensor src_num_zeros = zeros_like(x).index_add(dim, index,
      // src_zero.to(x.dtype())).index_select(dim, index); Tensor
      // src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);
      // // For src positions with src_single_zero, (grad *
      // result).index_select(dim,index) / source.masked_fill(src_zero, 1)
      // // would incorrectly propagate zeros as the gradient
      // Tensor masked_src = source.masked_fill(src_single_zero, 1);

      // Tensor masked_src_result = x.index_reduce(dim, index, masked_src,
      // reduce, include_self);
      if (index_type == DataType::INT32) {
        phi::funcs::cpu_gather_kernel<T, int32_t>(
            out_grad, axis, index, *value_grad, dev_ctx);
      } else if (index_type == DataType::INT64) {
        phi::funcs::cpu_gather_kernel<T, int64_t>(
            out_grad, axis, index, *value_grad, dev_ctx);
      }

      // Tensor grad_src1 = where(src_single_zero,
      //                          (grad * masked_src_result).index_select(dim,
      //                          index), (grad * result).index_select(dim,
      //                          index) / source.masked_fill(src_zero, 1));
      // if ((src_num_zeros > 1).any().item<bool>()) {
      //   auto node = std::make_shared<DelayedError>(
      //     "index_reduce(): Double backward is unsupported for source when >1
      //     zeros in source are scattered to the same position in x",
      //     /* num inputs */ 1);
      //   auto result = node->apply({ grad_src1 });
      //   grad_src = result[0];
      // } else {
      //   grad_src = grad_src1;
      // }
    }

  } else if (reduce == "mean") {
    // Tensor N = include_self ? ones_like(out_grad) : zeros_like(out_grad);
    auto zeros = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 0);
    auto ones = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 1);
    auto counts = include_self ? ones : zeros;

    // N = N.index_add(dim, index, ones_like(source));
    auto src_ones = Full<T, Context>(dev_ctx, vectorize(index.dims()), 1);
    auto src_cnts =
        IndexAdd<T, Context>(dev_ctx, counts, index, src_ones, axis);
    // N = N.scatter_add(dim, index, ones_like(src));

    // N.masked_fill_(N == 0, 1);
    auto mask = Equal<T, Context>(dev_ctx, src_cnts, zeros);

    auto N = Where<T, Context>(dev_ctx, mask, ones, src_cnts);

    if (x_grad) {
      // grad_self = grad / N;
      *x_grad = Divide<T, Context>(dev_ctx, out_grad, N);
    }

    if (value_grad) {
      // Tensor N_src = N.index_select(dim, index);
      DenseTensor N_src;
      N_src.Resize(source.dims());
      dev_ctx.template Alloc<T>(&N_src);
      IndexSelect<T, Context>(dev_ctx, N, index, &N_src, axis);
      // Tensor N_src = N.gather(dim, index);

      // grad_src = grad.index_select(dim, index) / N_src;
      DenseTensor grad_src;
      grad_src.Resize(source.dims());
      dev_ctx.template Alloc<T>(&grad_src);
      IndexSelect<T, Context>(dev_ctx, out_grad, index, &grad_src, axis);
      // grad_src = grad.gather(dim, index) / N_src;

      *value_grad = Divide<T, Context>(dev_ctx, grad_src, N_src);
    }

  } else if (reduce == "add") {
    if (x_grad) {
      if (include_self) {
        phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
      } else {
        *x_grad = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 0);
      }
    }

    if (value_grad) {
      IndexSelect<T, Context>(dev_ctx, out_grad, index, value_grad, axis);
    }
  } else if (reduce == "amin" || reduce == "amax") {
    // Tensor value = out.index_select(axis, index);
    DenseTensor value;
    value.Resize(source.dims());
    dev_ctx.template Alloc<T>(&value);
    IndexSelect<T, Context>(dev_ctx, out, index, &value, axis);
    // Tensor value = result.gather(dim, index);

    // Tensor self_is_result = (x == out).to(x.scalar_type());
    auto self_is_result = Equal<T, Context>(dev_ctx, x, out);

    // Tensor source_is_result = (source == value).to(x.scalar_type());
    auto source_is_result = Equal<T, Context>(dev_ctx, source, value);

    // Tensor N_to_distribute = self_is_result.index_add(axis, index,
    // source_is_result);
    auto N_to_distribute = IndexAdd<T, Context>(
        dev_ctx, self_is_result, index, source_is_result, axis);
    // Tensor N_to_distribute = self_is_result.scatter_add(dim, index,
    // src_is_result);

    // Tensor grad_distributed = grad / N_to_distribute;
    auto grad_distributed =
        Divide<T, Context>(dev_ctx, out_grad, N_to_distribute);

    if (x_grad) {
      //  grad_self = self_is_result * grad_distributed;
      *x_grad = Multiply<T, Context>(dev_ctx, self_is_result, grad_distributed);
    }

    if (value_grad) {
      // grad_src = source_is_result * grad_distributed.index_select(axis,
      // index);
      DenseTensor src_grad_dist;
      src_grad_dist.Resize(source.dims());
      dev_ctx.template Alloc<T>(&src_grad_dist);
      IndexSelect<T, Context>(
          dev_ctx, grad_distributed, index, &src_grad_dist, axis);
      //  grad_src = (src == value) * grad_distributed.gather(dim, index);

      *value_grad =
          Multiply<T, Context>(dev_ctx, source_is_result, src_grad_dist);
    }
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

  if (!include_self && x_grad) {
    // grad_self = grad_self.index_fill(axis, index, 0);
    auto self_dims = out_grad.dims();
    auto zeros = Full<T, Context>(dev_ctx, vectorize(self_dims), 0);
    auto ones = Full<T, Context>(dev_ctx, vectorize(self_dims), 1);

    auto src_ones = Full<T, Context>(dev_ctx, vectorize(index.dims()), 1);
    auto src_cnts = IndexAdd<T, Context>(dev_ctx, zeros, index, src_ones, axis);

    auto mask = Equal<T, Context>(dev_ctx, src_cnts, zeros);

    *x_grad = Where<T, Context>(dev_ctx, mask, out_grad, zeros);

    // grad_self = grad_self.scatter(dim, index, 0);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisGradKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t) {}
