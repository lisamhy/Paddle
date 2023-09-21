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
// #include "paddle/phi/kernels/gpu/unique_consecutive_functor.h"
#include "paddle/phi/kernels/bitwise_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/put_along_axis_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"

namespace phi {

// index_select() function for Tensor
template <typename Context, typename InT, typename IndexT>
void IndexSelect(const Context& context,
                 const DenseTensor& input,
                 const DenseTensor& index,
                 DenseTensor* output,
                 int dim) {
  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];

  std::vector<InT> input_vec;
  std::vector<IndexT> index_vec;
  phi::TensorToVector(input, context, &input_vec);
  phi::TensorToVector(index, context, &index_vec);
  std::vector<InT> out_vec(output->numel());

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_vec[i],
        0,
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i],
        input_dim[dim],
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_vec[i]));
  }

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_vec[j];
      for (auto k = 0; k < slice_size; k++) {
        out_vec[output_start_offset + j * slice_size + k] =
            input_vec[input_start_offset + index_value * slice_size + k];
      }
    }
  }
  context.template Alloc<InT>(output);
  phi::TensorFromVector(out_vec, context, output);
  output->Resize(output_dim);
}

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
    auto zeros = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 0);
    auto ones = Full<T, Context>(dev_ctx, vectorize(out_grad.dims()), 1);
    if (x_grad) {
      // Tensor masked_self = x.masked_fill(x == 0, 1);
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
      auto src_ones = Full<T, Context>(dev_ctx, vectorize(source.dims()), 1);
      auto src_zeros = Full<T, Context>(dev_ctx, vectorize(source.dims()), 1);
      auto src_zero = Equal<T, Context>(dev_ctx, source, src_zeros);

      // Tensor src_num_zeros = zeros_like(self).index_add(dim, index,
      // src_zero.to(self.dtype())).index_select(dim, index);
      auto src_num_zeros_inner =
          IndexAdd<T, Context>(dev_ctx, zeros, index, src_zero, axis);
      DenseTensor src_num_zeros;
      src_num_zeros.Resize(source.dims());
      dev_ctx.template Alloc<T>(&src_num_zeros);
      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx, src_num_zeros_inner, index, &src_num_zeros, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx, src_num_zeros_inner, index, &src_num_zeros, axis);
      }

      // src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);
      auto src_num_zeros_equal_one =
          Equal<T, Context>(dev_ctx, src_num_zeros, src_ones);
      auto src_single_zero =
          BitwiseAnd<T, Context>(dev_ctx, src_zero, src_num_zeros_equal_one);

      // // For src positions with src_single_zero, (grad *
      // result).index_select(dim,index) / source.masked_fill(src_zero, 1)
      // // would incorrectly propagate zeros as the gradient
      // Tensor masked_src = source.masked_fill(src_single_zero, 1);
      auto masked_src =
          Where<T, Context>(dev_ctx, src_single_zero, src_ones, source);

      // Tensor masked_src_result = x.index_reduce(dim, index, masked_src,
      // reduce, include_self);
      auto masked_src_result = PutAlongAxis(
          dev_ctx, x, index, masked_src, axis, reduce, include_self);

      // Tensor grad_src1 = where(src_single_zero,
      //                          (grad * masked_src_result).index_select(dim,
      //                          index), (grad * result).index_select(dim,
      //                          index) / source.masked_fill(src_zero, 1));
      auto grad_mul_masked_src_result =
          Multiply<T, Context>(dev_ctx, out_grad, masked_src_result);
      DenseTensor grad_mul_masked_src_result_index_select;
      grad_mul_masked_src_result_index_select.Resize(source.dims());
      dev_ctx.template Alloc<T>(&grad_mul_masked_src_result_index_select);
      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx,
            grad_mul_masked_src_result,
            index,
            &grad_mul_masked_src_result_index_select,
            axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx,
            grad_mul_masked_src_result,
            index,
            &grad_mul_masked_src_result_index_select,
            axis);
      }

      auto grad_mul_out = Multiply<T, Context>(dev_ctx, out_grad, out);
      DenseTensor grad_mul_out_index_select;
      grad_mul_out_index_select.Resize(source.dims());
      dev_ctx.template Alloc<T>(&grad_mul_out_index_select);
      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx, grad_mul_out, index, &grad_mul_out_index_select, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx, grad_mul_out, index, &grad_mul_out_index_select, axis);
      }
      auto src_masked_fill_one =
          Where<T, Context>(dev_ctx, src_zero, src_ones, source);
      auto where_2 = Divide<T, Context>(
          dev_ctx, grad_mul_out_index_select, src_masked_fill_one);

      auto grad_src1 =
          Where<T, Context>(dev_ctx,
                            src_single_zero,
                            grad_mul_masked_src_result_index_select,
                            where_2);

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
      auto tmp_ones =
          Full<T, Context>(dev_ctx, vectorize(src_num_zeros.dims()), 1);
      auto src_num_zeros_greater_one =
          GreaterThan<T, Context>(dev_ctx, src_num_zeros, tmp_ones);
      auto src_num_zeros_greater_one_any =
          Any<T, Context>(dev_ctx, src_num_zeros_greater_one, {}, false);
      if (src_num_zeros_greater_one_any.data<bool>()) {
        *value_grad = grad_src1;
      } else {
        *value_grad = grad_src1;
      }
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
      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(dev_ctx, N, index, &N_src, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(dev_ctx, N, index, &N_src, axis);
      }

      // Tensor N_src = N.gather(dim, index);

      // grad_src = grad.index_select(dim, index) / N_src;
      DenseTensor grad_src;
      grad_src.Resize(source.dims());
      dev_ctx.template Alloc<T>(&grad_src);

      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx, out_grad, index, &grad_src, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx, out_grad, index, &grad_src, axis);
      }
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
      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx, out_grad, index, value_grad, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx, out_grad, index, value_grad, axis);
      }
    }
  } else if (reduce == "amin" || reduce == "amax") {
    // Tensor value = out.index_select(axis, index);
    DenseTensor value;
    value.Resize(source.dims());
    dev_ctx.template Alloc<T>(&value);

    if (index_type == DataType::INT32) {
      IndexSelect<Context, T, int32_t>(dev_ctx, out, index, &value, axis);
    } else if (index_type == DataType::INT64) {
      IndexSelect<Context, T, int64_t>(dev_ctx, out, index, &value, axis);
    }

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

      if (index_type == DataType::INT32) {
        IndexSelect<Context, T, int32_t>(
            dev_ctx, grad_distributed, index, &src_grad_dist, axis);
      } else if (index_type == DataType::INT64) {
        IndexSelect<Context, T, int64_t>(
            dev_ctx, grad_distributed, index, &src_grad_dist, axis);
      }

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
