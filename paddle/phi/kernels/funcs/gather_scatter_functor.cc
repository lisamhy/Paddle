/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"

#include "glog/logging.h"

#include "paddle/phi/core/macros.h"

namespace phi {
namespace funcs {

class TensorAssign {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMax {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data =
        std::isnan(*src_data) ? *src_data : std::max(*self_data, *src_data);
  }
};
static ReduceMax reduce_max;

class ReduceMin {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data =
        std::isnan(*src_data) ? *src_data : std::min(*self_data, *src_data);
  }
};
static ReduceMin reduce_min;

class ReduceMultiply {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }
};
static ReduceMultiply reduce_mul;

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true,
          bool is_mean = false>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index UNUSED,
                  const phi::DenseTensor& src,
                  const std::string& method_name UNUSED,
                  const func_t& reduce_op,
                  const phi::DeviceContext& ctx UNUSED) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();
    auto* index_data = index.data<index_t>();
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    auto index_dims = index.dims();
    auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) {
      VLOG(3) << "zero size input found";
      phi::errors::InvalidArgument(
          "self_size, src_size, index_size cannot be 0");
      return;
    }

    phi::CPUContext cpu_ctx;
    auto self_cnt = Full<tensor_t, phi::CPUCOntext>(cpu_ctx, self_dims, 0);
    auto* self_cnt_data = self_cnt.data<tensor_t>();

    int64_t select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int replaced_select_dim_size =
        is_scatter_like ? self_dims[dim] : src_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }
    int64_t index_idx = 0;
    int64_t self_idx, src_idx;

    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];

          /*
            gather computation formula:

            self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
            self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
            self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

            scatter computation formula:

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

          */

          // This index might out of bound of index matrix's index, so here
          // multiply the replaced_select_dim_size.
          int64_t replace_index = k + index * outer_dim_size +
                                  i * outer_dim_size * replaced_select_dim_size;

          self_idx = is_scatter_like ? replace_index : index_idx;
          src_idx = is_scatter_like ? index_idx : replace_index;
          reduce_op((tensor_t*)(self_data + self_idx),  // NOLINT
                    (tensor_t*)(src_data + src_idx));   // NOLINT
          self_cnt_data[self_idx] += 1;
          index_idx++;
        }
      }
    }

    if (is_mean) {
      auto zeros = Full<tensor_t, phi::CPUCOntext>(cpu_ctx, self_dims, 0);
      auto ones = Full<tensor_t, phi::CPUCOntext>(cpu_ctx, self_dims, 0);
      phi::DenseTensor mask;
      EqualAllKernel<tensor_t, phi::CPUContext>(
          cpu_ctx, self_cnt, zeros, int axis, &mask);
      phi::DenseTensor cnt;
      WhereKernel<tensor_t, phi::CPUContext>(
          cpu_ctx, mask, ones, self_cnt, cnt);
      self = phi::Divide<tensor_t>(cpu_ctx, self, cnt);
    }
  }
};

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_out_cpu", tensor_assign, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_assign_cpu", tensor_assign, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_cpu", reduce_add, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true,
                             /*is_mean*/ true>()(
      self, dim, index, src, "scatter_mean_cpu", reduce_add, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_max_cpu", reduce_max, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_min_cpu", reduce_min, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_cpu", reduce_mul, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(phi::DenseTensor self UNUSED,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor output,
                                   const phi::DeviceContext& ctx UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* output_data = output.data<tensor_t>();

  auto index_dims = index.dims();
  auto output_dims = output.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t output_select_dim_size = output_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  int64_t index_idx = 0;
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index = k + index * outer_dim_size +
                                i * outer_dim_size * output_select_dim_size;
        output_data[replace_index] = 0;
        index_idx++;
      }
    }
  }
}

Instantiate_Template_Function(cpu_gather_kernel)
    Instantiate_Template_Function(cpu_scatter_assign_kernel)
        Instantiate_Template_Function(cpu_scatter_add_kernel)
            Instantiate_Template_Function(cpu_scatter_mul_kernel)
                Instantiate_Template_Function(cpu_scatter_input_grad_kernel)

}  // namespace funcs
}  // namespace phi
