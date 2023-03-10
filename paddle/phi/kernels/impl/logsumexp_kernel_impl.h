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

#pragma once
#include <type_traits>
#include <vector>
#include <bitset>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/logsumexp_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_amax_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/impl/compare_kernel_impl.h"
#include "paddle/phi/kernels/where_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"
#include "paddle/utils/array_ref.h"

namespace phi {

#define HANDLE_DIM(NDIM, RDIM)                                         \
  if (ndim == NDIM && rdim == RDIM) {                                  \
    funcs::ReduceFunctor<Context, T, NDIM, RDIM, LogsumexpFunctor<T>>( \
        dev_ctx, x, out, axis, keepdim);                               \
  }

template <typename T>
struct LogsumexpFunctor {
  template <typename Context, typename X, typename Y, typename Dim>
  void operator()(const Context& place, X* x, Y* y, const Dim& dim) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    auto x_dim = x->dimensions();
    auto t_dim = x_dim;
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      t_dim[dim[i]] = 1;
    }

    auto r_dim = x_dim;
    for (int i = 0; i < static_cast<int>(r_dim.size()); i++) {
      r_dim[i] = 1;
    }
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      r_dim[dim[i]] = x_dim[dim[i]];
    }

    auto x_mt = (*x).template cast<MT>();
    auto y_dim = y->dimensions();
    auto x_max = x_mt.maximum(dim).eval();
    y->device(place) =
        (x_max +
         (x_mt - x_max.reshape(t_dim).broadcast(r_dim)).exp().sum(dim).log())
            .reshape(y_dim)
            .template cast<T>();
  }
};


template <class T, class Context>
static DenseTensor Fill(const Context& ctx,
                        std::vector<int> shape,
                        float fill_value) {
  DenseTensor ret;
  ret.Resize(make_ddim(shape));
  ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(ctx, &ret, T(fill_value));
  return ret;
}


template <class T, class Context>
static DenseTensor Infinits(const Context& ctx, std::vector<int> shape) {
  auto value = static_cast<T>(std::numeric_limits<T>::infinity());
  return Fill<T, Context>(ctx, shape, value);
}

static DenseTensor Unsqueeze(const DenseTensor& x, int axis = 0) {
  // don't copy data, only change the dims
  DenseTensor out;
  out.ShareDataWith(x);
  std::vector<int> out_shape = phi::vectorize<int>(x.dims());
  if (axis >= 0) {
    auto index = (out_shape.begin() + axis);
    out_shape.insert(index, 1);
  } else if (axis < 0) {
    auto index = (out_shape.end() + axis + 1);
    out_shape.insert(index, 1);
  }
  out.Resize(phi::make_ddim(out_shape));
  return out;
}

static DenseTensor Squeeze(const DenseTensor& x, int axis = 0) {
  // don't copy data, only change the dims
  DenseTensor out;
  out.ShareDataWith(x);
  std::vector<int> out_shape = phi::vectorize<int>(x.dims());
  if (axis >= 0) {
    auto index = (out_shape.begin() + axis);
    out_shape.erase(index);
  } else if (axis < 0) {
    auto index = (out_shape.end() + axis + 1);
    out_shape.erase(index);
  }
  out.Resize(phi::make_ddim(out_shape));
  return out;
}

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t ndim,
                                            bool wrap_scalar = true) {
  if (ndim <= 0) {
    PADDLE_ENFORCE(!wrap_scalar,
        "dimension specified as ",
        dim,
        " but tensor has no dimensions");
    ndim = 1;  // this will make range [-1, 0]
  }

  int64_t min = -ndim;
  int64_t max = ndim - 1;

  PADDLE_ENFORCE(
      min <= dim && dim <= max,
      "Dimension out of range (expected to be in range of [",
      min,
      ", ",
      max,
      "], but got ",
      dim,
      ")");

  if (dim < 0) dim += ndim;
  return dim;
}

constexpr size_t dim_bitset_size = 64;

static inline std::bitset<dim_bitset_size> dim_list_to_bitset(
    IntArrayRef dims,
    int64_t ndims) {
  PADDLE_ENFORCE(
      ndims <= (int64_t)dim_bitset_size,
      "only tensors with up to ",
      dim_bitset_size,
      " dims are supported");
      
  std::bitset<dim_bitset_size> seen;
  
  for (size_t i = 0; i < dims.size(); ++i) {
    size_t dim = maybe_wrap_dim(dims[i], ndims);
    PADDLE_ENFORCE(
        !seen[dim], "dim ", dim, " appears multiple times in the list of dims");
        
    seen[dim] = true;
  }
  return seen;
}

static DenseTensor squeeze_multiple(const DenseTensor& self, IntArrayRef dims) {
  int ndims = self.dims().size();
  auto dims_to_squeeze = dim_list_to_bitset(dims, ndims);
  DenseTensor out = self;
  for (int i = ndims - 1; i >= 0; --i) {
    if (dims_to_squeeze[i]) {
      out = Squeeze(out, i);
    }
  }
  return out;
}

template <typename T, typename Context>
DenseTensor MaskedFill(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& mask,
                      const float& value) {
   PADDLE_ENFORCE(x.dims() == mask.dims(), 
      "incompatible dimensions."
      "x.shape != mask.shape, x.shape=%s, mask.shape=%s", 
      x.dims(), mask.dims()
   );
   PADDLE_ENFORCE(mask.dtype() == phi::DataType::BOOL, 
      "mask dtype must be bool, but is %s.", mask.dtype());
   auto yes = Fill<T, Context>(dev_ctx, vectorize<int>(x.dims()), value);
   auto out = Where<T,Context>(dev_ctx, mask, yes, x);
   return out;
}

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  // can't take max of empty tensor
  if(x.numel() != 0){
    // y = log(reducem_sum(exp(x - amax(x, dims)))) + amax(x, dims)
    // amax(x)
    auto maxes = AMax<T,Context>(dev_ctx, x,  axis, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, axis));


    std::vector<int> maxes_squeezed_shape = vectorize<int>(maxes_squeezed.dims());
    auto infinity = Fill<T, Context>(dev_ctx, maxes_squeezed_shape, INFINITY);

    auto abs = Abs<T, Context>(dev_ctx, maxes_squeezed);
    auto is_infinity = EqualAll<T, Context>(dev_ctx, abs, infinity);
    auto maxes_squeezed_filled = MaskedFill<T, Context>(dev_ctx, maxes_squeezed, is_infinity, 0);

    // x - amax(x)
    auto sub = Subtract<T, Context>(dev_ctx,x, maxes);

    // exp(x - amax(x))
    auto exp = Exp<T,Context>(dev_ctx, sub);
    // reduce_sum(exp(x - amax(x)))
    auto sum = Sum<T,Context>(dev_ctx, exp, axis, x.dtype(), keepdim);
    // log(exp(x - amax(x)))
    auto log = Log<T,Context>(dev_ctx, sum);
    // y = log(exp(x - amax(x))) + amax(x)
    *out = Add<T, Context>(dev_ctx, log, maxes_squeezed_filled);

    VLOG(1) << "logsumexp: " << *out;
    return;
  } else {
    auto exp = Exp<T,Context>(dev_ctx, x);
    auto sum = Sum<T,Context>(dev_ctx, exp, axis, x.dtype(), keepdim);
    *out = Log<T,Context>(dev_ctx, sum);
    VLOG(1) << "logsumexp numel=0: " << *out;
    return;
  }
  
  reduce_all = recompute_reduce_all(x, axis, reduce_all);

  if (reduce_all) {
    // Flatten and reduce 1-D tensor
    auto input = phi::EigenVector<T>::Flatten(x);
    auto output = phi::EigenScalar<T>::From(*out);
    auto& place = *dev_ctx.eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    LogsumexpFunctor<T>()(place, &input, &output, reduce_dim);
  } else {
    int ndim = x.dims().size();
    int rdim = axis.size();
    if (ndim > 4) {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported dimensions, please keep maximum dimensions of input "
          "data less than 4."));
    }

    // comments for accelerating compiling temporarily.
    // HANDLE_DIM(6, 5);
    // HANDLE_DIM(6, 4);
    // HANDLE_DIM(6, 3);
    // HANDLE_DIM(6, 2);
    // HANDLE_DIM(6, 1);
    // HANDLE_DIM(5, 4);
    // HANDLE_DIM(5, 3);
    // HANDLE_DIM(5, 2);
    // HANDLE_DIM(5, 1);
    HANDLE_DIM(4, 3);
    HANDLE_DIM(4, 2);
    HANDLE_DIM(4, 1);
    HANDLE_DIM(3, 2);
    HANDLE_DIM(3, 1);
    HANDLE_DIM(2, 1);
  }
}

}  // namespace phi
