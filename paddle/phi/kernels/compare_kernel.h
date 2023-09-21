/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/binary.h"

namespace phi {

#define DECALRE_COMPARE_KERNEL(name)      \
  template <typename T, typename Context> \
  void name##Kernel(const Context& ctx,   \
                    const DenseTensor& x, \
                    const DenseTensor& y, \
                    DenseTensor* out);

DECALRE_COMPARE_KERNEL(LessThan)
DECALRE_COMPARE_KERNEL(LessEqual)
DECALRE_COMPARE_KERNEL(GreaterThan)
DECALRE_COMPARE_KERNEL(GreaterEqual)
DECALRE_COMPARE_KERNEL(Equal)
DECALRE_COMPARE_KERNEL(NotEqual)
#undef DECALRE_COMPARE_KERNEL

#define DECALRE_COMPARE_ALL_KERNEL(compare_all)  \
  template <typename T, typename Context>        \
  void compare_all##Kernel(const Context& ctx,   \
                           const DenseTensor& x, \
                           const DenseTensor& y, \
                           DenseTensor* out);

DECALRE_COMPARE_ALL_KERNEL(EqualAll)
#undef DECALRE_COMPARE_KERNEL

template <typename T, typename Context>
DenseTensor Equal(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  MetaTensor meta_x(&x);
  MetaTensor meta_y(&y);
  CompareInferMeta(meta_x, meta_y, &meta_out);
  EqualKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor GreaterThan(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  MetaTensor meta_x(&x);
  MetaTensor meta_y(&y);
  CompareInferMeta(meta_x, meta_y, &meta_out);
  GreaterThanKernel<T, Context>(dev_ctx, x, y, &dense_out);
  return dense_out;
}

}  // namespace phi
