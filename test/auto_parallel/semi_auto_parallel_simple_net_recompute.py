# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
from semi_auto_parallel_simple_net import (
    DemoNet,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn


class TestSimpleNetWithRecomputeForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)
        self.init_input_data()
        self.init_single_card_net_result()

    def run_dynamic_recompute(self, layer, shard_input=False):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        image = paddle.to_tensor(self.image)
        if shard_input:
            image = dist.shard_tensor(
                image,
                dist_attr=dist.DistAttr(
                    mesh=self._mesh, sharding_specs=['x', None]
                ),
            )
        image.stop_gradient = False
        out = layer(image)

        label = paddle.to_tensor(self.label)
        loss = loss_fn(out, label)

        loss.backward()
        return loss, layer.parameters()

    def init_single_card_net_result(self):
        (
            self.base_loss,
            self.base_parameters,
        ) = self.run_dynamic_recompute(
            DemoNet("recompute_demo", is_recompute=True)
        )

    def test_dp_demo_net(self):
        (
            self.dp_loss,
            self.dp_parameters,
        ) = self.run_dynamic_recompute(
            DemoNet("recompute_dp_demo", is_recompute=True),
            shard_input=True,
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        for param, param_base in zip(self.dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp_demo_net(self):
        mp_layer = dist.shard_layer(
            DemoNet("recompute_mp_demo", is_recompute=True),
            self._mesh,
            self.shard_fn,
        )
        (
            self.mp_loss,
            self.mp_parameters,
        ) = self.run_dynamic_recompute(mp_layer)

        self.check_tensor_eq(self.mp_loss, self.base_loss)
        for param, param_base in zip(self.mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetWithRecomputeForSemiAutoParallel().run_test_case()
