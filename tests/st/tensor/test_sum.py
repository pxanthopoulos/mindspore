# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, axis=None, dtype=None, keepdims=False, initial=None):
        super(Net, self).__init__()
        self.axis = axis
        self.dtype = dtype
        self.keepdims = keepdims
        self.initial = initial

    def construct(self, x):
        return x.sum(self.axis, self.dtype, self.keepdims, self.initial)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sum(mode):
    """
    Feature: tensor.sum
    Description: Verify the result of sum
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[1, 2, 3], [2, 3, 4]]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = 15.0
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(axis=[0, 1], keepdims=True)
    output = net(x)
    expect_output = [[[3., 5., 7.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(axis=(2,), keepdims=False)
    output = net(x)
    expect_output = [[6., 9.]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(axis=2, keepdims=True)
    output = net(x)
    expect_output = [[[6.], [9.]]]
    assert np.allclose(output.asnumpy(), expect_output)

    net = Net(dtype=ms.bool_, initial=12)
    output = net(x)
    expect_output = True
    assert np.allclose(output.asnumpy(), expect_output)
