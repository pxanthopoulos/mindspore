# Copyright 2024 Huawei Technologies Co., Ltd
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

'''test all finite'''
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops


class Net(nn.Cell):
    def __init__(self,):
        super(Net, self).__init__()
        self.all_finite = ops.AllFinite()

    def construct(self, x):
        output = self.all_finite(x)
        return output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_all_finite(mode):
    """
    Feature: Add all_finite ops.
    Description: test all_finite ops.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    shape1 = [128, 128]
    shape2 = [12960, 65]
    inputs = [
        Tensor(np.full(shape1, -np.inf, np.float16)),
        Tensor(np.full(shape1, 0, np.float16)),
        Tensor(np.full(shape1, 40000, np.float16)),
        Tensor(np.full(shape2, 10, np.float16)),
        Tensor(np.full(shape2, np.inf, np.float16)),
    ]
    net = Net()
    output = net(inputs)
    assert output.asnumpy() == True

    inputs1 = [
        Tensor(np.full(shape1, 100, np.float32)),
        Tensor(np.full(shape1, 0, np.float32)),
        Tensor(np.full(shape1, 40000, np.float32)),
        Tensor(np.full(shape2, 10, np.float32)),
        Tensor(np.full(shape2, 256, np.float32)),
    ]
    output1 = net(inputs1)
    assert output1.asnumpy() == False




@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_all_finite_small(mode):
    """
    Feature: Add all_finite ops.
    Description: test all_finite ops.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    net = Net()

    in1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    fp16_in = Tensor(in1, ms.float16)
    output = net(fp16_in)
    assert output == False

    fp32_in = Tensor(in1, ms.float32)
    output = net(fp32_in)
    assert output == False

    in1[5] = np.inf

    fp16_in = Tensor(in1, ms.float16)
    output = net(fp16_in)
    assert output == True

    fp32_in = Tensor(in1, ms.float32)
    output = net(fp32_in)
    assert output == True

    in2 = [2]
    
    fp16_in = Tensor(in2, ms.float16)
    output = net(fp16_in)
    assert output == False

    fp32_in = Tensor(in2, ms.float32)
    output = net(fp32_in)
    assert output == False

    in2[0] = np.nan

    fp16_in = Tensor(in2, ms.float16)
    output = net(fp16_in)
    assert output == True

    fp32_in = Tensor(in2, ms.float32)
    output = net(fp32_in)
    assert output == True
    