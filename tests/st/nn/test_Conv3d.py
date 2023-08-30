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
    def __init__(self):
        super(Net, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 3, 3), dtype=ms.float16)

    def construct(self, x):
        out = self.conv3d(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv3d_para_customed_dtype(mode):
    """
    Feature: Conv3d
    Description: Verify the result of Conv3d specifying customed para dtype.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.ones([16, 3, 10, 32, 32]), ms.float16)
    output = net(x)
    expect_output_shape = (16, 32, 10, 32, 32)
    assert np.allclose(expect_output_shape, output.shape)
