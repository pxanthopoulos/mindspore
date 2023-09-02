/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "transform/acl_ir/acl_adapter_info.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace transform {
std::string CheckConvNdSupported(TypeId data_type, const std::vector<ShapeVector> &) {
  if (data_type != kNumberTypeFloat16) {
    return kOpFormat_DEFAULT;
  }
  return kOpFormat_NC1HWC0;
}

REGISTER_ACL_OP(Conv2D).Input(0, {"NCHW"}).Input(1, {"NCHW"}).Input(2, {"NCHW"}).OutputSelector(&CheckConvNdSupported);

REGISTER_ACL_OP(Conv3D).Input(0, {"NCHW"}).Input(1, {"NCHW"}).Input(2, {"NCHW"});

REGISTER_ACL_OP(Conv2DBackpropInput)
  .Input(0, {"NCHW"})
  .Input(1, {"NCHW"})
  .Input(2, {"NCHW"})
  .OutputSelector(&CheckConvNdSupported);

REGISTER_ACL_OP(Conv3DBackpropInput).Input(0, {"NCHW"}).Input(1, {"NCHW"}).Input(2, {"NCHW"});

REGISTER_ACL_OP(Conv2DBackpropFilter).Input(0, {"NCHW"}).Input(1, {"NCHW"}).Input(2, {"NCHW"});
}  // namespace transform
}  // namespace mindspore
