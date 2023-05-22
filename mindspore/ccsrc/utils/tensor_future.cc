/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "include/common/utils/tensor_future.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace pynative {
DeviceAddressFuture::~DeviceAddressFuture() {
  if (future_.valid()) {
    try {
      (void)future_.get();
    } catch (...) {
      MS_LOG(INFO) << "Find error and ignore when destroy future";
    }
  }
}

std::shared_ptr<DeviceSync> DeviceAddressFuture::Get() {
  if (future_.valid()) {
    GilReleaseWithCheck gil_release;
    auto future_data = future_.get();
    if (future_data->GetException() != nullptr) {
      std::rethrow_exception(future_data->GetException());
    }
    future_data_ = future_data;
  }
  if (future_data_ != nullptr) {
    return future_data_->GetData();
  } else {
    return nullptr;
  }
}
}  // namespace pynative
}  // namespace mindspore
