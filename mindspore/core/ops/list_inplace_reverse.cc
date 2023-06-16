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

#include "ops/list_inplace_reverse.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListInplaceReverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  constexpr size_t input_len = 1;
  constexpr size_t data_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           prim_name);
  auto data_abs = dyn_cast<abstract::AbstractList>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);

  abstract::AbstractListPtr ret;
  if (data_abs->dynamic_len()) {
    MS_LOG(INTERNAL_EXCEPTION) << "ListInplaceReverse do not support dynamic length list input.";
  }
  const auto &elements = data_abs->elements();
  abstract::AbstractBasePtrList new_elements;
  for (int i = elements.size() - 1; i >= 0; --i) {
    (void)new_elements.emplace_back(elements[i]);
  }
  ret = std::make_shared<abstract::AbstractList>(new_elements);

  if (data_abs->has_list_py_obj()) {
    ret = AbstractBroaden(ret)->cast<abstract::AbstractListPtr>();
    ret->set_list_user_data(data_abs->list_user_data());
  }

  return ret;
}
MIND_API_OPERATOR_IMPL(ListInplaceReverse, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ListInplaceReverse, prim::kPrimListInplaceReverse, ListInplaceReverseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
