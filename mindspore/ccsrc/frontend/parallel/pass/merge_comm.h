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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_MERGE_COMM_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_MERGE_COMM_H_

#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace parallel {
constexpr char MERGE_COMM_RUN_ONCE_ONLY[] = "merge_comm_run_once_only";
bool MergeComm(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_MERGE_COMM_H_
