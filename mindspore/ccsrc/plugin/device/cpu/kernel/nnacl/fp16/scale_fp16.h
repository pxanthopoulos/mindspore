/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_SCALE_FP16_H_
#define MINDSPORE_NNACL_SCALE_FP16_H_

#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/scale.h"

#ifdef __cplusplus
extern "C" {
#endif
void DoScaleFp16(const float16_t *src, const float16_t *scale, const float16_t *bias, float16_t *out,
                 const ScaleParameter *scale_param, const int block[C2NUM]);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_SCALE_FP16_H_
