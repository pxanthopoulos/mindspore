/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend platform.
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelAllFinite {
 public:
  __aicore__ inline KernelAllFinite() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum) {}
  __aicore__ inline void Process() {}
};

extern "C" __global__ __aicore__ void all_finite(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  KernelAllFinite op;
  op.Init(x, z, tilingData.totalLength, tilingData.tileNum);
  if (TILING_KEY_IS(1)) {
    op.Process();
  }
}
