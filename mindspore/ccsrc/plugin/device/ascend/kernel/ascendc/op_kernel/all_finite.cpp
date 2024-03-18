/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend platform.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t OUT_MIN_LEN = 16;

class KernelAllFinite {
 public:
  __aicore__ explicit KernelAllFinite() {}
  __aicore__ inline void setArgs(GM_ADDR in, GM_ADDR out, GM_ADDR tiling) {
    gm_x = reinterpret_cast<__gm__ uint16_t *>(in);
    gm_y = reinterpret_cast<__gm__ half *>(out);
    core_idx = get_block_idx();
    core_num = get_block_num();

    avg_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 0));
    avg_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 1));
    avg_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 2));
    avg_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 3));

    tail_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 4));
    tail_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 5));
    tail_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 6));
    tail_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 7));

    buffer_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 8));
  }

  __aicore__ inline void Process() {
    if (core_idx >= core_num) {
      return;
    }

    uint32_t ub_count = avg_block_ub_num;
    uint32_t ub_loop = avg_block_ub_loop;
    uint32_t ub_tail = avg_block_ub_tail;

    if (core_idx == core_num - 1) {
      ub_count = tail_block_ub_num;
      ub_loop = tail_block_ub_loop;
      ub_tail = tail_block_ub_tail;
    }

    Init(ub_count);

    pipe.InitBuffer(tmpQue, buffer_num, ub_count * sizeof(uint16_t));
    AscendC::LocalTensor<uint16_t> tmp_t = tmpQue.AllocTensor<uint16_t>();

    pipe.InitBuffer(maskQue, buffer_num, ub_count * sizeof(uint16_t));
    AscendC::LocalTensor<uint16_t> mask_t = maskQue.AllocTensor<uint16_t>();
    Duplicate(mask_t, (uint16_t)0x001F, ub_count);  //  0 00000 00000 11111

    pipe.InitBuffer(compQue, buffer_num, ub_count / 8 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> comp_t = compQue.AllocTensor<uint8_t>();

    uint32_t loop = 0;
    for (; loop < ub_loop - 1; loop++) {
      CopyIn(loop, ub_count, ub_count);
      Compute(ub_count, tmp_t, mask_t, comp_t, &loop);
    }

    /* for ub tail */
    if (ub_tail == 0 || loop >= ub_loop) {
      return;
    }
    CopyIn(loop, ub_count, ub_tail);
    Compute(ub_count, tmp_t, mask_t, comp_t, &loop);

    /* free tmp local tensor */
    tmpQue.FreeTensor(tmp_t);
    maskQue.FreeTensor(mask_t);
    compQue.FreeTensor(comp_t);
  }

 private:
  __aicore__ inline void Init(uint32_t count) {
    xGm.SetGlobalBuffer(gm_x + core_idx * avg_block_count);
    yGm.SetGlobalBuffer(gm_y);
    pipe.InitBuffer(xQue, buffer_num, count * sizeof(uint16_t));
  }

  __aicore__ inline void CopyIn(uint32_t idx, uint32_t stride, uint32_t count) {
    AscendC::LocalTensor<uint16_t> x = xQue.AllocTensor<uint16_t>();
    DataCopy(x, xGm[idx * stride], count);
    xQue.EnQue(x);
  }

  __aicore__ inline void CheckValid(uint32_t count, AscendC::LocalTensor<uint16_t> shift_t,
                                    AscendC::LocalTensor<uint16_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t) {
    AscendC::LocalTensor<uint16_t> in_t = xQue.DeQue<uint16_t>();

    AscendC::ShiftLeft<uint16_t>(shift_t, in_t, 1, count);
    pipe_barrier(PIPE_ALL);
    AscendC::ShiftRight<uint16_t>(shift_t, shift_t, 11, count);
    pipe_barrier(PIPE_ALL);

    xQue.FreeTensor(in_t);

    AscendC::LocalTensor<half> shift_half_t = shift_t.ReinterpretCast<half>();
    AscendC::LocalTensor<half> mask_half_t = mask_t.ReinterpretCast<half>();

    Compare(comp_t, shift_half_t, mask_half_t, AscendC::CMPMODE::EQ, count);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline void CombRes(uint32_t count, uint32_t *loop, AscendC::LocalTensor<uint8_t> comp_t,
                                 AscendC::LocalTensor<uint16_t> ui16_t) {
    AscendC::LocalTensor<half> half_comp_t = ui16_t.ReinterpretCast<half>();
    Cast(half_comp_t, comp_t, AscendC::RoundMode::CAST_NONE, count / 8);
    pipe_barrier(PIPE_ALL);

    const int mask = 128;
    int total_count = count / 8;
    int repeat = total_count / mask;

    while (repeat > 1) {
      WholeReduceSum(half_comp_t, half_comp_t, mask, repeat, 1, 1, 8);
      repeat = repeat / 128;
      total_count = total_count / 128;
      pipe_barrier(PIPE_ALL);
    }

    WholeReduceSum(half_comp_t, half_comp_t, total_count, 1, 1, 1, 8);
    pipe_barrier(PIPE_ALL);

    float result = half_comp_t.GetValue(0);
    if (result != 0) {
      ui16_t.SetValue(0, 1);
      AscendC::SetAtomicAdd<half>();
      DataCopy(yGm[0], half_comp_t, OUT_MIN_LEN);
      AscendC::SetAtomicNone();
      *loop = count;
    }
  }

  __aicore__ inline void Compute(uint32_t count, AscendC::LocalTensor<uint16_t> tmp_t,
                                 AscendC::LocalTensor<uint16_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t,
                                 uint32_t *loop) {
    CheckValid(count, tmp_t, mask_t, comp_t);
    CombRes(count, loop, comp_t, tmp_t);
  }

  AscendC::TPipe pipe;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> tmpQue, maskQue, compQue;

  AscendC::GlobalTensor<uint16_t> xGm;
  AscendC::GlobalTensor<half> yGm;

  __gm__ uint16_t *__restrict__ gm_x{nullptr};
  __gm__ half *__restrict__ gm_y{nullptr};

  uint32_t core_idx{0};
  uint32_t core_num{0};

  uint32_t buffer_num{0};

  uint32_t avg_block_count{0};
  uint32_t avg_block_ub_num{0};
  uint32_t avg_block_ub_tail{0};
  uint32_t avg_block_ub_loop{0};

  uint32_t tail_block_count{0};
  uint32_t tail_block_ub_num{0};
  uint32_t tail_block_ub_tail{0};
  uint32_t tail_block_ub_loop{0};
};

extern "C" __global__ __aicore__ void all_finite(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
  KernelAllFinite op;
  op.setArgs(x, z, tiling);
  op.Process();
}

