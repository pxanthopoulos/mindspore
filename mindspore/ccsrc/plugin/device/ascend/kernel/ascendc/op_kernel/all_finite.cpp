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

template <typename IN_TYPE>
class KernelAllFinite {
 public:
  __aicore__ explicit KernelAllFinite() {}
  __aicore__ inline void setArgs(GM_ADDR in, GM_ADDR out) {
    gm_x = reinterpret_cast<__gm__ IN_TYPE *>(in);
    gm_y = reinterpret_cast<__gm__ half *>(out);
    core_idx = get_block_idx();
    core_num = get_block_num();
  }
  __aicore__ inline void setTiling(uint32_t avg_block_count_in, uint32_t avg_block_ub_num_in,
                                   uint32_t avg_block_ub_tail_in, uint32_t avg_block_ub_loop_in,
                                   uint32_t tail_block_count_in, uint32_t tail_block_ub_num_in,
                                   uint32_t tail_block_ub_tail_in, uint32_t tail_block_ub_loop_in,
                                   uint32_t buffer_num_in, uint32_t in_dtype_in) {
    avg_block_count = avg_block_count_in;
    avg_block_ub_num = avg_block_ub_num_in;
    avg_block_ub_tail = avg_block_ub_tail_in;
    avg_block_ub_loop = avg_block_ub_loop_in;
    tail_block_count = tail_block_count_in;
    tail_block_ub_num = tail_block_ub_num_in;
    tail_block_ub_tail = tail_block_ub_tail_in;
    tail_block_ub_loop = tail_block_ub_loop_in;
    buffer_num = buffer_num_in;
    in_dtype = in_dtype_in;
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
    if (in_dtype == 1) {
      ProcessHalf(ub_count, ub_tail, ub_loop);
    } else if (in_dtype == 0) {
      ProcessFp32(ub_count, ub_tail, ub_loop);
    }
  }

 private:
  __aicore__ inline void ProcessHalf(uint32_t ub_count, uint32_t ub_tail, uint32_t ub_loop) {
    AscendC::LocalTensor<uint8_t> comp_t = compQue.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint16_t> tmp_t = tmpQue.AllocTensor<uint16_t>();
    AscendC::LocalTensor<uint16_t> mask_t = maskQue.AllocTensor<uint16_t>();
    Duplicate(mask_t, (uint16_t)0x001F, ub_count);  //  0 00000 00000 11111

    uint32_t loop = 0;
    for (; loop < ub_loop - 1; loop++) {
      CopyIn(loop, ub_count, ub_count);
      ComputeHalf(ub_count, tmp_t, mask_t, comp_t, &loop);
    }

    /* for ub tail */
    if (ub_tail == 0 || loop >= ub_loop) {
      return;
    }
    CopyIn(loop, ub_count, ub_tail);
    ComputeHalf(ub_count, tmp_t, mask_t, comp_t, &loop);

    /* free tmp local tensor */
    tmpQue.FreeTensor(tmp_t);
    maskQue.FreeTensor(mask_t);
    compQue.FreeTensor(comp_t);
  }

  __aicore__ inline void ProcessFp32(uint32_t ub_count, uint32_t ub_tail, uint32_t ub_loop) {
    AscendC::LocalTensor<uint8_t> comp_t = compQue.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint32_t> tmp_t = tmpQue.AllocTensor<uint32_t>();
    AscendC::LocalTensor<uint32_t> mask_t = maskQue.AllocTensor<uint32_t>();
    Duplicate(mask_t, (uint32_t)0x00FF, ub_count);  //  0 00000000 000 0000 0000 0000 1111 1111

    uint32_t loop = 0;
    for (; loop < ub_loop - 1; loop++) {
      CopyIn(loop, ub_count, ub_count);
      ComputeFp32(ub_count, tmp_t, mask_t, comp_t, &loop);
    }

    /* for ub tail */
    if (ub_tail == 0 || loop >= ub_loop) {
      return;
    }
    CopyIn(loop, ub_count, ub_tail);
    ComputeFp32(ub_count, tmp_t, mask_t, comp_t, &loop);

    /* free tmp local tensor */
    tmpQue.FreeTensor(tmp_t);
    maskQue.FreeTensor(mask_t);
    compQue.FreeTensor(comp_t);
  }

  __aicore__ inline void Init(uint32_t count) {
    xGm.SetGlobalBuffer(gm_x + core_idx * avg_block_count);
    yGm.SetGlobalBuffer(gm_y);
    pipe.InitBuffer(xQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(tmpQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(maskQue, buffer_num, count * sizeof(IN_TYPE));
    pipe.InitBuffer(compQue, buffer_num, count / 8 * sizeof(uint8_t));
  }

  __aicore__ inline void CopyIn(uint32_t idx, uint32_t stride, uint32_t count) {
    AscendC::LocalTensor<IN_TYPE> x = xQue.AllocTensor<IN_TYPE>();
    DataCopy(x, xGm[idx * stride], count);
    xQue.EnQue(x);
  }

  __aicore__ inline void CheckValidHalf(uint32_t count, AscendC::LocalTensor<uint16_t> shift_t,
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

  __aicore__ inline void CheckValidFp32(uint32_t count, AscendC::LocalTensor<uint32_t> shift_t,
                                        AscendC::LocalTensor<uint32_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t) {
    AscendC::LocalTensor<uint32_t> in_t = xQue.DeQue<uint32_t>();

    AscendC::ShiftLeft<uint32_t>(shift_t, in_t, 1, count);
    pipe_barrier(PIPE_ALL);
    AscendC::ShiftRight<uint32_t>(shift_t, shift_t, 24, count);
    pipe_barrier(PIPE_ALL);

    xQue.FreeTensor(in_t);

    AscendC::LocalTensor<float> shift_fp32_t = shift_t.ReinterpretCast<float>();
    AscendC::LocalTensor<float> mask_fp32_t = mask_t.ReinterpretCast<float>();

    Compare(comp_t, shift_fp32_t, mask_fp32_t, AscendC::CMPMODE::EQ, count);
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

  __aicore__ inline void ComputeHalf(uint32_t count, AscendC::LocalTensor<uint16_t> tmp_t,
                                     AscendC::LocalTensor<uint16_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t,
                                     uint32_t *loop) {
    CheckValidHalf(count, tmp_t, mask_t, comp_t);
    CombRes(count, loop, comp_t, tmp_t);
  }

  __aicore__ inline void ComputeFp32(uint32_t count, AscendC::LocalTensor<uint32_t> tmp_t,
                                     AscendC::LocalTensor<uint32_t> mask_t, AscendC::LocalTensor<uint8_t> comp_t,
                                     uint32_t *loop) {
    CheckValidFp32(count, tmp_t, mask_t, comp_t);
    CombRes(count, loop, comp_t, tmp_t.ReinterpretCast<uint16_t>());
  }

  AscendC::TPipe pipe;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> tmpQue, maskQue, compQue;

  AscendC::GlobalTensor<IN_TYPE> xGm;
  AscendC::GlobalTensor<half> yGm;

  __gm__ IN_TYPE *__restrict__ gm_x{nullptr};
  __gm__ half *__restrict__ gm_y{nullptr};

  uint32_t core_idx{0};
  uint32_t core_num{0};

  uint32_t buffer_num{0};
  uint32_t in_dtype{0};

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
  uint32_t avg_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 0));
  uint32_t avg_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 1));
  uint32_t avg_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 2));
  uint32_t avg_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 3));

  uint32_t tail_block_count = (uint32_t)(*((__gm__ uint32_t *)tiling + 4));
  uint32_t tail_block_ub_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 5));
  uint32_t tail_block_ub_tail = (uint32_t)(*((__gm__ uint32_t *)tiling + 6));
  uint32_t tail_block_ub_loop = (uint32_t)(*((__gm__ uint32_t *)tiling + 7));

  uint32_t buffer_num = (uint32_t)(*((__gm__ uint32_t *)tiling + 8));
  uint32_t in_dtype = (uint32_t)(*((__gm__ uint32_t *)tiling + 10));

  if (in_dtype == 0) {
    KernelAllFinite<uint32_t> op;
    op.setArgs(x, z);
    op.setTiling(avg_block_count, avg_block_ub_num, avg_block_ub_tail, avg_block_ub_loop, tail_block_count,
                 tail_block_ub_num, tail_block_ub_tail, tail_block_ub_loop, buffer_num, in_dtype);
    op.Process();
  } else if (in_dtype == 1) {
    KernelAllFinite<uint16_t> op;
    op.setArgs(x, z);
    op.setTiling(avg_block_count, avg_block_ub_num, avg_block_ub_tail, avg_block_ub_loop, tail_block_count,
                 tail_block_ub_num, tail_block_ub_tail, tail_block_ub_loop, buffer_num, in_dtype);
    op.Process();
  }
}
