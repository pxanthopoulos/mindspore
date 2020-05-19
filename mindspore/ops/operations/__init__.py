# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
Primitive operator classes.

A collection of operators to build nerual networks or computing functions.
"""

from .array_ops import (Argmax, Argmin, Cast, Concat, Pack, Unpack,
                        Diag, DiagPart, DType, ExpandDims, Eye,
                        Fill, GatherNd, GatherV2, InvertPermutation,
                        IsInstance, IsSubClass, ArgMaxWithValue, OnesLike, ZerosLike,
                        Rank, Reshape, ResizeNearestNeighbor, ArgMinWithValue, Range,
                        SameTypeShape, ScatterMax, ScatterUpdate,
                        ScalarToArray, ScalarToTensor, ScatterNd, ScatterNdUpdate, Select,
                        Shape, Size, Slice, Split,
                        Squeeze, StridedSlice, Tile,
                        Transpose, TruncatedNormal, TupleToArray, UnsortedSegmentMin,
                        UnsortedSegmentSum, SpaceToDepth, DepthToSpace, SpaceToBatch, BatchToSpace)
from .comm_ops import (AllGather, AllReduce, _AlltoAll, ReduceScatter, Broadcast,
                       _MirrorOperator, ReduceOp, _VirtualDataset,
                       _VirtualDiv, _GetTensorSlice)
from .debug_ops import (ImageSummary, InsertGradientOf, HookBackward, ScalarSummary,
                        TensorSummary, HistogramSummary, Print)
from .control_ops import ControlDepend, GeSwitch, Merge
from .inner_ops import ScalarCast
from .math_ops import (Abs, ACos, AddN, AssignAdd, AssignSub, Atan2, BatchMatMul,
                       ReduceMax, ReduceMin, ReduceMean, ReduceSum, ReduceAll, ReduceProd, CumProd,
                       Cos, Div, Equal, EqualCount, Exp, Erf, Erfc, Floor, FloorDiv, FloorMod, Acosh,
                       Greater, GreaterEqual, Less, LessEqual, Log, Log1p, LogicalAnd,
                       LogicalNot, LogicalOr, MatMul, Maximum,
                       Minimum, Mul, Neg, NMSWithMask, NotEqual,
                       NPUAllocFloatStatus, NPUClearFloatStatus,
                       NPUGetFloatStatus, Pow, RealDiv, IsNan, IsInf, IsFinite, FloatStatus,
                       Reciprocal, CumSum,
                       Sin, Sqrt, Rsqrt,
                       Square, Sub, TensorAdd, Sign, Round, SquareSumAll)
from .random_ops import (RandomChoiceWithMask)
from .nn_ops import (LSTM, SGD, Adam, ApplyMomentum, BatchNorm,
                     BiasAdd, Conv2D,
                     DepthwiseConv2dNative,
                     DropoutDoMask, DropoutGrad, Dropout,
                     DropoutGenMask, Flatten, FusedBatchNorm,
                     Gelu, Elu,
                     GetNext, L2Normalize, LayerNorm, L2Loss, CTCLoss,
                     LogSoftmax,
                     MaxPool,
                     AvgPool, Conv2DBackpropInput, ConfusionMulGrad,
                     MaxPoolWithArgmax, OneHot, Pad, MirrorPad, PReLU, ReLU, ReLU6, ReLUV2, HSwish, HSigmoid,
                     ResizeBilinear, Sigmoid,
                     SigmoidCrossEntropyWithLogits,
                     SmoothL1Loss, Softmax, Softplus,
                     SoftmaxCrossEntropyWithLogits, ROIAlign,
                     SparseSoftmaxCrossEntropyWithLogits, Tanh,
                     TopK, BinaryCrossEntropy, SparseApplyAdagrad, LARSUpdate, ApplyFtrl,
                     ApplyRMSProp, ApplyCenteredRMSProp)
from .other_ops import Assign, IOU, BoundingBoxDecode, BoundingBoxEncode, CheckValid, MakeRefKey, CheckBprop
from . import _quant_ops
from ._quant_ops import *

__all__ = [
    'TensorAdd',
    'Argmax',
    'Argmin',
    'ArgMaxWithValue',
    'ArgMinWithValue',
    'AddN',
    'Sub',
    'CumSum',
    'MatMul',
    'BatchMatMul',
    'Mul',
    'Pow',
    'Exp',
    'Rsqrt',
    'Sqrt',
    'Square',
    'Conv2D',
    'Flatten',
    'MaxPoolWithArgmax',
    'FusedBatchNorm',
    'BatchNorm',
    'MaxPool',
    'TopK',
    'Adam',
    'Softplus',
    'Softmax',
    'LogSoftmax',
    'SoftmaxCrossEntropyWithLogits',
    'ROIAlign',
    'ConfusionMulGrad',
    'SparseSoftmaxCrossEntropyWithLogits',
    'SGD',
    'ApplyMomentum',
    'ExpandDims',
    'Cast',
    'IsSubClass',
    'IsInstance',
    'Reshape',
    'Squeeze',
    'Transpose',
    'OneHot',
    'GatherV2',
    'Concat',
    'Pack',
    'Unpack',
    'Tile',
    'BiasAdd',
    'Gelu',
    'Minimum',
    'Maximum',
    'StridedSlice',
    'ReduceSum',
    'ReduceMean',
    'Range',
    'LayerNorm',
    'Rank',
    'Less',
    'LessEqual',
    'RealDiv',
    'Div',
    'TruncatedNormal',
    'Fill',
    'OnesLike',
    'ZerosLike',
    'Select',
    'Split',
    'ReLU',
    'ReLU6',
    'ReLUV2',
    'Elu',
    'Erf',
    'Erfc',
    'Sigmoid',
    'HSwish',
    'HSigmoid',
    'Tanh',
    'RandomChoiceWithMask',
    'ResizeBilinear',
    'ScalarSummary',
    'ImageSummary',
    'TensorSummary',
    'HistogramSummary',
    "Print",
    'InsertGradientOf',
    'HookBackward',
    'InvertPermutation',
    'Shape',
    'DropoutDoMask',
    'DropoutGenMask',
    'DropoutGrad',
    'Dropout',
    'Neg',
    'Slice',
    'DType',
    'NPUAllocFloatStatus',
    'NPUGetFloatStatus',
    'NPUClearFloatStatus',
    'IsNan',
    'IsFinite',
    'IsInf',
    'FloatStatus',
    'Reciprocal',
    'SmoothL1Loss',
    'L2Loss',
    'CTCLoss',
    'ReduceAll',
    'ScalarToArray',
    'ScalarToTensor',
    'TupleToArray',
    'ControlDepend',
    'GeSwitch',
    'Merge',
    'SameTypeShape',
    'CheckBprop',
    'CheckValid',
    'BoundingBoxEncode',
    'BoundingBoxDecode',
    'L2Normalize',
    'ScatterNd',
    'ScatterMax',
    'ResizeNearestNeighbor',
    'Pad',
    'MirrorPad',
    'GatherNd',
    'ScatterUpdate',
    'ScatterNdUpdate',
    'Floor',
    'NMSWithMask',
    'IOU',
    'MakeRefKey',
    'AvgPool',
    # Back Primitive
    'Equal',
    'EqualCount',
    'NotEqual',
    'Greater',
    'GreaterEqual',
    'LogicalNot',
    'LogicalAnd',
    'LogicalOr',
    'Size',
    'DepthwiseConv2dNative',
    'UnsortedSegmentSum',
    'UnsortedSegmentMin',
    "AllGather",
    "AllReduce",
    "ReduceScatter",
    "Broadcast",
    "ReduceOp",
    'ScalarCast',
    'GetNext',
    'ReduceMax',
    'ReduceMin',
    'ReduceProd',
    'CumProd',
    'Log',
    'SigmoidCrossEntropyWithLogits',
    'FloorDiv',
    'FloorMod',
    'Acosh',
    "PReLU",
    "Cos",
    "ACos",
    "Diag",
    "DiagPart",
    'Eye',
    'Assign',
    'AssignAdd',
    'AssignSub',
    "Sin",
    "LSTM",
    "Abs",
    "BinaryCrossEntropy",
    "SparseApplyAdagrad",
    "SpaceToDepth",
    "DepthToSpace",
    "Conv2DBackpropInput",
    "Sign",
    "LARSUpdate",
    "Round",
    "ApplyFtrl",
    "SpaceToBatch",
    "BatchToSpace",
    "Atan2",
    "ApplyRMSProp",
    "ApplyCenteredRMSProp",
    "SquareSumAll"
]

__all__.extend(_quant_ops.__all__)
__all__.sort()
