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
Functional Cells used in Bert finetune and evaluation.
"""

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.train.parallel_utils import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from .bert_model import BertModel
from .bert_for_pre_training import clip_grad
from .CRF import CRF

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class BertFinetuneCell(nn.Cell):
    """
    Especifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):

        super(BertFinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad',
                                    get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):


        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        if not self.gpu_target:
            init = self.alloc_status()
            clear_before_grad = self.clear_before_grad(init)
            F.control_depend(loss, init)
            self.depend_parameter_use(clear_before_grad, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            flag = self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
            F.control_depend(grads, flag)
            F.control_depend(flag, flag_sum)
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return F.depend(ret, succ)

class BertSquadCell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertSquadCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        weights = self.weights
        init = self.alloc_status()
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        clear_before_grad = self.clear_before_grad(init)
        F.control_depend(loss, init)
        self.depend_parameter_use(clear_before_grad, scaling_sens)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        flag = self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        F.control_depend(grads, flag)
        F.control_depend(flag, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return F.depend(ret, succ)


class BertRegressionModel(nn.Cell):
    """
    Bert finetune model for regression task
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.dense_1 = nn.Dense(config.hidden_size, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(mstype.float16)

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        return logits


class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is propotional to that of softmax.
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertCLSModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = \
            self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        log_probs = self.log_softmax(logits)
        return log_probs

class BertSquadModel(nn.Cell):
    """
    Bert finetune model for SQuAD v1.1 task
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=1)
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        batch_size, seq_length, hidden_size = P.Shape()(sequence_output)
        sequence = P.Reshape()(sequence_output, (-1, hidden_size))
        logits = self.dense1(sequence)
        logits = P.Cast()(logits, self.dtype)
        logits = P.Reshape()(logits, (batch_size, seq_length, self.num_labels))
        logits = self.log_softmax(logits)
        return logits

class BertNERModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is propotional to that of softmax.
    """
    def __init__(self, config, is_training, num_labels=11, use_crf=False, dropout_prob=0.0,
                 use_one_hot_embeddings=False):
        super(BertNERModel, self).__init__()
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.use_crf = use_crf
        self.origin_shape = (config.batch_size, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, _, _ = \
            self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        if self.use_crf:
            return_value = self.reshape(logits, self.origin_shape)
        else:
            return_value = self.log_softmax(logits)
        return return_value

class CrossEntropyCalculation(nn.Cell):
    """
    Cross Entropy loss
    """
    def __init__(self, is_training=True):
        super(CrossEntropyCalculation, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training

    def construct(self, logits, label_ids, num_labels):
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            one_hot_labels = self.onehot(label_ids, num_labels, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx))
            loss = self.reduce_mean(per_example_loss, self.last_idx)
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0
        return return_value

class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        log_probs = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(log_probs, label_ids, self.num_labels)
        return loss


class BertNER(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """
    def __init__(self, config, is_training, num_labels=11, use_crf=False, tag_to_index=None, dropout_prob=0.0,
                 use_one_hot_embeddings=False):
        super(BertNER, self).__init__()
        self.bert = BertNERModel(config, is_training, num_labels, use_crf, dropout_prob, use_one_hot_embeddings)
        if use_crf:
            if not tag_to_index:
                raise Exception("The dict for tag-index mapping should be provided for CRF.")
            self.loss = CRF(tag_to_index, config.batch_size, config.seq_length, is_training)
        else:
            self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.use_crf = use_crf
    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.use_crf:
            loss = self.loss(logits, label_ids)
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss

class BertSquad(nn.Cell):
    """
    Train interface for SQuAD finetuning task.
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquad, self).__init__()
        self.bert = BertSquadModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.seq_length = config.seq_length
        self.is_training = is_training
        self.total_num = Parameter(Tensor([0], mstype.float32), name='total_num')
        self.start_num = Parameter(Tensor([0], mstype.float32), name='start_num')
        self.end_num = Parameter(Tensor([0], mstype.float32), name='end_num')
        self.sum = P.ReduceSum()
        self.equal = P.Equal()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            unstacked_logits_0 = self.squeeze(logits[:, :, 0:1])
            unstacked_logits_1 = self.squeeze(logits[:, :, 1:2])
            start_loss = self.loss(unstacked_logits_0, start_position, self.seq_length)
            end_loss = self.loss(unstacked_logits_1, end_position, self.seq_length)
            total_loss = (start_loss + end_loss) / 2.0
        else:
            start_logits = self.squeeze(logits[:, :, 0:1])
            end_logits = self.squeeze(logits[:, :, 1:2])
            total_loss = (unique_id, start_logits, end_logits)
        return total_loss


class BertReg(nn.Cell):
    """
    Bert finetune model with loss for regression task
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertReg, self).__init__()
        self.bert = BertRegressionModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = nn.MSELoss()
        self.is_training = is_training
        self.sigmoid = P.Sigmoid()
        self.cast = P.Cast()
        self.mul = P.Mul()
    def construct(self, input_ids, input_mask, token_type_id, labels):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            loss = self.loss(logits, labels)
        else:
            loss = logits
        return loss
