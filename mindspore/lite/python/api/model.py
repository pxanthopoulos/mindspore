# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
Model API.
"""
from __future__ import absolute_import
import os
from enum import Enum

from mindspore_lite._checkparam import check_isinstance
from mindspore_lite.context import Context
from mindspore_lite.lib import _c_lite_wrapper
from mindspore_lite.tensor import Tensor

__all__ = ['ModelType', 'Model', 'ModelParallelRunner']


class ModelType(Enum):
    """
    The `ModelType` class defines the type of the model exported or imported in MindSpot Lite.

    Used in the following scenarios:

    1. When using `mindspore_lite.Converter`, set `save_type` parameter, `ModelType` used to define the model type
    generated by Converter.

    2. After using `mindspore_lite.Converter`, when loading or building a model from file for predicting, the
    `ModelType` is used to define Input model framework type.

    Currently, the following `ModelType` are supported:

    ===========================  =======================================================================
    Definition                    Description
    ===========================  =======================================================================
    `ModelType.MINDIR`           MindSpore model's framework type, which model uses .mindir as suffix.
    `ModelType.MINDIR_LITE`      MindSpore Lite model's framework type, which model uses .ms as suffix.
    ===========================  =======================================================================

    Examples:
        >>> # Method 1: Import mindspore_lite package
        >>> import mindspore_lite as mslite
        >>> print(mslite.ModelType.MINDIR_LITE)
        ModelType.MINDIR_LITE
        >>> # Method 2: from mindspore_lite package import ModelType
        >>> from mindspore_lite import ModelType
        >>> print(ModelType.MINDIR_LITE)
        ModelType.MINDIR_LITE
    """

    MINDIR = 0
    MINDIR_LITE = 4


model_type_py_cxx_map = {
    ModelType.MINDIR: _c_lite_wrapper.ModelType.kMindIR,
    ModelType.MINDIR_LITE: _c_lite_wrapper.ModelType.kMindIR_Lite,
}

model_type_cxx_py_map = {
    _c_lite_wrapper.ModelType.kMindIR: ModelType.MINDIR,
    _c_lite_wrapper.ModelType.kMindIR_Lite: ModelType.MINDIR_LITE,
}


class Model:
    """
    The `Model` class defines a MindSpore Lite's model, facilitating computational graph management.

    Examples:
        >>> import mindspore_lite as mslite
        >>> model = mslite.Model()
        >>> print(model)
        model_path: .
    """

    def __init__(self):
        self._model = _c_lite_wrapper.ModelBind()
        self.model_path_ = ""

    def __str__(self):
        res = f"model_path: {self.model_path_}."
        return res

    def build_from_file(self, model_path, model_type, context=None, config_path=""):
        """
        Load and build a model from file.

        Args:
            model_path (str): Path of the input model when build from file. For example, "/home/user/model.ms". Options
                are MindSpore model: "model.mindir" | MindSpore Lite model: "model.ms".
            model_type (ModelType): Define The type of input model file. Options are ModelType.MINDIR |
                ModelType.MINDIR_LITE. For details, see
                `ModelType <https://mindspore.cn/lite/api/en/r2.0/mindspore_lite/mindspore_lite.ModelType.html>`_ .
            context (Context, optional): Define the context used to transfer options during execution. Default: None.
                None means the Context with cpu target.
            config_path (str, optional): Define the config file path. the config file is used to transfer user defined
                options during build model. In the following scenarios, users may need to set the parameter.
                For example, "/home/user/config.txt". Default: "".

                - Usage 1: Set mixed precision inference. The content and description of the configuration file are as
                      follows:

                      .. code-block::

                          [execution_plan]
                          [op_name1]=data_Type: float16 (The operator named op_name1 sets the data type as Float16)
                          [op_name2]=data_Type: float32 (The operator named op_name2 sets the data type as Float32)

                - Usage 2: When GPU inference, set the configuration of TensorRT. The content and description of the
                      configuration file are as follows:

                      .. code-block::

                          [ms_cache]
                          serialize_Path=[serialization model path](storage path of serialization model)
                          [gpu_context]
                          input_shape=input_Name: [input_dim] (Model input dimension, for dynamic shape)
                          dynamic_Dims=[min_dim~max_dim] (dynamic dimension range of model input, for dynamic shape)
                          opt_Dims=[opt_dim] (the optimal input dimension of the model, for dynamic shape)

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `model_type` is not a ModelType.
            TypeError: `context` is neither a Context nor None.
            TypeError: `config_path` is not a str.
            RuntimeError: `model_path` does not exist.
            RuntimeError: `config_path` does not exist.
            RuntimeError: load configuration by `config_path` failed.
            RuntimeError: build from file failed.

        Examples:
            >>> # Testcase 1: build from file with default cpu context.
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> print(model)
            model_path: mobilenetv2.ms.
            >>> # Testcase 2: build from file with gpu context.
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE, context)
            >>> print(model)
            model_path: mobilenetv2.ms.
        """
        check_isinstance("model_path", model_path, str)
        check_isinstance("model_type", model_type, ModelType)
        if context is None:
            context = Context()
        check_isinstance("context", context, Context)
        check_isinstance("config_path", config_path, str)
        if not os.path.exists(model_path):
            raise RuntimeError(f"build_from_file failed, model_path does not exist!")
        self.model_path_ = model_path
        model_type_ = _c_lite_wrapper.ModelType.kMindIR_Lite
        if model_type is ModelType.MINDIR:
            model_type_ = _c_lite_wrapper.ModelType.kMindIR
        if config_path:
            if not os.path.exists(config_path):
                raise RuntimeError(f"build_from_file failed, config_path does not exist!")
            ret = self._model.load_config(config_path)
            if not ret.IsOk():
                raise RuntimeError(f"load configuration failed! Error is {ret.ToString()}")
        ret = self._model.build_from_file(self.model_path_, model_type_, context._context._inner_context)
        if not ret.IsOk():
            raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.

        Returns:
            list[Tensor], the input Tensor list of the model.

        Examples:
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> inputs = model.get_inputs()
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def predict(self, inputs):
        """
        Inference model.

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.

        Returns:
            list[Tensor], the output Tensor list of the model.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # 1. predict which indata is from file
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> #default context's target is cpu
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> inputs = model.get_inputs()
            >>> in_data = np.fromfile("input.bin", dtype=np.float32)
            >>> inputs[0].set_data_from_numpy(in_data)
            >>> outputs = model.predict(inputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
            >>> # 2. predict which indata is numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> inputs = model.get_inputs()
            >>> for input in inputs:
            ...     in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
            ...     input.set_data_from_numpy(in_data)
            ...
            >>> outputs = model.predict(inputs)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
            >>> # 3. predict which indata is from new MindSpore Lite's Tensor with numpy array
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> inputs = model.get_inputs()
            >>> input_tensors = []
            >>> for input in inputs:
            ...     input_tensor = mslite.Tensor()
            ...     input_tensor.dtype = input.dtype
            ...     input_tensor.shape = input.shape
            ...     input_tensor.format = input.format
            ...     input_tensor.name = input.name
            ...     in_data = np.arange(1 * 224 * 224 * 3, dtype=np.float32).reshape((1, 224, 224, 3))
            ...     input_tensor.set_data_from_numpy(in_data)
            ...     input_tensors.append(input_tensor)
            ...
            >>> outputs = model.predict(input_tensors)
            >>> for output in outputs:
            ...     data = output.get_data_to_numpy()
            ...     print("outputs' shape: ", data.shape)
            ...
            outputs' shape:  (1,1001)
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        outputs = self._model.get_outputs()
        ret = self._model.predict(_inputs, outputs, None, None)
        if not ret.IsOk():
            raise RuntimeError(f"predict failed! Error is {ret.ToString()}")
        predict_outputs = []
        for output in outputs:
            predict_outputs.append(Tensor(output))
        return predict_outputs

    def resize(self, inputs, dims):
        """
        Resizes the shapes of inputs. This method is used in the following scenarios:

        1. If multiple inputs of the same size need to predicted, you can set the batch dimension of `dims` to
           the number of inputs, then multiple inputs can be performed inference at the same time.

        2. Adjust the input size to the specify shape.

        3. When the input is a dynamic shape (a dimension of the shape of the model input contains -1), -1 must be
           replaced by a fixed dimension through `resize` .

        4. The shape operator contained in the model is dynamic shape (a dimension of the shape operator contains -1).

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.
            dims (list[list[int]]): A list that includes the new shapes of input Tensors, should be consistent with
                input Tensors' shape.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            TypeError: `dims` is not a list.
            TypeError: `dims` is a list, but the elements are not list.
            TypeError: `dims` is a list, the elements are list, but the element's elements are not int.
            ValueError: The size of `inputs` is not equal to the size of `dims` .
            RuntimeError: resize inputs failed.

        Examples:
            >>> import mindspore_lite as mslite
            >>> model = mslite.Model()
            >>> model.build_from_file("mobilenetv2.ms", mslite.ModelType.MINDIR_LITE)
            >>> inputs = model.get_inputs()
            >>> print("Before resize, the first input shape: ", inputs[0].shape)
            Before resize, the first input shape: [1, 224, 224, 3]
            >>> model.resize(inputs, [[1, 112, 112, 3]])
            >>> print("After resize, the first input shape: ", inputs[0].shape)
            After resize, the first input shape: [1, 112, 112, 3]
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        if not isinstance(dims, list):
            raise TypeError("dims must be list, but got {}.".format(type(dims)))
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
        for i, element in enumerate(dims):
            if not isinstance(element, list):
                raise TypeError(f"dims element must be list, but got "
                                f"{type(element)} at index {i}.")
            for j, dim in enumerate(element):
                if not isinstance(dim, int):
                    raise TypeError(f"dims element's element must be int, but got "
                                    f"{type(dim)} at {i}th dims element's {j}th element.")
        if len(inputs) != len(dims):
            raise ValueError(f"inputs' size does not match dims' size, but got "
                             f"inputs: {len(inputs)} and dims: {len(dims)}.")
        for _, element in enumerate(inputs):
            _inputs.append(element._tensor)
        ret = self._model.resize(_inputs, dims)
        if not ret.IsOk():
            raise RuntimeError(f"resize failed! Error is {ret.ToString()}")


class ModelParallelRunner:
    """
    The `ModelParallelRunner` class defines a MindSpore Lite's Runner, which support model parallelism. Compared with
    `model` , `model` does not support parallelism, but `ModelParallelRunner` supports parallelism. A Runner contains
    multiple workers, which are the units that actually perform parallel inferring. The primary use case is when
    multiple clients send inference tasks to the server, the server perform parallel inference, shorten the inference
    time, and then return the inference results to the clients.

    Examples:
        >>> # Use case: serving inference.
        >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
        >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
        >>> import mindspore_lite as mslite
        >>> model_parallel_runner = mslite.ModelParallelRunner()
        >>> print(model_parallel_runner)
        model_path: .
    """

    def __init__(self):
        if hasattr(_c_lite_wrapper, "ModelParallelRunnerBind"):
            self._model = _c_lite_wrapper.ModelParallelRunnerBind()
        else:
            raise RuntimeError(f"ModelParallelRunner init failed, If you want to use it, you need to build"
                               f"MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.")
        self.model_path_ = ""

    def __str__(self):
        return f"model_path: {self.model_path_}."

    def build_from_file(self, model_path, context=None):
        """
        build a model parallel runner from model path so that it can run on a device.

        Args:
            model_path (str): Define the model path.
            context (Context, optional): Define the config used to transfer context and options during building model.
                Default: None. None means the Context with cpu target. Context has the default parallel
                attribute.

        Raises:
            TypeError: `model_path` is not a str.
            TypeError: `context` is neither a Context nor None.
            RuntimeError: `model_path` does not exist.
            RuntimeError: ModelParallelRunner's init failed.

        Examples:
            >>> # Use case: serving inference.
            >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.parallel.workers_num = 4
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.ms", context=context)
            >>> print(model_parallel_runner)
            model_path: mobilenetv2.ms.
        """
        check_isinstance("model_path", model_path, str)
        if not os.path.exists(model_path):
            raise RuntimeError(f"ModelParallelRunner's build from file failed, model_path does not exist!")
        self.model_path_ = model_path
        if context is None:
            ret = self._model.init(self.model_path_, None)
        else:
            check_isinstance("context", context, Context)
            ret = self._model.init(self.model_path_, context.parallel._runner_config)
        if not ret.IsOk():
            raise RuntimeError(f"ModelParallelRunner's build from file failed! Error is {ret.ToString()}")

    def get_inputs(self):
        """
        Obtains all input Tensors of the model.

        Returns:
            list[Tensor], the input Tensor list of the model.

        Examples:
            >>> # Use case: serving inference.
            >>> # precondition 1: Building MindSpore Lite serving package by export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # precondition 2: install wheel package of MindSpore Lite built by precondition 1.
            >>> import mindspore_lite as mslite
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.parallel.workers_num = 4
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.ms", context=context)
            >>> inputs = model_parallel_runner.get_inputs()
        """
        inputs = []
        for _tensor in self._model.get_inputs():
            inputs.append(Tensor(_tensor))
        return inputs

    def predict(self, inputs):
        """
        Inference ModelParallelRunner.

        Args:
            inputs (list[Tensor]): A list that includes all input Tensors in order.

        Returns:
            list[Tensor], outputs, the model outputs are filled in the container in sequence.

        Raises:
            TypeError: `inputs` is not a list.
            TypeError: `inputs` is a list, but the elements are not Tensor.
            RuntimeError: predict model failed.

        Examples:
            >>> # Use case: serving inference.
            >>> # Precondition 1: Download MindSpore Lite serving package or building MindSpore Lite serving package by
            >>> #                 export MSLITE_ENABLE_SERVER_INFERENCE=on.
            >>> # Precondition 2: Install wheel package of MindSpore Lite built by precondition 1.
            >>> # The result can be find in the tutorial of runtime_parallel_python.
            >>> import time
            >>> from threading import Thread
            >>> import numpy as np
            >>> import mindspore_lite as mslite
            >>> # the number of threads of one worker.
            >>> # WORKERS_NUM * THREAD_NUM should not exceed the number of cores of the machine.
            >>> THREAD_NUM = 1
            >>> # In parallel inference, the number of workers in one `ModelParallelRunner` in server.
            >>> # If you prepare to compare the time difference between parallel inference and serial inference,
            >>> # you can set WORKERS_NUM = 1 as serial inference.
            >>> WORKERS_NUM = 3
            >>> # Simulate 5 clients, and each client sends 2 inference tasks to the server at the same time.
            >>> PARALLEL_NUM = 5
            >>> TASK_NUM = 2
            >>>
            >>>
            >>> def parallel_runner_predict(parallel_runner, parallel_id):
            ...     # One Runner with 3 workers, set model input, execute inference and get output.
            ...     task_index = 0
            ...     while True:
            ...         if task_index == TASK_NUM:
            ...             break
            ...         task_index += 1
            ...         # Set model input
            ...         inputs = parallel_runner.get_inputs()
            ...         in_data = np.fromfile("input.bin", dtype=np.float32)
            ...         inputs[0].set_data_from_numpy(in_data)
            ...         once_start_time = time.time()
            ...         # Execute inference
            ...         outputs = parallel_runner.predict(inputs)
            ...         once_end_time = time.time()
            ...         print("parallel id: ", parallel_id, " | task index: ", task_index, " | run once time: ",
            ...               once_end_time - once_start_time, " s")
            ...         # Get output
            ...         for output in outputs:
            ...             tensor_name = output.name.rstrip()
            ...             data_size = output.data_size
            ...             element_num = output.element_num
            ...             print("tensor name is:%s tensor size is:%s tensor elements num is:%s" % (tensor_name,
            ...                                                                                      data_size,
            ...                                                                                      element_num))
            ...
            ...             data = output.get_data_to_numpy()
            ...             data = data.flatten()
            ...             print("output data is:", end=" ")
            ...             for j in range(5):
            ...                 print(data[j], end=" ")
            ...             print("")
            ...
            >>> # Init RunnerConfig and context, and add CPU device info
            >>> context = mslite.Context()
            >>> context.target = ["cpu"]
            >>> context.cpu.enable_fp16 = False
            >>> context.cpu.thread_num = THREAD_NUM
            >>> context.cpu.inter_op_parallel_num = THREAD_NUM
            >>> context.parallel.workers_num = WORKERS_NUM
            >>> # Build ModelParallelRunner from file
            >>> model_parallel_runner = mslite.ModelParallelRunner()
            >>> model_parallel_runner.build_from_file(model_path="mobilenetv2.ms", context=context)
            >>> # The server creates 5 threads to store the inference tasks of 5 clients.
            >>> threads = []
            >>> total_start_time = time.time()
            >>> for i in range(PARALLEL_NUM):
            ...     threads.append(Thread(target=parallel_runner_predict, args=(model_parallel_runner, i,)))
            ...
            >>> # Start threads to perform parallel inference.
            >>> for th in threads:
            ...     th.start()
            ...
            >>> for th in threads:
            ...     th.join()
            ...
            >>> total_end_time = time.time()
            >>> print("total run time: ", total_end_time - total_start_time, " s")
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be list, but got {}.".format(type(inputs)))
        _inputs = []
        for i, element in enumerate(inputs):
            if not isinstance(element, Tensor):
                raise TypeError(f"inputs element must be Tensor, but got "
                                f"{type(element)} at index {i}.")
            _inputs.append(element._tensor)
        _outputs = self._model.predict(_inputs, [], None, None)
        if not isinstance(_outputs, list) or len(_outputs) == 0:
            raise RuntimeError(f"predict failed!")
        predict_outputs = []
        for _output in _outputs:
            predict_outputs.append(Tensor(_output))
        return predict_outputs
