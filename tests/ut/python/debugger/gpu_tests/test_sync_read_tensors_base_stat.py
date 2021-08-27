# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Read tensor base and statistics test script for offline debugger APIs.
"""
import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import compare_actual_with_expected, build_dump_structure

GENERATE_GOLDEN = False
test_name = "sync_read_tensors_base_stat"


def test_sync_read_tensors_base_stat():

    value_tensor = np.array([[7.5, 8.56, -9.78], [10.0, -11.0, 0.0]], np.float32)
    name1 = "Add.Add-op4.0.0."
    info1 = d.TensorInfo(node_name="Default/Add-op4",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    inf_tensor = np.array([[1., -np.inf, np.inf, -np.inf, np.inf], [np.inf, 1., -np.inf, np.inf, np.inf]], np.float32)
    name2 = "Reciprocal.Reciprocal-op3.0.0."
    info2 = d.TensorInfo(node_name="Default/Reciprocal-op3",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    nan_tensor = np.array([-2.1754317, 1.9901361, np.nan, np.nan, -1.8091936], np.float32)
    name3 = "ReduceMean.ReduceMean-op92.0.0."
    info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-MockModel/ReduceMean-op92",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    invalid_tensor = np.array([[1.1, -2.2], [3.3, -4.4]], np.float32)
    name4 = "Add.Add-op1.0.0."
    info4 = d.TensorInfo(node_name="invalid_name_for_test",
                         slot=0, iteration=0, rank_id=0, root_graph_id=0, is_output=True)

    tensor_info_1 = [info1]
    tensor_info_2 = [info2]
    tensor_info_3 = [info3]
    tensor_info_4 = [info4]
    tensor_info = [info1, info2, info3, info4]
    value_path = build_dump_structure([name1], [value_tensor], "Add", tensor_info_1)
    inf_path = build_dump_structure([name2], [inf_tensor], "Inf", tensor_info_2)
    nan_path = build_dump_structure([name3], [nan_tensor], "Nan", tensor_info_3)
    inv_path = build_dump_structure([name4], [invalid_tensor], "Inv", tensor_info_4)

    debugger_backend = d.DbgServices(
        dump_file_path=value_path, verbose=True)

    _ = debugger_backend.initialize(
        net_name="Add", is_sync_mode=True)

    debugger_backend_2 = d.DbgServices(
        dump_file_path=inf_path, verbose=True)

    _ = debugger_backend_2.initialize(
        net_name="Inf", is_sync_mode=True)

    debugger_backend_3 = d.DbgServices(
        dump_file_path=nan_path, verbose=True)

    _ = debugger_backend_3.initialize(
        net_name="Nan", is_sync_mode=True)

    debugger_backend_4 = d.DbgServices(
        dump_file_path=inv_path, verbose=True)

    _ = debugger_backend_4.initialize(
        net_name="Inv", is_sync_mode=True)

    tensor_base_data_list = debugger_backend.read_tensor_base(tensor_info_1)
    tensor_base_data_list_2 = debugger_backend_2.read_tensor_base(tensor_info_2)
    tensor_base_data_list.extend(tensor_base_data_list_2)
    tensor_base_data_list_3 = debugger_backend_3.read_tensor_base(tensor_info_3)
    tensor_base_data_list.extend(tensor_base_data_list_3)
    tensor_base_data_list_4 = debugger_backend_4.read_tensor_base(tensor_info_4)
    tensor_base_data_list.extend(tensor_base_data_list_4)

    tensor_stat_data_list = debugger_backend.read_tensor_stats(tensor_info_1)
    tensor_stat_data_list_2 = debugger_backend_2.read_tensor_stats(tensor_info_2)
    tensor_stat_data_list.extend(tensor_stat_data_list_2)
    tensor_stat_data_list_3 = debugger_backend_3.read_tensor_stats(tensor_info_3)
    tensor_stat_data_list.extend(tensor_stat_data_list_3)
    tensor_stat_data_list_4 = debugger_backend_4.read_tensor_stats(tensor_info_4)
    tensor_stat_data_list.extend(tensor_stat_data_list_4)

    shutil.rmtree(value_path)
    shutil.rmtree(inf_path)
    shutil.rmtree(nan_path)
    shutil.rmtree(inv_path)
    print_read_tensors(tensor_info, tensor_base_data_list, tensor_stat_data_list)
    if not GENERATE_GOLDEN:
        assert compare_actual_with_expected(test_name)


def print_read_tensors(tensor_info, tensor_base_data_list, tensor_stat_data_list):
    """Print read tensors info."""
    if GENERATE_GOLDEN:
        f_write = open(test_name + ".expected", "w")
    else:
        f_write = open(test_name + ".actual", "w")
    for x, (tensor_info_item, tensor_base, tensor_stat) in enumerate(zip(tensor_info,
                                                                         tensor_base_data_list,
                                                                         tensor_stat_data_list)):
        f_write.write(
            "-----------------------------------------------------------\n")
        f_write.write("tensor_info_" + str(x+1) + " attributes:\n")
        f_write.write("node name = " + tensor_info_item.node_name + "\n")
        f_write.write("slot = " + str(tensor_info_item.slot) + "\n")
        f_write.write("iteration = " + str(tensor_info_item.iteration) + "\n")
        f_write.write("rank_id = " + str(tensor_info_item.rank_id) + "\n")
        f_write.write("root_graph_id = " +
                      str(tensor_info_item.root_graph_id) + "\n")
        f_write.write("is_output = " +
                      str(tensor_info_item.is_output) + "\n")
        f_write.write("\n")
        f_write.write("tensor_base_info:\n")
        f_write.write(str(tensor_base) + "\n")
        f_write.write("\n")
        f_write.write("tensor_stat_info:\n")
        f_write.write(str(tensor_stat) + '\n')
    f_write.close()
