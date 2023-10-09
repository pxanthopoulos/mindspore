# Copyright 2023 Huawei Technologies Co., Ltd
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
Generate operator definition from ops.yaml
"""
import os
from gen_utils import py_licence_str, cc_license_str, \
    get_type_str, check_change_and_replace_file, merge_files, safe_load_yaml


def get_op_name(operator_name, class_def):
    """
    Get op name for python class Primitive or c++ OpDef name.
    """
    class_name = ''.join(word.capitalize() for word in operator_name.split('_'))
    if class_def is not None:
        item = class_def.get("name")
        if item is not None:
            class_name = item
    return class_name


def get_disable_flag(yaml_def):
    """
    Get class or functional api disable generate flag.
    """
    disable_flag = False
    if yaml_def is not None:
        item = yaml_def.get("disable")
        if item is not None:
            if item is not True and item is not False:
                raise TypeError(f"The disable label for function should be True or False, but get {item}.")
            disable_flag = item
    return disable_flag


def signature_get_rw_label(rw_op_name, write_list, read_list, ref_list):
    """
    Generate signature rw code
    """
    for op in write_list:
        if op == rw_op_name:
            return 'sig.sig_rw.RW_WRITE'
    for op in read_list:
        if op == rw_op_name:
            return 'sig.sig_rw.RW_READ'
    for op in ref_list:
        if op == rw_op_name:
            return 'sig.sig_rw.RW_REF'
    return ''


def signature_get_dtype_label(index):
    """
    Generate signature dtype code
    """
    dtype_index = ''
    if index > 0:
        dtype_index = f"""{index}"""
    return f"""dtype=sig.sig_dtype.T{dtype_index}"""


def generate_py_op_signature(args_signature):
    """
    Generate __mindspore_signature__
    """
    if args_signature is None:
        return ''

    signature_code = f"""    __mindspore_signature__ = """

    rw_write = args_signature.get('rw_write')
    rw_read = args_signature.get('rw_read')
    rw_ref = args_signature.get('rw_ref')
    dtype_group = args_signature.get('dtype_group')

    if rw_write is None and rw_read is None and rw_ref is None:
        signature_code += '(sig.sig_dtype.T, sig.sig_dtype.T)\n'
        return signature_code

    # init rw
    rw_write = rw_write.replace(' ', '')
    rw_read = rw_read.replace(' ', '')
    rw_ref = rw_ref.replace(' ', '')
    dtype_group = dtype_group.replace(' ', '')

    write_list = rw_write.split(",")
    read_list = rw_read.split(",")
    ref_list = rw_ref.split(",")
    rw_list = write_list + read_list + ref_list
    rw_items_used = [False for i in range(len(rw_list))]

    # init dtype group
    group_list = []
    same_type_parsed = dtype_group.split("(")
    for item in same_type_parsed:
        if ')' in item:
            parsed = item.split(")")
            group_list.append(parsed[0])

    signature_code += f""" (\n"""
    i = 0
    for dtype_group in group_list:
        dtype = signature_get_dtype_label(i)
        i = i + 1

        group_item = dtype_group.split(",")
        for same_type_op in group_item:
            find_writable = False
            for rw_index, rw_op in enumerate(rw_list):
                if rw_op == same_type_op:
                    find_writable = True
                    rw_items_used[rw_index] = True
                    rw_code = signature_get_rw_label(rw_op, write_list, read_list, ref_list)
                    signature_code += f"""     sig.make_sig('{rw_op}', {rw_code}, {dtype}),\n"""
            if not find_writable:
                signature_code += f"""     sig.make_sig('{same_type_op}', {dtype}),\n"""

    # item has writable but do not has same_type
    for used_index, used_item in enumerate(rw_items_used):
        if not used_item:
            item_name = rw_list[used_index]
            rw_code = signature_get_rw_label(item_name, write_list, read_list, ref_list)
            signature_code += f"""     sig.make_sig('{item_name}', {rw_code}),\n"""

    signature_code += f"""    )\n"""
    return signature_code


def generate_py_op_deprecated(deprecated):
    """
    Generate @deprecated
    """
    if deprecated is None:
        return ''
    version = deprecated.get("version")
    if version is None:
        raise ValueError("The version of deprecated can't be None.")
    substitute = deprecated.get("substitute")
    if substitute is None:
        raise ValueError("The substitute of deprecated can't be None.")
    use_substitute = deprecated.get("use_substitute")
    if use_substitute is None:
        raise ValueError("The use_substitute of deprecated can't be None.")
    if use_substitute is not True and use_substitute is not False:
        raise ValueError(f"The use_substitute must be True or False, but got {use_substitute}")

    deprecated = f"""    @deprecated("{version}", "{substitute}", {use_substitute})\n"""
    return deprecated


def generate_py_op_func(yaml_data, doc_data):
    """
    Generate operator python function api.
    """
    gen_py = ''

    op_desc_dict = {}
    for operator_name, operator_desc in doc_data.items():
        desc = operator_desc.get("description")
        op_desc_dict[operator_name] = desc

    for operator_name, operator_data in yaml_data.items():
        func_def = operator_data.get('function')
        func_name = operator_name
        if func_def is not None:
            func_disable = get_disable_flag(func_def)
            if func_disable:
                continue
            item = func_def.get("name")
            if item is not None:
                func_name = item

        description = op_desc_dict.get(operator_name)
        args = operator_data.get('args')
        class_name = get_op_name(operator_name, operator_data.get('class'))
        func_args = []
        init_args = []
        input_args = []
        for arg_name, arg_info in args.items():
            init_value = arg_info.get('init')

            if init_value is None:
                func_args.append(arg_name)
                input_args.append(arg_name)
            else:
                if init_value == 'NO_VALUE':
                    func_args.append(f"""{arg_name}""")
                    init_args.append(arg_name)
                else:
                    func_args.append(f"""{arg_name}={init_value}""")
                    init_args.append(arg_name)

        function_code = f"""\n
def {func_name}({', '.join(arg for arg in func_args)}):
    \"\"\"
    {description}
    \"\"\"
    {operator_name}_op = _get_cache_prim({class_name})({', '.join(arg_name for arg_name in init_args)})
    return {operator_name}_op({', '.join(arg_name for arg_name in input_args)})\n"""
        gen_py += function_code

    return gen_py


def process_args(args):
    """
    Process arg for yaml, get arg_name, init value, type cast, arg_handler, etc.
    """
    args_name = []
    args_assign = []
    init_args_with_default = []
    for arg_name, arg_info in args.items():
        dtype = arg_info.get('dtype')

        init_value = arg_info.get('init')
        if init_value is None:
            continue
        if init_value == 'NO_VALUE':
            init_args_with_default.append(f"""{arg_name}""")
        elif init_value == 'None':
            init_args_with_default.append(f"""{arg_name}={init_value}""")
        else:
            init_args_with_default.append(f"""{arg_name}={init_value}""")
        args_name.append(arg_name)

        assign_str = ""
        type_cast = arg_info.get('type_cast')
        if type_cast is not None:
            type_cast_tuple = tuple(ct.strip() for ct in type_cast.split(","))
            assign_str += f'type_it({arg_name}, '
            if len(type_cast_tuple) == 1:
                assign_str += get_type_str(type_cast_tuple[0]) + ', '
            else:
                assign_str += '(' + ', '.join(get_type_str(ct) for ct in type_cast_tuple) + '), '
            assign_str += get_type_str(dtype) + ')'
        else:
            assign_str += arg_name

        arg_handler = arg_info.get('arg_handler')
        if arg_handler is not None:
            assign_str = f'{arg_handler}({assign_str})'

        assign_str = f"""        self._set_prim_arg("{arg_name}", {assign_str})"""
        args_assign.append(assign_str)
    return args_name, args_assign, init_args_with_default


def generate_py_primitive(yaml_data):
    """
    Generate python primitive
    """
    gen_py = ''
    for operator_name, operator_data in yaml_data.items():
        class_def = operator_data.get('class')
        class_disable = get_disable_flag(class_def)
        if class_disable:
            continue

        signature_code = generate_py_op_signature(operator_data.get('args_signature'))
        deprecated_code = generate_py_op_deprecated(operator_data.get('deprecated'))

        args = operator_data.get('args')
        class_name = get_op_name(operator_name, class_def)
        init_args, args_assign, init_args_with_default = process_args(args)
        init_code = '\n'.join(assign for assign in args_assign)

        labels = operator_data.get('labels')
        if labels is not None:
            if init_code != "":
                init_code += "\n"
            init_code += \
                '\n'.join([f"""        self.add_prim_attr("{key}", {value})""" for key, value in labels.items()])
        if init_code == "":
            init_code = f"""        pass"""

        primitive_code = f"""\n
class {class_name}(Primitive):\n"""
        if signature_code != "":
            primitive_code += signature_code
        if deprecated_code != "":
            primitive_code += deprecated_code
        primitive_code += f"""    @prim_arg_register
    def __init__(self, {', '.join(init_args_with_default) if init_args_with_default else ''}):
{init_code}

    def __call__(self, *args):
        return super().__call__(*args, {', '.join([f'self.{arg}' for arg in init_args])})
"""

        gen_py += primitive_code
    return gen_py


def generate_op_name_opdef(yaml_data):
    """
    Generate op name
    """
    op_name_head = f"""
#ifndef MINDSPORE_CORE_OP_NAME_H_
#define MINDSPORE_CORE_OP_NAME_H_

namespace mindspore::ops {{
"""

    op_name_end = f"""}}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OP_NAME_H_
"""

    op_name_gen = ''
    op_name_gen += op_name_head
    for operator_name, operator_data in yaml_data.items():
        k_name_op = get_op_name(operator_name, operator_data.get('class'))
        op_name_gen += f"""constexpr auto kName{k_name_op} = "{k_name_op}";
"""

    op_name_gen += op_name_end
    return op_name_gen


def generate_op_prim_opdef(yaml_data):
    """
    Generate primitive c++ definition
    """
    ops_prim_head = f"""
#ifndef MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
#define MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/gen_ops_name.h"
#include "mindapi/base/macros.h"

namespace mindspore::prim {{
"""

    ops_prim_end = f"""}}  // namespace mindspore::prim
#endif  // MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
"""

    ops_prim_gen = ''
    ops_prim_gen += ops_prim_head
    for operator_name, operator_data in yaml_data.items():
        k_name_op = get_op_name(operator_name, operator_data.get('class'))
        ops_prim_gen += f"""GVAR_DEF(PrimitivePtr, kPrim{k_name_op}, std::make_shared<Primitive>(ops::kName{k_name_op}))
"""
    ops_prim_gen += ops_prim_end
    return ops_prim_gen


def generate_lite_ops(yaml_data):
    """
    Generate BaseOperator parameter set and get func
    """
    lite_ops_head = f"""
#ifndef MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
#define MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_

#include "ops/base_operator.h"
#include "ops/gen_ops_name.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {{
"""

    lite_ops_end = f"""}}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
"""

    lite_ops_gen = ''
    lite_ops_gen += lite_ops_head
    for operator_name, operator_data in yaml_data.items():
        OpName = get_op_name(operator_name, operator_data.get('class'))
        lite_ops_gen += f"""class MIND_API {OpName} : public BaseOperator {{
 public:
  {OpName}() : BaseOperator(kName{OpName}) {{}}\n"""
        args = operator_data.get('args')
        for _, (arg_name, arg_info) in enumerate(args.items()):
            init = arg_info.get('init')
            if init is None:
                continue

            dtype = arg_info.get('dtype')
            if dtype == "str":
                dtype = "std::string"
            if dtype == "tuple[int]":
                dtype = "std::vector<int64_t>"
            lite_ops_gen += f"""  void set_{arg_name}(const {dtype} &{arg_name}) {{
    (void)this->AddAttr("{arg_name}", api::MakeValue({arg_name}));
  }}\n"""
            lite_ops_gen += f"""  {dtype} get_{arg_name}() const {{
    return GetValue<{dtype}>(GetAttr("{arg_name}"));
  }}\n"""

        lite_ops_gen += f"""}};\n\n"""
    lite_ops_gen += lite_ops_end
    return lite_ops_gen


def generate_cc_opdef(yaml_data):
    """
    Generate c++ OpDef
    """
    gen_cc_code = f"""\n
namespace mindspore::ops {{"""
    gen_opdef_map = f"""
std::unordered_map<std::string, OpDefPtr> gOpDefTable = {{"""
    gen_include = f"""\n
#include \"ops/op_def.h\""""

    for operator_name, operator_data in yaml_data.items():
        args = operator_data.get('args')
        returns = operator_data.get('returns')
        class_name = get_op_name(operator_name, operator_data.get('class'))
        gen_include += f"""\n#include "ops/ops_func_impl/{operator_name}.h\""""
        opdef_cc = f"""\n{class_name}FuncImpl g{class_name}FuncImpl;""" + \
                   f"""\nOpDef g{class_name} = {{
    .name_ = "{class_name}",""" + \
                   f"""\n    .args_ = {{"""
        cc_index_str = f"""\n    .indexes_ = {{"""
        gen_opdef_map += f"""\n    {{"{class_name}", &g{class_name}}},"""

        for i, (arg_name, arg_info) in enumerate(args.items()):
            cc_index_str += f"""
                {{"{arg_name}", {i}}},"""
            dtype = arg_info.get('dtype')
            cc_dtype_str = 'DT_' + dtype.replace('[', '_').replace(']', '').upper()

            init = arg_info.get('init')
            init_flag = 0 if init is None else 1
            arg_handler = arg_info.get('arg_handler')
            arg_handler_str = "" if arg_handler is None else arg_handler

            type_cast = arg_info.get('type_cast')
            type_cast_str = "" if type_cast is None else \
                ', '.join('DT_' + type.replace('[', '_').replace(']', '').upper() for type in
                          (ct.strip() for ct in type_cast.split(",")))

            opdef_cc += f"""
                {{.arg_name_ = "{arg_name}", .arg_dtype_ = {cc_dtype_str}, .as_init_arg_ = {init_flag}, .arg_handler_ = "{arg_handler_str}", .cast_dtype_ = {{{type_cast_str}}}}},"""
        opdef_cc += f"""\n    }},"""
        opdef_cc += f"""\n    .returns_ = {{"""

        for return_name, return_info in returns.items():
            return_dtype = return_info.get('dtype')
            cc_return_type_str = 'DT_' + return_dtype.replace('[', '_').replace(']', '').upper()
            opdef_cc += f"""
                {{.arg_name_ = "{return_name}", .arg_dtype_ = {cc_return_type_str}}},"""
        opdef_cc += f"""\n    }},"""

        cc_index_str += f"""\n    }},"""
        opdef_cc += cc_index_str

        cc_func_impl_str = f"""\n    .func_impl_ = &g{class_name}FuncImpl,"""
        opdef_cc += cc_func_impl_str
        opdef_cc += f"""\n}};"""
        gen_cc_code += opdef_cc

    gen_opdef_map += f"""\n}};"""
    gen_cc_code += gen_opdef_map

    cc_opdef_end = f"""\n}}  // namespace mindspore::ops"""
    return gen_include + gen_cc_code + cc_opdef_end


ops_py_header = f"""
\"\"\"Operators definition generated by gen_os.py, includes functions and primitive classes.\"\"\"

from mindspore.ops.primitive import Primitive, prim_arg_register
from mindspore.ops import signature as sig
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops_generate.arg_dtype_cast import type_it
from mindspore.ops.auto_generate.gen_arg_handler import *
"""


def generate_ops_py_files(work_path, yaml_str, doc_str, file_pre):
    """
    Generate ops python file from yaml.
    """
    py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/{file_pre}_ops_def.py')
    tmp_py_path = os.path.join(work_path, f'mindspore/python/mindspore/ops/auto_generate/tmp_{file_pre}_ops_def.py')

    py_prim = generate_py_primitive(yaml_str)
    py_func = generate_py_op_func(yaml_str, doc_str)

    with open(tmp_py_path, 'w') as py_file:
        py_file.write(py_licence_str + ops_py_header + py_prim + py_func)
    check_change_and_replace_file(py_path, tmp_py_path)


def generate_ops_cc_files(work_path, yaml_str):
    """
    Generate ops c++ file from yaml.
    """
    # ops_def
    op_cc_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_def.cc')
    tmp_op_cc_path = os.path.join(work_path, 'mindspore/core/ops/tmp_gen_ops_def.cc')
    cc_def_code = generate_cc_opdef(yaml_str)
    with open(tmp_op_cc_path, 'w') as cc_file:
        cc_file.write(cc_license_str + cc_def_code)
    check_change_and_replace_file(op_cc_path, tmp_op_cc_path)

    # ops_primitive
    op_prim_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_primitive.h')
    tmp_op_prim_path = os.path.join(work_path, 'mindspore/core/ops/tmp_gen_ops_primitive.h')
    op_prim_code = generate_op_prim_opdef(yaml_str)
    with open(tmp_op_prim_path, 'w') as op_prim_file:
        op_prim_file.write(cc_license_str + op_prim_code)
    check_change_and_replace_file(op_prim_path, tmp_op_prim_path)

    # lite_ops
    lite_ops_path = os.path.join(work_path, 'mindspore/core/ops/gen_lite_ops.h')
    tmp_lite_ops_path = os.path.join(work_path, 'mindspore/core/ops/tmp_gen_lite_ops.h')
    lite_ops_code = generate_lite_ops(yaml_str)
    with open(tmp_lite_ops_path, 'w') as lite_ops_file:
        lite_ops_file.write(cc_license_str + lite_ops_code)
    check_change_and_replace_file(lite_ops_path, tmp_lite_ops_path)

    # ops_names
    op_name_path = os.path.join(work_path, 'mindspore/core/ops/gen_ops_name.h')
    tmp_op_name_path = os.path.join(work_path, 'mindspore/core/ops/tmp_gen_ops_name.h')
    op_name_code = generate_op_name_opdef(yaml_str)
    with open(tmp_op_name_path, 'w') as op_name_file:
        op_name_file.write(cc_license_str + op_name_code)
    check_change_and_replace_file(op_name_path, tmp_op_name_path)


def generate_py_labels(yaml_data):
    """
    Generate python labels
    """
    gen_label_py = f"""op_labels = {{"""
    for operator_name, operator_data in yaml_data.items():
        labels = operator_data.get('labels')
        if labels is not None:
            class_name = get_op_name(operator_name, operator_data.get('class'))
            gen_label_py += f"""
    "{class_name}": {{"""
            gen_label_py += f""", """.join([f""""{key}": {value}""" for key, value in labels.items()])
            gen_label_py += f"""}}, """
    gen_label_py += f"""
}}"""
    return gen_label_py


def generate_labels_file(work_path, yaml_str):
    """
    Generate labels python file from yaml.
    """
    op_py_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/gen_labels.py')
    tmp_op_py_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/tmp_gen_labels.py')
    py_labels = generate_py_labels(yaml_str)
    with open(tmp_op_py_path, 'w') as py_file:
        py_file.write(py_licence_str + "\n" + py_labels + "\n")
    check_change_and_replace_file(op_py_path, tmp_op_py_path)


eum_py_header = f"""
\"\"\"Operator argument enum definition.\"\"\"

from enum import Enum
"""

eum_cc_header = f"""
#ifndef MINDSPORE_CORE_OPS_GEN_ENUM_DEF_
#define MINDSPORE_CORE_OPS_GEN_ENUM_DEF_

#include <cstdint>

namespace mindspore::ops {{
"""

eum_cc_end = f"""}}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_GEN_ENUM_DEF_
"""


def generate_enum_code(yaml_data):
    """
    Generate python and c++ enum definition
    """
    gen_eum_py_func = ''
    gen_eum_py_def = eum_py_header
    gen_eum_cc_def = eum_cc_header
    for enum_name, enum_data in yaml_data.items():
        class_name = ''.join(word.capitalize() for word in enum_name.split('_'))
        gen_eum_py_func += f"""
def {enum_name}_to_enum({enum_name}_str):
    if not isinstance({enum_name}_str, str):
        raise TypeError(f"The {enum_name} should be string, but got {{{enum_name}_str}}")
    {enum_name}_str = {enum_name}_str.upper()\n"""
        gen_eum_py_def += f"""\n\nclass {class_name}(Enum):\n"""
        gen_eum_cc_def += f"""enum class {class_name} : int64_t {{\n"""

        for enum_key, enum_value in enum_data.items():
            gen_eum_py_func += f"""    if {enum_name}_str == "{enum_key}":
        return {enum_value}\n"""
            gen_eum_py_def += f"""    {enum_key} = {enum_value}\n"""
            gen_eum_cc_def += f"""    {enum_key} = {enum_value},\n"""

        gen_eum_py_func += f"""    raise ValueError(f"Invalid {class_name}: {{{enum_name}_str}}")\n\n"""
        gen_eum_cc_def += f"""}};\n\n"""
    gen_eum_cc_def += eum_cc_end

    return gen_eum_py_func, gen_eum_py_def, gen_eum_cc_def


def generate_enum_files(work_path):
    """
    Generate python function and c++ definition from enum yaml.
    """
    enum_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/enum.yaml')
    yaml_str = safe_load_yaml(enum_yaml_path)
    py_enum_func, py_enum_def, cc_enum_def = generate_enum_code(yaml_str)

    src_arg_handler_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/arg_handler.py')
    dst_arg_handler_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/gen_arg_handler.py')
    tmp_dst_arg_handler_path = os.path.join(work_path,
                                            'mindspore/python/mindspore/ops/auto_generate/tmp_gen_arg_handler.py')
    os.system(f'cp {src_arg_handler_path} {tmp_dst_arg_handler_path}')
    with open(tmp_dst_arg_handler_path, 'a') as py_file:
        py_file.write(py_enum_func)
    check_change_and_replace_file(dst_arg_handler_path, tmp_dst_arg_handler_path)

    enum_def_py_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/gen_enum_def.py')
    tmp_enum_def_py_path = os.path.join(work_path, 'mindspore/python/mindspore/ops/auto_generate/tmp_gen_enum_def.py')
    with open(tmp_enum_def_py_path, 'w') as cc_file:
        cc_file.write(py_licence_str + py_enum_def)
    check_change_and_replace_file(enum_def_py_path, tmp_enum_def_py_path)

    enum_def_cc_path = os.path.join(work_path, 'mindspore/core/ops/gen_enum_def.h')
    tmp_enum_def_cc_path = os.path.join(work_path, 'mindspore/core/ops/tmp_gen_enum_def.h')
    with open(tmp_enum_def_cc_path, 'w') as cc_file:
        cc_file.write(cc_license_str + cc_enum_def)
    check_change_and_replace_file(enum_def_cc_path, tmp_enum_def_cc_path)


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    work_path = os.path.join(current_path, '../../../../')

    # merge ops yaml
    ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops.yaml')
    doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/ops_doc.yaml')
    yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/')
    merge_files(yaml_dir_path, ops_yaml_path, '*op.yaml')
    merge_files(yaml_dir_path, doc_yaml_path, '*doc.yaml')

    # merge inner ops yaml
    inner_ops_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/inner_ops.yaml')
    inner_doc_yaml_path = os.path.join(work_path, 'mindspore/python/mindspore/ops_generate/inner_ops_doc.yaml')
    inner_yaml_dir_path = os.path.join(work_path, 'mindspore/core/ops/ops_def/inner')
    merge_files(inner_yaml_dir_path, inner_ops_yaml_path, '*op.yaml')
    merge_files(inner_yaml_dir_path, inner_doc_yaml_path, '*doc.yaml')

    # generate ops python files
    generate_ops_py_files(work_path, safe_load_yaml(ops_yaml_path), safe_load_yaml(doc_yaml_path), "gen")
    generate_ops_py_files(work_path, safe_load_yaml(inner_ops_yaml_path), safe_load_yaml(inner_doc_yaml_path),
                          "gen_inner")

    all_ops_str = {**safe_load_yaml(ops_yaml_path), **safe_load_yaml(inner_ops_yaml_path)}
    # generate ops c++ files
    generate_ops_cc_files(work_path, all_ops_str)
    # generate ops label python files
    generate_labels_file(work_path, all_ops_str)

    # generate enum code from enum.yaml
    generate_enum_files(work_path)


if __name__ == "__main__":
    main()
