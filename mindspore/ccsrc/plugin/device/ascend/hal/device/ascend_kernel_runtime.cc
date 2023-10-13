/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include <locale>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include "ops/ascend_op_name.h"
#include "ops/other_ops.h"
#include "include/common/utils/signal_util.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "utils/ms_context.h"
#include "runtime/rt.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include "plugin/device/ascend/hal/device/ascend_runtime_manager.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_load.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "kernel/oplib/op_info_utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_event.h"
#ifndef ENABLE_SECURITY
#include "toolchain/prof_api.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#endif
#include "toolchain/adx_datadump_server.h"
#include "utils/trace_base.h"
#include "external/acl/error_codes/rt_error_codes.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/common/mem_reuse/mem_reuse_checker.h"
#include "include/common/debug/env_config_parser.h"
#endif
#include "include/common/utils/config_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

#include "kernel/framework_utils.h"
#include "plugin/device/ascend/hal/common/platform_info_util.h"
using std::vector;
constexpr uint32_t kProfilingMaxTaskIdInStream = 65531;
constexpr uint32_t kDefaultHcclExecTimeout = 1800;

namespace mindspore::device::ascend {
static thread_local rtContext_t thread_local_rt_context{nullptr};
namespace {
void IntHandler(int, siginfo_t *, void *) {
  mindspore::kernel::AscendKernelBuildClient::Instance().Close();
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}

void AscendEnableDynamicRuntimeCache(const session::KernelGraph *graph) {
  const auto &node_list = FuncGraph::TopoSort(graph->get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    auto kernel_info = node->kernel_info();
    if (!kernel_info) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_info);
    auto runtime_cache = kernel_info->runtime_cache();
    runtime_cache.runtime_cache().set_is_valid(true);
  }
}
}  // namespace

struct TbeLaunchKernelModRegister {
  TbeLaunchKernelModRegister() {
    KernelRuntime::tbe_call_setter(
      [](const AnfNodePtr &kernel, const kernel::KernelMod *kernel_mod, std::vector<AddressPtr> *workspace_addr) {
        MS_EXCEPTION_IF_NULL(kernel);
        MS_EXCEPTION_IF_NULL(kernel_mod);
        MS_EXCEPTION_IF_NULL(workspace_addr);
        auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
        auto ms_context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(ms_context);
        auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
        MS_EXCEPTION_IF_NULL(runtime_instance);
        for (auto size : workspace_size_list) {
          auto device_address_ptr =
            std::make_shared<ascend::AscendDeviceAddress>(nullptr, size, kAscendDevice, device_id);
          device_address_ptr->set_is_ptr_persisted(true);
          auto ret = runtime_instance->GetMemoryManager()->MallocMemFromMemPool(device_address_ptr, size);
          if (!ret) {
            MS_LOG(EXCEPTION) << "MallocMem from memory pool failed. Node info :" << kernel->fullname_with_scope();
          }
          AddressPtr workspace_addr_ptr =
            std::make_shared<kernel::Address>(device_address_ptr->GetMutablePtr(), device_address_ptr->GetSize());
          (void)workspace_addr->emplace_back(workspace_addr_ptr);
        }
      });
  }
  TbeLaunchKernelModRegister(const TbeLaunchKernelModRegister &) = delete;
  TbeLaunchKernelModRegister &operator=(const TbeLaunchKernelModRegister &) = delete;
  ~TbeLaunchKernelModRegister() = default;
} tbe_launch_kernel_mod_register;

AscendKernelRuntime::~AscendKernelRuntime() {
  current_graph_ = nullptr;
  rt_context_ = nullptr;
}

void AscendKernelRuntime::SetContext() {
  ErrorManagerAdapter::BindToCurrentThread();
  if (rt_context_ == nullptr) {
    return;
  }
  if (thread_local_rt_context == rt_context_) {
    return;
  }
  auto ret = aclrtSetCurrentContext(rt_context_);
  thread_local_rt_context = rt_context_;
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::SetContextForce() {
  if (rt_context_ == nullptr) {
    return;
  }
  auto ret = aclrtSetCurrentContext(rt_context_);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
}

void AscendKernelRuntime::ClearGraphModelMap() {
  SetContextForce();
  graph_kernel_events_map_.clear();
  if (runtime_core_ != nullptr) {
    runtime_core_->UnloadModelCore();
  }
}

void AscendKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  SetContextForce();
  auto mem_scheduler = mem_scheduler_manager_.GetMemScheduler(graph_id);
  if (mem_scheduler != nullptr) {
    mem_scheduler->Clear();
  }
  const auto events_iter = graph_kernel_events_map_.find(graph_id);
  if (events_iter != graph_kernel_events_map_.end()) {
    (void)graph_kernel_events_map_.erase(events_iter);
  }
  if (runtime_core_ != nullptr) {
    runtime_core_->UnloadModelCore(graph_id);
  }
}

void *AscendKernelRuntime::GetModelStream(uint32_t graph_id) const {
  if (runtime_core_ != nullptr) {
    return runtime_core_->GetModelStreamCore(graph_id);
  }
  return nullptr;
}

void *AscendKernelRuntime::GetKernelStream(const AnfNodePtr &kernel) const {
  const auto stream = AscendStreamMng::GetInstance().GetStream(AnfAlgo::GetStreamId(kernel));
  if (stream == nullptr) {
    // Stream id may not be assigned in some scenarios, such as PyNative. Use the default stream in those cases.
    return stream_;
  }
  return stream;
}

void AscendKernelRuntime::ClearGlobalIdleMem() {
  if (mem_manager_ != nullptr) {
    mem_manager_->ClearGlobalIdleMem();
  }
}

bool AscendKernelRuntime::NeedDestroyHccl() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
    MS_LOG(INFO) << "Hccl is not enabled";
    return false;
  }
  // Note: make sure hcom_connectivity_detection api never be used.
  return true;
}

#ifndef ENABLE_SECURITY
void AsyncDataDumpUninit() {
  if (DumpJsonParser::GetInstance().async_dump_enabled()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    if (device_type == kAscendDevice) {
      // When it is A+M dump mode, wait until file save is finished.
      if (DumpJsonParser::GetInstance().FileFormatIsNpy()) {
        mindspore::ascend::AscendAsyncDumpManager::GetInstance().WaitForWriteFileFinished();
      }
    }
    if (AdxDataDumpServerUnInit() != 0) {
      MS_LOG(ERROR) << "Adx data dump server uninit failed";
    }
  }
}
#endif

void AscendKernelRuntime::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Ascend finalize start";
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->debugger_enabled()) {
    debugger_->SetTrainingDone(true);
    bool ret = debugger_->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  SetContextForce();

  // release ge runtime
  ClearGraphModelMap();

#ifndef ENABLE_SECURITY
  AsyncDataDumpUninit();
#endif

  PlatformInfoUtil::GetInstance().Finalize();

  mindspore::kernel::AicpuOpKernelLoad::GetInstance().FreeDeviceMemory();
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  // DestroyHccl must be called before FreeDeviceMemory
  (void)DestroyHccl();
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }

  if (runtime_core_ != nullptr) {
    runtime_core_->RegTaskFailCallback(true);
  }

  (void)ResetDevice(device_id);
  current_graph_ = nullptr;
  initialized_ = false;
  MS_LOG(INFO) << "Ascend finalize end";
}

#ifndef ENABLE_SECURITY
void AscendKernelRuntime::PreInit() {
  if (!ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
}
#endif

bool AscendKernelRuntime::Init() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  auto profiling_flag = profiler_manager->GetProfilingEnableFlag();
  if (execution_mode == kPynativeMode && profiling_flag) {
    pynative_mode_profiling_flag_ = true;
  }
#endif
  if (initialized_) {
    SetContextForce();
    return true;
  }

  auto soc_version = device::ascend::GetSocVersion();
  auto ascend_path = device::ascend::GetAscendPath();
  if (!mindspore::kernel::OpInfoUtils::GenerateOpInfos(soc_version, ascend_path)) {
    MS_LOG(EXCEPTION) << "Load op info form json config failed, version: " << soc_version;
  }
  if (!ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "Init ErrorManager failed.";
  }
  bool init_device = false;
  try {
    // Start up profiling before aclrtSetDevice
    bool ret = InitDevice();
    if (!ret) {
      return ret;
    }
    init_device = true;
#ifdef ENABLE_DEBUGGER
    SetDebugger();
#endif
    mem_manager_ = std::make_shared<AscendMemoryManager>();
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->Initialize();
    runtime_core_ = AscendRuntimeManager::Instance().GetAscendRuntime(kAscendVM, device_id_);
    if (runtime_core_ != nullptr) {
      runtime_core_->InitCore();
      runtime_core_->RegTaskFailCallback();
    }
    if (!PlatformInfoUtil::GetInstance().Init(soc_version)) {
      MS_LOG(EXCEPTION) << "PlatformInfo Initialization failed.";
    }
    uint32_t op_execute_timeout = ms_context->get_param<uint32_t>(MS_CTX_OP_TIMEOUT);
    std::string hccl_exec_timeout = common::GetEnv("HCCL_EXEC_TIMEOUT");
    uint32_t notify_wait_timeout;
    if (hccl_exec_timeout.empty()) {
      notify_wait_timeout = kDefaultHcclExecTimeout;
    } else {
      try {
        notify_wait_timeout = std::stoi(hccl_exec_timeout);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Parse environment variable HCCL_EXEC_TIMEOUT failed, value" << hccl_exec_timeout
                      << ", msg: " << e.what();
        return false;
      }
    }
    if (op_execute_timeout >= notify_wait_timeout) {
      MS_LOG(WARNING) << "OpExecuteTimeout should be less than NotifyWaitTimeout, but got OpExecuteTimeout "
                      << op_execute_timeout << ", notify_wait_timeout " << notify_wait_timeout << "."
                      << "1. You can set OpExecuteTimeout via mindspore.set_context(op_timeout=int)."
                      << "2. You can set NotifyWaitTimeout via environment variable HCCL_EXEC_TIMEOUT. ";
    }
    const uint32_t reserve_time = 180;
    uint32_t op_wait_timeout = kDefaultHcclExecTimeout + reserve_time;
    auto acl_ret = aclrtSetOpWaitTimeout(op_wait_timeout);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Set op wait timeout failed, error: " << acl_ret;
    }
    acl_ret = aclrtSetOpExecuteTimeOut(op_execute_timeout);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Set op execute timeout failed, error: " << acl_ret;
    }
  } catch (const std::exception &e) {
    if (init_device) {
      ResetDevice(device_id_);
    }
    MS_LOG(EXCEPTION) << "Ascend kernel runtime initialization failed. The details refer to 'Ascend Error Message'."
                      << "#dmsg#Framework Error Message:#dmsg#" << e.what();
  }

  initialized_ = true;
  return true;
}

bool AscendKernelRuntime::LoadData(const session::KernelGraph & /* graph */) {
  if (runtime_core_ != nullptr) {
    return runtime_core_->LoadDataCore();
  }
  return true;
}

bool AscendKernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  bool need_dump = false;
#ifndef ENABLE_SECURITY
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 1) {
    auto op_name = node->fullname_with_scope();
    if (dump_json_parser.NeedDump(op_name)) {
      need_dump = true;
    }
  }
#endif
  return need_dump;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr =
    std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id, kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(ascend_device_address_ptr);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

DeviceAddressPtr AscendKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                          TypeId type_id, const KernelWithIndex &node_index) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto ascend_device_address_ptr = std::make_shared<AscendDeviceAddress>(device_ptr, device_size, format, type_id,
                                                                         node_index, kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(ascend_device_address_ptr);
  ascend_device_address_ptr->set_is_ptr_persisted(true);
  return ascend_device_address_ptr;
}

bool AscendKernelRuntime::Load(const session::KernelGraph &graph, bool is_task_sink) {
  if (runtime_core_ != nullptr) {
    return runtime_core_->LoadCore(graph, is_task_sink);
  }
  return true;
}

bool AscendKernelRuntime::Run(const session::KernelGraph &graph, bool is_task_sink) {
  const uint64_t kUSecondInSecond = 1000000;
  SignalGuard sg(IntHandler);
  bool ret = false;

  if (is_task_sink) {
#if defined(_WIN32) || defined(_WIN64)
    auto start_time = std::chrono::steady_clock::now();
#else
    struct timeval start_time {};
    struct timeval end_time {};
    (void)gettimeofday(&start_time, nullptr);
#endif
    ret = RunTask(graph);
#if defined(_WIN32) || defined(_WIN64)
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
    MS_LOG(INFO) << "Call MS Run Success in " << cost.count() << " us";
#else
    (void)gettimeofday(&end_time, nullptr);
    uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
    cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
    MS_LOG(INFO) << "Call MS Run Success in " << cost << " us";
#endif
  } else {
    ret = LaunchKernels(graph);
  }

  return ret;
}

void AscendKernelRuntime::GetShadowBackendNodeMap(const session::KernelGraph &graph,
                                                  std::map<AnfNodePtr, AnfNodePtr> *shadow_backend_node_map) {
  auto &input_nodes = graph.input_nodes();
  MS_EXCEPTION_IF_NULL(shadow_backend_node_map);
  std::map<AnfNodePtr, AnfNodePtr> front_nodes_map;
  for (auto &node : input_nodes) {
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(node, graph);
    if (front_node == nullptr || common::AnfAlgo::IsTupleOutput(front_node)) {
      continue;
    }
    auto iter = front_nodes_map.find(front_node);
    if (iter != front_nodes_map.end()) {
      if (node != iter->second) {
        (void)shadow_backend_node_map->emplace(node, iter->second);
      }
    } else {
      (void)front_nodes_map.emplace(front_node, node);
    }
  }
}

DeviceAddressPtr AscendKernelRuntime::GetInternalDeviceAddress(const session::KernelGraph &graph,
                                                               const AnfNodePtr &node) {
  auto front_node = graph.GetFrontNodeByInternalParameter(node);
  if (front_node.first == nullptr) {
    return nullptr;
  }
  auto pre_graphs = graph.get_pre_graphs();
  for (auto pre_graph_item : pre_graphs) {
    auto pre_graph = pre_graph_item.second.lock();
    MS_EXCEPTION_IF_NULL(pre_graph);
    auto graph_output = pre_graph->GetGraphOutputByFrontNode(front_node);
    if (graph_output.first == nullptr) {
      continue;
    }
    if (!AnfAlgo::OutputAddrExist(graph_output.first, graph_output.second)) {
      return nullptr;
    }
    auto output_device_address = AnfAlgo::GetMutableOutputAddr(graph_output.first, graph_output.second);
    MS_EXCEPTION_IF_NULL(output_device_address);
    if (output_device_address->GetDeviceType() == DeviceType::kAscend) {
      return output_device_address;
    }
  }
  return nullptr;
}

void AscendKernelRuntime::GenKernelEvents(const session::KernelGraph &graph) {
  if (runtime_core_ != nullptr) {
    runtime_core_->GenKernelEventsCore(graph);
  }
}

bool AscendKernelRuntime::RunDynamicKernelAsync(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "RunExecutorAsync start. GraphId:" << graph.graph_id();
  AscendEnableDynamicRuntimeCache(&graph);

  const auto &kernels = graph.execution_order();
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::GetCNodeName(kernel) == kMemSetOpName) {
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto depends = abstract::GetValueDependArgIndices(kernel);
    if (!depends.empty() || AnfAlgo::GetKernelType(kernel) == KernelType::HCCL_KERNEL) {
      MS_LOG(INFO) << "Match Dynamic Kernel, Start SyncStream";
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
    }

    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      opt::InferOp(kernel);
      auto args = kernel->user_data<kernel::KernelArgs>();
      MS_EXCEPTION_IF_NULL(args);
      (void)kernel_mod->Resize(args->inputs, args->outputs, args->depend_tensor_map);
    }
    KernelLaunchInfo kernel_launch_info;
    device::KernelRuntime::GenLaunchArgs(*kernel_mod, kernel, &kernel_launch_info);
    // allocate workspace size
    std::vector<AddressPtr> workspace_addr;
    if (common::AnfAlgo::IsDynamicShape(kernel) && AnfAlgo::GetKernelType(kernel) == KernelType::TBE_KERNEL) {
      auto workspace_size_list = kernel_mod->GetWorkspaceSizeList();
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      auto runtime_instance = KernelRuntimeManager::Instance().GetSingleKernelRuntime(kAscendDevice, device_id);
      MS_EXCEPTION_IF_NULL(runtime_instance);

      for (auto size : workspace_size_list) {
        auto device_address_ptr = std::make_shared<AscendDeviceAddress>(nullptr, size, kAscendDevice, device_id);
        MS_EXCEPTION_IF_NULL(device_address_ptr);
        device_address_ptr->set_is_ptr_persisted(true);
        auto device_ptr = runtime_instance->MallocMem(MemType::kDynamicMem, size, device_address_ptr);
        if (device_ptr == nullptr) {
          MS_LOG(EXCEPTION) << "MallocMem from memory pool failed. Node info :" << kernel->fullname_with_scope();
        }

        AddressPtr workspace_addr_ptr =
          std::make_shared<kernel::Address>(device_address_ptr->GetMutablePtr(), device_address_ptr->GetSize());
        (void)workspace_addr.emplace_back(workspace_addr_ptr);
      }
    } else {
      workspace_addr = kernel_launch_info.workspaces_;
    }

    auto ret = kernel_mod->Launch(kernel_launch_info.inputs_, workspace_addr, kernel_launch_info.outputs_, stream_);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      return false;
    }
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      kernel::UpdateNodeShape(kernel);
    }
  }

  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }

  return true;
}

bool AscendKernelRuntime::RunTask(const session::KernelGraph &graph) {
  current_graph_ = &graph;
  SetContextForce();
  if (graph.is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic Shape Graph Run Task Async";
    return RunDynamicKernelAsync(graph);
  }
  if (runtime_core_ != nullptr) {
    return runtime_core_->RunTaskCore(graph);
  }
  return true;
}

bool AscendKernelRuntime::SyncStream() {
  SetContextForce();
  if (stream_ != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_);
    if (!AscendStreamMng::GetInstance().SyncStream(stream_)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  if (communication_stream_ != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(communication_stream_);
    if (!AscendStreamMng::GetInstance().SyncStream(communication_stream_)) {
      MS_LOG(ERROR) << "Sync default stream failed.";
      return false;
    }
  }
  return true;
}

bool AscendKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  SetContextForce();
  if (size == 0) {
    MS_LOG(DEBUG) << "rtMemcpyAsync size is 0, copy kind:" << kind;
    return true;
  }
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "MemcpyAsync failed. stream_ is nullptr";
    return false;
  }

  auto copy_kind = static_cast<rtMemcpyKind_t>(kind);
  if (copy_kind != RT_MEMCPY_HOST_TO_DEVICE_EX && copy_kind != RT_MEMCPY_DEVICE_TO_DEVICE) {
    MS_LOG(EXCEPTION) << "Memory copy async not support cache host buffer in kind: " << kind;
  }
  if (dst == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync dst ptr is null, copy kind:" << kind;
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "rtMemcpyAsync src ptr is null, copy kind:" << kind;
    return false;
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_);
  if (RT_ERROR_NONE != rtMemcpyAsync(dst, size, src, size, static_cast<rtMemcpyKind_t>(kind), stream_)) {
    MS_LOG(ERROR) << "Call runtime rtMemcpyAsync error.";
    return false;
  }
  return true;
}

void AscendKernelRuntime::SetRtDevice(uint32_t device_id) {
  MS_LOG(INFO) << "Enter SetRtDevice, current initialize device number:" << initialized_device_set_.size();
  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    MS_LOG(INFO) << "Device " << device_id << " has been set";
    return;
  }

  uint32_t device_count = 0;
  auto ret = aclrtGetDeviceCount(&device_count);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }

  ret = aclrtSetDevice(UintToInt(device_id));
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call aclrtSetDevice, ret[" << static_cast<int>(ret) << "]";
  }
  (void)initialized_device_set_.insert(device_id);
}

void AscendKernelRuntime::CreateDefaultStream() {
  size_t compute_stream_id;
  AscendStreamMng::GetInstance().CreateStreamWithFlags(&compute_stream_id, RT_STREAM_HUGE);
  MS_LOG(INFO) << "Create ascend default stream, stream id: " << compute_stream_id;
  stream_ = AscendStreamMng::GetInstance().GetStream(compute_stream_id);
  MS_EXCEPTION_IF_NULL(stream_);

  size_t communication_stream_id;
  AscendStreamMng::GetInstance().CreateStream(&communication_stream_id);
  MS_LOG(INFO) << "Create ascend communication stream, stream id: " << communication_stream_id;
  communication_stream_ = AscendStreamMng::GetInstance().GetStream(communication_stream_id);
  MS_EXCEPTION_IF_NULL(communication_stream_);
}

bool AscendKernelRuntime::InitDevice() {
  SetRtDevice(device_id_);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr == nullptr) {
    MS_LOG(ERROR) << "Get MsContext instance failed";
    return false;
  }

  // Context will be created by aclrtSetDevice
  const auto rt_ret = aclrtGetCurrentContext(&rt_context_);
  if (rt_ret != ACL_ERROR_NONE || rt_context_ == nullptr) {
    MS_LOG(ERROR) << "Call aclrtGetCurrentContext failed, ret[" << rt_ret << "]";
    return false;
  }

  CreateDefaultStream();
  return true;
}

bool AscendKernelRuntime::ResetDevice(uint32_t device_id) {
  SetContextForce();
  AscendStreamMng::GetInstance().DestroyAllRtEvents();
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    MS_LOG(ERROR) << "Fail to destroy all streams when reset device.";
    return false;
  }
  stream_ = nullptr;
  communication_stream_ = nullptr;

  if (initialized_device_set_.find(device_id) != initialized_device_set_.end()) {
    auto ret = aclrtResetDevice(UintToInt(device_id));
    if (ret != ACL_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call aclrtResetDevice, ret[" << ret << "]";
    }
    (void)initialized_device_set_.erase(device_id);
  }

  // set to nullptr as its not created, only bounded to existing context
  rt_context_ = nullptr;
  return true;
}

bool AscendKernelRuntime::DestroyHccl() {
  if (!NeedDestroyHccl()) {
    MS_LOG(INFO) << "Hccl is not enable, no need to close.";
    return true;
  }
  if (!AscendCollectiveCommLib::GetInstance().DestroyHcclComm()) {
    MS_LOG(ERROR) << "Hccl destroy failed.";
    return false;
  }
  MS_LOG(INFO) << "Hccl destroy successful.";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  context_ptr->set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  return true;
}

void AscendKernelRuntime::KernelLaunchProfiling(const std::string &kernel_name) {
#ifndef ENABLE_SECURITY
  auto profiler_manager = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manager);
  if (!profiler_manager->GetProfilingEnableFlag()) {
    return;
  }

  // save task info
  uint32_t stream_id;
  uint32_t task_id;
  auto rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Profiling get task_id stream_id failed";
  }
  std::pair<uint32_t, uint32_t> stream_task_pair = {stream_id, task_id};
  auto try_emplace_ret = stream_id_task_id_op_name_map_.try_emplace(stream_task_pair, kernel_name);
  if (!try_emplace_ret.second) {
    MS_LOG(WARNING) << "Profiling duplicate key, task_id:" << stream_task_pair.second
                    << " stream_id:" << stream_task_pair.first << " name:" << kernel_name;
  }
  if (stream_id_task_id_op_name_map_.size() > kProfilingMaxTaskIdInStream) {
    MS_LOG(EXCEPTION) << "Too many profiling data";
  }
#endif
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceEvent() {
  auto ascend_event = std::make_shared<AscendEvent>();
  MS_EXCEPTION_IF_NULL(ascend_event);
  return ascend_event;
}

std::shared_ptr<DeviceEvent> AscendKernelRuntime::CreateDeviceTimeEvent() {
  auto ascend_time_event = std::make_shared<AscendTimeEvent>();
  MS_EXCEPTION_IF_NULL(ascend_time_event);
  return ascend_time_event;
}

uint64_t AscendKernelRuntime::GetMsUsedHbmSize() const {
  auto ascend_mem_manager = std::dynamic_pointer_cast<AscendMemoryManager>(mem_manager_);
  MS_EXCEPTION_IF_NULL(ascend_mem_manager);
  return ascend_mem_manager->GetMsUsedHbmSize();
}

void AscendKernelRuntime::SetReuseCommunicationAddress(const session::KernelGraph &graph) {
  auto cnode_list = graph.execution_order();
  for (const auto &cnode : cnode_list) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, cnode)) {
      auto reuse_index = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrReuseCommunication);
      if (reuse_communication_address_.find(reuse_index) == reuse_communication_address_.end()) {
        (void)reuse_communication_address_.emplace(reuse_index, std::make_pair(nullptr, nullptr));
      }
    }
  }
}
}  // namespace mindspore::device::ascend
