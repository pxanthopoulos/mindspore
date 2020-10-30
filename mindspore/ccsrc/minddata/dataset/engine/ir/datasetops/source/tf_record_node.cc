/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/status.h"
#include "utils/system/crc32c.h"

namespace mindspore {
namespace dataset {
namespace api {

bool ValidateFirstRowCrc(const std::string &filename) {
  std::ifstream reader;
  reader.open(filename);
  if (!reader) {
    return false;
  }

  // read data
  int64_t record_length = 0;
  (void)reader.read(reinterpret_cast<char *>(&record_length), static_cast<std::streamsize>(sizeof(int64_t)));

  // read crc from file
  uint32_t masked_crc = 0;
  (void)reader.read(reinterpret_cast<char *>(&masked_crc), static_cast<std::streamsize>(sizeof(uint32_t)));

  // generate crc from data
  uint32_t generated_crc =
    system::Crc32c::GetMaskCrc32cValue(reinterpret_cast<char *>(&record_length), sizeof(int64_t));

  return masked_crc == generated_crc;
}

// Validator for TFRecordNode
Status TFRecordNode::ValidateParams() {
  std::vector<std::string> invalid_files(dataset_files_.size());
  auto it = std::copy_if(dataset_files_.begin(), dataset_files_.end(), invalid_files.begin(),
                         [](const std::string &filename) { return !ValidateFirstRowCrc(filename); });
  invalid_files.resize(std::distance(invalid_files.begin(), it));
  std::string err_msg;
  if (!invalid_files.empty()) {
    err_msg += "Invalid file, the following files either cannot be opened, or are not valid tfrecord files:\n";

    std::string accumulated_filenames = std::accumulate(
      invalid_files.begin(), invalid_files.end(), std::string(""),
      [](const std::string &accumulated, const std::string &next) { return accumulated + "    " + next + "\n"; });
    err_msg += accumulated_filenames;
  }
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
}

// Function to build TFRecordNode
std::vector<std::shared_ptr<DatasetOp>> TFRecordNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Sort the datasets file in a lexicographical order
  std::vector<std::string> sorted_dir_files = dataset_files_;
  std::sort(sorted_dir_files.begin(), sorted_dir_files.end());

  // Create Schema Object
  std::unique_ptr<DataSchema> data_schema = std::make_unique<DataSchema>();
  if (!schema_path_.empty()) {
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaFile(schema_path_, columns_list_));
  } else if (schema_obj_ != nullptr) {
    std::string schema_json_string = schema_obj_->to_json();
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaString(schema_json_string, columns_list_));
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // TFReaderOp by itself is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  std::shared_ptr<SamplerObj> sampler_ = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);

  // Create and initialize TFReaderOp
  std::shared_ptr<TFReaderOp> tf_reader_op =
    std::make_shared<TFReaderOp>(num_workers_, worker_connector_size_, rows_per_buffer_, num_samples_, sorted_dir_files,
                                 std::move(data_schema), connector_que_size_, columns_list_, shuffle_files, num_shards_,
                                 shard_id_, shard_equal_rows_, std::move(sampler_->Build()));

  RETURN_EMPTY_IF_ERROR(tf_reader_op->Init());

  if (cache_ == nullptr && shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp

    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(TFReaderOp::CountTotalRows(&num_rows, sorted_dir_files));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dir_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  // Add TFReaderOp
  node_ops.push_back(tf_reader_op);
  return node_ops;
}

// Get the shard id of node
Status TFRecordNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
