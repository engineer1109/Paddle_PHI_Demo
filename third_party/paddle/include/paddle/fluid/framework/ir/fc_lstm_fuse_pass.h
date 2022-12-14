// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

// The MulLstmFusePass and MulLstmFusePass will fuse to the same FusionLstm op.

// Just FC without bias
class Graph;

class FCLstmFusePass : public FusePassBase {
 public:
  FCLstmFusePass();
  virtual ~FCLstmFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope,
                  bool with_fc_bias) const;
  const std::string name_scope_{"fc_lstm_fuse"};
};

class MulLstmFusePass : public FCLstmFusePass {
 public:
  MulLstmFusePass();
  virtual ~MulLstmFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"fc_nobias_lstm_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
