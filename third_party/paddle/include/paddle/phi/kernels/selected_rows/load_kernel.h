// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/selected_rows.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void LoadKernel(const Context& dev_ctx,
                const std::string& file_path,
                int64_t seek,
                const std::vector<int64_t>& shape,
                bool load_as_fp16,
                SelectedRows* out);

}  // namespace sr
}  // namespace phi