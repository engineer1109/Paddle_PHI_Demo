#include <gtest/gtest.h>

#include <memory>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

using DDim = phi::DDim;

TEST(API, add_test)
{
    // 1. create tensor
    const auto alloc = std::make_unique<paddle::experimental::DefaultAllocator>(
        paddle::platform::CPUPlace());
    auto dense_x = std::make_shared<phi::DenseTensor>(
        alloc.get(),
        phi::DenseTensorMeta(phi::DataType::FLOAT32,
                             phi::make_ddim({3, 3}),
                             phi::DataLayout::NCHW));
    auto *dense_x_data =
        dense_x->mutable_data<float>(paddle::platform::CPUPlace());

    auto dense_y = std::make_shared<phi::DenseTensor>(
        alloc.get(),
        phi::DenseTensorMeta(phi::DataType::FLOAT32,
                             phi::make_ddim({3, 3}),
                             phi::DataLayout::NCHW));
    auto *dense_y_data =
        dense_y->mutable_data<float>(paddle::platform::CPUPlace());

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            dense_x_data[i * 3 + j] = float(i * 3 + j) * 1.0f;
            dense_y_data[i * 3 + j] = float(i * 3 + j) * 1.0f;
        }
    }

    paddle::experimental::Tensor x(dense_x);
    paddle::experimental::Tensor y(dense_y);

    std::vector<paddle::experimental::Tensor> inputs{x, y};

    // 2. test API
    auto out = paddle::experimental::add(x, y);

    // 3. check result
    ASSERT_EQ(out.dims().size(), 2);
    ASSERT_EQ(out.dims()[0], 3);
    ASSERT_EQ(out.dims()[1], 3);
    ASSERT_EQ(out.numel(), 9);
    ASSERT_EQ(out.is_cpu(), true);
    ASSERT_EQ(out.type(), phi::DataType::FLOAT32);
    ASSERT_EQ(out.layout(), phi::DataLayout::NCHW);
    ASSERT_EQ(out.initialized(), true);

    auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out.impl());
    auto out_data = dense_out->data<float>();

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            ASSERT_NEAR(out_data[i * 3 + j], float(i * 3 + j) * 2.0f, 1e-6f);
        }
    }

}
