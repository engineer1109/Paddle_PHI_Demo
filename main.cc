#include <gtest/gtest.h>

#include "paddle/fluid/platform/init.h"

int main(int argc, char **argv)
{
    //paddle::memory::allocation::UseAllocatorStrategyGFlag();
    testing::InitGoogleTest(&argc, argv);

    paddle::framework::InitDevices();
    // paddle::framework::InitDefaultKernelSignatureMap();

    int ret = RUN_ALL_TESTS();

    return ret;
}