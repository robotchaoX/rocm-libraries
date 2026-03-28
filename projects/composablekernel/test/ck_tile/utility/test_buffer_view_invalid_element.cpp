// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename T>
__global__ void kernel_read_invalid_customized_vector(T* data, T* out, T invalid_value)
{
    auto buffer_view = make_buffer_view<address_space_enum::global>(data, 4, invalid_value);
    auto vec         = buffer_view.template get<thread_buffer<T, 4>>(0, 0, false);

    out[0] = vec[0];
    out[1] = vec[1];
    out[2] = vec[2];
    out[3] = vec[3];
}

class BufferViewInvalidElementTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        if(hipSetDevice(0) != hipSuccess)
        {
            GTEST_SKIP() << "No GPU available for buffer_view invalid-element test";
        }
    }
};

TEST_F(BufferViewInvalidElementTest, GlobalBufferViewBroadcastsCustomizedInvalidValue)
{
    float* device_input  = nullptr;
    float* device_output = nullptr;

    ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device_input), 4 * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void**>(&device_output), 4 * sizeof(float)), hipSuccess);

    const float invalid_value = -7.0f;
    const float input[4]      = {1.0f, 2.0f, 3.0f, 4.0f};

    ASSERT_EQ(hipMemcpy(device_input, input, sizeof(input), hipMemcpyHostToDevice), hipSuccess);

    hipLaunchKernelGGL(kernel_read_invalid_customized_vector<float>,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       device_input,
                       device_output,
                       invalid_value);

    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    float output[4] = {};
    ASSERT_EQ(hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_FLOAT_EQ(output[0], invalid_value);
    EXPECT_FLOAT_EQ(output[1], invalid_value);
    EXPECT_FLOAT_EQ(output[2], invalid_value);
    EXPECT_FLOAT_EQ(output[3], invalid_value);

    EXPECT_EQ(hipFree(device_input), hipSuccess);
    EXPECT_EQ(hipFree(device_output), hipSuccess);
}

} // namespace ck_tile
