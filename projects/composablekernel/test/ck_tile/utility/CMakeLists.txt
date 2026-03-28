# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message("-- Adding: test/ck_tile/utility/")

add_gtest_executable(test_fill test_fill.cpp)
add_gtest_executable(test_buffer_view_invalid_element test_buffer_view_invalid_element.cpp)

# Add print tests
add_subdirectory(print)
