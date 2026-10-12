// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "../src/jaxbind_gpu/include/jaxbind_gpu.h"

int JbLaunchKernel(const jb_gpu_context*, const jb_gpu_buffer*, size_t,
                   jb_gpu_buffer*, size_t, const void*, size_t, char*, size_t);
