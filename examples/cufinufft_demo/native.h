// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "jaxbind_gpu.h"
#include <cstddef>

int JbCufinufft(const jb_gpu_context*, const jb_gpu_buffer*, size_t,
               jb_gpu_buffer*, size_t, const void*, size_t, char*, size_t);
void JbCufinufftSynchronize(bool clear);
void JbCufinufftConfigure(size_t limit);
size_t JbCufinufftCached();
