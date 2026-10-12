/* SPDX-License-Identifier: BSD-2-Clause
 * Standalone ABI for CUDA libraries, compilable as C or C++ without JAX headers.
 * Inputs and outputs are contiguous in row-major order. Input storage is read
 * only unless explicitly aliased by the binding. Buffer/shape/config descriptors
 * are valid only during the call. Queue all work on context->stream. Do not
 * destroy resources needed by queued work before that stream completes.
 */
#ifndef JAXBIND_GPU_H
#define JAXBIND_GPU_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define JB_GPU_ABI_VERSION 1
/* Numeric values match XLA FFI data types. */
enum jb_gpu_dtype {
  JB_GPU_BOOL = 1, JB_GPU_I8 = 2, JB_GPU_I16 = 3, JB_GPU_I32 = 4,
  JB_GPU_I64 = 5, JB_GPU_U8 = 6, JB_GPU_U16 = 7, JB_GPU_U32 = 8,
  JB_GPU_U64 = 9, JB_GPU_F16 = 10, JB_GPU_F32 = 11, JB_GPU_F64 = 12,
  JB_GPU_C64 = 15, JB_GPU_BF16 = 16, JB_GPU_C128 = 18,
  JB_GPU_F8E5M2 = 19, JB_GPU_F8E4M3FN = 20,
  JB_GPU_F8E4M3B11FNUZ = 23, JB_GPU_F8E5M2FNUZ = 24, JB_GPU_F8E4M3FNUZ = 25,
  JB_GPU_F8E4M3 = 28, JB_GPU_F8E3M4 = 29, JB_GPU_F8E8M0FNU = 33
};

typedef struct jb_gpu_context {
  uint32_t abi_version;
  int32_t device;
  void *stream;
} jb_gpu_context;

typedef struct jb_gpu_buffer {
  void *data;
  size_t nbytes;
  const int64_t *shape;
  size_t ndim;
  uint32_t dtype;
} jb_gpu_buffer;

/* Return zero for success. Otherwise write a NUL-terminated error message.
 * Do not let exceptions cross this boundary. A failing operation must leave
 * queued work safe to complete; CUDA errors that invalidate the context are
 * terminal. The host owns the error buffer and static configuration bytes.
 */
typedef int (*jb_gpu_operation)(
    const jb_gpu_context *context,
    const jb_gpu_buffer *inputs, size_t ninputs,
    jb_gpu_buffer *outputs, size_t noutputs,
    const void *config, size_t config_size,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
