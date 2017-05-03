#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void matrix_softmax_kernel(int nrow, int ncol, const float *input_a, float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  output += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input_a[x] - maxval) / sum;
  }
}

__global__ void array_relu_kernel(int64_t length, const float* input1, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = max(0.0f, input1[y]);
}

__global__ void array_relu_gradient_kernel(int64_t length, const float* input1, const float* in_grad, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = input1[y] >= 0.0f ? in_grad[y] : 0.0f;
}

__global__ void array_add_kernel(int64_t length, const float* input1, const float* input2, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = input1[y] + input2[y];
}

__global__ void array_add_by_const_kernel(int64_t length, const float* input1, float value, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = input1[y] + value;
}

__global__ void array_mul_kernel(int64_t length, const float* input1, const float* input2, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = input1[y] * input2[y];
}

__global__ void array_mul_by_const_kernel(int64_t length, const float* input1, float value, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = input1[y] * value;
}

__global__ void array_set_kernel(int64_t length, float value, float *output) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(y >= length) {
    return;
  }
  output[y] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < arr->ndim; i++) {
    length *= arr->shape[i];
  }
  if(length <= 1024) {
    array_set_kernel<<<1, length>>>(length, value, (float*)arr->data);
  } else {
    array_set_kernel<<<length / 1024 + 1, 1024>>>(length, value, (float*)arr->data);
  }
  return 0;
}

__global__ void broadcast_to_kernel(int64_t input_length, const float* input, float *output) {
  output += input_length * blockIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(x < input_length) {
    output[x] = input[x];
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t input_length = 1;
  for(int i = 0; i < input->ndim; i++) {
    input_length *= input->shape[i];
  }
  broadcast_to_kernel<<<dim3(input_length / 1024 + 1, output->shape[0]), 1024>>>(input_length, (const float*)input->data, (float*)output->data);
  return 0;
}


__global__ void reduce_sum_axis_zero_kernel(int64_t output_length, int reduce_size, const float* input, float *output) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  if(x >= output_length) return;
  float value = 0;
  for(int i = 0; i < reduce_size; i++) {
    value += input[i * output_length + x];
    __syncthreads();
  }
  output[x] = value;
}


int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int output_length = 1;
  for(int i = 0; i < output->ndim; i++) {
    output_length *= output->shape[i];
  }
  printf("DLGpuReduceSumAxisZero : %d, %d\n", output_length, input->shape[0]);
  reduce_sum_axis_zero_kernel<<<output_length / 1024 + 1, min(1024, output_length)>>>(output_length, input->shape[0], (float*)input->data, (float*)output->data);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_add_kernel<<<1, length>>>(length, (const float*)matA->data, (const float*)matB->data, (float*)output->data);
  } else {
    array_add_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)matA->data, (const float*)matB->data, (float*)output->data);
  }
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_add_by_const_kernel<<<1, length>>>(length, (const float*)input->data, val, (float*)output->data);
  } else {
    array_add_by_const_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)input->data, val, (float*)output->data);
  }
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_mul_kernel<<<1, length>>>(length, (const float*)matA->data, (const float*)matB->data, (float*)output->data);
  } else {
    array_mul_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)matA->data, (const float*)matB->data, (float*)output->data);
  }
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_mul_by_const_kernel<<<1, length>>>(length, (const float*)input->data, val, (float*)output->data);
  } else {
    array_mul_by_const_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)input->data, val, (float*)output->data);
  }
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  // op(A) * op(B) = C
  // op(B)T * op(A)T = CT

  cublasHandle_t handle;
  float one = 1.0f;
  float zero = 0.0f;
  int m = matC->shape[1];
  int n = matC->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];
  cublasCreate(&handle);
  cublasSgemm(handle,
    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
    m, n, k,
    &one,
    (const float*)matB->data, !transposeB ? m : k,
    (const float*)matA->data, !transposeA ? k : n,
    &zero,
    (float*)matC->data, m
  );
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_relu_kernel<<<1, length>>>(length, (const float*)input->data, (float*)output->data);
  } else {
    array_relu_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)input->data, (float*)output->data);
  }
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  dim3 blocks;
  int64_t length = 1;
  for(int i = 0; i < output->ndim; i++) {
    length *= output->shape[i];
  }
  if(length <= 1024) {
    array_relu_gradient_kernel<<<1, length>>>(length, (const float*)input->data, (const float*)in_grad->data, (float*)output->data);
  } else {
    array_relu_gradient_kernel<<<length / 1024 + 1, 1024>>>(length, (const float*)input->data, (const float*)in_grad->data, (float*)output->data);
  }
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 2);
  int nrow = input->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input->shape[1];
  const float *input_data_a = (const float *)input->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, output_data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
