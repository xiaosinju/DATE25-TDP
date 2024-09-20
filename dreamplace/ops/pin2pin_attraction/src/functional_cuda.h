#ifndef GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H
#define GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H

#include<iostream>

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void pin2pinAttractionCudaForward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];
        float dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        float dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        float distance = weights[idx] * (dx * dx + dy * dy);
        atomicAdd(total_distance, distance); // Atomic addition to accumulate the total distance
    }
}

template <typename T>
__global__ void pin2pinAttractionCudaBackward(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int pin_id1 = pairs[2 * idx];
        int pin_id2 = pairs[2 * idx + 1];
        float dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
        float dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
        float grad_x = 2 * (*grad_tensor) * weights[idx] * dx; // Assuming grad_input is a scalar
        float grad_y = 2 * (*grad_tensor) * weights[idx] * dy;

        atomicAdd(&grad_x_tensor[pin_id1], grad_x);
        atomicAdd(&grad_y_tensor[pin_id1], grad_y);
        atomicAdd(&grad_x_tensor[pin_id2], -grad_x);
        atomicAdd(&grad_y_tensor[pin_id2], -grad_y);
    }
}


DREAMPLACE_END_NAMESPACE

#endif