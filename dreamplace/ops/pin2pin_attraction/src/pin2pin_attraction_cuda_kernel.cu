#include <cfloat>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"
#include "pin2pin_attraction/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int pin2pinAttractionCudaLauncher(
  const T *pin_pos_x, const T *pin_pos_y,
  const int *pairs, // Pin pairs (flat array of indices)
  const T *weights, // Weights for each pair, updated to use T
  int num_pairs,
  T *total_distance, // Corrected parameter order and type
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
) {
  int thread_count = 64;
  int block_count = (num_pairs + thread_count - 1) / thread_count; // Correct calculation of blocks
  dim3 block_size(thread_count, 1, 1); // Simplified block dimensions

  if (grad_tensor) {
    pin2pinAttractionCudaBackward<<<block_count, block_size>>>(
        pin_pos_x, pin_pos_y,
        pairs, weights, num_pairs, total_distance,
        grad_tensor, grad_x_tensor, grad_y_tensor
    );
  } else {
    pin2pinAttractionCudaForward<<<block_count, block_size>>>(
        pin_pos_x, pin_pos_y,
        pairs, weights, num_pairs, total_distance,
        grad_tensor, grad_x_tensor, grad_y_tensor
    );
  }
  cudaDeviceSynchronize(); // Ensure completion before return
  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                         \
template int pin2pinAttractionCudaLauncher<T>(              \
    const T *pin_pos_x, const T *pin_pos_y,                 \
    const int *pairs,                                       \
    const T *weights,                                       \
    int num_pairs,                                          \
    T *total_distance,                                      \
    const T *grad_tensor,                                   \
    T *grad_x_tensor, T *grad_y_tensor                      \
)

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE