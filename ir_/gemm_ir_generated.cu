#include <cuda_fp16.h>
#include"/home/tusimple/Desktop/tvm_ir_test/wmma.h"
extern "C" __global__ void ir_test1_kernel0( float* __restrict__ ir_wmma,  half* __restrict__ placeholder,  half* __restrict__ placeholder1) {
  __shared__ half shmem[24000];
  __FRAGMENT_CH__();
  for (int b_for = 0; b_for < 13; ++b_for) {
    if (((int)blockIdx.x) < (1024 - (b_for * 80))) {
      for (int tile = 0; tile < 4; ++tile) {
        __INT4READ__(shmem[((((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile * 32))], ((float *)ir_wmma + ((((((((((int)threadIdx.x) / 64) * 131072) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile * 16))));
        __INT4READ__(shmem[(((((8 + ((((int)threadIdx.x) / 64) * 4224)) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile * 32))], ((float *)ir_wmma + (((((((4 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile * 16))));
      }
      __syncthreads();
      for (int tile1 = 0; tile1 < 4; ++tile1) {
        __LOADFRAG_C__(shmem[((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (tile1 * 32))], 0, tile1, 132);
      }
      __syncthreads();
      for (int tile2 = 0; tile2 < 4; ++tile2) {
        __INT4READ__(shmem[((((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile2 * 32))], ((float *)ir_wmma + (((((((65536 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile2 * 16))));
        __INT4READ__(shmem[(((((8 + ((((int)threadIdx.x) / 64) * 4224)) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile2 * 32))], ((float *)ir_wmma + (((((((65540 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile2 * 16))));
      }
      __syncthreads();
      for (int tile3 = 0; tile3 < 4; ++tile3) {
        __LOADFRAG_C__(shmem[((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (tile3 * 32))], 1, tile3, 132);
      }
      __syncthreads();
      for (int reduce_i = 0; reduce_i < 64; ++reduce_i) {
        __INT4READ__(shmem[(((((((int)threadIdx.x) / 64) * 2304) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 1152)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder + (((((((((int)threadIdx.x) / 64) * 131072) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 65536)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((9216 + ((((int)threadIdx.x) / 64) * 1152)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 4608)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder1 + (((((((((int)threadIdx.x) / 64) * 65536) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 262144)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((16 + ((((int)threadIdx.x) / 64) * 2304)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 1152)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder + ((((((16 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 65536)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((9232 + ((((int)threadIdx.x) / 64) * 1152)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 4608)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder1 + ((((((16 + ((((int)threadIdx.x) / 64) * 65536)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 262144)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((32 + ((((int)threadIdx.x) / 64) * 2304)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 1152)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder + ((((((32 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 65536)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((9248 + ((((int)threadIdx.x) / 64) * 1152)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 4608)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder1 + ((((((32 + ((((int)threadIdx.x) / 64) * 65536)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 262144)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((48 + ((((int)threadIdx.x) / 64) * 2304)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 1152)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder + ((((((48 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 65536)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (reduce_i * 64))));
        __INT4READ__(shmem[((((9264 + ((((int)threadIdx.x) / 64) * 1152)) + (((((int)threadIdx.x) % 32) / 2) * 72)) + (((((int)threadIdx.x) / 32) % 2) * 4608)) + (((((int)threadIdx.x) % 32) % 2) * 8))], ((half *)placeholder1 + ((((((48 + ((((int)threadIdx.x) / 64) * 65536)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 262144)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 524288)) + (reduce_i * 64))));
        __syncthreads();
        __LOADFRAG_A_CH__(shmem[((((int)threadIdx.x) / 64) * 2304)], 0, 0, 72);
        __LOADFRAG_A_CH__(shmem[(1152 + ((((int)threadIdx.x) / 64) * 2304))], 1, 0, 72);
        __LOADFRAG_B_CH__(shmem[(9216 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 0, 0, 72);
        __LOADFRAG_B_CH__(shmem[(10368 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 1, 0, 72);
        __LOADFRAG_B_CH__(shmem[(11520 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 2, 0, 72);
        __LOADFRAG_B_CH__(shmem[(12672 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 3, 0, 72);
        __LOADFRAG_A_CH__(shmem[(16 + ((((int)threadIdx.x) / 64) * 2304))], 0, 1, 72);
        __LOADFRAG_A_CH__(shmem[(1168 + ((((int)threadIdx.x) / 64) * 2304))], 1, 1, 72);
        __LOADFRAG_B_CH__(shmem[(9232 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 0, 1, 72);
        __LOADFRAG_B_CH__(shmem[(10384 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 1, 1, 72);
        __LOADFRAG_B_CH__(shmem[(11536 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 2, 1, 72);
        __LOADFRAG_B_CH__(shmem[(12688 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 3, 1, 72);
        __LOADFRAG_A_CH__(shmem[(32 + ((((int)threadIdx.x) / 64) * 2304))], 0, 2, 72);
        __LOADFRAG_A_CH__(shmem[(1184 + ((((int)threadIdx.x) / 64) * 2304))], 1, 2, 72);
        __LOADFRAG_B_CH__(shmem[(9248 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 0, 2, 72);
        __LOADFRAG_B_CH__(shmem[(10400 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 1, 2, 72);
        __LOADFRAG_B_CH__(shmem[(11552 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 2, 2, 72);
        __LOADFRAG_B_CH__(shmem[(12704 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 3, 2, 72);
        __LOADFRAG_A_CH__(shmem[(48 + ((((int)threadIdx.x) / 64) * 2304))], 0, 3, 72);
        __LOADFRAG_A_CH__(shmem[(1200 + ((((int)threadIdx.x) / 64) * 2304))], 1, 3, 72);
        __LOADFRAG_B_CH__(shmem[(9264 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 0, 3, 72);
        __LOADFRAG_B_CH__(shmem[(10416 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 1, 3, 72);
        __LOADFRAG_B_CH__(shmem[(11568 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 2, 3, 72);
        __LOADFRAG_B_CH__(shmem[(12720 + (((((int)threadIdx.x) / 32) % 2) * 4608))], 3, 3, 72);
        __WMMA_SYNC_CH__(0, 0, 0);
        __WMMA_SYNC_CH__(0, 1, 0);
        __WMMA_SYNC_CH__(0, 2, 0);
        __WMMA_SYNC_CH__(0, 3, 0);
        __WMMA_SYNC_CH__(1, 0, 0);
        __WMMA_SYNC_CH__(1, 1, 0);
        __WMMA_SYNC_CH__(1, 2, 0);
        __WMMA_SYNC_CH__(1, 3, 0);
        __WMMA_SYNC_CH__(0, 0, 1);
        __WMMA_SYNC_CH__(0, 1, 1);
        __WMMA_SYNC_CH__(0, 2, 1);
        __WMMA_SYNC_CH__(0, 3, 1);
        __WMMA_SYNC_CH__(1, 0, 1);
        __WMMA_SYNC_CH__(1, 1, 1);
        __WMMA_SYNC_CH__(1, 2, 1);
        __WMMA_SYNC_CH__(1, 3, 1);
        __WMMA_SYNC_CH__(0, 0, 2);
        __WMMA_SYNC_CH__(0, 1, 2);
        __WMMA_SYNC_CH__(0, 2, 2);
        __WMMA_SYNC_CH__(0, 3, 2);
        __WMMA_SYNC_CH__(1, 0, 2);
        __WMMA_SYNC_CH__(1, 1, 2);
        __WMMA_SYNC_CH__(1, 2, 2);
        __WMMA_SYNC_CH__(1, 3, 2);
        __WMMA_SYNC_CH__(0, 0, 3);
        __WMMA_SYNC_CH__(0, 1, 3);
        __WMMA_SYNC_CH__(0, 2, 3);
        __WMMA_SYNC_CH__(0, 3, 3);
        __WMMA_SYNC_CH__(1, 0, 3);
        __WMMA_SYNC_CH__(1, 1, 3);
        __WMMA_SYNC_CH__(1, 2, 3);
        __WMMA_SYNC_CH__(1, 3, 3);
        __syncthreads();
      }
      for (int tile4 = 0; tile4 < 4; ++tile4) {
        __STOREFRAG_C__(shmem[((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (tile4 * 32))], 0, tile4, 132);
      }
      for (int tile5 = 0; tile5 < 4; ++tile5) {
        __INT4WRITE__(shmem[((((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile5 * 32))], ((float *)ir_wmma + ((((((((((int)threadIdx.x) / 64) * 131072) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile5 * 16))));
        __INT4WRITE__(shmem[(((((8 + ((((int)threadIdx.x) / 64) * 4224)) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile5 * 32))], ((float *)ir_wmma + (((((((4 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile5 * 16))));
      }
      for (int tile6 = 0; tile6 < 4; ++tile6) {
        __STOREFRAG_C__(shmem[((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (tile6 * 32))], 1, tile6, 132);
      }
      for (int tile7 = 0; tile7 < 4; ++tile7) {
        __INT4WRITE__(shmem[((((((((int)threadIdx.x) / 64) * 4224) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile7 * 32))], ((float *)ir_wmma + (((((((65536 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile7 * 16))));
        __INT4WRITE__(shmem[(((((8 + ((((int)threadIdx.x) / 64) * 4224)) + (((((int)threadIdx.x) % 32) / 2) * 264)) + (((((int)threadIdx.x) / 32) % 2) * 128)) + (((((int)threadIdx.x) % 32) % 2) * 16)) + (tile7 * 32))], ((float *)ir_wmma + (((((((65540 + ((((int)threadIdx.x) / 64) * 131072)) + (((((int)threadIdx.x) % 32) / 2) * 4096)) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (((((int)threadIdx.x) % 32) % 2) * 8)) + (((((int)blockIdx.x) + (b_for * 80)) / 32) * 524288)) + (((((int)blockIdx.x) + (b_for * 16)) % 32) * 128)) + (tile7 * 16))));
      }
      __syncthreads();
    }
  }
}

