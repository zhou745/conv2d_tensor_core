#include <cuda_fp16.h>
#include"/home/tusimple/Desktop/tvm_ir_test/conv2dv6.h"
extern "C" __global__ void conv_kernel0( half* __restrict__ placeholder,  half* __restrict__ placeholder1,  half* __restrict__ placeholder2,  half* __restrict__ conv) {
  __shared__ half shmem_D[3072];
  __shared__ half shmem_F[3072];
  __shared__ half shmem_O[16384];
  SET_FRAGMENT_A(2);
  SET_FRAGMENT_B(4);
  SET_FRAGMENT_CF16(2, 4);
  DECLARE_PARA();
  INIT_PARA(1, 256, 256, 1, 1, 256, 256, 3, 3);
  POINTER_D(((half *)placeholder + 0), 64, 256, 256);
  POINTER_F(((half *)placeholder1 + 0), 64, 3, 3);
  for (int blk_id = 0; blk_id < 7; ++blk_id) {
    if (((int)blockIdx.x) < (512 - (blk_id * 80))) {
      LOAD_D(shmem_D[(((((int)threadIdx.x) / 2) * 24) + ((((int)threadIdx.x) % 2) * 8))], 256, 256);
      LOAD_F(shmem_F[(((((int)threadIdx.x) / 2) * 24) + ((((int)threadIdx.x) % 2) * 8))], 64);
      ADVANCE_PARA_P(64, 256, 256, 3, 3);
      for (int col_id = 0; col_id < 2; ++col_id) {
        for (int row_id = 0; row_id < 4; ++row_id) {
          FILLZERO_CF16(col_id, row_id);
        }
      }
      __syncthreads();
      __syncthreads();
      for (int reduce_crs = 0; reduce_crs < 36; ++reduce_crs) {
        for (int col = 0; col < 2; ++col) {
          LOADFRAG_A(shmem_F[(((((int)threadIdx.x) / 64) * 768) + (col * 384))], col, 24);
        }
        for (int row = 0; row < 4; ++row) {
          LOADFRAG_B(shmem_D[((((((int)threadIdx.x) % 64) / 32) * 1536) + (row * 384))], row, 24);
        }
        for (int col1 = 0; col1 < 2; ++col1) {
          for (int row1 = 0; row1 < 4; ++row1) {
            WMMA_SYNC(col1, row1);
          }
        }
        if (reduce_crs < 35) {
          LOAD_D(shmem_D[(((((int)threadIdx.x) / 2) * 24) + ((((int)threadIdx.x) % 2) * 8))], 256, 256);
          LOAD_F(shmem_F[(((((int)threadIdx.x) / 2) * 24) + ((((int)threadIdx.x) % 2) * 8))], 64);
          ADVANCE_PARA_P(64, 256, 256, 3, 3);
          __syncthreads();
          if (((int)blockIdx.x) < 1) {
            if (blk_id < 1) {
              if (reduce_crs < 16) {
                if (14 < reduce_crs) {
                  COPY(((half *)placeholder2 + 0), shmem_D[0], 3072);
                }
              }
            }
          }
        }
      }
      __syncthreads();
      for (int col_id1 = 0; col_id1 < 2; ++col_id1) {
        for (int row_id1 = 0; row_id1 < 4; ++row_id1) {
          STOREFRAG_C_F16(shmem_O[(((((((int)threadIdx.x) / 64) * 4096) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (col_id1 * 2048)) + (row_id1 * 16))], col_id1, row_id1, 128);
        }
      }
      store_O_matrix(((half *)conv + 0), shmem_O, ((((int)blockIdx.x) + (blk_id * 80)) / 512), ((((int)blockIdx.x) + (blk_id * 80)) % 512), (((int)threadIdx.x) / 32), (((int)threadIdx.x) % 32), 1, 64, 256, 256);
    }
  }
}

