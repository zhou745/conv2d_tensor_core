#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#include"/home/tusimple/Desktop/tvm_ir_test/conv2dv9.h"
extern "C" __global__ void conv_kernel0( half* __restrict__ placeholder,  half* __restrict__ placeholder1,  half* __restrict__ conv) {
  __shared__ half shmem[24576];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[2][2];
  for (int blk_id = 0; blk_id < 13; ++blk_id) {
    if (((int)blockIdx.x) < (1024 - (blk_id * 80))) {
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 0, 64);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 0, 64);
      for (int col_id_fi = 0; col_id_fi < 2; ++col_id_fi) {
        for (int row_id_fi = 0; row_id_fi < 2; ++row_id_fi) {
          wmma::fill_fragment(c_frag[col_id_fi][row_id_fi],static_cast<half>(0.000000e+00f));
        }
      }
      __syncthreads();
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      for (int col = 0; col < 2; ++col) {
        wmma::load_matrix_sync(a_frag[col], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col * 384)),24);
      }
      for (int row = 0; row < 2; ++row) {
        wmma::load_matrix_sync(b_frag[row], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row * 384)),24);
      }
      __syncthreads();
      for (int mma_col = 0; mma_col < 2; ++mma_col) {
        for (int mma_row = 0; mma_row < 2; ++mma_row) {
          wmma::mma_sync(c_frag[mma_col][mma_row],a_frag[mma_col],b_frag[mma_row],c_frag[mma_col][mma_row]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 1, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      __syncthreads();
      for (int col1 = 0; col1 < 2; ++col1) {
        wmma::load_matrix_sync(a_frag[col1], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col1 * 384)),24);
      }
      for (int row1 = 0; row1 < 2; ++row1) {
        wmma::load_matrix_sync(b_frag[row1], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row1 * 384)),24);
      }
      __syncthreads();
      for (int mma_col1 = 0; mma_col1 < 2; ++mma_col1) {
        for (int mma_row1 = 0; mma_row1 < 2; ++mma_row1) {
          wmma::mma_sync(c_frag[mma_col1][mma_row1],a_frag[mma_col1],b_frag[mma_row1],c_frag[mma_col1][mma_row1]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 2, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      __syncthreads();
      for (int col2 = 0; col2 < 2; ++col2) {
        wmma::load_matrix_sync(a_frag[col2], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col2 * 384)),24);
      }
      for (int row2 = 0; row2 < 2; ++row2) {
        wmma::load_matrix_sync(b_frag[row2], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row2 * 384)),24);
      }
      __syncthreads();
      for (int mma_col2 = 0; mma_col2 < 2; ++mma_col2) {
        for (int mma_row2 = 0; mma_row2 < 2; ++mma_row2) {
          wmma::mma_sync(c_frag[mma_col2][mma_row2],a_frag[mma_col2],b_frag[mma_row2],c_frag[mma_col2][mma_row2]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 3, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      __syncthreads();
      for (int col3 = 0; col3 < 2; ++col3) {
        wmma::load_matrix_sync(a_frag[col3], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col3 * 384)),24);
      }
      for (int row3 = 0; row3 < 2; ++row3) {
        wmma::load_matrix_sync(b_frag[row3], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row3 * 384)),24);
      }
      __syncthreads();
      for (int mma_col3 = 0; mma_col3 < 2; ++mma_col3) {
        for (int mma_row3 = 0; mma_row3 < 2; ++mma_row3) {
          wmma::mma_sync(c_frag[mma_col3][mma_row3],a_frag[mma_col3],b_frag[mma_row3],c_frag[mma_col3][mma_row3]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 4, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      __syncthreads();
      for (int col4 = 0; col4 < 2; ++col4) {
        wmma::load_matrix_sync(a_frag[col4], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col4 * 384)),24);
      }
      for (int row4 = 0; row4 < 2; ++row4) {
        wmma::load_matrix_sync(b_frag[row4], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row4 * 384)),24);
      }
      __syncthreads();
      for (int mma_col4 = 0; mma_col4 < 2; ++mma_col4) {
        for (int mma_row4 = 0; mma_row4 < 2; ++mma_row4) {
          wmma::mma_sync(c_frag[mma_col4][mma_row4],a_frag[mma_col4],b_frag[mma_row4],c_frag[mma_col4][mma_row4]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 5, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      __syncthreads();
      for (int col5 = 0; col5 < 2; ++col5) {
        wmma::load_matrix_sync(a_frag[col5], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col5 * 384)),24);
      }
      for (int row5 = 0; row5 < 2; ++row5) {
        wmma::load_matrix_sync(b_frag[row5], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row5 * 384)),24);
      }
      __syncthreads();
      for (int mma_col5 = 0; mma_col5 < 2; ++mma_col5) {
        for (int mma_row5 = 0; mma_row5 < 2; ++mma_row5) {
          wmma::mma_sync(c_frag[mma_col5][mma_row5],a_frag[mma_col5],b_frag[mma_row5],c_frag[mma_col5][mma_row5]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 6, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      __syncthreads();
      for (int col6 = 0; col6 < 2; ++col6) {
        wmma::load_matrix_sync(a_frag[col6], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col6 * 384)),24);
      }
      for (int row6 = 0; row6 < 2; ++row6) {
        wmma::load_matrix_sync(b_frag[row6], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row6 * 384)),24);
      }
      __syncthreads();
      for (int mma_col6 = 0; mma_col6 < 2; ++mma_col6) {
        for (int mma_row6 = 0; mma_row6 < 2; ++mma_row6) {
          wmma::mma_sync(c_frag[mma_col6][mma_row6],a_frag[mma_col6],b_frag[mma_row6],c_frag[mma_col6][mma_row6]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 7, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      __syncthreads();
      for (int col7 = 0; col7 < 2; ++col7) {
        wmma::load_matrix_sync(a_frag[col7], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col7 * 384)),24);
      }
      for (int row7 = 0; row7 < 2; ++row7) {
        wmma::load_matrix_sync(b_frag[row7], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row7 * 384)),24);
      }
      __syncthreads();
      for (int mma_col7 = 0; mma_col7 < 2; ++mma_col7) {
        for (int mma_row7 = 0; mma_row7 < 2; ++mma_row7) {
          wmma::mma_sync(c_frag[mma_col7][mma_row7],a_frag[mma_col7],b_frag[mma_row7],c_frag[mma_col7][mma_row7]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 8, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      __syncthreads();
      for (int col8 = 0; col8 < 2; ++col8) {
        wmma::load_matrix_sync(a_frag[col8], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col8 * 384)),24);
      }
      for (int row8 = 0; row8 < 2; ++row8) {
        wmma::load_matrix_sync(b_frag[row8], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row8 * 384)),24);
      }
      __syncthreads();
      for (int mma_col8 = 0; mma_col8 < 2; ++mma_col8) {
        for (int mma_row8 = 0; mma_row8 < 2; ++mma_row8) {
          wmma::mma_sync(c_frag[mma_col8][mma_row8],a_frag[mma_col8],b_frag[mma_row8],c_frag[mma_col8][mma_row8]);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 16, 64);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      for (int col9 = 0; col9 < 2; ++col9) {
        wmma::load_matrix_sync(a_frag[col9], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col9 * 384)),24);
      }
      for (int row9 = 0; row9 < 2; ++row9) {
        wmma::load_matrix_sync(b_frag[row9], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row9 * 384)),24);
      }
      __syncthreads();
      for (int mma_col9 = 0; mma_col9 < 2; ++mma_col9) {
        for (int mma_row9 = 0; mma_row9 < 2; ++mma_row9) {
          wmma::mma_sync(c_frag[mma_col9][mma_row9],a_frag[mma_col9],b_frag[mma_row9],c_frag[mma_col9][mma_row9]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      __syncthreads();
      for (int col10 = 0; col10 < 2; ++col10) {
        wmma::load_matrix_sync(a_frag[col10], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col10 * 384)),24);
      }
      for (int row10 = 0; row10 < 2; ++row10) {
        wmma::load_matrix_sync(b_frag[row10], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row10 * 384)),24);
      }
      __syncthreads();
      for (int mma_col10 = 0; mma_col10 < 2; ++mma_col10) {
        for (int mma_row10 = 0; mma_row10 < 2; ++mma_row10) {
          wmma::mma_sync(c_frag[mma_col10][mma_row10],a_frag[mma_col10],b_frag[mma_row10],c_frag[mma_col10][mma_row10]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      __syncthreads();
      for (int col11 = 0; col11 < 2; ++col11) {
        wmma::load_matrix_sync(a_frag[col11], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col11 * 384)),24);
      }
      for (int row11 = 0; row11 < 2; ++row11) {
        wmma::load_matrix_sync(b_frag[row11], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row11 * 384)),24);
      }
      __syncthreads();
      for (int mma_col11 = 0; mma_col11 < 2; ++mma_col11) {
        for (int mma_row11 = 0; mma_row11 < 2; ++mma_row11) {
          wmma::mma_sync(c_frag[mma_col11][mma_row11],a_frag[mma_col11],b_frag[mma_row11],c_frag[mma_col11][mma_row11]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      __syncthreads();
      for (int col12 = 0; col12 < 2; ++col12) {
        wmma::load_matrix_sync(a_frag[col12], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col12 * 384)),24);
      }
      for (int row12 = 0; row12 < 2; ++row12) {
        wmma::load_matrix_sync(b_frag[row12], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row12 * 384)),24);
      }
      __syncthreads();
      for (int mma_col12 = 0; mma_col12 < 2; ++mma_col12) {
        for (int mma_row12 = 0; mma_row12 < 2; ++mma_row12) {
          wmma::mma_sync(c_frag[mma_col12][mma_row12],a_frag[mma_col12],b_frag[mma_row12],c_frag[mma_col12][mma_row12]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      __syncthreads();
      for (int col13 = 0; col13 < 2; ++col13) {
        wmma::load_matrix_sync(a_frag[col13], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col13 * 384)),24);
      }
      for (int row13 = 0; row13 < 2; ++row13) {
        wmma::load_matrix_sync(b_frag[row13], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row13 * 384)),24);
      }
      __syncthreads();
      for (int mma_col13 = 0; mma_col13 < 2; ++mma_col13) {
        for (int mma_row13 = 0; mma_row13 < 2; ++mma_row13) {
          wmma::mma_sync(c_frag[mma_col13][mma_row13],a_frag[mma_col13],b_frag[mma_row13],c_frag[mma_col13][mma_row13]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      __syncthreads();
      for (int col14 = 0; col14 < 2; ++col14) {
        wmma::load_matrix_sync(a_frag[col14], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col14 * 384)),24);
      }
      for (int row14 = 0; row14 < 2; ++row14) {
        wmma::load_matrix_sync(b_frag[row14], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row14 * 384)),24);
      }
      __syncthreads();
      for (int mma_col14 = 0; mma_col14 < 2; ++mma_col14) {
        for (int mma_row14 = 0; mma_row14 < 2; ++mma_row14) {
          wmma::mma_sync(c_frag[mma_col14][mma_row14],a_frag[mma_col14],b_frag[mma_row14],c_frag[mma_col14][mma_row14]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      __syncthreads();
      for (int col15 = 0; col15 < 2; ++col15) {
        wmma::load_matrix_sync(a_frag[col15], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col15 * 384)),24);
      }
      for (int row15 = 0; row15 < 2; ++row15) {
        wmma::load_matrix_sync(b_frag[row15], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row15 * 384)),24);
      }
      __syncthreads();
      for (int mma_col15 = 0; mma_col15 < 2; ++mma_col15) {
        for (int mma_row15 = 0; mma_row15 < 2; ++mma_row15) {
          wmma::mma_sync(c_frag[mma_col15][mma_row15],a_frag[mma_col15],b_frag[mma_row15],c_frag[mma_col15][mma_row15]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      __syncthreads();
      for (int col16 = 0; col16 < 2; ++col16) {
        wmma::load_matrix_sync(a_frag[col16], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col16 * 384)),24);
      }
      for (int row16 = 0; row16 < 2; ++row16) {
        wmma::load_matrix_sync(b_frag[row16], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row16 * 384)),24);
      }
      __syncthreads();
      for (int mma_col16 = 0; mma_col16 < 2; ++mma_col16) {
        for (int mma_row16 = 0; mma_row16 < 2; ++mma_row16) {
          wmma::mma_sync(c_frag[mma_col16][mma_row16],a_frag[mma_col16],b_frag[mma_row16],c_frag[mma_col16][mma_row16]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      __syncthreads();
      for (int col17 = 0; col17 < 2; ++col17) {
        wmma::load_matrix_sync(a_frag[col17], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col17 * 384)),24);
      }
      for (int row17 = 0; row17 < 2; ++row17) {
        wmma::load_matrix_sync(b_frag[row17], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row17 * 384)),24);
      }
      __syncthreads();
      for (int mma_col17 = 0; mma_col17 < 2; ++mma_col17) {
        for (int mma_row17 = 0; mma_row17 < 2; ++mma_row17) {
          wmma::mma_sync(c_frag[mma_col17][mma_row17],a_frag[mma_col17],b_frag[mma_row17],c_frag[mma_col17][mma_row17]);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 32, 64);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      for (int col18 = 0; col18 < 2; ++col18) {
        wmma::load_matrix_sync(a_frag[col18], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col18 * 384)),24);
      }
      for (int row18 = 0; row18 < 2; ++row18) {
        wmma::load_matrix_sync(b_frag[row18], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row18 * 384)),24);
      }
      __syncthreads();
      for (int mma_col18 = 0; mma_col18 < 2; ++mma_col18) {
        for (int mma_row18 = 0; mma_row18 < 2; ++mma_row18) {
          wmma::mma_sync(c_frag[mma_col18][mma_row18],a_frag[mma_col18],b_frag[mma_row18],c_frag[mma_col18][mma_row18]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      __syncthreads();
      for (int col19 = 0; col19 < 2; ++col19) {
        wmma::load_matrix_sync(a_frag[col19], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col19 * 384)),24);
      }
      for (int row19 = 0; row19 < 2; ++row19) {
        wmma::load_matrix_sync(b_frag[row19], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row19 * 384)),24);
      }
      __syncthreads();
      for (int mma_col19 = 0; mma_col19 < 2; ++mma_col19) {
        for (int mma_row19 = 0; mma_row19 < 2; ++mma_row19) {
          wmma::mma_sync(c_frag[mma_col19][mma_row19],a_frag[mma_col19],b_frag[mma_row19],c_frag[mma_col19][mma_row19]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      __syncthreads();
      for (int col20 = 0; col20 < 2; ++col20) {
        wmma::load_matrix_sync(a_frag[col20], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col20 * 384)),24);
      }
      for (int row20 = 0; row20 < 2; ++row20) {
        wmma::load_matrix_sync(b_frag[row20], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row20 * 384)),24);
      }
      __syncthreads();
      for (int mma_col20 = 0; mma_col20 < 2; ++mma_col20) {
        for (int mma_row20 = 0; mma_row20 < 2; ++mma_row20) {
          wmma::mma_sync(c_frag[mma_col20][mma_row20],a_frag[mma_col20],b_frag[mma_row20],c_frag[mma_col20][mma_row20]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      __syncthreads();
      for (int col21 = 0; col21 < 2; ++col21) {
        wmma::load_matrix_sync(a_frag[col21], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col21 * 384)),24);
      }
      for (int row21 = 0; row21 < 2; ++row21) {
        wmma::load_matrix_sync(b_frag[row21], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row21 * 384)),24);
      }
      __syncthreads();
      for (int mma_col21 = 0; mma_col21 < 2; ++mma_col21) {
        for (int mma_row21 = 0; mma_row21 < 2; ++mma_row21) {
          wmma::mma_sync(c_frag[mma_col21][mma_row21],a_frag[mma_col21],b_frag[mma_row21],c_frag[mma_col21][mma_row21]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      __syncthreads();
      for (int col22 = 0; col22 < 2; ++col22) {
        wmma::load_matrix_sync(a_frag[col22], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col22 * 384)),24);
      }
      for (int row22 = 0; row22 < 2; ++row22) {
        wmma::load_matrix_sync(b_frag[row22], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row22 * 384)),24);
      }
      __syncthreads();
      for (int mma_col22 = 0; mma_col22 < 2; ++mma_col22) {
        for (int mma_row22 = 0; mma_row22 < 2; ++mma_row22) {
          wmma::mma_sync(c_frag[mma_col22][mma_row22],a_frag[mma_col22],b_frag[mma_row22],c_frag[mma_col22][mma_row22]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      __syncthreads();
      for (int col23 = 0; col23 < 2; ++col23) {
        wmma::load_matrix_sync(a_frag[col23], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col23 * 384)),24);
      }
      for (int row23 = 0; row23 < 2; ++row23) {
        wmma::load_matrix_sync(b_frag[row23], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row23 * 384)),24);
      }
      __syncthreads();
      for (int mma_col23 = 0; mma_col23 < 2; ++mma_col23) {
        for (int mma_row23 = 0; mma_row23 < 2; ++mma_row23) {
          wmma::mma_sync(c_frag[mma_col23][mma_row23],a_frag[mma_col23],b_frag[mma_row23],c_frag[mma_col23][mma_row23]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      __syncthreads();
      for (int col24 = 0; col24 < 2; ++col24) {
        wmma::load_matrix_sync(a_frag[col24], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col24 * 384)),24);
      }
      for (int row24 = 0; row24 < 2; ++row24) {
        wmma::load_matrix_sync(b_frag[row24], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row24 * 384)),24);
      }
      __syncthreads();
      for (int mma_col24 = 0; mma_col24 < 2; ++mma_col24) {
        for (int mma_row24 = 0; mma_row24 < 2; ++mma_row24) {
          wmma::mma_sync(c_frag[mma_col24][mma_row24],a_frag[mma_col24],b_frag[mma_row24],c_frag[mma_col24][mma_row24]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      __syncthreads();
      for (int col25 = 0; col25 < 2; ++col25) {
        wmma::load_matrix_sync(a_frag[col25], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col25 * 384)),24);
      }
      for (int row25 = 0; row25 < 2; ++row25) {
        wmma::load_matrix_sync(b_frag[row25], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row25 * 384)),24);
      }
      __syncthreads();
      for (int mma_col25 = 0; mma_col25 < 2; ++mma_col25) {
        for (int mma_row25 = 0; mma_row25 < 2; ++mma_row25) {
          wmma::mma_sync(c_frag[mma_col25][mma_row25],a_frag[mma_col25],b_frag[mma_row25],c_frag[mma_col25][mma_row25]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      __syncthreads();
      for (int col26 = 0; col26 < 2; ++col26) {
        wmma::load_matrix_sync(a_frag[col26], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col26 * 384)),24);
      }
      for (int row26 = 0; row26 < 2; ++row26) {
        wmma::load_matrix_sync(b_frag[row26], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row26 * 384)),24);
      }
      __syncthreads();
      for (int mma_col26 = 0; mma_col26 < 2; ++mma_col26) {
        for (int mma_row26 = 0; mma_row26 < 2; ++mma_row26) {
          wmma::mma_sync(c_frag[mma_col26][mma_row26],a_frag[mma_col26],b_frag[mma_row26],c_frag[mma_col26][mma_row26]);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 48, 64);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 64);
      __syncthreads();
      for (int col27 = 0; col27 < 2; ++col27) {
        wmma::load_matrix_sync(a_frag[col27], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col27 * 384)),24);
      }
      for (int row27 = 0; row27 < 2; ++row27) {
        wmma::load_matrix_sync(b_frag[row27], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row27 * 384)),24);
      }
      __syncthreads();
      for (int mma_col27 = 0; mma_col27 < 2; ++mma_col27) {
        for (int mma_row27 = 0; mma_row27 < 2; ++mma_row27) {
          wmma::mma_sync(c_frag[mma_col27][mma_row27],a_frag[mma_col27],b_frag[mma_row27],c_frag[mma_col27][mma_row27]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 64);
      __syncthreads();
      for (int col28 = 0; col28 < 2; ++col28) {
        wmma::load_matrix_sync(a_frag[col28], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col28 * 384)),24);
      }
      for (int row28 = 0; row28 < 2; ++row28) {
        wmma::load_matrix_sync(b_frag[row28], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row28 * 384)),24);
      }
      __syncthreads();
      for (int mma_col28 = 0; mma_col28 < 2; ++mma_col28) {
        for (int mma_row28 = 0; mma_row28 < 2; ++mma_row28) {
          wmma::mma_sync(c_frag[mma_col28][mma_row28],a_frag[mma_col28],b_frag[mma_row28],c_frag[mma_col28][mma_row28]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 64);
      __syncthreads();
      for (int col29 = 0; col29 < 2; ++col29) {
        wmma::load_matrix_sync(a_frag[col29], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col29 * 384)),24);
      }
      for (int row29 = 0; row29 < 2; ++row29) {
        wmma::load_matrix_sync(b_frag[row29], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row29 * 384)),24);
      }
      __syncthreads();
      for (int mma_col29 = 0; mma_col29 < 2; ++mma_col29) {
        for (int mma_row29 = 0; mma_row29 < 2; ++mma_row29) {
          wmma::mma_sync(c_frag[mma_col29][mma_row29],a_frag[mma_col29],b_frag[mma_row29],c_frag[mma_col29][mma_row29]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 64);
      __syncthreads();
      for (int col30 = 0; col30 < 2; ++col30) {
        wmma::load_matrix_sync(a_frag[col30], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col30 * 384)),24);
      }
      for (int row30 = 0; row30 < 2; ++row30) {
        wmma::load_matrix_sync(b_frag[row30], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row30 * 384)),24);
      }
      __syncthreads();
      for (int mma_col30 = 0; mma_col30 < 2; ++mma_col30) {
        for (int mma_row30 = 0; mma_row30 < 2; ++mma_row30) {
          wmma::mma_sync(c_frag[mma_col30][mma_row30],a_frag[mma_col30],b_frag[mma_row30],c_frag[mma_col30][mma_row30]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 64);
      __syncthreads();
      for (int col31 = 0; col31 < 2; ++col31) {
        wmma::load_matrix_sync(a_frag[col31], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col31 * 384)),24);
      }
      for (int row31 = 0; row31 < 2; ++row31) {
        wmma::load_matrix_sync(b_frag[row31], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row31 * 384)),24);
      }
      __syncthreads();
      for (int mma_col31 = 0; mma_col31 < 2; ++mma_col31) {
        for (int mma_row31 = 0; mma_row31 < 2; ++mma_row31) {
          wmma::mma_sync(c_frag[mma_col31][mma_row31],a_frag[mma_col31],b_frag[mma_row31],c_frag[mma_col31][mma_row31]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 64);
      __syncthreads();
      for (int col32 = 0; col32 < 2; ++col32) {
        wmma::load_matrix_sync(a_frag[col32], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col32 * 384)),24);
      }
      for (int row32 = 0; row32 < 2; ++row32) {
        wmma::load_matrix_sync(b_frag[row32], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row32 * 384)),24);
      }
      __syncthreads();
      for (int mma_col32 = 0; mma_col32 < 2; ++mma_col32) {
        for (int mma_row32 = 0; mma_row32 < 2; ++mma_row32) {
          wmma::mma_sync(c_frag[mma_col32][mma_row32],a_frag[mma_col32],b_frag[mma_row32],c_frag[mma_col32][mma_row32]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 64);
      __syncthreads();
      for (int col33 = 0; col33 < 2; ++col33) {
        wmma::load_matrix_sync(a_frag[col33], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col33 * 384)),24);
      }
      for (int row33 = 0; row33 < 2; ++row33) {
        wmma::load_matrix_sync(b_frag[row33], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row33 * 384)),24);
      }
      __syncthreads();
      for (int mma_col33 = 0; mma_col33 < 2; ++mma_col33) {
        for (int mma_row33 = 0; mma_row33 < 2; ++mma_row33) {
          wmma::mma_sync(c_frag[mma_col33][mma_row33],a_frag[mma_col33],b_frag[mma_row33],c_frag[mma_col33][mma_row33]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 64);
      __syncthreads();
      for (int col34 = 0; col34 < 2; ++col34) {
        wmma::load_matrix_sync(a_frag[col34], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col34 * 384)),24);
      }
      for (int row34 = 0; row34 < 2; ++row34) {
        wmma::load_matrix_sync(b_frag[row34], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row34 * 384)),24);
      }
      __syncthreads();
      for (int mma_col34 = 0; mma_col34 < 2; ++mma_col34) {
        for (int mma_row34 = 0; mma_row34 < 2; ++mma_row34) {
          wmma::mma_sync(c_frag[mma_col34][mma_row34],a_frag[mma_col34],b_frag[mma_row34],c_frag[mma_col34][mma_row34]);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((3136 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 64, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      im2col(shmem, ((1600 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 64);
      __syncthreads();
      for (int col35 = 0; col35 < 2; ++col35) {
        wmma::load_matrix_sync(a_frag[col35], shmem+((3136 + ((((int)threadIdx.x) / 64) * 768)) + (col35 * 384)),24);
      }
      for (int row35 = 0; row35 < 2; ++row35) {
        wmma::load_matrix_sync(b_frag[row35], shmem+((1600 + (((((int)threadIdx.x) % 64) / 32) * 768)) + (row35 * 384)),24);
      }
      __syncthreads();
      for (int mma_col35 = 0; mma_col35 < 2; ++mma_col35) {
        for (int mma_row35 = 0; mma_row35 < 2; ++mma_row35) {
          wmma::mma_sync(c_frag[mma_col35][mma_row35],a_frag[mma_col35],b_frag[mma_row35],c_frag[mma_col35][mma_row35]);
        }
      }
      __syncthreads();
      for (int col_id_st = 0; col_id_st < 2; ++col_id_st) {
        for (int row_id_st = 0; row_id_st < 2; ++row_id_st) {
          wmma::store_matrix_sync(shmem+(((((((int)threadIdx.x) / 64) * 2048) + (((((int)threadIdx.x) / 32) % 2) * 32)) + (col_id_st * 1024)) + (row_id_st * 16)),c_frag[col_id_st][row_id_st],64,wmma::mem_row_major);
        }
      }
      __syncthreads();
      store_output(((half *)conv + 0), shmem, blk_id, 1, 256, 256, 64, 64);
      __syncthreads();
    }
  }
}

