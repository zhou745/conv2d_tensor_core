#include <cuda_fp16.h>
#include"/home/tusimple/Desktop/tvm_ir_test/conv2dv8.h"
extern "C" __global__ void conv_kernel0( half* __restrict__ placeholder,  half* __restrict__ placeholder1,  half* __restrict__ conv) {
  __shared__ half shmem[24576];
  SET_FRAGMENT_A(2);
  SET_FRAGMENT_B(4);
  SET_FRAGMENT_CF16(2, 4);
  for (int blk_id = 0; blk_id < 7; ++blk_id) {
    if (((int)blockIdx.x) < (512 - (blk_id * 80))) {
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 0, 128);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 0, 128);
      for (int col_id = 0; col_id < 2; ++col_id) {
        for (int row_id = 0; row_id < 4; ++row_id) {
          FILLZERO_CF16(col_id, row_id);
        }
      }
      __syncthreads();
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      for (int col = 0; col < 2; ++col) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col * 384))], col, 24);
      }
      for (int row = 0; row < 4; ++row) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row * 384))], row, 24);
      }
      __syncthreads();
      for (int col1 = 0; col1 < 2; ++col1) {
        for (int row1 = 0; row1 < 4; ++row1) {
          WMMA_SYNC(col1, row1);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 1, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      __syncthreads();
      for (int col2 = 0; col2 < 2; ++col2) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col2 * 384))], col2, 24);
      }
      for (int row2 = 0; row2 < 4; ++row2) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row2 * 384))], row2, 24);
      }
      __syncthreads();
      for (int col3 = 0; col3 < 2; ++col3) {
        for (int row3 = 0; row3 < 4; ++row3) {
          WMMA_SYNC(col3, row3);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 2, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      __syncthreads();
      for (int col4 = 0; col4 < 2; ++col4) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col4 * 384))], col4, 24);
      }
      for (int row4 = 0; row4 < 4; ++row4) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row4 * 384))], row4, 24);
      }
      __syncthreads();
      for (int col5 = 0; col5 < 2; ++col5) {
        for (int row5 = 0; row5 < 4; ++row5) {
          WMMA_SYNC(col5, row5);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 3, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      __syncthreads();
      for (int col6 = 0; col6 < 2; ++col6) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col6 * 384))], col6, 24);
      }
      for (int row6 = 0; row6 < 4; ++row6) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row6 * 384))], row6, 24);
      }
      __syncthreads();
      for (int col7 = 0; col7 < 2; ++col7) {
        for (int row7 = 0; row7 < 4; ++row7) {
          WMMA_SYNC(col7, row7);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 4, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      __syncthreads();
      for (int col8 = 0; col8 < 2; ++col8) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col8 * 384))], col8, 24);
      }
      for (int row8 = 0; row8 < 4; ++row8) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row8 * 384))], row8, 24);
      }
      __syncthreads();
      for (int col9 = 0; col9 < 2; ++col9) {
        for (int row9 = 0; row9 < 4; ++row9) {
          WMMA_SYNC(col9, row9);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 5, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      __syncthreads();
      for (int col10 = 0; col10 < 2; ++col10) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col10 * 384))], col10, 24);
      }
      for (int row10 = 0; row10 < 4; ++row10) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row10 * 384))], row10, 24);
      }
      __syncthreads();
      for (int col11 = 0; col11 < 2; ++col11) {
        for (int row11 = 0; row11 < 4; ++row11) {
          WMMA_SYNC(col11, row11);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 6, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      __syncthreads();
      for (int col12 = 0; col12 < 2; ++col12) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col12 * 384))], col12, 24);
      }
      for (int row12 = 0; row12 < 4; ++row12) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row12 * 384))], row12, 24);
      }
      __syncthreads();
      for (int col13 = 0; col13 < 2; ++col13) {
        for (int row13 = 0; row13 < 4; ++row13) {
          WMMA_SYNC(col13, row13);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 7, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      __syncthreads();
      for (int col14 = 0; col14 < 2; ++col14) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col14 * 384))], col14, 24);
      }
      for (int row14 = 0; row14 < 4; ++row14) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row14 * 384))], row14, 24);
      }
      __syncthreads();
      for (int col15 = 0; col15 < 2; ++col15) {
        for (int row15 = 0; row15 < 4; ++row15) {
          WMMA_SYNC(col15, row15);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, ((((int)threadIdx.x) % 2) * 8), 8, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      __syncthreads();
      for (int col16 = 0; col16 < 2; ++col16) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col16 * 384))], col16, 24);
      }
      for (int row16 = 0; row16 < 4; ++row16) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row16 * 384))], row16, 24);
      }
      __syncthreads();
      for (int col17 = 0; col17 < 2; ++col17) {
        for (int row17 = 0; row17 < 4; ++row17) {
          WMMA_SYNC(col17, row17);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 16, 128);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      for (int col18 = 0; col18 < 2; ++col18) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col18 * 384))], col18, 24);
      }
      for (int row18 = 0; row18 < 4; ++row18) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row18 * 384))], row18, 24);
      }
      __syncthreads();
      for (int col19 = 0; col19 < 2; ++col19) {
        for (int row19 = 0; row19 < 4; ++row19) {
          WMMA_SYNC(col19, row19);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      __syncthreads();
      for (int col20 = 0; col20 < 2; ++col20) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col20 * 384))], col20, 24);
      }
      for (int row20 = 0; row20 < 4; ++row20) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row20 * 384))], row20, 24);
      }
      __syncthreads();
      for (int col21 = 0; col21 < 2; ++col21) {
        for (int row21 = 0; row21 < 4; ++row21) {
          WMMA_SYNC(col21, row21);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      __syncthreads();
      for (int col22 = 0; col22 < 2; ++col22) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col22 * 384))], col22, 24);
      }
      for (int row22 = 0; row22 < 4; ++row22) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row22 * 384))], row22, 24);
      }
      __syncthreads();
      for (int col23 = 0; col23 < 2; ++col23) {
        for (int row23 = 0; row23 < 4; ++row23) {
          WMMA_SYNC(col23, row23);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      __syncthreads();
      for (int col24 = 0; col24 < 2; ++col24) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col24 * 384))], col24, 24);
      }
      for (int row24 = 0; row24 < 4; ++row24) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row24 * 384))], row24, 24);
      }
      __syncthreads();
      for (int col25 = 0; col25 < 2; ++col25) {
        for (int row25 = 0; row25 < 4; ++row25) {
          WMMA_SYNC(col25, row25);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      __syncthreads();
      for (int col26 = 0; col26 < 2; ++col26) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col26 * 384))], col26, 24);
      }
      for (int row26 = 0; row26 < 4; ++row26) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row26 * 384))], row26, 24);
      }
      __syncthreads();
      for (int col27 = 0; col27 < 2; ++col27) {
        for (int row27 = 0; row27 < 4; ++row27) {
          WMMA_SYNC(col27, row27);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      __syncthreads();
      for (int col28 = 0; col28 < 2; ++col28) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col28 * 384))], col28, 24);
      }
      for (int row28 = 0; row28 < 4; ++row28) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row28 * 384))], row28, 24);
      }
      __syncthreads();
      for (int col29 = 0; col29 < 2; ++col29) {
        for (int row29 = 0; row29 < 4; ++row29) {
          WMMA_SYNC(col29, row29);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      __syncthreads();
      for (int col30 = 0; col30 < 2; ++col30) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col30 * 384))], col30, 24);
      }
      for (int row30 = 0; row30 < 4; ++row30) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row30 * 384))], row30, 24);
      }
      __syncthreads();
      for (int col31 = 0; col31 < 2; ++col31) {
        for (int row31 = 0; row31 < 4; ++row31) {
          WMMA_SYNC(col31, row31);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      __syncthreads();
      for (int col32 = 0; col32 < 2; ++col32) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col32 * 384))], col32, 24);
      }
      for (int row32 = 0; row32 < 4; ++row32) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row32 * 384))], row32, 24);
      }
      __syncthreads();
      for (int col33 = 0; col33 < 2; ++col33) {
        for (int row33 = 0; row33 < 4; ++row33) {
          WMMA_SYNC(col33, row33);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (16 + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      __syncthreads();
      for (int col34 = 0; col34 < 2; ++col34) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col34 * 384))], col34, 24);
      }
      for (int row34 = 0; row34 < 4; ++row34) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row34 * 384))], row34, 24);
      }
      __syncthreads();
      for (int col35 = 0; col35 < 2; ++col35) {
        for (int row35 = 0; row35 < 4; ++row35) {
          WMMA_SYNC(col35, row35);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 32, 128);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      for (int col36 = 0; col36 < 2; ++col36) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col36 * 384))], col36, 24);
      }
      for (int row36 = 0; row36 < 4; ++row36) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row36 * 384))], row36, 24);
      }
      __syncthreads();
      for (int col37 = 0; col37 < 2; ++col37) {
        for (int row37 = 0; row37 < 4; ++row37) {
          WMMA_SYNC(col37, row37);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      __syncthreads();
      for (int col38 = 0; col38 < 2; ++col38) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col38 * 384))], col38, 24);
      }
      for (int row38 = 0; row38 < 4; ++row38) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row38 * 384))], row38, 24);
      }
      __syncthreads();
      for (int col39 = 0; col39 < 2; ++col39) {
        for (int row39 = 0; row39 < 4; ++row39) {
          WMMA_SYNC(col39, row39);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      __syncthreads();
      for (int col40 = 0; col40 < 2; ++col40) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col40 * 384))], col40, 24);
      }
      for (int row40 = 0; row40 < 4; ++row40) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row40 * 384))], row40, 24);
      }
      __syncthreads();
      for (int col41 = 0; col41 < 2; ++col41) {
        for (int row41 = 0; row41 < 4; ++row41) {
          WMMA_SYNC(col41, row41);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      __syncthreads();
      for (int col42 = 0; col42 < 2; ++col42) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col42 * 384))], col42, 24);
      }
      for (int row42 = 0; row42 < 4; ++row42) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row42 * 384))], row42, 24);
      }
      __syncthreads();
      for (int col43 = 0; col43 < 2; ++col43) {
        for (int row43 = 0; row43 < 4; ++row43) {
          WMMA_SYNC(col43, row43);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      __syncthreads();
      for (int col44 = 0; col44 < 2; ++col44) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col44 * 384))], col44, 24);
      }
      for (int row44 = 0; row44 < 4; ++row44) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row44 * 384))], row44, 24);
      }
      __syncthreads();
      for (int col45 = 0; col45 < 2; ++col45) {
        for (int row45 = 0; row45 < 4; ++row45) {
          WMMA_SYNC(col45, row45);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      __syncthreads();
      for (int col46 = 0; col46 < 2; ++col46) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col46 * 384))], col46, 24);
      }
      for (int row46 = 0; row46 < 4; ++row46) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row46 * 384))], row46, 24);
      }
      __syncthreads();
      for (int col47 = 0; col47 < 2; ++col47) {
        for (int row47 = 0; row47 < 4; ++row47) {
          WMMA_SYNC(col47, row47);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      __syncthreads();
      for (int col48 = 0; col48 < 2; ++col48) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col48 * 384))], col48, 24);
      }
      for (int row48 = 0; row48 < 4; ++row48) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row48 * 384))], row48, 24);
      }
      __syncthreads();
      for (int col49 = 0; col49 < 2; ++col49) {
        for (int row49 = 0; row49 < 4; ++row49) {
          WMMA_SYNC(col49, row49);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      __syncthreads();
      for (int col50 = 0; col50 < 2; ++col50) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col50 * 384))], col50, 24);
      }
      for (int row50 = 0; row50 < 4; ++row50) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row50 * 384))], row50, 24);
      }
      __syncthreads();
      for (int col51 = 0; col51 < 2; ++col51) {
        for (int row51 = 0; row51 < 4; ++row51) {
          WMMA_SYNC(col51, row51);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (32 + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      __syncthreads();
      for (int col52 = 0; col52 < 2; ++col52) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col52 * 384))], col52, 24);
      }
      for (int row52 = 0; row52 < 4; ++row52) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row52 * 384))], row52, 24);
      }
      __syncthreads();
      for (int col53 = 0; col53 < 2; ++col53) {
        for (int row53 = 0; row53 < 4; ++row53) {
          WMMA_SYNC(col53, row53);
        }
      }
      __syncthreads();
      load_matrix_D(((half *)placeholder + 0), shmem, blk_id, 1, 256, 256, 64, 48, 128);
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 0, 128);
      __syncthreads();
      for (int col54 = 0; col54 < 2; ++col54) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col54 * 384))], col54, 24);
      }
      for (int row54 = 0; row54 < 4; ++row54) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row54 * 384))], row54, 24);
      }
      __syncthreads();
      for (int col55 = 0; col55 < 2; ++col55) {
        for (int row55 = 0; row55 < 4; ++row55) {
          WMMA_SYNC(col55, row55);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 1, 128);
      __syncthreads();
      for (int col56 = 0; col56 < 2; ++col56) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col56 * 384))], col56, 24);
      }
      for (int row56 = 0; row56 < 4; ++row56) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row56 * 384))], row56, 24);
      }
      __syncthreads();
      for (int col57 = 0; col57 < 2; ++col57) {
        for (int row57 = 0; row57 < 4; ++row57) {
          WMMA_SYNC(col57, row57);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 2, 128);
      __syncthreads();
      for (int col58 = 0; col58 < 2; ++col58) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col58 * 384))], col58, 24);
      }
      for (int row58 = 0; row58 < 4; ++row58) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row58 * 384))], row58, 24);
      }
      __syncthreads();
      for (int col59 = 0; col59 < 2; ++col59) {
        for (int row59 = 0; row59 < 4; ++row59) {
          WMMA_SYNC(col59, row59);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 3, 128);
      __syncthreads();
      for (int col60 = 0; col60 < 2; ++col60) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col60 * 384))], col60, 24);
      }
      for (int row60 = 0; row60 < 4; ++row60) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row60 * 384))], row60, 24);
      }
      __syncthreads();
      for (int col61 = 0; col61 < 2; ++col61) {
        for (int row61 = 0; row61 < 4; ++row61) {
          WMMA_SYNC(col61, row61);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 4, 128);
      __syncthreads();
      for (int col62 = 0; col62 < 2; ++col62) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col62 * 384))], col62, 24);
      }
      for (int row62 = 0; row62 < 4; ++row62) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row62 * 384))], row62, 24);
      }
      __syncthreads();
      for (int col63 = 0; col63 < 2; ++col63) {
        for (int row63 = 0; row63 < 4; ++row63) {
          WMMA_SYNC(col63, row63);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 5, 128);
      __syncthreads();
      for (int col64 = 0; col64 < 2; ++col64) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col64 * 384))], col64, 24);
      }
      for (int row64 = 0; row64 < 4; ++row64) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row64 * 384))], row64, 24);
      }
      __syncthreads();
      for (int col65 = 0; col65 < 2; ++col65) {
        for (int row65 = 0; row65 < 4; ++row65) {
          WMMA_SYNC(col65, row65);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 6, 128);
      __syncthreads();
      for (int col66 = 0; col66 < 2; ++col66) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col66 * 384))], col66, 24);
      }
      for (int row66 = 0; row66 < 4; ++row66) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row66 * 384))], row66, 24);
      }
      __syncthreads();
      for (int col67 = 0; col67 < 2; ++col67) {
        for (int row67 = 0; row67 < 4; ++row67) {
          WMMA_SYNC(col67, row67);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 7, 128);
      __syncthreads();
      for (int col68 = 0; col68 < 2; ++col68) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col68 * 384))], col68, 24);
      }
      for (int row68 = 0; row68 < 4; ++row68) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row68 * 384))], row68, 24);
      }
      __syncthreads();
      for (int col69 = 0; col69 < 2; ++col69) {
        for (int row69 = 0; row69 < 4; ++row69) {
          WMMA_SYNC(col69, row69);
        }
      }
      load_matrix_F(((half *)placeholder1 + 0), shmem, ((5952 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), blk_id, 128, 64, 1, 256, 256, (48 + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      im2col(shmem, ((2880 + ((((int)threadIdx.x) / 2) * 24)) + ((((int)threadIdx.x) % 2) * 8)), 8, 128);
      __syncthreads();
      for (int col70 = 0; col70 < 2; ++col70) {
        LOADFRAG_A(shmem[((5952 + ((((int)threadIdx.x) / 64) * 768)) + (col70 * 384))], col70, 24);
      }
      for (int row70 = 0; row70 < 4; ++row70) {
        LOADFRAG_B(shmem[((2880 + (((((int)threadIdx.x) % 64) / 32) * 1536)) + (row70 * 384))], row70, 24);
      }
      __syncthreads();
      for (int col71 = 0; col71 < 2; ++col71) {
        for (int row71 = 0; row71 < 4; ++row71) {
          WMMA_SYNC(col71, row71);
        }
      }
      __syncthreads();
      for (int col_id1 = 0; col_id1 < 2; ++col_id1) {
        for (int row_id1 = 0; row_id1 < 4; ++row_id1) {
          STOREFRAG_C_F16(shmem[(((((((int)threadIdx.x) / 64) * 4096) + (((((int)threadIdx.x) / 32) % 2) * 64)) + (col_id1 * 2048)) + (row_id1 * 16))], col_id1, row_id1, 128);
        }
      }
      __syncthreads();
      store_output(((half *)conv + 0), shmem, blk_id, 1, 256, 256, 128, 128);
      __syncthreads();
    }
  }
}

