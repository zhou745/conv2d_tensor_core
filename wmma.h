#include <iostream>
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;
#define M 16
#define N 16
#define K 16
#define WARP_COL_TILE 2
#define WARP_ROW_TILE 4
#define BLOCK_COL_WARP 4
#define BLOCK_ROW_WARP 2
#define ROW_ELE 4096
#define SHIEFT 8
#define CHUNK 4

#define __INIT_TILE_WARP__()

#define __FRAGMENT__()  wmma::fragment<wmma::matrix_a, M, 16, K, half, wmma::row_major> a_frag[WARP_COL_TILE];\
                        wmma::fragment<wmma::matrix_b, K, 16, N, half, wmma::col_major> b_frag[WARP_ROW_TILE];\
                        wmma::fragment<wmma::accumulator,M,16,N, float> c_frag[WARP_COL_TILE][WARP_ROW_TILE];

#define __FRAGMENT_F16__()  wmma::fragment<wmma::matrix_a, M, 16, K, half, wmma::row_major> a_frag[WARP_COL_TILE];\
                            wmma::fragment<wmma::matrix_b, K, 16, N, half, wmma::col_major> b_frag[WARP_ROW_TILE];\
                            wmma::fragment<wmma::accumulator,M,16,N, half> c_frag[WARP_COL_TILE][WARP_ROW_TILE];

#define __FRAGMENT_CH__()  wmma::fragment<wmma::matrix_a, M, 16, K, half, wmma::row_major> a_frag[CHUNK][WARP_COL_TILE];\
                           wmma::fragment<wmma::matrix_b, K, 16, N, half, wmma::col_major> b_frag[CHUNK][WARP_ROW_TILE];\
                           wmma::fragment<wmma::accumulator,M,16,N, float> c_frag[WARP_COL_TILE][WARP_ROW_TILE];

#define __INT4READ__(sh,gl) *(int4 *)&sh = *(int4 *)gl;

#define __INT4WRITE__(sh,gl) *(int4 *)gl = *(int4 *)&sh;

#define __LOADFRAG_C__(sh,index0,index1,row_ele_num) wmma::load_matrix_sync(c_frag[index0][index1], (float *)&sh,row_ele_num,wmma::mem_row_major);

#define __LOADFRAG_C_F16__(sh,index0,index1,row_ele_num) wmma::load_matrix_sync(c_frag[index0][index1], &sh,row_ele_num,wmma::mem_row_major);

#define __FILL_C_F16__(index0,index1) wmma::fill_fragment(c_frag[index0][index1],static_cast<half>(0.0));

#define __STOREFRAG_C__(sh,index0,index1,row_ele_num) wmma::store_matrix_sync((float*)&sh, c_frag[index0][index1],row_ele_num, wmma::mem_row_major);

#define __STOREFRAG_C_F16__(sh,index0,index1,row_ele_num) wmma::store_matrix_sync(&sh, c_frag[index0][index1],row_ele_num, wmma::mem_row_major);

#define __LOADFRAG_A__(sh,index0,sh_ele_num) wmma::load_matrix_sync(a_frag[index0], &sh,sh_ele_num);

#define __LOADFRAG_A_CH__(sh,index0,index2,sh_ele_num) wmma::load_matrix_sync(a_frag[index2][index0], &sh,sh_ele_num);


#define __LOADFRAG_B__(sh,index0,sh_ele_num) wmma::load_matrix_sync(b_frag[index0], &sh,sh_ele_num);

#define __LOADFRAG_B_CH__(sh,index0,index2,sh_ele_num) wmma::load_matrix_sync(b_frag[index2][index0], &sh,sh_ele_num);

//#define __WMMA_SYNC__(index0,index1)

#define __WMMA_SYNC__(index0,index1) wmma::mma_sync(c_frag[index0][index1], a_frag[index0], b_frag[index1], c_frag[index0][index1]);
                    
//wmma::mma_sync(c_frag[index0][index1], a_frag[index0], b_frag[index1], c_frag[index0][index1]);
#define __WMMA_SYNC_CH__(index0,index1,index2) wmma::mma_sync(c_frag[index0][index1], a_frag[index2][index0], b_frag[index2][index1], c_frag[index0][index1]);

__device__ void load_O_matrix(half * O,half * shmem,int bx,int by,int warp, int lane,int indexN,int indexK,int indexP,int indexQ)
{
    half * sh_warp_tile = shmem;  // point the pointer to the beginning of shmem
    sh_warp_tile+=warp/BLOCK_ROW_WARP*WARP_COL_TILE*16*BLOCK_ROW_WARP*WARP_ROW_TILE*16+warp%BLOCK_ROW_WARP*16*WARP_ROW_TILE; //point the location to current warp
    sh_warp_tile+=lane*WARP_ROW_TILE*BLOCK_ROW_WARP*16; //move the pointer to the location of current thread
    
    int dx = bx*BLOCK_COL_WARP*WARP_COL_TILE*16+warp/BLOCK_ROW_WARP*WARP_COL_TILE*16+lane;
    int dy = by*BLOCK_ROW_WARP*WARP_ROW_TILE*16+warp%BLOCK_ROW_WARP*16*WARP_ROW_TILE;
    //if this lane has reached bottom?
    bool not_bot = dx<indexK;
    if(not_bot){
        //compute the index
        int dN = dy/indexP/indexQ;
        int dP = (dy/indexQ)%indexP;
        int dQ = dy%indexQ;
        int dK = dx; 
        int total = indexN*indexP*indexQ;

        //move the source pointer to position
        half * source = O;
        source+=dN*indexP*indexQ*indexK+dK*indexP*indexQ+dP*indexQ+dQ;
        int remain = 64;
        while(remain>0){
            if(dy<total){
                if(dQ+8<indexQ&&remain-8>=0){
                   *(int4 *)sh_warp_tile = *(int4 *)source;
                   sh_warp_tile+=8;
                   source+=8;
                   remain-=8;
                   dQ+=8;
                   dy+=8;
                }else if(dQ+8<indexQ&&remain-8<0){
                    for(int idd=0;idd<remain;idd++){
                       *sh_warp_tile = *source;
                       sh_warp_tile+=1;
                       source+=1;
                    }
                }else{
                   for(int idd=0;idd<indexQ-dQ;idd++){
                       *sh_warp_tile = *source;
                       sh_warp_tile+=1;
                       source+=1;
                   }
                   remain-=(indexQ-dQ);
                   dy+=(indexQ-dQ);
                   dQ=0;
                   dP+=1;
                   if(dP==indexP){
                       dP=0;
                       dN+=1;
                   }
                   source = O+dN*indexP*indexQ*indexK+dK*indexP*indexQ+dP*indexQ+dQ;
                }
            } else {
                break;
            }
        }
    }
}

__device__ void store_O_matrix(half * O,half * shmem,int bx,int by,int warp, int lane,int indexN,int indexK,int indexP,int indexQ)
{
    half * sh_warp_tile = shmem;  // point the pointer to the beginning of shmem
    sh_warp_tile+=warp/BLOCK_ROW_WARP*WARP_COL_TILE*16*BLOCK_ROW_WARP*WARP_ROW_TILE*16+warp%BLOCK_ROW_WARP*16*WARP_ROW_TILE; //point the location to current warp
    sh_warp_tile+=lane*WARP_ROW_TILE*BLOCK_ROW_WARP*16; //move the pointer to the location of current thread
    
    int dx = bx*BLOCK_COL_WARP*WARP_COL_TILE*16+warp/BLOCK_ROW_WARP*WARP_COL_TILE*16+lane;
    int dy = by*BLOCK_ROW_WARP*WARP_ROW_TILE*16+warp%BLOCK_ROW_WARP*16*WARP_ROW_TILE;
    //if this lane has reached bottom?
    bool not_bot = dx<indexK;
    if(not_bot){
        //compute the index
        int dN = dy/indexP/indexQ;
        int dP = (dy/indexQ)%indexP;
        int dQ = dy%indexQ;
        int dK = dx; 
        int total = indexN*indexP*indexQ;

        //move the source pointer to position
        half * source = O;
        source+=dN*indexP*indexQ*indexK+dK*indexP*indexQ+dP*indexQ+dQ;
        int remain = 64;
        while(remain>0){
            if(dy<total){
                if(dQ+8<indexQ&&remain-8>=0){
                   *(int4 *)source=*(int4 *)sh_warp_tile;
                   sh_warp_tile+=8;
                   source+=8;
                   remain-=8;
                   dQ+=8;
                   dy+=8;
                }else if(dQ+8<indexQ&&remain-8<0){
                    for(int idd=0;idd<remain;idd++){
                       *source=*sh_warp_tile;
                       sh_warp_tile+=1;
                       source+=1;
                    }
                }else{
                   for(int idd=0;idd<indexQ-dQ;idd++){
                       *source=*sh_warp_tile;
                       sh_warp_tile+=1;
                       source+=1;
                   }
                   remain-=(indexQ-dQ);
                   dy+=(indexQ-dQ);
                   dQ=0;
                   dP+=1;
                   if(dP==indexP){
                       dP=0;
                       dN+=1;
                   }
                   source = O+dN*indexP*indexQ*indexK+dK*indexP*indexQ+dP*indexQ+dQ;
                }
            } else {
                break;
            }
        }
    }
}

__device__ void load_D_matrix(half * D,half * shmem,int col_id,int row_id,int tidx,\
                              int indexN, int indexC,int indexH,int indexW,\
                              int padh,int padw,int strh,int strw,\
                              int indexP,int indexQ,int row_ele)
{
    
    int warpid = tidx/32;
    int lane = tidx%32;
    int col_index = col_id*16*WARP_ROW_TILE*BLOCK_ROW_WARP;
    //move the index to current warp same column
    col_index+=warpid*16;
    //move the index to the specific lane
    col_index+=lane/2;
    //load the data into shared memory
    int lane_shief = 8*(lane%2);
    //compute the position
    int Q_loc = col_index%indexQ;
    int P_loc = (col_index/indexQ)%indexP;
    int N_loc = col_index/(indexP*indexQ);
    //move shared memory to the loading area
    half * sh_warp = shmem;
    sh_warp+=warpid*16*row_ele+(lane/2)*row_ele+lane_shief;
    //calculate the position 
    int W_loc = strw*Q_loc-padw;
    int H_loc = strh*P_loc-padh;
    //calculate the initial position in terms of C R S
    int R_ = 2*padh+1+indexH-strh*indexP;
    int S_ = 2*padw+1+indexW-strw*indexQ;
    int C_loc = (row_id*16+lane_shief)/(R_*S_);
    int R_loc = ((row_id*16+lane_shief)/S_)%R_;
    int S_loc = (row_id*16+lane_shief)%S_;

    int row_ = row_id*16+lane_shief;
    int row_b = R_*S_*indexC;
    //calculate the cordinate
    int dN = N_loc;
    int dc = C_loc;
    int dh = H_loc+R_loc;
    int dw = W_loc+S_loc;
    //
    int offsetN = dN*indexW*indexH*indexC;
    
    for(int id =0;id<8;id++){
        if(dh<0||dh>indexH-1||dw<0||dw>indexW-1||col_index>indexN*indexP*indexQ-1||row_>=row_b){
            *(sh_warp+id)=static_cast<half>(0.);
        } else {
            
            //*(sh_warp+id) = *(D+offsetN+dc*indexH*indexW+dh*indexW+dw);
            *(sh_warp+id) = *(D);
        }
        //advance the cordinate
        /*
        dw+=1;
        row_+=1;

        if(dw==(S_+W_loc)){
            dw=W_loc;
            dh+=1;
            if(dh==(R_+H_loc)){
                dh=H_loc;
                dc+=1;
            }
        }   
        */ 
    }
    
}

__device__ void load_F_matrix(half * F,half * shmem,int offsetF,int col_id,int row_id,int tidx,\
                                                    int indexK,int indexC,int indexR, int indexS,int row_ele)
{   
    
    int warpid = tidx/32;
    int lane = tidx%32;
    //move shared memory to the loading area
    int col_index = col_id*16*WARP_COL_TILE*BLOCK_COL_WARP;
    col_index+=warpid*16;
    col_index+=lane/2;
    int lane_shief = lane%2*8;
    
    half * sh_warp = shmem;
    sh_warp+=warpid*16*row_ele+(lane/2)*row_ele+lane_shief+offsetF;
    int K_loc = col_index;
    int C_loc = (row_id*16+lane_shief)/(indexR*indexS);
    int R_loc = ((row_id*16+lane_shief)/indexS)%indexR;
    int S_loc = (row_id*16+lane_shief)%indexS;
    int row_ = row_id*16+lane_shief;
    int row_b = indexC*indexR*indexS;
    int dk = K_loc;
    int dc = C_loc;
    int dr = R_loc;
    int ds = S_loc;

    int offsetK = dk *indexR*indexS*indexC;
    for(int id =0;id<8;id++){
        if(dk>indexK-1||row_>=row_b){
            *(sh_warp+id) = static_cast<half>(0.);
        } else {
            //*(sh_warp+id) = *(F+offsetK+dc*indexR*indexS+dr*indexS+ds);
           *(sh_warp+id) = *(F);
        }
        //advance the cordinate
        /*
        row_+=1;
        ds+=1;
        if(ds==indexS){
            ds=0;
            dr+=1;
            if(dr==indexR){
                dr=0;
                dc+=1;
            }
        }
        */
    }
    
}
 

__device__ void load_D_matrixb(half * D,half * shmem,int col_id,int row_id,int tidx,\
                              int indexN, int indexC,int indexH,int indexW,\
                              int padh,int padw,int strh,int strw,\
                              int indexP,int indexQ,int row_ele)
{
    int warpid = tidx/32;
    int lane = tidx%32;
    int col_index = col_id*16*WARP_ROW_TILE*BLOCK_ROW_WARP;
    //move the index to current warp same column
    col_index+=warpid*16;
    //move the index to the specific lane
    col_index+=lane/2;
    //load the data into shared memory
    int lane_shief = 8*(lane%2);
    //compute the position
    int Q_loc = col_index%indexQ;
    int P_loc = (col_index/indexQ)%indexP;
    int N_loc = col_index/(indexP*indexQ);
    //move shared memory to the loading area
    half * sh_warp = shmem;
    sh_warp+=warpid*16*row_ele+(lane/2)*row_ele+lane_shief;
    //calculate the position 
    int W_loc = strw*Q_loc-padw;
    int H_loc = strh*P_loc-padh;
    //calculate the initial position in terms of C R S
    int R_ = 2*padh+1+indexH-strh*indexP;
    int S_ = 2*padw+1+indexW-strw*indexQ;
    int C_loc = (row_id*16+lane_shief)/(R_*S_);
    int R_loc = ((row_id*16+lane_shief)/S_)%R_;
    int S_loc = (row_id*16+lane_shief)%S_;

    int row_ = row_id*16+lane_shief;
    int row_b = R_*S_*indexC;
    //calculate the cordinate
    int dN = N_loc;
    int dc = C_loc;
    int dh = H_loc+R_loc;
    int dw = W_loc+S_loc;
    //
    int offsetN = dN*indexW*indexH*indexC;
    
    for(int id =0;id<8;id++){
        if(dh<0||dh>indexH-1||dw<0||dw>indexW-1||col_index>indexN*indexP*indexQ-1||row_>=row_b){
            *(sh_warp+id)=static_cast<half>(0.);
        } else {
            *(sh_warp+id) = *(D+offsetN+dc*indexH*indexW+dh*indexW+dw);
        }
        //advance the cordinate

        dw+=1;
        row_+=1;
  
        if(dw==(S_+W_loc)){
            dw=W_loc;
            dh+=1;
            if(dh==(R_+H_loc)){
                dh=H_loc;
                dc+=1;
            }
        }
        
    }
    
}