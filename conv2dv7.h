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

#define SET_FRAGMENT_A(a_len) wmma::fragment<wmma::matrix_a, M, 16, K, half, wmma::row_major> a_frag[a_len];

#define SET_FRAGMENT_B(b_len) wmma::fragment<wmma::matrix_b, K, 16, N, half, wmma::col_major> b_frag[b_len];

#define SET_FRAGMENT_C(a_len,b_len) wmma::fragment<wmma::accumulator,M,16,N, float> c_frag[a_len][b_len];

#define SET_FRAGMENT_CF16(a_len,b_len) wmma::fragment<wmma::accumulator,M,16,N, half> c_frag[a_len][b_len];

#define SET_FRAGMENT_DF16(a_len,b_len) wmma::fragment<wmma::accumulator,M,16,N, half> d_frag[a_len][b_len];

#define LOADFRAG_A(sh,index0,sh_ele_num) wmma::load_matrix_sync(a_frag[index0], &sh,sh_ele_num);

#define LOADFRAG_B(sh,index0,sh_ele_num) wmma::load_matrix_sync(b_frag[index0], &sh,sh_ele_num);

#define FILLZERO_CF16(index0,index1) wmma::fill_fragment(c_frag[index0][index1],static_cast<half>(0.0));

#define FILLDATA_CF16(index0,index1) c_frag[index0][index1].x[0]=c_frag[index0][index1].x[0]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[1]=c_frag[index0][index1].x[1]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[2]=c_frag[index0][index1].x[2]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[3]=c_frag[index0][index1].x[3]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[4]=c_frag[index0][index1].x[4]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[5]=c_frag[index0][index1].x[5]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[6]=c_frag[index0][index1].x[6]+static_cast<half>(0.);\
                                     c_frag[index0][index1].x[7]=c_frag[index0][index1].x[7]+static_cast<half>(0.);

#define FILLDATA_CF16b(index0,index1) wmma::mma_sync(d_frag[index0][index1], a_frag[index0], b_frag[index1], c_frag[index0][index1]);

#define WMMA_SYNC(index0,index1) wmma::mma_sync(c_frag[index0][index1], a_frag[index0], b_frag[index1], c_frag[index0][index1]);

#define STOREFRAG_C_F16(sh,index0,index1,row_ele_num) wmma::store_matrix_sync(&sh, c_frag[index0][index1],row_ele_num, wmma::mem_row_major);

#define STOREFRAG_C_F16B(sh,index0,index1,row_ele_num) wmma::store_matrix_sync(sh, d_frag[index0][index1],row_ele_num, wmma::mem_row_major);

__device__ void load_temp(half *dst,half *src,int len_,int offset){
    for(int id=0;id<len_;id++){
        dst[id]=src[id+offset];
    }
}

__device__ void load_warp_0(half *D,half *shmem,int H,int W,int dh,int dw);

__device__ void load_warp_1_2(half *D,half *shmem,int H,int W,int dh,int dw,int warpid);

__device__ void load_main_D(half *D,half *shmem,int H,int W);

__device__ void load_warp_3(half *D,half *shmem,int H,int W,int dh,int dw);

__device__ void load_warp_4(half *D,half *shmem,int H,int W,int dh,int dw);

__device__ void load_warp_5_6(half *D,half *shmem,int H,int W,int dh,int dw,int warpid);

__device__ void load_warp_7(half *D,half *shmem,int H,int W,int dh,int dw);

__device__ void load_matrix_D(half * D,half * shmem,int blk_id,int H,int W,int C,int dc){
    //calculate the initial position of data to be loaded 
    //calculate the initial position of data to be loaded 
    int posi =10240*blk_id+blockIdx.x*128;
    int dN = posi/(H*W);
    posi=posi%(H*W)/128;
    int dh = posi/(W/16)*8;
    int dw = posi%(W/16)*16;
    //move D to current warp
    D+=(dN*H*W*C+dc*H*W+dh*W+dw);
   
    //sign id of the warp
    int warpid = threadIdx.x/32;
    
    //load the current warp based on the position of the warp
    load_main_D(D,shmem,H,W);

    if(warpid==0){
        load_warp_0(D,shmem,H,W,dh,dw);
    } else if(warpid==1||warpid==2){
        load_warp_1_2(D,shmem,H,W,dh,dw,warpid);
    } else if(warpid==3){
        load_warp_3(D,shmem,H,W,dh,dw);
    } else if(warpid==4){
        load_warp_4(D,shmem, H,W,dh,dw);
    } else if(warpid==5||warpid==6){
        load_warp_5_6(D,shmem,H,W,dh,dw,warpid);
    } else {
        load_warp_7(D,shmem,H, W, dh, dw);
    }

}

__device__ void move_data_int4(half *src,half *dst,int strd){
    //*((int4*)dst)=*((int4*)src);

    for(int id=0;id<8;id++){
        dst[id]=src[id*strd];
    }
}

__device__ void load_main_D(half *D,half *shmem,int H,int W){
    //calculate offset for shmem and D
    int row_id = threadIdx.x/32;
    int col_id = (threadIdx.x/2)%16;
    int lan_id = threadIdx.x%2;
    int offset_shmem=304+row_id*288+col_id*16+lan_id*8;
    int offset_D=row_id*W+col_id+lan_id*8*H*W;
    
    //find the position of current thread on shmem
    half* dst=shmem+offset_shmem;

    //move D to the continuous position
    half* src=D+offset_D;
    
    int strd=H*W;
    //move the data
    move_data_int4(src,dst,strd);

}

__device__ void load_warp_0(half *D,half *shmem,int H,int W,int dh,int dw){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the op edge
    half *dst=shmem+16;
    half *src;
    if(dh==0){//fill zero, now at edge
        dst[threadid*2]=static_cast<half>(0.);
        dst[threadid*2+1]=static_cast<half>(0.);
    } else {
        src=D-W;
        dst[threadid*2]=src[colid+rowid*H*W];
        dst[threadid*2+1]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the left edge
    dst=shmem+288;
    if(dw==0){//fill zero, now at edge
        dst[colid*288+rowid]=static_cast<half>(0.);
        dst[colid*288+rowid+1]=static_cast<half>(0.);
    } else {
        src=D-1;
        dst[colid*288+rowid]=src[colid*W+rowid*H*W];
        dst[colid*288+rowid+1]=src[colid*W+(rowid+1)*H*W];
    }
}

__device__ void load_warp_1_2(half *D,half *shmem,int H,int W,int dh,int dw,int warpid){
    int threadid=threadIdx.x%32;

    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the top edge
    half *dst=shmem+16+warpid*64;
    half *src;
    if(dh==0){//fill zero at boader
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[colid*16+rowid+1]=static_cast<half>(0.);
    } else {
        src=D-W+warpid*4;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[colid*16+rowid+1]=src[colid+(rowid+1)*H*W];
    }
    //fill the top left and top right point
    if(warpid==1){
        dst=shmem;
        if(dw==0||dh==0){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D-1-W;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    } else {
        dst=shmem+272;
        if(dw==W-16||dh==0){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D-W+16;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    }
}

__device__ void load_warp_3(half *D,half *shmem,int H,int W,int dh,int dw){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the top edge
    half *dst=shmem+208;
    half *src;
    if(dh==0){//fill zero, now at edge
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[1+colid*16+rowid]=static_cast<half>(0.);
    } else{
        src=D-W+12;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[1+colid*16+rowid]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the right edge
    dst=shmem+560;
    if(dw==W-16){//fill zero, now at edge
        dst[colid*288+rowid]=static_cast<half>(0.);
        dst[1+colid*288+rowid]=static_cast<half>(0.);
    } else {
        src=D+16;
        dst[colid*288+rowid]=src[colid*W+rowid*H*W];
        dst[1+colid*288+rowid]=src[colid*W+(rowid+1)*H*W];
    }
}

__device__ void load_warp_4(half *D,half *shmem,int H,int W,int dh,int dw){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the bot edge
    half *dst=shmem+2608;
    half *src;
    if(dh==H-8){//fill zero, now at edge
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[1+colid*16+rowid]=static_cast<half>(0.);
    } else{
        src=D+8*W;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[1+colid*16+rowid]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the left edge
    dst=shmem+1440;
    if(dw==0){//fill zero, now at edge
        dst[colid*288+rowid]=static_cast<half>(0.);
        dst[colid*288+rowid+1]=static_cast<half>(0.);
    } else {
        src=D-1+4*W;
        dst[colid*288+rowid]=src[colid*W+rowid*H*W];
        dst[colid*288+rowid+1]=src[colid*W+(rowid+1)*H*W];
    }
}

__device__ void load_warp_5_6(half *D,half *shmem,int H,int W,int dh,int dw,int warpid){
    int threadid=threadIdx.x%32;

    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the top edge
    half *dst=shmem+warpid*64+2352;
    half *src;
    if(dh==H-8){//fill zero at boader
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[1+colid*16+rowid]=static_cast<half>(0.);
    } else{
        src=D+8*W+4*warpid-16;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[1+colid*16+rowid]=src[colid+(rowid+1)*H*W];
    }
    //fill the top left and top right point
    if(warpid==5){
        dst=shmem+2592;
        if(dw==0||dh==H-8){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D-1+8*W;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    } else {
        dst=shmem+2864;
        if(dw==W-16||dh==H-8){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D+8*W+16;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    }
}

__device__ void load_warp_7(half *D,half *shmem,int H,int W,int dh,int dw){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the bot edge
    half *dst=shmem+2800;
    half *src;
    if(dh==H-8){//fill zero, now at edge
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[1+colid*16+rowid]=static_cast<half>(0.);
    } else{
        src=D+8*W+12;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[1+colid*16+rowid]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the right edge
    dst=shmem+1712;
    if(dw==W-16){//fill zero, now at edge
        dst[colid*288+rowid]=static_cast<half>(0.);
        dst[1+colid*288+rowid]=static_cast<half>(0.);
    } else {
        src=D+4*W+16;
        dst[colid*288+rowid]=src[colid*W+rowid*H*W];
        dst[1+colid*288+rowid]=src[colid*W+(rowid+1)*H*W];
    }
}

__device__ void load_matrix_F(half *F,half *shmem,int offset_F,int blk_id,int KK,int C,int H,int W,int dc,int dRS){
    //move shmem to the loading position of current thread
    half *dst=shmem+offset_F;

    //find the line of current location
    int posi=10240*blk_id+blockIdx.x*128;
    int dK=posi/(H*W)+threadIdx.x/2;
   
    //set the F pointer to loading location
    half *src=F+dK*9*C+dRS*C+dc;
    //fill the shared memory
    if(dK>=KK){//out of range for K
        for(int id=0;id<8;id++){
            dst[id]=static_cast<half>(0.);
        }
    } else{
        /*
        for(int id=0;id<8;id++){
            dst[id]=src[id*9];
        }*/
        *(int4 *)dst=*(int4*)src;
    }
}

__device__ void im2col(half *shmem,int offset_D,int ker_id){
    //find the position of current thread
    int colid = threadIdx.x/32;
    int rowid = (threadIdx.x/2)%16;
    int depth = threadIdx.x%2;
    half *src=shmem+ker_id/3*288+ker_id%3*16+colid*288+rowid*16+depth*8;
    
    half *dst=shmem+offset_D;

    *(int4*)dst=*(int4*)src;
    /*
    for(int id=0;id<8;id++){
        dst[id]=src[id*180];
    }*/
}

__device__ void store_output(half *O,half *shmem,int blk_id,int P,int Q,int KK){
    //calculate the initial position of data to be loaded 
    int posi =10240*blk_id+blockIdx.x*128;
    int dN = posi/(P*Q);
    posi=posi%(P*Q)/128;
    int dp = posi/(Q/16)*8+threadIdx.x/32;
    int dq = posi%(Q/16)*16+threadIdx.x%32/16*8;
    
    //set para for current thread    
    int dK=threadIdx.x%16*8;
    half *dst=O+dN*KK*P*Q+dK*P*Q+dp*Q+dq;
    half *src=shmem+dK*128+dp%8*16+dq%16;
   
    for(int id=0;id<8;id++){
        *(int4 *)(dst+id*Q*P)=*(int4*)(src+id*128);
    }
}

__device__ void temp_store(half *src,half *dst,int len,int offset){
    for(int id=0;id<len;id++){
        dst[id]=src[id+offset];
    }
}