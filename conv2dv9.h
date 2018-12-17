#include<cuda.h>
__device__ void load_temp(half *dst,half *src,int len_,int offset){
    for(int id=0;id<len_;id++){
        dst[id]=src[id+offset];
    }
}

__device__ void load_main_D(half *D,half *shmem,int H,int W,int offset_sh,int stride,int row_num,int row_thread);

__device__ void load_matrix_D_128(half * D,half * shmem,int blk_id,int NN,int H,int W,int C,int dc);

__device__ void load_corner(half *D,half *shmem,int H,int W,int dh,int dw,\
                            int h_sh,int v_sh,int h_d,int v_d,int h_c,int w_c);

__device__ void load_corner64(half *D,half *shmem,int H,int W,int dh,int dw,\
                              int h_sh,int v_sh,int h_d,int v_d,int h_c,int w_c,\
                              int cor_sh,int cor_d,int w_cor,int h_cor);

__device__ void load_edge(half *D,half *shmem,int H,int W,int dh,int dw,int warpid,int edge_sh,\
                     int edge_d,int cor1_sh,int cor1_d,int cor2_sh,int cor2_d,\
                     int h_0,int h_1,int h_2,int w_1,int w_2);

__device__ void load_matrix_D_64(half * D,half * shmem,int blk_id,int NN,int H,int W,int C,int dc,int block_num);

__device__ void load_matrix_D(half * D,half * shmem,int blk_id,int NN,int H,int W,int C,int dc,int version){
    //calculate the initial position of data to be loaded 
    if(version==128){
        load_matrix_D_128(D,shmem,blk_id,NN,H,W,C,dc);
    } else {
        load_matrix_D_64(D,shmem,blk_id,NN,H,W,C,dc,80);
    }
    
}


__device__ void load_matrix_D_64(half * D,half * shmem,int blk_id,int N,int H,int W,int C,int dc,int block_num){
    //calculate the initial position of data to be loaded
    int posi =(64*(block_num*blk_id+blockIdx.x))%(N*H*W);
    int dN = posi/(H*W);
    posi=posi%(H*W)/64;
    int dh = posi/(W/8)*8;
    int dw = posi%(W/8)*8;
    // move D to current warp
    D+=(dN*H*W*C+dh*W*C+dw*C+dc);
    //sign id of the warp
    int warpid = threadIdx.x/32;

    //load the current warp based on the position of the warp
    load_main_D(D,shmem,H,W,176,160,8,16);

    //fill the corner case
    if(warpid==0){
        load_corner64(D,shmem,H,W,dh,dw,16,160,-W,-1,0,0,0,-1-W,0,0);
    } else if(warpid==1) {
        load_corner64(D,shmem,H,W,dh,dw,80,304,4-W,8,0,W-8,144,8-W,0,W-8);
    } else if(warpid==2) {
        load_corner64(D,shmem,H,W,dh,dw,1456,800,8*W,4*W-1,H-8,0,1440,8*W-1,H-8,0);
    } else{
        load_corner64(D,shmem,H,W,dh,dw,1520,944,8*W+4,4*W+8,H-8,W-8,1584,8*W+8,H-8,W-8);
    }
}

__device__ void load_matrix_D_128(half * D,half * shmem,int blk_id,int NN,int H,int W,int C,int dc){
    //calculate the initial position of data to be loaded 
    int posi =(10240*blk_id+blockIdx.x*128)%(NN*H*W);
    int dN = posi/(H*W);
    posi=posi%(H*W)/128;
    int dh = posi/(W/16)*8;
    int dw = posi%(W/16)*16;
    //move D to current warp
    D+=(dN*H*W*C+dc*H*W+dh*W+dw);
   
    //sign id of the warp
    int warpid = threadIdx.x/32;
    
    //load the current warp based on the position of the warp
    load_main_D(D,shmem,H,W,304,288,16,32);

    if(warpid==0){
        load_corner(D,shmem,H,W,dh,dw,16,288,-W,-1,0,0);
    } else if(warpid==1||warpid==2){
        //load_warp_1_2(D,shmem,H,W,dh,dw,warpid);
        load_edge(D,shmem,H,W,dh,dw,warpid,16,-W,0,-1-W,272,16-W,0,0,0,0,W-16);
    } else if(warpid==3){
        load_corner(D,shmem,H,W,dh,dw,208,560,12-W,16,0,W-16);
    } else if(warpid==4){
        load_corner(D,shmem,H,W,dh,dw,2608,1440,8*W,4*W-1,H-8,0);
    } else if(warpid==5||warpid==6){
        //load_warp_5_6(D,shmem,H,W,dh,dw,warpid);
        load_edge(D,shmem,H,W,dh,dw,warpid,2352,8*W-16,2592,-1+8*W,2864,16+8*W,H-8,H-8,H-8,0,W-16);
    } else {
        load_corner(D,shmem,H,W,dh,dw,2800,1712,8*W+12,4*W+16,H-8,W-16);
    }
}

__device__ void move_data_int4(half *src,half *dst,int strd){
    //*((int4*)dst)=*((int4*)src);

    for(int id=0;id<8;id++){
        dst[id]=src[id*strd];
    }
}

__device__ void load_main_D(half *D,half *shmem,int W,int C,int offset_sh,int stride,int row_num,int row_thread){
    //calculate offset for shmem and D
    int row_id = threadIdx.x/row_thread;
    int col_id = (threadIdx.x/2)%row_num;
    int lan_id = threadIdx.x%2;
    int offset_shmem=offset_sh+row_id*stride+col_id*16+lan_id*8;
    int offset_D=row_id*W*C+col_id*C+lan_id*8;
    
    //find the position of current thread on shmem
    half* dst=shmem+offset_shmem;

    //move D to the continuous position
    half* src=D+offset_D;
    //move the data
    //move_data_int4(src,dst,strd);
    *(int4*)dst = *(int4 *)src;

}

__device__ void load_corner(half *D,half *shmem,int H,int W,int dh,int dw,\
                            int h_sh,int v_sh,int h_d,int v_d,int h_c,int w_c){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the op edge
    half *dst=shmem+h_sh;
    half *src;
    if(dh==h_c){//fill zero, now at edge
        dst[threadid*2]=static_cast<half>(0.);
        dst[threadid*2+1]=static_cast<half>(0.);
    } else {
        src=D+h_d;
        dst[threadid*2]=src[colid+rowid*H*W];
        dst[threadid*2+1]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the left edge
    dst=shmem+v_sh;
    if(dw==w_c){//fill zero, now at edge
        dst[colid*288+rowid]=static_cast<half>(0.);
        dst[colid*288+rowid+1]=static_cast<half>(0.);
    } else {
        src=D+v_d;
        dst[colid*288+rowid]=src[colid*W+rowid*H*W];
        dst[colid*288+rowid+1]=src[colid*W+(rowid+1)*H*W];
    }
}

__device__ void load_corner64(half *D,half *shmem,int H,int W,int dh,int dw,\
                              int h_sh,int v_sh,int h_d,int v_d,int h_c,int w_c,\
                              int cor_sh,int cor_d,int h_cor,int w_cor){
    int threadid=threadIdx.x%32;
    
    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the op edge
    half *dst=shmem+h_sh;
    half *src;
    if(dh==h_c){//fill zero, now at edge
        dst[threadid*2]=static_cast<half>(0.);
        dst[threadid*2+1]=static_cast<half>(0.);
    } else {
        src=D+h_d;
        dst[threadid*2]=src[colid+rowid*H*W];
        dst[threadid*2+1]=src[colid+(rowid+1)*H*W];
    }
    
    //fill the left edge
    dst=shmem+v_sh;
    if(dw==w_c){//fill zero, now at edge
        dst[colid*160+rowid]=static_cast<half>(0.);
        dst[colid*160+rowid+1]=static_cast<half>(0.);
    } else {
        src=D+v_d;
        dst[colid*160+rowid]=src[colid*W+rowid*H*W];
        dst[colid*160+rowid+1]=src[colid*W+(rowid+1)*H*W];
    }

    //fill the corner point
    dst=shmem+cor_sh;
    if(dw==w_cor||dh==h_cor){//fill zero
        if(threadid<16){
            dst[threadid]=static_cast<half>(0.);
        }
    } else {
        src=D+cor_d;
        if(threadid<16){
            dst[threadid]=src[threadid*H*W];
        }
    }
}

__device__ void load_edge(half *D,half *shmem,int H,int W,int dh,int dw,int warpid,int edge_sh,\
                     int edge_d,int cor1_sh,int cor1_d,int cor2_sh,int cor2_d,\
                     int h_0,int h_1,int h_2,int w_1,int w_2){
    int threadid=threadIdx.x%32;

    //now fill boaders
    int colid=threadid/8;
    int rowid=threadid%8*2;

    //fill the top edge
    half *dst=shmem+edge_sh+warpid*64;
    half *src;
    if(dh==h_0){//fill zero at boader
        dst[colid*16+rowid]=static_cast<half>(0.);
        dst[colid*16+rowid+1]=static_cast<half>(0.);
    } else {
        src=D+edge_d+warpid*4;
        dst[colid*16+rowid]=src[colid+rowid*H*W];
        dst[colid*16+rowid+1]=src[colid+(rowid+1)*H*W];
    }
    //fill the top left and top right point
    if(warpid%4==1){
        dst=shmem+cor1_sh;
        if(dw==w_1||dh==h_1){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D+cor1_d;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    } else {
        dst=shmem+cor2_sh;
        if(dw==w_2||dh==h_2){//fill zero
           if(threadid<16){
                dst[threadid]=static_cast<half>(0.);
           }
        } else {
            src=D+cor2_d;
            if(threadid<16){
                dst[threadid]=src[threadid*H*W];
            }
        }
    }
}

__device__ void load_matrix_F_128(half *F,half *shmem,int offset_F,int blk_id,int KK,int C,\
                             int NN,int H,int W,int dc,int dRS){
    //move shmem to the loading position of current thread
    half *dst=shmem+offset_F;

    //find the line of current location
    int posi=10240*blk_id+blockIdx.x*128;
    int dK=posi/(NN*H*W)+threadIdx.x/2;
   
    //set the F pointer to loading location
    half *src=F+dK*9*C+dRS*C+dc;
    //fill the shared memory
    if(dK>=KK){//out of range for K
        for(int id=0;id<8;id++){
            dst[id]=static_cast<half>(0.);
        }
    } else{
        *(int4 *)dst=*(int4*)src;
    }
}

__device__ void load_matrix_F_64(half *F,half *shmem,int offset_F,int blk_id,int KK,int C,\
                             int NN,int H,int W,int dc,int dRS){
    //move shmem to the loading position of current thread
    half *dst=shmem+offset_F;

    //find the line of current location
    int posi=5120*blk_id+blockIdx.x*64;
    int dK=posi/(NN*H*W)+threadIdx.x/2;
   
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

__device__ void load_matrix_F(half *F,half *shmem,int offset_F,int blk_id,int KK,int C,\
                             int NN,int H,int W,int dc,int dRS,\
                             int version){
    if(version==128) {
        load_matrix_F_128(F,shmem,offset_F,blk_id,KK,C,NN,H,W,dc,dRS);
    } else {
        load_matrix_F_64(F,shmem,offset_F,blk_id,KK,C,NN,H,W,dc,dRS);
    }
}

__device__ void im2col_128(half *shmem,int offset_D,int ker_id){
    //find the position of current thread
    int colid = threadIdx.x/32;
    int rowid = (threadIdx.x/2)%16;
    int depth = threadIdx.x%2;
    half *src=shmem+ker_id/3*288+ker_id%3*16+colid*288+rowid*16+depth*8;
    
    half *dst=shmem+offset_D;

    *(int4*)dst=*(int4*)src;
}

__device__ void im2col_64(half *shmem,int offset_D,int ker_id){
    //find the position of current thread
    int colid = threadIdx.x/16;
    int rowid = (threadIdx.x/2)%8;
    int depth = threadIdx.x%2;
    half *src=shmem+ker_id/3*160+ker_id%3*16+colid*160+rowid*16+depth*8;
    
    half *dst=shmem+offset_D;

    *(int4*)dst=*(int4*)src;
}

__device__ void im2col(half *shmem,int offset_D,int ker_id,int version){
    if(version==128){
        im2col_128(shmem,offset_D,ker_id);
    } else {
        im2col_64(shmem,offset_D,ker_id);
    }
}

__device__ void store_output_128(half *O,half *shmem,int blk_id,int NN,int P,int Q,int KK){
    //calculate the initial position of data to be loaded 
    int posi =(10240*blk_id+blockIdx.x*128)%(NN*P*Q);
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

__device__ void store_output_64(half *O,half *shmem,int blk_id,int NN,int P,int Q,int KK){
    //calculate the initial position of data to be loaded 
    int posi =(5120*blk_id+blockIdx.x*64)%(NN*P*Q);
    int dN = posi/(P*Q);
    posi=posi%(P*Q)/64;
    int dp = posi/(Q/8)*8+threadIdx.x/16;
    int dq = posi%(Q/8)*8;
    
    //set para for current thread    
    int dK=threadIdx.x%16*4;
    half *dst=O+dN*KK*P*Q+dK*P*Q+dp*Q+dq;
    half *src=shmem+dK*64+dp%8*8;
   
    for(int id=0;id<4;id++){
        *(int4 *)(dst+id*Q*P)=*(int4*)(src+id*64);
    }
}

__device__ void store_output(half *O,half *shmem,int blk_id,int NN,int P,int Q,int KK,int version){
    if(version ==128){
        store_output_128(O,shmem,blk_id,NN,P,Q,KK);
    } else {
        store_output_64(O,shmem,blk_id,NN,P,Q,KK);
    }
}