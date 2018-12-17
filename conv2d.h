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

#define FILLZERO_CF16(index0,index1) wmma::fill_fragment(c_frag[index0][index1],static_cast<half>(0.0));

#define LOADFRAG_A(sh,index0,sh_ele_num) wmma::load_matrix_sync(a_frag[index0], &sh,sh_ele_num);

#define LOADFRAG_B(sh,index0,sh_ele_num) wmma::load_matrix_sync(b_frag[index0], &sh,sh_ele_num);

#define STOREFRAG_C_F16(sh,index0,index1,row_ele_num) wmma::store_matrix_sync(&sh, c_frag[index0][index1],row_ele_num, wmma::mem_row_major);

#define WMMA_SYNC(index0,index1) wmma::mma_sync(c_frag[index0][index1], a_frag[index0], b_frag[index1], c_frag[index0][index1]);

#define ONEBYONE(sh,index0,index1,row_ele_num) for(int index=0;index<8;index++){\
                                                   c_frag[index0][index1].x[index];\
                                                }

#define SETVALUE(a) a=static_cast<half>(1.);

#define DECLARE_PARA() int dN;\
                       int dC;\
                       int dH;\
                       int dW;\
                       int dP;\
                       int dQ;\
                       int dR;\
                       int dS;\
                       int dK;\
                       int cordy;\
                       int cordx;\
                       int GAP_D;\
                       int GAP_F;

#define INIT_PARA(NN,PP,QQ,PH,PW,HH,WW,RR,SS) cordy = blockIdx.x*128%(NN*PP*QQ)+threadIdx.x/2;\
                            dN=cordy/(PP*QQ);\
                            dP=(cordy%PP)/QQ;\
                            dQ=cordy%QQ;\
                            cordx=blockIdx.x*128/(NN*PP*QQ)+threadIdx.x/2;\
                            dK=cordx;\
                            dH=dP-PH;\
                            dW=dQ-PW;\
                            if(threadIdx.x%2==1){\
                                dC=8;\
                            } else{\
                                dC=0;\
                            }\
                            dR=0;\
                            dS=0;\
                            GAP_D=HH*WW;\
                            GAP_F=RR*SS;

#define RESET_PARA(NN,PP,QQ,PH,PW,HH,WW,CC,RR,SS,blk_id,dp,fp) cordy=(80+80*blk_id+blockIdx.x)*128%(NN*PP*QQ)+threadIdx.x/2;\
                                               dN=cordy/(PP*QQ);\
                                               dP=(cordy%PP)/QQ;\
                                               dQ=cordy%QQ;\
                                               cordx=(blockIdx.x+80+80*blk_id)*128/(NN*PP*QQ)+threadIdx.x/2;\
                                               dK=cordx;\
                                               dH=dP-PH;\
                                               dW=dQ-PW;\
                                               if(threadIdx.x%2==1){\
                                                    dC=8;\
                                               } else{\
                                                    dC=0;\
                                               }\
                                               dR=0;\
                                               dS=0;\
                                               GAP_D=HH*WW;\
                                               GAP_F=RR*SS;\
                                               Dp=dp+dN*CC*HH*WW+dC*HH*WW+dH*WW+dW;\
                                               Fp=fp+dK*CC*RR*SS+dC*RR*SS;

#define POINTER_D(dp,CC,HH,WW) half * Dp=dp+dN*CC*HH*WW+dC*HH*WW+dH*WW+dW;

#define POINTER_F(fp,CC,RR,SS) half * Fp=fp+dK*CC*RR*SS+dC*RR*SS;

#define LOAD_D(sh_ele,HH,WW) if(dH<0||dH>HH-1||dW<0||dW>WW-1){\
                                        for(int id=0;id<8;id++){\
                                            (&sh_ele)[id]=static_cast<half>(0.);\
                                        }\
                                    }\
                                    else{\
                                        for(int id=0;id<8;id++){\
                                            (&sh_ele)[id]=Dp[id*GAP_D];\
                                        }\
                                    }\
                                    Dp+=GAP_D*7;

#define LOAD_F(sh_ele,KK) if(dK>KK-1){\
                                for(int id=0;id<8;id++){\
                                    (&sh_ele)[id]=static_cast<half>(0.);\
                                }\
                              } else{\
                                for(int id=0;id<8;id++){\
                                    (&sh_ele)[id]=Fp[id*GAP_F];\
                                }\
                              }\
                              Fp+=GAP_F*7;\

#define ADVANCE_PARA_P(CC,HH,WW,RR,SS) if(dC<CC-16&&GAP_D>0){\
                                dC+=16;\
                                Dp+=GAP_D*9;\
                                Fp+=GAP_F*9;\
                            } else if(dC>16&&GAP_D<0){\
                                dC-=16;\
                                Dp+=GAP_D*9;\
                                Fp+=GAP_F*9;\
                            } else if(dC>=CC-16&&GAP_D>0){\
                                dC=64;\
                                GAP_D=-GAP_D;\
                                GAP_F=-GAP_F;\
                                dS+=1;\
                                dW+=1;\
                                if(dS>=SS){\
                                  dS=0;\
                                  dW-=3;\
                                  dR+=1;\
                                  dH+=1;\
                                  Dp+=WW-3;\
                                }\
                                Dp+=1;\
                                Fp+=1;\
                            } else if(dC<=16&&GAP_D<0){\
                                dC=0;\
                                GAP_D=-GAP_D;\
                                GAP_F=-GAP_F;\
                                dS+=1;\
                                dW+=1;\
                                if(dS>=SS){\
                                  dS=0;\
                                  dW-=3;\
                                  dR+=1;\
                                  dH+=1;\
                                  Dp+=WW-3;\
                                }\
                                Dp+=1;\
                                Fp+=1;\
                            }

#define COPY(a,b,len_) for(int id=0;id<len_;id++){\
                           a[id]=(&b)[id];\
                       }
__device__ void load_matrix(half * D,half * shmem,int * id_table, int offset_sh)
{
    //move shared memory to the loading area
    half * sh_warp = shmem;
    sh_warp+=offset_sh;
    for(int id =0;id<8;id++){
        if(id_table[id]==-1){
            *(sh_warp+id)=static_cast<half>(0.);
        } else {
            *(sh_warp+id) = *(D+id_table[id]);
        }
    }
    id_table+=8;
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