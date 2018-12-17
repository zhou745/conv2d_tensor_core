# the input data are stored in the order of N C H W, namely, the element of rows are in consecutive memory location
# the filters are stored in the order of K C R S.

import functools
import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes
import sys

import mxnet as mx


TASK = 'conv_ir'
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target='ptx',arch='sm_70',options=['--maxrregcount', '128','-I /home/tusimple/Desktop/tvm_ir_test'])
    return ptx


def write_code(code, fname):
    with open(fname, 'w') as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("ir_"):
        os.mkdir("ir_")
    write_code(code, "ir_/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("ir_/%s_manual.cu" % TASK).read()
    return code

def convolutionf16(D,F,LOAD_INDEX_D,LOAD_INDEX_F,O):
    ib = tvm.ir_builder.create()
    
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',block_num)
    thread_x=tvm.thread_axis('threadIdx.x')
    ib.scope_attr(thread_x,'thread_extent',thread_num)
    
    bidx = block_x
    tidx = thread_x
    #set shared memory buffer for loading data as column
    shmem_O = ib.allocate("float16", 16384, name="shmem_O",scope = "shared")

    shmem_D = ib.allocate("float16", 3072, name="shmem_D",scope = "shared")

    shmem_F = ib.allocate("float16", 3072, name="shmem_F",scope = "shared")

    #sync thread model
    sync = tvm.call_extern("float32","__syncthreads")

    #declare matrix fragement
    Define_matrix_fragment_a = tvm.call_extern("float32","SET_FRAGMENT_A",warp_col_tile)
    ib.emit(Define_matrix_fragment_a)
    
    Define_matrix_fragment_b = tvm.call_extern("float32","SET_FRAGMENT_B",warp_row_tile)
    ib.emit(Define_matrix_fragment_b)

    Define_matrix_fragment_c = tvm.call_extern("float32","SET_FRAGMENT_CF16",warp_col_tile,warp_row_tile)
    ib.emit(Define_matrix_fragment_c)

    #set the loading index to current location
    #caculate the id of current warp
    warpid = tidx//32
    #caculate the id of current thread inside current warp
    lane = tidx%32
    #number of element in a row for shared memory
    o_row_num = warp_row_tile*block_row_warp*16
    # offset to point the pointer to the start of current warp in shared memory
    Dp = D.access_ptr("r")
    Fp = F.access_ptr("r")
    #offset_sh = bidx*thread_num*index_len+tidx*index_len
    #offset_sh = bidx+tidx+100000
    LDp = LOAD_INDEX_D.access_ptr("r")
    LFp = LOAD_INDEX_F.access_ptr("r")
    
    define_d_index = tvm.call_extern("float32","INDEXPOINTERD",LDp)
    define_f_index = tvm.call_extern("float32","INDEXPOINTERF",LFp)
    ib.emit(define_d_index)
    ib.emit(define_f_index)
    #loading parameter
    row_num = 16+shieft

    warp_offset_o = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*warp_col_tile*16*o_row_num

    offset_warp_row = warpid%block_row_warp*warp_row_tile*16*row_num

    offset_warp_col = warpid/block_row_warp*warp_col_tile*row_num*16
    
    offset_sh_load = warpid*16*row_num+(lane/2)*row_num+8*(lane%2)

    fragement_step = 16*row_num

    #main loop for computing the conv
    with ib.for_range(0,loop_len,name ="blk_id") as blk_id:
        with ib.if_scope(bidx+blk_id*block_num<rD*rF//block_len//block_len):
            #compute the location of current block   
            bx = (bidx+blk_id*block_num)//(rD//block_len)
            by = (bidx+blk_id*block_num)%(rD//block_len)
            #store the result from last computation
            """
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                    store_O_fragment = tvm.call_extern("float32","STOREFRAG_C_F16",shmem_O[warp_offset_o+col_id*16*o_row_num+row_id*16],\
                                                                                             col_id,row_id,o_row_num)
                    #onebyone_store = tvm.call_extern("float32","ONEBYONE",shmem_O[warp_offset_o+col_id*16*o_row_num+row_id*16],col_id,row_id,o_row_num)
                    ib.emit(store_O_fragment)   
       
            Op = O.access_ptr("w")
            store_O_matrix = tvm.call_extern("float32","store_O_matrix",\
                                                     Op,shmem_O,bx,by,warpid,lane,\
                                                     N,K,P,Q)
            ib.emit(store_O_matrix)
            """
            #set pointers                   

            #col_index_d = by*16*warp_row_tile*block_row_warp+warpid*16+lane/2
            #row_index_d = 8*(lane%2)
            #col_index_f = bx*16*warp_col_tile*block_col_warp+warpid*16+lane/2
            #row_index_f = 8*(lane%2)

            #LDp = LOAD_INDEX_D.access_ptr("r",offset = col_index_d*cD+row_index_d)
            #LFp = LOAD_INDEX_F.access_ptr("r",offset = col_index_f*cF+row_index_f)
            
            
            
            #now load F, D
            offset_sh = offset_sh_load
            load_D_matrix = tvm.call_extern("float32","LOAD_MATRIX_D",Dp,shmem_D[offset_sh])
            ib.emit(load_D_matrix)
            offset_sh = offset_sh_load
            load_F_matrix = tvm.call_extern("float32","LOAD_MATRIX_F",Fp,shmem_F[offset_sh])           
            ib.emit(load_F_matrix)
            #load the fragement
            ib.emit(sync)
            
           
            #load the out put matrix fragment
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:              
                    fill_O_zero = tvm.call_extern("float","FILLZERO_CF16",col_id,row_id)
                    ib.emit(fill_O_zero)
            ib.emit(sync)

            with ib.for_range(0,cD//16,name = "reduce_crs") as reduce_crs:
                offset_sh = offset_warp_col
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    #offset_sh+= col*16*row_num
                    load_matrix_frag_F = tvm.call_extern("float32","LOADFRAG_A",shmem_F[offset_sh],col,row_num)
                    ib.emit(load_matrix_frag_F)
                offset_sh = offset_warp_row
                with ib.for_range(0,warp_row_tile,name = "row") as row:
                    #offset_sh+= row*16*row_num
                    load_matrix_frag_D = tvm.call_extern("float32","LOADFRAG_B",shmem_D[offset_sh],row,row_num)
                    ib.emit(load_matrix_frag_D)
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        wmma_compute = tvm.call_extern("float32","WMMA_SYNC",col,row)
                        ib.emit(wmma_compute)
                #
                #ib.emit(sync)
                #load data of the next iteration if it is not the last
                with ib.if_scope(reduce_crs<cD//16-1):
                    #reset pointer location
                    #row_index_d = 16*(reduce_crs+1)+8*(lane%2)
                    #row_index_f = 16*(reduce_crs+1)+8*(lane%2)

                    #LDp = LOAD_INDEX_D.access_ptr("r",offset = col_index_d*cD+row_index_d)
                    #LFp = LOAD_INDEX_F.access_ptr("r",offset = col_index_f*cF+row_index_f)
                    
                    offset_sh = offset_sh_load
                    load_D_matrix = tvm.call_extern("float32","LOAD_MATRIX_D",Dp,shmem_D[offset_sh])
                    ib.emit(load_D_matrix)
                    offset_sh = offset_sh_load
                    load_F_matrix = tvm.call_extern("float32","LOAD_MATRIX_F",Fp,shmem_F[offset_sh])           
                    ib.emit(load_F_matrix)
                    """
                    #load the fragement
                    with ib.for_range(0,warp_col_tile,name = "col") as col:
                        offset_sh = col*16*row_num+warpid/block_row_warp*warp_col_tile*16*row_num
                        load_matrix_frag_F = tvm.call_extern("float32","LOADFRAG_A",shmem_F[offset_sh],col,row_num)
                        ib.emit(load_matrix_frag_F)

                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        offset_sh = row*16*row_num+warpid%block_row_warp*warp_row_tile*16*row_num
                        load_matrix_frag_D = tvm.call_extern("float32","LOADFRAG_B",shmem_D[offset_sh],row,row_num)
                        ib.emit(load_matrix_frag_D)
                    """
                ib.emit(sync)
              
                with ib.if_scope(reduce_crs == cD//16-1):
                    with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                        with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                            store_O_fragment = tvm.call_extern("float32","STOREFRAG_C_F16",shmem_O[warp_offset_o+col_id*16*o_row_num+row_id*16],\
                                                                                             col_id,row_id,o_row_num)
                            ib.emit(store_O_fragment)
              
                    Op = O.access_ptr("w")
                    store_O_matrix = tvm.call_extern("float32","store_O_matrix",\
                                                     Op,shmem_O,bx,by,warpid,lane,\
                                                     N,K,P,Q)
                    ib.emit(store_O_matrix)
                 
            
          
                
    body = ib.get()
    return(body)

@tvm.target.generic_func
def schedule_conv_fp16():
    raise NotImplemented()

@schedule_conv_fp16.register(['cuda'])
def _schedule_conv_fp16():
    s = tvm.create_schedule(O.op)
    return(s)

def single_dev_consist(d,f):
    data = mx.sym.var("data")
    weight = mx.sym.var("conv_weight")
    
    # bias = None, pad = (1, 1), stride = (1, 1)
    conv = Conv(name='conv', data=data, weight=weight, num_filter=K, pad=(ph, pw), kernel=(R, S))
    conv_exe = conv.simple_bind(mx.cpu(), data=(N,C,H,W), conv_weight=(K,C,R,S))

    conv_exe.forward(is_train=False, data=d, conv_weight=f)
    output = conv_exe.outputs[0].asnumpy()
    print('output',output.shape,output.dtype)
    return(output)
def compute_first(d,f):
    out = 0
    for ch in range(C):
        out+=d[0][ch][0][0]*f[0][ch][1][1]+d[0][ch][0][1]*f[0][ch][1][2]+d[0][ch][1][0]*f[0][ch][2][1]+d[0][ch][1][1]*f[0][ch][2][2]
    print(out)

def compute_index(ind):
    for cols in range(N*P*Q):
        dN = cols/(P*Q)
        dP = (cols/P)%Q
        dQ = cols%Q
        dH = dP*u-ph
        dW = dQ*v-pw
        for rows in range(C*R*S):
            dC = rows/(R*S)
            dR = (rows/S)%R
            dS = rows%S
            dH2=dH+dR
            dW2=dW+dS
            if(dH2>=0 and dW2>=0 and dH2<H and dW2<W):
                ind[cols][rows] = dN*C*H*W+dC*H*W+dH2*W+dW2

def compute_indexF(ind):
    for cols in range(K):
        dK = cols
        for rows in range(C*R*S):
            dC = rows/(R*S)
            dR = (rows/S)%R
            dS = rows%S
            ind[cols][rows] = dK*C*R*S+dC*R*S+dR*S+dS

if __name__ == "__main__":
    #bank shieft
    shieft = 8
    #input data shape
    N = 1
    C = 64
    H = 256
    W = 256

    #filter shape
    K = 64
    R = 3
    S = 3

    #conv setting
    u = 1 #vertical stride
    v = 1 #horizontal stride
    ph = 1 #vertical paddling
    pw = 1 #horizontal paddling

    #out put shape
    P = int(np.ceil(float(H-R+1+2*ph)/float(u)))
    Q = int(np.ceil(float(W-S+1+2*pw)/float(v)))

    #schedule parameters
    block_num = 80
    thread_num = 32*8
    
    block_row_warp = 2
    block_col_warp = 4
    warp_row_tile = 4
    warp_col_tile = 2
    block_len = 16*block_col_warp*warp_col_tile
    
    
    cD = ((R*S*C-1)//16+1)*16        
    cF = cD

    rD = ((P*Q*N-1)//block_len+1)*block_len
    rF = ((K-1)//block_len+1)*block_len

    loop_len = (rD*rF//block_len//block_len-1)//block_num+1              # length of the main loop
    index_len = loop_len*cD//16*8+8
    
    print(loop_len)
    print(P,Q,cD)

    OFFIND = np.ones((block_num,thread_num),dtype = np.int32)
    with tvm.target.create('cuda'):
        D = tvm.placeholder((N,C,H,W),dtype = 'float16')
        F = tvm.placeholder((K,C,R,S),dtype = 'float16')
        LOAD_INDEX_D = tvm.placeholder((block_num,thread_num,index_len),dtype = 'int32')
        LOAD_INDEX_F = tvm.placeholder((block_num,thread_num,index_len),dtype = 'int32')
        O = tvm.extern((N,K,P,Q),[D,F,LOAD_INDEX_D,LOAD_INDEX_F],lambda ins,outs:convolutionf16(ins[0],ins[1],ins[2],ins[3],outs[0]),name = "conv",dtype = 'float16')
        s = schedule_conv_fp16()
        
        print(tvm.lower(s,[D,F,LOAD_INDEX_D,LOAD_INDEX_F,O],name ='convf16',simple_mode = True))
        f = tvm.build(s, [D,F,LOAD_INDEX_D,LOAD_INDEX_F,O], target='cuda', name='conv')

        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(K,C,R,S)))
        c_np = np.zeros((N,K,P,Q), dtype=O.dtype)
        d1_np = -1*np.ones((block_num,thread_num,index_len),dtype = np.int32)
        d2_np = -1*np.ones((block_num,thread_num,index_len),dtype = np.int32)
        #d1_np = np.zeros((rD,cD),dtype = np.int32)
        #d2_np = np.zeros((rF,cF),dtype = np.int32)
        print("now start compute index")
        #compute_index(d1_np)
        #compute_indexF(d2_np)
        print("index computed")
        #print(d1_np[255])

        #print(d1_np[0])
        #print(d2_np[0])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d1 = tvm.nd.array(d1_np,ctx)
        d2 = tvm.nd.array(d2_np,ctx)

        f(a,b,d1,d2,c)
        result = c.asnumpy()
        #store = s.asnumpy()
        #
        amx = mx.nd.array(a_np)
        bmx = mx.nd.array(b_np)

        Conv = mx.symbol.Convolution
        result2 = single_dev_consist(amx,bmx)
        
        verify = False
        if verify:
            np.testing.assert_allclose(result,result2,rtol=1e-2)
            print("verify accuracy success")
        else:
            print("accuracy not verifyied")

        num_flops = P*Q*2*K*N*C*R*S
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
        t = timer_f(a, b, d1,d2,c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))





    
