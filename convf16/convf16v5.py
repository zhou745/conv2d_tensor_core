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

def convolutionf16(D,F,temp,O):
    ib = tvm.ir_builder.create()
    
    #define blocks and thread
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',block_num)
    thread_x=tvm.thread_axis('threadIdx.x')
    ib.scope_attr(thread_x,'thread_extent',thread_num)
    #neck name for block id and thread id
    bidx = block_x
    tidx = thread_x

    #set shared memory buffer for interacting with fragment
    shmem_O = ib.allocate("float16", 16384, name="shmem_O",scope = "shared")

    shmem_D = ib.allocate("float16", 3072, name="shmem_D",scope = "shared")

    shmem_F = ib.allocate("float16", 3072, name="shmem_F",scope = "shared")

    #sync thread syntex
    sync = tvm.call_extern("float32","__syncthreads")

    #declare matrix fragement
    Define_matrix_fragment_a = tvm.call_extern("float32","SET_FRAGMENT_A",warp_col_tile)
    ib.emit(Define_matrix_fragment_a)
    
    Define_matrix_fragment_b = tvm.call_extern("float32","SET_FRAGMENT_B",warp_row_tile)
    ib.emit(Define_matrix_fragment_b)

    Define_matrix_fragment_c = tvm.call_extern("float32","SET_FRAGMENT_CF16",warp_col_tile,warp_row_tile)
    ib.emit(Define_matrix_fragment_c)

    #loading parameter
    row_num = 16+shieft
    o_row_num = warp_row_tile*block_row_warp*16
    
    #define the parameters to calculate the position of data on local memory
    declare_cord = tvm.call_extern("float32","DECLARE_PARA")
    ib.emit(declare_cord)
    #initialize the parameter by tidx and bidx
    init_cord = tvm.call_extern("float32","INIT_PARA",N,P,Q,ph,pw,H,W,R,S)
    ib.emit(init_cord)

    #defclare pointer to F and D
    Dp=D.access_ptr("r")
    Fp=F.access_ptr("r")

    declare_d = tvm.call_extern("float32","POINTER_D",Dp,C,H,W)
    ib.emit(declare_d)

    declare_f = tvm.call_extern("float32","POINTER_F",Fp,C,R,S)
    ib.emit(declare_f)
    
    #the offset
    offset_df_load=tidx/2*24+tidx%2*8
    
    warpid=tidx/32
    lane=tidx%32
    warp_offset_o = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*warp_col_tile*16*o_row_num
    #now start main loop               
    with ib.for_range(0,loop_len,name ="blk_id") as blk_id:
        with ib.if_scope(bidx+blk_id*block_num<rD*rF//block_len//block_len):
            #load the first tile of matrix F and D
            bx = (bidx+blk_id*block_num)//(rD//block_len)
            by = (bidx+blk_id*block_num)%(rD//block_len)
            load_d_matrix=tvm.call_extern("float32","LOAD_D",shmem_D[offset_df_load],H,W)
            ib.emit(load_d_matrix)

            load_f_matrix=tvm.call_extern("float32","LOAD_F",shmem_F[offset_df_load],K)
            ib.emit(load_f_matrix)
            #advance the pointer
            advance_red = tvm.call_extern("float32","ADVANCE_PARA_P",C,H,W,R,S)
            ib.emit(advance_red)
            #load matrix fragment for C
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:              
                    fill_O_zero = tvm.call_extern("float","FILLZERO_CF16",col_id,row_id)
                    ib.emit(fill_O_zero)
    
            """
            #record the first shared memory
            with ib.if_scope(bidx<1):
                with ib.if_scope(blk_id<1):
                    tempp=temp.access_ptr("w")
                    load_sh = tvm.call_extern("float32","COPY",tempp,shmem_D[0],3072)
                    ib.emit(load_sh)
            """
            #sync
            ib.emit(sync)
            #loop for reduce at dimentison C R S
            with ib.for_range(0,cD//16,name = "reduce_crs") as reduce_crs:
                #load matrix fragment for A and B
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    load_matrix_frag_F = tvm.call_extern("float32","LOADFRAG_A",shmem_F[tidx/64*32*row_num+16*col*row_num],col,row_num)
                    ib.emit(load_matrix_frag_F)
        
                with ib.for_range(0,warp_row_tile,name = "row") as row:
                    load_matrix_frag_D = tvm.call_extern("float32","LOADFRAG_B",shmem_D[tidx%64/32*64*row_num+16*row*row_num],row,row_num)
                    ib.emit(load_matrix_frag_D)

                #start the compute
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        wmma_compute = tvm.call_extern("float32","WMMA_SYNC",col,row)
                        ib.emit(wmma_compute)
                #load the data for next time
                with ib.if_scope(reduce_crs<cD//16-1):
                    load_d_matrix=tvm.call_extern("float32","LOAD_D",shmem_D[offset_df_load],H,W)
                    ib.emit(load_d_matrix)

                    load_f_matrix=tvm.call_extern("float32","LOAD_F",shmem_F[offset_df_load],K)
                    ib.emit(load_f_matrix)
                    #advance the pointer
                    advance_red = tvm.call_extern("float32","ADVANCE_PARA_P",C,H,W,R,S)
                    ib.emit(advance_red)
                    ib.emit(sync)
                     #record the first shared memory
                    """
                    with ib.if_scope(bidx<1):
                        with ib.if_scope(blk_id<1):
                            with ib.if_scope(reduce_crs<16):
                                with ib.if_scope(reduce_crs>14):
                                    tempp=temp.access_ptr("w")
                                    load_sh = tvm.call_extern("float32","COPY",tempp,shmem_D[0],3072)
                                    ib.emit(load_sh)
                    """
            ib.emit(sync)
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

            reset_para = tvm.call_extern("float32","RESET_PARA",N,P,Q,ph,pw,H,W,C,R,S,blk_id,Dp,Fp)
            #ib.emit(reset_para)

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
    par_len = 17
    
    cD = ((R*S*C-1)//16+1)*16        
    cF = cD

    rD = ((P*Q*N-1)//block_len+1)*block_len
    rF = ((K-1)//block_len+1)*block_len

    loop_len = (rD*rF//block_len//block_len-1)//block_num+1              # length of the main loop
    index_len = loop_len*(cD//16)*8+8
    
    print(loop_len)
    print(P,Q,cD,rF)

    OFFIND = np.ones((block_num,thread_num),dtype = np.int32)
    with tvm.target.create('cuda'):
        D = tvm.placeholder((N,C,H,W),dtype = 'float16')
        F = tvm.placeholder((K,C,R,S),dtype = 'float16')
        temp = tvm.placeholder((3072,),dtype = 'float16')

        O = tvm.extern((N,K,P,Q),[D,F,temp],lambda ins,outs:convolutionf16(ins[0],ins[1],ins[2],\
                                                                      outs[0]),name = "conv",dtype = 'float16')
        s = schedule_conv_fp16()
        
        print(tvm.lower(s,[D,F,temp,O],name ='convf16',simple_mode = True))
        f = tvm.build(s, [D,F,temp,O], target='cuda', name='conv')

        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(K,C,R,S)))
        c_np = np.zeros((N,K,P,Q), dtype=O.dtype)
        t_np = np.zeros((3072),dtype=np.float16)
        #temp_np = np.zeros((3072),dtype = np.float16)
        print("now start compute index")

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        t = tvm.nd.array(t_np,ctx)
        #temp = tvm.nd.array(temp_np,ctx)
        f(a,b,t,c)
        result = c.asnumpy()
        #store = s.asnumpy()
        
        vit = t.asnumpy()
        print(vit[0:16])
        true_a=[]
        for idd in range(16):
            true_a.append(a_np[0][idd][0][0])
        
        print(true_a)
        print(vit[24:40])
        print(vit[48:64])
        print(vit[72:88])
        print(vit[96:112])
        print(vit[120:136])

        amx = mx.nd.array(a_np)
        bmx = mx.nd.array(b_np)

        Conv = mx.symbol.Convolution
        result2 = single_dev_consist(amx,bmx)
        
        verify = True
        if verify:
            np.testing.assert_allclose(result,result2,rtol=1e-2)
            print("verify accuracy success")
        else:
            print("accuracy not verifyied")

        num_flops = P*Q*2*K*N*C*R*S
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
        t = timer_f(a,b,t,c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))





    
