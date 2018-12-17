import functools
import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes
import sys

import mxnet as mx

TASK = 'conv2dv6'
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

    #define the computation architecture
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',block_num)
    thread_x=tvm.thread_axis('threadIdx.x')
    ib.scope_attr(thread_x,'thread_extent',thread_num)
    
    bidx=block_x
    tidx=thread_x
    warpid=tidx/32
    warp_offset_output = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*warp_col_tile*2048
    #define the double buffered shared memory
    declare_a = tvm.call_extern("float32","SET_FRAGMENT_A",warp_col_tile)
    declare_b = tvm.call_extern("float32","SET_FRAGMENT_B",warp_row_tile)
    declare_c = tvm.call_extern("float32","SET_FRAGMENT_CF16",warp_col_tile,warp_row_tile)
    ib.emit(declare_a)
    ib.emit(declare_b)
    ib.emit(declare_c)

    #define the shared memory for loading data and offset for loading the data
    shmem = ib.allocate("float16", 24576, name="shmem",scope = "shared")
    offset_D_warp = 2880+tidx/2*(16+shieft)+tidx%2*8
    offset_F_warp = 5952+tidx/2*(16+shieft)+tidx%2*8
    

    #sync thread syntex
    sync = tvm.call_extern("float32","__syncthreads")
    
    #since filter is usually small, load all filer used by the 

    #define the main loop and calculate
    with ib.for_range(0,loop_len,name ="blk_id") as blk_id:
        with ib.if_scope(bidx+blk_id*block_num<(rD*rF)/(block_zise*block_zise)):
            #set the pointer to beginning of D
            Dp=D.access_ptr("r")
            #load the first data from global memory for the reuse of 9 times
            load_first_data = tvm.call_extern("float32","load_matrix_D",Dp,shmem,blk_id,N,H,W,C,0,128)
            ib.emit(load_first_data)

            #set the pointer to beginning of F
            Fp=F.access_ptr("r")

            #load the first filter from global memory:
            load_filter=tvm.call_extern("float32","load_matrix_F",Fp,shmem,offset_F_warp,blk_id,K,C,N,H,W,tidx%2*8,0,128)
            ib.emit(load_filter)

            #load temp
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:              
                    fill_O_zero = tvm.call_extern("float","FILLZERO_CF16",col_id,row_id)
                    ib.emit(fill_O_zero)
            ib.emit(sync)

            #load the first data from shmem to im2col shmem
            im2col=tvm.call_extern("float32","im2col",shmem,offset_D_warp,0,128)
            ib.emit(im2col)
            ib.emit(sync)

            with ib.for_range(0,C/16,name="c_id",for_type = 'unroll') as c_id:
                with ib.for_range(0,9,name="ker_id",for_type = 'unroll') as ker_id:
                    #now load matrix fragment
                    with ib.for_range(0,warp_col_tile,name = "col") as col:
                        load_matrix_frag_F = tvm.call_extern("float32","LOADFRAG_A",shmem[5952+tidx/64*768+col*384],col,16+shieft)
                        ib.emit(load_matrix_frag_F)
        
                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        load_matrix_frag_D = tvm.call_extern("float32","LOADFRAG_B",shmem[2880+tidx%64/32*1536+row*384],row,16+shieft)
                        ib.emit(load_matrix_frag_D)

                    ib.emit(sync)
                    #now compute
                    with ib.for_range(0,warp_col_tile,name = "col") as col:
                        with ib.for_range(0,warp_row_tile,name = "row") as row:
                            wmma_compute = tvm.call_extern("float16","WMMA_SYNC",col,row)
                            ib.emit(wmma_compute)
            
                    with ib.if_scope(ker_id<8):
                        #load filer of the next ieration
                        load_filter=tvm.call_extern("float32","load_matrix_F",Fp,shmem,offset_F_warp,blk_id,K,C,N,H,W,c_id*16+tidx%2*8,ker_id+1,128)
                        ib.emit(load_filter)
                        #load data for next iteration
                        im2col=tvm.call_extern("float32","im2col",shmem,offset_D_warp,ker_id+1,128)
                        ib.emit(im2col)
                    ib.emit(sync)
     
                with ib.if_scope(c_id<C/16-1):
                    #load the next 9 iteration data from global memory
                    load_data = tvm.call_extern("float32","load_matrix_D",Dp,shmem,blk_id,N,H,W,C,c_id*16+16,128)
                    ib.emit(load_data)

                    #load filter for next cd iter
                    load_filter=tvm.call_extern("float32","load_matrix_F",Fp,shmem,offset_F_warp,blk_id,K,C,N,H,W,c_id*16+16+tidx%2*8,0,128)
                    ib.emit(load_filter)
                    ib.emit(sync)

                    #load the first data from shmem to im2col shmem
                    im2col=tvm.call_extern("float32","im2col",shmem,offset_D_warp,0,128)
                    ib.emit(im2col)
                    ib.emit(sync)
            #now start reload back to output
            #load fragment to shared memory first
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                    store_O_fragment = tvm.call_extern("float32","STOREFRAG_C_F16",shmem[warp_offset_output+col_id*2048+row_id*16],col_id,row_id,128)
                    ib.emit(store_O_fragment)
            ib.emit(sync)
            Op=O.access_ptr("w")
            store_O=tvm.call_extern("float32","store_output",Op,shmem,blk_id,N,P,Q,K,128)
            ib.emit(store_O)
            ib.emit(sync)
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
    #define tiling parameter
    block_row_warp=2
    block_col_warp=4

    warp_row_tile=4
    warp_col_tile=2
    
    #define block and thread
    block_num = 80
    thread_num = 256
    
    shieft=8
    #define conv size
    N=1
    C=64
    H=256
    W=256

    K=128
    R=3
    S=3
    
    ph=1
    pw=1

    sh=1
    sw=1

    P = (H-R+1+2*ph)/sh
    Q = (W-S+1+2*pw)/sw
    
    #image2col
    cD=R*S*C
    cF=cD
     
    block_zise=block_col_warp*warp_col_tile*16
    rD=((H*W*N-1)/block_zise+1)*block_zise
    rF=((K-1)/block_zise+1)*block_zise
    print(rD,rF)
    #number of blocks processed in the iteration for each block
    loop_len = (rD*rF-1)/(block_zise*block_zise*block_num)+1

    with tvm.target.create('cuda'):
        D = tvm.placeholder((N,C,H,W),dtype = 'float16')
        F = tvm.placeholder((K,R,S,C),dtype = 'float16')

        temp = tvm.placeholder((3072,),dtype='float16')

        O = tvm.extern((N,K,P,Q),[D,F,temp],lambda ins,outs:convolutionf16(ins[0],ins[1],ins[2],\
                                                                      outs[0]),name = "conv",dtype = 'float16')
        s = schedule_conv_fp16()
        
        print(tvm.lower(s,[D,F,temp,O],name ='convf16',simple_mode = True))
        f = tvm.build(s, [D,F,temp,O], target='cuda', name='conv')

        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(K,R,S,C)))
        c_np = np.zeros((N,K,P,Q), dtype=O.dtype)
        t_np = np.zeros((3072,),dtype=np.float16)
        print("now start compute index")

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        t = tvm.nd.array(t_np, ctx)

        f(a,b,t,c)
        result = c.asnumpy()
        #store = s.asnumpy()
        temp2=t.asnumpy()

        #print(temp2[2032:2048])

        dataf=[]
        for index in range(16):
            dataf.append(a_np[0][index][6][0])
        #print(dataf)
        amx = mx.nd.array(a_np)
        
        rb_np=np.transpose(b_np,(0,3,1,2))

        bmx = mx.nd.array(rb_np)

        Conv = mx.symbol.Convolution
        result2 = single_dev_consist(amx,bmx)
        print("result")
        #print(result[0][0][254][240:256])
        #print(result2[0][0][254][240:256])
        for row in range(256):
            for id in range(256):
                temp=(1.0*result[0][0][row][id]-1.0*result2[0][0][row][id])/result2[0][0][row][id]
                if temp>0.01:
                    print(row,id,result[0][0][row][id],result2[0][0][row][id])
        #print(result[0][0][255][240:256])
        #print(result2[0][0][255][240:256])

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
