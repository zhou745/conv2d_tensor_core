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

def convolutionf16(D,F,O):
    ib = tvm.ir_builder.create()
    
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',block_num)
    thread_x=tvm.thread_axis('threadIdx.x')
    ib.scope_attr(thread_x,'thread_extent',thread_num)
    
    bidx = block_x
    tidx = thread_x
    #set shared memory buffer for loading data as column
    shmem = ib.allocate("float16", 24576, name="shmem",scope = "shared")

    #sync thread model
    sync = tvm.call_extern("float32","__syncthreads")

    #declare matrix fragement
    Define_matrix_fragment = tvm.call_extern("float32","__FRAGMENT_F16__")
    ib.emit(Define_matrix_fragment)

    #cordinate number
    index0 = tvm.var("index0")
    index1 = tvm.var("index1")
    index2 = tvm.var("index2")
    index3 = tvm.var("index3")

    indexN = tvm.var("indexN")
    indexC = tvm.var("indexC")
    indexK = tvm.var("indexK")
    indexP = tvm.var("indexP")
    indexQ = tvm.var("indexQ")
    indexH = tvm.var("indexH")
    indexW = tvm.var("indexW")
    indexR = tvm.var("indexR")
    indexS = tvm.var("indexS")

    indexu = tvm.var("indexu")
    indexv = tvm.var("indexv")
    indexph = tvm.var("indexph")
    indexpw = tvm.var("indexpw")
    
    warpid = tidx//32
    lane = tidx%32
    o_row_num = warp_row_tile*block_row_warp*16
    warp_offset_o = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*warp_col_tile*16*o_row_num

    #the loop that loop through the blocks
    with ib.for_range(0,loop_len,name ="blk_id") as blk_id:
        with ib.if_scope(bidx+blk_id*block_num<rD*rF//block_len//block_len):
            with ib.for_range(0,cD//16,name = "reduce_crs") as reduce_crs:
                #the block location of current iteration
                bx = (bidx+blk_id*block_num)//(rD//block_len)
                by = (bidx+blk_id*block_num)%(rD//block_len)
                #load matrix O and load it to matrix fragment if reduce_crs is 0
                with ib.if_scope(reduce_crs==0):
                    Op = O.access_ptr("r")
                    index0 = bx
                    index1 = by
                    index2 = warpid
                    index3 = lane

                    indexN = N
                    indexK = K
                    indexP = P
                    indexQ = Q
                    load_O_matrix = tvm.call_extern("float32","load_O_matrix",\
                                                    Op,shmem,index0,index1,index2,index3,\
                                                    indexN,indexK,indexP,indexQ)
                    ib.emit(load_O_matrix)
                    with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                        with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                            index0 = col_id
                            index1 = row_id
                            
                            load_O_fragment = tvm.call_extern("float32","__LOADFRAG_C_F16__",shmem[warp_offset_o+col_id*16*o_row_num+row_id*16],index0,index1,o_row_num)
                            
                            #fill_O_zero = tvm.call_extern("float","__FILL_C_F16__",index0,index1)
                            ib.emit(load_O_fragment)
                    ib.emit(sync)
                #load filter and image block
                row_num = 16+shieft
                offset_F = 16*block_col_warp*warp_col_tile*row_num
                Dp = D.access_ptr("r")
                index0 = by
                index1 = reduce_crs
                index2 = tidx*1
                indexN = N
                indexC = C
                indexH = H
                indexW = W
                indexu = u
                indexv = v
                indexph = ph
                indexpw = pw
                indexP = P
                indexQ = Q
                indexR = R
                indexS = S
                load_D_matrix = tvm.call_extern("float32","load_D_matrix",Dp,shmem,index0,index1,index2,\
                                                                          indexN,indexC,indexH,indexW,\
                                                                          indexu,indexv,indexph,indexpw,\
                                                                          indexP,indexQ,row_num)

                ib.emit(load_D_matrix)

                Fp = F.access_ptr("r")
                index3 = bx
                load_F_matrix = tvm.call_extern("float32","load_F_matrix",Fp,shmem,offset_F,index3,index1,index2,\
                                                                          indexK,indexC,indexR,indexS,row_num)
                ib.emit(load_F_matrix)    
                ib.emit(sync)

                
                #start computing
                """
                with ib.if_scope(blk_id==0):
                    with ib.if_scope(reduce_crs==0):
                        with ib.if_scope(bidx<1):
                            len_ = 16*8*2*(16+8)
                            Sp = SH.access_ptr("r")
                            store_shared = tvm.call_extern("float32","sl_exchange",shmem,Sp,len_)
                            ib.emit(store_shared)
                            ib.emit(sync)
                """
                warp_offset_F = warpid//block_row_warp*16*warp_col_tile*row_num
                warp_offset_D = warpid%block_row_warp*16*warp_row_tile*row_num
         
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    index0 = col
                    load_matrix_frag_F = tvm.call_extern("float32","__LOADFRAG_A__",shmem[offset_F+warp_offset_F+col*16*row_num],index0,row_num)
                    ib.emit(load_matrix_frag_F)
                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        index1 = row
                        with ib.if_scope(col==0):
                            load_matrix_frag_D = tvm.call_extern("float32","__LOADFRAG_B__",shmem[warp_offset_D+row*16*row_num],index1,row_num)
                            ib.emit(load_matrix_frag_D)
                        wmma_compute = tvm.call_extern("float32","__WMMA_SYNC__",index0,index1)
                        ib.emit(wmma_compute)
                ib.emit(sync)
      
                #load O back to tnesor when computing finished
                with ib.if_scope(reduce_crs == cD//16-1):
                    with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                        with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                            index0 = col_id
                            index1 = row_id
                            store_O_fragment = tvm.call_extern("float32","__STOREFRAG_C_F16__",shmem[warp_offset_o+col_id*16*o_row_num+row_id*16],\
                                                                                             index0,index1,o_row_num)
                            ib.emit(store_O_fragment)
                    ib.emit(sync)

                    Op = O.access_ptr("w")
                    index0 = bx
                    index1 = by
                    index2 = warpid
                    index3 = lane

                    indexN = N
                    indexK = K
                    indexP = P
                    indexQ = Q
                    store_O_matrix = tvm.call_extern("float32","store_O_matrix",\
                                                    Op,shmem,index0,index1,index2,index3,\
                                                    indexN,indexK,indexP,indexQ)
                    ib.emit(store_O_matrix)
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
def compute_first(d,f):
    out = 0
    for ch in range(C):
        out+=d[0][ch][0][0]*f[0][ch][1][1]+d[0][ch][0][1]*f[0][ch][1][2]+d[0][ch][1][0]*f[0][ch][2][1]+d[0][ch][1][1]*f[0][ch][2][2]
    print(out)
    
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

    
    print(loop_len)
    print(P,Q)
    with tvm.target.create('cuda'):
        D = tvm.placeholder((N,C,H,W),dtype = 'float16')
        F = tvm.placeholder((K,C,R,S),dtype = 'float16')
        O = tvm.extern((N,K,P,Q),[D,F],lambda ins,outs:convolutionf16(ins[0],ins[1],outs[0]),name = "conv",dtype = 'float16')
        s = schedule_conv_fp16()
        
        print(tvm.lower(s,[D,F,O],name ='convf16',simple_mode = True))
        f = tvm.build(s, [D,F,O], target='cuda', name='conv')
        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(K,C,R,S)))
        c_np = np.zeros((N,K,P,Q), dtype=O.dtype)
        

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)

        f(a,b,c)
        result = c.asnumpy()
        #store = s.asnumpy()
        #
        amx = mx.nd.array(a_np)
        bmx = mx.nd.array(b_np)

        Conv = mx.symbol.Convolution
        result2 = single_dev_consist(amx,bmx)
          
        #np.testing.assert_allclose(result,result2,rtol=1e-2)
        #

        print("verify accuracy success")

        num_flops = P*Q*2*K*N*C*R*S
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
        t = timer_f(a, b, c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))




    
