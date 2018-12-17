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

def convolutionf16(D,F,BI,TI,DI,FI,TPAR,PAR,O):
    ib = tvm.ir_builder.create()
    
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',block_num)
    thread_x=tvm.thread_axis('threadIdx.x')
    ib.scope_attr(thread_x,'thread_extent',thread_num)
    #neck name 
    bidx = block_x
    tidx = thread_x

    #set the  memory cordinate to the first of DI and FI
    boffset = ib.allocate("int32", 1, name="boffset",scope = "local")
    toffset = ib.allocate("int32", 1, name="toffset",scope = "local")
    tpoffset = ib.allocate("int32", 1, name="tpoffset",scope = "local")
    pp = ib.allocate("int32", par_len, name="parameters",scope = "local")

    Bp = BI.access_ptr("r",offset=block_x)
    move_to_block = tvm.call_extern("float32","GETITEM",boffset[0],Bp)
    Tp = TI.access_ptr("r",offset=boffset[0]+thread_x)
    TPARp = TPAR.access_ptr("r",offset=boffset[0]+thread_x)
    move_to_thread = tvm.call_extern("float32","GETITEM",toffset[0],Tp)
    move_to_thread_par = tvm.call_extern("float32","GETITEM",tpoffset[0],TPARp)
    ib.emit(move_to_block)
    ib.emit(move_to_thread)
    ib.emit(move_to_thread_par)
    #pointer to index
    Dip = DI.access_ptr("r",offset = toffset[0])
    Fip = FI.access_ptr("r",offset = toffset[0])
    PARp = PAR.access_ptr("r",offset = tpoffset[0])
    #declare pointer 
    dip_set = tvm.call_extern("float32","POINTER_DIP",Dip)
    fip_set = tvm.call_extern("float32","POINTER_FIP",Fip)
    ib.emit(dip_set)
    ib.emit(fip_set)

    #set par_sh to the parameter
    set_pointer = tvm.call_extern("float32","SETPOINTER",pp,PARp,par_len)
    ib.emit(set_pointer)

    #pointer to data
    Dp = D.access_ptr("r")
    Fp = F.access_ptr("r")
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
    #loading parameter
    row_num = 16+shieft
    o_row_num = warp_row_tile*block_row_warp*16
    #caculate the id of current warp
    warpid = tidx//32
    #caculate the id of current thread inside current warp
    lane = tidx%32
    #paras


    #now start main loop
    with ib.for_range(0,loop_len,name ="blk_id") as blk_id:
        with ib.if_scope(bidx+blk_id*block_num<rD*rF//block_len//block_len):
            bx = (bidx+blk_id*block_num)//(rD//block_len)
            by = (bidx+blk_id*block_num)%(rD//block_len)

            #now load F, D for the first reduce iteration
            load_D_matrix = tvm.call_extern("float32","LOAD_MATRIX_D",Dp,shmem_D[pp[16]])
            ib.emit(load_D_matrix)
            load_F_matrix = tvm.call_extern("float32","LOAD_MATRIX_F",Fp,shmem_F[pp[16]])           
            ib.emit(load_F_matrix)
            #load the out put matrix fragment
            with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:              
                    fill_O_zero = tvm.call_extern("float","FILLZERO_CF16",col_id,row_id)
                    ib.emit(fill_O_zero)
            ib.emit(sync)

            with ib.for_range(0,cD//16,name = "reduce_crs") as reduce_crs:
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    load_matrix_frag_F = tvm.call_extern("float32","LOADFRAG_A",shmem_F[pp[8+col]],col,row_num)
                    ib.emit(load_matrix_frag_F)
        
                with ib.for_range(0,warp_row_tile,name = "row") as row:
                    load_matrix_frag_D = tvm.call_extern("float32","LOADFRAG_B",shmem_D[pp[8+warp_col_tile+row]],row,row_num)
                    ib.emit(load_matrix_frag_D)
                with ib.for_range(0,warp_col_tile,name = "col") as col:
                    with ib.for_range(0,warp_row_tile,name = "row") as row:
                        wmma_compute = tvm.call_extern("float32","WMMA_SYNC",col,row)
                        ib.emit(wmma_compute)
                with ib.if_scope(reduce_crs<cD//16-1):
                    load_D_matrix = tvm.call_extern("float32","LOAD_MATRIX_D",Dp,shmem_D[pp[16]])
                    ib.emit(load_D_matrix)
                    load_F_matrix = tvm.call_extern("float32","LOAD_MATRIX_F",Fp,shmem_F[pp[16]])           
                    ib.emit(load_F_matrix)
                ib.emit(sync)
                #load back to output
                with ib.if_scope(reduce_crs == cD//16-1):
                    with ib.for_range(0,warp_col_tile,name = "col_id") as col_id:
                        with ib.for_range(0,warp_row_tile, name = "row_id") as row_id:
                            store_O_fragment = tvm.call_extern("float32","STOREFRAG_C_F16",shmem_O[pp[col_id*warp_row_tile+row_id]],\
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

def compute_BI(bi_tb):
    for i in range(len(bi_tb)):
        bi_tb[i] = i*thread_num

def compute_TI(ti_tb):
    for i in range(len(ti_tb)):
        ti_tb[i]=i*index_len

def compute_TPAR(tpar_tb):
    for i in range(len(tpar_tb)):
        tpar_tb[i] =i*par_len

def compute_DFI(di_tb,fi_tb):
    blen_c = 16*warp_col_tile*block_col_warp
    blen_r = 16*warp_row_tile*block_row_warp
    for i in range(block_num*thread_num):
        bidx = i/thread_num
        tidx = i%thread_num
        warp = tidx/32
        lane = tidx%32
        shieft = lane%2
        if(tidx==0):
            print("finished %f"%(bidx/80.0))
        for j in range(index_len):
            blk_id = j/(cD//16*8)
            bx = (bidx+blk_id*block_num)//(rD//block_len)
            by = (bidx+blk_id*block_num)%(rD//block_len)

            idy = by*blen_r+warp*16+lane/2
            idx = bx*blen_c+warp*16+lane/2
            dN = idy/(P*Q)
            dP = (idy/P)%Q
            dQ = idy%Q
            dH = dP*u-ph
            dW = dQ*v-pw
            reduce_id=j%(cD//16*8)//8
            remains = j%(cD//16*8)%8
            rows = remains+reduce_id*16+shieft*8
            dC = rows/(R*S)
            dR = (rows/S)%R
            dS = rows%S
            dH2=dH+dR
            dW2=dW+dS
            if(dH2>=0 and dW2>=0 and dH2<H and dW2<W and dN<N and dC<C):
                di_tb[i][j] = dN*C*H*W+dC*H*W+dH2*W+dW2
            dK = idx
            if(dK<K and dC<C):
                fi_tb[i][j] = dK*C*R*S+dC*R*S+dR*S+dS
#the first 8 is used for load fragement
def compute_PAR(par_tb):
    o_row_num = warp_row_tile*block_row_warp*16
    row_num=16+shieft
    for i in range(len(par_tb)):
        bidx = i/thread_num
        tidx = i%thread_num
        warp = tidx/32
        lane = tidx%32
        warp_offset_o = warp%block_row_warp*16*warp_row_tile+warp/block_row_warp*warp_col_tile*16*o_row_num
        for index in range(8):
            col_id=index/warp_row_tile
            row_id=index%warp_row_tile
            par_tb[i][index]=warp_offset_o+col_id*16*o_row_num+row_id*16

        offset_warp_col = warp/block_row_warp*warp_col_tile*row_num*16
        for index in range(warp_col_tile):
            par_tb[i][index+8]=offset_warp_col+index*16*row_num
        
        offset_warp_row = warp%block_row_warp*warp_row_tile*16*row_num
        for index in range(warp_row_tile):
            par_tb[i][index+8+warp_col_tile]=offset_warp_row+index*16*row_num
        offset_sh_load = warp*16*row_num+(lane/2)*row_num+8*(lane%2)
        par_tb[i][16]=offset_sh_load
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
    print(P,Q,cD)

    OFFIND = np.ones((block_num,thread_num),dtype = np.int32)
    with tvm.target.create('cuda'):
        D = tvm.placeholder((N,C,H,W),dtype = 'float16')
        F = tvm.placeholder((K,C,R,S),dtype = 'float16')
        BI = tvm.placeholder((block_num,),dtype = 'int32')
        TI = tvm.placeholder((block_num*thread_num,),dtype ='int32')
        DI = tvm.placeholder((block_num*thread_num,index_len),dtype='int32')
        FI = tvm.placeholder((block_num*thread_num,index_len),dtype='int32')
        TPAR = tvm.placeholder((block_num*thread_num,),dtype ='int32')
        PAR = tvm.placeholder((block_num*thread_num,par_len),dtype = "int32")
        #temp = tvm.placeholder((3072,),dtype = "float16")
        O = tvm.extern((N,K,P,Q),[D,F,BI,TI,DI,FI,TPAR,PAR],lambda ins,outs:convolutionf16(ins[0],ins[1],\
                                                                                 ins[2],ins[3],\
                                                                                 ins[4],ins[5],\
                                                                                 ins[6],ins[7],\
                                                                                 outs[0]),name = "conv",dtype = 'float16')
        s = schedule_conv_fp16()
        
        print(tvm.lower(s,[D,F,BI,TI,DI,FI,TPAR,PAR,O],name ='convf16',simple_mode = True))
        f = tvm.build(s, [D,F,BI,TI,DI,FI,TPAR,PAR,O], target='cuda', name='conv')

        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(K,C,R,S)))
        c_np = np.zeros((N,K,P,Q), dtype=O.dtype)
        bi_np = np.zeros((block_num),dtype = np.int32)
        ti_np = np.zeros((block_num*thread_num),dtype = np.int32)
        di_np = -np.ones((block_num*thread_num,index_len),dtype = np.int32)
        fi_np = -np.ones((block_num*thread_num,index_len),dtype = np.int32)
        tpar_np = np.zeros((block_num*thread_num),dtype = np.int32)
        par_np = np.zeros((block_num*thread_num,par_len),dtype = np.int32)
        #temp_np = np.zeros((3072),dtype = np.float16)
        print("now start compute index")
        compute_BI(bi_np)
        compute_TI(ti_np)
        compute_TPAR(tpar_np)
        #compute_DFI(di_np,fi_np)
        compute_PAR(par_np)
        print("index computed")
        #print(par_np[0])
        #print(par_np[32])
        #print(fi_np[0][0:16])
        #print(fi_np[1][0:16])
        #print(d1_np[255])

        #print(d1_np[0])
        #print(d2_np[0])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        bi = tvm.nd.array(bi_np,ctx)
        ti = tvm.nd.array(ti_np,ctx)
        di = tvm.nd.array(di_np,ctx)
        fi = tvm.nd.array(fi_np,ctx)
        tpar = tvm.nd.array(tpar_np,ctx)
        par = tvm.nd.array(par_np,ctx)
        #temp = tvm.nd.array(temp_np,ctx)
        f(a,b,bi,ti,di,fi,tpar,par,c)
        result = c.asnumpy()
        #store = s.asnumpy()
        
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
        t = timer_f(a,b,bi,ti,di,fi,tpar,par,c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))





    
