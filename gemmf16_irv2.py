import functools
import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes

TASK = 'gemm_ir_f16'
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

def Gemm_ir_wmma(A,B,C):
    ib = tvm.ir_builder.create()
    block_x=tvm.thread_axis('blockIdx.x')   
    ib.scope_attr(block_x,'thread_extent',num_block)
    thread_x=tvm.thread_axis('threadIdx.x')   
    ib.scope_attr(thread_x,'thread_extent',num_thread)

    #declare shared memory
    shmem = ib.allocate("float16", 24576, name="shmem",scope = "shared")
    #sync thread model
    sync = tvm.call_extern("float32","__syncthreads")
    #define fragment 
    def_matrix_frag = tvm.call_extern("float32","__FRAGMENT_F16__")
    ib.emit(def_matrix_frag)
    #index number
    index0 = tvm.var("index0")
    index1 = tvm.var("index1")
    row_ele_num = tvm.var("row_ele_num")

    block_r = 16*block_row_warp*warp_row_tile
    block_c = 16*block_col_warp*warp_col_tile

    block_num = (rA/block_c)*(rB/block_r)
    block_re = block_num//num_block+1

    bidx = block_x
    tidx = thread_x
    #the loop that loop through the blocks
    with ib.for_range(0,block_re,name ="b_for") as b_for:
        with ib.if_scope(bidx+num_block*b_for<block_num):
            #block offset
            b_id_x = (bidx+num_block*b_for)//(rA/block_r)*block_c
            b_id_y = (bidx+num_block*b_for)%(rB/block_r)*block_r
            block_offset_c=b_id_x*rA+b_id_y
            #warp offset for shared memory
            warpid = tidx//32
            lane = tidx%32
            c_row_ele = 16*block_row_warp*warp_row_tile +shieft   #2 float 16 equals 1 float32
            row_ele_num = c_row_ele
            warp_offset_c_sh = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*16*c_row_ele
            #warp offset for C
            tile_offset_c = 16*rB
            warp_offset_c = warpid%block_row_warp*16*warp_row_tile+warpid/block_row_warp*warp_col_tile*16*rB
            #off set for line shared memory
            with ib.for_range(0,4,name = "tile") as tile:
                #copy the data to shared memory by int4
                Cp = C.access_ptr("r",offset = block_offset_c+warp_offset_c+tile*16+lane//2*rB+lane%2*8)
                read_c_int4a1 = tvm.call_extern("float32","__INT4READ__",shmem[warp_offset_c_sh+tile*16+lane//2*c_row_ele+lane%2*8],Cp)
                ib.emit(read_c_int4a1)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                index0=0
                index1=tile
                load_matrix_frag_c1 = tvm.call_extern("float32","__LOADFRAG_C_F16__",shmem[warp_offset_c_sh+tile*16],index0,index1,row_ele_num)
                ib.emit(load_matrix_frag_c1)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                #copy the data to shared memory by int4
                Cp = C.access_ptr("r",offset = block_offset_c+warp_offset_c+tile*16+lane//2*rB+lane%2*8+tile_offset_c)
                read_c_int4b1 = tvm.call_extern("float32","__INT4READ__",shmem[warp_offset_c_sh+tile*16+lane//2*c_row_ele+lane%2*8],Cp)
                ib.emit(read_c_int4b1)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                index0=1
                index1=tile
                load_matrix_frag_c2 = tvm.call_extern("float32","__LOADFRAG_C_F16__",shmem[warp_offset_c_sh+tile*16],index0,index1,row_ele_num)
                ib.emit(load_matrix_frag_c2)
            ib.emit(sync)
            sh_ele_num = 16*chunk+shieft
            #offset for a block
            block_offset_a = b_id_x*cA
            block_offset_b = b_id_y*cB

            #warp offset
            warp_offset_a_sh = warpid/block_row_warp*16*warp_col_tile*sh_ele_num
            warp_offset_a = warpid/block_row_warp*16*warp_col_tile*cA
            warp_offset_b_sh = warpid%block_row_warp*16*warp_row_tile*sh_ele_num
            warp_offset_b = warpid%block_row_warp*16*warp_row_tile*cB

            offset_b = block_col_warp*warp_col_tile*16*sh_ele_num
            row_ele_num = sh_ele_num
            with ib.for_range(0,cA//(chunk*16),name = "reduce_i") as reduce_i:
                with ib.for_range(0,chunk,name = "chunk_i",for_type='unroll') as chunk_i:                   
                    Ap = A.access_ptr("r",offset = block_offset_a+warp_offset_a+warpid%block_row_warp*16*cA+reduce_i*chunk*16+chunk_i*16+lane//2*cA+lane%2*8)
                    read_a_int4 = tvm.call_extern("float32","__INT4READ__",shmem[warp_offset_a_sh+warpid%block_row_warp*16*sh_ele_num+chunk_i*16+lane//2*sh_ele_num+lane%2*8],Ap)
                    Bp = B.access_ptr("r",offset = block_offset_b+warp_offset_b+warpid/block_row_warp*16*cB+reduce_i*chunk*16+chunk_i*16+lane//2*cB+lane%2*8)
                    read_b_int4 = tvm.call_extern("float32","__INT4READ__",shmem[offset_b+warp_offset_b_sh+warpid/block_row_warp*16*sh_ele_num+chunk_i*16+lane//2*sh_ele_num+lane%2*8],Bp)
                    ib.emit(read_a_int4)
                    ib.emit(read_b_int4)
                ib.emit(sync)
                with ib.for_range(0,chunk,name ="chunk_i",for_type='unroll') as chunk_i:
                    with ib.for_range(0,warp_col_tile,name ="col",for_type='unroll') as col:
                        index0 = col
                        load_matrix_frag_a = tvm.call_extern("float32","__LOADFRAG_A__",shmem[warp_offset_a_sh+16*chunk_i+col*16*sh_ele_num],index0,row_ele_num)
                        ib.emit(load_matrix_frag_a)
                        with ib.for_range(0,warp_row_tile,name = "row") as row:
                            index1 = row
                            with ib.if_scope(col==0):
                                load_matrix_frag_b = tvm.call_extern("float32","__LOADFRAG_B__",shmem[offset_b+warp_offset_b_sh+16*chunk_i+row*16*sh_ele_num],index1,row_ele_num)
                                ib.emit(load_matrix_frag_b)
                            wmma_compute = tvm.call_extern("float32","__WMMA_SYNC__",index0,index1)
                            ib.emit(wmma_compute)
                ib.emit(sync)
            row_ele_num = c_row_ele
            with ib.for_range(0,4,name = "tile") as tile:
                index0=0
                index1=tile
                store_matrix_frag_c1 = tvm.call_extern("float32","__STOREFRAG_C_F16__",shmem[warp_offset_c_sh+tile*16],index0,index1,row_ele_num)
                ib.emit(store_matrix_frag_c1)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                #copy the data to shared memory by int4
                Cp = C.access_ptr("w",offset = block_offset_c+warp_offset_c+tile*16+lane//2*rB+lane%2*8)
                write_c_int4a1 = tvm.call_extern("float32","__INT4WRITE__",shmem[warp_offset_c_sh+tile*16+lane//2*c_row_ele+lane%2*8],Cp)
                ib.emit(write_c_int4a1)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                index0=1
                index1=tile
                store_matrix_frag_c2 = tvm.call_extern("float32","__STOREFRAG_C_F16__",shmem[warp_offset_c_sh+tile*16],index0,index1,row_ele_num)
                ib.emit(store_matrix_frag_c2)
            ib.emit(sync)
            with ib.for_range(0,4,name = "tile") as tile:
                #copy the data to shared memory by int4
                Cp = C.access_ptr("w",offset = block_offset_c+warp_offset_c+tile*16+lane//2*rB+lane%2*8+tile_offset_c)
                write_c_int4b1 = tvm.call_extern("float32","__INT4WRITE__",shmem[warp_offset_c_sh+tile*16+lane//2*c_row_ele+lane%2*8],Cp)
                ib.emit(write_c_int4b1)
            ib.emit(sync)

            
    #ib.scope_attr(shmem, "storage_scope", "share")
    body = ib.get()

    return(body)

@tvm.target.generic_func
def schedule_gemm_fp16():
    raise NotImplemented()

@schedule_gemm_fp16.register(['cuda'])
def _schedule_gemm_fp16():
    s = tvm.create_schedule(C.op)
    return(s)

if __name__ == "__main__":
    #input matrix shape
    rA = 4096
    cA = 4096
    rB = 4096
    cB = 4096

    #schedule parameters
    num_block = 80
    chunk = 4
    shieft = 8

    block_col_warp = 4
    block_row_warp = 2

    warp_col_tile = 2
    warp_row_tile = 4
    block_tiles = block_col_warp*warp_col_tile
    block_warp = block_col_warp*block_row_warp

    num_thread = 32*block_col_warp*block_row_warp

    with tvm.target.create('cuda'):
        A = tvm.placeholder((rA,cA),dtype = 'float16')
        B = tvm.placeholder((rB,cB),dtype = 'float16')
        C = tvm.extern((rA,rB),[A,B],lambda ins,outs:Gemm_ir_wmma(ins[0],ins[1],outs[0]),name = "ir_wmma",dtype = 'float16')
        assert(cA==cB)
        s = schedule_gemm_fp16()

        print(tvm.lower(s,[A,B,C],name ='gemmf16',simple_mode = True))
        f = tvm.build(s, [A,B,C], target='cuda', name='ir_test1')
        print("build finished")
        ctx = tvm.context('cuda', 0)
        a_np = np.float16(np.random.uniform(0.,1.,size=(rA,cA)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(rB,cB)))
        c_np = np.zeros((rA, rB), dtype=C.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        f(a,b,c)
        #for item in c.asnumpy():
        #    print([item[i*128] for i in range(32)])
   
        np.testing.assert_allclose(c.asnumpy(),\
                                  np.dot(a_np,\
                                         np.transpose(b_np)),
                                  rtol=1e-3)

        print("verify the accuracy success")
        num_flops = 2 * rA * rB * cA
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
        t = timer_f(a, b, c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))
