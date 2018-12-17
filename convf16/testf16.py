import functools
import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes

TASK = 'gemm_ir'
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
    thread_x=tvm.thread_axis('threadIdx.x')   
    ib.scope_attr(thread_x,'thread_extent',num_thread)
    #declare shared memory
    offsetb = 16*16
    
    sync = tvm.call_extern("float32","__syncthreads")
    #define fragment 
    def_matrix_frag = tvm.call_extern("float32","__FRAGMENT_F16__")
    ib.emit(def_matrix_frag)
  

    
    load_matrix_frag_a = tvm.call_extern("float32","__LOADFRAG_A__",A,0,16)
    ib.emit(load_matrix_frag_a)
    load_matrix_frag_b = tvm.call_extern("float32","__LOADFRAG_B__",B,0,16)
    ib.emit(load_matrix_frag_b)
    ib.emit(sync)
    wmma_compute = tvm.call_extern("float32","__WMMA_SYNC__",0,0)
    ib.emit(wmma_compute)
    ib.emit(sync)
    store_matrix_frag_c1 = tvm.call_extern("float32","__STOREFRAG_C_F16__",C,0,0,16)
    ib.emit(store_matrix_frag_c1)
    ib.emit(sync)
           
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
    rA = 16
    cA = 16
    rB = 16
    cB = 16
    num_thread=32

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
                                  np.dot(np.float32(a_np),\
                                         np.float32(np.transpose(b_np))),
                                  rtol=1e-3)

        print("verify the accuracy success")
        num_flops = 2 * rA * rB * cA
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
        t = timer_f(a, b, c).mean
        TFLOPS = num_flops / (t * 1e3) / 1e9
        print("average time cost of %d runs = %g ms, %g TFLOPS." %
          (num_runs, t * 1e3, TFLOPS))
