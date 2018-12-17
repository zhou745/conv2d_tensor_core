import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes

TASK = 'ir'
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

def ir_warp(A,B):
    ib = tvm.ir_builder.create()
    A_ptr = ib.buffer_ptr(A)
    B_ptr = ib.buffer_ptr(B)
    
    tx = tvm.thread_axis('threadIdx.x')
    ib.scope_attr(tx,'thread_extent',10)
    i = tx
    with ib.if_scope(i %2 ==0):
        B_ptr[i] = A_ptr[i] +1

    o1 = tvm.call_extern("float32","__WMMA__")
    ib.emit(o1)
    body = ib.get()
    return(body)

A = tvm.placeholder((10,),dtype = 'float32')
B = tvm.extern((10,),[A],lambda ins,outs:ir_warp(ins[0],outs[0]),name ='B',dtype = 'float32')

s = tvm.create_schedule(B.op)


print(tvm.lower(s,[A,B],name ='ir_test',simple_mode = True))
"""
f = tvm.build(s, [A,B], target='cuda', name='ir_test1')
ctx = tvm.context('cuda', 0)
a_np = np.float32(np.random.uniform(0.,1.,size=(10,)))
b_np = np.float32(np.random.uniform(0.,1.,size=(10,)))
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
f(a,b)
print(a.asnumpy())
print(b.asnumpy())
"""
