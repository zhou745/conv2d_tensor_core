import functools
import tvm
from tvm.contrib import nvcc
import numpy as np
import os
import ctypes
#this task write a function that perform fp16 matrix computation
TASK = 'gemm'
USE_MANUAL_CODE = False


@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target='ptx',arch='sm_70',options=['--maxrregcount', '128','-I /root/zhoujq/gemmf16/wmma_gemm_f16/include'])
    return ptx


def write_code(code, fname):
    with open(fname, 'w') as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


@tvm.target.generic_func
def schedule_gemm_fp16():
    raise NotImplemented()

def mma_sync_wmma():
    factor1 = tvm.placeholder((1,),name='factor1',dtype='float16')
    factor2 = tvm.placeholder((1,),name='factor2',dtype='float16')
    product = tvm.placeholder((1,),name='product',dtype='float32')
    #product = tvm.placeholder((1,),name='product',dtype='float16')
    schedule = tvm.compute((1,),lambda _:(factor1[0]+factor2[0]+product[0].astype('float16')))
    
    def mma_sync(inputs,outputs): 
        print(inputs) 
        factor1_,factor2_,product_ = inputs
        schedule_=outputs[0]
        #get address for matrix A
        A_ptr = factor1_.access_ptr("r")
        #get address for matrix B
        B_ptr = factor2_.access_ptr("r")
        #get address for matrix C
        C_ptr = product_.access_ptr("w")

        body = tvm.call_extern('float32',"wmma_call",A_ptr,B_ptr,C_ptr)
        #body = tvm.call_extern('float32',"__INIT_TILE_WARP__")
        init = tvm.call_extern('float32',"__INIT_TILE_WARP__")
        #product_.vstore((0,0,0,0),0.)
        return body, init,body
    
    with tvm.build_config(data_alignment=1,offset_factor=1) as cfg:
        binds = {t: tvm.decl_buffer(t.shape, t.dtype, t.op.name,
                                    data_alignment=cfg.data_alignment, offset_factor=cfg.offset_factor,
                                    scope='global') for t in [factor1, factor2]}
        print(factor1.shape[0].dtype)
        binds.update({product:tvm.decl_buffer(product.shape, product.dtype, product.op.name,
                                    data_alignment=cfg.data_alignment, offset_factor=cfg.offset_factor,
                                    scope='global')})            
        binds.update({schedule:tvm.decl_buffer(schedule.shape, schedule.dtype, schedule.op.name,
                                    data_alignment=cfg.data_alignment, offset_factor=cfg.offset_factor,
                                    scope='global')})
        return tvm.decl_tensor_intrin(schedule.op, mma_sync, binds=binds)

@schedule_gemm_fp16.register(['cuda'])
def _schedule_gemm_fp16():
    s = tvm.create_schedule(D.op)
    bx,by,th,warp = s[D].op.axis
    s[D].tensorize(warp,mma_sync_wmma())
    b = s[D].fuse(bx,by)
    b,r = s[D].split(b,nparts = num_block)
    s[D].reorder(b,th,r)

    block_x = tvm.thread_axis('blockIdx.x')
    thread_x=tvm.thread_axis('threadIdx.x')

    s[D].bind(b,block_x)
    s[D].bind(th,thread_x)

    return(s)

if __name__ == "__main__":
    #input matrix shape
    rA = 4096
    cA = 4096
    rB = 4096
    cB = 4096

    #schedule parameters
    num_block = 80

    block_col_warp = 4
    block_row_warp = 2

    warp_col_tile = 2
    warp_row_tile = 4
    block_tiles = block_col_warp*warp_col_tile
    block_warp = block_col_warp*block_row_warp

    num_thread = 32

    with tvm.target.create('cuda'):
        A = tvm.placeholder((rA,cA),dtype = 'float16')
        B = tvm.placeholder((rB,cB),dtype = 'float16')
        C = tvm.placeholder((rA,rB),dtype = 'float32')
        #C = tvm.placeholder((rA,rB),dtype = 'float16')
        assert(cA==cB)
        D = tvm.compute((rA//(16*block_tiles),rB//(16*block_tiles),(num_thread*block_warp),1),\
                         lambda i,j,w,warp:(A[i*16*block_tiles+(w//(num_thread)//block_row_warp)*16*warp_col_tile,0]+\
                                              B[j*16*block_tiles+(w//(num_thread)%block_row_warp)*16*warp_row_tile,0]+\
                                              C[i*16*block_tiles+(w//num_thread//block_row_warp)*16*warp_col_tile,j*16*block_tiles+(w//num_thread%block_row_warp)*16*warp_row_tile].astype('float16')))
        s = schedule_gemm_fp16()
        print(tvm.lower(s,[A,B,C],name="matrix_dot",simple_mode=True))

        f = tvm.build(s, [A,B,C], target='cuda', name='gemm_fp16')
        print("build finished")

        ctx = tvm.context('cuda', 4)

        a_np = np.float16(np.random.uniform(0.,1.,size=(rA,cA)))
        b_np = np.float16(np.random.uniform(0.,1.,size=(rB,cB)))
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((rA, rB), dtype=C.dtype), ctx)
        #ci = tvm.nd.array(np.ones((rA//16, rB//16,16,16), dtype=C.dtype), ctx)
        f(a, b,c)
        ss = c.asnumpy()


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



