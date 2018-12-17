import tensorrt as trt 
import numpy as np
import mxnet as mx
import pycuda.driver as cuda
import pycuda.autoinit
import time

N = 1
C = 64
H = 256
W = 256
K = 128
R = 3
S = 3
ph=1
pw=1
u=4
v=4
P = int(np.ceil(float(H-R+1+2*ph)/float(u)))
Q = int(np.ceil(float(W-S+1+2*pw)/float(v)))

max_batch_size = N
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
hdata = np.float16(np.random.uniform(0.,1.,size=(N,C,H,W)))
filter_layer = np.float16(np.random.uniform(0.,1.,size=(K,C,R,S)))
bias_layer = np.zeros((K,), dtype=np.float16)
hout = np.zeros((N,K,P,Q), dtype=np.float32)

#create a single conv layer using fp16
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
    builder.fp16_mode = True
    input_tensor = network.add_input(name = "D",dtype = trt.float16,shape = (C,H,W))
    trtweight = trt.Weights(filter_layer)
    trtblas = trt.Weights()
    conv1 = network.add_convolution(input = input_tensor,num_output_maps = K, kernel_shape = (R,S),kernel = trtweight, bias=trtblas)

    conv1.stride = (u,v)
    conv1.padding = (ph,pw)

    network.mark_output(conv1.get_output(0))
    


    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = 1<<20
    with builder.build_cuda_engine(network) as engine:
        #hdata = cuda.pagelocked_empty((N,C,H,W),dtype=np.float16)
        #hout = cuda.pagelocked_empty((N,K,P,Q),dtype=np.float16)
        ddata = cuda.mem_alloc(hdata.nbytes)
        dout = cuda.mem_alloc(hout.nbytes)

        stream = cuda.Stream()
        num_runs =10
        with engine.create_execution_context() as context:
            #np.copyto(hdata,data)

            cuda.memcpy_htod_async(ddata,hdata,stream)
            stream.synchronize()
            print("now caculate")
            #run the inference
            t1=time.time()
            for iter_id in range(num_runs):
                context.execute_async(bindings = [int(ddata),int(dout)],stream_handle=stream.handle)
                stream.synchronize()
            dt = time.time()-t1
            print("now copy back")
            #copy back
            cuda.memcpy_dtoh_async(hout,dout,stream)

            stream.synchronize()
            print("now copy to out")
            #np.copyto(out,hout)
            num_flops = P*Q*2*K*N*C*R*S
            t= dt/num_runs
            TFLOPS = num_flops / (t * 1e3) / 1e9
            print("average time cost of %d runs = %g ms, %g TFLOPS." %(num_runs, t * 1e3, TFLOPS))

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

amx = mx.nd.array(hdata)
bmx = mx.nd.array(filter_layer)

Conv = mx.symbol.Convolution
result2 = single_dev_consist(amx,bmx)
          
#np.testing.assert_allclose(hout,result2,rtol=1e-2)
#print("verify accuracy success")

