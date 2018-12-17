import mxnet as mx
import numpy as np

Conv = mx.symbol.Convolution
w = np.random.normal(size=(64,64,3,3))
w = mx.nd.array(w, ctx=mx.cpu())
def single_dev_consist():
    data = mx.sym.var("data")
    weight = mx.sym.var("conv_weight")
    
    # bias = None, pad = (1, 1), stride = (1, 1)
    conv = Conv(name='conv', data=data, weight=weight, num_filter=64, pad=(1, 1), kernel=(3, 3))
    conv_exe = conv.simple_bind(mx.cpu(), data=(1,64,64,128), conv_weight=(64,64,3,3))

    d = mx.nd.array(np.random.normal(size=(1,64,64,128)), ctx=mx.cpu())
    conv_exe.forward(is_train=False, data=d, conv_weight=w)
    output = conv_exe.outputs[0].asnumpy()
    print('output',output.shape)
 

if __name__ == "__main__":
    single_dev_consist()
