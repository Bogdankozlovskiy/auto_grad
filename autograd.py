__all__ = ("TensorData", )


import numpy as np
from collections import defaultdict
from skimage.measure import block_reduce


class Tensor:
    def __init__(self, data=None, parents=None):
        self.data = data
        self.gradient = None
        self.parents = parents
        self.children = defaultdict(int)
        self.flag = False
    
    def forward(self):
        if self.flag:
            return self.result
        self.flag = True
        if self.parents is not None:
            for p in self.parents:
                p.children[self] += 1
            self.left_data = self.parents[0].forward()
    
    def bprop(self, gradient=None, child=None):
        self.flag = False
        if (self.gradient is None) or self.parents is not None:
            self.gradient = np.zeros_like(self.result, dtype=np.float64)
        if gradient is None:
            self.gradient += 1
        else:
            self.gradient += gradient
        if child is not None:
            self.children[child] -= 1
    
    def dot(self, other):
        return TensorDot(parents=[self, other])
    
    def sum(self, axis):
        return TensorSum(axis=axis, parents=[self])
    
    def drop_out(self):
        return TensorDropOut(parents=[self])
    
    def batch_normalize(self):
        return TensorBatchNormalize(parents=[self])
    
    def tanh(self):
        return TensorTanh(parents=[self])
    
    def sigmoid(self):
        return TensorSigmoid(parents=[self])
    
    def relu(self):
        return TensorRelu(parents=[self])
    
    def cross_entropy(self, target, num_classes=None):
        return TensorCrossEntropy(target=target, num_classes=num_classes, parents=[self])

    def binary_cross_entropy(self, target):
        return TensorBinaryCrossEntropy(target=target, parents=[self])

    def in_conv(self, rows, cols):
        return TensorInConv(rows=rows, cols=cols, parents=[self])
    
    def out_conv(self):
        return TensorOutConv(parents=[self])
    
    def flatten(self):
        return TensorFlatten(parents=[self])
    
    def max_poling(self, rows, cols):
        return TensorMaxPoling(rows=rows, cols=cols, parents=[self])

    @property
    def T(self):
        return TensorTranspouse(parents=[self])

    def to_iter(self, batch_size=1):
        return TensorIter(batch_size=batch_size, parents=[self])
    
    def rnn(self, wih, whh, who):
        return TensorRNN(wih=wih, whh=whh, who=who, parents=[self])
    
    def lstm(self, wxf, wxi, wxo, wxc, whf, whi, who, whc, wh2o):
        return TensorLSTM(wxf=wxf, wxi=wxi, wxo=wxo, wxc=wxc, whf=whf, whi=whi, who=who, whc=whc, wh2o=wh2o, parents=[self])
    
    def __add__(self, other):
        return TensorAdd(parents=[self, other])
    
    def __sub__(self, other):
        return TensorSub(parents=[self, other])
    
    def __mul__(self, other):
        if issubclass(other.__class__, Tensor):
            return TensorMul(parents=[self, other])
        elif isinstance(other, int) or isinstance(other, float):
            return TensorMulInt(digit=other, parents=[self])
    
    def __pow__(self, power):
        return TensorPow(power=power, parents=[self])
    
    def __lt__(self, other):
        '''for SVM mashine'''
        return TensorLess(digit=other, parents=[self])

    def __getitem__(self, other):
        return TensorGetItem(parents=[self, other])

    def __abs__(self):
        return TensorAbs(parents=[self])


class TensorDot(Tensor):
    def forward(self):
        super().forward()
        data_left = self.left_data
        data_right = self.parents[1].forward()
        dims = data_left.ndim if data_left.ndim < data_right.ndim else data_right.ndim
        dims -= 1 
        axis_l = range(-dims, 0)
        axis_r = range(dims)
        self.result = np.tensordot(data_left, data_right, axes=[axis_l, axis_r])
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            right = self.parents[1].result
            data = np.tensordot(self.gradient, right, axes=[(-1, ), (-1, )])
            self.parents[0].bprop(data, child=self)
            left = self.parents[0].result
            axes = tuple(range(self.gradient.ndim - 1))
            data = np.tensordot(left, self.gradient, axes=[axes, axes])
            self.parents[1].bprop(data, child=self)


class TensorSum(Tensor):
    def __init__(self, axis, data=None, parents=None):
        super().__init__(data, parents)
        self.axis = axis
    
    def forward(self):
        super().forward()
        self.expand = self.left_data.shape[self.axis]
        self.result = self.left_data.sum(self.axis)
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            new_shape = list(self.result.shape)
            new_shape.insert(self.axis, self.expand)
            new_grad = self.gradient.repeat(self.expand).reshape(new_shape)
            self.parents[0].bprop(new_grad, child=self)


class TensorDropOut(Tensor):
    def forward(self):
        super().forward()
        self.mask = np.random.randint(0, 2, size=self.left_data.shape)
        self.result = self.mask * self.left_data * 2
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.mask, child=self)


class TensorBatchNormalize(Tensor):
    def forward(self):
        super().forward()
        self.left_data -= self.left_data.mean(axis=1, keepdims=True)
        self.left_data /= self.left_data.std(axis=1, keepdims=True)
        self.result = self.left_data
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient, child=self)


class TensorTanh(Tensor):
    def forward(self):
        super().forward()
        self.result = np.tanh(self.left_data)
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * (1 - self.result ** 2), child=self)


class TensorSigmoid(Tensor):
    def forward(self):
        super().forward()
        self.result = 1 / (1 + np.exp(-self.left_data))
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.result * (1 - self.result) * self.gradient, child=self)


class TensorRelu(Tensor):
    def forward(self):
        super().forward()
        self.mask = self.left_data > 0
        self.result = self.mask * self.left_data
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.mask, child=self)


class TensorCrossEntropy(Tensor):
    def __init__(self, target, num_classes, data=None, parents=None):
        super().__init__(data, parents)
        numpy_fun = lambda : target
        tensor_fun = lambda : target.forward()
        self.target = tensor_fun if isinstance(target, Tensor) else numpy_fun
        self.num_classes = num_classes
    
    def forward(self):
        super().forward()
        tmp = np.exp(self.left_data - np.max(self.left_data, keepdims=True, axis=self.left_data.ndim - 1))
        self.softmax = tmp / np.sum(tmp, keepdims=True, axis=self.left_data.ndim - 1)
        self.decoded_target = np.eye(self.num_classes)[self.target()]
        loss = (-np.log(self.softmax) * self.decoded_target).sum()
        self.result = loss
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop((self.softmax - self.decoded_target) * self.gradient, child=self)


class TensorBinaryCrossEntropy(Tensor):
    def __init__(self, target, data=None, parents=None):
        super().__init__(data, parents)
        self.target = target

    def forward(self):
        super().forward()
        result = - self.target * np.log(self.left_data) \
                - (1 - self.target) * np.log(1 - self.left_data)
        self.result = result.sum()
        return self.result

    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop((self.left_data - self.target) * self.gradient, child=self)


class TensorAdd(Tensor):
    def forward(self):
        super().forward()
        self.result = self.left_data + self.parents[1].forward()
        return self.result

    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient, child=self)
            self.parents[1].bprop(self.gradient, child=self)


class TensorSub(Tensor):
    def forward(self):
        super().forward()
        self.result = self.left_data - self.parents[1].forward()
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient, child=self)
            self.parents[1].bprop(-self.gradient, child=self)


class TensorMul(Tensor):
    def forward(self):
        super().forward()
        self.result = self.left_data * self.parents[1].forward()
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.parents[1].result, child=self)
            self.parents[1].bprop(self.gradient * self.parents[0].result, child=self)


class TensorMulInt(Tensor):
    def __init__(self, digit, data=None, parents=None):
        super().__init__(data, parents)
        self.digit = digit
        
    def forward(self):
        super().forward()
        self.result = self.left_data * self.digit
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.digit, child=self)


class TensorPow(Tensor):
    def __init__(self, power, data=None, parents=None):
        super().__init__(data, parents)
        self.power = power
    
    def forward(self):
        super().forward()
        self.result = self.left_data ** self.power
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.power * self.gradient * self.parents[0].result, child=self)


class TensorAbs(Tensor):
    def forward(self):
        super().forward()
        self.result = abs(self.left_data)
        self.new_grad = (self.left_data > 0).astype("i") - (self.left_data < 0).astype("i")
        return self.result

    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.new_grad, child=self)


class TensorInConv(Tensor):
    def __init__(self, rows, cols, data=None, parents=None):
        super().__init__(data, parents)
        self.rows = rows
        self.cols = cols
    
    def forward(self):
        super().forward()
        self.result = self.chop_imgs(self.left_data, self.rows, self.cols)
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            result = self.parents[0].result
            new_grad = self.chop_imgs_bprop(result)
            self.parents[0].bprop(new_grad, child=self)

    def chop_imgs(self, data, rows, cols):
        len_vector = rows * cols
        len_data = data.shape[0]
        chanels = data.shape[-1]
        self.rows_data = data.shape[1] - rows + 1
        self.cols_data = data.shape[2] - cols + 1
        conv = (data[:, i:i + rows, j:j + cols].reshape((len_data, 1, len_vector, chanels))\
                                for i in range(self.rows_data) for j in range(self.cols_data))
        return np.concatenate(tuple(conv), axis=1)
    
    def chop_imgs_bprop(self, data):
        grad = np.zeros_like(data)
        k = 0
        for i in range(self.rows_data):
            for j in range(self.cols_data):
                grad[:, i:i + self.rows, j:j + self.cols] += \
                self.gradient[:, k].reshape((data.shape[0], self.rows, self.cols, data.shape[-1]))
                k += 1
        return grad


class TensorOutConv(Tensor):
    def forward(self):
        super().forward()
        grand2 = self.parents[0].parents[0].parents[0]
        grand1 = self.parents[0].parents[0]
        rows = grand2.result.shape[1] - grand1.rows + 1
        cols = grand2.result.shape[2] - grand1.cols + 1
        self.result = self.left_data.reshape((self.left_data.shape[0], rows, cols, self.left_data.shape[-1]))
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient.reshape(self.parents[0].result.shape), child=self)


class TensorFlatten(Tensor):
    def forward(self):
        super().forward()
        self.shape = self.left_data.shape
        self.result = self.left_data.reshape((self.left_data.shape[0], -1))
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient.reshape(self.shape), child=self)


class TensorMaxPoling(Tensor):
    def __init__(self, rows, cols, data=None, parents=None):
        super().__init__(data, parents)
        self.rows = rows
        self.cols = cols
    
    def forward(self):
        super().forward()
        self.result = block_reduce(self.left_data, (1, self.rows, self.cols, 1), np.max)
        poling_extend = np.repeat(np.repeat(self.result, self.rows, axis=1), self.cols, axis=2)
        self.mask = np.equal(self.left_data, poling_extend)
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            data = np.repeat(np.repeat(self.gradient, self.rows, axis=1), self.cols, axis=2) * self.mask
            self.parents[0].bprop(data, child=self)


class TensorData(Tensor):
    def forward(self):
        super().forward()
        self.result = self.data
        return self.result


class TensorTranspouse(Tensor): 
    def forward(self):
        super().forward()
        self.result = self.left_data.T
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient.T, child=self)


class TensorLess(Tensor):
    def __init__(self, digit, data=None, parents=None):
        super().__init__(data, parents)
        self.digit = digit
    
    def forward(self):
        super().forward()
        self.mask = self.left_data < self.digit
        self.result = self.left_data * self.mask
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.parents[0].bprop(self.gradient * self.mask, child=self)


class TensorGetItem(Tensor):
    def forward(self):
        super().forward()
        self.right_data = self.parents[1].forward()
        self.result = self.left_data[self.right_data]
        return self.result

    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            gradient = np.zeros_like(self.left_data)
            gradient[self.right_data] += self.gradient
            self.parents[0].bprop(gradient, child=self)


class TensorIter(Tensor):
    def __init__(self, batch_size, data=None, parents=None):
        super().__init__(data, parents)
        self.data = self.parents[0].data
        self.length = (self.data.shape[0] // batch_size) + 1
        self.batch_size = batch_size
        self.step = 0

    def forward(self):
        super().forward()
        self.result = self.data[self.step * self.batch_size:(self.step + 1) * self.batch_size]
        self.step += 1
        self.step %= self.length
        return self.result


class TensorRNN(Tensor):
    def __init__(self, wih, whh, who, data=None, parents=None):
        super().__init__(data, parents)
        self.wih = wih
        self.whh = whh
        self.who = who
        
    def forward(self):
        '''we have to use gradient parting in optimizer. because sigmoid in the forward'''
        super().forward()
        self.arr = []
        self.hidden = TensorData(np.zeros((self.left_data.shape[0], self.whh.data.shape[0])))
        for i in range(self.left_data.shape[1]):
            X = TensorData(self.left_data[:,i])
            self.arr.append(X)
            self.hidden = (X.dot(self.wih) + self.hidden.dot(self.whh)).sigmoid()
        self.tail = self.hidden.dot(self.who)
        self.result = self.tail.forward()
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.tail.bprop(self.gradient)
            length = self.left_data.shape[1]
            grad = np.concatenate([self.arr[i].gradient[:, np.newaxis] for i in range(length)], axis=1)
            self.parents[0].bprop(grad, child=self)


class TensorLSTM(Tensor):
    def __init__(self, wxf, wxi, wxo, wxc, whf, whi, who, whc, wh2o, data=None, parents=None):
        super().__init__(data, parents)
        self.wxf = wxf # input  forgot gate (inputs, hidden)
        self.wxi = wxi # input  input  gate (inputs, hidden)
        self.wxo = wxo # output input  gate (inputs, hidden)
        self.wxc = wxc # cell   input  gate (inputs, hidden)
        self.whf = whf # hidden forgot gate (hidden, hidden)
        self.whi = whi # input  hidden gate (hidden, hidden)
        self.who = who # output hidden gate (hidden, hidden)
        self.whc = whc # cell   hidden gate (hidden, hidden)
        self.wh2o = wh2o # hidden to output (hidden, output)
    
    def forward(self):
        '''we have to use gradient parting in optimizer. because sigmoid in the forward'''
        super().forward()
        self.arr = []
        hidden = TensorData(np.zeros((self.left_data.shape[0], self.whf.data.shape[0])))
        cell = TensorData(np.zeros((self.left_data.shape[0], self.whc.data.shape[0])))
        for i in range(self.left_data.shape[1]):
            X = TensorData(self.left_data[:, i])
            self.arr.append(X)
            F = (X.dot(self.wxf) + hidden.dot(self.whf)).sigmoid()
            I = (X.dot(self.wxi) + hidden.dot(self.whi)).sigmoid()
            O = (X.dot(self.wxo) + hidden.dot(self.who)).sigmoid()
            G = (X.dot(self.wxc) + hidden.dot(self.whc)).tanh()
            cell = (F * cell) + (I * G)
            hidden = O * cell.tanh()
        self.tail = hidden.dot(self.wh2o)
        self.result = self.tail.forward()
        return self.result
    
    def bprop(self, gradient=None, child=None):
        super().bprop(gradient=gradient, child=child)
        if not sum(self.children.values()):
            self.tail.bprop(self.gradient)
            length = self.left_data.shape[1]
            grad = np.concatenate([self.arr[i].gradient[:, np.newaxis] for i in range(length)], axis=1)
            self.parents[0].bprop(grad, child=self)
