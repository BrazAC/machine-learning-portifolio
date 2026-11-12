import random
from engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1), f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1,1), 'b')
        self.nonlin = nonlin

    def __call__(self, x):
        a = sum([w*x for w, x in zip(self.w, x)], self.b)
        return a.relu() if self.nonlin else a

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, non):
        self.neurons = [Neuron(nin) for _ in range(non)]

    def __call__(self, x):
        res = [n(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res

    def parameters(self):
        params = []
        for neuron in self.neurons:
           params += neuron.parameters()
        return params


class MLP:
    def __init__(self, nin, nouts):
        dim = [nin] + nouts
        self.layers  = [Layer(dim[i], dim[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
