import math
from collections import deque

class Value:
    def __init__(self, data, label, oper=None, prev=()):
        self.data = data
        self.label = label
        self._prev = set(prev)
        self._op = oper
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"<Value: {self.data}>"

    def __add__(self, other):
        if not isinstance(other, Value): other = Value(other, '')
        n = self.data + other.data
        o = Value(n, '', '+',(self, other))
        def _backward():
            self.grad += 1 * o.grad
            other.grad += 1 * o.grad
        o._backward = _backward

        return o

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if not isinstance(other, Value): other = Value(other,'')
        n = self.data * other.data
        o = Value(n, '', '*', (self, other))
        def _backward():
            self.grad += other.data * o.grad
            other.grad += self.data * o.grad
        o._backward = _backward
        return o

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)): raise TypeError("Second argument must be a int or float value")
        n = self.data ** other
        o = Value(n, '', f"**{other}", (self,))
        def _backward():
            self.grad += (other * math.pow(self.data, other - 1)) * o.grad
        o._backward = _backward
        return o

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        n = math.exp(self.data)
        o = Value(n, '', f"exp", (self,))
        def _backward():
            self.grad += o.data * o.grad
            print(o.data, o.grad, self.grad, self.label)
        o._backward = _backward
        return o

    def tanh(self):
        n = (math.exp(self.data) - math.exp(-self.data))/(math.exp(self.data) + math.exp(-self.data))
        o = Value(n, '', "tanh", (self,))
        def _backward():
            self.grad += (1 - math.tanh(self.data)**2) * o.grad
        o._backward = _backward
        return o

    def relu(self):
        n = 0 if self.data < 0 else self.data
        o = Value(n, '', "relu", (self,))
        def _backward():
            d = 0 if o.data < 0 else 1
            self.grad += d * o.grad
        o._backward = _backward
        return o

    def backward(self):
        # Topological sort
        sorted = []
        noDependcs = deque([self])
        while len(noDependcs) != 0:
            n = noDependcs.popleft()
            if n not in sorted:
                sorted.append(n)
                for prev in n._prev:
                    noDependcs.append(prev)

        # All nodes _backward
        self.grad = 1
        for value in sorted:
            value._backward()
