import math
class Value:
    def __init__(self, data, label, oper=None, prev=()):
        self.data = data
        self.label = label
        self._prev = set(prev)
        self._op = oper
        self.grad = 0

    def __repr__(self):
        return f"<Value object: label {self.label} | value {self.data} | oper '{self._op}'>"

    def __add__(self, other):
        n = self.data + other.data
        return Value(n, '', '+',(self, other))

    def __mul__(self, other):
        n = self.data * other.data
        return Value(n, '', '*', (self, other))

    def tanh(self):
        n = (math.exp(self.data) - math.exp(-self.data))/(math.exp(self.data) + math.exp(-self.data))
        return Value(n, '', "tanh", (self,))
