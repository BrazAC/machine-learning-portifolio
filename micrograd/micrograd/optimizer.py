class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)

    def zero_grad(self):
        for param in self.parameters: param.grad = 0

    def step(self):
        for param in self.parameters: param.data -= self.lr * param.grad

    def __repr__(self):
        print(f"<Optimzer SGD | lr = {self.lr}>")
