import numpy as np
from collections import defaultdict


class Node:
    def __init__(self, input_=[], name=None, istrainable=False):
        self.name = name
        self.input = input_
        self.output = []
        self.gradient = defaultdict(np.float64)
        self.istrainable = istrainable
        self.value = None

        for input_node in self.input:
            if input_node:
                input_node.output.append(self)

    def __repr__(self):
        return self.name

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Placeholder(Node):
    def __init__(self, x=None, name=None, istrainable=True):
        Node.__init__(self, [x], name=name, istrainable=istrainable)
        self.x = None

    def forward(self):
        # value will be assigned here again after init
        self.x = self.input[0]
        self.value = self.input[0]

    def backward(self):
        for node in self.output:
            cost = node.gradient[self]
            self.gradient[self] += cost * 1


class Linear(Node):
    def __init__(self, x, w, b, name=None, istrainable=False):
        Node.__init__(self, input_=[x, w, b], name=name, istrainable=istrainable)
        self.x, self.w, self.b = self.input

    def forward(self):
        self.value = self.x.value * self.w.value + self.b.value

    def backward(self):
        for node in self.output:
            cost = node.gradient[self]
            self.gradient[self.x] += cost * self.w.value
            self.gradient[self.w] += cost * self.x.value
            self.gradient[self.b] += cost * 1


class Sigmoid(Node):
    def __init__(self, input_=[], name=None, istrainable=False):
        Node.__init__(self, input_=input_, name=name, istrainable=istrainable)
        self.x = self.input[0]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.x.value)

    def patial_derivative(self):
        return self._sigmoid(self.x.value) * (1 - self._sigmoid(self.x.value))

    def backward(self):
        for node in self.output:
            cost = node.gradient[self]
            self.gradient[self.x] += cost * self.patial_derivative()


class MSELoss(Node):
    def __init__(self, y_, y, name=None, istrainable=False):
        Node.__init__(self, input_=[y_, y], name=name, istrainable=istrainable)
        self.y_ = self.input[0]
        self.y = self.input[1]
        self.y_val = None
        self.yval = None

    def forward(self):
        self.y_val = np.array(self.y_.value)  # prediction
        self.yval = np.array(self.y.value)
        self.value = np.mean((self.y_val - self.yval) ** 2)

    def backward(self):
        self.gradient[self.y_] = 2 * (self.y_val - self.yval)
        self.gradient[self.y] = 2 * (self.yval - self.y_val)


if __name__ == "__main__":
    pass