import numpy as np


class Node:
    def __init__(self, input_=[], name=None, istrainable=True):
        self.name = name
        self.input = input_
        self.output = []
        self.gradience = dict()
        self.istrainable = istrainable
        self.value = None

        for input_node in self.input:
            input_node.output.append(self)

    def __repr__(self):
        return self.name

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Placeholder(Node):
    def __init__(self, name=None, istrainable=True):
        Node.__init__(self, name=name, istrainable=istrainable)

    def forward(self, x):
        self.value = x

    def backward(self):
        pass


class Linear(Node):
    def __init__(self, x, w, b, name=None, istrainable=True):
        Node.__init__(self, input_=[x, w, b], name=name, istrainable=istrainable)
        self.x, self.w, self.b = self.input

    def forward(self):
        self.value = self.x.value * self.w.value + self.b.value

    def backward(self):
        pass


class Sigmoid(Node):
    def __init__(self, input_=[], name=None, istrainable=True):
        Node.__init__(self, input_=input_, name=name, istrainable=istrainable)

    def forward(self):
        self.value = 1/(1 + np.exp(-self.input[0].value))

    def backward(self):
        pass


class MSELoss(Node):
    def __init__(self, y_, y, name=None, istrainable=True):
        Node.__init__(self, input_=[y_, y], name=name, istrainable=istrainable)
        self.y_ = self.input[0]  # prediction
        self.y = self.input[1]

    def forward(self):
        self.value = ((self.y_.value-self.y)**2)/2

    def backward(self):
        pass


if __name__ == "__main__":
    pass