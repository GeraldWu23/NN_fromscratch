from collections import defaultdict


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
        None.__init__(self, name=name, istrainable=istrainable)

    def forward(self, x):
        self.value = x

    def backward(self):
        pass


class Linear(Node):
    def __init__(self, input_=[], name=None, istrainable=True):
        None.__init__(self, input_=input_, name=name, istrainable=istrainable)

    def forward(self):
        pass

    def backward(self):
        pass


class Sigmoid(Node):
    def __init__(self, input_=[], name=None, istrainable=True):
        None.__init__(self, input_=input_, name=name, istrainable=istrainable)

    def forward(self):
        pass

    def backward(self):
        pass


class MSELoss(Node):
    def __init__(self, input_=[], name=None, istrainable=True):
        None.__init__(self, input_=input_, name=name, istrainable=istrainable)

    def forward(self):
        pass

    def backward(self):
        pass


if __name__ == "__main__":
    pass