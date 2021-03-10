from collection import defaultdict


class Node:
    def __init__(self, name=None, value=None, istrainable=True):
        self.name = name
        self.value = value
        self.input = []
        self.output = []
        self.gradience = dict()
        self.istrainable = istrainable

    def __repr__(self):
        return self.name

    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Placeholder(Node):
    def __init__(self, name=None, value=None, istrainable=False):
        None.__init__(self, name=name, value=value, istrainable=istrainable)

    def forward(self):
        pass

    def backward(self):
        pass
