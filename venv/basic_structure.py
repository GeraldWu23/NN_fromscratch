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
    def __init__(self, name=None, istrainable=True):
        Node.__init__(self, name=name, istrainable=istrainable)

    def forward(self, value=None):
        # value will be assigned here again after init
        if value:
            self.value = value

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


def feed_dict_2_graph(feed_dict):
    nodes = [node for node in feed_dict]  # Placeholder nodes
    graph = defaultdict(list)

    while nodes:
        node = nodes.pop(0)
        if node in graph:
            continue  # avoid dead loop

        if isinstance(node, Placeholder):
            node.value = feed_dict[node]

        for next_node in node.output:
            graph[node].append(next_node)
            if next_node not in nodes:
                nodes.append(next_node)

    return graph


def topo_sorting(graph):
    order = []
    outfrom = set([node for node in graph])
    into = set()
    for node in outfrom:
        # print(graph[node])
        into = into.union(set(graph[node]))

    outfrom = outfrom.union(into - outfrom)  # nodes with no output

    while outfrom:
        no_in_node = outfrom - into
        order += no_in_node
        outfrom -= no_in_node  # generate next gen outfrom

        # generate next gen into
        into = set()
        for node in outfrom:
            left_out = set([node for node in graph[node] if node not in order])
            into = into.union(left_out)

    return order


def optimize(graph, lr=1e-2):
    for node in filter(lambda n: n.istrainable, graph):
        node.value += (-1) * node.gradient[node] * lr


def no_grad(graph):
    for node in graph:
        node.gradient = defaultdict(np.float64)


class NN:
    def __init__(self, input_):
        self.input = input_
        self.x = Placeholder(name='x', istrainable=False)
        self.y = Placeholder(name='y', istrainable=False)
        self.w = Placeholder(name='w')
        self.b = Placeholder(name='b')
        self.linear = Linear(self.x, self.w, self.b, name='linear')
        self.sigmoid = Sigmoid([self.linear], name='sigmoid')
        self.loss = MSELoss(self.sigmoid, self.y, name='MSE loss')

        feed_input = {self.x: self.input['x'],
                      self.y: self.input['y'],
                      self.w: self.input['w'],
                      self.b: self.input['b']}

        self.graph = feed_dict_2_graph(feed_input)
        self.order = topo_sorting(self.graph)

    def forward(self):
        for node in self.order:
            node.forward()

    def backward(self):
        for node in self.order[::-1]:
            node.backward()

    def run_one_epoch(self):
        self.forward()
        self.backward()

    def optimize(self, lr=1e-2):
        for node in filter(lambda n: n.istrainable, self.graph):
            node.value += (-1) * node.gradient[node] * lr


if __name__ == "__main__":
    x_, y_, w0_, b0_ = np.random.normal(size=(1, 4)).squeeze()
    input_ = {
        'x': x_,
        'y': y_,
        'w': w0_,
        'b': b0_
    }

    model = NN(input_)

    print(input_)
    model.forward()
    print([f'{node.name}: {node.value}' for node in model.graph])
    model.backward()
    model.optimize(lr=1e-1)
    print()
    print([f'{node.name}: {node.value}' for node in model.graph])
