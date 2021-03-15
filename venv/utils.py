from basic_structure import Placeholder, Linear, Sigmoid, MSELoss
import numpy as np
from collections import defaultdict


def feed_dict_2_graph(feed_dict):
    nodes = [node for node in feed_dict]  # Placeholder nodes
    graph = defaultdict(list)

    while nodes:
        node = nodes.pop(0)
        if node in graph:
            continue  # avoid dead loop

        if isinstance(node, Placeholder):
            node.input = [feed_dict[node]]

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


def optimize(graph, learning_rate):
    for node in graph:
        if node.istrainable:
            node.value += (-1) * learning_rate * node.gradient[node]


if __name__ == "__main__":
    x_, y_, w0_, b0_, w1_, b1_ = np.random.normal(size=(1, 6)).squeeze()
    x = Placeholder(name='x', istrainable=False)
    y = Placeholder(name='y', istrainable=False)
    w0 = Placeholder(name='w0')
    b0 = Placeholder(name='b0')

    feed_input = {x: x_,
                  y: y_,
                  w0: w0_,
                  b0: b0_,
                  }

    linear_out = Linear(x, w0, b0, name='linear')
    sigmoid_out = Sigmoid([linear_out], name='sigmoid')
    loss = MSELoss(sigmoid_out, y, name='loss')

    graph_ = feed_dict_2_graph(feed_input)
    order_ = topo_sorting(graph_)
    print(order_)

    print("forward:")
    for node in order_:
        node.forward()
        print(f"{node.name}: {node.value}")

    print("\nbackward:")
    for node in order_[::-1]:
        node.backward()
        print(f"{node.name}: {node.gradient}")

    optimize(graph_, learning_rate=1e-2)

    print("\noptimize:")
    for node in graph_:
        print(f"{node.name}: {node.value}")



