from engine import Value
import nn
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label = "{%s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

if __name__ == "__main__":
    # Mini dataset
    x = [[-0.45,-1.0,0.0],
        [1.3,2.9,0.3,1],
        [0.6,-0.9,0.9]]
    y_hat = [1.0, 0.0, 1.0]

    # Instantiate model
    mlp = nn.MLP(3, [5,5,1])

    # Training loop
    y = []
    epochs = 2000
    for _ in range(epochs):
        # Zero-grad
        for par in mlp.parameters():
            par.grad = 0

        # Foward
        y = [mlp(xs) for xs in x]

        # Loss
        L = sum([(y_p - y_h)**2 for y_p, y_h  in zip(y, y_hat)])/len(y_hat)
        print(L)
        # Backward
        L.backward()

        # Optimizer SGD
        lr = 0.01
        for par in mlp.parameters():
            par.data -= lr * par.grad

    print([value.data for value in y])
    print(y_hat)

#    dot1 = draw_dot(y)
#    dot1.render()
