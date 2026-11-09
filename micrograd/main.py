from micrograd import Value
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

def sorvete():
    h = 0.0001

    #y=tanh(x1w1+x2w2+b)
    x1 = Value(2 + h, 'x1')
    x2 = Value(3, 'x2')
    w1 = Value(0.5, 'w1')
    w2 = Value(0.2, 'w2')
    b = Value(1, 'b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    k = x1w1 + x2w2; k.label = 'x1w1 + x2w2'
    j = k + b; j.label = 'x1w1 + x2w2 + b'
    fxh = j.tanh(); y.label = 'y'
    fxh = fxh.data

    x1 = Value(2, 'x1')
    x2 = Value(3, 'x2')
    w1 = Value(0.5, 'w1')
    w2 = Value(0.2, 'w2')
    b = Value(1, 'b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    k = x1w1 + x2w2; k.label = 'x1w1 + x2w2'
    j = k + b; j.label = 'x1w1 + x2w2 + b'
    fx = j.tanh(); y.label = 'y'
    fx = fx.data

    deriv = (fxh - fx)/h
    print(deriv)

    dot = draw_dot(y)
    dot.render()

if __name__ == "__main__":
    # Foward pass em um neurônio
    # y=tanh(x1w1+x2w2+b)
    x1 = Value(2, 'x1')
    x2 = Value(0, 'x2')
    w1 = Value(-3, 'w1')
    w2 = Value(1, 'w2')
    b = Value(6.8813735870195432, 'b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    k = x1w1 + x2w2; k.label = 'x1w1 + x2w2'
    j = k + b; j.label = 'x1w1 + x2w2 + b'
    l = 2*j; l.label='2x'
    t = l.exp(); t.label = 'e'
    e1 = t - 1; e1.label = 'e1'
    e2 = t + 1; e2.label = 'e2'
    y = e1 / e2; y.label = 'y'
    #print(f"Antes: \t{y.data}")

    y.backward()
    # Backpropagation manual
    # y.grad = 1
    # j.grad = 0.02182
    # b.grad = 0.02182
    # k.grad = 0.02182
    # x2w2.grad = 0.02182
    # x1w1.grad = 0.02182
    # w2.grad = 0.065454
    # x2.grad = 0.00436
    # w1.grad = 0.0436409
    # x1.grad = 0.010911

    dot1 = draw_dot(y)
    dot1.render()
