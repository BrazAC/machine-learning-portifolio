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
    #y=tanh(x1w1+x2w2+b)
    x1 = Value(2, 'x1')
    x2 = Value(3, 'x2')
    w1 = Value(0.5, 'w1')
    w2 = Value(0.2, 'w2')
    b = Value(1, 'b')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    k = x1w1 + x2w2; k.label = 'x1w1 + x2w2'
    j = k + b; j.label = 'x1w1 + x2w2 + b'
    y = j.tanh(); y.label = 'y'
    print(f"Antes: \t{y.data}")

    # Backpropagation manual
    y.grad = 1
    j.grad = 0.02182
    b.grad = 0.02182
    k.grad = 0.02182
    x2w2.grad = 0.02182
    x1w1.grad = 0.02182
    w2.grad = 0.065454
    x2.grad = 0.00436
    w1.grad = 0.0436409
    x1.grad = 0.010911

    dot = draw_dot(y)
    dot.render()

    # Otimização
    u = 0.01
    w1.data = w1.data + u * w1.grad
    w2.data = w2.data + u * w2.grad
    b.data  = b.data + u *  b.grad

    # Foward pass pós otimização
    x1 = Value(2, 'x1')
    x2 = Value(3, 'x2')
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    k = x1w1 + x2w2; k.label = 'x1w1 + x2w2'
    j = k + b; j.label = 'x1w1 + x2w2 + b'
    y = j.tanh(); y.label = 'y'
    print(f"Depoi: \t{y.data}")

    dot1 = draw_dot(y)
    dot1.render()


#     # Derivando uma função de múltiplas variáveis
#     # f(a,b,c) = a+b*c+c
#     a = Value(3, 'a')
#     b = Value(4, 'b')
#     c = Value(-2, 'c')
#     d = b * c
#     e = a + d
#     y = e + c
#
#     #dy/dc = [f(a,b,c+h) - f(a,b,c)]/h
#     h = 0.0001
#
#     a = Value(3+h, 'a')
#     b = Value(4, 'b')
#     c = Value(-2, 'c')
#     d = b * c
#     e = a + d
#     fxh = e + c
#     print(fxh)
#
#     a = Value(3, 'a')
#     b = Value(4, 'b')
#     c = Value(-2, 'c')
#     d = b * c
#     e = a + d
#     fx = e + c
#     print(fx)
#
#     grad  = (fxh.value-fx.value)/h
#     print(f"dy/dc: {grad}")
