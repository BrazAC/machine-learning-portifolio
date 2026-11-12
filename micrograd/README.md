## Sobre o projeto
Micrograd é um pequeno motor de cálculo de gradientes de redes neurais simples. Opera sobre grafos direcionados acíclicos (DAG) construídos dinamicamente. Sua construção é inspirada no método backwards do PyTorch, visto que realiza a mesma tarefa de retropropagação. Porém, operando apenas dados escalares e não tensores.  
Micrograd é de autoria de Andrej Karpathy. Seu repositório do projeto original está disponível [aqui](https://github.com/karpathy/micrograd) e ele criou o vídeo [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1926s) com uma explicação detalhada de sua construção. 
Implementei a minha versão do projeto, documentando todo o processo de criação e aprendizados nesse README.

## Estrutura do repositório
- [micrograd/engine.py](./micrograd/engine.py) O motor micrograd com a classe Value e seus métodos.
- [microgra/nn.py](./micrograd/nn.py) API para criação de MLP's usando o micrograd.
- [test/test.py](./test/test.py) Testes do micrograd usando PyTorch

## Construindo o micrograd
### A intuição da derivada
Precisamos reforçar uma intuição muito importante: a **derivada de uma função**.  
A derivada de uma função do ponto de vista da geometria analítica é o coeficiente angular da reta tangente a curva do gráfico de uma função em um certo ponto. Em outras palavas, é um número que representa o quão inclinada é uma reta que tangencia o gráfico de uma função em um ponto (x,y). Isso é útil pois indica o "relevo" da função próximo a aquele ponto com variável x. E portanto a influência dessa variável x na saída da função.
Uma rede neural artificial é uma função matemática cujo valor calculado está em função de seus diversos parâmetros (pesos e viéses de seus neurônios).
No micrograd, a derivada é calculada usando sua definição por limite. Onde a derivada de uma função f(x) = y, onde y está em função de x, será:  
$$
\frac{dy}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{ h}
$$  
Porém, a saída da equação de um modelo está em função de N variáveis (seus parâmetros). Portanto, precisamos calcular as derivadas dessa função em relação a cada um de seus parâmetros. Em outras palavras, identificar o número (**gradiente**) que indica o quanto e de que forma (se boa ou ruim) cada parâmetro influencia na resposta do modelo.  

### Implementando grafo computacional
  
### Entendendo o backpropagation
  
### Implementando o backpropagation
   
## Construindo a MLP
### Implementando o neurônio
   
### Implementando a camada
  
### Implementando a MLP

## Treinando a MLP
  
### O método gradients
  
### O método zero-grad
  
### O otimizador
  
### A Loss
  
### Loop de treino
## Principais conclusões
-
