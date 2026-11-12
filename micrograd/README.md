## Sobre o projeto
Micrograd é um pequeno motor de cálculo de gradientes de redes neurais simples. Opera sobre grafos direcionados acíclicos (DAG) construídos dinamicamente. Sua construção é inspirada no método backwards do PyTorch, visto que realiza a mesma tarefa de retropropagação. Porém, operando apenas dados escalares e não tensores.  
Micrograd é de autoria de Andrej Karpathy. Seu repositório do projeto original está disponível [aqui](https://github.com/karpathy/micrograd) e ele criou o vídeo [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1926s) com uma explicação detalhada de sua construção. 
Implementei a minha versão do projeto implementando mais funcionalidades: 
- API para criação de redes neurais mais parecida com a do PyTorch.
- Todo: Otimizador SGD
- Todo: Regularização L1/L2
- Todo?: Gradient checking

## Estrutura do repositório
- [micrograd/engine.py](./micrograd/engine.py) O motor micrograd com a classe Value e seus métodos.
- [micrograd/nn.py](./micrograd/nn.py) API para criação de MLP's usando o micrograd.
- [test/test.py](./test/test.py) Testes do micrograd usando PyTorch.
- [micrograd/optmizer.py](./micrograd/optmizer.py) Implementação dos otimizadores. 

## Novas funcionalidades


## Principais conclusões
- Precisamos zerar os gradientes entre épocas. Gradientes indicam como cada parâmetro influencia na previsão do modelo, no final de cada época temos novos parâmetros com novas influências. Não zerar o gradiente é manter as influências antigas e incorretas que afetarão o cálculo das novas influências visto que os gradientes são **acumulados** e não iniciados a cada nova época.
- A justifica para acumularmos os gradientes é cobrir casos de cálculos de derivadas locais que envolvem usos múltiplos de uma mesma variável (regra da cadeia para cálculo multivariado). Além de permitir a execução de técnicas avançadas como gradient accumulation.
- Como o método foward funciona usando a classe value gerando o grafo computacional e cadastrando o cálculo dos gradientes locais.
- Como o método backwards funciona calculando os gradientes de forma "recursiva" usando o grafo computacional e o topological sort.
