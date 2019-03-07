# Melhores práticas para o projeto Boston Housing

## Links Importantes
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)


## Checklist (erros mais comuns)
- Utilizar a biblioteca numpy para executar os cálculos necessários em **Implementação: Calcular Estatísticas**


## Dicas
- O pacote `seaborn` possui algumas ferramentas úteis para visualização de dados (você pode instalar pelo terminal digitando: `pip install seaborn`). O seguinte código, fornece um sumário gráfico dos atributos:
```py
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
sns.pairplot(data, size=2)
plt.show()
```
![pairplot.png](https://udacity-reviews-uploads.s3.amazonaws.com/_attachments/38140/1495310213/pairplot.png)

- Outra ferramenta útil é o mapa de calor (heatmap) para imprimir uma matriz de correlação:
```
cols = data.columns
cm = data.corr()
sns.heatmap(cm, annot=True, square=True, yticklabels=cols, xticklabels=cols)
```
![cm.png](https://udacity-reviews-uploads.s3.amazonaws.com/_attachments/38140/1495310199/cm.png)

- As correlações podem ser revisadas:

```
import matplotlib.pyplot as plt

y = prices
plt.figure(figsize=(20, 5))
for i, col in enumerate(features):    
    x = features[col]
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit) 
    
    plt.subplot(1, 3, i+1)
    plt.plot(x, fit_fn(x), '-k')
    plt.scatter(x,y, alpha = 0.5)
    plt.xlabel(col)
```

![stats.png](https://udacity-github-sync-content.s3.amazonaws.com/_attachments/38140/1481316690/stats.png)
