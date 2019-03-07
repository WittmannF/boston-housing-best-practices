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

- O seguinte código compara diferentes scores R2:
```

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#%matplotlib inline

# Generate toy data
X, y, w_true = make_regression(n_samples=20, n_features=1, random_state=0, noise=1.0, coef=True)
w_bad = 0.5*w_true
w_verybad = -0.3*w_true

# Plot true data
plt.figure(figsize=(8,5))
plt.plot(X, y, '.')

# Plot a good, a bad and a very model with R2 scores
for w in [w_true, w_bad, w_verybad]:
    plt.plot(X, w*X, label="R2 Score = {:.3f}".format(r2_score(y, w*X)))
plt.legend(loc=0)
plt.show()
```
![download 2](https://user-images.githubusercontent.com/5733246/51628193-7c3df200-1f2b-11e9-906c-667c9184ce70.png)
