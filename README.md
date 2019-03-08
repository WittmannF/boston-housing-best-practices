# Melhores práticas para o projeto Boston Housing

## Links Importantes
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- https://github.com/udacity/br-machine-learning
- http://scikit-learn.org/stable/modules/cross_validation.html
- https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29#Purpose_of_cross-validation

## Checklist (erros mais comuns)
- Utilizar a biblioteca numpy para executar os cálculos necessários em **Implementação: Calcular Estatísticas**
- 


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

- Exemplos de curvas de aprendizado ([referência](https://sebastianraschka.com/faq/docs/ml-solvable.html))

![](https://sebastianraschka.com/images/faq/ml-solvable/bias-variance.png)

- Referências complementares sobre Curvas de Aprendizado:

    - [Katie’s lesson on Accuracy vs Training Set size](https://www.youtube.com/watch?v=9w1Yi5nMNgw)
    - [Andrew Ng. Class on Learning Curve](https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves)
    - https://www.blaenkdenum.com/notes/machine-learning/#learning-curves

- A seguinte imagem ilustra bias e variância elevados para classificação e regressão ([ref](https://www.youtube.com/watch?v=dBLZg-RqoLg)):
![maxresdefault](https://i.ytimg.com/vi/dBLZg-RqoLg/maxresdefault.jpg)

- Exemplo de curva de complexidade ([referência](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html)):
![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.03-validation-curve.png)

- A seguinte imagem sumariza os como os dados foram subdivididos neste projeto:

![K-foldCross.png](https://udacity-github-sync-content.s3.amazonaws.com/_attachments/38140/1485736387/K-foldCross.png)

- Este é o código que foi utilizado na análise de sensibilidade:
```
def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)
        
        # Fit the data
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print "Trial {}: ${:,.2f}".format(k+1, pred)

    # Display price range
    print "\nRange in prices: ${:,.2f}".format(max(prices) - min(prices))
```

- Alguns comentários:
    - A única coisa que mudou em cada *trial* foi o valor de `random_state` em `train_test_split`. 
    - Em todos os testes, o modelo está tentando prever um preço a partir dos primeiros atributos `client_data` no quais são `[5, 17, 15]`.
    - Este projeto não pediu para o aluno obter uma estimativa final do modelo a partir do conjunto de teste. Caso você queira calcular à parte, basta utilizar a função `performance_metric(y_test, reg.predict(X_test))`. O coeficiente de determinação R2 retornado será em torno de `0.77`.


