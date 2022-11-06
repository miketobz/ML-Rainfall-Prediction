
# Rainfall Prediction using Machine Learning

Rainfall Prediction is the application of meteorology and machine learning to predict the amount of rainfall over a region. It is important to exactly determine the rainfall for effective use of water resources, crop productivity and pre-planning of water structures.

Moreover, rainfall greatly affects human life in various sectors such as agriculture and transportation. Additionaly, it can cause natural disasters such as drought, floods, and landslides. The aim of this project is to build an accurate rainfall prediction model so that prescriptive measures can be made.

Governments, communities and individuals spend large amounts of money so that there is enough water available for everyone. Collecting and understanding rainfall data is important so that the right decisions are made.

Rainfall prediction has gained utmost research relevance in recent times due to its complexities and persistent applications such as flood forecasting and monitoring of pollutant concentration levels, among others. Existing models use complex statistical models that are often too costly, both computationally and budgetary, or are not applied to downstream applications. Therefore, approaches that use Machine Learning algorithms in conjunction with time-series data are being explored as an alternative to overcome these drawbacks.

The main objective of this project is to identify the relevant atmospheric features that cause rainfall and predict the intensity
of daily rainfall using various machine learning algorithms, such as Multivariate Linear Regression (MLR), Multilayer Perceptron Neural Network (MLPNN), KNeighbors Regressor, and Ridge Regression.

In addition to, Root Mean Squared Error and Mean Absolute Error metrics were used to measure the performance of the machine learning models.

The dataset was collected from Kaggle. It contains meteorological data of Austin, Texas. Further, Austin has a humid subtropical climate, which means it has long, hot summers and typically mild winters. In July and August, high temps often top out at around 100 degrees Fahrenheit (38 degrees Celsius), sometimes for several days in a row. The humidity is usually only at sauna-like levels just before a rainstorm, but even when it’s not raining, the humidity rarely dips below 30 percent.

Therefore, atmospheric features such as temperature, relative humidity, sea level pressure, and wind speed are seen as important features that predict rainfall accurately.




## Acknowledgements

 - [What Is a Regression Model?](https://www.imsl.com/blog/what-is-regression-model#:~:text=A%20regression%20model%20provides%20a,by%20a%20linear%20regression%20model.)
 - [Approaching (Almost) Any Machine Learning Problem Paperback](https://www.amazon.com/Approaching-Almost-Machine-Learning-Problem/dp/8269211508)
 - [Austin Weather](https://www.kaggle.com/datasets/grubenm/austin-weather)


## Appendix
![sample model](https://miro.medium.com/max/720/1*G1Y_-X14q2xMVHlUuaUUdA.png)

Linear regression is a basic predictive analytics technique that uses historical data to predict an output variable. It is popular for predictive modelling because it is easily understood and can be explained using plain English.

Linear regression models have many real-world applications in an array of industries such as economics (e.g. predicting growth), business (e.g. predicting product sales, employee performance), social science (e.g. predicting political leanings from gender or race), healthcare (e.g. predicting blood pressure levels from weight, disease onset from biological factors), and more.

Understanding how to implement linear regression models can unearth stories in data to solve important problems.

## Authors

- [@MichaelTobiko](https://github.com/miketobz)


## API Reference

#### LinearRegression()

```http
 from sklearn import linear_model
```
LinearRegression fits a linear model with coefficients w = (w_1, ..., w_p) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. Mathematically it solves a problem of the form:

min_{w} || X w - y||_2^2

#### Class MLPRegressor

```http
  from sklearn.neural_network import MLPRegressor
```
In regression, the output remains as f(x); therefore, output activation function is just the identity function. MLP uses different loss functions depending on the problem type. The loss function for classification is Average Cross-Entropy, which in binary case is given as,

Loss(\hat{y},y,W) = -\dfrac{1}{n}\sum_{i=0}^n(y_i \ln {\hat{y_i}} + (1-y_i) \ln{(1-\hat{y_i})}) + \dfrac{\alpha}{2n} ||W||_2^2

where 'alpha ||W||_2^2' is an L2-regularization term (aka penalty) that penalizes complex models; and 'alpha > 0' is a non-negative hyperparameter that controls the magnitude of the penalty.

#### GradientBoostingRegressor

```http
  from sklearn import ensemble.GradientBoostingRegressor
```
You can play with these parameters to see how the results change.

n_estimators : the number of boosting stages that will be performed. Later, we will plot deviance against boosting iterations.

max_depth : limits the number of nodes in the tree. The best value depends on the interaction of the input variables.

min_samples_split : the minimum number of samples required to split an internal node.

learning_rate : how much the contribution of each tree will shrink.

loss : loss function to optimize. The least squares function is used in this case however, there are many other options


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/michael-tobiko-1563a693)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](twitter.com/MichaelTobiko)


## Demo
![Demo](https://c.tenor.com/20n67BNCG9wAAAAC/linear-regression.gif)


![Demo1](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11069-021-05063-3/MediaObjects/11069_2021_5063_Fig2_HTML.png)



## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## Deployment

To deploy this project run

```bash
  npm run deploy
```


## Screenshots

![App Screenshot](https://media.geeksforgeeks.org/wp-content/uploads/20190610231245/precipitation.png)

![Screenshot](Rainfall_Prediction_BI_report.png)

## Lessons Learned
With high-dimensional datasets (meaning datasets with a large number of features), linear models become more powerful, and there is a higher chance of overfitting.

When comparing training set and test set scores, I found that the linear model predicts accurately on the training set, but the R^2 score on the test set is much worse; therefore, we should implement a model that allows us to control this complexity (such as Rigde Regression).

In conclusion, the machine learning models did not perform accurately due to poor-quality data; the target vector is highly skewed with zero (0) values.
