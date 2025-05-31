#project summary 

# Heart-attack-prediction model 

The dataset used in this project comes from Kaggle. The goal is to detect patients who have previously experienced a heart attack.

I started with an exploratory data analysis (EDA) to understand the data, visualize initial results, and remove outliers or noisy values.

Next, I built two models:

A logistic regression model that achieved around 80% accuracy in distinguishing the classes. I further optimized this model using cross-validation and grid search to improve performance.

A random forest model, which performed very well, reaching 99% accuracy. This indicated that the data relationships were nonlinear and better captured by the ensemble method.

Finally, I visualized the decision paths of the random forest with graphs to explain how it makes predictions.
