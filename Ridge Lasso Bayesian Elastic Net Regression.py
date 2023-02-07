#https://medium.com/codex/house-price-prediction-with-machine-learning-in-python-cf9df744f7ff

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
import sklearn.metrics

from termcolor import colored as cl 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import BayesianRidge 
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import explained_variance_score as evs 
from sklearn.metrics import r2_score as r2 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

df = pd.read_csv('C:\\Users\\dmcmo\\OneDrive\\Desktop\\Python - Ridge Lasso Bayesian Elastic Net Regression\\House-Price-Prediction-with-ML-master\\House-Price-Prediction-with-ML-master\\House_Data.csv')
df = df.drop('Id', axis=1)

df.head(5)

print(f"Rows: ", df.shape[0])
print(f"Columns: ", df.shape[1])

df.isnull().sum() #Only 8 out of 1460
df['MasVnrArea'].describe()
#Zero is the most frequest number
df['MasVnrArea'].mode()
sns.histplot(df['MasVnrArea'], binwidth = 2)
sns.distplot(df['MasVnrArea'], hist = True, kde = True)
#I remove the nans instead of replacing them with 0
df.dropna(inplace = True)

df.describe()

df.dtypes

df['MasVnrArea'] = df['MasVnrArea'].astype('int64')

sns.heatmap(df.corr(), annot = True, cmap = 'magma')

#scatter plots
def scatter_df(y_var):
    scatter_df = df.drop(y_var, axis = 1)
    i = df.columns
    
    plot1 = sns.scatterplot(x = i[0], y = y_var, data = df, color = 'orange', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[0]), fontsize = 16)
    plt.xlabel('{}'.format(i[0]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter1.png')
    plt.show()
    
    plot2 = sns.scatterplot(x = i[1], y = y_var, data = df, color = 'yellow', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[1]), fontsize = 16)
    plt.xlabel('{}'.format(i[1]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter2.png')
    plt.show()
    
    plot3 = sns.scatterplot(x = i[2], y = y_var, data = df, color = 'aquamarine', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[2]), fontsize = 16)
    plt.xlabel('{}'.format(i[2]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter3.png')
    plt.show()
    
    plot4 = sns.scatterplot(x = i[3], y = y_var, data = df, color = 'deepskyblue', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[3]), fontsize = 16)
    plt.xlabel('{}'.format(i[3]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter4.png')
    plt.show()
    
    plot5 = sns.scatterplot(x = i[4], y = y_var, data = df, color = 'crimson', edgecolor = 'white', s = 150)
    plt.title('{} / Sale Price'.format(i[4]), fontsize = 16)
    plt.xlabel('{}'.format(i[4]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter5.png')
    plt.show()
    
    plot6 = sns.scatterplot(x = i[5], y = y_var, data = df, color = 'darkviolet', edgecolor = 'white', s = 150)
    plt.title('{} / Sale Price'.format(i[5]), fontsize = 16)
    plt.xlabel('{}'.format(i[5]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter6.png')
    plt.show()
    
    plot7 = sns.scatterplot(x = i[6], y = y_var, data = df, color = 'khaki', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[6]), fontsize = 16)
    plt.xlabel('{}'.format(i[6]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter7.png')
    plt.show()
    
    plot8 = sns.scatterplot(x = i[7], y = y_var, data = df, color = 'gold', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[7]), fontsize = 16)
    plt.xlabel('{}'.format(i[7]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter8.png')
    plt.show()
    
    plot9 = sns.scatterplot(x = i[8], y = y_var, data = df, color = 'r', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[8]), fontsize = 16)
    plt.xlabel('{}'.format(i[8]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter9.png')
    plt.show()
    
    plot10 = sns.scatterplot(x = i[9], y = y_var, data = df, color = 'deeppink', edgecolor = 'b', s = 150)
    plt.title('{} / Sale Price'.format(i[9]), fontsize = 16)
    plt.xlabel('{}'.format(i[9]), fontsize = 14)
    plt.ylabel('Sale Price', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatter10.png')
    plt.show()
    
scatter_df('SalePrice')

#response variable disrtibution
sns.distplot(df['SalePrice'])
plt.title('Sale Price Distribution', fontsize = 16)
plt.xlabel('Sale Price', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

#explanatory matrix
X_var = df.drop('SalePrice', axis = 1)
#response
y_var = df['SalePrice']

#Data split into train and test stes
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

#Models withou tuning hyperparameters

#Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

#-----------------------------------------------------------------------------------------------------------------
#another linear regression
import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
model_yhat = model.predict(X_test)
r2(y_test, model_yhat)
#-----------------------------------------------------------------------------------------------------------------

#Regularised model, Ridge model
ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

#Regularised model, Lasso model
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

#Regularised model, Elastic Net model
en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)

#Bayesian model, Bayesian Ridge model
bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

#Models evaluation. Explained Variance score, r-squared, and RMSE.
exvarscor = pd.DataFrame({'Explainded Variance Score':['Linear Regression', #similar to R2 but calculated slightly differently
                                                       'Ridge Regression',
                                                       'Lasso Regression',
                                                       'Elastic Net',
                                                       'Bayesian Ridge'],
                    'Score':[evs(y_test, lr_yhat),
                             evs(y_test, ridge_yhat),
                             evs(y_test, lasso_yhat),
                             evs(y_test, en_yhat),
                             evs(y_test, bayesian_yhat)]})
exvarscor = exvarscor.sort_values('Score', ascending = False)
exvarscor

r2_df = pd.DataFrame({'Model':['Linear Regression',
                            'Ridge Regression',
                            'Lasso Regression',
                            'Elastic Net',
                            'Bayesian Ridge'],
                   'R Squared':[r2(y_test, lr_yhat),
                                r2(y_test, ridge_yhat),
                                r2(y_test, lasso_yhat),
                                r2(y_test, en_yhat),
                                r2(y_test, bayesian_yhat)]})
r2_df = r2_df.sort_values('R Squared', ascending = False)
r2_df

RMSE = pd.DataFrame({'Model':['Linear Regression',
                            'Ridge Regression',
                            'Lasso Regression',
                            'Elastic Net',
                            'Bayesian Ridge'],
                     'RMSE':[np.sqrt(mean_squared_error(y_test, lr_yhat)),
                             np.sqrt(mean_squared_error(y_test, ridge_yhat)),
                             np.sqrt(mean_squared_error(y_test, lasso_yhat)),
                             np.sqrt(mean_squared_error(y_test, en_yhat)),
                             np.sqrt(mean_squared_error(y_test, bayesian_yhat))]})
RMSE = RMSE.sort_values('RMSE', ascending = False)
RMSE

#Tune hyperparameters for the two best models
#Tune alpha for Elastic Net and Lasso
#Lasso model
param_grid = {'alpha': np.linspace(0.1, 100000, 2000)}
grid_search = GridSearchCV(Lasso(max_iter = 100000),
                           param_grid, scoring = ['r2'],
                           refit = 'r2')
grid_result = grid_search.fit(X_train, y_train)
LaR = pd.DataFrame(grid_result.cv_results_)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

#plot aplha values vs R2
plt.plot(LaR['param_alpha'], LaR['mean_test_r2'])
plt.axvline(x = LaR.loc[LaR.mean_test_r2 == LaR.mean_test_r2.max(), 'param_alpha'].item(), color = 'r')
plt.xlabel('Alpha Values')
plt.ylabel(' $R^2$')

#Lasso model with the best alpha value
lasso2 = Lasso(alpha = LaR.loc[LaR.mean_test_r2 == LaR.mean_test_r2.max(), 'param_alpha'].item())
lasso2.fit(X_train, y_train)
lasso_yhat2 = lasso2.predict(X_test)


#Elastic Net
param_grid = {'alpha': np.linspace(0.1, 100000, 2000)}
grid_search = GridSearchCV(ElasticNet(max_iter = 10000),
                           param_grid, scoring = ['r2'],
                           refit = 'r2')
grid_result = grid_search.fit(X_train, y_train)
ENR = pd.DataFrame(grid_result.cv_results_)
print(f'Best result: {grid_result.best_score_} for {grid_result.best_params_}')

#plot alpha values vs R2
plt.plot(ENR['param_alpha'], ENR['mean_test_r2'])
plt.axvline(x = ENR.loc[ENR.mean_test_r2 == ENR.mean_test_r2.max(), 'param_alpha'].item(), color = 'r')
plt.xlabel('Alpha Values')
plt.ylabel(' $R^2$')

#Elastic Net model with the best alpha value
en2 = ElasticNet(alpha = ENR.loc[ENR.mean_test_r2 == ENR.mean_test_r2.max(), 'param_alpha'].item())
en2.fit(X_train, y_train)
en_yhat2 = en2.predict(X_test)


evs(y_test, lasso_yhat2)
evs(y_test, en_yhat2)

r2(y_test, lasso_yhat2)
r2(y_test, en_yhat2)

np.sqrt(mean_squared_error(y_test, lasso_yhat2))
np.sqrt(mean_squared_error(y_test, en_yhat2))

#Lasso model seems to be the best



#Simple ANN. The results are not good because the model should be tuned but that require time.
model = keras.Sequential([
    # input layer
    keras.layers.Dense(60, input_shape = (10,), kernel_initializer = 'normal', activation = 'relu'),
    keras.layers.Dense(120, kernel_initializer = 'normal', activation = 'relu'),
    keras.layers.Dense(240, kernel_initializer = 'normal', activation = 'relu'),
    # we use sigmoid for binary output
    # output layer
    keras.layers.Dense(1, kernel_initializer = 'normal', activation = 'linear')])

optimizer = keras.optimizers.Adam(learning_rate = 0.001)

model.compile(optimizer = optimizer,
              loss = keras.losses.MeanSquaredError(),
              metrics = [keras.metrics.RootMeanSquaredError()])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience  = 3)]

history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, callbacks = callbacks, validation_split = 0.2)

plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.title('Training and Validation loss - Adam, lr = 0.001')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
plt.plot(history.history['root_mean_squared_error'], label='validation RMSE')
plt.title('Training and Validation RMSE - Adam, lr = 0.001')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()


ANN_yhat = model.predict(X_test)

ANN_RMSE = np.sqrt(mean_squared_error(y_test, ANN_yhat))
ANN_RMSE






