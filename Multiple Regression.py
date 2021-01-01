import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv('SampleBasketball.csv')
X = df[['X1', 'X2', 'X3', 'X4']]
y = df['X5']
est = sm.OLS(y, X).fit()
print(est.summary())

scaled = [6.4, 190 , 0.456, 0.761]
print(scaled)
predicted = est.predict(scaled)
print(predicted)
