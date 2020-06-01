#Having generated the data, I utilized a multiple linear regression analysis.
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv("4_dataset.csv")

#First, I wanted to check the correlations and tabulate them.
corr_mat = df.drop("Country", axis=1).corr()
plt.figure(figsize=(20,20))
sns_plot = sns.heatmap(corr_mat, annot=True, cmap=plt.cm.Reds)
plt.show()

sns_plot.figure.savefig("6_Correlation_Matrix.png")

#Then, I run the regression analysis and summarize the results.
X = df.drop(["Avg_stri","Country"], axis=1)
y = df["Avg_stri"]

reg = sm.OLS(y, X)
result = reg.fit()
result.summary()

#Finally, in my second model, I only included statistically significant 
#columns in the prior regression, and observed that their robustness increased.
X = df.drop(["Avg_stri","Country","Unemployment",
             "Fdi_Inflows","Fdi_Outflows","T_Union"], axis=1)
y = df["Avg_stri"]

reg = sm.OLS(y, X)
result = reg.fit()
result.summary()