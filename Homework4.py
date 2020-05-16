# Homework 4
# This is a cowork of Uğur İlker Darıcı & Oğuzhan İzmir

import numpy as np
import pandas as pd

import os
os.chdir('C:/Users/udarici19/Desktop/KocPython2020/in-classMaterial/day17')
tt = pd.read_csv('immSurvey.csv')
tt.head()

alphas = tt.stanMeansNewSysPooled
sample = tt.textToSend

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

from sklearn.model_selection import train_test_split
# It seems that "sklearn.cross_validation" turned to a new one  "sklearn.model_selection"
# At least, that is what we found when we google it.
Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas,
random_state=1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)

# Here correlation coefficient is 0.68
# Now, we use Bigrams

big_vec =  CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b',
min_df=1)
#Since ngram_range=(2,2) only includes bigrams and not the unigrams, we utilized it.
X = big_vec.fit_transform(sample)

pd.DataFrame(X.toarray(), columns=big_vec.get_feature_names())

Xtrain, Xtest, ytrain, ytest = train_test_split(X, alphas,
random_state=1)

rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-8)

gpr.fit(Xtrain.toarray(), ytrain)

mu_s, cov_s = gpr.predict(Xtest.toarray(), return_cov=True)

np.corrcoef(ytest, mu_s)

#after all, our firts correlation coeeficient, 0.68, turned out to be 0.45.
#analyzing the data through bigrams made the correlation coefficient lower. 
