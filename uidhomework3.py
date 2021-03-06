#Homework3
# This is a cowork of Oğuzhan İzmir & Uğur İlker Darıcı

import pystan
import pandas as pd
import numpy as np

data = pd.read_csv('trend2.csv')
data = data.dropna()
year = data.year

countries = data.country.str.strip()
unique_countries = countries.unique()
num_countries = len(unique_countries)
unique_year= year.unique()
num_year = len(unique_year)
countries_dict = dict(zip(unique_countries, range(num_countries)))
year_dict = dict(zip(unique_year, range(num_year)))

countries = countries.replace(countries_dict).values
year= year.replace(year_dict).values
religiosity = data.church2.values
inequality = data.gini_net.values
rgdpl = data.rgdpl.values

N = len(countries);
J1 = num_countries
J2 = num_year

#model 1:
model_one = """
data {
  int<lower=0> J1;
  int<lower=0>J2;
  int<lower=0> N;
  int<lower=1,upper=J1> country[N];
  int<lower=1,upper=J2> year[N];
  vector[N] x1; //inequality
  vector[N] x2; //rgdpl
  vector[N] y; //religiosity
}
parameters {
  vector[J1] a1;
  vector[J2] a2;
  real b1;
  real b2;
  real mu_a1;
  real mu_a2;
  real<lower=0,upper=100> sigma_a1;
  real<lower=0,upper=100> sigma_a2;
  real<lower=0,upper=100> sigma_y;
}
transformed_parameters {
  vector[N] y_hat;
  for (i in 1:N)
     y_hat[i] = alpha + a1[country[i]] + a2[year[i]] + x1[i]*b[1] + x2*b[2];
}
model {
  sigma_a1 ~ normal(0, 100);
  a ~ normal(mu_a1, sigma_a1);
  sigma_a2 ~ normal(0, 100);
  a ~ normal(mu_a2, sigma_a2);
  b1 ~ normal(0,1);
  b2 ~ normal(0,1);
  sigma_y ~ normal(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""

model_one_data = {'N': len(religiosity),
                 'J1': len(np.unique(countries)),
'J2': len(np.unique(year)),
                 'country': countries + 1,
        'year': year + 1,
                 'x1': inequality,
                 'x2': rgdpl,
                 'y': religiosity}

model_one_fit = pystan.stan(model_code=model_one, data=model_one_data, iter=1000, chains=2)

a_sample = pd.DataFrame(model_one_fit['a'])



#model 2:
model_two = """
data {
  int<lower=0> J1;
  int<lower=0> J2;
  int<lower=0> N;
  int<lower=1,upper=J1> country[N];
  int<lower=1,upper=J2> year[N];
  vector[N] x1; //inequality
  vector[N] x2; //rgdpl
  vector[N] y; //religiosity
}
parameters {
  vector[J1] a1;
  vector[J2] a2;
  real b1;
  real b2;
  real mu_a1;
  real mu_a2;
  real<lower=0,upper=100> sigma_a1;
  real<lower=0,upper=100> sigma_a2;
  real<lower=0,upper=100> sigma_y;
}
transformed_parameters {
  vector[N] y_hat;
  for (i in 1:N)
 y_hat[i] = alpha + a1[country[i]] + a2[year[i]] + x1[i]*b[1] + x2*b[2];
}
model {
  sigma_a1 ~ normal(0, 100);
  sigma_a2 ~ normal(0, 100);
  a ~ normal(mu_a1, sigma_a1);
  a ~ normal(mu_a2, sigma_a2);
  b1 ~ normal(100,100);
  b2 ~ normal(0,1);
  sigma_y ~ normal(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""

model_two_data = {'N': len(religiosity),
                 'J1': len(countries),
		 'J2': len(year),
                 'country': countries + 1,
                 'x1': inequality,
                 'x2': rgdpl,
                 'y': religiosity}

model_two_fit = pystan.stan(model_code=model_two, data=model_two_data, iter=1000, chains=2)

a_sample = pd.DataFrame(model_two_fit['a'])

#To make the beta estimate of the explanatory variable highly informative, we changed beta prior, such that b1~beta(100,100) in the second model.
#Unfortunately, we could not run our models even though we waited for long hours.
#In two different run,we expected to observe the posterior distribution to be highly determined by the data when prior distribution is noninformative
#and when the prior distribution is informative, we expected the dependence of the posterior distribution to the data to be decreased.
