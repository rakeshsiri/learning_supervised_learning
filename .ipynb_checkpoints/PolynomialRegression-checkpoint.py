# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.DataFrame({'Var_X':[-0.33532, 0.0216, -1.19438, -0.65046, -0.28001, 1.93258, 1.2262, 0.74727, 3.32853, 2.8745700000000003, -1.48662, 0.37629, 1.43918, 0.24183000000000002, -2.7914, 1.08176, 2.81555, 0.54924, 2.36449, -1.01925],
  'Var_Y':[6.668539999999999, 3.86398, 5.16161, 8.43823, 5.57201, -11.1327, -5.31226, -4.63725, 3.8065, -6.06084, 7.22328, 2.3888700000000003, -7.13415, 2.00412, 4.2979400000000005, -5.865530000000001, -5.20711, -3.52863, -10.16202, 5.31123] })

X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)
