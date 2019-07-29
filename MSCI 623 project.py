#!/usr/bin/env python
# coding: utf-8


#importing all required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


#reading the dataset's csv file into a dataset
ds = pd.read_csv('E:/fuel_consumption.csv')
#checking for NaN values
na_val = ds.isnull().sum()
#printing total NaN values
print(na_val)


#a peek into the structure of the data
ds.head(5)


#stats about the dataset
ds.describe()


#printing number of unique values for all categorical variables
print("---Unique Values---")
print("Make:\t\t",len(Counter(ds['make'])),"\nModel:\t\t",len(Counter(ds['model'])),"\nClass:\t\t",len(Counter(ds['class'])),
      "\nTransmission:\t",len(Counter(ds['transmission'])),"\nFuel Type:\t",len(Counter(ds['fuel_type'])))
#printing total rows in the dataset
print('\nTotal rows in dataset:\t',len(ds))




#implementing apriori algorithm on select few rows(make,class,transmission & fuel)
#convert dataframe into a list of lists
ds_list = ds[['make', 'class', 'transmission', 'fuel_type','model']].values.tolist()
#create instance for Transaction encoder which creates columns for each unique data point 
#and encodes them as boolean for each row
tenc = TransactionEncoder()
#encoding each data point as a boolean value
te_ds = tenc.fit(ds_list).transform(ds_list)
#loading encoded values as dataframe 
ds_df = pd.DataFrame(te_ds , columns = tenc.columns_)
#finding frequent sets using apriori algorithm for support of 10%
apriori_sets=apriori(ds_df,min_support=0.1, use_colnames=True)
#display results
apriori_sets



#generating association rules for itemsets from apriori for a min confidence of 10%
association_rules(apriori_sets, metric='confidence', min_threshold=0.1)




#handling categorical variables
#creating dummy variable with a binary column for each unique value in fuel_type(OneHotEncoding)
dummy_fuel = pd.get_dummies(ds['fuel_type'])
#add dummy variables to ds dataframe
ds = pd.concat([ds,dummy_fuel],axis = 1)
encode = LabelEncoder()#hashes each unique value to a numerical value
#transform make, model, class and transmission using LabelEncoder.
ds['make_code'] = encode.fit_transform(ds['make'])
ds['model_code'] = encode.fit_transform(ds['model'])
ds['class_code'] = encode.fit_transform(ds['class'])
ds['tr_code'] = encode.fit_transform(ds['transmission'])
#results after make, model, class and transmission are encoded
ds[['make', 'make_code' ,'model' ,'model_code' ,'class' ,'class_code' ,'transmission' ,'tr_code']].head(10)



#viewing first few rows of fuel_type and its dummy variables
ds[['fuel_type', 'Diesel', 'Ethanol', 'Natural_Gas', 'Prem_Gasoline','Reg_Gasoline']].head(5)



#displaying correlation matrix for checking multicollinearity
ds.corr()


#plot histogram for engine size
plt.hist(ds['engine_size'], bins = 30, color = "gray")
#set labels and title
plt.xlabel('engine size')
plt.ylabel('frequency')
plt.title('Histogram for engine_size')
#display plot
plt.show()



#scatterplot of outcome variable vs city
plt.scatter(ds['co2_emission'], ds['engine_size'], color = "gray",s = 10)
#set labels and title
plt.xlabel('co2_emission')
plt.ylabel('engine_size')
plt.title('Scatterplot: co2_emission vs engine_size')
#display plot
plt.show()


#segregate predictors and outcome variable
x = ds.iloc[:, [0, 4,  13, 14, 15, 16, 18, 19, 20,21 ]]
y = ds[['co2_emission']]
#first 5 rows of predictors
x.head(5)


#first 5 rows of outcome
y.head(5)


#Linear regression model
#dataset split into training and testing data at 80:20 ratio
x_tr1, x_te1, y_tr1, y_te1 = train_test_split(x, y, test_size=0.2, random_state = 3)
#model fit to training data and scored for testing data
lin_reg = LinearRegression()
lin_fit = lin_reg.fit(x_tr1 , y_tr1)
#printing accuracy score
print("Accuracy for Linear regression:",lin_fit.score(x_te1 , y_te1))


#getting values predicted by model for the testing data
y_reg = lin_fit.predict(x_te1)
#create a copy of y from testing set and insert column of predictions to it.
y_te1_lin=y_te1.copy()
y_te1_lin.insert(1, "predicted", y_reg)
y_te1_lin.head(5)



#calculate residuals with actuals and predicted from the the previous dataframe
resid_ = y_te1_lin['co2_emission'] - y_te1_lin['predicted']
#convert list of residuals to dataframe 
resid_ = pd.DataFrame(resid_, columns = ['residual'])
resid_.head(5)



#plot histograms for residuals with 30 bins in color gray
plt.hist(resid_['residual'], bins = 30, color = 'gray')
#set axes labels and title to histogram
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram: Linear Regression model residuals")
#display plot
plt.show()


#bar chart showing predicted vs actual for first 15 results
#set first 15 results to a dataframe
plot_df = y_te1_lin.head(15)
#specift plot type as bar chart
plot_df.plot(kind = 'bar')
#plot a grid
plt.grid(which = 'major', linewidth=0.7)
plt.grid(which = 'minor', linewidth=0.7)
#set labels for x and y axes 
plt.xlabel("Row Number")
plt.ylabel("Values")
#set title to plot
plt.title("Linear model:Predicted vs Actual")
#display plot
plt.show()


#creating a function that displays coefficient for alpha values in a dataframe
def det_alpha(a_vals):
  #create dataframe with one column containing predictor column names of x
    df_a = pd.DataFrame(np.array(list(x)),columns = [['Predictor']])
  #for loop for printing coefficient of predictors for different values of alpha.
    for a in a_vals: 
      # set a single alpha value that was parsed in function call as alpha in Lasso 
      lasso_reg = Lasso(alpha = a)
      #fit x and y to model
      lasso_reg.fit(x, y)
      #create var col to be used for column name
      col = "α = %f"%a
      #add coefficients of model to column named as value in col
      df_a[col] = lasso_reg.coef_
      #continue loop for all parsed values. exit loop after and return df_a 
    return df_a
#parsing alpha values as list in function call
det_alpha([0.001,0.01,0.05,0.1,0.5])


#Lasso regression model
#model fit to training data and scored for testing data
lasso_reg = Lasso(alpha = 0.05, max_iter = 1000)
lasso_fit = lasso_reg.fit(x_tr1 , y_tr1)
#printing accuracy score
print("Accuracy for Lasso regression:", lasso_reg.score(x_te1 , y_te1))


#getting values predicted by model for the testing data
y_las = lasso_fit.predict(x_te1)
#create a copy of y from testing set and insert column of predictions to it.
y_te1_las=y_te1.copy()
y_te1_las.insert(1, "predicted", y_las)
y_te1_las.head(5)



#calculate residuals with actuals and predicted from the the previous dataframe
resid_las = y_te1_las['co2_emission'] - y_te1_las['predicted']
#convert list of residuals to dataframe
resid_las = pd.DataFrame(resid_las, columns = ['residual'])
resid_las.head(5)


#plot histograms for residuals with 30 bins in color gray
plt.hist(resid_las['residual'], bins = 30, color = 'gray')
#set labels and titles
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram: Lasso Regression model residuals")
#display plot
plt.show()


#bar chart showing predicted vs actual for first 15 results
#set first 15 results to a dataframe
plot_df = y_te1_las.head(15)
#specift plot type as bar chart
plot_df.plot(kind = 'bar')
#plot a grid 
plt.grid(which = 'major', linewidth=0.7)
plt.grid(which = 'minor', linewidth=0.7)
#set labels for x and y axes 
plt.xlabel("Row Number")
plt.ylabel("Values")
#set title to plot
plt.title("Lasso model:Predicted vs Actual")
#display plot
plt.show()



# Calculate 46 fold cross validation results
print("46 fold cross-validation Mean Accuracy Results")
#parse linear model,x,y and number of folds
scores00 = cross_val_score(lin_reg, x, y, cv = 46)
#print mean of 46 results
print("Linear Regression: " ,scores00.mean())
#parse lasso model,x,y and number of folds
scores01 = cross_val_score(lasso_reg, x, y, cv = 46)
#print mean of 46 results
print("Lasso Regression: " ,scores01.mean())


#load  subset of x with the following predictors
x1=x[['engine_size','year','class_code','make_code','Prem_Gasoline']]
# Calculate 46 fold cross validation results for subset
print("46 fold cross-validation Mean Accuracy Results for new set of predictors")
#parse linear model,x1,y and number of folds
scores10 = cross_val_score(lin_reg, x1, y, cv = 46)
#print mean of 46 results
print("Linear Regression: " ,scores10.mean())
#parse lasso model,x1,y and number of folds
scores11 = cross_val_score(lasso_reg, x1, y, cv = 46)
#print mean of 46 results
print("Lasso Regression: " ,scores11.mean())


#load  subset of ds with the following predictors
x2=ds[['city']]
# Calculate 46 fold cross validation results for subset
print("46 fold cross-validation Mean Accuracy Results for city")
#parse linear model,x2,y and number of folds
scores10 = cross_val_score(lin_reg, x2, y, cv = 46)
#print mean of 46 results
print("Linear Regression: " ,scores10.mean())
#parse lasso model,x2,y and number of folds
scores11 = cross_val_score(lasso_reg, x2, y, cv = 46)
#print mean of 46 results
print("Lasso Regression: " ,scores11.mean())


#load  subset of ds with the following predictors
x3=ds[['city','Prem_Gasoline','Natural_Gas','Ethanol','Diesel']]
# Calculate 46 fold cross validation results for subset
print("46 fold cross-validation Mean Accuracy Results for city and fuel_type")
#parse linear model,x3,y and number of folds
scores10 = cross_val_score(lin_reg, x3, y, cv = 46)
#print mean of 46 results
print("Linear Regression: " ,scores10.mean())
#parse lasso model,x3,y and number of folds
scores11 = cross_val_score(lasso_reg, x3, y, cv = 46)
#print mean of 46 results
print("Lasso Regression: " ,scores11.mean())




