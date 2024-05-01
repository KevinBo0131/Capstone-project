# Installing select libraries:-

from gc import collect; # garbage collection to free up memory
from warnings import filterwarnings; # handle warning messages

import re # regular expressions

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import missingno as msno # Importing Missingno library for visualizing missing data

import numpy as np

import matplotlib
import matplotlib.pyplot as plt # data visualization
from matplotlib.ticker import FuncFormatter  # For custom formatting of ticks
from matplotlib.ticker import FormatStrFormatter  # For formatting ticks with a string
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D axes module
from matplotlib.colors import ListedColormap  # Importing colormap for scatter plot
import seaborn as sns # statistical data visualization

from wordcloud import WordCloud  # Importing WordCloud library for textual data visualization.
from datetime import datetime  # Importing datetime library for handle date & time

from scipy import stats # statistical functions
from scipy.stats import zscore # z-score
from scipy.stats import chi2_contingency # Chi-squared test
from scipy.stats import f_oneway # performing one-way ANOVA test
from statsmodels.formula.api import ols # creating Ordinary Least Squares (OLS) regression models
from statsmodels.stats.anova import anova_lm # computing ANOVA tables for linear regression models
from statsmodels.stats import weightstats as wstat  # Importing statsmodels for z tests
from scipy.stats.mstats import winsorize  # Importing winsorize function for handle outliers

from sklearn.linear_model import LinearRegression, LogisticRegression  # Import LinearRegression and LogisticRegression
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor for random forest
from sklearn.svm import SVR  # Import SVR for Support Vector Regression
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor for decision tree
from sklearn.neighbors import KNeighborsRegressor # implementing the K-Nearest Neighbors algorithm for continous value predict
from sklearn.model_selection import train_test_split # splitting data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # evaluating the accuracy of the classifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder # encode categorical labels into numerical labels
from sklearn.preprocessing import OneHotEncoder, StandardScaler # categorical variables into numerical format & Standardize the feature variables
from sklearn.metrics import mean_squared_error # mean squared error (MSE) between the actual and predicted values
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning
from sklearn.metrics import r2_score # evaluate the performance 


from itertools import combinations  # Igenerating combinations
from tabulate import tabulate  # Importing tabulate library for formatting tables

from IPython.display import display

from colorama import Fore

filterwarnings('ignore'); # Ignore warning messages
from IPython.display import display_html, clear_output; # displaying HTML content

plt.style.use("fivethirtyeight")
sns.set(rc={"figure.figsize":(10, 10)})
print(f"{Fore.GREEN}Successfully Configured libraries!{Fore.RESET}")

clear_output();
print();
collect();