## Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

from sklearn.model_selection import train_test_split, KFold, cross_val_score,  RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel, RFE, RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

## Data Import
train_orig = pd.read_csv('Data/train.csv')
test_orig = pd.read_csv('Data/test.csv')
sample_submission = pd.read_csv('Data/sample_submission.csv')

test = test_orig.copy()
train = train_orig.copy()


### EDA TEST
test.info()
test.describe().T
test.nunique().sort_values()

test.drop(['ArrivalYear', 'CompanyReservation'],axis = 1, inplace = True)
test[test.duplicated() == True]

test[test['BookingID'].duplicated()]

test.set_index(['BookingID'], inplace = True)



## Histograms

hist_test = test.drop(['WeekendStays', 'WeekdayStays', 'Adults', 'Children', 'Babies', 'ParkingSpacesBooked',
     'SpecialRequests', 'OrderedMealsPerDay', 'FloorAssigned', 'FloorReserved', '%PaidinAdvance', 'FirstTimeGuest',
     'AffiliatedCustomer', 'OnlineReservation', 'PartOfGroup', 'PreviousStays',
     'PreviousReservations', 'PreviousCancellations', 'DaysUntilConfirmation', 'BookingChanges'], axis=1)

fig, ax = plt.subplots(5,2, figsize=(15,15))
ax_row = 0
ax_col = 0

for var in hist_test:
    var = str(var)
    sns.histplot(data=test, x=var, kde=True, color='red', alpha=1, ax = ax[ax_col, ax_row])
    sns.histplot(data=test, x=var, kde=False, alpha=1, ax = ax[ax_col, ax_row])
    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 2:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()

## Boxplots
nrows, ncols = 5,2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
fig.subplots_adjust(wspace=0.35, hspace=0.25)

for i, f in enumerate(hist_test):
    test.boxplot(f, ax=axes.flatten()[i])

plt.tight_layout()

## Barplots
bar_var = ['WeekendStays','WeekdayStays','Adults','Children','Babies','ParkingSpacesBooked','SpecialRequests','OrderedMealsPerDay','FloorAssigned','FloorReserved','%PaidinAdvance']

fig, ax = plt.subplots(3,4, figsize=(15,15))

ax_row = 0
ax_col = 0

for var in bar_var:
    test[var].value_counts().plot(kind='bar', ax = ax[ax_col, ax_row])
    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 4:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()

## Pie Charts

# Create a list with all the continuous features
bi_variables = ['FirstTimeGuest','AffiliatedCustomer','OnlineReservation','PartOfGroup']


# Initialise a 3 by 3 plot
fig, ax = plt.subplots(2,2, figsize=(15,15))

# Store index of row and column for loop
ax_row = 0
ax_col = 0

for var in bi_variables:
    test[var].value_counts().plot(kind='pie', ax = ax[ax_col, ax_row])

    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 2:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()


## Checking for inconsistencies

test.loc[(test['FirstTimeGuest'] == 1) & (test['PreviousStays'] > 0)]

# First time guests with previous stays are disregarded and changed to not first time guests
test.loc[(test['FirstTimeGuest'] == 1) & (test['PreviousReservations'] > 0), 'FirstTimeGuest'] = 0

test.loc[(test['FirstTimeGuest'] == 0) & (test['PreviousReservations'] == 0)]

# customers who were being considered as not first time guests, but had no previous reservations were transformed into first time guests
test.loc[(test['FirstTimeGuest'] == 0) & (test['PreviousReservations'] == 0), 'FirstTimeGuest'] = 1

test.loc[(test['WeekendStays'] == 0) & (test['WeekdayStays'] == 0)]

test.loc[(test['PreviousReservations'] >0) & (test['PreviousStays'] ==0) & (test['PreviousCancellations'] ==0)]
##


### EDA TRAIN
train.info()
train.describe().T
train.nunique().sort_values()

train.drop(['ArrivalYear', 'CompanyReservation'],axis = 1, inplace = True)
train[train.duplicated() == True]

train[train['BookingID'].duplicated()]

train.set_index(['BookingID'], inplace = True)



## Histograms

hist_train = train.drop(['WeekendStays', 'WeekdayStays', 'Adults', 'Children', 'Babies', 'ParkingSpacesBooked',
     'SpecialRequests', 'OrderedMealsPerDay', 'FloorAssigned', 'FloorReserved', '%PaidinAdvance', 'FirstTimeGuest',
     'AffiliatedCustomer', 'OnlineReservation', 'PartOfGroup', 'PreviousStays',
     'PreviousReservations', 'PreviousCancellations', 'DaysUntilConfirmation', 'BookingChanges'], axis=1)

fig, ax = plt.subplots(5,2, figsize=(15,15))
ax_row = 0
ax_col = 0

for var in hist_train:
    var = str(var)
    sns.histplot(data=train, x=var, kde=True, color='red', alpha=1, ax = ax[ax_col, ax_row])
    sns.histplot(data=train, x=var, kde=False, alpha=1, ax = ax[ax_col, ax_row])
    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 2:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()

## Boxplots
nrows, ncols = 5,2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
fig.subplots_adjust(wspace=0.35, hspace=0.25)

for i, f in enumerate(hist_train):
    train.boxplot(f, ax=axes.flatten()[i])

plt.tight_layout()

## Barplots
bar_var = ['WeekendStays','WeekdayStays','Adults','Children','Babies','ParkingSpacesBooked','SpecialRequests','OrderedMealsPerDay','FloorAssigned','FloorReserved','%PaidinAdvance']

fig, ax = plt.subplots(3,4, figsize=(15,15))

ax_row = 0
ax_col = 0

for var in bar_var:
    train[var].value_counts().plot(kind='bar', ax = ax[ax_col, ax_row])
    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 4:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()

## Pie Charts

# Create a list with all the continuous features
bi_variables = ['FirstTimeGuest','AffiliatedCustomer','OnlineReservation','PartOfGroup']


# Initialise a 3 by 3 plot
fig, ax = plt.subplots(2,2, figsize=(15,15))

# Store index of row and column for loop
ax_row = 0
ax_col = 0

for var in bi_variables:
    train[var].value_counts().plot(kind='pie', ax = ax[ax_col, ax_row])

    ax_row += 1
    # Restart row when it reaches 3 and add a column
    if ax_row == 2:
        ax_row = 0
        ax_col += 1

fig.tight_layout()
fig.show()


## Checking for inconsistencies

train.loc[(train['FirstTimeGuest'] == 1) & (train['PreviousStays'] > 0)]

# First time guests with previous stays are disregarded and changed to not first time guests
train.loc[(train['FirstTimeGuest'] == 1) & (train['PreviousReservations'] > 0), 'FirstTimeGuest'] = 0

train.loc[(train['FirstTimeGuest'] == 0) & (train['PreviousReservations'] == 0)]

# customers who were being considered as not first time guests, but had no previous reservations were transformed into first time guests
train.loc[(train['FirstTimeGuest'] == 0) & (train['PreviousReservations'] == 0), 'FirstTimeGuest'] = 1

train.loc[(train['WeekendStays'] == 0) & (train['WeekdayStays'] == 0)]

train.loc[(train['PreviousReservations'] >0) & (train['PreviousStays'] ==0) & (train['PreviousCancellations'] ==0)]
##















































