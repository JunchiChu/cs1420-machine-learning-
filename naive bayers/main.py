#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.
   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.
   Brown CS142, Spring 2020
"""

import numpy as np
import pandas as pd
from models import NaiveBayes

def get_credit():
    """
    Gets and preprocesses German Credit data
    """
    data = pd.read_csv('./data/german_numerical-binsensitive.csv') # Reads file - may change

    # MONTH categorizing
    data['month'] = pd.cut(data['month'],3, labels=['month_1', 'month_2', 'month_3'], retbins=True)[0]
    # month bins: [ 3.932     , 26.66666667, 49.33333333, 72.        ]
    a = pd.get_dummies(data['month'])
    data = pd.concat([data, a], axis = 1)
    data = data.drop(['month'], axis=1)

    # CREDIT categorizing
    data['credit_amount'] = pd.cut(data['credit_amount'], 3, labels=['cred_amt_1', 'cred_amt_2', 'cred_amt_3'], retbins=True)[0]
    # credit bins: [  231.826,  6308.   , 12366.   , 18424.   ]
    a = pd.get_dummies(data['credit_amount'])
    data = pd.concat([data, a], axis = 1)
    data = data.drop(['credit_amount'], axis=1)

    for header in ['investment_as_income_percentage', 'residence_since', 'number_of_credits']:
        a = pd.get_dummies(data[header], prefix=header)
        data = pd.concat([data, a], axis = 1)
        data = data.drop([header], axis=1)

    # change from 1-2 classes to 0-1 classes
    data['people_liable_for'] = data['people_liable_for'] -1
    data['credit'] = -1*(data['credit']) + 2 # original encoding 1: good, 2: bad. we switch to 1: good, 0: bad

    # balance dataset
    data = data.reindex(np.random.permutation(data.index)) # shuffle
    pos = data.loc[data['credit'] == 1]
    neg = data.loc[data['credit'] == 0][:350]
    combined = pd.concat([pos, neg])

    y = data.iloc[:, data.columns == 'credit'].to_numpy()
    x = data.drop(['credit', 'sex', 'age', 'sex-age'], axis=1).to_numpy()

    # split into train and validation
    X_train, X_val, y_train, y_val = x[:350, :], x[351:526, :], y[:350, :].reshape([350,]), y[351:526, :].reshape([175,])

    # keep info about sex and age of validation rows for fairness portion
    x_sex = data.iloc[:, data.columns == 'sex'].to_numpy()[351:526].reshape([175,])
    x_age = data.iloc[:, data.columns == 'age'].to_numpy()[351:526].reshape([175,])
    x_sex_age = data.iloc[:, data.columns == 'sex-age'].to_numpy()[351:526].reshape([175,])

    return X_train, X_val, y_train, y_val, x_sex, x_age, x_sex_age

def main():

    np.random.seed(0)

    X_train, X_val, y_train, y_val, x_sex, x_age, x_sex_age = get_credit()

    model = NaiveBayes(2)

    model.train(X_train, y_train)
    model.accuracy(X_train, y_train)
    model.accuracy(X_val, y_val)

    print("------------------------------------------------------------")

    print("Train accuracy:")
    print(model.accuracy(X_train, y_train))

    print("------------------------------------------------------------")

    print("Test accuracy:")
    print(model.accuracy(X_val, y_val))

    print("------------------------------------------------------------")

    print("Fairness measures:")
    model.print_fairness(X_val, y_val, x_sex_age)

if __name__ == "__main__":
    main()
