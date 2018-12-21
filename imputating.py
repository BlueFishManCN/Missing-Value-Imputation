# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:53:02 2017

@author: jerry
"""

import math
import random
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def normalize(inputFileName='./input.csv', outputFileName='./output.csv', scalerRange=(0, 1)):
    """
    Created on Thu Nov  9 19:55:46 2017
    normalize data
    @author: jerry
    """
    original_data = pd.read_csv(inputFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    normalized_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    normalized_data.to_csv(outputFileName)


def introSingleMissingValues(inputFileName='./input.csv', outputFileName='./output.csv', missingPercentage=0.01):
    """
    Created on Thu Nov  9 20:56:06 2017
    intro missing values to data
    @author: jerry
    """
    original_data = pd.read_csv(inputFileName, index_col=0)
    time = int(original_data.size * missingPercentage)
    while time:
        x = random.randint(0, original_data.shape[0] * 0.9)
        y = random.randint(0, original_data.shape[1] - 1)
        original_data[original_data.columns[y]
                      ][original_data.index[x]] = np.nan
        time -= 1
    original_data.to_csv(outputFileName)


def introMultipleMissingValues(inputFileName='./input.csv', missingPercentageRange=(0.01)):
    """
    Created on Thu Nov  9 20:56:06 2017
    intro multi missing values to data
    @author: jerry
    """
    source_original_data = pd.read_csv(inputFileName, index_col=0)
    for mpri in missingPercentageRange:
        original_data = source_original_data.copy()
        time = int(original_data.size * mpri)
        while time:
            x = random.randint(0, original_data.shape[0] * 0.9)
            y = random.randint(0, original_data.shape[1] - 1)
            original_data[original_data.columns[y]
                          ][original_data.index[x]] = np.nan
            time -= 1
        original_data.to_csv('./' + str(mpri) + '.csv')


def ZeroI(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1)):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the zero imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = missing_data.fillna(0)  # handel the missing data
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'zeorI'
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./zeroI_' + missingDataFileName[2:])


def ColumnMeanI(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1)):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the column mean imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = missing_data.copy()
    for i in imputated_data.columns:
        temp = np.mean(imputated_data[i].dropna())
        imputated_data[i] = imputated_data[i].fillna(temp)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'ColumnMeanI'
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./ColumnMeanI_' + missingDataFileName[2:])


def RowMeanI(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1)):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the row mean imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = missing_data.copy()
    for i in imputated_data.index:
        temp = np.mean(imputated_data.loc[i].dropna())
        imputated_data.loc[i] = imputated_data.loc[i].fillna(temp)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'RowMeanI'
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./RowMeanI_' + missingDataFileName[2:])


def ColumnMedianI(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1)):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the column median imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = missing_data.copy()
    for i in imputated_data.columns:
        temp = np.median(imputated_data[i].dropna())
        imputated_data[i] = imputated_data[i].fillna(temp)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'ColumnMedianI'
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./ColumnMedianI_' + missingDataFileName[2:])


def RowMedianI(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1)):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the row median imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = missing_data.copy()
    for i in imputated_data.index:
        temp = np.median(imputated_data.loc[i].dropna())
        imputated_data.loc[i] = imputated_data.loc[i].fillna(temp)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'RowMedianI'
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./RowMedianI_' + missingDataFileName[2:])


def KNNI1(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1), K_neighbors=5):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the KNNI1 imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = pd.DataFrame([], columns=missing_data.columns)
    neigh = KNeighborsRegressor(
        n_neighbors=K_neighbors, weights='distance', n_jobs=-1)
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    c_data = pd.DataFrame([], columns=missing_data.columns)
    for i in missing_data.index:
        if True not in list(missing_data.loc[i].isnull()):
            c_data = c_data.append(missing_data.loc[i])

    for i in missing_data.index:
        if True not in list(missing_data.loc[i].isnull()):
            imputated_data = imputated_data.append(missing_data.loc[i])
        else:
            m_data = missing_data.loc[i].copy()
            m_dropna_data = missing_data.loc[i].dropna()
            for m in missing_data.columns:
                if pd.isnull(m_data[m]):
                    x = ss_X.fit_transform(c_data[m_dropna_data.index])
                    y = ss_y.fit_transform(np.array(c_data[m]).reshape(-1, 1))
                    p = ss_X.transform(np.array(m_dropna_data).reshape(1, -1))
                    neigh.fit(x, y)
                    m_data[m] = ss_y.inverse_transform(neigh.predict(p))
            imputated_data = imputated_data.append(m_data)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'KNNI1_' + str(K_neighbors)
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./KNNI1_' + str(K_neighbors) +
                          '_' + missingDataFileName[2:])


def KNNI2(originalDataFileName='./original_data.csv', missingDataFileName='./missing_data.csv', scalerRange=(0, 1), K_neighbors=5):
    """
    Created on Fri Nov 10 14:34:26 2017
    define the KNNI2 imputation alogrithm and evaluate its accracy
    @author: jerry
    """
    original_data = pd.read_csv(originalDataFileName, index_col=0)
    scaler = MinMaxScaler(feature_range=scalerRange)
    original_data = pd.DataFrame(scaler.fit_transform(
        original_data), index=original_data.index, columns=original_data.columns)
    missing_data = pd.read_csv(missingDataFileName, index_col=0)

    # head
    imputated_data = pd.DataFrame([], columns=missing_data.columns)
    neigh = KNeighborsRegressor(
        n_neighbors=K_neighbors, weights='distance', n_jobs=-1)
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    for i in missing_data.index:
        if True not in list(missing_data.loc[i].isnull()):
            imputated_data = imputated_data.append(missing_data.loc[i])
        else:
            m_data = missing_data.loc[i].copy()
            m_dropna_data = missing_data.loc[i].dropna()
            missing_dropna_data = missing_data[m_dropna_data.index]
            c_data = pd.DataFrame([], columns=missing_data.columns)
            for m in missing_dropna_data.index:
                if m != i and True not in list(missing_dropna_data.loc[m].isnull()):
                    c_data = c_data.append(missing_data.loc[m])

            for m in missing_data.columns:
                if pd.isnull(m_data[m]):
                    temp = pd.DataFrame([], columns=missing_data.columns)
                    for n in c_data.index:
                        if not pd.isnull(c_data[m][n]):
                            temp = temp.append(c_data.loc[n])
                    x = ss_X.fit_transform(temp[m_dropna_data.index])
                    y = ss_y.fit_transform(np.array(temp[m]).reshape(-1, 1))
                    p = ss_X.transform(np.array(m_dropna_data).reshape(1, -1))
                    neigh.fit(x, y)
                    m_data[m] = ss_y.inverse_transform(neigh.predict(p))
            imputated_data = imputated_data.append(m_data)
    # tail

    a = []
    b = []
    for i in imputated_data.columns:
        for j in imputated_data.index:
            if pd.isnull(missing_data[i][j]):
                a.append(original_data[i][j])
                b.append(imputated_data[i][j])
    evaluation = pd.read_csv('./evaluation.csv')
    c = pd.Series([None, None, None, None, None, None], index=[
                  'approach', 'missingPercentage', 'RMSE', 'RMAE', 'MSE', 'MAE'])
    c['approach'] = 'KNNI2_' + str(K_neighbors)
    c['missingPercentage'] = missingDataFileName
    c['RMSE'] = math.sqrt(mean_squared_error(a, b))
    c['RMAE'] = math.sqrt(mean_absolute_error(a, b))
    c['MSE'] = mean_squared_error(a, b)
    c['MAE'] = mean_absolute_error(a, b)
    evaluation = evaluation.append(c, ignore_index=True)
    evaluation.to_csv('./evaluation.csv', index=False)

    imputated_data = pd.DataFrame(scaler.inverse_transform(
        imputated_data), index=original_data.index, columns=original_data.columns)
    imputated_data.to_csv('./KNNI2_' + str(K_neighbors) +
                          '_' + missingDataFileName[2:])
