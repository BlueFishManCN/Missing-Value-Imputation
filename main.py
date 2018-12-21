# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:29:27 2017
call function in main.py to process the gene expression data obtained from UCI datasets
@author: jerry
"""

import imputating as ip

ip.normalize('./data.csv', './normalized_data.csv', (0, 1))

ip.introMultipleMissingValues(
    './normalized_data.csv', missingPercentageRange=(0.01, 0.05, 0.1, 0.15, 0.2))

ip.ZeroI('./data.csv', './0.01.csv', (0, 1))
ip.ZeroI('./data.csv', './0.05.csv', (0, 1))
ip.ZeroI('./data.csv', './0.1.csv', (0, 1))
ip.ZeroI('./data.csv', './0.15.csv', (0, 1))
ip.ZeroI('./data.csv', './0.2.csv', (0, 1))
print("1")
ip.ColumnMeanI('./data.csv', './0.01.csv', (0, 1))
ip.ColumnMeanI('./data.csv', './0.05.csv', (0, 1))
ip.ColumnMeanI('./data.csv', './0.1.csv', (0, 1))
ip.ColumnMeanI('./data.csv', './0.15.csv', (0, 1))
ip.ColumnMeanI('./data.csv', './0.2.csv', (0, 1))
print("2")
ip.RowMeanI('./data.csv', './0.01.csv', (0, 1))
ip.RowMeanI('./data.csv', './0.05.csv', (0, 1))
ip.RowMeanI('./data.csv', './0.1.csv', (0, 1))
ip.RowMeanI('./data.csv', './0.15.csv', (0, 1))
ip.RowMeanI('./data.csv', './0.2.csv', (0, 1))
print("3")
ip.ColumnMedianI('./data.csv', './0.01.csv', (0, 1))
ip.ColumnMedianI('./data.csv', './0.05.csv', (0, 1))
ip.ColumnMedianI('./data.csv', './0.1.csv', (0, 1))
ip.ColumnMedianI('./data.csv', './0.15.csv', (0, 1))
ip.ColumnMedianI('./data.csv', './0.2.csv', (0, 1))
print("4")
ip.RowMedianI('./data.csv', './0.01.csv', (0, 1))
ip.RowMedianI('./data.csv', './0.05.csv', (0, 1))
ip.RowMedianI('./data.csv', './0.1.csv', (0, 1))
ip.RowMedianI('./data.csv', './0.15.csv', (0, 1))
ip.RowMedianI('./data.csv', './0.2.csv', (0, 1))
print("5")
ip.KNNI1('./data.csv', './0.05.csv', (0, 1), 1)
ip.KNNI1('./data.csv', './0.05.csv', (0, 1))
ip.KNNI1('./data.csv', './0.05.csv', (0, 1), 10)
ip.KNNI1('./data.csv', './0.05.csv', (0, 1), 20)
print("6")
ip.KNNI1('./data.csv', './0.01.csv', (0, 1))
ip.KNNI1('./data.csv', './0.05.csv', (0, 1))
ip.KNNI1('./data.csv', './0.1.csv', (0, 1))
ip.KNNI1('./data.csv', './0.15.csv', (0, 1))
ip.KNNI1('./data.csv', './0.2.csv', (0, 1))
print("7")
ip.KNNI2('./data.csv', './0.01.csv', (0, 1))
ip.KNNI2('./data.csv', './0.05.csv', (0, 1))
ip.KNNI2('./data.csv', './0.1.csv', (0, 1))
ip.KNNI2('./data.csv', './0.15.csv', (0, 1))
ip.KNNI2('./data.csv', './0.2.csv', (0, 1))
