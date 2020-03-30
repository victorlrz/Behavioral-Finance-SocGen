import pandas as pd
import numpy as np
from decimal import *
from datetime import datetime
from dateutil import relativedelta


def main():

    # import the portfolio
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Data_Simulées_Projet.csv'
    df = pd.read_csv(filename, sep =';') # dataframe

    # import data bloomberg
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Data_Bloomberg.csv'
    df_bloomberg = pd.read_csv(filename, sep =';', index_col = 0) # dataframe 

    # import benchmark
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Data_Benchmark.csv'
    df_market = pd.read_csv(filename, sep =';', names = ['Date', 'Benchmark']) # dataframe

    # import transco
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Transco.csv'
    transco = pd.read_csv(filename, sep =';') # dataframe
    print(transco)

    # new dataset with only the columns we need
    df = df[['DATE', 'ISIN', 'operation_nature', 'quantity_ccy', 'last_price']]
    df['DATE'] = df['DATE'].str.replace('7', '9')
    df = pd.merge(df, transco, left_on='ISIN', right_on='ISIN')
    

    # Data bloomberg from string to decimal
    for i in range(df_bloomberg.shape[1]):
        # Convert last prices from string to decimal
        for j in range(df_bloomberg.shape[0]):
            df_bloomberg.iloc[j, i] = Decimal(df_bloomberg.iloc[j, i].replace(',', '.'))
    print(df_bloomberg)


    # Reorganize benchmark dataframe (delete NaN, convert date, set index)
    df_market = df_market.dropna()
    df_market['Date'] = pd.to_datetime(df_market['Date']).dt.strftime('%d/%m/%Y')
    df_market = df_market.set_index('Date')
    print(df_market)


    # Daily returns for bloomberg data and benchmark
    ri = daily_returns(df_bloomberg) # Returns for each stock i
    rm = daily_returns(df_market) # Market returns

    '''# the period Z
    start_date = datetime.strptime(df['DATE'].iloc[0], '%d/%m/%Y')
    end_date = datetime.strptime(df['DATE'].iloc[-1], '%d/%m/%Y')

    r = relativedelta.relativedelta(end_date, start_date)
    T = r.months + (12 * r.years) # number of months in the period
    print(T)'''

    # Average Cumulative Abnormal Returns (ACAR)
    ar, acar = avg_cum_abnormal_returns(ri, rm)
    print(acar)

    # list of all Ticker Bloomberg
    list_ticker = df['Ticker Bloomberg'].unique()

    for i in range (len(list_ticker)):

        # Create a subdataset for each stock i
        subdata = data_by_ticker(df, list_ticker[i])
        ar_i = ar[list_ticker[i]]
        indicator(subdata, ar_i)


def data_by_ticker(df, ticker):

    # Create a new dataframe with all the operations with the same ISIN
    subdata = df[df['Ticker Bloomberg'] == ticker]

    # Reindex the data
    subdata.reset_index(drop=True, inplace=True)
    
    # Convert last prices from string to decimal
    for i in range(subdata.shape[0]):
        subdata.loc[i, 'last_price'] = Decimal(subdata.loc[i, 'last_price'].replace(',', '.'))

    #print(subdata)
    return subdata


def daily_returns(df):

    # Create an empty dataframe with the same axes of bloomberg dataframe 
    df_returns = pd.DataFrame(index = df.index, columns = df.columns)
    df_returns = df_returns[:-1] # drop last row because we cannot calculate after J-1
    
    # Compute daily returns for each stock : (valeur J+1 / valeur J) - 1
    for i in range(df.shape[1]):
        for j in range(df.shape[0] - 1):
            daily_return = (df.iloc[j + 1, i] / df.iloc[j, i] - 1) * 100
            df_returns.iloc[j, i] = round(daily_return,2)

    print(df_returns)
    return df_returns


def avg_cum_abnormal_returns(ri, rm):

    # Create an empty dataframe with the same axes of bloomberg returns dataframe
    df_ar_returns = pd.DataFrame(index = ri.index, columns = ri.columns)

    # Join ri and rm by date
    # Add benchmark values to bloomberg data, associated by date
    df_join = ri.join(rm)
    print(df_join)

    # list of stocks i
    header = ri.columns.values.tolist()

    # AR : Abnormal Return (for each stock i)
    for i in range(len(header)):        
        df_ar_returns[header[i]] = df_join[header[i]].astype(float) - df_join['Benchmark'] # AR = Ri - Rm

    # CAR : Cumulative Abnormal Return
    df_returns = df_ar_returns.cumsum(axis = 0) # Cumulative sum for each column
    print(df_returns)

    # ACAR : Average Cumulative Abnormal Return
    df_returns = df_returns.mean(axis = 0) # Average of CAR for each column

    return df_ar_returns, df_returns


def indicator(subdata, ar_i):

    subdata = subdata.set_index('DATE')
    #print(subdata)
    #print(ar_i)
    df_indicator = pd.DataFrame(index = subdata.index, columns = ['Result', 'Operation', 'Score'])
    df_merge = pd.merge(ar_i, subdata['operation_nature'], right_index=True, left_index=True, how='outer')
    df_merge = df_merge.fillna('Nothing')

    df_join = subdata.join(ar_i)
    df_join = df_join.dropna()
    #print(df_join)

    for index, row in df_join.iterrows():
        if row[row['Ticker Bloomberg']] < 0:
            df_indicator.loc[index, 'Result'] = 'Underreaction'
        elif row[row['Ticker Bloomberg']] > 0:
            df_indicator.loc[index, 'Result'] = 'Overreaction'
        else:
            df_indicator.loc[index, 'Result'] = 'Nothing'

    df_indicator['Operation'] = df_join['operation_nature']

    for index, row in df_join.iterrows():
        if row[row['Ticker Bloomberg']] != 0 and row['operation_nature'] == 'Buy':
            df_indicator.loc[index, 'Score'] = 1 
        else: 
            df_indicator.loc[index, 'Score'] = 0

    '''for index, row in df_merge.iterrows():
        if row[0] < 0:
            df_indicator.loc[index, 'Result'] = 'Underreaction'
        elif row[0] > 0:
            df_indicator.loc[index, 'Result'] = 'Overreaction'
        else:
            df_indicator.loc[index, 'Result'] = 'Nothing'

    df_indicator['Operation'] = df_merge['operation_nature']

    for index, row in df_merge.iterrows():
        if row[0] != 0 and row[1] == 'Buy':
            df_indicator.loc[index, 'Score'] = 1 
        else: 
            df_indicator.loc[index, 'Score'] = 0'''

    print(df_indicator)

main()