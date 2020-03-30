import pandas as pd
import numpy as np
from decimal import *

def main():
    # import the full dataset
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Data_Simulées_Projet.csv'
    df = pd.read_csv(filename, sep =';') # dataframe
    print(df)

    # import data bloomberg
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Data_Bloomberg.csv'
    df_bloomberg = pd.read_csv(filename, sep =';', index_col = 0) # dataframe 
    df_bloomberg = df_bloomberg.tail(1)
    bloomberg_columns = df_bloomberg.columns

    # import transco
    filename = r'C:\Users\Valentine\Documents\ESILV\Année 4\PI²\Python\Transco.csv'
    transco = pd.read_csv(filename, sep =';') # dataframe

    # new dataset with only the columns we need
    df = df[['DATE', 'ISIN', 'operation_nature', 'quantity_ccy', 'last_price']]
    print(df)    

    # Data bloomberg from string to decimal
    for i in range(df_bloomberg.shape[1]):
        isin = transco.loc[transco['Ticker Bloomberg'] == bloomberg_columns[i]]
        df_bloomberg = df_bloomberg.rename(columns={bloomberg_columns[i]: isin.iloc[0]['ISIN']})

        # Convert last prices from string to decimal
        for j in range(df_bloomberg.shape[0]):
            df_bloomberg.iloc[j, i] = Decimal(df_bloomberg.iloc[j, i].replace(',', '.'))

    # list of all ISIN
    list_isin = df['ISIN'].unique()

    methods = ['FIFO', 'LIFO', 'Average']

    for i in range(len(methods)):

        # create a dataset for the output
        output_columns = ['ISIN', 'Total Gains', 'Total Losses', 'Number of cases', 'Number of operations', 'Gain/Loss', 'Monetary sum']
        df_output = pd.DataFrame(index=np.arange(len(list_isin) + 1), columns=output_columns)
        df_output['ISIN'] = np.append(list_isin, 'Total')

        df_output, gains, losses, paper_gains, paper_losses = calculation_method(df, df_bloomberg, list_isin, methods[i], df_output)

        print(df_output)
        
        print('\nTotal', methods[i], 'Gains:', gains, 'and Total', methods[i], 'Losses:', losses)
        print('Total', methods[i], 'Paper Gains:', paper_gains, 'and Total', methods[i], 'Paper Losses:', paper_losses, '\n')

        pgr = gains / (gains + paper_gains) * 100
        plr = losses / (losses + paper_losses) * 100
        WeberCamererDei = (gains - losses) / (gains + losses) * 100
        DharZhuDei = ((gains / losses) - (paper_gains / paper_losses)) * 100

        print(methods[i], 'PGR:', pgr, '% ; ', methods[i], 'PLR:', plr, '%')
        print('Disposition effect:', pgr - plr, '%')
        print('Disposition effect WeberCamerer measure :', WeberCamererDei, '%')
        print('Disposition effect DharZhu measure :', DharZhuDei , '%')  


def data_by_isin(df, isin):

    # Create a new dataframe with all the operations with the same ISIN
    subdata = df[df['ISIN'] == isin]

    # Reindex the data
    subdata.reset_index(drop=True, inplace=True)
    
    # Convert last prices from string to decimal
    for i in range(subdata.shape[0]):
        subdata.loc[i, 'last_price'] = Decimal(subdata.loc[i, 'last_price'].replace(',', '.'))

    #print(subdata)
    return subdata


def unique_operation(subdata, operation):

    # Check if a subdataframe contains only Buy or Sell operations
    # That means we can't calculate any realized gains or losses
    # But only paper gains or losses 
    for index, row in subdata.iterrows():
        if row['operation_nature'] == operation:
            return False 
    
    return True


def greater_than(first_in, first_out, column):

    # Check if the quantity or the price is greater for the first in or first out operation
    if (first_in.loc[first_in.index[0], column] > first_out.loc[first_out.index[0], column]):
        return first_in
    else:
        return first_out


def realized_profit(subdata, num_of_gains, num_of_losses, method, price):

    if method == 'FIFO':
        # select the first entrance's row (BUY) where quantity is not null
        first_in = subdata[(subdata['operation_nature'] == 'Buy') & (subdata['quantity_ccy'] != 0)].head(1)
        #print('FIRST IN:\n', first_in)
        entrance = first_in

        # if there is a buy operation with a quantity != 0, choose the first next sell operation with quantity != 0
        if not entrance.empty:
            first_out = subdata[(subdata['operation_nature'] == 'Sell') & (subdata['quantity_ccy'] != 0) & (subdata.index > entrance.index[0])].head(1)  
            #print('FIRST OUT:\n', first_out)
        else:
            first_out = pd.DataFrame(index=range(1,2)) # create a default empty dataframe

    
    elif method == 'LIFO':

        first_out = subdata[(subdata['operation_nature'] == 'Sell') & (subdata['quantity_ccy'] != 0)].head(1)
        #print('FIRST OUT:\n', first_out)

        if not first_out.empty:
            # select the last entrance's row (BUY) where quantity is not null
            last_in = subdata[(subdata['operation_nature'] == 'Buy') & (subdata['quantity_ccy'] != 0) & (subdata.index < first_out.index[0])].tail(1)
            #print('LAST IN:\n', last_in)     
            entrance = last_in
        else:
            entrance = pd.DataFrame(index=range(1,2)) # create a default empty dataframe
    

    # if there is a sell operation
    if not first_out.empty and not entrance.empty:

        # Max quantity between BUY and SELL
        max_quantity = greater_than(entrance, first_out, 'quantity_ccy')
        difference_quantity = abs(entrance.loc[entrance.index[0], 'quantity_ccy'] - first_out.loc[first_out.index[0], 'quantity_ccy'])

        # Max price between BUY and SELL 
        max_price = greater_than(entrance, first_out, 'last_price')
        difference_price = abs(entrance.loc[entrance.index[0], 'last_price'] - first_out.loc[first_out.index[0], 'last_price'])

        # Change the number of quantities

        # The operation which has the smaller quantity between buy and loss turn to 0 and this quantity becomes gains or losses
        # The other one gets the difference between the two quantitiess
        if max_quantity.loc[max_quantity.index[0], 'operation_nature'] == 'Buy':                
            subdata.loc[entrance.index[0], 'quantity_ccy'] = difference_quantity
            subdata.loc[first_out.index[0], 'quantity_ccy'] = 0
        else:
            subdata.loc[entrance.index[0], 'quantity_ccy'] = 0
            subdata.loc[first_out.index[0], 'quantity_ccy'] = difference_quantity

        #print(subdata)
        
        # If there is a gain or a loss
        if difference_price != 0:

            # If the buying price is greater than selling price
            if max_price.loc[max_price.index[0], 'operation_nature'] == 'Buy':

                # increment number of losses            
                num_of_losses += min(entrance.loc[entrance.index[0], 'quantity_ccy'], first_out.loc[first_out.index[0], 'quantity_ccy'])
                price -= min(entrance.loc[entrance.index[0], 'quantity_ccy'], first_out.loc[first_out.index[0], 'quantity_ccy']) * difference_price
            else:
                # increment number of gains
                num_of_gains += min(entrance.loc[entrance.index[0], 'quantity_ccy'], first_out.loc[first_out.index[0], 'quantity_ccy']) 
                price += min(entrance.loc[entrance.index[0], 'quantity_ccy'], first_out.loc[first_out.index[0], 'quantity_ccy']) * difference_price

            #print('Gains:', num_of_gains, 'Losses:', num_of_losses, '\n')
        return realized_profit(subdata, num_of_gains, num_of_losses, method, price)
    
    else:
        return subdata, num_of_gains, num_of_losses, price


def paper_profit(subdata, df_bloomberg, total_paper_gains, total_paper_losses, price):

    # Initialization: number of paper gains and losses in quantity
    num_of_paper_gains = 0
    num_of_paper_losses = 0

    # determine the current price for each ISIN 
    current_price = df_bloomberg.iloc[0][subdata.iloc[0]['ISIN']]

    # for each operation of the dataframe, we compare its price with the current price
    for index, row in subdata.iterrows():        

        # if the operation is a buy
        if row['operation_nature'] == 'Buy' and row['quantity_ccy'] != 0:

            # if its price is smaller than the current price
            if row['last_price'] < current_price:
                num_of_paper_gains += row['quantity_ccy'] # we have a potential gain
                price += row['quantity_ccy'] * abs(row['last_price'] - current_price)

            # if it's greater
            elif row['last_price'] > current_price:
                num_of_paper_losses += row['quantity_ccy'] # we have a potential loss
                price -= row['quantity_ccy'] * abs(row['last_price'] - current_price)

            subdata.loc[index, 'quantity_ccy'] = 0

            # if the price is equal we don't take it in account because we have neither a gain nor a loss

    total_paper_gains += num_of_paper_gains
    total_paper_losses += num_of_paper_losses

    return num_of_paper_gains, num_of_paper_losses, total_paper_gains, total_paper_losses, price


def calculation_method(df, df_bloomberg, list_isin, method, df_output):
    total_gains = 0
    total_losses = 0
    total_paper_gains = 0
    total_paper_losses = 0
    column_names = df_output.columns

    # for each ISIN, we compute its realized gains, realized losses, paper gains, paper losses 
    for i in range (len(list_isin)):

        num_of_gains = 0
        num_of_losses = 0
        price = 0
        
        # Create a subdataset for each isin and each method
        subdata = data_by_isin(df, list_isin[i])

        # Check if a ISIN is only buy or sell
        unique_sell = unique_operation(subdata, 'Buy')
        unique_buy = unique_operation(subdata, 'Sell')

        #print()

        # Only ISIN with buy & sell
        if not unique_buy | unique_sell:
            
            # FIFO and LIFO methods
            if method == 'FIFO' or method == 'LIFO':
                # do algorithm to compute realized profits - FIFO method
                subdata, num_of_gains, num_of_losses, price = realized_profit(subdata, num_of_gains, num_of_losses, method, price)
                total_gains += num_of_gains
                total_losses += num_of_losses
            
            # Average method
            else: 
                subdata, num_of_gains, num_of_losses, price = average(subdata, num_of_gains, num_of_losses, price)
                total_gains += num_of_gains
                total_losses += num_of_losses


        # If there are still some buys without sell, compute their paper profit
        if subdata['quantity_ccy'].sum() != 0:
            # do algorithm to compute paper profits
            num_of_paper_gains, num_of_paper_losses, total_paper_gains, total_paper_losses, price = paper_profit(subdata, df_bloomberg, total_paper_gains, total_paper_losses, price)
        
        df_output = output(df_output, subdata, i, num_of_gains, num_of_losses, num_of_paper_gains, num_of_paper_losses, price)

    # Total calculation for each column
    for j in range(df_output.shape[1]): 
        if j != 0:
            df_output.loc[df_output.shape[0] - 1, column_names[j]] = df_output[column_names[j]].sum()
    
    # Export the output dataframe to csv
    df_output.to_csv(method + '.csv', sep=';', index=False)

    return df_output, total_gains, total_losses, total_paper_gains, total_paper_losses


def average(subdata, num_of_gains, num_of_losses, price):
    
    buy_quantity = 0
    buy_price = 0
    average_price = 0

    for index, row in subdata.iterrows():

        if row['operation_nature'] == 'Buy':
            buy_quantity += row['quantity_ccy']
            buy_price += row['quantity_ccy'] * row['last_price']
            subdata.loc[index, 'quantity_ccy'] = 0      
            
            average_price = buy_price / buy_quantity            

        else:
            if average_price < row['last_price']:
                num_of_gains += row['quantity_ccy']
                price += row['quantity_ccy'] * abs(average_price - row['last_price'])
            elif average_price > row['last_price']:
                num_of_losses += row['quantity_ccy'] 
                price -= row['quantity_ccy'] * abs(average_price - row['last_price'])
            
            buy_quantity -= row['quantity_ccy']
            subdata.loc[index, 'quantity_ccy'] = 0
            

    # If there are still some buys without sell, put the corresponding remaining quantities
    if buy_quantity != 0:
        last_buy = subdata[(subdata['operation_nature'] == 'Buy')].tail(1)
        subdata.loc[last_buy.index[0], 'quantity_ccy'] = buy_quantity
        subdata.loc[last_buy.index[0], 'last_price'] = average_price

    #print(subdata)

    return subdata, num_of_gains, num_of_losses, price


def output(df_output, subdata, i, total_gains, total_losses, total_paper_gains, total_paper_losses, price):
    gains = total_gains + total_paper_gains
    losses = total_losses + total_paper_losses

    if gains > losses:
        df_output.loc[i, 'Gain/Loss'] = 1
    else:
        df_output.loc[i, 'Gain/Loss'] = 0

    df_output.loc[i, 'Total Gains'] = gains
    df_output.loc[i, 'Total Losses'] = losses
    df_output.loc[i, 'Number of operations'] = len(subdata)
    df_output.loc[i, 'Number of cases'] = gains + losses
    df_output.loc[i, 'Monetary sum'] = round(price, 2)
    
    return df_output


main()