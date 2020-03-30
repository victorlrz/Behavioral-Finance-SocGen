import pandas as pd
import numpy as np
import random as rd

df_stock  = pd.read_excel (r'Data_Simulées_Projet.xlsx')
df_stock  = df_stock.sort_values(['LIBELLE','DATE'], ascending=True)

#Give us stock's names to display correctly values
def Tab_Name_of_Stock(data):
	Tab_Name = []
	Tab_Name.append(data.iloc[0,3])
	for i in range(1, data.shape[0]):
		if(data.iloc[i,3] != Tab_Name[-1]):
			Tab_Name.append(data.iloc[i,3])
	return Tab_Name	

#Compute the total number of transactions for stock s durind period q
def Tab_Nqs(data, q, Tab_Name):
	Tab_Nqs = []
	for i in range(len(Tab_Name)):
		data_selected = data.loc[data['LIBELLE'] == Tab_Name[i], ['DATE','LIBELLE','operation_nature']]#sub dataframe for only one stock
		Nqs = data_selected.shape[0]
		if(Nqs > q):
			Nqs = q
		Tab_Nqs.append(Nqs)
	return Tab_Nqs

#Compute the number of BUY transactions for stock s durind period q
def Tab_Bqs(data, q, Tab_Name):
	Tab_Bqs = []
	for i in range(len(Tab_Name)):
		data_selected = data.loc[(data['LIBELLE'] == Tab_Name[i]) & (data['operation_nature'] == 'Buy')]#sub dataframe for only one stock with only BUY transactions
		Bqs = data_selected.shape[0]
		if(Bqs > q):
			Bqs = q
		Tab_Bqs.append(Bqs)
	return Tab_Bqs	

#Compute Pi^q for each period q
def Compute_Rq(data, Tab_Bqs, Tab_Nqs):
	Sum_Bqs = np.sum(Tab_Bqs)
	Sum_Nqs = np.sum(Tab_Nqs)
	Rq = Sum_Bqs/Sum_Nqs
	return Rq

#Simulate Binomial law with parameters : B(Pi^q, N^qs) and return a Binomial B^qs
def Compute_Binomial_Bqs(Rq, Nqs):
	Binomial_Bqs = 0
	for i in range(Nqs):
		rand = rd.random()
		if(rand <= Rq):
			Binomial_Bqs +=1
	return Binomial_Bqs

#Compute expected value of |(Binomial B^qs/N^qs) - Pi^q|
def Compute_AFqs(data, Tab_Bqs, Tab_Nqs, Rq):
	Tab_AFqs = []
	for i in range(len(Tab_Nqs)):
		Binomial_Bqs = Compute_Binomial_Bqs(Rq, Tab_Nqs[i])
		AFqs = abs((Binomial_Bqs/Tab_Nqs[i]) - Rq)
		Tab_AFqs.append(AFqs)
	return Tab_AFqs

#Compute H1^qs
def Herding_effect_1(data, q):
    tab_Name = Tab_Name_of_Stock(data)
    tab_Bqs = Tab_Bqs(data, q, tab_Name)
    tab_Nqs = Tab_Nqs(data, q, tab_Name)
    Rq = Compute_Rq(data, tab_Bqs, tab_Nqs)
    tab_AFqs = Compute_AFqs(data, tab_Bqs, tab_Nqs, Rq)
    print('')
    print('Result Herding_effect_1 :')
    print('')
    for i in range(len(tab_Name)):
        Herding_effect_1 = (abs((tab_Bqs[i]/tab_Nqs[i])-Rq)-tab_AFqs[i])*100
        print(str(tab_Name[i]) + " : " + str(Herding_effect_1) + " %")

#Compute H2^qs
def Herding_effect_2(data, q):
    tab_Name = Tab_Name_of_Stock(data)
    tab_Bqs = Tab_Bqs(data, q, tab_Name)
    tab_Nqs = Tab_Nqs(data, q, tab_Name)
    Rq = Compute_Rq(data, tab_Bqs, tab_Nqs)
    print('')
    print('Result Herding_effect_2 :')
    print('')
    for i in range(len(tab_Name)):
        Herding_effect_2_numerator = (tab_Bqs[i] - Rq*tab_Nqs[i])**2 - tab_Nqs[i]*Rq*(1 - Rq)
        Herding_effect_2_denominator = tab_Nqs[i]*(tab_Nqs[i] - 1)
        Herding_effect_2 = (Herding_effect_2_numerator/Herding_effect_2_denominator)*100
        print(str(tab_Name[i]) + " : " + str(Herding_effect_2) + " %")

Herding_effect_1(df_stock, 4)
Herding_effect_2(df_stock, 4)