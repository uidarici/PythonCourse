# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 02:01:46 2020

@author: udarici19
"""

import wbdata
import datetime

data_date = (datetime.datetime(1960, 1, 1), datetime.datetime(2019, 1, 1))
wbdata.search_countries('United States') #USA

wbdata.search_indicators('Labor force participation rate, female') #SL.TLF.CACT.FE.NE.ZS
wbdata.search_indicators('Fertility Rate') #SP.DYN.TFRT.IN
wbdata.search_indicators('GDP per capita') #NY.GDP.PCAP.CD

df = wbdata.get_dataframe({"NY.GDP.PCAP.CD":"GDPPC","SP.DYN.TFRT.IN":"FR","SL.TLF.CACT.FE.NE.ZS":"FLFP"},
                          country="USA", data_date=data_date)

df.to_csv('data.csv')
df.describe()
