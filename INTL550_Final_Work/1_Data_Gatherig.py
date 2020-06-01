#To test my hypotheses mentioned in the report, firstly, I imported relavant
#World Bank data using World Bank API
import wbdata
import datetime
import pandas as pd

#Due to data availability, a cross-sectional study containing 36 OECD countries
#and 10 partnering countries for the year of 2018 has been designed.
data_date = (datetime.datetime(2018, 1, 1), datetime.datetime(2018, 12, 31))

wbdata.search_indicators('Unemployment') #SL.UEM.TOTL.ZS
wbdata.search_indicators('Tariff Rate') #TM.TAX.MRCH.WM.AR.ZS
wbdata.search_indicators('GDP Growth') #NY.GDP.MKTP.KD.ZG
wbdata.search_indicators('Fdi Inflows') #BX.KLT.DINV.WD.GD.ZS
wbdata.search_indicators('Fdi Outflows') #BM.KLT.DINV.WD.GD.ZS
wbdata.search_indicators('Exports of goods') #BX.GSR.GNFS.CD
wbdata.search_indicators('Imports of goods') #BM.GSR.GNFS.CD

df = wbdata.get_dataframe({"SL.UEM.TOTL.ZS" : "Unemployment",
                           "TM.TAX.MRCH.WM.AR.ZS" : "Tariff_rate",
                           "NY.GDP.MKTP.KD.ZG" : "Gdp_Growth",
                           "BX.KLT.DINV.WD.GD.ZS" : "Fdi_Inflows",
                           "BM.KLT.DINV.WD.GD.ZS" : "Fdi_Outflows",
                           "BX.GSR.GNFS.CD" : "Export",
                           "BM.GSR.GNFS.CD" : "Import" },
                           country={"AUS","AUT","BRA","BEL","CAN","CHE",
                           "CHL","CHN","COL","CRI","CZE","DEU","DNK",
                           "ESP","EST","FIN","FRA","GBR","GRC","HUN",
                           "IDN","IND","IRL","ISL","ISR","ITA","JPN",
                           "KOR","LTU","LUX","LVA","MEX","MYS","NLD",
                           "NOR","NZL","POL","PRT","RUS","SVK","SVN",
                           "SWE","THA","TUR","USA","ZAF"},
                           data_date=data_date)

#To test the Import/Export ratio hypothesis, Import numbers of the selected
#countires have been divided by their Export numbers by using dataframes.
#Also, since all the remaining independent variables are in percentages
#to normalize the data, the outcome ratio multiplied by 100.
df["Import/Export"] = 100 * df["Import"] / df["Export"]
df.drop(columns=["Import","Export"], inplace = True)

#Since the OECD does not present a handy API, I downloded Service Trade Restictiveness
#Index and formatted cells so that it can be merged with World Bank data.
#Then, I calculated the mean of service trade restrictiveness and turned them
#to percentages.
df_stri2018 = pd.read_csv("2_STRI2018.csv", index_col = "Country")
df_stri2018["Avg_stri"] = df_stri2018.mean(axis=1)*100

df = pd.merge(df_stri2018["Avg_stri"], df, left_index= True, right_index = True)

#Due to above-mentioned reasons, I applied same procedure for the OECD's
#Trade Union Density ratio.
df_t_union = pd.read_csv("3_T_Union.csv", index_col = "Country")

df = pd.merge(df, df_t_union, left_index= True, right_index = True)

#If I dropp null values, the data would shrink so much so that its explanatory
#power for a qunatitative analysis would be in danger. So, I filled null values
#with the means of every column.
df = df.fillna(df.mean())

#Finally, I wrote the final data to a csv.
df.to_csv("4_dataset.csv", index = True, index_label="Country")
