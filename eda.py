#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import GammaRegressor as gmreg
#import statsmodels.api as sm   

#read in csv dataset
real_est_zip = pd.read_csv("RDC_Inventory_Core_Metrics_Zip_History.csv", low_memory = False)
real_est_zip.head() 
real_est_zip.shape        #table dimension
real_est_zip.describe()   #summary statistics

#Find sum of missing values by column
real_est_zip.isnull().sum()

# remove columns from data that have mostly missing values
real_est_zip.drop(columns = ['flag', 'price_increased_count_mm', 'price_increased_count_yy'], axis = 1, inplace = True)

#drop all other missing rows from data
real_est_zip.dropna(inplace = True)

#make sure there are no null values in table
real_est_zip.isnull().sum()
real_est_zip.shape

#convert date to datetime value and generate year and month features
real_est_zip['YearMonth'] = pd.to_datetime(real_est_zip['month_date_yyyymm'], format = '%Y%m').dt.to_period('M')
real_est_zip['Year'] = real_est_zip['YearMonth'].dt.year
real_est_zip['Month'] = real_est_zip['YearMonth'].dt.month

#convert postal_code to numeric value
real_est_zip['postal_code'] = real_est_zip['postal_code'].astype('int')

#create a new column named state by extracting state from zip name
real_est_zip['State'] = real_est_zip['zip_name'].str[-2:]
real_est_zip['State'] = real_est_zip['State'].str.upper()

#create a new column named city by extracting city from zip name
real_est_zip['City'] = real_est_zip['zip_name'].str[:-4]
real_est_zip['City'] = real_est_zip['City'].str.capitalize()

#drop zip_name column since information is now derivable from columns City and State
real_est_zip.drop('zip_name', axis = 1, inplace = True)

#plot histograms to see feature distribution: they both have distributions between -1 and 1
plt.style.use('ggplot')
plt.hist(data = real_est_zip, x = "median_listing_price_mm", color = 'red', bins = 40)
plt.hist(data = real_est_zip, x = "median_listing_price_yy", color = 'blue', bins = 40)

plt.show

#view distribution of average_listing_price
sns.histplot(data = real_est_zip, x = real_est_zip["average_listing_price"], color = "blue", bins = 40)
#data was right skewed so I decided to plot a log transformation of it to see if it would be approximately normally
#distributed and the log transformation fixed the skewness of the variable
sns.histplot(data = real_est_zip, x = np.log(real_est_zip["average_listing_price"]), color = "blue", bins = 40)

#remove mm and yy columns since their values are not intuitive and they contribute no real explanation to the data
real_est_zip = real_est_zip[real_est_zip.columns.drop(list(real_est_zip.filter(regex='mm')))]
real_est_zip = real_est_zip[real_est_zip.columns.drop(list(real_est_zip.filter(regex='yy')))]

#plot scatterplots for all numeric columns vs. average_listing_price(target_variable)
num_cols = real_est_zip.select_dtypes([np.number]).columns.tolist()
count=1
plt.subplots(figsize=(50, 50))
for i in num_cols:
    plt.subplot(5,3,count)
    sns.scatterplot(data = real_est_zip, x = 'average_listing_price', y = real_est_zip[i], color = 'blue')
    count+=1

plt.show()
#plot scatterplots for Year and State features vs. average_listing_price(target_variable)
cat_cols = ["Year", "State"]

count=1
plt.subplots(figsize=(50, 50))
for i in cat_cols:
    plt.subplot(3,1,count)
    sns.boxplot(data = real_est_zip, x = real_est_zip[i], y = 'average_listing_price', color = 'red')
    count+=1

plt.show()

#from the scatterplots, we see that abouth 3 features are strongly correlated with price to include median_square_foot
#which is an intuitive result. It also seems like most variables like median_days_on_market have higher values for lower price
#listing and they decrease as the average listing price increases, this means houses that are listed for lower values
#typically are on the market for a longer time probably due to bad quality. For year, it appears 2021 has the largest variance in
#average listing price amongst all the years. For states, Michigan and Oklahoma seems to have the highest variance in average listing
#price. However from the box plots we can see a presence of several outliers

#Looking at the distribution of observations by state, for modeling purposes, we might think about giving more weight to
#states with fewer observations to ensure that we have balanced data.
real_est_zip["State"].value_counts()
real_est_zip["Year"].value_counts()
real_est_zip.dtypes

#create a region feature from the state feature, to see if there are any trends by region
north_east = ["ME", "VT", "NH", "NY", "PA", "MA", "RI", "NJ", "CT"]
mid_west = ["ND", "MN", "WI", "MI", "SD", "NE", "IA", "IL", "IN", "OH", "KS", "MO"]
south = ["OK", "TX", "AR", "LA", "MS", "AL", "TN", "KY", "WV","VA","DC", "MD", "DE", "NC", "SC", "GA", "FL"]
west = ["WA", "MT", "OR", "ID", "WY", "CA", "NV", "UT", "CO", "AZ", "NM", "AK", "HI"]
reg_dict = {'North East': north_east,'Midwest': mid_west, 'South': south,'West':west}
dict_cond_values= {key:real_est_zip['State'].isin(reg_dict[key]) for key in reg_dict}
real_est_zip['Region']=np.select(dict_cond_values.values(),dict_cond_values.keys())
real_est_zip.head()

#plot region against average price
sns.boxplot(data = real_est_zip, x = "Region", y = 'average_listing_price', color = 'red', width = 0.6)

#Even with a lot of outliers, we can see that real estate is most expensive in the west and cheapest in the midwest which
#is intuitive considering what states make up these regions.


