# Exploratory-Data-Analysis-on-Diwali-Sales-Dataset.

# Objective -

# Improve Customer Experience by Analyzing Sales Data.
# Increase Revenue.

# Install Libraries

!pip install Numpy
!pip install Pandas
!pip install Matplotlib
!pip install Seaborn

# import python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # visualizing data
%matplotlib inline
import seaborn as sns

# import csv file

df = pd.read_csv('Diwali Sales Data.csv', encoding= 'unicode_escape')
![Screenshot (1)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/078a4d31-a555-46f9-bc04-1575db510991)

df.shape
![Screenshot (5)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/6bd1271e-582e-4ca4-a07d-e6ce8870c500)


df.head()
![Screenshot (6)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/1f9f44d5-2592-4c5b-adaa-02a148292f9e)


df.info()
![Screenshot (33)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/8d6ed2e9-fe32-428f-862e-bf802075778d)


# Column Non-Null Count Dtype
--- ------ -------------- -----
0 User_ID 11251 non-null int64
1 Cust_name 11251 non-null object
2 Product_ID 11251 non-null object
3 Gender 11251 non-null object
4 Age Group 11251 non-null object
5 Age 11251 non-null int64
6 Marital_Status 11251 non-null int64
7 State 11251 non-null object
8 Zone 11251 non-null object
9 Occupation 11251 non-null object
10 Product_Category 11251 non-null object
11 Orders 11251 non-null int64
12 Amount 11239 non-null float64
13 Status 0 non-null float64
14 unnamed1 0 non-null float64
dtypes: float64(3), int64(4), object(8)
memory usage: 1.3+ MB


# drop unrelated/blank columns

df.drop(['Status', 'unnamed1'], axis=1, inplace=True)

# check for null values
pd.isnull(df).sum
![Screenshot (29)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/2e2bc1ec-9b67-4fe6-896f-98e7437b245f)


# drop null values
df.dropna(inplace=True)
![Screenshot (31)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/f869f868-1bc6-431b-a741-551c35cb911e)


# change data type
df['Amount'] = df['Amount'].astype('int')

df['Amount'].dtypes
![Screenshot (10)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/8e6f7024-2b14-4cfa-923a-2db61db8c861)

df.columns

# rename column
df.rename(columns= {'Marital_Status':'Shaadi'})
Out[7]:
User_ID 0
Cust_name 0
Product_ID 0
Gender 0
Age Group 0
Age 0
Marital_Status 0
State 0
Zone 0
Occupation 0
Product_Category 0
Orders 0
Amount 12

dtype: int64
dtype('int32')

df.columns
Index(['User_ID', 'Cust_name', 'Product_ID', 'Gender', 'Age Group', 'Age',
'Marital_Status', 'State', 'Zone', 'Occupation', 'Product_Category',
'Orders', 'Amount'],

dtype='object')
Out[12]:

User_ID Cust_name Product_ID Gender
Age
Group
Age Shaadi State Zone Occupation Product_Category0 1002903 Sanskriti P00125942 F 26-35 28 0 Maharashtra Western Healthcare Auto1 1000732 Kartik P00110942 F 26-35 35 1 Andhra Pradesh Southern Govt Auto2 1001990 Bindu P00118542 F 26-35 35 1 Uttar Pradesh Central Automobile Auto3 1001425 Sudevi P00237842 M 0-17 16 0 Karnataka Southern Construction Auto

# describe() method returns description of the data in the DataFrame (i.e. count, mean, s
td, etc)
df.describe()

# use describe() for specific columns
df[['Age', 'Orders', 'Amount']].describe()

# Exploratory Data Analysis
# Gender

3 1001425 Sudevi P00237842 M 0-17 16 0 Karnataka Southern Construction Auto4 1000588 Joni P00057942 M 26-35 28 1 Gujarat Western
Food
Processing
Auto... ... ... ... ... ... ... ... ... ... ...
11246 1000695 Manning P00296942 M 18-25 19 1 Maharashtra Western Chemical Office11247 1004089 Reichenbach P00171342 M 26-35 33 0 Haryana Northern Healthcare Veterinary11248 1001209 Oshin P00201342 F 36-45 40 0
Madhya
Pradesh
Central Textile Office11249 1004023 Noonan P00059442 M 36-45 37 0 Karnataka Southern Agriculture Office11250 1002744 Brumley P00281742 F 18-25 19 0 Maharashtra Western Healthcare OfficeUser_ID Cust_name Product_ID Gender
Age
Group
Age Shaadi State Zone Occupation Product_Category11239 rows × 13 columns
Out[13]:
User_ID Age Marital_Status Orders Amount
count 1.123900e+04 11239.000000 11239.000000 11239.000000 11239.000000
mean 1.003004e+06 35.410357 0.420055 2.489634 9453.610553
std 1.716039e+03 12.753866 0.493589 1.114967 5222.355168
min 1.000001e+06 12.000000 0.000000 1.000000 188.000000
25% 1.001492e+06 27.000000 0.000000 2.000000 5443.000000
50% 1.003064e+06 33.000000 0.000000 2.000000 8109.000000
75% 1.004426e+06 43.000000 1.000000 3.000000 12675.000000
max 1.006040e+06 92.000000 1.000000 4.000000 23952.000000
Out[14]:
Age Orders Amount
count 11239.000000 11239.000000 11239.000000
mean 35.410357 2.489634 9453.610553
std 12.753866 1.114967 5222.355168
min 12.000000 1.000000 188.000000
25% 27.000000 2.000000 5443.000000
50% 33.000000 2.000000 8109.000000
75% 43.000000 3.000000 12675.000000
max 92.000000 4.000000 23952.000000

# plotting a bar chart for Gender and it's count

ax = sns.countplot(x = 'Gender',data = df)
for bars in ax.containers:
ax.bar_label(bars
![Screenshot (14)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/a2d1235a-a0dc-4d90-b21d-149a54ef8a50)

# plotting a bar chart for gender vs total amount
sales_gen = df.groupby(['Gender'], as_index=False)['Amount'].sum().sort_values(by='Amoun
t', ascending=False)
sns.barplot(x = 'Gender',y= 'Amount' ,data = sales_gen)
Out[16]:
<Axes: xlabel='Gender', ylabel='Amount'>
![Screenshot (15)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/9884eacb-b1ac-4df0-ae8a-2da8ab2663aa)

# From above graphs we can see that most of the buyers are females and even the purchasing power of females are greater than men

# Age

ax = sns.countplot(data = df, x = 'Age Group', hue = 'Gender')
for bars in ax.containers:
ax.bar_label(bars)
![Screenshot (16)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/4f8f2a34-8eff-41ee-a2c6-9ba092bf54c8)


# Total Amount vs Age Group

sales_age = df.groupby(['Age Group'], as_index=False)['Amount'].sum().sort_values(by='Am
ount', ascending=False)
sns.barplot(x = 'Age Group',y= 'Amount' ,data = sales_age)
Out[18]:
<Axes: xlabel='Age Group', ylabel='Amount'>
![Screenshot (17)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/50bfdc05-4bc5-49e3-8c83-d61239f7a23c)

# From above graphs we can see that most of the buyers are of age group between 26-35 yrs female

# State
# total number of orders from top 10 states

sales_state = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orde
rs', ascending=False).head(10)
sns.set(rc={'figure.figsize':(15,5)})
sns.barplot(data = sales_state, x = 'State',y= 'Orders')
![Screenshot (18)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/22f85fc8-4180-4e7a-b982-09aebf684a7c)

# total amount/sales from top 10 states

sales_state = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amou
nt', ascending=False).head(10)
sns.set(rc={'figure.figsize':(15,5)})
sns.barplot(data = sales_state, x = 'State',y= 'Amount')
<Axes: xlabel='State', ylabel='Orders'>
![Screenshot (19)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/ea8a23c0-c215-46f6-9623-921edbf390bc)

sales_state = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)
sns.set(rc={'figure.figsize':(15,5)})
sns.barplot(data = sales_state, x = 'State',y= 'Amount')
<Axes: xlabel='State', ylabel='Amount'>
![Screenshot (20)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/87dbee5a-cbe2-4dfe-85b6-bfc97c48a69d)
# From above graphs we can see that most of the orders & total sales/amount are from Uttar Pradesh, Maharashtra and Karnataka respectively

# Marital Status
ax = sns.countplot(data = df, x = 'Marital_Status')
sns.set(rc={'figure.figsize':(7,5)})
for bars in ax.containers:
ax.bar_label(bars)
![Screenshot (21)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/7de94e73-3b58-45c4-a716-e295220a0bd5)


sales_state = df.groupby(['Marital_Status', 'Gender'], as_index=False)['Amount'].sum().s
ort_values(by='Amount', ascending=False)
sns.set(rc={'figure.figsize':(6,5)})
sns.barplot(data = sales_state, x = 'Marital_Status',y= 'Amount', hue='Gender')


<Axes: xlabel='Marital_Status', ylabel='Amount'>
# From above graphs we can see that most of the buyers are married (women) and they have high purchasing power

# Occupation
sns.set(rc={'figure.figsize':(20,5)})
ax = sns.countplot(data = df, x = 'Occupation')
for bars in ax.containers:
ax.bar_label(bars)
 ![Screenshot (22)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/549701d7-4878-4484-bdb7-c260a742d6fe)

sales_state = df.groupby(['Occupation'], as_index=False)['Amount'].sum().sort_values(by=
'Amount', ascending=False)
sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(data = sales_state, x = 'Occupation',y= 'Amount')
![Screenshot (26)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/c75fefc1-12fa-43c5-b1ed-fc916d6dba5d)

# From above graphs we can see that most of the buyers are working in IT, Healthcare and Aviation sector

# Product Category

<Axes: xlabel='Occupation', ylabel='Amount'>
sns.set(rc={'figure.figsize':(20,5)})
ax = sns.countplot(data = df, x = 'Product_Category')
for bars in ax.containers:
ax.bar_label(bars)

sales_state = df.groupby(['Product_Category'], as_index=False)['Amount'].sum().sort_valu
es(by='Amount', ascending=False).head(10)
sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(data = sales_state, x = 'Product_Category',y= 'Amount')
![Screenshot (23)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/5c56dd0b-a663-4e79-a4ab-f208fc11a8d3)

# From above graphs we can see that most of the sold products are from Food, Clothing and Electronics category

sales_state = df.groupby(['Product_ID'], as_index=False)['Orders'].sum().sort_values(by=
'Orders', ascending=False).head(10)
sns.set(rc={'figure.figsize':(20,5)})
sns.barplot(data = sales_state, x = 'Product_ID',y= 'Orders')
<Axes: xlabel='Product_Category', ylabel='Amount'>
![Screenshot (27)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/0c7a37d4-db0c-4732-86c3-4b256b228fe2)

<Axes: xlabel='Product_ID', ylabel='Orders'>
![Screenshot (28)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/ac8a71a7-8923-4106-9a8c-354afe8b8c64)

# top 10 most sold products (same thing as above)
fig1, ax1 = plt.subplots(figsize=(12,7))
df.groupby('Product_ID')['Orders'].sum().nlargest(10).sort_values(ascending=False).plot(
kind='bar')
![Screenshot (25)](https://github.com/sanketshintre/Exploratory-Data-Analysis-on-Diwali-Sales-Dataset/assets/123626990/9cbf4f27-d974-4c59-8c8c-69624696651c)



# Conclusion:
# Married women age group 26-35 yrs from UP, Maharastra and Karnataka working in IT, Healthcare and Aviationare more likely to buy products from Food, Clothing and Electronics category.
Thank you!

