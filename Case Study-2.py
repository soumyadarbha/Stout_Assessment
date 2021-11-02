#!/usr/bin/env python
# coding: utf-8

# ## Case Study - 2

# In[42]:


# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# In[43]:


Total_Revenue=pd.read_csv("casestudy.csv")
Total_Revenue.head()


# In[44]:


Total_Revenue.tail()


# ### 1. Total revenue for the current year

# In[45]:


# Displaying the total revenue for each year i.e years - 2015, 2016 and 2017

Total_Revenue['year'].value_counts()


# In[46]:


print("Total Revenue for each Year:")
Total_Revenue.groupby('year').agg({'net_revenue':'sum'})


# ### 2. New Customer Revenue e.g., new customers not present in previous year only

# In[47]:


Total_Revenue_17 = Total_Revenue[Total_Revenue['year']==2017]
Total_Revenue_16 = Total_Revenue[Total_Revenue['year']==2016]
Total_Revenue_15 = Total_Revenue[Total_Revenue['year']==2015]


# In[48]:


# new customer revnue in 2017

Total_Revenue_17.loc[~Total_Revenue_17['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum()


# In[49]:


# new customer revnue in 2016

Total_Revenue_16.loc[~Total_Revenue_16['customer_email'].isin(Total_Revenue_15['customer_email']),'net_revenue'].sum()


# ### 3. Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# In[50]:


# Existing customer growth in 2017
Total_Revenue_17.loc[Total_Revenue_17['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum() - Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'net_revenue'].sum()


# In[51]:


# existing customer growth in 2016
Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_15['customer_email']),'net_revenue'].sum() - Total_Revenue_15.loc[Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum()


# ### 4. Revenue lost from attrition

# In[52]:


# Revenue lost from attrition for year 2017

Total_Revenue_16.loc[~Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'net_revenue'].sum()


# In[53]:


# Revenue lost from attrition for year 2016

Total_Revenue_15.loc[~Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum()


# ### 5. Existing Customer Revenue Current Year

# In[54]:


# existing customer revnue for current year 2017

Total_Revenue_17.loc[Total_Revenue_17['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum()


# In[55]:


# existing customer revnue for current year 2016

Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_15['customer_email']),'net_revenue'].sum()


# ### 6. Existing Customer Revenue Prior Year

# In[56]:


# existing customer revenue prior year 2016

Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'net_revenue'].sum()


# In[57]:


# existing customer revenue prior year 2015

Total_Revenue_15.loc[Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'net_revenue'].sum()


# ### 7. Total Customers Current Year

# In[58]:


# Total customers for current year 2017

Total_Revenue_17.loc[Total_Revenue_17['customer_email'].isin(Total_Revenue_16['customer_email']),'customer_email'].nunique()


# In[59]:


# Total customers for current year 2016

Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_15['customer_email']),'customer_email'].nunique()


# In[60]:


# Total customers for year 2017

Total_Revenue.loc[Total_Revenue['year']==2017,'customer_email'].nunique()


# In[61]:


# Total customers for year 2016

Total_Revenue.loc[Total_Revenue['year']==2016,'customer_email'].nunique()


# In[62]:


# Total customers for year 2015

Total_Revenue.loc[Total_Revenue['year']==2015,'customer_email'].nunique()


# ### 8. Total Customers Previous Year

# In[63]:


# Total Customers for previous year 2016

Total_Revenue_16.loc[Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'customer_email'].nunique()


# In[64]:


# Total Customers for previous year 2015

Total_Revenue_15.loc[Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'customer_email'].nunique()


# ### 9. New customers

# In[65]:


# new customers in year 2017

Total_Revenue_17.loc[~Total_Revenue_17['customer_email'].isin(Total_Revenue_16['customer_email']),'customer_email'].nunique()


# In[66]:


# new customers in year 2016

Total_Revenue_16.loc[~Total_Revenue_16['customer_email'].isin(Total_Revenue_15['customer_email']),'customer_email'].nunique()


# ### 10. Lost Customers

# In[67]:


# lost customers in year 2017

Total_Revenue_16.loc[~Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'customer_email'].nunique()


# In[68]:


# lost customers in year 2016

Total_Revenue_15.loc[~Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'customer_email'].nunique()


# In[69]:


# lost customers in year 2017

Total_Revenue_16.loc[~Total_Revenue_16['customer_email'].isin(Total_Revenue_17['customer_email']),'customer_email'].unique()


# In[70]:


# lost customers in year 2016

Total_Revenue_15.loc[~Total_Revenue_15['customer_email'].isin(Total_Revenue_16['customer_email']),'customer_email'].unique()


# ### Additionally, generate a few unique plots highlighting some information from the dataset. 

# In[71]:


# The below plot shows the trends of each column in the dataset for each year

import matplotlib.pyplot as plt
import pandas as pd

Total_Revenue.groupby('year').nunique().plot(kind='bar')
plt.show()


# In[72]:


# The below plot shows the frequency of net_revenue. As shown below, there is uniform distribution of net_revenue.

Total_Revenue[['net_revenue']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8)
plt.show()


# In[73]:


YearbyRevenue = Total_Revenue.groupby("year")["net_revenue"].sum().sort_values()


# In[74]:


YearbyRevenue


# In[75]:


# The below plot displays the total net_revenue for each year

YearbyRevenue.plot(kind="barh", fontsize=8)


# In[78]:


# The below plot indicates the number of times each customer_email has been repeated. 

Total_Revenue['customer_email'].value_counts().hist()


# In[85]:


# The below plot shows the top 10 customers with highest net_revenue (in descending order)

Top_Customers=Total_Revenue.groupby('customer_email').agg({'net_revenue':'sum'}).sort_values('net_revenue', ascending=False).head(10)
Top_Customers


# In[84]:


Top_Customers.plot(kind="bar", fontsize=8)


# ### Some of the interesting observations from above plots and graphs are :
# ### 1. The year 2017 has the maximum net_revenue. 
# ### 2. Maximum number of customer emails are unique. 
# ### 3. There is uniform distribution in the frequency of net_revenue.
# ### 4. Along with net_revenue, highest number of customer emails have been registered for the year 2017

# In[ ]:




