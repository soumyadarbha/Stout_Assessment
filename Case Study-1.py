#!/usr/bin/env python
# coding: utf-8

# ### Case Study - 1

# In[1]:


#importing the required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Reading the csv file 

loans_data=pd.read_csv("loans_full_schema.csv")
type(loans_data)


# ## 1. Describe the dataset and any issues with it.

# #### As shown below, the dataset contains 10000 rows and 55 coulmns.

# In[3]:


#Printing the number of rows of dataset
print(len(loans_data))

#Printing the dimensionality of the dataset
print(loans_data.shape)


# #### Displaying all the columns and their data types as shown below.

# In[3]:


loans_data.info()


# In[4]:


# Displaying the first 5 rows of dataset

loans_data.head()


# In[6]:


# Displaying the last five rows of dataset

loans_data.tail()


# #### Displaying the descriptive statistics for all columns as displayed below.

# In[7]:


loans_data.describe()


# ### Issues with respect to the dataset:

# #### 1. Detecting the null values:

# In[8]:


# To detect the missing values which returns True for NA values else False.

loans_data.isna()


# In[9]:


# This prints the total number of missing values for each column.

loans_data.isna().sum()


# In[10]:


# Total number of missing values in the entire dataset.

loans_data.isna().sum().sum()


# ### Solutions to fix the issues wrt null values:

# In[11]:


# Drops the rows containg the null values

clean_loans_data=loans_data.dropna()
clean_loans_data.shape


# ###### As shown above, the number of rows has been reduced to 201. Hence, by dropping the rows containing null values can result in loss of necessary information.

# In[12]:


# The other way to fix this issue is by filling the null values with a fixed value like zero

clean_data=loans_data.fillna(value=0, inplace=True)


# #### 2. Detecting duplicates:

# In[13]:


# Apart from missing data, there can also be duplicate rows in a dataframe.

duplicate_loans_data= loans_data[loans_data.duplicated()]
duplicate_loans_data.shape


# ### Solution to fix the issues wrt duplicates:

# In[14]:


# But if there are any duplicates, it can be removed as shown below.

loans_data.drop_duplicates()


# #### 3. Another aspect would be detecting outliers in dataset which is an important segment in Exploratory Data Analysis. Outliers can play havoc when we want to apply Machine Learning algorithms for predictions.

# ## 2. Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. 

# ### 1. Trends of Annual Income and Loan Amounts

# #### The below distribution plot the variation in the distribution of Annual Income and Loan Amounts. We can see from the below graphs that the Annual Income mostly lie between 0 and 500000 and the maximum number of Loan Amount taken is approx 10000.

# In[80]:


Amounts = ['annual_income','loan_amount']
for i in Amounts:
    sns.distplot(loans_data[i],color="red",bins=20)
    plt.title("Trends of "+ i, fontsize=15)
    plt.xlabel(i)
    plt.ylabel('count')
    plt.show()


# ### 2. Percentage of Number of Applicants Statewise Distribution Raking

# #### The below pie chart gives the ranking of top 5 states with highest percentage of number of applicants.

# In[16]:


# Number of times each state has been mentioned in the dataset

loans_data['state'].value_counts()


# In[17]:


# Percentage of Number of Applicants State wise distribution ranking (Top 5 states)

size=[1330,806,793,732,382]
labels="3","2","4","5","1"
colors=['grey','lightpink','yellow','lightblue','purple']

graph = plt.Circle((0,0), 0.2, color='white')

plt.rcParams['figure.figsize']=(10,10)
plt.pie(size, colors=colors, labels=labels, shadow=True, autopct="%.2f%%")
plt.title("Number of Applicants State wise distribution ranking", fontsize= 20)
p=plt.gcf()
p.gca().add_artist(graph)
plt.show()


# ### 3. Distribution of Application Types

# #### The below pie chart shows the percentage of distribution of application types i.e Individual and Joint application types. From below graph, we can say that maximum percentage of application type is Individual application type.

# In[18]:


# Distribution of Application type

loans_data['application_type'].value_counts()


# In[19]:


size=[8505,1495]
labels="Individual", "Joint"
colors= ["grey","lightgrey"]
explode= [0,0.1]
plt.rcParams['figure.figsize']= (10,10)
plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
plt.title('Distribution of Application type', fontsize=15)
plt.axis('off')
plt.show()


# ### 4. Count of Loan Status

# #### The below bar graph gives us the count of each loan status. And from the below graph, we can say that current loan status has the highest count.

# In[20]:


# Loan status

sns.countplot(loans_data['loan_status'],color="lightblue")
plt.title("Loan Status", fontsize=15)
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()


# ### 5. Correlation Matrix for all variables in the dataset

# #### The below Correlation Matrix gives the correlation of all variables in the dataset.

# In[21]:


#Correlation Matrix: Correlation of all variables

fig, ax = plt.subplots(figsize=(30,30))
sns.heatmap(loans_data.corr(), annot=True,cmap="mako")


# ### 6. Word Cloud

# #### The below word cloud displays the most prominent or frequent words in the entire dataset.

# In[22]:


#Word Cloud 

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
import json
import numpy as np

with open('loans_full_schema.json', errors="ignore") as f:
    data = json.load(f)

# get the data in json format
text = []
for row in data:
    if (row != ""):
        text.append(row)

while('' in text) :
    text.remove('')

# text = np.delete(text['type'], 1, 0)
# print(text)


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=2000,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(text)


# ### 7. Dashboard of Loans Trends:

# #### - The below dashboard gives the background information for different states which is displaying the common information for the selected states in the dashboard. 
# #### - The dashboard has 4 graphs included. 
# #### - The first graph displays the loan amounts range in each state. The second graph displays the range of interest rates for each state. The third graph displays the range of annual income in each state. Finally, the fourth graph displays the debt range for each state.
# #### - Have to apply the below graphs to multiple states to see the changes in the trends of the bar graphs.

# In[ ]:


import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
global data
data = loans_data


#assets_external_path='/style.css'
app = dash.Dash(__name__)
server = app.server

global dict_products
def create_dict_list_of_product():
    dictlist = []
    unique_list = loans_data.state.unique()
    for state in unique_list:
        dictlist.append({'label': state, 'value': state})
    return dictlist

def dict_product_list(dict_list):
    product_list = []
    for dict in dict_list:
        product_list.append(dict.get('value'))
    return product_list


dict_products = create_dict_list_of_product()



app.layout = html.Div([
    html.Div([
        html.H1('Loans Trend Dashboard'),
        html.H2('Choose a State'),
        dcc.Dropdown(
            id='state-dropdown',
            options=[{'label':'Delaware','value':'Delaware'},
{'label':'Pennsylvania','value':'Pennsylvania'},
{'label':'New Jersey','value':'New Jersey'},

{'label':'Georgia','value':'GA'},
{'label':'Connecticut','value':'CT'},
{'label':'Massachusetts','value':'MA'},
{'label':'Maryland','value':'MD'},
{'label':'South Carolina','value':'SC'},
{'label':'New Hampshire','value':'NH'},
{'label':'Virginia','value':'VA'},
{'label':'New York','value':'NY'},
{'label':'North Carolina','value':'NC'},
{'label':'Rhode Island','value':'RI'},
{'label':'Vermont','value':'VT'},
{'label':'Kentucky','value':'KY'},
{'label':'Tennessee','value':'TN'},
{'label':'Ohio','value':'OH'},
{'label':'Louisiana','value':'LA'},
{'label':'Indiana','value':'IN'},
{'label':'Mississippi','value':'MS'},
{'label':'Illinois','value':'IL'},
{'label':'Alabama','value':'AL'},
{'label':'Maine','value':'ME'},
{'label':'Missouri','value':'MO'},
{'label':'Arkansas','value':'AR'},
{'label':'Michigan','value':'MI'},
{'label':'Florida','value':'FL'},
{'label':'Texas','value':'TX'},
{'label':'Iowa','value':'IA'},
{'label':'Wisconsin','value':'WI'},
{'label':'California','value':'CA'},
{'label':'Minnesota','value':'MN'},
{'label':'Oregon','value':'OR'},
{'label':'Kansas','value':'Kansas'},
{'label':'West Virginia','value':'WV'},
{'label':'Nevada','value':'NV'},
{'label':'Nebraska','value':'NE'},
{'label':'Colorado','value':'CO'},
{'label':'North Dakota','value':'ND'},
{'label':'South Dakota','value':'SD'},
{'label':'Montana','value':'MT'},
{'label':'Washington','value':'WA'},
{'label':'Idaho','value':'ID'},
{'label':'Wyoming','value':'WY'},
{'label':'Utah','value':'UT'},
{'label':'Oklahoma','value':'OK'},
{'label':'New Mexico','value':'NM'},
{'label':'Arizona','value':'AZ'},
{'label':'Alaska','value':'AK'},
{'label':'Hawaii','value':'HI'}],
            multi=True,
            value = ["GA"],
            searchable = True,
        ),

    ], style={'width': '40%', 'display': 'inline-block'}),
    html.Div([
        html.H2('Background Information of Selected States'),
        html.Table(id='my-table'),
        html.P(''),
    ], style={'width': '55%', 'float': 'right', 'display': 'inline-block'}),
    html.Div([
        html.H2('Counts of Loan Amounts  '),
        dcc.Graph(id='loanamount-graph'),
        html.P('')
    ], style={'width': '50%',  'display': 'inline-block'}),

    html.Div([
        html.H2('Counts of Interest Rates'),
        dcc.Graph(id='intrate-graph'),
        html.P('')
    ], style={'width': '50%',  'display': 'inline-block'}),

html.Div([
    html.H2('Counts of Annual Income'),
    dcc.Graph(id='other-graph'),
    html.P('')
], style={'width': '50%',  'display': 'inline-block'}),

html.Div([
    html.H2('Counts of Debts to Income'),
    dcc.Graph(id='multiple-graph'),
    html.P('')
], style={'width': '50%',  'display': 'inline-block'}),



])

@app.callback(Output('my-table', 'children'), [Input('state-dropdown', 'value')])
def generate_table(selected_dropdown_value, max_rows=5):

    df_filter = data[(data['state'].isin(selected_dropdown_value))]


    return [html.Tr([html.Th(col) for col in df_filter.columns])] + [html.Tr([
        html.Td(df_filter.iloc[i][col]) for col in df_filter.columns])
        for i in range(min(len(df_filter), max_rows))]


@app.callback(Output('loanamount-graph', 'figure'), [Input('state-dropdown', 'value')])

def update_graph(selected_dropdown_value):

    fig = loans_data.loc[(loans_data['state'].isin(selected_dropdown_value))]

    fig1 = px.bar(fig, x="state", y ='loan_amount')

    return fig1
@app.callback(Output('intrate-graph', 'figure'), [Input('state-dropdown', 'value')])

def update_graph(selected_dropdown_value):

    fig = loans_data.loc[(loans_data['state'].isin(selected_dropdown_value))]

    fig1 = px.bar(fig, x="state", y ='interest_rate')

    return fig1

@app.callback(Output('other-graph', 'figure'), [Input('state-dropdown', 'value')])

def update_graph(selected_dropdown_value):

    fig = loans_data.loc[(loans_data['state'].isin(selected_dropdown_value))]

    fig1 = px.bar(fig, x="state", y ='annual_income')

    return fig1

@app.callback(Output('multiple-graph', 'figure'), [Input('state-dropdown', 'value')])

def update_graph(selected_dropdown_value):

    fig = loans_data.loc[(loans_data['state'].isin(selected_dropdown_value))]

    fig1 = px.bar(fig, x="state", y ='debt_to_income')

    return fig1





if __name__ == '__main__':
    app.run_server(debug=False)


# ### 3. Create a feature set and create a model which predicts interest rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.

# ### Linear Regression Model

# In[23]:


from sklearn.preprocessing import LabelEncoder 
import matplotlib.pylab as plt 
import numpy as np 
from scipy import sparse 
from sklearn.datasets import make_classification, make_blobs, load_boston 
from sklearn.decomposition import PCA 
from sklearn.model_selection import ShuffleSplit, train_test_split 
from sklearn import metrics 
from sklearn.model_selection import learning_curve 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import GradientBoostingRegressor 
from pprint import pprint 
import pandas as pd 
import urllib 
import seaborn 


# Converting the string type of data to numeric data type

loans_data=loans_data._convert(numeric=True) 
loans_data.head(2) 


# In[24]:


# Loading the interest rate into y variable

y=loans_data.interest_rate.values

# Performing Data Cleaning by dropping interest_rate and all the non-numeric columns

del loans_data['interest_rate']

loans_data=loans_data.drop(['emp_title','state','homeownership','verified_income','verification_income_joint','loan_purpose','application_type','grade','sub_grade','issue_month','loan_status','initial_listing_status','disbursement_method'],axis=1)


# In[31]:


print(y)


# In[73]:


# Loading the features/dataset values into the variable x

x=loans_data.values


# In[74]:


# Now training the test split

X_train, X_test, y_train, y_test = train_test_split(x,y)

# Fitting the Linear Regression model to the training set

linr=LinearRegression().fit(X_train, y_train)

# Printing the parameters we have learned

print ("Coefficients (theta_1..theta_n)")
print (linr.coef_)
print ("Y Intercept(theta0)")
print (linr.intercept_)

print ("R-squared for Train: %.2f" %linr.score(X_train, y_train))
print ("R-squared for Test: %.2f" %linr.score(X_test, y_test))


# In[76]:


# Fitting the Linear regression model with normalize=True to the training set

linr=LinearRegression(normalize=True).fit(X_train, y_train)

# Getting the parameters we have learned

print ("Coefficients (theta_1..theta_n)")
print (linr.coef_)
print ("Y Intercept(theta0)") 
print (linr.intercept_)

print ("R-squared for Train: %.2f" %linr.score(X_train, y_train))
print ("R-squared for Test: %.2f" %linr.score(X_test, y_test))


# #### The data cleansing to be performed are remove the missing or null values. Drop the interest rate after assigning the values to Y variable and delete all the non-numeric values from the dataset.Remove the special characters if there are any. Finally, convert all the values in the dataset to numberic to make sure that every value is numeric type. The mentioned steps have been shown above.

# ### 4. Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.

# ### In the above models, the R-squared value on the test set is about 72%, which is not great but understandable considering the data must be much more sophisticated than a straight line. The only other thing we can do with this regressor is to normalize the data before training so that all values are in the same range from 0 to 1. If I had more time, I would explore more sophisticated regressors and convert the non-numeric/string data types to numeric values while building the model.

# In[1]:





# In[ ]:




