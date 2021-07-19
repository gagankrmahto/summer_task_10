#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# In[2]:


from datetime import datetime
import pytz

def parse_str(x):
    """
    Returns the string delimited by two characters.

    Example:
        `>>> parse_str('[my string]')`
        `'my string'`
    """
    return x[1:-1]

def parse_datetime(x):
    '''
    Parses datetime with timezone formatted as:
        `[day/month/year:hour:minute:second zone]`

    Example:
        `>>> parse_datetime('13/Nov/2015:11:45:42 +0000')`
        `datetime.datetime(2015, 11, 3, 11, 45, 4, tzinfo=<UTC>)`

    Due to problems parsing the timezone (`%z`) with `datetime.strptime`, the
    timezone will be obtained using the `pytz` library.
    '''
    dt = datetime.strptime(x[1:-7], '%d/%b/%Y:%H:%M:%S')
    dt_tz = int(x[-6:-3])*60+int(x[-3:-1])
    return dt.replace(tzinfo=pytz.FixedOffset(dt_tz))


# In[3]:


import re
import pandas as pd

data = pd.read_csv(
    'https://my-dataset-collection.s3.ap-south-1.amazonaws.com/stjgps_access.log.1',
    sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
    engine='python',
    na_values='-',
    header=None,
    usecols=[0, 3, 4, 5, 6, 7, 8],
    names=['ip', 'time', 'request', 'status', 'size', 'referer', 'user_agent'],
    converters={'time': parse_datetime,
                'request': parse_str,
                'status': int,
                'size': int,
                'referer': parse_str,
                'user_agent': parse_str})


# In[4]:


# data.head(5)


# In[5]:


# ip = data['ip'].unique()


# In[6]:


# ip


# In[7]:


# data.info()


# In[8]:


clean_data = data[["ip","status"]]


# In[9]:


# clean_data


# In[10]:


ip_status_data = clean_data.groupby(["ip",'status']).size().reset_index(name="visited")


# In[11]:


# ip_status_data


# In[12]:


ip_status_data['ip'].value_counts()


# In[13]:


ip_status_data['status'].value_counts()


# In[14]:


final_train_data = ip_status_data.drop(['ip'],axis=1)


# In[15]:


# final_train_data


# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[17]:


data_scaled = sc.fit_transform(final_train_data)


# In[18]:


data_scaled


# In[ ]:





# In[19]:


wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
# for i in range(1, 20):  
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 32)  
#     kmeans.fit(data_scaled)  
#     wcss_list.append(kmeans.inertia_)  
# plt.plot(range(1, 20), wcss_list)  
# plt.title('The Elbow Method Graph')  
# plt.xlabel('Number of clusters(k)')  
# plt.ylabel('wcss_list')  
# plt.show() 


# In[20]:


model = KMeans(n_clusters=5)


# In[21]:


pred = model.fit_predict(data_scaled)


# In[22]:


final_predicted_data = pd.DataFrame(data_scaled,columns=['Status_Scaled','Frequency_scaled'])
final_predicted_data['Clusters'] = pred
final_data = pd.concat([ip_status_data,final_predicted_data],axis=1)


# In[23]:





# In[ ]:





# In[24]:


# final_data.plot.bar(x='ip',y='visited',rot=0)


# In[25]:


# plt.scatter(final_data['ip'],final_data['visited'],c=final_data['Clusters'])


# In[26]:


Block_IP = []
Block_ip_list = []
for key, value in final_data.iloc[:,[0,2,5]].iterrows():
    if value.visited > 40:
        Block_IP.append([value.Clusters,value.ip, value.visited])
        Block_ip_list.append(value.ip)
# Block_IP
# Block_ip_list

# final_list = []
# for sub_list in Block_IP:
#     final_list.append(Block_IP[0][1])
# final_list[0]


# In[27]:


# final_data


# In[28]:


print(Block_ip_list)


# In[ ]:




