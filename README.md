# summer_task_10
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
```


```python
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
```


```python
import re
import pandas as pd

data = pd.read_csv(
    'apache2/stjgps_access.log.1',
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
```


```python
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ip</th>
      <th>time</th>
      <th>request</th>
      <th>status</th>
      <th>size</th>
      <th>referer</th>
      <th>user_agent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66.249.75.8</td>
      <td>2021-07-13 00:09:29+05:30</td>
      <td>GET /school_club/ HTTP/1.1</td>
      <td>200</td>
      <td>12102</td>
      <td>NaN</td>
      <td>Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Bu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66.249.75.26</td>
      <td>2021-07-13 00:19:25+05:30</td>
      <td>GET /school_strength/ HTTP/1.1</td>
      <td>200</td>
      <td>11590</td>
      <td>NaN</td>
      <td>Mozilla/5.0 (compatible; Googlebot/2.1; +http:...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66.249.75.8</td>
      <td>2021-07-13 00:26:18+05:30</td>
      <td>GET / HTTP/1.1</td>
      <td>200</td>
      <td>18137</td>
      <td>NaN</td>
      <td>Mozilla/5.0 (compatible; Googlebot/2.1; +http:...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>192.241.223.158</td>
      <td>2021-07-13 00:42:18+05:30</td>
      <td>GET /actuator/health HTTP/1.1</td>
      <td>404</td>
      <td>5028</td>
      <td>NaN</td>
      <td>Mozilla/5.0 zgrab/0.x</td>
    </tr>
    <tr>
      <th>4</th>
      <td>178.159.37.139</td>
      <td>2021-07-13 00:43:28+05:30</td>
      <td>GET /contact.php HTTP/1.0</td>
      <td>404</td>
      <td>5246</td>
      <td>https://stjgps.org/contact.php</td>
      <td>Mozilla/5.0 (Windows NT 10.0; Win64; x64) Appl...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ip = data['ip'].unique()
```


```python
# ip
```


```python
# data.info()
```


```python
clean_data = data[["ip","status"]]
```


```python
clean_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ip</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66.249.75.8</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66.249.75.26</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66.249.75.8</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>192.241.223.158</td>
      <td>404</td>
    </tr>
    <tr>
      <th>4</th>
      <td>178.159.37.139</td>
      <td>404</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1620</th>
      <td>106.207.23.21</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1621</th>
      <td>66.249.75.25</td>
      <td>304</td>
    </tr>
    <tr>
      <th>1622</th>
      <td>106.207.23.21</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1623</th>
      <td>106.207.23.21</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1624</th>
      <td>106.207.23.21</td>
      <td>408</td>
    </tr>
  </tbody>
</table>
<p>1625 rows × 2 columns</p>
</div>




```python
ip_status_data = clean_data.groupby(["ip",'status']).size().reset_index(name="visited")
```


```python
ip_status_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ip</th>
      <th>status</th>
      <th>visited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.15.175.155</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.36.111.167</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103.159.178.199</td>
      <td>404</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>104.143.83.241</td>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>106.207.23.21</td>
      <td>200</td>
      <td>108</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>74.120.14.55</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>226</th>
      <td>91.188.215.198</td>
      <td>301</td>
      <td>1</td>
    </tr>
    <tr>
      <th>227</th>
      <td>91.247.220.24</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>228</th>
      <td>94.16.121.91</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>229</th>
      <td>94.16.121.91</td>
      <td>404</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 3 columns</p>
</div>




```python
ip_status_data['ip'].value_counts()
```




    202.168.84.90    7
    106.207.23.21    5
    157.42.211.38    5
    66.249.75.25     4
    157.42.99.210    4
                    ..
    171.13.14.52     1
    23.251.102.90    1
    113.31.108.14    1
    171.13.14.76     1
    157.35.238.50    1
    Name: ip, Length: 153, dtype: int64




```python
ip_status_data['status'].value_counts()
```




    200    109
    404     75
    301     18
    408     11
    304      7
    403      4
    400      3
    206      2
    302      1
    Name: status, dtype: int64




```python
final_train_data = ip_status_data.drop(['ip'],axis=1)
```


```python
final_train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status</th>
      <th>visited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>404</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200</td>
      <td>108</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>226</th>
      <td>301</td>
      <td>1</td>
    </tr>
    <tr>
      <th>227</th>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>228</th>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>229</th>
      <td>404</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 2 columns</p>
</div>




```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
```


```python
data_scaled = sc.fit_transform(final_train_data)
```


```python
print(data_scaled)
```

    [[-0.98277442 -0.35493994]
     [-0.98277442 -0.29641938]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.29641938]
     [-0.98277442  5.90676042]
     [-0.92015998 -0.17937825]
     [ 0.07123527 -0.12085769]
     [ 1.14611643 -0.06233712]
     [ 1.18785939  0.2302657 ]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.29641938]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.17937825]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.23789882]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.17937825]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.23789882]
     [-0.98277442 -0.29641938]
     [ 0.07123527 -0.29641938]
     [ 1.14611643 -0.23789882]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.17937825]
     [-0.98277442 -0.29641938]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.06233712]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.29641938]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  0.69843021]
     [ 1.14611643 -0.35493994]
     [ 1.10437347 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.10437347 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442  2.86369108]
     [ 1.18785939 -0.35493994]
     [ 1.18785939 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  3.56593785]
     [ 1.14611643 -0.29641938]
     [ 1.18785939 -0.23789882]
     [-0.98277442  1.63475924]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.18785939 -0.06233712]
     [-0.98277442  5.6141576 ]
     [ 0.07123527 -0.29641938]
     [ 0.10254248  0.93251247]
     [ 1.14611643 -0.35493994]
     [ 1.18785939 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442  0.8739919 ]
     [ 0.07123527 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.18785939 -0.29641938]
     [-0.98277442  0.05470401]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  0.05470401]
     [ 1.14611643 -0.35493994]
     [-0.98277442  0.81547134]
     [-0.98277442 -0.06233712]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442  0.40582739]
     [-0.98277442  0.2302657 ]
     [-0.98277442 -0.23789882]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.12085769]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.23789882]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.23789882]
     [-0.98277442 -0.17937825]
     [-0.98277442 -0.29641938]
     [ 0.07123527 -0.35493994]
     [ 1.10437347 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.29641938]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.13568069 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  3.56593785]
     [-0.92015998  0.17174513]
     [ 0.07123527 -0.29641938]
     [ 0.081671   -0.29641938]
     [ 0.10254248  2.04440319]
     [ 1.14611643 -0.35493994]
     [ 1.18785939 -0.06233712]
     [-0.98277442 -0.35493994]
     [-0.98277442  0.17174513]
     [-0.98277442  0.34730683]
     [-0.98277442  0.05470401]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.14611643  0.81547134]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.13568069 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.17937825]
     [ 1.13568069 -0.35493994]
     [-0.98277442  0.05470401]
     [ 0.10254248 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  0.05470401]
     [-0.98277442  0.05470401]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.23789882]
     [-0.98277442  0.17174513]
     [-0.98277442  2.16144431]
     [-0.98277442  0.11322457]
     [-0.98277442  2.27848544]
     [-0.98277442  0.75695078]
     [-0.98277442  2.10292375]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.06233712]
     [ 1.14611643  1.16659472]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.23789882]
     [ 0.07123527 -0.23789882]
     [ 1.14611643 -0.29641938]
     [-0.98277442  4.61930801]
     [ 1.14611643 -0.35493994]
     [ 1.18785939 -0.35493994]
     [-0.98277442  4.38522575]
     [ 1.18785939 -0.29641938]
     [-0.98277442  5.67267816]
     [ 0.10254248  0.93251247]
     [ 1.14611643 -0.29641938]
     [ 1.18785939 -0.12085769]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [ 1.13568069 -0.29641938]
     [ 1.14611643 -0.29641938]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.29641938]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442  1.51771811]
     [ 0.07123527 -0.35493994]
     [ 0.10254248 -0.23789882]
     [ 1.14611643 -0.23789882]
     [-0.98277442  2.10292375]
     [ 0.10254248 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442  1.51771811]
     [ 0.07123527 -0.35493994]
     [ 0.10254248 -0.23789882]
     [ 1.14611643 -0.17937825]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.17937825]
     [ 1.14611643 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.29641938]
     [-0.98277442 -0.35493994]
     [ 1.14611643 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 0.07123527 -0.35493994]
     [-0.98277442 -0.35493994]
     [-0.98277442 -0.35493994]
     [ 1.14611643  0.05470401]]
    


```python

```


```python
wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
for i in range(1, 20):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 32)  
    kmeans.fit(data_scaled)  
    wcss_list.append(kmeans.inertia_)  
plt.plot(range(1, 20), wcss_list)  
plt.title('The Elbow Method Graph')  
plt.xlabel('Number of clusters(k)')  
plt.ylabel('wcss_list')  
plt.show() 
```


    
![png](output_19_0.png)
    



```python
model = KMeans(n_clusters=5)
```


```python
pred = model.fit_predict(data_scaled)
```


```python
final_predicted_data = pd.DataFrame(data_scaled,columns=['Status_Scaled','Frequency_scaled'])
final_predicted_data['Clusters'] = pred
final_data = pd.concat([ip_status_data,final_predicted_data],axis=1)
```


```python
final_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ip</th>
      <th>status</th>
      <th>visited</th>
      <th>Status_Scaled</th>
      <th>Frequency_scaled</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.15.175.155</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.36.111.167</td>
      <td>200</td>
      <td>2</td>
      <td>-0.982774</td>
      <td>-0.296419</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103.159.178.199</td>
      <td>404</td>
      <td>1</td>
      <td>1.146116</td>
      <td>-0.354940</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>104.143.83.241</td>
      <td>200</td>
      <td>2</td>
      <td>-0.982774</td>
      <td>-0.296419</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>106.207.23.21</td>
      <td>200</td>
      <td>108</td>
      <td>-0.982774</td>
      <td>5.906760</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>74.120.14.55</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>226</th>
      <td>91.188.215.198</td>
      <td>301</td>
      <td>1</td>
      <td>0.071235</td>
      <td>-0.354940</td>
      <td>4</td>
    </tr>
    <tr>
      <th>227</th>
      <td>91.247.220.24</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>228</th>
      <td>94.16.121.91</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>229</th>
      <td>94.16.121.91</td>
      <td>404</td>
      <td>8</td>
      <td>1.146116</td>
      <td>0.054704</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 6 columns</p>
</div>




```python

```


```python
final_data.plot.bar(x='ip',y='visited',rot=0)
```




    <AxesSubplot:xlabel='ip'>




    
![png](output_25_1.png)
    



```python
plt.scatter(final_data['ip'],final_data['visited'],c=final_data['Clusters'])
```




    <matplotlib.collections.PathCollection at 0x159793fdf10>




    
![png](output_26_1.png)
    



```python
Block_IP = []
Block_ip_list = []
for key, value in final_data.iloc[:,[0,2,5]].iterrows():
    if value.visited > 40:
        Block_IP.append([value.Clusters,value.ip, value.visited])
        Block_ip_list.append(value.ip)
Block_IP
Block_ip_list

# final_list = []
# for sub_list in Block_IP:
#     final_list.append(Block_IP[0][1])
# final_list[0]

```




    ['106.207.23.21',
     '157.35.231.66',
     '157.35.241.23',
     '157.42.211.38',
     '202.168.84.90',
     '202.168.84.90',
     '42.236.10.114',
     '42.236.10.75',
     '42.236.10.93',
     '47.9.241.160',
     '47.9.244.121',
     '49.37.76.200',
     '66.249.75.26']




```python
final_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ip</th>
      <th>status</th>
      <th>visited</th>
      <th>Status_Scaled</th>
      <th>Frequency_scaled</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.15.175.155</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.36.111.167</td>
      <td>200</td>
      <td>2</td>
      <td>-0.982774</td>
      <td>-0.296419</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103.159.178.199</td>
      <td>404</td>
      <td>1</td>
      <td>1.146116</td>
      <td>-0.354940</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>104.143.83.241</td>
      <td>200</td>
      <td>2</td>
      <td>-0.982774</td>
      <td>-0.296419</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>106.207.23.21</td>
      <td>200</td>
      <td>108</td>
      <td>-0.982774</td>
      <td>5.906760</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>225</th>
      <td>74.120.14.55</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>226</th>
      <td>91.188.215.198</td>
      <td>301</td>
      <td>1</td>
      <td>0.071235</td>
      <td>-0.354940</td>
      <td>4</td>
    </tr>
    <tr>
      <th>227</th>
      <td>91.247.220.24</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>228</th>
      <td>94.16.121.91</td>
      <td>200</td>
      <td>1</td>
      <td>-0.982774</td>
      <td>-0.354940</td>
      <td>2</td>
    </tr>
    <tr>
      <th>229</th>
      <td>94.16.121.91</td>
      <td>404</td>
      <td>8</td>
      <td>1.146116</td>
      <td>0.054704</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>230 rows × 6 columns</p>
</div>




```python
Block_ip_list
```




    ['106.207.23.21',
     '157.35.231.66',
     '157.35.241.23',
     '157.42.211.38',
     '202.168.84.90',
     '202.168.84.90',
     '42.236.10.114',
     '42.236.10.75',
     '42.236.10.93',
     '47.9.241.160',
     '47.9.244.121',
     '49.37.76.200',
     '66.249.75.26']




```python

```
