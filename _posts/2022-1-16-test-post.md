---
layout: post
title: Blog Post 1
---


### 1. Create a Database

>Because we are going to create a database which is include three tables,
so we should load these three data first. Then, add them into our database.
First we can import some libraries and create an empty database.

```python
import pandas as pd
import sqlite3
conn = sqlite3.connect("database.db")
```
>Load data one by one.

```python
# load temperature data.
temperatures_iter = pd.read_csv("temps.csv", chunksize = 100000)
temperatures = temperatures_iter.__next__()
# add temperature into my data base.
for temperatures in temperatures_iter:
    temperatures.to_sql("temperatures", conn, if_exists = "append", index = False)

# load stations data.
stations = pd.read_csv("station-metadata.csv")
# add stations into my data base
stations.to_sql("stations", conn, if_exists = "replace", index = False)

# load countries data.
countries = pd.read_csv("countries.csv")
countries.rename(columns = {"ISO 3166":"ISO_3166", "FIPS 10-4": "FIPS_10-4"})
# add countires into my data base
countries.to_sql("countries", conn, if_exists = "replace", index = False)
```
>Last step, let's check what our database includes.

```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
```
```
[('temperatures',), ('stations',), ('countries',)]
```
<br />

---

### 2. Write a Query Function
>We need to write CMD to extract data from sql.

```python
def query_climate_database(country, year_begin, year_end, month):
    df = pd.read_csv("temps.csv")
    df["FIPS 10-4"] = df["ID"].str[0:2]
    df = pd.merge(df, countries, on = ["FIPS 10-4"])
    df = df.drop(["FIPS 10-4", "ISO 3166"], axis = 1)
    df = df.set_index(keys=["ID","Year", "Name"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_3"  : "Month" , 0 : "Temp", "Name": "Country"})
    df["Month"] = df["Month"].str[5:].astype(int)
    df = pd.merge(df, stations, on = "ID")
    df = df.drop(["ID", "STNELEV"], axis = 1)
    df = df[["NAME", "LATITUDE", "LONGITUDE", "Country", "Year", "Month", "Temp"]]
    df = df[df["Year"] <= year_end]
    df = df[df["Year"] >= year_begin]
    df = df[df["Country"] == country]
    df = df[df["Month"] == month]
    df["Temp"] = df["Temp"]/100
    df.index = list(range(0,df.shape[0]))
    return df
```
>Let's try for the result.

```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```

```
+----+---------------+------------+-------------+-----------+--------+---------+--------+
|    | NAME          |   LATITUDE |   LONGITUDE | Country   |   Year |   Month |   Temp |
+====+===============+============+=============+===========+========+=========+========+
|  0 | PBO_ANANTAPUR |     14.583 |      77.633 | India     |   1980 |       1 |  23.48 |
+----+---------------+------------+-------------+-----------+--------+---------+--------+
|  1 | PBO_ANANTAPUR |     14.583 |      77.633 | India     |   1981 |       1 |  24.57 |
+----+---------------+------------+-------------+-----------+--------+---------+--------+
|  2 | PBO_ANANTAPUR |     14.583 |      77.633 | India     |   1982 |       1 |  24.19 |
+----+---------------+------------+-------------+-----------+--------+---------+--------+
|  3 | PBO_ANANTAPUR |     14.583 |      77.633 | India     |   1983 |       1 |  23.51 |
+----+---------------+------------+-------------+-----------+--------+---------+--------+
|  4 | PBO_ANANTAPUR |     14.583 |      77.633 | India     |   1984 |       1 |  24.81 |
+----+---------------+------------+-------------+-----------+--------+---------+--------+
```
<br />

---

### 3. Write a Geographic Scatter Function for Yearly Temperature Increases
```python
from plotly import express as px
import numpy as np
from sklearn.linear_model import LinearRegression

def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return round(LR.coef_[0], 4)


color_map = px.colors.diverging.RdGy_r
def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    df = query_climate_database(country = country, 
                                year_begin = year_begin, 
                                year_end = year_end,
                                month = month)   
    df = df[df.groupby(["NAME"])["Year"].transform(len) >= min_obs]
    df4 = pd.concat([df.groupby(["NAME"])["LATITUDE"].aggregate([np.mean]),
                     df.groupby(["NAME"])["LONGITUDE"].aggregate([np.mean]),
                     pd.DataFrame(df.groupby(["NAME"]).apply(coef))], axis=1)
    column_names = df4.columns.values
    column_names[0] = "LATITUDE"
    column_names[1] = "LONGITUDE"
    column_names[2] = "Estimated yearly increase(℃)"
    df4.columns = column_names
    df4["NAME"] = df4.index
    fig = px.scatter_mapbox(df4, 
                        lat = "LATITUDE",
                        lon = "LONGITUDE",
                        hover_name = "NAME",
                        color = "Estimated yearly increase(℃)",
                        **kwargs)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
```

>Let's try for the result.

```python
fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)
```

![newplot.png]({{ site.baseurl }}/images/newplot.png)



