---
layout: post
title: Blog Post 1
---


### 1. Create a Database

Because we are going to create a database which is include three tables,
so we should load these three data first. Then, add them into our database.
First we can import some libraries and create an empty database.

```python
import pandas as pd
import sqlite3
conn = sqlite3.connect("database.db")
```
Load data one by one.
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
Last step, let's check what our database includes.
```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
```
```
[('temperatures',), ('stations',), ('countries',)]
```


### 2. Write a Query Function

Thanks for the class notes.
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
    df.index = list(range(0,df.shape[0]))
    return df
```
Let's try for the result.
```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




