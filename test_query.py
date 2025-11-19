#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 15:03:08 2025
A simple test for database query
Queries database: "observatory_operation"

@author: Ron Micha
"""

'''Arbitrary Query program'''


import pandas as pd
# import numpy as np
# import sys
# from datetime import datetime, timedelta
# from openpyxl import load_workbook
# import json
# import math
# import csv
# import time
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from sklearn.linear_model import LinearRegression
# import matplotlib.ticker as ticker
# from collections import deque
# from io import StringIO
# import subprocess
# import argparse
from clickhouse_driver import Client
import clickhouse_connect

# select the computer from which to run: LAST_0 or euclid

euclidhost = '10.150.28.18'
euclidport = 9000
eucliduser ='default'
euclidpw = 'PassRoot'

last0host = '10.23.1.25'
last0port = 8123
last0user = 'last_user'
last0pw = 'physics'
lastdatabase = 'observatory_operation'

# Here is a query to see if there are duplicates in the db
query_string1 = """SELECT 
                rediskey, time, value, count(*) AS cnt
                FROM operation_strings
                WHERE startsWith(rediskey, 'unitCS.set.FocusData:')
                  AND time > '2025-09-29 12:00:00'
                  AND time < '2025-09-30 12:00:00'
                  AND value LIKE '%LoopCompleted%:%true%'
                GROUP BY rediskey, time, value
                HAVING cnt > 1
                ORDER BY cnt DESC"""
                
query_string2 = """SELECT * FROM operation_strings 
                      WHERE startsWith(rediskey, 'XerxesMountBinary.get.Status')
                      AND time > '2025-09-11 19:50:00' 
                      AND time < '2025-09-11 20:07:00' 
                      ORDER BY time DESC"""
query_string = query_string1
host = 'euclid' #'LAST_0'   # or 'euclid'


if host == 'last0':
    client = clickhouse_connect.get_client(host=last0host, port=last0port, \
             username= last0user, password= last0pw, database=lastdatabase) 
    result = client.execute(query_string) 
    df2 = pd.DataFrame(result.result_rows, columns=['rediskey', 'time', 'value', 'count']).set_index('rediskey')
elif host == 'euclid':
    client = Client(host=euclidhost, port=euclidport, \
         user=eucliduser, password= euclidpw, database=lastdatabase)              
    result = client.execute(query_string)        
    df2 = pd.DataFrame(result, columns=["rediskey", "time", "value", 'count']).set_index("rediskey")
print(df2)
# query_string = """SELECT * FROM operation_strings
#         WHERE startsWith(rediskey, 'XerxesMountBinary.get.Status')
#         AND "value" LIKE 'tracking'
#         AND time > '2025-10-19 12:00:00'
#         AND time < '2025-09-20 12:00:00'
#         ORDER BY time DESC
#                       """


                      
# '''        

# if database == 'LAST_0':    

# elif database == 'euclid':
#     result = client.execute(query_string)        
#     df2 = pd.DataFrame(result, columns=["rediskey", "time", "value", 'count']).set_index("rediskey")
# df2 = df2.sort_index().sort_values(by="time")
# df2 = df2.sort_values(by=["time", df2.index])

# '''to export into csv, uncomment the following + insert correct name'''
# Save DataFrame to CSV
#df2.to_csv('/home/ocs/Documents/read_DB/df_Temp_2.csv' , index=True)   # keep index as first column

#df2 = df2.assign(_index=df2.index).sort_values(by=["_index", "time"])


                      
# if database == 'LAST_0':    
#     result2 = client.query(query_string)
#     df_alt = pd.DataFrame(result2.result_rows, columns=['rediskey', 'time', 'value']).set_index('rediskey')
# elif database == 'euclid':
#     result2 = client.execute(query_string)        
#     df_alt = pd.DataFrame(result2, columns=["rediskey", "time", "value"]).set_index("rediskey")

# df_alt = df_alt.assign(_index=df_alt.index).sort_values(by=["_index", "time"])

''''
Here are the queries for downloading from DB to csv, for example for September, I added Sep.
Focus:
    WHERE startsWith(rediskey, 'unitCS.set.FocusData:')
    AND value like '%LoopCompleted%:%true%' 
    Focus_Sep.csv
    
FWHM:
    WHERE startsWith(rediskey, 'camera.set.FWHMellipse:')
    FWHM_Sep.csv
        
Tracking status:
    WHERE startsWith(rediskey, 'XerxesMountBinary.get.Status')
    AND "value" LIKE 'tracking'
    Track_only.csv
    
now from operation_numbers:
RA:
    WHERE startsWith(rediskey, 'XerxesMountBinary.get.RA')
    RA_Sep.csv
Dec:
    WHERE startsWith(rediskey, 'XerxesMountBinary.get.Dec')
    Dec_Sep.csv
RA:
    WHERE startsWith(rediskey, 'XerxesMountBinary.get.Az')
    Az_Sep.csv
RA:
    WHERE startsWith(rediskey, 'XerxesMountBinary.get.Alt')
    Alt_Sep.csv    

Temperature:
    WHERE startsWith(rediskey, 'unitCS.get.Temperature:')
    AND endsWith(rediskey, '.1')
    df_Temp_1.csv
    
    WHERE startsWith(rediskey, 'unitCS.get.Temperature:')
    AND endsWith(rediskey, '.2')
    df_Temp_2.csv
'''