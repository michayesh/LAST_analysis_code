#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 14:36:02 2025

@author: micha
"""
import pandas as pd
import pyLAST as pla
import clickhouse_connect
client = clickhouse_connect.get_client(host='10.150.28.18', port=8123, username='last_user', password='physics')
# query_result = client.query('SELECT top 2 * FROM last.visit_images WHERE camnum == 1')
# query_result = client.query('SELECT top 10 cropid, dateobs FROM last.visit_images WHERE cropid  == 1 or cropid ==24')
df=client.query_df('''SELECT dateobs,mountnum,camnum,ra,dec,cropid,fwhm,med_a,med_b,med_th,airmass
                   FROM last.visit_images 
                   WHERE dateobs > '2025-11-20 00:00:00'
                   AND dateobs < '2025-11-23 00:00:00'
                   ''')
  #AND time < '{end_str}'    '''
  #'select top 10 id_visit, dec from visit_images where dec > 0'
# print (query_result.result_set)
print('Finished')