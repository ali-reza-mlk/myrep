# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:08:05 2021

@author: alire
"""
import pickle
dest = 'D:\\Project\\BCI\\batches'
for ID in range(1):
            with open(dest+'\\batches'+str(ID)+'.pkl', 'rb') as f:
                loaded_obj = pickle.load(f)