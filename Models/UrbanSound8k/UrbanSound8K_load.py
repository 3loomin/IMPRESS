#!/usr/bin/env python
# coding: utf-8

# In[2]:


import UrbanSound8k_features as F
import numpy as np


# In[3]:


parent_dir="C:\\Users\\tlsck\\IMPRESS\\Data\\UrbanSound8K\\audio"
tr_sub_dirs=['fold1','fold2']
ts_sub_dirs=['fold3']
tr_features,tr_labels=F.parse_audio_files(parent_dir,tr_sub_dirs)
ts_features,ts_labels=F.parse_audio_files(parent_dir,ts_sub_dirs)

tr_labels=F.one_hot_encode(tr_labels)
ts_labels=F.one_hot_encode(ts_labels)


# In[ ]:




