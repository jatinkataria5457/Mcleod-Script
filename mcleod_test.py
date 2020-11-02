# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:18:08 2020

@author: Jatin Kataria
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#cons=pd.read_excel('McLeod total Consumption data.xlsx')
cons=pd.read_excel('Mcleod knee consumption data.xlsx')
item=pd.read_excel('item_master.xlsx')
rev=pd.read_excel('McLeod Revenue data  (2019, Jan-Jul 2020).xlsx')
dem=pd.read_excel('McLeod demography data (2019, Jan-Jul 2020).xlsx')

dataset_list=[cons,rev,dem,item]
dataset_name=['Consumption Dataset','Revenue Dataset',
              'Demography Dataset','Item Master']
merge_keys=['Patient ID','Patient ID' ,'Model Nbr' ]
NullChecklist=['Patient ID','Qty Used','Cost','Model Nbr']
MappingChecklist={dataset_name[0]:['Patient ID','Case ID'],
                  dataset_name[1]:['Patient ID','Charge'],
                  dataset_name[2]:['Patient ID','Medical Record'],
                  dataset_name[3]:['Model Nbr']}



def NullCheck(df):
    null_series=pd.isnull(df).sum().sort_values(ascending=False)
    print(null_series)
    null_col=null_series[null_series>0].index.tolist()
    print('The columns containing null values in this dataset are: \n',null_col)
    null_df=[]
    df_wo_null=df
    count=0
    for i in NullChecklist:
        for j in null_col:
            if i==j:
                print('\nRemoving null rows from this dataset')
                null_df.append(df[pd.isnull(df[i])])
                df_wo_null=df[pd.notnull(df[i])]
                count+=1
               
        df=df_wo_null
   
    if count==0:
        print('\nNo null values found in this dataset with respect to NullChecklist')
    
    return null_df,df_wo_null
   
    


null_df_combined=[]
df_wo_null_combined=[]
for i in range(len(dataset_list)):
    print('\nPerforming null check on %s'%(dataset_name[i]))
    NullCheck_results=NullCheck(dataset_list[i])
    null_df_combined.append((dataset_name[i],NullCheck_results[0]))
    df_wo_null_combined.append(NullCheck_results[1])
    
df_wo_null_zipped=dict(list(zip(dataset_name,df_wo_null_combined)))




def MappingCheck(df):
   
    for i in MappingChecklist.keys():
        
        print('\nPerforming mapping check on %s vis-à-vis %s\n'%(i,MappingChecklist[i]))
        if i==dataset_name[2]:
            data=df[dataset_name[2]]
            data["is_duplicate"]=data.duplicated(subset=MappingChecklist[dataset_name[2]], keep='first')
            grp=data.groupby(MappingChecklist[i][0])
            mapping_series=(grp["is_duplicate"].unique()).agg(np.size).sort_values(ascending=False)
            print(mapping_series)  
            if len(mapping_series[mapping_series>1])==0:
                print('Conclusion--> 1 to 1 mapping')
            else:
                print('Conclusion--> 1 to many mapping')
                print('\nCorrecting many mapping into 1 to 1 mapping ')
                many_relation_items=mapping_series[mapping_series>1].index.tolist()
                df[dataset_name[2]].drop_duplicates(subset=MappingChecklist[dataset_name[2]], keep="first", inplace=True)
                df[dataset_name[2]].drop(['is_duplicate'],axis=1,inplace=True)
                
        elif i==dataset_name[3]:
            data=df[dataset_name[3]]
            data["is_duplicate"]=data.duplicated(subset=MappingChecklist[dataset_name[3]], keep='first')
            grp=data.groupby(MappingChecklist[i][0])
            mapping_series=(grp["is_duplicate"]).agg(np.size).sort_values(ascending=False)
            print(mapping_series) 
            print(len(mapping_series))
            if len(mapping_series[mapping_series>1])==0:
                print('Conclusion--> 1 to 1 mapping')
            else:
                print('Conclusion--> 1 to many mapping')
                print('\nCorrecting many mapping into 1 to 1 mapping ')
                many_relation_items=mapping_series[mapping_series>1].index.tolist()
                df[dataset_name[3]].drop_duplicates(subset=MappingChecklist[dataset_name[3]], keep="first", inplace=True)
                df[dataset_name[3]].drop(['is_duplicate'],axis=1,inplace=True)
       
        else: 
            grp=df[i].groupby(MappingChecklist[i][0])
            mapping_series=(grp[MappingChecklist[i][1]].unique()).agg(np.size).sort_values(ascending=False)
            print(mapping_series)
            if len(mapping_series[mapping_series>1])==0:
                print('Conclusion--> 1 to 1 mapping')
            else:
                print('Conclusion--> 1 to many mapping')
                print('\nCorrecting many mapping into 1 to 1 mapping ')
              
                many_relation_items=mapping_series[mapping_series>1].index.tolist()
                if np.issubdtype(df[i][MappingChecklist[i][1]].dtype, np.number)==False:
                    many_relation_items=mapping_series[mapping_series>1].index.tolist()
                    corrections=[':'.join(grp.get_group(items)[MappingChecklist[i][1]].unique()) for items in many_relation_items]
                    map_reference=list(zip(many_relation_items,corrections))
                    #print(map_reference)
                    df_correction=df[i]
                    df_correction['Adjusted '+MappingChecklist[i][1]]=df_correction[MappingChecklist[i][1]]
                    
                    common_mapping_key=[value for value in list(MappingChecklist.values())[0] if value in list(MappingChecklist.values())[1]][0]
                    
                    
                    for ref in map_reference:
                      df_correction.loc[df_correction[common_mapping_key] == ref[0], 'Adjusted '+MappingChecklist[i][1]] = ref[1]
                    
        
                else:
                    many_relation_items=mapping_series[mapping_series>1].index.tolist()
                    corrections=[np.sum(grp.get_group(items)[MappingChecklist[i][1]].unique()) for items in many_relation_items]
                    #print(corrections)
                    map_reference=list(zip(many_relation_items,corrections))
                    #print(map_reference)
                    
                    df_correction=df[i]
                    df_correction['Adjusted '+MappingChecklist[i][1]]=df_correction[MappingChecklist[i][1]]
                    
                    common_mapping_key=[value for value in list(MappingChecklist.values())[0] if value in list(MappingChecklist.values())[1]][0]
            
                    for ref in map_reference:
                      df_correction.loc[df_correction[common_mapping_key] == ref[0], 'Adjusted '+MappingChecklist[i][1]] = ref[1]
     
       

MappingCheck(df_wo_null_zipped)





def MergeDatasets(df,merge_keys):
    print('\nMerging all the datasets')
    base_dataset=df[dataset_name[0]]
    for i in range(len(merge_keys)):
        df['Merged Dataset']=pd.merge(base_dataset,df[dataset_name[i+1]],
                                      on=merge_keys[i],how='left')
        base_dataset=df['Merged Dataset']
    print('\nMerging complete. Access the merged dataset by opening--> df_wo_null_zipped')


MergeDatasets(df_wo_null_zipped, merge_keys)


