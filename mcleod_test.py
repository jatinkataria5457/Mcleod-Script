# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:18:08 2020

@author: Jatin Kataria
"""
import sys
import re
import pandas as pd 
import numpy as np
import string
import copy
import warnings
warnings.filterwarnings("ignore")


#cons=pd.read_excel('McLeod total Consumption data.xlsx')
cons=pd.read_excel('Mcleod knee consumption data.xlsx')
item=pd.read_excel('item_master.xlsx')
#item=pd.read_excel('item_master_unclean.xlsx')
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

merged_data_params=['Merged Dataset','AttnMD','Case ID','com_bom_main_category','com_bom_sub_category','Cost','com_manufacturer','com_brand']

filter_keys=['data','filter_col','date_filter_pattern','config_file','patient_type']
filter_params={filter_keys[0]:merged_data_params[0],filter_keys[1]:['Surgery Date','Patient Type_y'],
               filter_keys[2]:'\d{4}[/-]\d{2}[/-]\d{2} \d{2}[:]\d{2}[:]\d{2}',
               filter_keys[3]:'mcleod_date_config.txt',
               filter_keys[4]:'Outpatient'}



bom_filter_params={filter_keys[1]:'Adjusted '+merged_data_params[2],
                   filter_keys[2]:'\d+[.]\d*',
                   filter_keys[3]:'BOM_rare_products_cutoff.txt'}
 
pdt_usage_params=['AttnMD','Qty Used','Cost','com_bom_main_category',
                  'com_bom_sub_category','com_manufacturer','com_brand']

margin_payer_params=['Standard Payer','AttnMD','Charge','Net Rev','Op Margin']


def IntegratedAnalysis():

    def CleanCheck(df):
        null_series=pd.isnull(df).sum().sort_values(ascending=False)
        print(null_series)
        null_col=null_series[null_series>0].index.tolist()
        print('The columns containing null values in this dataset are: \n',null_col)
        unclean_df=[]
        df_clean=df
        count=0
        for i in NullChecklist:
            for j in null_col:
                if i==j:
                    print('\nRemoving unclean rows from this dataset')
                    #null_df.append(df[pd.isnull(df[i])])
                    unclean=(df[(df[i]==0)|(df[i]=='Null')|(df[i]=='na')])
                    unclean_df.extend([df[pd.isnull(df[i])],unclean])
                    df_clean=df[(pd.notnull(df[i]))]
                    df_clean=df_clean.drop(unclean.index.tolist(), inplace = False).reset_index(drop=True)
                    count+=1
       
            if count==0 and i in df.columns.tolist():
                unclean=(df[(df[i]==0)|(df[i]=='Null')|(df[i]=='na')])
                if len(unclean)!=0:
                    print('\nFound %d unclean records in this dataset with respect to %s'%(len(unclean),i))
                    print('Following are the found unclean records:\n',unclean)
                    print('\nRemoving unclean rows from this dataset with respect to %s'%i)
                    unclean_df.append(unclean)
                    df_clean=df.drop(unclean.index.tolist(), inplace = False).reset_index(drop=True)
                else:
                    print('\nNo unclean values found in this dataset with respect to %s'%i)
                    #df_clean=df
                   
            df=df_clean
            
            
        return unclean_df,df_clean
       
    
    
    unclean_df_combined=[]
    df_clean_combined=[]
    for i in range(len(dataset_list)):
        print('\nPerforming clean check on %s'%(dataset_name[i]))
        CleanCheck_results=CleanCheck(dataset_list[i])
        unclean_df_combined.append((dataset_name[i],CleanCheck_results[0]))
        df_clean_combined.append(CleanCheck_results[1])
        
    df_clean_zipped=dict(list(zip(dataset_name,df_clean_combined)))
    
    
    
    
    
    
    
    def ExtraClean(df):
        print('\nPerforming irregular whitespaces trimming and non-printable characters/line breaks cleaning on all the datasets\n')
        for data in df.values():
            cat_col=data.select_dtypes(include='object').columns.tolist()
            count_trim=0
            count_clean=0
            for col in cat_col:
                string_list=[]
                for s in data[col].tolist():
                    if s is not np.nan and type(s)==str:
                        pre_trim_len=len(s)
                        trim_string=(' '.join(s.split()))
                        post_trim_len=len(trim_string)
                        pre_clean_len=post_trim_len
                        clean_string=(''.join(filter(lambda x:x in string.printable, trim_string)))
                        clean_string=clean_string.replace('\n','').replace('\r',' ')
                        post_clean_len=len(clean_string)
                        string_list.append(clean_string)
                        if pre_trim_len!=post_trim_len:
                            count_trim+=1
                        if pre_clean_len!=post_clean_len:
                            count_clean+=1
                    elif s is np.nan:
                        string_list.append(np.nan)
                    elif type(s)!=str:
                        string_list.append(s)
                data[col]=pd.Series(string_list)
                #print(data[col])
            print('Irregular whitespaces found in %d records. Non-printable characters/line breaks found in %d records'%(count_trim,count_clean))
        print('\nTrimming and Cleaning process completed')
    
    
    ExtraClean(df_clean_zipped) 
        
    
    
    
       
    
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
         
           
    
    MappingCheck(df_clean_zipped)
    
    
    
    
    def extractInformation(df):
        phy=df[merged_data_params[1]].unique().tolist()
        phy=[i for i in phy if i is not np.nan]
        print('The number of unique physicians are: %d'%len(phy))
        if 'Adjusted '+merged_data_params[2] in df.columns.tolist():
            surgery_per_doc=[len(df['Adjusted '+merged_data_params[2]][df[merged_data_params[1]]==i].unique().tolist()) for i in phy]
        else:
            surgery_per_doc=[len(df[merged_data_params[2]][df[merged_data_params[1]]==i].unique().tolist()) for i in phy]
        
        print('The number of unique surgeries performed are: %d'%sum(surgery_per_doc))
        surgery=dict(list(zip(phy,surgery_per_doc)))
        grp=df.groupby([pdt_usage_params[0]])
        mat_cost=(grp[pdt_usage_params[2]]).agg(np.sum)                  
        mat_cost=pd.DataFrame(mat_cost)
        mat_cost['Material Cost per procedure']=[np.nan for i in range(len(mat_cost))]
        for key in surgery.keys():
            mat_cost.loc[key,'Material Cost per procedure']=mat_cost.loc[key,pdt_usage_params[2]].tolist()/surgery[key]
        
        mat_cost=mat_cost.sort_values(by='Material Cost per procedure',ascending=True)
        surgery=sorted(surgery.items(), key=lambda x: x[1], reverse=True)
        main_cat_pdt=df[merged_data_params[3]].unique().tolist()
        sub_cat_pdt=df[merged_data_params[4]].unique().tolist()
        print('The number of unique main category and sub-category of products are: %d and %d respectively'%(len(main_cat_pdt),len(sub_cat_pdt)))
        return phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt
      
    
    
    
    def MergeDatasets(df,merge_keys):
        print('\nMerging all the datasets')
        base_dataset=df[dataset_name[0]]
        for i in range(len(merge_keys)):
            df[merged_data_params[0]]=pd.merge(base_dataset,df[dataset_name[i+1]],
                                          on=merge_keys[i],how='inner')
            base_dataset=df['Merged Dataset']
        print('\nMerging complete. Access the merged dataset by opening--> df_clean_zipped')
        
        
        print('\nExtracting important information from merged dataset\n')
        return extractInformation(df[merged_data_params[0]])
    
    
    phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt=MergeDatasets(df_clean_zipped, merge_keys)
    
    
    
    
    def filterData(df,filter_params):
        content=open(filter_params[filter_keys[3]], 'r').read()
        pattern = filter_params[filter_keys[2]]
        found_data = re.findall(pattern, content)
        df_filtered=df[filter_params[filter_keys[0]]][(df[filter_params[filter_keys[0]]][filter_params[filter_keys[1]][0]]>=found_data[0])&(df[filter_params[filter_keys[0]]][filter_params[filter_keys[1]][0]]<=found_data[1])]
        
        loop_cond=1
        while loop_cond>0:
            user_choice=input('Do you want to filter data on patient type?\nEnter [y/n]:')
            if user_choice=='y':
                df_filtered=df_filtered[df_filtered[filter_params[filter_keys[1]][1]]==filter_params[filter_keys[4]]]
        
                break
            elif user_choice=='n':
                break
            else:
                print('Invalid choice. Kindly enter a valid option')
        
        print('\nExtracting important information from filtered dataset\n')
        phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt=extractInformation(df_filtered)
        return phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt,df_filtered
    
    phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt,df_filtered=filterData(df_clean_zipped,filter_params)
    
    
    
    def renameProcedure(df_filtered):
        loop_cond=1
        while loop_cond>0:
            user_choice=input('Do you want to rename all the procedures to a particular name?\nEnter [y/n]:')
            if user_choice=='y':
                rename=input('Enter the new name you want:')
                df_filtered['Procedure']=[rename]*len(df_filtered)
        
                break
            elif user_choice=='n':
                break
            else:
                print('Invalid choice. Kindly enter a valid option')
                
    renameProcedure(df_filtered)
    
    
    
    def pdtClassGrouping(df,main_cat_pdt):
        df_pdt_cat=df[[merged_data_params[3],merged_data_params[4]]]
        pdt_cat_list=[]
        for cat in main_cat_pdt:
            pdt_cat_list.append((cat,df_pdt_cat[df_pdt_cat[merged_data_params[3]]==cat][merged_data_params[4]].unique().tolist()))
        pdt_cat=dict(pdt_cat_list)
        return pdt_cat
    
    pdt_cat=pdtClassGrouping(df_filtered,main_cat_pdt)
    
    
    
    def pdtManufacturerGrouping(df):
        mft_cat=df[merged_data_params[-2]].unique().tolist()
        brand_cat=df[merged_data_params[-1]].unique().tolist()
                
        df_mft=df[[merged_data_params[-2],merged_data_params[-1]]]
        mft_cat_list=[]
        for cat in mft_cat:
            mft_cat_list.append((cat,df_mft[df_mft[merged_data_params[-2]]==cat][merged_data_params[-1]].unique().tolist()))
        mft_brand_cat=dict(mft_cat_list)
        return mft_cat,brand_cat,mft_brand_cat
    
    mft_cat,brand_cat,mft_brand_cat=pdtManufacturerGrouping(df_filtered)
    
    
    
    def costDistribution(df_filtered):
        df_filtered['Cost Distribution']=df_filtered[merged_data_params[-3]]
        
        for docphy in phy:
            for k in pdt_cat.keys():
                for v in pdt_cat[k]:
                    #print(docphy,k,v)
                    df=df_filtered
                    df_interest=df[(df[pdt_usage_params[0]]==docphy)&
                        (df[pdt_usage_params[3]]==k)&
                        (df[pdt_usage_params[4]]==v)]
                    #print(df_interest.shape)
                    if len(df_interest)!=0:
                        
                        cost_freq=(df_interest[merged_data_params[-3]].value_counts().sort_values(ascending=False))
                        cost_freq_dict=dict(list(zip(cost_freq.index.tolist(),cost_freq.tolist())))
                        
                        #print(cost_freq_dict)
                        for i in range(len(df_interest.index.tolist())):
                            
                            df_filtered.loc[df_interest.index.tolist()[i],'Cost Distribution']=str(df_filtered.loc[df_interest.index.tolist()[i],'Cost Distribution'])+' ('+str(cost_freq_dict[df_filtered.loc[df_interest.index.tolist()[i],'Cost Distribution']])+')'
                            
     
        return df_filtered       
    df_filtered =costDistribution(df_filtered) 
    
    
    
    
    
    
    def pdtUsageAnalysis(df,surgery):
        surgery_dict=dict(surgery)
        df_columns=[merged_data_params[3],merged_data_params[4]]
        df_columns.extend([i+'_avg_cost' for i in phy])
        df_columns.extend([i+'_mkt_share' for i in phy])
        df_columns.extend([i+'_avg_usage' for i in phy])
        df_index=[i for i in range(len(sub_cat_pdt))]
        df_usage_analysis=pd.DataFrame(columns=df_columns,index=df_index)
        
        df_mkt_share_per_list=[]
        
        for doc in phy:
            avg_unit_cost=[]
            mkt_seg=[]
            avg_usage=[]
            for key in pdt_cat.keys():
                for val in pdt_cat[key]:
                    #df=df_filtered
                    df_interest=df[(df[pdt_usage_params[0]]==doc)&
                        (df[pdt_usage_params[3]]==key)&
                        (df[pdt_usage_params[4]]==val)]
                    if len(df_interest)!=0:
                        group=(df_interest.groupby([pdt_usage_params[-2],pdt_usage_params[-1]])[pdt_usage_params[1],pdt_usage_params[2]]).agg([np.sum])
                        avg_unit_cost.append([key,val,group.sum().tolist()[1]/group.sum().tolist()[0]])
                        avg_usage.append([key,val,round(group.sum().tolist()[0]/surgery_dict[doc],2)])
                        
                        group_percent =  (group/group.sum().tolist()[0])*100
                        grp_idx=[list(i) for i in group_percent.index.tolist()]
                        grp_per=[round(i,2) for i in (group_percent[pdt_usage_params[1]]['sum']).tolist()]
                        [grp_idx[i].append(grp_per[i]) for i in range(len(grp_per))]
                        for i in range(len(grp_idx)):
                            grp_idx[i][1]=grp_idx[i][1]+'-'+str(grp_idx[i][2])+'%'
                            del grp_idx[i][-1]
                        
                        mft_distinct=(group.sum(level=0, axis=0)).index.tolist()
                       
                        mkt_share_per=[(group.index.tolist()[gil][0],group.index.tolist()[gil][1],grp_per[gil]) for gil in range(len(group.index.tolist()))]
                        df_mkt_share_per_list.append((doc+';'+key+';'+val,pd.DataFrame(mkt_share_per,columns=['Manufacturer','Brand','percent_share'])))
                       
                        mkt_share_list=[]
                        for i in mft_distinct:
                            mkt_share=''
                            for j in grp_idx:
                                if j[0]==i:
                                    mkt_share=mkt_share+j[1]+','
                            mkt_share=mkt_share[:-1]
                            mkt_share_list.append(i+'('+mkt_share+')')
                            
                        mkt_share='--'.join(mkt_share_list)
                        mkt_seg.append([key,val,mkt_share])
                    else:
                        avg_unit_cost.append([key,val,np.nan])
                        mkt_seg.append([key,val,np.nan])
                        avg_usage.append([key,val,np.nan])
            df_usage_analysis[pdt_usage_params[3]]=[i[0] for i in avg_unit_cost]
            df_usage_analysis[pdt_usage_params[4]]=[i[1] for i in avg_unit_cost]
            df_usage_analysis[doc+'_avg_cost']=[i[2] for i in avg_unit_cost]
            df_usage_analysis[doc+'_mkt_share']=[i[2] for i in mkt_seg]
            df_usage_analysis[doc+'_avg_usage']=[i[2] for i in avg_usage]
            
        return df_usage_analysis,dict(df_mkt_share_per_list)
    
    df_usage_analysis,mkt_share_per=pdtUsageAnalysis(df_filtered,surgery)
    
    
    
    
    
    
    
    
    
    
    def MarginPayerAnalysis(df):
        df_copy=df.copy()
        df_copy=df_copy.drop_duplicates(subset=['Adjusted '+merged_data_params[2]], keep='first')
        
        op_margin=(df_copy.groupby([margin_payer_params[1]])['Adjusted '+merged_data_params[2],margin_payer_params[-1]]).agg({'Adjusted '+merged_data_params[2]:'nunique',margin_payer_params[-1]:['sum']})
        op_margin['Margin per procedure']=op_margin.apply(lambda x: x[margin_payer_params[-1]]['sum']/ x['Adjusted '+merged_data_params[2]]['nunique'], axis=1)   
        
        
        
        group_payer_type=(df_copy.groupby([margin_payer_params[0],margin_payer_params[1]])['Adjusted '+merged_data_params[2],margin_payer_params[2],margin_payer_params[-2],margin_payer_params[-1]]).agg({'Adjusted '+merged_data_params[2]:'nunique',margin_payer_params[2]:['sum'],margin_payer_params[-2]:['sum'],margin_payer_params[-1]:['sum']})
        grp_payer_type_col=group_payer_type.columns.tolist()
        for i in range(len(grp_payer_type_col)):
            if i!=0:
                group_payer_type['Avg_'+grp_payer_type_col[i][0]]=group_payer_type.apply(lambda x: x[grp_payer_type_col[i][0]][grp_payer_type_col[i][1]]/ x[grp_payer_type_col[0][0]][grp_payer_type_col[0][1]], axis=1)   
        group_payer_type['Revenue/Charge Ratio in %']=group_payer_type.apply(lambda x: 100*(x[grp_payer_type_col[2][0]][grp_payer_type_col[2][1]]/ x[grp_payer_type_col[1][0]][grp_payer_type_col[1][1]]), axis=1)   
        
        return op_margin,group_payer_type
    
    op_margin,group_payer_type=MarginPayerAnalysis(df_filtered)
    
    
    
    
    
    
    import time
    start_time = time.time()
    
    df_bom_columns=['AttnMD','Procedure Name','BOM Rank','com_bom_main_category','com_bom_sub_category','Avg Usage','Number of products','Total Coverage','Market Share']  
    df_bom_analysis=pd.DataFrame(columns=df_bom_columns)
    
    content=open(bom_filter_params[filter_keys[3]], 'r').read()
    pattern = bom_filter_params[filter_keys[2]]
    rare_case_cutoff = float(re.findall(pattern, content)[0])
    
    rare_pdt_combined=[]  
    for doctor in phy:
        
        df_filtered_bom_copy=df_filtered.copy()
        t=df_filtered_bom_copy
        
        procedures_per_phy=(t[(t['AttnMD']==doctor)].groupby(['AttnMD','Procedure'])).agg({'Adjusted Case ID':'nunique'}).index
        procedures_per_phy=[i[1] for i in procedures_per_phy]
         
        
        
        
        for i in range(len(procedures_per_phy)):
            
            
            t=df_filtered_bom_copy
            t=t[(t['AttnMD']==doctor)&(t['Procedure']==procedures_per_phy[i])]
            g=(t.groupby(['AttnMD','Procedure'])).agg({'Adjusted Case ID':'nunique'})
            
            group=(t.groupby(['AttnMD','Procedure','com_bom_main_category','com_bom_sub_category'])).agg({'Adjusted Case ID':'nunique','Qty Used':'sum'})
            
            group['Avg Usage']=group[group.columns.tolist()[1]]/g['Adjusted Case ID'].tolist()[0]
            
            
            group_std=group[group['Avg Usage']>rare_case_cutoff]
            group_rare=group[group['Avg Usage']<=rare_case_cutoff]
            
            rare_pdt=[(gr[2],gr[3]) for gr in group_rare.index.tolist()]
            rare_pdt_combined.append((doctor,rare_pdt))
        
            if len(rare_pdt)!=0:
                idx_rare=[]
                idx_rare_pdt=[]
                for rp in range(len(rare_pdt)):
                    idx_rare.extend(t[(t['com_bom_main_category']==rare_pdt[rp][0])&(t['com_bom_sub_category']==rare_pdt[rp][1])].index.tolist())
                t.drop(idx_rare , inplace=True) 
            else:
                group_std=group
            
            
            
            
            
            aci=(t.groupby(['AttnMD','Procedure','Adjusted Case ID'])).agg({'Adjusted Case ID':'nunique'})
            aci=[ac[2] for ac in aci.index.tolist()]
            freq_unique_procedures=sorted(group_std['Adjusted Case ID'].unique().tolist(),reverse=True)
            
            
            basket=[]
            baskets=[]
            basket_collection=[]
            
            
            
            
            for freq in freq_unique_procedures:
             
                basket=[]
                basket=[(gs[2],gs[3]) for gs in group_std[group_std['Adjusted Case ID']==freq].index.tolist()]
                baskets.extend(basket)
                
              
               
                treatment=0
                for ci in aci:
                    t_sel=t[t['Adjusted Case ID']==ci][['com_bom_main_category','com_bom_sub_category']]
                    t_sel=t_sel.drop_duplicates(subset=['com_bom_sub_category'], keep='first')
                    p_aci=list(zip(t_sel['com_bom_main_category'].tolist(),t_sel['com_bom_sub_category'].tolist()))    
                    if(set(p_aci).issubset(set(baskets))):
                        treatment+=1
                
                coverage=round((treatment/g['Adjusted Case ID'].tolist()[0])*100,2)
                
               
                basket_collection.append(('Coverage',coverage,len(baskets),baskets)) 
            basket_collection=[(b[0],b[1],b[2],b[3][:b[2]]) for b in basket_collection]    
            
           
            num_pdt=[p[2] for p in basket_collection]
            
            
            for loop in range(len(num_pdt)):
                for num in range(num_pdt[loop]):
                    
                    new_row={'AttnMD':doctor,'Procedure Name':procedures_per_phy[i],
                             'BOM Rank':loop+1,'com_bom_main_category':basket_collection[loop][3][num][0],
                             'com_bom_sub_category':basket_collection[loop][3][num][1],
                             'Avg Usage': round(group_std.loc[(group_std.index.get_level_values('com_bom_main_category')== basket_collection[loop][3][num][0]) & (group_std.index.get_level_values('com_bom_sub_category') == basket_collection[loop][3][num][1])]['Avg Usage'].tolist()[0],2),
                             'Number of products':num_pdt[loop],
                             'Total Coverage':basket_collection[loop][1],
                             'Market Share':df_usage_analysis[(df_usage_analysis['com_bom_main_category']==basket_collection[loop][3][num][0])&(df_usage_analysis['com_bom_sub_category']==basket_collection[loop][3][num][1])][doctor+'_mkt_share'].tolist()[0]}
                    
                    df_bom_analysis = df_bom_analysis.append(new_row, ignore_index=True)
        
         
    end_time=time.time()
    print("Execution time is: %.2f minutes" % ((end_time - start_time)/60))
    
    
    
    
    
    
    
    
    df_outcome_score=df_filtered.copy()
    df_outcome_score=df_outcome_score[['AttnMD','Adjusted Case ID','Arith LOS Observed','Complication Observed','Readmissions Observed (All IP)','Mortality Observed']]
    
    df_outcome_score=df_outcome_score.drop_duplicates(subset=['Adjusted '+merged_data_params[2]], keep='first')
    
    mean_los=np.mean(df_outcome_score['Arith LOS Observed'])   
    
    
    #print(pd.isnull(df_outcome_score).sum().sort_values(ascending=False))
    null_outcome_params=pd.isnull(df_outcome_score).sum().sort_values(ascending=False)
    if null_outcome_params[null_outcome_params>0].size!=0:
        print('Null values found\nReplacing all null values by 0 using the assumption that if there was any complication,readmission or mortality it would have been mentioned')
        df_outcome_score=df_outcome_score.fillna(0)
        df_outcome_score.drop(df_outcome_score[df_outcome_score['AttnMD'] == 0].index,inplace=True)
    
    
    
    
    def OutcomeScore(c1,c2,c3,c4):
        total=0
        if c1>mean_los:
            total+=0
        else:
             total+=2.5
        if c2==0:
            total+=2.5
        else:
            total+=0 
        if c3==0:
            total+=2.5
        else:
            total+=0 
        if c4==0:
            total+=2.5
        else:
            total+=0 
           
        return total
    
    
    df_outcome_score['total score'] = df_outcome_score[['Arith LOS Observed','Complication Observed','Readmissions Observed (All IP)','Mortality Observed']].apply(lambda row : OutcomeScore(row['Arith LOS Observed'],row['Complication Observed' ],row['Readmissions Observed (All IP)' ],row['Mortality Observed']),axis=1) 
    outcome_score_leaderboard=df_outcome_score.groupby('AttnMD').agg({'total score':'sum'})
    surgery_dict=dict(surgery)
    
    total_procedures_list=[surgery_dict[phy_name] for phy_name in outcome_score_leaderboard.index.tolist()]
    outcome_score_leaderboard['total procedures']=total_procedures_list
    outcome_score_leaderboard['avg outcome score']=outcome_score_leaderboard.apply(lambda row : row['total score']/row['total procedures'],axis=1) 
    
    outcome_score_leaderboard=outcome_score_leaderboard.sort_values(by='avg outcome score',ascending=False)
    
    
    
    
    
    
    
    
    
    df_bom_physicians_combined=[]
    for doct in phy:
        d_copy=df_usage_analysis.copy()
        col_sel=['com_bom_main_category','com_bom_sub_category',doct+'_avg_usage',
                 doct+'_avg_cost',doct+'_mkt_share']
        df_bom_physicians=d_copy[col_sel]
        df_bom_physicians.rename(columns = {col_sel[2]:'avg_usage',col_sel[-2]:'actual_avg_cost' ,col_sel[-1]:'actual_mkt_share'}, inplace = True)
        df_bom_physicians.insert(loc=0, column='AttnMD', value=[doct]*len(df_bom_physicians)) 
        df_bom_physicians.dropna(inplace=True)
        df_bom_physicians.reset_index(drop=True, inplace=True)
        df_bom_physicians.insert(loc=5, column='actual_total_cost', value=df_bom_physicians.apply(lambda row : row['avg_usage']*row['actual_avg_cost'],axis=1)) 
    
        df_bom_physicians_combined.append(df_bom_physicians)
    
    df_bom=pd.concat(df_bom_physicians_combined, sort=False) 
    
    df_bom=df_bom.round(decimals=2)
    df_bom.reset_index(drop=True,inplace=True)
    
    bom_group=df_bom.groupby(['com_bom_main_category','com_bom_sub_category','AttnMD']).agg({'actual_avg_cost':'min'})
    temp=(bom_group.min(level=1,axis=0))
    
    #print(bom_group.index.get_level_values('com_bom_sub_category'))
    bom_pdt_dict=(pd.Series(bom_group.index.get_level_values('com_bom_sub_category').tolist()).value_counts()).to_dict()
    
    
    z=list(zip(bom_group.index.tolist(),bom_group['actual_avg_cost'].tolist()))
    min_bom_list=[]
    
    for i in list(zip(temp.index.tolist(),temp['actual_avg_cost'].tolist())):
        count=0
        for j in range(len(z)):
            if z[j][0][1]==i[0] and z[j][1]==i[1]:
                phy_name=z[j][0][2]
                count+=1
        if count==bom_pdt_dict[i[0]]:
            min_bom_list.append((i[0],i[1],'self'))
        else:
            min_bom_list.append((i[0],i[1],phy_name))
            
            
    df_bom.insert(loc=6, column='ideal_avg_cost', value=np.nan) 
    df_bom.insert(loc=7, column='ideal_total_cost', value=np.nan) 
    df_bom.insert(loc=9, column='ideal_mkt_share', value=np.nan)
    
    
    min_bom_dict=dict([(mbl[0],[mbl[1],mbl[2]]) for mbl in min_bom_list])
    #print(df_bom.loc[0,'ideal_avg_cost'])
    for bom_loop in range(len(df_bom)):
        df_bom.loc[bom_loop,'ideal_avg_cost']=min_bom_dict[df_bom.loc[bom_loop,'com_bom_sub_category']][0]
        df_bom.loc[bom_loop,'ideal_total_cost']=round(df_bom.loc[bom_loop,'ideal_avg_cost']*df_bom.loc[bom_loop,'avg_usage'],2)
        if min_bom_dict[df_bom.loc[bom_loop,'com_bom_sub_category']][1]=='self':
            df_bom.loc[bom_loop,'ideal_mkt_share']=df_bom.loc[bom_loop,'actual_mkt_share']
        else:
            df_bom.loc[bom_loop,'ideal_mkt_share']=df_usage_analysis[(df_usage_analysis['com_bom_sub_category']==df_bom.loc[bom_loop,'com_bom_sub_category'])][min_bom_dict[df_bom.loc[bom_loop,'com_bom_sub_category']][1]+'_mkt_share'].tolist()[0]
            
    
    
    def bomfullCoverage(df_bom):
        idx_combined=[]
        for i in range(len(phy)):
            df=df_bom.copy()
            for j in range(len(rare_pdt_combined[i][1])):
                idx_combined.extend(df[(df['AttnMD']==phy[i])&(df['com_bom_main_category']==rare_pdt_combined[i][1][j][0])&(df['com_bom_sub_category']==rare_pdt_combined[i][1][j][1])].index.tolist())
        df=df_bom.copy()
        df.drop(idx_combined, inplace = True) 
        df.reset_index(drop=True, inplace=True)
        return df
    
    df_bom_full_coverage=bomfullCoverage(df_bom)
    
    
    
    #BOM Leaders
    def bomLeader(df_bom_full_coverage):
        bom_leaderboard=df_bom_full_coverage.groupby(['AttnMD']).agg({'actual_total_cost':'sum'}).sort_values(by='actual_total_cost',ascending=True)
        bom_leaderboard.rename(columns = {'actual_total_cost':'avg_procedure_cost'}, inplace = True) 
        bom_leaderboard['total procedures']=[surgery_dict[dphy] for dphy in bom_leaderboard.index.tolist()]
        bom_leaderboard['total cost']=bom_leaderboard.apply(lambda row : row['avg_procedure_cost']*row['total procedures'],axis=1) 
    
        return bom_leaderboard
    bom_leaderboard=bomLeader(df_bom_full_coverage)
    
    
    
    
    
    

    
    return phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt,pdt_cat,df_usage_analysis,mkt_share_per,op_margin,group_payer_type,outcome_score_leaderboard,bom_leaderboard,df_bom_full_coverage,df_filtered


phy,surgery,mat_cost,main_cat_pdt,sub_cat_pdt,pdt_cat,df_usage_analysis,mkt_share_per,op_margin,group_payer_type,outcome_score_leaderboard,bom_leaderboard,df_bom_full_coverage,df_filtered=IntegratedAnalysis()


sys.exit()


#%%

#for top leaders

i=0
count=0
leader_names=[]
while i==0:
    
    leader_names.append(input('Enter the name of physician you want to replace as Leaders:'))
    loop_cond=1
    while loop_cond>0:
        ans=input('Have you mentioned all the names? [y/n]:')
        if ans=='y':
            count+=1
            break
        elif ans=='n':
            count=0
            break
        else:
            count=0
            print('Invalid choice. Enter the correct option')
    if count>0:
        break
    else:
        i=count
  

dem_incl_leaders=dem.copy()
for dl_loop in range(len(dem_incl_leaders)):
    if dem_incl_leaders.loc[dl_loop,'AttnMD'] in leader_names:
        dem_incl_leaders.loc[dl_loop,'AttnMD']='Top_'+str(len(leader_names))+'_Leaders'



dataset_list=[cons,rev,dem_incl_leaders,item]
phy_incl_leaders,surgery_incl_leaders,mat_cost_incl_leaders,main_cat_pdt_incl_leaders,sub_cat_pdt_incl_leaders,pdt_cat_incl_leaders,df_usage_analysis_incl_leaders,mkt_share_per_incl_leaders,op_margin_incl_leaders,group_payer_type_incl_leaders,outcome_score_leaderboard_incl_leaders,bom_leaderboard_incl_leaders,df_bom_full_coverage_incl_leaders,df_filtered_incl_leaders=IntegratedAnalysis()



def SavingsAnalysis(df_filtered_incl_leaders,pdt_cat_incl_leaders,df_bom_full_coverage_incl_leaders,bom_leaderboard_incl_leaders,mkt_share_per_incl_leaders):

    def brandwiseUnitCost(df_filtered_incl_leaders):
        brand_unit_price_coll=[]
        for key in pdt_cat_incl_leaders.keys():
            for val in pdt_cat_incl_leaders[key]:
                df=df_filtered_incl_leaders.copy()
                df_interest=df[
                    (df[pdt_usage_params[3]]==key)&
                    (df[pdt_usage_params[4]]==val)]
                if len(df_interest)!=0:
                    group_brand=(df_interest.groupby([pdt_usage_params[-2],pdt_usage_params[-1]])[pdt_usage_params[1],pdt_usage_params[2]]).agg([np.sum])
                    group_brand['brand_avg_unit_cost']=group_brand.apply(lambda row : round(row['Cost']['sum']/row['Qty Used']['sum'],2),axis=1) 
                    brand_unit_price_coll.append((key+','+val,group_brand))
        
        brand_unit_price_dict=dict(brand_unit_price_coll)
        brand_unit_price_dict=dict([(b_key,brand_unit_price_dict[b_key].sort_values(by='brand_avg_unit_cost',ascending=True)) for b_key in brand_unit_price_dict.keys()])
        return brand_unit_price_dict
         
    brand_unit_price_dict=brandwiseUnitCost(df_filtered_incl_leaders) 
        
        
        
    #Savings Options
    df_savings=df_bom_full_coverage_incl_leaders.copy()
    df_savings=df_savings[df_savings['AttnMD']=='Top_'+str(len(leader_names))+'_Leaders']
    df_savings.reset_index(drop=True, inplace=True)
    df_savings.insert(loc=9, column='actual_avg_procedure_cost', value=[bom_leaderboard_incl_leaders.loc['Top_'+str(len(leader_names))+'_Leaders','avg_procedure_cost']]*len(df_savings))
    df_savings['option1_avg_procedure_cost']= [np.sum(df_savings['ideal_total_cost'])]*len(df_savings)
    
    
    #grp_test=df_bom_full_coverage_incl_leaders.groupby(['com_bom_main_category','com_bom_sub_category','actual_mkt_share']).agg({'actual_avg_cost':'min'})
    
    
    
    df_savings['overall_min_avg_cost']=[np.nan]*len(df_savings)
    df_savings['overall_min_mkt_share']=[np.nan]*len(df_savings)
    for b_key in brand_unit_price_dict.keys():
        for s_loop in range(len(df_savings)):
            if df_savings.loc[s_loop,'com_bom_main_category']==b_key.split(',')[0] and df_savings.loc[s_loop,'com_bom_sub_category']==b_key.split(',')[1]:
                data=brand_unit_price_dict[b_key]
                data_min=data[['brand_avg_unit_cost']].idxmin().tolist()
                data_min_cost=data[['brand_avg_unit_cost']].min().tolist()[0]
                data_min=[(data_min[i][0]+'('+data_min[i][1]+'-100%'+')') for i in range(len(data_min))]
                data_min=' / '.join(data_min)
                df_savings.loc[s_loop,'overall_min_mkt_share']=data_min
                df_savings.loc[s_loop,'overall_min_avg_cost']=data_min_cost
                
    df_savings['overall_min_total_cost']=df_savings.apply(lambda row : row['overall_min_avg_cost']*row['avg_usage'],axis=1)             
    df_savings['option2_avg_procedure_cost']= [np.sum(df_savings['overall_min_total_cost'])]*len(df_savings)
    
    
    
    
    
    
    
    
    
    
    def marketShareSorted(mkt_share_per_incl_leaders):
        for msp_key in mkt_share_per_incl_leaders.keys():
            mkt_share_per_incl_leaders[msp_key]
            
            ref_order=brand_unit_price_dict[','.join(msp_key.split(';')[1:])].index.tolist()
            actual_pdt_combo=[tuple(msp) for msp in mkt_share_per_incl_leaders[msp_key][['Manufacturer','Brand']].to_numpy().tolist()]
            
            ref_idx=[ref_order.index(apc) for apc in actual_pdt_combo]
            df_brand_unit_price=brand_unit_price_dict[','.join(msp_key.split(';')[1:])]
            apc_cost=[df_brand_unit_price[(df_brand_unit_price.index.get_level_values(0) == apc_e[0])&(df_brand_unit_price.index.get_level_values(1) == apc_e[1])]['brand_avg_unit_cost'].tolist()[0] for apc_e in actual_pdt_combo]
            
            mkt_share_per_incl_leaders[msp_key]['brand_avg_unit_cost']=apc_cost
            mkt_share_per_incl_leaders[msp_key]['Index']=ref_idx
            mkt_share_per_incl_leaders[msp_key].set_index('Index',inplace=True)
            mkt_share_per_incl_leaders[msp_key].sort_index(inplace=True)
        return mkt_share_per_incl_leaders
         
    mkt_share_per_incl_leaders=marketShareSorted(mkt_share_per_incl_leaders)
    
    
    
    
    
    def savingsLogic2MftMax(mkt_share_per_incl_leaders,df_savings):
        msp_mft2=copy.deepcopy(mkt_share_per_incl_leaders)
        for mem_key in msp_mft2.keys():
            dist_mft=msp_mft2[mem_key]['Manufacturer'].unique().tolist()
            #print(dist_mft)
            ref_elim=[(dm,np.sum(msp_mft2[mem_key][msp_mft2[mem_key]['Manufacturer']==dm].index.tolist())) for dm in dist_mft]
            ref_elim.sort(key = lambda x: x[1]) 
            #print(ref_elim)
            #print(ref_elim[2:])
            if len(ref_elim)>2:
                for elim_ele in ref_elim[2:]:
                    df_elim=msp_mft2[mem_key][msp_mft2[mem_key]['Manufacturer']==elim_ele[0]]
                    per_shift=np.sum(df_elim['percent_share'])
                    #print(per_shift)
                    
                    msp_mft2[mem_key]=msp_mft2[mem_key][msp_mft2[mem_key]['Manufacturer']!=elim_ele[0]]
                    msp_mft2[mem_key].reset_index(drop=True,inplace = True)
                    msp_mft2[mem_key].loc[0,'percent_share']=per_shift+msp_mft2[mem_key].loc[0,'percent_share']
                    #print(msp_elim_mft1[mem_key])
            
        
            msp_mft2[mem_key]['contri_unit_cost'] = msp_mft2[mem_key].apply(lambda row : 0.01*row['percent_share']*row['brand_avg_unit_cost'],axis=1)        
            msp_mft2[mem_key]['avg_unit_cost']=[round(np.sum(msp_mft2[mem_key]['contri_unit_cost']),2)]*len(msp_mft2[mem_key])
            
            msp_mft2_grp=msp_mft2[mem_key].groupby(['Manufacturer','Brand']).groups
            msp_mft2_mkt_list=[]
            for mft in msp_mft2[mem_key]['Manufacturer'].unique().tolist():
                msp_mft2_mkt_seg=''
                for key_grp in msp_mft2_grp.keys():
                    if mft==key_grp[0]:
                        msp_mft2_mkt_seg=msp_mft2_mkt_seg+key_grp[1]+'-'+str(round(msp_mft2[mem_key][msp_mft2[mem_key]['Brand']==key_grp[1]]['percent_share'].tolist()[0],2))+'%'+','
                msp_mft2_mkt_list.append(mft+'('+msp_mft2_mkt_seg[:-1]+')')
            msp_mft2_mkt_share='--'.join(msp_mft2_mkt_list)
            msp_mft2[mem_key]['mkt_share']=[msp_mft2_mkt_share]*len(msp_mft2[mem_key])
            
        
        df_savings['max2mft_avg_cost']=[np.nan]*len(df_savings)
        df_savings['max2mft_mkt_share']=[np.nan]*len(df_savings)
        for m2_key in msp_mft2.keys():
            if m2_key.split(';')[0]=='Top_3_Leaders_for_TKR':
                for s_loop in range(len(df_savings)):  
                    if df_savings.loc[s_loop,'com_bom_main_category']==m2_key.split(';')[1] and df_savings.loc[s_loop,'com_bom_sub_category']==m2_key.split(';')[2]:
                        df_savings.loc[s_loop,'max2mft_avg_cost']=msp_mft2[m2_key]['avg_unit_cost'].unique()[0]
                        df_savings.loc[s_loop,'max2mft_mkt_share']=msp_mft2[m2_key]['mkt_share'].unique()[0]          
        
        df_savings['max2mft_total_cost']=df_savings.apply(lambda row : row['max2mft_avg_cost']*row['avg_usage'],axis=1)             
        df_savings['option3_avg_procedure_cost']= [np.sum(df_savings['max2mft_total_cost'])]*len(df_savings)
    
    
        return msp_mft2,df_savings
    
    msp_mft2,df_savings=savingsLogic2MftMax(mkt_share_per_incl_leaders,df_savings)
    
    
    
    
    
    def savingsLogic1BrandMax(mkt_share_per_incl_leaders,df_savings):
        msp_1brand = copy.deepcopy(mkt_share_per_incl_leaders)
        for mb_key in msp_1brand.keys():
            dist_mft=msp_1brand[mb_key]['Manufacturer'].unique().tolist()
            #print(dist_mft)
            ref_idx_per_shift=[]
            ref_idx_elim=[]
            for dm in dist_mft:
                ref_idx=msp_1brand[mb_key][msp_1brand[mb_key]['Manufacturer']==dm].index.tolist()
                per_shift=round(np.sum(msp_1brand[mb_key].loc[ref_idx[1:],'percent_share']),2)
                ref_idx_per_shift.append((ref_idx[0],per_shift))
                ref_idx_elim.extend(ref_idx[1:])
            #print(ref_idx_per_shift)
            #print(ref_idx_elim)
            msp_1brand[mb_key].drop(ref_idx_elim,inplace=True)        
            
            for rips in ref_idx_per_shift:
                msp_1brand[mb_key].loc[rips[0],'percent_share']=rips[1]+msp_1brand[mb_key].loc[rips[0],'percent_share']
            
            msp_1brand[mb_key]['contri_unit_cost'] = msp_1brand[mb_key].apply(lambda row : 0.01*row['percent_share']*row['brand_avg_unit_cost'],axis=1)        
            msp_1brand[mb_key]['avg_unit_cost']=[round(np.sum(msp_1brand[mb_key]['contri_unit_cost']),2)]*len(msp_1brand[mb_key])
            
            msp_1brand_grp=msp_1brand[mb_key].groupby(['Manufacturer','Brand']).groups
            msp_1brand_mkt_list=[]
            for mft in msp_1brand[mb_key]['Manufacturer'].unique().tolist():
                msp_1brand_mkt_seg=''
                for key_grp in msp_1brand_grp.keys():
                    if mft==key_grp[0]:
                        msp_1brand_mkt_seg=msp_1brand_mkt_seg+key_grp[1]+'-'+str(round(msp_1brand[mb_key][msp_1brand[mb_key]['Brand']==key_grp[1]]['percent_share'].tolist()[0],2))+'%'+','
                msp_1brand_mkt_list.append(mft+'('+msp_1brand_mkt_seg[:-1]+')')
            msp_1brand_mkt_share='--'.join(msp_1brand_mkt_list)
            msp_1brand[mb_key]['mkt_share']=[msp_1brand_mkt_share]*len(msp_1brand[mb_key])
            
        df_savings['max1brand_avg_cost']=[np.nan]*len(df_savings)
        df_savings['max1brand_mkt_share']=[np.nan]*len(df_savings)
        for m1_key in msp_1brand.keys():
            if m1_key.split(';')[0]=='Top_'+str(len(leader_names))+'_Leaders':
                for s_loop in range(len(df_savings)):  
                    if df_savings.loc[s_loop,'com_bom_main_category']==m1_key.split(';')[1] and df_savings.loc[s_loop,'com_bom_sub_category']==m1_key.split(';')[2]:
                        df_savings.loc[s_loop,'max1brand_avg_cost']=msp_1brand[m1_key]['avg_unit_cost'].unique()[0]
                        df_savings.loc[s_loop,'max1brand_mkt_share']=msp_1brand[m1_key]['mkt_share'].unique()[0]          
        
        df_savings['max1brand_total_cost']=df_savings.apply(lambda row : row['max1brand_avg_cost']*row['avg_usage'],axis=1)             
        df_savings['option4_avg_procedure_cost']= [np.sum(df_savings['max1brand_total_cost'])]*len(df_savings)
    
        return msp_1brand,df_savings
     
     
    msp_1brand,df_savings=savingsLogic1BrandMax(mkt_share_per_incl_leaders,df_savings)
    
    
    
    
    
    
    def savingsLogic3MftMax(mkt_share_per_incl_leaders,df_savings):
        msp_top_brands=copy.deepcopy(mkt_share_per_incl_leaders)
        
        def compute3mft(dist_mft,msp_top_brands,mb_key):
            ref_mft=dist_mft[2:]
            #print(ref_mft)
            if len(ref_mft)>0:
                
                df_brand_price=brand_unit_price_dict[','.join(mb_key.split(';')[1:])]
                #print(df_brand_price.index.tolist())
                #print(df_brand_price[df_brand_price.index.get_level_values(0) == 'JOHNSON & JOHNSON']['brand_avg_unit_cost'].tolist()[0])
                all_dist_mft=[m for i,m in enumerate([i[0] for i in df_brand_price.index.tolist()]) if m not in [i[0] for i in df_brand_price.index.tolist()][:i]]
                for rm in ref_mft:
                    better_mft=[]
                    for adm in all_dist_mft[:all_dist_mft.index(rm)]:
                        if adm!=rm:
                            better_mft.append(adm)
                    #print(better_mft)
                    repl_mft=[]
                    for bm in better_mft:
                        #print(dist_mft[:dist_mft.index(rm)])
                        if bm not in dist_mft[:dist_mft.index(rm)]:
                            repl_mft.append(bm)
                            
                            #repl_avg_unit_cost=df_brand_price[(df_brand_price.index.get_level_values(0) == bm[0])&(df_brand_price.index.get_level_values(1) == bm[1])]['brand_avg_unit_cost'].tolist()[0]
                   
                    if repl_mft==[] and len(better_mft)>=3:
                        repl_mft=[better_mft[2]]
                    #print(repl_mft)
                    
                    if len(repl_mft)!=0:
                        best_repl_mft_brand_idx = [i for i, tupl in enumerate(df_brand_price.index.tolist()) if tupl[0] == repl_mft[0]][0]
                        best_repl_mft_brand=df_brand_price.index.tolist()[best_repl_mft_brand_idx]
                        repl_avg_unit_cost=df_brand_price[(df_brand_price.index.get_level_values(0) == best_repl_mft_brand[0])&(df_brand_price.index.get_level_values(1) == best_repl_mft_brand[1])]['brand_avg_unit_cost'].tolist()[0]
                        
                        df_elim=msp_top_brands[mb_key][msp_top_brands[mb_key]['Manufacturer']==rm]
                        per_shift=np.sum(df_elim['percent_share'])
                        best_repl=dict([('Manufacturer',best_repl_mft_brand[0]),('Brand',best_repl_mft_brand[1]),('percent_share',per_shift),('brand_avg_unit_cost',repl_avg_unit_cost)])
                        msp_top_brands[mb_key]=msp_top_brands[mb_key][msp_top_brands[mb_key]['Manufacturer']!=rm].append(best_repl, ignore_index=True)
                        
                        msp_dupl = msp_top_brands[mb_key][msp_top_brands[mb_key].duplicated(subset=['Manufacturer','Brand'])]
                        msp_dupl_grp=msp_dupl.groupby(['Manufacturer','Brand']).agg({'percent_share':'sum'})
                        #print(msp_dupl_grp.index.tolist())
                        msp_dupl_per=[]
                        for mgi in msp_dupl_grp.index.tolist():
                            msp_dupl_per.append(msp_dupl_grp[(msp_dupl_grp.index.get_level_values(0) == mgi[0])&(msp_dupl_grp.index.get_level_values(1) == mgi[1])]['percent_share'].tolist()[0])
                            
                        msp_dupl_grp_per=[(msp_dupl_grp.index.tolist()[mdgil][0],msp_dupl_grp.index.tolist()[mdgil][1],msp_dupl_per[mdgil]) for mdgil in range(len(msp_dupl_grp.index.tolist()))]
                        #print(msp_dupl_grp_per)
                        
                        msp_top_brands[mb_key].drop_duplicates(subset=['Manufacturer','Brand'], keep='first',inplace=True)
                        for mdgp in msp_dupl_grp_per:
                            for mtb_loop in range(len(msp_top_brands[mb_key])):
                                if  msp_top_brands[mb_key].loc[mtb_loop,['Manufacturer','Brand']].tolist()==[mdgp[0],mdgp[1]]:
                                    msp_top_brands[mb_key].loc[mtb_loop,'percent_share']=mdgp[2]+msp_top_brands[mb_key].loc[mtb_loop,'percent_share']
                            
            return msp_top_brands[mb_key]
        
        for mb_key in msp_top_brands.keys():
            #print(mb_key)
            dist_mft=msp_top_brands[mb_key]['Manufacturer'].unique().tolist()
            #print(dist_mft)
            
            mft_cond=len(dist_mft)
            if mft_cond==3:
                msp_top_brands[mb_key]=compute3mft(dist_mft,msp_top_brands,mb_key)
    
            elif mft_cond>3:
                while mft_cond>3:
                    msp_top_brands[mb_key]=compute3mft(dist_mft,msp_top_brands,mb_key)
                    dist_mft=msp_top_brands[mb_key]['Manufacturer'].unique().tolist()
                    mft_cond=len(dist_mft)
    
      
    
            msp_top_brands[mb_key]['contri_unit_cost'] = msp_top_brands[mb_key].apply(lambda row : 0.01*row['percent_share']*row['brand_avg_unit_cost'],axis=1)        
            msp_top_brands[mb_key]['avg_unit_cost']=[round(np.sum(msp_top_brands[mb_key]['contri_unit_cost']),2)]*len(msp_top_brands[mb_key])
            
            msp_top_brands_grp=msp_top_brands[mb_key].groupby(['Manufacturer','Brand']).groups
            msp_top_brands_mkt_list=[]
            for mft in msp_top_brands[mb_key]['Manufacturer'].unique().tolist():
                msp_top_brands_mkt_seg=''
                for key_grp in msp_top_brands_grp.keys():
                    if mft==key_grp[0]:
                        msp_top_brands_mkt_seg=msp_top_brands_mkt_seg+key_grp[1]+'-'+str(round(msp_top_brands[mb_key][msp_top_brands[mb_key]['Brand']==key_grp[1]]['percent_share'].tolist()[0],2))+'%'+','
                msp_top_brands_mkt_list.append(mft+'('+msp_top_brands_mkt_seg[:-1]+')')
            msp_top_brands_mkt_share='--'.join(msp_top_brands_mkt_list)
            msp_top_brands[mb_key]['mkt_share']=[msp_top_brands_mkt_share]*len(msp_top_brands[mb_key])
            
        df_savings['max3mft_avg_cost']=[np.nan]*len(df_savings)
        df_savings['max3mft_mkt_share']=[np.nan]*len(df_savings)
        for m1_key in msp_top_brands.keys():
            if m1_key.split(';')[0]=='Top_'+str(len(leader_names))+'_Leaders':
                for s_loop in range(len(df_savings)):  
                    if df_savings.loc[s_loop,'com_bom_main_category']==m1_key.split(';')[1] and df_savings.loc[s_loop,'com_bom_sub_category']==m1_key.split(';')[2]:
                        df_savings.loc[s_loop,'max3mft_avg_cost']=msp_top_brands[m1_key]['avg_unit_cost'].unique()[0]
                        df_savings.loc[s_loop,'max3mft_mkt_share']=msp_top_brands[m1_key]['mkt_share'].unique()[0]          
        
        df_savings['max3mft_total_cost']=df_savings.apply(lambda row : row['max3mft_avg_cost']*row['avg_usage'],axis=1)             
        df_savings['option5_avg_procedure_cost']= [np.sum(df_savings['max3mft_total_cost'])]*len(df_savings)
    
        
        return msp_top_brands,df_savings
     
    msp_mft3,df_savings=savingsLogic3MftMax(mkt_share_per_incl_leaders,df_savings)


    return brand_unit_price_dict,mkt_share_per_incl_leaders,df_savings


brand_unit_price_dict_incl_leaders,mkt_share_per_incl_leaders,df_savings_incl_leaders=SavingsAnalysis(df_filtered_incl_leaders,pdt_cat_incl_leaders,df_bom_full_coverage_incl_leaders,bom_leaderboard_incl_leaders,mkt_share_per_incl_leaders)






#Savings for every physician using saving options
savcol=df_savings_incl_leaders.columns.tolist()
sav_col=[scol.split('_')[0] for scol in savcol if len(scol.split('_'))>3 and scol.split('_')[1]=='avg' and scol.split('_')[2]=='procedure' and scol.split('_')[3]=='cost']   

bom_board=bom_leaderboard.copy()
bom_board.drop(['total cost'],axis=1,inplace=True)


saving_options=[]
for sc in sav_col:
    bb=bom_board.copy()
    bb['Default BOM of Leaders']=[df_savings_incl_leaders[sc+'_avg_procedure_cost'].unique().tolist()[0]]*len(bb)
    saving_options.append((sc,bb))
saving_options=dict(saving_options)    
    
for skey in saving_options.keys():
    saving_options[skey]['Savings Per Procedure']= saving_options[skey].apply(lambda row : row['avg_procedure_cost']-row['Default BOM of Leaders'],axis=1)            
    saving_options[skey]['Total Savings']= saving_options[skey].apply(lambda row : row['Savings Per Procedure']*row['total procedures'],axis=1)            

    
