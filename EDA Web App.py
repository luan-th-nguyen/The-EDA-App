# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:18:44 2021

@author: ChiGa
"""
# ---------------------------------------------------------------- # 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
# ---------------------------------------------------------------- # 
# Web App Title
st.markdown('''# **The EDA App**''')
st.markdown("---")
st.subheader("This web app performs the basic **Exploratory Data Analysis** that includes:")
c1, c2 = st.columns(2)
with c1:
    st.markdown('''
                * Read Data
                * Initial Inspections
                * Summary of Numeric type columms
                ''')
with c2:
       st.markdown('''
                * Missing Values by Columns
                * Data types by Columns
                * Data Visualization
                ''') 
st.markdown("---")
# ---------------------------------------------------------------- #
# Upload CSV data
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# ---------------------------------------------------------------- #
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    st.header("Summay:")
    col1, col2, col3 = st.columns(3)
# ---------------------------------------------------------------- #
    with col1:

        col1.subheader('**Input DataFrame**')
        col1.dataframe(df.head(df.shape[1]))
        col1.subheader('**Summary: Numeric Columns **')
        df_num = df.select_dtypes([int,float])
        col1.write(df_num.describe())
        col1.subheader('**Object Columns Data**')
        df_object = df.select_dtypes(object)
        col1.write(df_object.head(5))
        
# ---------------------------------------------------------------- #    
    with col3:
        col3.subheader('**Data Types by Columns**')
        df_data_type = pd.DataFrame({'Columns':df.columns,'Data Type':str(df.columns.dtype)})
        col3.dataframe(df_data_type)
        col3.subheader('**Null Values by Columns**')
        df_null = df.isnull().sum().to_frame('Null Values')
        df_null['Null Values %'] = round(df_null['Null Values']/df.shape[0],4)*100
        col3.dataframe(df_null)
        col3.subheader('**Columns by Data types**')
        count_int = df.select_dtypes(int).shape[1]
        count_float = df.select_dtypes(float).shape[1]
        count_datetime = df.select_dtypes('datetime').shape[1]
        count_object = df.select_dtypes(object).shape[1]
        df_by_dtype = pd.DataFrame({'Data Type':['Int','Float','Data Time','Object'],'Count':[count_int,count_float,count_datetime,count_object]})        
        col3.dataframe(df_by_dtype)
    
    st.markdown("---")
    st.header("Data Visualization:") 
#Progress        
    my_bar = st.progress(0)

# ---------------------------------------------------------------- #       
    st.subheader("Pair Plots:")
    st.markdown(" ")
    num_cols = df.select_dtypes([int,float]).columns.tolist()
    my_bar.progress(20)
    def plot_box(cols_list):
        #dff=dataframe
        data = df[num_cols]
        fig = sns.pairplot(data)
        st.pyplot(fig)
    plot_box(num_cols)
#Progress    
    
    my_bar.progress(100)
# ---------------------------------------------------------------- #
    my_bar = st.progress(0)
#Creating the correlation matrix
    st.subheader("Correlation matrix:")
    st.markdown(" ")
    corr_mat = df[num_cols].corr()
    corr_mat_mask = np.array(corr_mat)
#Creating a heatmap
    corr_mat_mask[np.tril_indices_from(corr_mat)] = False
    fig, ax = plt.subplots(constrained_layout=True,figsize=[20,20])
    ax = sns.heatmap(corr_mat, mask=corr_mat_mask, vmax=.8, square=True, annot=True, cmap='RdYlGn_r');
    st.pyplot(fig)
    my_bar.progress(100)
# ---------------------------------------------------------------- #   
else:
    st.info('Waiting for CSV file to be uploaded.')
    if st.button('Press here to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['Col A', 'Col B', 'Col C', 'Col D', 'Col E']
            )
            return a
        df = load_data()
        st.header('**Input DataFrame**')
        st.dataframe(df)
        st.write('---')

# ---------------------------------------------------------------- # 
st.markdown("---")
st.write('_Thanks for visiting!_')
st.markdown('''
            * Author   : Chinmay Gaikwad
            * Email   : chinmaygaikwad123@gmail.com
            ''')
st.markdown("---")