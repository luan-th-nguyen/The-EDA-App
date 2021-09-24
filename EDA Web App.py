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
import math
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
# ---------------------------------------------------------------- # 
# Web App Title
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

st.markdown('''# **The EDA App**''')
st.markdown("---")
st.subheader("This web app performs the basic **Exploratory Data Analysis** that includes:")
c1, c2 = st.columns(2)
with c1:
    st.markdown('''
                * Sample Data and Dimension
                * Summary of Object type columns
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
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file as Input data", type=["csv"])
    st.sidebar.markdown("""
[Demo CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
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
        df_object = df.select_dtypes([object,np.object])
        col1.write(df_object.head(5))
        
    st.subheader('**Data Types by Columns**')
    data = dict(df.dtypes)
    df_data_type = data
    st.write(df_data_type)
        
# ---------------------------------------------------------------- #    
    with col3:
        col3.subheader('**DataFrame Dimensions**')
        df_dim = pd.DataFrame({'Count':[df.shape[0],df.shape[1]]},index=['Rows','Columns'])
        col3.dataframe(df_dim)
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
# ---------------------------------------------------------------- #    
    figs = []   
#Progress 
    st.subheader("Pair Plots:")       
    my_bar = st.progress(0)    
    st.markdown(" ")
    num_cols = df.select_dtypes([int,float]).columns.tolist()
    my_bar.progress(20)
    def plot_box(cols_list):
        #dff=dataframe
        data = df[num_cols]
        fig = sns.pairplot(data,corner=True)
        figs.append(fig)
        st.pyplot(fig)
    plot_box(num_cols)
#Progress    
    my_bar.progress(100)
# ---------------------------------------------------------------- #
#Creating the Boxplots matrix
    st.subheader("Boxplots:")
    num_cols = df.select_dtypes([int,float]).columns.tolist()
    def box_plot(df,col_list):
        plt.figure(figsize=[30,30])
        _ = math.ceil(math.sqrt(len(num_cols)))
        fig, axs = plt.subplots(_, _, sharey=True)
        for i, _c in enumerate(num_cols):
            ax = axs.flat[i]
            ax.boxplot(df[[_c]],autorange =True,meanline =True,vert=False)
            ax.set_title(_c)
        fig.tight_layout(pad=2.5)
        figs.append(fig)
        st.pyplot(fig)
    box_plot(df,num_cols)
    # ---------------------------------------------------------------- #
#Creating the Histograms matrix
    st.subheader("Histograms:")
    obj_cols = df.select_dtypes([object]).columns.tolist()
    def box_plot(df,col_list):
        plt.figure(figsize=[60,60])
        _ = math.ceil(math.sqrt(len(obj_cols)))
        fig, axs = plt.subplots(_, _, sharey=True,figsize=[20,20])
        for i, _c in enumerate(obj_cols):
            ax = df[_c].value_counts().plot.barh()
        fig.tight_layout(pad=2.0)
        figs.append(fig)
        st.pyplot(fig)
    box_plot(df,obj_cols)
# ---------------------------------------------------------------- #
#Creating the correlation matrix
    st.subheader("Correlation matrix:")
    my_bar = st.progress(0)
    st.markdown(" ")
    my_bar.progress(10)
    corr_mat = df[num_cols].corr()
    corr_mat_mask = np.array(corr_mat)
    my_bar.progress(30)
#Creating a heatmap
    corr_mat_mask[np.tril_indices_from(corr_mat)] = False
    fig, ax = plt.subplots(constrained_layout=True,figsize=[20,20])
    my_bar.progress(50)
    ax = sns.heatmap(corr_mat, mask=corr_mat_mask, vmax=.8, square=True, annot=True, cmap='RdYlGn_r')
    figs.append(fig)
    st.pyplot(fig)
    my_bar.progress(100)
# ---------------------------------------------------------------- #  
    st.markdown("---")
    st.markdown(" ")
    st.markdown(" ")
    button = st.button('Press here to export the report')
    if button:
        pdf = FPDF()
        for fig in figs:
           pdf.add_page()
           with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name,x=50, y=40, w = 100, h = 100)
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "EDA Report")
        st.markdown(html, unsafe_allow_html=True)
# ---------------------------------------------------------------- #   
else:
    st.info('Waiting for CSV file to be uploaded.')
    if st.button('Press here to use Example Dataset '):
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

# ---------------------------------------------------------------- # 
st.markdown("---")
st.write('_Thanks for visiting!_')
st.markdown('''
            * Author   : Chinmay Gaikwad
            * Email   : chinmaygaikwad123@gmail.com
            ''')
st.markdown("---")