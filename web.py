import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="划分数据集")

# 展示一级标题
st.header('划分数据集')
#数据
st.subheader('1.选择光谱数据')#二级
data_xs = st.file_uploader("请选择文件：",\
    accept_multiple_files =True, type=["csv"])

for data_x in data_xs:
    x_data = np.array(pd.read_csv(data_x, header=None)[1:])
    st.dataframe(x_data,height=250)
    #标签
    st.subheader('2.选择标签')
    data_ys = st.file_uploader("请选择文件: ",\
        accept_multiple_files = False, type=["csv"])

    if data_ys is not None:
        y_data = np.array(pd.read_csv(data_ys, header=None))

        x_col = np.transpose(x_data)
        fig, ax = plt.subplots()
        ax.plot(range(900,1701),x_col,)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig)
        with col2:
            st.dataframe(y_data,height=250)
        st.subheader(f'3.划分比例 ')  # 二级
        testsize = st.number_input('划分比例：',0.3)

        st.write(f'划分比例 {(1-testsize)*10}:{(testsize)*10}')

        #划分
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,\
            test_size = testsize, random_state=2)
        st.subheader("4.保存文件")  # 二级
        #保存
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        x_train = pd.DataFrame(x_train)
        x_train = convert_df(x_train)

        x_test = pd.DataFrame(x_test)
        x_test = convert_df(x_test)

        y_train = pd.DataFrame(y_train)
        y_train = convert_df(y_train)

        y_test = pd.DataFrame(y_test)
        y_test = convert_df(y_test)

        but1,but2,but3,but4 =st.columns(4)

        with but1:
            st.download_button(label="x_train",data=x_train,file_name='x_train.csv',mime='text/csv')

        with but2:
            st.download_button(label="x_test",data=x_test,file_name='x_test.csv',mime='text/csv')

        with but3:
            st.download_button(label="y_train",data=y_train,file_name='y_train.csv',mime='text/csv')

        with but4:
            st.download_button(label="y_test",data=y_test,file_name='y_test.csv',mime='text/csv')

