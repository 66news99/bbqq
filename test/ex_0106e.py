#
# Streamlit의 Iris Flower 예측 APP.
#

import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.title("Iris Flower 예측 App")



# 붓꽃 데이터를 불러온다.
my_data = load_iris()



# Y의 유형을 list로 저장해 둔다.
y_labels = ["setosa", "versicolor", "virginica"]

# X 데이터를 불러온다.
my_X = my_data["data"]

# Feature 이름.
my_features_X = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

my_df_X = pd.DataFrame(data=my_X, columns=my_features_X)


my_sepal_length = st.sidebar.slider('Sepal length', float(my_df_X.sepal_length.min()), float(my_df_X.sepal_length.max()), float(my_df_X.sepal_length.mean()))
my_sepal_width = st.sidebar.slider('Sepal width', float(my_df_X.sepal_width.min()), float(my_df_X.sepal_width.max()), float( my_df_X.sepal_width.mean()))
my_petal_length = st.sidebar.slider('Petal length', float(my_df_X.petal_length.min()), float( my_df_X.petal_length.max()), float(my_df_X.petal_length.mean()))
my_petal_width = st.sidebar.slider('Petal width', float(my_df_X.petal_width.min()), float(my_df_X.petal_width.max()), float( my_df_X.petal_width.mean()))

# 입력된 X 데이터.
st.header("입력된 X 데이터:")
my_X_raw = np.array([[my_sepal_length,my_sepal_width,my_petal_length,my_petal_width]])
my_df_X_raw = pd.DataFrame(data=my_X_raw, columns=my_features_X)
st.write(my_df_X_raw)


# 전처리된 X 데이터.
with open("../../Desktop/AIschool/3.Big Data/8월 25일/04_시각화와 분석_업데이트 2/data/my_iris_scaler.pkl", "rb") as f:
    my_scaler = pickle.load(f)
my_X_scaled = my_scaler.transform(my_X_raw)     # fit_transform이 아닌 transform!!

st.header("전처리된 X 데이터:")
my_df_X_scaled = pd.DataFrame(data=my_X_scaled, columns=my_features_X)
st.write(my_df_X_scaled)

# 예측.
with open("../../Desktop/AIschool/3.Big Data/8월 25일/04_시각화와 분석_업데이트 2/data/my_iris_classifier.pkl", "rb") as f:
    my_classifier = pickle.load(f)

my_proba = my_classifier.predict_proba(my_X_scaled)
my_Y_pred = y_labels[my_classifier.predict(my_X_scaled)[0]]

st.header("예측 결과:")
my_proba_df = pd.DataFrame(data=my_proba, columns=y_labels)
st.write("유형별 예측 확률:  ", my_proba_df)
st.write("가장 유력한 유형:  ", my_Y_pred)