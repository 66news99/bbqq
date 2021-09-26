
import pandas as pd
import numpy as np
import pickle


# 예측.
with open("testpickle.pkl", "rb") as f:
    my_classifier = pickle.load(f)
my_X_scaled = "ㄴㅇㅍㅁㄴㅇㄹ"
my_proba = my_classifier.predict(my_X_scaled)
print(my_proba)

#
# st.header("예측 결과:")
# my_proba_df = pd.DataFrame(data=my_proba, columns=y_labels)
# st.write("유형별 예측 확률:  ", my_proba_df)
# st.write("가장 유력한 유형:  ", my_Y_pred)