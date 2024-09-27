import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from prediction import get_prediction

st.set_page_config(page_title='Hospital Patient Survival Prediction', page_icon="🏥", layout="wide")

model = load_model('keras_model01.h5')

# Các đặc điểm cho dropdown
features = ['apache_3j_diagnosis', 'gcs_motor_apache', 'd1_lactate_max',
            'd1_lactate_min', 'apache_4a_hospital_death_prob',
            'apache_4a_icu_death_prob', 'gcs-eyes-apache']


# Hàm thêm sidebar
def add_sidebar():
    st.sidebar.header("Predict the input for following features:")

    apache_3j_diagnosis = st.sidebar.slider('apache_3j_diagnosis', 0.0300, 2201.05, value=1.0000, format="%f")
    gcs_motor_apache = st.sidebar.slider('gcs_motor_apache', 1.0000, 6.0000, value=1.0000, format="%f")
    d1_lactate_max = st.sidebar.selectbox('d1_lactate_max:', [3.970, 5.990, 0.900, 1.100, 1.200, 2.927, 9.960, 19.500])
    d1_lactate_min = st.sidebar.selectbox('d1_lactate_min:', [2.380, 2.680, 6.860, 0.900, 1.100, 1.200, 1.000, 2.125])
    apache_4a_hospital_death_prob = st.sidebar.selectbox('apache_4a_hospital_death_prob:',
                                                         [0.990, 0.980, 0.950, 0.040, 0.030, 0.086, 0.020, 0.010])
    apache_4a_icu_death_prob = st.sidebar.selectbox('apache_4a_icu_death_prob:',
                                                    [0.950, 0.940, 0.920, 0.030, 0.043, 0.030, 0.043, 0.020, 0.010])
    gcs_eyes_apache = st.sidebar.selectbox('gcs-eyes-apache:', [1.0, 2.0, 3.0, 3.4650492139135083, 4.0])

    input_data = np.array(
        [apache_3j_diagnosis, gcs_motor_apache, d1_lactate_max, d1_lactate_min,
         apache_4a_hospital_death_prob, apache_4a_icu_death_prob, gcs_eyes_apache]).reshape(1, -1)

    return input_data


# Hàm tạo biểu đồ radar
def get_radar_chart(input_data):
    categories = ['apache_3j_diagnosis', 'gcs_motor_apache', 'd1_lactate_max',
                  'd1_lactate_min', 'apache_4a_hospital_death_prob',
                  'apache_4a_icu_death_prob', 'gcs_eyes_apache']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=input_data[0],
        theta=categories,
        fill='toself',
        name='Patient Data'
    ))

    std_err_values = [0.1] * len(categories)  # Giá trị giả định cho Standard Error
    worst_values = [0.9] * len(categories)  # Giá trị giả định cho Worst Value

    fig.add_trace(go.Scatterpolar(
        r=std_err_values,
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=worst_values,
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3]
            )),
        showlegend=True,
        title='Radar Chart for Patient Features',
        template='plotly_dark'  # Thêm template tối để hiển thị rõ hơn
    )

    st.plotly_chart(fig)


# Hàm tạo biểu đồ dự đoán
def get_prediction_wave_chart(pred):
    fig = go.Figure()

    outcomes = ['Survival', 'Death']
    probabilities = [pred[0][0], 1 - pred[0][0]]

    fig.add_trace(go.Scatter(
        x=outcomes,
        y=probabilities,
        mode='lines+markers',
        name='Probability',
        line=dict(shape='spline', smoothing=1.3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        title='Prediction Results',
        xaxis_title='Outcome',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        template='plotly_dark'
    )

    st.plotly_chart(fig)


# Hàm chính
def main():
    # Thêm CSS để trang trí nút Predict
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #218838;
        }
        </style>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>The Predicted Patient Survival</h1>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Please connect this app to your hospital system to help predict patient survival based on medical data. "
            "You can also update the measurements manually using the sliders in the sidebar.</p>",
            unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    input_data = add_sidebar()  # Nhận dữ liệu đầu vào

    # Tạo hai cột để hiển thị hai biểu đồ song song
    col1, col2 = st.columns(2)

    with col1:
        get_radar_chart(input_data)  # Vẽ biểu đồ radar trong cột đầu tiên

    submit = st.sidebar.button("Predict")

    if submit:
        pred = get_prediction(data=input_data, model=model)

        survival = 'Yes' if pred[0][0] > 0.5 else 'No'

        # Thêm màu sắc dựa vào kết quả dự đoán
        if survival == 'Yes':
            st.markdown(
                f"<h3 style='text-align: center; color: #28a745;'>The predicted Patient Survival is: {survival}</h3>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<h3 style='text-align: center; color: #dc3545;'>The predicted Patient Survival is: {survival}</h3>",
                unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align: center;'>This app can assist medical professionals in making a diagnosis, "
            "but should not be used as a substitute for a professional diagnosis.</p>", unsafe_allow_html=True)

        # Hiển thị biểu đồ dự đoán trong cột thứ hai
        with col2:
            get_prediction_wave_chart(pred)


if __name__ == '__main__':
    main()
