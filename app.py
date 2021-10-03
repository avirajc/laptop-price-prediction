import streamlit as st
import pickle
import numpy as np

# model
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))
st.title(" Laptop Predictor")

# brand
company = st.selectbox("Brand", df["Company"].unique())
# type of laptop
type = st.selectbox("Type", df["TypeName"].unique())
# Ram
Ram = st.selectbox("Ram (in GB)", [0, 2, 4, 6, 12, 16, 24, 32, 64])

# weight of laptop
weight = st.number_input("Weight of laptop ")
# touchscreen
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# Ips
ips = st.selectbox("IPS", ["No", 'Yes'])

# screen size
screen_size = st.number_input("Screen Size")

# resolution

resolution = st.selectbox("Screen Resolution", ["1920x1080", "1366x768", "1600x900",
                                                "2304x1440", "3840x1800", "2880x1800"])

# cpu

cpu = st.selectbox("Cpu", df["Cpu brand"].unique())
# hdd
hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
# sdd
ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# gpu
gpu = st.selectbox("GPU", df["Gpu brand"].unique())

# os
os = st.selectbox("OS", df["os"].unique())

if st.button("Predict Price"):
    # query
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == "Yes":
        ips = 1
    else:
        ips = 0
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res*2) + (Y_res*2))*0.5/screen_size
    query = np.array([company,type,Ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The Predicted Laptop Price : " + str(int(np.exp(pipe.predict(query)[0]))))

