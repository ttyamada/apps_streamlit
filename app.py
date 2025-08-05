# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Sin 波を描くアプリ")

freq = st.slider("周波数", 1, 10, 3)
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(freq * x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
