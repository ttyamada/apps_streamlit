import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

st.title("matplotlib の 3D プロット")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# データ作成
theta = np.linspace(0, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='3D spiral line')
ax.legend()

st.pyplot(fig)