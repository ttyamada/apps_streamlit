import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("2つの 3D グラフを表示")

# 1つ目のグラフ：3D Surface
x = y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(np.sqrt(X**2 + Y**2))

fig1 = go.Figure(data=[go.Surface(z=Z1, x=X, y=Y)])
fig1.update_layout(title="Surface Plot 1", height=400)

st.plotly_chart(fig1)

# 2つ目のグラフ：3D Scatter
theta = np.linspace(0, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x2 = r * np.sin(theta)
y2 = r * np.cos(theta)

fig2 = go.Figure(data=[go.Scatter3d(x=x2, y=y2, z=z, mode='lines')])
fig2.update_layout(title="3D Spiral Line", height=400)

st.plotly_chart(fig2)
