import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("Plotly でインタラクティブな 3D プロット")

# メッシュデータ
X, Y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))

# PlotlyのSurfaceプロット
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

fig.update_layout(title='3D Surface', autosize=True)

st.plotly_chart(fig)
