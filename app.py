import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("左右に表示：2つの正二十面体")

# 黄金比 τ
tau = (1 + np.sqrt(5)) / 2

# 正二十面体の頂点定義（中心位置指定）
def create_icosahedron(center=(0, 0, 0)):
    v = np.array([
        [-1,  tau,  0], [ 1,  tau,  0], [-1, -tau,  0], [ 1, -tau,  0],
        [ 0, -1,  tau], [ 0,  1,  tau], [ 0, -1, -tau], [ 0,  1, -tau],
        [ tau,  0, -1], [ tau,  0,  1], [-tau,  0, -1], [-tau,  0,  1]
    ])
    v /= np.linalg.norm(v[0])  # 半径1に正規化
    v += np.array(center)      # 中心ずらし
    return v

# 三角形面（共通）
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
    [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
]

# Plotly メッシュ作成
def plot_icosahedron(vertices, color="skyblue"):
    x, y, z = vertices.T
    i, j, k = zip(*faces)
    return go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=0.8,
            flatshading=True
        )
    ])

# カラムで分割
col1, col2 = st.columns(2)

with col1:
    st.subheader("Icosahedron A")
    v1 = create_icosahedron(center=(0, 0, 0))
    fig1 = plot_icosahedron(v1, color="lightblue")
    fig1.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Icosahedron B")
    v2 = create_icosahedron(center=(0, 0, 0))  # 同じでもOK（色だけ違う）
    fig2 = plot_icosahedron(v2, color="salmon")
    fig2.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)
