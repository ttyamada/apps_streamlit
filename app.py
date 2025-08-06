import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("正二十面体 + インタラクティブな五回回転軸")

# ✅ UI：チェックボックス
show_fivefold = st.checkbox("五回回転軸を表示する", value=True)

# 黄金比 τ
tau = (1 + np.sqrt(5)) / 2

# 頂点定義（正規化）
vertices = np.array([
    [-1,  tau,  0], [ 1,  tau,  0], [-1, -tau,  0], [ 1, -tau,  0],
    [ 0, -1,  tau], [ 0,  1,  tau], [ 0, -1, -tau], [ 0,  1, -tau],
    [ tau,  0, -1], [ tau,  0,  1], [-tau,  0, -1], [-tau,  0,  1]
])
vertices /= np.linalg.norm(vertices[0])

# 面（三角形）
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
    [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
]

# 頂点リスト
x, y, z = vertices.T
i, j, k = zip(*faces)

# Mesh（面）
mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightblue',
    opacity=0.8,
    flatshading=True,
    name="Icosahedron"
)

# 頂点番号ラベル
labels = go.Scatter3d(
    x=x, y=y, z=z,
    mode="text",
    text=[str(idx) for idx in range(len(vertices))],
    textfont=dict(size=12, color='black'),
    showlegend=False
)

# ==== 五回回転軸（6本）の定義 ====
fivefold_pairs = [
    ([0, 11, 5, 1, 7], [10, 11, 0, 10, 7]),
    ([1, 5, 9, 8, 7], [6, 7, 10, 2, 3]),
    ([2, 11, 10, 0, 4], [6, 2, 4, 3, 9]),
    ([3, 9, 4, 2, 6], [0, 11, 4, 5, 1]),
    ([8, 9, 1, 5, 7], [3, 6, 2, 4, 0]),
    ([10, 7, 1, 0, 11], [2, 6, 3, 9, 8])
]

fivefold_lines = []
if show_fivefold:
    for faceA, faceB in fivefold_pairs:
        centerA = np.mean([vertices[i] for i in faceA], axis=0)
        centerB = np.mean([vertices[i] for i in faceB], axis=0)
        fivefold_lines.append(
            go.Scatter3d(
                x=[centerA[0], centerB[0]],
                y=[centerA[1], centerB[1]],
                z=[centerA[2], centerB[2]],
                mode='lines',
                line=dict(color='purple', width=5, dash='dash'),
                name='5-fold axis'
            )
        )

# ==== 合成して描画 ====
fig = go.Figure(data=[mesh, labels] + fivefold_lines)

fig.update_layout(
    scene=dict(
        aspectmode="data",
        camera=dict(
            projection=dict(type="orthographic")
        )
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    title="正二十面体 + 五回回転軸（インタラクティブ）"
)

st.plotly_chart(fig, use_container_width=True)
