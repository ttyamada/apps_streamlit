import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("正二十面体 + インタラクティブな五回回転軸")

# チェックボックス：軸を表示するか
show_axes = st.checkbox("五回回転軸を表示する", value=True)

# 黄金比
tau = (1 + np.sqrt(5)) / 2

# 頂点定義（正規化）
vertices = np.array([
    [-1,  tau,  0], [ 1,  tau,  0], [-1, -tau,  0], [ 1, -tau,  0],
    [ 0, -1,  tau], [ 0,  1,  tau], [ 0, -1, -tau], [ 0,  1, -tau],
    [ tau,  0, -1], [ tau,  0,  1], [-tau,  0, -1], [-tau,  0,  1]
])
vertices /= np.linalg.norm(vertices[0])

# 面（三角形インデックス）
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
    [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
]

# 頂点
x, y, z = vertices.T
i, j, k = zip(*faces)

# 正二十面体（面）
mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightblue',
    opacity=0.8,
    name='Icosahedron'
)

# 頂点番号表示
labels = go.Scatter3d(
    x=x, y=y, z=z,
    mode='text',
    text=[str(n) for n in range(len(vertices))],
    textfont=dict(size=12, color='black'),
    showlegend=False
)

# 対の頂点インデックス（6本の五回回転軸）
fivefold_pairs = [
    (0, 3), (1, 2), (4, 7),
    (5, 6), (8, 11), (9, 10)
]

# 軸の描画
fivefold_lines = []
if show_axes:
    for a, b in fivefold_pairs:
        va = vertices[a]
        vb = vertices[b]
        line = go.Scatter3d(
            x=[va[0], vb[0]],
            y=[va[1], vb[1]],
            z=[va[2], vb[2]],
            mode='lines',
            line=dict(color='purple', width=5, dash='dot'),
            name='5-fold axis'
        )
        fivefold_lines.append(line)

# 描画オブジェクト合成
fig = go.Figure(data=[mesh, labels] + fivefold_lines)

# 描画設定（並行投影）
fig.update_layout(
    scene=dict(
        aspectmode="data",
        camera=dict(projection=dict(type="orthographic"))
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    title="正二十面体と五回回転軸"
)

# 表示
st.plotly_chart(fig, use_container_width=True)
