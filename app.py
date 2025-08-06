import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("正二十面体の回転対称軸")

# 回転軸の切り替え
show_2fold = st.checkbox("2回回転軸を表示", value=True)
show_3fold = st.checkbox("3回回転軸を表示", value=True)
show_5fold = st.checkbox("5回回転軸を表示", value=True)

# 黄金比
tau = (1 + np.sqrt(5)) / 2

# 頂点（正規化）
vertices = np.array([
    [-1,  tau,  0], [ 1,  tau,  0], [-1, -tau,  0], [ 1, -tau,  0],
    [ 0, -1,  tau], [ 0,  1,  tau], [ 0, -1, -tau], [ 0,  1, -tau],
    [ tau,  0, -1], [ tau,  0,  1], [-tau,  0, -1], [-tau,  0,  1]
])
vertices /= np.linalg.norm(vertices[0])

# 面（三角形のインデックス）
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
    [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
]

x, y, z = vertices.T
i, j, k = zip(*faces)

# 正二十面体
mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightblue',
    opacity=0.7,
    name="Icosahedron"
)

# 頂点番号
labels = go.Scatter3d(
    x=x, y=y, z=z,
    mode="text",
    text=[str(n) for n in range(len(vertices))],
    textfont=dict(size=12, color='black'),
    showlegend=False
)

# 軸描画用リスト
axis_lines = []

# ✅ 5回回転軸（頂点ペア）
if show_5fold:
    fivefold_pairs = [(0,3), (1,2), (4,7), (5,6), (8,11), (9,10)]
    for a, b in fivefold_pairs:
        va = vertices[a]
        vb = vertices[b]
        axis_lines.append(go.Scatter3d(
            x=[va[0], vb[0]],
            y=[va[1], vb[1]],
            z=[va[2], vb[2]],
            mode='lines',
            line=dict(color='purple', width=6, dash='dot'),
            name='5-fold axis'
        ))

# ✅ 3回回転軸（正三角形の面の重心ペア）
if show_3fold:
    # 20三角形 → 10ペアの重心を結ぶ（遠い順）
    centers = [np.mean([vertices[a], vertices[b], vertices[c]], axis=0) for a,b,c in faces]
    used = set()
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            c1, c2 = centers[i], centers[j]
            dist = np.linalg.norm(c1 - c2)
            if 1.9 < dist < 2.1:  # 長めの距離（同軸の面同士）
                key = tuple(sorted((i,j)))
                if key not in used:
                    axis_lines.append(go.Scatter3d(
                        x=[c1[0], c2[0]],
                        y=[c1[1], c2[1]],
                        z=[c1[2], c2[2]],
                        mode='lines',
                        line=dict(color='green', width=4),
                        name='3-fold axis'
                    ))
                    used.add(key)

# ✅ 2回回転軸（辺の中点ペア）
if show_2fold:
    # 全辺（頂点ペア）を収集
    edges = set()
    for a, b, c in faces:
        edges.update({tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))})
    edge_centers = [0.5 * (vertices[a] + vertices[b]) for a, b in edges]
    used = set()
    for i in range(len(edge_centers)):
        for j in range(i+1, len(edge_centers)):
            c1, c2 = edge_centers[i], edge_centers[j]
            dist = np.linalg.norm(c1 - c2)
            if 1.4 < dist < 1.6:  # 中距離（対辺）
                key = tuple(sorted((i,j)))
                if key not in used:
                    axis_lines.append(go.Scatter3d(
                        x=[c1[0], c2[0]],
                        y=[c1[1], c2[1]],
                        z=[c1[2], c2[2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='2-fold axis'
                    ))
                    used.add(key)

# 図の合成と表示
fig = go.Figure(data=[mesh, labels] + axis_lines)

fig.update_layout(
    scene=dict(
        aspectmode='data',
        camera=dict(projection=dict(type='orthographic'))
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    title="正二十面体の対称軸"
)

st.plotly_chart(fig, use_container_width=True)
