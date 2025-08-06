import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("正二十面体の回転対称軸（正確版）")

# チェックボックスで軸の表示切替
show_2fold = st.checkbox("2回回転軸を表示", value=True)
show_3fold = st.checkbox("3回回転軸を表示", value=True)
show_5fold = st.checkbox("5回回転軸を表示", value=True)

# 黄金比
tau = (1 + np.sqrt(5)) / 2

# 頂点の定義（正規化）
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

# 正二十面体メッシュ
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

axis_lines = []

# ----- 5回回転軸（対頂点） -----
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

# ----- 3回回転軸（面の重心の対ペア） -----
if show_3fold:
    # 面の重心
    face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])
    # 中心対称な重心ペアを抽出
    threefold_pairs = []
    used = set()
    for i, c1 in enumerate(face_centers):
        for j, c2 in enumerate(face_centers):
            if i >= j:
                continue
            # 中心対称なので c1 + c2 ≈ 0 のはず
            if np.allclose(c1 + c2, np.zeros(3), atol=1e-5):
                if (j,i) not in used:
                    threefold_pairs.append((c1, c2))
                    used.add((i,j))
    # 描画
    for c1, c2 in threefold_pairs:
        axis_lines.append(go.Scatter3d(
            x=[c1[0], c2[0]],
            y=[c1[1], c2[1]],
            z=[c1[2], c2[2]],
            mode='lines',
            line=dict(color='green', width=4),
            name='3-fold axis'
        ))

# ----- 2回回転軸（辺の中点の対称ペア） -----
if show_2fold:
    # 辺集合
    edges = set()
    for a,b,c in faces:
        edges.add(tuple(sorted((a,b))))
        edges.add(tuple(sorted((b,c))))
        edges.add(tuple(sorted((c,a))))
    edge_centers = np.array([0.5*(vertices[a] + vertices[b]) for a,b in edges])
    # 中心対称ペアを抽出
    twofold_pairs = []
    used = set()
    for i, e1 in enumerate(edge_centers):
        for j, e2 in enumerate(edge_centers):
            if i >= j:
                continue
            if np.allclose(e1 + e2, np.zeros(3), atol=1e-5):
                if (j,i) not in used:
                    twofold_pairs.append((e1, e2))
                    used.add((i,j))
    # 描画
    for e1, e2 in twofold_pairs:
        axis_lines.append(go.Scatter3d(
            x=[e1[0], e2[0]],
            y=[e1[1], e2[1]],
            z=[e1[2], e2[2]],
            mode='lines',
            line=dict(color='blue', width=3),
            name='2-fold axis'
        ))

# 描画設定
fig = go.Figure(data=[mesh, labels] + axis_lines)
fig.update_layout(
    scene=dict(
        aspectmode='data',
        camera=dict(projection=dict(type='orthographic'))
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    title="正二十面体の回転対称軸（正確版）"
)

import streamlit as st
st.plotly_chart(fig, use_container_width=True)
