import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("正二十面体：面・辺・頂点・ベクトルの全表示")

# 黄金比 τ
tau = (1 + np.sqrt(5)) / 2

# 頂点座標（正規化）
vertices = np.array([
    [-1,  tau,  0], [ 1,  tau,  0], [-1, -tau,  0], [ 1, -tau,  0],
    [ 0, -1,  tau], [ 0,  1,  tau], [ 0, -1, -tau], [ 0,  1, -tau],
    [ tau,  0, -1], [ tau,  0,  1], [-tau,  0, -1], [-tau,  0,  1]
])
vertices /= np.linalg.norm(vertices[0])

# 面（三角形）インデックス
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
    [1, 5, 9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
]

# 面（Mesh3D）
x, y, z = vertices.T
i, j, k = zip(*faces)
mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightblue',
    opacity=0.8,
    flatshading=True,
    name='Icosahedron'
)

# 原点から頂点へのベクトル（赤）
vectors = [
    go.Scatter3d(
        x=[0, vx], y=[0, vy], z=[0, vz],
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    )
    for vx, vy, vz in vertices
]

# 頂点番号ラベル（黒テキスト）
labels = go.Scatter3d(
    x=x, y=y, z=z,
    mode='text',
    text=[str(i) for i in range(len(vertices))],
    textposition='top center',
    textfont=dict(size=14, color='black'),
    showlegend=False
)

# 辺（重複を避けるため、set を使ってユニークな辺を抽出）
edge_set = set()
for face in faces:
    for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
        edge = tuple(sorted((a, b)))  # 順序を正規化
        edge_set.add(edge)

# 辺を線として描画（黒線）
edges = []
for a, b in edge_set:
    ax, ay, az = vertices[a]
    bx, by, bz = vertices[b]
    edges.append(
        go.Scatter3d(
            x=[ax, bx], y=[ay, by], z=[az, bz],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )

# すべてまとめて描画
fig = go.Figure(data=[mesh] + edges + vectors + [labels])

fig.update_layout(
    scene=dict(aspectmode='data'),
    margin=dict(l=0, r=0, t=30, b=0),
    title="正二十面体：辺・ベクトル・番号付き"
)

st.plotly_chart(fig, use_container_width=True)
