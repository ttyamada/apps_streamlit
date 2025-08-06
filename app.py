import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("Icosahedron")

# チェックボックス群
show_2fold = st.checkbox("2-fold axis", value=False)
show_3fold = st.checkbox("3-fold axis", value=False)
show_5fold = st.checkbox("5-fold axis", value=False)
show_coords = st.checkbox("XYZ Coordinates", value=True)

tau = (1 + np.sqrt(5)) / 2

# 頂点定義
vertices = np.array([
    [-1,  tau,  0], 
    [ 1,  tau,  0], 
    [-1, -tau,  0], 
    [ 1, -tau,  0],
    [ 0, -1,  tau], 
    [ 0,  1,  tau], 
    [ 0, -1, -tau], 
    [ 0,  1, -tau],
    [ tau,  0, -1], 
    [ tau,  0,  1], 
    [-tau,  0, -1], 
    [-tau,  0,  1]
])
# 正規化
#vertices /= np.linalg.norm(vertices[0])

# 面（三角形）インデックス
faces = [
    [  0, 11,  5],
    [  0,  5,  1],
    [  0,  1,  7],
    [  0,  7, 10],
    [  0, 10, 11],
    [  1,  5,  9],
    [  5, 11,  4],
    [ 11, 10,  2],
    [ 10,  7,  6],
    [  7,  1,  8],
    [  3,  9,  4],
    [  3,  4,  2],
    [  3,  2,  6],
    [  3,  6,  8],
    [  3,  8,  9],
    [  4,  9,  5],
    [  2,  4, 11],
    [  6,  2, 10],
    [  8,  6,  7],
    [  9,  8,  1]
]

#x, y, z = vertices.T
#i, j, k = zip(*faces)

# 正二十面体メッシュ
#mesh = go.Mesh3d(
#    x=x, y=y, z=z,
#    i=i, j=j, k=k,
#    color='lightblue',
#    opacity=0.1,
#    name="Icosahedron"
#)

# 面（Mesh3D）
x, y, z = vertices.T
i, j, k = zip(*faces)
mesh = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightblue',
    opacity=0.1,
    flatshading=True,
    name='Icosahedron'
)

# 辺（重複を避けるため、set を使ってユニークな辺を抽出）
edge_set = set()
for face in faces:
    for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
        edge = tuple(sorted((a, b)))  # 順序を正規化
        edge_set.add(edge)

# 辺を線として描画（黒線）
edges_ico = []
for a, b in edge_set:
    ax, ay, az = vertices[a]
    bx, by, bz = vertices[b]
    edges_ico.append(
        go.Scatter3d(
            x=[ax, bx], y=[ay, by], z=[az, bz],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )


# 頂点番号は常に表示
vertex_labels = [str(i) for i in range(len(vertices))]

# hover用テキスト（頂点座標）表示切替
if show_coords:
    hover_texts = [f"vertex {i}<br>x={v[0]:.3f}<br>y={v[1]:.3f}<br>z={v[2]:.3f}" for i, v in enumerate(vertices)]
else:
    hover_texts = [""] * len(vertices)

# 頂点Scatter3d
points = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    marker=dict(size=4, color='blue'),
    text=vertex_labels,
    textposition='top center',
    hoverinfo='text',
    hovertext=hover_texts,
    name="Vertices"
)

axis_lines = []

# 5回軸（対頂点）
if show_5fold:
    fivefold_pairs = [(0,3), (1,2), (4,7), (5,6), (8,11), (9,10)]
    for a,b in fivefold_pairs:
        va = vertices[a]
        vb = vertices[b]
        axis_lines.append(go.Scatter3d(
            x=[va[0], vb[0]],
            y=[va[1], vb[1]],
            z=[va[2], vb[2]],
            mode='lines',
            line=dict(color='red', width=4),
            name='5fold axis'
        ))

# 3回軸（面の重心の対称ペア）
if show_3fold:
    face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])
    threefold_pairs = []
    used = set()
    for i, c1 in enumerate(face_centers):
        for j, c2 in enumerate(face_centers):
            if i >= j:
                continue
            if np.allclose(c1 + c2, np.zeros(3), atol=1e-5):
                if (j,i) not in used:
                    threefold_pairs.append((c1, c2))
                    used.add((i,j))
    for c1, c2 in threefold_pairs:
        axis_lines.append(go.Scatter3d(
            x=[c1[0], c2[0]],
            y=[c1[1], c2[1]],
            z=[c1[2], c2[2]],
            mode='lines',
            line=dict(color='green', width=4),
            name='3fold axis'
        ))

# 2回軸（辺の中点の対称ペア）
if show_2fold:
    edges = set()
    for a,b,c in faces:
        edges.add(tuple(sorted((a,b))))
        edges.add(tuple(sorted((b,c))))
        edges.add(tuple(sorted((c,a))))
    edge_centers = np.array([0.5*(vertices[a]+vertices[b]) for a,b in edges])
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
    for e1, e2 in twofold_pairs:
        axis_lines.append(go.Scatter3d(
            x=[e1[0], e2[0]],
            y=[e1[1], e2[1]],
            z=[e1[2], e2[2]],
            mode='lines',
            line=dict(color='blue', width=4),
            name='2fold axis'
        ))

fig = go.Figure(data=[mesh, points] + edges_ico + axis_lines)

fig.update_layout(
    scene=dict(
        aspectmode='data',
        camera=dict(projection=dict(type='orthographic'))
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    title="Icosahedron"
)

st.plotly_chart(fig, use_container_width=True)
