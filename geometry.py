import numpy as np
import trimesh

def create_high_res_icosphere(subdivisions=3, radius=0.5):
    """
    Генерирует вершины, грани и adjacency для ICOSPHERE.
    
    Args:
        subdivisions (int): Количество подразделений (3 = 642 вершины)
        radius (float): Радиус сферы (0.5 для соответствия нормализации GT)
    
    Returns:
        vertices: [N, 3] координаты вершин
        faces: [F, 3] индексы треугольников
        adjacency: [N, N] нормализованная матрица смежности
        edges: [E, 2] уникальные ребра
    """
    # Создаем сферу через trimesh
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    
    vertices = np.array(mesh.vertices, dtype=np.float32)  # [N, 3]
    faces = np.array(mesh.faces, dtype=np.int64)          # [F, 3]
    
    # Создаем матрицу смежности (Adjacency Matrix)
    # 1 если вершины соединены ребром, иначе 0
    n_vertices = vertices.shape[0]
    adjacency = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            adjacency[v1, v2] = 1
            adjacency[v2, v1] = 1
            
    # Добавляем само-петли (self-loops) для GCN
    np.fill_diagonal(adjacency, 1)
            
    # Нормализация матрицы смежности (D^-0.5 * A * D^-0.5)
    degree = np.sum(adjacency, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adjacency = np.dot(np.dot(d_mat_inv_sqrt, adjacency), d_mat_inv_sqrt)

    edges = np.array(mesh.edges_unique, dtype=np.int64)  # [E, 2]
    
    print(f"[geometry.py] Создана сетка: {n_vertices} вершин, {len(faces)} граней, {len(edges)} ребер")
    
    return vertices, faces, adjacency, edges