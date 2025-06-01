import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

st.set_page_config(
    page_title="Graf & Lintasan Euler UNRI-PMC",
    page_icon="🗺️",
    layout="wide"
)

st.title("Visualisasi Graf & 3 Lintasan Euler UNRI Panam - RS PMC")
st.write("3 Jalur Berbeda dari Rektorat ke PMC dan Kembali (Lintasan Euler Kecil)")

def create_graph():
    G = nx.Graph()
    
    nodes = {
        'A': {'name': 'Rektorat UNRI', 'pos': (0, 0)},
        'B': {'name': 'Bundaran SM Amin', 'pos': (1, 0.5)},
        'C': {'name': 'The Premiere Hotel', 'pos': (4, 1)},
        'D': {'name': 'RS Awal Bros', 'pos': (3, 2)},
        'E': {'name': 'RS Prima Pekanbaru', 'pos': (4, 0.5)},
        'F': {'name': 'Hotel Aryaduta', 'pos': (6, 1)},
        'G': {'name': 'Pekanbaru Eye Center', 'pos': (4, 0)},
        'H': {'name': 'Grand Central Hotel', 'pos': (8, 0)},
        'I': {'name': 'RS PMC', 'pos': (9, 0)},
        'J': {'name': 'Wrung Makan Uni Mega', 'pos': (2.5, 1)},
    }
    
    for node_id, node_data in nodes.items():
        G.add_node(node_id, name=node_data['name'], pos=node_data['pos'])
    
    edges = [
        ('A', 'B', 4.2),
        ('C', 'D', 1.3),
        ('C', 'H', 1.4),
        ('H', 'I', 1.8),
        ('F', 'I', 1.8),
        ('A', 'G', 6.6),
        ('G', 'H', 6.1),
        ('G', 'B', 5.0),
        ('B', 'D', 7.3),
        ('B', 'E', 0.85),
        ('E', 'H', 7.5),
        ('C', 'E', 6.1),
        ('D', 'F', 1.5),
        ('C', 'F', 0.9),
        ('B', 'J', 4.5),
        ('J', 'C', 2.3),
    ]
    
    for u, v, w in edges:
        if G.has_edge(u, v):
            G[u][v]['weight'] = min(G[u][v]['weight'], w)
        else:
            G.add_edge(u, v, weight=w)
    
    return G, nodes

def bellman_ford(G, source):
    distances = {}
    predecessors = {}
    
    for node in G.nodes():
        distances[node] = float('inf')
        predecessors[node] = None
    distances[source] = 0
    
    for _ in range(len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
            
            if distances[v] + weight < distances[u]:
                distances[u] = distances[v] + weight
                predecessors[u] = v
    
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if distances[u] + weight < distances[v] or distances[v] + weight < distances[u]:
            raise ValueError("Graf mengandung siklus negatif")
    
    return distances, predecessors

def reconstruct_path(predecessors, source, target):
    path = []
    current = target
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    path.reverse()
    
    if path[0] != source:
        return None
    
    return path

def get_bellman_ford_route(G):
    try:
        distances_from_A, predecessors_from_A = bellman_ford(G, 'A')
        path_A_to_I = reconstruct_path(predecessors_from_A, 'A', 'I')
        
        if not path_A_to_I:
            return None
        
        G_remaining = G.copy()
        used_edges = set()
        
        for i in range(len(path_A_to_I) - 1):
            u, v = path_A_to_I[i], path_A_to_I[i+1]
            used_edges.add((u, v))
            used_edges.add((v, u))
            if G_remaining.has_edge(u, v):
                G_remaining.remove_edge(u, v)
        
        if nx.is_connected(G_remaining) and G_remaining.has_node('I') and G_remaining.has_node('A'):
            try:
                path_I_to_A = nx.shortest_path(G_remaining, 'I', 'A', weight='weight')
                distance_I_to_A = nx.shortest_path_length(G_remaining, 'I', 'A', weight='weight')
            except nx.NetworkXNoPath:
                path_I_to_A = find_alternative_path(G, path_A_to_I, 'I', 'A')
                if path_I_to_A:
                    distance_I_to_A = calculate_route_distance(G, path_I_to_A)
                else:
                    return None
        else:
            path_I_to_A = find_alternative_path(G, path_A_to_I, 'I', 'A')
            if path_I_to_A:
                distance_I_to_A = calculate_route_distance(G, path_I_to_A)
            else:
                return None
        
        path_I_to_A_edges = set()
        for i in range(len(path_I_to_A) - 1):
            u, v = path_I_to_A[i], path_I_to_A[i+1]
            edge = tuple(sorted([u, v]))
            path_I_to_A_edges.add(edge)
        
        path_A_to_I_edges = set()
        for i in range(len(path_A_to_I) - 1):
            u, v = path_A_to_I[i], path_A_to_I[i+1]
            edge = tuple(sorted([u, v]))
            path_A_to_I_edges.add(edge)
        
        common_edges = path_A_to_I_edges.intersection(path_I_to_A_edges)
        if common_edges:
            st.warning(f"Peringatan: Ada sisi yang digunakan berulang: {common_edges}")
        
        complete_path = path_A_to_I + path_I_to_A[1:]
        
        return {
            'path': complete_path,
            'distance_A_to_I': distances_from_A['I'],
            'distance_I_to_A': distance_I_to_A,
            'total_distance': distances_from_A['I'] + distance_I_to_A,
            'path_A_to_I': path_A_to_I,
            'path_I_to_A': path_I_to_A,
            'algorithm': 'Bellman-Ford + Alternative Path',
            'used_edges_A_to_I': path_A_to_I_edges,
            'used_edges_I_to_A': path_I_to_A_edges,
            'common_edges': common_edges
        }
        
    except Exception as e:
        st.error(f"Error dalam Bellman-Ford: {e}")
        return None

def find_alternative_path(G, used_path, start, end):
    used_edges = set()
    for i in range(len(used_path) - 1):
        u, v = used_path[i], used_path[i+1]
        used_edges.add(tuple(sorted([u, v])))
    
    alternative_paths = [
        ['I', 'H', 'G', 'A'],
        ['I', 'F', 'D', 'B', 'A'],
        ['I', 'H', 'C', 'J', 'B', 'A'],
        ['I', 'F', 'C', 'E', 'B', 'A'],
    ]
    
    for path in alternative_paths:
        if path[0] == start and path[-1] == end:
            path_valid = True
            path_edges = set()
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted([u, v]))
                
                if not G.has_edge(u, v):
                    path_valid = False
                    break
                    
                path_edges.add(edge)
            
            if path_valid and len(path_edges.intersection(used_edges)) == 0:
                return path
    
    return None

def get_euler_routes():
    routes = {
        'Route 2 (Manual)': ['A', 'B', 'E', 'C', 'D', 'F', 'I', 'H', 'G', 'A'],
        'Route 3 (Manual)': ['A', 'G', 'H', 'I', 'F', 'C', 'E', 'B', 'A']
    }
    return routes

def verify_euler_path(G, route):
    used_edges = set()
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        edge = tuple(sorted([u, v]))
        
        if edge in used_edges:
            return False, f"Sisi {u}-{v} digunakan berulang"
        
        if not G.has_edge(u, v):
            return False, f"Sisi {u}-{v} tidak ada dalam graf"
            
        used_edges.add(edge)
    
    return True, "Lintasan Euler valid"

def calculate_faces(G):
    if not nx.is_connected(G):
        return 1
    
    V = G.number_of_nodes()
    E = G.number_of_edges()
    
    if E == 0:
        return 1
    
    F = 2 - V + E
    return F

def calculate_route_distance(G, route):
    total_distance = 0
    for i in range(len(route) - 1):
        if G.has_edge(route[i], route[i+1]):
            total_distance += G[route[i]][route[i+1]]['weight']
    return total_distance

def visualize_euler_route(G, nodes, route, route_name, algorithm_info=None):
    plt.figure(figsize=(16, 12))
    
    pos = nx.get_node_attributes(G, 'pos')
    
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#95a5a6', alpha=0.6, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.3, edge_color='#bdc3c7')
    
    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route_edges)))
    
    for i, (edge, color) in enumerate(zip(route_edges, colors)):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=4.0, edge_color=[color], alpha=0.8)
        
        u, v = edge
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        plt.text(x, y, str(i+1), fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='circle,pad=0.1'))
    
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=1200, node_color='#27ae60', alpha=0.9, edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=[route[-1]], node_size=1200, node_color='#e67e22', alpha=0.9, edgecolors='black')
    
    intermediate_nodes = list(set(route[1:-1]))
    if intermediate_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=intermediate_nodes, node_size=1000, 
                             node_color='#3498db', alpha=0.7, edgecolors='black')
    
    node_labels = {node: f"{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', font_color='white')
    
    for node, (x, y) in pos.items():
        plt.text(x, y-0.2, nodes[node]['name'], fontsize=7, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#2c3e50')
    
    title = f"Lintasan Euler - {route_name}"
    if algorithm_info:
        title += f" ({algorithm_info})"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def visualize_all_graphs(G, nodes, bellman_route):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    pos = nx.get_node_attributes(G, 'pos')
    manual_routes = get_euler_routes()
    
    axes[0, 0].set_title("Graf Dasar", fontsize=14, fontweight='bold')
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#3498db', alpha=0.8, ax=axes[0, 0])
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='#95a5a6', ax=axes[0, 0])
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white', ax=axes[0, 0])
    axes[0, 0].axis('off')
    
    ax = axes[0, 1]
    route = bellman_route['path']
    ax.set_title("Route 1 (Bellman-Ford)", fontsize=14, fontweight='bold')
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#95a5a6', alpha=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color='#bdc3c7', ax=ax)
    
    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route_edges)))
    
    for edge, color in zip(route_edges, colors):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=3.0, edge_color=[color], alpha=0.8, ax=ax)
    
    nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=800, node_color='#27ae60', alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[route[-1]], node_size=800, node_color='#e67e22', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white', ax=ax)
    ax.axis('off')
    
    route_axes = [(1, 0), (1, 1)]
    for idx, (route_name, route) in enumerate(manual_routes.items()):
        ax = axes[route_axes[idx]]
        ax.set_title(f"{route_name}", fontsize=14, fontweight='bold')
        
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#95a5a6', alpha=0.5, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color='#bdc3c7', ax=ax)
        
        route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(route_edges)))
        
        for edge, color in zip(route_edges, colors):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=3.0, edge_color=[color], alpha=0.8, ax=ax)
        
        nx.draw_networkx_nodes(G, pos, nodelist=[route[0]], node_size=800, node_color='#27ae60', alpha=0.9, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[route[-1]], node_size=800, node_color='#e67e22', alpha=0.9, ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white', ax=ax)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    G, nodes = create_graph()
    
    bellman_route = get_bellman_ford_route(G)
    manual_routes = get_euler_routes()
    
    st.sidebar.header("Pilihan Visualisasi")
    
    view_option = st.sidebar.radio(
        "Pilih tampilan:",
        ["Semua Route Sekaligus", "Route Individual", "Analisis Bellman-Ford", "Analisis Detail"]
    )
    
    if view_option == "Semua Route Sekaligus":
        st.subheader("Perbandingan Semua Lintasan Euler")
        
        if bellman_route:
            fig_all = visualize_all_graphs(G, nodes, bellman_route)
            st.pyplot(fig_all)
            
            st.subheader("Ringkasan 3 Lintasan Euler Kecil (A ke I dan kembali ke A)")
            
            route_display = ' → '.join(bellman_route['path'])
            st.write(f"**Route 1 (Bellman-Ford):** {route_display}")
            st.write(f"**Jarak:** {bellman_route['total_distance']:.2f} km | **Langkah:** {len(bellman_route['path'])-1}")
            st.write("---")
            
            for route_name, route in manual_routes.items():
                distance = calculate_route_distance(G, route)
                route_display = ' → '.join(route)
                st.write(f"**{route_name}:** {route_display}")
                st.write(f"**Jarak:** {distance:.2f} km | **Langkah:** {len(route)-1}")
                st.write("---")
    
    elif view_option == "Route Individual":
        all_routes = {}
        if bellman_route:
            all_routes['Route 1 (Bellman-Ford)'] = bellman_route['path']
        all_routes.update(manual_routes)
        
        selected_route = st.sidebar.selectbox("Pilih Route:", list(all_routes.keys()))
        
        route = all_routes[selected_route]
        distance = calculate_route_distance(G, route)
        
        st.subheader(f"Lintasan Euler - {selected_route}")
        st.write(f"**Rute:** {' → '.join(route)}")
        st.write(f"**Total Jarak:** {distance:.2f} km")
        st.write(f"**Jumlah Langkah:** {len(route) - 1}")
        
        algorithm_info = "Bellman-Ford Algorithm" if "Bellman-Ford" in selected_route else None
        fig_individual = visualize_euler_route(G, nodes, route, selected_route, algorithm_info)
        st.pyplot(fig_individual)
        
        st.subheader("Detail Langkah")
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            if G.has_edge(u, v):
                weight = G[u][v]['weight']
                st.write(f"Langkah {i+1}: {nodes[u]['name']} → {nodes[v]['name']} ({weight} km)")
    
    elif view_option == "Analisis Bellman-Ford":
        st.subheader("Analisis Algoritma Bellman-Ford untuk Route 1")
        
        if bellman_route:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Jalur A → I (Bellman-Ford)")
                path_A_to_I = bellman_route['path_A_to_I']
                st.write(f"**Jalur:** {' → '.join(path_A_to_I)}")
                st.write(f"**Jarak:** {bellman_route['distance_A_to_I']:.2f} km")
                
                st.markdown("### Detail Langkah A → I")
                for i in range(len(path_A_to_I) - 1):
                    u, v = path_A_to_I[i], path_A_to_I[i+1]
                    if G.has_edge(u, v):
                        weight = G[u][v]['weight']
                        st.write(f"{i+1}. {nodes[u]['name']} → {nodes[v]['name']} ({weight} km)")
            
            with col2:
                st.markdown("### Jalur I → A (Bellman-Ford)")
                path_I_to_A = bellman_route['path_I_to_A']
                st.write(f"**Jalur:** {' → '.join(path_I_to_A)}")
                st.write(f"**Jarak:** {bellman_route['distance_I_to_A']:.2f} km")
                
                st.markdown("### Detail Langkah I → A")
                for i in range(len(path_I_to_A) - 1):
                    u, v = path_I_to_A[i], path_I_to_A[i+1]
                    if G.has_edge(u, v):
                        weight = G[u][v]['weight']
                        st.write(f"{i+1}. {nodes[u]['name']} → {nodes[v]['name']} ({weight} km)")
            
            st.markdown("### Ringkasan Route 1")
            st.success(f"**Total Jarak Route 1:** {bellman_route['total_distance']:.2f} km")
            st.info(f"**Jalur Lengkap:** {' → '.join(bellman_route['path'])}")
            
            fig_bf = visualize_euler_route(G, nodes, bellman_route['path'], "Route 1", "Bellman-Ford Algorithm")
            st.pyplot(fig_bf)
            
            st.markdown("### Analisis Penggunaan Sisi")
            st.write(f"**Sisi A→I:** {len(bellman_route['used_edges_A_to_I'])} sisi")
            st.write(f"**Sisi I→A:** {len(bellman_route['used_edges_I_to_A'])} sisi")
            
            if bellman_route['common_edges']:
                st.error(f"❌ Sisi yang digunakan berulang: {bellman_route['common_edges']}")
            else:
                st.success("✅ Tidak ada sisi yang digunakan berulang - Lintasan Euler valid!")
            
            with st.expander("Detail Sisi yang Digunakan"):
                st.write("**Sisi Jalur A→I:**")
                for edge in sorted(bellman_route['used_edges_A_to_I']):
                    st.write(f"- {edge[0]} ↔ {edge[1]}")
                
                st.write("**Sisi Jalur I→A:**")
                for edge in sorted(bellman_route['used_edges_I_to_A']):
                    st.write(f"- {edge[0]} ↔ {edge[1]}")
            
            st.markdown("### Tentang Algoritma Bellman-Ford + Jalur Alternatif")
            st.write("""
            **Strategi yang Digunakan:**
            1. **Bellman-Ford** untuk jalur terpendek A→I
            2. **Algoritma Alternatif** untuk jalur I→A yang tidak menggunakan sisi yang sama
            
            **Keunggulan:**
            - Memenuhi syarat Lintasan Euler (tidak ada sisi berulang)
            - Jalur A→I tetap optimal (Bellman-Ford)
            - Jalur I→A dipilih dari alternatif yang tersedia
            
            **Proses:**
            1. Hitung jalur terpendek A→I dengan Bellman-Ford
            2. Hapus sisi yang sudah digunakan dari graf
            3. Cari jalur I→A dengan sisi yang tersisa
            4. Verifikasi tidak ada sisi yang berulang
            """)
        else:
            st.error("Gagal menghitung route menggunakan Bellman-Ford")
    
    else:
        st.subheader("Analisis Detail Graf dan Lintasan Euler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Informasi Graf")
            V = G.number_of_nodes()
            E = G.number_of_edges()
            F = calculate_faces(G)
            
            st.write(f"Jumlah Simpul (V): {V}")
            st.write(f"Jumlah Sisi (E): {E}")
            st.write(f"Jumlah Muka (F): {F}")
            
            st.markdown("### Rumus Euler untuk Graf Planar")
            euler_result = V - E + F
            st.write(f"V - E + F = {V} - {E} + {F} = {euler_result}")
            
            if euler_result == 2:
                st.success("✓ Rumus Euler terpenuhi (V - E + F = 2)")
            else:
                st.warning(f"⚠ Rumus Euler: V - E + F = {euler_result} (≠ 2)")
            
            degrees = dict(G.degree())
            odd_nodes = [node for node, degree in degrees.items() if degree % 2 == 1]
            even_nodes = [node for node, degree in degrees.items() if degree % 2 == 0]
            
            st.write(f"Simpul berderajat ganjil: {len(odd_nodes)}")
            st.write(f"Simpul berderajat genap: {len(even_nodes)}")
            
            if len(odd_nodes) == 0:
                st.success("Graf memiliki sirkuit Euler!")
            elif len(odd_nodes) == 2:
                st.info("Graf memiliki lintasan Euler!")
            else:
                st.warning("Graf tidak memiliki lintasan Euler!")
        
        with col2:
            st.markdown("### Derajat Setiap Simpul")
            for node in sorted(G.nodes()):
                degree = G.degree(node)
                status = "Ganjil" if degree % 2 == 1 else "Genap"
                st.write(f"{node}: {degree} ({status})")
        
        st.subheader("Perbandingan Efisiensi Route")
        
        route_data = []
        
        if bellman_route:
            is_valid, validation_msg = verify_euler_path(G, bellman_route['path'])
            route_data.append({
                'Route': 'Route 1 (Bellman-Ford)',
                'Jarak (km)': bellman_route['total_distance'],
                'Langkah': len(bellman_route['path']) - 1,
                'Algoritma': 'Bellman-Ford + Alternative',
                'Valid Euler': '✅' if is_valid else '❌'
            })
        
        for route_name, route in manual_routes.items():
            distance = calculate_route_distance(G, route)
            steps = len(route) - 1
            is_valid, validation_msg = verify_euler_path(G, route)
            route_data.append({
                'Route': route_name,
                'Jarak (km)': distance,
                'Langkah': steps,
                'Algoritma': 'Manual',
                'Valid Euler': '✅' if is_valid else '❌'
            })
        
        import pandas as pd
        df = pd.DataFrame(route_data)
        st.table(df)
        
        if route_data:
            valid_routes = [r for r in route_data if '✅' in r['Valid Euler']]
            if valid_routes:
                best_route = min(valid_routes, key=lambda x: x['Jarak (km)'])
                st.success(f"Route terpendek (valid): **{best_route['Route']}** dengan jarak {best_route['Jarak (km)']:.2f} km")
            else:
                st.warning("Tidak ada route yang valid sebagai Lintasan Euler!")
        
        st.subheader("Validasi Lintasan Euler")
        
        all_routes = {}
        if bellman_route:
            all_routes['Route 1 (Bellman-Ford)'] = bellman_route['path']
        all_routes.update(manual_routes)
        
        for route_name, route in all_routes.items():
            is_valid, validation_msg = verify_euler_path(G, route)
            if is_valid:
                st.success(f"✅ **{route_name}**: {validation_msg}")
            else:
                st.error(f"❌ **{route_name}**: {validation_msg}")

if __name__ == "__main__":
    main()
