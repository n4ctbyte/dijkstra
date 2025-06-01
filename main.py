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

def find_euler_path(G, start_node, end_node):
    temp_G = G.copy()
    
    odd_degree_nodes = [node for node in temp_G.nodes() if temp_G.degree(node) % 2 == 1]
    
    if len(odd_degree_nodes) == 0:
        try:
            euler_path = list(nx.eulerian_path(temp_G, source=start_node))
            path_nodes = [start_node]
            for edge in euler_path:
                if edge[0] == path_nodes[-1]:
                    path_nodes.append(edge[1])
                else:
                    path_nodes.append(edge[0])
            return path_nodes
        except:
            return None
    elif len(odd_degree_nodes) == 2 and start_node in odd_degree_nodes and end_node in odd_degree_nodes:
        try:
            euler_path = list(nx.eulerian_path(temp_G, source=start_node))
            path_nodes = [start_node]
            for edge in euler_path:
                if edge[0] == path_nodes[-1]:
                    path_nodes.append(edge[1])
                else:
                    path_nodes.append(edge[0])
            return path_nodes
        except:
            return None
    else:
        missing_edges = []
        if len(odd_degree_nodes) > 2:
            for i in range(0, len(odd_degree_nodes), 2):
                if i + 1 < len(odd_degree_nodes):
                    u, v = odd_degree_nodes[i], odd_degree_nodes[i+1]
                    if not temp_G.has_edge(u, v):
                        temp_G.add_edge(u, v, weight=0)
                        missing_edges.append((u, v))
        
        try:
            euler_path = list(nx.eulerian_path(temp_G, source=start_node))
            path_nodes = [start_node]
            for edge in euler_path:
                if edge[0] == path_nodes[-1]:
                    path_nodes.append(edge[1])
                else:
                    path_nodes.append(edge[0])
            return path_nodes
        except:
            return None

def get_euler_routes():
    routes = {
        'Route 1 (Original)': ['A', 'B', 'J', 'C', 'F', 'I', 'H', 'G', 'A'],
        'Route 2': ['A', 'B', 'E', 'C', 'D', 'F', 'I', 'H', 'G', 'A'],
        'Route 3': ['A', 'G', 'H', 'I', 'F', 'C', 'E', 'B', 'A']
    }
    return routes

def calculate_route_distance(G, route):
    total_distance = 0
    for i in range(len(route) - 1):
        if G.has_edge(route[i], route[i+1]):
            total_distance += G[route[i]][route[i+1]]['weight']
    return total_distance

def visualize_euler_route(G, nodes, route, route_name):
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
    
    plt.title(f"Lintasan Euler - {route_name}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def visualize_all_graphs(G, nodes):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    pos = nx.get_node_attributes(G, 'pos')
    routes = get_euler_routes()
    
    axes[0, 0].set_title("Graf Dasar", fontsize=14, fontweight='bold')
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='#3498db', alpha=0.8, ax=axes[0, 0])
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='#95a5a6', ax=axes[0, 0])
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white', ax=axes[0, 0])
    axes[0, 0].axis('off')
    
    route_axes = [(0, 1), (1, 0), (1, 1)]
    for idx, (route_name, route) in enumerate(routes.items()):
        ax = axes[route_axes[idx]]
        ax.set_title(f"Lintasan Euler - {route_name}", fontsize=14, fontweight='bold')
        
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
    routes = get_euler_routes()
    
    st.sidebar.header("Pilihan Visualisasi")
    
    view_option = st.sidebar.radio(
        "Pilih tampilan:",
        ["Semua Route Sekaligus", "Route Individual", "Analisis Detail"]
    )
    
    if view_option == "Semua Route Sekaligus":
        st.subheader("Perbandingan Semua Lintasan Euler")
        fig_all = visualize_all_graphs(G, nodes)
        st.pyplot(fig_all)
        
        st.subheader("Ringkasan 3 Lintasan Euler Kecil (A ke I dan kembali ke A)")
        for route_name, route in routes.items():
            distance = calculate_route_distance(G, route)
            route_display = ' → '.join(route)
            st.write(f"**{route_name}:** {route_display}")
            st.write(f"**Jarak:** {distance:.2f} km | **Langkah:** {len(route)-1}")
            st.write("---")
    
    elif view_option == "Route Individual":
        selected_route = st.sidebar.selectbox("Pilih Route:", list(routes.keys()))
        
        route = routes[selected_route]
        distance = calculate_route_distance(G, route)
        
        st.subheader(f"Lintasan Euler - {selected_route}")
        st.write(f"**Rute:** {' → '.join(route)}")
        st.write(f"**Total Jarak:** {distance:.2f} km")
        st.write(f"**Jumlah Langkah:** {len(route) - 1}")
        
        fig_individual = visualize_euler_route(G, nodes, route, selected_route)
        st.pyplot(fig_individual)
        
        st.subheader("Detail Langkah")
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            if G.has_edge(u, v):
                weight = G[u][v]['weight']
                st.write(f"Langkah {i+1}: {nodes[u]['name']} → {nodes[v]['name']} ({weight} km)")
    
    else:
        st.subheader("Analisis Detail Graf dan Lintasan Euler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Informasi Graf")
            st.write(f"Jumlah Simpul: {G.number_of_nodes()}")
            st.write(f"Jumlah Sisi: {G.number_of_edges()}")
            
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
        for route_name, route in routes.items():
            distance = calculate_route_distance(G, route)
            steps = len(route) - 1
            unique_edges = len(set([(route[i], route[i+1]) for i in range(len(route)-1)]))
            route_data.append({
                'Route': route_name,
                'Jarak (km)': distance,
                'Langkah': steps,
                'Sisi Unik': unique_edges
            })
        
        import pandas as pd
        df = pd.DataFrame(route_data)
        st.table(df)
        
        best_route = min(route_data, key=lambda x: x['Jarak (km)'])
        st.success(f"Route terpendek: **{best_route['Route']}** dengan jarak {best_route['Jarak (km)']:.2f} km")

if __name__ == "__main__":
    main()
