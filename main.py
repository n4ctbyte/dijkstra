import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

st.set_page_config(
    page_title="Graf & Lintasan Terpendek UNRI-PMC",
    page_icon="🗺️",
    layout="wide"
)

st.title("Visualisasi Graf & Lintasan Terpendek UNRI Panam - RS PMC")
st.write("Tugas Akhir Matematika Diskrit")

def create_graph():
    G = nx.Graph()
    
    nodes = {
        'A': {'name': 'Rektorat UNRI', 'pos': (0, 0)},
        'B': {'name': 'Bundaran SM Amin', 'pos': (1, 1)},
        'C': {'name': 'The Premiere Hotel', 'pos': (3, 1)},
        'D': {'name': 'RS Awal Bros', 'pos': (3, 2)},
        'E': {'name': 'RS Prima Pekanbaru', 'pos': (4, 0.5)},
        'F': {'name': 'Hotel Aryaduta', 'pos': (5, 1)},
        'G': {'name': 'Pekanbaru Eye Center', 'pos': (1, 0)},
        
        'I': {'name': 'Grand Central Hotel', 'pos': (8, 0)},
        'J': {'name': 'RS PMC', 'pos': (9, 0)},
    }
    
    for node_id, node_data in nodes.items():
        G.add_node(node_id, name=node_data['name'], pos=node_data['pos'])
    
    edges = [
        ('A', 'B', 4.2),
        ('B', 'C', 6.6),
        ('C', 'D', 1.3),
        ('C', 'I', 1.4),
        ('I', 'J', 1.7),
        ('F', 'J', 1.8),
        ('A', 'G', 6.6),
        ('G', 'I', 6.1),
        ('G', 'B', 5.0),
        ('B', 'D', 7.3),
        ('A', 'B', 4.2),
        ('B', 'E', 1.2),
        ('B', 'E', 0.8),
        ('E', 'I', 7.0),
        ('I', 'J', 0.8),
        ('C', 'E', 6.1),
        ('D', 'F', 1.5),
        ('C', 'F', 0.9),
    ]
    
    for u, v, w in edges:
        if G.has_edge(u, v):
            G[u][v]['weight'] = min(G[u][v]['weight'], w)
        else:
            G.add_edge(u, v, weight=w)
    
    return G, nodes

def calculate_euler_formula(G):
    V = G.number_of_nodes()
    E = G.number_of_edges()
    F = 2 + E - V
    return V, E, F

def visualize_graph(G, nodes, path=None):
    plt.figure(figsize=(14, 10))
    
    pos = nx.get_node_attributes(G, 'pos')
    
    node_sizes = [900 for _ in G.nodes()]
    node_colors = ['#3498db' for _ in G.nodes()]
    edge_colors = ['#95a5a6' for _ in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color=edge_colors)
    
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4.0, edge_color='#e74c3c', alpha=0.9)
        
        path_nodes = path[1:-1]
        highlight_sizes = [node_sizes[0] * 1.2 for _ in path_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_size=highlight_sizes, node_color='#f39c12', alpha=0.85, edgecolors='black')
        
        nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=node_sizes[0] * 1.4, node_color='#2ecc71', alpha=0.9, edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=node_sizes[0] * 1.4, node_color='#e74c3c', alpha=0.9, edgecolors='black')
    
    node_labels = {node: f"{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', font_color='white')
    
    for node, (x, y) in pos.items():
        plt.text(x, y-0.15, nodes[node]['name'], fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v} km" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#2c3e50')
    
    plt.title("Graf Jalur UNRI Panam - RS PMC Pekanbaru", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def find_shortest_path(G, source, target):
    try:
        path = nx.dijkstra_path(G, source=source, target=target, weight='weight')
        length = nx.dijkstra_path_length(G, source=source, target=target, weight='weight')
        return path, length
    except nx.NetworkXNoPath:
        return None, float('inf')

def main():
    G, nodes = create_graph()
    
    st.sidebar.header("Konfigurasi")
    
    start_node = st.sidebar.selectbox("Pilih Titik Awal:", list(nodes.keys()), 
                                      format_func=lambda x: f"{x}: {nodes[x]['name']}")
    end_node = st.sidebar.selectbox("Pilih Titik Akhir:", list(nodes.keys()), 
                                    format_func=lambda x: f"{x}: {nodes[x]['name']}", index=8)
    
    calculate_return = st.sidebar.checkbox("Hitung jalur pulang juga")
    
    if st.sidebar.button("Cari Lintasan Terpendek", key="find_path"):
        path_to_pmc, length_to_pmc = find_shortest_path(G, start_node, end_node)
        
        if calculate_return:
            path_from_pmc, length_from_pmc = find_shortest_path(G, end_node, start_node)
            total_length = length_to_pmc + length_from_pmc
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Jalur Pergi")
                if path_to_pmc:
                    st.markdown(f"**Jalur Pergi:** {' → '.join(path_to_pmc)}")
                    st.markdown(f"**Jarak:** {length_to_pmc:.2f} km")
                    
                    fig_to = visualize_graph(G, nodes, path_to_pmc)
                    st.pyplot(fig_to)
                else:
                    st.error("Tidak ada jalur yang ditemukan!")
            
            with col2:
                st.subheader("Jalur Pulang")
                if path_from_pmc:
                    st.markdown(f"**Jalur Pulang:** {' → '.join(path_from_pmc)}")
                    st.markdown(f"**Jarak:** {length_from_pmc:.2f} km")
                    
                    fig_from = visualize_graph(G, nodes, path_from_pmc)
                    st.pyplot(fig_from)
                else:
                    st.error("Tidak ada jalur yang ditemukan!")
            
            st.success(f"Total jarak (pergi-pulang): {total_length:.2f} km")
            
        else:
            if path_to_pmc:
                st.markdown(f"### Lintasan Terpendek")
                st.markdown(f"**Rute:** {' → '.join(path_to_pmc)}")
                st.markdown(f"**Jarak:** {length_to_pmc:.2f} km")
                
                fig = visualize_graph(G, nodes, path_to_pmc)
                st.pyplot(fig)
            else:
                st.error("Tidak ada jalur yang ditemukan!")
    
    st.sidebar.subheader("Informasi Graf")
    v, e, f = calculate_euler_formula(G)
    st.sidebar.write(f"Jumlah Simpul (V): {v}")
    st.sidebar.write(f"Jumlah Sisi (E): {e}")
    st.sidebar.write(f"Jumlah Muka (F): {f}")
    st.sidebar.write(f"Rumus Euler: V - E + F = {v} - {e} + {f} = {v - e + f}")
    
    # Tampilkan graf dasar hanya jika belum ada hasil pencarian yang ditampilkan
    if "find_path" not in st.session_state or not st.session_state.find_path:
        st.subheader("Visualisasi Graf Dasar")
        fig_base = visualize_graph(G, nodes)
        st.pyplot(fig_base)
    
    with st.expander("Informasi Tentang Graf Ini"):
        st.write("""
        Graf ini memodelkan jalur-jalur potensial dari Kampus UNRI Panam ke RS PMC Pekanbaru.
        
        - **Simpul (Vertex)**: Merepresentasikan lokasi penting seperti perempatan, pertigaan, atau landmark.
        - **Sisi (Edge)**: Merepresentasikan jalan yang menghubungkan dua lokasi.
        - **Bobot (Weight)**: Merepresentasikan jarak dalam kilometer.
        
        Graf ini adalah contoh graf bidang (planar graph) yang memenuhi rumus Euler:
        
        V - E + F = 2
        
        di mana:
        - V = jumlah simpul
        - E = jumlah sisi
        - F = jumlah muka (termasuk muka tak terhingga)
        """)
    
    with st.expander("Daftar Simpul dan Sisi"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Simpul (Vertex)")
            for node_id, node_data in nodes.items():
                st.write(f"{node_id}: {node_data['name']}")
        
        with col2:
            st.subheader("Sisi (Edge)")
            for u, v, data in G.edges(data=True):
                st.write(f"{u} -- {v}: {data['weight']} km")

if __name__ == "__main__":
    main()