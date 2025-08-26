import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA  
import os

def run_unsupervised():
    print("\n=== INIZIO APPRENDIMENTO NON SUPERVISIONATO ===")
    try:
        print("... Caricamento dati ...")
        df = pd.read_parquet("reddit_data/cleaned_posts.parquet")
        
        if df.empty:
            raise ValueError("Dataset vuoto")
        print(f"- Dati caricati. Shape: {df.shape}")
        
    except Exception as e:
        print(f"- Errore: {str(e)}")
        print("Verifica che il file 'cleaned_posts.parquet' esista e non sia vuoto")
        return

    # Feature selezionate
    features = [
        'upvote_ratio', 
        'author_impact', 
        'post_hour', 
        'title_length', 
        'upvotes', 
        'num_comments'
    ]
    
    # Standardizzazione
    X = df[features].fillna(df[features].median())
    X_scaled = StandardScaler().fit_transform(X)

    # Ricerca k ottimale
    best_k = find_best_k(X_scaled)
    print(f"- Numero cluster ottimali: {best_k}")

    # Clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, algorithm='elkan')
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Analisi risultati del cluster
    analyze_clusters(df, features, kmeans, X_scaled)  

# Funzione per il k ottimale
def find_best_k(X_scaled, max_k=8):
    best_score = -1
    best_k = 2
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, algorithm='elkan')
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k

def analyze_clusters(df, features, kmeans, X_scaled):
    # Statistiche dei cluster
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'median'])
    cluster_stats['count'] = df.groupby('cluster').size()
    print("\n- Statistiche dei Cluster:")
    print(cluster_stats.round(2))
    
    # Analisi del tipo di contenuto
    content_dist = df.groupby('cluster')['content_type'].value_counts(normalize=True).unstack().fillna(0)
    print("\n- Distribuzione Tipi di Contenuto per Cluster:")
    print(content_dist.round(2))

    # Salvataggio del report
    os.makedirs("reddit_data/results", exist_ok=True)
    txt_path = "reddit_data/results/cluster_result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Statistiche dei Cluster:\n")
        f.write(cluster_stats.round(2).to_string())
        f.write("\n\nDistribuzione Tipi di Contenuto per Cluster:\n")
        f.write(content_dist.round(2).to_string())
    print(f"\n- Output cluster salvato in '{txt_path}'")

    # Stampa dei grafici
    plt.figure(figsize=(10, 6))
    content_dist.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Distribuzione Tipi di Contenuto per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Percentuale')
    plt.legend(title='Tipo Contenuto', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs("reddit_data/plots", exist_ok=True)
    plt.savefig('reddit_data/plots/clusters_content.png', dpi=150)
    plt.close()
    print("\n- Grafico contenuti salvato in 'reddit_data/plots/clusters_content.png'")
    
    # Grafico con centroidi 
    # Si usa PCA per ridurre le dimensioni delle feature in modo da visulaizzare i cluster
    plt.figure(figsize=(12, 8))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    clusters = df['cluster'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
    for i, cluster in enumerate(clusters):
        mask = df['cluster'] == cluster
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=colors[i],
            alpha=0.6,
            s=30,
            label=f'Cluster {cluster}'
        )
    plt.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        marker='X', s=250, c='red',
        edgecolor='black', linewidth=1.5,
        label='Centroidi'
    )
    plt.title('Distribuzione Cluster con Centroidi', fontsize=14)
    plt.xlabel('Componente Principale 1', fontsize=12)
    plt.ylabel('Componente Principale 2', fontsize=12)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.figtext(0.5, 0.01, 
                "Visualizzazione mediante PCA dei dati proiettati in 2D",
                ha="center", fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('reddit_data/plots/clusters_distribution.png', dpi=150)
    plt.close()
    print("- Grafico distribuzione salvato in 'reddit_data/plots/clusters_distribution.png'")

    df.to_parquet("reddit_data/results/clustered_data.parquet", index=False)
    print("\n- Dati salvati in 'reddit_data/results/clustered_data.parquet'")

if __name__ == "__main__":
    run_unsupervised()