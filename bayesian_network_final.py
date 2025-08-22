import os
import pandas as pd
import logging
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import networkx as nx
import matplotlib.pyplot as plt

# Comando per disattivare warning
logging.getLogger("pgmpy").setLevel(logging.WARNING)


def add_new_features(df):
    # Definizione delle nuove feature
    df = df.copy()
    df['engagement_score'] = df['upvotes'] + df['num_comments']
    df['title_complexity'] = df['title_length'] / df['title_length'].mean()
    return df


def calculate_is_viral(df):
    # Definizione di is_viral (come nella fasi precedenti)
    df = df.copy()
    viral_criteria = [
        df['upvote_ratio'] >= 0.92,
        df['author_impact'] >= 9.8,
        df['upvotes'] >= 2000,
        df['num_comments'] >= 180,
        df['title_length'] >= 40
    ]
    df['is_viral'] = (sum(viral_criteria) >= 3).astype(int)
    return df


def discretize_variables(df):
    # Discretizzazione delle variabili
    df = df.copy()
    bins_dict = {
        'upvotes': [0, 2000, 4000, float('inf')],
        'num_comments': [0, 100, 300, float('inf')],
        'upvote_ratio': [0, 0.5, 0.75, 1.0],
        'title_length': [0, 40, 80, float('inf')],
        'author_impact': [0, 5, 10, float('inf')],
        'engagement_score': [0, 2000, 5000, float('inf')],
        'title_complexity': [0, 0.5, 1.0, float('inf')]
    }

    for col, bins in bins_dict.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
    return df


def preprocess_data(df):
    # Processa i dati
    df = df.dropna().copy()
    df = add_new_features(df)
    df = calculate_is_viral(df)
    df = discretize_variables(df)
    feature_cols = [
        'upvotes', 'num_comments', 'upvote_ratio', 'title_length', 'author_impact',
        'engagement_score', 'title_complexity', 'is_viral'
    ]
    return df[feature_cols]


def save_graph_as_image(dag, output_path):
    # Salva il grafo orientato come immagine con layout circolare
    G = nx.DiGraph()
    G.add_nodes_from(sorted(dag.nodes()))
    G.add_edges_from(dag.edges())

    plt.figure(figsize=(12, 8))
    pos = nx.circular_layout(G)

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrows=True,
        arrowstyle='-|>',
        arrowsize=30,
        connectionstyle='arc3,rad=0.1'
    )

    plt.axis("off")
    plt.savefig(output_path, format="PNG", dpi=300)
    plt.close()
    print(f"Immagine grafo salvata in: {output_path}")


def learn_structure_with_hill_climbing(df, graph_output_path):
    # Apprendimento della struttura
    print("Apprendimento della struttura in corso...")
    hc = HillClimbSearch(df)
    dag = hc.estimate()

    if 'is_viral' not in dag.nodes():
        dag.add_node('is_viral')

    for col in ['upvotes', 'num_comments', 'upvote_ratio', 'title_length', 'author_impact']:
        if col in dag.nodes():
            try:
                dag.add_edge(col, 'is_viral')
                # Verifica se l'arco crea un ciclo
                if not nx.is_directed_acyclic_graph(dag):
                    dag.remove_edge(col, 'is_viral')
            except ValueError:
                pass

    save_graph_as_image(dag, graph_output_path)

    model = DiscreteBayesianNetwork(dag.edges())

    model.fit(df, estimator=MaximumLikelihoodEstimator)

    return model


def evaluate_model(model, X_test, y_test, output_path):
    # Valutazione del modello
    inference = VariableElimination(model)
    y_pred_probs = []

    for _, row in X_test.iterrows():
        evidence = {k: v for k, v in row.to_dict().items() if k in model.nodes()}
        try:
            prob = inference.query(variables=['is_viral'], evidence=evidence)
            if len(prob.values) < 2:
                y_pred_probs.append(0)
            else:
                y_pred_probs.append(prob.values[1])  # Calcoliamo la prob di is_viral = 1
        except Exception:
            y_pred_probs.append(0)

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if len(set(y_test)) > 1:
        bce = log_loss(y_test, y_pred_probs)
    else:
        bce = "Non calcolabile (una sola classe presente in y_test)"

    report = (
        f"=== Risultati del modello ===\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- Precision: {precision:.4f}\n"
        f"- Recall: {recall:.4f}\n"
        f"- F1-score: {f1:.4f}\n"
        f"- Binary Cross-Entropy: {bce}\n"
    )

    with open(output_path, 'a') as f:
        f.write("\n")
        f.write(report)

    print(report)


def main(
    input_path='reddit_data/cleaned_posts.parquet',
    output_path='reddit_data/results/hill_climbing_results.txt',
    graph_output_dir=r'reddit_data\plots_models'
):

    if not os.path.exists(input_path):
        print(f"Errore: Il file di input {input_path} non esiste.")
        return

    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_parquet(input_path)
    df = preprocess_data(df)

    X = df.drop(columns=['is_viral'])
    y = df['is_viral']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Apprendimento della struttura
    graph_output_path = os.path.join(graph_output_dir, "learned_structure.png")
    model = learn_structure_with_hill_climbing(pd.concat([X_train, y_train], axis=1), graph_output_path)

    # Stampa la struttura appresa
    print("Struttura appresa:")
    for edge in model.edges():
        print(f"{edge[0]} -> {edge[1]}")

    # Salva la struttura appresa in un file
    with open(output_path, 'w') as f:
        f.write("=== Struttura appresa ===\n")
        for edge in model.edges():
            f.write(f"{edge[0]} -> {edge[1]}\n")

    # Valutazione del modello
    evaluate_model(model, X_test, y_test, output_path)


if __name__ == '__main__':
    main()