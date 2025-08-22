import os
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


def calculate_is_viral(df):
    # Definizione di is_viral (come nella fasi precedenti)
    viral_criteria = [
        df['upvote_ratio'] >= 0.92,
        df['author_impact'] >= 9.8,
        df['upvotes'] >= 2000,
        df['num_comments'] >= 180,
        df['title_length'] >= 40
    ]
    df['is_viral'] = (np.sum(viral_criteria, axis=0) >= 3).astype(int)
    return df


def discretize_variables(df):
    # Discretizzazione delle variabili
    bins_dict = {
        'upvotes': [0, 2000, 4000, np.inf],  
        'num_comments': [0, 180, 500, np.inf],  
        'upvote_ratio': [0, 0.5, 0.75, 0.92, 1.0],  
        'title_length': [0, 40, 80, np.inf],  
        'author_impact': [0, 5, 9.8, np.inf]  
    }

    for col, bins in bins_dict.items():
        try:
            df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
        except ValueError:
            print(f"Impossibile discretizzare la variabile '{col}'. Controlla i dati.")
    return df


def preprocess_data(df, model_type='full'):
    # Processa i dati
    if 'is_viral' not in df:
        df = calculate_is_viral(df)

    # Rimozione dei valori mancanti e riempimento
    df = df.dropna(subset=['upvotes', 'num_comments', 'upvote_ratio', 'title_length', 'author_impact'])
    df = df.fillna(0)

    # Discretizzazione delle variabili
    df = discretize_variables(df)

    # Manteniamo solo le feature utili in base al modello
    feature_cols = [
        'upvotes', 'num_comments', 'upvote_ratio', 'title_length', 'author_impact', 'is_viral'
    ]
    return df[feature_cols]


def define_bayesian_model_full():
    # Definizione del modello full
    model = DiscreteBayesianNetwork([
        ('upvotes', 'is_viral'),
        ('num_comments', 'is_viral'),
        ('upvote_ratio', 'is_viral'),
        ('title_length', 'is_viral'),
        ('author_impact', 'is_viral'),

        ('upvotes', 'num_comments'),
        ('upvotes', 'upvote_ratio'),
        ('author_impact', 'upvotes'),
        ('author_impact', 'num_comments'),
        ('num_comments', 'upvote_ratio'),
        ('title_length', 'num_comments')
    ])
    return model


def define_bayesian_model_simple():
    # Definizione del modello simple
    model = DiscreteBayesianNetwork([
        ('upvotes', 'is_viral'),
        ('num_comments', 'is_viral'),
        ('upvote_ratio', 'is_viral'),
        ('title_length', 'is_viral'),
        ('author_impact', 'is_viral')
    ])
    return model


def save_results_to_file(report, model_type, output_path='reddit_data/results/bayesian_results.txt'):
    # Salvataggio Report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write(f"\n=== Risultati per il modello {model_type.capitalize()} ===\n")
        f.write(report)
        f.write("\n")


def evaluate_model(model, X_test, y_test, model_type):
    # Valutazione dei due modelli
    inference = VariableElimination(model)
    y_pred_probs = []
    for _, row in X_test.iterrows():
        evidence = {k: v for k, v in row.to_dict().items() if k in model.nodes()}
        prob = inference.query(variables=['is_viral'], evidence=evidence, show_progress=False)
        y_pred_probs.append(prob.values[1])

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_probs]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bce = log_loss(y_test, y_pred_probs)

    report = (
        f"- Accuracy: {accuracy:.4f}\n"
        f"- Precision: {precision:.4f}\n"
        f"- Recall: {recall:.4f}\n"
        f"- F1-score: {f1:.4f}\n"
        f"- Binary Cross-Entropy: {bce:.4f}\n"
    )

    print(f"\nMetriche per il modello {model_type.capitalize()}:\n{report}")

    save_results_to_file(report, model_type)


def run_bayesian_network(input_path='reddit_data/cleaned_posts.parquet', model_type='full'):
    # Avvio dell'addestramento
    if not os.path.exists(input_path):
        print(f"Errore: File non trovato: {input_path}")
        return
    df = pd.read_parquet(input_path)

    df = preprocess_data(df, model_type=model_type)

    X = df.drop(columns=['is_viral'])
    y = df['is_viral']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_type == 'full':
        model = define_bayesian_model_full()
    elif model_type == 'simple':
        model = define_bayesian_model_simple()
    else:
        print("Errore: Tipo di modello non valido. Utilizzare 'full' o 'simple'.")
        return

    model.fit(pd.concat([X_train, y_train], axis=1), estimator=MaximumLikelihoodEstimator)
    evaluate_model(model, X_test, y_test, model_type)


if __name__ == '__main__':
    results_path = 'reddit_data/results/bayesian_results.txt'
    if os.path.exists(results_path):
        os.remove(results_path)

    run_bayesian_network(model_type='full')
    run_bayesian_network(model_type='simple')
