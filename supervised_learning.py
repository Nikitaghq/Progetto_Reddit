import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import random

RANDOM_STATE = 42 # Random Seed per riproddurre gli stessi risultati ad ogni avvio
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def run_supervised(
    input_path='reddit_data/cleaned_posts.parquet',
    output_report='reddit_data/results/supervised_result.txt'
):
    print("\n=== INIZIO APPRENDIMENTO SUPERVISIONATO ===")

    if not os.path.exists(input_path):
        print(f"- File non trovato: {input_path}")
        return
    df = pd.read_parquet(input_path)
    df = df[df['author_impact'].notna()]
    print(f"- Dati caricati: {df.shape[0]} righe, {df.shape[1]} colonne")

    # Creazione di is_viral
    if 'is_viral' not in df:
        viral_criteria = [
            df['upvote_ratio'] >= 0.92,
            df['author_impact'] >= 9.8,
            df['upvotes'] >= 2000,
            df['num_comments'] >= 180,
            df['title_length'] >= 40
        ]
        df['is_viral'] = (np.sum(viral_criteria, axis=0) >= 3).astype(int)

    # Creazione delle colonne mancanti
    if 'post_hour_sin' not in df or 'post_hour_cos' not in df:
        if 'post_hour' in df:
            post_hour = df['post_hour'].fillna(0).astype(int)
            df['post_hour_sin'] = np.sin(2 * np.pi * post_hour / 24)
            df['post_hour_cos'] = np.cos(2 * np.pi * post_hour / 24)
        else:
            raise KeyError("Colonna 'post_hour' non trovata. Non è possibile calcolare 'post_hour_sin' e 'post_hour_cos'.")

    if 'is_weekend' not in df:
        if 'post_weekday' in df:
            weekday = df['post_weekday'].fillna(0).astype(int)
            df['is_weekend'] = (weekday >= 5).astype(int)
        else:
            raise KeyError("Colonna 'post_weekday' non trovata. Non è possibile calcolare 'is_weekend'.")

    if 'has_question_mark' not in df:
        if 'title' in df:
            df['has_question_mark'] = df['title'].astype(str).str.contains('\?').astype(int)
        else:
            raise KeyError("Colonna 'title' non trovata. Non è possibile calcolare 'has_question_mark'.")

    # Gestione Feature
    feature_cols = [
        'upvotes', 'num_comments', 'upvote_ratio',
        'title_length', 'author_impact',
        'post_hour_sin', 'post_hour_cos', 'is_weekend', 'has_question_mark',
        'content_image', 'content_video', 'content_text', 'content_other'
    ]
    X = df[feature_cols]
    y = df['is_viral']

    X_selected = X
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"- Dimensione training set: {X_train.shape[0]} campioni")
    print(f"- Dimensione test set: {X_test.shape[0]} campioni")

    # Configurazione dei modelli e dei parametri
    models_config = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 150],
                'max_depth': [4, 6, 8],
                'min_samples_split': [10, 20, 30],
                'max_features': ['sqrt', 0.5]
            }
        },
        'SVC': {
            'model': SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1, 1]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.5],
                'estimator': [  
                    DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                    DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
                ]
            }
        }
    }

    report_lines = []
    roc_data = []

    # Outer loop: validazione esterna
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Per ogni modello e modalità (con e senza SMOTE)
    for model_name, config in models_config.items():
        for smote_mode in ['SMOTE', 'NoSMOTE']:
            print(f"\n--- Nested GridSearch per {model_name} ({smote_mode}) ---")

            # Controllo SMOTE
            if smote_mode == 'SMOTE':
                class_counts = y_train.value_counts()
                if class_counts.min() < 2:
                    print(f"- [ATTENZIONE] SMOTE non applicabile: classe minoritaria troppo piccola.")
                    continue
                smote = SMOTE(random_state=RANDOM_STATE)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Inner loop: GridSearchCV
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                scoring='accuracy',
                cv=5,  # Inner loop: 5-fold cross-validation
                n_jobs=-1,
                verbose=1
            )

            # Nested cross-validation
            nested_scores = cross_val_score(
                grid_search,
                X_train_resampled,
                y_train_resampled,
                scoring='accuracy',
                cv=outer_cv,  # Outer loop
                n_jobs=-1
            )

            # Calcolo media e deviazione standard delle prestazioni
            mean_nested_score = nested_scores.mean()
            std_nested_score = nested_scores.std()

            print(f"- Nested CV Accuracy: {mean_nested_score:.4f} (+/- {std_nested_score:.4f})")

            # Esegui la GridSearch sui dati di training completi
            grid_search.fit(X_train_resampled, y_train_resampled)

            # Miglior modello e parametri
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Aggiungi al report
            report_lines.append(f"=== {model_name} ({smote_mode}) ===\n")
            report_lines.append(f"Migliori parametri: {best_params}\n")
            report_lines.append(f"Nested CV Accuracy: {mean_nested_score:.4f} (+/- {std_nested_score:.4f})\n")
            report_lines.append("\n")

            # Valutazione sul test set
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            cm = confusion_matrix(y_test, y_pred)

            # Aggiungi l'accuratezza al report
            report_lines.append(f"Accuracy (Test Set): {accuracy:.4f}\n")
            report_lines.append(f"AUC-ROC (Test Set): {auc:.4f}\n")

            # Aggiungi il classification report al report
            classification_rep = classification_report(y_test, y_pred, digits=4, zero_division=0)
            report_lines.append("Classification Report (Test Set):\n")
            report_lines.append(classification_rep + "\n")

            # Learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_errors = []
            test_errors = []
            for frac in train_sizes:
                n_samples = int(frac * len(X_train))
                X_frac = X_train[:n_samples]
                y_frac = y_train[:n_samples]
                best_model.fit(X_frac, y_frac)
                train_errors.append(1 - accuracy_score(y_frac, best_model.predict(X_frac)))
                test_errors.append(1 - accuracy_score(y_test, best_model.predict(X_test)))

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes * len(X_train), train_errors, 'o-', color="r", label="Errore Training")
            plt.plot(train_sizes * len(X_train), test_errors, 'o-', color="g", label="Errore Test")
            plt.title(f"Learning Curve - {model_name} ({smote_mode})")
            plt.xlabel("Numero di campioni di training")
            plt.ylabel("Errore (1 - Accuratezza)")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'reddit_data/plots_models/learning_curve_{model_name}_{smote_mode}.png')
            plt.close()

            # Matrice di confusione
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Virale', 'Virale'], yticklabels=['Non Virale', 'Virale'])
            plt.xlabel('Predetto')
            plt.ylabel('Reale')
            plt.title(f'Matrice di Confusione - {model_name} ({smote_mode})')
            plt.tight_layout()
            plt.savefig(f'reddit_data/plots_models/confusion_matrix_{model_name}_{smote_mode}.png')
            plt.close()

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
            roc_data.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc,
                'label': f'{model_name} ({smote_mode})'
            })

    # Salva report completo
    with open(output_report, 'w') as f:
        f.writelines(report_lines)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    for data in roc_data:
        plt.plot(data['fpr'], data['tpr'], label=f"{data['label']} (AUC = {data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Confronto Modelli')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reddit_data/plots_models/roc_curve.png')
    plt.close()

    print("\n=== OPERAZIONE COMPLETATA ===")
    print(f"- Report completo salvato in '{output_report}'")
    print(f"- Grafici salvati in 'reddit_data/plots_models/'")

if __name__ == '__main__':
    run_supervised()