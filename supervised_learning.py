import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib
matplotlib.use('Agg')
import random

RANDOM_STATE = 42
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

    # Gestione Feature
    if 'post_hour_sin' not in df or 'post_hour_cos' not in df:
        if 'post_hour' in df:
            post_hour = df['post_hour'].fillna(0).astype(int)
            df['post_hour_sin'] = np.sin(2 * np.pi * post_hour / 24)
            df['post_hour_cos'] = np.cos(2 * np.pi * post_hour / 24)

    if 'is_weekend' not in df:
        if 'post_weekday' in df:
            weekday = df['post_weekday'].fillna(0).astype(int)
            df['is_weekend'] = (weekday >= 5).astype(int)

    if 'has_question_mark' not in df:
        df['has_question_mark'] = df['title'].astype(str).str.contains('\?').astype(int)

    feature_cols = [
        'upvotes', 'num_comments', 'upvote_ratio',
        'title_length', 'author_impact',
        'post_hour_sin', 'post_hour_cos', 'is_weekend', 'has_question_mark',
        'content_image', 'content_video', 'content_text', 'content_other'
    ]
    X = df[feature_cols]
    y = df['is_viral']

    selector = SelectKBest(f_classif, k=8)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"- Feature selezionate ({len(selected_features)}): {list(selected_features)}")

    rng = np.random.RandomState(RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"- Dimensione training set: {X_train.shape[0]} campioni")
    print(f"- Dimensione test set: {X_test.shape[0]} campioni")

    models = {
        'RandomForest': {
            'estimator': RandomForestClassifier(
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_samples=0.8
            ),
            'param_grid': {
                'n_estimators': [100, 150],
                'max_depth': [4, 6, 8],
                'min_samples_split': [10, 20, 30],
                'max_features': ['sqrt', 0.5]
            }
        },
        'SVC': {
            'estimator': __import__('sklearn.svm').svm.SVC(
                probability=True, kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE
            ),
            'param_grid': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1, 1]
            }
        },
        'AdaBoost': {
            'estimator': AdaBoostClassifier(
                random_state=RANDOM_STATE
            ),
            'param_grid': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.5],
                'estimator': [
                    DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                    DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
                ]
            }
        }
    }

    best_models = {}
    results = []
    report_lines = []
    roc_data = []

    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, conf in models.items():
        for smote_mode in ['SMOTE', 'NoSMOTE']:
            print(f"\n- GridSearchCV per {name} ({smote_mode})...")

            class_counts = y_train.value_counts()
            if len(class_counts) < 2:
                print(f"- [ATTENZIONE] {name} ({smote_mode}) saltato: il training set contiene una sola classe ({class_counts.index[0]}: {class_counts.iloc[0]} campioni). Modello non addestrato.")
                report_lines.append(f"=== {name} ({smote_mode}) ===\n")
                report_lines.append(f"Modello saltato: il training set contiene una sola classe. Distribuzione: {class_counts.to_dict()}\n\n")
                continue
            if smote_mode == 'SMOTE':
                if class_counts.min() < 2:
                    print(f"- [ATTENZIONE] SMOTE saltato per {name} ({smote_mode}): la classe minoritaria ha meno di 2 campioni. Modello non addestrato.")
                    report_lines.append(f"=== {name} ({smote_mode}) ===\n")
                    report_lines.append(f"SMOTE non applicabile: classe minoritaria troppo piccola. Distribuzione: {class_counts.to_dict()} Modello saltato.\n\n")
                    continue
                k_neighbors = min(5, class_counts.min() - 1)
                X_train_fit, y_train_fit = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors).fit_resample(X_train, y_train)
            else:
                X_train_fit, y_train_fit = X_train, y_train

            try:
                grid = GridSearchCV(
                    conf['estimator'],
                    conf['param_grid'],
                    scoring='roc_auc',
                    cv=cv_splitter,
                    n_jobs=-1,
                    verbose=1
                )
                grid.fit(X_train_fit, y_train_fit)
                best_models[f'{name}_{smote_mode}'] = grid.best_estimator_
                print(f"- Migliori parametri {name} ({smote_mode}): {grid.best_params_}")
                print(f"- Miglior score (AUC-ROC, crossval): {grid.best_score_:.4f}")
            except Exception as e:
                print(f"- ERRORE durante GridSearchCV per {name} ({smote_mode}): {e}")
                report_lines.append(f"=== {name} ({smote_mode}) ===\n")
                report_lines.append(f"Errore durante GridSearchCV: {e}\n\n")
                continue

            train_sizes = np.linspace(0.1, 1.0, 10)
            n_train_samples = len(X_train)
            train_errors = []
            test_errors = []
            for frac in train_sizes:
                n_samples = int(frac * n_train_samples)
                indices = rng.choice(n_train_samples, n_samples, replace=False)
                X_frac = X_train[indices]
                y_frac = y_train.iloc[indices]
                unique, counts = np.unique(y_frac, return_counts=True)
                if len(unique) < 2 or np.min(counts) < 2:
                    train_errors.append(np.nan)
                    test_errors.append(np.nan)
                    continue
                if smote_mode == 'SMOTE':
                    X_frac_res, y_frac_res = SMOTE(random_state=RANDOM_STATE, k_neighbors=1).fit_resample(X_frac, y_frac)
                else:
                    X_frac_res, y_frac_res = X_frac, y_frac
                clf = grid.best_estimator_
                clf.fit(X_frac_res, y_frac_res)
                train_acc = accuracy_score(y_frac_res, clf.predict(X_frac_res))
                train_errors.append(1 - train_acc)
                test_acc = accuracy_score(y_test, clf.predict(X_test))
                test_errors.append(1 - test_acc)

            # Valutazione su test e train
            model = grid.best_estimator_
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train_fit)
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train = accuracy_score(y_train_fit, y_pred_train)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            y_proba_train = model.predict_proba(X_train_fit)[:, 1]
            auc_test = roc_auc_score(y_test, y_proba_test)
            auc_train = roc_auc_score(y_train_fit, y_proba_train)

            # Plot learning curve errori
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes * n_train_samples, train_errors, 'o-', color="r", label="Errore Training")
            plt.plot(train_sizes * n_train_samples, test_errors, 'o-', color="g", label="Errore Test")
            plt.title(f"Learning Curve Errori - {name} ({smote_mode})")
            plt.xlabel("Numero di campioni di training")
            plt.ylabel("Errore (1 - Accuratezza)")
            plt.legend(loc="best")
            plt.grid(True)
            plt.tight_layout()
            err_path = f'reddit_data/plots_models/learning_curve_{name}_{smote_mode}.png'
            plt.savefig(err_path, dpi=300)
            plt.close()
            print(f"- Learning curve errori salvata per {name} ({smote_mode}) in {err_path}")

            # Plot e matrice di confusione
            cm = confusion_matrix(y_test, y_pred_test)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Virale', 'Virale'], yticklabels=['Non Virale', 'Virale'])
            plt.xlabel('Predetto')
            plt.ylabel('Reale')
            plt.title(f'Matrice di Confusione (Test Set)\n{name} ({smote_mode})')
            cm_path = f'reddit_data/plots_models/confusion_matrix_{name}_{smote_mode}.png'
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"- Matrice di confusione salvata per {name} ({smote_mode}) in {cm_path}")

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba_test)
            roc_data.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc_test,
                'label': f'{name} ({smote_mode})'
            })

            # Report
            report_lines.append(f"=== {name} ({smote_mode}) ===\n")
            report_lines.append(f"Migliori parametri: {grid.best_params_}\n")
            report_lines.append(f"AUC-ROC TRAIN: {auc_train:.4f}\n")
            report_lines.append(f"AUC-ROC TEST: {auc_test:.4f}\n")
            report_lines.append(f"Accuracy TRAIN: {acc_train:.4f}\n")
            report_lines.append(f"Accuracy TEST: {acc_test:.4f}\n")
            report_lines.append("Classification Report TEST:\n")
            report_lines.append(classification_report(y_test, y_pred_test, digits=4, zero_division=0))
            report_lines.append("\nLearning curve errori (1-accuracy):\n")
            report_lines.append("Frazione_trainset\tErrore_train\tErrore_test\n")
            for frac, err_tr, err_te in zip(train_sizes, train_errors, test_errors):
                report_lines.append(f"{frac:.2f}\t{err_tr:.4f}\t{err_te:.4f}\n")
            report_lines.append("\n\n")

            results.append({
                'modello': f'{name} ({smote_mode})',
                'AUC-ROC': auc_test,
                'Accuracy': acc_test,
                'ClassificationReport': classification_report(y_test, y_pred_test, digits=4, zero_division=0)
            })

    # Salva report completo
    with open(output_report, 'w') as f:
        f.writelines(report_lines)

    # Plot ROC curve combinata
    plt.figure(figsize=(10, 8))
    for data in roc_data:
        plt.plot(data['fpr'], data['tpr'], label=f"{data['label']} (AUC = {data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Confronto Modelli')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = 'reddit_data/plots_models/roc_curve.png'
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"- ROC Curve salvata in {roc_path}")
    print("\n=== OPERAZIONE COMPLETATA ===")
    print(f"- Report completo salvato in '{output_report}'")
    print(f"- Grafici salvati in 'reddit_data/'")

if __name__ == '__main__':
    run_supervised()