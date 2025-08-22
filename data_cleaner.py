import pandas as pd
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

def clean_reddit_data(input_path='reddit_data/reddit_viral_posts.json',
                     output_path='reddit_data/cleaned_posts.parquet'):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"- Errore durante la lettura del file {input_path}")
        print(f"Dettaglio: {str(e)}")
        return None

    # Standardizzazione
    required_columns = ['title', 'score', 'num_comments', 'upvote_ratio', 'created_utc']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"- Colonne mancanti: {missing_cols}")
        return None

    os.makedirs('reddit_data', exist_ok=True)

    def clean_text(text):
        if not isinstance(text, str):
            return ''
        return re.sub(r'[^\w\s]', '', text).strip()

    cleaned_data = []
    for _, row in df.iterrows():
        try:
            title = clean_text(row.get('title', ''))
            text = clean_text(row.get('text', ''))
            
            # Gestione datetime (manteniamo solo hour, weekday come numerici)
            created_utc = row.get('created_utc')
            try:
                dt = datetime.fromisoformat(created_utc) if pd.notna(created_utc) else None
                hour = dt.hour if dt else None
                weekday = dt.weekday() if dt else None
            except:
                hour = weekday = None

            # Calcolo metriche autore (solo author_impact mantenuto)
            link_karma = max(float(row.get('author_link_karma', 0)), 0)
            comment_karma = max(float(row.get('author_comment_karma', 0)), 0)
            author_impact = np.log1p(link_karma) * 0.6 + np.log1p(comment_karma) * 0.4

            cleaned_post = {
                'id': str(row.get('id', '')),
                'subreddit': str(row.get('subreddit', 'unknown')),
                'title': title,
                'text': text,
                'upvotes': int(row.get('score', 0)),
                'num_comments': int(row.get('num_comments', 0)),
                'upvote_ratio': float(row.get('upvote_ratio', 0)),
                'title_length': len(title),
                'post_hour': hour,
                'post_weekday': weekday,
                'content_type': str(row.get('content_type', 'unknown')),
                'nsfw': int(bool(row.get('nsfw', False))),
                'author_impact': author_impact
            }
            cleaned_data.append(cleaned_post)
        except Exception as e:
            print(f"- Errore nel post {row.get('id', 'unknown')}: {str(e)}")
            continue

    df_clean = pd.DataFrame(cleaned_data)

    # Codifica solo i content_type principali (altri vengono raggruppati)
    main_content_types = ['image', 'video', 'text']
    df_clean['content_type'] = df_clean['content_type'].apply(
        lambda x: x if x in main_content_types else 'other'
    )
    content_dummies = pd.get_dummies(df_clean['content_type'], prefix='content')
    df_clean = pd.concat([df_clean, content_dummies], axis=1)

    # Visualizzazioni modificate per le nuove feature
    plot_distributions(df_clean)
    
    df_clean.to_parquet(output_path, index=False)
    print(f"- Dati puliti salvati in {output_path} ({len(df_clean)} righe)")
    return df_clean

def plot_distributions(df):
    # Stampa Grafici
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['upvotes'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribuzione Upvotes')
    plt.xlabel('Upvotes')
    plt.ylabel('Frequenza')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(df['num_comments'], bins=50, color='salmon', edgecolor='black')
    plt.title('Distribuzione Commenti')
    plt.xlabel('Commenti')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    df['content_type'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Tipi di Contenuto')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    plt.scatter(df['upvotes'], df['num_comments'], alpha=0.5, color='purple')
    plt.title('Upvotes vs Commenti')
    plt.xlabel('Upvotes')
    plt.ylabel('Commenti')
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('reddit_data/data_distributions.png', dpi=300)
    plt.close()
    print("- Grafici salvati in 'reddit_data/data_distributions.png'")

if __name__ == "__main__":
    clean_reddit_data()