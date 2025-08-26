# Menu del programma
def mostra_menu():
    print("\n" + "="*50)
    print("MENU PRINCIPALE - PROGETTO REDDIT")
    print("="*50)
    print("1) Esegui Apprendimento Non Supervisionato")
    print("2) Esegui Apprendimento Supervisionato")
    print("3) Esegui Bayesian Network (full & simple)")
    print("4) Esegui Bayesian Network con Hill Climbing")
    print("5) Esci")
    print("="*50)

# Verifica dei dati necessari per il funzionamento
def verifica_dati():
    """Controlla la presenza dei file necessari"""
    from pathlib import Path
    if not Path("reddit_data/cleaned_posts.parquet").exists():
        print("\n- File cleaned_posts.parquet non trovato!")
        print("- Esegui prima data_cleaner.py dalla console")
        return False
    return True