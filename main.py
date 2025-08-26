import sys
sys.dont_write_bytecode = True
from menu import mostra_menu, verifica_dati
from unsupervised_learning import run_unsupervised
from supervised_learning import run_supervised
from bayesian_network import run_bayesian_network
from bayesian_network_final import main as run_hill_climbing


# Funzione main
def main():
   while True:
    mostra_menu()
    scelta = input("Scelta: ").strip()

    match scelta:
        case "1":
            if verifica_dati():
                run_unsupervised()
        case "2":
            if verifica_dati():
                run_supervised()
        case "3":
            if verifica_dati():
                run_bayesian_network(model_type='full')
                run_bayesian_network(model_type='simple')
        case "4":
            if verifica_dati():
                run_hill_climbing()
        case "5":
            print("Uscita dal programma.")
            break
        case _:
            print("Scelta non valida. Riprovare.")

if __name__ == "__main__":
    import sys
    sys.dont_write_bytecode = True
    main()