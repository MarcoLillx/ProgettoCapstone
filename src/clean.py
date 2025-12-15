import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cartelle da svuotare (ma non cancellare la cartella stessa)
DIRS_TO_CLEAN = [
    os.path.join(BASE_DIR, 'models'),
    os.path.join(BASE_DIR, 'adv_examples'),
    os.path.join(BASE_DIR, 'logs'),
    os.path.join(BASE_DIR, 'figure'),
    # Nota: Non cancelliamo 'dataset' per non perdere i file raw o processed che richiedono tempo
]

def clean_project():
    print("--- PULIZIA AMBIENTE ---")
    for directory in DIRS_TO_CLEAN:
        if os.path.exists(directory):
            print(f"Pulizia: {directory}...")
            # Rimuove tutto il contenuto
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        if filename != ".gitkeep": # Preserva il gitkeep
                            os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Errore eliminazione {file_path}: {e}")
        else:
            os.makedirs(directory)
            print(f"Creata cartella mancante: {directory}")
            
    print("--- AMBIENTE PULITO E PRONTO ---")

if __name__ == "__main__":
    confirm = input("Sei sicuro di voler cancellare tutti i modelli, log e figure generati? (s/n): ")
    if confirm.lower() == 's':
        clean_project()
    else:
        print("Operazione annullata.")