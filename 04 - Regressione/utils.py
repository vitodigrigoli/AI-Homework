import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(file):
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return torch.tensor(x, dtype=torch.float32),\
            torch.tensor(y, dtype=torch.float32)


# Pre-inizializzazione della figura e degli assi
plot_count = 0
rows, cols = 2, 3  # Definisci il numero di righe e colonne dei subplot
fig, axs = plt.subplots(rows, cols, figsize=(16, 9))
def plot(X, y, model, title="", pause=False):
    global plot_count, fig, axs
    
    # Determina l'asse corrente basandosi sul conteggio dei plot
    ax = axs[plot_count // cols, plot_count % cols]
    plt.sca(ax)  # Seleziona il subplot corretto
    
    ax.clear()  # Pulisce solo il subplot attuale
    
    xmin, xmax = min(X), max(X)
    x = np.linspace(xmin, xmax, 100)
    ax.scatter(X, y)  # Usa 'ax' per chiamare funzioni specifiche del subplot
    ax.plot(x, model.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy(), color="red")
    ax.set_title(title)

    plot_count += 1  # Incrementa il conteggio dei plot dopo ogni chiamata
    
    if pause or plot_count == rows * cols:  # Mostra e blocca l'output se necessario
        plt.show()
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))  # Resetta fig e axs se necessario
        plot_count = 0  # Resetta il contatore dei plot
    else:
        plt.pause(1)  # Mostra brevemente il plot e continua l'esecuzione
