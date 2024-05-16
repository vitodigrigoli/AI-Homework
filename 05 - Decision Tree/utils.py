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
rows, cols = 1, 1  # Definisci il numero di righe e colonne dei subplot
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



def classification_stats(y, yhat):
    accuracy = (yhat == y).sum() / len(y)
    precision = ((yhat == 1) * (y == 1)).sum() / (yhat == 1).sum()
    recall = ((yhat == 1) * (y == 1)).sum() / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall)
    print( f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}")
    
    
    
def plot3(X,y, model, title="", pause=False):
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.025)
    x2grid = np.arange(min2, max2, 0.025)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    # horizontal stack vectors to create x1,x2 input for the model
    grid = [[x1, x2] for x1, x2 in zip(r1,r2)]
    # make predictions for the grid
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='black')
    plt.title(title)
    # show the plot
    plt.pause(2)
    if pause:
        plt.show()
    plt.clf()
