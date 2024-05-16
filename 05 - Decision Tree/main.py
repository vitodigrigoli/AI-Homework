from mynet import *
from utils import *
from decision_tree import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

FILE = 'dataset.dat'


def main_old():
    
    x, y = load_data(FILE)
    
    x_train = x.float()
    y_train = y.long()
    print(type(x), type(x_train))
    
    net = Net()
    
    net.train(x_train, y_train)
    plot3(x_train,y_train, net, title="Neural Network", pause=True)
    # classification_stats(y_train.numpy(), net.predict(x_train))


def print_tree(node, depth=0, label="Root"):
    """ Funzione ricorsiva per stampare l'albero decisionale. """
    if node is not None:
        if node.value is not None:  # È un nodo foglia
            print(f"{'|   ' * depth}{label} - Leaf: Class={node.value}")
        else:  # Nodo decisionale
            feature_info = f"Feature {node.feature_index}" if node.feature_index is not None else "No feature"
            threshold_info = f"< {node.threshold:.2f}" if node.threshold is not None else "No threshold"
            print(f"{'|   ' * depth}{label} - [{feature_info} {threshold_info}]")
            print_tree(node.left, depth + 1, "Left")
            print_tree(node.right, depth + 1, "Right")



def main():
    # Carica il dataset Iris
    data = load_iris()
    X = data.data
    y = data.target

    # Divide i dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X.shape)

    # Crea un'istanza dell'albero decisionale
    tree = DecisionTree(max_depth=3)

    # Addestra l'albero con il set di training
    tree.fit(X_train, y_train)

    # Usa l'albero per fare previsioni sul set di test
    predictions = tree.predict(X_test)

    # Calcola l'accuratezza delle previsioni
    accuracy = np.mean(predictions == y_test)
    print(f'Accuracy: {accuracy:.2f}')

    # Stampa le previsioni e le vere classi
    print("Predictions:", predictions)
    print("True labels:", y_test)
    
    # Stampa l'albero
    print("\nDecision Tree:")
    print_tree(tree.root)

main()