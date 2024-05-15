from net import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

def main():
    
    # carica il dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # dividi in dati in train e test
    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # converti i dati in tensori
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    
    # crea un DataLoader per il training set
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # inizializzazione del modello
    input_size = X_train.shape[1]
    hidden_size = 6
    output_size = len(np.unique(y_train))

	# addrestramento del modello
    model = AEClassifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    losses = model.train_classifier(train_loader, epoches=300)
    
    # predizione sui dati di test
    test_predictions = model.classify(X_test)
    accuracy = (test_predictions == y_test).float().mean()
    print(f'Accuracy: {accuracy:.2f}')
    
    # calcola e stampa la matrice di confusione
    conf_matrix = confusion_matrix(y_test, test_predictions)
    print(f'Matrice di confusione:\n{conf_matrix}')
    
    
    
if __name__ == '__main__':
    main()