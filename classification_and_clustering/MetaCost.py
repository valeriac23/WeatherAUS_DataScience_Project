import pandas as pd
import numpy as np
from sklearn.base import clone


class MetaCost(object):

    """
    Implementazione del meta-algoritmo MetaCost per rendere un classificatore
    sensibile ai costi d'errore tramite relabeling basato su bootstrap ensemble.
    
    """
    def __init__(self, S, L, C, m=50, n=1, p=True, q=True):
        """
        :param S: Training set
        :param L: Algoritmo/modello di classificazione
        :param C: Matrice dei costi 
        :param q: True per usare tutti i modelli per predire
        :param m: Numero di campioni bootstrap da generare (default 50)
        :param n: Numero di esempi in ogni campione bootstrap
        :param p: True se il modello L genera probabilità (predict_proba)
        """
        if not isinstance(S, pd.DataFrame):
            raise ValueError('S must be a DataFrame object')
        new_index = list(range(len(S)))
        S.index = new_index
        self.S = S
        self.L = L
        self.C = C
        self.m = m
        self.n = len(S) * n
        self.p = p
        self.q = q

    def fit(self, flag, num_class):
        """
        Calcola le nuove etichette ad alto costo e addestra il modello finale.
        
        :param flag: Nome della colonna target in S
        :param num_class: Numero totale di classi distinte
        """
        # Separazione feature (X) e target (y)
        col = [c for c in self.S.columns if c != flag]
        X_full = self.S[col].values
        y_full = self.S[flag].values
        N = len(self.S)
        
        # Accumulatori per le probabilità e i conteggi dei modelli
        P_sum = np.zeros((N, num_class))
        counts = np.zeros((N, 1))

        # 1. GENERAZIONE DELL'ENSEMBLE E STIMA DELLE PROBABILITÀ
        for i in range(self.m):
            # Estrazione con reinserimento (Bootstrap)
            sample_idx = np.random.choice(N, size=self.n, replace=True)
            X_boot = X_full[sample_idx]
            y_boot = y_full[sample_idx]

            # Clona e addestra il modello base sul campione attuale
            model = clone(self.L)
            model.fit(X_boot, y_boot)

            # Estrazione delle probabilità di classe sul dataset intero
            if self.p:
                probas = model.predict_proba(X_full)
            else:
                # Se il modello non ha predict_proba, trasforma la predizione hard in One-Hot
                preds = model.predict(X_full)
                probas = np.eye(num_class)[preds]

            # Accumulo delle probabilità
            if self.q:
                # Usa tutte le predizioni
                P_sum += probas
                counts += 1
            else:
                # Usa solo le predizioni Out-Of-Bag (OOB) per evitare overfitting
                is_in_bag = np.zeros(N, dtype=bool)
                is_in_bag[sample_idx] = True
                is_oob = ~is_in_bag
                
                P_sum[is_oob] += probas[is_oob]
                counts[is_oob] += 1

        # Evita divisioni per zero per eventuali punti mai rimasti OOB
        counts[counts == 0] = 1 
        
        # Calcolo della probabilità media di ogni classe per ogni record P(j|x)
        P_avg = P_sum / counts

        # Moltiplica le probabilità per la matrice dei costi
        expected_risks = P_avg.dot(self.C.T)
        
        # RELABELING: Assegna a ciascun record la classe che minimizza il costo atteso d'errore
        new_labels = np.argmin(expected_risks, axis=1)

    
        # Fit del modello finale sul dataset originale ma con le nuove etichette
        model_new = clone(self.L)
        model_new.fit(X_full, new_labels)

        return model_new