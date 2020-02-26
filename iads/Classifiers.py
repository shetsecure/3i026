# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        N = len(desc_set)
        predictions = [self.predict(x) for x in desc_set]
        
        acc = [1 if predictions[i] * label_set[i] > 0 else 0 for i in range(N)]
        
        return float(sum(acc) / N)
    
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.weights = np.random.normal(0, 0.5, input_dimension)
        #self.weights = np.random.randn(input_dimension)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.weights)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
    
    
# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, max_iter=1e2):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.weights = np.random.normal(0, 0.5, input_dimension)
        #self.weights = np.zeros(input_dimension)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ordre = np.arange(len(label_set))
        np.random.shuffle(ordre)
        wrong = True
        counter = 0
        
        while wrong and counter < self.max_iter:
            wrong = False
            
            for i in ordre:
                if self.predict(desc_set[i]) * label_set[i] < 0:
                    wrong = True
                    self.weights += self.learning_rate * desc_set[i] * label_set[i]
                    
            counter += 1
            
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.weights)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) > 0 else -1
    
class ClassifierOneStepPerceptron(ClassifierLineaireRandom):
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.weights = np.random.normal(0, 0.5, input_dimension)
        self.trained = False
    
    # override training method, to calculate pseudo-inverse matrix as the solution to the MSE cost function
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        if not self.trained:
            X = desc_set
            y = label_set
            w = np.linalg.lstsq(np.matmul(X.T, X), np.matmul(X.T, y), rcond=None)
            w = np.reshape(w[0], (1, w[0].shape[0]))[0]
            self.weights = w
            
        self.trained = True

class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension,learning_rate, kernel, max_iter=1e3):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.weights = self.kernel.transform(np.random.normal(0, 0.5, input_dimension))
        self.max_iter = max_iter
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.weights, self.kernel.transform(x))    
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """       
        wrong = True
        counter = 0
        
        while wrong and counter < self.max_iter:
            wrong = False
            
            for i in range(len(desc_set)):
                if (self.predict(desc_set[i]) * label_set[i]) < 0:
                    wrong = True
                    self.weights += self.learning_rate * label_set[i] * self.kernel.transform(desc_set[i])
                    
            counter += 1
                

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
    
    def dist_euc(self, x, y):
        dist = sum([(yi - xi) ** 2 for yi, xi in zip(y, x)])
        return (dist ** 0.5)
    
    """
    def score(self,x):
         rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        
        dist_array = np.array([self.dist_euc(x,y) for y in self.desc_set])
        sorted_indexs = np.argsort(dist_array)
        #sorted_dist_array = dist_array(sorted_indexs)
        
        proportion = sum([self.label_set[i] for i in sorted_indexs[:self.k] if self.label_set[i] == 1]) / self.k
        
        return proportion
        
    def predict(self, x):
         rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        
        
        if self.score(x) > 0.5:
            return 1
        
        return -1
    """
    # optimizing score
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist_array = np.array([self.dist_euc(x,y) for y in self.desc_set])
        # The k-th element will be in its final sorted position and all smaller
        # elements will be moved before it and all larger elements behind it. 
        # perform partial sort ( in O(n) time as opposed to full sort that is O(n) * log(n))
        sorted_indexs = np.argpartition(dist_array, self.k)[:self.k]
        
        somme = sum([self.label_set[i] for i in sorted_indexs])
        
        return somme
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        
        if self.score(x) > 0:
            return 1
        
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set.copy()
        self.label_set = label_set.copy()