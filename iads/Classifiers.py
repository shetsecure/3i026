# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
from . import utils as ut
from copy import copy

import multiprocessing
import concurrent

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
        self.w_init = self.weights
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
    
    def reset(self):
        self.weights = self.w_init
    
    
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
        self.weights = np.random.normal(0, 0.5, input_dimension + 1)
        self.w_init = self.weights
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
                    
                    # adjusting dim for the description (adding artificial 1 for the bias)
                    x = list(desc_set[i])
                    x.insert(0, 1)
                    x = np.asarray(x)
                    
                    self.weights += self.learning_rate * x * label_set[i]
                    
            counter += 1
            
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # adjusting dim for the description (adding artificial 1 for the bias)
        x = list(x)
        x.insert(0, 1)
        x = np.asarray(x)
        
        return np.dot(x, self.weights)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        
        return 1 if self.score(x) > 0 else -1
    
    def reset(self):
        self.weights = self.w_init
    
    def toString(self):
        s = "ClassifierPerceptron: \n"
        
        for key, val in self.__dict__.items():
            if str(key) != "w_init":
                s += "\t [+] " + str(key) + " = " + str(val) + '\n'
            
        return s
    
class ClassifierOneStepPerceptron(ClassifierPerceptron):
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.weights = np.random.normal(0, 0.5, input_dimension + 1)
        self.w_init = self.weights
    
    # override training method, to calculate pseudo-inverse matrix as the solution to the MSE cost function
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        X = np.c_[np.ones(desc_set.shape[0]), desc_set]
        y = label_set
        w = np.linalg.lstsq(np.matmul(X.T, X), np.matmul(X.T, y), rcond=None)
        w = np.reshape(w[0], (1, w[0].shape[0]))[0]
        self.weights = w
        
    def reset(self):
        self.weights = self.w_init
        
    def toString(self):
        s = "ClassifierOneStepPerceptron: \n"
        
        for key, val in self.__dict__.items():
            if str(key) != "w_init":
                s += "\t [+] " + str(key) + " = " + str(val) + '\n'
            
        return s

class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, kernel, max_iter=1e3):
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
        self.w_init = self.weights
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
            
    def reset(self):
        self.weights = self.w_init
        
    def toString(self):
        s = "ClassifierPerceptronKernel: \n"
        
        for key, val in self.__dict__.items():
            if str(key) != "w_init":
                if str(key) == "kernel":
                    val = "polynomial's degree is " + str(self.kernel.degree)
                    
                s += "\t [+] " + str(key) + " = " + str(val) + '\n'
            
        return s
                

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
        self._last_predictions = None
        self._last_predicted_desc = None
    
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

        dist_array = np.asarray([self.dist_euc(x,y) for y in self.desc_set])
        # The k-th element will be in its final sorted position and all smaller
        # elements will be moved before it and all larger elements behind it. 
        # perform partial sort ( in O(n) time as opposed to full sort that is O(n * log(n)) )
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
        self.desc_set = desc_set
        self.label_set = label_set

    def reset(self):
        pass

    def toString(self):
        s = "ClassifierKNN: \n"
        s += "\t [+] k = " + str(self.k) + '\n'
        
        return s
    
    def predict_all(self, desc_set):
        """
            Predict all labels of desc_set using multiple processes.
        """
        try:
            N = len(desc_set)

            cpu_count = multiprocessing.cpu_count()
            executor = concurrent.futures.ProcessPoolExecutor(cpu_count)

            futures = [executor.submit(self.predict, item) for item in desc_set]
            concurrent.futures.wait(futures)

            predictions = [f.result() for f in futures]

            self._last_predictions = predictions
            self._last_predicted_desc = desc_set

            return predictions
        
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            
    
    def fast_accuracy(self, desc_set, label_set):
        """ 
        Calculate accuracy using multiple processes.
        
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        N = len(desc_set)
        
        if np.all(self._last_predicted_desc == desc_set):
            predictions = self._last_predictions
        else:
            predictions = self.predict_all(desc_set)
        
        acc = [1 if predictions[i] * label_set[i] > 0 else 0 for i in range(N)]
        
        return float(sum(acc) / N)
    
    
class MultiClassKNN(ClassifierKNN):
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
        Argument:
            - intput_dimension (int) : dimension d'entrée des exemples
            - k (int) : nombre de voisins à considérer
        Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension, k)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        dist_array = np.asarray([self.dist_euc(x,y) for y in self.desc_set])
        sorted_indices = np.argpartition(dist_array, self.k)[:self.k]

        possible_labels = [self.label_set[i][0] for i in sorted_indices]
        
        # return most occured label
        return max( [(possible_labels.count(label), label) for label in set(possible_labels)] )[1]
    
    
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = ut.construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        root = copy(self.racine)
        assert(len(self.LNoms) == len(x))
        
        try:
            while not root.est_feuille():
                root_index = list(self.LNoms).index(root.nom_attribut)
                root = copy(root.Les_fils[x[root_index]])
        except KeyError:
            print("*** Warning: attribut " + str(root.nom_attribut) + " -> Valeur inconnue: " + str(x[root_index]))
            return 0 # inspired from below (la classification de certains exemples produit un warning)
            
        return root.classe

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)