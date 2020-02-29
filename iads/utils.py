# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ------------------------ 
def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o') # 'o' pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])    
    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
        par défaut: binf vaut -1 et bsup vaut 1
    """
    assert(isinstance(n, int) and isinstance(n, int))
    assert(n >= 0 and p > 0)
    assert(n%2 == 0)
    
    data_desc = np.random.uniform(binf, bsup, (n, p))
    data_labels = np.asarray([-1 for i in range(n//2)] + [1 for i in range(n//2)])
    
    return data_desc, data_labels
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    
    negatif_pts = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    positif_pts = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    data_desc = np.concatenate((negatif_pts, positif_pts))
    data_labels = np.asarray([-1 for i in range(nb_points)] + [1 for i in range(nb_points)])
    
    return data_desc, data_labels
# ------------------------ 
def create_XOR(nb_points, var):
    pos_1 = np.random.multivariate_normal(np.array([0,0]), np.array([[var, 0], [0, var]]), nb_points)
    pos_2 = np.random.multivariate_normal(np.array([1,1]), np.array([[var, 0], [0, var]]), nb_points)
    pos = np.concatenate((pos_1, pos_2))
    
    neg_1 = np.random.multivariate_normal(np.array([1,0]), np.array([[var, 0], [0, var]]), nb_points)
    neg_2 = np.random.multivariate_normal(np.array([0,1]), np.array([[var, 0], [0, var]]), nb_points)
    neg = np.concatenate((neg_1, neg_2))
    
    data = np.concatenate((pos, neg))
    labels = np.asarray([1 for i in range(nb_points*2)] + [-1 for i in range(nb_points*2)])
    
    return data, labels
 # ------------------------ 
    
class KernelPoly:
    def transform(self, x, degree = 2): # what polynomial will we return
        assert(isinstance(degree, int) and degree >= 1 and degree <= len(x))
        
        l = [1]
        l.extend([xx for xx in x])
        start_from = 1
        
        if degree > 1:
            # now we will add the combinaisons
            for i in range(2, degree+1):
                l.extend([np.product(list(x)) for x in  list(combinations(x, i))])
                
        return np.asarray(l)