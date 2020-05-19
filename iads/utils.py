# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz as gv
from . import evaluation as ev # we used from . because of sys.path.append('../')
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

def categories_2_numeriques(DF,nom_col_label =''):
    """ DataFrame * str -> DataFrame
        nom_col_label est le nom de la colonne Label pour ne pas la transformer
        si vide, il n'y a pas de colonne label
        rend l'équivalent numérique de DF
    """
    dfloc = DF.copy()  # pour ne pas modifier DF
    L_new_cols = []    # pour mémoriser le nom des nouvelles colonnes créées
    Noms_cols = [nom for nom in dfloc.columns if nom != nom_col_label]
     
    for c in Noms_cols:
        if dfloc[c].dtypes != 'object':  # pour détecter un attribut non catégoriel
            L_new_cols.append(c)  # on garde la colonne telle quelle dans ce cas
        else:
            for v in dfloc[c].unique():
                nom_col = c + '_' + v    # nom de la nouvelle colonne à créer
                dfloc[nom_col] = 0
                dfloc.loc[dfloc[c] == v, nom_col] = 1
                L_new_cols.append(nom_col)
            
    return dfloc[L_new_cols]  # on rend que les valeurs numériques

class KernelPoly: # For kernelized perceptron
    def __init__(self, degree=2):
        assert(int(degree) >= 1)
        self.degree = degree
    
    def transform(self, x, degree = None): # what polynomial will we return
        if degree is None:
            degree = self.degree
            
        assert(int(degree) >= 1)
        
        if (int(degree) > len(x)):
            print(" Cannot create a 3rd kernel polynomial if the number of features < ", degree)
        assert(int(degree) <= len(x))
        
        l = [1]
        l.extend([xx for xx in x])
        start_from = 1
        
        if degree > 1:
            # now we will add the combinaisons
            for i in range(2, degree+1):
                l.extend([np.product(list(x)) for x in  list(combinations(x, i))])
                
        return np.asarray(l)

############### FOR DECISION TREE ###################

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    if k <= 1:
        return 0.0
    
    return  -sum([i * math.log(i, k) if i > 0 else 0 for i in P])
    
def entropie(labels):
    labels = np.asarray(labels)
    
    valeurs, nb_fois = np.unique(labels ,return_counts=True)
    k = len(labels)
    
    return shannon([count_label / k for count_label in nb_fois])

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    
    return valeurs[np.argmax(nb_fois)]


class AdaptateurCategoriel:
    """ Classe pour adapter un dataframe catégoriel par l'approche one-hot encoding
    """
    def __init__(self, DF, nom_col_label=''):
        """ Constructeur
            Arguments: 
                - DataFrame représentant le dataset avec des attributs catégoriels
                - str qui donne le nom de la colonne du label (que l'on ne doit pas convertir)
                  ou '' si pas de telle colonne. On mémorise ce nom car il permettra de toujours
                  savoir quelle est la colonne des labels.
        """
        self.DF = DF  # on garde le DF original  (rem: on pourrait le copier)
        self.nom_col_label = nom_col_label 
        
        # Conversion des colonnes catégorielles en numériques:
        self.DFcateg = categories_2_numeriques(DF, nom_col_label)
        
        # Pour faciliter les traitements, on crée 2 variables utiles:
        self.data_desc = self.DFcateg.values
        self.data_label = self.DF[nom_col_label].values
        # Dimension du dataset convertit (sera utile pour définir le classifieur)
        self.dimension = self.data_desc.shape[1]
                
    def get_dimension(self):
        """ rend la dimension du dataset dé-catégorisé 
        """
        return self.dimension
        
        
    def train(self,classifieur):
        """ Permet d'entrainer un classifieur sur les données dé-catégorisées 
        """        
        classifieur.train(self.data_desc, self.data_label)
    
    def accuracy(self,classifieur):
        """ Permet de calculer l'accuracy d'un classifieur sur les données
            dé-catégorisées de l'adaptateur.
            Hypothèse: le classifieur doit avoir été entrainé avant sur des données
            similaires (mêmes colonnes/valeurs)
        """
        return classifieur.accuracy(self.data_desc,self.data_label)

    def converti_categoriel(self, x):
        """ transforme un exemple donné sous la forme d'un dataframe contenant
            des attributs catégoriels en son équivalent dé-catégorisé selon le 
            DF qui a servi à créer cet adaptateur
            rend le dataframe numérisé correspondant             
        """
        # Assuming that the example is one line.
        # IMPORTANT: this will work assuming that new columns are constructed in the same way we did before:
        # nom_col = c + '_' + v
        
        x_copy = list(x.values[0])
        all_values = [c.split('_')[-1] for c in list(self.DFcateg.columns)]
        
        # casting to set will not change the len, because the columns are uniques anyways
        intersection = list(set(x_copy).intersection(set(all_values)))
        
        one_columns = [all_values.index(i) for i in intersection] # the indices of columns that will have a value of 1
        
        values = [0] * len(all_values)
        
        for index in one_columns:
            values[index] = 1
            
        cols = list(self.DFcateg.columns)
            
        # Adding the label
        if self.nom_col_label != '':
            values.append(x[self.nom_col_label][0])
            cols.append(self.nom_col_label)
        else: # we'll consider it as the value of the last column
            values.append(x_copy[-1])
            cols.append('LabelX') 
        
        return  pd.DataFrame([values], columns=cols)
    
    def predict(self, x, classifieur):
        """ rend la prédiction de x avec le classifieur donné
            Avant d'être classifié, x doit être converti
        """
        x_df = self.converti_categoriel(x)
        return classifieur.predict(x_df[self.DFcateg.columns].values[0])
    
    def cross_validation(self, classifier, m):
        if m == 1:
            return ev.leave_one_out(classifier, (self.data_desc, self.data_label))
        
        return ev.crossvalidation(classifier, (self.data_desc, self.data_label), m)
    
    def compare_classifiers(self, classif_dict, m=10, show_res=True, plot=False):
        return ev.compare(self.data_desc, self.data_label, classif_dict, m, show_res, plot)

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g
    
    
def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        ############################# DEBUT ########
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        X_entropies = []
        
        for i in range(X.shape[1]): # pour chaque attribut Xj qui décrit les exemples de X
            # pour chacune des valeurs vjl de Xj construire l'ensemble des exemples de X qui possède la valeur vjl 
            s = pd.Series(X[:, i]) # ith column
            d = s.groupby(s).groups # dict[value = vjl] = indices ( a list Int64 index)
            
            vjls_ent = []
            # pour chacune des valeurs vjl de Xj
            for vjl, indices in d.items():
                vjl_labels = Y[indices] # ainsi que l'ensemble de leurs labels.
                
                vjl_entropie = entropie(vjl_labels) # HS(Y|vjl) 
                p_vjl = len(indices) / len(s)
                vjls_ent.append(vjl_entropie * p_vjl)
                
            # HS(Y|Xj) 
            Xj_entropie = sum(vjls_ent)
            
            X_entropies.append(Xj_entropie)
            
        
        min_entropie = min(X_entropies)
        i_best = X_entropies.index(min_entropie)
        Xbest_valeurs = np.unique(X[:, i_best])
        
        ############################# FIN ######## 
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
    
############################################################################################