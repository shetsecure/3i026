import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

def crossvalidation(C, DS, m=10):
    """ Classifieur * tuple[array, array] * int -> tuple[tuple[float,float], tuple[float,float]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    
    
    ############################ IMPORTANT ####################################
    # 𝐷𝑆1 comme dataset de test et les autres datasets 𝐷𝑆0, 𝐷𝑆3,...,𝐷𝑆9 comme dataset pour apprentissage 
    # each iteration we pick 1 for testing and the others for training
    ###########################################################################
    
    data_desc, data_labels = DS[0], DS[1]
    indices = [i for i in range(len(data_desc))]
    train_accs = []
    test_accs = []
    m_indices = []
    
    length = len(data_desc) // m
    
    for i in range(m): # we're not treating the case where m%len(DS) != 0 (if the rest has a size = 1, it'll kill the avg)
        # random tirage
        np.random.shuffle(indices)
        
        m_indices.append([i for i in indices[:length]])

        # remove the first length indices so that we don't take the same description twice
        for j in range(length):
            indices.pop(0)
    
    for test_index in range(m):
        # getting the training indices in each iteration
        train_indices = list(chain.from_iterable([m_indices[i] for i in range(m) if i != test_index]))
        
        # append training accuracy to the training_set
        C.train(data_desc[train_indices], data_labels[train_indices])   
        train_accs.append(C.accuracy(data_desc[train_indices], data_labels[train_indices]))

        # append test accuracy to the test_set
        test_accs.append(C.accuracy(data_desc[m_indices[test_index]], data_labels[m_indices[test_index]]))    
        
    print("[Info debug] liste accuracies Apprentissage: ", train_accs)
    print("[Info debug] liste accuracies Test         : ", test_accs)
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    
    return (train_accs.mean(), train_accs.std()), (test_accs.mean(), test_accs.std())

def leave_one_out(C, DS):
    """ 
        Classifieur * tuple[array, array] -> float
    """
    
    data_desc, data_labels = DS[0], DS[1]
    indices = [i for i in range(len(data_desc))]
    np.random.shuffle(indices)
    
    marked_points = 0 # nombre de points marqués
    
    # grab the first index in the list and make it the one who's out (for test) (out)
    # train on the rest
    # test on the out (index)
    # append out to the end of the list (indices)
    
    for test_index in range(len(indices)):
        out = indices.pop(0)
        
        # training
        C.train(data_desc[indices], data_labels[indices])
        
        # transforming the args to list, to make it compatible for the accuracy method
        marked_points += C.accuracy([data_desc[out]], [data_labels[out]]) 
        
        indices.append(out) # add to the end of the list
        
    return marked_points / len(data_desc)

def crossvalidation_multiple(LC, DS, m, debug=True):
    """ List[Classifieur] * tuple[array, array] * int ->  List[tuple[tuple[float,float], tuple[float,float]]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    
    data_desc, data_labels = DS[0], DS[1]
    indices = [i for i in range(len(data_desc))]
    train_accs = []
    test_accs = []
    m_indices = []
    length = len(data_desc) // m
    
    for i in range(len(LC)):
        train_accs.append([])
        test_accs.append([])
    
        
    for i in range(m): # we're not treating the case where m%len(DS) != 0 (if the rest has a size = 1, it'll kill the avg)
        # random tirage
        np.random.shuffle(indices)
        
        m_indices.append([i for i in indices[:length]])

        # remove the first length indices so that we don't take the same description twice
        for j in range(length):
            indices.pop(0)
            
    # take one for training, the others for testing
    for test_index in range(m):
        # getting the training indices in each iteration
        train_indices = list(chain.from_iterable([m_indices[i] for i in range(m) if i != test_index])) 
        
        
        for i in range(len(LC)):
            LC[i].train(data_desc[train_indices], data_labels[train_indices])

            # append training accuracy to the training_set
            train_accs[i].append(LC[i].accuracy(data_desc[train_indices], data_labels[train_indices]))

            # append test accuracy to the test_set
            test_accs[i].append(LC[i].accuracy(data_desc[m_indices[test_index]], data_labels[m_indices[test_index]]))    
            
            # reset the classifier, to have a fresh one next iteration
            LC[i].reset()
    
    if debug:
        print("Il y a ", len(LC), "classifieurs à comparer.")
        for i in range(len(LC)):
            print("[Info debug] Classifieur   ", i)
            print("[Info debug] liste accuracies Apprentissage: ", train_accs[i])
            print("[Info debug] liste accuracies Test         : ", test_accs[i])
            print()

    for i in range(len(LC)):
        train_accs[i] = np.array(train_accs[i])
        test_accs[i] = np.array(test_accs[i])
    
    returned_values = [((tr.mean(), tr.std()), (ts.mean(), ts.std())) for tr, ts in zip(train_accs, test_accs)]
    
    return returned_values

def compare(X, y, classif_dict, m=10, show_res=True, plot=False): 
    """
          Comparing multiple classifiers on a single dataset using m-fold cross validation.
          
          @params:
              [+] X: desc data (np ndarray)
              [+] y: labels    (np ndarray)
              [+] classif_dict : dictionnary of classifiers (dict where keys are str (name of classifier) and value are objects)
              [+] m : how many folds for cross_validation (int)
              [+] show_res: show detailed results of the comparaison
              [+] plot: boolean value for plotting the results
    """

    Resultats = crossvalidation_multiple(list(classif_dict.values()), (X, y), m, debug=False)
    
    def plot_comparaison():
        perf_train = []
        perf_test = []
        classifs = [i for i in range(len(classif_dict))]

        for i in range(len(classif_dict)):
            perf_train.append(Resultats[i][0][0])
            perf_test.append(Resultats[i][1][0])

        plt.figure(figsize=(7, 3))
        plt.plot(classifs, perf_train, label='Training performane')
        plt.plot(classifs, perf_test, label='Testing performane')
        plt.ylabel('Performane')
        plt.xlabel('Classifiers')
        plt.xticks(np.arange(len(classif_dict)), list(classif_dict.keys()), rotation='vertical')
        
        best_index = perf_test.index(max(perf_test))
        names = list(classif_dict.keys())
        
        plt.title("Comparing multiple classifiers\nBest classifier is %s with a test accuracy of %.2f%%" 
                  %(names[best_index], perf_test[best_index]*100))
        plt.legend()
        plt.show()
        
        print("Best classifier details: ")
        print(classif_dict[names[best_index]].toString())
    
    if show_res:
        print("\n*****\nAffichage des résultats:")

        i = 0
        for key in classif_dict.keys():
            print("Classifieur ", classif_dict[key].toString())
            print("\t(moyenne, std) pour apprentissage :", Resultats[i][0])
            print("\t(moyenne, std) pour test          :", Resultats[i][1])
            i += 1
            
    if plot:
        plot_comparaison()
        
    return Resultats


def errorBarComparisonPlot(Resultats, classif_dict, m=10):
    #Resultats = np.asarray(crossvalidation_multiple(list(classif_dict.values()), (X, y), m, debug=False))
    Resultats = np.asarray(Resultats)
    
    train_perf = []
    test_perf = []
    for i in range(len(Resultats)):
        train_perf.append(Resultats[i][0]*100)
        test_perf.append(Resultats[i][1]*100)

    train_perf = np.asarray(train_perf)
    test_perf = np.asarray(test_perf)
    
    df_train = pd.DataFrame.from_dict({
        "model" : list(classif_dict.keys()),
        "mean": train_perf[:, 0],
        "std": train_perf[:, 1]
    })

    df_test = pd.DataFrame.from_dict({
        "model" : list(classif_dict.keys()),
        "mean": test_perf[:, 0],
        "std": test_perf[:, 1]
    })
    
    plt.ylabel('Performane')
    plt.xlabel('Classifiers')
    plt.xticks(np.arange(len(classif_dict)), list(classif_dict.keys()), rotation='vertical')
 
    plt.errorbar(df_train['model'], df_train['mean'], df_train['std'], lw=2, label='Training performane')
    plt.legend()
    plt.show()
    
    
    plt.ylabel('Performane')
    plt.xlabel('Classifiers')
    plt.xticks(np.arange(len(classif_dict)), list(classif_dict.keys()), rotation='vertical')
 
    plt.errorbar(df_test['model'], df_test['mean'], df_test['std'], lw=2, label='Testing performane', color='#ff7f0e')
    plt.legend()
    plt.show()
    