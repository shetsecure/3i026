import numpy as np
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

def crossvalidation(LC, DS, m, debug=True):
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

def compare(X, y, classif_dict, m=10, show_res=True, plot=False, returnRes=False):   

    Resultats = crossvalidation(list(classif_dict.values()), (X, y), m, debug=False)
    
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
        
    if returnRes:
        return Resultats
