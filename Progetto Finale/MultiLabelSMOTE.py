import numpy as np
import pandas as pd
import random
from sklearn.neighbors import  NearestNeighbors
from sklearn.exceptions import NotFittedError
class MultiLabelSMOTE:

    def __init__(self, n_neighbors: int = 5):
        """
        Class constructor

        args
        n_neighbors: int (default 5), neighbors number, to be used in NearestNeighBors instance.

        """
        self.nbs=NearestNeighbors(n_neighbors=n_neighbors, metric='cityblock', algorithm='kd_tree')
        self.X_sub = pd.DataFrame()
        self.y_sub = pd.DataFrame()
        self.fitted = False

    def fit_resample(self, X: pd.DataFrame, y: pd.DataFrame, n_sample: int = 100):
        """
        Fit the MLSMOTE and resample data. 

        args
        X: pandas.DataFrame, input vector DataFrame
        y: pandas.DataFrame, feature vector dataframe
        n_sample: int (default 100), number of newly generated sample

        return
        new_X: pandas.DataFrame, augmented feature vector data
        target: pandas.DataFrame, augmented target vector data
        """
        self.fit(X, y)
        return self.resample(X, y, n_sample)

    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X_sub, self.y_sub = self.get_minority_instace(X, y) # X_sub e y_sub sono sotto matrici di X e y
        self.nbs.fit(self.X_sub)
        self.fitted = True

    def resample(self, X: pd.DataFrame, y: pd.DataFrame, n_sample : int = 5):
        """
        Give the augmented data using MLSMOTE algorithm
        
        args
        X: pandas.DataFrame, input vector DataFrame
        y: pandas.DataFrame, feature vector dataframe
        n_sample: int, number of newly generated sample
        
        return
        new_X: pandas.DataFrame, augmented feature vector data
        target: pandas.DataFrame, augmented target vector data
        """

        #X_sub, y_sub = self.get_minority_instace(X, y) # X_sub e y_sub sono sotto matrici di X e y

        if not(self.fitted):
            raise NotFittedError("This MultiLabelSMOTE instance is not fitted yet. Call 'fit' with appropriate arguments before using this sampler.")

        indices2 = self.nearest_neighbour(self.X_sub)
        n = len(indices2)
        new_X = np.zeros((n_sample, self.X_sub.shape[1]))
        target = np.zeros((n_sample, self.y_sub.shape[1]))
        for i in range(n_sample):
            reference = random.randint(0,n-1)
            neighbour = np.random.choice(indices2[reference,1:]) #prendo un vicino random tra i k selezionati
            all_point = indices2[reference]
            nn_df = self.y_sub[self.y_sub.index.isin(all_point)]
            ser = nn_df.sum(axis = 0, skipna = True)
            target[i] = np.array([1 if val>2 else 0 for val in ser])
            ratio = random.random() #lambda
            # x_i = X_sub.loc[reference,:]
            # x_zi = self.X_sub.loc[neighbour,:] (vicino selezionato)
            gap = self.X_sub.loc[reference,:] - self.X_sub.loc[neighbour,:]
            # new_X[i] = np.array(self.X_sub.loc[reference,:] + ratio * gap)
            new_X[i] = np.round(np.abs(self.X_sub.loc[reference,:] + ratio * gap)).astype(int)
        new_X = pd.DataFrame(new_X, columns=self.X_sub.columns)
        target = pd.DataFrame(target, columns=self.y_sub.columns)
        # new_X = pd.concat([X_sub, new_X], axis=0)
        # target = pd.concat([y_sub, target], axis=0)

        #TODO: mettere insieme X + X_new e y + target (fare un concat) - fatto
        new_X = pd.concat([X, new_X], axis=0)
        target = pd.concat([y, target], axis=0)
        return new_X, target
        


    def get_tail_label(self, df: pd.DataFrame):
        """
        Give tail label colums of the given target dataframe
        
        args
        df: pandas.DataFrame, target label df whose tail label has to identified
        
        return
        tail_label: list, a list containing column name of all the tail label
        """
        columns = df.columns
        n = len(columns)
        irpl = np.zeros(n)
        for column in range(n):
            irpl[column] = df[columns[column]].value_counts()[1]
        irpl = max(irpl)/irpl
        mir = np.average(irpl)
        tail_label = []
        for i in range(n):
            if irpl[i] > mir:
                tail_label.append(columns[i])
        return tail_label

    def get_index(self, df: pd.DataFrame):
        """
        give the index of all tail_label rows
        args
        df: pandas.DataFrame, target label df from which index for tail label has to identified
            
        return
        index: list, a list containing index number of all the tail label
        """
        # tail_labels = self.get_tail_label(df)
        tail_labels = df.columns
        index = set()
        for tail_label in tail_labels:
            sub_index = set(df[df[tail_label]==1].index)
            index = index.union(sub_index)
        return list(index)

    def get_minority_instace(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Give minority dataframe containing all the tail labels
        
        args
        X: pandas.DataFrame, the feature vector dataframe
        y: pandas.DataFrame, the target vector dataframe
        
        return
        X_sub: pandas.DataFrame, the feature vector minority dataframe
        y_sub: pandas.DataFrame, the target vector minority dataframe
        """
        index = self.get_index(y)
        X_sub = X[X.index.isin(index)].reset_index(drop = True)
        y_sub = y[y.index.isin(index)].reset_index(drop = True)
        return X_sub, y_sub
    
    def nearest_neighbour(self, X: pd.DataFrame):
        """
        Give index of self.n_neighbors nearest neighbor of all the instance
        
        args
        X: pandas.DataFrame, array whose nearest neighbor has to find
        
        return
        indices: list of list, index of self.n_neighbors NN of each element in X
        """
        indices = self.nbs.kneighbors(X, return_distance=False)
        return indices
    