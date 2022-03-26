# Data wrangling
import pandas as pd 

# OS traversal 
import os 

# Infinity constant 
from math import inf 


class Tree():
    """
    Class to fit a regression tree to the given data 
    """
    def __init__(
        self, 
        d: pd.DataFrame,
        y_var: str,
        x_vars: list,
        max_depth: int = 4,
        min_sample_leaf: int = 2
    ):
        """
        Class to create the regression tree object. 

        Arguments
        ---------
        d: pd.DataFrame
            The dataframe to create the tree from
        y_var: str
            The target values
        x_vars: dict
            The features to use in the tree
        max_depth: int
            The maximum depth of the tree
        min_sample_leaf: int 
            The minimum number of observations in each of the subtrees after 
            spliting
        """
        # Saving the names of y variable and X features
        self.y_var = y_var 
        self.features = x_vars

        # Saving the node data to memory 
        self.d = d[[y_var] + x_vars].copy()

        # Saving the data to the node 
        self.Y = d[y_var].values.tolist()

        # Saving the number of observations in the node 
        self.n = len(d)

        # Initiating the depth counter
        self.depth = 0

        # Saving the maximum depth of the tree
        self.max_depth = max_depth

        # Saving the minimum samples in the dataframe after spliting 
        self.min_sample_leaf = min_sample_leaf

        # Calculating the mse of the node 
        self.get_y_mse()

        # Infering the best split 
        self.get_best_split()

        # Saving to memory the y mean (prediction of the node)
        self.get_y_mean()

    @staticmethod
    def get_mean(x: list) -> float:
        """
        Calculates the mean over a list of float elements
        """
        # Initiating the sum counter 
        _sum = 0 

        # Infering the lenght of list 
        _n = len(x)

        if _n == 0:
            return inf

        # Iterating through the y values
        for _x in x:
            _sum += _x

        # Returning the mean 
        return _sum / _n 

    def get_y_mean(self) -> None:
        """
        Saves the current node's mean
        """
        self.y_mean = self.get_mean(self.Y)

    def get_mse(self, x: list) -> float: 
        """
        Calculates the mse of a given list by subtracting the mean, 
        summing and dividing by n
        """
        # Infering the lenght of list 
        _n = len(x)

        if _n == 0:
            return inf

        # Calculating the mean 
        _mean = self.get_mean(x)

        # Getting the residuals 
        residuals = [_x - _mean for _x in x]

        # Squaring the residuals
        residuals = [r ** 2 for r in residuals]

        # Summing the residuals 
        _r_sum = 0 
        for r in residuals:
            _r_sum += r

        # Returning the mean squared error 
        return _r_sum / _n      

    def get_y_mse(self) -> None:
        """
        Method to calculate the MSE of the current node
        """ 
        self.mse = self.get_mse(self.Y)

    def get_mse_weighted(self, y_left: list, y_right: list):
        """
        Calculates the weighted mse given two lists
        """
        # Calculating the lenth of both values
        _n_left = len(y_left)
        _n_right = len(y_right)
        _n_total = _n_left + _n_right

        # Calculating the mse of each sides
        _mse_left = self.get_mse(y_left)
        _mse_right = self.get_mse(y_right)

        # Calculating the weighted mse 
        return (_mse_left * _n_left / _n_total) + (_mse_right * _n_right / _n_total) 

    def get_best_split(self):
        """
        Method to find the best split among the features 
        
        The logic is to find the feature and the feature value which reduces 
        the objects mse the most 
        """
        # Setting initial values
        _best_mse = self.mse
        _best_feature = None
        _best_feature_value = None

        # Creating lists of categorical and numeric features
        _cat_features = [ft for ft in self.d.columns if self.d[ft].dtype == 'category']
        _num_features = list(set(self.features) - set(_cat_features))

        # Going through the categorical features
        for _cat_feature in _cat_features:
            # Infering the levels of the categorical feature 
            _levels = self.d[_cat_feature].unique()
            
            for _level in _levels:
                # Spliting the data into two parts: one that is equal to the categorical level
                # and one that is not 
                _y_left = self.d.loc[self.d[_cat_feature]==_level, self.y_var].values
                _y_right = self.d.loc[self.d[_cat_feature]!=_level, self.y_var].values

                if len(_y_left) >= self.min_sample_leaf and len(_y_right) >= self.min_sample_leaf:
                    # Calculating the weighted mse 
                    _mse_w = self.get_mse_weighted(_y_left, _y_right)

                    # Checking the clause 
                    if _mse_w < _best_mse:
                        _best_mse = _mse_w 
                        _best_feature = _cat_feature
                        _best_feature_value = str(_level) # Specificaly adding the type for later spliting

        # Going through the numerical features
        for _num_feature in _num_features:
            # Getting the values 
            _values = self.d[_num_feature].values

            # Getting the unique entries
            _values = list(set(_values))

            # Sorting the values
            _values.sort()

            # Getting the rolling average values of the feature 
            # and spliting the dataset by that value
            for i in range(len(_values) - 1): 
                # Roling average
                _left = _values[i]
                _right = _values[i + 1]
                _mean = (_left + _right) / 2

                # Iterating over the values and calculating the mse 
                _y_left = self.d.loc[self.d[_num_feature]<=_mean, self.y_var].values
                _y_right = self.d.loc[self.d[_num_feature]>_mean, self.y_var].values

                if len(_y_left) >= self.min_sample_leaf and len(_y_right) >= self.min_sample_leaf:
                    # Getting the weighted mse 
                    _mse_w = self.get_mse_weighted(_y_left, _y_right)

                    # Checking the clause 
                    if _mse_w < _best_mse:
                        _best_mse = _mse_w 
                        _best_feature = _num_feature
                        _best_feature_value = _mean

        # Saving the best splits to object memory 
        self.best_feature = _best_feature
        self.best_feature_value = _best_feature_value

    def fit(self):
        """
        The recursive method to fit a regression tree on the data provided
        """
        if self.depth < self.max_depth and self.best_feature is not None:
            # Spliting the data depending on the found best splits 
            _best_feature = self.best_feature
            _best_feature_value = self.best_feature_value

            # Spliting the data for the creation of additional sub trees
            _d_left = pd.DataFrame()
            _d_right = pd.DataFrame()
            if isinstance(_best_feature_value, str):
                _d_left = self.d[self.d[_best_feature]==_best_feature_value].copy()
                _d_right = self.d[self.d[_best_feature]!=_best_feature_value].copy()
            else:
                _d_left = self.d[self.d[_best_feature]<=_best_feature_value].copy()
                _d_right = self.d[self.d[_best_feature]>_best_feature_value].copy()

            # Creating the tree instances 
            _left_tree = Tree(
                d = _d_left.copy(),
                y_var = self.y_var,
                x_vars = self.features,
                min_sample_leaf = self.min_sample_leaf,
                max_depth = self.max_depth
                )
            
            _right_tree = Tree(
                d = _d_right.copy(),
                y_var = self.y_var,
                x_vars = self.features,
                min_sample_leaf = self.min_sample_leaf,
                max_depth = self.max_depth
                )

            # Setting the depths 
            _left_tree.depth = self.depth + 1
            _right_tree.depth = self.depth + 1

            # Defining the rules for the left and right subtrees
            _left_symbol = '<='
            _right_symbol = '>'
            if isinstance(_best_feature_value, str):
                _left_symbol = '=='
                _right_symbol = '!='

            _rule_left = f"{_best_feature} {_left_symbol} {_best_feature_value}"
            _rule_right = f"{_best_feature} {_right_symbol} {_best_feature_value}"

            _left_tree.rule = _rule_left
            _right_tree.rule = _rule_right

            # Saving the pointers in memory 
            self.left = _left_tree
            self.right = _right_tree

            # Continuing the recursive process
            self.left.fit()
            self.right.fit()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.depth == 0:
            print(f"Root (level {self.depth})")
        else:
            print(f"|{spaces} Split rule: {self.rule} (level {self.depth})")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.y_mean, 3)}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.depth < self.max_depth: 
            self.left.print_tree()
            self.right.print_tree()

    def predict(self, x: dict) -> float:
        """
        Returns the predict Y value based on the X values

        Arguments
        ---------
        x: dict 
            Dictionary of the structure: 
            {
                "feature_name": value,
                ...
            }
        
        Returns
        -------
        The mean Y based on the x and fitted 
        """
        # Infering the node 
        _node = self
        while _node.depth < self.max_depth and _node.best_feature is not None:
            
            # Extracting the best split feature and values 
            _best_feature = _node.best_feature
            _best_feature_value = _node.best_feature_value

            # Checking if the feature is categorical or numerical
            if isinstance(_best_feature_value, str):
                if x[_best_feature] == _best_feature_value:
                    _node = _node.left
                else:
                    _node = _node.right
            else:
                if x[_best_feature] <= _best_feature_value:
                    _node = _node.left
                else:
                    _node = _node.right
        
        # Returning the prediction
        return _node.y_mean

if __name__ == '__main__':
    # Infering the file location 
    _current_dir = os.path.dirname(os.path.abspath(__file__))

    # Defining the data directory 
    _data_dir = os.path.join(_current_dir, 'data')

    # Reading the data
    d = pd.read_csv(os.path.join(_data_dir, 'auto-mpg.csv'))

    # Encoding the correct types for categorical variables
    d['origin'] = d['origin'].astype('category')

    # Defining the y_var 
    _y_var = 'mpg'

    # Defining the features
    _x_vars = ['origin', 'weight', 'cylinders']

    # Spliting the dataset into Train and Test 
    train = d.sample(frac=0.8, random_state=0)
    test = d.drop(train.index)

    # Initiating the ungrown tree 
    tree = Tree(
        d = train,
        y_var = _y_var,
        x_vars = _x_vars,
        max_depth = 3
    )

    # Fitting the tree 
    tree.fit()

    # Printing out the tree 
    tree.print_tree()

    # Getting the predictions 
    _inputs = test[_x_vars].to_dict('records')
    _yhat = [tree.predict(x) for x in _inputs]

    # Extracting the true values
    _y_test = test[_y_var].to_list()

    # Calculating the mse 
    _mse = 0 
    for i, y_true in enumerate(_y_test):
        _mse += (y_true - _yhat[i])**2
    _mse /= len(_y_test)
    print(f"MSE on the test set: {_mse}")