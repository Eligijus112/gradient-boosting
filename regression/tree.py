# Data wrangling
import pandas as pd 

# OS traversal 
import os 


class Tree():
    """
    Class to fit a regression tree to the given data 
    """
    def __init__(
        self, 
        d: pd.DataFrame,
        y_var: str,
        x_vars: list,
        max_depth: int = 4
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
        """
        # Saving the names of y variable and X features
        self.y_var = y_var 
        self.features = x_vars

        # Saving the node data to memory 
        self.d = d[[y_var] + x_vars] 

        # Saving the data to the node 
        self.Y = d[y_var].values.tolist()

        # Saving the number of observations in the node 
        self.n = len(d)

        # Initiating the depth counter
        self.depth = 0

        # Saving the maximum depth of the tree
        self.max_depth = max_depth

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

    # Initiating the ungrown tree 
    tree = Tree(
        d,
        _y_var,
        _x_vars,
    )

    # Root node's best split 
    print(f"Root feature to split:\n{tree.best_feature}\nValue:\n{tree.best_feature_value}")