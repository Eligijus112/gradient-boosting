# Gradient boosting 

Implementation of gradient boosting algorithm using Python. The dependency on third party packages it put to a minimum. 

The original paper for gradient boosting by Jerome H. Friedman https://jerryfriedman.su.domains/ftp/trebst.pdf.

# Regression 

If the response variable is continuous, the gradient boosting algorithm is sometimes called regression boosting. The implementation of the algorithm is in the `regression` directory. 

A single regression tree implementation is in the `regression/tree.py` file.

The implementation of the gradient boosting algorithm for regression is in the `regression/boosting.py` file.

## Data 

The data to showcase the regression boosting algorithm is in the `regression/data/auto-mpg.csv` file. The target variable is the `mpg` - miles per galon - column. 