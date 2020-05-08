import numpy as np
from qp import solve_QP

def linear_kernel(xj, xk):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :return: float64
    """
    #TODO
    pass

def rbf_kernel(xj, xk, gamma = 0.1):
    """
    Kernel Function, radial basis function kernel or gaussian kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param gamma: parameter of the RBF kernel.
    :return: float64
    """
    # TODO

def polynomial_kernel(xj, xk, c = 0, d = 1):
    """
    Kernel Function, polynomial kernel

    :param xj: an input sample (np array)
    :param xk: an input sample (np array)
    :param c: mean of the polynomial kernel (np array)
    :param d: exponent of the polynomial (np array)
    :return: float64
    """
    #TODO
    pass

class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels
        Q, c = self._objective_function()
        A, b = self._inequality_constraints()
        E, d = self._equality_constraints()
        # TODO: Uncomment the next line when you have implemented _objective_function(),
        # _inequality_constraints() and _equality_constraints().
        # self.alphas = solve_QP(Q, c, A, b, E, d)

        #TODO: Given the alphas computed by the quadratic solver, compute the bias
        pass


    def _objective_function(self):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        For specifics on the values for Q and c, see the objective function in the handout.

        :return: two numpy arrays, Q and c which fully specify the objective function.
        """

        #TODO
        return None, None

    def _equality_constraints(self):
        """
        Generate the equality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ex = d.

        For specifics on the values for E and d, see the constraints in the handout

        :return: two numpy arrays, E, the coefficients, and d, the values
        """

        #TODO
        return None, None

    def _inequality_constraints(self):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        For specifics on the values of A and b, see the constraints in the handout

        :return: two numpy arrays, A, the coefficients, and b, the values
        """
        #TODO
        return None, None

    def predict(self, input):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        pass

    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO
        pass
