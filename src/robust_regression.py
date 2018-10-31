from params import *
from scipy.optimize import linprog

class RobustRegression():
    def __init__(self):
        self.theta = None

    def fit(self, phi, y):
        d = phi.shape[0]
        N = phi.shape[1]
        In = np.identity(N)
        A = np.vstack((np.hstack((-phi.T, -In)), np.hstack((phi.T, -In))))
        b = np.vstack((-y, y)).reshape((2 * N,))
        f = np.vstack((np.zeros((d, 1)), np.ones((N, 1)))).reshape(((d + N),))

        res = linprog(c=f, A_ub=A, b_ub=b, bounds=(None, None), method='interior-point')
        self.theta = np.asarray(res['x'][:d])

    def predict(self, phi):
        return phi.T.dot(self.theta)

    def save(self, model_path):
        np.save(model_path, self.theta)

    def load(self, model_path):
        self.theta = np.load(model_path)