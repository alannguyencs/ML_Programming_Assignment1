from params import *
from scipy.optimize import linprog
from cvxopt import matrix, solvers


class Estimator:
	def __init__(self):
		self.theta = None
		self.name = None

	def save(self, model_path):
		np.save(model_path, self.theta)

	def load(self, model_path):
		self.theta = np.load(model_path)


class LeastSquares(Estimator):
	def __init__(self):
		self.name = "LeastSquares"
    
	def fit(self, phi, y):
		self.theta = np.linalg.inv(phi.dot(phi.T)).dot(phi).dot(y)

	def predict(self, phi):
		return phi.T.dot(self.theta)

    


class RegularizedLeastSquares(Estimator):
    def __init__(self):
        self.name = "RegularizedLeastSquares"

    def fit(self, phi, y, _lambda):
        d = phi.shape[0]
        lambda_I = _lambda * np.identity(d)
        self.theta = np.linalg.inv(phi.dot(phi.T) + lambda_I).dot(phi).dot(y)

    def predict(self, phi):
        return phi.T.dot(self.theta)



class Lasso(Estimator):
    def __init__(self):
        self.name = "Lasso"

    def fit(self, phi, y, _lambda):
        dimension = phi.shape[0]
        phi_phi_T = phi.dot(phi.T)
        H = np.hstack((np.vstack((phi_phi_T, -phi_phi_T)), np.vstack((phi_phi_T, phi_phi_T))))
        phi_Y = phi.dot(y)
        f = _lambda * np.ones((2 * dimension, 1)) - np.vstack((phi_Y, -phi_Y))
        G = - np.identity(2 * dimension)
        h = np.zeros(2 * dimension)

        H = matrix(H)
        f = matrix(f)
        G = matrix(G)
        h = matrix(h)

        res = solvers.qp(H, f, G, h, options={'show_progress': False})
        res = np.array(res['x'])
        self.theta = res[:dimension] - res[dimension:]

    def predict(self, phi):
        return phi.T.dot(self.theta)


class RobustRegression(Estimator):
    def __init__(self):
        self.name = "RobustRegression"

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



class BayesianRegression(Estimator):

    def __init__(self):
        self.name = "BayesianRegression"
        self.mean = None
        self.variance = None


    def fit(self, phi, y, _alpha):
        d = phi.shape[0]
        I_d = 1./_alpha * np.identity(d)
        self.variance = np.linalg.inv(I_d + phi.dot(phi.T) * (1./(_alpha**2)))
        self.mean = 1./(_alpha**2) * self.variance.dot(phi).dot(y)

    def predict(self, phi):
        mean_star = (phi.T).dot(self.mean)
        variance_star = ((phi.T).dot(self.variance).dot(phi)).diagonal()
        variance_star = variance_star[:, np.newaxis]
        std_star = np.asarray([math.sqrt(variance_star[i]) for i in range(variance_star.shape[0])])
        return mean_star, std_star
