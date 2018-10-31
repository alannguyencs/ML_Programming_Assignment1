from params import *
from least_squares import LeastSquares
from regularized_least_squares import RegularizedLeastSquares
# from lasso import Lasso
from robust_regression import RobustRegression
from bayesian_regression import BayesianRegression
from dataset import DataSet
from common import *

result_sub_path = result_path + 'part1b/'
#get data
sample_data = DataSet(sample_x_path, sample_y_path)
sample_x, sample_y = sample_data.x, sample_data.y
sample_phi = generate_polynomial_features(sample_x)

gt_data = DataSet(gt_x_path, gt_y_path)
gt_x, gt_y = gt_data.x, gt_data.y
gt_phi = generate_polynomial_features(gt_x)

def deploy_least_square():
    title = 'LEAST SQUARES'
    log = open(result_sub_path + title + '.txt', 'w')
    img_path = result_sub_path + title + '.png'

    least_square = LeastSquares()
    least_square.fit(sample_phi, sample_y)

    pred_y = least_square.predict(gt_phi)

    error = compute_mean_squared_error(pred_y, gt_y)
    log.write("MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)

def deploy_regularized_least_squares():
    title = 'REGULARIZED LEAST SQUARES'
    log = open(result_sub_path + title + '.txt', 'w')
    img_path = result_sub_path + title + '.png'

    candidate_pred_y, candidate_error, candidate_lambda = [],[],[]
    _lambda = 0.000001
    _lambda_step = 1.05
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(1000):
        regularized_least_square = RegularizedLeastSquares()
        regularized_least_square.fit(sample_phi, sample_y, _lambda)

        pred_y = regularized_least_square.predict(gt_phi)

        error = compute_mean_squared_error(pred_y, gt_y)
        candidate_error.append((error, candidate_id))
        candidate_lambda.append(_lambda)
        candidate_pred_y.append(pred_y)

        _lambda *= _lambda_step

    log.write(str(_lambda) + '\n')
    candidate_error.sort()

    (error, best_id) = candidate_error[0]
    pred_y = candidate_pred_y[best_id]
    _lambda = candidate_lambda[best_id]

    log.write("lambda: " + str(_lambda) + '\n')
    log.write("MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)

def deploy_lasso():
    title = 'LASSO'
    log = open(result_sub_path + title + '.txt', 'w')
    img_path = result_sub_path + title + '.png'

    candidate_pred_y, candidate_error, candidate_lambda = [],[],[]
    _lambda = 0.000001
    _lambda_step = 1.05
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(1000):
        lasso = Lasso()
        lasso.fit(sample_phi, sample_y, _lambda)

        pred_y = lasso.predict(gt_phi)

        error = compute_mean_squared_error(pred_y, gt_y)
        candidate_error.append((error, candidate_id))
        candidate_lambda.append(_lambda)
        candidate_pred_y.append(pred_y)

        _lambda *= _lambda_step

    log.write(str(_lambda) + '\n')
    candidate_error.sort()

    (error, best_id) = candidate_error[0]
    pred_y = candidate_pred_y[best_id]
    _lambda = candidate_lambda[best_id]

    log.write("lambda: " + str(_lambda) + '\n')
    log.write("MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)

def deploy_robust_regression():
    title = 'ROBUST REGRESSION'
    log = open(result_sub_path + title + '.txt', 'w')
    img_path = result_sub_path + title + '.png'

    robust_regression = RobustRegression()
    robust_regression.fit(sample_phi, sample_y)

    pred_y = robust_regression.predict(gt_phi)

    error = compute_mean_squared_error(pred_y, gt_y)
    log.write("MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)


def bayesian_regression():
    title = 'Bayesian Regression'
    log = open(result_sub_path + title + '.txt', 'w')
    img_path = result_sub_path + title + '.png'

    candidate_pred_y, candidate_pred_std, candidate_error, candidate_alpha = [],[],[], []
    _alpha = 0.00001
    _alpha_step = 1.05
    log.write("range of alpha: " + str(_alpha) + ' : ')

    for candidate_id in range(1000):
        bayesian_regression = BayesianRegression()
        bayesian_regression.fit(sample_phi, sample_y, _alpha)

        pred_y, pred_std = bayesian_regression.predict(gt_phi)

        error = compute_mean_squared_error(pred_y, gt_y)
        candidate_error.append((error, candidate_id))
        candidate_alpha.append(_alpha)
        candidate_pred_y.append(pred_y)
        candidate_pred_std.append(pred_std)

        _alpha *= _alpha_step

    log.write(str(_alpha) + '\n')
    candidate_error.sort()

    (error, best_id) = candidate_error[0]
    pred_y = candidate_pred_y[best_id]
    print (pred_y.shape)
    pred_std = candidate_pred_std[best_id]

    print (pred_std.shape)
    print (pred_std)
    _alpha = candidate_alpha[best_id]

    log.write("alpha: " + str(_alpha) + '\n')
    log.write("MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path, pred_std)

#========question: how to compute standard deviation here????



# deploy_least_square()
# deploy_regularized_least_squares()
# deploy_lasso()
# deploy_robust_regression()
bayesian_regression()