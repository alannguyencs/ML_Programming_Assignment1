from params import *
from least_squares import LeastSquares
from regularized_least_squares import RegularizedLeastSquares
# from lasso import Lasso
from robust_regression import RobustRegression
from bayesian_regression import BayesianRegression
from dataset import DataSet
from common import *

result_sub_path = result_path + 'part1c/'
#get data
gt_data = DataSet(gt_x_path, gt_y_path)
gt_x, gt_y = gt_data.x, gt_data.y
gt_phi = generate_polynomial_features(gt_x)

sample_percent = [10, 25, 50, 75]
num_sub_sample = 5

sample_data_list = []
for i in range(4):
    for j in range(num_sub_sample):
        sample_data = DataSet(sample_x_path, sample_y_path, percent_sample=sample_percent[i])
        sample_data_list.append(sample_data)

def deploy_least_square(sample_id):
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    title = 'LEAST SQUARES'
    log = open(result_sub_path + title + '_' + str(sample_id) + '.txt', 'w')
    img_path = result_sub_path + title + '_' + str(sample_id) + '.png'

    model = LeastSquares()
    model.fit(sample_phi, sample_y)

    pred_sample_y = model.predict(sample_phi)
    pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)
    return sample_error, error

def deploy_regularized_least_squares(sample_id):
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    title = 'REGULARIZED LEAST SQUARES'
    log = open(result_sub_path + title + '_' + str(sample_id) + '.txt', 'w')
    img_path = result_sub_path + title + '_' + str(sample_id) + '.png'

    candidate_pred_y, candidate_error, candidate_sample_error, candidate_lambda = [],[],[], []
    _lambda = 0.000001
    _lambda_step = 1.05
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(1000):
        model = RegularizedLeastSquares()
        model.fit(sample_phi, sample_y, _lambda)

        pred_sample_y = model.predict(sample_phi)
        pred_y = model.predict(gt_phi)

        sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
        error = compute_mean_squared_error(pred_y, gt_y)

        candidate_sample_error.append((sample_error, candidate_id))
        candidate_error.append(error)
        candidate_lambda.append(_lambda)
        candidate_pred_y.append(pred_y)

        _lambda *= _lambda_step

    log.write(str(_lambda) + '\n')
    candidate_error.sort()

    (sample_error, best_id) = candidate_sample_error[0]
    error = candidate_error[best_id]
    pred_y = candidate_pred_y[best_id]
    _lambda = candidate_lambda[best_id]

    log.write("lambda: " + str(_lambda) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)
    return sample_error, error

def deploy_lasso(sample_id):
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    title = 'LASSO'

    log = open(result_sub_path + title + '_' + str(sample_id) + '.txt', 'w')
    img_path = result_sub_path + title + '_' + str(sample_id) + '.png'

    candidate_pred_y, candidate_error, candidate_sample_error, candidate_lambda = [], [], [], []
    _lambda = 0.000001
    _lambda_step = 1.05
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(1000):
        model = Lasso()
        model.fit(sample_phi, sample_y, _lambda)

        pred_sample_y = model.predict(sample_phi)
        pred_y = model.predict(gt_phi)

        sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
        error = compute_mean_squared_error(pred_y, gt_y)

        candidate_sample_error.append((sample_error, candidate_id))
        candidate_error.append(error)
        candidate_lambda.append(_lambda)
        candidate_pred_y.append(pred_y)

        _lambda *= _lambda_step

    log.write(str(_lambda) + '\n')
    candidate_error.sort()

    (sample_error, best_id) = candidate_sample_error[0]
    error = candidate_error[best_id]
    pred_y = candidate_pred_y[best_id]
    _lambda = candidate_lambda[best_id]

    log.write("lambda: " + str(_lambda) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)
    return sample_error, error

def deploy_robust_regression(sample_id):
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    title = 'ROBUST REGRESSION'
    log = open(result_sub_path + title + '_' + str(sample_id) + '.txt', 'w')
    img_path = result_sub_path + title + '_' + str(sample_id) + '.png'

    model = RobustRegression()
    model.fit(sample_phi, sample_y)

    pred_sample_y = model.predict(sample_phi)
    pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path)
    return sample_error, error


def bayesian_regression(sample_id):
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    title = 'BAYESIAN REGRESSION'
    log = open(result_sub_path + title + '_' + str(sample_id) + '.txt', 'w')
    img_path = result_sub_path + title + '_' + str(sample_id) + '.png'

    candidate_pred_y, candidate_pred_std, candidate_error, \
                    candidate_sample_error, candidate_alpha = [],[],[], [], []
    _alpha = 0.00001
    _alpha_step = 1.05
    log.write("range of alpha: " + str(_alpha) + ' : ')

    for candidate_id in range(1000):
        model = BayesianRegression()
        model.fit(sample_phi, sample_y, _alpha)

        pred_sample_y, pred_sample_std = model.predict(sample_phi)
        pred_y, pred_std = model.predict(gt_phi)

        sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
        error = compute_mean_squared_error(pred_y, gt_y)

        candidate_sample_error.append((sample_error, candidate_id))
        candidate_error.append(error)
        candidate_alpha.append(_alpha)
        candidate_pred_y.append(pred_y)
        candidate_pred_std.append(pred_std)

        _alpha *= _alpha_step

    log.write(str(_alpha) + '\n')
    candidate_sample_error.sort()

    (sample_error, best_id) = candidate_sample_error[0]
    error = candidate_error[best_id]
    pred_y = candidate_pred_y[best_id]
    pred_std = candidate_pred_std[best_id]
    _alpha = candidate_alpha[best_id]

    log.write("alpha: " + str(_alpha) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')


    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, title, img_path, pred_std)
    return sample_error, error

#================================================================================
summary_file = open(result_sub_path + 'summary.txt', 'w')
sample_error_list = [[] for _ in range(5)]
error_list = [[] for _ in range(5)]
for sample_id in range(4 * num_sub_sample):
    sample_error, error = deploy_least_square(sample_id)
    sample_error_list[0].append(sample_error)
    error_list[0].append(error)

    sample_error, error = deploy_regularized_least_squares(sample_id)
    sample_error_list[1].append(sample_error)
    error_list[1].append(error)

    sample_error, error = deploy_lasso(sample_id)
    sample_error_list[2].append(sample_error)
    error_list[2].append(error)

    sample_error, error = deploy_robust_regression(sample_id)
    sample_error_list[3].append(sample_error)
    error_list[3].append(error)

    sample_error, error = bayesian_regression(sample_id)
    sample_error_list[4].append(sample_error)
    error_list[4].append(error)

for i in range(1):
    for j in range(4):
        s1 = sum(sample_error_list[i][(j*num_sub_sample):((j+1)*num_sub_sample)])
        s1 /= num_sub_sample

        s2 = sum(error_list[i][(j * num_sub_sample):((j + 1) * num_sub_sample)])
        s2 /= num_sub_sample

        summary_file.write(str(s1) + ' ' + str(s2) + '\n')
    summary_file.write('-----------------------\n')