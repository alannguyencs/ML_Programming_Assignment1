from params import *
from least_squares import LeastSquares
from regularized_least_squares import RegularizedLeastSquares
from lasso import Lasso
from robust_regression import RobustRegression
from bayesian_regression import BayesianRegression
from dataset import DataSet
from common import *

result_sub_path = result_path + 'part2b/'
#get data
sample_data = DataSet(cnt_sample_x_path, cnt_sample_y_path)
sample_x, sample_y = sample_data.x, sample_data.y
sample_phi = generate_counting_features(sample_x)
# print (np.amax(sample_phi), np.amin(sample_phi))
print (sample_x.shape, sample_y.shape, sample_phi.shape)

gt_data = DataSet(cnt_gt_x_path, cnt_gt_y_path)
gt_x, gt_y = gt_data.x, gt_data.y
gt_phi = generate_counting_features(gt_x)
# print (np.amax(gt_phi), np.amin(gt_phi))

def deploy_least_square():
    title = 'LEAST SQUARES'
    log = open(result_sub_path + title + '_2.txt', 'w')
    img_path = result_sub_path + title + '_2.png'

    model = LeastSquares()
    model.fit(sample_phi, sample_y)

    pred_sample_y = model.predict(sample_phi)
    pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')

    visualize_people_counting(gt_y, pred_y, title, img_path)

def deploy_regularized_least_squares():
    title = 'REGULARIZED LEAST SQUARES'
    log = open(result_sub_path + title + '_2.txt', 'w')
    img_path = result_sub_path + title + '_2.png'

    candidate_pred_y, candidate_error, candidate_sample_error, candidate_lambda = [],[],[], []
    _lambda = 0.1
    _lambda_step = 1.28
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(10):
        model = RegularizedLeastSquares()
        model.fit(sample_phi, sample_y, _lambda)

        pred_sample_y = model.predict(sample_phi)
        pred_y = model.predict(gt_phi)

        sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
        error = compute_mean_squared_error(pred_y, gt_y)

        candidate_sample_error.append((sample_error, candidate_id))
        candidate_error.append((error, candidate_id))
        candidate_lambda.append(_lambda)
        candidate_pred_y.append(pred_y)

        _lambda *= _lambda_step

    log.write(str(_lambda) + '\n')
    candidate_error.sort()

    (error, best_id) = candidate_error[0]
    # error = candidate_error[best_id]
    pred_y = candidate_pred_y[best_id]
    good_lambda = candidate_lambda[best_id]
    print (candidate_lambda)
    print (best_id)
    print (candidate_sample_error)

    log.write("lambda: " + str(good_lambda) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')

    visualize_people_counting(gt_y, pred_y, title, img_path)

def deploy_lasso():
    title = 'LASSO'
    log = open(result_sub_path + title + '_2.txt', 'w')
    img_path = result_sub_path + title + '_2.png'

    candidate_pred_y, candidate_error, candidate_sample_error, candidate_lambda = [], [], [], []
    _lambda = 1.0
    _lambda_step = 1.05
    log.write("range of lambda: " + str(_lambda) + ' : ')

    for candidate_id in range(10):
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

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')


    visualize_people_counting(gt_y, pred_y, title, img_path)
#
def deploy_robust_regression():
    title = 'ROBUST REGRESSION'
    log = open(result_sub_path + title + '_2.txt', 'w')
    img_path = result_sub_path + title + '_2.png'

    model = RobustRegression()
    model.fit(sample_phi, sample_y)

    pred_sample_y = model.predict(sample_phi)
    pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')

    visualize_people_counting(gt_y, pred_y, title, img_path)


def bayesian_regression():
    title = 'BAYESIAN REGRESSION'
    log = open(result_sub_path + title + '_2.txt', 'w')
    img_path = result_sub_path + title + '_2.png'

    candidate_pred_y, candidate_pred_std, candidate_error, \
                    candidate_sample_error, candidate_alpha = [],[],[], [], []
    _alpha = 0.01
    _alpha_step = 1.05
    log.write("range of alpha: " + str(_alpha) + ' : ')

    for candidate_id in range(10):
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

    sample_error = compute_mean_absolute_error(pred_sample_y, sample_y)
    error = compute_mean_absolute_error(pred_y, gt_y)

    log.write("Training MAE: " + str(sample_error) + '\n')
    log.write("Testing MAE: " + str(error) + '\n')


    visualize_people_counting(gt_y, pred_y, title, img_path)

#
# #================================================================================
#
deploy_least_square()
deploy_regularized_least_squares()
deploy_lasso()
deploy_robust_regression()
bayesian_regression()