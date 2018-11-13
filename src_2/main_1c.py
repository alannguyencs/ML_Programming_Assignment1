from params import *
from estimators import *
from dataset import DataSet
from common import *

result_sub_path = result_path + 'part1c/'
gt_data = DataSet(gt_x_path, gt_y_path)
gt_x, gt_y = gt_data.x, gt_data.y
gt_phi = generate_polynomial_features(gt_x)



def deploy_model(model, sample_phi, sample_y, _lambda=None):
    print ("--------------------------------------------------------")
    print (model.name)
    log = open(result_sub_path + model.name + '.txt', 'w')
    img_path = result_sub_path + model.name + '.png'

    if _lambda == None:
        model.fit(sample_phi, sample_y)
    else:
        model.fit(sample_phi, sample_y, _lambda)
        print ("Lambda:", _lambda)
    
    pred_std=None
    if model.name == "BayesianRegression":
        pred_sample_y, pred_sample_std = model.predict(sample_phi)
        pred_y, pred_std = model.predict(gt_phi)
    else:
        pred_sample_y = model.predict(sample_phi)
        pred_y = model.predict(gt_phi)

    sample_error = compute_mean_squared_error(pred_sample_y, sample_y)
    error = compute_mean_squared_error(pred_y, gt_y)

    print ("Training MSE: " + str(sample_error))
    print ("Testing MSE: " + str(error))
    log.write("Lambda: " + str(_lambda) + '\n')
    log.write("Training MSE: " + str(sample_error) + '\n')
    log.write("Testing MSE: " + str(error) + '\n')

    visualize(sample_x, sample_y, gt_x, gt_y, pred_y, model.name, img_path, pred_std)
    return sample_error, error


def estimate_best_lambda(model, val_sample_phi, val_sample_y, val_gt_phi, val_gt_y):

    def estimate_k_fold_error(_lambda):
        sample_error = 0

        if model.name == "RegularizedLeastSquares":
            for fold_id in range(num_fold):
                model.fit(val_sample_phi[fold_id].T, val_sample_y[fold_id], _lambda)
                pred_y = model.predict(val_gt_phi[fold_id].T)
                sample_error += compute_mean_squared_error(pred_y, val_gt_y[fold_id])

        return sample_error / num_fold

    val_error = []
    for _lambda in candidate_hyperparams:
        val_error.append(estimate_k_fold_error(_lambda=_lambda))

    return candidate_hyperparams[val_error.index(min(val_error))]

#========================================= MAIN ====================================================
summary_file = open(result_sub_path + 'summary.txt', 'w')
sample_error_list = [[] for _ in range(5)]
error_list = [[] for _ in range(5)]

sample_percent = [75, 50, 25, 15]
num_sub_sample = 5

sample_data_list = []
for i in range(4):
    for _ in range(num_sub_sample):
        sample_data = DataSet(sample_x_path, sample_y_path, percent_sample=sample_percent[i])
        sample_data_list.append(sample_data)

for sample_id in range(4 * num_sub_sample):
    print ("sample_id:", sample_id)
    sample_data = sample_data_list[sample_id]
    sample_x, sample_y = sample_data.x, sample_data.y
    sample_phi = generate_polynomial_features(sample_x)

    val_sample_phi, val_sample_y, val_gt_phi, val_gt_y \
                        = generate_validation_set(sample_phi, sample_y)

    #LEAST SQUARES
    sample_error, error = deploy_model(model=LeastSquares(), sample_phi=sample_phi, sample_y=sample_y)
    sample_error_list[0].append(sample_error)
    error_list[0].append(error)

    #REGULARIZED LEAST SQUARES
    best_lambda = estimate_best_lambda(RegularizedLeastSquares(), \
                val_sample_phi, val_sample_y, val_gt_phi, val_gt_y)
    sample_error, error = deploy_model(model=RegularizedLeastSquares(), sample_phi=sample_phi, sample_y=sample_y, _lambda=best_lambda)
    sample_error_list[1].append(sample_error)
    error_list[1].append(error)
    
    #LASSO
    best_lambda = estimate_best_lambda(Lasso(), \
                val_sample_phi, val_sample_y, val_gt_phi, val_gt_y)
    sample_error, error = deploy_model(model=Lasso(), sample_phi=sample_phi, sample_y=sample_y, _lambda=best_lambda)
    sample_error_list[2].append(sample_error)
    error_list[2].append(error)

    #ROBUST REGRESSION
    sample_error, error = deploy_model(model=RobustRegression(), sample_phi=sample_phi, sample_y=sample_y)
    sample_error_list[3].append(sample_error)
    error_list[3].append(error)

    #BAYESIAN REGRESSION
    best_lambda = estimate_best_lambda(BayesianRegression(), \
                val_sample_phi, val_sample_y, val_gt_phi, val_gt_y)
    sample_error, error = deploy_model(model=BayesianRegression(), sample_phi=sample_phi, sample_y=sample_y, _lambda=best_lambda)
    sample_error_list[4].append(sample_error)
    error_list[4].append(error)

for i in range(5):
    for j in range(4):
        s1 = sum(sample_error_list[i][(j*num_sub_sample):((j+1)*num_sub_sample)])
        s1 /= num_sub_sample

        s2 = sum(error_list[i][(j * num_sub_sample):((j + 1) * num_sub_sample)])
        s2 /= num_sub_sample

        summary_file.write(str(s1) + ' ' + str(s2) + '\n')
    summary_file.write('-----------------------\n')














