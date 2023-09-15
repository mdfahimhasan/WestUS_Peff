import numpy as np
import pandas as pd
from Codes.utils.ML_utils import calculate_rmse
from scipy.optimize import minimize


def create_input_LM(predictor_csv, observed_csv):
    x_df = pd.read_csv(predictor_csv)
    y_df = pd.read_csv(observed_csv)

    y_df['fips'] = [int(str(i)[:-4]) for i in y_df['fips_years']]
    y_df['Year'] = [int(str(i)[-4:]) for i in y_df['fips_years']]

    obsv_df = y_df[y_df['Year'] == 3020]
    x_df = x_df[['GPM_PRECIP', 'GPW_Pop', 'MODIS_Day_LST', 'SSEBOP_ET', 'USDA_cropland', 'USDA_developed', 'fips']]

    variables_arr = x_df.values[:, :-1]
    fips_arr = x_df.values[:, -1:]

    return variables_arr, fips_arr, obsv_df


# Training data load
train_data = '../../Data_main/Model_csv/train_data.csv'
train_obsv= '../../Data_main/Model_csv/train_obsv.csv'

variables_arr, fips_arr, obsv_df = create_input_LM(train_data, train_obsv)


def calc_gw(param_space):
    a, b, c, d, e, f, intercept = param_space

    y_pixel = a * variables_arr[:, 0:1] + b * variables_arr[:, 1:2] + c * variables_arr[:, 2:3] + \
              d * variables_arr[:, 3:4] + e * variables_arr[:, 4:5] + f * variables_arr[:, 5:6] + intercept

    y_pixel_fips = np.hstack((fips_arr, y_pixel))

    y_pixel_df = pd.DataFrame(y_pixel_fips, columns=['fips', 'y_pixel'])
    y_sum_df = y_pixel_df.groupby(by='fips')['y_pixel'].sum().reset_index()
    y_sum_df = y_sum_df.sort_values(by=['fips'], ascending=True)
    y_sum_df = y_sum_df.merge(obsv_df, on='fips', how='left')
    y_sum_df = y_sum_df.drop(columns=['fips_years', 'Year'])

    return y_sum_df


def objective_func(param_space):
    y_sum_df = calc_gw(param_space)
    rmse_val = calculate_rmse(Y_pred=y_sum_df['y_pixel'], Y_obsv=y_sum_df['total_gw_observed'])

    return rmse_val


# Initial guesses
x0 = np.zeros(7)
# Bounds on parameters
bnds = ((-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-50, 50))
# Optimize
solution = minimize(objective_func, x0, method='Nelder-Mead', bounds=bnds)
params = solution.x
y_county = calc_gw(params)
a, b, c, d, e, f, interp = params

# Final Objective
print(f'Final RMSE: {objective_func(param_space=params)}')

# Print solution
print('Solution')
print(f'a = {a}')
print(f'b = {b}')
print(f'c = {c}')
print(f'd = {d}')
print(f'e = {e}')
print(f'f = {f}')
print(f'intercept = {interp}')











