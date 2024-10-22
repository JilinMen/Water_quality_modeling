# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:12:45 2024

@author: jmen
"""
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import joblib

# ... (保留之前的 read_file, exponential_func, fit_models, plot_results, print_formulas 函数)
def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel file.")

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def fit_models(X, y):
    models = {
        'Linear': LinearRegression(),
        'Polynomial (degree 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Exponential': None,  # We'll implement this separately
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
        'SVM': SVR(kernel='rbf'),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'XGBoost': XGBRegressor(n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        if name != 'Exponential':
            model.fit(X, y)
            y_pred = model.predict(X)
            # Save the model
            joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.joblib')
        else:
            # Exponential model: y = a * exp(b * x)
            popt, _ = curve_fit(exponential_func, X.ravel(), y)
            y_pred = exponential_func(X.ravel(), *popt)
            # Save the parameters
            np.save('exponential_params.npy', popt)
        
        results[name] = {
            'R2': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAPE': np.mean(np.abs((y - y_pred) / y)) * 100
        }
    
    return results, models

def plot_results(X, y, results, models):
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, color='blue', label='Data')
    
    for name, model in models.items():
        if name != 'Exponential':
            y_pred = model.predict(X)
        else:
            popt = np.load('exponential_params.npy')
            y_pred = exponential_func(X.ravel(), *popt)
        
        plt.plot(X, y_pred, label=f'{name} fit')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Model Fitting Comparison')
    plt.savefig('model_comparison.png')
    plt.close()

def print_formulas(X, y, models):
    # Linear model
    linear_model = models['Linear']
    print(f"Linear model: y = {linear_model.intercept_:.4f} + {linear_model.coef_[0]:.4f}x")
    
    # Polynomial models
    for degree in [2, 3]:
        poly_model = models[f'Polynomial (degree {degree})']
        coeffs = poly_model.named_steps['linearregression'].coef_
        intercept = poly_model.named_steps['linearregression'].intercept_
        formula = f"y = {intercept:.4f}"
        for i, coef in enumerate(coeffs):
            if i == 0:
                formula += f" + {coef:.4f}x"
            else:
                formula += f" + {coef:.4f}x^{i+1}"
        print(f"Polynomial (degree {degree}) model: {formula}")
    
    # Exponential model
    popt = np.load('exponential_params.npy')
    print(f"Exponential model: y = {popt[0]:.4f} * exp({popt[1]:.4f}x)")
    
def read_raster_data(file_path):
    if file_path.endswith('.nc'):
        # 读取netCDF文件
        ds = xr.open_dataset(file_path)
        data = ds.to_array().values
        # 假设第一个维度是波段，我们需要将其移到最后
        data = np.moveaxis(data, 0, -1)
        return data, ds
    elif file_path.endswith('.tif'):
        # 读取多个tif文件
        files = [file_path] if isinstance(file_path, str) else file_path
        data_list = []
        for file in files:
            with rasterio.open(file) as src:
                data_list.append(src.read(1))
        data = np.dstack(data_list)
        return data, src
    else:
        raise ValueError("Unsupported file format. Please use netCDF or TIF file.")

def apply_model(model, data):
    original_shape = data.shape
    # 重塑数据为2D数组，每行代表一个像素，每列代表一个波段
    data_2d = data.reshape(-1, data.shape[-1])
    
    if isinstance(model, (LinearRegression, SVR, RandomForestRegressor, XGBRegressor)):
        predictions = model.predict(data_2d)
    elif isinstance(model, MLPRegressor):
        predictions = model.predict(data_2d)
    else:  # Polynomial models
        predictions = model.predict(data_2d)
    
    # 将预测结果重塑回原始形状
    return predictions.reshape(original_shape[:-1])

def save_results(predictions, output_file, reference_data):
    if output_file.endswith('.nc'):
        # 保存为netCDF
        if isinstance(reference_data, xr.Dataset):
            # 创建一个新的DataArray
            da = xr.DataArray(predictions, dims=reference_data.dims[1:], 
                              coords={dim: reference_data[dim] for dim in reference_data.dims[1:]})
            da.to_netcdf(output_file)
        else:
            raise ValueError("Reference data must be an xarray Dataset for netCDF output")
    elif output_file.endswith('.tif'):
        # 保存为GeoTIFF
        if isinstance(reference_data, rasterio.DatasetReader):
            with rasterio.open(output_file, 'w', driver='GTiff',
                               height=predictions.shape[0], width=predictions.shape[1],
                               count=1, dtype=predictions.dtype,
                               crs=reference_data.crs, transform=reference_data.transform) as dst:
                dst.write(predictions, 1)
        else:
            raise ValueError("Reference data must be a rasterio DatasetReader for TIF output")
    else:
        raise ValueError("Unsupported output format. Please use netCDF or TIF.")

def main():
    # 训练模型部分
    train_file = input("请输入训练数据文件路径: ")
    x_column = input("请输入自变量列名: ")
    y_column = input("请输入因变量列名: ")
    
    df = read_file(train_file)
    X = df[[x_column]].values
    y = df[y_column].values
    
    results, models = fit_models(X, y)
    
    print("\n模型评估结果:")
    for name, metrics in results.items():
        print(f"\n{name} 模型:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    plot_results(X, y, results, models)
    print("\n模型比较图已保存为 'model_comparison.png'")
    
    print("\n模型公式:")
    print_formulas(X, y, models)
    
    # 应用模型部分
    input_file = input("请输入要应用模型的遥感数据文件路径 (如果是多个tif文件，请用逗号分隔): ")
    if ',' in input_file:
        input_files = [f.strip() for f in input_file.split(',')]
    else:
        input_files = input_file

    data, reference = read_raster_data(input_files)
    
    model_choice = input("请选择要应用的模型 (Linear, Polynomial2, Polynomial3, Exponential, NeuralNetwork, SVM, RandomForest, XGBoost): ")
    if model_choice == 'Exponential':
        popt = np.load('exponential_params.npy')
        predictions = exponential_func(data, *popt)
    else:
        model = joblib.load(f'{model_choice.lower()}_model.joblib')
        predictions = apply_model(model, data)
    
    output_file = input("请输入输出文件路径 (.nc 或 .tif): ")
    save_results(predictions, output_file, reference)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()