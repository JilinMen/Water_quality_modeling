# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:52:10 2024

@author: jmen
"""
import pandas as pd
import numpy as np
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

def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel file.")

def exponential_func(x, a0, a1, a2, a3):
    return 10**(a0 + a1 * np.log10(x) + a2 * np.log10(x)**2 + a3 * np.log10(x)**3)

def fit_models(X, y):
    models = {
        'Linear': LinearRegression(),
        'Polynomial (degree 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Exponential': None,  # We'll implement this separately
        'Neural Network': MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000),
        'SVM': SVR(kernel='rbf'),
        'Random Forest': RandomForestRegressor(n_estimators=10),
        'XGBoost': XGBRegressor(n_estimators=10)
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
            try:
                # 使用对数变换来提高数值稳定性
                log_y = np.log10(y)
                log_x = np.log10(X.ravel())
                
                # 使用多项式拟合对数变换后的数据
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(log_x.reshape(-1, 1))
                linear_model = LinearRegression()
                linear_model.fit(X_poly, log_y)
                # 提取系数作为初始猜测
                initial_guess = [linear_model.intercept_] + list(linear_model.coef_[1:])
                
                popt, _ = curve_fit(exponential_func, X.ravel(), y, p0=initial_guess, maxfev=10000)
                y_pred = exponential_func(X.ravel(), *popt)
                # Save the parameters
                np.save('exponential_params.npy', popt)
            except RuntimeError:
                print("警告：指数模型拟合失败。可能需要调整初始参数或使用其他拟合方法。")
                continue
        
        results[name] = {
            'R2': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAPE': np.mean(np.abs((y - y_pred) / y)) * 100
        }
    
    return results, models

def plot_results(results):
    metrics = ['R2', 'RMSE', 'MAPE']
    n_metrics = len(metrics)
    n_models = len(results)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    bar_width = 0.8
    index = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        axes[i].bar(index, values, bar_width, label=metric)
        axes[i].set_title(metric)
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(results.keys(), rotation=45, ha='right')
        axes[i].set_xlabel('Models')
        
        # 设置 y 轴范围和标签
        if metric == 'R2':
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel('R² Score')
        elif metric == 'RMSE':
            max_rmse = max(values) * 1.1  # 给顶部留些空间
            axes[i].set_ylim(0, max_rmse)
            axes[i].set_ylabel('RMSE')
        else:  # MAPE
            max_mape = max(values) * 1.1  # 给顶部留些空间
            axes[i].set_ylim(0, max_mape)
            axes[i].set_ylabel('MAPE (%)')
        
        # 在柱子上方添加数值标签
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
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

def main():
    # input
    file_path = r"C:\Users\jmen\Box\ERSL_FieldDatabase\LakeHarsha\Matchups_Rrs_wq.xlsx"
    B = "Rrs_482"
    G = "Rrs_561"
    R = "Rrs_654"
    
    # output
    y_column = "Chl ug/L"
    
    df = read_file(file_path)
    
    Blue = df[[B]].values
    Green = df[[G]].values
    Red = df[[R]].values
    
    y = df[y_column].values
    
    # CI = Green - (Blue + (561 - 482)/(654 - 482) * (Red - Blue))
    
    CI = Blue / Green
    
    results, models = fit_models(CI, y)
    
    print("\n模型评估结果:")
    for name, metrics in results.items():
        print(f"\n{name} 模型:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    plot_results(results)
    print("\n模型比较图已保存为 'model_performance_comparison.png'")
    
    print("\n模型公式:")
    print_formulas(CI, y, models)
    
    print("\n所有模型已保存。")

if __name__ == "__main__":
    main()
