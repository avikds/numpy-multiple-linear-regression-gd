# NumPy Multiple Linear Regression GD

Build a from-scratch multiple linear regression trainer in pure NumPy: standardize features, minimize MSE with batch gradient descent and early stopping, compare against the normal equation, and report MAE, RMSE, and R-squared via a reusable model API.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** shuffle_xy
- [x] **2.** split_train_val_test
- [x] **3.** compute_feature_stats
- [x] **4.** standardize_features
- [x] **5.** add_bias_column
- [x] **6.** prepare_design_matrix
- [x] **7.** predict_linear
- [x] **8.** mse_loss
- [x] **9.** mse_gradient
- [x] **10.** normal_equation
- [x] **11.** initialize_weights
- [x] **12.** gd_step
- [x] **13.** epoch_train_val_losses
- [x] **14.** update_early_stop_state
- [x] **15.** init_training_state
- [x] **16.** run_one_epoch
- [x] **17.** train_batch_gd
- [x] **18.** mean_absolute_error
- [x] **19.** root_mean_squared_error
- [x] **20.** r_squared
- [x] **21.** evaluate_regression
- [x] **22.** learning_curve_data
- [x] **23.** weights_l2_distance
- [x] **24.** create_lr_model
- [x] **25.** fit_lr_model
- [x] **26.** predict_lr_model
- [x] **27.** score_lr_model
- [x] **28.** compare_with_normal_equation

## Output

```text

Splits: 90 30 30

Sample preds: [-4.9349 -2.1614 -1.1088 -3.5913  1.7047]

Sample trues: [-4.8688 -2.2391 -0.8235 -3.5582  1.6763]

Test MAE/RMSE/R2: {'mae': 0.09220147681923264, 'rmse': 0.12202512653831676, 'r2': 0.9985479787094949}

GD vs normal-eq L2 gap: 0.005860423059766929

Final train/val MSE: 0.010137750731524809 0.012253596422242757

Epochs run: 81

```

## Results

| Metric | Value |
|---|---:|
| Train samples | 90 |
| Validation samples | 30 |
| Test samples | 30 |
| Test MAE | 0.092201 |
| Test RMSE | 0.122025 |
| Test R² | 0.998548 |
| GD vs Normal Equation L2 Gap | 0.005860 |
| Final Train MSE | 0.010138 |
| Final Validation MSE | 0.012254 |
| Epochs Run | 81 |
