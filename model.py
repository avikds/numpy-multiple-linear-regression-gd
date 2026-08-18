"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
        Shuffled feature matrix.
    y_shuffled : np.ndarray, shape (n,)
        Shuffled target vector.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(X.shape[0])

    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    return X_shuffled, y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    """Split already-shuffled data into train, validation, and test sets.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    train_frac : float, optional
        Fraction of data for training (default 0.6).
    val_frac : float, optional
        Fraction of data for validation (default 0.2).

    Returns
    -------
    X_train, y_train : training data
    X_val, y_val : validation data
    X_test, y_test : test data
    """
    n = X.shape[0]

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    """Compute per-feature mean and population standard deviation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Training feature matrix.

    Returns
    -------
    mean : np.ndarray, shape (d,)
        Per-feature means.
    std : np.ndarray, shape (d,)
        Per-feature population standard deviations, with zeros replaced by 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1.0, std)

    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    """Apply z-score normalization using precomputed feature statistics.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    mean : np.ndarray, shape (d,)
        Per-feature means computed from the training data.
    std : np.ndarray, shape (d,)
        Per-feature standard deviations computed from the training data.

    Returns
    -------
    np.ndarray, shape (n, d)
        Standardized feature matrix.
    """
    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    """Prepend a column of ones to the feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.

    Returns
    -------
    np.ndarray, shape (n, d + 1)
        Feature matrix with a leading column of ones.
    """
    bias = np.ones((X.shape[0], 1))
    return np.hstack((bias, X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    """Standardize features and prepend a bias column.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    mean : np.ndarray, shape (d,)
        Per-feature means computed from the training data.
    std : np.ndarray, shape (d,)
        Per-feature standard deviations computed from the training data.

    Returns
    -------
    np.ndarray, shape (n, d + 1)
        Standardized design matrix with a leading column of ones.
    """
    X_standardized = standardize_features(X, mean, std)
    return add_bias_column(X_standardized)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    """Compute mean squared error between true and predicted values.

    Args:
        y_true: True target values.
        y_pred: Model predictions.

    Returns:
        Mean squared error as a scalar float.
    """
    return float(np.mean((y_true - y_pred) ** 2))

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    """Compute the analytic gradient of MSE with respect to model weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y_true: True target values of shape (n,).
        y_pred: Model predictions of shape (n,).

    Returns:
        Gradient vector of shape (d_in,).
    """
    n = X.shape[0]
    return (2.0 / n) * (X.T @ (y_pred - y_true))

# Step 10 - normal_equation
def normal_equation(X, y):
    """Solve for the closed-form least-squares weights.

    Parameters
    ----------
    X : np.ndarray, shape (n, d_in)
        Design matrix.
    y : np.ndarray, shape (n,)
        Target vector.

    Returns
    -------
    np.ndarray, shape (d_in,)
        Least-squares weight vector.
    """
    XtX = X.T @ X
    Xty = X.T @ y

    return np.linalg.solve(XtX, Xty)

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

