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

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    """Initialize weights from a normal distribution N(0, 0.01).

    Parameters
    ----------
    n_features : int
        Number of weights to initialize.
    seed : int or None, optional
        Random seed for reproducibility. If None, no reseeding is performed.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Randomly initialized weight vector.
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(loc=0.0, scale=0.01, size=n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    y_pred = predict_linear(X, weights)
    gradient = mse_gradient(X, y, y_pred)

    return weights - lr * gradient

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    train_pred = predict_linear(X_train, weights)
    val_pred = predict_linear(X_val, weights)

    train_loss = mse_loss(y_train, train_pred)
    val_loss = mse_loss(y_val, val_pred)

    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(
    val_loss, best_val_loss, wait, weights, best_weights, patience
):
    """Update early-stopping state after a validation evaluation.

    Parameters
    ----------
    val_loss : float
        Current validation loss.
    best_val_loss : float
        Best validation loss seen so far.
    wait : int
        Number of consecutive evaluations without improvement.
    weights : np.ndarray
        Current model weights.
    best_weights : np.ndarray
        Best model weights seen so far.
    patience : int
        Number of consecutive non-improving evaluations allowed.

    Returns
    -------
    best_val_loss : float
        Updated best validation loss.
    wait : int
        Updated wait counter.
    best_weights : np.ndarray
        Updated best weights.
    stop : bool
        Whether early stopping should occur.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_weights = weights.copy()
        stop = False
    else:
        wait += 1
        stop = wait >= patience

    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    """Build the initial training-state dictionary for batch GD.

    Parameters
    ----------
    n_features : int
        Number of model weights.
    seed : int or None, optional
        Random seed used to initialize the weights.

    Returns
    -------
    dict
        Initial training state containing weights, early-stopping state,
        loss histories, and stopped flag.
    """
    weights = initialize_weights(n_features, seed=seed)

    return {
        "weights": weights,
        "best_weights": weights.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False,
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # Take one gradient descent step
    state["weights"] = gd_step(
        X_train, y_train, state["weights"], lr
    )

    # Compute train and validation losses
    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state["weights"]
    )

    # Log losses
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)

    # Update early-stopping state
    (
        state["best_val_loss"],
        state["wait"],
        state["best_weights"],
        state["stopped"],
    ) = update_early_stop_state(
        val_loss,
        state["best_val_loss"],
        state["wait"],
        state["weights"],
        state["best_weights"],
        patience,
    )

    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    """Train weights with full-batch GD and early stopping.

    Parameters
    ----------
    X_train : np.ndarray
        Training design matrix.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Validation design matrix.
    y_val : np.ndarray
        Validation targets.
    lr : float
        Learning rate.
    epochs : int
        Maximum number of training epochs.
    patience : int
        Number of consecutive non-improving validation epochs allowed.
    seed : int or None
        Random seed used only for weight initialization.

    Returns
    -------
    weights : np.ndarray
        Best weights found during training.
    train_losses : list
        Training MSE history.
    val_losses : list
        Validation MSE history.
    """
    state = init_training_state(X_train.shape[1], seed=seed)

    for _ in range(epochs):
        state = run_one_epoch(
            state,
            X_train,
            y_train,
            X_val,
            y_val,
            lr,
            patience,
        )

        if state["stopped"]:
            break

    return (
        state["best_weights"],
        state["train_losses"],
        state["val_losses"],
    )

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    """Compute the mean absolute error between true targets and predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Mean absolute error as a scalar float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Return the root mean squared error between y_true and y_pred.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Root mean squared error as a scalar float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    """Compute the coefficient of determination R^2.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        R^2 score as a scalar float.
        Returns nan when the total sum of squares is zero.
    """
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)

    if total_sum_of_squares == 0:
        return float("nan")

    return float(
        1.0 - residual_sum_of_squares / total_sum_of_squares
    )

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    """Bundle MAE, RMSE, and R^2 into a metrics dictionary.

    Args:
        y_true: True target values of shape (n,).
        y_pred: Predicted target values of shape (n,).

    Returns:
        Dictionary containing 'mae', 'rmse', and 'r2' metrics.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    """Return epoch indices and loss series for external plotting.

    Args:
        train_losses: Training loss sequence.
        val_losses: Validation loss sequence.

    Returns:
        Tuple of (epochs, train, val) as plain Python lists.
    """
    epochs = list(range(1, len(train_losses) + 1))
    train = np.asarray(train_losses).tolist()
    val = np.asarray(val_losses).tolist()

    return epochs, train, val

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    """Compute the L2 distance between two weight vectors.

    Args:
        w_gd: Weight vector obtained from gradient descent.
        w_closed: Weight vector obtained from the closed-form solution.

    Returns:
        Euclidean distance as a scalar float.
    """
    return float(np.linalg.norm(w_gd - w_closed))

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    """Build the initial LinearRegressionGD-style model dictionary.

    Parameters
    ----------
    learning_rate : float, optional
        Gradient descent learning rate.
    epochs : int, optional
        Maximum number of training epochs.
    patience : int, optional
        Early-stopping patience.
    seed : int, optional
        Random seed for weight initialization.

    Returns
    -------
    dict
        Unfitted linear regression model state.
    """
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "weights": None,
        "normal_weights": None,
        "mean": None,
        "std": None,
        "train_losses": [],
        "val_losses": [],
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    """Fit model with train stats, design matrices, GD, and normal eq.

    Parameters
    ----------
    model : dict
        Linear regression model dictionary containing hyperparameters.
    X_train : np.ndarray
        Raw training feature matrix.
    y_train : np.ndarray
        Training targets.
    X_val : np.ndarray
        Raw validation feature matrix.
    y_val : np.ndarray
        Validation targets.

    Returns
    -------
    dict
        The same model dictionary, updated with fitted parameters.
    """
    # Compute feature statistics using training data only
    model["mean"], model["std"] = compute_feature_stats(X_train)

    # Standardize features and add bias column
    X_train_design = prepare_design_matrix(
        X_train, model["mean"], model["std"]
    )
    X_val_design = prepare_design_matrix(
        X_val, model["mean"], model["std"]
    )

    # Train with batch gradient descent and early stopping
    (
        model["weights"],
        model["train_losses"],
        model["val_losses"],
    ) = train_batch_gd(
        X_train_design,
        y_train,
        X_val_design,
        y_val,
        lr=model["learning_rate"],
        epochs=model["epochs"],
        patience=model["patience"],
        seed=model["seed"],
    )

    # Solve the closed-form normal equation.
    # If X^T X is singular, use least squares as a robust fallback.
    try:
        model["normal_weights"] = normal_equation(
            X_train_design, y_train
        )
    except np.linalg.LinAlgError:
        model["normal_weights"] = np.linalg.lstsq(
            X_train_design, y_train, rcond=None
        )[0]

    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    """Return predicted targets for raw X using the fitted model.

    Parameters
    ----------
    model : dict
        Fitted linear regression model containing 'mean', 'std', and 'weights'.
    X : np.ndarray, shape (n, d)
        Raw feature matrix.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted target values.
    """
    X_design = prepare_design_matrix(
        X, model["mean"], model["std"]
    )

    return predict_linear(X_design, model["weights"])

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

