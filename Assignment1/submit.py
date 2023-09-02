import numpy as np
from sklearn.linear_model import (
    SGDClassifier,
)
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length


def my_func(x):
    return 8 * x[0] + 4 * x[1] + 2 * x[2] + x[3]


def prep_data(data):  # Text file data converted to integer data type
    File_data = data
    # Specify the index of the column to insert
    i = 63

    # Get the last column of the array
    last_col = File_data[:, -1]

    # Remove the last column from the array
    arr_without_last_col = File_data[:, :-1]

    # Insert the last column between columns i and i+1
    new_arr = np.hstack(
        (
            arr_without_last_col[:, : i + 1],
            last_col.reshape(-1, 1),
            arr_without_last_col[:, i + 1 :],
        )
    )

    new_col = np.apply_along_axis(my_func, axis=1, arr=new_arr[:, [65, 66, 67, 68]])

    new_col2 = np.apply_along_axis(my_func, axis=1, arr=new_arr[:, [69, 70, 71, 72]])

    # Delete the original columns from the array
    new_arr = np.delete(new_arr, [65, 66, 67, 68, 69, 70, 71, 72], axis=1)

    new_arr = np.hstack((new_arr, new_col.reshape(-1, 1)))

    new_arr = np.hstack((new_arr, new_col2.reshape(-1, 1)))

    return new_arr


def prep_tst_data(data):  # Text file data converted to integer data type

    File_data = data

    new_col = np.apply_along_axis(my_func, axis=1, arr=File_data[:, [64, 65, 66, 67]])

    new_col2 = np.apply_along_axis(my_func, axis=1, arr=File_data[:, [68, 69, 70, 71]])

    # Delete the original columns from the array
    new_arr = np.delete(File_data, [64, 65, 66, 67, 68, 69, 70, 71], axis=1)

    new_arr = np.hstack((new_arr, new_col.reshape(-1, 1)))

    new_arr = np.hstack((new_arr, new_col2.reshape(-1, 1)))

    return new_arr


################################
# Non Editable Region Starting #
################################
def my_fit(Z_train):
    ################################
    #  Non Editable Region Ending  #
    ################################
    models = {}
    new_arr = prep_data(Z_train)
    # print(len(new_arr[0]))
    unique_vals = np.unique(new_arr[:, [65, 66]], axis=0)
    x = new_arr[:, :64]
    y = new_arr[:, 64]

    # Train a model for each unique combination of values
    for vals in unique_vals:
        # Select rows with the current combination of values
        rows = np.all(new_arr[:, [65, 66]] == vals, axis=1)

        # Train a linear regression model
        model = SGDClassifier(alpha=0.001).fit(x[rows], y[rows])

        # Store the model in a dictionary with the current combination of values as key
        models[tuple(vals)] = model

    # Use this method to train your model using training CRPs
    # The first 64 columns contain the config bits
    # The next 4 columns contain the select bits for the first mux
    # The next 4 columns contain the select bits for the second mux
    # The first 64 + 4 + 4 = 72 columns constitute the challenge
    # The last column contains the response

    return models  # Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, models):
    ################################
    #  Non Editable Region Ending  #
    ################################
    new_examples = prep_tst_data(X_tst)
    x_test = new_examples[:, :64]

    # Get unique combinations of values
    vals, inv = np.unique(new_examples[:, [64, 65]], axis=0, return_inverse=True)

    # Initialize output array
    all_pred = np.zeros(len(X_tst))

    # Iterate over unique combinations of values
    for i, v in enumerate(vals):
        # Select rows with current combination of values
        rows = inv == i
        # Get model for current combination of values
        model = models[tuple(v)]
        # Predict for selected rows
        y_pred = model.predict(x_test[rows])
        # Store predictions in output array
        all_pred[rows] = y_pred

    return all_pred
