# ------------------------- UTILITY CONSTANTS ---------------------------- #
# Null Types for native data types
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Types Nulls
IntNull = 0
FloatNull = 0.00
StringNull = ''

ListNull = []
TupleNull = ()
SetNull: set = {None}
DictNull: dict = {None: None}
MatrixNull = []

# Types Aliases
Matrix = List[List[float]]
Curve = List[Tuple[float]]


# ------------------------- UTILITY FUNCTIONS ---------------------------- #

# --------------- NUMERICAL ANALYSIS AND SCIENCE OPERATIONS -------------- #
# Compute the difference formula for f'(a) with step size h.
# f : Vectorized function of one variable
# a : Compute derivative at x = a
# method : Difference formula: 'forward', 'backward' or 'central'
# h : Step size in difference formula
# TODO: TEST THIS IMPLEMENTATION
def delta(f, a: float, method='central', h=0.000001) -> float:
    if method == 'central':
        return (f(a + h) - f(a - h)) / (2 * h)
    elif method == 'forward':
        return (f(a + h) - f(a)) / h
    elif method == 'backward':
        return (f(a) - f(a - h)) / h


def conductivity(): pass


def resistivity(): pass


# ------------------------- DATA STRUCTURES OPERATIONS ---------------------------- #
# Print list to standard output
def show(a: list) -> None:
    print("########## START: Show List")
    print(a)
    print("########## END: Show List")


# Check if the list is sorted
def is_sorted(a: list, low: int, high: int) -> bool:
    print("########## START CHECK: List from", low, "to", high, "if is sorted")
    # Traverse all the list
    for i in range(low + 1, high):
        # Compare each one of the element
        if a[i] < a[i - 1]:
            print("########## END CHECK: List is NOT sorted")
            return False
    print("########## END CHECK: List is sorted")
    return True


# Exchange two elements of a list
def swap(a: list, i: int, j: int) -> None:
    print("########## START: Swap Elements", a[i], "<", a[j])
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
    print("########## END: Swapped Elements", a[i], ">", a[j])


# LOW LEVEL API MIMICKING ALLOCATION BEHAVIOR
def allocate(null_type: Any, size: int) -> list:
    print("########## START: Allocate a Null-Typed List")
    if size < 0:
        raise Exception("########## ERROR: length is not a positive integer")
    else:
        print("########## | ALLOC: NULL-TYPE LIST")
        print("########## END: Allocate a Null-Typed List \n")
        return [null_type] * size


def deallocate(obj: object):
    print("########## START: Deallocate an Object")
    if obj is None:
        print("########## | DEALLOC: EMPTY OBJECT")
        print("########## END: Allocate a Null-Typed List \n")
        del obj
    else:
        print("########## | DEALLOC: NON-EMPTY OBJECT")
        print("########## END: Deallocate an Object \n")
        del obj


# Function to find the partition position
def partition(a: list, low: int, high: int):
    # choose the rightmost element as pivot
    pivot = a[high]

    # pointer for greater element
    i = low - 1

    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if a[j] <= pivot:
            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            # (array[i], array[j]) = (array[j], array[i])
            swap(a, i, j)

    # Swap the pivot element with the greater element specified by i
    # (array[i + 1], array[high]) = (array[high], array[i + 1])
    swap(a, i + 1, high)

    # Return the position from where partition is done
    return i + 1


# Merges two sub-arrays of a[].
# First subarray is a[l ... m]
# Second subarray is a[m+1 ... r]
def merge_sub(a: list, left: int, mid: int, right: int):
    L = []
    R = []
    n = mid - left + 1
    m = right - mid

    # Create temp arrays with zeros
    for i in range(0, n):
        L.append(0)
    for j in range(0, m):
        R.append(0)

    # Copy data to temp arrays L[] and R[]
    for i in range(0, n):
        L[i] = a[left + i]

    for j in range(0, m):
        R[j] = a[mid + 1 + j]

    # Merge the temp arrays back into arr[l ... r]
    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    k = left  # Initial index of merged subarray

    while i < n and j < m:
        if L[i] <= R[j]:
            a[k] = L[i]
            i = i + 1
        else:
            a[k] = R[j]
            j = j + 1
        k = k + 1

    # Copy the remaining elements of L[]
    while i < n:
        a[k] = L[i]
        i = i + 1
        k = k + 1

    # Copy the remaining elements of R[]
    while j < m:
        a[k] = R[j]
        j = j + 1
        k = k + 1


# HEAPIFY OPERATIONS
# Bottom-Up Swim
def swim(a: list, k: int):
    while k > 1 and k // 2 < k:
        swap(a, k // 2, k)
        k = k // 2


# Top-Down Sink
def sink(a: list, k: int):
    n = len(a)

    while 2 * k <= n:
        j = 2 * k

        if j < n and j < j + 1:
            j = j + 1

        if not k < j:
            break

        swap(a, k, j)
        k = j


# DATA SCIENCE OPERATIONS ###########################################
def load_file(file_path: str, file_format='csv'):
    if file_format == 'csv':
        return pd.read_csv(file_path)
    elif file_format == 'json':
        return pd.read_json(file_path)
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    else:
        Exception('Specify file format')


def load_data(file_path: str,
              x_row_start: int, x_row_end: int,
              x_column_start: int, x_column_end: int,
              y_row_start: int, y_row_end: int,
              y_column_start: int, y_column_end: int
              ):
    data = load_file(file_path)
    x = data.iloc[x_row_start:x_row_end, x_column_start:x_column_end]
    y = data.iloc[y_row_start:y_row_end, y_column_start:y_column_end]
    return x.values, y.values


def split(x: np.ndarray, y: np.ndarray, test_size=1 / 3, random_state=0):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=test_size,
        random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def load_images(image_path: str):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    np.expand_dims(img, axis=0)
    return img


def load_images_split(train_path: str, test_path: str):
    x_train_gen = ImageDataGenerator(rescale=1. / 255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)
    x_train = x_train_gen.flow_from_directory(train_path,
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')
    x_test_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    x_test = x_test_gen.flow_from_directory(test_path,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

    return x_train, x_test
