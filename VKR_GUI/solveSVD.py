import numpy as np
import matrixTransformation


def matrix_mult(U, vector):
    result_vector = [0.0] * len(U)
    for i in range(0, len(U)):
        value = 0.0
        for j in range(0, len(U[i])):
            value += U[i][j] * vector[j]
        result_vector[i] = value
    return result_vector


def solve(matrix, vector, accuracy, file_results):
    U, s, V = np.linalg.svd(matrix)

    # Округление матриц
    U = matrixTransformation.rounding_matrix(U, accuracy)
    V = matrixTransformation.rounding_matrix(V, accuracy)
    s = matrixTransformation.rounding_vector(s, accuracy)

    file_results.write("Левая ортогональная матрица U:\n" + str(U) + "\n\n")
    file_results.write("Сингулярные числа исходной матрицы A:\n" + str(s) + "\n\n")
    file_results.write("Правая ортогональная матрица VT:\n" + str(V) + "\n\n")
    x = [0] * len(V)
    w = matrix_mult(U.transpose(), vector)

    for i in range(0, len(V)):
        wi = w[i]
        alpha = s[i]
        if alpha != 0.0:
            alpha = 1.0 / alpha
        w[i] = alpha * w[i]

    x = matrix_mult(V.transpose(), w)
    matrixTransformation.rounding_vector(x, accuracy)
    file_results.write("Полученное решение:\n" + str(x) + "\n\n")
    return U, s, x, V


def calculation_of_trial_solutions(matrix, vector, accuracy):
    U, s, V = np.linalg.svd(matrix)

    U = matrixTransformation.rounding_matrix(U, accuracy)
    V = matrixTransformation.rounding_matrix(V, accuracy)
    s = matrixTransformation.rounding_vector(s, accuracy)

    w = np.array(matrix_mult(U.transpose(), vector))
    nonzero_singular_values = np.count_nonzero(s)
    trial_solutions = [[0.0] * len(s) for i in range(len(s))]

    #if len(w) > len(s):
    #    w = w[0:len(s) - 1]
    for i in range(1, nonzero_singular_values + 1):
        p = [0.0] * i
        for j in range(0, i):
            p[j] = w[j] / s[j]
            trial_solutions[i-1] += p[j] * V[j]
    w = matrixTransformation.rounding_vector(w, accuracy)

    vectorP = [0.0] * len(s)
    for i in range(0, len(s)):
        if w[i] == 0.0 or s[i] == 0.0:
            vectorP[i] = 0.0
        else:
            vectorP[i] = w[i] / s[i]
    vectorP = matrixTransformation.rounding_vector(vectorP, accuracy)

    trial_solutions = trial_solutions[0:nonzero_singular_values]
    trial_solutions = matrixTransformation.rounding_matrix(trial_solutions, accuracy)
    return trial_solutions, w, vectorP


def accuracy_solution(matrix, vectorX, vectorB, file_results):
    norm = 0
    acc = 0
    valuesAx = matrix_mult(matrix, vectorX)
    for i in range(0, len(vectorB)):
        norm += (vectorB[i] - valuesAx[i]) ** 2
        if vectorB[i] != 0.0 or valuesAx[i] != 0.0:
            if abs(vectorB[i]) > abs(valuesAx[i]):
                acc += abs(valuesAx[i] / vectorB[i])
            else:
                acc += abs(vectorB[i] / valuesAx[i])
    acc /= len(vectorB)
    fault = 1 - acc

    norm = matrixTransformation.rounding_number(np.sqrt(norm) ** 2, 10)
    fault = matrixTransformation.rounding_number(fault, 10)
    acc = matrixTransformation.rounding_number(acc * 100, 10)
    file_results.write("Квадрат нормы невязки полученного решения: " + str(norm) + "\n")
    file_results.write("Погрешность полученного решения: " + str(fault) + "\n")
    file_results.write("Точность решения: " + str(acc) + "%\n\n")
    return norm, fault, acc
