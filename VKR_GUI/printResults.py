from time import clock
from PyQt5 import QtCore
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QTableWidgetItem
from alignDelegate import AlignDelegate

import matrixTransformation
import numpy as np
import solveSVD


def print_solve(ui_MainWindow, original_matrix, vector, accuracy, file_results):
    s, x, V = solveSVD.solve(original_matrix, vector, accuracy, file_results)
    size_V = len(V)

    # Вывод матрицы V
    V = V.transpose()
    ui_MainWindow.matrixV.setRowCount(size_V)
    ui_MainWindow.matrixV.setColumnCount(size_V)
    ui_MainWindow.matrixV.setGeometry(QtCore.QRect(10, 40, size_V * 90 + 17, size_V * 40 + 27))

    ui_MainWindow.matrixV.clear()
    for i in range(0, size_V):
        for j in range(0, size_V):
            ui_MainWindow.matrixV.horizontalHeader().resizeSection(i, 90)
            ui_MainWindow.matrixV.verticalHeader().resizeSection(j, 40)

            item = QTableWidgetItem()
            item.setText(str(V[i][j]))
            item.setFlags(QtCore.Qt.ItemIsEditable)
            item.setForeground(QBrush(QColor(0, 0, 0)))
            ui_MainWindow.matrixV.setItem(i, j, item)

    delegate = AlignDelegate(ui_MainWindow.matrixV)
    ui_MainWindow.matrixV.setItemDelegate(delegate)

    # Вывод сингулярных чисел
    ui_MainWindow.singularValues.setRowCount(size_V)
    ui_MainWindow.singularValues.setColumnCount(1)
    ui_MainWindow.singularValuesLabel.setGeometry(QtCore.QRect(size_V * 90 + 40, 20, 121, 16))
    ui_MainWindow.singularValues.setGeometry(QtCore.QRect(size_V * 90 + 40, 40, 107, size_V * 40 + 27))

    ui_MainWindow.singularValues.clear()
    for i in range(0, size_V):
        ui_MainWindow.singularValues.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.singularValues.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(s[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.singularValues.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.singularValues)
    ui_MainWindow.singularValues.setItemDelegate(delegate)

    # Вывод вектора решения
    ui_MainWindow.vectorSolution.setRowCount(size_V)
    ui_MainWindow.vectorSolution.setColumnCount(1)
    ui_MainWindow.solveVector.setGeometry(QtCore.QRect((size_V + 1) * 90 + 80, 20, 101, 16))
    ui_MainWindow.vectorSolution.setGeometry(QtCore.QRect((size_V + 1) * 90 + 80, 40, 107, size_V * 40 + 27))

    ui_MainWindow.vectorSolution.clear()
    for i in range(0, size_V):
        ui_MainWindow.vectorSolution.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorSolution.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(x[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorSolution.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorSolution)
    ui_MainWindow.vectorSolution.setItemDelegate(delegate)

    # Вывод времени и памяти
    ui_MainWindow.timeLabel.setGeometry(QtCore.QRect(20, size_V * 40 + 80, 161, 16))
    ui_MainWindow.timeResult.setGeometry(QtCore.QRect(180, size_V * 40 + 80, 101, 16))
    ui_MainWindow.memoryLabel.setGeometry(QtCore.QRect(20, size_V * 40 + 110, 161, 16))
    ui_MainWindow.memoryResult.setGeometry(QtCore.QRect(180, size_V * 40 + 110, 101, 16))

    # Вывод ранга, нормы невязки и погрешности
    ui_MainWindow.rankLabel.setGeometry(QtCore.QRect(310, size_V * 40 + 80, 161, 16))
    ui_MainWindow.rankResult.setGeometry(QtCore.QRect(470, size_V * 40 + 80, 101, 16))
    ui_MainWindow.rankResult.setText(str(np.linalg.matrix_rank(original_matrix)))

    norm, fault, acc = solveSVD.accuracy_solution(original_matrix, x, vector, file_results)
    ui_MainWindow.residualNormSquaredLabel.setGeometry(QtCore.QRect(310, size_V * 40 + 110, 161, 16))
    ui_MainWindow.residualNormSquared.setGeometry(QtCore.QRect(470, size_V * 40 + 110, 101, 16))
    ui_MainWindow.residualNormSquared.setText(str(norm))

    ui_MainWindow.faultLabel.setGeometry(QtCore.QRect(310, size_V * 40 + 140, 161, 16))
    ui_MainWindow.faultResult.setGeometry(QtCore.QRect(470, size_V * 40 + 140, 101, 16))
    ui_MainWindow.faultResult.setText(str(fault))

    ui_MainWindow.accuracyLabel_2.setGeometry(QtCore.QRect(310, size_V * 40 + 170, 161, 16))
    ui_MainWindow.accuracyResult.setGeometry(QtCore.QRect(470, size_V * 40 + 170, 101, 16))
    ui_MainWindow.accuracyResult.setText(str(acc) + "%")

    # Вывод анализа обусловленности
    ui_MainWindow.conditionLabel.setGeometry(QtCore.QRect(580, size_V * 40 + 70, 141, 16))
    ui_MainWindow.conditionDescription.setGeometry(QtCore.QRect(580, size_V * 40 + 90, 231, 347 - size_V * 40))

    file_results.write("Число обусловленности: " +
                       str(matrixTransformation.rounding_number(s[size_V - 1] / s[0], accuracy + 3)) + "\n")
    ui_MainWindow.conditionDescription.setText("Число обусловленности матрицы: " +
                                               str(matrixTransformation.rounding_number(s[size_V - 1] / s[0], accuracy + 3)))
    ui_MainWindow.conditionDescription.setReadOnly(True)

    if s[size_V - 1] != 0.0:
        file_results.write("Исходная матрица хорошо обусловлена\n")
        ui_MainWindow.conditionDescription.append("Исходная матрица хорошо обусловлена")
    else:
        file_results.write("Исходная матрица плохо обусловлена\n")
        ui_MainWindow.conditionDescription.append("Исходная матрица плохо обусловлена")
        cond = [0.0] * (size_V - 1)
        file_results.write("Отношение сингулярных чисел:\n")
        ui_MainWindow.conditionDescription.append("Отношение сингулярных чисел:")
        for i in range(0, len(cond)):
            if s[i + 1] == 0.0:
                cond[i] = 10 ** (-accuracy + 3)
            cond[i] = s[i] / s[i + 1]
            file_results.write("s[" + str(i) + "]/s[" + str(i+1) + "] = " + str(cond[i]) + "\n")
            ui_MainWindow.conditionDescription.setText("s[" + str(i) + "]/s[" + str(i+1) + "] = " + str(cond[i]))
        if cond[size_V - 2] == 'inf' or max(cond) / min(cond) >= 10:
            file_results.write("\nОбусловленность 'хорошая' плохая т.к. есть резкое различие в значениях\n")
            ui_MainWindow.conditionDescription.append("Обусловленность 'хорошая' плохая т.к. есть резкое различие в значениях")
        else:
            file_results.write("\nОбусловленность 'плохая' плохая т.к. сингулярные числа уменьшаются без заметных скачков\n")
            ui_MainWindow.conditionDescription.append("Обусловленность 'плохая' плохая т.к. сингулярные числа уменьшаются без заметных скачков")
    return s


def print_trial_solutions(ui_MainWindow, original_matrix, vector, accuracy, file_results):
    trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(original_matrix, vector, accuracy)
    norm_vector = []
    printed_solutions = 0
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 1:\n" +
                           str(trial_solutions[0]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[0], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution, trial_solutions[0],
                                    ui_MainWindow.residualNormSquaredTrialSol1, norm,
                                    ui_MainWindow.accuracyTrialSol1, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution, 20, ui_MainWindow.residualNormSquaredTrialSol1,
                           ui_MainWindow.accuracyTrialSol1)
        print_empty_widget(ui_MainWindow.trialSolution2, 150, ui_MainWindow.residualNormSquaredTrialSol2,
                           ui_MainWindow.accuracyTrialSol2)
        print_empty_widget(ui_MainWindow.trialSolution3, 280, ui_MainWindow.residualNormSquaredTrialSol3,
                           ui_MainWindow.accuracyTrialSol3)
        print_empty_widget(ui_MainWindow.trialSolution4, 410, ui_MainWindow.residualNormSquaredTrialSol4,
                           ui_MainWindow.accuracyTrialSol4)
        print_empty_widget(ui_MainWindow.trialSolution5, 540, ui_MainWindow.residualNormSquaredTrialSol5,
                           ui_MainWindow.accuracyTrialSol5)
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    printed_solutions += 1
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 2:\n" +
                           str(trial_solutions[1]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[1], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution2, trial_solutions[1],
                                    ui_MainWindow.residualNormSquaredTrialSol2, norm,
                                    ui_MainWindow.accuracyTrialSol2, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution2, 150, ui_MainWindow.residualNormSquaredTrialSol2,
                           ui_MainWindow.accuracyTrialSol2)
        print_empty_widget(ui_MainWindow.trialSolution3, 280, ui_MainWindow.residualNormSquaredTrialSol3,
                           ui_MainWindow.accuracyTrialSol3)
        print_empty_widget(ui_MainWindow.trialSolution4, 410, ui_MainWindow.residualNormSquaredTrialSol4,
                           ui_MainWindow.accuracyTrialSol4)
        print_empty_widget(ui_MainWindow.trialSolution5, 540, ui_MainWindow.residualNormSquaredTrialSol5,
                           ui_MainWindow.accuracyTrialSol5)
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    printed_solutions += 1
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 3:\n" +
                           str(trial_solutions[2]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[2], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution3, trial_solutions[2],
                                    ui_MainWindow.residualNormSquaredTrialSol3, norm,
                                    ui_MainWindow.accuracyTrialSol3, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution3, 280, ui_MainWindow.residualNormSquaredTrialSol3,
                           ui_MainWindow.accuracyTrialSol3)
        print_empty_widget(ui_MainWindow.trialSolution4, 410, ui_MainWindow.residualNormSquaredTrialSol4,
                           ui_MainWindow.accuracyTrialSol4)
        print_empty_widget(ui_MainWindow.trialSolution5, 540, ui_MainWindow.residualNormSquaredTrialSol5,
                           ui_MainWindow.accuracyTrialSol5)
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    printed_solutions += 1
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 4:\n" +
                           str(trial_solutions[3]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[3], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution4, trial_solutions[3],
                                    ui_MainWindow.residualNormSquaredTrialSol4, norm,
                                    ui_MainWindow.accuracyTrialSol4, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution4, 410, ui_MainWindow.residualNormSquaredTrialSol4,
                           ui_MainWindow.accuracyTrialSol4)
        print_empty_widget(ui_MainWindow.trialSolution5, 540, ui_MainWindow.residualNormSquaredTrialSol5,
                           ui_MainWindow.accuracyTrialSol5)
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    printed_solutions += 1
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 5:\n" +
                           str(trial_solutions[4]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[4], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution5, trial_solutions[4],
                                    ui_MainWindow.residualNormSquaredTrialSol5, norm,
                                    ui_MainWindow.accuracyTrialSol5, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution5, 540, ui_MainWindow.residualNormSquaredTrialSol5,
                           ui_MainWindow.accuracyTrialSol5)
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    printed_solutions += 1
    if len(trial_solutions) != printed_solutions:
        file_results.write("\nПробное решение 6:\n" +
                           str(trial_solutions[5]) +
                           "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[5], vector, file_results)
        norm_vector.append(norm)
        print_vector_trial_solution(ui_MainWindow.trialSolution6, trial_solutions[5],
                                    ui_MainWindow.residualNormSquaredTrialSol6, norm,
                                    ui_MainWindow.accuracyTrialSol6, acc, printed_solutions)
    else:
        print_empty_widget(ui_MainWindow.trialSolution6, 670, ui_MainWindow.residualNormSquaredTrialSol6,
                           ui_MainWindow.accuracyTrialSol6)
        return w, vectorP, norm_vector, trial_solutions
    return w, vectorP, norm_vector, trial_solutions


def print_vector_trial_solution(trialSolUi, vector, normSquaredUi, norm, accuracyUi, acc, printed_solutions):
    trialSolUi.setRowCount(len(vector))
    trialSolUi.setColumnCount(1)
    trialSolUi.setGeometry(QtCore.QRect(130 * printed_solutions + 20, 40, 107, len(vector) * 40 + 27))
    for i in range(0, len(vector)):
        trialSolUi.horizontalHeader().resizeSection(1, 90)
        trialSolUi.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(vector[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        trialSolUi.setItem(i, 0, item)

    delegate = AlignDelegate(trialSolUi)
    trialSolUi.setItemDelegate(delegate)

    if str(norm) != 'nan' or str(norm) != 'inf':
        normSquaredUi.setText(str(norm))
    else:
        normSquaredUi.setText(str(0.0))
    if str(acc) != 'nan' or str(acc) != 'inf':
        accuracyUi.setText(str(acc) + "%")
    else:
        accuracyUi.setText("0.0%")
    accuracyUi.setText(str(acc) + "%")
    printed_solutions += 1


def print_empty_widget(trial_solution, x, norm, accuracy):
    trial_solution.setRowCount(0)
    trial_solution.setColumnCount(0)
    trial_solution.setGeometry(QtCore.QRect(x, 40, 107, 267))
    norm.setText("0.0")
    accuracy.setText("0.0%")


def print_data_to_file(original_matrix, vector, accuracy, dimension):
    start_time = clock()
    file_results = open("matrix" + str(dimension[0]) + "x" + str(dimension[1]) + ".txt", 'w')
    file_results.write("Размерность исходной матрицы: " + str(dimension[0]) + "x" + str(dimension[1]) + ".\n")
    file_results.write("Точность расчётов: " + str(accuracy) + " знаков после запятой.\n")
    file_results.write("Исходная матрица:\n" + str(original_matrix) + "\n")
    file_results.write("Вектор правых членов:\n" + str(vector) + "\n")
    s, x, V = solveSVD.solve(original_matrix, vector, accuracy, file_results)
    solveSVD.accuracy_solution(original_matrix, x, vector, file_results)
    if s[len(V) - 1] != 0.0:
        file_results.write("Исходная матрица хорошо обусловлена\n\n")
    else:
        file_results.write("Исходная матрица плохо обусловлена\n")
        cond = [0.0] * (len(V) - 1)
        file_results.write("Отношение сингулярных чисел:\n")
        for i in range(0, len(cond)):
            if s[i + 1] == 0.0:
                cond[i] = 10 ** (-accuracy + 3)
            cond[i] = s[i] / s[i + 1]
            file_results.write("s[" + str(i) + "]/s[" + str(i+1) + "] = " + str(cond[i]) + "\n")
        if cond[len(V) - 2] == 'inf' or max(cond) / min(cond) >= 10:
            file_results.write("\nОбусловленность 'хорошая' плохая т.к. есть резкое различие в значениях\n\n")
        else:
            file_results.write("\nОбусловленность 'плохая' плохая т.к. сингулярные числа уменьшаются без заметных скачков\n\n")

    trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(original_matrix, vector, accuracy)
    norm_vector = []
    for i in range(0, len(trial_solutions)):
        file_results.write("Пробное решение " + str(i+1) + ":\n" + str(trial_solutions[i]) + "\n")
        norm, fault, acc = solveSVD.accuracy_solution(original_matrix, trial_solutions[i], vector, file_results)
        norm_vector.append(norm)

    w = w[0:len(trial_solutions)]
    if len(w) - np.count_nonzero(w) != 0:
        file_results.write("Последнее(ие) " + str(len(w) - np.count_nonzero(w)) +
                           " значение(я) вектора G меньше заданной точности, поэтому считаются нулевыми.\n")
    else:
        file_results.write("Нулевых значений в векторе G нет.\n")
    for i in range(len(trial_solutions), 0, -1):
        if w[i - 1] != 0.0:
            file_results.write("Пробный вектор " + str(i) + " считается наиболее удовлитворительным решением.\n\n")
            break
        else:
            continue

    NSRCSS = [0.0] * len(trial_solutions)
    for i in range(0, len(trial_solutions)):
        NSRCSS[i] = np.sqrt(norm_vector[i] / (len(original_matrix) - i + 1))
    NSRCSS = matrixTransformation.rounding_vector(NSRCSS, accuracy + 1)
    min_NSRCSS = NSRCSS[0]
    for item in NSRCSS:
        if abs(item - (10 ** -accuracy)) < abs(min_NSRCSS - (10 ** -accuracy)):
            min_NSRCSS = item
    file_results.write("Анализ вектора N.S.R.C.S.S:\n")
    file_results.write("Вектор N.S.R.C.S.S:\n" + str(NSRCSS) + "\n")

    file_results.write("Ближайшее значение в векторе N.S.R.C.S.S к входной точности " +
                       str(10 ** (-accuracy)) +
                       " - это значение " +
                       str(min_NSRCSS) +
                       ", поэтому пробный вектор " +
                       str(NSRCSS.index(min_NSRCSS) + 1) +
                       " является предпочтительным пробным решением.\n")

    file_results.write("\nАнализ вектора P:\nСравнение |P| и |acc/s|\n")
    file_results.write("Вектор P:\n" + str(vectorP) + "\n")
    min_rat_index = 0
    for i in range(0, len(trial_solutions)):
        acc_div_s = matrixTransformation.rounding_number(abs(10 ** -accuracy / s[i]), accuracy + 3)
        if abs(vectorP[i]) > acc_div_s:
            file_results.write(str(abs(vectorP[i])) + " > " + str(acc_div_s) + "\n")
            min_rat_index = i
        else:
            file_results.write(str(abs(vectorP[i])) + " < " + str(acc_div_s) + "\n")

    file_results.write("Последнее неравенство, в котором левая часть больше правой: пробное решение " + str(
        min_rat_index + 1) + "\n\n")

    trial_solutions_norm = [0.0] * len(trial_solutions[0])
    for i in range(0, len(trial_solutions)):
        for j in range(0, len(trial_solutions[i])):
            trial_solutions_norm[i] += trial_solutions[i][j] ** 2
    min_norm_vector = min(norm_vector)
    file_results.write("Анализ векторов YNorm и RNorm:\n")
    file_results.write("Вектор YNorm:\n" + str(norm_vector))
    file_results.write("\nВектор RNorm:\n" + str(trial_solutions_norm))
    file_results.write("\nМинимальное значение в векторе YNorm - минимальное значение нормы невязки: " +
                       str(min_norm_vector))
    file_results.write("\nНаилучшее решение: пробное решение с нормой невязки " + str(min_norm_vector))
    time = matrixTransformation.rounding_number(clock() - start_time, accuracy + 3)
    memory = len(original_matrix) * len(original_matrix[0]) * 6 * 16 + len(original_matrix[0]) * 17 * 16 + 13 * 16
    matrixTransformation.rounding_number(memory / 1024, accuracy + 3)
    file_results.write("\nЗатраченное время: " + str(time) + " сек")
    file_results.write("\nЗатраченная память: " + str(memory / 1024) + " Кб")