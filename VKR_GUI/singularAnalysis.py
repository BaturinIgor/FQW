import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QTableWidgetItem

import matrixTransformation
from alignDelegate import AlignDelegate


def print_vector_G(ui_MainWindow, w, trial_solutions_count, file_results):
    # Печать вектора G
    ui_MainWindow.vectorGResult.setRowCount(trial_solutions_count)
    ui_MainWindow.vectorGResult.setColumnCount(1)
    ui_MainWindow.vectorGLabel.setGeometry(QtCore.QRect(20, 10, 121, 16))
    ui_MainWindow.vectorGResult.setGeometry(QtCore.QRect(20, 30, 107, trial_solutions_count * 40 + 27))

    ui_MainWindow.vectorGResult.clear()
    for i in range(0, trial_solutions_count):
        ui_MainWindow.vectorGResult.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorGResult.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(w[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorGResult.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorGResult)
    ui_MainWindow.vectorGResult.setItemDelegate(delegate)

    ui_MainWindow.vectorGDescription.clear()
    ui_MainWindow.vectorGDescription.setGeometry(20, trial_solutions_count * 40 + 64, 171, 377 - trial_solutions_count * 40)
    ui_MainWindow.vectorGDescription.append("Анализ вектора G:\n")
    file_results.write("\nАнализ вектора G:\n")
    file_results.write("Вектор G:\n" + str(w) + "\n")
    ui_MainWindow.vectorGDescription.setReadOnly(True)

    w = w[0:trial_solutions_count]
    if len(w) - np.count_nonzero(w) != 0:
        file_results.write("Последнее(ие) " + str(len(w) - np.count_nonzero(w)) +
                                                " значение(я) вектора G меньше заданной точности, поэтому считаются нулевыми.\n")
        ui_MainWindow.vectorGDescription.append("Последнее(ие) " + str(len(w) - np.count_nonzero(w)) +
                                                " значение(я) вектора G меньше заданной точности, поэтому считаются нулевыми.\n")
    else:
        file_results.write("Нулевых значений в векторе G нет.\n")
        ui_MainWindow.vectorGDescription.append("Нулевых значений в векторе G нет.\n")

    for i in range(trial_solutions_count, 0, -1):
        if w[i - 1] != 0.0:
            file_results.write("Пробный вектор " + str(i) + " считается наиболее удовлитворительным решением.\n\n")
            ui_MainWindow.vectorGDescription.append(
                "Пробный вектор " + str(i) + " считается наиболее удовлитворительным решением")
            break
        else:
            continue


def print_vector_NSRCSS(ui_MainWindow, norm_vector, m, trial_solutions_count, accuracy, file_results):
    # Печать вектора NSRCSS
    NSRCSS = [0.0] * trial_solutions_count
    for i in range(0, trial_solutions_count):
        NSRCSS[i] = np.sqrt(norm_vector[i] / (m - i + 1))
    NSRCSS = matrixTransformation.rounding_vector(NSRCSS, accuracy + 1)
    ui_MainWindow.vectorNSRCSSResult.setRowCount(trial_solutions_count)
    ui_MainWindow.vectorNSRCSSResult.setColumnCount(1)
    ui_MainWindow.vectorNSRCSSLabel.setGeometry(QtCore.QRect(200, 10, 121, 16))
    ui_MainWindow.vectorNSRCSSResult.setGeometry(QtCore.QRect(200, 30, 107, trial_solutions_count * 40 + 27))

    ui_MainWindow.vectorNSRCSSResult.clear()
    for i in range(0, trial_solutions_count):
        ui_MainWindow.vectorNSRCSSResult.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorNSRCSSResult.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(NSRCSS[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorNSRCSSResult.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorNSRCSSResult)
    ui_MainWindow.vectorNSRCSSResult.setItemDelegate(delegate)

    min_NSRCSS = NSRCSS[0]
    for item in NSRCSS:
        if abs(item - (10 ** -accuracy)) < abs(min_NSRCSS - (10 ** -accuracy)):
            min_NSRCSS = item

    ui_MainWindow.vectorNSRCSSDescription.clear()
    ui_MainWindow.vectorNSRCSSDescription.setGeometry(200, trial_solutions_count * 40 + 64, 171,
                                                 377 - trial_solutions_count * 40)
    ui_MainWindow.vectorNSRCSSDescription.append("Анализ вектора N.S.R.C.S.S:\n")
    file_results.write("Анализ вектора N.S.R.C.S.S:\n")
    file_results.write("Вектор N.S.R.C.S.S:\n" + str(NSRCSS) + "\n")
    ui_MainWindow.vectorNSRCSSDescription.setReadOnly(True)

    ui_MainWindow.vectorNSRCSSDescription.append("Ближайшее значение в векторе N.S.R.C.S.S к входной точности " +
                                                 str(10 ** (-accuracy)) +
                                                 " - это значение " +
                                                 str(min_NSRCSS) +
                                                 ", поэтому пробный вектор " +
                                                 str(NSRCSS.index(min_NSRCSS) + 1) +
                                                 " является предпочтительным пробным решением.")
    file_results.write("Ближайшее значение в векторе N.S.R.C.S.S к входной точности " +
                       str(10 ** (-accuracy)) +
                       " - это значение " +
                       str(min_NSRCSS) +
                       ", поэтому пробный вектор " +
                       str(NSRCSS.index(min_NSRCSS) + 1) +
                       " является предпочтительным пробным решением.\n")


def print_vector_P(ui_MainWindow, trial_solutions_count, vectorP, accuracy, s, file_results):
    ui_MainWindow.vectorPResult.setRowCount(trial_solutions_count)
    ui_MainWindow.vectorPResult.setColumnCount(1)
    ui_MainWindow.vectorPLabel.setGeometry(QtCore.QRect(380, 10, 121, 16))
    ui_MainWindow.vectorPResult.setGeometry(QtCore.QRect(380, 30, 107, trial_solutions_count * 40 + 27))

    ui_MainWindow.vectorPResult.clear()
    for i in range(0, trial_solutions_count):
        ui_MainWindow.vectorPResult.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorPResult.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(vectorP[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorPResult.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorPResult)
    ui_MainWindow.vectorPResult.setItemDelegate(delegate)

    ui_MainWindow.vectorPDescription.clear()
    ui_MainWindow.vectorPDescription.setGeometry(380, trial_solutions_count * 40 + 64, 171,
                                                      377 - trial_solutions_count * 40)
    ui_MainWindow.vectorPDescription.setReadOnly(True)
    ui_MainWindow.vectorPDescription.append("Анализ вектора P:\nСравнение |P| и |acc/s|\n")
    file_results.write("\nАнализ вектора P:\nСравнение |P| и |acc/s|\n")
    file_results.write("Вектор P:\n" + str(vectorP) + "\n")
    min_rat_index = 0
    for i in range(0, trial_solutions_count):
        acc_div_s = matrixTransformation.rounding_number(abs(10 ** -accuracy / s[i]), accuracy+3)
        if abs(vectorP[i]) > acc_div_s:
            file_results.write(str(abs(vectorP[i])) + " > " + str(acc_div_s) + "\n")
            ui_MainWindow.vectorPDescription.append(str(abs(vectorP[i])) + " > " + str(acc_div_s))
            min_rat_index = i
        else:
            file_results.write(str(abs(vectorP[i])) + " < " + str(acc_div_s) + "\n")
            ui_MainWindow.vectorPDescription.append(str(abs(vectorP[i])) + " < " + str(acc_div_s))

    file_results.write("Последнее неравенство, в котором левая часть больше правой: пробное решение " + str(min_rat_index + 1) + "\n\n")
    ui_MainWindow.vectorPDescription.append("Последнее неравенство, в котором левая часть больше правой: пробное решение " + str(min_rat_index + 1))


def print_vectors_RYNorm(ui_MainWindow, norm_vector, trial_solutions, accuracy, file_results):
    # Печать вектора YNorm
    ui_MainWindow.vectorYNormResult.setRowCount(len(trial_solutions))
    ui_MainWindow.vectorYNormResult.setColumnCount(1)
    ui_MainWindow.vectorYNormLabel.setGeometry(QtCore.QRect(560, 10, 121, 16))
    ui_MainWindow.vectorYNormResult.setGeometry(QtCore.QRect(560, 30, 107, len(trial_solutions) * 40 + 27))

    ui_MainWindow.vectorYNormResult.clear()
    for i in range(0, len(trial_solutions)):
        ui_MainWindow.vectorYNormResult.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorYNormResult.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(norm_vector[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorYNormResult.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorYNormResult)
    ui_MainWindow.vectorYNormResult.setItemDelegate(delegate)

    # Печать вектора RNorm
    trial_solutions_norm = [0.0] * len(trial_solutions[0])
    for i in range(0, len(trial_solutions)):
        for j in range(0, len(trial_solutions[i])):
            trial_solutions_norm[i] += trial_solutions[i][j] ** 2
        trial_solutions_norm[i] = np.sqrt(trial_solutions_norm[i])
    matrixTransformation.rounding_vector(trial_solutions_norm, accuracy)
    ui_MainWindow.vectorRNormResult.setRowCount(len(trial_solutions))
    ui_MainWindow.vectorRNormResult.setColumnCount(1)
    ui_MainWindow.vectorRNormLabel.setGeometry(QtCore.QRect(700, 10, 121, 16))
    ui_MainWindow.vectorRNormResult.setGeometry(QtCore.QRect(700, 30, 107, len(trial_solutions) * 40 + 27))

    ui_MainWindow.vectorRNormResult.clear()
    for i in range(0, len(trial_solutions)):
        ui_MainWindow.vectorRNormResult.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorRNormResult.verticalHeader().resizeSection(i, 40)

        item = QTableWidgetItem()
        item.setText(str(trial_solutions_norm[i]))
        item.setFlags(QtCore.Qt.ItemIsEditable)
        item.setForeground(QBrush(QColor(0, 0, 0)))
        ui_MainWindow.vectorRNormResult.setItem(i, 0, item)

    delegate = AlignDelegate(ui_MainWindow.vectorRNormResult)
    ui_MainWindow.vectorRNormResult.setItemDelegate(delegate)

    ui_MainWindow.vectorNormDescription.clear()
    ui_MainWindow.vectorNormDescription.setGeometry(560, len(trial_solutions) * 40 + 64, 251,
                                                    377 - len(trial_solutions) * 40)
    min_norm_vector = min(norm_vector)
    ui_MainWindow.vectorNormDescription.setReadOnly(True)
    file_results.write("Анализ векторов YNorm и RNorm:\n")
    file_results.write("Вектор YNorm:\n" + str(norm_vector))
    file_results.write("\nВектор RNorm:\n" + str(trial_solutions_norm))
    file_results.write("\nМинимальное значение в векторе YNorm - минимальное значение нормы невязки: " +
                       str(min_norm_vector))
    file_results.write("\nНаилучшее решение: пробное решение с нормой невязки " + str(min_norm_vector))
    ui_MainWindow.vectorNormDescription.append("Анализ векторов YNorm и RNorm:\n")
    ui_MainWindow.vectorNormDescription.append("Минимальное значение в векторе YNorm - минимальное значение нормы невязки: " +
                                               str(min_norm_vector))
    ui_MainWindow.vectorNormDescription.append("Наилучшее решение: пробное решение с нормой невязки " + str(min_norm_vector))
