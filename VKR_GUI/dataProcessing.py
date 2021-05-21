import numpy as np
import matrixTransformation
import singularAnalysis
import printResults

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QFileDialog
from alignDelegate import AlignDelegate
from time import clock

start_time = 0


def generate_matrix(ui_MainWindow):
    current_size = ui_MainWindow.matrixSize.currentText()
    m = int(current_size[0])
    n = int(current_size[2])
    ui_MainWindow.matrixCoefficients.setRowCount(m)
    ui_MainWindow.matrixCoefficients.setColumnCount(n)
    ui_MainWindow.matrixCoefficients.setGeometry(QtCore.QRect(20, 120, n * 90 + 17, m * 40 + 27))
    for i in range(0, n):
        for j in range(0, m):
            ui_MainWindow.matrixCoefficients.horizontalHeader().resizeSection(i, 90)
            ui_MainWindow.matrixCoefficients.verticalHeader().resizeSection(j, 40)

    ui_MainWindow.matrixCoefficients.clear()

    delegate = AlignDelegate(ui_MainWindow.matrixCoefficients)
    ui_MainWindow.matrixCoefficients.setItemDelegate(delegate)

    ui_MainWindow.vectorCoeffLabel.setGeometry(QtCore.QRect(n * 90 + 50, 100, 141, 16))

    ui_MainWindow.vectorCoefficients.setRowCount(m)
    ui_MainWindow.vectorCoefficients.setColumnCount(1)
    ui_MainWindow.vectorCoefficients.setGeometry(QtCore.QRect(n * 90 + 50, 120, 117, m * 40 + 27))
    for i in range(0, m):
        ui_MainWindow.vectorCoefficients.horizontalHeader().resizeSection(1, 90)
        ui_MainWindow.vectorCoefficients.verticalHeader().resizeSection(i, 40)
    ui_MainWindow.vectorCoefficients.clear()

    delegate = AlignDelegate(ui_MainWindow.vectorCoefficients)
    ui_MainWindow.vectorCoefficients.setItemDelegate(delegate)

    ui_MainWindow.solveButton.setGeometry(QtCore.QRect(n * 90 + 60, m * 40 + 155, 100, 25))


def reading_data(ui_MainWindow):
    start_time = clock()
    m = ui_MainWindow.matrixCoefficients.rowCount()
    n = ui_MainWindow.matrixCoefficients.columnCount()

    file_results = open("matrix" + str(m) + "x" + str(n) + ".txt", 'w')
    original_matrix = [[0.0] * n for i in range(m)]
    vector = [0.0] * m

    file_results.write("Размерность исходной матрицы: " + str(m) + "x" + str(n) + ".\n")

    if ui_MainWindow.accuracyInput.text() != "":
        accuracy = int(ui_MainWindow.accuracyInput.text())
    else:
        warning_window("Введите значение точности")
        return
    file_results.write("Точность расчётов: " + str(accuracy) + " знаков после запятой.\n")
    if accuracy <= 1:
        warning_window("Значение точности должно быть больше 1!")
    else:
        try:
            for i in range(0, m):
                for j in range(0, n):
                    item = ui_MainWindow.matrixCoefficients.item(i, j)
                    if item:
                        value = float(item.text())
                        original_matrix[i][j] = value
                    else:
                        warning_window("Заполните матрицу")
                        return
            original_matrix = np.array(original_matrix)
            file_results.write("Исходная матрица:\n" + str(original_matrix) + "\n")
            for i in range(0, m):
                item = ui_MainWindow.vectorCoefficients.item(i, 0)
                if item and item.text():
                    value = float(item.text())
                    vector[i] = value
            vector = np.array(vector)
            file_results.write("Вектор правых членов:\n" + str(vector) + "\n")

            s = printResults.print_solve(ui_MainWindow, original_matrix, vector, accuracy, file_results)
            ui_MainWindow.mainWidget.setCurrentWidget(ui_MainWindow.results)
            g, vectorP, norm_vector, trial_solutions = printResults.print_trial_solutions(ui_MainWindow, original_matrix, vector, accuracy, file_results)
            singularAnalysis.print_vector_G(ui_MainWindow, g, len(trial_solutions), file_results)
            singularAnalysis.print_vector_NSRCSS(ui_MainWindow, norm_vector, m, len(trial_solutions), accuracy, file_results)
            singularAnalysis.print_vector_P(ui_MainWindow, len(trial_solutions), vectorP, accuracy, s, file_results)
            singularAnalysis.print_vectors_RYNorm(ui_MainWindow, norm_vector, trial_solutions, accuracy, file_results)
            time = matrixTransformation.rounding_number(clock() - start_time, accuracy + 3)
            ui_MainWindow.timeResult.setText(str(time) + " сек")
            memory = len(original_matrix) * len(original_matrix[0]) * 6 * 16 + len(original_matrix[0]) * 17 * 16 + 13 * 16
            ui_MainWindow.memoryResult.setText(str(matrixTransformation.rounding_number(memory / 1024, accuracy + 3)) + " Кб")
            file_results.write("\nЗатраченное время: " + str(time) + " сек")
            file_results.write("\nЗатраченная память: " + str(memory / 1024) + " Кб")
        except ValueError:
            warning_window("Введите корректные числа в матрице!")


def data_processing(ui_MainWindow, original_matrix, vector, m, accuracy):
    s = printResults.print_solve(ui_MainWindow, original_matrix, vector, accuracy)
    ui_MainWindow.mainWidget.setCurrentWidget(ui_MainWindow.results)
    w, vectorP, norm_vector, trial_solutions = printResults.print_trial_solutions(ui_MainWindow, original_matrix,
                                                                                  vector, accuracy)
    singularAnalysis.print_vector_G(ui_MainWindow, w, len(trial_solutions))
    singularAnalysis.print_vector_NSRCSS(ui_MainWindow, norm_vector, m, len(trial_solutions), accuracy)
    singularAnalysis.print_vector_P(ui_MainWindow, len(trial_solutions), vectorP, accuracy, s)
    singularAnalysis.print_vectors_RYNorm(ui_MainWindow, norm_vector, trial_solutions, accuracy)


def upload_data(ui_MainWindow):
    file_name = QFileDialog.getOpenFileName()

    f = open(file_name[0], 'r')
    result = []
    while True:
        line = f.readline()
        if not line:
            break
        result.append(line)
    result[0] = result[0].replace("\n", "")
    dimension = result[0].split(" ")
    result[1] = result[1].replace("\n", "")
    accuracy = result[1].split(" ")
    result[2] = result[2].replace("\n", "")
    original_matrix_str = result[2].split(" ")
    original_matrix = [[0.0] * int(dimension[1]) for i in range(int(dimension[0]))]
    for i in range(0, int(dimension[0])):
        for j in range(0, int(dimension[1])):
            original_matrix[i][j] = float(original_matrix_str[i*int(dimension[1]) + j])
    original_matrix = np.array(original_matrix)

    result[3] = result[3].replace("\n", "")
    vector_str = result[3].split(" ")
    vector = [0.0] * int(dimension[0])
    for i in range(0, int(dimension[0])):
        vector[i] = float(vector_str[i])
    vector = np.array(vector)

    if int(dimension[0]) > 6 or int(dimension[1]) > 6:
        printResults.print_data_to_file(original_matrix, vector, int(accuracy[0]), dimension)
    else:
        print_initial_data(ui_MainWindow, dimension, accuracy, original_matrix, vector)
    f.close()


def print_initial_data(ui_MainWindow, dimension, accuracy, original_matrix, vector):
    ui_MainWindow.matrixSize.setCurrentIndex(ui_MainWindow.matrixSize.findText(dimension[0] + "x" + dimension[1]))
    ui_MainWindow.accuracyInput.setText(str(accuracy[0]))
    generate_matrix(ui_MainWindow)
    for i in range(0, int(dimension[0])):
        for j in range(0, int(dimension[1])):
            item = QTableWidgetItem()
            item.setText(str(original_matrix[i][j]))
            ui_MainWindow.matrixCoefficients.setItem(i, j, item)
    for i in range(0, int(dimension[0])):
        item = QTableWidgetItem()
        item.setText(str(vector[i]))
        ui_MainWindow.vectorCoefficients.setItem(i, 0, item)


def close_window(MainWindow):
    MainWindow.close()


def warning_window(text):
    warning_accuracy_number = QMessageBox()
    warning_accuracy_number.setWindowTitle("Ошибка")
    warning_accuracy_number.setText(text)
    warning_accuracy_number.setIcon(QMessageBox.Warning)
    warning_accuracy_number.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    warning_accuracy_number.exec_()