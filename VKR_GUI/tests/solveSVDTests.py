import sys
import unittest

import numpy as np
from PyQt5 import QtWidgets

import gui
import solveSVD
from printResults import print_solve, print_trial_solutions


class SolveSVDTests(unittest.TestCase):
    original_matrix_2x2 = [[1, 3],
                           [3, 4]]

    original_matrix_4x2 = [[3, 4],
                           [6, 1],
                           [5, 1],
                           [1, 2]]

    original_matrix_3x3 = [[1, 3, 4],
                           [1, 2, 3],
                           [3, 4, 5]]

    original_matrix_4x4 = [[1, 3, 4, 2],
                           [1, 2, 3, 1],
                           [3, 4, 5, 1],
                           [3, 6, 2, 1]]

    original_matrix_5x5 = [[1, 3, 4, 7, 1],
                           [1, 2, 3, 5, 5],
                           [3, 4, 5, 4, 9],
                           [3, 6, 2, 4, 1],
                           [4, 3, 6, 2, 1]]

    original_matrix_6x6 = [[1, 3, 4, 3, 1, 7],
                           [1, 2, 3, 5, 7, 6],
                           [3, 4, 5, 5, 9, 2],
                           [3, 6, 2, 3, 5, 1],
                           [4, 7, 8, 3, 6, 2],
                           [1, 2, 3, 4, 5, 6]]

    vector_2x1 = [1, 3]
    vector_3x1 = [1, 2, 4]
    vector_4x1 = [1, 2, 3, 4]
    vector_4x2 = [2, 3, 5, 6]
    vector_5x1 = [4, 1, 4, 5, 1]
    vector_6x1 = [1, 3, 4, 6, 7, 1]

    vector_s_2x1 = np.array([5.8541, 0.8541])
    vector_s_3x1 = np.array([9.42616, 1.05204, 0.20168])
    vector_s_4x1 = np.array([11.5762, 3.19882, 1.32385, 0.0816])
    vector_s_4x2 = np.array([9.02793, 3.39064])
    vector_s_5x1 = np.array([18.46259, 6.52064, 4.78808, 3.55198, 0.26814])
    vector_s_6x1 = np.array([24.79181, 8.14367, 5.65312, 2.96712, 0.51324, 0.14729])

    vector_x_2x1 = [1.0, 0.0]
    vector_x_3x1 = [2.00004, -3.00005, 2.00002]
    vector_x_4x1 = [-6.24969, 4.24988, 2.49985, -7.7497]
    vector_x_4x2 = [0.5443, 0.65421]
    vector_x_5x1 = [-5.05266, 3.65018, 2.23126, -1.5391, -0.04918]
    vector_x_6x1 = [-7.28913, 3.67189, 0.88281, 0.75, 0.62502, -1.30469]

    matrix_V_2x2 = np.array([[-0.52573, -0.85065],
                             [0.85065, -0.52573]])
    matrix_V_3x3 = np.array([[-0.33631, -0.571, -0.7489],
                             [0.92598, -0.05559, -0.37345],
                             [0.17161, -0.81907, 0.54743]])
    matrix_V_4x4 = np.array([[-0.37168, -0.67736, -0.60003, -0.20737],
                             [0.27428, 0.57223, -0.74579, -0.20278],
                             [-0.63867, 0.30604, -0.1853, 0.68125],
                             [-0.6154, 0.34653, 0.22231, -0.67214]])
    matrix_V_4x2 = np.array([[-0.92194, -0.38734],
                             [-0.38734, 0.92194]])
    matrix_V_5x5 = np.array([[-0.28656, -0.42825, -0.48491, -0.52381, -0.47431],
                             [0.16493, 0.33511, 0.13199, 0.29955, -0.86796],
                             [-0.53895, -0.15203, -0.37772, 0.73653, 0.03565],
                             [0.10936, 0.69917, -0.67958, -0.13105, 0.14215],
                             [-0.76698, 0.43859, 0.37808, -0.27612, -0.01421]])
    matrix_V_6x6 = np.array([[-0.22444, -0.40105, -0.4306, -0.37912, -0.57035, -0.36652],
                             [0.28494, 0.38968, 0.13725, -0.18801, 0.16703, -0.82757],
                             [0.12261, 0.42622, 0.46359, -0.24649, -0.68655, 0.23723],
                             [-0.12365, -0.60803, 0.74743, -0.1502, 0.10756, -0.14909],
                             [0.28779, -0.19333, 0.06148, 0.80822, -0.40249, -0.2466],
                             [0.86907, -0.31396, -0.13562, -0.2905, 0.04338, 0.20366]])

    norm_2x2 = 0.0
    norm_3x3 = 1.3e-09
    norm_4x4 = 5.3e-08
    norm_4x2 = 25.7459978702
    norm_5x5 = 1.64e-08
    norm_6x6 = 1.46e-08

    fault_2x2 = 0.0
    fault_3x3 = 1.16667e-05
    fault_4x4 = 3.56243e-05
    fault_4x2 = 0.4450375343
    fault_5x5 = 3.65974e-05
    fault_6x6 = 1.91663e-05

    acc_2x2 = 100.0
    acc_3x3 = 99.9988333342
    acc_4x4 = 99.9964375689
    acc_4x2 = 55.4962465698
    acc_5x5 = 99.9963402598
    acc_6x6 = 99.9980833723

    trial_solutions_2x2 = [[0.27639, 0.44721],
                           [1., 0.]]
    trial_solutions_3x3 = [[0.15384, 0.2612, 0.34258],
                           [1.33156, 0.1905, -0.1324],
                           [2.00004, -3.00005, 2.00002]]
    trial_solutions_4x4 = [[0.16677, 0.30393, 0.26923, 0.09305],
                           [0.28329, 0.54703, -0.0476, 0.0069],
                           [0.57107, 0.40913, 0.03589, -0.30006],
                           [-6.24969, 4.24988, 2.49985, -7.7497]]
    trial_solutions_4x2 = [[0.69626, 0.29252],
                           [0.5443, 0.65421]]
    trial_solutions_5x5 = [[0.10538, 0.15748, 0.17832, 0.19262, 0.17442],
                           [0.14925, 0.24662, 0.21343, 0.2723, -0.05647],
                           [0.04682, 0.21773, 0.14164, 0.41229, -0.04969],
                           [0.12095, 0.6917, -0.31906, 0.32345, 0.04667],
                           [-5.05266, 3.65018, 2.23126, -1.5391, -0.04918]]
    trial_solutions_6x6 = [[0.08514, 0.15214, 0.16335, 0.14383, 0.21637, 0.13905],
                           [0.23279, 0.35406, 0.23447, 0.04641, 0.30292, -0.28976],
                           [0.25205, 0.42102, 0.30731, 0.00768, 0.19504, -0.25248],
                           [0.31664, 0.73863, -0.08311, 0.08614, 0.13886, -0.17461],
                           [-0.2811, 1.14018, -0.2108, -1.59254, 0.97483, 0.33758],
                           [-7.28913, 3.67189, 0.88281, 0.75, 0.62502, -1.30469]]

    vector_w_2x2 = np.array([-3.07768, 0.72654])
    vector_w_3x3 = np.array([-4.31192, 1.33805, 0.78561])
    vector_w_4x2 = np.array([-6.81798, 1.3302, 2.52369, 4.40191])
    vector_w_4x4 = np.array([-5.19418, 1.35895, -0.59651, 0.90441])
    vector_w_5x5 = np.array([-6.78922, 1.73455, 0.91001, 2.40792, 1.80872])
    vector_w_6x6 = np.array([-9.40516, 4.21961, 0.88823, -1.54988, -1.066, -1.18772])

    vector_P_2x2 = [-0.52573, 0.85065]
    vector_P_3x3 = [-0.45744, 1.27186, 3.89533]
    vector_P_4x2 = [-0.75521, 0.39232]
    vector_P_4x4 = [-0.44869, 0.42483, -0.45059, 11.08346]
    vector_P_5x5 = [-0.36773, 0.26601, 0.19006, 0.67791, 6.74543]
    vector_P_6x6 = [-0.37937, 0.51815, 0.15712, -0.52235, -2.077, -8.06382]

    norm_vector_2x2 = [0.5278650805, 0.0]
    norm_vector_3x3 = [2.4074814664, 0.6171686036, 1.3e-09]
    norm_vector_4x2 = [27.5155107884, 25.7459978702]
    norm_vector_4x4 = [3.0205070462, 1.1737785655, 0.8179318321, 5.3e-08]
    norm_vector_5x5 = [12.9067617612, 9.8980241675, 9.0698566571, 3.2716400143, 1.64e-08]
    norm_vector_6x6 = [23.5439457937, 5.7379477303, 4.9491057194, 2.5469439425, 1.4106373785, 1.46e-08]

    file_results = open("test.txt", 'w')

    def test_solve_matrix_2x2(self):
        s, x, V = solveSVD.solve(self.original_matrix_2x2, self.vector_2x1, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_2x1))
        self.assertEqual(str(x), str(self.vector_x_2x1))
        self.assertEqual(str(V), str(self.matrix_V_2x2))

    def test_solve_matrix_3x3(self):
        s, x, V = solveSVD.solve(self.original_matrix_3x3, self.vector_3x1, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_3x1))
        self.assertEqual(str(x), str(self.vector_x_3x1))
        self.assertEqual(str(V), str(self.matrix_V_3x3))

    def test_solve_matrix_4x4(self):
        s, x, V = solveSVD.solve(self.original_matrix_4x4, self.vector_4x1, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_4x1))
        self.assertEqual(str(x), str(self.vector_x_4x1))
        self.assertEqual(str(V), str(self.matrix_V_4x4))

    def test_solve_matrix_4x2(self):
        s, x, V = solveSVD.solve(self.original_matrix_4x2, self.vector_4x2, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_4x2))
        self.assertEqual(str(x), str(self.vector_x_4x2))
        self.assertEqual(str(V), str(self.matrix_V_4x2))

    def test_solve_matrix_5x5(self):
        s, x, V = solveSVD.solve(self.original_matrix_5x5, self.vector_5x1, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_5x1))
        self.assertEqual(str(x), str(self.vector_x_5x1))
        self.assertEqual(str(V), str(self.matrix_V_5x5))

    def test_solve_matrix_6x6(self):
        s, x, V = solveSVD.solve(self.original_matrix_6x6, self.vector_6x1, 5, self.file_results)
        self.assertEqual(str(s), str(self.vector_s_6x1))
        self.assertEqual(str(x), str(self.vector_x_6x1))
        self.assertEqual(str(V), str(self.matrix_V_6x6))

    def test_accuracy_solution_matrix_2x2(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_2x2,
                                                      self.vector_x_2x1,
                                                      self.vector_2x1,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_2x2))
        self.assertEqual(str(fault), str(self.fault_2x2))
        self.assertEqual(str(acc), str(self.acc_2x2))

    def test_accuracy_solution_matrix_3x3(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_3x3,
                                                      self.vector_x_3x1,
                                                      self.vector_3x1,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_3x3))
        self.assertEqual(str(fault), str(self.fault_3x3))
        self.assertEqual(str(acc), str(self.acc_3x3))

    def test_accuracy_solution_matrix_4x4(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_4x4,
                                                      self.vector_x_4x1,
                                                      self.vector_4x1,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_4x4))
        self.assertEqual(str(fault), str(self.fault_4x4))
        self.assertEqual(str(acc), str(self.acc_4x4))

    def test_accuracy_solution_matrix_4x2(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_4x2,
                                                      self.vector_x_4x2,
                                                      self.vector_4x2,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_4x2))
        self.assertEqual(str(fault), str(self.fault_4x2))
        self.assertEqual(str(acc), str(self.acc_4x2))

    def test_accuracy_solution_matrix_5x5(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_5x5,
                                                      self.vector_x_5x1,
                                                      self.vector_5x1,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_5x5))
        self.assertEqual(str(fault), str(self.fault_5x5))
        self.assertEqual(str(acc), str(self.acc_5x5))

    def test_accuracy_solution_matrix_6x6(self):
        norm, fault, acc = solveSVD.accuracy_solution(self.original_matrix_6x6,
                                                      self.vector_x_6x1,
                                                      self.vector_6x1,
                                                      self.file_results)
        self.assertEqual(str(norm), str(self.norm_6x6))
        self.assertEqual(str(fault), str(self.fault_6x6))
        self.assertEqual(str(acc), str(self.acc_6x6))

    def test_calculation_of_trial_solutions_2x2(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_2x2,
                                                                              self.vector_2x1,
                                                                              5)
        self.assertEqual(str(w), str(self.vector_w_2x2))
        self.assertEqual(str(vectorP), str(self.vector_P_2x2))

    def test_calculation_of_trial_solutions_3x3(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_3x3,
                                                                              self.vector_3x1,
                                                                              5)
        for i in range(0, len(trial_solutions)):
            for j in range(0, len(trial_solutions[0])):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_3x3[i][j])
        self.assertEqual(str(w), str(self.vector_w_3x3))
        self.assertEqual(str(vectorP), str(self.vector_P_3x3))

    def test_calculation_of_trial_solutions_4x2(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_4x2,
                                                                              self.vector_4x2,
                                                                              5)
        for i in range(0, len(trial_solutions)):
            for j in range(0, len(trial_solutions[0])):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_4x2[i][j])
        self.assertEqual(str(w), str(self.vector_w_4x2))
        self.assertEqual(str(vectorP), str(self.vector_P_4x2))

    def test_calculation_of_trial_solutions_4x4(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_4x4,
                                                                              self.vector_4x1,
                                                                              5)
        for i in range(0, len(trial_solutions)):
            for j in range(0, len(trial_solutions[0])):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_4x4[i][j])
        self.assertEqual(str(w), str(self.vector_w_4x4))
        self.assertEqual(str(vectorP), str(self.vector_P_4x4))

    def test_calculation_of_trial_solutions_5x5(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_5x5,
                                                                              self.vector_5x1,
                                                                              5)
        for i in range(0, len(trial_solutions)):
            for j in range(0, len(trial_solutions[0])):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_5x5[i][j])
        self.assertEqual(str(w), str(self.vector_w_5x5))
        self.assertEqual(str(vectorP), str(self.vector_P_5x5))

    def test_calculation_of_trial_solutions_6x6(self):
        trial_solutions, w, vectorP = solveSVD.calculation_of_trial_solutions(self.original_matrix_6x6,
                                                                              self.vector_6x1,
                                                                              5)
        for i in range(0, len(trial_solutions)):
            for j in range(0, len(trial_solutions[0])):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_6x6[i][j])
        self.assertEqual(str(w), str(self.vector_w_6x6))
        self.assertEqual(str(vectorP), str(self.vector_P_6x6))

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = gui.Ui_MainWindow()

    ui.setupUi(MainWindow)

    def test_printSolve_2x2(self):
        s = print_solve(self.ui,
                        self.original_matrix_2x2,
                        self.vector_2x1,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_2x2.transpose()
        for i in range(0, 2):
            for j in range(0, 2):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 2):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_2x1[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_2x1[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_2x2))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_2x2))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_2x2) + "%")
        self.assertEqual(str(s), str(self.vector_s_2x1))

    def test_printSolve_3x3(self):
        s = print_solve(self.ui,
                        self.original_matrix_3x3,
                        self.vector_3x1,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_3x3.transpose()
        for i in range(0, 3):
            for j in range(0, 3):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 3):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_3x1[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_3x1[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_3x3))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_3x3))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_3x3) + "%")
        self.assertEqual(str(s), str(self.vector_s_3x1))

    def test_printSolve_4x4(self):
        s = print_solve(self.ui,
                        self.original_matrix_4x4,
                        self.vector_4x1,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_4x4.transpose()
        for i in range(0, 4):
            for j in range(0, 4):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 4):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_4x1[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_4x1[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_4x4))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_4x4))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_4x4) + "%")
        self.assertEqual(str(s), str(self.vector_s_4x1))

    def test_printSolve_4x2(self):
        s = print_solve(self.ui,
                        self.original_matrix_4x2,
                        self.vector_4x2,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_4x2.transpose()
        for i in range(0, 2):
            for j in range(0, 2):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 2):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_4x2[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_4x2[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_4x2))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_4x2))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_4x2) + "%")
        self.assertEqual(str(s), str(self.vector_s_4x2))

    def test_printSolve_5x5(self):
        s = print_solve(self.ui,
                        self.original_matrix_5x5,
                        self.vector_5x1,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_5x5.transpose()
        for i in range(0, 5):
            for j in range(0, 5):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 5):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_5x1[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_5x1[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_5x5))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_5x5))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_5x5) + "%")
        self.assertEqual(str(s), str(self.vector_s_5x1))

    def test_printSolve_6x6(self):
        s = print_solve(self.ui,
                        self.original_matrix_6x6,
                        self.vector_6x1,
                        5,
                        self.file_results)
        matrixV = self.matrix_V_6x6.transpose()
        for i in range(0, 6):
            for j in range(0, 6):
                item = self.ui.matrixV.item(i, j)
                self.assertEqual(str(item.text()), str(matrixV[i][j]))
        for i in range(0, 6):
            item = self.ui.singularValues.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_s_6x1[i]))
            item = self.ui.vectorSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.vector_x_6x1[i]))
        self.assertEqual(str(self.ui.residualNormSquared.text()), str(self.norm_6x6))
        self.assertEqual(str(self.ui.faultResult.text()), str(self.fault_6x6))
        self.assertEqual(str(self.ui.accuracyResult.text()), str(self.acc_6x6) + "%")
        self.assertEqual(str(s), str(self.vector_s_6x1))

    def test_printTrialSolutions_2x2(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_2x2,
                                                                         self.vector_2x1,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 2):
            self.assertEqual(str(w[i]), str(self.vector_w_2x2[i]))
            self.assertEqual(str(vectorP[i]), str(self.vector_P_2x2[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_2x2[i]))
        for i in range(0, 2):
            for j in range(0, 2):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_2x2[i][j])
        for i in range(0, 2):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_2x2[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_2x2[1][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_2x2[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_2x2[1]))

    def test_printTrialSolutions_3x3(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_3x3,
                                                                         self.vector_3x1,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 3):
            self.assertEqual(str(w[i]), str(self.vector_w_3x3[i]))
            self.assertEqual(str(vectorP[i]), str(self.vector_P_3x3[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_3x3[i]))
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_3x3[i][j])
        for i in range(0, 3):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_3x3[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_3x3[1][i]))
            item = self.ui.trialSolution3.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_3x3[2][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_3x3[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_3x3[1]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol3.text()), str(self.norm_vector_3x3[2]))

    def test_printTrialSolutions_4x2(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_4x2,
                                                                         self.vector_4x2,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 4):
            self.assertEqual(str(w[i]), str(self.vector_w_4x2[i]))
        for i in range(0, 2):
            self.assertEqual(str(vectorP[i]), str(self.vector_P_4x2[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_4x2[i]))
        for i in range(0, 2):
            for j in range(0, 2):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_4x2[i][j])
        for i in range(0, 2):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x2[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x2[1][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_4x2[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_4x2[1]))

    def test_printTrialSolutions_4x4(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_4x4,
                                                                         self.vector_4x1,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 4):
            self.assertEqual(str(w[i]), str(self.vector_w_4x4[i]))
            self.assertEqual(str(vectorP[i]), str(self.vector_P_4x4[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_4x4[i]))
        for i in range(0, 4):
            for j in range(0, 4):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_4x4[i][j])
        for i in range(0, 4):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x4[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x4[1][i]))
            item = self.ui.trialSolution3.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x4[2][i]))
            item = self.ui.trialSolution4.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_4x4[3][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_4x4[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_4x4[1]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol3.text()), str(self.norm_vector_4x4[2]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol4.text()), str(self.norm_vector_4x4[3]))

    def test_printTrialSolutions_5x5(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_5x5,
                                                                         self.vector_5x1,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 5):
            self.assertEqual(str(w[i]), str(self.vector_w_5x5[i]))
            self.assertEqual(str(vectorP[i]), str(self.vector_P_5x5[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_5x5[i]))
        for i in range(0, 5):
            for j in range(0, 5):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_5x5[i][j])
        for i in range(0, 5):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_5x5[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_5x5[1][i]))
            item = self.ui.trialSolution3.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_5x5[2][i]))
            item = self.ui.trialSolution4.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_5x5[3][i]))
            item = self.ui.trialSolution5.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_5x5[4][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_5x5[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_5x5[1]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol3.text()), str(self.norm_vector_5x5[2]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol4.text()), str(self.norm_vector_5x5[3]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol5.text()), str(self.norm_vector_5x5[4]))

    def test_printTrialSolutions_6x6(self):
        w, vectorP, norm_vector, trial_solutions = print_trial_solutions(self.ui,
                                                                         self.original_matrix_6x6,
                                                                         self.vector_6x1,
                                                                         5,
                                                                         self.file_results)

        for i in range(0, 6):
            self.assertEqual(str(w[i]), str(self.vector_w_6x6[i]))
            self.assertEqual(str(vectorP[i]), str(self.vector_P_6x6[i]))
            self.assertEqual(str(norm_vector[i]), str(self.norm_vector_6x6[i]))
        for i in range(0, 6):
            for j in range(0, 6):
                self.assertEqual(trial_solutions[i][j], self.trial_solutions_6x6[i][j])
        for i in range(0, 6):
            item = self.ui.trialSolution.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[0][i]))
            item = self.ui.trialSolution2.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[1][i]))
            item = self.ui.trialSolution3.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[2][i]))
            item = self.ui.trialSolution4.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[3][i]))
            item = self.ui.trialSolution5.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[4][i]))
            item = self.ui.trialSolution6.item(i, 0)
            self.assertEqual(str(item.text()), str(self.trial_solutions_6x6[5][i]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol1.text()), str(self.norm_vector_6x6[0]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol2.text()), str(self.norm_vector_6x6[1]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol3.text()), str(self.norm_vector_6x6[2]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol4.text()), str(self.norm_vector_6x6[3]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol5.text()), str(self.norm_vector_6x6[4]))
        self.assertEqual(str(self.ui.residualNormSquaredTrialSol6.text()), str(self.norm_vector_6x6[5]))


if __name__ == '__main__':
    unittest.main()