# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIntValidator

import dataProcessing


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedWidth(830)
        MainWindow.setFixedHeight(560)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.mainWidget.setEnabled(True)
        self.mainWidget.setGeometry(QtCore.QRect(0, 40, 830, 480))
        self.mainWidget.setMinimumSize(QtCore.QSize(830, 480))
        self.mainWidget.setMaximumSize(QtCore.QSize(830, 480))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.mainWidget.setFont(font)
        self.mainWidget.setObjectName("mainWidget")
        self.inputData = QtWidgets.QWidget()
        self.inputData.setEnabled(True)
        self.inputData.setObjectName("inputData")
        self.size = QtWidgets.QLabel(self.inputData)
        self.size.setGeometry(QtCore.QRect(20, 20, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.size.setFont(font)
        self.size.setObjectName("size")
        self.matrixSize = QtWidgets.QComboBox(self.inputData)
        self.matrixSize.setGeometry(QtCore.QRect(110, 20, 111, 22))
        self.matrixSize.setObjectName("matrixSize")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.matrixSize.addItem("")
        self.accuracyLabel = QtWidgets.QLabel(self.inputData)
        self.accuracyLabel.setGeometry(QtCore.QRect(20, 60, 91, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracyLabel.setFont(font)
        self.accuracyLabel.setObjectName("accuracyLabel")
        self.accuracyInput = QtWidgets.QLineEdit(self.inputData)
        self.accuracyInput.setGeometry(QtCore.QRect(110, 60, 111, 20))
        self.accuracyInput.setObjectName("accuracyInput")
        self.accuracyInput.setValidator(QIntValidator(1, 9))
        self.setSizeButton = QtWidgets.QPushButton(self.inputData)
        self.setSizeButton.setGeometry(QtCore.QRect(240, 20, 151, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setSizeButton.setFont(font)
        self.setSizeButton.setObjectName("setSizeButton")
        self.solveButton = QtWidgets.QPushButton(self.inputData)
        self.solveButton.setGeometry(QtCore.QRect(590, 400, 101, 23))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.solveButton.setFont(font)
        self.solveButton.setObjectName("solveButton")
        self.matrCoeffLabel = QtWidgets.QLabel(self.inputData)
        self.matrCoeffLabel.setGeometry(QtCore.QRect(20, 100, 191, 16))
        self.matrCoeffLabel.setObjectName("matrCoeffLabel")
        self.vectorCoeffLabel = QtWidgets.QLabel(self.inputData)
        self.vectorCoeffLabel.setGeometry(QtCore.QRect(590, 100, 141, 16))
        self.vectorCoeffLabel.setObjectName("vectorCoeffLabel")
        self.matrixCoefficients = QtWidgets.QTableWidget(self.inputData)
        self.matrixCoefficients.setGeometry(QtCore.QRect(20, 120, 557, 267))
        self.matrixCoefficients.setObjectName("matrixCoefficients")
        self.matrixCoefficients.setColumnCount(0)
        self.matrixCoefficients.setRowCount(0)
        self.vectorCoefficients = QtWidgets.QTableWidget(self.inputData)
        self.vectorCoefficients.setGeometry(QtCore.QRect(590, 120, 107, 267))
        self.vectorCoefficients.setObjectName("vectorCoefficients")
        self.vectorCoefficients.setColumnCount(0)
        self.vectorCoefficients.setRowCount(0)
        self.mainWidget.addTab(self.inputData, "")
        self.results = QtWidgets.QWidget()
        self.results.setEnabled(True)
        self.results.setObjectName("results")
        self.rankLabel = QtWidgets.QLabel(self.results)
        self.rankLabel.setGeometry(QtCore.QRect(310, 330, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rankLabel.setFont(font)
        self.rankLabel.setObjectName("rankLabel")
        self.rankResult = QtWidgets.QLabel(self.results)
        self.rankResult.setGeometry(QtCore.QRect(470, 330, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rankResult.setFont(font)
        self.rankResult.setText("")
        self.rankResult.setObjectName("rankResult")
        self.faultResult = QtWidgets.QLabel(self.results)
        self.faultResult.setGeometry(QtCore.QRect(470, 390, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.faultResult.setFont(font)
        self.faultResult.setText("")
        self.faultResult.setObjectName("faultResult")
        self.faultLabel = QtWidgets.QLabel(self.results)
        self.faultLabel.setGeometry(QtCore.QRect(310, 390, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.faultLabel.setFont(font)
        self.faultLabel.setObjectName("faultLabel")
        self.timeResult = QtWidgets.QLabel(self.results)
        self.timeResult.setGeometry(QtCore.QRect(180, 330, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.timeResult.setFont(font)
        self.timeResult.setText("")
        self.timeResult.setObjectName("timeResult")
        self.timeLabel = QtWidgets.QLabel(self.results)
        self.timeLabel.setGeometry(QtCore.QRect(20, 330, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.timeLabel.setFont(font)
        self.timeLabel.setObjectName("timeLabel")
        self.memoryLabel = QtWidgets.QLabel(self.results)
        self.memoryLabel.setGeometry(QtCore.QRect(20, 360, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.memoryLabel.setFont(font)
        self.memoryLabel.setObjectName("memoryLabel")
        self.memoryResult = QtWidgets.QLabel(self.results)
        self.memoryResult.setGeometry(QtCore.QRect(180, 360, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.memoryResult.setFont(font)
        self.memoryResult.setText("")
        self.memoryResult.setObjectName("memoryResult")
        self.singularValuesLabel = QtWidgets.QLabel(self.results)
        self.singularValuesLabel.setGeometry(QtCore.QRect(580, 20, 121, 16))
        self.singularValuesLabel.setObjectName("singularValuesLabel")
        self.singularValues = QtWidgets.QTableWidget(self.results)
        self.singularValues.setGeometry(QtCore.QRect(580, 40, 107, 267))
        self.singularValues.setObjectName("singularValues")
        self.singularValues.setColumnCount(0)
        self.singularValues.setRowCount(0)
        self.vectorSolution = QtWidgets.QTableWidget(self.results)
        self.vectorSolution.setGeometry(QtCore.QRect(710, 40, 107, 267))
        self.vectorSolution.setObjectName("vectorSolution")
        self.vectorSolution.setColumnCount(0)
        self.vectorSolution.setRowCount(0)
        self.solveVector = QtWidgets.QLabel(self.results)
        self.solveVector.setGeometry(QtCore.QRect(710, 20, 101, 16))
        self.solveVector.setObjectName("solveVector")
        self.residualNormSquared = QtWidgets.QLabel(self.results)
        self.residualNormSquared.setGeometry(QtCore.QRect(470, 360, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.residualNormSquared.setFont(font)
        self.residualNormSquared.setText("")
        self.residualNormSquared.setObjectName("residualNormSquared")
        self.residualNormSquaredLabel = QtWidgets.QLabel(self.results)
        self.residualNormSquaredLabel.setGeometry(QtCore.QRect(310, 360, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.residualNormSquaredLabel.setFont(font)
        self.residualNormSquaredLabel.setObjectName("residualNormSquaredLabel")
        self.matrixV = QtWidgets.QTableWidget(self.results)
        self.matrixV.setGeometry(QtCore.QRect(10, 40, 557, 267))
        self.matrixV.setObjectName("matrixV")
        self.matrixV.setColumnCount(0)
        self.matrixV.setRowCount(0)
        self.matrixVLabel = QtWidgets.QLabel(self.results)
        self.matrixVLabel.setGeometry(QtCore.QRect(10, 20, 81, 16))
        self.matrixVLabel.setObjectName("matrixVLabel")
        self.accuracyResult = QtWidgets.QLabel(self.results)
        self.accuracyResult.setGeometry(QtCore.QRect(470, 420, 101, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracyResult.setFont(font)
        self.accuracyResult.setText("")
        self.accuracyResult.setObjectName("accuracyResult")
        self.accuracyLabel_2 = QtWidgets.QLabel(self.results)
        self.accuracyLabel_2.setGeometry(QtCore.QRect(310, 420, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracyLabel_2.setFont(font)
        self.accuracyLabel_2.setObjectName("accuracyLabel_2")
        self.conditionDescription = QtWidgets.QTextEdit(self.results)
        self.conditionDescription.setGeometry(QtCore.QRect(580, 340, 231, 101))
        self.conditionDescription.setObjectName("conditionDescription")
        self.conditionLabel = QtWidgets.QLabel(self.results)
        self.conditionLabel.setGeometry(QtCore.QRect(580, 320, 141, 16))
        self.conditionLabel.setObjectName("conditionLabel")
        self.mainWidget.addTab(self.results, "")
        self.trialSolutions = QtWidgets.QWidget()
        self.trialSolutions.setEnabled(True)
        self.trialSolutions.setObjectName("trialSolutions")
        self.residualNormSquaredLabelSingAnal = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredLabelSingAnal.setGeometry(QtCore.QRect(20, 320, 301, 20))
        self.residualNormSquaredLabelSingAnal.setObjectName("residualNormSquaredLabelSingAnal")
        self.residualNormSquaredTrialSol1 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol1.setGeometry(QtCore.QRect(20, 350, 111, 16))
        self.residualNormSquaredTrialSol1.setObjectName("residualNormSquaredTrialSol1")
        self.residualNormSquaredTrialSol2 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol2.setGeometry(QtCore.QRect(150, 350, 111, 16))
        self.residualNormSquaredTrialSol2.setObjectName("residualNormSquaredTrialSol2")
        self.accuracyTrialSol2 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol2.setGeometry(QtCore.QRect(150, 420, 111, 16))
        self.accuracyTrialSol2.setObjectName("accuracyTrialSol2")
        self.accuracyLabelSingAnal = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyLabelSingAnal.setGeometry(QtCore.QRect(20, 390, 301, 20))
        self.accuracyLabelSingAnal.setObjectName("accuracyLabelSingAnal")
        self.accuracyTrialSol1 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol1.setGeometry(QtCore.QRect(20, 420, 111, 16))
        self.accuracyTrialSol1.setObjectName("accuracyTrialSol1")
        self.trialSolution2 = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution2.setGeometry(QtCore.QRect(150, 40, 107, 267))
        self.trialSolution2.setObjectName("trialSolution2")
        self.trialSolution2.setColumnCount(0)
        self.trialSolution2.setRowCount(0)
        self.trialSolution2Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution2Label.setGeometry(QtCore.QRect(150, 20, 131, 16))
        self.trialSolution2Label.setObjectName("trialSolution2Label")
        self.trialSolution = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution.setGeometry(QtCore.QRect(20, 40, 107, 267))
        self.trialSolution.setObjectName("trialSolution")
        self.trialSolution.setColumnCount(0)
        self.trialSolution.setRowCount(0)
        self.trialSolution1Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution1Label.setGeometry(QtCore.QRect(20, 20, 121, 16))
        self.trialSolution1Label.setObjectName("trialSolution1Label")
        self.trialSolution3 = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution3.setGeometry(QtCore.QRect(280, 40, 107, 267))
        self.trialSolution3.setObjectName("trialSolution3")
        self.trialSolution3.setColumnCount(0)
        self.trialSolution3.setRowCount(0)
        self.trialSolution3Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution3Label.setGeometry(QtCore.QRect(280, 20, 121, 16))
        self.trialSolution3Label.setObjectName("trialSolution3Label")
        self.trialSolution4Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution4Label.setGeometry(QtCore.QRect(410, 20, 121, 16))
        self.trialSolution4Label.setObjectName("trialSolution4Label")
        self.trialSolution4 = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution4.setGeometry(QtCore.QRect(410, 40, 107, 267))
        self.trialSolution4.setObjectName("trialSolution4")
        self.trialSolution4.setColumnCount(0)
        self.trialSolution4.setRowCount(0)
        self.trialSolution5Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution5Label.setGeometry(QtCore.QRect(540, 20, 121, 16))
        self.trialSolution5Label.setObjectName("trialSolution5Label")
        self.trialSolution5 = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution5.setGeometry(QtCore.QRect(540, 40, 107, 267))
        self.trialSolution5.setObjectName("trialSolution5")
        self.trialSolution5.setColumnCount(0)
        self.trialSolution5.setRowCount(0)
        self.trialSolution6Label = QtWidgets.QLabel(self.trialSolutions)
        self.trialSolution6Label.setGeometry(QtCore.QRect(670, 20, 121, 16))
        self.trialSolution6Label.setObjectName("trialSolution6Label")
        self.trialSolution6 = QtWidgets.QTableWidget(self.trialSolutions)
        self.trialSolution6.setGeometry(QtCore.QRect(670, 40, 107, 267))
        self.trialSolution6.setObjectName("trialSolution6")
        self.trialSolution6.setColumnCount(0)
        self.trialSolution6.setRowCount(0)
        self.residualNormSquaredTrialSol4 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol4.setGeometry(QtCore.QRect(410, 350, 111, 16))
        self.residualNormSquaredTrialSol4.setObjectName("residualNormSquaredTrialSol4")
        self.residualNormSquaredTrialSol3 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol3.setGeometry(QtCore.QRect(280, 350, 111, 16))
        self.residualNormSquaredTrialSol3.setObjectName("residualNormSquaredTrialSol3")
        self.residualNormSquaredTrialSol5 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol5.setGeometry(QtCore.QRect(540, 350, 111, 16))
        self.residualNormSquaredTrialSol5.setObjectName("residualNormSquaredTrialSol5")
        self.residualNormSquaredTrialSol6 = QtWidgets.QLabel(self.trialSolutions)
        self.residualNormSquaredTrialSol6.setGeometry(QtCore.QRect(670, 350, 111, 16))
        self.residualNormSquaredTrialSol6.setObjectName("residualNormSquaredTrialSol6")
        self.accuracyTrialSol3 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol3.setGeometry(QtCore.QRect(280, 420, 111, 16))
        self.accuracyTrialSol3.setObjectName("accuracyTrialSol3")
        self.accuracyTrialSol4 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol4.setGeometry(QtCore.QRect(410, 420, 111, 16))
        self.accuracyTrialSol4.setObjectName("accuracyTrialSol4")
        self.accuracyTrialSol6 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol6.setGeometry(QtCore.QRect(670, 420, 111, 16))
        self.accuracyTrialSol6.setObjectName("accuracyTrialSol6")
        self.accuracyTrialSol5 = QtWidgets.QLabel(self.trialSolutions)
        self.accuracyTrialSol5.setGeometry(QtCore.QRect(540, 420, 111, 16))
        self.accuracyTrialSol5.setObjectName("accuracyTrialSol5")
        self.mainWidget.addTab(self.trialSolutions, "")
        self.singularAnalysis = QtWidgets.QWidget()
        self.singularAnalysis.setObjectName("singularAnalysis")
        self.vectorNormDescription = QtWidgets.QTextEdit(self.singularAnalysis)
        self.vectorNormDescription.setGeometry(QtCore.QRect(560, 310, 251, 131))
        self.vectorNormDescription.setObjectName("vectorNormDescription")
        self.vectorGDescription = QtWidgets.QTextEdit(self.singularAnalysis)
        self.vectorGDescription.setGeometry(QtCore.QRect(20, 310, 171, 131))
        self.vectorGDescription.setObjectName("vectorGDescription")
        self.vectorNSRCSSDescription = QtWidgets.QTextEdit(self.singularAnalysis)
        self.vectorNSRCSSDescription.setGeometry(QtCore.QRect(200, 310, 171, 131))
        self.vectorNSRCSSDescription.setObjectName("vectorNSRCSSDescription")
        self.vectorPDescription = QtWidgets.QTextEdit(self.singularAnalysis)
        self.vectorPDescription.setGeometry(QtCore.QRect(380, 310, 171, 131))
        self.vectorPDescription.setObjectName("vectorPDescription")
        self.vectorGResult = QtWidgets.QTableWidget(self.singularAnalysis)
        self.vectorGResult.setGeometry(QtCore.QRect(20, 30, 107, 267))
        self.vectorGResult.setObjectName("vectorGResult")
        self.vectorGResult.setColumnCount(0)
        self.vectorGResult.setRowCount(0)
        self.vectorGLabel = QtWidgets.QLabel(self.singularAnalysis)
        self.vectorGLabel.setGeometry(QtCore.QRect(20, 10, 121, 16))
        self.vectorGLabel.setObjectName("vectorGLabel")
        self.vectorNSRCSSResult = QtWidgets.QTableWidget(self.singularAnalysis)
        self.vectorNSRCSSResult.setGeometry(QtCore.QRect(200, 30, 107, 267))
        self.vectorNSRCSSResult.setObjectName("vectorNSRCSSResult")
        self.vectorNSRCSSResult.setColumnCount(0)
        self.vectorNSRCSSResult.setRowCount(0)
        self.vectorNSRCSSLabel = QtWidgets.QLabel(self.singularAnalysis)
        self.vectorNSRCSSLabel.setGeometry(QtCore.QRect(200, 10, 121, 16))
        self.vectorNSRCSSLabel.setObjectName("vectorNSRCSSLabel")
        self.vectorPResult = QtWidgets.QTableWidget(self.singularAnalysis)
        self.vectorPResult.setGeometry(QtCore.QRect(380, 30, 107, 267))
        self.vectorPResult.setObjectName("vectorPResult")
        self.vectorPResult.setColumnCount(0)
        self.vectorPResult.setRowCount(0)
        self.vectorPLabel = QtWidgets.QLabel(self.singularAnalysis)
        self.vectorPLabel.setGeometry(QtCore.QRect(380, 10, 121, 16))
        self.vectorPLabel.setObjectName("vectorPLabel")
        self.vectorRNormResult = QtWidgets.QTableWidget(self.singularAnalysis)
        self.vectorRNormResult.setGeometry(QtCore.QRect(700, 30, 107, 267))
        self.vectorRNormResult.setObjectName("vectorRNormResult")
        self.vectorRNormResult.setColumnCount(0)
        self.vectorRNormResult.setRowCount(0)
        self.vectorRNormLabel = QtWidgets.QLabel(self.singularAnalysis)
        self.vectorRNormLabel.setGeometry(QtCore.QRect(700, 10, 141, 16))
        self.vectorRNormLabel.setObjectName("vectorRNormLabel")
        self.vectorYNormResult = QtWidgets.QTableWidget(self.singularAnalysis)
        self.vectorYNormResult.setGeometry(QtCore.QRect(560, 30, 107, 267))
        self.vectorYNormResult.setObjectName("vectorYNormResult")
        self.vectorYNormResult.setColumnCount(0)
        self.vectorYNormResult.setRowCount(0)
        self.vectorYNormLabel = QtWidgets.QLabel(self.singularAnalysis)
        self.vectorYNormLabel.setGeometry(QtCore.QRect(560, 10, 141, 16))
        self.vectorYNormLabel.setObjectName("vectorYNormLabel")
        self.mainWidget.addTab(self.singularAnalysis, "")
        self.topMainWidget = QtWidgets.QLabel(self.centralwidget)
        self.topMainWidget.setGeometry(QtCore.QRect(10, 10, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.topMainWidget.setFont(font)
        self.topMainWidget.setObjectName("topMainWidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 835, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.upload = QtWidgets.QAction(MainWindow)
        self.upload.setObjectName("upload")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.menu.addAction(self.upload)
        self.menu.addAction(self.exit)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.mainWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.button_handler(MainWindow)

    def button_handler(self, MainWindow):
        self.setSizeButton.clicked.connect(lambda: dataProcessing.generate_matrix(self))
        self.solveButton.clicked.connect(lambda: dataProcessing.reading_data(self))
        self.upload.triggered.connect(lambda: dataProcessing.upload_data(self))
        self.exit.triggered.connect(lambda: dataProcessing.close_window(MainWindow))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SVD"))
        self.size.setText(_translate("MainWindow", "Размерность:"))
        self.matrixSize.setItemText(0, _translate("MainWindow", "2x2"))
        self.matrixSize.setItemText(1, _translate("MainWindow", "3x2"))
        self.matrixSize.setItemText(2, _translate("MainWindow", "4x2"))
        self.matrixSize.setItemText(3, _translate("MainWindow", "5x2"))
        self.matrixSize.setItemText(4, _translate("MainWindow", "6x2"))
        self.matrixSize.setItemText(5, _translate("MainWindow", "3x3"))
        self.matrixSize.setItemText(6, _translate("MainWindow", "4x3"))
        self.matrixSize.setItemText(7, _translate("MainWindow", "5x3"))
        self.matrixSize.setItemText(8, _translate("MainWindow", "6x3"))
        self.matrixSize.setItemText(9, _translate("MainWindow", "4x4"))
        self.matrixSize.setItemText(10, _translate("MainWindow", "5x4"))
        self.matrixSize.setItemText(11, _translate("MainWindow", "6x4"))
        self.matrixSize.setItemText(12, _translate("MainWindow", "5x5"))
        self.matrixSize.setItemText(13, _translate("MainWindow", "6x5"))
        self.matrixSize.setItemText(14, _translate("MainWindow", "6x6"))
        self.accuracyLabel.setText(_translate("MainWindow", "Точность:"))
        self.setSizeButton.setText(_translate("MainWindow", "Задать размерность"))
        self.solveButton.setText(_translate("MainWindow", "Решить"))
        self.matrCoeffLabel.setText(_translate("MainWindow", "Матрица коэффициентов:"))
        self.vectorCoeffLabel.setText(_translate("MainWindow", "Вектор правых членов:"))
        self.mainWidget.setTabText(self.mainWidget.indexOf(self.inputData), _translate("MainWindow", "Входные данные"))
        self.rankLabel.setText(_translate("MainWindow", "Ранг:"))
        self.faultLabel.setText(_translate("MainWindow", "Погрешность:"))
        self.timeLabel.setText(_translate("MainWindow", "Затраченное время:"))
        self.memoryLabel.setText(_translate("MainWindow", "Потребляемая память:"))
        self.singularValuesLabel.setText(_translate("MainWindow", "Сингулярные числа:"))
        self.solveVector.setText(_translate("MainWindow", "Вектор решения:"))
        self.residualNormSquaredLabel.setText(_translate("MainWindow", "Квадрат нормы невязки:"))
        self.matrixVLabel.setText(_translate("MainWindow", "Матрица V:"))
        self.conditionLabel.setText(_translate("MainWindow", "Обусловленность:"))
        self.mainWidget.setTabText(self.mainWidget.indexOf(self.results),
                                   _translate("MainWindow", "Полученные результаты"))
        self.residualNormSquaredLabelSingAnal.setText(_translate("MainWindow", "Квадрат нормы невязки:"))
        self.residualNormSquaredTrialSol1.setText(_translate("MainWindow", "0.0"))
        self.residualNormSquaredTrialSol2.setText(_translate("MainWindow", "0.0"))
        self.accuracyTrialSol2.setText(_translate("MainWindow", "0.0%"))
        self.accuracyLabelSingAnal.setText(_translate("MainWindow", "Точность решения:"))
        self.accuracyTrialSol1.setText(_translate("MainWindow", "0.0%"))
        self.trialSolution2Label.setText(_translate("MainWindow", "Пробное решение 2:"))
        self.trialSolution1Label.setText(_translate("MainWindow", "Пробное решение 1:"))
        self.trialSolution3Label.setText(_translate("MainWindow", "Пробное решение 3:"))
        self.trialSolution4Label.setText(_translate("MainWindow", "Пробное решение 4:"))
        self.trialSolution5Label.setText(_translate("MainWindow", "Пробное решение 5:"))
        self.trialSolution6Label.setText(_translate("MainWindow", "Пробное решение 6:"))
        self.residualNormSquaredTrialSol4.setText(_translate("MainWindow", "0.0"))
        self.residualNormSquaredTrialSol3.setText(_translate("MainWindow", "0.0"))
        self.residualNormSquaredTrialSol5.setText(_translate("MainWindow", "0.0"))
        self.residualNormSquaredTrialSol6.setText(_translate("MainWindow", "0.0"))
        self.accuracyTrialSol3.setText(_translate("MainWindow", "0.0%"))
        self.accuracyTrialSol4.setText(_translate("MainWindow", "0.0%"))
        self.accuracyTrialSol6.setText(_translate("MainWindow", "0.0%"))
        self.accuracyTrialSol5.setText(_translate("MainWindow", "0.0%"))
        self.mainWidget.setTabText(self.mainWidget.indexOf(self.trialSolutions),
                                   _translate("MainWindow", "Пробные решения"))
        self.vectorGLabel.setText(_translate("MainWindow", "Вектор G:"))
        self.vectorNSRCSSLabel.setText(_translate("MainWindow", "Вектор N.S.R.C.S.S:"))
        self.vectorPLabel.setText(_translate("MainWindow", "Вектор P:"))
        self.vectorRNormLabel.setText(_translate("MainWindow", "RNORM:"))
        self.vectorYNormLabel.setText(_translate("MainWindow", "YNORM:"))
        self.mainWidget.setTabText(self.mainWidget.indexOf(self.singularAnalysis),
                                   _translate("MainWindow", "Сингулярный анализ"))
        self.topMainWidget.setText(_translate("MainWindow", "Сингулярное разложение марицы"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.upload.setText(_translate("MainWindow", "Загрузить"))
        self.exit.setText(_translate("MainWindow", "Выход"))