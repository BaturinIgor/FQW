## Инструкция по запуску

    git clone https://github.com/BaturinIgor/FQW.git  
    git checkout dev
    cd FQW
    python main.py

## Входная информация

    accuracy - точность, кол-во знаков после запятой
    m - кол-во строк
    n - кол-во столбцов
    matrix_coefficients - коэффициенты матрицы А в виде одномерного массива
    vector_coefficients - коэффициенты вектора b в виде одномерного массива

## Выходная информация

    В качестве выходной информации в папке results генерируется файл с названием matrix_(размерность).txt в котором хранится следующая информация:
    - определитель, в случае, если матрица квадратная;
    - ранг матрицы для последующего сравнения с рангом, полученным посредством сингулярного разложения;
    - сингулярное разожение;
    - свойства сингулярного разложения;
    - число обусловленности;
    - расчёт погрешности полученных результатов;
    - изменение входных данных для сравнения полученных результатов.

## Описание функций

1. array_to_matrix(matr, number_of_rows, number_of_columns, file) - преобразование одномерного массива matr в матрицу размерности number_of_rows на number_of_columns
2. array_to_vector(vec, number_of_rows, file) - преобразование одномерного массива vec в вектор размерности number_of_rows на 1
3. rounding_number(value, accuracy) - округление числа до accuracy значащих цифр
4. rounding_vector(value, accuracy) - округление числел в векторе до accuracy значащих цифр
5. rounding_matrix(value, accuracy) - округление числел в матрице до accuracy значащих цифр
6. matrix_dimension(matrix, file) -  определяет квадратная матрица или прямоугольная
7. matrix_rank(matrix, file) - определяет ранг матрицы с последующим выводом о том является ли матрица матрицей полного ранга или нет
8. decomposition_matrices(matrix, accuracy, file) - получение сингулярного разложения и 3-х матриц, перемножение которых даёт исходную матрицу
9. matrix_mult(transp, alpha, U, vector, w) - умножение матрицы на вектор
10. ortonormalization(sqr_matrix, accuracy, file) - проверка ортогонализации столбцов и строк квадратной матрицы
11. svd_properties(U, s, V, accuracy, file) - получение свойств сингулярного разложения
12. solve(U, s, V, vector, accuracy, file) - получение корней СЛАУ
13. solution_check(matrix, vector, x, accuracy, file) - проверка полученных результатов с исходным вектором правых членов и расчёт погрешностей
