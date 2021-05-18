## Инструкция по запуску

    git clone https://github.com/BaturinIgor/FQW.git  
    cd FQW/VKR_GUI
    python main.py

В случае необходимости установить библиотеки PyQt5 и numpy:

    pip install numpy
    pip install PyQt5

## Входная информация

    m - кол-во строк
    n - кол-во столбцов
    accuracy - точность, кол-во знаков после запятой
    matrix_coefficients - коэффициенты матрицы А в виде одномерного массива
    vector_coefficients - коэффициенты вектора b в виде одномерного массива
    
Если данные считываются с файла, то должна соблюдаться строгая структура данных при считывании:

    m n
    accyracy
    matrix_coefficients[0][0] matrix_coefficients[0][1] matrix_coefficients[1][0] matrix_coefficients[1][1]
    vector_coefficients[0] vector_coefficients[1]
    
Например:

    2 2
    5
    1 2 3 4
    4 5
    
Тестовые входные данные находятся в файлах inputData1.txt, inputData2.txt и inputData3.txt
