def rounding_number(value, accuracy):
    return round(value, accuracy)


def rounding_vector(value, accuracy):
    for i in range(0, len(value)):
        value[i] = round(value[i], accuracy)
    return value


def rounding_matrix(value, accuracy):
    for i in range(0, len(value)):
        for j in range(0, len(value[0])):
            value[i][j] = round(value[i][j], accuracy)
    return value