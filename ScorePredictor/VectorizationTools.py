import numpy
import math


def SimpleVectorizedResult(j):
    j = int(j)
    result = numpy.zeros((10))
    result[j] = 1.0
    return result.tolist()


def NormalizedLogisticVectorizedResult(j):
    j = int(j)
    result = numpy.zeros((10))
    sum = 0
    for i in range(0, 10):
        result[i] = math.exp(-1.5 * (abs(j - i)))
        sum = sum + result[i]

    for i in range(0, 10):
        result[i] = result[i] / sum

    return result


def VectorizationMethod(j):
    return VectorizationClass.VectorizationMethod(j)


class VectorizationClass:
    VectorizationMethod = SimpleVectorizedResult
