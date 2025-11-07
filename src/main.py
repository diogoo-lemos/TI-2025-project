import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import huffmancodec as huffc
from collections import Counter


def mpgVsVars(data, names, symbol):
    numIndVars = len(names) - 1
    numLinhasPlot = math.ceil(numIndVars / 2)

    for i in range(0, numIndVars):
        plt.subplot(numLinhasPlot, 2, i + 1)
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
        plt.plot(data[:, i], data[:, -1], symbol)
        plt.xlabel(names[i])
        plt.ylabel(names[-1])
        plt.title(names[-1] + " VS " + names[i])
    plt.show()


def setA(data):
    numVars = np.shape(data)[1]
    A = np.zeros((2, numVars), dtype='uint16')

    for i in range(numVars):
        A[0, i] = data[:, i].min()
        A[1, i] = data[:, i].max()
    return (A)


def setNoX(data, A):
    numVars = np.shape(data)[1]
    listr = []
    for i in range(numVars):
        maxVarAValue = A[1][i]
        minVarAValue = A[0][i]
        numEleVar = np.shape(data[:, i])[0]
        numEleA = maxVarAValue - minVarAValue + 1
        holder = np.zeros(numEleA, dtype='int16')
        for j in range(numEleVar):
            # print(int(data[j][i])-minVarAValue)
            holder[int(data[j][i]) - minVarAValue] += 1
        listr.append(holder)
    return listr


def setNoXNZ(noX):
    listR = []
    for i in noX:
        listR.append(i[i > 0])
    return listR


def histogram(A, noX, lVars):
    for i in lVars:
        noXnz = noX[i][noX[i] > 0]
        B = np.arange(int(A[0][i]), int(A[1][i] + 1))
        B = B[noX[i] > 0]
        plt.bar(B.astype('str'), noXnz, width=0.5)
        plt.show()


def binningV2(data, A, noX, lVars, w, hVars):
    for i in lVars:
        minV = int(A[0][i])
        maxV = int(A[1][i])

        # +1 para obter o numero de indices
        numCic = math.ceil((maxV - minV + 1) / float(w))
        B = np.arange(minV, maxV + 1)

        print("Before binning: ")
        print("A: ", A[0][i], "  ", A[1][i], '\n')
        print(np.unique(data[:, i]), '\n\n\n')

        for j in range(0, numCic):
            mInd = noX[i][j * w:j * w + w].argmax() + j * w
            minR = B[j * w:j * w + w][0]
            maxR = B[j * w:j * w + w][-1]
            data[:, i][np.logical_and(
                data[:, i] >= minR, data[:, i] <= maxR)] = B[mInd]

        A[0][i] = data[:, i].min()
        A[1][i] = data[:, i].max()  # Atualizar o min/max de cada var

        print("After binning: ")
        print("A: ", A[0][i], "  ", A[1][i], '\n')
        print(np.unique(data[:, i]), '\n\n\n')

    if len(hVars) > 0:
        binnedNoX = setNoX(data, A)
        histogram(A, binnedNoX, hVars)


def calcP(B, data, noXNZ, lVars, listP):
    for i in lVars:
        sampleSize = np.shape(data[:, i])[0]
        for j in range(0, np.shape(B[i])[0]):
            numOX = noXNZ[i][j]
            listP[i][j] = numOX / sampleSize


def calcH(lVars, listP, listH):
    for i in lVars:
        listH[i] = -np.sum(listP[i] * np.log2(listP[i]))


def calcHuffman(B, listP, noXNZ, dataNP, lVars, varNames):
    print("\n\n")
    for alphabet, i in zip(B, range(len(B))):
        codec = huffc.HuffmanCodec.from_data(dataNP[:, i])
        symbols, lenght = codec.get_code_len()
        m = np.dot(listP[i], np.array(lenght))
        v = (np.dot(noXNZ[i], ((lenght - m)) ** 2)) / len(dataNP[:, i])
        print(m, " | ", v)
    print("\n\n")


def calcular_correlacoes(dataNP):
    idx_MPG = np.shape(dataNP)[1] - 1  # última coluna
    mpg = dataNP[:, idx_MPG]

    print("\n")
    for i in range(idx_MPG):  # exclui a própria MPG
        var = dataNP[:, i]
        coef_matrix = np.corrcoef(mpg, var)
        coef = coef_matrix[0, 1]
        print(coef)


def mutual_information(x, y):
    joint_hist, _, _ = np.histogram2d(x, y, bins=(np.unique(x).size, np.unique(y).size))

    joint_prob = joint_hist / np.sum(joint_hist)
    px = np.sum(joint_prob, axis=1)  # X
    py = np.sum(joint_prob, axis=0)  # Y

    nz = joint_prob > 0
    MI = np.sum(joint_prob[nz] * np.log2(joint_prob[nz] / (px[:, None] * py)[nz]))

    return MI


def calc_MI_all_vars(data, varNames):
    idx_MPG = len(varNames) - 1
    mpg = data[:, idx_MPG]

    for i in range(len(varNames) - 1):  # Exclui MPG
        x = data[:, i]
        MI = mutual_information(x, mpg)
        print(MI)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def avaliar_modelo(dataNP):
    Acceleration = dataNP[:, 0]
    Cylinders = dataNP[:, 1]
    Displacement = dataNP[:, 2]
    Horsepower = dataNP[:, 3]
    Model = dataNP[:, 4]
    Weight = dataNP[:, 5]
    mpg = dataNP[:, 6]

    MPGpred = (-5.5241
               - 0.146 * Acceleration
               - 0.4909 * Cylinders
               + 0.0026 * Displacement
               - 0.0045 * Horsepower
               + 0.6725 * Model
               - 0.0059 * Weight)
    # print("\n")
    # print(MPGpred[:10])

    mean_acc = np.mean(Acceleration)
    MPGpred_c = (-5.5241
                 - 0.146 * mean_acc
                 - 0.4909 * Cylinders
                 + 0.0026 * Displacement
                 - 0.0045 * Horsepower
                 + 0.6725 * Model
                 - 0.0059 * Weight)

    # print("\n")
    # print(MPGpred_c[:10])

    mean_weight = np.mean(Weight)
    MPGpred_d = (-5.5241
                 - 0.146 * Acceleration
                 - 0.4909 * Cylinders
                 + 0.0026 * Displacement
                 - 0.0045 * Horsepower
                 + 0.6725 * Model
                 - 0.0059 * mean_weight)

    # print("\n")
    # print(MPGpred_d[:10])

    rmse_original = rmse(mpg, MPGpred)
    rmse_acc_mean = rmse(mpg, MPGpred_c)
    rmse_weight_mean = rmse(mpg, MPGpred_d)

    mae_original = mae(mpg, MPGpred)
    mae_acc_mean = mae(mpg, MPGpred_c)
    mae_weight_mean = mae(mpg, MPGpred_d)

    print("\n")
    print(mae_original)
    print(rmse_original)
    print("\n")
    print(mae_acc_mean)
    print(rmse_acc_mean)
    print("\n")
    print(mae_weight_mean)
    print(rmse_weight_mean)


"-----------------------------------------------------------------"


def main():
    data = pd.read_excel(r"Data\CarDataset.xlsx")  # path to file
    dataNP = data.to_numpy()
    varNames = data.columns.values.tolist()
    lVarsInd = np.arange(np.shape(dataNP)[1] - 1)
    lVarsDep = np.array([6]);
    lVars = np.concatenate((lVarsInd, lVarsDep))

    mpgVsVars(dataNP, varNames, 'og')

    dataNP = dataNP.astype('uint16')
    A = setA(dataNP)

    noX = setNoX(dataNP, A)

    histogram(A, noX, lVars)
    binningV2(dataNP, A, noX, [5, ], 40, [])
    binningV2(dataNP, A, noX, [2, 3], 5, [])
    noX = setNoX(dataNP, A)
    noXNZ = setNoXNZ(noX)

    B = []
    listP = []
    listH = np.zeros(len(lVars))
    for i in lVars:
        AMin = A[0][i]
        AMax = A[1][i]
        B.append(np.arange(AMin, AMax + 1))
        B[i] = B[i][noX[i] > 0]
        listP.append(np.zeros(np.shape(B[i])[0]))

    calcP(B, dataNP, noXNZ, lVars, listP)

    calcH(lVars, listP, listH)
    for i in listH:
        print(i)

    calcHuffman(B, listP, noXNZ, dataNP, lVars, varNames)

    calcular_correlacoes(dataNP)

    print("\n")
    calc_MI_all_vars(dataNP, varNames)

    avaliar_modelo(dataNP)


if __name__ == '__main__':
    main()
