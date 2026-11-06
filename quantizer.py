import math

class Int8Quantizer:
    @staticmethod
    def quantizeMatrix(matrix):
        rowCount = len(matrix)
        colCount = len(matrix[0])
        maxAbsVal = 1e-9
        for row in matrix:
            for val in row:
                absVal = abs(val)
                if absVal > maxAbsVal:
                    maxAbsVal = absVal

        scale = maxAbsVal / 127.0
        quantizedMatrix = []
        dequantizedMatrix = []

        for row in matrix:
            quantizedRow = []
            dequantizedRow = []
            for val in row:
                intVal = int(round(val / scale))
                clampedVal = max(-128, min(127, intVal))
                quantizedRow.append(clampedVal)
                dequantizedRow.append(clampedVal * scale)
            quantizedMatrix.append(quantizedRow)
            dequantizedMatrix.append(dequantizedRow)

        return quantizedMatrix, dequantizedMatrix, scale

    @staticmethod
    def calculatePhaseDistortionDegrees(originalInPhase, originalQuadrature, quantizedInPhase, quantizedQuadrature):
        rowCount = len(originalInPhase)
        colCount = len(originalInPhase[0])
        phaseErrorSum = 0.0
        elementCount = rowCount * colCount

        for rowIndex in range(rowCount):
            for colIndex in range(colCount):
                trueAngle = math.atan2(originalQuadrature[rowIndex][colIndex], originalInPhase[rowIndex][colIndex])
                quantAngle = math.atan2(quantizedQuadrature[rowIndex][colIndex], quantizedInPhase[rowIndex][colIndex])
                angleDifference = abs(trueAngle - quantAngle)
                if angleDifference > math.pi:
                    angleDifference = (2.0 * math.pi) - angleDifference
                phaseErrorSum += (angleDifference * 180.0 / math.pi)

        return phaseErrorSum / elementCount

class ModelProfiler:
    @staticmethod
    def calculateParameterCount(model, exitStage="final"):
        parameterCount = 0
        patchEmbeddingParams = (model.patchEmbedding.patchVectorLength * model.patchEmbedding.hiddenDimension)
        patchEmbeddingParams += model.patchEmbedding.hiddenDimension
        patchEmbeddingParams += (model.patchEmbedding.tokenCount * model.patchEmbedding.hiddenDimension)
        parameterCount += patchEmbeddingParams

        activeLayers = 2 if exitStage == "early" else model.layerCount
        singleLayerParams = (
            (model.hiddenDimension * model.hiddenDimension * 4) +
            (model.hiddenDimension * (model.hiddenDimension * 2)) +
            ((model.hiddenDimension * 2) * model.hiddenDimension)
        )
        parameterCount += (activeLayers * singleLayerParams)

        headParams = (model.hiddenDimension * model.classCount) + model.classCount
        parameterCount += headParams
        return parameterCount

    @staticmethod
    def calculateMemoryFootprintMb(parameterCount, bitPrecision=32):
        bytesPerParameter = bitPrecision / 8.0
        totalBytes = parameterCount * bytesPerParameter
        return totalBytes / (1024.0 * 1024.0)

    @staticmethod
    def calculateTheoreticalFlops(model, tokenCount=16, exitStage="final"):
        activeLayers = 2 if exitStage == "early" else model.layerCount
        dim = model.hiddenDimension
        patchFlops = tokenCount * (model.patchEmbedding.patchVectorLength * dim)

        layerFlops = (
            (3 * tokenCount * dim * dim) +
            (tokenCount * tokenCount * dim) +
            (tokenCount * tokenCount * dim) +
            (tokenCount * dim * dim) +
            (2 * tokenCount * dim * (dim * 2))
        )
        headFlops = tokenCount * dim + (dim * model.classCount)
        totalFlops = patchFlops + (activeLayers * layerFlops) + headFlops
        return totalFlops / 1e6

