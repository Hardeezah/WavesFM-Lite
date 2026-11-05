import math
import random

class PatchEmbedding:
    def __init__(self, patchDimension=16, channelCount=2, hiddenDimension=128, inputHeight=64, inputWidth=64):
        self.patchDimension = patchDimension
        self.channelCount = channelCount
        self.hiddenDimension = hiddenDimension
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.patchesPerRow = inputWidth // patchDimension
        self.patchesPerColumn = inputHeight // patchDimension
        self.tokenCount = self.patchesPerRow * self.patchesPerColumn
        self.patchVectorLength = patchDimension * patchDimension * channelCount

        random.seed(101)
        scale = math.sqrt(2.0 / self.patchVectorLength)
        self.projectionWeights = [
            [random.gauss(0.0, scale) for _ in range(self.hiddenDimension)]
            for _ in range(self.patchVectorLength)
        ]
        self.projectionBias = [0.0 for _ in range(self.hiddenDimension)]
        self.positionalEmbeddings = [
            [random.gauss(0.0, 0.02) for _ in range(self.hiddenDimension)]
            for _ in range(self.tokenCount)
        ]

    def forward(self, radioGrid):
        extractedTokens = []
        tokenIndex = 0
        for patchRowIndex in range(self.patchesPerColumn):
            rowStart = patchRowIndex * self.patchDimension
            rowEnd = rowStart + self.patchDimension
            for patchColIndex in range(self.patchesPerRow):
                colStart = patchColIndex * self.patchDimension
                colEnd = colStart + self.patchDimension

                flattenedPatch = []
                for channelIndex in range(self.channelCount):
                    for rowIndex in range(rowStart, rowEnd):
                        for colIndex in range(colStart, colEnd):
                            flattenedPatch.append(radioGrid[channelIndex][rowIndex][colIndex])

                projectedToken = [self.projectionBias[dimIndex] for dimIndex in range(self.hiddenDimension)]
                for elementIndex in range(self.patchVectorLength):
                    inputValue = flattenedPatch[elementIndex]
                    for dimIndex in range(self.hiddenDimension):
                        projectedToken[dimIndex] += inputValue * self.projectionWeights[elementIndex][dimIndex]

                for dimIndex in range(self.hiddenDimension):
                    projectedToken[dimIndex] += self.positionalEmbeddings[tokenIndex][dimIndex]

                extractedTokens.append(projectedToken)
                tokenIndex += 1
        return extractedTokens

class MultiHeadAttentionBlock:
    def __init__(self, hiddenDimension=128, headCount=4):
        self.hiddenDimension = hiddenDimension
        self.headCount = headCount
        self.headDimension = hiddenDimension // headCount
        self.scale = 1.0 / math.sqrt(self.headDimension)

        random.seed(202)
        weightScale = math.sqrt(2.0 / hiddenDimension)
        self.queryWeights = [[random.gauss(0.0, weightScale) for _ in range(hiddenDimension)] for _ in range(hiddenDimension)]
        self.keyWeights = [[random.gauss(0.0, weightScale) for _ in range(hiddenDimension)] for _ in range(hiddenDimension)]
        self.valueWeights = [[random.gauss(0.0, weightScale) for _ in range(hiddenDimension)] for _ in range(hiddenDimension)]
        self.outWeights = [[random.gauss(0.0, weightScale) for _ in range(hiddenDimension)] for _ in range(hiddenDimension)]

        self.feedForwardWeights1 = [[random.gauss(0.0, weightScale) for _ in range(hiddenDimension * 2)] for _ in range(hiddenDimension)]
        self.feedForwardWeights2 = [[random.gauss(0.0, math.sqrt(2.0 / (hiddenDimension * 2))) for _ in range(hiddenDimension)] for _ in range(hiddenDimension * 2)]

    def forward(self, inputTokens):
        tokenCount = len(inputTokens)
        projectedQueries = self.matrixMultiply(inputTokens, self.queryWeights)
        projectedKeys = self.matrixMultiply(inputTokens, self.keyWeights)
        projectedValues = self.matrixMultiply(inputTokens, self.valueWeights)

        attentionScores = [[0.0 for _ in range(tokenCount)] for _ in range(tokenCount)]
        for tokenIndexA in range(tokenCount):
            for tokenIndexB in range(tokenCount):
                dotProduct = 0.0
                for dimIndex in range(self.hiddenDimension):
                    dotProduct += projectedQueries[tokenIndexA][dimIndex] * projectedKeys[tokenIndexB][dimIndex]
                attentionScores[tokenIndexA][tokenIndexB] = dotProduct * self.scale

        attentionWeights = []
        for tokenIndexA in range(tokenCount):
            rowValues = attentionScores[tokenIndexA]
            maxVal = max(rowValues)
            expValues = [math.exp(val - maxVal) for val in rowValues]
            sumExp = sum(expValues)
            normalizedRow = [val / sumExp for val in expValues]
            attentionWeights.append(normalizedRow)

        contextTokens = [[0.0 for _ in range(self.hiddenDimension)] for _ in range(tokenCount)]
        for tokenIndexA in range(tokenCount):
            for tokenIndexB in range(tokenCount):
                weight = attentionWeights[tokenIndexA][tokenIndexB]
                for dimIndex in range(self.hiddenDimension):
                    contextTokens[tokenIndexA][dimIndex] += weight * projectedValues[tokenIndexB][dimIndex]

        projectedAttention = self.matrixMultiply(contextTokens, self.outWeights)

        residual1 = []
        for tokenIndex in range(tokenCount):
            residualRow = []
            for dimIndex in range(self.hiddenDimension):
                residualRow.append(inputTokens[tokenIndex][dimIndex] + projectedAttention[tokenIndex][dimIndex])
            residual1.append(residualRow)

        hiddenIntermediate = self.matrixMultiply(residual1, self.feedForwardWeights1)
        for tokenIndex in range(tokenCount):
            for dimIndex in range(self.hiddenDimension * 2):
                value = hiddenIntermediate[tokenIndex][dimIndex]
                hiddenIntermediate[tokenIndex][dimIndex] = 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))

        feedForwardOut = self.matrixMultiply(hiddenIntermediate, self.feedForwardWeights2)

        outputTokens = []
        for tokenIndex in range(tokenCount):
            outputRow = []
            for dimIndex in range(self.hiddenDimension):
                outputRow.append(residual1[tokenIndex][dimIndex] + feedForwardOut[tokenIndex][dimIndex])
            outputTokens.append(outputRow)
        return outputTokens

    def matrixMultiply(self, matrixA, matrixB):
        rowCountA = len(matrixA)
        colCountA = len(matrixA[0])
        colCountB = len(matrixB[0])
        result = [[0.0 for _ in range(colCountB)] for _ in range(rowCountA)]
        for rowIndex in range(rowCountA):
            for innerIndex in range(colCountA):
                valueA = matrixA[rowIndex][innerIndex]
                if valueA != 0.0:
                    for colIndex in range(colCountB):
                        result[rowIndex][colIndex] += valueA * matrixB[innerIndex][colIndex]
        return result

class OutputClassifierHead:
    def __init__(self, hiddenDimension=128, classCount=4, seedValue=303):
        self.hiddenDimension = hiddenDimension
        self.classCount = classCount
        random.seed(seedValue)
        weightScale = math.sqrt(2.0 / hiddenDimension)
        self.headWeights = [[random.gauss(0.0, weightScale) for _ in range(classCount)] for _ in range(hiddenDimension)]
        self.headBias = [0.0 for _ in range(classCount)]

    def forward(self, tokenSequence):
        pooledRepresentation = [0.0 for _ in range(self.hiddenDimension)]
        tokenCount = len(tokenSequence)
        for token in tokenSequence:
            for dimIndex in range(self.hiddenDimension):
                pooledRepresentation[dimIndex] += token[dimIndex] / tokenCount

        logits = [self.headBias[classIndex] for classIndex in range(self.classCount)]
        for dimIndex in range(self.hiddenDimension):
            pooledVal = pooledRepresentation[dimIndex]
            for classIndex in range(self.classCount):
                logits[classIndex] += pooledVal * self.headWeights[dimIndex][classIndex]

        maxLogit = max(logits)
        expLogits = [math.exp(val - maxLogit) for val in logits]
        sumExp = sum(expLogits)
        probabilities = [val / sumExp for val in expLogits]
        return probabilities

class WavesFMLiteModel:
    def __init__(self, layerCount=4, hiddenDimension=128, classCount=4):
        self.layerCount = layerCount
        self.hiddenDimension = hiddenDimension
        self.classCount = classCount
        self.patchEmbedding = PatchEmbedding(patchDimension=16, channelCount=2, hiddenDimension=hiddenDimension)
        self.encoderLayers = [MultiHeadAttentionBlock(hiddenDimension=hiddenDimension) for _ in range(layerCount)]
        self.earlyExitHead = OutputClassifierHead(hiddenDimension=hiddenDimension, classCount=classCount, seedValue=404)
        self.finalExitHead = OutputClassifierHead(hiddenDimension=hiddenDimension, classCount=classCount, seedValue=505)

    def forward(self, radioGrid, exitStage="all"):
        tokens = self.patchEmbedding.forward(radioGrid)
        earlyExitProbabilities = None
        finalExitProbabilities = None

        for layerIndex in range(self.layerCount):
            tokens = self.encoderLayers[layerIndex].forward(tokens)
            if layerIndex == 1:
                earlyExitProbabilities = self.earlyExitHead.forward(tokens)
                if exitStage == "early":
                    return earlyExitProbabilities

        finalExitProbabilities = self.finalExitHead.forward(tokens)
        if exitStage == "final":
            return finalExitProbabilities
        return earlyExitProbabilities, finalExitProbabilities
