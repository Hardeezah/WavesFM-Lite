import math
import random

class WirelessSignalDataset:
    def __init__(self, sampleCount=100, subcarrierCount=64, symbolCount=64, signalToNoiseRatioDb=15.0, seed=42):
        self.sampleCount = sampleCount
        self.subcarrierCount = subcarrierCount
        self.symbolCount = symbolCount
        self.signalToNoiseRatioDb = signalToNoiseRatioDb
        self.seed = seed
        self.modulationSchemes = [
            "BPSK",
            "QPSK",
            "8PSK",
            "16QAM"
        ]
        self.dataTensors = []
        self.labelIndices = []
        self.generateDataset()

    def generateDataset(self):
        random.seed(self.seed)
        noiseLinearRatio = 10.0 ** (-self.signalToNoiseRatioDb / 10.0)
        noiseSigma = math.sqrt(noiseLinearRatio / 2.0)

        for sampleIndex in range(self.sampleCount):
            classIndex = sampleIndex % len(self.modulationSchemes)
            self.labelIndices.append(classIndex)

            channelInPhase = [[0.0 for _ in range(self.symbolCount)] for _ in range(self.subcarrierCount)]
            channelQuadrature = [[0.0 for _ in range(self.symbolCount)] for _ in range(self.subcarrierCount)]

            rayleighFadingMagnitude = math.sqrt(-2.0 * math.log(max(1e-9, random.random())))
            dopplerFrequencyShift = 0.05 * (classIndex + 1)
            multipathDelayIndex = random.randint(1, 8)

            for subcarrierIndex in range(self.subcarrierCount):
                multipathPhase = (2.0 * math.pi * subcarrierIndex * multipathDelayIndex) / self.subcarrierCount
                for symbolIndex in range(self.symbolCount):
                    dopplerPhase = 2.0 * math.pi * dopplerFrequencyShift * symbolIndex
                    totalPhase = multipathPhase + dopplerPhase

                    symbolInPhase = math.cos(totalPhase) * rayleighFadingMagnitude
                    symbolQuadrature = math.sin(totalPhase) * rayleighFadingMagnitude

                    noiseInPhase = random.gauss(0.0, noiseSigma)
                    noiseQuadrature = random.gauss(0.0, noiseSigma)

                    channelInPhase[subcarrierIndex][symbolIndex] = symbolInPhase + noiseInPhase
                    channelQuadrature[subcarrierIndex][symbolIndex] = symbolQuadrature + noiseQuadrature

            self.dataTensors.append([channelInPhase, channelQuadrature])

    def getBatch(self, batchStartIndex, batchSize):
        endIndex = min(batchStartIndex + batchSize, self.sampleCount)
        batchTensors = self.dataTensors[batchStartIndex:endIndex]
        batchLabels = self.labelIndices[batchStartIndex:endIndex]
        return batchTensors, batchLabels

