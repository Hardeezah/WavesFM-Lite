import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wavesfm_lite.dataset import WirelessSignalDataset
from wavesfm_lite.model import WavesFMLiteModel
from wavesfm_lite.quantizer import Int8Quantizer, ModelProfiler

class TestWavesFMLitePipeline(unittest.TestCase):
    def testDatasetGeneration(self):
        dataset = WirelessSignalDataset(sampleCount=5, subcarrierCount=64, symbolCount=64)
        self.assertEqual(len(dataset.dataTensors), 5)
        self.assertEqual(len(dataset.dataTensors[0]), 2)
        self.assertEqual(len(dataset.dataTensors[0][0]), 64)
        self.assertEqual(len(dataset.dataTensors[0][0][0]), 64)
        self.assertEqual(len(dataset.labelIndices), 5)

    def testModelPatchEmbeddingAndForward(self):
        dataset = WirelessSignalDataset(sampleCount=2, subcarrierCount=64, symbolCount=64)
        model = WavesFMLiteModel(layerCount=4, hiddenDimension=128, classCount=4)
        grid = dataset.dataTensors[0]

        earlyProbabilities = model.forward(grid, exitStage="early")
        self.assertEqual(len(earlyProbabilities), 4)
        self.assertAlmostEqual(sum(earlyProbabilities), 1.0, places=4)

        finalProbabilities = model.forward(grid, exitStage="final")
        self.assertEqual(len(finalProbabilities), 4)
        self.assertAlmostEqual(sum(finalProbabilities), 1.0, places=4)

    def testInt8QuantizationAndPhaseError(self):
        matrix = [[0.85, -0.42], [0.12, 0.99]]
        quantMatrix, dequantMatrix, scale = Int8Quantizer.quantizeMatrix(matrix)

        self.assertGreater(scale, 0.0)
        self.assertEqual(len(quantMatrix), 2)
        self.assertEqual(len(quantMatrix[0]), 2)
        for row in quantMatrix:
            for val in row:
                self.assertTrue(-128 <= val <= 127)

        inPhaseOriginal = [[1.0, 0.0], [0.5, -0.5]]
        quadratureOriginal = [[0.0, 1.0], [0.5, 0.5]]
        _, dequantI, _ = Int8Quantizer.quantizeMatrix(inPhaseOriginal)
        _, dequantQ, _ = Int8Quantizer.quantizeMatrix(quadratureOriginal)

        phaseError = Int8Quantizer.calculatePhaseDistortionDegrees(
            inPhaseOriginal, quadratureOriginal, dequantI, dequantQ
        )
        self.assertLess(phaseError, 3.0)

    def testProfilerCalculations(self):
        model = WavesFMLiteModel(layerCount=4, hiddenDimension=128, classCount=4)
        earlyParams = ModelProfiler.calculateParameterCount(model, exitStage="early")
        finalParams = ModelProfiler.calculateParameterCount(model, exitStage="final")
        self.assertLess(earlyParams, finalParams)

        earlyMemory = ModelProfiler.calculateMemoryFootprintMb(earlyParams, bitPrecision=8)
        finalMemory = ModelProfiler.calculateMemoryFootprintMb(finalParams, bitPrecision=32)
        self.assertLess(earlyMemory, finalMemory)

if __name__ == "__main__":
    unittest.main()
