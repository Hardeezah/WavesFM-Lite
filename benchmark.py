import copy
import time
from wavesfm_lite.dataset import WirelessSignalDataset
from wavesfm_lite.model import WavesFMLiteModel
from wavesfm_lite.quantizer import Int8Quantizer, ModelProfiler

class BenchmarkRunner:
    def __init__(self, testSampleCount=10):
        self.testSampleCount = testSampleCount
        self.dataset = WirelessSignalDataset(sampleCount=testSampleCount, subcarrierCount=64, symbolCount=64)
        self.model = WavesFMLiteModel(layerCount=4, hiddenDimension=128, classCount=4)
        self.quantizedModel = self._buildQuantizedModelCopy(self.model)

    def _quantizeVector(self, vector):
        _, dequantizedMatrix, _ = Int8Quantizer.quantizeMatrix([vector])
        return dequantizedMatrix[0]

    def _buildQuantizedModelCopy(self, sourceModel):
        quantizedModel = copy.deepcopy(sourceModel)
        quantizedModel.patchEmbedding.projectionWeights = Int8Quantizer.quantizeMatrix(
            quantizedModel.patchEmbedding.projectionWeights
        )[1]
        quantizedModel.patchEmbedding.projectionBias = self._quantizeVector(
            quantizedModel.patchEmbedding.projectionBias
        )
        quantizedModel.patchEmbedding.positionalEmbeddings = Int8Quantizer.quantizeMatrix(
            quantizedModel.patchEmbedding.positionalEmbeddings
        )[1]

        weightMatrixNames = (
            "queryWeights",
            "keyWeights",
            "valueWeights",
            "outWeights",
            "feedForwardWeights1",
            "feedForwardWeights2"
        )
        for layer in quantizedModel.encoderLayers:
            for attrName in weightMatrixNames:
                setattr(layer, attrName, Int8Quantizer.quantizeMatrix(getattr(layer, attrName))[1])

        for head in (quantizedModel.earlyExitHead, quantizedModel.finalExitHead):
            head.headWeights = Int8Quantizer.quantizeMatrix(head.headWeights)[1]
            head.headBias = self._quantizeVector(head.headBias)

        return quantizedModel

    def forwardQuantized(self, radioGrid, exitStage="early"):
        quantizedInPhase = Int8Quantizer.quantizeMatrix(radioGrid[0])[1]
        quantizedQuadrature = Int8Quantizer.quantizeMatrix(radioGrid[1])[1]
        tokens = self.quantizedModel.patchEmbedding.forward([quantizedInPhase, quantizedQuadrature])
        tokens = Int8Quantizer.quantizeMatrix(tokens)[1]

        activeLayerCount = 2 if exitStage == "early" else self.quantizedModel.layerCount
        for layerIndex in range(activeLayerCount):
            tokens = self.quantizedModel.encoderLayers[layerIndex].forward(tokens)
            tokens = Int8Quantizer.quantizeMatrix(tokens)[1]

        selectedHead = self.quantizedModel.earlyExitHead if exitStage == "early" else self.quantizedModel.finalExitHead
        return selectedHead.forward(tokens)

    def _predictLabel(self, radioGrid, exitStage, isQuantized=False):
        probabilities = (
            self.forwardQuantized(radioGrid, exitStage=exitStage)
            if isQuantized
            else self.model.forward(radioGrid, exitStage=exitStage)
        )
        return probabilities.index(max(probabilities))

    def evaluateAccuracy(self, exitStage, isQuantized=False):
        correctCount = 0
        for sampleIndex in range(self.testSampleCount):
            radioGrid = self.dataset.dataTensors[sampleIndex]
            trueLabel = self.dataset.labelIndices[sampleIndex]
            predictedLabel = self._predictLabel(radioGrid, exitStage=exitStage, isQuantized=isQuantized)
            if predictedLabel == trueLabel:
                correctCount += 1

        return (correctCount / self.testSampleCount) * 100.0

    def evaluateTop1Agreement(self, exitStage="early"):
        matchCount = 0
        for sampleIndex in range(self.testSampleCount):
            radioGrid = self.dataset.dataTensors[sampleIndex]
            fp32Prediction = self._predictLabel(radioGrid, exitStage=exitStage, isQuantized=False)
            int8Prediction = self._predictLabel(radioGrid, exitStage=exitStage, isQuantized=True)
            if fp32Prediction == int8Prediction:
                matchCount += 1

        return (matchCount / self.testSampleCount) * 100.0

    def measureLatencyMs(self, exitStage, isQuantized=False, iterations=3):
        warmupGrid = self.dataset.dataTensors[0]
        if isQuantized:
            self.forwardQuantized(warmupGrid, exitStage=exitStage)
        else:
            self.model.forward(warmupGrid, exitStage=exitStage)

        startTime = time.perf_counter()
        for iterationIndex in range(iterations):
            grid = self.dataset.dataTensors[iterationIndex % self.testSampleCount]
            if isQuantized:
                self.forwardQuantized(grid, exitStage=exitStage)
            else:
                self.model.forward(grid, exitStage=exitStage)
        elapsedTime = time.perf_counter() - startTime

        return (elapsedTime / iterations) * 1000.0

    def computePhaseErrorDegrees(self):
        totalPhaseError = 0.0
        for sampleIndex in range(self.testSampleCount):
            sampleGrid = self.dataset.dataTensors[sampleIndex]
            originalI = sampleGrid[0]
            originalQ = sampleGrid[1]

            _, dequantizedI, _ = Int8Quantizer.quantizeMatrix(originalI)
            _, dequantizedQ, _ = Int8Quantizer.quantizeMatrix(originalQ)

            totalPhaseError += Int8Quantizer.calculatePhaseDistortionDegrees(
                originalI, originalQ, dequantizedI, dequantizedQ
            )

        return totalPhaseError / self.testSampleCount

    def runFullBenchmark(self):
        results = []

        paramsFull = ModelProfiler.calculateParameterCount(self.model, exitStage="final")
        memoryFull = ModelProfiler.calculateMemoryFootprintMb(paramsFull, bitPrecision=32)
        flopsFull = ModelProfiler.calculateTheoreticalFlops(self.model, exitStage="final")
        latencyFull = self.measureLatencyMs(exitStage="final", isQuantized=False)
        accuracyFull = self.evaluateAccuracy(exitStage="final", isQuantized=False)

        results.append({
            "name": "Full ViT (Layer 4)",
            "precision": "FP32",
            "weightsMb": round(memoryFull, 2),
            "flopsM": round(flopsFull, 1),
            "latencyMs": round(latencyFull, 2),
            "accuracy": round(accuracyFull, 1),
            "phaseError": 0.0,
            "agreementPct": 100.0
        })

        paramsEarly = ModelProfiler.calculateParameterCount(self.model, exitStage="early")
        memoryEarly = ModelProfiler.calculateMemoryFootprintMb(paramsEarly, bitPrecision=32)
        flopsEarly = ModelProfiler.calculateTheoreticalFlops(self.model, exitStage="early")
        latencyEarly = self.measureLatencyMs(exitStage="early", isQuantized=False)
        accuracyEarly = self.evaluateAccuracy(exitStage="early", isQuantized=False)

        results.append({
            "name": "Early-Exit (Layer 2)",
            "precision": "FP32",
            "weightsMb": round(memoryEarly, 2),
            "flopsM": round(flopsEarly, 1),
            "latencyMs": round(latencyEarly, 2),
            "accuracy": round(accuracyEarly, 1),
            "phaseError": 0.0,
            "agreementPct": 100.0
        })

        memoryInt8 = ModelProfiler.calculateMemoryFootprintMb(paramsEarly, bitPrecision=8)
        flopsInt8 = flopsEarly
        latencyInt8 = self.measureLatencyMs(exitStage="early", isQuantized=True)
        accuracyInt8 = self.evaluateAccuracy(exitStage="early", isQuantized=True)
        phaseInt8 = self.computePhaseErrorDegrees()
        agreementInt8 = self.evaluateTop1Agreement(exitStage="early")

        results.append({
            "name": "Early-Exit + INT8",
            "precision": "INT8",
            "weightsMb": round(memoryInt8, 2),
            "flopsM": round(flopsInt8, 1),
            "latencyMs": round(latencyInt8, 2),
            "accuracy": round(accuracyInt8, 1),
            "phaseError": round(phaseInt8, 2),
            "agreementPct": round(agreementInt8, 1)
        })

        return results

    def printFormattedReport(self, results):
        print("=" * 100)
        print("                        WAVESFM-LITE: 6G EDGE BENCHMARK REPORT")
        print("=" * 100)
        header = f"{'Configuration':<24} {'Precision':<11} {'Weights (MB)':<14} {'FLOPs (M)':<11} {'Latency (ms)':<14} {'Raw Acc':<10} {'Phase Err':<11}"
        print(header)
        print("-" * 100)
        for row in results:
            line = f"{row['name']:<24} {row['precision']:<11} {str(row['weightsMb']) + ' MB':<14} {str(row['flopsM']) + ' M':<11} {str(row['latencyMs']) + ' ms':<14} {str(row['accuracy']) + '%':<10} {str(row['phaseError']) + ' deg':<11}"
            print(line)
        print("-" * 100)

        memReduction = round((1.0 - (results[2]['weightsMb'] / results[0]['weightsMb'])) * 100.0, 1)
        flopsReduction = round((1.0 - (results[2]['flopsM'] / results[0]['flopsM'])) * 100.0, 1)
        earlyExitSpeedup = round(results[0]['latencyMs'] / max(0.01, results[1]['latencyMs']), 1)
        print(
            f"Weight Memory: -{memReduction}% | FLOPs: -{flopsReduction}% | "
            f"Early-Exit CPU Speedup: {earlyExitSpeedup}x | "
            f"FP32/INT8 Agreement: {results[2]['agreementPct']}% | "
            f"Mean Phase Error: {results[2]['phaseError']} deg"
        )
        print("Note: Untrained random weights. Simulated INT8 in pure Python has no SIMD kernels; speedup is early-exit depth skipping only.")
        print("=" * 100)
