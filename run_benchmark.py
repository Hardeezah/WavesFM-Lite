import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from wavesfm_lite.benchmark import BenchmarkRunner

def main():
    runner = BenchmarkRunner(testSampleCount=10)
    benchmarkResults = runner.runFullBenchmark()
    runner.printFormattedReport(benchmarkResults)

if __name__ == "__main__":
    main()
