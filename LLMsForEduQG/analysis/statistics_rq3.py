# compute_rq3_stats.py

import os
import sys
from Metrics import Metrics
from Statistics import Statistics

if __name__ == "__main__":
    results_dir = sys.argv[1]   # pass the results folder as argument
    input_filename = sys.argv[2]  # the original test csv (needed for Statistics init)

    metrics = Metrics()
    stats = Statistics(input_filename, results_dir, metrics)
    stats.compute_statistics_cross_model()

    print("Done! Check: {}".format(stats.RESULTS_STATISTICS))