import csv
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import scipy.stats as ss
from bisect import bisect_left
from typing import List
from matplotlib_venn import venn3, venn3_circles
import plotly.express as px
from Metrics import Metrics
from pathlib import Path

class Statistics:
    metrics: Metrics
    INPUT_FILENAME = ""
    RESULTS_DIR = "results/"
    RESULTS_FILENAME = "results/generated_questions.csv"
    RESULTS_STATISTICS = "results/statistics.csv"
    RESULTS_SUMMARY = "results/summary.csv"

    def __init__(self,input_filename,results_dir,metrics):
        self.INPUT_FILENAME = input_filename
        self.RESULTS_DIR = results_dir
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.RESULTS_FILENAME = self.RESULTS_DIR+"/generated_questions.csv"
        self.RESULTS_STATISTICS = self.RESULTS_DIR+"/statistics.csv"
        self.RESULTS_SUMMARY = self.RESULTS_DIR + "/summary.csv"
        self.metrics = metrics

    def VD_A(self,treatment: List[float], control: List[float]):
        """
        Computes Vargha and Delaney A index
        A. Vargha and H. D. Delaney.
        A critique and improvement of the CL common language
        effect size statistics of McGraw and Wong.
        Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

        The formula to compute A has been transformed to minimize accuracy errors
        See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

        :param treatment: a numeric list
        :param control: another numeric list

        :returns the value estimate and the magnitude
        """
        m = len(treatment)
        n = len(control)

        if m != n:
            raise ValueError("Data d and f must have the same length")

        r = ss.rankdata(treatment + control)
        r1 = sum(r[0:m])

        # Compute the measure
        # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
        A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

        levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
        magnitude = ["negligible", "small", "medium", "large"]
        scaled_A = (A - 0.5) * 2

        magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
        estimate = A

        return estimate, magnitude

    def compute_statistics(self):
        df_all = pd.read_csv(self.RESULTS_FILENAME)

        # # summary file
        # summary_file = open(self.RESULTS_SUMMARY, 'w')
        # summary_writer = csv.writer(summary_file)
        # summary_writer.writerow(['model', 'prompt_id','metric', 'median', 'mean', 'min','max'])
        #statistics file
        statistics_file = open(self.RESULTS_STATISTICS, 'w', newline='', encoding='utf-8')
        statistics_writer = csv.writer(statistics_file)
        statistics_writer.writerow(
            ['model_id', 'metric', 'treatment', 'control', 'median_treatment', 'mean_treatment', 'median_control',
             'mean_control', 'wilcoxon', 'kendall', 'A12'])
        for mid in df_all['model_id'].unique():
            mid_indexes = df_all['model_id'] == mid
            df = df_all[mid_indexes]
            for metric in self.metrics.get_available_metrics():
                for treatment in df['prompt_id'].unique():
                    treatment_values = df[metric][df['prompt_id'] == treatment].values.astype(float)
                    # summary_writer.writerow(["gpt-4o-mini", treatment, metric,
                    #              "{:.2f}".format(np.median(treatment_values)),
                    #              "{:.2f}".format(np.mean(treatment_values)),
                    #              "{:.2f}".format(np.median(treatment_values)),
                    #              "{:.2f}".format(np.mean(treatment_values))])
                    for control in df['prompt_id'].unique():
                        if treatment == control:
                            continue
                        control_values = df[metric][df['prompt_id'] == control].values.astype(float)
                        wil_p = 0
                        ken_p = 0
                        A12 = 0
                        if len(treatment_values) > 0 and len(treatment_values) == len(
                                control_values) and not np.allclose(treatment_values,control_values):
                            _, wil_p = ss.wilcoxon(treatment_values, control_values)
                            _, ken_p = ss.kendalltau(treatment_values, control_values)
                            A12, _ = self.VD_A(treatment_values, control_values)
                        stats_row = [mid, metric, treatment, control,
                                     "{:.2f}".format(np.median(treatment_values)),
                                     "{:.2f}".format(np.mean(treatment_values)),
                                     "{:.2f}".format(np.median(control_values)), "{:.2f}".format(np.mean(control_values)),
                                     "{:.2f}".format(wil_p), "{:.2f}".format(ken_p), "{:.2f}".format(A12)]
                        statistics_writer.writerow(stats_row)
                        statistics_file.flush()

        statistics_file.close()

    def generate_summary(self):
        df = pd.read_csv(self.RESULTS_FILENAME)
        # # summary file
        summary_file = open(self.RESULTS_SUMMARY, 'w', newline='', encoding='utf-8')
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(['prompt_id','model_id'] + self.metrics.get_available_metrics())
        for pid in df['prompt_id'].unique():
            pid_indexes =df['prompt_id'] == pid
            df_pid = df[pid_indexes]
            for mid in df_pid['model_id'].unique():
                stats_row = [pid,mid]
                for metric in self.metrics.get_available_metrics():
                    treatment_values = df_pid[metric][df_pid['model_id'] == mid].values.astype(float)
                    stats_row.append("{:.2f}".format(np.mean(treatment_values)))
                summary_writer.writerow(stats_row)
                summary_file.flush()

        summary_file.close()

    def generate_plots(self):
        df = pd.read_csv(self.RESULTS_FILENAME)

        for metric in self.metrics.get_available_metrics():
            print("Metric: {}".format(metric))
            y_data = []
            x_data = []
            # df_metric = df[df['metric'] == metric]
            for pid in df['prompt_id'].unique():
                df_pid = df[df['prompt_id'] == pid]
                for mid in df['model_id'].unique():
                    df_mid = df_pid[df_pid['model_id'] == mid]
                    pid_mid_values = df_mid[metric].values
                    pid_mid_median = np.median(pid_mid_values)
                    pid_mid_mean = np.mean(pid_mid_values)
                    x_data.append(pid+"_"+mid)
                    y_data.append(list(pid_mid_values))
                    #fig.add_trace(go.Box(y=pid_values, name=pid, boxpoints='outliers')) #marker_color='#3D6270',
                    print("Prompt: {}, Model: {}, median: {}, mean: {}".format(pid, mid, pid_mid_median, pid_mid_mean))

            print(">>>")
            print(">>>")
            fig, ax = plt.subplots()
            ax.set_ylabel("{}".format(metric.upper()))
            ax.set_xlabel("Prompts + Model")
            ax.boxplot(y_data, tick_labels=x_data, showmeans=True)
            plt.style.use('default')
            # ax.set_xticklabels(x_data)
            plt.savefig(self.RESULTS_DIR+"/"+metric+".pdf")
            # fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=10, r=10, b=10, t=10),
            #                   xaxis_title='<b>Prompts</b>',
            #                   yaxis_title="<b>{}</b>".format(metric.upper()),
            #                   )
            #
            # fig.write_image(self.RESULTS_DIR+"/"+metric+".pdf")
