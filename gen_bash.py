import glob
import os
from os.path import join
import numpy as np

DIR_DATA = join(".", "datasets")
DIR_SAVE = os.path.join(os.environ["HOME"], "Soroosh/results_full")
DATASETS = glob.glob(DIR_DATA + "/*.txt")
DATASETS = [f_name for f_name in DATASETS if "_test.txt" not in f_name]
DATASETS.sort()

rho = np.hstack([np.round(np.linspace(1, 9, 9) * 1e-3, 3),
                 np.round(np.linspace(1, 9, 9) * 1e-2, 2),
                 np.round(np.linspace(1, 9, 9) * 1e-1, 1)])
cv = 5
repeat = 100


def file_writer_1(b_file, command):
    """ command writer in a file """
    for dataset in DATASETS:
        for rho_1 in rho:
            f_name = dataset[11:-4] + "_" + command.split("\"")[1] + "_" \
                     + str(rho_1) + "_" + str(rho_1) + ".csv"
            f_name = os.path.join(DIR_SAVE, f_name)
            if not os.path.exists(f_name):
                print(command + "\"{}\" --rho {:0.3f} {:0.3f}".format(
                    dataset, rho_1, rho_1), file=b_file)
    return


def file_writer_2(b_file, command):
    """ command writer in a file """
    for dataset in DATASETS:
        for rho_1 in rho:
            for rho_2 in rho:
                f_name = dataset[11:-4] + "_" + command.split("\"")[1] + "_" \
                         + str(rho_1) + "_" + str(rho_2) + ".csv"
                f_name = os.path.join(DIR_SAVE, f_name)
                if not os.path.exists(f_name):
                    print(command + "\"{}\" --rho {:0.3f} {:0.3f}".format(
                        dataset, rho_1, rho_2), file=b_file)
    return


with open("./tester.sh", "w") as bash_file:
    for dataset in DATASETS:
        print("python tester.py --method \"freg\" --cv {} --repeat {} --dataset \"{}\"".format(cv, repeat, dataset),
              file=bash_file)
    for method in ["wass", "reg", "sparse"]:
        cmd = "python tester.py --method \"{}\" --cv {} --repeat {} --dataset ".format(
            method, cv, repeat)
        file_writer_1(bash_file, cmd)
    for method in ["FR", "KL", "mean"]:
        cmd = "python tester.py --method \"{}\" --cv {} --repeat {} --dataset ".format(
            method, cv, repeat)
        file_writer_2(bash_file, cmd)
