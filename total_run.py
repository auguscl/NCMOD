import sys
import os
import utils
import mvod_main
from competitors.iforest import if_main
from competitors.ocsvm import os_main
from competitors.hoad import hoad_main
from competitors.crmod import dmod_main, crmod_main
from competitors.muvad import muvad_main
from competitors.moddis import moddis_main
from itertools import combinations, permutations

import matplotlib
import numpy as np
import time


def main():

    data_root_path = 'datasets/'

    fo_other = open("results_other.csv", "w")
    fo_our = open("results_our.csv", "w")

    n_objects = 1000

    # for dim in (list(range(10, 55, 5)) + list(range(100, 550, 50))):
    for dataset_name in ['mnist', 'reuters', 'ttc']:  # mnist, reuters, ttc, test
        for num_views in [2, 3]:
            for (rate_attribute_outlier, rate_class_outlier, rate_class_attribute_outlier) \
                    in list(permutations([0.02, 0.05, 0.08], 3)):

                # file_name = 'syn-1000-' + str(dim) + '-10-10-10'

                file_name = dataset_name + '-' + str(n_objects) + '-' + str(num_views) + '-' \
                            + str(rate_attribute_outlier) + '-' + str(rate_class_outlier) + '-' \
                            + str(rate_class_attribute_outlier)

                # file_name = 'mnist-1000-3-0.05-0.08-0.02'
                utils.set_num_views(num_views)

                od_data = np.loadtxt(data_root_path + file_name + '.csv', delimiter=',')
                xs = od_data[:, 0:-1]
                xs = xs.astype(np.float32)
                labels = od_data[:, -1]
                labels = labels.astype(np.int8)

                utils.set_dim(xs.shape[1])

                print('fea num:', xs.shape[1])
                utils.set_outlier_ratio(np.sum(labels) / xs.shape[0])

                matplotlib.font_manager._rebuild()
                #
                # for num_neibs in [4, 6, 10, 12, 14]:
                # for module_weight in [0.25, 0.5, 2, 4]:
                for albation_set in [(0, 1), (1, 0)]:
                    auc, f1 = mvod_main.run(data_root_path, file_name, 20, 0.0001, 8, 1, albation_set)
                    fo_our.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # time.sleep(10)

                # auc, f1 = os_main.run(data_root_path, file_name)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # time.sleep(10)
                #
                # auc, f1 = hoad_main.run(data_root_path, file_name)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # # time.sleep(10)
                #
                # auc, f1 = dmod_main.run(data_root_path, file_name)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # # time.sleep(30)
                # #
                # auc, f1 = crmod_main.run(data_root_path, file_name)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # # time.sleep(30)
                #
                # auc, f1 = muvad_main.run(data_root_path, file_name)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # time.sleep(10)

                # auc, f1 = moddis_main.run(data_root_path, file_name, 16, 0.0001)
                # fo_other.write(str(auc.round(3)) + ',' + str(f1.round(3)) + ',')
                # # time.sleep(10)

                # fo_other.write('\n')
                # fo_other.flush()
                fo_our.write('\n')
                fo_our.flush()
        # fo_other.write('\n')
        fo_our.write('\n')

    fo_other.close()
    fo_our.close()

    return


if __name__ == '__main__':
    main()
