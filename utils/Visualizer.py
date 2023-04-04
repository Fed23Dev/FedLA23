from typing import List, Tuple

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from utils.DataExtractor import Extractor
from env.running_env import *
from utils.objectIO import pickle_load
from utils.objectIO import fetch_path_id


def text_info(title: str, x_label: str, y_label: str):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

# axis mode:
'''
    F:FLOPs
    A:Acc
    T:Turn
    R:Rate
'''

# form mode:
'''
    h:histogram-4
    k:kernel density estimation-2
    r:rug-1
'''


def adjust_font(ax, x, y, t):
    font_size = 30
    plt.legend(loc=2, prop={'size': 40})
    # ax.set_xticklabels(x_p, fontsize=font_size)
    # ax.set_yticklabels(y_p, fontsize=font_size)
    ax.set_ylabel(x, fontsize=font_size)
    ax.set_xlabel(y, fontsize=font_size)
    ax.set_title(t, fontsize=font_size)


class HRankBoard:
    def __init__(self):
        plt.switch_backend('agg')
        sns.set(style='darkgrid', color_codes=True, font_scale=1)

    def simp_rank_img(self, path: str):
        plt.figure(figsize=(20, 15))
        hrank_list = pickle_load(path)
        hr = []
        for rank in hrank_list:
            hr.extend(rank.tolist())
        layer_prefix = []
        for ind, rank in enumerate(hrank_list):
            layer_prefix.extend([f"conv{ind+1}" for _ in rank])

        df = pd.DataFrame(list(zip(hr, layer_prefix)), columns=['Rank', 'Layer'])
        sns.boxplot(x='Layer', y='Rank', data=df)
        label = fetch_path_id(path)
        plt.savefig(file_repo.new_img(f"{label}-box")[0])
        plt.clf()
        sns.swarmplot(x='Layer', y='Rank', data=df)
        plt.savefig(file_repo.new_img(f"{label}-scatter")[0])
        plt.clf()

    # key->value; key:str value:str
    # key:acc_name, value:acc_txt_filepath
    def simp_acc_compare_img(self, **paths):
        x_label = "Epoch"
        y_label = "Acc"
        title = "Acc Compare"

        pd_ori_data = []
        for key, path in paths.items():
            pickle_obj = pickle_load(path)
            sub_list = [[num, ind + 1, key] for ind, num in enumerate(pickle_obj)]
            pd_ori_data.extend(sub_list)
        df = pd.DataFrame(pd_ori_data, columns=[y_label, x_label, 'Class'])
        ax = sns.lineplot(data=df, x=x_label, y=y_label, hue="Class", style="Class", markers=True)
        # adjust_font(ax, x_label, y_label, title)
        plt.savefig(file_repo.new_img("Acc-compare")[0])
        plt.clf()


class VisBoard:
    ERROR_MESS1 = "The axis length cannot exceed 4 characters."
    ERROR_MESS2 = "The form length cannot exceed 3 characters."
    ERROR_MESS3 = "The mode str contains not defined character."
    map_dict = {"h": 4, "k": 2, "r": 1}
    AXIS_KEYS = ['F', 'A', 'I', 'R']
    KEYS = ['FLOPs', 'Acc', 'Interval', 'Rate']
    SNS_STYLE = "darkgrid"
    PALETTE = "rainbow"

    HIST_FLAG = 0
    KDE_FLAG = 1
    RUG_FLAG = 2

    def __init__(self):
        self.io = Extractor()
        # figSize and font
        sns.set_style(self.SNS_STYLE)
        # sns.set_palette(self.PALETTE)

    def check_axis_form(self, axis: str, form: str):
        assert len(axis) < 5, self.ERROR_MESS1
        assert all([ch in self.AXIS_KEYS for ch in axis]), self.ERROR_MESS3

        assert len(form) < 4, self.ERROR_MESS2
        assert all([ch in self.map_dict.keys() for ch in form]), self.ERROR_MESS3

    def map_bools(self, form: str) -> Tuple[bool, bool, bool]:
        ret = 0
        for ch in form:
            ret += self.map_dict[ch]
        hist_flag = False if ret < 4 else True
        kde_flag = False if ret // 2 % 2 == 0 else True
        rug_flag = False if ret % 2 == 0 else True
        return hist_flag, kde_flag, rug_flag

    def map_list(self, mode: str) -> List[int]:
        indices = []
        for ch in mode:
            indices.append(self.AXIS_KEYS.index(ch))
        return indices

    def init_graph_contex(self, axis: str, form: str) -> Tuple[pd.DataFrame, List, Tuple]:
        self.check_axis_form(axis, form)
        graph_type = self.map_bools(form)
        data_key = self.map_list(axis)
        data_frame = self.io.map_vars(data_key)
        return data_frame, data_key, graph_type

    def end_graph_contex(self, indices: List[int]):
        if len(indices) == 1:
            title = f"{self.KEYS[indices[0]]} Graph"
            text_info(title, self.KEYS[indices[0]], "frequency")
        else:
            title = f"{self.KEYS[indices[0]]}~{self.KEYS[indices[1]]} Graph"
            text_info(title, self.KEYS[indices[0]], self.KEYS[indices[1]])
        plt.savefig(file_repo.img(title))

    # 直方图和密度图必须二选一，rug可以按需选择
    def single_var_dist(self, axis: str, form: str = 'k'):
        df, indices, mode = self.init_graph_contex(axis, form)
        SINGLE = 0
        if mode[self.HIST_FLAG]:
            sns.displot(df, x=self.KEYS[indices[SINGLE]],
                        kde=mode[self.KDE_FLAG],
                        rug=mode[self.RUG_FLAG],
                        hue="class")
        else:
            sns.displot(df, x=self.KEYS[indices[SINGLE]],
                        kind="kde",
                        rug=[self.RUG_FLAG],
                        hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_dist"))

    # def single_var_sequence(self, axis: str, form: str = 'k'):
    #     df, indices, mode = self.init_graph_contex(axis, form)
    #     SINGLE = 0
    #     sns.lineplot(x=df.index,
    #                  y=df[self.KEYS[indices[SINGLE]]],
    #                  ci=None)
    #     plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

    def single_var_sequence(self, axis: str, form: str = 'k'):
        # df, indices, mode = self.init_graph_contex(axis, form)
        # SINGLE = 0
        # sns.lineplot(data=df, x=self.KEYS[indices[0]],
        #              y=self.KEYS[indices[1]],
        #              ci=None, hue="Interval")
        # text_info('Top-Acc1 Retrain', 'Retrain batch', 'Acc/%')
        # plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

        df, indices, mode = self.init_graph_contex(axis, form)
        SINGLE = 0
        sns.lineplot(x=df.index,
                     y=df[self.KEYS[indices[SINGLE]]],
                     ci=None)
        text_info('pairwise_distances', 'Round', 'Magnitude')
        plt.savefig(file_repo.img(f"{self.KEYS[indices[SINGLE]]}_sequencer_chart"))

    def double_vars_relation(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.lineplot(data=df, x=self.KEYS[indices[DOUBLE_X]],
                     y=self.KEYS[indices[DOUBLE_Y]],
                     ci=None, hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_relation"))

    def double_vars_dist(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.jointplot(data=df, x=self.KEYS[indices[DOUBLE_X]], y=self.KEYS[indices[DOUBLE_Y]], hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_dist"))

    def double_vars_regression(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        DOUBLE_X = 0
        DOUBLE_Y = 1
        sns.regplot(data=df, x=self.KEYS[indices[DOUBLE_X]], y=self.KEYS[indices[DOUBLE_Y]], hue="class")
        plt.savefig(file_repo.img(f"{self.KEYS[indices[DOUBLE_X]]}~{self.KEYS[indices[DOUBLE_Y]]}_dist"))

    def multi_vars_regression(self, axis: str, form: str):
        df, indices, mode = self.init_graph_contex(axis, form)
        sns.lmplot()
