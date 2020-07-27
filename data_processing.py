# # Install TensorFlow
# from __future__ import absolute_import, division, print_function, unicode_literals
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
#
# # time series
# from statsmodels.tsa.arima_model import ARIMA
# from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import scipy.optimize

from tools import CustomTool
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.colors as mcolors



tool = CustomTool()


def calculate_1d_mean(_data):
    sum = 0
    count = 0
    for val in _data:
        if np.isfinite(val):
            sum += val
            count += 1
    if count == 0:
        return np.nan
    return sum/count


def nanmean(_data, _axis=0):
    mean_list = []
    if _axis == 0:
        for sublist_idx in range(_data.shape[1]):
            annual_avg = calculate_1d_mean(_data[:, sublist_idx])
            mean_list.append(annual_avg)
    if _axis == 1:
        for sublist_idx in range(_data.shape[0]):
            annual_avg = calculate_1d_mean(_data[sublist_idx, :])
            mean_list.append(annual_avg)
    return np.array(mean_list)


def plot_each(_nd_columns, _nd_data, _idx_list, log_transform=True):
    _labels = _nd_columns[_idx_list]
    _exp_data = _nd_data[:, _idx_list].astype(float)
    if log_transform:
        _exp_data = np.log(_nd_data[:, _idx_list].astype(float))
    x_pos = list(range(len(_labels)))
    plt.figure(figsize=(8, 6))
    plt.xticks(x_pos, _labels, rotation=90, ha='center')
    for cty_idx in range(_exp_data.shape[0]):
        plt_data = _exp_data[cty_idx, :]
        # plt.plot(x_pos, plt_data, marker='o')
        plt.plot(x_pos, plt_data)

    # plt.plot('x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot('x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot('x', 'y2', data=df, marker='', color='olive', linewidth=2)
    # plt.plot('x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    # plt.legend()

    plt.show()


def plot_average(_nd_columns, _nd_data, _idx_list, log_transform=True):
    _labels = _nd_columns[_idx_list]
    _exp_data = _nd_data[:, _idx_list].astype(float)
    if log_transform:
        _exp_data = np.log(_nd_data[:, _idx_list].astype(float))
    x_pos = list(range(len(_labels)))
    plt.figure(figsize=(8, 6))
    plt.xticks(x_pos, _labels, rotation=90, ha='center')

    plt_data = nanmean(_exp_data, _axis=0)
    plt.plot(x_pos, plt_data, marker='o')

    plt.show()


def plot_all(_nd_columns, _nd_data, _idx_list, log_transform=True):
    _labels = _nd_columns[_idx_list]
    _exp_data = _nd_data[:, _idx_list].astype(float)
    if log_transform:
        _exp_data = np.log(_nd_data[:, _idx_list].astype(float))
    x_pos = list(range(len(_labels)))
    plt.figure(figsize=(8, 6))
    plt.xticks(x_pos, _labels, rotation=90, ha='center')

    for cty_idx in range(_exp_data.shape[0]):
        plt_data = _exp_data[cty_idx, :]
        plt.plot(x_pos, plt_data)

    # line for the average
    plt_data = nanmean(_exp_data, _axis=0)
    plt.plot(x_pos, plt_data, marker='o')

    plt.show()


def plot_specific(_nd_columns, _nd_data, _idx_list, target_index, log_transform=True, add_average=False):
    _labels = _nd_columns[_idx_list]
    _exp_data = _nd_data[:, _idx_list].astype(float)
    if log_transform:
        _exp_data = np.log(_nd_data[:, _idx_list].astype(float))
    x_pos = list(range(len(_labels)))
    plt.figure(figsize=(8, 6))
    plt.xticks(x_pos, _labels, rotation=90, ha='center')

    try:
        plt_data = _exp_data[target_index, :]
        plt.plot(x_pos, plt_data)
    except:
        pass

    # line for the average
    plt_data = nanmean(_exp_data, _axis=0)
    plt.plot(x_pos, plt_data, marker='o')

    plt.show()


def retrieve_non_nan_years(_columns, _data):
    new_columns = []
    new_data = []
    new_x_pos = []
    flipped_cols = np.flip(_columns)
    flipped_data = np.flip(_data)

    valid_flag = False
    for i in range(len(flipped_data)):
        annual_val = flipped_data[i]
        if np.isfinite(annual_val):
            valid_flag = True
            new_columns.append(flipped_cols[i])
            new_data.append(annual_val)
            new_x_pos.append(i)
        elif valid_flag:
            break

    return np.flip(new_columns), np.flip(new_data), new_x_pos


def retrieve_non_nan_years_and_hundredmore_case(_columns, _data, _thrshd=100):
    new_columns = []
    new_data = []
    new_x_pos = []
    flipped_cols = np.flip(_columns)
    flipped_data = np.flip(_data)

    valid_flag = False
    for i in range(len(flipped_data)):
        annual_val = flipped_data[i]
        if np.isfinite(annual_val) and annual_val >= _thrshd:
            valid_flag = True
            new_columns.append(flipped_cols[i])
            new_data.append(annual_val)
            new_x_pos.append(i)
        elif valid_flag:
            break

    return np.flip(new_columns), np.flip(new_data), new_x_pos


def create_x_tick(time_data):
    list_converted_date_txt = []
    dim1, dim2 = time_data.shape
    for i in range(dim1):
        converted_date_txt = ""
        for j in range(dim2):
            converted_date_txt = converted_date_txt + str(time_data[i, j]) + " "
        list_converted_date_txt.append(converted_date_txt)
    return np.array(list_converted_date_txt)


def merge_flags(flag_list):
    merged_flags = None
    for flags in flag_list:
        if merged_flags is None:
            merged_flags = np.array(flags)
        else:
            merged_flags = merged_flags + flags
    merged_flags = merged_flags > 0
    return merged_flags


def extract_PM25_data(nd_data, fixed_year, fixed_month, fixed_day, fixed_hour, data_index):
    original_data = nd_data.copy()
    if fixed_hour is not None:
        flag_list = []
        for hour in fixed_hour:
            flag_list.append(nd_data[:, 4]==hour)
        nd_data = nd_data[merge_flags(flag_list)]
    if fixed_day is not None:
        flag_list = []
        for day in fixed_day:
            flag_list.append(nd_data[:, 3]==day)
        nd_data = nd_data[merge_flags(flag_list)]
    if fixed_month is not None:
        flag_list = []
        for month in fixed_month:
            flag_list.append(nd_data[:, 2]==month)
        nd_data = nd_data[merge_flags(flag_list)]
    if fixed_year is not None:
        flag_list = []
        for year in fixed_year:
            flag_list.append(nd_data[:, 1]==year)
        nd_data = nd_data[merge_flags(flag_list)]

    return nd_data


def impute_missing(original_data):
    float_data = original_data.astype(float)
    nan_count = np.sum(np.isnan(float_data))
    non_nan_count = float_data.size - nan_count
    if non_nan_count > 2 and nan_count > 0:
        # find the nan indices
        nan_indices = np.array(list(range(float_data.size)))[np.isnan(float_data)]
        imp_vals = []
        for nan_index in nan_indices:
            imp_val = knn_impute(float_data, nan_index)
            imp_vals.append(imp_val)
        for i in range(len(nan_indices)):
            nan_index = nan_indices[i]
            imp_val = imp_vals[i]
            float_data[nan_index] = imp_val
    return float_data


def linear_intepretation(val1, val2, post_val=True):
    if post_val:
        return val2 - (val1 - val2)
    else:
        return val1 - (val2 - val1)


def non_linear_intepretation(val1, val2, post_val=True):
    if post_val:
        if val1 == 0:
            return 0
        return math.pow(val2, 2) / val1
    else:
        if val2 == 0:
            return 0
        return math.pow(val1, 2) / val2


def parse_date(date_str, date_pattern='%m/%d/%Y'):
    return datetime.datetime.strptime(date_str, date_pattern)


def knn_impute(_data, current_pointer, k=5):
    if not np.isnan(_data[current_pointer]):
        return _data[current_pointer]

    last_index = len(_data) - 1
    if current_pointer == 0:
        if not np.isnan(_data[1]) and not np.isnan(_data[2]):
            imp_val = linear_intepretation(_data[1], _data[2], post_val=False)
        else:
            imp_val = np.nanmean(_data)
    elif current_pointer == last_index:
        if not np.isnan(_data[last_index - 1]) and not np.isnan(_data[last_index - 2]):
            imp_val = linear_intepretation(_data[last_index - 2], _data[last_index - 1], post_val=True)
        else:
            imp_val = np.nanmean(_data)
    else:
        dist = 1
        candidates = []
        weights = []
        prev_idx = current_pointer - dist
        next_idx = current_pointer + dist
        prev_val = _data[prev_idx]
        next_val = _data[next_idx]
        if not np.isnan(prev_val):
            candidates.append(prev_val)
            weights.append(1 / dist / dist)
        if not np.isnan(next_val):
            candidates.append(next_val)
            weights.append(1 / dist / dist)

        while len(candidates) < k:
            dist = dist + 1
            prev_idx = current_pointer - dist
            next_idx = current_pointer + dist
            if not prev_idx < 0 and not np.isnan(_data[prev_idx]):
                candidates.append(_data[prev_idx])
                weights.append(1 / dist / dist)
            if not next_idx > last_index and not np.isnan(_data[next_idx]):
                candidates.append(_data[next_idx])
                weights.append(1 / dist / dist)
            if prev_idx < 0 and next_idx > last_index:
                break
        imp_val = np.sum(np.array(candidates) * np.array(weights)) / np.sum(weights)

    if imp_val < 0:
        imp_val = 0
    return imp_val


def load_data(_data_path):
    time_series_df = tool.read_from_file(_data_path)

    return time_series_df


def find_index_by_input(_input, list_index, list_name):
    try:
        if isinstance(int(_input), int):
            return int(_input)
    except:
        sub_idx = []
        sub_name = []
        for idx in list_index:
            if _input.lower() in list_name[idx].lower():
                sub_idx.append(idx)
                sub_name.append(list_name[idx])
        if len(sub_idx) == 0:
            _input = input("No match, please type again:")
            user_index = find_index_by_input(_input, list_index, list_name)
        elif len(sub_idx) == 1:
            return sub_idx[0]
        else:
            for idx in sub_idx:
                print(str(idx) + ": " + list_name[idx])
            user_input = input("Type index:")
            while len(user_input) == 0:
                user_input = input("Type index:")

            try:
                if isinstance(int(user_input), int):
                    return int(user_input)
            except:
                user_index = find_index_by_input(user_input, list_index, list_name)
        return user_index


def CoV(_df, _df_dth, _df_rcv):
    try:
        if input("Use log data? (T or F)").lower() == 't':
            log_toggle = True
        else:
            log_toggle = False
    except:
        log_toggle = False

    try:
        if input("Use case threshold? (T or F)").lower() == 't':
            hundred_toggle = True
            case_threshold = int(input("Case threshold: "))
        else:
            hundred_toggle = False
    except:
        hundred_toggle = True
        case_threshold = 100

    try:
        if input("Run world cases without China? (T or F)").lower() == 't':
            world_rest_toggle = True
        else:
            world_rest_toggle = False
    except:
        world_rest_toggle = True

    if world_rest_toggle:
        rest_index = _df['Country/Region'].map(lambda x: 'China' not in x)
        rest_df = _df[rest_index].sum()
        rest_df_dth = _df_dth[rest_index].sum()
        rest_df_rcv = _df_rcv[rest_index].sum()
        nd_columns = rest_df.index.values
        nd_data = rest_df.values
        nd_dth_data = rest_df_dth.values
        nd_rcv_data = rest_df_rcv.values

        if log_toggle:
            case_threshold = math.log10(case_threshold)
        idx_list = []
        datetime_pattern = '%m/%d/%y'
        for clm_idx in range(nd_columns.size):
            try:
                the_date = parse_date(nd_columns[clm_idx], datetime_pattern)
                if isinstance(the_date, datetime.datetime):
                    idx_list.append(clm_idx)
            except:
                pass
        labels = nd_columns[idx_list]
        if log_toggle:
            series = np.log10(nd_data[idx_list].astype(float))
            series_dth = np.log10(nd_dth_data[idx_list].astype(float))
            series_rcv = np.log10(nd_rcv_data[idx_list].astype(float))
            series[series == -np.inf] = 0
            series_dth[series_dth == -np.inf] = 0
            series_rcv[series_rcv == -np.inf] = 0
        else:
            series = nd_data[idx_list].astype(float)
            series_dth = nd_dth_data[idx_list].astype(float)
            series_rcv = nd_rcv_data[idx_list].astype(float)
        if hundred_toggle:
            len_labels = len(labels)
            labels, series, x_pos = retrieve_non_nan_years_and_hundredmore_case(labels, series, _thrshd=case_threshold)
            series_dth = series_dth[len_labels + (np.flip(x_pos) + 1) * -1]
            series_rcv = series_rcv[len_labels + (np.flip(x_pos) + 1) * -1]
        else:
            labels, series, x_pos = retrieve_non_nan_years(labels, series)
            xx1, series_dth, xx3 = retrieve_non_nan_years(labels, series_dth)
            xx1, series_rcv, xx3 = retrieve_non_nan_years(labels, series_rcv)

        loop = True
        first_plot = True
        while(loop):
            if first_plot:
                plot_indices = input('What to plot? 1:total, 2:death, 3:recovery. For all, type "1,2,3" ')
                plot_flags = [1, 2, 3]
                if len(plot_indices) != 0:
                    try:
                        plot_flags = list(map(int, plot_indices.split(',')))
                    except:
                        pass
            if 1 in plot_flags:
                plt.plot(x_pos, series, 'b.-')
            if 2 in plot_flags:
                plt.plot(x_pos, series_dth, 'r.-')
            if 3 in plot_flags:
                plt.plot(x_pos, series_rcv, 'g.-')
            plt.xticks(x_pos, labels, rotation=90, ha='center')
            plt.title("Cases outside China")
            plt.show()
            plot_indices = input('What to plot? 1:total, 2:death, 3:recovery. For all, type "1,2,3" ')
            while len(plot_indices) == 0:
                plot_indices = input('Type next index or -1 (if you want to end this):')

            if len(plot_indices.strip()) != 0:
                plot_flags = [1, 2, 3]
                try:
                    plot_flags = list(map(int, plot_indices.split(',')))
                    if int(plot_indices) == -1:
                        loop = False
                except:
                    pass
                first_plot = False

    else:
        try:
            if input("Location to country level? (T or F)").lower() == 'f':
                country_toggle = False
            else:
                country_toggle = True
        except:
            country_toggle = True

        loop = True
        first_input = True
        df = _df
        nd_columns = df.columns.values

        if country_toggle:
            nd_location = np.unique(_df.values[:, 1])
            df = _df.groupby(nd_columns[1]).aggregate(sum)
            nd_data = df.values
            nd_dth_data = _df_dth.groupby(nd_columns[1]).aggregate(sum).values
            nd_rcv_data = _df_rcv.groupby(nd_columns[1]).aggregate(sum).values
            nd_columns = df.columns.values
        else:
            nd_location = [str(row[1]) + (", " + str(row[0]) if str(row[0]) != "nan" else "") for row in _df.values[:, :]]
            df = _df
            nd_data = df.values
            nd_dth_data = _df_dth.values
            nd_rcv_data = _df_rcv.values

        if log_toggle:
            case_threshold = math.log10(case_threshold)
        idx_list = []
        datetime_pattern = '%m/%d/%y'
        for clm_idx in range(nd_columns.size):
            try:
                the_date = parse_date(nd_columns[clm_idx], datetime_pattern)
                if isinstance(the_date, datetime.datetime):
                    idx_list.append(clm_idx)
            except:
                pass

        loc_idx = list(range(len(nd_location)))
        loc_name = nd_location
        _labels = nd_columns[idx_list]

        while(loop):
            if first_input:
                for locidx in loc_idx:
                    print(str(locidx) + ': ' + loc_name[locidx])
                input_end = input("Type country name or index: ")
                unit_index = find_index_by_input(input_end, loc_idx, loc_name)
            # series = np.log(nd_data[:, idx_list].astype(float))
            # series = nanmean(series, _axis=0)
            if log_toggle:
                series = np.log10(nd_data[unit_index, idx_list].astype(float))
                series_dth = np.log10(nd_dth_data[unit_index, idx_list].astype(float))
                series_rcv = np.log10(nd_rcv_data[unit_index, idx_list].astype(float))
                series[series == -np.inf] = 0
                series_dth[series_dth == -np.inf] = 0
                series_rcv[series_rcv == -np.inf] = 0
            else:
                series = nd_data[unit_index, idx_list].astype(float)
                series_dth = nd_dth_data[unit_index, idx_list].astype(float)
                series_rcv = nd_rcv_data[unit_index, idx_list].astype(float)
            if hundred_toggle:
                len_labels = len(_labels)
                labels, series, x_pos = retrieve_non_nan_years_and_hundredmore_case(_labels, series,
                                                                                    _thrshd=case_threshold)
                series_dth = series_dth[len_labels + (np.flip(x_pos) + 1) * -1]
                series_rcv = series_rcv[len_labels + (np.flip(x_pos) + 1) * -1]
            else:
                labels, series, x_pos = retrieve_non_nan_years(_labels, series)
                xx1, series_dth, xx3 = retrieve_non_nan_years(_labels, series_dth)
                xx1, series_rcv, xx3 = retrieve_non_nan_years(_labels, series_rcv)

            # x_pos = list(range(len(series)))
            plot_indices = input('What to plot? 1:total, 2:death, 3:recovery. For all, type "1,2,3" ')
            plot_flags = [1, 2, 3]
            if len(plot_indices) != 0:
                try:
                    plot_flags = list(map(int, plot_indices.split(',')))
                except:
                    pass
            if 1 in plot_flags:
                plt.plot(x_pos, series, 'b.-')
            if 2 in plot_flags:
                plt.plot(x_pos, series_dth, 'r.-')
            if 3 in plot_flags:
                plt.plot(x_pos, series_rcv, 'g.-')
            plt.xticks(x_pos, labels, rotation=90, ha='center')
            plt.title("Location: " + loc_name[unit_index])
            plt.show()
            first_input = False

            input_end = input('Type next index or -1 (if you want to end this):')
            while len(input_end) == 0:
                input_end = input('Type next index or -1 (if you want to end this):')

            if len(input_end.strip()) != 0:
                unit_index = find_index_by_input(input_end, loc_idx, loc_name)
                if unit_index == -1:
                    loop = False


file_name_cnf = "time_series_covid19_confirmed_global.csv"
file_name_dth = "time_series_covid19_deaths_global.csv"
file_name_rcv = "time_series_covid19_recovered_global.csv"
data_path = "../COVID-19/csse_covid_19_data/csse_covid_19_time_series/"

data_cnf = load_data(data_path + file_name_cnf)
data_dth = load_data(data_path + file_name_dth)
data_rcv = load_data(data_path + file_name_rcv)
CoV(data_cnf, data_dth, data_rcv)

# Taiwan: 34, Germany: 53
print("Finished")
