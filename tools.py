import random
import time
from numpy import genfromtxt
import numpy as np
import numpy.core.defchararray as np_f
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


class CustomTool():
    is_print = False

    def read_from_file(self, datapath, delimeter=','):
        read_df = pd.read_csv(datapath)
        return read_df

    def parse_to_float(self, to_parse):
        try:
            if isinstance(to_parse, str):
                return_value = float(to_parse.replace(',', '.'))
            else:
                return_value = float(to_parse)
        except ValueError:
            return None
        else:
            return return_value

    def filter_out_non_numerical(self, data, lbl_idx):
        def is_float(val):
            try:
                float(val)
            except ValueError:
                return False
            else:
                return True

        # is_numeric_2 = lambda x: np.array(map(is_float, x))
        # is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))

        process_data = data.copy()
        rows, dimen = process_data.shape
        selected_indices = []
        # original_indices = []

        for i in range(dimen):  # check if numeric
            data_single_dim = process_data[:, i]
            check_numeric = True
            for index, x in np.ndenumerate(data_single_dim):
                if not is_float(x):
                    check_numeric = False

            if check_numeric:
                # index_to_append = i
                # if i >= lbl_idx:
                #     index_to_append = i + 1
                selected_indices.append(i)
                # original_indices.append(index_to_append)

        return selected_indices, process_data[:, selected_indices]

    def remove_nan_feature(self, data, labels, threshold=0,
                           matching_id=None, check_classwise=True, remove_bad_case=False, file_name=''):
        # missing threshold as percentage per class to keep the feature
        # missing values replaced by ___

        threshold = 0 if threshold < 0 else 100 if threshold > 100 else threshold
        counting_nan = np.isnan(data.T.tolist())
        dimen = counting_nan.shape[0]
        count = counting_nan.shape[1]
        selected_indices = []

        unique, counts = np.unique(labels, return_counts=True)

        if check_classwise:
            thr_counts = counts * threshold/100
        else:
            thr_count = count * threshold/100

        missing_rate = []
        total_missing_rate = []
        for i in range(dimen):
            check_is_kept = True
            processed_data = data[:, i]

            if check_classwise:
                temp_max_missing = 0
                class_idx = 0
                for cls in unique:
                    if check_is_kept or to_plot:
                        subset_x = np.array(processed_data[labels == cls])
                        nan_count = np.sum(np.isnan(subset_x.tolist()))
                        temp_missing = round((nan_count / counts[class_idx])*100, 3)
                        if temp_missing > temp_max_missing:
                            temp_max_missing = temp_missing
                        if nan_count > thr_counts[class_idx]:
                            check_is_kept = False

                    class_idx = class_idx + 1
                missing_rate.append(temp_max_missing)
                total_missing_rate.append(round((np.sum(np.isnan(subset_x.tolist())) / count)*100, 3))
            else:
                nan_count = np.sum(np.isnan(processed_data.tolist()))
                if nan_count > thr_count:
                    check_is_kept = False

            if check_is_kept:
                selected_indices.append(i)

        selected_data = data[:, selected_indices]
        new_dimen = selected_data.shape[1]
        case_missing_list = []
        if remove_bad_case:
            case_keep_indexes = []
            # case_threshold = threshold/2
            case_threshold = 15
            for i in range(count):
                subset_y = np.array(selected_data[i, :])
                nan_count = np.sum(np.isnan(subset_y.tolist()))
                case_missing = round((nan_count / new_dimen) * 100, 3)
                case_missing_list.append(case_missing)
                # print(str(case_missing) + '%')
                if case_missing < case_threshold:
                    case_keep_indexes.append(i)

            selected_data = selected_data[case_keep_indexes, :]
            selected_labels = labels[case_keep_indexes]
            matching_id = matching_id[case_keep_indexes]
        # show missing intensity of kept features
        # self.check_missing_intensity(selected_data)

        if remove_bad_case:
            return selected_indices, selected_data, selected_labels, matching_id
        else:
            return selected_indices, selected_data

    def remove_missing_feature_case(self, data, labels, original_columns, threshold=0, check_classwise=True, feature_prob=0.5):
        # missing threshold as percentage per class to keep the feature
        # missing values replaced by ___

        threshold = 0 if threshold < 0 else 100 if threshold > 100 else threshold
        counting_nan = np.isnan(data.T.tolist())
        dimen = counting_nan.shape[0]
        count = counting_nan.shape[1]
        selected_indices = []
        self.check_missing_dimension(data, axis=0)
        choice_prob = random.uniform(0, 1)

        feature_number = dimen
        case_number = count
        is_finished_feature = False
        is_finished_case = False

        while is_finished_feature:
            if choice_prob < feature_prob:
                # try to remove feature
                data, is_finished = self.remove_missing_feature(data, labels, threshold)
            else:
                # try to remove case
                data, labels, is_finished = self.remove_missing_case(data, labels, threshold)

        unique, counts = np.unique(labels, return_counts=True)

        if check_classwise:
            thr_counts = counts * threshold / 100
        else:
            thr_count = count * threshold / 100

        for i in range(dimen):
            check_is_kept = True

        case_missing_list = []
        for sample_i in range(count):
            case_missing_count = np.sum(counting_nan[:, sample_i])
            case_missing_list.append(case_missing_count)
        if self.is_print:
            print(case_missing_list)
        np.argsort(-np.array(case_missing_list))

        selected_data = data[:, selected_indices]

        # show missing intensity of kept features
        self.check_missing_intensity(selected_data)

        # missing imputation by class
        # ======= pending =======
        return original_columns, selected_indices, selected_data, labels

    def remove_missing_feature(self, data, labels, missing_threshold):
        is_finished = False
        unique, counts = np.unique(labels, return_counts=True)

        # find top missing value
        for cls in unique:
            subset_x = np.array(data[labels == cls])
            missing_result = self.check_missing_dimension(subset_x, axis=1)

        remove_indices = []
        for cls in unique:
            for i in range(missing_result.size):
                data[i, :]
            subset_x = np.array(data[labels == cls])

            # if nan_count > thr_counts[class_idx]:
            #     check_is_kept = False

        return data, is_finished

    def remove_missing_case(self, data, labels, missing_threshold):
        is_finished = False
        missing_result = self.check_missing_dimension(data, axis=0)
        unique, counts = np.unique(labels, return_counts=True)

        # find top missing value
        for cls in unique:
            subset_missing = missing_result[labels == cls]
            if subset_missing.size > 1:
                np.sort(subset_missing)[-2]
            else:
                np.sort(subset_missing)[-1]

        remove_indices = []
        for cls in unique:
            for i in range(missing_result.size):
                data[i, :]

            subset_x = np.array(data[labels == cls])
            target_indices = np.argwhere(subset_x > missing_threshold)
            if target_indices.size > 0:
                pass

        return data, labels, is_finished

    def check_missing_dimension(self, data, axis=0):
        dimension_len = data.shape[axis]
        missing_array = np.isnan(data.tolist())
        if axis == 0:
            missing_array = missing_array.T

        total_count = missing_array.shape[0]
        result_array = np.empty(dimension_len)
        for i in range(dimension_len):
            result = np.sum(missing_array[:, i]) / total_count
            result_array[i] = result

        return result_array

    def missing_imputation(self, data, labels=None, create_missing_set=False, matching_id=None):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        ind_list = np.argsort(labels)
        data = data[ind_list]
        labels = np.array(labels)[ind_list]
        if matching_id is not None:
            matching_id = np.array(matching_id)[ind_list]

        if labels is None:
            imp.fit(data)
            imputed_data = imp.transform(data)
            return imputed_data, matching_id
        elif create_missing_set:
            imputed_data, new_y, sorted_data, class_mean_list = self.missing_tempt_imputation_by_class(data, labels)
            return imputed_data, new_y, sorted_data, class_mean_list, matching_id
        else:
            init_shape = data.shape
            unique, counts = np.unique(labels, return_counts=True)
            new_x = np.array([])
            new_y = np.array([])

            cls_idx = 0
            for cls in unique:
                subset_x = data[labels == cls]
                imp.fit(subset_x)
                subset_x = imp.transform(subset_x)
                subset_y = np.ones([counts[cls_idx], 1]) * cls
                new_x = np.append(new_x, subset_x)
                new_y = np.append(new_y, subset_y)
                cls_idx = cls_idx + 1
            imputed_data = new_x.reshape(new_x.size // init_shape[1], init_shape[1])
            return imputed_data, new_y, matching_id

    def missing_tempt_imputation_by_class(self, data, labels):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

        init_shape = data.shape
        unique, counts = np.unique(labels, return_counts=True)
        new_x = np.array([])
        imputed_x = np.array([])
        new_y = np.array([])
        missing_set_list = []
        class_mean_list = []

        # reorder the dataset
        cls_idx = 0
        for cls in unique:
            subset_x = data[labels == cls]
            subset_y = np.ones([counts[cls_idx], 1]) * cls
            imp.fit(subset_x)
            imputed_subset_x = imp.transform(subset_x)
            new_x = np.append(new_x, subset_x)
            imputed_x = np.append(imputed_x, imputed_subset_x)
            new_y = np.append(new_y, subset_y)
            cls_idx = cls_idx + 1
        imputed_data = imputed_x.reshape(imputed_x.size // init_shape[1], init_shape[1])
        sorted_data = new_x.reshape(new_x.size // init_shape[1], init_shape[1])

        cls_idx = 0
        for cls in unique:
            class_mean_all_features = np.nanmean(sorted_data[:, ][new_y == cls].astype(float), axis=0)
            class_mean_list.append(class_mean_all_features.copy())

            temp_data = sorted_data.copy()
            for ft_i in range(init_shape[1]):
                temp_data[:, ft_i][np.isnan(temp_data[:, ft_i].astype(float))] = class_mean_all_features[ft_i]  # = xxx
            # class_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=class_mean)
            # temp_data = class_imp.fit_transform(sorted_data)
            missing_set_list.append(temp_data)
            cls_idx = cls_idx + 1

        return imputed_data, new_y, sorted_data, class_mean_list

    def max_features(self, _omega):
        if _omega.ndim == 1 or _omega.shape[0] != _omega.shape[1]:
            diagonal_values = _omega
        else:
            diagonal_values = self.get_diagonal_values(_omega)
        sorted_indices = np.flip(np.argsort(diagonal_values), axis=0)  # large to small
        return sorted_indices, np.array(diagonal_values)[sorted_indices]

    def get_diagonal_values(self, matx):
        size = matx.shape[0]
        diagonal_values = []
        for i in range(size):
            diagonal_values.append(matx[i, i])
        return diagonal_values

    def check_missing_intensity(self, data):
        data_to_check = data.astype(float).ravel()
        missing_intensity = np.isnan(data_to_check).sum() / data_to_check.size
        if self.is_print:
            print("Current missing intensity: " + str(missing_intensity))
        return missing_intensity

    def transform_missing_bool(self, data):
        return np.isnan(data)

    def remove_collinearity(self, data, thresh=5.0):
        if self.is_print:
            print("Remove collinear features VIF higher than: " + str(thresh))
        time_start = time.time()

        variables = list(range(data.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(data[:, variables], ix)
                   for ix in range(data[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                del variables[maxloc]
                dropped = True

        if self.is_print:
            print('Collinear features removed! Time elapsed: {} seconds'.format(time.time() - time_start))
        # print('Remaining variables:')
        # print(X.columns[variables])
        data = data[:, variables]
        return data, variables

    def dimen_reduction_pca(self, data, kept_variance=0.95, testset=None):
        time_start = time.time()
        pca = PCA()

        if testset is None:
            pca_result = pca.fit_transform(data)
            var_ratio = pca.explained_variance_ratio_

            accu_var = 0
            num_comp = 0
            for comp_var in var_ratio:
                accu_var = accu_var + comp_var
                if accu_var < kept_variance:
                    num_comp = num_comp + 1
                else:
                    break
            # print(accu_var)
            # print(num_comp)

            pca_result = pca_result[:, 0:num_comp+1]
            # pca = PCA(n_components=k)
            # pca_result = pca.fit_transform(data)
            if self.is_print:
                print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

            return pca_result

        else:
            pca.fit(data)
            data = pca.transform(data)
            testset = pca.transform(testset)
            var_ratio = pca.explained_variance_ratio_

            accu_var = 0
            num_comp = 0
            for comp_var in var_ratio:
                accu_var = accu_var + comp_var
                if accu_var < kept_variance:
                    num_comp = num_comp + 1
                else:
                    break
            data = data[:, 0:num_comp+1]
            testset = testset[:, 0:num_comp+1]
            if self.is_print:
                print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))
                print(data.shape)
            return data, testset


    def dimen_reduction_lda(self, data, labels, testset=None):
        clf = LinearDiscriminantAnalysis()
        if testset is None:
            data = clf.fit_transform(data, labels)
            return data
        else:
            clf.fit(data, labels)
            data = clf.transform(data)
            testset = clf.transform(testset)
            return data, testset

    def dimen_reduction_tsne(self, data, random_seed=795, k=3):
        time_start = time.time()
        if k > 3:
            tsne_result = TSNE(random_state=random_seed, n_components=k, method='exact').fit_transform(data)
        else:
            tsne_result = TSNE(random_state=random_seed, n_components=k).fit_transform(data)

        if self.is_print:
            print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        # rndperm = np.random.permutation(data.shape[0])
        # n_sne = 7000
        #
        # time_start = time.time()
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # tsne_results = tsne.fit_transform(data[rndperm[:n_sne], feat_cols].values)
        return tsne_result

    def normalize_attr(self, toy_data):
        le = preprocessing.LabelEncoder()
        process_data = toy_data.copy()
        rows, dimen = process_data.shape

        for i in range(dimen):  # 46
            # if i == 46:
            #     a=1
            data_single_dim = process_data[:, i]
            data_single_dim = data_single_dim.astype(float)
            mean = np.nanmean(data_single_dim) # np.nanmean(data_single_dim)
            std = np.nanstd(data_single_dim)
            process_data[:, i] = (data_single_dim - mean) / std

        return process_data

    def relabel_data(self, labels, n_classes):
        process_data = labels.copy()
        unit = 100 / n_classes
        current_rank = -1
        rank_begin_label = np.inf
        rank_end_label = np.inf
        for i in range(n_classes):
            # percentile of boundary in the beginning and in the end
            bound_begin = i * unit
            bound_end = (i+1) * unit
            if bound_end > 100:
                bound_end = 100
            label_begin = np.percentile(labels, bound_begin)
            label_end = np.percentile(labels, bound_end)

            if rank_begin_label != label_begin and rank_end_label != label_end:
                current_rank += 1
                rank_begin_label = label_begin
                rank_end_label = label_end

            if label_begin == label_end:
                process_data[labels == label_end] = current_rank
            elif i == 0:
                process_data[(labels <= label_end) & (labels >= label_begin)] = current_rank
            else:
                process_data[(labels <= label_end) & (labels > label_begin)] = current_rank

        return process_data.astype(int)


    # 5-fold cross validation in default
    def cross_validation(self, data, labels, fold=5):
        unique, counts = np.unique(labels, return_counts=True)
        min_count = int(counts.min())
        length = len(data)
        if min_count < fold:
            raise ValueError(
                " fold number {} is larger than data length {}".format(fold, length))
        kf = KFold(n_splits=fold, random_state=None, shuffle=True)

        train_list = []
        test_list = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            Y_train, Y_test = labels[train_index], labels[test_index]
            train_list.append([X_train, Y_train])
            test_list.append([X_test, Y_test])

        return train_list, test_list

    # 5-fold cross validation in default
    def cross_validation_by_class(self, data, labels, data_test=None, fold=5, upsample=True, upsample_train_only=True,
                                  SMOTE=True, k_value=1, is_dr=False, PCA=0.995, matching_id=None):
        unique, counts = np.unique(labels, return_counts=True)
        min_count = int(counts.min())

        if matching_id is not None:
            is_id = True
        else:
            is_id = False

        if data_test is None:
            data_test = data
        # length = len(data)
        if min_count < fold:
            raise ValueError(
                " fold number {} is larger than smallest number of class {}".format(fold, min_count))
        kf = KFold(n_splits=fold, random_state=None, shuffle=True)

        temp_train_list = []
        temp_test_list = []
        # split each class
        for idx in range(len(unique)):
            cls = unique[idx]
            cls_data = data[labels == cls]
            cls_data_test = data_test[labels == cls]
            if is_id:
                cls_data_id = matching_id[labels == cls]
            splited_index = kf.split(cls_data)
            cls_train_list = []
            cls_test_list = []
            for train_index, test_index in splited_index:
                if is_id:
                    X_train, X_test, test_id = cls_data[train_index], cls_data_test[test_index], cls_data_id[test_index]
                else:
                    X_train, X_test = cls_data[train_index], cls_data_test[test_index]
                Y_train, Y_test = np.ones(X_train.shape[0]) * cls, np.ones(X_test.shape[0]) * cls
                cls_train_list.append([X_train, Y_train])

                if is_id:
                    cls_test_list.append([X_test, Y_test, test_id])
                else:
                    cls_test_list.append([X_test, Y_test])
            temp_train_list.append(cls_train_list)
            temp_test_list.append(cls_test_list)

        train_list = []
        test_list = []
        for f in range(fold):
            # fold_train_data = np.array([])
            # fold_train_label= np.array([])
            # fold_test_data = np.array([])
            # fold_test_label = np.array([])
            for idx in range(len(unique)):
                cls = int(unique[idx])
                if idx == 0:
                    fold_train_data = temp_train_list[cls][f][0]
                    fold_train_label = temp_train_list[cls][f][1]
                    fold_test_data = temp_test_list[cls][f][0]
                    fold_test_label = temp_test_list[cls][f][1]
                    if is_id:
                        fold_test_id = temp_test_list[cls][f][2]

                else:
                    fold_train_data = np.concatenate((fold_train_data, temp_train_list[cls][f][0]))
                    fold_train_label = np.concatenate((fold_train_label, temp_train_list[cls][f][1]))
                    fold_test_data = np.concatenate((fold_test_data, temp_test_list[cls][f][0]))
                    fold_test_label = np.concatenate((fold_test_label, temp_test_list[cls][f][1]))
                    if is_id:
                        fold_test_id = np.concatenate((fold_test_id, temp_test_list[cls][f][2]))

            if is_dr:
                # dimensionality reduction
                ### fold_train_data, fold_test_data = self.dimen_reduction_lda(fold_train_data, fold_train_label, fold_test_data)
                fold_train_data, fold_test_data = self.dimen_reduction_pca(fold_train_data, PCA, fold_test_data)

            if upsample:
                if SMOTE:
                    # generate with SMOTE
                    fold_train_data, fold_train_label = self.up_sample_SMOTE(fold_train_data, fold_train_label,
                                                                             k_value)
                    if not upsample_train_only and not is_id:
                        fold_test_data, fold_test_label = self.up_sample_SMOTE(fold_test_data, fold_test_label,
                                                                               k_value)
                else:
                    # simple duplication
                    fold_train_data, fold_train_label = self.up_sample(fold_train_data, fold_train_label)
                    if not upsample_train_only and not is_id:
                        fold_test_data, fold_test_label = self.up_sample(fold_test_data, fold_test_label)

            train_list.append([fold_train_data, fold_train_label])
            if is_id:
                test_list.append([fold_test_data, fold_test_label, fold_test_id])
            else:
                test_list.append([fold_test_data, fold_test_label])

        return train_list, test_list

    def loocv(self, training, test, labels):
        pass

    def anova(self, _data, _labels):
        if _data.ndim > 1 and _data.shape[1] > 1:
            return self.anova_multiple_attributes(_data, _labels)
        else:
            classes = np.unique(_labels)
            grouped_list = []
            for cls in classes:
                grouped_list.append(_data[_labels == cls])

            F, p_value = stats.f_oneway(*grouped_list)
            # print(p_value)
            return F, p_value

    def anova_multiple_attributes(self, _data, _labels):
        results = []
        for i in range(_data.shape[1]):
            results.append(self.anova(_data[:, i], _labels))

        return results

    def filter_anova_results(self, indices, results, data, threshold_pvalue=0.05, threshold_number=None):
        # resluts: list of tuple(F, p value)

        if threshold_number is not None:

            filtered_indices = np.argsort([tp[1] for tp in results])[0:threshold_number]
            sig_indices = np.array(indices)[filtered_indices]
            sig_results = np.array(results)[filtered_indices]
            sig_data = data[:, filtered_indices]
            second_filtered_indices = []
            for i in range(len(sig_results)):
                result_tuple = sig_results[i]
                if result_tuple[1] <= threshold_pvalue:
                    second_filtered_indices.append(i)

            filtered_indices = filtered_indices[second_filtered_indices]
            sig_indices = sig_indices[second_filtered_indices]
            sig_results = sig_results[second_filtered_indices]
            sig_data = sig_data[:, second_filtered_indices]
            # len_current = len(sig_results)
            # if len_current > threshold_number:
            #     threshold_pvalue = threshold_pvalue / 2
            # elif len_current < threshold_number:
            #     threshold_pvalue = threshold_pvalue * 1.5
            # else:

        else:
            filtered_indices = []
            for i in range(len(results)):
                result_tuple = results[i]
                if result_tuple[1] <= threshold_pvalue:
                    filtered_indices.append(i)

            sig_indices = np.array(indices)[filtered_indices]
            sig_results = np.array(results)[filtered_indices]
            sig_data = data[:, filtered_indices]

        return sig_indices, sig_data, sig_results, filtered_indices

    def filter_by_anova(self, _indices, _data, _labels, threshold_pvalue=0.05, threshold_number=None):
        if threshold_pvalue == 1:
            sig_indices = _indices
            sig_data = _data
            sig_results = None
            filtered_indices = list(range(_data.shape[1]))
        else:
            results = self.anova(_data, _labels)
            sig_indices, sig_data, sig_results, filtered_indices = self.filter_anova_results(_indices, results, _data, threshold_pvalue=threshold_pvalue, threshold_number=threshold_number)
        return sig_indices, sig_data, sig_results, filtered_indices

    def up_sample(self, x, y):
        init_shape = x.shape
        unique, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        new_x = np.array([])
        new_y = np.array([])

        for cls in unique:
            subset_x = np.array(x[y == cls])
            subset_y = np.ones([max_count, 1]) * cls
            subset_x = resample(subset_x, n_samples=max_count, random_state=0)
            new_x = np.append(new_x, subset_x)
            new_y = np.append(new_y, subset_y)

        new_x = new_x.reshape(new_x.size // init_shape[1], init_shape[1])
        return new_x, new_y

    def up_sample_SMOTE(self, x, y, k_value=1):
        init_shape = x.shape
        unique, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        new_x = np.array([])
        new_y = np.array([])

        for cls in unique:
            subset_x = np.array(x[y == cls])
            subset_y = np.ones([max_count, 1]) * cls
            sample_size_subX = subset_x.shape[0]

            if sample_size_subX < max_count:
                self.SMOTE(subset_x, max_count, k_value=k_value)

            subset_x = resample(subset_x, n_samples=max_count, random_state=0)
            new_x = np.append(new_x, subset_x)
            new_y = np.append(new_y, subset_y)

        new_x = new_x.reshape(new_x.size // init_shape[1], init_shape[1])
        return new_x, new_y

    def linear_new_datapoint(self, x1, x2):
        unit_vec = random.uniform(0, 1)
        return np.array([x1 * (1 - unit_vec) + x2 * unit_vec])

    def find_knn_list(self, data, k_value):
        sample_size = data.shape[0]
        knn_list = []

        for i in range(sample_size):
            datapoint = np.array([data[i, :]])
            rest1 = list(range(0, i))
            rest2 = list(range(i+1, sample_size))
            rest = rest1 + rest2
            others = data[rest, :]
            distance_list = self._squared_euclidean(datapoint, others).flatten()
            min_indices = np.argsort(distance_list, axis=0)[0: k_value]
            knn_points = data[min_indices]
            knn_list.append((datapoint, knn_points))
        return knn_list  # return list of 2darray (k, tuple(datapoint, feature_dimension))

    def select_from_knn_list(self, knn_list, number_to_generate):
        # knn_list = list of 2darray (k, feature_dimension)
        candidate_list = []
        pre_candidates = random.choices(knn_list, k=number_to_generate)
        for (datapoint, knn_points) in pre_candidates:
            if len(knn_points) == 0:
                candidate_list.append((datapoint[0], datapoint[0]))
            else:
                candidate_list.append((datapoint[0], random.choice(knn_points)))
        return candidate_list

    def SMOTE(self, data, target_number, k_value=1):
        data_shape = data.shape
        knn_list = self.find_knn_list(data, k_value)
        number_to_generate = target_number - len(knn_list)
        candidate_list = self.select_from_knn_list(knn_list, number_to_generate)
        for candidate in candidate_list:
            original_x = candidate[0]
            neighbor_x = candidate[1]
            generated_x = self.linear_new_datapoint(original_x, neighbor_x)
            data = np.append(data, generated_x, axis=0)
        # print(data)

    def _squared_euclidean(self, a, b=None):
        if b is None:
            d = np.sum(a * a, 1)[np.newaxis].T + np.sum(a * a, 1) - 2 * a.dot(
                a.T)
        else:
            d = np.sum(a * a, 1)[np.newaxis].T + np.sum(b * b, 1) - 2 * a.dot(
                b.T)
        return np.maximum(d, 0)

    def artificial_data(self, sample_size, list_center, list_label, list_matrix, if_normalize=False):
        nb_ppc = sample_size
        k = len(list_center)
        for i in range(k):
            if i == 0:
                toy_label = np.ones(nb_ppc) * list_label[i]
                toy_data = np.random.multivariate_normal(list_center[i], np.array(list_matrix[i]), size=nb_ppc)

            else:
                index = i
                toy_label = np.append(toy_label, np.ones(nb_ppc) * list_label[i], axis=0)
                toy_data = np.append(toy_data,
                                     np.random.multivariate_normal(list_center[index], np.array(list_matrix[index]),
                                                                       size=nb_ppc), axis=0)

        if if_normalize:
            mean0 = toy_data[:, 0].mean()
            std0 = toy_data[:, 0].std()
            toy_data[:, 0] = (toy_data[:, 0] - mean0) / std0

            mean1 = toy_data[:, 1].mean()
            std1 = toy_data[:, 1].std()
            toy_data[:, 1] = (toy_data[:, 1] - mean1) / std1

        return toy_data, toy_label

    def get_iteration(self, gtol, initial_lr, final_lr, max_iter=2500):
        return min(int(initial_lr / (final_lr * gtol) + 1 - 1 / gtol), max_iter)

    def set_iteration(self, iter, initial_lr, final_lr):
        gtol = (initial_lr - final_lr)/((iter-1)*final_lr)
        if gtol >= 1:
            raise ValueError(
                " iteration number ({}) is not large enough for learning rate to decrease".format(iter))
        return gtol

    def combine_stds(self, _mean_list, _std_list):
        mean_list = np.array(_mean_list)
        std_list = np.array(_std_list)
        combined_mean = np.mean(mean_list)
        temp_arr = np.square(mean_list - combined_mean) + np.square(std_list)
        combined_std = np.sqrt(temp_arr.mean())
        return combined_mean, combined_std

    def mean_confidence_interval(self, val_list, confidence=0.95, axis=0):
        a = 1.0 * np.array(val_list)
        if a.ndim == 2:
            if axis == 0:
                temp_a = a
            else:
                temp_a = a.T
            length = temp_a.shape[1]
            res_list = []
            for i in range(length):
                temp_list = temp_a[:, i]
                res_list.append(self.mean_confidence_interval(temp_list, confidence=confidence))
            return res_list
        else:
            n = len(a)
            m, se = np.mean(a), stats.sem(a)
            h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
            return m, m - h, m + h, h

    def rank_array(self, _list, descending=True):
        array = np.array(_list)
        if array.ndim == 2:
            res_list = []
            for sublist in array:
                res_list.append(self.rank_array(sublist, descending))
            return res_list
        else:
            if descending:
                array = array * -1

            temp = array.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(array))
            return ranks

    def to_2d_data(self, data, mtx):
        # u, s, vh = np.linalg.svd(np.array(mtx).T.dot(np.array(mtx)), full_matrices=True)
        # x_vec = u[0:2, :]
        # y_vec = vh[0:2, :]
        w, vec = np.linalg.eig(mtx)

        transformed_example = []
        for datapoint in data:
            # temp_x = x_vec.dot(datapoint)
            # temp_y = y_vec.dot(datapoint)
            temp_point = [vec[0].dot(datapoint).real, vec[1].dot(datapoint).real]
            transformed_example.append(temp_point)
        return transformed_example
