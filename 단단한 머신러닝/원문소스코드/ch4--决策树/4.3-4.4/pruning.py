import pandas as pd
import numpy as np


def post_pruning(X_train, y_train, X_val, y_val, tree_=None):
    if tree_.is_leaf:
        return tree_

    if X_val.empty:         # 유효성 검사 세트가 비어 있으면 더 이상 가지치기가 수행되지 않습니다.
        return tree_

    most_common_in_train = pd.value_counts(y_train).index[0]
    current_accuracy = np.mean(y_val == most_common_in_train)  # 현재 노드에서 검증 세트 샘플의 정확도

    if tree_.is_continuous:
        up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
        up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

        up_subtree = post_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                  y_val[up_part_val],
                                  tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
        tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree
        down_subtree = post_pruning(X_train[down_part_train], y_train[down_part_train],
                                    X_val[down_part_val], y_val[down_part_val],
                                    tree_.subtree['< {:.3f}'.format(tree_.split_value)])
        tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree

        tree_.high = max(up_subtree.high, down_subtree.high) + 1
        tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

        if up_subtree.is_leaf and down_subtree.is_leaf:
            def split_fun(x):
                if x >= tree_.split_value:
                    return '>= {:.3f}'.format(tree_.split_value)
                else:
                    return '< {:.3f}'.format(tree_.split_value)

            val_split = X_val.loc[:, tree_.feature_name].map(split_fun)
            right_class_in_val = y_val.groupby(val_split).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            if current_accuracy > split_accuracy:  # 현재 노드가 리프 노드일 때의 정확도가 가지치기를 하지 않은 정확도보다 크면 가지치기 작업을 수행 - 현재 노드를 리프 노드로 설정
                set_leaf(pd.value_counts(y_train).index[0], tree_)
    else:
        max_high = -1
        tree_.leaf_num = 0
        is_all_leaf = True  # 현재 노드 아래의 모든 하위 트리가 리프 노드인지 확인

        for key in tree_.subtree.keys():
            this_part_train = X_train.loc[:, tree_.feature_name] == key
            this_part_val = X_val.loc[:, tree_.feature_name] == key

            tree_.subtree[key] = post_pruning(X_train[this_part_train], y_train[this_part_train],
                                              X_val[this_part_val], y_val[this_part_val], tree_.subtree[key])
            if tree_.subtree[key].high > max_high:
                max_high = tree_.subtree[key].high
            tree_.leaf_num += tree_.subtree[key].leaf_num

            if not tree_.subtree[key].is_leaf:
                is_all_leaf = False
        tree_.high = max_high + 1

        if is_all_leaf:  # 모든 자식 노드가 리프 노드인 경우 가지치기 여부를 고려합니다.
            right_class_in_val = y_val.groupby(X_val.loc[:, tree_.feature_name]).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            if current_accuracy > split_accuracy:  # 현재 노드가 리프 노드일 때의 정확도가 가지치기를 하지 않은 정확도보다 크면 가지치기 작업을 수행 - 현재 노드를 리프 노드로 설정
                set_leaf(pd.value_counts(y_train).index[0], tree_)

    return tree_


def pre_pruning(X_train, y_train, X_val, y_val, tree_=None):
    if tree_.is_leaf:  # 현재 노드가 이미 리프 노드인 경우 직접 return
        return tree_

    if X_val.empty: # 유효성 검사 세트가 비어 있으면 더 이상 가지치기가 수행되지 않습니다.
        return tree_
    # 정확도를 계산할 때 수박 데이터 세트로 인해 좋은 멜론과 나쁜 멜론의 개수는 같을 것입니다. 50%).
    # 이것은 불안정한 정확도로 이어지며, 물론 숫자가 클 때 거의 발생하지 않습니다.

    most_common_in_train = pd.value_counts(y_train).index[0]
    current_accuracy = np.mean(y_val == most_common_in_train)

    if tree_.is_continuous:  # 값이 연속적일 때 분할 후 정확한 비율을 계산하기 위해 샘플을 두 부분으로 나누어야 합니다.

        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val,
                                                  split_value=tree_.split_value)

        if current_accuracy >= split_accuracy:  # 현재 노드가 리프 노드일 때 정확도 비율이 나눗셈 후 정확도 비율보다 크거나 나누지 않도록 선택합니다.
            set_leaf(pd.value_counts(y_train).index[0], tree_)

        else:
            up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
            up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

            up_subtree = pre_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                     y_val[up_part_val],
                                     tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
            tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree
            down_subtree = pre_pruning(X_train[down_part_train], y_train[down_part_train],
                                       X_val[down_part_val],
                                       y_val[down_part_val],
                                       tree_.subtree['< {:.3f}'.format(tree_.split_value)])
            tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree

            tree_.high = max(up_subtree.high, down_subtree.high) + 1
            tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

    else:  # 불연속 값인 경우 변수의 모든 값은 분할 후 정확한 비율을 계산합니다.

        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val)

        if current_accuracy >= split_accuracy:
            set_leaf(pd.value_counts(y_train).index[0], tree_)

        else:
            max_high = -1
            tree_.leaf_num = 0
            for key in tree_.subtree.keys():
                this_part_train = X_train.loc[:, tree_.feature_name] == key
                this_part_val = X_val.loc[:, tree_.feature_name] == key
                tree_.subtree[key] = pre_pruning(X_train[this_part_train], y_train[this_part_train],
                                                 X_val[this_part_val],
                                                 y_val[this_part_val], tree_.subtree[key])
                if tree_.subtree[key].high > max_high:
                    max_high = tree_.subtree[key].high
                tree_.leaf_num += tree_.subtree[key].leaf_num
            tree_.high = max_high + 1
    return tree_


def set_leaf(leaf_class, tree_):
    # 노드를 리프 노드로 설정
    tree_.is_leaf = True  # 나눗셈 전의 정확한 비율이 나눗셈 후의 정확한 비율보다 큰 경우. 그런 다음 분할하지 않고 현재 노드를 리프 노드로 설정합니다.
    tree_.leaf_class = leaf_class
    tree_.feature_name = None
    tree_.feature_index = None
    tree_.subtree = {}
    tree_.impurity = None
    tree_.split_value = None
    tree_.high = 0  # 상위 노드 및 리프 노드 수 재설정
    tree_.leaf_num = 1


def val_accuracy_after_split(feature_train, y_train, feature_val, y_val, split_value=None):
    # 연속적인 값이면 절단점에 따라 피처를 그룹화해야 하고, 불연속 값이면 처리할 필요가 없습니다.
    if split_value is not None:
        def split_fun(x):
            if x >= split_value:
                return '>= {:.3f}'.format(split_value)
            else:
                return '< {:.3f}'.format(split_value)

        train_split = feature_train.map(split_fun)
        val_split = feature_val.map(split_fun)

    else:
        train_split = feature_train
        val_split = feature_val

    majority_class_in_train = y_train.groupby(train_split).apply(
        lambda x: pd.value_counts(x).index[0])  # 각 기능에서 가장 많은 샘플이 있는 범주를 계산합니다.
    right_class_in_val = y_val.groupby(val_split).apply(
        lambda x: np.sum(x == majority_class_in_train[x.name]))  # 각 범주에 해당하는 숫자를 계산합니다.

    return right_class_in_val.sum() / y_val.shape[0]  # 반환 정확도
