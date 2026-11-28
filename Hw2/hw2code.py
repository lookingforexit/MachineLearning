import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    x = np.asarray(feature_vector, dtype=float)
    y = np.asarray(target_vector)
    
    if len(x) < 2:
        return np.array([]), np.array([]), None, None
    
    order = np.argsort(x, kind='stable')
    x_sorted = x[order]
    y_sorted = (y[order] == 1).astype(int)

    is_valid = (x_sorted[:-1] < x_sorted[1:])
    if not np.any(is_valid):
        return np.array([]), np.array([]), None, None
    
    thresholds = x_sorted[:-1][is_valid] / 2 + x_sorted[1:][is_valid] / 2

    left_sizes = np.arange(1, len(x))[is_valid]
    right_sizes = len(x) - left_sizes

    pos_prefix = np.cumsum(y_sorted)
    left_pos = pos_prefix[:-1][is_valid]
    right_pos = pos_prefix[-1] - left_pos

    left_p = left_pos / left_sizes
    right_p = right_pos / right_sizes

    left_odd = 2 * left_p * (1 - left_p)
    right_odd = 2 * right_p * (1 - right_p)

    gini = -(left_sizes / len(x) * left_odd + right_sizes / len(x) * right_odd)

    return thresholds, gini, float(thresholds[np.argmax(gini)]), float(gini[np.argmax(gini)])


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_types=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if feature_types is not None and np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]) \
            or (self._max_depth is not None and depth >= self._max_depth) \
            or len(sub_y) < self._min_split_count \
            or len(sub_y) < 2 * self._min_leaf_count: # тут было != у проверки классов
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # для доп. задания пришлось изменить
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): # тут был перебор не всех признаков, а начиная со 1го
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # тут были перепутаны переменные местами
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # тут бралась доля категории, а не название
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) # тут передавался map, а не список
            else:
                raise ValueError

            """
            if len(feature_vector) == 3:
                continue
                
            тут странное удаление признаков по которым нельзя разделить выборку по группам
            """

            thresholds, ginies, _, _ = find_best_split(feature_vector, sub_y) # нормальная проверка на удаление признаков по которым не делим (включая доп. задание)
            
            left_sizes = np.searchsorted(np.sort(feature_vector), thresholds, side='left')
            right_sizes = len(sub_y) - left_sizes

            is_valid = (left_sizes >= self._min_leaf_count) & (right_sizes >= self._min_leaf_count)
            if not np.any(is_valid):
                continue

            thresholds = thresholds[is_valid]
            ginies = ginies[is_valid]

            best_i = np.argmax(ginies)
            threshold = thresholds[best_i]
            gini = ginies[best_i]

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": # название типа было с большой буквы
                    threshold_best = list(map(lambda x: x[0], filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # тут хранился список пар, а не класс
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1) # тут для правого поддерева была левая часть sub_y

    def _predict_node(self, x, node):
        if node['type'] == "terminal":
            return node['class']
        
        feature = node['feature_split']

        if self._feature_types[feature] == 'real':
            is_left = x[feature] < node['threshold']
        else:
            is_left = x[feature] in node['categories_split']

        if is_left:
            return self._predict_node(x, node["left_child"])
        
        return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._feature_types = ['real'] * X.shape[1] if self.feature_types is None else self.feature_types
        if len(self._feature_types) != X.shape[1]:
            raise ValueError("incorrect count of types")

        self._max_depth = self.max_depth
        self._min_samples_split = self.min_samples_split
        self._min_samples_leaf = self.min_samples_leaf

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self._tree = {}

        if self._min_samples_split is None:
            self._min_split_count = 2
        elif isinstance(self._min_samples_split, (float, np.floating)):
            if not 0 < self._min_samples_split <= 1:
                raise ValueError("incorrect min_samples_split")
            
            self._min_split_count = max(2, int(np.ceil(self._min_samples_split * len(y))))
        elif (isinstance(self._min_samples_split, (int, np.integer)) and self._min_samples_split >= 2):
            self._min_split_count = self._min_samples_split
        else:
            raise ValueError("incorrect min_samples_split")
        
        if self._min_samples_leaf is None:
            self._min_leaf_count = 1
        elif (isinstance(self._min_samples_leaf, (float, np.floating))):
            if not 0 < self._min_samples_leaf <= 0.5:
                raise ValueError("incorrect min_samples_leaf")
            
            self._min_leaf_count = int(np.ceil(self._min_samples_leaf * len(y)))
        elif (isinstance(self._min_samples_leaf, (int, np.integer)) and self._min_samples_leaf >= 1):
            self._min_leaf_count = self._min_samples_leaf
        else:
            raise ValueError("incorrect min_samples_leaf")
        
        if self._max_depth is not None and not (isinstance(self._max_depth, (int, np.integer)) and self._max_depth >= 1):
            raise ValueError("incorrect max_depth")

        self._fit_node(X, y, self._tree)

        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def _depth(node):
        if node['type'] == 'terminal':
            return 0
        
        return 1 + max(DecisionTree._depth(node['left_child']), DecisionTree._depth(node["right_child"]))