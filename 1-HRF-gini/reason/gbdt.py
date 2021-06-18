import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
import json

from sklearn.tree import DecisionTreeClassifier as mod

class GradientBoostingDecisionTree():
    def __init__(self):
        self.tree = None

    def fit(self, score, label, depth = 3, gamma = 0):

        # dtrain = xgb.DMatrix(score, label=label)
        # param = {
        #     "max_depth": depth, 
        #     "eta": 1, 
        #     "objective": "binary:logistic",
        #     "gamma": gamma
        # }
        # num_round = 1
        # self.tree = xgb.train(param, dtrain, num_boost_round = num_round)


        self.mod = mod(
            criterion="gini", 
            splitter="best", 
            max_depth=3, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0., 
            max_features=None, 
            random_state=None, 
            max_leaf_nodes=None, 
            min_impurity_decrease=0., 
            min_impurity_split=None, 
            class_weight=None, 
            ccp_alpha=0.0
        )
        self.tree = self.mod.fit(score.copy(), label.copy())

    def pred(self, score):
        assert self.tree is not None
        # dtest = xgb.DMatrix(score)
        dtest = np.array(score)
        preds = self.tree.predict(dtest)
        preds[np.where(preds > 0.5)] = 1
        preds[np.where(preds <= 0.5)] = 0
        return preds
    
    def f1(self, label, preds):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i in range(len(label)):
            x = label[i]
            y = preds[i]
            if x and x == y:
                true_positive += 1
            elif x and x != y:
                false_positive += 1
            elif x == 0 and x == y:
                true_negative += 1
            elif x == 0 and x != y:
                false_negative += 1
        prec = true_positive / (true_positive + false_positive + 1e-16) + 1e-16
        rec = true_positive / (true_positive + false_negative + 1e-16) + 1e-16
        f1 = 2 / (1 / prec + 1 / rec)
        return f1, prec, rec

    def eval(self, score, label, case = False):
        assert self.tree is not None
        preds = self.pred(score)
        preds = preds.tolist()
        # f1 = f1_score(label, preds)
        f1, prec, rec = self.f1(label, preds)
        if not case:
            return f1, prec, rec

        error_case = [(r, int(preds[r])) for r in range(len(preds)) if preds[r] != label[r]]
        return f1, prec, rec, error_case


    def get_attribute_weight(self, fmap, importance_type = "binary"):
        """
        importance_type could be 
        "binary": as xgboost gives a binary tree, we use the binary tree to assign a score to each attributes
        "gain": gain from the optimization process
        """
        assert importance_type in ["binary", "gain"]
        weight = defaultdict(int)
        
        if importance_type == "binary":
            tree_structure = self.tree.get_dump(fmap=fmap, with_stats=False, dump_format="json")
            tree_structure = tree_structure[0]
            tree_structure = json.loads(tree_structure)
            self.__get_weight(tree_structure, 1., weight)

        elif importance_type == "gain":
            scores = self.tree.get_score(fmap = fmap, importance_type = "gain")
            total_score = 0
            for a in scores:
                s = scores[a]
                a = a.replace("-dist", "").replace("-cosine", "").replace("-pearson", "")
                weight[a] += s
                total_score += s
            for x in weight:
                weight[x] /= total_score
            
        return weight
 
    def __get_weight(self, tree_structure, remaining_weights, weight_dict):
        child_count = 0
        for child in tree_structure["children"]:
            if "leaf" not in child:
                child_count += 1
        
        weight_assign_root = remaining_weights * (1 - 0.25 * child_count)
        weight_dict[tree_structure["split"].replace("-dist", "").replace("-cosine", "")] += weight_assign_root

        for child in tree_structure["children"]:
            if "leaf" not in child:
                self.__get_weight(child, remaining_weights = 0.25 * remaining_weights, weight_dict = weight_dict)

    def plot(self):
        pass

    def load_model(self, model_file_name):
        self.tree = xgb.Booster()
        self.tree.load_model(model_file_name)
        # self.tree = xgb.load_model(model_file_name)