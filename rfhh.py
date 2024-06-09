import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from typing import List
from phermes import HyperHeuristic, Problem

class RFHH(HyperHeuristic):

    def __init__(self, features: List[str], heuristics: List[str], n_estimators: int = 100):
        super().__init__(features, heuristics)
        self._model = RandomForestClassifier(n_estimators=n_estimators)
        self._n_estimators = n_estimators

    def train(self, filename: str) -> None:
        data = pd.read_csv(filename, header=0)
        columns = ["INSTANCE", "BEST", "ORACLE"] + self._heuristics
        X = data.drop(columns, axis=1).values
        y = data["BEST"].values
        for i in range(len(self._heuristics)):
            y[y == self._heuristics[i]] = i
        y = y.astype("int")

        # Generate learning curve
        self._plot_learning_curve(X, y)

    def _plot_learning_curve(self, X, y):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_title("Learning Curve (RandomForest)")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.1)
        ax.grid()
        train_scores_mean_list = []
        test_scores_mean_list = []
        train_sizes_list = []

        skf = StratifiedKFold(n_splits=5)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_sizes = np.linspace(0.1, 1.0, 10) * len(X_train)
            train_scores = []
            test_scores = []

            for size in train_sizes:
                size = int(size)
                self._model.fit(X_train[:size], y_train[:size])
                train_score = self._model.score(X_train[:size], y_train[:size])
                test_score = self._model.score(X_test, y_test)
                train_scores.append(train_score)
                test_scores.append(test_score)

            train_sizes_list.append(train_sizes)
            train_scores_mean_list.append(np.mean(train_scores))
            test_scores_mean_list.append(np.mean(test_scores))

            ax.plot(train_sizes, train_scores, 'o-', color="r", label="Training score" if len(train_scores_mean_list) == 1 else "")
            ax.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score" if len(test_scores_mean_list) == 1 else "")

            ax.legend(loc="best")
            plt.draw()
            plt.pause(0.01)  # Pause to update the plot

        plt.ioff()  # Turn off interactive mode
        plt.show()

    def getHeuristic(self, problem: Problem) -> str:
        state = pd.DataFrame()
        for i in range(len(self._features)):
            state[self._features[i]] = [problem.getFeature(self._features[i])]
        prediction = self._model.predict(state.values)
        return self._heuristics[prediction[0]]
