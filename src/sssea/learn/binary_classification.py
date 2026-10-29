"""Binary classification utilities."""

import logging
from typing import Any, NamedTuple, Self, cast

import numpy as np
from numpy._typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

__all__ = [
    "BinaryClassifierResult",
    "BinaryClassifiers",
    "predict_binary",
]

logger = logging.getLogger(__name__)


class BinaryClassifierResult(NamedTuple):
    """Results from training and evaluating a binary classifier."""

    key: str
    classifier: str
    roc_auc: float


class BinaryClassifiers(NamedTuple):
    """A tuple containing a variety of binary classifiers."""

    random_forest: RandomForestClassifier
    logistic_regression: LogisticRegression
    linear_svc: LinearSVC
    rbf_svc: SVC

    # decision tree not great here, don't bother using it

    @classmethod
    def make(cls) -> Self:
        """Construct an ensemble of classifiers."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        return cls(
            random_forest=RandomForestClassifier(),
            logistic_regression=LogisticRegression(),
            linear_svc=LinearSVC(),
            rbf_svc=CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False),
        )

    def fit_evaluate(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.bool],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.bool],
    ) -> list[BinaryClassifierResult]:
        """Fit all classifiers."""
        from sklearn.metrics import roc_auc_score

        results = []
        for key, classifier in zip(BinaryClassifiers._fields, self, strict=False):
            classifier.fit(x_train, y_train)
            roc_auc = roc_auc_score(y_test, predict_binary(classifier, x_test))
            logger.info("%s ROC-AUC: %.2f", key, roc_auc)
            results.append(BinaryClassifierResult(key, classifier, roc_auc))
        return results


def predict_binary(
    classifier: Any, x: NDArray[np.float64] | NDArray[np.str_]
) -> NDArray[np.float64]:
    """Predict with a classifier."""
    if hasattr(classifier, "predict_proba"):
        return cast(NDArray[np.float64], classifier.predict_proba(x)[:, 1])
    elif hasattr(classifier, "decision_function"):
        return cast(NDArray[np.float64], classifier.decision_function(x))
    else:
        raise TypeError(f"classifier {classifier} neither has predict_proba nor decision function")
