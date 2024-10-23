from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# 创建SVM分类器
svm = SVC(kernel='linear', C=1.0, random_state=42)
class SVM(SVC):
    def __init__(
            self,
            *,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,):
        super(SVM, self).__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,)
        
        self.iseval = None
        
    def __call__(self, X, y=None):
        if self.iseval:
            return self.predict(X)
        return self.fit(X, y)
    
    def train(self):
        self.iseval = False
        return None
    
    def eval(self):
        self.iseval = True
        return None
    
    def to(self, opt):
        return self
