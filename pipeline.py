from os import pipe
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

class pipeline:
    def __init__(self, df, split) -> None:
        self.df = df
        self.split = split
    
    def build_pipeline(self):
        self.pipe = Pipeline([('scaler', StandardScaler()), ('decision_tree', DecisionTreeRegressor())])
    
    def fit_test(self, x_columns, y_columns):
        X = self.df.drop(y_columns, axis=1)
        y = self.df[y_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split)
        self.x_test = X_test
        self.y_test = y_test
        self.pipe.fit(X_train, y_train)
        score = self.pipe.score(X_test, y_test)
        mae = mean_absolute_error(y_test, self.pipe.predict(X_test))
        return [score,mae]

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    df = pd.read_csv('sample_data.csv')
    ml_pipeline = pipeline(df, ['x1', 'x2', 'x3', 'x4'], ['y'], 0.3)
    ml_pipeline.build_pipeline()
    sc,mae = ml_pipeline.fit_test()
    print(sc,mae)