from sklearn.model_selection import train_test_split
import pandas as pd

def split_from_separately(x_tra, x_te, y_tra, y_te):
    df_x_tra = pd.DataFrame(x_tra)
    df_x_te = pd.DataFrame(x_te)
    df_x = pd.concat([df_x_tra, df_x_te])
    df_y_tra = pd.DataFrame(y_tra)
    df_y_te = pd.DataFrame(y_te)
    df_y = pd.concat([df_y_tra, df_y_te])
    x = df_x.values
    y = df_y.values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42)
    
    return x_train, x_test, y_train, y_test


def split_from_combined(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=42)
    
    return x_train, x_test, y_train, y_test




