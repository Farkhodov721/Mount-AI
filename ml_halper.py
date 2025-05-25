import pickle
from re import X

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from models import MLModel


def clothes_to_numeric(clothes):
    clothes_mapping = {
        "Jeans va futbolka": 0,
        "Kostyum": 1,
        "Sport kiyim": 2,
        "Yengil ko'ylak": 3,
        "Libos": 4,
        "Ofis kiyim": 5,
        "Shortik va futbolka": 6,
        "Polo va shim": 7,
        "Klassik kostyum": 8,
        "Futbolka va shortik": 9
    }
    return clothes_mapping.get(clothes, 0)


def prepare_customer_data(customers, visits):
    data = []
    for v in visits:
        cust = next((c for c in customers if c.id == v.customer_id), None)
        if cust:
            arrival_hour = cust.arrival_time.hour + cust.arrival_time.minute / 60.0
            data.append({
                'age': cust.age,
                'gender': 1 if cust.gender == "Erkak" else 0,
                'clothes': clothes_to_numeric(cust.clothes),
                'arrival_time': arrival_hour,
                'duration': (v.exit_time - v.entry_time).total_seconds() / 3600,
                'purchase': 1 if cust.purchase else 0
            })
    return pd.DataFrame(data)


def train_model(df):
    X = df[['age', 'gender', 'clothes', 'arrival_time', 'duration']]
    y = df['purchase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    accuracy = rf.score(X_test, y_test)
    return rf, accuracy, X.columns.tolist()


class X_test:
    pass


def save_model(model, db_session, y_test=None):
    model_binary = pickle.dumps(model)
    ml_model = MLModel(
        model_data=model_binary,
        accuracy=model.score(X_test, y_test),
        features=",".join(X.columns)
    )
    db_session.add(ml_model)
    db_session.commit()


def load_latest_model(db_session):
    latest_model = MLModel.query.order_by(MLModel.created_at.desc()).first()
    if latest_model:
        return pickle.loads(latest_model.model_data)
    return None


def predict_purchase(model, customer_data):
    df = pd.DataFrame([customer_data])
    return model.predict_proba(df)[0][1]