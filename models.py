from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    clothes = db.Column(db.String(100))
    arrival_time = db.Column(db.Time)
    car_type = db.Column(db.String(50), nullable=True)
    purchase = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(100))
    visits = db.relationship('Visit', backref='customer', lazy=True)

class Visit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    purpose = db.Column(db.String(100))
    suggested_cars = db.Column(db.String(200))
    purchase_prob = db.Column(db.Float)
    ml_purchase_prob = db.Column(db.Float)
    car_type = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(100))

class MLModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_data = db.Column(db.LargeBinary)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    accuracy = db.Column(db.Float)
    features = db.Column(db.String(500))