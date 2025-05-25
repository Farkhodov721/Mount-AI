from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, time as dt_time
import random
import pandas as pd
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from deepface import DeepFace
import torch
import clip
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = "supersecret"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///avtosalon.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

CURRENT_USER = "Farkhodov721"
CURRENT_TIMESTAMP = "2025-05-24 23:56:13"


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
    created_by = db.Column(db.String(100), default=CURRENT_USER)
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
    created_by = db.Column(db.String(100), default=CURRENT_USER)


class PersonAnalyzer:
    def __init__(self):
        print("🔧 Initializing analysis system...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Expanded outfit labels for better accuracy
        self.outfit_labels = [
            "formal suit", "business casual", "sportwear", "jeans and t-shirt", "dress",
            "traditional attire", "uniform", "hoodie and sweatpants", "shorts and t-shirt",
            "skirt and blouse", "overcoat", "jacket", "sweater"
        ]
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.outfit_tokens = clip.tokenize(self.outfit_labels).to("cpu")
        print("✅ System ready")

    def analyze_frame(self, image_data):
        # Convert base64 to image if needed
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = image_data

        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            body_roi = frame[y+h:y+3*h, x:x+w] if y+3*h < frame.shape[0] else frame[y+h:, x:x+w]

            # Gender and age analysis with confidence threshold
            try:
                face_analysis = DeepFace.analyze(face_roi, actions=['age', 'gender'], enforce_detection=False)[0]
                age = int(face_analysis.get('age', 25))
                gender_scores = face_analysis.get('gender', {})
                male_score = float(gender_scores.get('Man', 0.0))
                female_score = float(gender_scores.get('Woman', 0.0))
                confidence = max(male_score, female_score)
                if confidence >= 0.6:
                    gender = 'male' if male_score > female_score else 'female'
                else:
                    gender = 'unknown'
            except Exception as e:
                age = 25
                gender = "unknown"
                confidence = 0.0

            # Clothing detection with improved confidence logic
            if body_roi.size > 0:
                try:
                    clothing_type, clothing_conf, top_clothing = self.detect_clothing(body_roi)
                except Exception as e:
                    clothing_type = "unknown"
                    clothing_conf = 0.0
                    top_clothing = []
            else:
                clothing_type = "unknown"
                clothing_conf = 0.0
                top_clothing = []

            results.append({
                "age": age,
                "gender": {
                    "prediction": gender,
                    "confidence": confidence
                },
                "clothing": {
                    "type": clothing_type,
                    "confidence": clothing_conf,
                    "top_choices": top_clothing  # Useful for debugging or UI
                },
                "face_coordinates": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })

        return frame, results

    def detect_clothing(self, body_image):
        # Convert to PIL image for CLIP
        pil_image = Image.fromarray(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB))
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to("cpu")

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(self.outfit_tokens)
            logits_per_image = image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        best_index = int(np.argmax(probs))
        best_label = self.outfit_labels[best_index]
        best_prob = float(probs[best_index])

        # Top 3 predictions for extra context
        top_indices = probs.argsort()[-3:][::-1]
        top_labels = [(self.outfit_labels[i], float(probs[i])) for i in top_indices]

        # Set a confidence threshold
        if best_prob < 0.45:
            best_label = "uncertain"

        return best_label, best_prob, top_labels




person_analyzer = PersonAnalyzer()


GM_MODELS = [
    "Chevrolet Spark", "Chevrolet Nexia 3", "Chevrolet Cobalt", "Chevrolet Lacetti (Gentra)",
    "Chevrolet Malibu", "Chevrolet Malibu 2", "Chevrolet Onix", "Chevrolet Tracker",
    "Chevrolet Equinox", "Chevrolet Traverse", "Chevrolet Tahoe", "Chevrolet Damas", "Chevrolet Labo"
]


def predict_purpose(age, gender, duration):
    if age < 26 and duration > 40:
        return "test drive"
    elif duration < 20:
        return "yangi mashina ko'rish"
    else:
        return "servis"


def predict_purchase_prob(age, gender, duration, purpose):
    prob = 0.3
    if purpose == 'test drive':
        prob += 0.25
    if duration > 60:
        prob += 0.15
    if age < 30:
        prob += 0.10
    if gender == "Erkak":
        prob += 0.07
    if gender == "Ayol":
        prob += 0.03
    if 20 <= age <= 30 and purpose == "test drive":
        prob += 0.1
    prob += random.uniform(-0.08, 0.08)
    prob = min(prob, 0.98)
    prob = max(prob, 0.03)
    return round(prob, 2)


def suggest_gm_models(age, gender, prob, purpose):
    """
    Suggest Chevrolet models based on age, gender, probability (confidence), and purpose.
    - Returns: (main_suggestion: str, alternatives: list)
    """
    # Normalize gender input
    gender = gender.lower()
    if gender == "erkak":
        gender = "male"
    elif gender == "ayol":
        gender = "female"

    # Define model groups for better targeting
    young_male = ["Chevrolet Onix", "Chevrolet Malibu", "Chevrolet Lacetti (Gentra)"]
    young_female = ["Chevrolet Spark", "Chevrolet Nexia 3", "Chevrolet Tracker"]
    mature_male = ["Chevrolet Malibu", "Chevrolet Tahoe", "Chevrolet Traverse"]
    mature_female = ["Chevrolet Malibu", "Chevrolet Tracker", "Chevrolet Cobalt"]
    senior = ["Chevrolet Tahoe", "Chevrolet Malibu 2", "Chevrolet Traverse"]
    general = ["Chevrolet Cobalt", "Chevrolet Nexia 3", "Chevrolet Onix"]
    test_drive_list = ["Chevrolet Malibu", "Chevrolet Lacetti (Gentra)", "Chevrolet Onix"]

    low_conf_young = ["Chevrolet Spark", "Chevrolet Nexia 3", "Chevrolet Cobalt"]
    low_conf_mid = ["Chevrolet Cobalt", "Chevrolet Onix", "Chevrolet Tracker"]
    low_conf_senior = ["Chevrolet Damas", "Chevrolet Malibu", "Chevrolet Labo"]

    fallback_young = ["Chevrolet Spark", "Chevrolet Nexia 3", "Chevrolet Cobalt"]
    fallback_mid = ["Chevrolet Nexia 3", "Chevrolet Cobalt", "Chevrolet Damas"]
    fallback_senior = ["Chevrolet Damas", "Chevrolet Labo", "Chevrolet Malibu 2"]

    # High confidence
    if prob > 0.8:
        # Only one main model for very high confidence
        if 20 <= age <= 30 and gender == "male":
            main = "Chevrolet Onix"
            alts = ["Chevrolet Malibu", "Chevrolet Lacetti (Gentra)"]
        elif 20 <= age <= 30 and gender == "female":
            main = "Chevrolet Spark"
            alts = ["Chevrolet Nexia 3", "Chevrolet Tracker"]
        elif age > 35 and gender == "male":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Tahoe", "Chevrolet Traverse"]
        elif age > 35 and gender == "female":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Tracker", "Chevrolet Cobalt"]
        elif age > 45:
            main = "Chevrolet Tahoe"
            alts = ["Chevrolet Malibu 2", "Chevrolet Traverse"]
        elif purpose and purpose.lower() == "test drive":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Lacetti (Gentra)", "Chevrolet Onix"]
        else:
            main = "Chevrolet Onix"
            alts = ["Chevrolet Cobalt", "Chevrolet Nexia 3"]
        return main, alts[:2]

    # Good confidence
    elif prob > 0.6:
        if 20 <= age <= 30 and gender == "male":
            main = "Chevrolet Onix"
            alts = ["Chevrolet Malibu", "Chevrolet Lacetti (Gentra)"]
        elif 20 <= age <= 30 and gender == "female":
            main = "Chevrolet Spark"
            alts = ["Chevrolet Nexia 3", "Chevrolet Tracker"]
        elif age > 35 and gender == "male":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Tahoe", "Chevrolet Traverse"]
        elif age > 35 and gender == "female":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Tracker", "Chevrolet Cobalt"]
        elif age > 45:
            main = "Chevrolet Tahoe"
            alts = ["Chevrolet Malibu 2", "Chevrolet Traverse"]
        elif purpose and purpose.lower() == "test drive":
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Lacetti (Gentra)", "Chevrolet Onix"]
        else:
            main = "Chevrolet Cobalt"
            alts = ["Chevrolet Nexia 3", "Chevrolet Onix"]
        return main, alts[:2]

    # Moderate confidence
    elif prob > 0.4:
        if age < 30:
            main = "Chevrolet Spark"
            alts = ["Chevrolet Nexia 3", "Chevrolet Cobalt"]
        elif age < 40:
            main = "Chevrolet Cobalt"
            alts = ["Chevrolet Onix", "Chevrolet Tracker"]
        else:
            main = "Chevrolet Malibu"
            alts = ["Chevrolet Damas", "Chevrolet Labo"]
        return main, alts[:2]

    # Low confidence / fallback
    else:
        if age < 30:
            main = "Chevrolet Spark"
            alts = ["Chevrolet Nexia 3", "Chevrolet Cobalt"]
        elif age < 45:
            main = "Chevrolet Nexia 3"
            alts = ["Chevrolet Cobalt", "Chevrolet Damas"]
        else:
            main = "Chevrolet Damas"
            alts = ["Chevrolet Labo", "Chevrolet Malibu 2"]
        return main, alts[:2]


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


def ai_suggestion_text(age, gender, prob, cars, purpose, top3_actual):
    """
    Generates an AI suggestion text for car models based on probability and user group.
    If probability is very high, shows only the most accurate model as a strong recommendation.
    """
    # If cars is a tuple (main, alternatives), use that; otherwise assume list
    if isinstance(cars, tuple):
        main_model = cars[0]
        alt_models = cars[1] if len(cars) > 1 else []
    else:
        main_model = cars[0] if cars else ""
        alt_models = cars[1:4] if len(cars) > 1 else []

    actual_list = ', '.join(top3_actual)

    if prob > 0.8:
        # Very high probability: recommend only the top model, emphasize accuracy
        text = (f"AI eng mos modelni aniqladi: {main_model}. "
                f"Sizga o'xshashlar orasida ham aynan shu model eng ko'p tanlangan. "
                f"Bu guruh uchun bu model trendda va tavsiya etiladi.")
    elif prob > 0.6:
        # High probability: show main and alternatives
        alt_list = ', '.join(alt_models)
        text = (f"AI tavsiya qilgan modellar: {main_model}" +
                (f", {alt_list}" if alt_list else "") +
                f". Sizga o'xshashlar orasida eng ko'p tanlanganlar: {actual_list}. "
                f"Bu guruh uchun aynan shu modellar trendda.")
    elif prob > 0.4:
        # Medium probability: soften recommendation, emphasize popularity
        model_list = ', '.join([main_model] + alt_models)
        text = (f"Xarid qilish ehtimoli o'rtacha. "
                f"Siz uchun ommabop variantlar: {model_list}. "
                f"Bu guruhda ko'p xarid qilinganlar: {actual_list}.")
    else:
        # Low probability: recommend affordable, reliable, encourage test drive
        model_list = ', '.join([main_model] + alt_models)
        text = (f"Ehtimol pastroq. Siz uchun arzon va ishonchli variantlar — {model_list}. "
                f"Bu guruhda so'nggi xaridlar: {actual_list}. Test drive tavsiya qilamiz.")

    return text


def stats_comparison(df, age, gender, purpose):
    """
    Compares user to their demographic group for a given purpose.
    - Handles age restrictions for each purpose.
    - Returns a meaningful Uzbek message.
    """
    # Example: age restrictions for each purpose (customize as needed)
    age_restrictions = {
        "test drive": (18, 70),
        "leasing": (21, 65),
        "purchase": (18, 75),
        # ... add more if needed
    }
    gender = gender.lower().capitalize()  # Ensure proper formatting

    # Check age restriction for the given purpose
    min_age, max_age = age_restrictions.get(purpose, (16, 99))
    if not (min_age <= age <= max_age):
        return (f"{purpose.title()} uchun yaroqli yosh oralig'i: {min_age}-{max_age} yil. "
                f"Sizning yoshingiz bu oralikda emas.")

    # Compute age group
    df = df.copy()
    df['age_group'] = (df['age'] // 5) * 5
    user_age_group = (age // 5) * 5

    # Filter by gender and age group
    group_df = df[(df['gender'].str.lower() == gender.lower()) & (df['age_group'] == user_age_group)]

    if len(group_df) == 0:
        return f"{user_age_group}-{user_age_group + 4} yoshli {gender.lower()}lar uchun '{purpose}' ma'lumotlari topilmadi."

    # Calculate percentage, avoid division by zero
    total = len(group_df)
    match_count = (group_df['purpose'] == purpose).sum()
    percent = int(100 * match_count / total) if total > 0 else 0

    return (f"{user_age_group}-{user_age_group + 4} yoshli {gender.lower()}lar ichida '{purpose}' ni tanlaganlar: {percent}%")


from collections import Counter
import numpy as np


def get_top3_models_by_demographic(df, age, gender, model_embeddings=None, car_list=None, clip_model=None,
                                   clip_preprocess=None):
    """
    Returns the top-3 car models for a demographic group.
    Uses semantic similarity (CLIP or other AI) if available and enough real data, else falls back to frequency.
    - model_embeddings: dict mapping car model to embedding vector (if using AI/CLIP)
    - car_list: list of all possible car models (for fallback)
    - clip_model/clip_preprocess: CLIP model/preprocess for semantic similarity
    """
    df = df.copy()
    df['age_group'] = (df['age'] // 5) * 5
    user_age_group = (age // 5) * 5
    group_df = df[(df['gender'].str.lower() == gender.lower()) & (df['age_group'] == user_age_group)]

    # If there are enough samples for AI ranking, use CLIP or other AI
    if not group_df.empty and model_embeddings and clip_model and clip_preprocess:
        all_cars = []
        for cars in group_df['suggested_cars']:
            all_cars.extend(cars)
        unique_cars = list(set(all_cars))
        # Compute popularity as weight
        car_counter = Counter(all_cars)
        car_weights = np.array([car_counter[car] for car in unique_cars], dtype=np.float32)
        car_weights /= car_weights.sum()
        # Get text embeddings for demographic description
        desc = f"{user_age_group}-{user_age_group + 4} {gender.lower()} recommended car"
        import torch
        with torch.no_grad():
            desc_tokens = clip_preprocess(desc).unsqueeze(0)
            desc_features = clip_model.encode_text(desc_tokens)
            car_features = np.stack([model_embeddings[car] for car in unique_cars])
            # Cosine similarity
            desc_features_np = desc_features.cpu().numpy()
            sims = np.dot(car_features, desc_features_np.T).flatten()
            # Weighted score: combine similarity and popularity
            scores = 0.7 * sims + 0.3 * car_weights
            top_indices = np.argsort(scores)[::-1][:3]
            models = [unique_cars[i] for i in top_indices]
        return models

    # Fallback to simple frequency for smaller datasets or if no AI available
    if not group_df.empty:
        all_cars = []
        for cars in group_df['suggested_cars']:
            all_cars.extend(cars)
        counter = Counter(all_cars)
        models = [c for c, _ in counter.most_common(3)]
        return models

    # If no data, return empty or a generic list if car_list provided
    if car_list:
        return car_list[:3]
    return []

def get_funnel_stats(df):
    total = len(df)
    test_drive = len(df[df['purpose'] == "test drive"])
    purchases = len(df[df['purchase'] == True])
    return total, test_drive, purchases


def get_repeat_stats(df):
    name_counts = df['name'].value_counts()
    repeat_customers = name_counts[name_counts > 1]
    repeat_ids = df[df['name'].isin(repeat_customers.index)]['name'].unique()
    repeat_purchases = df[df['name'].isin(repeat_ids) & (df['purchase'] == True)]
    return len(repeat_ids), len(repeat_purchases)


def get_time_trends(df):
    if 'entry_time' not in df or df.empty:
        return [], []
    df['day'] = df['entry_time'].dt.strftime('%Y-%m-%d')
    trend = df.groupby('day')['purchase'].sum().tail(14)
    return list(trend.index), list(trend.values)


def get_model_comparison(df, models):
    res = []
    for m in models:
        filtered = df[df['suggested_cars'].apply(lambda l: m in l)]
        if not filtered.empty:
            avg_age = int(filtered['age'].mean())
            gender = filtered['gender'].value_counts().idxmax()
            purchase_rate = round(filtered['purchase'].mean() * 100)
            res.append({'model': m, 'avg_age': avg_age, 'top_gender': gender, 'purchase_rate': purchase_rate})
    return res


def get_stats():
    visits = Visit.query.all()
    customers = {c.id: c for c in Customer.query.all()}
    rows = []
    for v in visits:
        cust = customers[v.customer_id]
        duration = (v.exit_time - v.entry_time).total_seconds() / 60
        suggested_car_list = v.suggested_cars.split(",") if v.suggested_cars else []
        rows.append({
            "name": cust.name,
            "age": cust.age,
            "gender": cust.gender,
            "phone": cust.phone,
            "purchase": cust.purchase,
            "entry_time": v.entry_time,
            "exit_time": v.exit_time,
            "duration": duration,
            "purpose": v.purpose,
            "suggested_cars": suggested_car_list,
            "purchase_prob": v.purchase_prob
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
    if df.empty:
        return {}, []
    # Filter out 'Unknown' or 'Noma\'lum' or empty/None from gender before counting!
    df_gender_filtered = df[df["gender"].isin(["Erkak", "Ayol"])]
    purpose_counts = df["purpose"].value_counts().to_dict()
    gender_counts = df_gender_filtered["gender"].value_counts().to_dict()
    purchase_counts = df["purchase"].value_counts().to_dict()
    avg_duration = round(df["duration"].mean(), 1)
    avg_age = round(df["age"].mean(), 1)
    car_flat = [c for sublist in df["suggested_cars"] for c in sublist]
    car_counts = pd.Series(car_flat).value_counts().to_dict()
    funnel_total, funnel_testdrive, funnel_purchases = get_funnel_stats(df)
    repeat_customers, repeat_purchases = get_repeat_stats(df)
    trend_labels, trend_data = get_time_trends(df)
    stats = {
        "purpose_counts": purpose_counts,
        "gender_counts": gender_counts,
        "purchase_counts": purchase_counts,
        "avg_duration": avg_duration,
        "avg_age": avg_age,
        "purpose_labels": list(purpose_counts.keys()),
        "purpose_data": list(purpose_counts.values()),
        "gender_labels": list(gender_counts.keys()),
        "gender_data": list(gender_counts.values()),
        "purchase_labels": ["Ha" if k else "Yo'q" for k in purchase_counts.keys()],
        "purchase_data": list(purchase_counts.values()),
        "durations": df["duration"].tolist(),
        "ages": df["age"].tolist(),
        "car_labels": list(car_counts.keys()),
        "car_data": list(car_counts.values()),
        "funnel": [funnel_total, funnel_testdrive, funnel_purchases],
        "repeat_customers": repeat_customers,
        "repeat_purchases": repeat_purchases,
        "trend_labels": trend_labels,
        "trend_data": trend_data
    }
    stats['df'] = df
    return stats, rows

def get_car_gender_percent(df):
    """
    Returns a dict: model -> {'Erkak': %, 'Ayol': %, 'total': %}
    Only includes customers with purchase=True and gender in ["Erkak", "Ayol"].
    Percentages are out of all purchases.
    """
    from collections import defaultdict

    # Only consider rows with purchase True and gender valid
    df_p = df[(df['purchase'] == True) & (df['gender'].isin(["Erkak", "Ayol"]))]
    if df_p.empty:
        return {}

    # Count buyers by model and gender
    model_gender_count = defaultdict(lambda: {"Erkak": 0, "Ayol": 0})
    total_by_gender = {"Erkak": 0, "Ayol": 0}
    total_purchases = 0

    for _, row in df_p.iterrows():
        gender = row["gender"]
        models = row["suggested_cars"] if isinstance(row["suggested_cars"], list) else []
        if not models:
            continue
        for m in models:
            model_gender_count[m][gender] += 1
        total_by_gender[gender] += 1
        total_purchases += 1

    # Calculate percents
    results = {}
    for model, gdict in model_gender_count.items():
        erkak_percent = round(100 * gdict["Erkak"] / total_by_gender["Erkak"], 1) if total_by_gender["Erkak"] else 0.0
        ayol_percent = round(100 * gdict["Ayol"] / total_by_gender["Ayol"], 1) if total_by_gender["Ayol"] else 0.0
        total_model = gdict["Erkak"] + gdict["Ayol"]
        total_percent = round(100 * total_model / total_purchases, 1) if total_purchases else 0.0
        results[model] = {"Erkak": erkak_percent, "Ayol": ayol_percent, "total": total_percent}
    return results


@app.route('/')
def index():
    stats, rows = get_stats()
    df = stats.get('df', pd.DataFrame())
    model_comparison = []
    top3_models = []
    if not df.empty:
        top3_models = get_top3_models_by_demographic(df, 30, "Erkak")
        model_comparison = get_model_comparison(df, top3_models)
    return render_template('index.html', stats=stats, visits=rows, gm_models=GM_MODELS,
                           top3_models=top3_models, model_comparison=model_comparison)


@app.route('/add_customer', methods=['GET', 'POST'])
def add_customer():
    if request.method == 'POST':
        try:
            name = request.form['name']
            age = int(request.form['age'])
            gender = request.form['gender']
            phone = request.form['phone']
            clothes = request.form['clothes']
            purchase = request.form.get('purchase') == 'yes'
            entry_time = request.form['entry_time']
            purpose = request.form['predicted_purpose']
            cars = request.form['suggested_cars']
            purchase_prob = float(request.form['purchase_prob'])

            # Remove exit_time input!
            entry_time_dt = datetime.strptime(entry_time, "%Y-%m-%dT%H:%M")
            predicted_duration_min = int(random.uniform(15, 90))
            exit_time_dt = entry_time_dt + timedelta(minutes=predicted_duration_min)

            # Ensure cars is a string
            if isinstance(cars, str):
                cars_str = cars
            elif isinstance(cars, list):
                cars_str = ",".join(str(x) for x in cars if isinstance(x, str))
            else:
                cars_str = str(cars)

            # ...rest as before...
            customer = Customer(
                name=name,
                age=age,
                gender=gender,
                phone=phone,
                clothes=clothes,
                purchase=purchase,
                created_by=CURRENT_USER,
                created_at=datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S")
            )
            db.session.add(customer)
            db.session.flush()

            visit = Visit(
                customer_id=customer.id,
                entry_time=entry_time_dt,
                exit_time=exit_time_dt,
                purpose=purpose,
                suggested_cars=cars_str,
                purchase_prob=purchase_prob,
                ml_purchase_prob=random.uniform(0.2, 0.8),
                created_by=CURRENT_USER,
                created_at=datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S")
            )
            db.session.add(visit)
            db.session.commit()
            flash('Mijoz va tashrif ma\'lumotlari saqlandi!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            db.session.rollback()
            flash('Xatolik: {}'.format(str(e)), 'danger')
            return redirect(url_for('add_customer'))
    return render_template('add_customer.html')


def seed_fake_data():
    import random
    from datetime import datetime, timedelta, time as dt_time

    # Current timestamp from parameters
    CURRENT_TIMESTAMP = "2025-05-24 14:40:01"
    CURRENT_USER = "Farkhodov721"

    clothes_options = [
        "Jeans va futbolka", "Kostyum", "Sport kiyim", "Yengil ko'ylak",
        "Libos", "Ofis kiyim", "Shortik va futbolka", "Polo va shim",
        "Klassik kostyum", "Futbolka va shortik"
    ]

    gm_models = [
        "Chevrolet Spark", "Chevrolet Nexia 3", "Chevrolet Cobalt", "Chevrolet Lacetti",
        "Chevrolet Gentra", "Chevrolet Malibu", "Chevrolet Malibu 2", "Chevrolet Onix",
        "Chevrolet Tracker", "Chevrolet Equinox", "Chevrolet Traverse", "Chevrolet Tahoe",
        "Chevrolet Damas", "Chevrolet Labo"
    ]

    if Customer.query.count() > 0:
        return

    names = [
        "Ali", "Vali", "Aziza", "Malika", "Jasur", "Dilshod", "Gulnoza", "Sardor", "Nodira", "Javlon",
        "Bobur", "Nargiza", "Shahnoza", "Sherzod", "Umida", "Shoxrux", "Nigora", "Ulugbek", "Kamola", "Sanjar"
    ]
    surnames = [
        "Tursunov", "Rahmonov", "Karimova", "Sattorov", "Najmiddinov", "Ibragimov", "Xolmatova", "Mirzayev"
    ]
    genders = ["Erkak", "Ayol"]
    purposes = ["test drive", "servis", "yangi mashina ko'rish"]

    def probable_purchase(age, gender, clothes, purpose):
        base = 0.5
        if gender == "Ayol":
            base -= 0.05
        if 26 <= age <= 40:
            base += 0.08
        if age > 50:
            base -= 0.10
        if "kostyum" in clothes.lower() or "klassik" in clothes.lower():
            base += 0.08
        if "sport" in clothes.lower():
            base -= 0.05
        if purpose == "test drive":
            base += 0.08
        if purpose == "servis":
            base -= 0.08
        base = max(0.15, min(0.85, base))
        return random.random() < base

    def choose_gm_model(age, gender, clothes, purpose):
        pool = gm_models.copy()
        weights = []
        for model in gm_models:
            w = 1
            if model == "Chevrolet Spark" and age < 30: w += 2
            if model == "Chevrolet Nexia 3" and age < 30: w += 1
            if model in ["Chevrolet Malibu", "Chevrolet Malibu 2"] and (
                    "kostyum" in clothes.lower() or age > 30): w += 2
            if model == "Chevrolet Gentra" and 25 <= age <= 40: w += 2
            if model == "Chevrolet Cobalt" and gender == "Erkak": w += 1
            if model == "Chevrolet Tracker" and 22 <= age <= 35: w += 1
            if model in ["Chevrolet Equinox", "Chevrolet Traverse", "Chevrolet Tahoe"] and age > 35: w += 1
            if model in ["Chevrolet Damas", "Chevrolet Labo"] and (age > 28 and random.random() < 0.3): w += 1
            if purpose == "servis" and model in ["Chevrolet Damas", "Chevrolet Labo"]: w += 1
            weights.append(max(w, 1))
        chosen = random.choices(pool, weights=weights, k=1)[0]
        alt_models = random.sample([m for m in gm_models if m != chosen], k=2)
        return [chosen] + alt_models

    def predict_purchase_prob(age, gender, duration, purpose):
        # Placeholder for probability calculation, you can adjust as needed
        base = 0.5
        if gender == "Ayol":
            base -= 0.05
        if 26 <= age <= 40:
            base += 0.08
        if age > 50:
            base -= 0.10
        if duration > 60:
            base += 0.07
        if purpose == "test drive":
            base += 0.08
        if purpose == "servis":
            base -= 0.08
        base = max(0.15, min(0.85, base))
        return round(base, 2)

    current_timestamp = datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S")

    for i in range(500):  # 500 users in DB, but you can display only 20 on frontend
        name = random.choice(names) + " " + random.choice(surnames)
        age = random.choices(
            population=[random.randint(18, 25), random.randint(26, 35), random.randint(36, 50), random.randint(51, 65)],
            weights=[0.29, 0.35, 0.23, 0.13]
        )[0]
        gender = random.choices(genders, weights=[0.56, 0.44])[0]
        phone = "+9989" + str(random.randint(100000000, 999999999))
        clothes = random.choice(clothes_options)

        # Generate a random arrival_time between 09:00:00 and 21:00:00
        hour = random.randint(9, 20)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        arrival_time = dt_time(hour=hour, minute=minute, second=second)

        # Purpose
        purpose = random.choices(purposes, weights=[0.3, 0.55, 0.15])[0] if age > 35 else \
            random.choices(purposes, weights=[0.45, 0.35, 0.2])[0]

        # Purchase
        purchase = probable_purchase(age, gender, clothes, purpose)

        # Model selection
        gm_selected = choose_gm_model(age, gender, clothes, purpose)
        customer_car_type = gm_selected[0] if purchase else None

        # Create customer
        customer = Customer(
            name=name,
            age=age,
            gender=gender,
            phone=phone,
            clothes=clothes,
            arrival_time=arrival_time,
            car_type=customer_car_type,
            purchase=purchase,
            created_at=current_timestamp,
            created_by=CURRENT_USER
        )
        db.session.add(customer)
        db.session.flush()

        # Visit data
        days_ago = random.randint(1, 180)
        base_date = current_timestamp - timedelta(days=days_ago)
        entry_hour = random.randint(9, 20)
        entry_min = random.randint(0, 59)
        entry_sec = random.randint(0, 59)
        entry_dt = base_date.replace(hour=entry_hour, minute=entry_min, second=entry_sec, microsecond=0)

        duration = random.randint(10, 110)
        exit_dt = entry_dt + timedelta(minutes=duration)

        if exit_dt.hour > 21 or (exit_dt.hour == 21 and exit_dt.minute > 0):
            exit_dt = exit_dt.replace(hour=21, minute=0, second=0)

        # Calculate probabilities
        prob = predict_purchase_prob(age, gender, duration, purpose)
        ml_prob = random.uniform(0.2, 0.8)  # Placeholder for ML prediction

        visit = Visit(
            customer_id=customer.id,
            entry_time=entry_dt,
            exit_time=exit_dt,
            purpose=purpose,
            suggested_cars=",".join(gm_selected),
            purchase_prob=prob,
            ml_purchase_prob=ml_prob,
            car_type=customer_car_type,
            created_at=current_timestamp,
            created_by=CURRENT_USER
        )
        db.session.add(visit)

        # Commit every 100 records
        if i % 100 == 0:
            db.session.commit()

    db.session.commit()

def flatten_str_list(lst):
    """Recursively flattens a list of strings/lists into a simple string list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_str_list(item))
        else:
            result.append(str(item))
    return result

@app.route('/generate_statistics', methods=['POST'])
def generate_statistics():
    try:
        data = request.json
        name = data.get('name', '').strip()
        age = int(data.get('age', 0))
        gender = data.get('gender', '')
        clothes = data.get('clothes', '')
        entry_time = data.get('entry_time', '')

        if not name or not gender or not entry_time:
            return jsonify({'error': "Barcha maydonlarni to'ldiring!"}), 400
        if age < 18:
            return jsonify({'error': "Yosh 18 dan kichik bo'lishi mumkin emas!"}), 400

        try:
            entry_time_dt = datetime.strptime(entry_time, "%Y-%m-%dT%H:%M")
        except Exception:
            return jsonify({'error': "Vaqt formatida xatolik!"}), 400

        if not (9 <= entry_time_dt.hour <= 21):
            return jsonify({'error': "Ish vaqti 09:00-21:00 oralig'ida bo'lishi kerak!"}), 400
        if entry_time_dt.weekday() == 6:
            return jsonify({'error': "Yakshanba dam olish kuni, boshqa kun tanlang!"}), 400

        # Duration: you may want to set a default or use a random value
        # For demo, set duration as 35 minutes (or as you see fit)
        duration = 35

        purpose = predict_purpose(age, gender, duration)

        # Get both predictions
        heuristic_prob = predict_purchase_prob(age, gender, duration, purpose)

        # ML prediction
        X = pd.DataFrame([{
            'age': age,
            'gender': 1 if gender == "Erkak" else 0,
            'clothes': clothes_to_numeric(clothes),
            'duration': duration
        }])
        ml_prob = random.uniform(0.2, 0.8)  # Placeholder for ML prediction

        # Combined prediction (70% heuristic, 30% ML)
        final_prob = 0.7 * heuristic_prob + 0.3 * ml_prob

        cars = suggest_gm_models(age, gender, final_prob, purpose)
        stats, _ = get_stats()
        df = pd.DataFrame(_)

        purchase_total = len(df)
        purchase_count = int(df['purchase'].sum()) if not df.empty else 0
        purchase_percent = (purchase_count / purchase_total * 100) if purchase_total else 0

        group_purchase_count = 0
        group_total = 0
        group_purchase_percent = 0
        model_stats = []

        if not df.empty:
            df['age_group'] = (df['age'] // 5) * 5
            user_age_group = (age // 5) * 5
            group_df = df[(df.gender == gender) & (df.age_group == user_age_group)]
            group_total = len(group_df)
            group_purchase_count = int(group_df['purchase'].sum())
            group_purchase_percent = (group_purchase_count / group_total * 100) if group_total else 0

            all_models = []
            for cars_list in group_df['suggested_cars']:
                all_models.extend(flatten_str_list(cars_list))
            if all_models:
                from collections import Counter
                model_counter = Counter(all_models)
                group_model_total = sum(model_counter.values())
                for model, cnt in model_counter.most_common(3):
                    percent = 100 * cnt / group_model_total if group_model_total > 0 else 0
                    model_stats.append({'model': model, 'percent': percent})

            group_df = group_df.sort_values('age')
            line_labels = group_df['age'].tolist()
            line_data = [int(100 * p) for p in group_df['purchase_prob']]
        else:
            line_labels = [age]
            line_data = [int(final_prob * 100)]

        top3_actual = [m['model'] for m in model_stats[:3]] if model_stats else ["Chevrolet Cobalt",
                                                                                 "Chevrolet Nexia 3", "Chevrolet Spark"]

        percent = int(100 * (group_purchase_count / group_total)) if group_total else int(final_prob * 100)
        comparison = f"{(age // 5) * 5}-{(age // 5) * 5 + 4} yoshli {gender.lower()}lar ichida xarid qilish ehtimoli: {percent}%"

        suggestion_text = ai_suggestion_text(age, gender, final_prob, cars, purpose, top3_actual)

        return jsonify({
            "purpose": purpose,
            "purpose_text": f"{name} ({gender}, {age} yosh) uchun AI taxmini: {purpose} (salonda {int(duration)} daqiqa)",
            "comparison": comparison,
            "suggested_cars": ", ".join(flatten_str_list(cars)),
            "heuristic_prob": int(heuristic_prob * 100),
            "ml_prob": int(ml_prob * 100),
            "final_prob": int(final_prob * 100),
            "suggestion_text": suggestion_text,
            "line_labels": line_labels,
            "line_data": line_data,
            "top3_actual": ", ".join(flatten_str_list(top3_actual)),
            "purchase_total": purchase_total,
            "purchase_count": purchase_count,
            "purchase_percent": purchase_percent,
            "group_purchase_count": group_purchase_count,
            "group_total": group_total,
            "group_purchase_percent": group_purchase_percent,
            "model_stats": model_stats
        })
    except Exception as e:
        return jsonify({'error': f"Ichki xatolik: {str(e)}"}), 500

@app.route('/camera_monitor')
def camera_monitor():
    return render_template('camera_monitor.html',
                         current_user=CURRENT_USER,
                         current_time=CURRENT_TIMESTAMP)



@app.route('/api/analyze', methods=['POST'])
def analyze_person():
    try:
        data = request.get_json()
        frame, results = person_analyzer.analyze_frame(data['image'])

        if not results:
            return jsonify({
                'status': 'error',
                'message': 'No faces detected'
            }), 400

        gender_mapping = {
            'male': 'Erkak',
            'female': 'Ayol',
            'unknown': "Noma'lum"
        }

        # Use all detected faces for real-time UI
        formatted_results = []
        for det in results:
            gender_pred = det['gender']['prediction']
            gender_conf = det['gender']['confidence']
            if gender_pred in ['male', 'female'] and gender_conf >= 0.6:
                mapped_gender = gender_mapping[gender_pred]
            else:
                mapped_gender = gender_mapping['unknown']

            clothing_type = det['clothing']['type']
            clothing_conf = det['clothing']['confidence']
            # If you have Uzbek labels directly from detect_clothing, use as is.
            # Otherwise, you can add a mapping for English->Uzbek (not hardcoded default!).
            mapped_clothes = clothing_type if clothing_conf >= 0.45 else "Noma'lum"

            formatted_results.append({
                "age": det['age'],
                "gender": {
                    "prediction": mapped_gender,
                    "confidence": gender_conf
                },
                "clothing": {
                    "type": mapped_clothes,
                    "confidence": clothing_conf
                },
                "face_coordinates": det['face_coordinates']
            })

        # Register the most confident detection (gender+clothing) as customer
        best_detection = max(
            results,
            key=lambda d: d['gender']['confidence'] + d['clothing']['confidence']
        )

        best_gender_pred = best_detection['gender']['prediction']
        best_gender_conf = best_detection['gender']['confidence']
        best_gender = gender_mapping[best_gender_pred] if best_gender_pred in ['male', 'female'] and best_gender_conf >= 0.6 else gender_mapping['unknown']

        best_clothing_type = best_detection['clothing']['type']
        best_clothing_conf = best_detection['clothing']['confidence']
        best_clothes = best_clothing_type if best_clothing_conf >= 0.45 else "Noma'lum"

        customer = Customer(
            name=f"Camera_Detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            age=int(best_detection['age']),
            gender=best_gender,
            phone="+998900000000",
            clothes=best_clothes,
            arrival_time=datetime.now().time(),
            created_at=datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S"),
            created_by=CURRENT_USER
        )
        db.session.add(customer)
        db.session.flush()

        entry_time = datetime.now()
        visit = Visit(
            customer_id=customer.id,
            entry_time=entry_time,
            exit_time=entry_time + timedelta(minutes=30),
            purpose="yangi mashina ko'rish",
            suggested_cars="",
            purchase_prob=0.5,
            created_at=datetime.strptime(CURRENT_TIMESTAMP, "%Y-%m-%d %H:%M:%S"),
            created_by=CURRENT_USER
        )
        db.session.add(visit)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Person data saved successfully',
            'id': customer.id,
            'results': formatted_results
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

import random

def flatten_str_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_str_list(item))
        else:
            result.append(str(item))
    return result

def generate_camera_statistics(age, gender, clothes):
    # 1. Predict purpose (randomly for demo)
    purposes = ["servis", "sotib olish", "test drive", "ma'lumot olish"]
    purpose = random.choice(purposes)

    # 2. Duration (fixed for demo)
    duration = 35

    # 3. Predict probabilities
    heuristic_prob = round(random.uniform(0.2, 0.5), 2)  # e.g. 0.35
    ml_prob = round(random.uniform(0.2, 0.5), 2)         # e.g. 0.24
    final_prob = round(0.7 * heuristic_prob + 0.3 * ml_prob, 2)
    final_prob_pct = int(final_prob * 100)

    # 4. Suggest models based on gender/age/final_prob/purpose
    if gender == "Erkak":
        suggested_cars = ["Chevrolet Nexia 3", "Chevrolet Cobalt", "Chevrolet Damas"]
        recent = ["Chevrolet Labo", "Chevrolet Malibu", "Chevrolet Malibu 2"]
    else:
        suggested_cars = ["Chevrolet Spark", "Chevrolet Tracker", "Chevrolet Onix"]
        recent = ["Chevrolet Matiz", "Chevrolet Tracker", "Chevrolet Onix"]
    suggestion_text = (
        f"Ehtimol pastroq. Siz uchun arzon va ishonchli variantlar — {', '.join(suggested_cars)}. "
        f"Bu guruhda so'nggi xaridlar: {', '.join(recent)}. Test drive tavsiya qilamiz."
    )

    # 5. Age group text
    age_group_start = (age // 5) * 5
    age_group_end = age_group_start + 4
    comparison = f"{age_group_start}-{age_group_end} yoshli {gender.lower()}lar ichida xarid qilish ehtimoli: {final_prob_pct}%"

    # 6. Compose text
    result = {
        "purpose": purpose,
        "purpose_text": f"{gender}, {age} yosh uchun AI taxmini: {purpose} (salonda {duration} daqiqa)",
        "comparison": comparison,
        "heuristic_prob": int(heuristic_prob * 100),
        "ml_prob": int(ml_prob * 100),
        "final_prob": final_prob_pct,
        "suggested_cars": ", ".join(flatten_str_list(suggested_cars)),
        "suggested_cars_list": flatten_str_list(suggested_cars),
        "suggestion_text": suggestion_text,
        "actual_top3": recent
    }
    return result

# --- Example usage ---
if __name__ == "__main__":
    age = 43
    gender = "Erkak"
    clothes = "Kostyum"
    stats = generate_camera_statistics(age, gender, clothes)
    print(stats["purpose_text"])
    print(stats["comparison"])
    print(f"Heuristic model taxmini: {stats['heuristic_prob']}%")
    print(f"ML model taxmini: {stats['ml_prob']}%")
    print(f"Umumiy xarid ehtimoli: {stats['final_prob']}%")
    print(f"Tavsiya etilgan modellar:\n\n{stats['suggested_cars']}")
    print(stats["suggestion_text"])


if __name__ == '__main__':
    if os.path.exists("avtosalon.db"):
        os.remove("avtosalon.db")
    with app.app_context():
        db.create_all()
        seed_fake_data()
    app.run(debug=True)