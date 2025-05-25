# camera_analyzer.py
import cv2
import numpy as np
from datetime import datetime
import requests

import json
import base64
import os

class PersonAnalyzer:
    def __init__(self):
        # Инициализация каскадов для детекции
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # URL вашего Flask API
        self.api_url = "http://localhost:5000/api/analyze"
    
    def detect_gender(self, face_roi):
        """
        Простая эвристика для определения гендера на основе размеров лица
        В реальном проекте используйте предобученные модели (dlib, OpenCV DNN)
        """
        height, width = face_roi.shape[:2]
        # Простая эвристика - в реальности нужна обученная модель
        ratio = width / height
        if ratio > 0.8:
            return "male", 0.6
        else:
            return "female", 0.6
    
    def analyze_clothing(self, body_roi):
        """
        Анализ одежды на основе цветовых характеристик
        В реальном проекте используйте CNN модели для классификации одежды
        """
        if body_roi is None or body_roi.size == 0:
            return "unknown", 0.0
        
        # Анализ доминирующих цветов
        colors = cv2.mean(body_roi)[:3]
        
        # Простая классификация по цвету (в реальности нужна CNN)
        if colors[2] > colors[1] and colors[2] > colors[0]:  # Красный канал
            clothing_type = "red_clothing"
        elif colors[1] > colors[0] and colors[1] > colors[2]:  # Зеленый канал
            clothing_type = "casual_wear"
        elif colors[0] > colors[1] and colors[0] > colors[2]:  # Синий канал
            clothing_type = "formal_wear"
        else:
            clothing_type = "mixed_clothing"
        
        return clothing_type, 0.7
    
    def send_to_api(self, data):
        """Отправка данных в Flask API"""
        try:
            response = requests.post(self.api_url, json=data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Данные успешно отправлены: {response.json()}")
                return True
            else:
                print(f"❌ Ошибка API: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка соединения: {e}")
            return False
    
    def encode_image(self, image):
        """Кодирование изображения в base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_frame(self, frame):
        """Анализ кадра и извлечение данных"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Детекция тел (для анализа одежды)
        bodies = self.body_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
        )
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Извлечение области лица
            face_roi = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]
            
            # Анализ гендера
            gender, gender_confidence = self.detect_gender(face_roi)
            
            # Поиск соответствующего тела для анализа одежды
            clothing_type, clothing_confidence = "unknown", 0.0
            body_roi = None
            
            for (bx, by, bw, bh) in bodies:
                # Проверяем, что лицо находится в верхней части тела
                if (x >= bx and x + w <= bx + bw and 
                    y >= by and y <= by + bh // 3):
                    body_roi = frame[by:by+bh, bx:bx+bw]
                    clothing_type, clothing_confidence = self.analyze_clothing(body_roi)
                    break
            
            # Рисование прямоугольников
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Отображение информации
            info_text = f"Gender: {gender} ({gender_confidence:.2f})"
            cv2.putText(frame, info_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            clothing_text = f"Clothing: {clothing_type}"
            cv2.putText(frame, clothing_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Подготовка данных для API
            person_data = {
                "id": f"person_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "face_coordinates": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "gender": {
                    "prediction": gender,
                    "confidence": float(gender_confidence)
                },
                "clothing": {
                    "type": clothing_type,
                    "confidence": float(clothing_confidence)
                },
                "face_image": self.encode_image(face_color) if face_color.size > 0 else None
            }
            
            results.append(person_data)
        
        return frame, results
    
    def start_capture(self):
        """Основной цикл захвата и анализа"""
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("❌ Не удалось открыть камеру")
            return
        
        print("🎥 Камера запущена. Нажмите 'q' для выхода, 's' для сохранения кадра в БД")
        
        frame_count = 0
        save_next = False
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("❌ Не удалось получить кадр")
                break
            
            # Анализ каждого 10-го кадра для производительности
            if frame_count % 10 == 0 or save_next:
                analyzed_frame, results = self.analyze_frame(frame.copy())
                
                # Отправка данных в API при нажатии 's' или автоматически
                if save_next and results:
                    for person_data in results:
                        self.send_to_api(person_data)
                    save_next = False
                
                cv2.imshow('Person Analyzer', analyzed_frame)
            else:
                cv2.imshow('Person Analyzer', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_next = True
                print("📸 Сохранение данных в следующем кадре...")
            
            frame_count += 1
        
        camera.release()
        cv2.destroyAllWindows()
        print("👋 Камера остановлена")


# flask_app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///person_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Модель базы данных
class PersonDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Координаты лица
    face_x = db.Column(db.Integer)
    face_y = db.Column(db.Integer)
    face_width = db.Column(db.Integer)
    face_height = db.Column(db.Integer)
    
    # Данные о гендере
    gender = db.Column(db.String(20))
    gender_confidence = db.Column(db.Float)
    
    # Данные об одежде
    clothing_type = db.Column(db.String(50))
    clothing_confidence = db.Column(db.Float)
    
    # Изображение лица (base64)
    face_image = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'timestamp': self.timestamp.isoformat(),
            'face_coordinates': {
                'x': self.face_x,
                'y': self.face_y,
                'width': self.face_width,
                'height': self.face_height
            },
            'gender': {
                'prediction': self.gender,
                'confidence': self.gender_confidence
            },
            'clothing': {
                'type': self.clothing_type,
                'confidence': self.clothing_confidence
            }
        }

@app.route('/api/analyze', methods=['POST'])
def analyze_person():
    try:
        data = request.get_json()
        
        # Создание записи в БД
        detection = PersonDetection(
            person_id=data['id'],
            face_x=data['face_coordinates']['x'],
            face_y=data['face_coordinates']['y'],
            face_width=data['face_coordinates']['width'],
            face_height=data['face_coordinates']['height'],
            gender=data['gender']['prediction'],
            gender_confidence=data['gender']['confidence'],
            clothing_type=data['clothing']['type'],
            clothing_confidence=data['clothing']['confidence'],
            face_image=data.get('face_image')
        )
        
        db.session.add(detection)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Person data saved successfully',
            'id': detection.id
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Получение всех детекций"""
    detections = PersonDetection.query.order_by(PersonDetection.timestamp.desc()).all()
    return jsonify([detection.to_dict() for detection in detections])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Статистика по детекциям"""
    total = PersonDetection.query.count()
    male_count = PersonDetection.query.filter_by(gender='male').count()
    female_count = PersonDetection.query.filter_by(gender='female').count()
    
    clothing_stats = db.session.query(
        PersonDetection.clothing_type,
        db.func.count(PersonDetection.clothing_type)
    ).group_by(PersonDetection.clothing_type).all()
    
    return jsonify({
        'total_detections': total,
        'gender_distribution': {
            'male': male_count,
            'female': female_count
        },
        'clothing_distribution': dict(clothing_stats)
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("🚀 Flask API запущен на http://localhost:5000")
    print("📊 Доступные эндпоинты:")
    print("  POST /api/analyze - сохранение данных о персоне")
    print("  GET /api/detections - получение всех детекций")
    print("  GET /api/stats - статистика")
    
    app.run(debug=True, port=5000)


# run_camera.py - запуск только камеры
if __name__ == '__main__':
    print("🎥 Запуск камеры анализа персон")
    print("📋 Убедитесь, что Flask API запущен на localhost:5000")
    print("⌨️  Управление:")
    print("   's' - сохранить данные в БД")
    print("   'q' - выйти")
    
    analyzer = PersonAnalyzer()
    analyzer.start_capture()


