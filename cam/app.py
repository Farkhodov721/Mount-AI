from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///person_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class PersonDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Face coordinates
    face_x = db.Column(db.Integer)
    face_y = db.Column(db.Integer)
    face_width = db.Column(db.Integer)
    face_height = db.Column(db.Integer)
    
    # Age and gender
    age = db.Column(db.String(20))
    gender = db.Column(db.String(20))
    gender_confidence = db.Column(db.Float)
    
    # Clothing
    clothing_type = db.Column(db.String(50))
    clothing_confidence = db.Column(db.Float)
    
    # Face image
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
            'age': self.age,
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
        
        detection = PersonDetection(
            person_id=data['id'],
            face_x=data['face_coordinates']['x'],
            face_y=data['face_coordinates']['y'],
            face_width=data['face_coordinates']['width'],
            face_height=data['face_coordinates']['height'],
            age=data.get('age', 'unknown'),
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
    detections = PersonDetection.query.order_by(PersonDetection.timestamp.desc()).all()
    return jsonify([detection.to_dict() for detection in detections])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total = PersonDetection.query.count()
    male_count = PersonDetection.query.filter_by(gender='male').count()
    female_count = PersonDetection.query.filter_by(gender='female').count()
    
    age_stats = db.session.query(
        PersonDetection.age,
        db.func.count(PersonDetection.age)
    ).group_by(PersonDetection.age).all()
    
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
        'age_distribution': dict(age_stats),
        'clothing_distribution': dict(clothing_stats)
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Person Detection API',
        'endpoints': {
            'POST /api/analyze': 'Save person detection data',
            'GET /api/detections': 'Get all detections',
            'GET /api/stats': 'Get statistics'
        }
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=5000)