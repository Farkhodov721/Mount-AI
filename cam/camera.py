import cv2
import numpy as np
from datetime import datetime
import requests
import base64
from deepface import DeepFace
import torch
import clip
from PIL import Image

class PersonAnalyzer:
    def __init__(self):
        print("🔧 Инициализация системы анализа...")

        self.api_url = "http://127.0.0.1:5000/api/analyze"
        self.check_api()

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.outfit_labels = [
            "formal",
            "casual",
            "homewear",
            "dress"
        ]
        self.outfit_tokens = clip.tokenize(self.outfit_labels).to("cpu")

        print("✅ Система готова к работе")

    def check_api(self):
        try:
            requests.get("http://127.0.0.1:5000/", timeout=2)
            print("✅ API доступен")
            return True
        except:
            print("❌ API недоступен - сначала запустите app.py")
            return False

    def detect_age_gender(self, face_image):
        try:
            result = DeepFace.analyze(face_image, actions=['age', 'gender'], enforce_detection=False)[0]
            print("📦 Ответ DeepFace:", result)

            raw_age = result.get('age', -1)
            age = str(int(raw_age)) if isinstance(raw_age, (int, float, np.integer)) else str(raw_age)

            dominant_gender = result.get('dominant_gender', 'unknown').lower()
            gender_scores = result.get('gender', {})
            gender_conf = float(gender_scores.get(dominant_gender.capitalize(), 0.0))

            return age, dominant_gender, gender_conf
        except Exception as e:
            print(f"⚠️ Ошибка DeepFace: {e}")
            return "unknown", "unknown", 0.0

    def detect_clothing(self, body_image):
        try:
            pil_image = Image.fromarray(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB))
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to("cpu")

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(self.outfit_tokens)
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            best_index = int(np.argmax(probs))
            label = self.outfit_labels[best_index]
            confidence = float(probs[best_index])
            return label, confidence
        except Exception as e:
            print(f"⚠️ Ошибка определения одежды: {e}")
            return "unknown", 0.0

    def send_to_api(self, data):
        try:
            response = requests.post(self.api_url, json=data, timeout=3)
            if response.status_code == 200:
                print(f"✅ Данные сохранены (ID: {response.json().get('id', 'unknown')})")
                return True
            print(f"❌ Ошибка API: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка отправки: {e}")
        return False

    def start_capture(self):
        print("🎥 Запуск камеры...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Не удалось открыть камеру")
            return

        print("⌨️ Управление:")
        print("   s - сохранить текущие лица")
        print("   q - выйти")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("❌ Ошибка чтения кадра")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                body_roi = frame[y + h:y + 3 * h, x:x + w] if y + 3 * h < frame.shape[0] else frame[y + h:, x:x + w]

                if face_roi.size > 0:
                    age, gender, gender_conf = self.detect_age_gender(face_roi)
                    if body_roi.size > 0:
                        clothing_type, clothing_conf = self.detect_clothing(body_roi)
                    else:
                        clothing_type, clothing_conf = "unknown", 0.0

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{gender} ({gender_conf:.1f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Age: {age}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"{clothing_type}", (x, y + h + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.putText(frame, "Нажмите 's' для сохранения, 'q' для выхода", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Person Analyzer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("📸 Сохранение данных...")
                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = frame[y:y + h, x:x + w]
                    body_roi = frame[y + h:y + 3 * h, x:x + w] if y + 3 * h < frame.shape[0] else frame[y + h:, x:x + w]

                    if face_roi.size > 0:
                        age, gender, gender_conf = self.detect_age_gender(face_roi)
                        if body_roi.size > 0:
                            clothing_type, clothing_conf = self.detect_clothing(body_roi)
                        else:
                            clothing_type, clothing_conf = "unknown", 0.0

                        _, buffer = cv2.imencode('.jpg', face_roi)
                        face_b64 = base64.b64encode(buffer).decode('utf-8')

                        self.send_to_api({
                            "id": f"person_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "face_coordinates": {
                                "x": int(x),
                                "y": int(y),
                                "width": int(w),
                                "height": int(h)
                            },
                            "age": age,
                            "gender": {
                                "prediction": gender,
                                "confidence": float(gender_conf)
                            },
                            "clothing": {
                                "type": clothing_type,
                                "confidence": float(clothing_conf)
                            },
                            "face_image": face_b64
                        })

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Анализ завершен")

def main():
    print("=" * 50)
    print("🎯 Система анализа лиц с определением возраста, пола и одежды")
    print("=" * 50)
    analyzer = PersonAnalyzer()
    analyzer.start_capture()

if __name__ == '__main__':
    main()
