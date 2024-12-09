import cv2
import numpy as np

# Load pre-trained models
face_proto = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/opencv_face_detector.pbtxt"
face_model = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/opencv_face_detector_uint8.pb"
age_proto = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/deploy_age.prototxt"
age_model = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/age_net.caffemodel"
gender_proto = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/deploy_gender.prototxt"
gender_model = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/gender_net.caffemodel"

# Model mean values for age and gender classification
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender classes
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

# Load networks
face_net = cv2.dnn.readNet(face_model, face_proto)
try:
    age_net = cv2.dnn.readNet(age_model, age_proto)
    print("Model loaded successfully!")
except cv2.error as e:
    print("Error loading model:", e)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    return face_net.forward()

def main(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    detected_faces = detect_faces(image)
    count = 0
    for i in range(detected_faces.shape[2]):
        confidence = detected_faces[0, 0, i, 2]
        if confidence > 0.5:
            count += 1
            box = detected_faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)

            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classes[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_classes[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 1)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    font_scale = 0.5
    font_thickness = 2
    text = f"People detected: {count}"
    print(count)
            
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    bottom_right = (w - text_width - 1, h - 1)

    cv2.putText(image, text, bottom_right, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)       
    
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the image path
image_path = "C:/Users/Aryan Raj/Documents/projects/python/opencv_face_age_detection/Screenshot 2024-11-30 104602.png"
main(image_path)
