import cv2
import torch
import time
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

id2label = {
    0: "drinking",
    1: "other_activities",
    2: "safe_driving",
    3: "sleeping",
    4: "talking_phone",
    5: "texting_phone",
    6: "turning",
    7: "yawning"
}

model_name = "Ganaa614/vit-tiny-patch16-224cabin_activity_recognition"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

device = "cpu"
model.to(device)

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_id = preds.argmax(-1).item()
        label = id2label[label_id]
        confidence = preds[0][label_id].item()

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Cabin Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
