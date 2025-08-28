import sys, json, numpy as np, tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (128,128)
ARTIFACTS_DIR = "artifacts"

if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    sys.exit(1)

# load model & class names
model = tf.keras.models.load_model(f"{ARTIFACTS_DIR}/model.keras")
with open(f"{ARTIFACTS_DIR}/class_names.json","r") as f:
    class_names = json.load(f)

# load & preprocess image
img = load_img(sys.argv[1], target_size=IMG_SIZE)
x = img_to_array(img)
x = np.expand_dims(x, axis=0).astype("float32")
x = preprocess_input(x)

# predict
prob = float(model.predict(x, verbose=0)[0][0])
pred_idx = int(prob >= 0.5)
print(f"Prediction: {class_names[pred_idx]} (confidence={prob:.4f})")
