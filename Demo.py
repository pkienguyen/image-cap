from textwrap import wrap
import numpy as np
import pandas as pd
import os
from keras import Model
from keras.src.applications.densenet import DenseNet201
from keras.src.saving import load_model
from keras_preprocessing.image import load_img, img_to_array
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot as plt


# Đường dẫn đến ảnh và mô hình
image_file = "dog1.jpg"
model_path = "model.keras"

# Hàm load ảnh, chuyển cỡ sang 224x224
def readImage(path, img_size=224):
    img = load_img(path, color_mode="rgb", target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.
    return img

# Hàm chuyển index sang từ
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Hàm dự đoán caption
def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text

# Load model
caption_model = load_model(model_path)

# Xử lý file captions.txt để lấy tokenizer và max_length
data = pd.read_csv("flickr8k/captions.txt")

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

data = text_preprocessing(data)
captions = data['caption'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Trích xuất đặc trưng ảnh sử dụng DenseNet201
weights_path = "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
base_model = DenseNet201(weights=weights_path, include_top=False, pooling="avg")  # Thêm pooling='avg'
fe = Model(inputs=base_model.input, outputs=base_model.output)  # Đầu ra giờ là (None, 1920)

# Load ảnh và trích xuất đặc trưng
img_path = os.path.join(image_file)
img = readImage(img_path)

# Thêm batch dimension
img = np.expand_dims(img, axis=0)

feature = fe.predict(img, verbose=0)

# Lưu đặc trưng vào dictionary
features = {image_file: feature}

# Dự đoán caption
caption = predict_caption(caption_model, image_file, tokenizer, max_length, features)

# Hiển thị ảnh và caption
plt.figure(figsize=(8, 8))
plt.imshow(img[0])  # hiển thị ảnh đầu tiên trong batch
plt.title("\n".join(wrap(caption, 40)))  # hiển thị caption với wrap text
plt.axis("off")
plt.show()

print(caption)
