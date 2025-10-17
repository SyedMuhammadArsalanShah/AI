import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="centered")

st.title("🧠 MNIST Handwritten Digit Classifier")
st.markdown("### Built with TensorFlow 2.0 and Streamlit\n"
            "Draw or upload a digit (0–9) and let the model predict!")

# -------------------- Model Definition --------------------
@st.cache_resource
def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.sidebar.success(f"✅ Model trained (Accuracy: {test_acc:.2%})")
    return model

model = train_model()

# -------------------- Drawing / Upload Section --------------------
st.sidebar.header("🖋️ Input Options")
option = st.sidebar.radio("Select input mode:", ["Draw a digit", "Upload an image"])

if option == "Draw a digit":
    from streamlit_drawable_canvas import st_canvas

    st.write("Draw a **digit (0–9)** below 👇")
    canvas_result = st_canvas(
        fill_color="#000000", stroke_width=15, stroke_color="#FFFFFF",
        background_color="#000000", height=280, width=280, drawing_mode="freedraw", key="canvas"
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        img_array = np.array(img).reshape(1, 28, 28) / 255.0

        if st.button("🔍 Predict"):
            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction)
            st.success(f"### Predicted Digit: {pred_class}")
            st.bar_chart(prediction[0])

elif option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image of a digit (28x28 or larger):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img = ImageOps.invert(image.resize((28, 28)))
        img_array = np.array(img).reshape(1, 28, 28) / 255.0

        if st.button("🔍 Predict"):
            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction)
            st.success(f"### Predicted Digit: {pred_class}")
            st.bar_chart(prediction[0])

st.markdown("---")
st.caption("Developed by **Syed Muhammad Arsalan Shah Bukhari** – Powered by TensorFlow ⚡")
