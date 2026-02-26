import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Judul & Header Kece
st.set_page_config(page_title="AI Diagnosa Paru Adam", layout="centered")
st.title("🩺 AI Deteksi Pneumonia")
st.write("Upload foto rontgen untuk mendapatkan hasil diagnosa dan tingkat keyakinan AI.")

# 2. Load Model Baru (Pastikan nama file .h5 lu bener ya!)
@st.cache_resource
def load_my_model():
    # Ganti nama file di bawah kalau lu kasih nama lain pas save tadi
    return tf.keras.models.load_model('model_pneumonia_v2_1000data.h5')

model = load_my_model()

# 3. Sidebar / Upload Area
uploaded_file = st.file_uploader("Pilih file gambar rontgen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar yang diupload
    img = Image.open(uploaded_file).convert('RGB') # Cegah error grayscale
    st.image(img, caption='Gambar Rontgen Terpilih', use_column_width=True)
    
    # Preprocessing (Ukuran harus 150x150 sesuai training)
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Tombol Analisis
    if st.button('Mulai Diagnosa'):
        with st.spinner('AI sedang menganalisa gambar...'):
            prediction = model.predict(img_array)
            skor_mentah = prediction[0][0]
            
            # 4. Logika Persentase & Diagnosa
            if skor_mentah > 0.5:
                diagnosa = "PNEUMONIA"
                persentase = skor_mentah * 100
                st.error(f"### HASIL DIAGNOSA: {diagnosa}")
            else:
                diagnosa = "NORMAL"
                persentase = (1 - skor_mentah) * 100
                st.success(f"### HASIL DIAGNOSA: {diagnosa}")

            # Tampilan Persentase Rapi
            st.metric(label="Tingkat Keyakinan AI", value=f"{persentase:.2f}%")
            
            # Catatan Tambahan
            st.info("Catatan: Ini adalah hasil prediksi AI. Harap konsultasikan kembali dengan dokter spesialis.")