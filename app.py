import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# NLP INDONESIA
# =========================================================
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================================================
# MACHINE LEARNING
# =========================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# KONFIGURASI STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Analisis Sentimen Tim Tarsius",
    page_icon="👮",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.title("👮 ANALISIS SENTIMEN KOMENTAR MASYARAKAT")
st.subheader(
    "Terhadap Tim Tarsius Polres Bitung pada Media Sosial YouTube "
    "Menggunakan Metode Support Vector Machine (SVM)"
)

st.markdown("""
### Metode Penelitian:
- Case Folding
- Cleansing
- Stopword Removal
- Stemming
- Pelabelan Otomatis
- TF-IDF
- Support Vector Machine (SVM)
""")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

# =========================================================
# INISIALISASI NLP
# =========================================================
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# =========================================================
# KAMUS SENTIMEN
# =========================================================
positive_words = [
    "bagus", "mantap", "hebat", "keren", "baik",
    "luar biasa", "top", "cepat", "sigap",
    "berhasil", "apresiasi", "terbaik",
    "salut", "aman", "percaya", "membantu"
]

negative_words = [
    "buruk", "jelek", "lambat", "parah",
    "gagal", "takut", "kasar", "jahat",
    "kecewa", "marah", "bohong", "rusak",
    "tidak becus", "kriminal", "preman"
]

# =========================================================
# FUNGSI PREPROCESSING
# =========================================================
def preprocessing(text):

    text = str(text).lower()

    # hapus url
    text = re.sub(r"http\S+|www\S+", "", text)

    # hapus mention
    text = re.sub(r"@\w+", "", text)

    # hapus hashtag
    text = re.sub(r"#\w+", "", text)

    # hapus angka
    text = re.sub(r"\d+", "", text)

    # hapus simbol
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # stopword removal
    text = stopword.remove(text)

    # stemming
    text = stemmer.stem(text)

    return text

# =========================================================
# FUNGSI LABEL OTOMATIS
# =========================================================
def auto_labeling(text):

    text = text.lower()

    positive_score = 0
    negative_score = 0

    for word in positive_words:
        if word in text:
            positive_score += 1

    for word in negative_words:
        if word in text:
            negative_score += 1

    if positive_score > negative_score:
        return "Positif"

    elif negative_score > positive_score:
        return "Negatif"

    else:
        return "Netral"

# =========================================================
# MAIN PROGRAM
# =========================================================
if uploaded_file is not None:

    # =====================================================
    # MEMBACA CSV
    # =====================================================
    try:
        df = pd.read_csv(uploaded_file)

    except:
        df = pd.read_csv(uploaded_file, encoding='latin1')

    st.success("✅ Dataset berhasil dimuat")

    # =====================================================
    # TAMPILKAN KOLOM
    # =====================================================
    st.subheader("📋 Informasi Dataset")

    st.write("Kolom Dataset:")
    st.write(list(df.columns))

    # =====================================================
    # VALIDASI KOLOM
    # =====================================================
    if 'comment_text' not in df.columns:

        st.error(
            "❌ Kolom 'comment_text' tidak ditemukan."
        )

        st.stop()

    # =====================================================
    # HAPUS DATA KOSONG
    # =====================================================
    df = df.dropna(subset=['comment_text'])

    # =====================================================
    # PREPROCESSING
    # =====================================================
    with st.spinner("Melakukan preprocessing..."):

        df['clean_text'] = df['comment_text'].apply(preprocessing)

    # =====================================================
    # LABEL OTOMATIS
    # =====================================================
    with st.spinner("Melakukan pelabelan otomatis..."):

        df['sentiment'] = df['clean_text'].apply(auto_labeling)

    st.success("✅ Pelabelan otomatis selesai")

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset",
        "⚙️ Preprocessing",
        "🤖 SVM"
    ])

    # =====================================================
    # TAB 1 DATASET
    # =====================================================
    with tab1:

        st.subheader("Preview Dataset")

        st.dataframe(
            df[
                [
                    'comment_text',
                    'sentiment'
                ]
            ].head(20),
            use_container_width=True
        )

        st.subheader("Jumlah Dataset")

        st.write(f"Total komentar: {len(df)}")

        # =================================================
        # VISUALISASI SENTIMEN
        # =================================================
        st.subheader("Sebaran Sentimen")

        sentiment_count = df['sentiment'].value_counts()

        fig1, ax1 = plt.subplots(figsize=(6, 6))

        ax1.pie(
            sentiment_count,
            labels=sentiment_count.index,
            autopct='%1.1f%%',
            startangle=90
        )

        ax1.axis('equal')

        st.pyplot(fig1)

        # =================================================
        # BAR CHART
        # =================================================
        st.subheader("Grafik Jumlah Sentimen")

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        sns.barplot(
            x=sentiment_count.index,
            y=sentiment_count.values,
            ax=ax2
        )

        ax2.set_xlabel("Kategori Sentimen")
        ax2.set_ylabel("Jumlah Komentar")

        st.pyplot(fig2)

    # =====================================================
    # TAB 2 PREPROCESSING
    # =====================================================
    with tab2:

        st.subheader("Hasil Preprocessing")

        st.dataframe(
            df[
                [
                    'comment_text',
                    'clean_text',
                    'sentiment'
                ]
            ].head(20),
            use_container_width=True
        )

    # =====================================================
    # TAB 3 SVM
    # =====================================================
    with tab3:

        st.subheader("Klasifikasi Sentimen Menggunakan SVM")

        # =================================================
        # FITUR DAN LABEL
        # =================================================
        X = df['clean_text']
        y = df['sentiment']

        # =================================================
        # TF-IDF
        # =================================================
        tfidf = TfidfVectorizer()

        X_tfidf = tfidf.fit_transform(X)

        # =================================================
        # SPLIT DATA
        # =================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=0.2,
            random_state=42
        )

        # =================================================
        # MODEL SVM
        # =================================================
        svm_model = SVC(kernel='linear')

        svm_model.fit(X_train, y_train)

        # =================================================
        # PREDIKSI
        # =================================================
        y_pred = svm_model.predict(X_test)

        # =================================================
        # AKURASI
        # =================================================
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"✅ Akurasi Model: {accuracy:.2f}")

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        # =================================================
        # CONFUSION MATRIX
        # =================================================
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig3, ax3 = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax3
        )

        ax3.set_xlabel("Prediksi")
        ax3.set_ylabel("Aktual")

        st.pyplot(fig3)

        # =================================================
        # INPUT PREDIKSI MANUAL
        # =================================================
        st.subheader("🔍 Prediksi Sentimen Komentar")

        user_input = st.text_area(
            "Masukkan komentar:"
        )

        if st.button("Prediksi Sentimen"):

            clean_input = preprocessing(user_input)

            vector_input = tfidf.transform([clean_input])

            prediction = svm_model.predict(vector_input)[0]

            st.success(
                f"Hasil Prediksi Sentimen: {prediction}"
            )