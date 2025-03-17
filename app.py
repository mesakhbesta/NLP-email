import os
os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"  # Nonaktifkan file watcher Streamlit

import asyncio
import streamlit as st
import pandas as pd
import re
import joblib
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import asyncio


# Sidebar setup
st.sidebar.title("Aplikasi Klasifikasi Kendala 📧")
input_method = st.sidebar.radio("Pilih Metode Input", ["Upload File", "Input Manual"])
st.sidebar.write("Analisis Klasifikasi Email")
st.sidebar.write("Developed by Mesakh Besta 🚀")

# Judul aplikasi dan instruksi
st.title("Email Complaint Processing and Topic Classification")
st.write("Upload file Excel atau CSV atau ketik manual complaint untuk lihat distribusi topik beserta full teks Cleaned Complaint.")
st.markdown('<p style="color:red;">(Isi file Excel: Incident Number, Summary, Notes)</p>', unsafe_allow_html=True)

# Input data: file upload atau input manual
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload your Excel atau CSV file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success("File berhasil diupload!")
    else:
        st.info("Silakan upload file Excel atau CSV untuk memulai proses.")
        df = None
else:
    manual_input = st.text_area("Ketik complaint di sini", "")
    if manual_input:
        df = pd.DataFrame([{'Incident Number': 'Manual', 'Summary': '', 'Notes': manual_input}])
        st.success("Complaint berhasil dimasukkan!")
    else:
        st.info("Silakan ketik complaint di text box untuk memulai proses.")
        df = None

if df is not None:
    # Fungsi untuk menghapus "Dear Bapak/Ibu Helpdesk OJK"
    def remove_dear_ojk(text):
        if isinstance(text, str):
            return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", text)
        return text

    df['Notes'] = df['Notes'].apply(remove_dear_ojk)

    # Fungsi untuk mengekstrak bagian complaint
    def extract_complaint(text):
        if not isinstance(text, str):
            return "Bagian komplain tidak ditemukan."
        start_patterns = [
            r"PERHATIAN: E-mail ini berasal dari pihak di luar OJK.*?attachment.*?link.*?yang terdapat pada e-mail ini."
        ]
        end_patterns = [
            r"(From\s*.*?From|Best regards|Salam|Atas perhatiannya|Regards|Best Regards|Mohon\s*untuk\s*melengkapi\s*data\s*.*tabel\s*dibawah).*",
            r"From:\s*Direktorat\s*Pelaporan\s*Data.*"
        ]
        start_match = None
        for pattern in start_patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                start_match = matches[-1].end()
        if start_match:
            text = text[start_match:].strip()
        for pattern in end_patterns:
            end_match = re.search(pattern, text, re.DOTALL)
            if end_match:
                text = text[:end_match.start()].strip()
        sensitive_info_patterns = [
            r"Nama\s*Terdaftar\s*.*", r"Email\s*.*", r"No\.\s*Telp\s*.*", r"User\s*Id\s*/\s*User\s*Name\s*.*",
            r"No\.\s*KTP\s*.*", r"Nama\s*Perusahaan\s*.*", r"Nama\s*Pelapor\s*.*", r"No\.\s*Telp\s*Pelapor\s*.*",
            r"Internal", r"Dengan\s*hormat.*", r"Jenis\s*Usaha\s*.*", r"Keterangan\s*.*", r"No\.\s*SK\s*.*",
            r"Alamat\s*website/URL\s*.*", r"Selamat\s*(Pagi|Siang|Sore).*", r"Kepada\s*Yth\.\s*Bapak/Ibu.*",
            r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*", r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*",
            r"No\.\s*NPWP\s*Perusahaan\s*.*", r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*", r"Yth\s*.*",
            r"demikian\s*.*", r"Demikian\s*.*", r"Demikianlah\s*.*"
        ]
        for pattern in sensitive_info_patterns:
            text = re.sub(pattern, "", text)
        return text if text else "Bagian komplain tidak ditemukan."

    df['Complaint'] = df.apply(lambda row: extract_complaint(row['Notes']), axis=1)

    # Memotong teks complaint berdasarkan kata kunci tertentu
    def cut_off_general(complaint):
        cut_off_keywords = [
            "PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo",
            "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan",
            "h._________", "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus",
            "KamiBapak/Ibu untuk pencerahannya", "kami ucapkan h. ,", "-- , DANA PENSIUN BPD JAWA", "sDian PENYANGKALAN.",
            "------------------------Dari: Adrian", "hormat saya RidwanForwarded", "--h, DANA PENSIUN WIJAYA", "Mohon InfonyahKantor",
            "an arahannya dari Bapak/ Ibu", "ya untuk di check ya.Thank", "Kendala:Thank youAddelin", ",Sekretaris DAPENUrusan Humas & ProtokolTazkya",
            "Mohon arahannya.Berikut screenshot", "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024", "Annie Clara DesiantyComplianceIndonesia",
            "Dian Rosmawati RambeCompliance", "Beararti apakah,Tri WahyuniCompliance DeptPT.", "Dengan alamat email yang didaftarkan",
            "dan arahan", ",AJB Bumiputera", "’h sebelumnya Afriyanty", "PENYANGKALAN.", "h Dana Pensiun PKT", ", h , Tasya PT.",
            "Contoh: 10.00", "hAnnisa Adelya SerawaiPT Fazz", "sebagaimana gambar di bawah ini", "PT Asuransi Jiwa SeaInsure On Fri",
            "hJana MaesiatiBanking ReportFinance", "Tembusan", "Sebagai referensi", "hAdriansyah", "h atas bantuannya Dwi Anggina",
            "PT Asuransi Jiwa SeaInsure", "dengan notifikasi dibawah ini", "Terima ksh", ": DISCLAIMER", "Sebagai informasi",
            "nya. h.Kind s,Melati", ": DISCLAIMER", "Petugas AROPT", "h,Julianto", "h,Hernawati", "Dana Pensiun Syariah",
            ",Tria NoviatyStrategic"
        ]
        for keyword in cut_off_keywords:
            if keyword in complaint:
                complaint = complaint.split(keyword)[0]
        return complaint

    df['Complaint'] = df['Complaint'].apply(cut_off_general)
    # Gabungkan Complaint dan Summary (pastikan tidak ada nilai NaN)
    df['Complaint'] = df['Complaint'].fillna('') + " " + df['Summary'].fillna('')
    subject_pattern = r"(?i)Subject:\s*(Re:\s*FW:|RE:|FW:|PTAsuransiAllianzUtamaIndonesiaPT Asuransi Allianz Utama Indonesia)?\s*"
    df['Complaint'] = df['Complaint'].str.replace(subject_pattern, "", regex=True)
    
    # Pembersihan teks menggunakan stopword dari Sastrawi
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stopword_factory = StopWordRemoverFactory()
    stopwords = stopword_factory.get_stop_words()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    # Bersihkan teks complaint awal
    df['Cleaned_Complaint'] = df['Complaint'].apply(clean_text)

    # Menghapus kata-kata yang tidak penting
    words_to_remove = r"\b(terima|kasih|mohon|silakan|untuk|dan|atau|saya|kami|helpdesk|bapak|ibu|segera|harap|apakah|kapan|dapat|tidak|dan)\b"
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(words_to_remove, "", regex=True)

    # Melatih model BERTopic baru menggunakan data yang telah dibersihkan
    def train_topic_model(documents):
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(embedding_model=sentence_model)
        topics, probs = topic_model.fit_transform(documents)
        return topic_model, topics, probs

    with st.spinner("Melatih model BERTopic baru..."):
        topic_model, topics, probs = train_topic_model(df['Cleaned_Complaint'].tolist())
        df['Topic'] = topics

    topic_counts = df['Topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    st.write("#### Distribusi Topik Final:")
    st.dataframe(topic_counts)

    unique_topics = sorted(df['Topic'].unique().tolist())
    selected_topic = st.selectbox("Pilih Topik untuk ditampilkan", ["Semua Topik"] + unique_topics)
    if selected_topic != "Semua Topik":
        df_filtered = df[df['Topic'] == selected_topic]
    else:
        df_filtered = df

    st.write("#### Full Text of Cleaned Complaint per Incident:")
    for idx, row in df_filtered.iterrows():
        st.write("=" * 135)
        st.write(f"**Incident Number**: {row['Incident Number']}")
        st.write(f"**Cleaned Complaint**: {row['Cleaned_Complaint']}")
        st.write("=" * 135)
