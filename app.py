import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import joblib
from bertopic import BERTopic
from huggingface_hub import hf_hub_download
import asyncio

# Fix asyncio issue
try:
    asyncio.get_event_loop()
except RuntimeError as e:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Sidebar Setup
st.sidebar.title("Aplikasi Klasifikasi Kendala 📧")
input_method = st.sidebar.radio("Pilih Metode Input", ["Upload File", "Input Manual"])
st.sidebar.write("Analisis Klasifikasi Email")
st.sidebar.write("Developed by Mesakh Besta 🚀")
st.title("Email Complaint Processing and Topic Classification")
st.write("Upload file Excel atau CSV atau ketik manual complaint untuk lihat distribusi topik beserta full teks Cleaned Complaint.")
st.markdown('<p style="color:red;">(Isi file Excel: Incident Number, Summary, Notes)</p>', unsafe_allow_html=True)

# Handle file input or manual input
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

# Process Data if DataFrame is loaded
if df is not None:
    def remove_dear_ojk(text):
        if isinstance(text, str):
            return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", text)
        return text
    
    # Remove 'Dear Bapak/Ibu Helpdesk OJK' from Notes
    df['Notes'] = df['Notes'].apply(remove_dear_ojk)

    def extract_complaint(text):
        if not isinstance(text, str):
            return "Bagian komplain tidak ditemukan."
        start_patterns = [r"PERHATIAN: E-mail ini berasal dari pihak di luar OJK.*?attachment.*?link.*?yang terdapat pada e-mail ini."]
        end_patterns = [r"(From\s*.*?From|Best regards|Salam|Atas perhatiannya|Regards|Best Regards|Mohon\s*untuk\s*melengkapi\s*data\s*.*tabel\s*dibawah).*", r"From:\s*Direktorat\s*Pelaporan\s*Data.*"]
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
        
        # Remove sensitive information
        sensitive_info_patterns = [r"Nama\s*Terdaftar\s*.*", r"Email\s*.*", r"No\.\s*Telp\s*.*", r"User\s*Id\s*/\s*User\s*Name\s*.*", r"No\.\s*KTP\s*.*", r"Nama\s*Perusahaan\s*.*", r"Nama\s*Pelapor\s*.*", r"No\.\s*Telp\s*Pelapor\s*.*", r"Internal", r"Dengan\s*hormat.*", r"Jenis\s*Usaha\s*.*", r"Keterangan\s*.*", r"No\.\s*SK\s*.*", r"Alamat\s*website/URL\s*.*", r"Selamat\s*(Pagi|Siang|Sore).*", r"Kepada\s*Yth\.\s*Bapak/Ibu.*", r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*", r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*", r"No\.\s*NPWP\s*Perusahaan\s*.*", r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*", r"Yth\s*.*", r"demikian\s*.*", r"Demikian\s*.*", r"Demikianlah\s*.*"]
        for pattern in sensitive_info_patterns:
            text = re.sub(pattern, "", text)
        return text if text else "Bagian komplain tidak ditemukan."
    
    df['Complaint'] = df.apply(lambda row: extract_complaint(row['Notes']), axis=1)

    # Additional Cleanup and Cut-off of general noise
    def cut_off_general(complaint):
        cut_off_keywords = ["PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo", "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan", "h._________", "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus"]
        for keyword in cut_off_keywords:
            if keyword in complaint:
                complaint = complaint.split(keyword)[0]
        return complaint
    
    df['Complaint'] = df['Complaint'].apply(cut_off_general)

    # Add Summary to Complaint
    df['Complaint'] = df['Complaint'] + " " + df['Summary']

    # Remove subject prefixes
    subject_pattern = r"(?i)Subject:\s*(Re:\s*FW:|RE:|FW:|PTAsuransiAllianzUtamaIndonesiaPT Asuransi Allianz Utama Indonesia)?\s*"
    df['Complaint'] = df['Complaint'].str.replace(subject_pattern, "", regex=True)

    # Remove stopwords
    stopword_factory = StopWordRemoverFactory()
    stopwords = stopword_factory.get_stop_words()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text
    
    df['Cleaned_Complaint'] = df['Complaint'].apply(clean_text)

    # Additional Text Replacements
    words_to_remove = r"\b(" \
    "terima|kasih|mohon|silakan|untuk|dan|atau|saya|kami|helpdesk|bapak|ibu|segera|harap|apakah|kapan|dapat|tidak|" \
    "dana|pensiun|sampaikan|konvensional|djakarta|delta|asuransi|ventura|modal|absensi|arahannya|" \
    "bapak|ibu|berikut|selalu|maksud|mrisikodapenbuncoid|sistem|mencoba|dibawah|lbbpr|kejadian|" \
    "arahan|lamp|berhasil|ringkasan|publikasi|sosialisasi|pelaporanid|sultra|penyampaian|" \
    "surat|yg|satyadhika|bakti|penamaan|menjumpai|progo|group|diisi|terh|login|file|gambar|screenshot|panduan|" \
    "perhutani|selfassessment|umum|status|keperluan|ulang|publik|" \
    "lembaga|nomor|petunjuk|dikirimkan|maksud|astra|gb|mesin|" \
    "terjadi|selfassessment|tanya|reliance|" \
    "unit|terdaftar|jl|ii|put|nama|muncul|dimaksud|kegiatan|waktu|desember|pkap|life|" \
    "mengenai|monitoring|ditinjau|dosbnb|pmo|wisma|apuppt|pergada|tombol|bpd|oss|insidentil|" \
    "psak|pelaksanaan|perkembangan|format|berdasarkan|luar|penguna|hpt|ppt|bambu|nasabah|team|marga|" \
    "cipayun|star|dipo|finance|menyampaikan|lapor|jasa|pengawasan|dokumen|asuransi|rencana|permohonan|" \
    "indonesia|allianz|keuangan|otoritas|no|penggunaan|antifraud|penerapan|strategi|fraud|anti|realisasi|saf|tersebut|" \
    "nya|data|terdapat|periode|melalui|perusahaan|sesuai|melakukan|hak|komplek|laporan|pelaporan|modul|apolo|sebut|" \
    "terap|email|pt|mohon|sampai|ikut|usaha|dapat|tahun|kini|lalu|kendala|ojk|laku|guna|aplikasi|atas|radius|prawiro|" \
    "jakarta pusat)\b"
    
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(words_to_remove, "", regex=True)

    # Final Clean up for extra spaces and trim
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\s+", " ", regex=True).str.strip()

    st.write("### Final Cleaned Complaint Data")
    st.dataframe(df[['Incident Number', 'Summary', 'Cleaned_Complaint']].head(4))

    @st.cache_data(show_spinner=False)
    def load_models():
        repo_id = 'mesakhbesta/NLP_OJK'
        model_utama = hf_hub_download(repo_id=repo_id, filename='bertopic_utama.joblib')
        model_utama = joblib.load(model_utama)
        model_sub_1 = hf_hub_download(repo_id=repo_id, filename='sub_topic_1.joblib')
        model_sub_1 = joblib.load(model_sub_1)
        model_sub_min1 = hf_hub_download(repo_id=repo_id, filename='sub_topic_min1.joblib')
        model_sub_min1 = joblib.load(model_sub_min1)
        return model_utama, model_sub_1, model_sub_min1

    # Load models
    topic_model, sub_topic_model_1, sub_topic_model_min1 = load_models()

    # Function to classify sub-topics
    def classify_sub_topic_min1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        if sub_topics[0] == 2:
            return 2
        else:
            return -1

    def classify_sub_topic_1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        if sub_topics[0] == 1:
            return 1
        else:
            return 0

    # Function to get main topic
    def get_main_topic(topic_model, data):
        topics, probs = topic_model.transform(data)
        return topics[0]
        
    # Generate predictions
    df['Main Topic'] = df['Cleaned_Complaint'].apply(lambda x: get_main_topic(topic_model, x))
    df['Sub Topic -1'] = df['Cleaned_Complaint'].apply(lambda x: classify_sub_topic_min1(sub_topic_model_min1, x))
    df['Sub Topic 1'] = df['Cleaned_Complaint'].apply(lambda x: classify_sub_topic_1(sub_topic_model_1, x))
    st.write(df[['Incident Number', 'Summary', 'Cleaned_Complaint', 'Main Topic', 'Sub Topic -1', 'Sub Topic 1']])

