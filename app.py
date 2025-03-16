import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import joblib
from bertopic import BERTopic
from huggingface_hub import hf_hub_download
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

st.sidebar.title("Aplikasi Klasifikasi Kendala 📧")

input_method = st.sidebar.radio("Pilih Metode Input", ["Upload File", "Input Manual"])
st.sidebar.write("Analisis Klasifikasi Email")
st.sidebar.write("Developed by Mesakh Besta 🚀")
st.title("Email Complaint Processing and Topic Classification")
st.write("Upload file Excel atau CSV atau ketik manual complaint untuk lihat distribusi topik beserta full teks Cleaned Complaint.")
st.markdown('<p style="color:red;">(Isi file Excel: Incident Number, Summary, Notes)</p>', unsafe_allow_html=True)

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
    def remove_dear_ojk(text):
        if isinstance(text, str):
            return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", text)
        return text
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
        sensitive_info_patterns = [r"Nama\s*Terdaftar\s*.*", r"Email\s*.*", r"No\.\s*Telp\s*.*", r"User\s*Id\s*/\s*User\s*Name\s*.*", r"No\.\s*KTP\s*.*", r"Nama\s*Perusahaan\s*.*", r"Nama\s*Pelapor\s*.*", r"No\.\s*Telp\s*Pelapor\s*.*", r"Internal", r"Dengan\s*hormat.*", r"Jenis\s*Usaha\s*.*", r"Keterangan\s*.*", r"No\.\s*SK\s*.*", r"Alamat\s*website/URL\s*.*", r"Selamat\s*(Pagi|Siang|Sore).*", r"Kepada\s*Yth\.\s*Bapak/Ibu.*", r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*", r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*", r"No\.\s*NPWP\s*Perusahaan\s*.*", r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*", r"Yth\s*.*", r"demikian\s*.*", r"Demikian\s*.*", r"Demikianlah\s*.*"]
        for pattern in sensitive_info_patterns:
            text = re.sub(pattern, "", text)
        return text if text else "Bagian komplain tidak ditemukan."
    df['Complaint'] = df.apply(lambda row: extract_complaint(row['Notes']), axis=1)

    def cut_off_general(complaint):
        cut_off_keywords = [
            "PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo", "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan",
            "h._________", "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus",
            "KamiBapak/Ibu untuk pencerahannya", "kami ucapkan h. ,", "-- , DANA PENSIUN BPD JAWA", "sDian PENYANGKALAN.", "------------------------Dari: Adrian", "hormat saya RidwanForwarded",
            "--h, DANA PENSIUN WIJAYA", "Mohon InfonyahKantor", "an arahannya dari Bapak/ Ibu", "ya untuk di check ya.Thank", "Kendala:Thank youAddelin", ",Sekretaris DAPENUrusan Humas & ProtokolTazkya", "Mohon arahannya.Berikut screenshot",
            "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024", "Annie Clara DesiantyComplianceIndonesia", "Dian Rosmawati RambeCompliance", "Beararti apakah,Tri WahyuniCompliance DeptPT."
        ]
        for keyword in cut_off_keywords:
            if keyword in complaint:
                complaint = complaint.split(keyword)[0]
        return complaint
    df['Complaint'] = df['Complaint'].apply(cut_off_general)
    df['Complaint'] = df['Complaint'] + " " + df['Summary']
    subject_pattern = r"(?i)Subject:\s*(Re:\s*FW:|RE:|FW:|PTAsuransiAllianzUtamaIndonesiaPT Asuransi Allianz Utama Indonesia)?\s*"
    df['Complaint'] = df['Complaint'].str.replace(subject_pattern, "", regex=True)

    stopword_factory = StopWordRemoverFactory()
    stopwords = stopword_factory.get_stop_words()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    df['Cleaned_Complaint'] = df['Complaint'].apply(lambda x: clean_text(x))

    # Make sure this part is optimized as it is dealing with regex replacements
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bmasuk\b", "login", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\blog-in\b", "login", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\brenbis\b", "rencana bisnis", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bapkap\b", "ap kap", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\baro\b", "administrator responsible officer", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bro\b", "responsible officer", regex=True)

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

    topic_model, sub_topic_model_1, sub_topic_model_min1 = load_models()

    def classify_sub_topic_min1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        if sub_topics[0] == 2:
            return 2
        else:
            return -1

    def classify_sub_topic_1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        return sub_topics[0]

    def get_final_topic(complaint):
        topics, probs = topic_model.transform([complaint])
        main_topic = topics[0]
        if main_topic == -1:
            sub_topic = classify_sub_topic_min1(sub_topic_model_min1, [complaint])
            if sub_topic == 2:
                return 'Kendala Install Aplikasi Client'
            else:
                return 'Outlier'
        elif main_topic == 1:
            sub_topic = classify_sub_topic_1(sub_topic_model_1, [complaint])
            if sub_topic == 0:
                return 'Kendala Akses Aplikasi/Authorized Error'
            elif sub_topic == 1:
                return 'Lupa Password/Reset Password'
            elif sub_topic == 2:
                return 'Kendala Username Salah'
        elif main_topic == 0:
            return 'Permintaan Akses ARO/RO'
        elif main_topic == 2:
            return 'Kesalahan File/ Validasi'
        elif main_topic == 3:
            return 'Perpanjangan Waktu Pengumpulan'
        elif main_topic == 4:
            return 'Gagal Upload Pelaporan'
        elif main_topic == 5:
            return 'Kendala pelaporan APOLO'
        return 'Unknown'

    @st.cache_data
    def get_topik_pengaduan(row):
        return get_final_topic(row)

    df['Topik_Kendala'] = df['Cleaned_Complaint'].apply(get_topik_pengaduan)
    
    # Display table or filtering
    st.subheader("Topik Klasifikasi Keluhan")
    filter_topic = st.selectbox("Pilih Topik", df['Topik_Kendala'].unique())
    filtered_df = df[df['Topik_Kendala'] == filter_topic]
    st.write(filtered_df)

else:
    st.warning("Tidak ada data untuk diproses.")
