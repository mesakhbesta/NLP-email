import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import joblib
from bertopic import BERTopic
from huggingface_hub import hf_hub_download
import asyncio
from umap import UMAP

# Set event loop
try:
    asyncio.get_event_loop()
except RuntimeError as e:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit UI Setup
st.sidebar.title("Aplikasi Klasifikasi Kendala 📧")
st.sidebar.write("Analisis Klasifikasi Email")
st.sidebar.write("Developed by Mesakh Besta 🚀")
st.title("Email Complaint Processing and Topic Classification")
st.write("Upload file Excel atau CSV untuk lihat distribusi topik beserta full teks Cleaned Complaint.")
st.markdown('<p style="color:red;">(Isi file Excel: Incident Number, Summary, Notes)</p>', unsafe_allow_html=True)

# Input method (file upload only)
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

# Proses jika dataframe tidak kosong
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
        cut_off_keywords = ["PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo", "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan", "h._________", "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus", "KamiBapak/Ibu untuk pencerahannya", "kami ucapkan h. ,", "-- , DANA PENSIUN BPD JAWA", "sDian PENYANGKALAN.", "------------------------Dari: Adrian", "hormat saya RidwanForwarded", "--h, DANA PENSIUN WIJAYA", "Mohon InfonyahKantor", "an arahannya dari Bapak/ Ibu", "ya untuk di check ya.Thank", "Kendala:Thank youAddelin", ",Sekretaris DAPENUrusan Humas & ProtokolTazkya", "Mohon arahannya.Berikut screenshot", "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024", "Annie Clara DesiantyComplianceIndonesia", "Dian Rosmawati RambeCompliance", "Beararti apakah,Tri WahyuniCompliance DeptPT.", "Dengan alamat email yang didaftarkan", "dan arahan", ",AJB Bumiputera", "’h sebelumnya Afriyanty", "PENYANGKALAN.", "h Dana Pensiun PKT", ", h , Tasya PT.", "Contoh: 10.00", "hAnnisa Adelya SerawaiPT Fazz", "sebagaimana gambar di bawah ini", "PT Asuransi Jiwa SeaInsure On Fri", "hJana MaesiatiBanking ReportFinance", "Tembusan", "Sebagai referensi", "hAdriansyah", "h atas bantuannya Dwi Anggina", "PT Asuransi Jiwa SeaInsure", "dengan notifikasi dibawah ini", "Terima ksh", ": DISCLAIMER", "Sebagai informasi", "nya. h.Kind s,Melati", ": DISCLAIMER", "Petugas AROPT", "h,Julianto", "h,Hernawati", "Dana Pensiun Syariah", ",Tria NoviatyStrategic"]
        for keyword in cut_off_keywords:
            if keyword in complaint:
                complaint = complaint.split(keyword)[0]
        return complaint
    
    df['Complaint'] = df['Complaint'].apply(cut_off_general)
    df['Complaint'] = df['Complaint'] + " " + df['Summary']
    
    # Remove subject prefix from complaints
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
    
    df['Cleaned_Complaint'] = df['Complaint'].apply(clean_text)

    words_to_remove = r"\b(" \
    "terima|kasih|mohon|silakan|untuk|dan|atau|saya|kami|helpdesk|bapak|ibu|segera|harap|apakah|kapan|dapat|tidak|" \
    "dana|pensiun|sampaikan|djakarta|delta|asuransi|ventura|modal|absensi|arahannya|" \
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
    "jakarta pusatlapor|client|jasa|web|pengawasan|dokumen|asuransi|rencana|permohonan|indonesia|allianz|keuangan|otoritas|no|penggunaan|antifraud|penerapan|strategi|fraud|anti|realisasi|saf|tersebut|nya|data|terdapat|periode|melalui|perusahaan|sesuai|melakukan|hak|komplek|laporan|pelaporan|modul|apolo|sebut|terap|email|pt|mohon|sampai|ikut|usaha|dapat|tahun|kini|lalu|kendala|ojk|laku|guna|aplikasi|atas|radius|prawiro|jakarta pusat)\b"
    
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(words_to_remove, "", regex=True)

    # Display cleaned complaints
    st.write("### Final Cleaned Complaint Data")
    st.dataframe(df[['Incident Number', 'Summary', 'Cleaned_Complaint']].head(4))

    # UMAP model for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        low_memory=False,
        random_state=1337
    )

    # Initialize BERTopic with UMAP model
    topic_model = BERTopic(
        language="indonesian",
        umap_model=umap_model,
        calculate_probabilities=True,
    )
    topics, probabilities = topic_model.fit_transform(df['Cleaned_Complaint'])

    # Display the topics
    topic_info = pd.DataFrame(topic_model.get_topic_info())
    st.write("### Topics found in the data")
    st.write(topic_info)

    st.write("### Complete Data with Identified Topics")
    df['Topic'] = topics
    st.dataframe(df[['Incident Number', 'Summary', 'Cleaned_Complaint', 'Topic']])
