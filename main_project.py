import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from collections import Counter
import re
import pickle
from scipy.sparse import csr_matrix
import io

# 設定網頁標題和圖表字型
st.set_page_config(page_title="評論分類與文字雲", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 載入停用詞清單
@st.cache_data
def load_stopwords(path='stopwords.txt'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        st.error(f"錯誤: '{path}' 檔案未找到。請確保此檔案存在於應用程式同一個目錄。")
        return set()

# 載入預先訓練好的模型和 TF-IDF 特徵器
@st.cache_data
def load_model_and_vectorizer():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('mnb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        st.error("錯誤: 'tfidf_vectorizer.pkl' 或 'mnb_model.pkl' 檔案未找到。請先執行訓練腳本以生成模型檔案。")
        return None, None

# 預處理和斷詞函數
def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r'[0-9a-zA-Z\W_]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def jieba_cut(text, stopwords):
    text = str(text).strip()
    if not text:
        return ""
    words = jieba.cut(text)
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

# 主應用程式介面
st.title("自動評論分類與文字雲分析")

# 使用者上傳檔案
uploaded_file = st.file_uploader("請上傳您的 Excel 評論資料 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("檔案上傳成功！")
        
        if "評論內容" not in df.columns:
            st.error("上傳的檔案必須包含 '評論內容' 欄位。")
            st.stop()
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤：{e}")
        st.stop()

    st.subheader("1. 原始評論資料預覽")
    st.dataframe(df.head())

    st.subheader("2. 預測主題與斷詞結果")
    with st.spinner('正在進行評論分類與斷詞...'):
        stopwords = load_stopwords()
        tfidf, model = load_model_and_vectorizer()

        if tfidf and model:
            df["評論內容"] = df["評論內容"].astype(str)
            df["斷詞"] = df["評論內容"].apply(lambda x: jieba_cut(x, stopwords))
            
            # 確保有斷詞資料後再進行預測
            if not df["斷詞"].str.strip().eq("").all():
                X_predict = tfidf.transform(df["斷詞"])
                df["預測主題"] = model.predict(X_predict)
                st.dataframe(df[["評論內容", "預測主題"]])
            else:
                st.warning("斷詞後沒有有效的文字數據，無法進行預測。")
                st.stop()
            
            st.markdown("---")
            st.markdown("### 下載預測結果")
            
            csv_data = df.to_csv(index=False).encode('utf-8-sig') # 使用 'utf-8-sig' 避免中文亂碼
            excel_data = io.BytesIO()
            df.to_excel(excel_data, index=False)
            excel_data.seek(0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="下載為 CSV",
                    data=csv_data,
                    file_name="評論分類結果.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    label="下載為 Excel",
                    data=excel_data,
                    file_name="評論分類結果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        
    st.subheader("3. 資料視覺化分析")
    
    # 總體 TF-IDF 文字雲
    st.markdown("#### 總體評論關鍵字文字雲")
    with st.spinner('正在生成文字雲...'):
        word_scores = X_predict.sum(axis=0).A1
        words = tfidf.get_feature_names_out()
        tfidf_dict = dict(zip(words, word_scores))
        
        # 確保字典不為空，如果為空則顯示警告
        if tfidf_dict and any(tfidf_dict.values()):
            wordcloud = WordCloud(
                font_path="msjh.ttc",
                background_color="white",
                width=1000,
                height=600,
                max_words=200,
                min_font_size=1,
                margin=2,
                prefer_horizontal=0.9
            )
            wordcloud.generate_from_frequencies(tfidf_dict)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("沒有足夠的文字數據來生成文字雲。")
            
    # 預測主題數量長條圖 (針對長條圖重疊進行優化)
    st.markdown("#### 預測主題數量分佈")
    if '預測主題' in df.columns and not df['預測主題'].empty:
        # 取得標籤數量，並根據數量決定圖表大小
        num_labels = len(df['預測主題'].unique())
        fig_width = max(10, num_labels * 1.5)  # 每個標籤留1.5的寬度
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        df['預測主題'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
        
        ax.set_title("各預測主題評論數量", fontsize=16)
        ax.set_xlabel("預測主題", fontsize=12)
        ax.set_ylabel("評論數量", fontsize=12)
        
        # 根據標籤數量決定旋轉角度
        if num_labels > 5:
            ax.tick_params(axis='x', rotation=45, ha='right')
        else:
            ax.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("沒有可用的預測主題資料來生成長條圖。")

else:
    st.info("請上傳您的 Excel 檔案以開始分析。")

st.markdown("---")
st.write("此應用程式需依賴 `mnb_model.pkl` 和 `tfidf_vectorizer.pkl` 模型檔案，請先運行訓練腳本以生成這些檔案。")





