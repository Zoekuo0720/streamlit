import streamlit as st
import pandas as pd
import numpy as np
import pickle
import jieba
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib as mpl
from matplotlib import font_manager
import os
from scipy.sparse import csr_matrix
import io
from docx import Document

# --- 程式碼重構與優化 ---

def preprocess_text(text, stopwords):
    text = str(text).strip()
    if not text:
        return ""
    words = jieba.cut(text)
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

@st.cache_resource
def load_resources():
    """載入所有必要的模型、停用詞與字體，並進行 Matplotlib 設定。"""
    
    # --- 檢查與載入字體 ---
    font_file_name = None
    for filename in ['NotoSansTC-Regular.ttf', 'NotoSansCJKtc-Regular.otf']:
        if os.path.exists(filename):
            font_file_name = filename
            break

    if not font_file_name:
        st.sidebar.error(f"❌ 警告：找不到字體檔案。請將 '{filename}' 上傳至專案根目錄。")
        st.stop()
    
    FONT_PATH = os.path.join(os.path.dirname(__file__), font_file_name)
    
    try:
        # 清除舊的 Matplotlib 字體快取檔案，這是解決字體問題的關鍵步驟。
        cache_dir = mpl.get_cachedir()
        if os.path.isdir(cache_dir):
            for font_cache_file in os.listdir(cache_dir):
                if font_cache_file.startswith('fontlist-'):
                    os.remove(os.path.join(cache_dir, font_cache_file))
        
        # 載入新字體並設定 Matplotlib 參數。
        font_manager.fontManager.addfont(FONT_PATH)
        font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        mpl.rcParams['font.sans-serif'] = [font_name, 'Arial Unicode MS', 'Microsoft JhengHei']
        mpl.rcParams['axes.unicode_minus'] = False
        st.sidebar.success("✅ 成功載入專案字體。")
        
    except Exception as e:
        st.sidebar.error(f"❌ 載入字體時發生錯誤：{e}")
        st.stop()

    # --- 載入模型與停用詞 ---
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('mnb_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])

        return vectorizer, model, stopwords
    
    except FileNotFoundError as e:
        st.error(f"❌ 錯誤：找不到必要的檔案。請確保 'tfidf_vectorizer.pkl', 'mnb_model.pkl', 和 'stopwords.txt' 都在相同目錄下。")
        st.stop()
    except Exception as e:
        st.error(f"載入資源時發生錯誤：{e}")
        st.stop()

def get_docx_text(file):
    document = Document(file)
    full_text = []
    for para in document.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

vectorizer, model, stopwords = load_resources()

class_labels = model.classes_

if 'classified_df_for_display' not in st.session_state:
    st.session_state.classified_df_for_display = None
if 'has_ground_truth' not in st.session_state:
    st.session_state.has_ground_truth = False
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = None

st.set_page_config(layout="wide", page_title="評論分析儀表板")
st.title("員工評論智能分析儀表板")
st.markdown("本儀表板運用先進的機器學習模型，旨在智能分析員工評論，自動識別潛在問題主題，並以直觀的視覺化方式呈現關鍵洞察，助力企業優化管理決策。")
st.markdown("---")
st.header("1. 評論即時分類")
st.markdown("您可以選擇輸入單條評論進行即時分類，或上傳批量文件進行自動分析。")

tab1, tab2 = st.tabs(["單條評論分析", "批量文件分析"])

with tab1:
    st.subheader("單條評論快速分析")
    user_input = st.text_area("請輸入您想分析的評論內容：", "報到流程很混亂，文件準備不齊全，窗口也不明確。", height=100)
    
    if st.button("分析評論", key="single_analysis_button"):
        if user_input:
            with st.spinner("正在分析評論，請稍候..."):
                processed_input = preprocess_text(user_input, stopwords)
                input_vector = vectorizer.transform([processed_input])
                
                prediction = model.predict(input_vector)[0]
                st.success(f"**預測的評論主題是：** `{prediction}`")
        else:
            st.warning("請輸入評論內容。")

with tab2:
    st.subheader("批量文件自動分類")
    st.markdown("請上傳包含評論的 Excel 檔案 (.xlsx) 或 Word 檔案 (.docx)。")
    uploaded_file = st.file_uploader("選擇檔案", type=["xlsx", "docx"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        df_uploaded = None
        comment_column = None
        
        try:
            if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df_uploaded = pd.read_excel(uploaded_file)
                st.info("Excel 檔案上傳成功！請確認下方數據預覽並選擇評論欄位。")
                st.dataframe(df_uploaded.head())
                column_options = df_uploaded.columns.tolist()
                
                default_index = column_options.index('評論內容') if '評論內容' in column_options else 0
                comment_column = st.selectbox("請選擇包含評論內容的欄位：", column_options, index=default_index, key="excel_column_select")
            
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                st.info("Word 檔案上傳成功！系統將提取文件所有文本內容進行分析。")
                doc_text = get_docx_text(uploaded_file)
                if doc_text:
                    df_uploaded = pd.DataFrame({'評論內容': [doc_text]})
                    comment_column = '評論內容'
                    st.text_area("Word 文件內容預覽 (前500字):", doc_text[:500] + "..." if len(doc_text) > 500 else doc_text, height=150)
                else:
                    st.warning("上傳的 Word 文件中沒有可讀取的文本內容。")
                    df_uploaded = None
            else:
                st.error("不支援的檔案類型。請上傳 .xlsx 或 .docx 檔案。")
                df_uploaded = None

            if df_uploaded is not None and comment_column is not None:
                if st.button("開始批量分類", key="batch_analysis_button"):
                    with st.spinner("正在分析評論，請稍候..."):
                        predictions = []
                        st.session_state.classified_df_for_display = None
                        st.session_state.has_ground_truth = False
                        st.session_state.edited_df = None
                        df_uploaded['processed_review'] = df_uploaded[comment_column].astype(str).apply(lambda x: preprocess_text(x, stopwords))
                        for index, row in df_uploaded.iterrows():
                            processed_comment = row['processed_review']
                            if processed_comment:
                                input_vector = vectorizer.transform([processed_comment])
                                prediction = model.predict(input_vector)[0]
                            else:
                                prediction = "無法分類 (內容空缺)"
                            predictions.append(prediction)
                        
                        df_results = pd.DataFrame({
                            '原始評論內容': df_uploaded[comment_column].astype(str).tolist(),
                            '預測負評主題': predictions
                        })
                        
                        if '分類標籤' in df_uploaded.columns:
                            df_results['真實主題'] = df_uploaded['分類標籤'].astype(str).tolist()
                            st.session_state.has_ground_truth = True
                        else:
                            st.session_state.has_ground_truth = False

                        st.session_state.classified_df_for_display = df_results
                        st.success("批量分類完成！請查看下方結果並可選擇下載。")
                        
                        @st.cache_data
                        def convert_df_to_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False, sheet_name='Classified Reviews')
                            processed_data = output.getvalue()
                            return processed_data

                        excel_data = convert_df_to_excel(df_results)
                        st.download_button(
                            label="下載分類結果 (Excel)",
                            data=excel_data,
                            file_name="classified_reviews.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            elif df_uploaded is None:
                pass
            else:
                st.warning("請選擇包含評論內容的欄位，或確保 Word 文件有內容。")

        except Exception as e:
            st.error(f"讀取或處理檔案時發生錯誤：{e}")
            st.info("請確認您上傳的是有效的 Excel 或 Word 檔案，且包含預期的評論欄位。")

st.markdown("---")
if st.session_state.classified_df_for_display is not None and st.session_state.has_ground_truth:
    st.header("2. 模型效能評估報告")
    st.markdown("以下報告和圖表是根據您上傳的 Excel 數據，比較模型預測結果與真實標籤所生成。")
    y_true = st.session_state.classified_df_for_display['真實主題'].astype(str)
    y_pred = st.session_state.classified_df_for_display['預測負評主題'].astype(str)
    
    with st.expander("點擊查看分類報告 (Classification Report)"):
        st.subheader("分類報告")
        st.markdown("此報告提供了模型在您的數據集上對各類別的精準率、召回率和 F1-分數。")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))

    with st.expander("點擊查看混淆矩陣 (Confusion Matrix)"):
        st.subheader("混淆矩陣")
        st.markdown("混淆矩陣直觀展示了模型在您的數據集上，各類別的分類正確與錯誤情況。")
        fig, ax = plt.subplots(figsize=(6, 5))
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
        ax.set_xlabel('預測類別 (Predicted Label)', fontweight='bold')
        ax.set_ylabel('真實類別 (True Label)', fontweight='bold')
        ax.set_title('混淆矩陣', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

elif st.session_state.classified_df_for_display is not None and not st.session_state.has_ground_truth:
    st.warning("您的檔案不包含 **'分類標籤'** 欄位。分類報告和混淆矩陣需要此欄位來評估模型的準確性。您可以自行為評論加上標籤後再上傳，以啟用這些報告。")
    
st.markdown("---")
st.header("3. 評論關鍵詞與主題洞察")
st.markdown("此文字雲是基於當前數據集中的評論生成，幫助您快速掌握各主題的關鍵詞語。")

if st.session_state.classified_df_for_display is not None:
    source_df_wc = st.session_state.classified_df_for_display
    category_column_wc = '預測負評主題'
    # 這裡確保有 processed_review 欄位
    if 'processed_review' not in source_df_wc.columns:
        source_df_wc['processed_review'] = source_df_wc['原始評論內容'].astype(str).apply(lambda x: preprocess_text(x, stopwords))
    st.info("當前文字雲顯示的是您**上傳檔案並分類後**的評論關鍵詞。")
else:
    source_df_wc = pd.DataFrame(columns=['評論內容', '分類標籤'])
    category_column_wc = '分類標籤'
    st.info("請先上傳檔案進行分析，以生成文字雲。")

selected_category_options = source_df_wc[category_column_wc].unique().tolist()
if selected_category_options:
    selected_category = st.selectbox(
        "請選擇您想查看文字雲的評論主題：",
        options=selected_category_options
    )
    
    # 確保選中的類別有數據
    category_reviews_processed = source_df_wc[source_df_wc[category_column_wc] == selected_category]['processed_review']
    text_for_wordcloud = " ".join(category_reviews_processed.dropna())

    if text_for_wordcloud:
        # 增加 min_font_size 參數，確保即使頻率低也能顯示
        wordcloud = WordCloud(
            font_path=FONT_PATH,
            width=500,
            height=250,
            background_color='white',
            collocations=False,
            max_words=25,
            min_font_size=1
        ).generate(text_for_wordcloud)

        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.write("此類別暫無足夠評論生成文字雲。")
else:
    st.write("沒有可供選擇的評論主題。請先上傳檔案並進行分析。")

st.markdown("---")
st.header("4. 評論主題分佈概覽")
st.markdown("此直條圖展示了評論數據中各主題的數量分佈。")

if st.session_state.classified_df_for_display is not None:
    st.info("當前圖表顯示的是您**上傳檔案並分類後**的評論主題分佈。")
    source_df_dist = st.session_state.classified_df_for_display
    category_column_dist = '預測負評主題'
else:
    source_df_dist = pd.DataFrame(columns=['預測負評主題'])
    category_column_dist = '預測負評主題'
    st.info("請先上傳檔案進行分析，以生成主題分佈圖。")

if not source_df_dist.empty:
    category_counts = source_df_dist[category_column_dist].value_counts().sort_values(ascending=False)
    
    # 檢查是否所有類別都是空的，以避免繪圖錯誤
    if not category_counts.empty:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax_dist, palette='Blues_d')
        ax_dist.set_title('各評論主題數量分佈', fontweight='bold')
        ax_dist.set_xlabel('評論主題', fontweight='bold')
        ax_dist.set_ylabel('評論數', fontweight='bold')
        
        # 根據主題數量動態調整標籤旋轉
        if len(category_counts) > 5:
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks(rotation=0)
            
        plt.tight_layout()
        st.pyplot(fig_dist)
    else:
        st.write("沒有可供繪製的數據。")
else:
    st.write("沒有可供繪製的數據。")

st.markdown("---")
st.write("© 分類互動模型")


