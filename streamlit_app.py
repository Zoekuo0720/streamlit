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

# --- 專案專用中文字體設定，確保跨平台部署的相容性 ---
# 警告：您必須將 'NotoSansCJKtc-Regular.otf' 檔案上傳到與此 Python 檔案相同的 GitHub 目錄下。
FONT_FILE = 'NotoSansCJKtc-Regular.otf'
FONT_PATH = os.path.join(os.path.dirname(__file__), FONT_FILE)

if os.path.exists(FONT_PATH):
    # 載入字體
    font_manager.fontManager.addfont(FONT_PATH)
    font_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
    mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['axes.unicode_minus'] = False 
    st.sidebar.success("✅ 成功載入專案字體。")
else:
    st.sidebar.error(f"❌ 警告：找不到字體檔案 '{FONT_FILE}'。請務必將此檔案上傳至您的 GitHub 專案根目錄，與 `streamlit_app.py` 檔案並列。")

# --- 載入模型和評論資料 ---
@st.cache_resource
def load_resources():
    """
    載入預訓練的 TF-IDF Vectorizer、分類模型，以及原始評論資料。
    """
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('mnb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        df_reviews = pd.DataFrame() # 初始化為空，只處理上傳的檔案
        
        def load_stopwords_internal(path='stopwords.txt'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return set([line.strip() for line in f if line.strip()])
            except FileNotFoundError:
                st.error("錯誤：找不到 'stopwords.txt' 檔案。請確保它在您的專案目錄中。")
                return set()

        stopwords = load_stopwords_internal()

        def preprocess_text(text, stopwords):
            text = str(text).strip()
            if not text:
                return ""
            words = jieba.cut(text)
            return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

        return vectorizer, model, df_reviews, stopwords
    except FileNotFoundError as e:
        st.error(f"❌ 錯誤：找不到必要的模型檔案。請確保 'tfidf_vectorizer.pkl', 'mnb_model.pkl', 和 'stopwords.txt' 都在應用程式的相同目錄下。詳細錯誤：{e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"載入資源時發生錯誤：{e}")
        return None, None, None, None

vectorizer, model, df_reviews, stopwords = load_resources()

# 檢查資源是否成功載入
if vectorizer is None or model is None:
    st.stop()

class_labels = model.classes_

def preprocess_text_for_prediction(text):
    text = str(text).strip()
    if not text:
        return ""
    words = jieba.cut(text)
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

def get_docx_text(file):
    document = Document(file)
    full_text = []
    for para in document.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

# 初始化 session state
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
                processed_input = preprocess_text_for_prediction(user_input)
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
                        
                        # 處理評論並進行預測
                        df_uploaded['processed_review'] = df_uploaded[comment_column].apply(lambda x: preprocess_text_for_prediction(x))

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
                        
                        # 檢查檔案中是否有分類標籤，如果存在則新增到結果DataFrame
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
    if 'processed_review' not in source_df_wc.columns:
        source_df_wc['processed_review'] = source_df_wc['原始評論內容'].apply(lambda x: preprocess_text_for_prediction(x))
    st.info("當前文字雲顯示的是您**上傳檔案並分類後**的評論關鍵詞。")
else:
    # 這裡可以載入預設資料，但為了簡化部署，我們假設沒有預設檔案，只處理上傳
    source_df_wc = pd.DataFrame(columns=['評論內容', '分類標籤'])
    category_column_wc = '分類標籤'
    st.info("請先上傳檔案進行分析，以生成文字雲。")


selected_category_options = source_df_wc[category_column_wc].unique().tolist()
if selected_category_options:
    selected_category = st.selectbox(
        "請選擇您想查看文字雲的評論主題：",
        options=selected_category_options
    )

    if selected_category:
        st.subheader(f"{selected_category} 文字雲")
        category_reviews_processed = source_df_wc[source_df_wc[category_column_wc] == selected_category]['processed_review']
        text_for_wordcloud = " ".join(category_reviews_processed.dropna())

        if text_for_wordcloud and os.path.exists(FONT_PATH):
            wordcloud = WordCloud(
                font_path=FONT_PATH,
                width=500,
                height=250,
                background_color='white',
                collocations=False,
                max_words=25
            ).generate(text_for_wordcloud)

            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.write("此類別暫無足夠評論生成文字雲，或找不到適合的中文字體。")
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
    # 這裡也假設沒有預設檔案
    source_df_dist = pd.DataFrame(columns=['預測負評主題'])
    category_column_dist = '預測負評主題'
    st.info("請先上傳檔案進行分析，以生成主題分佈圖。")

if not source_df_dist.empty:
    category_counts = source_df_dist[category_column_dist].value_counts().sort_values(ascending=False)

    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax_dist, palette='Blues_d')
    ax_dist.set_title('各評論主題數量分佈', fontweight='bold')
    ax_dist.set_xlabel('評論主題', fontweight='bold')
    ax_dist.set_ylabel('評論數', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_dist)
else:
    st.write("沒有可供繪製的數據。")

st.markdown("---")
st.write("© 分類互動模型")



