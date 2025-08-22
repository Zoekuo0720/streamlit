# 1.載入套件與基本設定
# ==============================================================================
# 導入所需的函式庫，這些函式庫用於數據處理、文字斷詞、機器學習、視覺化等
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay
import re
import pickle 
from scipy.sparse import csr_matrix

# 由於 wordcloud 在中文顯示上可能需要指定字型，這裡指定一個常見的字型。
# 如果您使用的作業系統沒有此字型 (例如 macOS 或 Linux)，請修改為系統上可用的中文字型路徑。
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 2.載入資料與預處理
# ==============================================================================
# 載入停用詞清單的函數
# 腳本需要一個名為 'stopwords.txt' 的檔案，其中包含每行一個停用詞
def load_stopwords(path='stopwords.txt'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        print(f"錯誤: '{path}' 檔案未找到。請確保此檔案存在於腳本同一個目錄。")
        return set()
stopwords = load_stopwords()

# 載入評論資料 Excel 檔案
# 腳本需要一個名為 '評論資料.xlsx' 的檔案
try:
    df = pd.read_excel("評論資料.xlsx")
    print("--- 成功載入 '評論資料.xlsx' ---")
except FileNotFoundError:
    print("錯誤: '評論資料.xlsx' 檔案未找到。請確保此檔案存在於腳本同一個目錄。")
    exit()

# **重要修正：處理資料中的空值和型別問題**
# 1. 移除評論內容或分類標籤中的標準空值 (NaN)
df.dropna(subset=["評論內容", "分類標籤"], inplace=True)
# 2. 移除可能以字串形式存在的 'nan' 值
df = df[df['分類標籤'].astype(str) != 'nan']
# 3. 重新設定索引
df.reset_index(drop=True, inplace=True)
print(f"--- 數據清洗完成，移除空值後剩餘 {len(df)} 筆資料 ---")

# **新增修正：強制將「評論內容」和「分類標籤」轉換為字串**
# 這能確保 TF-IDF 和 SMOTE 處理時不會因為資料型別不一致而產生錯誤
df["評論內容"] = df["評論內容"].astype(str)
df["分類標籤"] = df["分類標籤"].astype(str)

# 定義去除雜訊的函數，這裡用來清理文本中的非中文和多餘空格
def preprocess_text(text):
    text = str(text).strip()
    # 移除數字、英文字母和大部分特殊符號
    text = re.sub(r'[0-9a-zA-Z\W_]+', '', text)
    # 將多個空格替換為單個空格，並移除頭尾空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 自訂斷詞函數（含停用詞過濾）
def jieba_cut(text):
    text = str(text).strip()
    if not text:
        return ""
    # 使用 jieba 進行斷詞
    words = jieba.cut(text)
    # 過濾掉停用詞和空白字元
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

# 對「評論內容」欄位進行斷詞並建立新欄位
df["斷詞"] = df["評論內容"].apply(jieba_cut)

# 顯示前幾筆資料，檢查斷詞結果
print("--- 前 5 筆評論內容及斷詞結果 ---")
print(df[["評論內容", "斷詞"]].head())

# 3.特徵工程 (TF-IDF N-gram)
# ==============================================================================
if 'df' not in locals() or df.empty:
    print("請先成功載入資料。")
    exit()

# 初始化 TF-IDF 特徵器，設定 ngram_range=(1, 3) 來捕捉單詞和詞組
tfidf = TfidfVectorizer(ngram_range=(1, 3))

# 使用斷詞後的文本來建構 TF-IDF 特徵矩陣
X = tfidf.fit_transform(df["斷詞"])
y = df["分類標籤"]

# 檢查 TF-IDF 特徵器提取的詞彙
print("\n--- TF-IDF 提取的前 50 個特徵詞（含 N-gram）---")
print(tfidf.get_feature_names_out()[:50])
print(f"TF-IDF 特徵總數：{len(tfidf.get_feature_names_out())}")

# 4.數據切分與不均衡處理 (SMOTE)
# ==============================================================================
if 'X' not in locals() or 'y' not in locals():
    print("請先執行特徵工程步驟。")
    exit()

# 切分訓練與測試資料（80% 用於訓練，20% 用於測試）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 顯示訓練集原始類別分佈
print("\n--- 訓練集原始類別分佈 (SMOTE 之前) ---")
print(Counter(y_train))

# 使用 SMOTE (Synthetic Minority Over-sampling Technique) 進行過採樣
# 目的在於平衡類別分佈，避免模型偏向多數類
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 顯示 SMOTE 後的訓練集類別分佈
print("\n--- SMOTE 後訓練集類別分佈 (SMOTE 之後) ---")
print(Counter(y_train_smote))

# 5.建立與訓練分類模型
# ==============================================================================
if 'X_train_smote' not in locals() or 'y_train_smote' not in locals():
    print("請先執行數據切分與不均衡處理步驟。")
    exit()

# 建立 Naive Bayes 分類模型，特別適用於文字分類任務
model = MultinomialNB()

# 使用 SMOTE 過採樣後的數據來訓練模型
model.fit(X_train_smote, y_train_smote)
print("\n模型訓練完成！")

# 6.模型預測與評估
# ==============================================================================
if 'model' not in locals() or 'X_test' not in locals():
    print("請先執行模型訓練步驟。")
    exit()

# 使用測試集數據進行預測
y_pred = model.predict(X_test)

# 顯示模型的分類準確度
print("\n--- 模型分類準確度 ---")
print("分類準確度：", accuracy_score(y_test, y_pred))

# 顯示詳細的分類報告，包含每個類別的精準度、召回率和 F1-score
print("\n--- 模型分類報告 ---")
print(classification_report(y_test, y_pred))

# 7.結果視覺化
# ==============================================================================
if 'model' not in locals() or 'df' not in locals() or 'tfidf' not in locals():
    print("請先執行所有前置步驟。")
    exit()

# 1. 各分類標籤數量長條圖 (顯示原始數據分佈)
plt.figure(figsize=(10, 6))
df['分類標籤'].value_counts().plot(kind='bar', color='skyblue')
plt.title('各分類標籤原始評論數量', fontsize=16)
plt.xlabel("分類標籤", fontsize=12)
plt.ylabel("評論數量", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout() 
plt.show()

# 2. 混淆矩陣圖
# 視覺化模型預測結果，可看出哪些類別容易被混淆
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, xticks_rotation='vertical', cmap=plt.cm.Blues)
plt.title("分類預測混淆矩陣", fontsize=16)
plt.xlabel("預測標籤", fontsize=12)
plt.ylabel("真實標籤", fontsize=12)
plt.tight_layout()
plt.show()

# 3. 總體 TF-IDF 文字雲
# 根據 TF-IDF 權重生成文字雲，視覺化最重要的關鍵字
word_scores = X.sum(axis=0).A1
words = tfidf.get_feature_names_out()
tfidf_dict = dict(zip(words, word_scores))
font_path = "msjh.ttc" # 如果您在 Mac 或 Linux 上，請改為 'simhei.ttf' 或其他中文字型

wordcloud = WordCloud(
    font_path=font_path,
    background_color="white",
    width=1000,
    height=600,
    max_words=200,
    margin=2,
    prefer_horizontal=0.9
)
wordcloud.generate_from_frequencies(tfidf_dict)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("總體評論關鍵字 TF-IDF 文字雲", fontsize=18)
plt.show()

# 8.模型與數據持久化
# ==============================================================================
if 'model' not in locals() or 'tfidf' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
    print("請先執行所有前置步驟。")
    exit()

# 保存 TF-IDF 特徵器，以便之後對新數據進行同樣的轉換
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("\nTF-IDF 特徵器已保存為 tfidf_vectorizer.pkl")

# 保存訓練好的模型，以便之後直接載入並使用，無需重新訓練
with open('mnb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("分類模型已保存為 mnb_model.pkl")

# 保存測試集的特徵矩陣 (X_test) 和真實標籤 (y_test)，方便後續測試或評估
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
print("測試集特徵矩陣 X_test 已保存為 X_test.pkl")

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
print("測試集真實標籤 y_test 已保存為 y_test.pkl")




