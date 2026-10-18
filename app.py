#python -m streamlit run app.py
#python -m pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
st.set_page_config(
    page_title="Student Learning Clustering",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Learning Clustering")
st.write("เว็บไซต์วิเคราะห์และจัดกลุ่มรูปแบบการเรียนของนักศึกษา")

# 1. อ่านข้อมูล
st.header("1. Data Understanding")

uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์ Excel",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name="Raw_Data")
else:
    try:
        df = pd.read_excel(
            "CL01_student_learning_clustering.xlsx",
            sheet_name="Raw_Data"
        )
    except FileNotFoundError:
        st.warning("กรุณาวางไฟล์ Excel ไว้ในโฟลเดอร์เดียวกับ app.py")
        st.stop()

st.write("ตัวอย่างข้อมูล")
st.dataframe(df.head(10), width="stretch")

col1, col2 = st.columns(2)
col1.metric("จำนวนแถว", len(df))
col2.metric("จำนวนคอลัมน์", len(df.columns))

# 2. ตรวจสอบคุณภาพข้อมูล
st.header("2. Data Quality")

quality = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Unique Values": df.nunique()
})

st.dataframe(quality, width="stretch")

st.write("จำนวนแถวซ้ำ:", int(df.duplicated().sum()))
st.write("จำนวน Missing Values ทั้งหมด:", int(df.isna().sum().sum()))
st.caption("การตรวจ Outlier ใช้เกณฑ์ IQR เพื่อตรวจหาค่าที่อาจผิดปกติ โดยไม่ลบค่าเหล่านั้นอัตโนมัติ เพราะค่าที่สูงหรือต่ำอาจเป็นข้อมูลจริงได้")

# 3. Data Cleaning
st.header("3. Data Cleaning")

data = df.drop_duplicates().copy()

numeric_columns = data.select_dtypes(
    include=np.number
).columns.tolist()
# ตรวจสอบ Outlier ด้วยวิธี IQR
outlier_results = []

for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (data[col] < lower) |
        (data[col] > upper)
    ).sum()

    outlier_results.append({
        "Column": col,
        "Outlier Count": int(count)
    })

st.subheader("ผลการตรวจสอบ Outlier")
st.dataframe(
    pd.DataFrame(outlier_results),
    width="stretch"
)

st.write("จำนวนแถวหลังลบข้อมูลซ้ำ:", len(data))
st.caption("เหตุผลในการทำความสะอาด: ลบแถวที่ซ้ำกันเพื่อไม่ให้นับข้อมูลเดิมซ้ำ และเติม Missing ของฟีเจอร์ตัวเลขที่เลือกด้วยค่ามัธยฐาน (Median) เพื่อลดผลกระทบจากค่าสุดโต่งเมื่อเทียบกับค่าเฉลี่ย")

# 4. Feature Selection
st.header("4. Feature Selection")

# ไม่ใช้ student_id เป็น Feature
features = [
    c for c in numeric_columns
    if c.lower() not in ["student_id", "id"]
]

selected_features = st.multiselect(
    "เลือกตัวแปรสำหรับจัดกลุ่ม",
    features,
    default=features
)

if len(selected_features) < 2:
    st.info("กรุณาเลือก Features อย่างน้อย 2 ตัว")
    st.stop()


# เติม Missing ด้วย Median
X_df = data[selected_features].copy()

for col in selected_features:
    X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df[col] = X_df[col].fillna(X_df[col].median())

# ตรวจสอบ Missing Values หลังเติม Median
if X_df.isna().any().any():
    st.error(
        "ยังมี Missing Values หลังเติม Median "
        "กรุณาตรวจสอบข้อมูลหรือเลือก Features อื่น"
    )
    st.stop()
    
# 5. EDA
st.header("5. Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs([
    "Histogram",
    "Correlation Heatmap",
    "Boxplot"
])

with tab1:
    feature = st.selectbox(
        "เลือก Feature",
        selected_features
    )
    fig, ax = plt.subplots()
    sns.histplot(X_df[feature], kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        X_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=X_df, ax=ax)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
    plt.close(fig)

# 6. Feature Scaling
st.header("6. Feature Scaling")

scaler = StandardScaler()
X = scaler.fit_transform(X_df)

st.write("จำนวนข้อมูลทั้งหมด:", len(X_df))
st.success("ทำ Feature Scaling เรียบร้อยแล้ว")

st.caption(
    "ประเมิน Clustering จากข้อมูลทั้งหมด "
    "โดยทำ Feature Scaling ก่อนสร้างโมเดล"
)

# 7. Elbow Method และเลือกจำนวน Cluster
st.header("7. Elbow Method")
st.write("ใช้ค่า Inertia ของ K-Means เปรียบเทียบจำนวนกลุ่มหลายค่า เพื่อช่วยพิจารณาจุดที่การลด Inertia เริ่มช้าลง")
max_k = min(8, len(X_df) - 1)
elbow_k_values = list(range(1, max_k + 1))
inertias = []
for candidate_k in elbow_k_values:
    elbow_model = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
    elbow_model.fit(X)
    inertias.append(elbow_model.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(elbow_k_values, inertias, marker="o")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
ax.set_xticks(elbow_k_values)
st.pyplot(fig)
plt.close(fig)
st.caption("เลือก K จากบริเวณที่กราฟเริ่มแบนลงร่วมกับ Silhouette Score และความหมายของแต่ละกลุ่ม ไม่ควรใช้กราฟ Elbow เพียงอย่างเดียว")

# 8. เลือกจำนวน Cluster
st.header("เลือกจำนวน Cluster และเลือกโมเดล")

k = st.slider(
    "จำนวนกลุ่ม (K)",
    min_value=2,
    max_value=min(8, len(X_df) - 1),
    value=min(3, len(X_df) - 1)
)

model_name = st.radio(
    "เลือกโมเดลเพื่อดูผล",
    ["K-Means", "Hierarchical Clustering"],
    horizontal=True
)

# 9. Model Evaluation
st.header("เปรียบเทียบคะแนนของทั้งสองโมเดล")

# จัดกลุ่มข้อมูลชุดเดียวกันด้วยทั้งสองโมเดล
kmeans_all = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
kmeans_all_labels = kmeans_all.fit_predict(X)


hierarchical_all = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)
hierarchical_all_labels = hierarchical_all.fit_predict(X)

# เลือก Label สำหรับแสดงผล
if model_name == "K-Means":
    labels = kmeans_all_labels
else:
    labels = hierarchical_all_labels

result = data.copy()
result["cluster"] = labels

evaluation = []

for name, model_labels in [
    ("K-Means", kmeans_all_labels),
    ("Hierarchical", hierarchical_all_labels)
]:
    evaluation.append({
        "Model": name,
        "Silhouette Score": silhouette_score(X, model_labels),
        "Davies-Bouldin Index": davies_bouldin_score(X, model_labels),
        "Calinski-Harabasz Index": calinski_harabasz_score(X, model_labels)
    })

evaluation_df = pd.DataFrame(evaluation)
st.dataframe(evaluation_df, width="stretch")

st.caption(
    "ทั้งสองโมเดลจัดกลุ่มข้อมูลชุดเดียวกัน "
    "เพื่อเปรียบเทียบคุณภาพการแบ่งกลุ่มโดยใช้ Metrics ชุดเดียวกัน "
    "Silhouette และ Calinski-Harabasz ค่าสูงมักดีกว่า "
    "ส่วน Davies-Bouldin ค่าต่ำมักดีกว่า"
)


# 11. Visualization และ Cluster Profiling
st.header("แสดงกราฟ PCA")

pca = PCA(n_components=2)
coordinates = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": coordinates[:, 0],
    "PC2": coordinates[:, 1],
    "Cluster": labels.astype(str)
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    ax=ax
)
ax.set_title(f"{model_name} Clusters")
st.pyplot(fig)
plt.close(fig)

st.header("แสดงค่าเฉลี่ยและจำนวนสมาชิกแต่ละกลุ่ม")

profile = result.groupby("cluster")[selected_features].mean().round(2)

st.write("ค่าเฉลี่ยของ Features ในแต่ละกลุ่ม")
st.dataframe(profile, width="stretch")

st.write("จำนวนสมาชิกแต่ละกลุ่ม")
st.dataframe(
    result["cluster"].value_counts().sort_index().rename(
        "จำนวนสมาชิก"
    )
)

st.download_button(
    "ดาวน์โหลดผลการจัดกลุ่ม CSV",
    data=result.to_csv(index=False).encode("utf-8-sig"),
    file_name="student_clustering_result.csv",
    mime="text/csv"
)

st.subheader("สรุปโมเดลที่มี Silhouette Score สูงกว่า")
valid_scores = evaluation_df.dropna(subset=["Silhouette Score"])
if not valid_scores.empty:
    best_row = valid_scores.loc[valid_scores["Silhouette Score"].idxmax()]
    st.write(
        f"เมื่อพิจารณา Silhouette Score บนข้อมูลทั้งหมด โมเดลที่ได้คะแนนสูงสุดในการทดลองนี้คือ **{best_row['Model']}** "
        f"(Silhouette = {best_row['Silhouette Score']:.4f}) ทั้งนี้ควรพิจารณา Davies-Bouldin Index, "
        "Calinski-Harabasz Index, กราฟ และความหมายของ Cluster Profiling ร่วมด้วย"
    )
else:
    st.warning(
        "ไม่สามารถคำนวณคะแนนประเมินได้ "
        "กรุณาตรวจสอบจำนวนกลุ่มและข้อมูลที่เลือก"
    )
st.success("วิเคราะห์ข้อมูลเสร็จแล้ว")


# ==============================
# แสดงผลลัพธ์ใน Terminal ของ VS Code
# ==============================

print("\n===== STUDENT LEARNING CLUSTERING =====")

print("\n--- Data Preview ---")
print(data.head())

print("\n--- Data Information ---")
print("จำนวนแถว:", len(data))
print("จำนวนคอลัมน์:", len(data.columns))

print("\n--- Data Cleaning Result ---")
print("Missing Values หลังจัดการ:",
      X_df.isnull().sum().sum())

print("\n--- Selected Features ---")
print(selected_features)

print("\n--- Cluster Result ---")
print(result[["cluster"]].value_counts().sort_index())

print("\n--- Cluster Profile ---")
print(result.groupby("cluster")[selected_features].mean())

print("\n===== END OF RESULT =====")
