import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_FILES = {
    "KNN": "k-nearest_neighbors_pipeline.joblib",
    "LogisticRegression": "logistic_regression_pipeline.joblib",
    "隨機森林": "randomforest_classifier_pipeline.joblib",
    "XgBoost": "xgboost_classifier_pipeline.joblib",
}

@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def main() -> None:
    st.set_page_config(page_title="信用卡違約模型預測", layout="wide")
    st.sidebar.title("模型選擇")
    model_name = st.sidebar.selectbox(
        "請選擇一個模型",
        list(MODEL_FILES.keys()),
        index=0,
    )

    st.title("信用卡違約預測 App")
    st.markdown("本應用使用 UCI_Credit_Card 資料集，並依選擇的模型進行隨機抽測預測。")

    data_path = Path(__file__).parent / "UCI_Credit_Card.csv"
    df = load_data(data_path)

    st.subheader("資料前 10 筆")
    st.dataframe(df.head(10))

    st.subheader("特徵與目標變數設定")
    if "ID" not in df.columns or "default.payment.next.month" not in df.columns:
        st.error("資料集缺少必要欄位：ID 或 default.payment.next.month")
        return

    X = df.drop(columns=["ID", "default.payment.next.month"])
    y = df["default.payment.next.month"]

    st.write("- X 欄位數量：", X.shape[1])
    st.write("- y 欄位名稱： default.payment.next.month")

    st.subheader("y 分類數量")
    counts = y.value_counts().sort_index()
    counts_df = counts.rename_axis("class").reset_index(name="count")
    st.table(counts_df)
    st.bar_chart(counts)

    st.subheader("隨機抽測並進行預測")
    if st.button("開始預測"):
        sample = df.sample(n=1, random_state=None)
        sample_X = sample.drop(columns=["ID", "default.payment.next.month"])

        model_path = Path(__file__).parent / MODEL_FILES[model_name]
        try:
            model = load_model(model_path)
        except Exception as ex:
            st.error(f"模型讀取失敗：{ex}")
            return

        try:
            prediction = model.predict(sample_X)
        except Exception as ex:
            st.error(f"模型預測失敗：{ex}")
            return

        predicted_label = int(prediction[0])
        label_text = "違約" if predicted_label == 1 else "未違約"

        st.markdown("### 抽測資料")
        st.dataframe(sample)

        st.markdown("### 預測結果")
        st.write(f"選擇模型：**{model_name}**")
        st.write(f"預測類別：**{predicted_label}** ({label_text})")

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(sample_X)[0]
                st.write("置信度：")
                st.write({f"類別 {i}": float(p) for i, p in enumerate(proba)})
            except Exception:
                pass


if __name__ == "__main__":
    main()
