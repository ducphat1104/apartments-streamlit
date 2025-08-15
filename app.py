# app.py
import streamlit as st
import pandas as pd
from joblib import load
import json
import datetime as dt

st.set_page_config(page_title="Dự đoán giá thuê căn hộ", layout="wide")

# =========================
# Nạp mô hình & metadata
# =========================
@st.cache_resource
def load_assets():
    rf = load("rf_pipeline.joblib")
    xgb = load("xgb_pipeline.joblib")
    with open("X_columns.json", "r", encoding="utf-8") as f:
        X_cols = json.load(f)
    return rf, xgb, X_cols

try:
    rf_model, xgb_model, X_COLUMNS = load_assets()
except Exception as e:
    st.error(f"Không nạp được mô hình/metadata: {e}")
    st.stop()

# =========================
# Danh sách feature tiện ích & thú cưng (key phải trùng lúc train)
# =========================
amen_cols = [
    "amenities_Parking","amenities_Pool","amenities_Gym","amenities_Patio/Deck",
    "amenities_Washer Dryer","amenities_Storage","amenities_Clubhouse",
    "amenities_Dishwasher","amenities_AC","amenities_Refrigerator","amenities_Fireplace",
    "amenities_Cable or Satellite","amenities_Playground","amenities_Internet Access",
    "amenities_Wood Floors","amenities_Gated","amenities_Tennis","amenities_TV",
    "amenities_Elevator","amenities_Basketball","amenities_Hot Tub",
    "amenities_Garbage Disposal","amenities_View","amenities_Alarm",
    "amenities_Doorman","amenities_Luxury",
]
pet_cols = ["pets_allowed_Cats","pets_allowed_Dogs","pets_allowed_None"]

# Label tiếng Việt cho tiện ích
AMEN_VN = {
    "amenities_Parking": "Bãi đỗ xe",
    "amenities_Pool": "Hồ bơi",
    "amenities_Gym": "Phòng gym",
    "amenities_Patio/Deck": "Sân/Hiên",
    "amenities_Washer Dryer": "Máy giặt/sấy",
    "amenities_Storage": "Kho chứa/Storage",
    "amenities_Clubhouse": "Nhà CLB/Clubhouse",
    "amenities_Dishwasher": "Máy rửa bát",
    "amenities_AC": "Điều hoà",
    "amenities_Refrigerator": "Tủ lạnh",
    "amenities_Fireplace": "Lò sưởi",
    "amenities_Cable or Satellite": "Truyền hình cáp/vệ tinh",
    "amenities_Playground": "Sân chơi",
    "amenities_Internet Access": "Internet",
    "amenities_Wood Floors": "Sàn gỗ",
    "amenities_Gated": "Khu có cổng",
    "amenities_Tennis": "Sân tennis",
    "amenities_TV": "TV",
    "amenities_Elevator": "Thang máy",
    "amenities_Basketball": "Sân bóng rổ",
    "amenities_Hot Tub": "Bồn nước nóng",
    "amenities_Garbage Disposal": "Nghiền rác",
    "amenities_View": "View đẹp",
    "amenities_Alarm": "Báo động",
    "amenities_Doorman": "Bảo vệ cửa/Doorman",
    "amenities_Luxury": "Cao cấp/Luxury",
}
PET_VN = {
    "pets_allowed_Cats": "Cho nuôi mèo",
    "pets_allowed_Dogs": "Cho nuôi chó",
    "pets_allowed_None": "Không rõ thông tin thú cưng",
}

# =========================
# Sidebar
# =========================
st.sidebar.title("Tuỳ chọn")
model_name = st.sidebar.selectbox("Chọn mô hình", ["XGBoost (đã tinh chỉnh)", "Random Forest (đã tinh chỉnh)"])
model = xgb_model if "XGBoost" in model_name else rf_model

st.sidebar.write("📦 Tài sản đã nạp:")
st.sidebar.code("rf_pipeline.joblib\nxgb_pipeline.joblib\nX_columns.json")

# =========================
# Tiêu đề chính
# =========================
st.title("Dự đoán giá thuê căn hộ (USD / tháng)")
st.caption("Nhập thông tin căn hộ, mô hình sẽ ước lượng giá thuê theo tháng.")

# =========================
# Khu nhập liệu
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    bathrooms = st.number_input("Số phòng tắm", min_value=1.0, max_value=3.0, value=2.0, step=0.5, format="%.1f")
    bedrooms  = st.number_input("Số phòng ngủ",  min_value=1.0, max_value=4.0, value=2.0, step=1.0, format="%.0f")
    square_feet = st.number_input("Diện tích (ft²)", min_value=200, max_value=3000, value=900, step=25)

with col2:
    latitude  = st.number_input("Vĩ độ",  min_value=26.0, max_value=48.0, value=37.2258, step=0.0001, format="%.4f")
    longitude = st.number_input("Kinh độ", min_value=-123.0, max_value=-70.0, value=-84.5623, step=0.0001, format="%.4f")
    today = dt.date(2019, 9, 18)  # khớp phân bố ngày trong tập train
    d = st.date_input("Ngày đăng tin", value=today)

with col3:
    fee = st.selectbox("Phí dịch vụ", ["No", "Yes"])
    has_photo = st.selectbox("Tình trạng ảnh", ["Yes", "Thumbnail", "No"])
    price_type = st.selectbox("Đơn vị giá", ["Monthly", "Weekly", "Monthly|Weekly"])
    cityname = st.text_input("Thành phố", value="Dallas")
    state = st.text_input("Bang (2 ký tự)", value="TX")
    source = st.text_input("Nguồn dữ liệu", value="RentDigs.com")

st.markdown("---")
st.subheader("Tiện ích & Thú cưng")

cols = st.columns(4)
checks = {}
for i, key in enumerate(amen_cols):
    label = AMEN_VN.get(key, key.replace("amenities_", "Tiện ích: "))
    checks[key] = cols[i % 4].checkbox(label, value=False)

pet1, pet2, pet3 = st.columns(3)
checks["pets_allowed_Cats"] = pet1.checkbox(PET_VN["pets_allowed_Cats"], value=False)
checks["pets_allowed_Dogs"] = pet2.checkbox(PET_VN["pets_allowed_Dogs"], value=False)
checks["pets_allowed_None"] = pet3.checkbox(PET_VN["pets_allowed_None"], value=True)

st.markdown("---")

# =========================
# Build hàng input theo đúng schema X_COLUMNS
# =========================
def build_input_row():
    row = {}
    # default: numeric = 0, object = "Unknown"
    obj_cols = {"fee","has_photo","price_type","cityname","state","source"}
    for c in X_COLUMNS:
        row[c] = "Unknown" if c in obj_cols else 0

    # gán giá trị thực từ UI
    row["bathrooms"] = float(bathrooms)
    row["bedrooms"] = float(bedrooms)
    row["square_feet"] = float(square_feet)
    row["latitude"] = float(latitude)
    row["longitude"] = float(longitude)
    row["fee"] = fee
    row["has_photo"] = has_photo
    row["price_type"] = price_type
    row["cityname"] = cityname
    row["state"] = state
    row["source"] = source

    # tách ngày thành year/month/day (đã dùng lúc train)
    row["year"] = d.year
    row["month"] = d.month
    row["day"] = d.day

    # tiện ích & thú cưng (0/1)
    for c in amen_cols + pet_cols:
        if c in X_COLUMNS:
            row[c] = 1 if checks.get(c, False) else 0

    # đảm bảo đúng thứ tự cột
    return pd.DataFrame([row], columns=X_COLUMNS)

# =========================
# Dự đoán
# =========================
if st.button("Dự đoán giá thuê"):
    X_one = build_input_row()
    try:
        pred = model.predict(X_one)[0]
        st.success(f"📈 Giá thuê dự đoán: **${pred:,.0f} / tháng**  ({model_name})")
        with st.expander("Xem dữ liệu đầu vào (đã chuẩn hoá schema)"):
            st.dataframe(X_one)
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")

