# ===== PHẦN 1: Import thư viện =====
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ===== PHẦN 2: Tạo dataset =====
data = {
    "Hours": [1, 2, 3, 4, 5, 6],
    "Score": [2, 4, 5, 6, 7, 9]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ===== PHẦN 3: Tách Input và Output =====
X = df[["Hours"]]   # input (feature)
y = df["Score"]     # output (label)

# ===== PHẦN 4: Tạo model =====
model = LinearRegression()

# ===== PHẦN 5: Train model =====
model.fit(X, y)

print("\nHệ số góc (a):", model.coef_[0])
print("Hệ số chặn (b):", model.intercept_)

# ===== PHẦN 6: Dự đoán dữ liệu mới =====
hours = [[7]]   # học 7 giờ
predicted_score = model.predict(hours)

print("\nDự đoán điểm khi học 7 giờ:", predicted_score[0])

# ===== PHẦN 7: Dự đoán nhiều giá trị =====
new_data = [[4.5], [6.5], [8]]
predictions = model.predict(new_data)

print("\nDự đoán nhiều giá trị:")
for i, val in enumerate(new_data):
    print(f"Học {val[0]} giờ -> Điểm: {predictions[i]}")

# ===== PHẦN 8: Vẽ biểu đồ =====
plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.title("Hours vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")

plt.show()

# ===== PHẦN 9: Đánh giá model =====
y_pred = model.predict(X)
score = r2_score(y, y_pred)

print("\nĐộ chính xác R2:", score)