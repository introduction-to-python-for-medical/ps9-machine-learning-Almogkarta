import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

parkinsons_df = pd.read_csv('parkinsons.csv')
parkinsons_df = parkinsons_df.dropna()  # Remove rows with missing values

selected_features = ['PPE', 'DFA']
target = 'status'
x = parkinsons_df[selected_features]
y = parkinsons_df[target]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


svc = SVC()
svc.fit(x_train, y_train)


y_pred = svc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


joblib.dump(svc, 'svc_model.joblib')
