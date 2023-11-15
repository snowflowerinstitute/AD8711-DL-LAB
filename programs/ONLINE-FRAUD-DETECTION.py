import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('../datasets/6-FRAUD-DETECTION.csv')
type_encoder = LabelEncoder()
df['type'] = type_encoder.fit_transform(df['type'])
X = df.drop('isFraud', axis=1)
y = df['isFraud']

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=420)
X_train_resampled, y_train_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled,
                                                    test_size=0.2, random_state=420)
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
_ = model.fit(X_train, y_train, epochs=10, batch_size=int(1e5), verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=int(1e5), verbose=0)
print(f'Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')
