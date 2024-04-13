import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from preprocessing import Preprocessing
from model import BaseModel
from modified_model import ModifiedModel
from plotting import Plotting
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Main script
data_path = '/content/language.csv'

# Preprocessing
preprocessor = Preprocessing()
x, y, data = preprocessor.preprocess_data(data_path)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Model
base_model = BaseModel(input_size=x_train.shape[1], output_size=len(data['language_encoded'].unique()))
base_model.compile_model()
hist = base_model.train_model(x_train, y_train, epochs=10, batch_size=128, validation_split=0.3)

# Plotting
plotter = Plotting()
plotter.plot_learning_curve(hist)
plotter.plot_accuracy_curve(hist)

# Confusion Matrix
y_pred = base_model.model.predict(x_test)
y_pred = [i.argmax() for i in y_pred]
cm = confusion_matrix(y_test, y_pred)
plotter.plot_confusion_matrix(cm, preprocessor.le.inverse_transform([i for i in range(len(data['language_encoded'].unique()))]))

# Modified Model
modified_model = ModifiedModel(input_size=x_train.shape[1], output_size=len(data['language_encoded'].unique()))
modified_model.compile_model()
es = EarlyStopping(monitor='accuracy', patience=1)
hist = modified_model.train_model(x_train, y_train, epochs=8, batch_size=256, validation_split=0.3, callbacks=[es])

# Plotting Modified Model
plotter.plot_learning_curve(hist)
plotter.plot_accuracy_curve(hist)

# Confusion Matrix for Modified Model
y_pred = modified_model.model.predict(x_test)
y_pred = [i.argmax() for i in y_pred]
cm = confusion_matrix(y_test, y_pred)
plotter.plot_confusion_matrix(cm, preprocessor.le.inverse_transform([i for i in range(len(data['language_encoded'].unique()))]))

# Sample prediction
sample_text = "Hello world!" 
clean_text = preprocessor.clean_text(sample_text)
sample_vector = preprocessor.cv.transform([clean_text])
sample_vector = sample_vector.astype('uint8')

# Predict using the base model
base_model_prediction = base_model.model.predict(sample_vector)
base_model_prediction_language = preprocessor.le.inverse_transform([base_model_prediction.argmax()])[0]
print("Prediction using Base Model:", base_model_prediction_language)

# Predict using the modified model
modified_model_prediction = modified_model.model.predict(sample_vector)
modified_model_prediction_language = preprocessor.le.inverse_transform([modified_model_prediction.argmax()])[0]
print("Prediction using Modified Model:", modified_model_prediction_language)
