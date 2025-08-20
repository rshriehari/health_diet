site-https://healthdietprediction.streamlit.app/
# README: Health and Fitness Prediction App

## Overview
The **Health and Fitness Prediction App** is a Streamlit-based application designed to predict daily calorie requirements and potential health risks based on user inputs such as age, gender, dietary habits, and activity levels. The app leverages machine learning models for regression (calorie prediction) and classification (disease risk).

## Features
- **User Login**: Simple login functionality to access the app.
- **Health Metrics Input**: Users can input personal health metrics like age, height, weight, dietary preferences, and activity level.
- **Calorie Prediction**: Predicts daily calorie requirements using a regression model.
- **Disease Risk Prediction**: Identifies potential health risks using a classification model.
- **Reports and Visualizations**: 
  - Tabular display of input features and predictions.
  - Bar chart and line chart visualizations for insights into calorie predictions.
  - Annotated visual feedback for user understanding.
- **Customizable Background**: Dynamic background image for an improved user experience.

## Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io)
- **Backend**: Python
- **Machine Learning**:
  - Regression Model: XGBoost Regressor
  - Classification Model: XGBoost Classifier
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy

## Installation and Setup
1. **Prerequisites**:
   - Python 3.8+
   - Required Python packages (see below)

2. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Models**:
   - Place the pre-trained `xgb_regressor_model.pkl` and `xgb_classifier_model.pkl` files in the `models` directory.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the App**:
   Open `http://localhost:8501` in your browser.

## Usage
1. **Login**: Enter your credentials to access the app.
2. **Input Details**: Fill out the form with your personal health metrics.
3. **Prediction**:
   - Submit the form to get predicted calorie requirements and disease risks.
   - View detailed reports in tabular and graphical formats.
4. **Analysis**: Review insights into factors influencing your health and fitness.

## File Structure
```plaintext
├── app.py                  # Main application file
├── models/
│   ├── xgb_regressor_model.pkl  # Calorie prediction model
│   ├── xgb_classifier_model.pkl # Disease classification model
├── image.webp              # Background image
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Dependencies
```plaintext
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

## Future Enhancements
- Add more diseases and metrics for prediction.
- Enable user-specific history and trend analysis.
- Integrate additional machine learning models for personalized recommendations.

## Contributing
Feel free to contribute by submitting issues or pull requests to improve the app. Ensure to follow the coding standards and test changes thoroughly.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the open-source community for providing tools and libraries used in this project.

---

Enjoy exploring your health and fitness with the **Health and Fitness Prediction App**!
