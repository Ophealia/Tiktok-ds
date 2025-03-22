from pyexpat import features
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

def load_processed_data(filepath):
    
    return pd.read_csv(filepath)


def prepare_features(df):
    
    features = [
        "video_category_num", "hour", "dayofweek", 
        "total_views", "like_ratio_x", "user_category_like_ratio",
        "total_watched", "like_ratio_y", "share_ratio",
    ]

    X = df[features]
    y = df['like_type']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, features

def train_and_evaluate_model(X, y, features):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    importances = model.feature_importances_
    feature_names = features 
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(feature_importance_df.sort_values(by='Importance', ascending=False))


if __name__ == "__main__":

    df = load_processed_data("data/data_process.csv")

    X, y, features = prepare_features(df)

    train_and_evaluate_model(X, y, features)