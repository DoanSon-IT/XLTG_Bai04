import time
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from utils import prepare_data

def train_and_log_results(model, model_name, X_train, X_test, y_train, y_test, log_file="README.md"):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

    # Ghi kết quả vào README.md
    with open(log_file, "a") as f:
        f.write(f"{model_name} - Time: {end_time - start_time:.2f}s, Accuracy: {accuracy:.2f}, "
                f"Precision: {precision:.2f}, Recall: {recall:.2f}\n")
    
    print(f"{model_name} - Time: {end_time - start_time:.2f}s, Accuracy: {accuracy:.2f}, "
          f"Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Lưu mô hình vào file .pkl
    with open(f"{model_name}.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

def main():
    X, y = prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    svm_model = SVC()
    knn_model = KNeighborsClassifier(n_neighbors=3)  
    decision_tree_model = DecisionTreeClassifier()

    # Huấn luyện và ghi kết quả cho từng mô hình
    train_and_log_results(svm_model, "SVM", X_train, X_test, y_train, y_test)
    train_and_log_results(knn_model, "KNN", X_train, X_test, y_train, y_test)
    train_and_log_results(decision_tree_model, "Decision Tree", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
