import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def plot_decision_boundary(X, y, models, feature_names):
    plt.figure(figsize=(12, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    colors = ['red', 'blue']
    labels = ['Logistic Regression', 'Linear SVM']
    
    for idx, (model, color, label) in enumerate(zip(models, colors, labels)):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contour(xx, yy, Z, levels=[0], colors=color, 
                   linestyles='-', linewidths=2)
        plt.contour(xx, yy, Z, levels=[-1, 1], colors=color, 
                   linestyles='--', linewidths=1)
        
        plt.plot([], [], color=color, label=label, linewidth=2)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', linewidth=1, alpha=0.8)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundaries Comparison:\nLogistic Regression vs Linear SVM')
    plt.legend(loc='upper right')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Loading data...")
    data = pd.read_csv("./prepare_dataset/data/dataset_new_cat.csv")
    
    feature_idx1, feature_idx2 = 1, 6
    class_idx1, class_idx2 = 0, 2  # Select first two classes
    
    feature_names = data.columns[:-1].tolist()
    X = data.iloc[:, [feature_idx1, feature_idx2]].values
    y = data.iloc[:, -1].values
    
    print(f"Selected features: {feature_names[feature_idx1]} and {feature_names[feature_idx2]}")
    
    mask = (y == class_idx1) | (y == class_idx2)
    X = X[mask]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining models...")
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Train SVM
    svm_model = SVC(kernel='linear', C=0.5, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    print("\nPlotting decision boundaries...")
    selected_features = [feature_names[feature_idx1], feature_names[feature_idx2]]
    plot_decision_boundary(
        X_train_scaled, 
        y_train, 
        [lr_model, svm_model],
        selected_features
    )
    
    # Print performance metrics
    print("\nModel Performance:")
    print("-" * 50)
    print("Logistic Regression:")
    print(f"Training accuracy: {lr_model.score(X_train_scaled, y_train):.4f}")
    print(f"Testing accuracy: {lr_model.score(X_test_scaled, y_test):.4f}")
    
    print("\nLinear SVM:")
    print(f"Training accuracy: {svm_model.score(X_train_scaled, y_train):.4f}")
    print(f"Testing accuracy: {svm_model.score(X_test_scaled, y_test):.4f}")

if __name__ == "__main__":
    main()