import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([[0],[0],[0],[1],[1],[1],[1],[1],[1],[1]])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Model accuracy: {accuracy}")

#make prediction on unseen data
X=np.array([[7.5],[3.9],[50],[1]])
predictions=model.predict(X)
hours=np.array([7.5,3.9,50,1])
for h,r in zip(hours,predictions):
    print(f"if study hours is {h}, hours:{" pass " if r else " fail "}")
