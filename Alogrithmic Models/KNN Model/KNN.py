import math
from collections import Counter

# Training Data
X = [
    [150, 45],
    [155, 50],
    [160, 55],
    [175, 75],
    [180, 80],
    [185, 90]
]

y = [
    "Student",
    "Student",
    "Student",
    "Adult",
    "Adult",
    "Adult"
]

def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)

def knn_predict(X_train, y_train, test_point, k=3):
    distances = []

    # Calculate distance from test point to each training sample
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Take k nearest labels
    k_neighbors = [label for _, label in distances[:k]]

    # Majority vote
    prediction = Counter(k_neighbors).most_common(1)[0][0]

    return prediction

# Test
new_person = [165, 60]
result = knn_predict(X, y, new_person, k=3)

print("Prediction:", result)