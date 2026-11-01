import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

data = {
    "StudyHours": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8,
        9, 9, 10, 10, 11
    ],

    "Attendance": [
        50, 55, 60, 65, 68,
        70, 72, 75, 78, 80,
        82, 84, 85, 87, 88,
        90, 92, 93, 95, 96
    ],

    "PreviousMarks": [
        30, 35, 38, 40, 42,
        45, 48, 50, 52, 55,
        58, 60, 62, 65, 68,
        70, 73, 75, 78, 82
    ],

    "Result": [
        0, 0, 0, 0, 0,
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)

print("Student Dataset:")
print(df)





X = df[["StudyHours", "Attendance", "PreviousMarks"]]

y = df["Result"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




model = LogisticRegression()


model.fit(X_train, y_train)




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")


print("\nEnter Student Details")

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
previous_marks = float(input("Previous marks: "))


new_student = pd.DataFrame(
    [[study_hours, attendance, previous_marks]],
    columns=["StudyHours", "Attendance", "PreviousMarks"]
)

prediction = model.predict(new_student)

prediction = model.predict(new_student)



if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")