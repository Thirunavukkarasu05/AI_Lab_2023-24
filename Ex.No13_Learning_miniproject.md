# Implementation-of-SVM-For-Spam-Mail-Detection
## Register Number:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.
## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
## Program:
```

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()data.isnull().sum
x=data['v1'].values
y=data['v2'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## data head
![Screenshot 2024-04-29 141243](https://github.com/Thirunavukkarasu05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119291645/82e1870e-f060-43eb-a23b-d1e6842a6d2f)
## data info
![Screenshot 2024-04-29 141256](https://github.com/Thirunavukkarasu05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119291645/0b2ee05c-8ce0-40f7-855a-b46e3bcd6099)
## isnull
![Screenshot 2024-04-29 141351](https://github.com/Thirunavukkarasu05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119291645/ed760152-4e4b-4dc0-8a84-a81196193979)
## y_pred
![Screenshot 2024-04-29 141405](https://github.com/Thirunavukkarasu05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119291645/8936a0dd-2fad-41c5-bc53-79282e5c577e)
## accuracy
![Screenshot 2024-04-29 141410](https://github.com/Thirunavukkarasu05/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119291645/ebe7e0a4-ae02-4abb-8b8f-4a371b861cc0)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
