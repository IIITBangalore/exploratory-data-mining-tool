
## **Exploratory data mining:**
A flask based web application which enables users to upload custom csv based datasets and run various data mining algorithms on it. As of now the following techniques are implemented:

 - **Association rule mining** :
	 - Apriori algorithm
	 - Force directed plot
	 - Parallel cordinates plot
- **Clustering algorithms** :
	- Agglomerative
	- DBSCAN
	- K-means
	- K-modes
	- K-prototypes
## **Installing dependencies:**

 1. cd into the root directory of the project.
 2. Run `pip3 install -r requirements.txt`

## **Steps to run:**

**1. Setup environment**
  - cd into root directory of the project
  - Generate a SECRET_KEY by running the following commands in python:
```
>>> import os
>>> os.urandom(24)
'\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
```
-  Create a file `.env` and paste the following content into it:
  ```
FLASK_ENV=development
UPLOAD_FOLDER='./uploads/dataset'
SECRET_KEY='paste the generated key here'
```

**2. Run the flask app**
- cd into the root directory of the project
 - Run `python3 project.py`


**Outcome:** The web app will run on `0.0.0.0:5000`

### Libraries
 - PyClustering - Andrei Novikov
 - Apyori - Yu Mochizuki
 - KModes - Nico de Vos
