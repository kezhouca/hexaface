# HexaFace

This is the project for CMPT733 - Programming for Big Data II.

# Requirements:

- Latest version of Anaconda (https://www.anaconda.com/)

- TensorFlow 1.15 [conda create --name tf1 tensorflow=1.15]

- Keras [conda install keras]

- OpenCV [conda install opencv]

- Django 1.10 [pip install django==1.10]

- MTCNN [pip install mtcnn]

- keras-vggface [pip install keras-vggface] 

- Scikit-Learn [conda install scikit-learn]

- matplotlib [conda install matplotlib]

# Run Application:
In order to run the product, you need to go to the HexaFace/django/ folder in command line.<br>
Now, run the following command to run the server:<br>
python manage.py runserver<br><br>

if you get any errors when starting, please run:<br>
python manage.py migrate<br><br>

This command runs the application server on http://127.0.0.1:8000 <br>
Pay attention that port 8000 may be occupied on your system.

# Deploy on Google Cloud:
http://hexaface.ddns.net