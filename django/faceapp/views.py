import sys
import cv2
import os
import numpy as np
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.core.urlresolvers import reverse
#from django.urls import reverse
#from django.urls import reverse
from faceapp.models import PicUpload
from faceapp.forms import ImageForm, DoubleImageForm
from mtcnn import MTCNN

# Create your views here.
def index(request):
    return render(request,'index.html')

def FaceDetection(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile = request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('FaceDetection'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request,'FaceDetection.html',
    {'documents':documents,'image_path1': image_path1,'form':form}
    )




def GenderPrediction(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile = request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('GenderPrediction'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request,'GenderPrediction.html',
    {'documents':documents,'image_path1':image_path1,'form':form}
    )



def AgeEstimation(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile = request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('AgeEstimation'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request,'AgeEstimation.html',
    {'documents':documents,'image_path1':image_path1,'form':form}
    )

def FacialEmotionRecognition(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile = request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('FacialEmotionRecognition'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request,'FacialEmotionRecognition.html',
    {'documents':documents,'image_path1':image_path1,'form':form}
    )

def FaceVerification(request):
    image_paths = np.empty((0))
    image_path1 = ''
    image_path2 = ''

    if request.method == "POST":
        form = DoubleImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc1 = PicUpload(imagefile = request.FILES['imagefile1'])
            newdoc1.save()

            newdoc2 = PicUpload(imagefile = request.FILES['imagefile2'])
            newdoc2.save()

            return HttpResponseRedirect(reverse('FaceVerification'))
    else:
        form = DoubleImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        current_image_path = '/' + document.imagefile.name
        document.delete()
        image_paths = np.append(image_paths, np.array([current_image_path]), axis=0)

    if (len(image_paths) == 2):
        image_path1 = image_paths[0]
        image_path2 = image_paths[1]

        request.session['image_path1'] = image_path1
        request.session['image_path2'] = image_path2

    return render(request,'FaceVerification.html',
    {
        'documents':documents,
        'image_path1':image_path1,
        'image_path2':image_path2,
        'form':form}
    )

def FaceGeneration(request):
    image_path = ''
    image_path1 = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imagefile = request.FILES['imagefile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('FaceGeneration'))
    else:
        form = ImageForm()

    documents = PicUpload.objects.all()

    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request,'FaceGeneration.html',
    {'documents':documents,'image_path1':image_path1,'form':form}
    )

import os
import json
import base64
from io import BytesIO

import h5py
import numpy as np
import pickle as pk
from PIL import Image

from keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
#import tensorflow.python.keras.backend as K
from keras import backend as K
import tensorflow as tf

def prepare_img_224(img_path):
    img = load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def prepare_age_img_224(img_path):
    img = load_img(img_path,target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    return x

def prepare_img_array(img_path,width,height,color_mode='rgb'):
    img = load_img(img_path,target_size=(width,height),color_mode=color_mode)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

with open('static/cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)

cat_list = [k for k, v in cat_counter.most_common()[:27]]

global graph
graph = tf.get_default_graph()

def prepare_flat(img_224):
    base_model = load_model('static/vgg16.h5')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    feature = model.predict(img_224)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    return flat

CLASS_INDEX_PATH = 'static/imagenet-class-index.json'

def get_predictions(preds, top=5):
    global CLASS_INDEX

    CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x:x[2], reverse=True)
        results.append(result)
    return results

def face_categories_check(img_224):
    first_check = load_model('static/vgg16.h5')
    print("Validating that this is a face picture...")
    out = first_check.predict(img_224)
    top = get_predictions(out, top=5)
    for j in top[0]:
        if j[0:2] in cat_list:
            print("Face check passed!")
            print('\n')
            return True
    return False

def gender_prediction(img_flat):
    second_check = pk.load(open('static/gender_classifier.pickle', 'rb'))
    print("Predicting the gender...")
    train_labels = ['male', 'female']
    preds = second_check.predict(img_flat)
    prediction = train_labels[preds[0]]
    print("This is a face of" + train_labels[preds[0]])
    print ("Task complete.")
    print("\n")
    print ("Thank you for using HexaFace!")
    return prediction

#
# def gender_prediction(img_224):
#     print('LOADING GENDER MODEL')
#     with open('static/gender.json','r') as f:
#         model_json = json.load(f)
#     model = model_from_json(json.dumps(model_json))
#     model.load_weights('static/gender.h5')
#     print('LOADING GENDER MODEL COMPLETE')
#     print('PREDICTING EMOTION')
#     preds = model.predict(img_224)
#     #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
#     train_labels = ['male', 'female']
#     index = np.argwhere(preds[0]>=1)
#     prediction = train_labels[index[0,0]]
#     print(preds)
#     print('GENDER PREDICTION COMPLETE')
#     return prediction

def emotion_detection(img_224):
    print('LOADING EMOTION MODEL')
    with open('static/fer.json','r') as f:
        model_json = json.load(f)
    model = model_from_json(json.dumps(model_json))
    model.load_weights('static/fer.h5')
    print('LOADING EMOTION MODEL COMPLETE')
    print('PREDICTING EMOTION')
    preds = model.predict(img_224)
    #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    train_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    index = np.argwhere(preds[0]>=1)
    prediction = train_labels[index[0,0]]
    print(preds)
    print('PREDICTION EMOTION COMPLETE')
    return prediction

def age_estimation(img_224):
    print('LOADING AGE MODEL')
    model = load_model('static/age_model_final.h5')
    print('LOADING AGE Estimation MODEL COMPLETE')
    print('PREDICTING AGE')
    preds = model.predict(img_224)
    prediction=np.argmax(preds)
    #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    #train_labels = map(str, range(100 + 1))
    #index = np.argwhere(preds[0]>=1)
    #prediction = train_labels[index[0,0]]
    #print(preds)
    print('Age Estimation Complete')
    print("most dominant age class: ",prediction)
    return str(prediction)

def generate_one_face():
    model = tf.keras.models.load_model('static/face_gen.h5',compile=False)
    noise = tf.random.normal([1, 300])
    face = model.predict(noise,steps=1)
    return face[0]

# draw an image with detected objects
def draw_image_with_boxes(data, result_list):
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()
    
def engine_1(request):
    hface=request.session['image_path']
    img_path = hface
    request.session.pop('image_path', None)
    request.session.modified = True
    img = cv2.imread(img_path)
    detector = MTCNN()
    result=detector.detect_faces(img)
    color=(0,0,255)
    for r in result:
        bounding_box = r['box']
        keypoints = r['keypoints']

        cv2.rectangle(img,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color)
        cv2.putText(img, '{:.2f}'.format(r['confidence']),
                       (bounding_box[0], bounding_box[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,color)
        cv2.circle(img,(keypoints['left_eye']), 2, color)
        cv2.circle(img,(keypoints['right_eye']), 2, color)
        cv2.circle(img,(keypoints['nose']), 2, color)
        cv2.circle(img,(keypoints['mouth_left']), 2, color)
        cv2.circle(img,(keypoints['mouth_right']), 2,color)

    base_64 = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)
    K.clear_session()

    context = {'img': 'data:image/jpg;base64,' + base_64}

    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')




def engine_2(request):

    hface=request.session['image_path']
    img_path = hface
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        img_224 = prepare_img_224(img_path)
        img_flat = prepare_flat(img_224)
        g2 = gender_prediction(img_flat)

    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)
    K.clear_session()

    #return render(request,'results_2.html', context={'g1_pic':g1_pic, 'gender':g2})
    context={'gender':g2}

    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')


# def engine_3(request):
#
#     hface=request.session['image_path']
#     img_path = hface
#     request.session.pop('image_path', None)
#     request.session.modified = True
#     with graph.as_default():
#
#         img_224 = prepare_img_224(img_path)
#         img_flat = prepare_flat(img_224)
#         g1 = face_categories_check(img_224)
#
#         while True:
#             try:
#
#                 if g1 is False:
#                     g1_pic = "Are you sure it is a face?"
#                     break
#                 else:
#                     g1_pic = "it is a face!"
#
#                     g2 = gender_prediction(img_flat)
#                     break
#             except:
#                 break
#
#     src= 'pic_upload/'
#     import os
#     for image_file_name in os.listdir(src):
#         if image_file_name.endswith(".jpg"):
#             os.remove(src + image_file_name)
#     K.clear_session()
#
#     return render(request,'results_3.html', context={'g1_pic':g1_pic, 'gender':g2})

def engine_3(request):

    hface=request.session['image_path']
    img_path = hface
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        #img_224 = prepare_img_224(img_path)
        #img_flat = prepare_flat(img_224)
        #fer model requires grayscale 48 x 48
        #img_age = prepare_img_array(img_path,width=224,height=224,color_mode='grayscale')
        # g1 = face_categories_check(img_224)
        #g2 = age_estimation(img_age)
        img_age=prepare_age_img_224(img_path)
        g2 = age_estimation(img_age)
        # if g1 is False:
        #     g1_pic = "Are you sure it is a face?"
        # else:
        #     g1_pic = "it is a face!"
        #     g2 = emotion_detection(img_fer)
        # """
        # while True:
        #     try:
        #         if g1 is False:
        #             g1_pic = "Are you sure it is a face?"
        #             break
        #         else:
        #             g1_pic = "it is a face!"
        #
        #             g2 = emotion_detection(img_224)
        #             print('ITS A FACE CONTINUES')
        #             break
        #     except:
        #         break
        # """

    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)
    K.clear_session()
    context={'age':g2}
    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')

def engine_4(request):
    print('test')
    hface=request.session['image_path']
    img_path = hface
    request.session.pop('image_path', None)
    request.session.modified = True
    with graph.as_default():

        # img_224 = prepare_img_224(img_path)
        #img_flat = prepare_flat(img_224)
        #fer model requires grayscale 48 x 48
        img_fer = prepare_img_array(img_path,width=48,height=48,color_mode='grayscale')
        # g1 = face_categories_check(img_224)
        g2 = emotion_detection(img_fer)

        # if g1 is False:
        #     g1_pic = "Are you sure it is a face?"
        # else:
        #     g1_pic = "it is a face!"
        #     g2 = emotion_detection(img_fer)
        # """
        # while True:
        #     try:
        #         if g1 is False:
        #             g1_pic = "Are you sure it is a face?"
        #             break
        #         else:
        #             g1_pic = "it is a face!"
        #
        #             g2 = emotion_detection(img_224)
        #             print('ITS A FACE CONTINUES')
        #             break
        #     except:
        #         break
        # """

    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg"):
            os.remove(src + image_file_name)
    K.clear_session()
    context={'fer':g2}

    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')

def engine_5(request):
    import mtcnn
    import keras
    import matplotlib.pyplot as plt
    import numpy as np

    from PIL import Image
    from mtcnn import MTCNN
    from numpy import asarray, expand_dims

    from keras.models import Model, load_model
    from keras import Sequential
    from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
    from keras.models import model_from_json
    from keras_vggface.utils import preprocess_input, decode_predictions

    def detect_face(image_path):
        # create detector
        detector = MTCNN()

        # read image
        pixels = plt.imread(image_path)

        # detect face in the input image
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height

        # get face pixels from input image
        face_pixels = pixels[y1:y2, x1:x2]

        # convert pixels array to image and resize to (224, 224)
        face_image = Image.fromarray(face_pixels)
        face_image = face_image.resize([224,224])

        # again back to array
        face_array = asarray(face_image)

        return face_array

    def preprocess(image_array):
        # convert to float32
        image_array = image_array.astype('float32')

        # expand dimensions
        processed_array = expand_dims(image_array, axis=0)

        # normalize
        processed_array = preprocess_input(processed_array, version=2)

        return processed_array

    def findCosineDistance(source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def findEuclideanDistance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    epsilon = 0.40 #cosine similarity
    #epsilon = 120 #euclidean distance

    def verify_face(img1, img2):
        # read json file
        json_file = open('static/face_verification.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # load model
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights("static/face_verification.h5")

        vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

        # extract face from each image
        face1 = detect_face(img1)
        face2 = detect_face(img2)

        # save detected images
        img_to_save1 = Image.fromarray(face1)
        img_to_save1.save('pic_upload/detected_face1.jpeg')
        img_to_save2 = Image.fromarray(face2)
        img_to_save2.save('pic_upload/detected_face2.jpeg')

        # preprocess each face (normalize)
        face1_preprocessed = preprocess(face1)
        face2_preprocessed = preprocess(face2)

        # get feature vectors of each face
        img1_representation = vgg_face_descriptor.predict(face1_preprocessed)[0,:]
        img2_representation = vgg_face_descriptor.predict(face2_preprocessed)[0,:]

        # compute similarity scores
        cosine_similarity = findCosineDistance(img1_representation, img2_representation)
        euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

        # decide if they are match or not
        if(cosine_similarity < epsilon):
            return(True, cosine_similarity, euclidean_distance)
        else:
            return(False, cosine_similarity, euclidean_distance)

    hface1=request.session['image_path1']
    hface2=request.session['image_path2']

    img_path1 = hface1[1:]
    img_path2 = hface2[1:]

    request.session.pop('image_path1', None)
    request.session.pop('image_path2', None)

    request.session.modified = True

    with graph.as_default():
        is_match, cosine_similarity, euclidean_distance = verify_face(img_path1, img_path2)

    src= 'pic_upload/'
    import os
    for image_file_name in os.listdir(src):
        if image_file_name.endswith(".jpg") and not image_file_name.startswith('detected_face') :
            os.remove(src + image_file_name)

    result_text = ''

    if is_match == True:
        result_text = 'Verified. They are same person.'
    else:
        result_text = 'Not Verified. They are not same person.'

    context={'result_text':result_text,
             'cosine_similarity': str(cosine_similarity),
             'euclidean_distance': str(euclidean_distance)
             }
    results = json.dumps(context)

    return HttpResponse(results, content_type='application/json')

def engine_6(request):

    hface=request.session['image_path']
    img_path = hface
    """
    batch_complete = int(request.GET['batch_complete'])
    if(batch_complete == 1):
        print('poppin---------')
        request.session.pop('image_path', None)
    """
    request.session.modified = True

    num_faces = 3
    imgs = []
    genders = []

    for i in range(num_faces):
        with graph.as_default():
            print('GENERATING FACE {0}'.format(i+1))
            raw_img = generate_one_face()
            de_norm = np.uint8( ((raw_img + 1) / 2.0) * 255 )
            im = Image.fromarray(de_norm)
            output = BytesIO()
            im.save(output, format='JPEG')
            im_data = output.getvalue()
            base_64 = base64.b64encode(im_data).decode('utf-8')
            imgs.append('data:image/jpg;base64,'+base_64)

            img_224 = cv2.resize(de_norm, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            img_flat = prepare_flat(np.expand_dims(img_224, axis=0))
            gender = gender_prediction(img_flat)
            genders.append(gender)
    """
    if(batch_complete == 1):
        src= 'pic_upload/'
        for image_file_name in os.listdir(src):
            if image_file_name.endswith(".jpg") or image_file_name.endswith(".png"):
                os.remove(src + image_file_name)
    """

    context={'imgs':imgs,'genders':genders}

    results = json.dumps(context)
    return HttpResponse(results, content_type='application/json')

    #return render(request,'results_6.html', context={'imgs':imgs})
