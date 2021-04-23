from flask import url_for,render_template,redirect,request,flash
from flaskpackage import app
from flaskpackage.form import ImageUploadForm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import os

def yolo_loss(y_true,y_pred):
   no_obj = (1 - y_true[:,:,:,0])*0.5
   coords = y_true[:,:,:,0]*5
   c_loss = K.sum(K.square(y_true[:,:,:,0]-y_pred[:,:,:,0])*no_obj)
   c_loss1 =  K.sum(K.square(y_true[:,:,:,0]-y_pred[:,:,:,0])*coords)
   w_loss = K.sum(K.square(K.sqrt(y_true[:,:,:,3])-K.sqrt(y_pred[:,:,:,3]))*coords)
   h_loss = K.sum(K.square(K.sqrt(y_true[:,:,:,4])-K.sqrt(y_pred[:,:,:,4]))*coords)
   x_loss = K.sum(K.square(y_true[:,:,:,1]-y_pred[:,:,:,1])*coords)
   y_loss = K.sum(K.square(y_true[:,:,:,2]-y_pred[:,:,:,2])*coords)
   loss = c_loss + c_loss1 + w_loss + h_loss + x_loss + y_loss
   return loss

#app = Flask(__name__)


model =keras.models.load_model(os.path.join(app.config['MODEL_FOLDER'], 'yolo_model.h5'),
                                    custom_objects={'yolo_loss': yolo_loss})

#from project import gen_bounding_boxes
from flaskpackage.project import gen_bounding_boxes

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    form = ImageUploadForm()
    if form.validate_on_submit():
        if allowed_file(form.image.data.filename):
            form.image.data.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], 'answer.jpg')):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'answer.jpg'))

            text = gen_bounding_boxes(model)
            return render_template('result.html',text=text)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif','danger')
    return render_template('home.html',form=form)


@app.route('/about')
def about():
    return render_template('about.html')