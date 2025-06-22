from flask import Flask
from importlib import reload
from flask import request,redirect,render_template,url_for
from werkzeug.utils import secure_filename
import sqlite3
import os
from FinalYearProject.dense_python import predict_image
from FinalYearProject.binary_model_copy import test_model_on_image

import sys
reload(sys)

"Create table in database TextExtractor.db"
conn = sqlite3.connect('TextExtractor.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS Contact
                (name text, email text,message text)''')
conn.commit()
conn.close()

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__,template_folder='templates',static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submitImage',methods=['POST',])
def submitImage():
    image = request.files['ocrImage']
    filename = secure_filename(image.filename)
    if(filename == ''):
        print('No file selected')
        return redirect(url_for('index'))

    elif(filename.split('.')[1] != 'jpg' and filename.split('.')[1] != 'jpeg' and filename.split('.')[1] != 'png' and filename.split('.')[1] != 'gif' and filename.split('.')[1] != 'avif' and filename.split('.')[1] != 'bmp' and filename.split('.')[1] != 'tiff' and filename.split('.')[1] != 'webp'):
        print('Invalid file type')
        return redirect(url_for('index'))

    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # get image path
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # append ./static to front
    img_path = './static/' + filename
  
    if filename.startswith('Apple'):
       maturity , quality = predict_image(img_path,model_path="densenet_modelapple.h5", le_main_classes_path="le_main_classes_densenetapple.npy", le_subclass_classes_path="le_subclass_classes_densenetapple.npy")
       banana_on_tree = test_model_on_image(img_path,model="MobileNetv2_model.h5") 
    else:
      maturity , quality = predict_image(img_path,model_path="densenet_model.h5", le_main_classes_path="le_main_classes_densenet.npy", le_subclass_classes_path="le_subclass_classes_densenet.npy")
      banana_on_tree = test_model_on_image(img_path,model="MobileNetv2_model.h5")
    return render_template('textFile.html',maturity=maturity,quality=quality,banana_on_tree=banana_on_tree,filename=filename)
   



@app.route('/history',methods=['GET',])
def history():
    conn = sqlite3.connect('TextExtractor.db')
    c = conn.cursor()
    c.execute("SELECT * FROM TextExtractor")
    data = c.fetchall()
    return render_template('history.html',data=data)


@app.route('/contact_us',methods=['GET','POST'])
def contact_us():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        conn = sqlite3.connect('TextExtractor.db')
        c = conn.cursor()
        c.execute("INSERT INTO Contact (name,email,message) VALUES (?,?,?)",(name,email,message))
        conn.commit()
        conn.close()
        return render_template('contact_us.html',message='Thank you for contacting us. We will get back to you soon.')
    return render_template('contact_us.html')

@app.route('/message',methods=['GET','POST'])
def message():
    conn = sqlite3.connect('TextExtractor.db')
    c = conn.cursor()
    c.execute("SELECT * FROM Contact")
    data = c.fetchall()
    return render_template('message.html',messages=data)

if __name__ == '__main__':
    app.run('0.0.0.0',8000, debug=True)


