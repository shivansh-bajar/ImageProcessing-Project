from flask import Flask,render_template,request
from sklearn.cluster import KMeans
import numpy as np
import cv2
import os


app=Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    imgfile=request.files['imagefile']
    cluster=request.form.get('cluster')
    image_path="./images/"+ imgfile.filename
    file, file_extension = os.path.splitext(image_path)
    imgfile.save(image_path)
    cluster=int(cluster)

    img = cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    rows = img.shape[0]
    cols = img.shape[1]

    img = img.reshape(rows*cols, 3)

    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(img)

    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    compressed_image = compressed_image.reshape(rows, cols, 3)

    final_img=cv2.cvtColor(compressed_image,cv2.COLOR_RGB2BGR)
    directory= r'C:\Users\Shivansh Bajar\Desktop'
    os.chdir(directory)
    filename="compressed_img"+file_extension
    cv2.imwrite(filename,final_img)

    return render_template('output.html')



if __name__ == '__main__':
    app.run(port=3000,debug=True)

