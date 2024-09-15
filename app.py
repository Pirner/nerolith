import os
import io
import json
import base64

import cv2
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap5
from flask_wtf import CSRFProtect
import numpy as np
from PIL import Image

from src.config.DTO import ModelConfig
from src.forms.PredictionForm import NameForm
from src.modeling.inference.BinarySegmentationPipeline import BinarySegmentationInferencePipeline
from src.vision.rendering import VisionRendering


app = Flask(__name__)
app.config.from_pyfile('config.py')

# Bootstrap-Flask requires this line
bootstrap = Bootstrap5(app)
# Flask-WTF requires this line
csrf = CSRFProtect(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect/', methods=('GET', 'POST'))
def create():
    form = NameForm()
    if form.validate_on_submit():
        photo = request.files['photo']
        in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        im = cv2.imdecode(data, color_image_flag)
        # TODO clean up the code at this point
        # perform inference
        experiment_path = r'C:\project_data\nerolith\alpha_model'
        with open(os.path.join(experiment_path, 'config.json')) as json_data:
            config = json.load(json_data)
            json_data.close()

        renderer = VisionRendering()
        config = ModelConfig(**config)
        pipeline = BinarySegmentationInferencePipeline(config=config,
                                                       model_path=os.path.join(experiment_path, 'model.pth'))
        segmentation_map = pipeline.predict_image(im=im)
        segmentation_map = cv2.resize(segmentation_map, (im.shape[1], im.shape[0]), cv2.INTER_LINEAR_EXACT)
        overlay = renderer.render_binary_mask(im=im, mask=segmentation_map)
        cv2.imwrite('tmp.png', im)
        cv2.imwrite('overlay.png', overlay)

        # pil_im = Image.open("test.jpg")
        pil_im = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        data = io.BytesIO()

        # First save image as in-memory.
        pil_im.save(data, "JPEG")

        # Then encode the saved image file.
        encoded_img_data = base64.b64encode(data.getvalue())
        # end inference

        return render_template('predicted.html', form=form, img_data=encoded_img_data.decode('utf-8'))
    else:
        return render_template('prediction_request.html', form=form)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
