from app import app
from flask import flash, send_from_directory, render_template, request, redirect, url_for
from config import ALLOWED_EXT
from werkzeug.utils import secure_filename
from torchvision.transforms.functional import to_tensor, normalize
from app.load_models import DIM, model, mean, std, INT_TO_NAME
from PIL import Image
from torch.nn.functional import softmax
from torch import max
from flask import jsonify
import os


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("NO FILE PART")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("NO SELECTED FILE")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # if os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"],filename)):

            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for('predict', img=filename))
    return render_template("index.html")


# @app.route("/upload/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename=filename)

@app.route("/predict/<img>")
def predict(img):

    ts_X = to_tensor(Image.open(os.path.join(app.config["UPLOAD_FOLDER"], img)).convert(
        'RGB').resize((DIM, DIM))).view(-1, 3, DIM, DIM)

    for i in range(len(ts_X)):
        ts_X[i] = normalize(ts_X[i], mean=mean, std=std)

    model.eval()
    pred = softmax(model(ts_X), dim=1)

    payload = {"label":
               INT_TO_NAME[max(pred, dim=1).indices.item()]}
    return render_template("predict.html", payload=payload)
    # return jsonify(f"{INT_TO_NAME[max(pred, dim=1).indices.item()]}")
