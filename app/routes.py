from app import app
from flask import flash, send_from_directory, render_template, request, redirect, url_for
from config import ALLOWED_EXT
from werkzeug.utils import secure_filename
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
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template("index.html")


@app.route("/upload/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename=filename)
