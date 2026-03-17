from flask import Flask, render_template, request
from model_pipeline import run_correlation_pipeline
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "" and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            result = run_correlation_pipeline(save_path)

            # ⭐ FIXED PATH
            image_path = save_path.replace("\\", "/")

    return render_template(
        "index.html",
        result=result,
        image=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)