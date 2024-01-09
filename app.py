# from ultralytics import YOLO
import yolov5
from flask import request, Response, Flask, render_template
from waitress import serve
from PIL import Image
import json
import yolov5

app = Flask(__name__, static_url_path='/assets', static_folder='assets')

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["GET","POST"])
def detect():
    buf = request.files["image_file"]
    image = Image.open(buf.stream)
    boxes = detect_objects_on_image(image)
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )

def detect_objects_on_image(buf):
    model = yolov5.load("best.pt")
    results = model(buf)
    result = results.xyxy[0]  # Updated syntax for accessing bounding boxes

    output = []
    for box in result:
        print("Box:", box)
        x1, y1, x2, y2 = [round(x) for x in box[:4].tolist()]
        class_id = int(box[5])
        prob = round(box[4].item(), 2)
        output.append([x1, y1, x2, y2, model.names[class_id], prob])


    print(output)  # Tambahkan ini jika Anda ingin melihat output pada konsol
    return output


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

