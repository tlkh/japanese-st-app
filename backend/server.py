print("Starting server script")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import flask
from flask_cors import CORS
import torch
import transformers
import traceback

app = flask.Flask(__name__)
app.config['TESTING'] = True
cors = CORS(app)

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = transformers.MBartForConditionalGeneration.from_pretrained(model_name)
model = model.eval()
tokenizer = transformers.MBart50Tokenizer.from_pretrained(model_name)


def translate(in_text, src="en_XX", tgt="ja_XX"):
    if len(in_text) > 0:
        tokenizer.src_lang = src
        encoded = tokenizer(in_text, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**encoded,
                                       forced_bos_token_id=tokenizer.lang_code_to_id[tgt])
        translation = tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]
        translation = translation.strip()
    else:
        translation = ""
    return translation


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}
    try:
        request = flask.request.get_json(force=True)
        src = request["src"]
        tgt = request["tgt"]
        in_text = request["text"]
        translation = translate(in_text, src, tgt)
        data["success"] = True
        data["translation"] = translation
    except Exception as e:
        error_string = str(e) + " - " + str(traceback.format_exc())
        print("Error:", error_string)
        data["error"] = error_string
    return flask.jsonify(data)


if __name__ == "__main__":
    print("Starting Flask server")
    try:
        app.run(host="0.0.0.0", port=5000)
    except:
        pass
