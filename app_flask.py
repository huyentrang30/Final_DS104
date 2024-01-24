from flask import Flask, render_template, request
from pyngrok import ngrok

from flask_app import utils_flask as f_utils
import utils as u


app = Flask(__name__, template_folder='./flask_app/templates')

# config for ngrok colab
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))
app.config["BASE_URL"] = public_url

# init Spark and read data
spark, sc = u.initialize_spark()
pd_df = f_utils.read_data(spark)

list_model = f_utils.init_ml_model()

@app.route("/")
def index():
    data = f_utils.create_dashboard(pd_df)
    return render_template("dashboard.html", data = data)

# @app.get('/')
# def index():
#     return render_template("dashboard.html", data = {})

@app.get('/predict')
def predict_site():
    df_html = pd_df.head(50).to_html()
    
    return render_template('model.html', data = {'df' : df_html})

@app.post('/get_prediction')
def get_prediction():
    req = request.get_json(silent=True)

    insert_type = req['insert_type']
    model = int(req['model'])
    data = req['inserted_data']

    res = f_utils.get_prediction(spark, pd_df, insert_type, list_model[model], data)

    return {'df_html' : res.to_html()}

    

if __name__ == "__main__":

    app.run(debug=False)
    # app.run(debug=True)