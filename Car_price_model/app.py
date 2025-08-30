from flask import Flask,render_template,request  #we use render_template to access html in flask ans show it on frontend
import pandas as pd
import pickle



app = Flask(__name__)
model = pickle.load(open('CAR_price_predictormodel.pkl','rb'))
car = pd.read_csv('new_quikr_cleaned_cardataset')

@app.route('/')

def index():
    company = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year= sorted(car['year'].unique())
    fuel_type =sorted(car['fuel_type'].unique())
    return render_template('index.html',companies =company,car_models=car_model,years =year, fuel_types=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('Company')
    car_model = request.form.get('name')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('driven'))
    print(company,car_model,year, fuel_type,kms_driven);
    df= pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven.1','fuel_type'])
    prediction = model.predict(df);
    print(prediction)
    
    
    return f"{prediction[0]:.4f}"

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug = True)