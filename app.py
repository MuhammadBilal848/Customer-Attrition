import uvicorn
from fastapi import FastAPI , Path
from typing import Optional
from pydantic import BaseModel,conint
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

with open('model_with_scaler.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

app = FastAPI()

class Prediction(BaseModel):
    Gender: conint(ge=0, le=1)
    HasPartner: conint(ge=0, le=1)
    HasDependents: conint(ge=0, le=1)
    HasPhoneService: conint(ge=0, le=1)
    HasMultiplePhoneServices: conint(ge=0, le=2)
    InternetServiceType: conint(ge=0, le=2)
    HasCloudSecurity: conint(ge=0, le=2)
    HasCloudBackup: conint(ge=0, le=2)
    HasDeviceCoverage: conint(ge=0, le=2)
    HasTechSupport: conint(ge=0, le=2)
    HasStreamingTV: conint(ge=0, le=2)
    HasStreamingMovies: conint(ge=0, le=2)
    SubscriptionType: conint(ge=0, le=2)
    HasElectronicBilling: conint(ge=0, le=1)
    PaymentMethod: conint(ge=0, le=3)
    MonthlySubscriptionFee: float
    IsSenior: conint(ge=0, le=1)
    ServiceDuration: int
    TotalSubscriptionCost: float

@app.get('/')
def index(name:str = None): 
    if name == None:
        return {'Welcome': 'Sych!!!'} 

@app.post('/post-predict/')
def prediction(prediction : Prediction):
    prediction = prediction.dict()  
    columns_to_scale = ['ServiceDuration', 'MonthlySubscriptionFee', 'TotalSubscriptionCost']
    data_to_scale = np.array([[prediction[col] for col in columns_to_scale]])
    scaler = loaded_model['scaler']
    model = loaded_model['model']
    scaled_data = scaler.transform(data_to_scale)
    for i, col in enumerate(columns_to_scale):
        prediction[col] = scaled_data[0][i]
        final_pred = model.predict([np.array(list(prediction.values()))])
    return f'You prediction is {final_pred}'

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
