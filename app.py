import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr


df=pd.read_excel("Salary.xlsx")
df.head(3)

y_dep=df.Salary
x_ind=df.drop(["Salary","Student"],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_ind,y_dep, test_size = 0.2,random_state=1)

model = LinearRegression() 
model.fit(x_train, y_train) 
y_pred = model.predict(x_test)

def salary(school_ranking,gpa,experience):
    input1=np.array([school_ranking,gpa,experience])
    output1=model.predict([input1])
    return output1[0].round()

interface = gr.Interface(fn = salary,
inputs=[gr.inputs.Number(default=1, label="School Ranking"), gr.inputs.Slider(1,10,step=0.1,label = "G.P.A"),gr.inputs.Slider(1,15,step=1,label = "Experience")], 
outputs = [gr.outputs.Textbox( label="Expected Salary")],description="SALARY PREDICTION")

interface.launch()