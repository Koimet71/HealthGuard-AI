
from flask import Flask, render_template, request
import pandas as pd
import os
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

df = pd.read_csv("dataset.csv")

ALL_SYMPTOMS=[
"itching","skin_rash","nodal_skin_eruptions","continuous_sneezing",
"shivering","chills","joint_pain","stomach_pain","acidity",
"ulcers_on_tongue","muscle_wasting","vomiting","burning_micturition",
"spotting_urination","fatigue","weight_gain","anxiety",
"cold_hands_and_feets","mood_swings","weight_loss"
]

binary_df=pd.DataFrame()
for symptom in ALL_SYMPTOMS:
    binary_df[symptom]=df.iloc[:,1:].apply(
        lambda row:int(symptom in [str(v).strip().lower() for v in row.values if pd.notna(v)]),
        axis=1
    )

binary_df["Disease"]=df["Disease"]
X=binary_df.drop(columns=["Disease"])
y=binary_df["Disease"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=300,random_state=42)
model.fit(X_train,y_train)
accuracy=accuracy_score(y_test,model.predict(X_test))

HIGH={"Pneumonia","Malaria","Typhoid","Dengue"}
MEDIUM={"Common Cold","Allergy","Flu"}

def map_risk(d):
    if d in HIGH:return "High"
    if d in MEDIUM:return "Medium"
    return "Low"

@app.route("/",methods=["GET","POST"])
def index():
    disease=""
    result=""
    confidence=0
    advice=""
    selected_symptoms=[]
    timestamp=""
    if request.method=="POST":
        inputs=[]
        for s in ALL_SYMPTOMS:
            val=int(request.form.get(s,0))
            inputs.append(val)
            if val:
                selected_symptoms.append(s.replace("_"," ").title())
        disease=model.predict([inputs])[0]
        confidence=round(model.predict_proba([inputs])[0].max()*100,2)
        result=map_risk(disease)
        if result=="Low":
            advice="Continue monitoring your symptoms, rest well and stay hydrated."
        elif result=="Medium":
            advice="Monitor your symptoms closely and consult a healthcare professional if they worsen."
        else:
            advice="Seek immediate medical attention. Avoid self-medication."

        timestamp=datetime.datetime.now().strftime("%d %b %Y • %H:%M")
        row=pd.DataFrame([{
            "Timestamp":timestamp,
            "Disease":disease,
            "Risk":result,
            "Confidence":confidence,
            "Symptoms":", ".join(selected_symptoms)
        }])
        if os.path.exists("prediction_history.csv"):
            old=pd.read_csv("prediction_history.csv")
            row=pd.concat([old,row],ignore_index=True)
        row.to_csv("prediction_history.csv",index=False)

    if os.path.exists("prediction_history.csv"):
        history=pd.read_csv("prediction_history.csv").iloc[::-1].head(5).to_dict("records")
    else:
        history=[]

    return render_template(
        "index.html",
        symptoms=ALL_SYMPTOMS,
        disease=disease,
        result=result,
        confidence=confidence,
        advice=advice,
        selected_symptoms=selected_symptoms,
        timestamp=timestamp,
        accuracy=round(accuracy*100,2),
        history=history
    )

if __name__=="__main__":
    app.run(debug=True)
