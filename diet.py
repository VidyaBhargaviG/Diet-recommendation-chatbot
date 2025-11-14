import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

df = pd.read_csv("diet_data.csv")
model = pickle.load(open("diet_model.pkl", "rb"))# Load trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_diet', methods=['POST'])
def get_diet():
    try:
        age = int(request.form['age'])
        gender = request.form['gender'].lower()
        bmi = float(request.form['bmi'])
        chronic_disease = request.form['chronic_disease'].lower()
        blood_pressure_sys = int(request.form['blood_pressure_sys'])
        blood_pressure_dia = int(request.form['blood_pressure_dia'])
        cholesterol = int(request.form['cholesterol'])
        blood_sugar = int(request.form['blood_sugar'])
        allergies = request.form.getlist('allergies')
        diet_preference = request.form['diet_preference'].lower()

        # Convert disease name to number
        disease_mapping = {str(d).lower(): i for i, d in enumerate(df["Chronic_Disease"].dropna().unique())}
        chronic_disease_num = disease_mapping.get(chronic_disease, -1)

        if chronic_disease_num == -1:
            return jsonify({"error": "Disease not found in dataset"}), 404

        # Convert gender to numerical value (Male: 0, Female: 1)
        gender_num = 0 if gender == "male" else 1

        
        user_input = [[age, gender_num, bmi, chronic_disease_num, blood_pressure_sys, blood_pressure_dia, cholesterol, blood_sugar]]

        
        prediction = model.predict(user_input)
        diet_plan = {
            "Recommended_Calories": int(prediction[0][0]),
            "Recommended_Protein": int(prediction[0][1]),
            "Recommended_Carbs": int(prediction[0][2]),
            "Recommended_Fats": int(prediction[0][3])
        }

        
        meal_plan = df.loc[
            (df["Chronic_Disease"].str.lower() == chronic_disease) & 
            (df["Age"] == age) & 
            (df["Gender"].str.lower() == gender), 
            "Recommended_Meal_Plan"
        ]

        if not meal_plan.empty:
            diet_plan["Meal Plan"] = meal_plan.values[0]
        else:
            diet_plan["Meal Plan"] = "No specific meal plan available."

        
        if allergies:
            diet_plan["Avoid"] = ", ".join(allergies)

        return render_template("index.html", diet=diet_plan)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  

if __name__ == '__main__':
    app.run(debug=True)
