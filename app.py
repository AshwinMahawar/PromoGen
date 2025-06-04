from flask import Flask, request, jsonify, render_template,redirect,url_for
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from flask_sqlalchemy import SQLAlchemy

fine_tuned_model_path = "promogen_final_model"

model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    device_map="auto",
    # torch_dtype="auto"
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////PromoGen/database.db'
db = SQLAlchemy(app)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))
 
with app.app_context():    
    db.create_all()

@app.route("/")
def homepage():
    return render_template("home-1.html")

@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        
        login = user.query.filter_by(username=uname, password=passw).first()
        
        return redirect(url_for("homepage"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username = uname, email = mail, password = passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/team",methods = ['POST', 'GET'])
def team():
    ad_script = None
    cta = None
    if request.method == 'POST':
        company_name = request.form.get("company_name")
        product_name = request.form.get("product_name")
        product_description = request.form.get("product_description")
        print(company_name)
        print(product_name)
        print(product_description)
    # Prompt format as per your fine-tuning
        input_text = (
            "### Instruction: Write a highly engaging advertisement script and a strong call to action for the product below.\n"
            "The ad script should creatively describe the product's features and benefits, and the call to action should motivate the audience to take action immediately.\n\n"
            f"Company Name: {company_name}\n"
            f"Product Name: {product_name}\n"
            f"Product Description: {product_description}\n\n"
            "Ad Script:\n"
        )
        print(input_text)

    # Tokenize and move to GPU
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

    # Generate the ad content
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Extract Ad Script and CTA using regex
        pattern = r"Ad Script:\s*(.*?)\s*CTA:\s*(.*?)(?:\nCompany Name:|\Z)"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            ad_script = match.group(1).strip()
            cta = match.group(2).strip()
        else:
            ad_script = "Failed to extract Ad Script."
            cta = "Failed to extract CTA."
        

    return render_template("h.html", ad_script=ad_script, cta=cta)


if __name__ == "__main__":
    app.run(debug=True)