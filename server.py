import os
import csv
import pickle
import pandas as pd
from flask import Flask, request, render_template, flash, redirect, session, url_for
from datetime import datetime
from analytics import write_to_csv_departments, write_to_csv_teachers
from analytics import get_counts, get_tables, get_titles
from teacherdashboard import get_feedback_counts
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import functools
import numpy as np
import csv
import re
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration (update with your database URI)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect to the login page if not authenticated

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Load user by ID from the database

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Define CSV_FILE location
CSV_FILE = 'users.csv'  # Update the path as per your project structure

@app.route('/student_registration', methods=['GET', 'POST'])
def student_registration():
    if request.method == 'POST':
        # Capture only the USN and password
        username = request.form.get('usn', '').strip()  # Ensure this matches the form input name
        password = request.form.get('password', '').strip()

        # USN Validation pattern (first character should be 3)
        usn_pattern = r'^[3]{1}[a-zA-Z0-9]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{3}$'
        if not re.match(usn_pattern, username):
            flash("Invalid USN format. Format: 3BR21AI001 (first character must be 3).", 'danger')
            return redirect(url_for('student_registration'))

        # Password Validation pattern
        password_pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_pattern, password):
            flash("Password must be at least 8 characters long, include 1 uppercase letter, 1 lowercase letter, 1 number, and 1 special character.", 'danger')
            return redirect(url_for('student_registration'))

        # Check if the user already exists in the CSV file
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row
            for row in reader:
                if row[0] == username:
                    flash("User already exists. Please login.", 'warning')
                    return redirect(url_for('login'))

        # Register the user: only store USN and hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, hashed_password])  # Only store USN and password

        flash("Registration successful! Please log in.", 'success')
        return redirect(url_for('login'))

    return render_template('student_registration.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Input validation
        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for('login'))

        # USN validation pattern (for students)
        usn_pattern = r'^[3]{1}[a-zA-Z0-9]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{3}$'
        if not re.match(usn_pattern, username) and username not in ['admin', 'hod'] and not username.startswith('teacher'):
            flash("Invalid username format.", "danger")
            return redirect(url_for('login'))

        # Check for admin, HOD, or teacher credentials
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            flash("Admin login successful!", "success")
            return redirect(url_for('root'))
        elif username == 'hod' and password == 'hod':
            session['logged_in'] = True
            flash("HOD login successful!", "success")
            return redirect(url_for('hoddashboard'))
        elif username.startswith('teacher') and username[7:].isdigit() and password == username:
            session['logged_in'] = True
            teacher_id = int(username[7:])
            flash(f"Teacher {teacher_id} login successful!", "success")
            return redirect(url_for('teacherdashboard', teacher_id=teacher_id))

        # Validate student credentials from the CSV
        try:
            with open(CSV_FILE, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    if row[0] == username and check_password_hash(row[1], password):
                        session['logged_in'] = True
                        flash("Student login successful!", "success")
                        return redirect(url_for('feedback_system'))
        except FileNotFoundError:
            flash("No users registered yet. Please register first.", "danger")
            return redirect(url_for('student_registration'))

        # If credentials are invalid
        flash("Invalid username or password. Please try again.", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    if request.method == 'POST':
        usn = request.form.get('usn', '').strip()

        # Validate USN
        usn_pattern = r'^[3]{1}[a-zA-Z0-9]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{3}$'
        if not re.match(usn_pattern, usn):
            flash("Invalid USN format. Please try again.", "danger")
            return redirect(url_for('forget_password'))

        # Check if USN exists in the CSV or database
        user_found = False
        try:
            with open(CSV_FILE, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['USN'] == usn:
                        user_found = True
                        break
        except FileNotFoundError:
            flash("Data file not found.", "danger")
            return redirect(url_for('forget_password'))

        if user_found:
            # Redirect to Set Password page with the USN as a query parameter
            return redirect(url_for('set_new_password', usn=usn))
        else:
            flash("USN not found. Please try again.", "warning")
            return redirect(url_for('forget_password'))

    return render_template('forget_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        usn = request.form.get('usn', '').strip()

        # Validate USN format
        usn_pattern = r'^[3]{1}[a-zA-Z0-9]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{3}$'
        if not re.match(usn_pattern, usn):
            flash("Invalid USN format.", "danger")
            return redirect(url_for('reset_password'))

        # Check if USN exists in your database or CSV
        user_found = False
        try:
            with open(CSV_FILE, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['USN'] == usn:
                        user_found = True
                        break
        except FileNotFoundError:
            flash("Data file not found.", "danger")
            return redirect(url_for('reset_password'))

        if user_found:
            # Redirect to the Set Password page with USN as a query parameter
            return redirect(url_for('set_new_password', usn=usn))
        else:
            flash("USN not found. Please try again.", "warning")
            return redirect(url_for('reset_password'))
    
    return render_template('reset_password.html')

@app.route('/set_new_password', methods=['GET', 'POST'])
def set_new_password():
    usn = request.args.get('usn')

    if not usn or not re.match(r'^[3]{1}[a-zA-Z0-9]{2}[0-9]{2}[a-zA-Z]{2}[0-9]{3}$', usn):
        flash("Invalid or missing USN. Please try again.", "danger")
        return redirect(url_for('reset_password'))

    if request.method == 'GET':
        return render_template('set_password.html', usn=usn)

    elif request.method == 'POST':
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Password validation logic
        if new_password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return render_template('set_password.html', usn=usn)

        password_pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
        if not re.match(password_pattern, new_password):
            flash("Password does not meet the required criteria.", "danger")
            return render_template('set_password.html', usn=usn)

        # Update password in CSV or database
        try:
            updated_rows = []
            user_found = False

            with open(CSV_FILE, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['USN'] == usn:
                        row['Password'] = generate_password_hash(new_password, method='pbkdf2:sha256')
                        user_found = True
                    updated_rows.append(row)

            if user_found:
                with open(CSV_FILE, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['USN', 'Password'])
                    writer.writeheader()
                    writer.writerows(updated_rows)
                flash("Password reset successful! Please log in.", "success")
                return redirect(url_for('login'))
            else:
                flash("USN not found.", "danger")
                return redirect(url_for('reset_password'))

        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('reset_password'))

def teacherdashboard(teachernumber):
    ttf, teachers_total_positive_feedbacks, teachers_total_negative_feedbacks, teachers_total_neutral_feedbacks, teachers_li = get_feedback_counts()
    ttp, ttn, ttneu, tcp, tcn, tcneu, tep, ten, teneu, tlwp, tlwn, tlwneu, tlfp, tlfn, tlfneu, tecp, tecn, tecneu = teachers_li

    ttp = int(round(ttp/ttf *100));ttn = int(round(ttn/ttf *100));ttneu = int(round(ttneu/ttf *100))
    tcp = int(round(tcp / ttf * 100));tcn = int(round(tcn/ttf *100));tcneu = int(round(tcneu/ttf *100))
    tep = int(round(tep / ttf * 100));ten = int(round(ten/ttf *100));teneu = int(round(teneu/ttf *100))
    tlwp = int(round(tlwp / ttf * 100));tlwn = int(round(tlwn/ttf *100));tlwneu = int(round(tlwneu/ttf *100))
    tlfp = int(round(tlfp / ttf * 100));tlfn = int(round(tlfn/ttf *100));tlfneu = int(round(tlfneu/ttf *100))
    tecp = int(round(tecp / ttf * 100));tecn = int(round(tecn/ttf *100));tecneu = int(round(tecneu/ttf *100))

    if teachernumber == 1:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=ttp, ttn=ttn, ttneu=ttneu)
    elif teachernumber == 2:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=tcp, ttn=tcn, ttneu=tcneu)
    elif teachernumber == 3:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=tep, ttn=ten, ttneu=teneu)
    elif teachernumber == 4:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=tlwp, ttn=tlwn, ttneu=tlwneu)
    elif teachernumber == 5:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=tlfp, ttn=tlfn, ttneu=tlfneu)
    else:
        return render_template('teacherdashboard.html',ttf=ttf,ttp=tecp, ttn=tecn, ttneu=tecneu)

# @app.route('/login', methods=['GET'])
# def login():
#    return render_template('login.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Get the data from the form
    teaching = request.form['teaching']
    placements = request.form['placements']
    collaboration_with_companies = request.form['collaboration_with_companies']
    infrastructure = request.form['infrastructure']
    hostel = request.form['hostel']
    library = request.form['library']

    teacher1 = request.form['teacher1']
    teacher2 = request.form['teacher2']
    teacher3 = request.form['teacher3']
    teacher4 = request.form['teacher4']
    teacher5 = request.form['teacher5']
    teacher6 = request.form['teacher6']

    # Optionally, predict scores using your model
    model = pickle.load(open('SVM classifier.pkl ', 'rb'))
    teachingscore = model.predict(pd.array([teaching]))
    placementsscore = model.predict(pd.array([placements]))
    collaboration_with_companiesscore = model.predict(pd.array([collaboration_with_companies]))
    infrastructurescore = model.predict(pd.array([infrastructure]))
    hostelscore = model.predict(pd.array([hostel]))
    libraryscore = model.predict(pd.array([library]))

    teacher1score = model.predict(pd.array([teacher1]))
    teacher2score = model.predict(pd.array([teacher2]))
    teacher3score = model.predict(pd.array([teacher3]))
    teacher4score = model.predict(pd.array([teacher4]))
    teacher5score = model.predict(pd.array([teacher5]))
    teacher6score = model.predict(pd.array([teacher6]))

    # Get current time
    time = datetime.now().strftime("%m/%d/%Y (%H:%M:%S)")

    # Store feedback data in CSV files
    # Write to department feedback CSV
    write_to_csv_departments(time, teachingscore[0], teaching, placementsscore[0], placements,
                             collaboration_with_companiesscore[0], collaboration_with_companies, infrastructurescore[0], infrastructure,
                             hostelscore[0], hostel, libraryscore[0], library)

    # Write to teacher feedback CSV
    write_to_csv_teachers(teacher1, teacher1score[0], teacher2, teacher2score[0], teacher3, teacher3score[0],
                          teacher4, teacher4score[0], teacher5, teacher5score[0], teacher6, teacher6score[0])

    return render_template('thankyoupage.html')

def write_to_csv_departments(time, teachingscore, teaching, placementsscore, placements,
                              collaboration_with_companiesscore, collaboration_with_companies,
                              infrastructurescore, infrastructure, hostelscore, hostel,
                              libraryscore, library):
    with open('dataset/database.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the feedback data
        writer.writerow([time, teachingscore, teaching, placementsscore, placements,
                              collaboration_with_companiesscore, collaboration_with_companies,
                              infrastructurescore, infrastructure, hostelscore, hostel,
                              libraryscore, library])

def write_to_csv_teachers(teacher1, teacher1score, teacher2, teacher2score, teacher3, teacher3score,
                           teacher4, teacher4score, teacher5, teacher5score, teacher6, teacher6score):
    with open('dataset/teacherdb.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write a row with the teacher feedback data
        writer.writerow([teacher1, teacher1score, teacher2, teacher2score, teacher3, teacher3score,
                         teacher4, teacher4score, teacher5, teacher5score, teacher6, teacher6score])
        
@app.route('/public_department_dashboard')
def public_department_dashboard():
    try:
        total_feedbacks, total_positive_feedbacks, total_negative_feedbacks, total_neutral_feedbacks, li = get_counts()
        teachers_total_feedbacks, teachers_total_positive_feedbacks, teachers_total_negative_feedbacks, teachers_total_neutral_feedbacks, teachers_li = get_feedback_counts()

        # Handle unexpected data lengths safely
        tp, tn, tneu, cp, cn, cneu, ep, en, eneu, lwp, lwn, lwneu, lfp, lfn, lfneu, ecp, ecn, ecneu = li[:18] + [None]*(18 - len(li))

        # Extract teacher data
        ttp, ttn, ttneu, tcp, tcn, tcneu, tep, ten, teneu, tlwp, tlwn, tlwneu, tlfp, tlfn, tlfneu, tecp, tecn, tecneu = teachers_li

        return render_template('public_department_dashboard.html',
                               tf=total_feedbacks, tpf=total_positive_feedbacks, tnegf=total_negative_feedbacks,
                               tneuf=total_neutral_feedbacks, tp=tp, tn=tn, tneu=tneu, cp=cp, cn=cn, cneu=cneu,
                               ep=ep, en=en, eneu=eneu, lwp=lwp, lwn=lwn, lwneu=lwneu, lfp=lfp, lfn=lfn, lfneu=lfneu,
                               ecp=ecp, ecn=ecn, ecneu=ecneu,
                               ttf=teachers_total_feedbacks, ttpf=teachers_total_positive_feedbacks,
                               ttnegf=teachers_total_negative_feedbacks, ttneuf=teachers_total_neutral_feedbacks,
                               ttp=ttp, ttn=ttn, ttneu=ttneu, tcp=tcp, tcn=tcn, tcneu=tcneu, tep=tep, ten=ten,
                               teneu=teneu, tlwp=tlwp, tlwn=tlwn, tlwneu=tlwneu, tlfp=tlfp, tlfn=tlfn, tlfneu=tlfneu,
                               tecp=tecp, tecn=tecn, tecneu=tecneu)
    except Exception as e:
        flash(f"Error loading public department dashboard: {str(e)}")
        return redirect('/')
    
@app.route('/admin')
def root():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        total_feedbacks, total_positive_feedbacks, total_negative_feedbacks, total_neutral_feedbacks, li = get_counts()
        tp,tn,tneu,cp,cn,cneu,ep,en,eneu,lwp,lwn,lwneu,lfp,lfn,lfneu,ecp,ecn,ecneu = li
        teachers_total_feedbacks, teachers_total_positive_feedbacks, teachers_total_negative_feedbacks, teachers_total_neutral_feedbacks, teachers_li = get_feedback_counts()
        ttp, ttn, ttneu, tcp, tcn, tcneu, tep, ten, teneu, tlwp, tlwn, tlwneu, tlfp, tlfn, tlfneu, tecp, tecn, tecneu = teachers_li

        return render_template('admin.html',tf = total_feedbacks,tpf = total_positive_feedbacks,tnegf = total_negative_feedbacks, tneuf= total_neutral_feedbacks,
                               tp=tp,tn=tn,tneu=tneu,cp=cp,cn=cn,cneu=cneu,ep=ep,en=en,eneu=eneu,
                               lwp=lwp,lwn=lwn,lwneu=lwneu,lfp=lfp,lfn=lfn,lfneu=lfneu,ecp=ecp,
                               ecn=ecn,ecneu=ecneu,
                               ttf = teachers_total_feedbacks, ttpf = teachers_total_positive_feedbacks, ttnegf = teachers_total_negative_feedbacks,
                               ttneuf = teachers_total_neutral_feedbacks,ttp = ttp, ttn = ttn, ttneu = ttneu, tcp = tcp, tcn = tcn,
                               tcneu = tcneu, tep = tep, ten = ten, teneu = teneu,tlwp = tlwp, tlwn = tlwn,
                               tlwneu = tlwneu, tlfp = tlfp, tlfn = tlfn, tlfneu = tlfneu, tecp = tecp,tecn = tecn, tecneu = tecneu
                               )


@app.route("/hoddashboard")
def hoddashboard():
    if not session.get('logged_in'):
        return render_template('login.html')
    else :
        teachers_total_feedbacks, teachers_total_positive_feedbacks, teachers_total_negative_feedbacks, teachers_total_neutral_feedbacks, teachers_li = get_feedback_counts()
        ttp, ttn, ttneu, tcp, tcn, tcneu, tep, ten, teneu, tlwp, tlwn, tlwneu, tlfp, tlfn, tlfneu, tecp, tecn, tecneu = teachers_li
        return render_template('hoddashboard.html',
                               ttf=teachers_total_feedbacks, ttpf=teachers_total_positive_feedbacks,
                               ttnegf=teachers_total_negative_feedbacks,
                               ttneuf=teachers_total_neutral_feedbacks, ttp=ttp, ttn=ttn, ttneu=ttneu, tcp=tcp, tcn=tcn,
                               tcneu=tcneu, tep=tep, ten=ten, teneu=teneu, tlwp=tlwp, tlwn=tlwn,
                               tlwneu=tlwneu, tlfp=tlfp, tlfn=tlfn, tlfneu=tlfneu, tecp=tecp, tecn=tecn, tecneu=tecneu
                               )


@app.route("/displayteacherfeedbacks")
def displayteacherfeedbacks():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        df1 = pd.read_csv('dataset\teacher_feedback.csv')
        return render_template('teacherfeedbacks.html', tables=[df1.to_html(classes='data', header="true")])


@app.route("/display")
def display():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        try:
            # Check if the file exists
            import os
            file_path = 'dataset/database.csv'
            if not os.path.exists(file_path):
                return "Error: The dataset file does not exist."

            # Load the CSV file
            df = pd.read_csv(file_path)

            # # Debug: Print DataFrame to ensure it loaded correctly
            # print("Initial DataFrame:")
            # print(df)

            # Check for empty DataFrame
            if df.empty:
                return "Error: The dataset file is empty or improperly formatted."

            # List of aspects to extract feedback from
            aspects = [
                "teaching", "placements", "collaboration_with_companies",
                "infrastructure", "hostel", "library", "sports", "events"
            ]

            # Reshape the data: Convert each feedback into a separate row
            feedback_data = []
            for _, row in df.iterrows():
                for aspect in aspects:
                    if aspect in df.columns:  # Check if the aspect column exists
                        feedback_data.append({
                            "timestamp": row.get("Timestamp", ""),  # Default to empty if missing
                            "feedback": row.get(aspect, ""),  # Default to empty if missing
                            "type_of_aspect": aspect,
                            "priority": row.get(f"{aspect}score", 0)  # Default score if missing
                        })

            # Debug: Print reshaped feedback data
            # print("Reshaped Feedback Data:")
            # print(feedback_data)

            # Sort the feedback data by priority (High -> Neutral -> Low)
            feedback_data = sorted(feedback_data, key=lambda x: x["priority"])

            # # Debug: Print sorted feedback data
            # print("Sorted Feedback Data:")
            # print(feedback_data)

            # Pass feedback data to the template
            return render_template('feedbacks.html', feedbacks=feedback_data)

        except Exception as e:
            return f"An error occurred: {str(e)}"
    
@app.route('/feedback_system')
def feedback_system():
    # Your logic for the feedback system
    return render_template('feedback_system.html')

app.secret_key = os.urandom(12)
app.run(port=5978, host='0.0.0.0', debug=True)