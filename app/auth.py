import bcrypt
from flask import Blueprint, render_template, request, redirect, session, url_for
from .utils import query_db, execute_db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Retrieve user from the database by email
        user = query_db("SELECT * FROM authentication WHERE email = ?", (email,), one=True)

        # Check if user exists and the password is correct
        if user and bcrypt.checkpw(password.encode('utf-8'), user[4].encode('utf-8')):
            # Store user info in the session
            session['user_ID'] = user[0]
            session['user_name'] = user[1]
            return redirect(url_for('main.home'))
            
        # If login fails, show an error message
        return render_template('login.html', msg="Invalid email or password.")
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fname = request.form['first_name']
        lname = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        cpassword = request.form['con-password']

        # Check if passwords match
        if password != cpassword:
            return render_template('register.html', msg="Passwords do not match.")

        # Check if the email is already registered
        if query_db("SELECT email FROM authentication WHERE email = ?", (email,), one=True):
            return render_template('register.html', msg="An account with this email already exists.")

        # Hash the password before storing it
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Insert the new user into the database
        execute_db(
            """
            INSERT INTO authentication (Fname, Lname, Email, Password)
            VALUES (?, ?, ?, ?)
            """, (fname, lname, email, hashed_pw.decode('utf-8'))
        )
        # Redirect to login page with a success message
        return render_template('login.html', msg="Registration successful. You can now log in.")
    
    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    # Clear the session to log the user out
    session.clear()
    return redirect(url_for('main.home'))