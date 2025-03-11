from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
from datetime import datetime, timedelta
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import functools
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meals.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize model variables (will be loaded on first use)
feature_extractor = None
model = None
device = None

def get_model():
    """Lazy loading of the model to save resources"""
    global feature_extractor, model, device
    
    if model is None:
        # Use nateraw/food" model from Hugging Face
        model_name = "nateraw/food"
        
        # Set device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load feature extractor and model
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
        
    return feature_extractor, model

# Enhanced database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    logs = db.relationship('FoodLog', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class FoodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    calories = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)

class PasswordReset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)
    user = db.relationship('User', backref='reset_tokens')

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def identify_food(file_path):
    """Identify food in an image using Hugging Face model"""
    try:
        # Load the image using PIL
        image = Image.open(file_path).convert('RGB')
        
        # Get model and feature extractor
        feature_extractor, model = get_model()
        
        # Prepare image for the model
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get the predicted class
        predicted_class_idx = logits.argmax(-1).item()
        
        # Get the confidence score (using softmax to convert logits to probabilities)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0, predicted_class_idx].item() * 100
        
        # Get the predicted label
        predicted_label = model.config.id2label[predicted_class_idx]
        food_name = predicted_label.replace('_', ' ')
        
        return {
            'food': food_name,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error identifying food: {e}")
        # For debugging, return the error message
        return {'food': f'Error: {str(e)}', 'confidence': 0.0}

# Jinja filter for nice date formatting
@app.template_filter('format_date')
def format_date(value):
    return value.strftime('%B %d, %Y at %I:%M %p')

# Routes
@app.route('/')
def index():
    logs = []
    if 'user_id' in session:
        logs = FoodLog.query.filter_by(user_id=session['user_id']).order_by(FoodLog.date.desc()).all()
    return render_template('index.html', logs=logs)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            # Set session
            session['user_id'] = user.id
            session['username'] = user.username
            
            if remember:
                session.permanent = True
            
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            flash(f'Welcome back, {username}!', 'success')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate form data
        if not username or not password or not email:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
            
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
            
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Identify the food
        result = identify_food(file_path)
        
        # Save to database
        log = FoodLog(
            food=result['food'],
            confidence=result['confidence'],
            image_path=f"uploads/{unique_filename}",
            user_id=session['user_id']
        )
        db.session.add(log)
        db.session.commit()
        
        flash(f"Identified as {result['food']} with {result['confidence']:.2f}% confidence", 'success')
    else:
        flash('File type not allowed. Please upload a JPG, JPEG or PNG image.', 'error')
    
    return redirect(url_for('index'))

@app.route('/delete/<int:log_id>')
@login_required
def delete_log(log_id):
    log = FoodLog.query.get_or_404(log_id)
    
    # Check if this log belongs to the logged in user
    if log.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('index'))
    
    # Delete the image file
    try:
        os.remove(os.path.join('static', log.image_path))
    except Exception as e:
        print(f"Error removing file: {e}")
        # Continue even if file removal fails
    
    # Delete the log
    db.session.delete(log)
    db.session.commit()
    
    flash('Log deleted successfully', 'success')
    return redirect(url_for('index'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Email is required', 'error')
            return redirect(url_for('forgot_password'))
        
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Don't reveal if email exists or not for security
            flash('If your email is registered, you will receive a password reset link', 'info')
            return redirect(url_for('login'))
        
        # Generate a unique token
        token = str(uuid.uuid4())
        
        # Create a password reset token that expires in 24 hours
        reset_token = PasswordReset(
            user_id=user.id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
        db.session.add(reset_token)
        db.session.commit()
        
        # In a real app, send an email here
        # For now, just flash the reset link (for development purposes)
        reset_url = url_for('reset_password', token=token, _external=True)
        flash(f'Password reset link: {reset_url}', 'info')
        flash('In a production environment, this link would be emailed to you.', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Find the token
    reset = PasswordReset.query.filter_by(token=token, used=False).first()
    
    if not reset or reset.expires_at < datetime.utcnow():
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not password or not confirm_password:
            flash('All fields are required', 'error')
            return redirect(url_for('reset_password', token=token))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('reset_password', token=token))
        
        # Update the user's password
        user = User.query.get(reset.user_id)
        user.set_password(password)
        
        # Mark token as used
        reset.used = True
        
        db.session.commit()
        
        flash('Password has been reset successfully. Please login with your new password.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.template_filter('format_confidence')
def format_confidence(value):
    return f"{value:.2f}"  # For 2 decimal places
    # OR
    # return f"{int(value)}"  # For whole numbers

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        # Update profile information
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if email and email != user.email:
            # Check if email already exists
            if User.query.filter_by(email=email).first() and User.query.filter_by(email=email).first().id != user.id:
                flash('Email already in use', 'error')
                return redirect(url_for('profile'))
            
            user.email = email
            flash('Email updated successfully', 'success')
        
        # Change password if requested
        if current_password and new_password:
            if not user.check_password(current_password):
                flash('Current password is incorrect', 'error')
                return redirect(url_for('profile'))
            
            if new_password != confirm_password:
                flash('New passwords do not match', 'error')
                return redirect(url_for('profile'))
            
            user.set_password(new_password)
            flash('Password updated successfully', 'success')
        
        db.session.commit()
        return redirect(url_for('profile'))
    
    # Get user stats
    stats = {
        'total_logs': FoodLog.query.filter_by(user_id=user.id).count(),
        'member_since': user.created_at.strftime('%B %d, %Y'),
        'last_login': user.last_login.strftime('%B %d, %Y at %I:%M %p') if user.last_login else 'Never'
    }
    
    return render_template('profile.html', user=user, stats=stats)

@app.route('/edit-log/<int:log_id>', methods=['GET', 'POST'])
@login_required
def edit_log(log_id):
    log = FoodLog.query.get_or_404(log_id)
    
    # Check if this log belongs to the logged in user
    if log.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Update log information
        log.food = request.form.get('food', log.food)
        log.calories = request.form.get('calories', log.calories)
        log.notes = request.form.get('notes', log.notes)
        
        db.session.commit()
        flash('Log updated successfully', 'success')
        return redirect(url_for('index'))
    
    return render_template('edit_log.html', log=log)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

with app.app_context():
    # Create database tables
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)