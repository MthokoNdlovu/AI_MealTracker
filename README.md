# AI_MealTracker
# AI Meal Tracker

A Flask web application that uses AI to identify food from photos and track your meals.

## Overview

AI Meal Tracker is a modern web application that allows users to:
- Upload images of their food
- Get AI-powered identification of food items
- Keep a log of meals with timestamps
- Track meals over time with a visual history

The application uses the Hugging Face `nateraw/food` model to identify food items from images and provides an intuitive user interface for managing your meal history.

## Features

- **AI Food Recognition**: Upload photos of your meals and let our AI identify what you're eating
- **User Authentication**: Secure registration and login system
- **Meal History**: View your past meals with images, dates, and confidence scores
- **Responsive Design**: Works on desktop and mobile devices
- **Password Management**: Secure password reset functionality

## Installation

### Prerequisites

- Python 3.7 to 3.9
- Flask
- SQLAlchemy
- PyTorch
- Transformers (Hugging Face)
- Pillow
- Werkzeug

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/MthokoNdlovu/AI_MealTracker.git
   cd ai-meal-tracker
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```
   python create_db.py
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Visit `http://localhost:5000` in your browser

## Project Structure

```
ai-meal-tracker/
├── app.py               # Main application file
├── create_db.py         # Database initialization script
├── migrations.py        # Database migration script
├── static/              # Static files (CSS, JS, uploads)
│   ├── css/
│   ├── js/
│   └── uploads/         # User uploaded food images
├── templates/           # HTML templates
│   ├── index.html       # Home page
│   ├── login.html       # Login page
│   ├── register.html    # Registration page
│
└── requirements.txt     # Project dependencies
```

## AI Model

The application uses the Hugging Face `nateraw/food` model for food classification. The model is loaded on first use to save resources. Food images are processed to extract features, which the model uses to identify the food items with a confidence score.

## Database Models

- **User**: Stores user account information (username, email, password)
- **FoodLog**: Stores food entries (food name, image path, confidence score, date)
- **PasswordReset**: Manages password reset tokens

## Development

### Create Database

```
python create_db.py
```

### Database Migrations

If you make changes to the database models, run:

```
flask db migrate -m "Description of changes"
flask db upgrade
```

## Future Enhancements

- Add nutritional information for identified foods
- Implement food search functionality
- Add meal planning features
- Implement social sharing of meals

## Security Notes

- User passwords are securely hashed using Werkzeug's security functions
- Password reset tokens expire after 24 hours
- File uploads are validated and sanitized

## Acknowledgements

- Food classification model: [nateraw/food](https://huggingface.co/nateraw/food)
- Icons: [Font Awesome](https://fontawesome.com/)
- Flask framework and extensions
