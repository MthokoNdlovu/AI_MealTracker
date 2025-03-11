from app import app, db

with app.app_context():
    print("Creating new database with current models...")
    db.create_all()
    print("Database reset complete!")