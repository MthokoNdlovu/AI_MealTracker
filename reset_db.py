from app import app, db
import os

def reset_database():
    # Find the database file path
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    
    if db_uri.startswith('sqlite:///'):
        # Extract path for SQLite file
        db_path = db_uri.replace('sqlite:///', '')
        
        # For relative paths in Flask
        if not os.path.isabs(db_path):
            # Check if it's in the instance folder
            instance_path = os.path.join(app.instance_path, db_path)
            if os.path.exists(instance_path):
                db_path = instance_path
            else:
                # Try in the current directory
                current_dir_path = os.path.join(os.getcwd(), db_path)
                if os.path.exists(current_dir_path):
                    db_path = current_dir_path
        
        # Delete the file if it exists
        if os.path.exists(db_path):
            print(f"Removing database file: {db_path}")
            os.remove(db_path)
        else:
            print(f"Database file not found at {db_path}")
    
    # Create all tables from current models
    with app.app_context():
        print("Creating new database with current models...")
        db.create_all()
        print("Database reset complete!")

if __name__ == "__main__":
    reset_database()