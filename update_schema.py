import sqlite3

# Path to your database file - adjust if needed
DB_PATH = 'instance/meals.db'

def add_missing_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if calories column exists
    cursor.execute("PRAGMA table_info(food_log)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'calories' not in columns:
        print("Adding calories column...")
        cursor.execute("ALTER TABLE food_log ADD COLUMN calories INTEGER")
    
    if 'notes' not in columns:
        print("Adding notes column...")
        cursor.execute("ALTER TABLE food_log ADD COLUMN notes TEXT")
    
    conn.commit()
    conn.close()
    print("Schema update complete!")

if __name__ == "__main__":
    add_missing_columns()