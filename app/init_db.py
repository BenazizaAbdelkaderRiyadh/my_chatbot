import sqlite3

# Connect to the database file (it will be created if it doesn't exist)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create the 'authentication' table if it doesn't already exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS authentication (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Fname TEXT NOT NULL,
    Lname TEXT NOT NULL,
    Email TEXT NOT NULL UNIQUE,
    Password TEXT NOT NULL
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("✅ 'database.db' is ready with the 'authentication' table.")