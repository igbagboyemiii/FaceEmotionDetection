import sqlite3

# Connect to (or create) database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Create table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    image_path TEXT NOT NULL,
    emotion TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("✅ Database created successfully and ready to use!")