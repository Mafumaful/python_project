import sqlite3

# Connect to the database
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL
    )
''')

# Insert data
cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', ('Alice', 25))
conn.commit()

# Query data
cursor.execute('SELECT * FROM users')
rows = cursor.fetchall()
for row in rows:
    print(row)

# Update data
cursor.execute('UPDATE users SET age = ? WHERE name = ?', (26, 'Alice'))
conn.commit()

# Delete data
cursor.execute('DELETE FROM users WHERE name = ?', ('Alice',))
conn.commit()

# Close connection
conn.close()
