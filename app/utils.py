# utils.py
import sqlite3

def query_db(query, args=(), one=False):
    """
    Queries the database and returns results.
    `one=True` returns the first result, otherwise returns all.
    """
    with sqlite3.connect('database.db') as con:
        cur = con.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    """
    Executes a database command (INSERT, UPDATE, DELETE) and commits the change.
    """
    with sqlite3.connect('database.db') as con:
        cur = con.cursor()
        cur.execute(query, args)
        con.commit()