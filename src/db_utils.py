import argparse
import sqlite3

# Utility functions


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


def insert_many(conn, data, table, count):
    """
    Insert multiple rows into table
    :param conn: the Connection object
    :return:
    """

    values = (1,) * count
    values = str(values).replace("1", "?")

    cur = conn.cursor()
    cur.executemany(f"INSERT INTO {table} VALUES {values}", data)


def retrieve_query(conn, query):
    """
    Execute SQL query and returns output
    :param conn: the Connection object
    :return:
    """
    try:
        cur = conn.cursor()
        cur.execute(query)
    except sqlite3.Error as e:
        print(e)

    rows = cur.fetchall()

    return rows


def execute_sql(conn, sql):
    """ Execute multiple SQL statements without return
    :param conn: Connection object
    :param sql: a string of SQL statements
    :return:
    """
    try:
        c = conn.cursor()
        c.executescript(sql)
    except sqlite3.Error as e:
        print(e)
