from configparser import ConfigParser
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
import os

config = ConfigParser()
config.read(os.path.join(os.path.expanduser('~'), 'config.ini'))


class Database(object):
    def __init__(self, database):
        self.database = database.lower()
        engine_dict = config._sections[database.lower()]
        self.engine = create_engine(URL(**engine_dict))

    def __enter__(self):
        self.conn = self.engine.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.conn.close()

    def execute(self, query):
        self.table = pd.read_sql_query(query, self.conn)

    def get_table(self):
        return self.table
