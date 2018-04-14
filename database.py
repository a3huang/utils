from configparser import ConfigParser
from sqlalchemy import create_engine
import pandas as pd
import boto3
import os
import pyathena
import re

config = ConfigParser()
config.read(os.path.join(os.path.expanduser('~'), 'config.ini'))


class Database(object):
    '''
    ex) with Database('mysql') as db:
            db.execute("select * from table")
            df = db.get_table()
    '''
    def __init__(self, database):
        self.database = database.lower()
        self.engine_dict = {'mysql': 'mysql+pymysql',
                            'postgres': 'postgresql+psycopg2',
                            'redshfit': 'redshift+psycopg2'}

        self.host = config.get(database, 'host')
        self.port = int(config.get(database, 'port'))
        self.user = config.get(database, 'user')
        self.password = config.get(database, 'password')
        self.db = config.get(database, 'db')
        self.driver = self.engine_dict[database]

        connection_params = (self.driver, self.user, self.password,
                             self.host, self.port, self.db)
        connection_string = '%s://%s:%s@%s:%s/%s' % connection_params
        self.engine = create_engine(connection_string)

    def __enter__(self):
        self.conn = self.engine.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.conn.close()

    def execute(self, query):
        self.table = pd.read_sql_query(query, self.conn)

    def get_table(self):
        return self.table
