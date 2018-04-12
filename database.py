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

        if self.database == 'athena':
            self.aws_access_key_id = config.get('aws', 'aws_access_key_id')
            self.aws_secret_access_key = config.get(
                'aws', 'aws_secret_access_key')
            self.s3_staging_bucket = config.get('aws', 's3_staging_bucket')
            self.region_name = config.get('aws', 'region_name')

            self.conn = pyathena.connect(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                s3_staging_dir='s3://' + self.s3_staging_bucket,
                region_name=self.region_name)

        else:
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
        if self.database != 'athena':
            self.conn = self.engine.connect()
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.database == 'athena':
            self.remove_metadata_from_bucket(self.s3_staging_bucket)

        self.cur.close()
        self.conn.close()

    def execute(self, query):
        self.cur.execute(query)

        if self.database in ['postgres', 'redshift']:
            columns = [i.name for i in self.cur.description]

        elif self.database in ['mysql', 'athena']:
            columns = [i[0] for i in self.cur.description]

        self.table = pd.DataFrame(list(self.cur.fetchall()), columns=columns)

    def get_table(self):
        return self.table

    def remove_metadata_from_bucket(self, bucket):
        string = "[0-F]{8}-[0-F]{4}-[0-F]{4}-[0-F]{4}-[0-F]{12}"
        pattern = re.compile(string, re.I)

        client = boto3.client('s3', aws_access_key_id=self.aws_access_key_id,
                              aws_secret_access_key=self.aws_secret_access_key,
                              region_name=self.region_name)
        paginator = client.get_paginator('list_objects')
        page_iterator = paginator.paginate(Bucket=bucket)

        pages = []
        for page in page_iterator:
            try:
                pages.append(page['Contents'])
            except:
                pages.append([])

        files = [item for sublist in pages for item in sublist]

        files_to_delete = {'Objects': []}

        for f in files:
            if pattern.search(f['Key']):
                files_to_delete['Objects'].append({'Key': f['Key']})

        client.delete_objects(Bucket=bucket, Delete=files_to_delete)
