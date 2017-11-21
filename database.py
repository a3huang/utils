import pandas as pd

import boto3
import ConfigParser
import MySQLdb
import os
import psycopg2
import pyathena
import re

config = ConfigParser.ConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.ini'))

def remove_metadata_from_bucket(aws_access_key_id, aws_secret_access_key, bucket,
        region_name):
    pattern = re.compile("[0-F]{8}-[0-F]{4}-[0-F]{4}-[0-F]{4}-[0-F]{12}", re.I)

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key, region_name=region_name)
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

class Database(object):
    '''
    ex) with Database('mysql') as db:
            db.execute("select * from table")
            df = db.get_table()

    ex) with Database('mysql') as db:
            cur = db.get_cursor()
            cur.execute("QUERY_STRING")
            rows = cur.fetchmany(1000)
    '''

    def __init__(self, database):
        self.database = database.lower()

        if self.database == 'athena':
            self.aws_access_key_id = config.get('aws', 'aws_access_key_id')
            self.aws_secret_access_key = config.get('aws', 'aws_secret_access_key')
            self.s3_staging_bucket = config.get('aws', 's3_staging_bucket')
            self.region_name = config.get('aws', 'region_name')

            conn = pyathena.connect(aws_access_key_id=self.aws_access_key_id,
                                    aws_secret_access_key=self.aws_secret_access_key,
                                    s3_staging_dir='s3://' + self.s3_staging_bucket,
                                    region_name=self.region_name)

        else:
            host = config.get(database, 'host')
            port = int(config.get(database, 'port'))
            user = config.get(database, 'user')
            pswd = config.get(database, 'password')
            db = config.get(database, 'db')

            if self.database == 'mysql':
                conn = MySQLdb.connect(host=host, port=port, user=user, passwd=pswd,
                                       db=db)

            elif self.database in ['postgres', 'redshift']:
                conn = psycopg2.connect(host=host, port=port, user=user, password=pswd,
                                        dbname=db)

        self.conn = conn

    def __enter__(self):
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.database == 'athena':
            remove_metadata_from_bucket(self.s3_staging_bucket, self.aws_access_key_id,
                self.aws_secret_access_key, self.region_name)

        self.cur.close()
        self.conn.close()

    def execute(self, query):
        self.cur.execute(query)

        if self.database in ['postgres', 'redshift']:
            columns = [i.name for i in self.cur.description]

        elif self.database in ['mysql', 'athena']:
            columns = [i[0] for i in self.cur.description]

        self.table = pd.DataFrame(list(self.cur.fetchall()), columns=columns)

    def get_cursor(self):
        return self.cur

    def get_table(self):
        return self.table
