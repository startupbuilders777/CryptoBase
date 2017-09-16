#!/bin/sh
set -ex

cockroach sql --insecure -e 'DROP DATABASE IF EXISTS example_flask_sqlalchemy'
cockroach sql --insecure -e 'CREATE DATABASE example_flask_sqlalchemy'
cockroach sql --insecure -e 'GRANT ALL ON DATABASE example_flask_sqlalchemy TO example'

python -c 'import cryptobase; cryptobase.db.create_all()'
