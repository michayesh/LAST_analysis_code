#!/bin/bash

SERVER="euclid"
DB="observatory_operation"
TABLE="operation_numbers"
USER="default"
PASSWORD="PassRoot"

clickhouse-client --host "$SERVER" --user "$USER" --password "$PASSWORD" --query "SELECT COUNT(*) FROM $DB.$TABLE"

echo "Connection successful"

