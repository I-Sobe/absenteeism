# confirm that database nae doesn't exist
DROP DATABASE IF EXISTS predicted_outputs;
# create the database
CREATE DATABASE IF NOT EXISTS predicted_outputs;
# make the database available for use
USE predicted_outputs;
# creating a database
DROP TABLE IF EXISTS predicted_outputs;
#rule states that you must designate the names of all columns, it's data types and put a data constraints like NOT NULL
CREATE TABLE predicted_outputs 
(
	reason_1 BIT NOT NULL,
    reason_2 BIT NOT NULL,
    reason_3 BIT NOT NULL,
    reason_4 BIT NOT NULL,
    month INT NOT NULL,
    transportation_expense INT NOT NULL,
    age INT NOT NULL,
    body_mass_index INT NOT NULL,
    education BIT NOT NULL,
    children INT NOT NULL,
    pets INT NOT NULL,
    probability FLOAT NOT NULL,
    prediction BIT NOT NULL
);

SELECT
	*
FROM
	predicted_outputs;