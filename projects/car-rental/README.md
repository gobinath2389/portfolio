
python3 -m venv ~/car-venv
source ~/car-venv/bin/activate

install local posgres db
brew install postgresql@17
brew services start postgresql@17
install pgvector extenstion
brew install pgvector
brew services restart postgresql@17

Create a table car under car_inventory schema

-- car_inventory.cars definition

-- Drop table

-- DROP TABLE car_inventory.cars;

CREATE TABLE car_inventory.cars (
car_id serial4 NOT NULL,
car_number varchar(10) NOT NULL,
manufacturer varchar(50) NOT NULL,
model varchar(50) NOT NULL,
"type" varchar(20) NOT NULL,
num_seats int2 NOT NULL,
color varchar(20) NOT NULL,
daily_rate numeric(10, 2) NOT NULL,
embedding car_inventory.vector NULL,
CONSTRAINT cars_car_number_key UNIQUE (car_number),
CONSTRAINT cars_daily_rate_check CHECK ((daily_rate > (0)::numeric)),
CONSTRAINT cars_pkey PRIMARY KEY (car_id)
);
CREATE INDEX cars_embedding_idx ON car_inventory.cars USING hnsw (embedding car_inventory.vector_l2_ops);

-- Permissions

ALTER TABLE car_inventory.cars OWNER TO postgres;
GRANT ALL ON TABLE car_inventory.cars TO postgres;

uvicorn app.api.car:app --reload
streamlit run ui/streamlit_app.py --server.port 8501

Certainly! Here’s a well-formatted version of your README.md content, with clear sections, code blocks, and explanations for each step.

# Car Inventory Project Setup

This guide will help you set up the development environment, install dependencies, configure the PostgreSQL database, and run the application.

## 1. Create a Python Virtual Environment

```bash
python3 -m venv ~/car-venv
source ~/car-venv/bin/activate
```


## 2. Install and Configure Local PostgreSQL Database

### Install PostgreSQL 17

```bash
brew install postgresql@17
brew services start postgresql@17
```


### Install `pgvector` Extension

```bash
brew install pgvector
brew services restart postgresql@17
```


## 3. Create Database Schema and Table

### Create `car` Table under `car_inventory` Schema

```sql
-- car_inventory.cars definition

-- Drop table if it exists
-- DROP TABLE car_inventory.cars;

CREATE TABLE car_inventory.cars (
    car_id serial4 NOT NULL,
    car_number varchar(10) NOT NULL,
    manufacturer varchar(50) NOT NULL,
    model varchar(50) NOT NULL,
    "type" varchar(20) NOT NULL,
    num_seats int2 NOT NULL,
    color varchar(20) NOT NULL,
    daily_rate numeric(10, 2) NOT NULL,
    embedding car_inventory.vector NULL,
    CONSTRAINT cars_car_number_key UNIQUE (car_number),
    CONSTRAINT cars_daily_rate_check CHECK ((daily_rate > (0)::numeric)),
    CONSTRAINT cars_pkey PRIMARY KEY (car_id)
);

CREATE INDEX cars_embedding_idx ON car_inventory.cars USING hnsw (embedding car_inventory.vector_l2_ops);
```


### Set Permissions

```sql
ALTER TABLE car_inventory.cars OWNER TO postgres;
GRANT ALL ON TABLE car_inventory.cars TO postgres;
```


## 4. Run the Application

### Start the FastAPI Backend

```bash
uvicorn app.api.car:app --reload
```


### Start the Streamlit Frontend

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

**Current date:** Sunday, July 13, 2025, 5:27 PM EDT

Feel free to copy and use this as your README.md! If you need further formatting or additional sections, just let me know.

