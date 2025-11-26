import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import pandas as pd
import pytesseract
import cv2
import numpy as np
import re
import io
import random
import string
import os
import traceback
import logging
import sys

# ==========================================
# 1. LOGGING & CONFIGURATION
# ==========================================

# Setup Production Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[sys.stdout]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
CSV_FILE = "parcels_10000.csv"

# Global Cache
ROUTE_CACHE = {}

# Database Setup with Production Pooling
engine = None
SessionLocal = None

try:
    if not DATABASE_URL:
        logger.critical("DATABASE_URL is missing in .env file!")
    else:
        # pool_pre_ping=True is CRITICAL for production (auto-reconnects if DB closes connection)
        engine = create_engine(
            DATABASE_URL, 
            pool_size=10, 
            max_overflow=20, 
            pool_pre_ping=True 
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database engine initialized with connection pooling.")
except Exception as e:
    logger.critical(f"Database Connection Failed: {e}")
    if DEBUG_MODE:
        traceback.print_exc()

# ==========================================
# 2. MODELS
# ==========================================

class LoginRequest(BaseModel):
    username: str
    password: str

class NewUserRequest(BaseModel):
    email: str

class UpdateProfileRequest(BaseModel):
    user_id: int
    new_username: str
    new_password: str

class CityRequest(BaseModel):
    city_name: str

class DataPointRequest(BaseModel):
    source_city_id: int
    source_city_name: str
    destination_city_id: int
    destination_city_name: str
    parcel_type: int
    route_direction: int

class FindRouteRequest(BaseModel):
    source_city_id: int
    source_city_name: str | None = None 
    destination_city_id: int
    destination_city_name: str | None = None
    parcel_type: int

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_db():
    if not SessionLocal:
        raise HTTPException(status_code=500, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def refresh_route_cache():
    global ROUTE_CACHE
    temp_cache = {}
    
    logger.info("Refreshing route cache...")
    
    # 1. Load CSV (Fallback)
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            for _, row in df.iterrows():
                key = (int(row['source_city_id']), int(row['destination_city_id']), int(row['parcel_type']))
                temp_cache[key] = int(row['route_direction'])
            logger.info(f"Loaded {len(df)} rules from CSV.")
        except Exception as e:
            logger.error(f"CSV Load Error: {e}")

    # 2. Load DB (Primary)
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS datapoints (
                        id SERIAL PRIMARY KEY,
                        source_city_id INTEGER, source_city_name TEXT,
                        destination_city_id INTEGER, destination_city_name TEXT,
                        parcel_type INTEGER, route_direction INTEGER
                    )
                """))
                conn.commit()

                result = conn.execute(text("SELECT source_city_id, destination_city_id, parcel_type, route_direction FROM datapoints"))
                rows = result.fetchall()
                for row in rows:
                    key = (int(row[0]), int(row[1]), int(row[2]))
                    temp_cache[key] = int(row[3])
                logger.info(f"Loaded {len(rows)} rules from DB.")
        except Exception as e:
            logger.error(f"DB Cache Load Error: {e}", exc_info=True)

    ROUTE_CACHE = temp_cache
    logger.info(f"Total Active Rules in Cache: {len(ROUTE_CACHE)}")
    return len(ROUTE_CACHE)

def process_image_cv(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        custom_config = r'--oem 3 --psm 6' 
        text_out = pytesseract.image_to_string(thresh, config=custom_config)
        
        data = {"source_city": None, "source_id": None, "dest_city": None, "dest_id": None, "type": 0}
        clean_text = re.sub(r'[|/\[\]]', ' ', text_out)
        
        # Debug Log for OCR
        if DEBUG_MODE:
            logger.debug(f"OCR Extracted Text: {clean_text[:100]}...")

        # Regex Logic
        src_match = re.search(r'SOURCE(?!\s*_?ID)\s+([A-Z]+)', clean_text, re.IGNORECASE)
        data['source_city'] = src_match.group(1) if src_match else None
        
        src_id_match = re.search(r'SOURCE_ID\s+(\d+)', clean_text, re.IGNORECASE)
        if src_id_match:
            data['source_id'] = int(src_id_match.group(1))
        elif data['source_city']:
            fallback = re.search(rf"{re.escape(data['source_city'])}.*?(\d+)", clean_text, re.IGNORECASE)
            if fallback: data['source_id'] = int(fallback.group(1))

        dst_match = re.search(r'DESTINATION(?!\s*_?ID)\s+([A-Z]+)', clean_text, re.IGNORECASE)
        data['dest_city'] = dst_match.group(1) if dst_match else None

        dst_id_match = re.search(r'DESTINATION_ID\s+(\d+)', clean_text, re.IGNORECASE)
        if dst_id_match:
            data['dest_id'] = int(dst_id_match.group(1))
        elif data['dest_city']:
            fallback = re.search(rf"{re.escape(data['dest_city'])}.*?(\d+)", clean_text, re.IGNORECASE)
            if fallback: data['dest_id'] = int(fallback.group(1))

        type_match = re.search(r'PARCEL_TYPE\s+([0-9loO]+)', clean_text, re.IGNORECASE)
        if type_match:
            val = type_match.group(1)
            if val.lower() in ['lo', 'o', 'l']:
                data['type'] = 0
            else:
                digits = re.search(r'\d+', val)
                data['type'] = int(digits.group(0)) if digits else 0

        return data
    except Exception as e:
        logger.error(f"Image Processing Error: {e}", exc_info=True)
        raise e

def sync_sequences(db: Session):
    """Resets the Auto-Increment sequence to prevent duplicate key errors."""
    try:
        logger.info("Syncing Database Sequences...")
        db.execute(text("SELECT setval(pg_get_serial_sequence('datapoints', 'id'), COALESCE((SELECT MAX(id) + 1 FROM datapoints), 1), false);"))
        db.commit()
        logger.info("Sequences Synced Successfully.")
    except Exception as e:
        logger.warning(f"Could not sync sequences (Table might not exist yet): {e}")
        db.rollback()

def init_db_data(db: Session):
    """Handles Table creation and Seeding"""
    logger.info("Initializing Database Schema & Seeds...")
    try:
        # Create Tables
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS "admin-logins" (
                id SERIAL PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT
            );
            CREATE TABLE IF NOT EXISTS "employee-logins" (
                id SERIAL PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT
            );
            CREATE TABLE IF NOT EXISTS cities (
                unique_id INTEGER PRIMARY KEY, cityname TEXT
            );
            CREATE TABLE IF NOT EXISTS datapoints (
                id SERIAL PRIMARY KEY,
                source_city_id INTEGER, source_city_name TEXT,
                destination_city_id INTEGER, destination_city_name TEXT,
                parcel_type INTEGER, route_direction INTEGER
            );
        """))
        db.commit()

        # Seed Admin
        admin_chk = db.execute(text("SELECT * FROM \"admin-logins\"")).fetchone()
        if not admin_chk:
            logger.info("Seeding default admin account...")
            db.execute(text("INSERT INTO \"admin-logins\" (username, password, email) VALUES ('admin', 'admin123', 'admin@sys.com')"))
            db.commit()

        # Seed Cities
        city_count = db.execute(text("SELECT COUNT(*) FROM cities")).scalar()
        if city_count == 0 and os.path.exists(CSV_FILE):
            logger.info("Seeding cities from CSV...")
            df = pd.read_csv(CSV_FILE)
            src = df[['source_city_name', 'source_city_id']].rename(columns={'source_city_name': 'name', 'source_city_id': 'id'})
            dst = df[['destination_city_name', 'destination_city_id']].rename(columns={'destination_city_name': 'name', 'destination_city_id': 'id'})
            cities = pd.concat([src, dst]).drop_duplicates(subset=['id'])
            
            for _, row in cities.iterrows():
                clean_name = str(row['name']).replace("'", "''")
                db.execute(text(f"INSERT INTO cities (unique_id, cityname) VALUES ({row['id']}, '{clean_name}') ON CONFLICT DO NOTHING"))
            db.commit()
            
        sync_sequences(db)

    except Exception as seed_err:
        logger.error(f"Seeding Failed: {seed_err}", exc_info=True)

# ==========================================
# 4. LIFESPAN MANAGER
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    logger.info("Application Starting Up...")
    if engine:
        db = SessionLocal()
        try:
            init_db_data(db)
        except Exception as e:
            logger.error(f"Startup DB Error: {e}", exc_info=True)
        finally:
            db.close()
        
        refresh_route_cache()
    else:
        logger.warning("No Database Engine - Skipping DB Tasks")
    
    yield
    
    # --- SHUTDOWN ---
    logger.info("Application Shutting Down...")
    if engine:
        engine.dispose()

app = FastAPI(title="Smart Conveyor Belt System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In strict production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. ENDPOINTS
# ==========================================

@app.get("/health")
def health_check():
    """Simple endpoint to check if server is running"""
    return {"status": "ok", "database": "connected" if engine else "disconnected"}

@app.post("/login/{role}")
def login(role: str, creds: LoginRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    try:
        # Parameterized query for security
        res = db.execute(text(f"SELECT id, username FROM \"{table}\" WHERE username=:u AND password=:p"), 
                         {"u": creds.username, "p": creds.password}).fetchone()
        if res:
            logger.info(f"User {creds.username} logged in as {role}")
            return {"status": "success", "user_id": res[0], "username": res[1], "role": role}
        
        logger.warning(f"Failed login attempt for {creds.username} in {role}")
    except Exception as e:
        logger.error("Login DB Error", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/users/{role}")
def get_users(role: str, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    res = db.execute(text(f"SELECT id, username, email FROM \"{table}\"")).fetchall()
    return [{"id": r[0], "username": r[1], "email": r[2]} for r in res]

@app.post("/add-user/{role}")
def add_user(role: str, req: NewUserRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    username = 'user_' + ''.join(random.choices(string.digits, k=4))
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    try:
        db.execute(text(f"INSERT INTO \"{table}\" (username, password, email) VALUES (:u, :p, :e)"), 
                   {"u": username, "p": password, "e": req.email})
        db.commit()
        logger.info(f"Added new user {username} to {role}")
        return {"username": username, "password": password}
    except Exception as e:
        logger.error("Add User Error", exc_info=True)
        raise HTTPException(status_code=400, detail="Could not add user. Email or User might exist.")

@app.delete("/remove-user/{role}/{uid}")
def remove_user(role: str, uid: int, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    db.execute(text(f"DELETE FROM \"{table}\" WHERE id=:id"), {"id": uid})
    db.commit()
    return {"status": "deleted"}

@app.put("/edit-profile/{role}")
def edit_profile(role: str, req: UpdateProfileRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    check = db.execute(text(f"SELECT id FROM \"{table}\" WHERE username=:u AND id!=:id"), 
                       {"u": req.new_username, "id": req.user_id}).fetchone()
    if check:
        raise HTTPException(status_code=400, detail="Username taken")
    
    db.execute(text(f"UPDATE \"{table}\" SET username=:u, password=:p WHERE id=:id"), 
               {"u": req.new_username, "p": req.new_password, "id": req.user_id})
    db.commit()
    return {"status": "updated"}

@app.get("/cities")
def get_cities(db: Session = Depends(get_db)):
    res = db.execute(text("SELECT unique_id, cityname FROM cities ORDER BY unique_id")).fetchall()
    return [{"id": r[0], "name": r[1]} for r in res]

@app.post("/add-city")
def add_city(req: CityRequest, db: Session = Depends(get_db)):
    max_id = db.execute(text("SELECT MAX(unique_id) FROM cities")).scalar()
    new_id = (max_id if max_id else 5000) + 1
    try:
        db.execute(text("INSERT INTO cities (unique_id, cityname) VALUES (:id, :n)"), {"id": new_id, "n": req.city_name})
        db.commit()
        return {"status": "success", "new_id": new_id}
    except Exception as e:
        logger.error(f"Add City Error: {e}")
        raise HTTPException(status_code=400, detail="City likely exists")

@app.get("/datapoints")
def get_datapoints(db: Session = Depends(get_db)):
    res = db.execute(text("SELECT source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction FROM datapoints ORDER BY id DESC LIMIT 100")).fetchall()
    return [{"source_id": r[0], "source_name": r[1], "dest_id": r[2], "dest_name": r[3], "type": r[4], "route": r[5]} for r in res]

@app.post("/add-datapoint")
def add_datapoint(d: DataPointRequest, db: Session = Depends(get_db)):
    try:
        logger.debug(f"Inserting: {d.source_city_id} -> {d.destination_city_id} ({d.parcel_type})")
        
        db.execute(text("""
            INSERT INTO datapoints (source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction)
            VALUES (:sid, :sn, :did, :dn, :pt, :rd)
        """), {
            "sid": d.source_city_id, 
            "sn": d.source_city_name, 
            "did": d.destination_city_id, 
            "dn": d.destination_city_name, 
            "pt": d.parcel_type, 
            "rd": d.route_direction
        })
        db.commit()
        
        refresh_route_cache()
        return {"status": "success"}
    except Exception as e:
        logger.error("Error in add_datapoint", exc_info=True)
        
        # Emergency auto-fix for sequence issues
        if "UniqueViolation" in str(e) or "duplicate key" in str(e):
            logger.warning("Duplicate Key Detected - Attempting Auto-Fix")
            db.rollback()
            sync_sequences(db)
            raise HTTPException(status_code=500, detail="Database error (Auto-Corrected). Please Try Again.")

        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/refresh-data")
def refresh_data_endpoint():
    count = refresh_route_cache()
    return {"status": "success", "total_rules_loaded": count}

@app.post("/find-route")
def find_route(req: FindRouteRequest):
    key = (req.source_city_id, req.destination_city_id, req.parcel_type)
    
    if key in ROUTE_CACHE:
        route_idx = ROUTE_CACHE[key]
        mapping = {0: "Straight", 1: "Left", 2: "Right"}
        return {"found": True, "route_code": route_idx, "direction": mapping.get(route_idx, "Unknown")}
    else:
        logger.info(f"Route not found for key: {key}")
        return {"found": False, "route_code": -1, "direction": "Route Not Found in Database"}

@app.post("/extract-from-image")
async def extract_api(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = process_image_cv(content)
        return data
    except Exception as e:
        logger.error("Image Extraction Failed", exc_info=True)
        if DEBUG_MODE:
             # In debug mode, show the full error
            raise HTTPException(status_code=500, detail=str(e))
        else:
             # In prod, show a generic error
            raise HTTPException(status_code=500, detail="Error processing image")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)