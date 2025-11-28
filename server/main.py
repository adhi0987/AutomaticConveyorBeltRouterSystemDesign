import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv  # IMPORT THIS
import pandas as pd
import pytesseract
import cv2
import numpy as np
# explicitly set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
import re
import io
import random
import string
import os
import traceback

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================

# Load environment variables from the .env file
load_dotenv()



# Get the URL securely
DATABASE_URL = os.getenv("DATABASE_URL")

CSV_FILE = "parcels_10000.csv"

# Global Cache for O(1) Routing Lookup
ROUTE_CACHE = {}

try:
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set. Please check your .env file.")
        
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    print("Database engine created successfully.")
except Exception as e:
    print(f"CRITICAL: Database Connection Error: {e}")
    traceback.print_exc()
    engine = None



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
    if not engine:
        raise HTTPException(status_code=500, detail="Database not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def refresh_route_cache():
    global ROUTE_CACHE
    temp_cache = {}
    
    print("[Cache] Refreshing route cache...")
    
    # 1. Load CSV (Fallback/Seed data)
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            for _, row in df.iterrows():
                key = (int(row['source_city_id']), int(row['destination_city_id']), int(row['parcel_type']))
                temp_cache[key] = int(row['route_direction'])
            print(f"[Cache] Loaded {len(df)} rules from CSV.")
        except Exception as e:
            print(f"[Cache] CSV Load Error: {e}")

    # 2. Load DB (Overrides CSV)
    if engine:
        try:
            with engine.connect() as conn:
                # Ensure table exists before querying
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
                print(f"[Cache] Loaded {len(rows)} rules from DB.")
        except Exception as e:
            print(f"[Cache] DB Load Error: {e}")
            traceback.print_exc()

    ROUTE_CACHE = temp_cache
    print(f"[Cache] Total Active Rules: {len(ROUTE_CACHE)}")
    return len(ROUTE_CACHE)

def process_image_cv(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    custom_config = r'--oem 3 --psm 6' 
    text_out = pytesseract.image_to_string(thresh, config=custom_config)
    
    data = {"source_city": None, "source_id": None, "dest_city": None, "dest_id": None, "type": 0}
    clean_text = re.sub(r'[|/\[\]]', ' ', text_out)

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

def sync_sequences(db: Session):
    """
    FIX: Resets the Auto-Increment sequence for datapoints.
    This prevents 'duplicate key value' errors.
    """
    try:
        print("[Startup] Syncing Database Sequences...")
        # This SQL command tells Postgres to set the ID counter to the MAX ID + 1
        db.execute(text("SELECT setval(pg_get_serial_sequence('datapoints', 'id'), COALESCE((SELECT MAX(id) + 1 FROM datapoints), 1), false);"))
        db.commit()
        print("[Startup] Sequences Synced.")
    except Exception as e:
        print(f"[Startup Warning] Could not sync sequences: {e}")
        db.rollback()

def init_db_data(db: Session):
    """Handles Table creation and Seeding"""
    print("Running startup DB tasks...")
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
            print("Seeding default admin...")
            db.execute(text("INSERT INTO \"admin-logins\" (username, password, email) VALUES ('admin', 'admin123', 'admin@sys.com')"))
            db.commit()

        # Seed Cities
        city_count = db.execute(text("SELECT COUNT(*) FROM cities")).scalar()
        if city_count == 0 and os.path.exists(CSV_FILE):
            print("Seeding cities from CSV...")
            df = pd.read_csv(CSV_FILE)
            src = df[['source_city_name', 'source_city_id']].rename(columns={'source_city_name': 'name', 'source_city_id': 'id'})
            dst = df[['destination_city_name', 'destination_city_id']].rename(columns={'destination_city_name': 'name', 'destination_city_id': 'id'})
            cities = pd.concat([src, dst]).drop_duplicates(subset=['id'])
            
            for _, row in cities.iterrows():
                clean_name = str(row['name']).replace("'", "''")
                db.execute(text(f"INSERT INTO cities (unique_id, cityname) VALUES ({row['id']}, '{clean_name}') ON CONFLICT DO NOTHING"))
            db.commit()
            
        # Fix sequences after creating tables/seeding
        sync_sequences(db)

    except Exception as seed_err:
        print(f"Seeding Error: {seed_err}")
        traceback.print_exc()

# ==========================================
# 4. LIFESPAN MANAGER
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    if engine:
        db = SessionLocal()
        try:
            init_db_data(db)
        finally:
            db.close()
        
        refresh_route_cache()
    else:
        print("Skipping startup DB tasks - No Engine")
    
    yield
    # --- SHUTDOWN LOGIC ---
    if engine:
        engine.dispose()

app = FastAPI(title="Smart Conveyor Belt System", lifespan=lifespan)

origins = ["http://localhost:5173", "https://automaticconveyorbeltroutersystemdesign-l3l8.onrender.com","https://automaticconveyorbeltroutersystemdesign.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. ENDPOINTS
# ==========================================

@app.post("/login/{role}")
def login(role: str, creds: LoginRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    try:
        res = db.execute(text(f"SELECT id, username FROM \"{table}\" WHERE username=:u AND password=:p"), 
                         {"u": creds.username, "p": creds.password}).fetchone()
        if res:
            return {"status": "success", "user_id": res[0], "username": res[1], "role": role}
    except Exception as e:
        print(f"Login DB Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Database error during login")
        
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
        return {"username": username, "password": password}
    except Exception as e:
        print(f"Add User Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

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
        print(f"Add City Error: {e}")
        raise HTTPException(status_code=400, detail="City likely exists")

@app.get("/datapoints")
def get_datapoints(db: Session = Depends(get_db)):
    # LIMIT to 100 to prevent overloading
    res = db.execute(text("SELECT source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction FROM datapoints ORDER BY id DESC LIMIT 100")).fetchall()
    return [{"source_id": r[0], "source_name": r[1], "dest_id": r[2], "dest_name": r[3], "type": r[4], "route": r[5]} for r in res]

@app.post("/add-datapoint")
def add_datapoint(d: DataPointRequest, db: Session = Depends(get_db)):
    try:
        print(f"Attempting to insert: Src={d.source_city_id}, Dst={d.destination_city_id}, Type={d.parcel_type}")
        
        # INSERT without 'id' - let Postgres handle it (now that sequence is fixed)
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
        print("Insert successful. Refreshing cache...")
        
        refresh_route_cache()
        return {"status": "success"}
    except Exception as e:
        print("!!! ERROR IN ADD_DATAPOINT !!!")
        traceback.print_exc()
        
        # If insertion fails due to PK error, try one emergency sync
        if "UniqueViolation" in str(e) or "duplicate key" in str(e):
            print("Detected Sequence Desync. Attempting emergency fix...")
            db.rollback()
            sync_sequences(db)
            raise HTTPException(status_code=500, detail="Database sequence error. Please try again (System auto-corrected).")

        raise HTTPException(status_code=500, detail=str(e))

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
        return {"found": False, "route_code": -1, "direction": "Route Not Found in Database"}

@app.post("/extract-from-image")
async def extract_api(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = process_image_cv(content)
        return data
    except Exception as e:
        print(f"Image Extraction Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)