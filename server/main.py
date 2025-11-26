import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from sklearn.model_selection import train_test_split
import pytesseract
from PIL import Image
import io
import random
import string
import os

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================

# Your Supabase Connection URL
# CSV File Path
CSV_FILE = "parcels_10000.csv"
MODEL_PATH = "routing_model.h5"
MAX_CITY_ID = 10000  # Adjust if city IDs exceed this

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI App Setup
app = FastAPI(title="Smart Conveyor Belt System")

# CORS - Allow connections from React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. DATA MODELS (Pydantic)
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

class PredictionRequest(BaseModel):
    source_city_id: int
    destination_city_id: int
    parcel_type: int

# ==========================================
# 3. HELPER FUNCTIONS & DEPENDENCIES
# ==========================================

def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_ai_model():
    """Loads the trained Keras model if it exists."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

# Global Model Variable
model = load_ai_model()

def train_model_pipeline():
    """
    Reads data from CSV and Database, combines them,
    trains a new Neural Network, and saves it.
    """
    print("Starting Model Retraining...")
    
    # 1. Load CSV Data
    if os.path.exists(CSV_FILE):
        df_csv = pd.read_csv(CSV_FILE)
    else:
        df_csv = pd.DataFrame(columns=['source_city_ID', 'destination_city_ID', 'parcel_type', 'route'])

    # 2. Load DB Data
    db = SessionLocal()
    try:
        query = text("SELECT source_city_id, destination_city_id, parcel_type, route_direction FROM datapoints")
        results = db.execute(query).fetchall()
        
        if results:
            df_db = pd.DataFrame(results, columns=['source_city_ID', 'destination_city_ID', 'parcel_type', 'route'])
            full_df = pd.concat([df_csv, df_db], ignore_index=True)
        else:
            full_df = df_csv
    except Exception as e:
        print(f"Database read error during training: {e}")
        full_df = df_csv
    finally:
        db.close()

    if len(full_df) < 100:
        print("Not enough data to train.")
        return 0

    # 3. Preprocess
    X_src = full_df['source_city_ID'].values
    X_dst = full_df['destination_city_ID'].values
    X_type = full_df['parcel_type'].values
    y = full_df['route'].values

    # 4. Build Neural Network
    src_input = Input(shape=(1,), name='src_in')
    dst_input = Input(shape=(1,), name='dst_in')
    type_input = Input(shape=(1,), name='type_in')

    embedding = Embedding(input_dim=MAX_CITY_ID + 5000, output_dim=32)
    src_emb = Flatten()(embedding(src_input))
    dst_emb = Flatten()(embedding(dst_input))

    merged = Concatenate()([src_emb, dst_emb, type_input])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)

    new_model = Model(inputs=[src_input, dst_input, type_input], outputs=output)
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 5. Train
    new_model.fit([X_src, X_dst, X_type], y, epochs=5, batch_size=32, verbose=0)
    
    # 6. Save
    new_model.save(MODEL_PATH)
    print("Model saved.")
    
    return len(full_df)

# ==========================================
# 4. STARTUP EVENT (Seeding)
# ==========================================

@app.on_event("startup")
def startup_event():
    """Runs when server starts. Seeds cities table if empty."""
    db = SessionLocal()
    try:
        # Check if cities exist
        try:
            count = db.execute(text("SELECT COUNT(*) FROM cities")).scalar()
        except:
            # Table might not exist yet, skip seeding (assume user ran SQL script)
            print("Cities table not found or error checking count.")
            return

        if count == 0 and os.path.exists(CSV_FILE):
            print("Seeding cities from CSV...")
            df = pd.read_csv(CSV_FILE)
            
            # Extract unique source cities
            src = df[['source_city', 'source_city_ID']].rename(columns={'source_city': 'name', 'source_city_ID': 'id'})
            # Extract unique dest cities
            dst = df[['destination_city', 'destination_city_ID']].rename(columns={'destination_city': 'name', 'destination_city_ID': 'id'})
            
            # Combine and drop duplicates
            cities = pd.concat([src, dst]).drop_duplicates(subset=['id'])
            
            # Insert into DB
            for _, row in cities.iterrows():
                clean_name = str(row['name']).replace("'", "''") # Escape single quotes
                sql = text(f"INSERT INTO cities (unique_id, cityname) VALUES ({row['id']}, '{clean_name}') ON CONFLICT DO NOTHING")
                db.execute(sql)
            
            db.commit()
            print(f"Seeded {len(cities)} cities.")
    except Exception as e:
        print(f"Startup error: {e}")
    finally:
        db.close()

# ==========================================
# 5. API ENDPOINTS
# ==========================================

# --- Auth ---

@app.post("/login/{role}")
def login(role: str, creds: LoginRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    # Note: Using parameterized queries to prevent SQL Injection
    query = text(f"SELECT id, username FROM \"{table}\" WHERE username=:u AND password=:p")
    result = db.execute(query, {"u": creds.username, "p": creds.password}).fetchone()
    
    if result:
        return {"status": "success", "user_id": result[0], "username": result[1], "role": role}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# --- User Management (Admin) ---

@app.get("/users/{role}")
def get_users(role: str, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    result = db.execute(text(f"SELECT id, username, email FROM \"{table}\"")).fetchall()
    return [{"id": r[0], "username": r[1], "email": r[2]} for r in result]

@app.post("/add-user/{role}")
def add_user(role: str, req: NewUserRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    
    # Generate random credentials
    username = 'user_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    try:
        query = text(f"INSERT INTO \"{table}\" (username, password, email) VALUES (:u, :p, :e)")
        db.execute(query, {"u": username, "p": password, "e": req.email})
        db.commit()
        return {"status": "created", "username": username, "password": password}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")

@app.delete("/remove-user/{role}/{user_id}")
def remove_user(role: str, user_id: int, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    db.execute(text(f"DELETE FROM \"{table}\" WHERE id=:id"), {"id": user_id})
    db.commit()
    return {"status": "deleted"}

@app.put("/edit-profile/{role}")
def edit_profile(role: str, req: UpdateProfileRequest, db: Session = Depends(get_db)):
    table = "admin-logins" if role == "admin" else "employee-logins"
    
    # Check uniqueness of new username
    check_query = text(f"SELECT id FROM \"{table}\" WHERE username=:u AND id != :id")
    existing = db.execute(check_query, {"u": req.new_username, "id": req.user_id}).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
        
    update_query = text(f"UPDATE \"{table}\" SET username=:u, password=:p WHERE id=:id")
    db.execute(update_query, {"u": req.new_username, "p": req.new_password, "id": req.user_id})
    db.commit()
    return {"status": "updated"}

# --- City Management ---

@app.get("/cities")
def get_cities(db: Session = Depends(get_db)):
    res = db.execute(text("SELECT unique_id, cityname FROM cities ORDER BY cityname")).fetchall()
    return [{"id": r[0], "name": r[1]} for r in res]

@app.post("/add-city")
def add_city(req: CityRequest, db: Session = Depends(get_db)):
    # Calculate next ID
    max_id = db.execute(text("SELECT MAX(unique_id) FROM cities")).scalar()
    new_id = (max_id if max_id else 5000) + 1
    
    try:
        db.execute(text("INSERT INTO cities (unique_id, cityname) VALUES (:id, :name)"), 
                   {"id": new_id, "name": req.city_name})
        db.commit()
        return {"status": "success", "new_id": new_id, "name": req.city_name}
    except:
        raise HTTPException(status_code=400, detail="City name likely exists")

# --- Data Points & Training ---

@app.get("/datapoints")
def get_datapoints(db: Session = Depends(get_db)):
    # Get DB data
    db_res = db.execute(text("SELECT source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction FROM datapoints ORDER BY id DESC LIMIT 500")).fetchall()
    
    db_data = []
    for r in db_res:
        db_data.append({
            "source_city_ID": r[0], "source_city": r[1],
            "destination_city_ID": r[2], "destination_city": r[3],
            "parcel_type": r[4], "route": r[5], "origin": "DB"
        })
        
    # Optional: Mix with some CSV data for display
    # (Skipping heavy CSV load here for performance, showing mostly DB data)
    return db_data

@app.post("/add-datapoint")
def add_datapoint(data: DataPointRequest, db: Session = Depends(get_db)):
    query = text("""
        INSERT INTO datapoints 
        (source_city_id, source_city_name, destination_city_id, destination_city_name, parcel_type, route_direction)
        VALUES (:sid, :sn, :did, :dn, :pt, :rd)
    """)
    db.execute(query, {
        "sid": data.source_city_id, "sn": data.source_city_name,
        "did": data.destination_city_id, "dn": data.destination_city_name,
        "pt": data.parcel_type, "rd": data.route_direction
    })
    db.commit()
    return {"status": "success"}

@app.post("/retrain")
def retrain_endpoint():
    try:
        total_samples = train_model_pipeline()
        # Reload model globally
        global model
        model = load_ai_model()
        return {"status": "success", "total_samples_trained": total_samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Prediction & Image ---

@app.post("/predict")
def predict_route(req: PredictionRequest):
    if model is None:
        return {"route_code": -1, "direction": "Error: Model not trained/loaded"}
    
    src = np.array([req.source_city_id])
    dst = np.array([req.destination_city_id])
    typ = np.array([req.parcel_type])
    
    pred = model.predict([src, dst, typ])
    route_idx = int(np.argmax(pred[0]))
    
    mapping = {0: "Straight", 1: "Left", 2: "Right"}
    return {"route_code": route_idx, "direction": mapping.get(route_idx, "Unknown")}

@app.post("/extract-from-image")
async def extract_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform OCR
        # Note: Requires Tesseract installed on system
        text_out = pytesseract.image_to_string(image)
        
        # Simple parsing heuristic
        data = {"source_id": "", "dest_id": "", "type": "0"}
        lines = text_out.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            # Extract numbers from line
            nums = [int(s) for s in line.split() if s.isdigit()]
            if not nums: continue
            
            if "source" in line_lower:
                data["source_id"] = nums[0]
            elif "dest" in line_lower:
                data["dest_id"] = nums[0]
            elif "type" in line_lower or "parcel" in line_lower:
                data["type"] = nums[0]
                
        return {"extracted_text": text_out, "parsed_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)