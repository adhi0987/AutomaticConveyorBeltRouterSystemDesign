import uvicorn
import os
import io
import random
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================

# Your Supabase Connection URL
# CSV File Path
CSV_FILE = "parcels_10000.csv"


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
# 3. HELPER FUNCTIONS & DEPENDENCIES
# ==========================================

def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
