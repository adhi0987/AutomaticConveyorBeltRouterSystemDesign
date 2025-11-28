import { useState, useEffect, ChangeEvent } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, Navigate } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// const API_URL = "http://localhost:8000";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";  
// --- COMPONENTS ---

const Navbar = ({ role, onLogout }: { role: string | null, onLogout: () => void }) => (
  <nav className="navbar">
    <div className="nav-brand">📦 AutoRouter</div>
    <div className="nav-links">
      <Link to="/">Home</Link>
      <Link to="/about">  About</Link>
      {!role && (
        <>
          <Link to="/login/admin">Admin</Link>
          <Link to="/login/employee">Employee</Link>
        </>
      )}
      {role === 'admin' && <Link to="/admin">Dashboard</Link>}
      {role === 'employee' && <Link to="/employee">Dashboard</Link>}
      {role && <button onClick={onLogout} className="logout-btn">Logout</button>}
    </div>
  </nav>
);

const About = () => (
  <div className="page-container">
    <h1>About Smart Conveyor Router</h1>
    <p>
      This system uses Computer Vision to read parcel labels and a highly efficient 
      Database Lookup Engine to determine the routing direction (Left, Right, Straight).
    </p>
    
    <div className="features-list">
      <h3>Key Features:</h3>
      <ul>
        <li><b>Admin Portal:</b> Manage staff access securely.</li>
        <li><b>Employee Portal:</b> Update routing rules and city databases.</li>
        <li><b>Computer Vision:</b> OCR extraction from cardboard labels.</li>
        <li><b>Instant Routing:</b> O(1) Time Complexity lookup (No ML latency).</li>
      </ul>
    </div>

    <div className="video-placeholder">
      <div className="video-box">
        <p>▶ Demo Video Placeholder</p>
        <small>(Insert your YouTube Embed or Video File here)</small>
      </div>
    </div>
  </div>
);

const Login = ({ type, setRole, setUserId }: { type: string, setRole: any, setUserId: any }) => {
  const [creds, setCreds] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const res = await axios.post(`${API_URL}/login/${type}`, creds);
      setRole(res.data.role);
      setUserId(res.data.user_id);
      navigate(type === 'admin' ? '/admin' : '/employee');
    } catch (err) {
      setError("Invalid Credentials. Please check username/password.");
    }
  };

  return (
    <div className="login-container">
      <div className="card login-card">
        <h2>{type === 'admin' ? 'Admin' : 'Employee'} Portal</h2>
        <input placeholder="Username" onChange={e => setCreds({...creds, username: e.target.value})} />
        <input type="password" placeholder="Password" onChange={e => setCreds({...creds, password: e.target.value})} />
        <button onClick={handleLogin}>Login</button>
        {error && <p className="error-msg">{error}</p>}
      </div>
    </div>
  );
};

// --- ADMIN DASHBOARD ---
const AdminDashboard = ({ userId }: { userId: number }) => {
  const [view, setView] = useState('admins');
  const [users, setUsers] = useState<any[]>([]);
  const [newUserEmail, setNewUserEmail] = useState('');
  const [createdCreds, setCreatedCreds] = useState<any>(null);

  const fetchUsers = async (role: string) => {
    const res = await axios.get(`${API_URL}/users/${role}`);
    setUsers(res.data);
  };

  useEffect(() => {
    setCreatedCreds(null);
    if (view === 'admins') fetchUsers('admin');
    if (view === 'employees') fetchUsers('employee');
  }, [view]);

  const addUser = async (role: string) => {
    if(!newUserEmail) return alert("Enter email");
    const res = await axios.post(`${API_URL}/add-user/${role}`, { email: newUserEmail });
    setCreatedCreds(res.data);
    fetchUsers(role);
  };

  const removeUser = async (role: string, id: number) => {
    if(!confirm("Are you sure?")) return;
    await axios.delete(`${API_URL}/remove-user/${role}/${id}`);
    fetchUsers(role);
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <button onClick={() => setView('admins')} className={view==='admins'?'active':''}>Manage Admins</button>
        <button onClick={() => setView('employees')} className={view==='employees'?'active':''}>Manage Employees</button>
        <button onClick={() => setView('profile')} className={view==='profile'?'active':''}>Edit Profile</button>
      </div>
      <div className="content">
        {view === 'profile' ? <EditProfile role="admin" userId={userId} /> : (
          <>
            <h3>Manage {view}</h3>
            <div className="action-bar">
              <input placeholder="New User Email" value={newUserEmail} onChange={e => setNewUserEmail(e.target.value)} />
              <button onClick={() => addUser(view === 'admins' ? 'admin' : 'employee')}>+ Add User</button>
            </div>
            
            {createdCreds && (
              <div className="success-box">
                <p>User Created Successfully!</p>
                <p>Username: <b>{createdCreds.username}</b></p>
                <p>Password: <b>{createdCreds.password}</b></p>
                <small>(Please copy these credentials now)</small>
              </div>
            )}

            <table className="data-table">
              <thead><tr><th>ID</th><th>Username</th><th>Email</th><th>Action</th></tr></thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id}>
                    <td>{u.id}</td><td>{u.username}</td><td>{u.email}</td>
                    <td><button className="danger-btn" onClick={() => removeUser(view === 'admins' ? 'admin' : 'employee', u.id)}>Remove</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
};

// --- EMPLOYEE DASHBOARD ---
const EmployeeDashboard = ({ userId }: { userId: number }) => {
  const [view, setView] = useState('datapoints');
  const [cities, setCities] = useState<any[]>([]);
  const [datapoints, setDatapoints] = useState<any[]>([]);
  
  const [formData, setFormData] = useState({ source: '', dest: '', type: '0', route: '0' });

  useEffect(() => {
    loadCities();
    loadDatapoints();
  }, []);

  const loadCities = async () => { const res = await axios.get(`${API_URL}/cities`); setCities(res.data); };
  const loadDatapoints = async () => { const res = await axios.get(`${API_URL}/datapoints`); setDatapoints(res.data); };

  const handleAddCity = async () => {
    const name = prompt("Enter new city name:");
    if (name) {
      await axios.post(`${API_URL}/add-city`, { city_name: name });
      loadCities();
    }
  };

  const handleAddData = async () => {
    const srcCity = cities.find(c => c.name === formData.source);
    const dstCity = cities.find(c => c.name === formData.dest);
    
    if (!srcCity || !dstCity) return alert("Please select valid cities from the list.");

    await axios.post(`${API_URL}/add-datapoint`, {
      source_city_id: srcCity.id, source_city_name: srcCity.name,
      destination_city_id: dstCity.id, destination_city_name: dstCity.name,
      parcel_type: parseInt(formData.type), route_direction: parseInt(formData.route)
    });
    alert("Data Point Added!");
    loadDatapoints();
  };

  const refreshData = async () => {
    const res = await axios.post(`${API_URL}/refresh-data`);
    alert(`System Updated! Total Rules Active: ${res.data.total_rules_loaded}`);
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <button onClick={() => setView('datapoints')} className={view==='datapoints'?'active':''}>Data Points</button>
        <button onClick={() => setView('cities')} className={view==='cities'?'active':''}>Cities Database</button>
        <button onClick={() => setView('profile')} className={view==='profile'?'active':''}>My Profile</button>
        <button onClick={refreshData} className="warning-btn">↻ Refresh System</button>
      </div>
      <div className="content">
        {view === 'profile' && <EditProfile role="employee" userId={userId} />}
        
        {view === 'cities' && (
          <>
            <div className="header-flex">
               <h3>City Database</h3>
               <button onClick={handleAddCity}>+ Add New City</button>
            </div>
            <div className="list-container">
              <table>
                <td>
                  {cities.map(c => <div className="list-item" key={c.id}>
                    <td>{c.id}</td><td>{c.name}</td>
                  </div>)}
                </td>
              </table>
            </div>
          </>
        )}
        
        {view === 'datapoints' && (
          <>
            <h3>Add New Routing Rule</h3>
            <div className="data-entry-form">
              <div className="form-group">
                <label>Source City</label>
                <input list="cities" value={formData.source} onChange={e => setFormData({...formData, source: e.target.value})} />
              </div>
              <div className="form-group">
                <label>Destination City</label>
                <input list="cities" value={formData.dest} onChange={e => setFormData({...formData, dest: e.target.value})} />
              </div>
              <datalist id="cities">{cities.map(c => <option key={c.id} value={c.name} />)}</datalist>
              
              <div className="form-group">
                <label>Parcel Type</label>
                <select onChange={e => setFormData({...formData, type: e.target.value})}>
                  <option value="0">Normal (0)</option><option value="1">Fast (1)</option>
                </select>
              </div>
              <div className="form-group">
                <label>Route</label>
                <select onChange={e => setFormData({...formData, route: e.target.value})}>
                  <option value="0">Straight</option><option value="1">Left</option><option value="2">Right</option>
                </select>
              </div>
              <button className="add-btn" onClick={handleAddData}>Add Entry</button>
            </div>
            
            <h3>Recent Data Entries</h3>
            <table className="data-table">
              <thead><tr><th>Source</th><th>Dest</th><th>Type</th><th>Route</th></tr></thead>
              <tbody>
                {datapoints.map((d, i) => (
                  <tr key={i}><td>{d.source_name}</td><td>{d.dest_name}</td><td>{d.type}</td><td>{d.route}</td></tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
};

// --- USER HOME (PREDICTION) ---
const UserHome = () => {
  const [file, setFile] = useState<File | null>(null);
  const [extracted, setExtracted] = useState<any>(null);
  const [routeResult, setRouteResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setRouteResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${API_URL}/extract-from-image`, formData);
      setExtracted(res.data);
    } catch(e) { alert("Error processing image"); }
    setLoading(false);
  };

  const handleFindRoute = async () => {
    if (!extracted) return;
    try {
      const res = await axios.post(`${API_URL}/find-route`, {
        source_city_id: extracted.source_id,
        destination_city_id: extracted.dest_id,
        parcel_type: extracted.type
      });
      setRouteResult(res.data);
    } catch(e) { alert("Error finding route"); }
  };

  return (
    <div className="home-container">
      <header className="hero">
        <h1>Smart Conveyor Routing</h1>
        <p>Upload a parcel image to automatically determine the conveyor direction.</p>
      </header>

      <div className="upload-section card">
        <h3>1. Scan Parcel</h3>
        <input type="file" onChange={(e: ChangeEvent<HTMLInputElement>) => { if (e.target.files) setFile(e.target.files[0]) }} />
        <button onClick={handleUpload} disabled={loading}>{loading ? 'Scanning...' : 'Extract Data'}</button>
      </div>

      {extracted && (
        <div className="results-grid">
          <div className="card info-card">
            <h3>2. Extracted Info</h3>
            <p><b>Source:</b> {extracted.source_city || "Unknown"} (ID: {extracted.source_id})</p>
            <p><b>Destination:</b> {extracted.dest_city || "Unknown"} (ID: {extracted.dest_id})</p>
            <p><b>Type:</b> {extracted.type === 0 ? "Normal" : "Fast"}</p>
            <button className="predict-btn" onClick={handleFindRoute}>Find Route</button>
          </div>
          
          {routeResult && (
             <div className={`card result-card ${routeResult.found ? 'found' : 'error'}`}>
               <h3>3. Routing Decision</h3>
               {routeResult.found ? (
                 <>
                   <div className="direction-arrow">
                      {routeResult.direction === 'Left' && '⬅️ LEFT'}
                      {routeResult.direction === 'Right' && 'RIGHT ➡️'}
                      {routeResult.direction === 'Straight' && '⬆️ STRAIGHT'}
                   </div>
                   <p>Route Code: {routeResult.route_code}</p>
                 </>
               ) : (
                 <p className="not-found">Route Not Found in Database</p>
               )}
             </div>
          )}
        </div>
      )}
    </div>
  );
};

const EditProfile = ({ role, userId }: { role: string, userId: number }) => {
  const [data, setData] = useState({ new_username: '', new_password: '' });
  const update = async () => {
    try {
      await axios.put(`${API_URL}/edit-profile/${role}`, { user_id: userId, ...data });
      alert("Profile Updated Successfully!");
    } catch(e) { alert("Error: Username likely taken"); }
  };
  return (
    <div className="card profile-card">
      <h3>Edit Profile</h3>
      <input placeholder="New Username" onChange={e => setData({...data, new_username: e.target.value})} />
      <input placeholder="New Password" onChange={e => setData({...data, new_password: e.target.value})} />
      <button onClick={update}>Save Changes</button>
    </div>
  );
};

function App() {
  const [role, setRole] = useState<string | null>(null);
  const [userId, setUserId] = useState<number>(0);

  const logout = () => { setRole(null); setUserId(0); };

  return (
    <Router>
      <div className="App">
        <Navbar role={role} onLogout={logout} />
        <div className="main-wrapper">
          <Routes>
            <Route path="/" element={<UserHome />} />
            <Route path="/about" element={<About />} />
            <Route path="/login/admin" element={<Login type="admin" setRole={setRole} setUserId={setUserId} />} />
            <Route path="/login/employee" element={<Login type="employee" setRole={setRole} setUserId={setUserId} />} />
            <Route path="/admin" element={role === 'admin' ? <AdminDashboard userId={userId} /> : <Navigate to="/login/admin" />} />
            <Route path="/employee" element={role === 'employee' ? <EmployeeDashboard userId={userId} /> : <Navigate to="/login/employee" />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;