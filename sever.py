from fastapi import FastAPI, Request, Form, Response, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse # <--- Đã thêm JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from collections import defaultdict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# FAVICON FALLBACK (CỰC QUAN TRỌNG)
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/AI_Smart_Monitor.ico")
templates = Jinja2Templates(directory="templates")

logs_by_class = defaultdict(list)

# --- CẤU HÌNH TÀI KHOẢN ---
ADMIN_USER = "admin"
ADMIN_PASS = "123456"

# [ĐÃ LOẠI BỎ] Các danh sách hành vi phân loại (NEGATIVE_EMOTIONS, DISTRACTION_BEHAVIORS) 
# vì chúng không còn được sử dụng trong logic lọc mới.

# CẬP NHẬT MODEL: Thêm date và scan_mode
class Incident(BaseModel):
    class_id: str
    zone_id: str
    issue_type: str
    start_time: str
    end_time: str
    duration_seconds: float
    date: str          # Ngày xảy ra sự cố (YYYY-MM-DD)
    scan_mode: str     # Chế độ quét (epd_full hoặc epd_distraction)

# --- HÀM HỖ TRỢ LỌC DỮ LIỆU LOG THEO CHẾ ĐỘ & NGÀY (ĐÃ ĐƠN GIẢN HÓA) ---
def filter_logs(mode: str, class_id: str, date: str):
    """
    Lọc log theo chế độ và ngày. 
    Chỉ hiển thị log nếu log.scan_mode khớp với mode được yêu cầu.
    """
    all_logs = logs_by_class.get(class_id, [])
    
    # 1. Lọc theo ngày
    daily_logs = [log for log in all_logs if log.get('date') == date]
    
    # 2. Lọc theo chế độ quét (scan_mode)
    # QUAN TRỌNG: Chỉ giữ lại log có scan_mode TRÙNG KHỚP với mode được yêu cầu từ dashboard
    return [log for log in daily_logs if log.get('scan_mode') == mode]


# --- API CLIENT GỬI DỮ LIỆU (Không cần đăng nhập) ---
@app.post("/log_incident/")
async def log_incident(incident: Incident):
    data = incident.dict()
    logs_by_class[incident.class_id].insert(0, data)
    
    if len(logs_by_class[incident.class_id]) > 500: 
        logs_by_class[incident.class_id].pop()
    
    print(f"📡 {incident.class_id} ({incident.scan_mode}) | Ngày {incident.date}: HS-{incident.zone_id} +{incident.duration_seconds} điểm")
    return {"status": "success"}

# --- API LẤY LOGS CHI TIẾT (Giữ nguyên) ---
@app.get("/api/get_logs/{mode}/{class_id}/{date}")
async def get_logs(mode: str, class_id: str, date: str, request: Request):
    if not check_auth(request): return JSONResponse(content=[], status_code=200) 
    
    filtered_logs = filter_logs(mode, class_id, date)
    return JSONResponse(content=filtered_logs)

# --- API LẤY BẢNG XẾP HẠNG (Giữ nguyên) ---
@app.get("/api/get_risk_ranking/{mode}/{class_id}/{date}")
async def get_risk_ranking(mode: str, class_id: str, date: str, request: Request):
    if not check_auth(request): return JSONResponse(content=[], status_code=200)

    filtered_logs = filter_logs(mode, class_id, date)
    
    risk_scores_daily = defaultdict(float)
    
    for log in filtered_logs:
        duration = log.get('duration_seconds', 0)
        risk_scores_daily[log['zone_id']] += duration 

    sorted_students = sorted(risk_scores_daily.items(), key=lambda item: item[1], reverse=True)
    return [{"id": k, "score": int(v)} for k, v in sorted_students]


# --- Hàm kiểm tra đăng nhập (Giữ nguyên) ---
def check_auth(request: Request):
    token = request.cookies.get("access_token")
    if token != "logged_in_secret_key":
        return False
    return True

# --- TRANG LOGIN (GET) --- (Giữ nguyên)
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# --- XỬ LÝ LOGIN (POST) --- (Giữ nguyên)
@app.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USER and password == ADMIN_PASS:
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(key="access_token", value="logged_in_secret_key")
        return resp
    else:
        return RedirectResponse(url="/login", status_code=303)

# --- TRANG CHỦ (Đã bảo vệ) --- (Giữ nguyên)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    token = request.cookies.get("access_token")
    if token != "logged_in_secret_key":
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("index.html", {"request": request})

# --- ĐĂNG XUẤT --- (Giữ nguyên)
@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login")
    resp.delete_cookie("access_token")
    return resp
