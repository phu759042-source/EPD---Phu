from fastapi import FastAPI, Request, Form, Response, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse # <--- Đã thêm JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from collections import defaultdict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Dữ liệu mẫu (logs_by_class sẽ lưu tất cả logs, bao gồm cả ngày)
logs_by_class = defaultdict(list)
# Đã loại bỏ biến risk_scores cũ, điểm rủi ro sẽ được tính toán động khi truy cập

# --- CẤU HÌNH TÀI KHOẢN ---
ADMIN_USER = "admin"
ADMIN_PASS = "123456"

# Danh sách hành vi được coi là "Mất tập trung" cho mode 'epd_distraction'
DISTRACTION_BEHAVIORS = [
    'Ngủ', 
    'Mất tập trung', 
    'Cúi nhiều (Bad)',
    'Hơi cúi (Warning)' 
    # Thêm các loại vi phạm khác nếu cần
]

# CẬP NHẬT MODEL: Thêm date và scan_mode
class Incident(BaseModel):
    class_id: str
    zone_id: str
    issue_type: str
    start_time: str
    end_time: str
    duration_seconds: float
    date: str          # <--- [MỚI] Ngày xảy ra sự cố (YYYY-MM-DD)
    scan_mode: str     # <--- [MỚI] Chế độ quét (epd_full hoặc epd_distraction)

# --- HÀM HỖ TRỢ LỌC DỮ LIỆU LOG THEO CHẾ ĐỘ & NGÀY ---
def filter_logs(mode: str, class_id: str, date: str):
    """Lọc log theo chế độ, lớp và ngày."""
    all_logs = logs_by_class.get(class_id, [])
    
    # 1. Lọc theo ngày
    daily_logs = [log for log in all_logs if log.get('date') == date]
    
    if mode == 'epd_full':
        # Chế độ Full: Trả về tất cả các log của ngày đó
        return daily_logs
    
    elif mode == 'epd_distraction':
        # Chế độ Mất tập trung: Chỉ lọc các log có hành vi trong DISTRACTION_BEHAVIORS
        return [log for log in daily_logs if log.get('issue_type') in DISTRACTION_BEHAVIORS]
    
    return [] 

# --- API CLIENT GỬI DỮ LIỆU (Không cần đăng nhập) ---
@app.post("/log_incident/")
async def log_incident(incident: Incident):
    data = incident.dict()
    logs_by_class[incident.class_id].insert(0, data)
    
    # Tăng giới hạn log lên 500 (hoặc cao hơn) để lưu trữ được nhiều ngày
    if len(logs_by_class[incident.class_id]) > 500: 
        logs_by_class[incident.class_id].pop() # Loại bỏ log cũ nhất
    
    # In thông báo kèm theo scan_mode và date
    print(f"📡 {incident.class_id} ({incident.scan_mode}) | Ngày {incident.date}: HS-{incident.zone_id} +{incident.duration_seconds} điểm")
    return {"status": "success"}

# --- API LẤY LOGS CHI TIẾT (Cập nhật route: /api/get_logs/{mode}/{class_id}/{date}) ---
@app.get("/api/get_logs/{mode}/{class_id}/{date}")
async def get_logs(mode: str, class_id: str, date: str, request: Request):
    # Dùng JSONResponse để trả về mảng rỗng nếu chưa đăng nhập
    if not check_auth(request): return JSONResponse(content=[], status_code=200) 
    
    # Sử dụng hàm lọc
    filtered_logs = filter_logs(mode, class_id, date)
    return JSONResponse(content=filtered_logs)

# --- API LẤY BẢNG XẾP HẠNG (Cập nhật route: /api/get_risk_ranking/{mode}/{class_id}/{date}) ---
@app.get("/api/get_risk_ranking/{mode}/{class_id}/{date}")
async def get_risk_ranking(mode: str, class_id: str, date: str, request: Request):
    # Dùng JSONResponse để trả về mảng rỗng nếu chưa đăng nhập
    if not check_auth(request): return JSONResponse(content=[], status_code=200)

    # 1. Lọc log tương ứng với chế độ và ngày
    filtered_logs = filter_logs(mode, class_id, date)
    
    # 2. Tính toán điểm rủi ro chỉ cho các log đã lọc
    risk_scores_daily = defaultdict(float)
    
    for log in filtered_logs:
        duration = log.get('duration_seconds', 0)
        risk_scores_daily[log['zone_id']] += duration 

    # 3. Sắp xếp kết quả
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)