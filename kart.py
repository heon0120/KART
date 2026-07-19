import sys
import os
import re
import requests
import numpy as np
import logging
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from PySide6.QtCore import QTimer, Slot, Qt, QRunnable, QThreadPool, Signal, QObject, QDate
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, 
                               QWidget, QLabel, QPushButton, QComboBox, 
                               QGroupBox, QGridLayout, QFrame, QSizePolicy, 
                               QDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                               QDateEdit, QFileDialog, QMessageBox, QProgressBar, QDoubleSpinBox, QLineEdit)

# Matplotlib을 PySide6와 연동
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams['font.family'] = 'Consolas'  # Matplotlib 모노스페이스 폰트 적용
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- 상세 로깅 설정 ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)



# =====================

KART_VERSION = "1.0.4"
CONFIG_FILE = "config.xml"
BUILD_DATE = "N/A"
BUILD_SERIAL = "N/A"
Intended_Use = "Community Version"
usage_disclaimer = "이 소프트웨어는 오픈소스로 공개된 커뮤니티 버전입니다. 상업적 용도로 사용 시 KMA API 이용 약관을 준수해야 합니다."
# =====================

def load_auth_key():
    """config.xml에서 인증키를 불러옵니다."""
    if os.path.exists(CONFIG_FILE):
        try:
            tree = ET.parse(CONFIG_FILE)
            root = tree.getroot()
            auth_key = root.find('auth_key').text
            return auth_key if auth_key else None
        except Exception as e:
            logger.error(f"Error loading config.xml: {e}")
            return None
    return None

def save_auth_key(auth_key):
    """인증키를 config.xml에 저장합니다."""
    try:
        root = ET.Element("config")
        auth_elem = ET.SubElement(root, "auth_key")
        auth_elem.text = auth_key
        tree = ET.ElementTree(root)
        tree.write(CONFIG_FILE, encoding='utf-8', xml_declaration=True)
        logger.info("Auth key saved to config.xml")
    except Exception as e:
        logger.error(f"Error saving config.xml: {e}")

def mask_auth_key_in_url(url):
    return re.sub(r'(authKey=)[^&\s]+', r'\1[AuthKey Masked]', url)

# 다크 모드 전역 스타일시트
DARK_QSS = """
QMainWindow, QDialog, QFrame {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QWidget#central_widget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QLabel {
    background: transparent;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    margin-top: 12px;
    padding-top: 12px;
    color: #cccccc;
    font-weight: bold;
    background-color: transparent;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QTableWidget {
    background-color: #252525;
    alternate-background-color: #2e2e2e;
    color: #e0e0e0;
    gridline-color: #444444;
    selection-background-color: #0078d7;
    selection-color: #ffffff;
    border: 1px solid #3a3a3a;
}
QHeaderView::section {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #444444;
    padding: 4px;
    font-weight: bold;
}
"""

def parse_radar_text(api_text):
    """기상청 텍스트 스트림 데이터 분석 및 메타데이터 추출 (예외 처리 강화)"""
    logger.debug("Starting to parse radar text data...")
    lines = [line.strip() for line in api_text.split('\n') if line.strip()]
    azimuths = []
    radar_matrix = []
    current_ray_data = []
    is_reading_data = False
    expected_bins = 600  # 초기 가정값
    
    metadata = {
        'lat': '-', 'lon': '-', 'ht': '-', 'Vn': '-', 'sw': '-', 
        'evel': '-', 'gate': '-', 'bin1': '-', 'width': '-', 'nyq_vel': '-'
    }

    try:
        for line in lines:
            if line.startswith('/DATA/'):
                try:
                    parts = [p.strip() for p in line.split(':')]
                    if len(parts) >= 5:
                        metadata['lat'] = parts[2]
                        metadata['lon'] = parts[3]
                        metadata['ht'] = parts[4]
                        last_parts = parts[-1].split()
                        if len(last_parts) == 4:
                            metadata['Vn'] = last_parts[0]
                            metadata['sw'] = last_parts[1]
                            metadata['nray'] = last_parts[2]
                            metadata['evel'] = last_parts[3]
                except Exception as e:
                    logger.warning(f"Error parsing metadata line '{line}': {e}")
                continue
                
            if line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue

            if len(parts) == 7:
                try:
                    _ = int(parts[0])
                    azimuth_val = float(parts[1])
                    expected_bins = int(parts[2])
                    
                    if metadata['gate'] == '-':
                        metadata['gate'] = parts[3]
                        metadata['bin1'] = parts[4]
                        metadata['width'] = parts[5]
                        metadata['nyq_vel'] = parts[6]
                    
                    if current_ray_data:
                        if len(current_ray_data) < expected_bins:
                            current_ray_data.extend([np.nan] * (expected_bins - len(current_ray_data)))
                        radar_matrix.append(current_ray_data[:expected_bins])
                        current_ray_data = []
                        
                    azimuths.append(azimuth_val)
                    is_reading_data = True
                    continue
                except ValueError as e:
                    logger.warning(f"ValueError parsing header parts '{line}': {e}")
                    continue

            if is_reading_data:
                try:
                    current_ray_data.extend([float(x) for x in parts])
                except ValueError:
                    continue

        if current_ray_data:
            if len(current_ray_data) < expected_bins:
                current_ray_data.extend([np.nan] * (expected_bins - len(current_ray_data)))
            radar_matrix.append(current_ray_data[:expected_bins])

        if not radar_matrix:
            logger.error("Radar matrix is empty after parsing.")
            return np.array([]), np.zeros((0, expected_bins)), metadata

        logger.debug(f"Parsing complete. Azimuths: {len(azimuths)}, Matrix Shape: {len(radar_matrix)}x{expected_bins}")
        return np.array(azimuths), np.array(radar_matrix), metadata
        
    except Exception as e:
        logger.error(f"Critical error parsing radar text: {e}\n{traceback.format_exc()}")
        return np.array([]), np.zeros((0, 0)), metadata


def parse_uf_inf(api_text):
    """nph-rdr_uf_inf API 응답을 파싱하여 사용 가능한 Volume과 Sweep 목록을 반환"""
    logger.debug("Parsing UF inf data...")
    volumes = []
    sweeps_per_vol = {}
    is_data_section = False
    
    try:
        for line in api_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('#Vn'):
                is_data_section = True
                continue
                
            if not is_data_section:
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vn = int(parts[0])
                    sw = int(parts[2])
                    
                    if vn not in sweeps_per_vol:
                        sweeps_per_vol[vn] = set()
                        volumes.append(vn)
                    sweeps_per_vol[vn].add(sw)
                except ValueError as e:
                    logger.debug(f"Skipping inf line due to ValueError: {line} - {e}")
                    continue
                    
        for vn in sweeps_per_vol:
            sweeps_per_vol[vn] = sorted(list(sweeps_per_vol[vn]))
            
        volumes.sort()
        logger.info(f"Inf parse result -> Volumes: {volumes}, Sweeps per Vol: {sweeps_per_vol}")
        return volumes, sweeps_per_vol
        
    except Exception as e:
        logger.error(f"Critical error parsing UF inf: {e}\n{traceback.format_exc()}")
        return [], {}


def parse_uf_list(api_text):
    """UF 파일 목록 텍스트 파싱"""
    logger.debug("Parsing UF file list...")
    files = []
    try:
        for line in api_text.split('\n'):
            line = line.strip()
            if line.startswith('RDR_') and '.uf' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    fname = parts[0].strip()
                    size_str = parts[1].strip()
                    try:
                        # 1. 파일명을 언더바(_)로 분리
                        name_parts = fname.split('_')
                        
                        # 2. 지점명(stn)은 항상 'RDR_' 바로 뒤인 1번 인덱스
                        stn = name_parts[1]
                        
                        # 3. 날짜/시간(dt_str)은 항상 가장 마지막 요소에서 '.uf'를 제거한 것
                        dt_str = name_parts[-1].replace('.uf', '')
                        
                        size = int(size_str)
                        
                        dt_obj = datetime.strptime(dt_str, "%Y%m%d%H%M")
                        dt_display = dt_obj.strftime("%Y-%m-%d %H:%M")
                        
                        files.append({
                            'filename': fname,
                            'station': stn,
                            'datetime': dt_str,
                            'dt_display': dt_display,
                            'size': size
                        })
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Failed to parse UF list line '{line}': {e}")
                        continue
        logger.info(f"Found {len(files)} UF files.")
        return files
    except Exception as e:
        logger.error(f"Critical error parsing UF list: {e}\n{traceback.format_exc()}")
        return files

    
# --- 백그라운드 쓰레드 작업을 위한 클래스들 ---
class RadarSignals(QObject):
    result = Signal(str, object, object, object, datetime)
    error = Signal(str, str, str) 

class RadarFetchTask(QRunnable):
    def __init__(self, station, tm_str, vol_val, sw_val, auth_key, qcd_val):
        super().__init__()
        self.station = station
        self.tm_str = tm_str
        self.vol_val = vol_val
        self.sw_val = sw_val
        self.auth_key = auth_key
        self.qcd_val = qcd_val
        self.signals = RadarSignals()

    @Slot()
    def run(self):
        logger.info(f"[{self.station}] Starting data fetch for TM={self.tm_str}, Vol={self.vol_val}, Sw={self.sw_val}, QCD={self.qcd_val}")
        base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-rdr_uf_data"
        params = {
            "tm": self.tm_str,
            "stn": self.station,
            "vol": self.vol_val,
            "sw": self.sw_val,
            "qcd": self.qcd_val,
            "mode": "A",
            "authKey": self.auth_key
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            logger.debug(f"[{self.station}] Response Status: {response.status_code}, Length: {len(response.text)} bytes")
            
            if response.status_code == 200:
                if "데이터가 존재하지 않습니다" in response.text or not response.text.strip():
                    logger.warning(f"[{self.station}] No data exists for TM={self.tm_str}")
                    self.signals.error.emit(self.station, "No data yet", self.tm_str)
                    return
                
                azimuths, radar_array, meta = parse_radar_text(response.text)
                if radar_array.size == 0:
                    logger.error(f"[{self.station}] Parse error: radar array size is 0")
                    self.signals.error.emit(self.station, "Parse error", self.tm_str)
                    return

                # 결측치 처리 (-99.0을 np.nan으로 변환하여 시각적 왜곡 방지)
                radar_array[radar_array == -99.0] = np.nan
                target_dt = datetime.strptime(self.tm_str, "%Y%m%d%H%M")
                logger.info(f"[{self.station}] Data fetch successful. Emitting result.")
                self.signals.result.emit(self.station, azimuths, radar_array, meta, target_dt)
            else:
                logger.error(f"[{self.station}] HTTP Error: {response.status_code}")
                self.signals.error.emit(self.station, f"HTTP {response.status_code}", self.tm_str)
        except requests.exceptions.Timeout:
            logger.error(f"[{self.station}] Request timed out.")
            self.signals.error.emit(self.station, "Timeout", self.tm_str)
        except Exception as e:
            logger.error(f"[{self.station}] Exception during data fetch: {e}\n{traceback.format_exc()}")
            self.signals.error.emit(self.station, str(e), self.tm_str)


# --- Volume/Sweep 제한을 위한 Info API Task ---
class RadarInfoSignals(QObject):
    result = Signal(str, object, object, str) 
    error = Signal(str, str, str)

class RadarInfoFetchTask(QRunnable):
    def __init__(self, station, tm_str, auth_key, qcd_val):
        super().__init__()
        self.station = station
        self.tm_str = tm_str
        self.auth_key = auth_key
        self.qcd_val = qcd_val
        self.signals = RadarInfoSignals()

    @Slot()
    def run(self):
        logger.info(f"[{self.station}] Starting INF fetch for TM={self.tm_str}, QCD={self.qcd_val}")
        base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-rdr_uf_inf"
        params = {
            "help": 0,
            "tm": self.tm_str,
            "stn": self.station,
            "qcd": self.qcd_val,
            "authKey": self.auth_key
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            logger.debug(f"[{self.station}] Inf Response Status: {response.status_code}, Length: {len(response.text)} bytes")
            
            if response.status_code == 200:
                if "데이터가 존재하지 않습니다" in response.text or not response.text.strip():
                    logger.warning(f"[{self.station}] No inf data exists for TM={self.tm_str}")
                    self.signals.error.emit(self.station, "No info yet", self.tm_str)
                    return
                
                volumes, sweeps_per_vol = parse_uf_inf(response.text)
                if not volumes:
                    logger.error(f"[{self.station}] Inf parse error: no volumes found")
                    self.signals.error.emit(self.station, "Info parse error", self.tm_str)
                    return
                
                logger.info(f"[{self.station}] Inf fetch successful. Volumes: {volumes}")
                self.signals.result.emit(self.station, volumes, sweeps_per_vol, self.tm_str)
            else:
                logger.error(f"[{self.station}] Inf HTTP Error: {response.status_code}")
                self.signals.error.emit(self.station, f"HTTP {response.status_code}", self.tm_str)
        except requests.exceptions.Timeout:
            logger.error(f"[{self.station}] Inf request timed out.")
            self.signals.error.emit(self.station, "Timeout", self.tm_str)
        except Exception as e:
            logger.error(f"[{self.station}] Exception during inf fetch: {e}\n{traceback.format_exc()}")
            self.signals.error.emit(self.station, str(e), self.tm_str)


# --- UF 목록 조회 및 다운로드를 위한 백그라운드 클래스 ---
class UFListSignals(QObject):
    result = Signal(list)
    error = Signal(str)

class UFListFetchTask(QRunnable):
    def __init__(self, tm_str, auth_key):
        super().__init__()
        self.tm_str = tm_str
        self.auth_key = auth_key
        self.signals = UFListSignals()

    @Slot()
    def run(self):
        logger.info(f"Fetching UF list for TM={self.tm_str}")
        base_url = "https://apihub.kma.go.kr/api/typ01/url/rdr_file_list.php"
        params = {
            "rdr": "UF",
            "qcd": 2,
            "tm": self.tm_str,
            "authKey": self.auth_key
        }
        logger.debug(f"request URL: {mask_auth_key_in_url(base_url + '?' + '&'.join(f'{k}={v}' for k, v in params.items()))}")

 
        try:
            # 타임아웃 세분화: 연결(Connection)은 5초 내에, 응답(Read)은 데이터가 많을 수 있으므로 25초까지 대기
            response = requests.get(base_url, params=params, timeout=(5, 25))
            logger.debug(f"UF List Response Status: {response.status_code}, Length: {len(response.text)} bytes")
            
            if response.status_code == 200:
                files = parse_uf_list(response.text)
                if not files:
                    logger.warning("UF List empty or parse failed.")
                    self.signals.error.emit("데이터가 없거나 파싱 실패")
                    return
                logger.info(f"UF List fetch successful. Emitting {len(files)} files.")
                self.signals.result.emit(files)
            else:
                logger.error(f"UF List HTTP Error: {response.status_code}")
                self.signals.error.emit(f"HTTP {response.status_code}")

        # 1. 타임아웃 에러를 먼저 별도로 잡아내어 친절한 문구 송출
        except requests.exceptions.Timeout as te:
            logger.error(f"Timeout during UF list fetch: {te}")
            self.signals.error.emit("기상청 서버 응답 지연 (타임아웃). 잠시 후 다시 시도해 주세요.")

        # 2. 기타 requests 관련 에러 (네트워크 끊김 등)
        except requests.exceptions.RequestException as re:
            logger.error(f"Network error during UF list fetch: {re}")
            self.signals.error.emit(f"네트워크 오류: {re}")

        # 3. 그 외 예상치 못한 예외 처리 (JSON 파싱 에러, 로직 에러 등)
        except Exception as e:
            logger.error(f"Exception during UF list fetch: {e}\n{traceback.format_exc()}")
            self.signals.error.emit(f"시스템 오류: {str(e)}")

class UFDownloadSignals(QObject):
    progress = Signal(str, int, int)
    finished = Signal(int, int)
    error = Signal(str, str)

class UFDownloadTask(QRunnable):
    def __init__(self, files, save_dir, auth_key):
        super().__init__()
        self.files = files
        self.save_dir = save_dir
        self.auth_key = auth_key
        self.signals = UFDownloadSignals()

    @Slot()
    def run(self):
        total = len(self.files)
        success = 0
        logger.info(f"Starting UF download task. Total files: {total}, Save Dir: {self.save_dir}")
        
        for i, f in enumerate(self.files):
            self.signals.progress.emit(f['filename'], i + 1, total)
            base_url = "https://apihub.kma.go.kr/api/typ01/url/rdr_file_down.php"
            params = {
                "rdr": "UF",
                "stn": f['station'],
                "qcd": 2,
                "tm": f['datetime'],
                "authKey": self.auth_key
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30, stream=True)
                if response.status_code == 200:
                    save_path = os.path.join(self.save_dir, f['filename'])
                    with open(save_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    success += 1
                    logger.debug(f"Successfully downloaded {f['filename']}")
                else:
                    logger.error(f"Failed to download {f['filename']}. HTTP {response.status_code}")
                    self.signals.error.emit(f['filename'], f"HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"Exception downloading {f['filename']}: {e}")
                self.signals.error.emit(f['filename'], str(e))
        
        logger.info(f"UF download task finished. Success: {success}/{total}")
        self.signals.finished.emit(success, total)


# --- 재사용 가능한 파라미터 라벨 위젯 ---
class RadarInfoLabels(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        self.lbls = {}
        keys = ["Status", "Target", "Display", "Lat/Lon", "Alt", "Vol/Sw", "Elev", "nray", "nbin", "Shape", "Gate", "Width", "Nyq"]
        for i, k in enumerate(keys):
            lbl_key = QLabel(f"{k}:")
            lbl_key.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_val = QLabel("-")
            layout.addWidget(lbl_key, i, 0)
            layout.addWidget(lbl_val, i, 1)
            self.lbls[k] = lbl_val
            
    def update_data(self, azimuths, radar_array, meta, target_dt):
        time_display = target_dt.strftime("%Y-%m-%d %H:%M KST")
        self.lbls["Status"].setText("✅ Success")
        self.lbls["Target"].setText(target_dt.strftime("%Y%m%d%H%M"))
        self.lbls["Display"].setText(time_display)
        self.lbls["Lat/Lon"].setText(f"{meta['lat']} / {meta['lon']}")
        self.lbls["Alt"].setText(f"{meta['ht']} m")
        self.lbls["Vol/Sw"].setText(f"{meta['Vn']} / {meta['sw']}")
        self.lbls["Elev"].setText(f"{meta['evel']}°")
        self.lbls["nray"].setText(str(len(azimuths)))
        self.lbls["nbin"].setText(str(radar_array.shape[1]))
        self.lbls["Shape"].setText(str(radar_array.shape))
        self.lbls["Gate"].setText(f"{meta['gate']}m / {meta['bin1']}m")
        self.lbls["Width"].setText(f"{meta['width']}°")
        self.lbls["Nyq"].setText(f"{meta['nyq_vel']} m/s")
        
    def update_status(self, msg):
        self.lbls["Status"].setText(msg)


# --- ALL 모드용 통합 파라미터 패널 ---
class ParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)
        
        stations = ["DJK", "MIL", "SRI"]
        
        for c, stn in enumerate(stations):
            lbl = QLabel(f"■ {stn} Info")
            lbl.setStyleSheet("font-weight: bold; font-size: 13px; border-bottom: 2px solid #555; padding-bottom: 3px;")
            layout.addWidget(lbl, 0, c)
            
        keys = [
            ("Status", "Status: "), ("Target", "Tgt: "), ("Display", "Disp: "),
            ("Lat/Lon", "Lat/Lon: "), ("Alt", "Alt: "), ("Vol/Sw", "V/S: "),
            ("Elev", "Elev: "), ("nray", "nray: "), ("nbin", "nbin: "),
            ("Shape", "Shape: "), ("Gate", "Gate: "), ("Width", "Width: "), ("Nyq", "Nyq: ")
        ]
        
        self.labels = {stn: {} for stn in stations}
        
        for r, (k, prefix) in enumerate(keys, 1):
            for c, stn in enumerate(stations):
                lbl = QLabel(f"{prefix}-")
                lbl.setStyleSheet("font-size: 11px; padding: 0px;")
                layout.addWidget(lbl, r, c)
                self.labels[stn][k] = lbl

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_data(self, station, azimuths, radar_array, meta, target_dt):
        if station in self.labels:
            time_display = target_dt.strftime("%Y-%m-%d %H:%M KST")
            self.labels[station]["Status"].setText("Status: ✅")
            self.labels[station]["Target"].setText(f"Tgt: {target_dt.strftime('%Y%m%d%H%M')}")
            self.labels[station]["Display"].setText(f"Disp: {time_display}")
            self.labels[station]["Lat/Lon"].setText(f"Lat/Lon: {meta['lat']} / {meta['lon']}")
            self.labels[station]["Alt"].setText(f"Alt: {meta['ht']} m")
            self.labels[station]["Vol/Sw"].setText(f"V/S: {meta['Vn']} / {meta['sw']}")
            self.labels[station]["Elev"].setText(f"Elev: {meta['evel']}°")
            self.labels[station]["nray"].setText(f"nray: {len(azimuths)}")
            self.labels[station]["nbin"].setText(f"nbin: {radar_array.shape[1]}")
            self.labels[station]["Shape"].setText(f"Shape: {radar_array.shape}")
            self.labels[station]["Gate"].setText(f"Gate: {meta['gate']}m / {meta['bin1']}m")
            self.labels[station]["Width"].setText(f"Width: {meta['width']}°")
            self.labels[station]["Nyq"].setText(f"Nyq: {meta['nyq_vel']} m/s")
            
    def update_status(self, station, msg):
        if station in self.labels:
            self.labels[station]["Status"].setText(f"Status: {msg}")


# --- 개별 스테이션 뷰어 위젯 ---
class StationViewerWidget(QWidget):
    def __init__(self, station_name):
        super().__init__()
        self.station = station_name
        self.cbar = None  
        self.mesh = None
        self.current_mode = "PPI"
        self.plan_view_radius_km = 90.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 5))
        self.figure.subplots_adjust(left=0.08, right=0.86, bottom=0.08, top=0.92)
        self.figure.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        
        self.ax.tick_params(colors='#e0e0e0')
        for spine in self.ax.spines.values():
            spine.set_color('#e0e0e0')
        self.ax.xaxis.label.set_color('#e0e0e0')
        self.ax.yaxis.label.set_color('#e0e0e0')
        self.ax.title.set_color('#e0e0e0')
        
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('E-W (km)')
        self.ax.set_ylabel('N-S (km)')
        self.ax.set_title(f"{self.station} Initializing...")
        self.ax.grid(True, alpha=0.3, color='#555555')

        self.overlay = QFrame(self)
        self.overlay.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 20, 220);
                color: #FFFFFF;
                border-radius: 5px;
            }
            QLabel {
                color: #FFFFFF;
                background: transparent;
                font-size: 11px;
            }
        """)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(10, 10, 10, 10)
        overlay_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.info_labels = RadarInfoLabels()
        overlay_layout.addWidget(self.info_labels)
        
        self.overlay.hide()
        self.update_overlay_size()

    def _style_axes(self):
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='#e0e0e0')
        for spine in self.ax.spines.values():
            spine.set_color('#e0e0e0')
        self.ax.xaxis.label.set_color('#e0e0e0')
        self.ax.yaxis.label.set_color('#e0e0e0')
        self.ax.title.set_color('#e0e0e0')
        self.ax.grid(True, alpha=0.3, color='#555555')

    def _reset_plan_view(self, mode):
        self.ax.clear()
        self._style_axes()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('E-W (km)')
        self.ax.set_ylabel('N-S (km)')
        self._lock_plan_view_limits()

    def _lock_plan_view_limits(self):
        self.ax.set_xlim(-self.plan_view_radius_km, self.plan_view_radius_km)
        self.ax.set_ylim(-self.plan_view_radius_km, self.plan_view_radius_km)
        self.ax.set_autoscale_on(False)

    def _reset_rhi_view(self):
        self.ax.clear()
        self._style_axes()
        self.ax.set_aspect('auto')
        self.ax.set_xlabel('Distance (km)')
        self.ax.set_ylabel('Height (km)')
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        self.ax.set_autoscale_on(False)

    def _lock_rhi_view_limits(self, max_r, max_h):
        self.ax.set_xlim(0, max_r)
        self.ax.set_ylim(0, max(max_h, 10.0))
        self.ax.set_autoscale_on(False)

    def update_overlay_size(self):
        self.overlay.adjustSize()
        fixed_width = max(self.overlay.sizeHint().width(), 280)
        self.overlay.setFixedWidth(fixed_width)
        self.overlay.adjustSize()

    def resizeEvent(self, event):
        self.overlay.move(10, 10)
        self.overlay.raise_()
        super().resizeEvent(event)

    def _init_plot(self, mode):
        """플롯 초기화 (모드 전환 또는 최초 생성 시 호출)"""
        if self.mesh is not None:
            self.mesh.remove()
            self.mesh = None
            
        self.current_mode = mode
        if mode == "PPI" or mode == "CAPPI":
            self._reset_plan_view(mode)
        else: # RHI
            self._reset_rhi_view()

    def _rebuild_mesh_needed(self, mode, data_shape):
        """현재 메쉬를 재사용할 수 있는지 판단한다."""
        shape_attr = f"_{mode.lower()}_data_shape"
        cached_shape = getattr(self, shape_attr, None)
        return self.mesh is None or self.current_mode != mode or cached_shape != data_shape

    def update_data_ppi(self, azimuths, radar_array, meta, target_dt):
        """PPI 모드 데이터 시각화 (최적화 및 좌표계 보정)"""
        logger.debug(f"[{self.station}] Updating PPI data. Array shape: {radar_array.shape}")
        theta = np.radians(azimuths)
        
        # 거리 엣지 계산
        r_edges = (75 + 150 * np.arange(radar_array.shape[1] + 1)) / 1000.0 
        
        # 각도 엣지 계산 (원형 데이터 매끄러운 연결)
        if len(theta) > 1:
            dtheta = np.diff(theta).mean()
            theta_edges = np.concatenate([[theta[0] - dtheta/2], theta + dtheta/2])
            theta_edges = np.where(theta_edges < 0, theta_edges + 2*np.pi, theta_edges)
        else:
            theta_edges = theta
            
        X = r_edges[np.newaxis, :] * np.sin(theta_edges[:, np.newaxis])
        Y = r_edges[np.newaxis, :] * np.cos(theta_edges[:, np.newaxis])

        if self._rebuild_mesh_needed("PPI", radar_array.shape):
            self._init_plot("PPI")
            cmap = matplotlib.cm.turbo.copy()
            cmap.set_bad(color='#1e1e1e')
            self.mesh = self.ax.pcolormesh(X, Y, radar_array, cmap=cmap, vmin=0, vmax=60, shading='auto')
            self._ppi_data_shape = radar_array.shape
            
            if self.cbar is None:
                self.cbar = self.figure.colorbar(self.mesh, ax=self.ax, label='dBZ')
                self.cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
                self.cbar.ax.yaxis.label.set_color('#e0e0e0')
                for label in self.cbar.ax.get_yticklabels():
                    label.set_color('#e0e0e0')
            else:
                self.cbar.update_normal(self.mesh)
        else:
            self.mesh.set_array(radar_array.ravel())

        self._lock_plan_view_limits()

        self.canvas.draw_idle()

        time_display = target_dt.strftime("%Y-%m-%d %H:%M KST")
        self.info_labels.update_data(azimuths, radar_array, meta, target_dt)
        self.update_overlay_size()
        self.ax.set_title(f"{self.station} PPI ({time_display})")

    def update_data_rhi(self, sweeps_data, target_azimuth, target_dt):
        """RHI 모드 데이터 시각화 (지구 곡률 반영 및 스케일 최적화)"""
        logger.debug(f"[{self.station}] Updating RHI data. Sweeps: {len(sweeps_data)}")
        
        sorted_sweeps = sorted(sweeps_data.items(), key=lambda item: float(item[1][2]['evel']))
        if not sorted_sweeps: return

        max_bins = 0
        for sw, (az, arr, meta) in sorted_sweeps:
            if arr.shape[1] > max_bins: max_bins = arr.shape[1]
                
        r_centers = (150 + 150 * np.arange(max_bins)) / 1000.0 
        
        rhi_matrix, X_grid, Y_grid = [], [], []
        
        for sw, (az, arr, meta) in sorted_sweeps:
            try: 
                elev = float(meta['evel'])
            except ValueError: 
                continue
                
            # 가장 가까운 방위각 Ray 찾기
            az_diff = np.abs(az - target_azimuth)
            az_diff = np.minimum(az_diff, 360 - az_diff)
            closest_idx = np.argmin(az_diff)
            
            reflectivity = arr[closest_idx, :]
            if len(reflectivity) < max_bins:
                pad_len = max_bins - len(reflectivity)
                reflectivity = np.pad(reflectivity, (0, pad_len), 'constant', constant_values=np.nan)
                
            rhi_matrix.append(reflectivity)
            X_grid.append(r_centers)
            
            # 4/3 지구 반경 모델 적용 (표준 대기 굴절)
            h = r_centers * np.sin(np.radians(elev)) + (r_centers**2) / (2 * 8500)
            Y_grid.append(h)
            
        rhi_matrix = np.array(rhi_matrix) 
        X_grid = np.array(X_grid)          
        Y_grid = np.array(Y_grid)          
        
        if self._rebuild_mesh_needed("RHI", rhi_matrix.shape):
            self._init_plot("RHI")
            cmap = matplotlib.cm.turbo.copy()
            cmap.set_bad(color='#1e1e1e')
            self.mesh = self.ax.pcolormesh(X_grid, Y_grid, rhi_matrix, cmap=cmap, vmin=0, vmax=60, shading='nearest')
            self._rhi_data_shape = rhi_matrix.shape
            
            # RHI 축 범위 설정: 가로(거리)는 0~최대거리, 세로(고도)는 0~최대고도(최소 10km 보장)
            max_r = np.nanmax(X_grid) if X_grid.size > 0 else 150.0
            max_h = np.nanmax(Y_grid) if Y_grid.size > 0 else 10.0
            self._lock_rhi_view_limits(max_r, max_h)
            
            if self.cbar is None:
                self.cbar = self.figure.colorbar(self.mesh, ax=self.ax, label='dBZ')
                self.cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
                self.cbar.ax.yaxis.label.set_color('#e0e0e0')
                for label in self.cbar.ax.get_yticklabels(): label.set_color('#e0e0e0')
            else:
                self.cbar.update_normal(self.mesh)
        else:
            self.mesh.set_array(rhi_matrix.ravel())
            max_r = np.nanmax(X_grid) if X_grid.size > 0 else 150.0
            max_h = np.nanmax(Y_grid) if Y_grid.size > 0 else 10.0
            self._lock_rhi_view_limits(max_r, max_h)

        self.canvas.draw_idle()

        time_display = target_dt.strftime("%Y-%m-%d %H:%M KST")
        self.info_labels.update_status(f"✅ RHI Az:{target_azimuth}°")
        self.info_labels.lbls["Display"].setText(time_display)
        self.update_overlay_size()
        self.ax.set_title(f"{self.station} RHI (Az: {target_azimuth}°, {time_display})")

    def update_data_cappi(self, sweeps_data, target_altitude, target_dt):
        """CAPPI 모드 데이터 시각화 (Numpy 벡터화 및 빔 교차 방지)"""
        logger.debug(f"[{self.station}] Updating CAPPI data. Sweeps: {len(sweeps_data)}")
        
        sorted_sweeps = sorted(sweeps_data.items(), key=lambda item: float(item[1][2]['evel']))
        if not sorted_sweeps: return

        max_bins = 0
        for sw, (az, arr, meta) in sorted_sweeps:
            if arr.shape[1] > max_bins: max_bins = arr.shape[1]
                
        r_centers = (150 + 150 * np.arange(max_bins)) / 1000.0 
        nray = sorted_sweeps[0][1][0].shape[0]
        
        Z_sw = np.full((len(sorted_sweeps), nray, max_bins), np.nan)
        H_sw = np.zeros((len(sorted_sweeps), max_bins))
        theta = None
        
        for i, (sw, (az, arr, meta)) in enumerate(sorted_sweeps):
            elev = float(meta['evel'])
            # 4/3 지구 반경 모델 적용
            H_sw[i, :] = r_centers * np.sin(np.radians(elev)) + (r_centers**2) / (2 * 8500)
            current_nray, current_nbins = arr.shape
            if theta is None: theta = np.radians(az)
            
            temp_arr = arr.copy().astype(float)
            Z_sw[i, :current_nray, :current_nbins] = temp_arr
            
        Z_cappi = np.full((nray, max_bins), np.nan)
        
        # 벡터화된 고도 보간 (거리 빈마다 수행)
        for j in range(max_bins):
            heights_at_r = H_sw[:, j]
            valid_mask = ~np.isnan(heights_at_r)
            if np.sum(valid_mask) < 2:
                continue
            
            # 고도순 정렬 (빔 교차 현상 방지)
            sort_idx = np.argsort(heights_at_r[valid_mask])
            h_sorted = heights_at_r[valid_mask][sort_idx]
            z_sorted = Z_sw[valid_mask, :, j][sort_idx, :] # (스윕, 방위각)
            
            # 각 방위각마다 np.interp 수행
            Z_cappi[:, j] = np.apply_along_axis(
                lambda z_col: np.interp(target_altitude, h_sorted, z_col, left=np.nan, right=np.nan),
                0,
                z_sorted
            )
        
        # 시각화를 위한 극좌표 -> 직교좌표 변환
        r_edges = (75 + 150 * np.arange(max_bins + 1)) / 1000.0
        
        if len(theta) > 1:
            dtheta = np.diff(theta).mean()
            theta_edges = np.concatenate([[theta[0] - dtheta/2], theta + dtheta/2])
            theta_edges = np.where(theta_edges < 0, theta_edges + 2*np.pi, theta_edges)
        else:
            theta_edges = theta
            
        X = r_edges[np.newaxis, :] * np.sin(theta_edges[:, np.newaxis])
        Y = r_edges[np.newaxis, :] * np.cos(theta_edges[:, np.newaxis])
        
        if self._rebuild_mesh_needed("CAPPI", Z_cappi.shape):
            self._init_plot("CAPPI")
            cmap = matplotlib.cm.turbo.copy()
            cmap.set_bad(color='#1e1e1e')
            self.mesh = self.ax.pcolormesh(X, Y, Z_cappi, cmap=cmap, vmin=0, vmax=60, shading='auto')
            self._cappi_data_shape = Z_cappi.shape
            
            if self.cbar is None:
                self.cbar = self.figure.colorbar(self.mesh, ax=self.ax, label='dBZ')
                self.cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
                self.cbar.ax.yaxis.label.set_color('#e0e0e0')
                for label in self.cbar.ax.get_yticklabels(): label.set_color('#e0e0e0')
            else:
                self.cbar.update_normal(self.mesh)
        else:
            self.mesh.set_array(Z_cappi.ravel())

        self._lock_plan_view_limits()

        self.canvas.draw_idle()

        time_display = target_dt.strftime("%Y-%m-%d %H:%M KST")
        self.info_labels.update_status(f"✅ CAPPI Alt:{target_altitude}km")
        self.info_labels.lbls["Display"].setText(time_display)
        self.update_overlay_size()
        self.ax.set_title(f"{self.station} CAPPI (Alt: {target_altitude}km, {time_display})")

    def update_status(self, msg):
        self.info_labels.update_status(msg)
        self.update_overlay_size()
        self.ax.set_title(f"{self.station} - {msg}")


# --- 인증키 입력 다이얼로그 ---
class AuthKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("기상청 API 인증키 입력")
        self.resize(450, 150)
        
        layout = QVBoxLayout(self)
        
        lbl = QLabel("기상청 APIHub 인증키를 입력하세요.\n(입력된 키는 config.xml에 저장되어 다음 실행부터 자동 로드됩니다.)")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        
        self.txt_auth = QLineEdit()
        self.txt_auth.setPlaceholderText("인증키를 여기에 붙여넣기 하세요.")
        layout.addWidget(self.txt_auth)
        
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("저장 후 시작")
        btn_save.clicked.connect(self.on_save)
        btn_cancel = QPushButton("종료")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def on_save(self):
        key = self.txt_auth.text().strip()
        if key:
            save_auth_key(key)
            self.accept()
        else:
            QMessageBox.warning(self, "경고", "인증키를 입력해주세요.")
            
    def get_auth_key(self):
        return self.txt_auth.text().strip()


# --- UF 다운로드 다이얼로그 ---
class UFDownloadDialog(QDialog):
    def __init__(self, auth_key, parent=None):
        super().__init__(parent)
        self.auth_key = auth_key
        self.setWindowTitle("UF 원본 데이터 다운로드")
        self.resize(800, 600)
        
        self.thread_pool = QThreadPool()
        
        layout = QVBoxLayout(self)
        
        top_ctrl = QHBoxLayout()
        top_ctrl.addWidget(QLabel("조회 날짜:"))
        
        self.date_edit = QDateEdit()
        self.date_edit.setMaximumDate(QDate.currentDate())
        self.date_edit.setDisplayFormat("yyyyMMdd")
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        top_ctrl.addWidget(self.date_edit)
        
        self.btn_fetch = QPushButton("목록 조회")
        self.btn_fetch.clicked.connect(self.fetch_list)
        top_ctrl.addWidget(self.btn_fetch)
        
        top_ctrl.addStretch()
        
        self.lbl_status = QLabel("대기 중...")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        top_ctrl.addWidget(self.lbl_status)
        
        layout.addLayout(top_ctrl)
        
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["선택", "파일명", "관측소", "시간", "용량"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)
        
        bottom_ctrl = QHBoxLayout()
        
        self.btn_sel_all = QPushButton("전체 선택")
        self.btn_sel_all.clicked.connect(lambda: self.toggle_all(True))
        bottom_ctrl.addWidget(self.btn_sel_all)
        
        self.btn_deselect_all = QPushButton("선택 해제")
        self.btn_deselect_all.clicked.connect(lambda: self.toggle_all(False))
        bottom_ctrl.addWidget(self.btn_deselect_all)
        
        bottom_ctrl.addStretch()
        
        self.btn_download = QPushButton("선택 항목 다운로드")
        self.btn_download.clicked.connect(self.start_download)
        bottom_ctrl.addWidget(self.btn_download)
        
        layout.addLayout(bottom_ctrl)
        
        # 기존에 선언되어 있던 프로그레스 바 (그대로 유지)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def toggle_all(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item:
                item.setCheckState(state)

    def fetch_list(self):
        # text() 대신 qdate 객체에서 직접 "yyyyMMdd" 문자열을 추출하여 안전하게 날짜를 가져옵니다.
        tm_str = self.date_edit.date().toString("yyyyMMdd")
        self.table.setRowCount(0)
        self.lbl_status.setText(f"{tm_str} 목록 불러오는 중...")
        self.btn_fetch.setEnabled(False)

        # [재탕 1 단계] 기존 프로그레스 바를 켜고, 좌우 무한 왕복(0, 0) 모드로 전환합니다.
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)  # 조회 중에는 퍼센트(%) 텍스트 숨김
        self.progress.setVisible(True)

        task = UFListFetchTask(tm_str, self.auth_key)
        task.signals.result.connect(self.on_list_fetched)
        task.signals.error.connect(self.on_list_error)
        self.thread_pool.start(task)

    def on_list_fetched(self, files):
        # [재탕 2 단계 - 성공] 목록 조회가 끝났으므로 기존 프로그레스 바를 숨깁니다.
        self.progress.setVisible(False)

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(files))
        
        stn_order = {"DJK": 0, "MIL": 1, "SRI": 2}
        files.sort(key=lambda x: (stn_order.get(x['station'], 99), x['datetime']))
        
        for r, f in enumerate(files):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            self.table.setItem(r, 0, chk_item)
            
            self.table.setItem(r, 1, QTableWidgetItem(f['filename']))
            self.table.setItem(r, 2, QTableWidgetItem(f['station']))
            self.table.setItem(r, 3, QTableWidgetItem(f['dt_display']))
            
            size_mb = f['size'] / (1024 * 1024)
            size_item = NumericTableWidgetItem(f"{size_mb:.2f} MB")
            size_item.setData(Qt.UserRole, f['size'])
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 4, size_item)
            
            chk_item.setData(Qt.UserRole, f)

        self.table.setSortingEnabled(True)
        self.lbl_status.setText(f"총 {len(files)}개 파일 검색됨")
        self.btn_fetch.setEnabled(True)

    def on_list_error(self, err_msg):
        # [재탕 2 단계 - 에러] 에러가 발생해도 기존 프로그레스 바를 안전하게 숨깁니다.
        self.progress.setVisible(False)

        self.lbl_status.setText(f"오류: {err_msg}")
        self.btn_fetch.setEnabled(True)
        QMessageBox.critical(self, "오류", err_msg)

    def start_download(self):
        selected_files = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                f_data = item.data(Qt.UserRole)
                if f_data:
                    selected_files.append(f_data)
        
        if not selected_files:
            QMessageBox.information(self, "알림", "다운로드할 파일을 선택하세요.")
            return
            
        save_dir = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if not save_dir:
            return
            
        # [재탕 3 단계] 기존 프로그레스 바를 다시 켜고, 실제 다운로드 개수 범위로 재설정합니다.
        self.progress.setVisible(True)
        self.progress.setRange(0, len(selected_files))  # 실제 파일 개수로 범위 지정
        self.progress.setValue(0)
        self.progress.setTextVisible(True)              # 다운로드 진행률(%) 텍스트 복구
        
        self.lbl_status.setText("다운로드 시작...")
        self.btn_download.setEnabled(False)
        self.btn_fetch.setEnabled(False)
        
        task = UFDownloadTask(selected_files, save_dir, self.auth_key)
        task.signals.progress.connect(self.on_download_progress)
        task.signals.error.connect(self.on_download_error)
        task.signals.finished.connect(self.on_download_finished)
        self.thread_pool.start(task)

    def on_download_progress(self, fname, curr, total):
        self.progress.setValue(curr)
        self.lbl_status.setText(f"다운로드 중... ({curr}/{total}) - {fname}")

    def on_download_error(self, fname, err):
        logger.error(f"Download error for {fname}: {err}")

    def on_download_finished(self, success, total):
        self.lbl_status.setText(f"다운로드 완료: 성공 {success}건 / 실패 {total - success}건")
        self.progress.setVisible(False)
        self.btn_download.setEnabled(True)
        self.btn_fetch.setEnabled(True)
        QMessageBox.information(self, "완료", f"총 {total}건 중 {success}건 다운로드 완료.")

class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        val1 = self.data(Qt.UserRole)
        val2 = other.data(Qt.UserRole)
        if val1 is not None and val2 is not None:
            return val1 < val2
        return super().__lt__(other)


# --- 메인 윈도우 ---
class RadarLiveViewer(QMainWindow):
    def __init__(self, auth_key):
        super().__init__()
        logger.info("Initializing RadarLiveViewer...")
        self.setWindowTitle(f"KART: KMA Analysis Radar Tool - {Intended_Use}")
        self.resize(1400, 1000)

        self.auth_key = auth_key
        
        self.stations_1min = ["DJK", "MIL", "SRI"]
        self.stations_5min = ["BRI", "BSL", "GAS", "GDK", "GNG", "GRS", "GSN", "IIA", "JNI", "KSN", "KWK", "MHS", "MYN", "PSN", "SBS", "SDS", "SSP", "YBS"]
        
        self.active_stations = ["MIL"]
        
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(10)
        logger.debug(f"Thread pool max count: {self.thread_pool.maxThreadCount()}")

        self.inf_cache = {}
        self.vol_data_cache = {} 

        self.central_widget = QWidget()
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- 상단 컨트롤 바 ---
        ctrl_group = QGroupBox("Control Panel")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.lbl_sys_time = QLabel("System Time: -")
        self.lbl_sys_time.setStyleSheet("font-size: 14px; font-weight: bold; padding: 0 20px;")
        ctrl_layout.addWidget(self.lbl_sys_time)
        
        self.lbl_next_run = QLabel("Next Check: -")
        self.lbl_next_run.setStyleSheet("font-size: 13px; padding: 0 20px;")
        ctrl_layout.addWidget(self.lbl_next_run)

        ctrl_layout.addStretch()

        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems(["PPI", "RHI", "CAPPI"])
        self.combo_view_mode.currentIndexChanged.connect(self.on_view_mode_changed)
        ctrl_layout.addWidget(QLabel("View Mode:"))
        ctrl_layout.addWidget(self.combo_view_mode)

        self.combo_station = QComboBox()
        self.combo_station.addItem("MIL (1분)", "MIL")
        for stn in self.stations_1min:
            if stn != "MIL":
                self.combo_station.addItem(f"{stn} (1분)", stn)
                
        self.combo_station.insertSeparator(self.combo_station.count())
        for stn in self.stations_5min:
            self.combo_station.addItem(f"{stn} (5분)", stn)
            
        self.combo_station.insertSeparator(self.combo_station.count())
        self.combo_station.addItem("ALL (1분: DJK, MIL, SRI)", "ALL")
        
        self.combo_station.currentIndexChanged.connect(self.on_station_changed)
        ctrl_layout.addWidget(QLabel("Station:"))
        ctrl_layout.addWidget(self.combo_station)

        self.combo_vol = QComboBox()
        self.combo_vol.setEnabled(False)
        self.combo_vol.currentIndexChanged.connect(self.on_vol_changed)
        ctrl_layout.addWidget(QLabel("Volume:"))
        ctrl_layout.addWidget(self.combo_vol)

        self.lbl_sw = QLabel("Sweep:")
        ctrl_layout.addWidget(self.lbl_sw)
        self.combo_sw = QComboBox()
        self.combo_sw.setEnabled(False)
        self.combo_sw.currentIndexChanged.connect(self.on_sw_changed)
        ctrl_layout.addWidget(self.combo_sw)

        self.lbl_az = QLabel("Azimuth(°):")
        self.lbl_az.hide()
        ctrl_layout.addWidget(self.lbl_az)
        self.spin_az = QDoubleSpinBox()
        self.spin_az.setRange(0.0, 360.0)
        self.spin_az.setValue(90.0)
        self.spin_az.setSingleStep(1.0)
        self.spin_az.hide()
        self.spin_az.valueChanged.connect(self.update_view_only)
        ctrl_layout.addWidget(self.spin_az)

        self.lbl_alt = QLabel("Altitude(km):")
        self.lbl_alt.hide()
        ctrl_layout.addWidget(self.lbl_alt)
        self.spin_alt = QDoubleSpinBox()
        self.spin_alt.setRange(0.1, 10.0)
        self.spin_alt.setValue(1.0)
        self.spin_alt.setSingleStep(0.1)
        self.spin_alt.hide()
        self.spin_alt.valueChanged.connect(self.update_view_only)
        ctrl_layout.addWidget(self.spin_alt)

        self.btn_apply = QPushButton("즉시 적용 (Apply)")
        self.btn_apply.clicked.connect(self.manual_refresh)
        ctrl_layout.addWidget(self.btn_apply)

        self.btn_uf_download = QPushButton("UF 다운로드")
        self.btn_uf_download.clicked.connect(self.open_uf_dialog)
        ctrl_layout.addWidget(self.btn_uf_download)
        
        self.btn_change_auth = QPushButton("인증키 변경")
        self.btn_change_auth.clicked.connect(self.change_auth_key)
        ctrl_layout.addWidget(self.btn_change_auth)

        self.btn_credit = QPushButton("Credit")
        self.btn_credit.setFixedWidth(72)
        self.btn_credit.clicked.connect(self.show_credit)
        ctrl_layout.addWidget(self.btn_credit)

        self.main_layout.addWidget(ctrl_group)

        # --- 레이더 뷰어 컨테이너 ---
        self.viewer_container = QWidget()
        self.viewer_layout = QGridLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viewer_layout.setRowStretch(0, 1)
        self.viewer_layout.setRowStretch(1, 1)
        self.viewer_layout.setColumnStretch(0, 1)
        self.viewer_layout.setColumnStretch(1, 1)
        
        self.main_layout.addWidget(self.viewer_container, stretch=1)

        self.station_widgets = {}
        self.param_panel = ParamPanel()
        
        self.inf_results = {}
        self.inf_errors = 0
        self.inf_expected = 0
        self.target_tm_str = ""
        self.current_qcd_val = 0
        self.current_sweeps_per_vol = {}
        
        self.vol_render_cache = {} 
        self.vol_expected_sweeps = {}
        
        self.rebuild_viewer_ui()

        self.fetch_timer = QTimer()
        self.fetch_timer.setSingleShot(True)
        self.fetch_timer.timeout.connect(self.check_new_data)

        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_sys_time)
        self.sys_timer.start(1000)
        self.update_sys_time()

        logger.info("Initialization complete. Triggering first data check.")
        self.check_new_data()

    def get_current_station_code(self):
        return self.combo_station.currentData()

    def change_auth_key(self):
        dialog = AuthKeyDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.auth_key = dialog.get_auth_key()
            QMessageBox.information(self, "알림", "인증키가 변경되었습니다. 데이터를 다시 불러옵니다.")
            self.check_new_data()

    def open_uf_dialog(self):
        dialog = UFDownloadDialog(self.auth_key, self)
        dialog.exec()

    def show_credit(self):
        credit()

    def rebuild_viewer_ui(self):
        logger.debug("Rebuilding viewer UI...")
        for i in reversed(range(self.viewer_layout.count())): 
            w = self.viewer_layout.itemAt(i).widget()
            if w is not None:
                self.viewer_layout.removeWidget(w)
                w.hide()
                w.setParent(None)

        selected = self.get_current_station_code()
        
        if selected == "ALL":
            self.active_stations = self.stations_1min
            for stn in self.active_stations:
                if stn not in self.station_widgets:
                    self.station_widgets[stn] = StationViewerWidget(stn)
            
            self.viewer_layout.addWidget(self.station_widgets["DJK"], 0, 0)
            self.viewer_layout.addWidget(self.station_widgets["MIL"], 0, 1)
            self.viewer_layout.addWidget(self.station_widgets["SRI"], 1, 0)
            self.viewer_layout.addWidget(self.param_panel, 1, 1)
            
            for stn in self.active_stations:
                self.station_widgets[stn].overlay.hide()
                self.station_widgets[stn].show()
                self.station_widgets[stn].canvas.draw_idle()
            
            self.param_panel.show()
        else:
            self.active_stations = [selected]
            if selected not in self.station_widgets:
                self.station_widgets[selected] = StationViewerWidget(selected)
                
            self.viewer_layout.addWidget(self.station_widgets[selected], 0, 0, 2, 2)
            
            self.station_widgets[selected].overlay.show()
            self.station_widgets[selected].show()
            self.station_widgets[selected].canvas.draw_idle()

    def on_view_mode_changed(self):
        mode = self.combo_view_mode.currentText()
        is_ppi = mode == "PPI"
        is_rhi = mode == "RHI"
        is_cappi = mode == "CAPPI"
        
        if (is_rhi or is_cappi) and self.get_current_station_code() == "ALL":
            QMessageBox.information(self, "알림", f"{mode} 모드는 단일 관측소에서만 지원됩니다.")
            self.combo_view_mode.blockSignals(True)
            self.combo_view_mode.setCurrentIndex(0)
            self.combo_view_mode.blockSignals(False)
            return
            
        self.lbl_sw.setVisible(is_ppi)
        self.combo_sw.setVisible(is_ppi)
        self.lbl_az.setVisible(is_rhi)
        self.spin_az.setVisible(is_rhi)
        self.lbl_alt.setVisible(is_cappi)
        self.spin_alt.setVisible(is_cappi)
        
        if not self.manual_refresh_from_cache():
            self.manual_refresh()

    def on_station_changed(self):
        self.combo_vol.blockSignals(True)
        self.combo_sw.blockSignals(True)
        self.combo_vol.clear()
        self.combo_sw.clear()
        self.combo_vol.setEnabled(False)
        self.combo_sw.setEnabled(False)
        self.combo_vol.blockSignals(False)
        self.combo_sw.blockSignals(False)
        self.rebuild_viewer_ui()
        self.manual_refresh()

    def update_sys_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lbl_sys_time.setText(f"System Time: {now}")

    def schedule_next_fetch(self):
        mode = self.combo_view_mode.currentText()
        if mode in ["RHI", "CAPPI"]:
            self.fetch_timer.stop()
            self.lbl_next_run.setText(f"Next Check: Manual ({mode} Mode)")
            return

        now = datetime.now()
        selected = self.get_current_station_code()
        
        if selected in self.stations_5min:
            base_minute = (now.minute // 5) * 5
            base_time = now.replace(minute=base_minute, second=0, microsecond=0)
            target_time = base_time + timedelta(minutes=5, seconds=2) if now > base_time + timedelta(seconds=2) else base_time + timedelta(seconds=2)
            label_text = f"Next Check (FQC2 5min): {target_time.strftime('%H:%M:%S')}"
        else:
            next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            target_time = next_minute + timedelta(seconds=2)
            label_text = f"Next Check (1min): {target_time.strftime('%H:%M:%S')}"
        
        if target_time <= now:
            target_time = now + timedelta(seconds=1)
            
        delta = target_time - now
        delay_ms = int(delta.total_seconds() * 1000)
        
        self.lbl_next_run.setText(label_text)
        self.fetch_timer.stop()
        self.fetch_timer.start(delay_ms)

    def manual_refresh(self):
        self.fetch_timer.stop()
        self.check_new_data()

    def manual_refresh_from_cache(self):
        mode = self.combo_view_mode.currentText()
        vol_val = self.combo_vol.currentData()
        target_dt = datetime.strptime(self.target_tm_str, "%Y%m%d%H%M") if self.target_tm_str else None
        
        if not target_dt or vol_val is None:
            return False
            
        if mode in ["RHI", "CAPPI"]:
            all_sweeps_available = True
            for stn in self.active_stations:
                cache_key = (stn, self.target_tm_str, vol_val)
                if cache_key not in self.vol_data_cache or not self.current_sweeps_per_vol.get(vol_val):
                    all_sweeps_available = False
                    break
                for sw in self.current_sweeps_per_vol[vol_val]:
                    if sw not in self.vol_data_cache[cache_key]:
                        all_sweeps_available = False
                        break
            
            if all_sweeps_available:
                target_val = self.spin_az.value() if mode == "RHI" else self.spin_alt.value()
                for stn in self.active_stations:
                    if stn in self.station_widgets:
                        cache_key = (stn, self.target_tm_str, vol_val)
                        if mode == "RHI":
                            self.station_widgets[stn].update_data_rhi(self.vol_data_cache[cache_key], target_val, target_dt)
                        else:
                            self.station_widgets[stn].update_data_cappi(self.vol_data_cache[cache_key], target_val, target_dt)
                return True
        else:
            sw_val = self.combo_sw.currentData()
            if sw_val is None: return False
                
            all_sw_available = True
            for stn in self.active_stations:
                cache_key = (stn, self.target_tm_str, vol_val)
                if cache_key not in self.vol_data_cache or sw_val not in self.vol_data_cache[cache_key]:
                    all_sw_available = False
                    break
            
            if all_sw_available:
                for stn in self.active_stations:
                    if stn in self.station_widgets:
                        cache_key = (stn, self.target_tm_str, vol_val)
                        az, arr, meta = self.vol_data_cache[cache_key][sw_val]
                        self.station_widgets[stn].update_data_ppi(az, arr, meta, target_dt)
                        if self.get_current_station_code() == "ALL":
                            self.param_panel.update_data(stn, az, arr, meta, target_dt)
                return True
        return False

    def update_view_only(self):
        mode = self.combo_view_mode.currentText()
        if mode not in ["RHI", "CAPPI"]:
            return
            
        vol_val = self.combo_vol.currentData()
        if vol_val is None or not self.target_tm_str: return
            
        target_dt = datetime.strptime(self.target_tm_str, "%Y%m%d%H%M")
        target_val = self.spin_az.value() if mode == "RHI" else self.spin_alt.value()
        
        for stn in self.active_stations:
            if stn in self.station_widgets:
                cache_key = (stn, self.target_tm_str, vol_val)
                if cache_key in self.vol_data_cache and self.vol_data_cache[cache_key]:
                    if mode == "RHI":
                        self.station_widgets[stn].update_data_rhi(self.vol_data_cache[cache_key], target_val, target_dt)
                    else:
                        self.station_widgets[stn].update_data_cappi(self.vol_data_cache[cache_key], target_val, target_dt)

    @Slot()
    def check_new_data(self):
        now = datetime.now()
        selected = self.get_current_station_code()
        
        if selected in self.stations_5min:
            floored_minute = (now.minute // 5) * 5
            target_dt = now.replace(minute=floored_minute, second=0, microsecond=0)
            qcd_val = 2
        else:
            target_dt = now.replace(second=0, microsecond=0)
            qcd_val = 0
            
        tm_str = target_dt.strftime("%Y%m%d%H%M")
        
        self.target_tm_str = tm_str
        self.current_qcd_val = qcd_val
        self.inf_results = {}
        self.inf_errors = 0
        self.inf_expected = 0
        
        keys_to_remove = [k for k in self.vol_data_cache.keys() if k[1] < tm_str]
        for k in keys_to_remove:
            del self.vol_data_cache[k]

        if len(self.inf_cache) > 50:
            self.inf_cache.clear()

        pending_inf_stations = []
        
        for stn in self.active_stations:
            if stn not in self.station_widgets: continue
                
            cache_key = (stn, tm_str, qcd_val)
            if cache_key in self.inf_cache:
                self.inf_results[stn] = self.inf_cache[cache_key]
            else:
                pending_inf_stations.append(stn)
                
        self.inf_expected = len(pending_inf_stations)
        
        if self.inf_expected == 0:
            self.update_vol_sw_ui()
            self.fetch_actual_data()
        else:
            for stn in pending_inf_stations:
                widget = self.station_widgets[stn]
                widget.update_status(f"Fetching Info (QCD:{qcd_val})...")
                if selected == "ALL":
                    self.param_panel.update_status(stn, "Fetching Info...")
                
                inf_task = RadarInfoFetchTask(stn, tm_str, self.auth_key, qcd_val)
                inf_task.signals.result.connect(self.on_inf_result)
                inf_task.signals.error.connect(self.on_inf_error)
                self.thread_pool.start(inf_task)

        self.schedule_next_fetch()

    @Slot(str, object, object, str)
    def on_inf_result(self, station, volumes, sweeps_per_vol, tm_str):
        if tm_str != self.target_tm_str: return
            
        cache_key = (station, tm_str, self.current_qcd_val)
        self.inf_cache[cache_key] = (volumes, sweeps_per_vol)
        self.inf_results[station] = (volumes, sweeps_per_vol)
        self.check_inf_complete()

    @Slot(str, str, str)
    def on_inf_error(self, station, error_msg, tm_str):
        if tm_str != self.target_tm_str: return
            
        self.inf_errors += 1
        if station in self.station_widgets:
            self.station_widgets[station].update_status(f"⚠️ Info: {error_msg}")
        if self.get_current_station_code() == "ALL":
            self.param_panel.update_status(station, f"⚠️ Info: {error_msg}")
        self.check_inf_complete()

    def check_inf_complete(self):
        total_received = len(self.inf_results) + self.inf_errors
        if total_received >= self.inf_expected:
            if self.inf_results:
                self.update_vol_sw_ui()
                self.fetch_actual_data()
            else:
                for stn in self.active_stations:
                    if stn in self.station_widgets:
                        self.station_widgets[stn].update_status("⚠️ No Vol/Sw Info")
                    if self.get_current_station_code() == "ALL":
                        self.param_panel.update_status(stn, "⚠️ No Info")

    def update_vol_sw_ui(self):
        common_volumes = None
        common_sweeps = {}
        
        for stn, (vols, swps_per_vol) in self.inf_results.items():
            if common_volumes is None:
                common_volumes = set(vols)
            else:
                common_volumes &= set(vols)
                
            for v, sw_list in swps_per_vol.items():
                if v not in common_sweeps:
                    common_sweeps[v] = set(sw_list)
                else:
                    common_sweeps[v] &= set(sw_list)
                    
        volumes = sorted(list(common_volumes)) if common_volumes else []
        sweeps_per_vol = {v: sorted(list(s)) for v, s in common_sweeps.items() if v in volumes}
        
        current_vol_items = [self.combo_vol.itemText(i) for i in range(self.combo_vol.count())]
        new_vol_items = [str(v) for v in volumes]
        
        if current_vol_items == new_vol_items and self.current_sweeps_per_vol == sweeps_per_vol:
            return
        
        prev_vol = self.combo_vol.currentText()
        prev_sw = self.combo_sw.currentText()
        
        self.combo_vol.blockSignals(True)
        self.combo_vol.clear()
        for v in volumes:
            self.combo_vol.addItem(str(v), v)
            
        vol_idx = self.combo_vol.findText(prev_vol)
        if vol_idx != -1:
            self.combo_vol.setCurrentIndex(vol_idx)
        elif self.combo_vol.count() > 0:
            self.combo_vol.setCurrentIndex(0)
            
        self.current_sweeps_per_vol = sweeps_per_vol
        self.update_sweep_combo(prev_sw)
        self.combo_vol.blockSignals(False)
        
        self.combo_vol.setEnabled(self.combo_vol.count() > 0)
        self.combo_sw.setEnabled(self.combo_sw.count() > 0)

    def update_sweep_combo(self, prev_sw=""):
        self.combo_sw.blockSignals(True)
        self.combo_sw.clear()
        vol_val = self.combo_vol.currentData()
        if vol_val is not None and vol_val in self.current_sweeps_per_vol:
            for s in self.current_sweeps_per_vol[vol_val]:
                self.combo_sw.addItem(str(s), s)
                
            sw_idx = self.combo_sw.findText(prev_sw)
            if sw_idx != -1:
                self.combo_sw.setCurrentIndex(sw_idx)
            elif self.combo_sw.count() > 0:
                self.combo_sw.setCurrentIndex(0)
        self.combo_sw.blockSignals(False)

    def on_vol_changed(self):
        self.update_sweep_combo(self.combo_sw.currentText())
        QTimer.singleShot(100, self.manual_refresh)

    def on_sw_changed(self):
        QTimer.singleShot(100, self.manual_refresh)

    def fetch_actual_data(self):
        if self.combo_vol.count() == 0: return
            
        vol_val = self.combo_vol.currentData()
        tm_str = self.target_tm_str
        qcd_val = self.current_qcd_val
        mode = self.combo_view_mode.currentText()

        if mode in ["RHI", "CAPPI"]:
            sweeps_to_fetch = self.current_sweeps_per_vol.get(vol_val, [])
            if not sweeps_to_fetch: return

            self.vol_render_cache = {stn: {} for stn in self.active_stations}
            self.vol_expected_sweeps = {stn: 0 for stn in self.active_stations}
            
            pending_tasks = []
            for stn in self.active_stations:
                if stn not in self.station_widgets: continue
                    
                cache_key = (stn, tm_str, vol_val)
                if cache_key not in self.vol_data_cache:
                    self.vol_data_cache[cache_key] = {}
                
                widget = self.station_widgets[stn]
                widget.update_status(f"Fetching {mode} V{vol_val}...")

                for sw in sweeps_to_fetch:
                    if sw not in self.vol_data_cache[cache_key]:
                        pending_tasks.append((stn, sw))
                        self.vol_expected_sweeps[stn] += 1
                    else:
                        self.vol_render_cache[stn][sw] = self.vol_data_cache[cache_key][sw]

            if not pending_tasks:
                target_dt = datetime.strptime(tm_str, "%Y%m%d%H%M")
                target_val = self.spin_az.value() if mode == "RHI" else self.spin_alt.value()
                for stn in self.active_stations:
                    if stn in self.station_widgets and self.vol_render_cache[stn]:
                        if mode == "RHI":
                            self.station_widgets[stn].update_data_rhi(self.vol_render_cache[stn], target_val, target_dt)
                        else:
                            self.station_widgets[stn].update_data_cappi(self.vol_render_cache[stn], target_val, target_dt)
            else:
                for stn, sw in pending_tasks:
                    task = RadarFetchTask(stn, tm_str, vol_val, sw, self.auth_key, qcd_val)
                    task.signals.result.connect(self.on_fetch_result_volume)
                    task.signals.error.connect(self.on_fetch_error_volume)
                    self.thread_pool.start(task)
        else:
            if self.combo_sw.count() == 0: return

            sw_val = self.combo_sw.currentData()
            pending_tasks = []

            for stn in self.inf_results.keys():
                if stn not in self.station_widgets: continue
                    
                cache_key = (stn, tm_str, vol_val)
                if cache_key not in self.vol_data_cache:
                    self.vol_data_cache[cache_key] = {}
                
                widget = self.station_widgets[stn]
                if sw_val in self.vol_data_cache[cache_key]:
                    az, arr, meta = self.vol_data_cache[cache_key][sw_val]
                    target_dt = datetime.strptime(tm_str, "%Y%m%d%H%M")
                    widget.update_data_ppi(az, arr, meta, target_dt)
                    if self.get_current_station_code() == "ALL":
                        self.param_panel.update_data(stn, az, arr, meta, target_dt)
                else:
                    widget.update_status(f"Fetching V{vol_val}/S{sw_val}...")
                    if self.get_current_station_code() == "ALL":
                        self.param_panel.update_status(stn, f"Fetching V{vol_val}/S{sw_val}...")
                    pending_tasks.append(stn)

            for stn in pending_tasks:
                task = RadarFetchTask(stn, tm_str, vol_val, sw_val, self.auth_key, qcd_val)
                task.signals.result.connect(self.on_fetch_result_ppi)
                task.signals.error.connect(self.on_fetch_error_ppi)
                self.thread_pool.start(task)

    @Slot(str, object, object, object, datetime)
    def on_fetch_result_ppi(self, station, azimuths, radar_array, meta, target_dt):
        if target_dt.strftime("%Y%m%d%H%M") != self.target_tm_str: return
            
        vol_val = self.combo_vol.currentData()
        sw_val = self.combo_sw.currentData()
        cache_key = (station, self.target_tm_str, vol_val)
        if cache_key not in self.vol_data_cache:
            self.vol_data_cache[cache_key] = {}
        self.vol_data_cache[cache_key][sw_val] = (azimuths, radar_array, meta)
        
        if station in self.station_widgets:
            self.station_widgets[station].update_data_ppi(azimuths, radar_array, meta, target_dt)
        if self.get_current_station_code() == "ALL":
            self.param_panel.update_data(station, azimuths, radar_array, meta, target_dt)

    @Slot(str, str, str)
    def on_fetch_error_ppi(self, station, error_msg, tm_str):
        if tm_str != self.target_tm_str: return
        if station in self.station_widgets:
            self.station_widgets[station].update_status(f"⚠️ {error_msg}")
        if self.get_current_station_code() == "ALL":
            self.param_panel.update_status(station, f"⚠️ {error_msg}")

    @Slot(str, object, object, object, datetime)
    def on_fetch_result_volume(self, station, azimuths, radar_array, meta, target_dt):
        if target_dt.strftime("%Y%m%d%H%M") != self.target_tm_str: return

        vol_val = self.combo_vol.currentData()
        sw_val = meta['sw']
        cache_key = (station, self.target_tm_str, vol_val)
        if cache_key not in self.vol_data_cache:
            self.vol_data_cache[cache_key] = {}
        self.vol_data_cache[cache_key][sw_val] = (azimuths, radar_array, meta)
        
        if station not in self.vol_render_cache:
            self.vol_render_cache[station] = {}
        self.vol_render_cache[station][sw_val] = (azimuths, radar_array, meta)
        
        if len(self.vol_render_cache[station]) >= self.vol_expected_sweeps.get(station, 999):
            mode = self.combo_view_mode.currentText()
            target_val = self.spin_az.value() if mode == "RHI" else self.spin_alt.value()
            
            if station in self.station_widgets:
                if mode == "RHI":
                    self.station_widgets[station].update_data_rhi(self.vol_render_cache[station], target_val, target_dt)
                else:
                    self.station_widgets[station].update_data_cappi(self.vol_render_cache[station], target_val, target_dt)

    @Slot(str, str, str)
    def on_fetch_error_volume(self, station, error_msg, tm_str):
        if tm_str != self.target_tm_str: return
            
        if station in self.vol_expected_sweeps:
            self.vol_expected_sweeps[station] -= 1
            
            if len(self.vol_render_cache.get(station, {})) >= self.vol_expected_sweeps[station]:
                mode = self.combo_view_mode.currentText()
                target_val = self.spin_az.value() if mode == "RHI" else self.spin_alt.value()
                if station in self.station_widgets and self.vol_render_cache.get(station):
                    target_dt = datetime.strptime(tm_str, "%Y%m%d%H%M")
                    if mode == "RHI":
                        self.station_widgets[station].update_data_rhi(self.vol_render_cache[station], target_val, target_dt)
                    else:
                        self.station_widgets[station].update_data_cappi(self.vol_render_cache[station], target_val, target_dt)
                else:
                    self.station_widgets[station].update_status(f"⚠️ {mode} Fail")

def credit():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowTitle("KART 정보")
    
    # 볼드체로 크게 보여줄 메인 타이틀
    msg.setText("KART: KMA Analysis Radar Tool")
    
    # 아래쪽에 정렬되어 들어갈 상세 정보
    msg.setInformativeText(
        f"개발자: 조성헌\n"
        f"Division: AVUX LAB\n"
        f"Version: {KART_VERSION}\n"
        f"Build Date: {BUILD_DATE}\n"
        f"Build Serial: {BUILD_SERIAL}\n\n"
        f"Intended Use: {usage_disclaimer}\n\n"
        f"Copyright (c) 2026 조성헌 & AVUX LAB. Licensed under the MIT License.\n\n"
        f"Disclaimer: This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software."
    )
    
    msg.setStandardButtons(QMessageBox.StandardButton.Ok) # PySide6 정석 상수로 변경
    msg.exec()


LOGO = r"""
  ______   __     __  __    __  __    __        __         ______   _______
 /      \ |  \   |  \|  \  |  \|  \  |  \      |  \       /      \ |       \
|  $$$$$$\| $$   | $$| $$  | $$| $$  | $$      | $$      |  $$$$$$\| $$$$$$$\
| $$__| $$| $$   | $$| $$  | $$ \$$\/  $$      | $$      | $$__| $$| $$__/ $$
| $$    $$ \$$\ /  $$| $$  | $$  >$$  $$       | $$      | $$    $$| $$    $$
| $$$$$$$$  \$$\  $$ | $$  | $$ /  $$$$\       | $$      | $$$$$$$$| $$$$$$$\
| $$  | $$   \$$ $$  | $$__/ $$|  $$ \$$\      | $$_____ | $$  | $$| $$__/ $$
| $$  | $$    \$$$    \$$    $$| $$  | $$      | $$     \| $$  | $$| $$    $$
 \$$   \$$     \$      \$$$$$$  \$$   \$$       \$$$$$$$$ \$$   \$$ \$$$$$$$


                                                                   By ARON CENTRAL
"""
credit_text = f"""

    KART: KMA Analysis Radar Tool
    =============================

    Developer: 조성헌
    Division: AVUX LAB
    Version: {KART_VERSION}
    Build Date: {BUILD_DATE}
    Build Serial: {BUILD_SERIAL}

    Intended Use: {usage_disclaimer}

    Copyright (c) 2026 조성헌 & AVUX LAB. Licensed under the MIT License.

    Disclaimer: This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.
"""
print("="*80)
print(LOGO)
print("="*80)
print(credit_text)
print("="*80)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Consolas", 9)
    app.setFont(font)
    app.setStyleSheet(DARK_QSS)
    
    # 인증키 로드 및 검증
    auth_key = load_auth_key()
    if not auth_key:
        dialog = AuthKeyDialog()
        if dialog.exec() == QDialog.Accepted:
            auth_key = dialog.get_auth_key()
        else:
            logger.info("Application exited: No auth key provided.")
            sys.exit(0)
            
    logger.info("Starting application event loop.")
    viewer = RadarLiveViewer(auth_key)
    viewer.show()
    
    exit_code = app.exec()
    logger.info(f"Application exited with code {exit_code}.")
    sys.exit(exit_code)
