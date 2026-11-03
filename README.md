# mujoco_teleoperation

Ubuntu 데스크톱에서 프로젝트 전체(`assets/`, `pyproject.toml`, `uv.lock`,
`.python-version` 포함)를 받은 뒤 실행합니다. `.venv`는 복사하지 않습니다.

```bash
uv sync --locked
uv run python collect_data.py
```

Python 3.10을 사용하며 instruction 입력창은 PySide6입니다. UI의 Python
의존성은 uv가 함께 설치하므로 Tkinter를 별도로 설치할 필요가 없습니다.
H 키 또는 `Instruction 변경` 버튼으로 입력창을 엽니다. 확인하면 반영되고,
취소하면 기존 값이 유지됩니다. 녹화 중 변경하면 에피소드가 분리됩니다.

화면이 있는 데스크톱 세션과 동작하는 OpenGL 드라이버가 필요합니다.
Ubuntu PC에 Qt/X11 런타임이나 한글 폰트가 없다면 최초 1회 설치합니다.
이 OS 패키지들은 `uv sync`의 관리 대상이 아닙니다.

```bash
sudo apt-get update
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 fonts-noto-cjk
```

Qt의 Linux 시스템 의존성: https://doc.qt.io/qt-6.8/linux-requirements.html
