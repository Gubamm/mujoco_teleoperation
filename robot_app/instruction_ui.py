"""Instruction text editing and Unicode header rendering for the collection window."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER_HEIGHT = 100
_qt_app = None


def edit_instruction(current):
    """Modal dialog; simulation stays paused until Apply/Cancel. None means cancelled."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDialog, QInputDialog

    # Keep the application alive across dialogs while GLFW owns the main loop.
    global _qt_app
    if _qt_app is None:
        _qt_app = QApplication.instance() or QApplication([])
        _qt_app.setQuitOnLastWindowClosed(False)
    dialog = QInputDialog()
    dialog.setWindowTitle('Instruction 변경')
    dialog.setLabelText(
        '현재 작업의 instruction을 입력하세요.\n'
        '녹화 중 변경하면 현재 에피소드를 저장하고 새 에피소드로 이어갑니다.')
    dialog.setInputMode(QInputDialog.InputMode.TextInput)
    dialog.setTextValue(current)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialog.resize(540, 160)
    try:
        return dialog.textValue().strip() if dialog.exec() == QDialog.DialogCode.Accepted else None
    finally:
        dialog.deleteLater()


def font(size):
    for path in (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        'C:/Windows/Fonts/malgun.ttf',
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class InstructionHeader:
    def __init__(self):
        self.title_font = font(13)
        self.text_font = font(21)
        self.button_font = font(16)
        self.cache_key = None
        self.pixels = None
        self.button = None  # Pixel rectangle in top-left image coordinates.

    def image(self, width, instruction):
        key = (width, instruction)
        if key == self.cache_key:
            return self.pixels
        image = Image.new('RGB', (width, HEADER_HEIGHT), '#182a31')
        draw = ImageDraw.Draw(image)
        button_width = min(190, width // 2)
        self.button = (width - button_width - 12, 26, width - 12, 72)
        draw.rounded_rectangle(self.button, radius=8, fill='#50dac5')
        draw.text((self.button[0] + 12, 39), 'Instruction 변경 [H]', font=self.button_font, fill='#08221e')
        draw.text((16, 8), '현재 INSTRUCTION', font=self.title_font, fill='#8cb6bd')
        text = ' '.join(instruction.split()) or '(미설정 — 버튼으로 입력하세요)'
        available = max(20, self.button[0] - 32)
        lines = ['']
        for character in text:
            if draw.textlength(lines[-1] + character, font=self.text_font) > available:
                if len(lines) == 2:
                    while lines[-1] and draw.textlength(lines[-1] + '…', font=self.text_font) > available:
                        lines[-1] = lines[-1][:-1]
                    lines[-1] += '…'
                    break
                lines.append('')
            lines[-1] += character
        for index, line in enumerate(lines):
            draw.text((16, 30 + index * 29), line, font=self.text_font, fill='#e3eef3')
        self.cache_key = key
        self.pixels = np.ascontiguousarray(np.asarray(image)[::-1])
        return self.pixels
