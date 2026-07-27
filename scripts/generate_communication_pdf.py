from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "LeakGuard_Communication_Architecture_KO.pdf"
REGULAR_FONT = Path(r"C:\Windows\Fonts\malgun.ttf")
BOLD_FONT = Path(r"C:\Windows\Fonts\malgunbd.ttf")

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1677B8")
LIGHT_BLUE = colors.HexColor("#EAF4FB")
CYAN = colors.HexColor("#DDF4F2")
GREEN = colors.HexColor("#16836B")
LIGHT_GREEN = colors.HexColor("#E8F5F0")
YELLOW = colors.HexColor("#F7C948")
LIGHT_YELLOW = colors.HexColor("#FFF8DB")
RED = colors.HexColor("#C23B3B")
LIGHT_RED = colors.HexColor("#FCEBEC")
GRAY = colors.HexColor("#52606D")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
BORDER = colors.HexColor("#C7D1DA")


def register_fonts() -> None:
    for path in (REGULAR_FONT, BOLD_FONT):
        if not path.is_file():
            raise FileNotFoundError(f"Korean font not found: {path}")
    pdfmetrics.registerFont(TTFont("Malgun", str(REGULAR_FONT)))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(BOLD_FONT)))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Malgun-Bold",
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Malgun",
            fontSize=12,
            leading=19,
            textColor=GRAY,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "Heading1KO",
            parent=base["Heading1"],
            fontName="Malgun-Bold",
            fontSize=16,
            leading=23,
            textColor=NAVY,
            spaceBefore=4 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2KO",
            parent=base["Heading2"],
            fontName="Malgun-Bold",
            fontSize=12,
            leading=18,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyKO",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=9.4,
            leading=15.5,
            textColor=colors.HexColor("#243B53"),
            spaceAfter=2.2 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "SmallKO",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8,
            leading=12,
            textColor=GRAY,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletKO",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=9.1,
            leading=14.5,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=1 * mm,
            textColor=colors.HexColor("#243B53"),
            spaceAfter=1.2 * mm,
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "CalloutKO",
            parent=base["BodyText"],
            fontName="Malgun-Bold",
            fontSize=10.2,
            leading=16,
            textColor=NAVY,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableKO",
            parent=base["BodyText"],
            fontName="Malgun",
            fontSize=8.3,
            leading=12,
            textColor=colors.HexColor("#243B53"),
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadKO",
            parent=base["BodyText"],
            fontName="Malgun-Bold",
            fontSize=8.4,
            leading=12,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"• {text}", styles["bullet"])


def callout(
    text: str,
    styles: dict[str, ParagraphStyle],
    background: colors.Color = LIGHT_YELLOW,
    border: colors.Color = YELLOW,
) -> Table:
    table = Table([[p(text, styles["callout"])]], colWidths=[169 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 1, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
) -> Table:
    data = [
        [p(escape(value), styles["table_head"]) for value in headers]
    ] + [
        [p(escape(value).replace("\n", "<br/>"), styles["table"]) for value in row]
        for row in rows
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY)
            )
    table.setStyle(TableStyle(commands))
    return table


def code_block(text: str) -> Table:
    code = Preformatted(
        text.strip(),
        ParagraphStyle(
            "Code",
            fontName="Courier",
            fontSize=7.2,
            leading=9.3,
            textColor=colors.HexColor("#E6EDF3"),
            leftIndent=0,
        ),
    )
    table = Table([[code]], colWidths=[169 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#172A3A")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#344E5C")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def architecture_table(styles: dict[str, ParagraphStyle]) -> Table:
    def box(title: str, body: str, bg: colors.Color) -> Table:
        inner = Table(
            [
                [p(title, styles["callout"])],
                [p(body, styles["small"])],
            ],
            colWidths=[39 * mm],
        )
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), bg),
                    ("BOX", (0, 0), (-1, -1), 1, BLUE),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        return inner

    arrow_style = ParagraphStyle(
        "Arrow",
        fontName="Malgun-Bold",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=BLUE,
    )
    table = Table(
        [
            [
                box("Jetson Nano", "Python 3<br/>U-Net + NDJSON", LIGHT_BLUE),
                p("TCP / NDJSON<br/>-&gt;", arrow_style),
                box(
                    "Raspberry Pi",
                    "Python 3 asyncio<br/>Server + 변환",
                    CYAN,
                ),
                p("TCP / ASCII<br/>-&gt;", arrow_style),
                box("STM32", "C<br/>기존 ASCII Parser", LIGHT_GREEN),
            ]
        ],
        colWidths=[39 * mm, 26 * mm, 39 * mm, 26 * mm, 39 * mm],
        hAlign="CENTER",
    )
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return table


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    page_width, page_height = A4
    if doc.page > 1:
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, page_height - 15 * mm, page_width - 20 * mm, page_height - 15 * mm)
        canvas.setFont("Malgun-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(20 * mm, page_height - 11.5 * mm, "LeakGuard 통신 시스템 설계안")
        canvas.setFont("Malgun", 8)
        canvas.setFillColor(GRAY)
        canvas.drawRightString(
            page_width - 20 * mm, page_height - 11.5 * mm, "Team Handoff v1.0"
        )
    canvas.setStrokeColor(BORDER)
    canvas.line(20 * mm, 14 * mm, page_width - 20 * mm, 14 * mm)
    canvas.setFont("Malgun", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(20 * mm, 9.5 * mm, "U-Net 기반 색상 액체 누출 감지 및 원격 안전 제어 시스템")
    canvas.drawRightString(page_width - 20 * mm, 9.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    register_fonts()
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="LeakGuard 분산 통신 시스템 설계안",
        author="LeakGuard AI Team",
        subject="Raspberry Pi, Jetson Nano, STM32 TCP/IP 통신 설계",
    )
    story = []

    # Cover
    story.extend(
        [
            Spacer(1, 24 * mm),
            p("LeakGuard 분산 통신<br/>시스템 설계안", styles["cover_title"]),
            p(
                "Raspberry Pi Server - Jetson Nano Client - STM32 Client",
                styles["cover_subtitle"],
            ),
            Spacer(1, 12 * mm),
            architecture_table(styles),
            Spacer(1, 12 * mm),
            callout(
                "NDJSON은 프로그래밍 언어가 아니며 Node.js를 요구하지 않습니다. "
                "Jetson Nano와 Raspberry Pi는 Python 3, STM32는 기존 C 코드를 "
                "유지하는 구성이 가장 적합합니다.",
                styles,
                LIGHT_YELLOW,
                YELLOW,
            ),
            Spacer(1, 18 * mm),
            data_table(
                ["장치", "권장 언어", "통신 역할"],
                [
                    ["Jetson Nano", "Python 3", "LeakStatus를 NDJSON으로 전송"],
                    ["Raspberry Pi", "Python 3", "TCP 서버, 검증, NDJSON/ASCII 변환"],
                    ["STM32", "C", "기존 ASCII 파서와 안전 FSM 유지"],
                ],
                [44 * mm, 38 * mm, 87 * mm],
                styles,
            ),
            Spacer(1, 18 * mm),
            p("문서 버전 1.0  |  2026-07-27  |  팀 공유용", styles["cover_subtitle"]),
            PageBreak(),
        ]
    )

    # 1. Format and responsibilities
    story.extend(
        [
            p("1. 핵심 아키텍처와 언어 선택", styles["h1"]),
            p(
                "현재 Raspberry Pi - STM32 구간에서 ASCII 기반 C 파서를 개발하고 "
                "있으므로 이를 폐기하고 STM32에 JSON 파서를 새로 넣을 이유가 없습니다. "
                "Raspberry Pi가 두 형식 사이를 변환하면 장치별 장점을 유지할 수 있습니다.",
                styles["body"],
            ),
            callout(
                "권장안: Jetson -> Raspberry Pi는 TCP/NDJSON, Raspberry Pi -> "
                "STM32는 TCP/ASCII를 사용합니다. Raspberry Pi가 중앙 서버와 "
                "프로토콜 변환기를 겸합니다.",
                styles,
                LIGHT_BLUE,
                BLUE,
            ),
            p("1.1 NDJSON의 의미", styles["h2"]),
            p(
                "NDJSON(Newline-Delimited JSON)은 JSON 객체 하나를 한 줄에 하나씩 "
                "전송하고 줄바꿈 문자로 메시지 경계를 표시하는 형식입니다. Python, C, "
                "JavaScript, Java 등 어느 언어에서도 사용할 수 있습니다.",
                styles["body"],
            ),
            code_block(
                '{"v":1,"type":"status","device_id":"jetson-01","seq":152}\n'
                '{"v":1,"type":"heartbeat","device_id":"jetson-01","seq":153}'
            ),
            Spacer(1, 3 * mm),
            p("1.2 화면 문자열을 파싱하지 않는 이유", styles["h2"]),
            p(
                "화면의 'Leak area: 2.364%' 같은 문구는 사람에게 보여주기 위한 "
                "표현입니다. 공백이나 문구가 변경되면 파서가 깨질 수 있습니다. Jetson의 "
                "LeakStatus 객체를 직접 JSON으로 직렬화해야 합니다.",
                styles["body"],
            ),
            code_block(
                "LeakStatus object\n"
                "  -> json.dumps(status_message) + \"\\n\"\n"
                "  -> Raspberry Pi TCP server\n"
                "  -> json.loads(line)\n"
                "  -> STM32 ASCII packet"
            ),
            p("1.3 장치별 책임", styles["h2"]),
            data_table(
                ["장치", "주요 책임", "장애 시 동작"],
                [
                    [
                        "Jetson Nano",
                        "U-Net 추론, 면적/확산/Danger ROI/4단계 판단, 최신 상태 전송",
                        "네트워크가 끊겨도 추론 지속, 자동 재접속",
                    ],
                    [
                        "Raspberry Pi",
                        "다중 TCP 연결, 장치 등록, 검증, 변환, 라우팅, 로그",
                        "한 연결 오류가 서버 전체를 중단하지 않음",
                    ],
                    [
                        "STM32",
                        "ASCII 파싱, 로컬 FSM, 액추에이터와 최종 안전 판단",
                        "통신 타임아웃 시 Fail-safe, DANGER latch 유지",
                    ],
                ],
                [34 * mm, 84 * mm, 51 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # 2. Protocol
    story.extend(
        [
            p("2. 권장 메시지 프로토콜", styles["h1"]),
            p("2.1 연결 등록 - HELLO", styles["h2"]),
            p(
                "모든 클라이언트는 연결 직후 역할, 장치 ID와 프로토콜 버전을 등록해야 "
                "합니다. 서버는 등록되지 않은 연결의 상태 메시지를 거부합니다.",
                styles["body"],
            ),
            code_block(
                "{\n"
                '  "v": 1,\n'
                '  "type": "hello",\n'
                '  "role": "jetson",\n'
                '  "device_id": "jetson-01",\n'
                '  "software_version": "1.0"\n'
                "}"
            ),
            p("2.2 Jetson 상태 - STATUS", styles["h2"]),
            code_block(
                "{\n"
                '  "v": 1, "type": "status", "device_id": "jetson-01",\n'
                '  "seq": 152, "timestamp": 1721800000.250,\n'
                '  "detected": true, "leak_ratio": 2.364,\n'
                '  "instant_ratio": 2.401, "spreading": false,\n'
                '  "spreading_delta": 0.031,\n'
                '  "danger_overlap": false, "danger_overlap_pixels": 0,\n'
                '  "level": "WARNING", "leak_pixels": 4085,\n'
                '  "inference_ms": 24.8, "fps": 18.5\n'
                "}"
            ),
            p("필수 필드와 검증", styles["h2"]),
            data_table(
                ["필드", "의미", "서버 검증"],
                [
                    ["v", "프로토콜 버전", "지원 버전인지 확인"],
                    ["type", "hello/status/heartbeat", "허용 메시지인지 확인"],
                    ["device_id", "장치 고유 ID", "등록된 연결과 일치"],
                    ["seq", "순서 번호", "중복 및 역순 거부"],
                    ["leak_ratio", "Monitoring ROI 면적 비율(%)", "0~100 범위"],
                    ["level", "4단계 상태", "허용된 4개 문자열"],
                    ["timestamp", "상태 생성 시각", "오래된 상태 TTL 검사"],
                ],
                [32 * mm, 72 * mm, 65 * mm],
                styles,
            ),
            p("2.3 Raspberry Pi -> STM32 ASCII", styles["h2"]),
            code_block(
                "LEAK,<seq>,<detected>,<ratio>,<spreading>,<danger>,<level>\\n\n\n"
                "LEAK,152,1,2.364,0,0,WARNING\\n\n"
                "HEARTBEAT,981,1721800000\\n\n"
                "CMD,381,RESET\\n\n"
                "ACK,381,RESET,OK\\n"
            ),
            PageBreak(),
        ]
    )

    # 3. TCP details
    story.extend(
        [
            p("3. TCP 구현에서 놓치기 쉬운 부분", styles["h1"]),
            callout(
                "TCP는 메시지 단위가 아니라 연속된 바이트 스트림입니다. recv() 한 번이 "
                "메시지 하나라고 가정하면 안 됩니다.",
                styles,
                LIGHT_RED,
                RED,
            ),
            p("3.1 수신 framing", styles["h2"]),
            bullet("수신 바이트를 연결별 buffer에 계속 누적합니다.", styles),
            bullet("줄바꿈(\\n)을 찾고 한 줄씩 분리합니다.", styles),
            bullet("완전한 한 줄만 UTF-8 decode와 JSON parse를 수행합니다.", styles),
            bullet("남은 불완전 데이터는 다음 recv()까지 보관합니다.", styles),
            bullet("메시지 최대 크기를 예를 들어 4096 bytes로 제한합니다.", styles),
            code_block(
                "buffer += await reader.read(4096)\n"
                "while b\"\\n\" in buffer:\n"
                "    line, buffer = buffer.split(b\"\\n\", 1)\n"
                "    message = json.loads(line.decode(\"utf-8\"))\n"
                "    validate_and_route(message)"
            ),
            p("3.2 전송 주기와 queue", styles["h2"]),
            data_table(
                ["데이터", "권장 주기", "정책"],
                [
                    ["Jetson STATUS", "5~10 Hz", "최신 상태만 유지"],
                    ["Jetson HEARTBEAT", "1 Hz", "연결 생존 확인"],
                    ["단계 변경 / DANGER", "즉시", "일반 주기를 기다리지 않음"],
                    ["STM32 STATE", "2~10 Hz", "FSM 및 액추에이터 상태"],
                    ["CSV 일반 로그", "1 Hz", "단계 변경은 즉시 기록"],
                ],
                [48 * mm, 35 * mm, 86 * mm],
                styles,
            ),
            p(
                "상태 스트림은 오래된 값을 쌓지 않고 최신값으로 덮어씁니다. RESET 같은 "
                "명령과 단계 변경 이벤트는 별도의 제한된 queue에 넣고 ACK를 요구합니다.",
                styles["body"],
            ),
            p("3.3 재접속", styles["h2"]),
            data_table(
                ["구성 요소", "권장 동작"],
                [
                    [
                        "Jetson",
                        "추론과 통신 thread 분리, 1/2/4/8초 backoff, 재연결 후 HELLO와 최신 상태 전송",
                    ],
                    [
                        "Raspberry Pi",
                        "연결별 task 격리, 마지막 수신 시각 관리, 무응답 장치 OFFLINE 처리",
                    ],
                    [
                        "STM32",
                        "1~2초 상태 미수신 시 COMM_ERROR 후보, 안전 출력으로 전환",
                    ],
                ],
                [38 * mm, 131 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # 4. Safety and security
    story.extend(
        [
            p("4. 안전, 장애 대응과 보안", styles["h1"]),
            p("4.1 Raspberry Pi 단일 장애점", styles["h2"]),
            p(
                "Raspberry Pi가 꺼지면 Jetson 상태가 STM32에 전달되지 않습니다. "
                "따라서 STM32는 서버나 네트워크가 없어도 독립적으로 안전 상태로 "
                "전환해야 합니다.",
                styles["body"],
            ),
            callout(
                "STM32가 최종 안전 책임을 갖습니다. 통신 타임아웃 시 펌프 정지, "
                "밸브 차단, 경고 출력 등 Fail-safe를 수행하고 DANGER latch는 "
                "명시적인 안전 RESET 전까지 유지합니다.",
                styles,
                LIGHT_RED,
                RED,
            ),
            p("선택 가능한 보조 안전 경로", styles["h2"]),
            code_block(
                "Jetson -> Raspberry Pi TCP : monitoring / logging / remote control\n"
                "Jetson -> STM32 UART       : emergency leak / danger signal\n"
                "STM32                     : final safety FSM and actuators"
            ),
            p(
                "UART 보조 경로를 사용하지 않는 경우 Raspberry Pi 전원 차단, LAN 케이블 "
                "분리, 서버 프로세스 강제 종료 시험을 필수 항목으로 포함해야 합니다.",
                styles["body"],
            ),
            p("4.2 네트워크와 보안", styles["h2"]),
            bullet("Raspberry Pi에는 고정 IP 또는 DHCP 예약을 사용합니다.", styles),
            bullet("서버 IP, port, device_id는 설정 파일로 분리합니다.", styles),
            bullet("서버를 인터넷에 직접 노출하지 않습니다.", styles),
            bullet("메시지 종류, 길이, 수치 범위와 문자열 값을 검사합니다.", styles),
            bullet("필요하면 장치별 공유 token 또는 HMAC을 추가합니다.", styles),
            bullet("Raspberry Pi와 Jetson 프로그램을 systemd 서비스로 등록합니다.", styles),
            p("4.3 STM32 네트워크 인터페이스 결정", styles["h2"]),
            p(
                "내장 Ethernet + lwIP, W5500, ESP32/ESP8266 중 어떤 방식을 사용하는지 "
                "확정해야 합니다. 선택에 따라 socket API, RTOS task, 수신 buffer와 "
                "재접속 구현이 달라집니다.",
                styles["body"],
            ),
            data_table(
                ["방식", "고려사항"],
                [
                    ["내장 Ethernet + lwIP", "STM32 native socket, RTOS task 및 timeout 설계"],
                    ["W5500", "하드웨어 socket 수, SPI 처리, 재접속 상태 관리"],
                    ["ESP 계열", "UART 또는 AT 명령 계층, 모듈 재부팅과 Wi-Fi 재접속"],
                ],
                [52 * mm, 117 * mm],
                styles,
            ),
            PageBreak(),
        ]
    )

    # 5. Implementation
    story.extend(
        [
            p("5. 권장 구현 순서", styles["h1"]),
            data_table(
                ["단계", "작업", "완료 기준"],
                [
                    ["1", "프로토콜 v1 필드와 예제 확정", "팀 전체가 동일 문서 승인"],
                    ["2", "Raspberry Pi asyncio TCP 서버", "다중 client 등록과 연결 관리"],
                    ["3", "Python 가짜 Jetson/STM32 client", "장비 없이 양방향 통신"],
                    ["4", "NDJSON/ASCII 변환 테스트", "분할, 결합, 오류 입력 통과"],
                    ["5", "Jetson LeakStatus 연결", "추론과 독립적으로 5~10Hz 전송"],
                    ["6", "STM32 C TCP client 연결", "기존 ASCII parser와 FSM 연동"],
                    ["7", "장애 주입 시험", "재부팅, cable 분리, timeout, 재접속"],
                    ["8", "systemd와 로그", "부팅 자동 실행과 이벤트 추적"],
                ],
                [16 * mm, 75 * mm, 78 * mm],
                styles,
            ),
            p("5.1 권장 폴더 구조", styles["h2"]),
            code_block(
                "Raspberry Pi/\n"
                "  server.py              # asyncio TCP server\n"
                "  config.yaml            # IP, port, timeout\n"
                "  protocol.py            # NDJSON / ASCII codec\n"
                "  client_registry.py     # role and device registry\n"
                "  state_router.py        # Jetson -> STM32 routing\n"
                "  event_logger.py\n"
                "  tests/\n\n"
                "Jetson Nano/leakguard/\n"
                "  tcp_client.py\n"
                "  protocol.py\n\n"
                "STM32/\n"
                "  tcp_client.c\n"
                "  protocol_parser.c\n"
                "  safety_fsm.c"
            ),
            p("5.2 통합 시험 체크리스트", styles["h2"]),
        ]
    )
    checklist = [
        "Jetson과 STM32가 HELLO로 올바르게 등록된다.",
        "4단계 상태와 Danger ROI 이벤트가 STM32까지 전달된다.",
        "TCP 데이터가 분할되거나 합쳐져도 정상 파싱된다.",
        "잘못된 JSON, ASCII, 범위 초과 값이 거부된다.",
        "sequence 중복 및 역순 메시지가 거부된다.",
        "Jetson과 Raspberry Pi 재부팅 후 자동 재접속한다.",
        "STM32 통신 타임아웃과 Fail-safe가 동작한다.",
        "장시간 실행 시 queue와 메모리가 계속 증가하지 않는다.",
        "연결, 해제, 경고 단계 변경이 로그에 기록된다.",
    ]
    checklist_rows = [
        ["□", item] for item in checklist
    ]
    story.extend(
        [
            data_table(
                ["확인", "시험 항목"],
                checklist_rows,
                [18 * mm, 151 * mm],
                styles,
            ),
            Spacer(1, 5 * mm),
            callout(
                "다음 즉시 작업: Raspberry Pi asyncio 서버와 Python 가짜 Jetson, "
                "가짜 STM32 클라이언트를 먼저 구현해 프로토콜을 장비 없이 검증합니다.",
                styles,
                LIGHT_GREEN,
                GREEN,
            ),
            Spacer(1, 6 * mm),
            p(
                "상세 텍스트 문서: docs/communication/"
                "LeakGuard_Communication_Architecture_KO.txt",
                styles["small"],
            ),
        ]
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
