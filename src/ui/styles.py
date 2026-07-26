"""Centralized Qt style sheets for EspectroApp."""

MAIN_WINDOW_STYLE = r"""
QWidget {
    background-color: #F4F7F6;
    color: #12231E;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 14px;
}

QWidget#mainPanel {
    background-color: #073E34;
    border: none;
}

QLabel#appTitle {
    background-color: transparent;
    color: #FFFFFF;
    padding: 12px 8px 22px 8px;
}

QLabel#menuSectionLabel {
    background-color: transparent;
    color: #67D8C0;
    font-size: 11px;
    font-weight: 700;
    padding: 14px 8px 6px 8px;
}

QScrollArea#mainScroll,
QScrollArea#mainScroll QWidget#qt_scrollarea_viewport {
    background-color: #073E34;
    border: none;
}

QFrame#workspace {
    background-color: #FFFFFF;
    border: none;
}

QLabel#workspaceTitle {
    background-color: transparent;
    color: #101F1A;
    font-size: 30px;
    font-weight: 750;
}

QLabel#workspaceSubtitle {
    background-color: transparent;
    color: #485D55;
    font-size: 14px;
    padding-bottom: 8px;
}

QStackedWidget#workspaceStack,
QFrame#workspacePage {
    background-color: #FFFFFF;
    border: none;
}

QLabel#welcomeTitle {
    background-color: transparent;
    color: #101F1A;
    font-size: 23px;
    font-weight: 750;
}

QLabel#welcomeDescription {
    background-color: transparent;
    color: #4A5E56;
    font-size: 14px;
    padding-bottom: 6px;
}

QFrame#quickStartCard {
    background-color: #FFFFFF;
    border: 1px solid #D8E0DC;
    border-radius: 11px;
}

QLabel#quickStartTitle {
    background-color: transparent;
    color: #10231C;
    font-size: 18px;
    font-weight: 700;
}

QLabel#quickStartText {
    background-color: transparent;
    color: #375047;
    font-size: 14px;
}

QPushButton#acceptButton, QPushButton#deleteButton, QPushButton#backButton {
    background-color: #FFFFFF;
    color: #18382F;
    border: 1px solid #CAD6D1;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton#acceptButton:hover, QPushButton#deleteButton:hover, QPushButton#backButton:hover {
    background-color: #EDF7F4;
    border-color: #65CDB6;
}

QSplitter::handle {
    background-color: #DDE4E1;
    width: 1px;
}

QFrame#datasetsStatCard,
QFrame#operationsStatCard,
QFrame#modelsStatCard {
    border: none;
    border-radius: 11px;
}

QFrame#datasetsStatCard {
    background-color: #DDF3EC;
}

QFrame#operationsStatCard {
    background-color: #E3F3EF;
}

QFrame#modelsStatCard {
    background-color: #FFF0D9;
}

QLabel#statCardTitle {
    background-color: transparent;
    color: #08735F;
    font-size: 13px;
    font-weight: 650;
}

QFrame#modelsStatCard QLabel#statCardTitle {
    color: #9B6500;
}

QLabel#statCardValue {
    background-color: transparent;
    color: #10261F;
    font-size: 25px;
    font-weight: 700;
}

QFrame#historyCard {
    background-color: #FFFFFF;
    border: 1px solid #D7DEDB;
    border-radius: 13px;
}

QScrollArea#historyScroll,
QScrollArea#historyScroll QWidget#qt_scrollarea_viewport,
QWidget#historyContainer {
    background-color: transparent;
    border: none;
}

QFrame#historyEmptyState {
    background-color: #E1F5EF;
    border: 1px dashed #58CDB5;
    border-radius: 10px;
}

QLabel#historyEmptyIcon {
    background-color: transparent;
    color: #079979;
    font-size: 42px;
    font-weight: 700;
}

QLabel#historyEmptyTitle {
    background-color: transparent;
    color: #0E2B22;
    font-size: 18px;
    font-weight: 700;
}

QLabel#historyEmptyText {
    background-color: transparent;
    color: #07836B;
    font-size: 14px;
    font-weight: 550;
}

QPushButton#settingsButton {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(213, 235, 231, 0.12);
    border-radius: 9px;
    min-width: 38px;
    max-width: 38px;
    min-height: 38px;
    max-height: 38px;
    padding: 0px;
}

QPushButton#settingsButton:hover {
    background-color: #155548;
    border-color: rgba(213, 235, 231, 0.28);
}

QPushButton#settingsButton:pressed {
    background-color: #1B6254;
}

QMenu#settingsMenu,
QMenu#languageMenu,
QMenu {
    background-color: #FFFFFF;
    color: #17342C;
    border: 1px solid #CAD8D3;
    border-radius: 8px;
    padding: 6px;
    font-size: 13px;
}

QMenu::item {
    background-color: transparent;
    border-radius: 6px;
    padding: 8px 28px 8px 12px;
    margin: 2px;
}

QMenu::item:selected {
    background-color: #DDF3EC;
    color: #075F4F;
}

QMenu::indicator {
    width: 14px;
    height: 14px;
}

QMenu::indicator:checked {
    background-color: #17A884;
    border: 2px solid #FFFFFF;
    border-radius: 7px;
}

QMenu::right-arrow {
    width: 8px;
    height: 8px;
}

"""

MENU_BUTTON_STYLE = r"""
QPushButton#menuButton {
    background-color: transparent;
    color: #E0EFEB;
    border: none;
    border-radius: 8px;
    padding: 9px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
}

QPushButton#menuButton:hover {
    background-color: #155548;
    color: #FFFFFF;
}

QPushButton#menuButton:pressed {
    background-color: #1B6254;
}

QPushButton#menuButton:checked {
    background-color: #1A5C4F;
    color: #FFFFFF;
    font-weight: 700;
}
"""

# Dashboard additions are appended to MAIN_WINDOW_STYLE in the generated file below.
