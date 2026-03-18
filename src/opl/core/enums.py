from enum import StrEnum


class Sex(StrEnum):
    MALE = "M"
    FEMALE = "F"
    MX = "Mx"


class Equipment(StrEnum):
    RAW = "Raw"
    WRAPS = "Wraps"
    SINGLE_PLY = "Single-ply"
    MULTI_PLY = "Multi-ply"
    UNLIMITED = "Unlimited"
    STRAPS = "Straps"


class Event(StrEnum):
    SBD = "SBD"
    BD = "BD"
    SD = "SD"
    SB = "SB"
    S = "S"
    B = "B"
    D = "D"
