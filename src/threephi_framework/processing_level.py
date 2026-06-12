from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    # Python < 3.11 compatibility.
    class StrEnum(str, Enum):
        pass


class ProcessingLevel(StrEnum):
    RAW = "raw"
    CLEANED = "cleaned"
    CLEANED_AND_CORRECTED = "cleaned_and_corrected"
