from enum import StrEnum


class ProcessingLevel(StrEnum):
    RAW = "raw"
    CLEANED = "cleaned"
    CLEANED_AND_CORRECTED = "cleaned_and_corrected"
