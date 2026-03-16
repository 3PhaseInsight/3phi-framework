from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaseConfig:
    def to_dict(self) -> dict:
        return asdict(self)
