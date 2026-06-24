import hashlib

from models.state import MigrationState


def generate_key(state: MigrationState) -> str:
    content = (
        f"{state.source_code}|"
        f"{state.source_language}|{state.source_version}|"
        f"{state.target_language}|{state.target_version}"
    )
    return f"migrate:{hashlib.sha256(content.encode()).hexdigest()}"
