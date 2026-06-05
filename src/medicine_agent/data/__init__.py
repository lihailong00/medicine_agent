"""Data ingestion and LIANA processing lane."""

__all__ = ["process_data_dir", "summarize_liana_files"]


def __getattr__(name: str):
    if name in __all__:
        from .liana import process_data_dir, summarize_liana_files

        return {"process_data_dir": process_data_dir, "summarize_liana_files": summarize_liana_files}[name]
    raise AttributeError(name)
