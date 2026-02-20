import tempfile
import os

def read_eqdsk_from_bytes(raw_bytes, reader_func):
    """
    Call an existing eqdsk reader that expects a filename,
    but feed it in-memory bytes instead of a file on disk.
    """
    # Write raw bytes to a temporary file, then pass its path
    # to the reader exactly as it expects.
    with tempfile.NamedTemporaryFile(
        mode="wb",          # write bytes
        suffix=".eqdsk",
        delete=False,       # keep alive until we've finished reading
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        result = reader_func(tmp_path)
    finally:
        os.remove(tmp_path)  # clean up no matter what

    return result