"""Resumable sequential downloader for the selected 500 TCGA-BRCA SVS files."""
import csv
import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import openslide


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata"
SELECTION_CSV = METADATA / "selected_500_slides.csv"
STATUS_CSV = METADATA / "download_status_500.csv"
OUTPUT_ROOT = ROOT / "data" / "images" / "tcga_brca_500"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data/"
MAX_ATTEMPTS = 3
CHUNK_SIZE = 1024 * 1024
STATUS_FIELDS = [
    "patient_id", "case_uuid", "slide_file_uuid", "slide_file_name",
    "status", "downloaded_bytes", "expected_bytes", "validation_result",
    "slide_width", "slide_height", "error_message",
]


def load_selection():
    with SELECTION_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 500:
        raise ValueError("Expected exactly 500 selected slides, found %d" % len(rows))
    uuids = [row["slide_file_uuid"] for row in rows]
    if len(set(uuids)) != 500:
        raise ValueError("Selected CSV contains duplicate slide UUIDs")
    for row in rows:
        if not all(row.get(key) for key in
                   ("patient_id", "case_uuid", "slide_file_uuid", "slide_file_name")):
            raise ValueError("Selected CSV contains an incomplete row")
        if Path(row["slide_file_name"]).name != row["slide_file_name"]:
            raise ValueError("Unsafe filename in selected CSV")
    return rows


def read_prior_status():
    if not STATUS_CSV.exists():
        return {}
    with STATUS_CSV.open(newline="") as handle:
        return {row["slide_file_uuid"]: row for row in csv.DictReader(handle)}


def save_status(rows):
    temporary = STATUS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(STATUS_CSV)


def get_expected_sizes(session, rows):
    sizes = {}
    for offset in range(0, len(rows), 100):
        ids = [row["slide_file_uuid"] for row in rows[offset:offset + 100]]
        filters = {"op": "in", "content": {"field": "file_id", "value": ids}}
        response = session.get(
            GDC_FILES_ENDPOINT,
            params={"filters": json.dumps(filters), "format": "JSON", "size": 100,
                    "fields": "file_id,file_name,file_size,data_format"},
            timeout=(20, 90),
        )
        response.raise_for_status()
        data = response.json()["data"]
        for hit in data["hits"]:
            if hit.get("data_format") != "SVS":
                raise ValueError("GDC UUID %s is not SVS format" % hit.get("file_id"))
            if hit.get("file_name", "").lower()[-4:] != ".svs":
                raise ValueError("GDC UUID %s has a non-SVS filename" % hit.get("file_id"))
            sizes[hit["file_id"]] = int(hit["file_size"])
    missing = set(row["slide_file_uuid"] for row in rows) - set(sizes)
    if missing:
        raise ValueError("GDC metadata missing %d selected UUIDs" % len(missing))
    return sizes


def verify_svs(path, expected_size):
    if not path.is_file():
        raise ValueError("downloaded file does not exist")
    actual_size = path.stat().st_size
    if actual_size <= 0:
        raise ValueError("downloaded file is empty")
    if actual_size != expected_size:
        raise ValueError("size mismatch: got %d, expected %d" % (actual_size, expected_size))
    slide = openslide.OpenSlide(str(path))
    try:
        width, height = slide.dimensions
        if width <= 0 or height <= 0:
            raise ValueError("OpenSlide returned invalid dimensions %r" % (slide.dimensions,))
        if not slide.level_dimensions or any(w <= 0 or h <= 0
                                             for w, h in slide.level_dimensions):
            raise ValueError("OpenSlide returned invalid level dimensions")
        return width, height
    finally:
        slide.close()


def download_one(session, row, expected_size, status_row, all_status):
    patient_dir = OUTPUT_ROOT / row["patient_id"]
    patient_dir.mkdir(parents=True, exist_ok=True)
    destination = patient_dir / row["slide_file_name"]
    partial = destination.with_name(destination.name + ".part")

    if destination.exists() and destination.stat().st_size == expected_size:
        try:
            width, height = verify_svs(destination, expected_size)
            status_row.update(status="skipped_existing", downloaded_bytes=expected_size,
                              expected_bytes=expected_size, validation_result="passed",
                              slide_width=width, slide_height=height, error_message="")
            save_status(all_status)
            return
        except Exception as exc:
            # This file was never recorded as a valid completed download. Preserve it
            # until a verified replacement is ready; completed downloads are not removed.
            status_row.update(validation_result="failed", slide_width="", slide_height="",
                              error_message="Existing file failed verification: %s" % exc)

    status_row.update(status="pending", downloaded_bytes=0,
                      expected_bytes=expected_size, error_message="")
    save_status(all_status)
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        status_row["status"] = "downloading"
        status_row["error_message"] = ""
        status_row["downloaded_bytes"] = partial.stat().st_size if partial.exists() else 0
        save_status(all_status)
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                partial.unlink()
                offset = 0
            headers = {"Range": "bytes=%d-" % offset} if offset else {}
            response = session.get(
                urljoin(GDC_DATA_ENDPOINT, row["slide_file_uuid"]),
                headers=headers, stream=True, timeout=(30, 180),
            )
            response.raise_for_status()
            append_mode = False
            if offset and response.status_code == 206:
                # GDC has returned both RFC-style "bytes START-END/TOTAL" and
                # bare "START-END/TOTAL" Content-Range values. Validate the start
                # offset in either representation before appending.
                content_range = response.headers.get("Content-Range", "")
                range_value = content_range.removeprefix("bytes ")
                range_start = range_value.split("-", 1)[0]
                if range_start != str(offset):
                    raise ValueError("invalid Content-Range for resume: %s" % content_range)
                append_mode = True
            elif offset and response.status_code == 200:
                # Range unsupported: safely restart this one partial download.
                offset = 0
            elif not offset and response.status_code not in (200, 206):
                raise ValueError("unexpected HTTP status %d" % response.status_code)

            mode = "ab" if append_mode else "wb"
            with partial.open(mode) as output:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        output.write(chunk)
                        status_row["downloaded_bytes"] = partial.stat().st_size
                        # Keep the CSV useful during long transfers without writing on
                        # every chunk.
                        if status_row["downloaded_bytes"] % (16 * CHUNK_SIZE) < len(chunk):
                            save_status(all_status)
                output.flush()
                os.fsync(output.fileno())
            response.close()

            if partial.stat().st_size != expected_size:
                raise ValueError("size mismatch: got %d, expected %d" %
                                 (partial.stat().st_size, expected_size))
            # Atomic replacement ensures a prior file is kept until this candidate is
            # complete and size-checked. If there was a verified completed file, it was
            # skipped above and is never replaced.
            os.replace(str(partial), str(destination))
            width, height = verify_svs(destination, expected_size)
            status_row.update(status="completed", downloaded_bytes=expected_size,
                              expected_bytes=expected_size, validation_result="passed",
                              slide_width=width, slide_height=height, error_message="")
            save_status(all_status)
            return
        except Exception as exc:
            last_error = "%s (attempt %d/%d)" % (exc, attempt, MAX_ATTEMPTS)
            status_row.update(status="failed" if attempt == MAX_ATTEMPTS else "pending",
                              downloaded_bytes=partial.stat().st_size if partial.exists() else 0,
                              expected_bytes=expected_size, error_message=last_error)
            if destination.exists():
                status_row.update(validation_result="failed", slide_width="", slide_height="")
            save_status(all_status)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 8))
    status_row["status"] = "failed"
    status_row["error_message"] = last_error
    save_status(all_status)


def main():
    rows = load_selection()
    previous = read_prior_status()
    statuses = []
    status_by_uuid = {}
    for row in rows:
        item = {key: row[key] for key in ("patient_id", "case_uuid", "slide_file_uuid",
                                          "slide_file_name")}
        prior = previous.get(row["slide_file_uuid"], {})
        item.update(status="pending", downloaded_bytes=prior.get("downloaded_bytes", 0),
                    expected_bytes=prior.get("expected_bytes", ""),
                    validation_result=prior.get("validation_result", "not_checked"),
                    slide_width=prior.get("slide_width", ""),
                    slide_height=prior.get("slide_height", ""), error_message="")
        statuses.append(item)
        status_by_uuid[row["slide_file_uuid"]] = item
    save_status(statuses)

    session = requests.Session()
    session.headers.update({"User-Agent": "tcga-brca-selected-slide-downloader/1.0"})
    # No token is required for the public TCGA-BRCA diagnostic slide files. If access
    # changes, raise_for_status records the API error; credentials are never guessed.
    expected_sizes = get_expected_sizes(session, rows)
    for index, row in enumerate(rows, start=1):
        item = status_by_uuid[row["slide_file_uuid"]]
        item["expected_bytes"] = expected_sizes[row["slide_file_uuid"]]
        save_status(statuses)
        download_one(session, row, expected_sizes[row["slide_file_uuid"]], item, statuses)
        print("[%d/%d] %s %s (%s/%s bytes)" % (
            index, len(rows), row["slide_file_uuid"], item["status"],
            item["downloaded_bytes"], item["expected_bytes"]), flush=True)
    counts = {status: sum(row["status"] == status for row in statuses)
              for status in ("completed", "skipped_existing", "failed")}
    print("completed=%d skipped=%d failed=%d total=%d" % (
        counts["completed"], counts["skipped_existing"], counts["failed"], len(rows)))


if __name__ == "__main__":
    main()
