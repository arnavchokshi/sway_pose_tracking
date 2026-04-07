#!/usr/bin/env python3
"""
Serve the batch output folder over HTTP so the review page can load local MP4s.

Opening review/index.html as file:// often shows a blank video in Chrome/Safari.
Use this instead (works offline; no network required).

  cd sway_pose_mvp
  python review_app/serve_review.py output/flight_batch

Then open: http://localhost:8899/review/index.html

HTTP byte-range (206) is required for video scrubbing/seeking. Python 3.11's
SimpleHTTPRequestHandler does not implement Range — this server adds it.

The review page can POST to /__sway__/delete-sample (same origin) to delete the
batch input video listed in batch_manifest.json, remove this clip's output
folder, update the manifest, and regenerate review/index.html. That only works
when using this server, not file://.
"""

from __future__ import annotations

import argparse
import datetime
import email.utils
import errno
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import urllib.parse
from http import HTTPStatus
from pathlib import Path


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Like SimpleHTTPRequestHandler but supports Range: bytes=… for seeking in MP4s.
    """

    def copyfile(self, source, outputfile) -> None:  # noqa: ANN001
        # Client may close mid-stream (seek / new Range); ignore pipe/reset errors.
        try:
            rng = getattr(self, "_range_copy_len", None)
            if rng is not None:
                nleft: int = rng
                bufsize = 64 * 1024
                while nleft > 0:
                    chunk = source.read(min(bufsize, nleft))
                    if not chunk:
                        break
                    outputfile.write(chunk)
                    nleft -= len(chunk)
                self._range_copy_len = None
            else:
                shutil.copyfileobj(source, outputfile)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass
        except OSError as e:
            # Some stacks surface EPIPE/ECONNRESET as OSError instead.
            if e.errno not in (errno.EPIPE, errno.ECONNRESET):
                raise

    def send_head(self):
        """Common code for GET and HEAD; handles Range for partial content."""
        self._range_copy_len = None
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            parts = urllib.parse.urlsplit(self.path)
            if not parts.path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                new_url = urllib.parse.urlunsplit(
                    (parts[0], parts[1], parts[2] + "/", parts[3], parts[4])
                )
                self.send_header("Location", new_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.isfile(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        if path.endswith("/"):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            file_len = fs.st_size

            if "If-Modified-Since" in self.headers and "If-None-Match" not in self.headers:
                try:
                    ims = email.utils.parsedate_to_datetime(
                        self.headers["If-Modified-Since"]
                    )
                except (TypeError, IndexError, OverflowError, ValueError):
                    pass
                else:
                    if ims.tzinfo is None:
                        ims = ims.replace(tzinfo=datetime.timezone.utc)
                    if ims.tzinfo is datetime.timezone.utc:
                        last_modif = datetime.datetime.fromtimestamp(
                            fs.st_mtime, datetime.timezone.utc
                        ).replace(microsecond=0)
                        if last_modif <= ims:
                            self.send_response(HTTPStatus.NOT_MODIFIED)
                            self.end_headers()
                            f.close()
                            return None

            rng = _parse_single_byte_range(self.headers.get("Range"), file_len)
            if rng is not None:
                start, end = rng
                if start >= file_len or start > end:
                    f.close()
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_len}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                end = min(end, file_len - 1)
                f.seek(start)
                chunk_len = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-type", ctype)
                self.send_header("Content-Length", str(chunk_len))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_len}")
                self.send_header(
                    "Last-Modified", self.date_time_string(fs.st_mtime)
                )
                self.end_headers()
                self._range_copy_len = chunk_len
                return f

            self.send_response(HTTPStatus.OK)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(file_len))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise


def _parse_single_byte_range(
    range_header: str | None, file_len: int
) -> tuple[int, int] | None:
    """
    Parse one Range: bytes=… spec. Returns (start, end) inclusive, or None to
    serve the full file (ignore Range).
    """
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[6:].strip().split(",")[0].strip()
    if not spec or spec.count("-") != 1:
        return None
    left, right = spec.split("-", 1)
    try:
        if left == "":
            suffix = int(right)
            if suffix <= 0:
                return None
            start = max(0, file_len - suffix)
            end = file_len - 1
        elif right == "":
            start = int(left)
            end = file_len - 1
        else:
            start = int(left)
            end = int(right)
    except ValueError:
        return None
    if start < 0:
        return None
    return start, end


_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _strict_subpath(root: Path, candidate: Path) -> bool:
    """True if candidate is root or a directory/file strictly inside root."""
    try:
        r = root.resolve()
        c = candidate.resolve()
        c.relative_to(r)
        return c != r
    except ValueError:
        return False


def _regenerate_review_index(output_root: Path) -> None:
    gen = Path(__file__).resolve().parent / "generate_review_index.py"
    if not gen.is_file():
        return
    subprocess.run(
        [sys.executable, str(gen), str(output_root)],
        check=False,
        capture_output=True,
        text=True,
    )


def _delete_sample(output_root: Path, sample_id: str) -> dict:
    """
    Delete the batch input video (if manifest lists it) and remove the sample
    output directory + manifest entry, then refresh review/index.html.
    """
    if not _SAMPLE_ID_RE.match(sample_id):
        raise ValueError("invalid sample_id")

    root = output_root.resolve()
    sample_dir = root / sample_id
    if not _strict_subpath(root, sample_dir):
        raise ValueError("invalid output path")

    deleted_input = False
    deleted_output = False
    manifest_path = root / "batch_manifest.json"
    manifest: dict | None = None
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    input_path: Path | None = None
    input_dir: Path | None = None
    if manifest:
        samples = manifest.get("samples")
        if isinstance(samples, dict):
            entry = samples.get(sample_id)
            if isinstance(entry, dict):
                ip = entry.get("input_path")
                if ip:
                    input_path = Path(str(ip))
                idir = manifest.get("input_dir")
                if idir:
                    input_dir = Path(str(idir))

    if input_path and input_dir and input_path.is_file():
        idir_res = input_dir.resolve()
        ip_res = input_path.resolve()
        try:
            ip_res.relative_to(idir_res)
        except ValueError:
            pass
        else:
            if ip_res != idir_res:
                ip_res.unlink()
                deleted_input = True

    if sample_dir.is_dir():
        shutil.rmtree(sample_dir, ignore_errors=False)
        deleted_output = True
    elif sample_dir.exists():
        sample_dir.unlink()
        deleted_output = True

    if manifest is not None:
        samples = manifest.get("samples")
        if isinstance(samples, dict) and sample_id in samples:
            del samples[sample_id]
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    _regenerate_review_index(root)

    return {
        "ok": True,
        "sample_id": sample_id,
        "deleted_input": deleted_input,
        "deleted_output": deleted_output,
    }


class SwayReviewHTTPRequestHandler(RangeHTTPRequestHandler):
    """Adds POST /__sway__/delete-sample when serving a batch output root."""

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/__sway__/delete-sample":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        root = getattr(self.server, "sway_output_root", None)
        if not root or not isinstance(root, Path):
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Server misconfigured")
            return

        length_hdr = self.headers.get("Content-Length", "0")
        try:
            length = int(length_hdr)
        except ValueError:
            length = 0
        if length > 1_000_000:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Body too large")
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") if raw else "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        sample_id = body.get("sample_id")
        if not isinstance(sample_id, str):
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing sample_id")
            return

        try:
            payload = _delete_sample(root, sample_id.strip())
        except ValueError as e:
            self.send_error(HTTPStatus.BAD_REQUEST, str(e))
            return
        except OSError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
            return

        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP server for Sway review UI")
    ap.add_argument(
        "output_root",
        help="Batch folder containing review/ (e.g. output/flight_batch)",
    )
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.output_root))
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")
    os.chdir(root)

    port = args.port
    httpd = None
    for _ in range(32):
        try:
            httpd = ReusableTCPServer(("", port), SwayReviewHTTPRequestHandler)
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            port += 1
    if httpd is None:
        raise SystemExit(
            f"Could not bind to ports {args.port}–{args.port + 31}. "
            "Stop other servers or pass --port with a free port."
        )

    httpd.sway_output_root = Path(root)  # type: ignore[attr-defined]

    with httpd:
        if port != args.port:
            print(f"Note: port {args.port} was in use; using {port} instead.")
        print(f"Serving: {root}")
        print(f"Open:  http://localhost:{port}/review/index.html")
        print("Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
