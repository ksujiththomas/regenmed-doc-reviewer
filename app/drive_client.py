"""Google Drive access for Away mode.

Auth: the app's own OAuth client (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET).
The user connects once via /drive/connect; the refresh token is stored in
the local DB. All calls below take an authorized service object.
"""
import io
import os
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"

_client_id = None
_client_secret = None


def client_configured():
    return bool(os.environ.get("GOOGLE_CLIENT_ID")
                and os.environ.get("GOOGLE_CLIENT_SECRET"))


def _client_id():
    global _client_id
    if _client_id is None:
        _client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    return _client_id


def _client_secret():
    global _client_secret
    if _client_secret is None:
        _client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    return _client_secret


def creds_from_refresh_token(refresh_token):
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_client_id(),
        client_secret=_client_secret(),
        scopes=SCOPES,
    )
    creds.refresh(Request())  # raises RefreshError when revoked/expired
    return creds


def drive_service(refresh_token):
    return build("drive", "v3",
                 credentials=creds_from_refresh_token(refresh_token),
                 cache_discovery=False)


def account_email(svc):
    about = svc.about().get(fields="user(emailAddress)").execute()
    return about.get("user", {}).get("emailAddress", "")


def list_pdfs(svc, folder_id, page_size=50):
    """PDFs directly inside folder_id, not trashed. Returns list of dicts."""
    q = (f"'{folder_id}' in parents and trashed=false "
         f"and mimeType='application/pdf'")
    out, page_token = [], None
    while True:
        resp = svc.files().list(
            q=q, fields="files(id,name,parents,modifiedTime,size),nextPageToken",
            pageSize=page_size, pageToken=page_token,
            orderBy="modifiedTime").execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token or len(out) >= 200:
            break
    return out


def download_pdf(svc, file_id):
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def move_file(svc, file_id, from_parent_id, to_parent_id):
    svc.files().update(
        fileId=file_id,
        addParents=to_parent_id,
        removeParents=from_parent_id,
        fields="id,parents").execute()


def ensure_folder(svc, name, parent_id):
    """Return the id of a child folder called name, creating it if needed."""
    q = (f"'{parent_id}' in parents and trashed=false "
         f"and mimeType='{FOLDER_MIME}' and name='{name}'")
    resp = svc.files().list(q=q, fields="files(id)", pageSize=1).execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    created = svc.files().create(
        body={"name": name, "mimeType": FOLDER_MIME,
              "parents": [parent_id]},
        fields="id").execute()
    return created["id"]


def upload_report(svc, name, content_bytes, parent_id,
                  mimetype="text/csv"):
    media = MediaIoBaseUpload(io.BytesIO(content_bytes),
                              mimetype=mimetype, resumable=True)
    created = svc.files().create(
        body={"name": name, "parents": [parent_id]},
        media_body=media, fields="id").execute()
    return created["id"]


def get_folder(svc, folder_id):
    """Validate an id is a readable folder; returns its name."""
    meta = svc.files().get(
        fileId=folder_id,
        fields="id,name,mimeType").execute()
    if meta.get("mimeType") != FOLDER_MIME:
        raise ValueError("That link is not a folder.")
    return meta["name"]


def list_top_folders(svc, page_size=50):
    """Folders the user owns / can see, for the picker."""
    q = f"mimeType='{FOLDER_MIME}' and trashed=false"
    resp = svc.files().list(
        q=q, fields="files(id,name,modifiedTime)",
        pageSize=page_size, orderBy="modifiedTime desc").execute()
    return resp.get("files", [])


def extract_folder_id(text):
    """Accept a raw id or a Drive folder URL; return the id or ''."""
    text = (text or "").strip()
    if not text:
        return ""
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", text)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", text):
        return text
    return ""
