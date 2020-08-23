import os

basedir = os.path.abspath(os.path.dirname('__file__'))
ALLOWED_EXT = {'jpg', 'png', 'jpeg'}


class Config(object):
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dede1234"
    UPLOAD_FOLDER = os.path.join(basedir, "uploads")
    MAX_CONTENT_LENGTH = 16*1024*1024
