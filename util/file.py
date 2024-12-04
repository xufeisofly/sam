# coding: utf-8

def get_file_name_without_ext(file_path: str) -> str:
    return file_path.split("/")[-1].split(".")[0]