
import os
from github import Github, Repository, ContentFile
import requests
from argparse import ArgumentParser, Namespace


def download(c: ContentFile, out: str):
    r = requests.get(c.download_url)
    output_path = f'{out}/{c.path}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        print(f'downloading {c.path} to {out}')
        f.write(r.content)


def download_demo(repo, folder, data_dir: str, recursive: bool):
    g = Github()
    contents = g.get_repo(repo).get_contents(folder)
    for c in contents:
        if c.download_url is None:
            if recursive:
                download_demo(repo, c.path, data_dir, recursive)
            continue
        download(c, data_dir)


def download_file(repo: Repository, folder: str, out: str):
    c = repo.get_contents(folder)
    download(c, out)
