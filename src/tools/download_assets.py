"""Download a reviewed model repository snapshot at an explicit revision."""
import argparse
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', required=True, help='Team-selected Hugging Face model repository')
    p.add_argument('--revision', required=True, help='Prefer an immutable commit hash')
    p.add_argument('--output', default='weights')
    a = p.parse_args()
    snapshot_download(repo_id=a.repo_id, revision=a.revision, local_dir=a.output,
                      allow_patterns=['*.pth','*.parquet','*.json','README.md'])


if __name__ == '__main__':
    main()
