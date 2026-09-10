#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_교안 -> CICD, 02_이미지 -> CICD/images 자동 동기화 스크립트.

CICD 폴더 안에서 이 스크립트를 실행하면(파이썬 표준 라이브러리만 사용, 설치 불필요),
원본 교안(01_교안)과 이미지(02_이미지) 폴더를 지켜보다가 파일이 새로 생기거나
바뀔 때마다 CICD 폴더 안의 사본에 자동으로 복사합니다.
VS Code에서 CICD 폴더만 열어 작업할 때, Live Server 등으로 CICD를 서빙 중이면
이 스크립트가 사본을 갱신하는 순간 브라우저가 자동으로 새로고침됩니다.

사용법 (CICD 폴더의 VS Code 터미널에서):
    python3 sync.py          # 계속 감시하며 자동 동기화 (Ctrl+C로 종료)
    python3 sync.py --once   # 지금 한 번만 동기화하고 종료
"""

import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # CICD 폴더
PROJECT_ROOT = BASE_DIR.parent                        # 01_교안, 02_이미지가 있는 상위 폴더
SRC_MD_DIR = PROJECT_ROOT / "01_교안"
SRC_IMG_DIR = PROJECT_ROOT / "02_이미지"
DEST_IMG_DIR = BASE_DIR / "images"

POLL_SECONDS = 1.0


def sync_dir(src_dir: Path, dest_dir: Path, suffixes=None):
    """src_dir의 파일을 dest_dir로 복사한다. 새 파일이거나 수정시간이 더 최신이면 복사."""
    if not src_dir.is_dir():
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    synced = []

    for src_file in src_dir.iterdir():
        if not src_file.is_file():
            continue
        if suffixes and src_file.suffix.lower() not in suffixes:
            continue

        dest_file = dest_dir / src_file.name
        needs_copy = (
            not dest_file.exists()
            or src_file.stat().st_mtime > dest_file.stat().st_mtime
        )
        if needs_copy:
            shutil.copy2(src_file, dest_file)
            synced.append(src_file.name)

    return synced


def run_once():
    md_synced = sync_dir(SRC_MD_DIR, BASE_DIR, suffixes={".md"})
    img_synced = sync_dir(SRC_IMG_DIR, DEST_IMG_DIR)

    if md_synced or img_synced:
        stamp = time.strftime("%H:%M:%S")
        for name in md_synced:
            print(f"[{stamp}] 교안 동기화: {name}")
        for name in img_synced:
            print(f"[{stamp}] 이미지 동기화: {name}")
    return md_synced, img_synced


def main():
    if not SRC_MD_DIR.is_dir():
        print(f"원본 교안 폴더를 찾을 수 없습니다: {SRC_MD_DIR}")
        sys.exit(1)

    once = "--once" in sys.argv

    if once:
        md_synced, img_synced = run_once()
        if not md_synced and not img_synced:
            print("변경된 파일이 없습니다. 이미 최신 상태입니다.")
        return

    print(f"교안 폴더 감시 시작: {SRC_MD_DIR}")
    print(f"이미지 폴더 감시 시작: {SRC_IMG_DIR}")
    print("저장하는 즉시 CICD 폴더로 자동 복사됩니다. 종료하려면 Ctrl+C를 누르세요.\n")

    try:
        while True:
            run_once()
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n동기화를 종료합니다.")


if __name__ == "__main__":
    main()
