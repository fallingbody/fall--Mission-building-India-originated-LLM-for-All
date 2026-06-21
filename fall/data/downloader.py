"""
Distributed downloader for FALL training data.
Handles GitHub repos, Common Crawl, security databases, and academic papers.
"""
import asyncio
import aiohttp
import aiofiles
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self, config, num_workers=64):
        self.config = config
        self.num_workers = num_workers
        self.session: Optional[aiohttp.ClientSession] = None
        self.downloaded_hashes = set()
        self.stats = {"downloaded": 0, "failed": 0, "skipped": 0, "bytes": 0}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def download_github_repos(self, repo_list: List[str], output_dir: str):
        """Download and extract GitHub repositories."""
        os.makedirs(output_dir, exist_ok=True)
        semaphore = asyncio.Semaphore(self.num_workers)

        async def download_one(repo_url):
            async with semaphore:
                try:
                    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                    repo_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:16]
                    output_path = os.path.join(output_dir, f"{repo_name}_{repo_hash}")

                    if os.path.exists(output_path):
                        self.stats["skipped"] += 1
                        return

                    # Clone via subprocess (aiohttp for API, git for repos)
                    process = await asyncio.create_subprocess_exec(
                        "git", "clone", "--depth=1", repo_url, output_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await process.communicate()

                    if process.returncode == 0:
                        self.stats["downloaded"] += 1
                        logger.info(f"Downloaded: {repo_url}")
                    else:
                        self.stats["failed"] += 1
                        logger.error(f"Failed: {repo_url} - {stderr.decode()[:200]}")
                except Exception as e:
                    self.stats["failed"] += 1
                    logger.error(f"Error downloading {repo_url}: {e}")

        tasks = [download_one(repo) for repo in repo_list]
        await asyncio.gather(*tasks)

    async def download_common_crawl(self, warc_paths: List[str], output_dir: str):
        """Download and extract Common Crawl WARC files."""
        os.makedirs(output_dir, exist_ok=True)
        semaphore = asyncio.Semaphore(self.num_workers)

        async def download_one(warc_url):
            async with semaphore:
                try:
                    filename = warc_url.split("/")[-1]
                    output_path = os.path.join(output_dir, filename)
                    if os.path.exists(output_path):
                        self.stats["skipped"] += 1
                        return
                    async with self.session.get(warc_url) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            async with aiofiles.open(output_path, "wb") as f:
                                await f.write(content)
                            self.stats["downloaded"] += 1
                            self.stats["bytes"] += len(content)
                        else:
                            self.stats["failed"] += 1
                except Exception as e:
                    self.stats["failed"] += 1

        tasks = [download_one(url) for url in warc_paths]
        await asyncio.gather(*tasks)

    async def download_security_corpus(self, output_dir: str):
        """Download CVE database, Exploit-DB, and security papers."""
        sources = [
            "https://cve.mitre.org/data/downloads/allitems.csv",
            "https://raw.githubusercontent.com/offensive-security/exploitdb/master/files_exploits.csv",
        ]
        os.makedirs(output_dir, exist_ok=True)
        for url in sources:
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        filename = url.split("/")[-1]
                        async with aiofiles.open(os.path.join(output_dir, filename), "w") as f:
                            await f.write(content)
                        self.stats["downloaded"] += 1
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")

    def get_stats(self):
        return dict(self.stats)