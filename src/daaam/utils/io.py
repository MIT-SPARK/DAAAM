from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
	"""Yield a temporary sibling of target and move it into place once the block succeeds.

	A process killed mid-write (e.g. by the shutdown watchdog) leaves the previous complete
	file untouched instead of a truncated one. The suffix is preserved so writers that pick
	the format from the extension (spark_dsg) behave identically.
	"""
	tmp = target.with_name(f"{target.stem}.tmp{target.suffix}")
	try:
		yield tmp
	except BaseException:
		tmp.unlink(missing_ok=True)
		raise
	os.replace(tmp, target)


def atomic_yaml_dump(target: Path, data: object) -> None:
	with atomic_path(target) as tmp:
		with open(tmp, "w") as f:
			yaml.safe_dump(data, f)
