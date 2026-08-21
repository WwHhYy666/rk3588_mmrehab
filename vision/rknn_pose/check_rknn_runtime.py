from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    print("python:", sys.executable)
    print("version:", sys.version.replace("\n", " "))
    try:
        import rknnlite
    except Exception as exc:
        print(f"rknnlite import failed: {exc}")
        return 2

    root = pathlib.Path(rknnlite.__file__).resolve().parent
    print("rknnlite root:", root)

    package_sos = sorted(root.rglob("*.so*"))
    print("\n[rknnlite package .so files]")
    if not package_sos:
        print("none")
    for path in package_sos:
        print(path)

    print("\n[ldd links mentioning rknn]")
    ldd = shutil.which("ldd")
    if ldd is None:
        print("ldd not found")
    else:
        for path in package_sos:
            print(f"==== {path} ====")
            result = subprocess.run([ldd, str(path)], text=True, capture_output=True, check=False)
            lines = [line for line in result.stdout.splitlines() if "rknn" in line.lower()]
            if lines:
                print("\n".join(lines))
            else:
                print("no rknn link shown by ldd")
            if result.stderr.strip():
                print(result.stderr.strip())

    print("\n[system librknnrt candidates]")
    find = shutil.which("find")
    if find is None:
        print("find not found")
    else:
        result = subprocess.run(
            [find, "/", "-name", "librknnrt.so*"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            print(line)
        if result.returncode not in (0, 1):
            print(result.stderr.strip())

    print("\n[ldconfig rknn entries]")
    ldconfig = shutil.which("ldconfig")
    if ldconfig is None:
        print("ldconfig not found")
    else:
        result = subprocess.run([ldconfig, "-p"], text=True, capture_output=True, check=False)
        lines = [line for line in result.stdout.splitlines() if "rknn" in line.lower()]
        print("\n".join(lines) if lines else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
